import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { assertArchiveSafe, extractArchive } from "./archive.mjs";
import { getLatestPreview } from "./artifacts.mjs";
import { loadConfig, resolveColmapFlagStyle } from "./config.mjs";
import { openDatabase, createJobStore } from "./db.mjs";
import { inspectInput } from "./dataset.mjs";
import { createLogger } from "./logger.mjs";
import { parseFinalMetrics, readTail } from "./logs.mjs";
import { getPreset } from "./presets.mjs";
import { clamp, ensureDir, fileExists, safeFileName } from "./utils.mjs";

const PHASE_PROGRESS = {
  validating: 10,
  converting_text_model: 20,
  extracting_features: 35,
  matching: 55,
  mapping: 65,
  selecting_model: 70,
  training: 70,
  finalizing: 100
};
const SMALL_RAW_RECONSTRUCTION_IMAGE_COUNT = 8;

class CancelledJobError extends Error {
  constructor() {
    super("Job cancelled");
    this.name = "CancelledJobError";
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isCommandPath(command) {
  return command.includes("/") || command.startsWith(".");
}

function buildRunArgs(job) {
  const preset = getPreset(job.preset);
  if (!preset) {
    throw new Error(`Unknown preset: ${job.preset}`);
  }

  const outputDir = path.dirname(job.outputPath);
  const outputPlyPath = path.join(outputDir, "final.ply");

  return [
    job.datasetRoot,
    "-o",
    job.outputPath,
    "--export-ply",
    outputPlyPath,
    ...preset.args,
    "--val-render",
    job.previewsDir
  ];
}

function killProcessGroup(child) {
  if (!child?.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    try {
      child.kill("SIGTERM");
    } catch {
      // ignore
    }
  }
}

async function writeLogBanner(logPath, message) {
  await fsp.appendFile(logPath, `${message}\n`, "utf8");
}

async function resetDirectory(dirPath) {
  await fsp.rm(dirPath, { recursive: true, force: true });
  await ensureDir(dirPath);
}

async function findNumericModelDirs(sparseRoot) {
  const entries = await fsp.readdir(sparseRoot, { withFileTypes: true }).catch(() => []);
  const numeric = entries
    .filter((entry) => entry.isDirectory() && /^\d+$/.test(entry.name))
    .map((entry) => path.join(sparseRoot, entry.name))
    .sort((left, right) => left.localeCompare(right));

  if (numeric.length > 0) {
    return numeric;
  }

  if (
    (await fileExists(path.join(sparseRoot, "cameras.bin"))) &&
    (await fileExists(path.join(sparseRoot, "images.bin")))
  ) {
    return [sparseRoot];
  }

  return [];
}

function parseAnalyzerStats(output) {
  const registeredImages = Number(output.match(/Registered images:\s*([0-9]+)/)?.[1] ?? 0);
  const points = Number(output.match(/Points:\s*([0-9]+)/)?.[1] ?? 0);
  return { registeredImages, points };
}

function matchesConfiguredCommand(command, configuredCommand) {
  if (command === configuredCommand) {
    return true;
  }
  if (isCommandPath(command) && isCommandPath(configuredCommand)) {
    return path.resolve(command) === path.resolve(configuredCommand);
  }
  return false;
}

function buildColmapGpuArgs(style) {
  if (style === "legacy") {
    return {
      featureExtraction: ["--SiftExtraction.use_gpu", "0"],
      featureMatching: ["--SiftMatching.use_gpu", "0"]
    };
  }
  return {
    featureExtraction: ["--FeatureExtraction.use_gpu", "0"],
    featureMatching: ["--FeatureMatching.use_gpu", "0"]
  };
}

function buildMatcherArgs(databasePath, requestedMode, imageCount, gpuArgs) {
  const mode = requestedMode === "exhaustive" ? "exhaustive" : "sequential";
  const args = [
    mode === "exhaustive" ? "exhaustive_matcher" : "sequential_matcher",
    "--database_path",
    databasePath,
    ...gpuArgs.featureMatching
  ];

  if (mode === "sequential" && imageCount <= SMALL_RAW_RECONSTRUCTION_IMAGE_COUNT) {
    // Small batches need denser pair coverage than COLMAP's default sequential schedule.
    args.push(
      "--SequentialMatching.overlap",
      String(Math.max(3, imageCount - 1)),
      "--SequentialMatching.quadratic_overlap",
      "0"
    );
  }

  return args;
}

function buildMapperArgs(databasePath, imageRoot, sparseRoot, imageCount) {
  const args = [
    "mapper",
    "--database_path",
    databasePath,
    "--image_path",
    imageRoot,
    "--output_path",
    sparseRoot
  ];

  if (imageCount <= SMALL_RAW_RECONSTRUCTION_IMAGE_COUNT) {
    // The UI accepts tiny photo batches, so lower COLMAP's 10-image/100-inlier defaults.
    args.push(
      "--Mapper.min_model_size",
      "3",
      "--Mapper.init_min_num_inliers",
      "30",
      "--Mapper.abs_pose_min_num_inliers",
      "15",
      "--Mapper.init_min_tri_angle",
      "4",
      "--Mapper.tri_ignore_two_view_tracks",
      "0"
    );
  }

  return args;
}

export async function createWorker(overrides = {}) {
  const config = {
    ...loadConfig(),
    ...overrides
  };
  config.colmapFlagStyle = resolveColmapFlagStyle(config.colmapFlagStyle);
  const logger = createLogger("worker");
  await ensureDir(config.jobsDir);
  const db = overrides.db ?? await openDatabase(config.databasePath);
  const store = overrides.store ?? createJobStore(db);
  const ownDb = !overrides.db;

  let shuttingDown = false;
  let dbClosed = false;
  let activeChild = null;
  const lastPreviewStepByJob = new Map();

  function closeDbIfNeeded() {
    if (!ownDb || dbClosed) return;
    db.close();
    dbClosed = true;
  }

  function currentProgress(job, fallback = 0) {
    return Number.isFinite(job.progressPercent) ? job.progressPercent : fallback;
  }

  async function assertNotCancelled(jobId) {
    const latest = store.getJob(jobId);
    if (!latest || latest.cancelRequested || shuttingDown) {
      throw new CancelledJobError();
    }
    return latest;
  }

  async function setPhase(jobId, phase, phaseMessage, progressPercent, extra = {}) {
    const previous = store.getJob(jobId);
    const next = store.markPhase(jobId, phase, phaseMessage, progressPercent, extra.progressStep ?? 0, extra);
    const progressStep = extra.progressStep ?? 0;

    if (
      !previous ||
      previous.phase !== phase ||
      previous.phaseMessage !== phaseMessage ||
      previous.progressStep !== progressStep
    ) {
      logger.info("job phase updated", {
        jobId,
        phase,
        phaseMessage,
        progressPercent: Number(progressPercent?.toFixed?.(1) ?? progressPercent),
        progressStep
      });
    }

    return next;
  }

  async function runLoggedCommand(job, command, args, options = {}) {
    const {
      phase,
      phaseMessage,
      progressPercent,
      onInterval,
      captureStdout = false
    } = options;

    const startedAt = Date.now();
    const commandName = isCommandPath(command) ? path.basename(command) : command;
    const commandPath = isCommandPath(command) ? path.resolve(command) : command;
    const isColmapCommand = matchesConfiguredCommand(command, config.colmapBin);
    const childEnv = { ...process.env };
    if (commandPath === path.resolve(config.msplatBin)) {
      childEnv.MSPLAT_METALLIB_PATH = path.join(path.dirname(config.msplatBin), "default.metallib");
    }
    if (isColmapCommand) {
      childEnv.FAKE_COLMAP_FLAG_STYLE = config.colmapFlagStyle;
    }

    await setPhase(job.id, phase, phaseMessage, progressPercent ?? currentProgress(job), {
      phaseMessage,
      progressPercent: progressPercent ?? currentProgress(job)
    });
    logger.info("command starting", {
      jobId: job.id,
      phase,
      command: commandName,
      args: args.join(" "),
      colmapFlagStyle: isColmapCommand ? config.colmapFlagStyle : undefined,
      metallibPath: childEnv.MSPLAT_METALLIB_PATH
    });
    await writeLogBanner(
      job.logPath,
      `>>> [${phase}] START ${command} ${args.join(" ")}${isColmapCommand ? ` [colmap_flag_style=${config.colmapFlagStyle}]` : ""}`
    );
    await assertNotCancelled(job.id);

    const logStream = fs.createWriteStream(job.logPath, { flags: "a" });
    const child = spawn(command, args, {
      cwd: config.projectRoot,
      detached: true,
      env: childEnv,
      stdio: ["ignore", "pipe", "pipe"]
    });

    activeChild = child;
    store.setRunnerPid(job.id, child.pid);

    let captured = "";
    child.stdout.on("data", (chunk) => {
      if (captureStdout) captured += String(chunk);
    });
    child.stdout.pipe(logStream, { end: false });
    child.stderr.pipe(logStream, { end: false });

    const closePromise = new Promise((resolve, reject) => {
      child.on("error", (error) => {
        if (error.code === "ENOENT") {
          reject(new Error(`${command === config.colmapBin ? "COLMAP_BIN" : "MSPLAT_BIN"} not found: ${command}`));
          return;
        }
        reject(error);
      });
      child.on("close", (code, signal) => resolve({ code, signal }));
    });

    try {
      while (true) {
        const result = await Promise.race([
          closePromise.then((value) => ({ done: true, value })),
          sleep(250).then(() => ({ done: false }))
        ]);

        const latest = store.getJob(job.id);
        if (!latest || latest.cancelRequested || shuttingDown) {
          killProcessGroup(child);
        }

        if (onInterval) {
          await onInterval(latest ?? job);
        }

        if (result.done) {
          if ((store.getJob(job.id)?.cancelRequested ?? false) || shuttingDown || result.value.signal === "SIGTERM") {
            const durationMs = Date.now() - startedAt;
            logger.warn("command cancelled", {
              jobId: job.id,
              phase,
              command: commandName,
              durationMs
            });
            await writeLogBanner(job.logPath, `<<< [${phase}] CANCELLED ${commandName} duration_ms=${durationMs}`);
            throw new CancelledJobError();
          }
          if (result.value.code !== 0) {
            const durationMs = Date.now() - startedAt;
            logger.error("command failed", {
              jobId: job.id,
              phase,
              command: commandName,
              code: result.value.code,
              signal: result.value.signal,
              durationMs,
              colmapFlagStyle: isColmapCommand ? config.colmapFlagStyle : undefined
            });
            await writeLogBanner(
              job.logPath,
              `<<< [${phase}] FAILED ${commandName} code=${result.value.code ?? "null"} signal=${result.value.signal ?? "null"} duration_ms=${durationMs}${isColmapCommand ? ` colmap_flag_style=${config.colmapFlagStyle}` : ""}`
            );
            if (result.value.signal) {
              throw new Error(`${path.basename(command)} exited with signal ${result.value.signal}`);
            }
            throw new Error(`${path.basename(command)} exited with code ${result.value.code ?? "unknown"}`);
          }
          const durationMs = Date.now() - startedAt;
          logger.info("command finished", {
            jobId: job.id,
            phase,
            command: commandName,
            durationMs
          });
          await writeLogBanner(job.logPath, `<<< [${phase}] DONE ${commandName} duration_ms=${durationMs}`);
          return captured;
        }
      }
    } finally {
      activeChild = null;
      store.setRunnerPid(job.id, null);
      logStream.end();
    }
  }

  async function prepareDatasetRoot(targetRoot, imageRoot, imageFiles, modelDir) {
    await resetDirectory(targetRoot);
    const targetImagesRoot = path.join(targetRoot, "images");
    const targetSparseRoot = path.join(targetRoot, "sparse", "0");
    await ensureDir(targetImagesRoot);
    await ensureDir(targetSparseRoot);

    for (const sourcePath of imageFiles) {
      const relativePath = path.relative(imageRoot, sourcePath);
      const destinationPath = path.join(targetImagesRoot, relativePath);
      await ensureDir(path.dirname(destinationPath));
      await fsp.copyFile(sourcePath, destinationPath);
    }

    for (const entry of await fsp.readdir(modelDir, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const sourcePath = path.join(modelDir, entry.name);
      const destinationPath = path.join(targetSparseRoot, entry.name);
      await fsp.copyFile(sourcePath, destinationPath);
    }

    return targetRoot;
  }

  async function stageArchive(job) {
    const extractionDir = path.join(job.jobDir, "dataset");
    await resetDirectory(extractionDir);
    logger.info("extracting archive", {
      jobId: job.id,
      uploadPath: job.uploadPath,
      extractionDir
    });
    await setPhase(job.id, "validating", "Extracting archive", 5);
    await assertArchiveSafe(job.uploadPath);
    await extractArchive(job.uploadPath, extractionDir);
    await assertNotCancelled(job.id);
    return extractionDir;
  }

  async function chooseBestModel(job, sparseRoot) {
    const modelDirs = await findNumericModelDirs(sparseRoot);
    if (modelDirs.length === 0) {
      throw new Error("COLMAP mapper did not produce a sparse model");
    }
    if (modelDirs.length === 1) {
      logger.info("single COLMAP model found", {
        jobId: job.id,
        modelDir: modelDirs[0]
      });
      return modelDirs[0];
    }

    logger.info("multiple COLMAP models found", {
      jobId: job.id,
      count: modelDirs.length,
      sparseRoot
    });
    await setPhase(job.id, "selecting_model", "Selecting the best COLMAP model", 68);

    let best = null;
    for (const modelDir of modelDirs) {
      const output = await runLoggedCommand(job, config.colmapBin, ["model_analyzer", "--path", modelDir], {
        phase: "selecting_model",
        phaseMessage: `Analyzing ${path.basename(modelDir)}`,
        progressPercent: 69,
        captureStdout: true
      });
      const stats = parseAnalyzerStats(output);
      logger.info("COLMAP model analyzed", {
        jobId: job.id,
        modelDir,
        registeredImages: stats.registeredImages,
        points: stats.points
      });
      if (
        !best ||
        stats.registeredImages > best.stats.registeredImages ||
        (stats.registeredImages === best.stats.registeredImages && stats.points > best.stats.points)
      ) {
        best = { modelDir, stats };
      }
    }

    logger.info("COLMAP model selected", {
      jobId: job.id,
      modelDir: best.modelDir,
      registeredImages: best.stats.registeredImages,
      points: best.stats.points
    });
    await writeLogBanner(
      job.logPath,
      `>>> [selecting_model] selected ${best.modelDir} registered_images=${best.stats.registeredImages} points=${best.stats.points}`
    );
    return best.modelDir;
  }

  async function convertTextModel(job, info) {
    const workspace = path.join(job.jobDir, "colmap-workspace");
    const convertedDir = path.join(workspace, "converted");
    await resetDirectory(workspace);
    await ensureDir(convertedDir);

    logger.info("converting COLMAP TXT dataset", {
      jobId: job.id,
      modelDir: info.modelDir,
      imageCount: info.imageFiles.length
    });
    await runLoggedCommand(job, config.colmapBin, [
      "model_converter",
      "--input_path",
      info.modelDir,
      "--output_path",
      convertedDir,
      "--output_type",
      "BIN"
    ], {
      phase: "converting_text_model",
      phaseMessage: "Converting COLMAP TXT model to BIN",
      progressPercent: PHASE_PROGRESS.converting_text_model
    });

    const normalizedDatasetRoot = path.join(job.jobDir, "normalized-dataset");
    await prepareDatasetRoot(normalizedDatasetRoot, info.imageRoot, info.imageFiles, convertedDir);

    const dbPath = await fileExists(path.join(info.datasetRoot, "database.db"))
      ? path.join(info.datasetRoot, "database.db")
      : null;

    return {
      datasetRoot: normalizedDatasetRoot,
      datasetType: "colmap",
      sourceFormat: "colmap_txt",
      colmapWorkspacePath: workspace,
      colmapDatabasePath: dbPath,
      colmapModelPath: path.join(normalizedDatasetRoot, "sparse", "0"),
      phase: "converting_text_model",
      phaseMessage: "COLMAP TXT model converted"
    };
  }

  async function normalizeRawImages(job, info) {
    const workspace = path.join(job.jobDir, "colmap-workspace");
    const normalizedImagesDir = path.join(workspace, "images");
    await resetDirectory(workspace);
    await ensureDir(normalizedImagesDir);

    logger.info("normalizing raw images", {
      jobId: job.id,
      imageCount: info.imageFiles.length,
      sourceRoot: info.datasetRoot,
      workspace
    });
    const normalizedFiles = [];
    for (let index = 0; index < info.imageFiles.length; index += 1) {
      const sourcePath = info.imageFiles[index];
      const extension = path.extname(sourcePath) || ".jpg";
      const baseName = safeFileName(path.basename(sourcePath, extension)) || `image-${index + 1}`;
      const destinationName = `${String(index + 1).padStart(6, "0")}_${baseName}${extension}`;
      const destinationPath = path.join(normalizedImagesDir, destinationName);
      await fsp.copyFile(sourcePath, destinationPath);
      normalizedFiles.push(destinationPath);
    }

    return {
      workspace,
      imageRoot: normalizedImagesDir,
      imageFiles: normalizedFiles,
      databasePath: path.join(workspace, "database.db"),
      sparseRoot: path.join(workspace, "sparse")
    };
  }

  async function reconstructRawImages(job, info) {
    const colmapMode = job.colmapMode || "sequential";
    const gpuArgs = buildColmapGpuArgs(config.colmapFlagStyle);
    const normalized = await normalizeRawImages(job, info);
    const imageCount = normalized.imageFiles.length;
    logger.info("reconstructing raw images with COLMAP", {
      jobId: job.id,
      colmapMode,
      colmapFlagStyle: config.colmapFlagStyle,
      imageCount,
      workspace: normalized.workspace
    });
    await writeLogBanner(job.logPath, `>>> [validating] colmap_flag_style=${config.colmapFlagStyle}`);

    await runLoggedCommand(job, config.colmapBin, [
      "feature_extractor",
      "--database_path",
      normalized.databasePath,
      "--image_path",
      normalized.imageRoot,
      "--ImageReader.camera_model",
      "SIMPLE_RADIAL",
      ...gpuArgs.featureExtraction
    ], {
      phase: "extracting_features",
      phaseMessage: "Extracting COLMAP features",
      progressPercent: PHASE_PROGRESS.extracting_features
    });

    await runLoggedCommand(job, config.colmapBin, buildMatcherArgs(normalized.databasePath, colmapMode, imageCount, gpuArgs), {
      phase: "matching",
      phaseMessage: colmapMode === "exhaustive" ? "Running exhaustive matching" : "Running sequential matching",
      progressPercent: PHASE_PROGRESS.matching
    });

    const runMapper = async (phaseMessage) => {
      await resetDirectory(normalized.sparseRoot);
      await runLoggedCommand(job, config.colmapBin, buildMapperArgs(
        normalized.databasePath,
        normalized.imageRoot,
        normalized.sparseRoot,
        imageCount
      ), {
        phase: "mapping",
        phaseMessage,
        progressPercent: PHASE_PROGRESS.mapping
      });
    };

    try {
      await runMapper("Building sparse COLMAP model");
    } catch (error) {
      const shouldRetryWithExhaustive =
        colmapMode === "sequential" && imageCount <= SMALL_RAW_RECONSTRUCTION_IMAGE_COUNT;

      if (!shouldRetryWithExhaustive) {
        throw error;
      }

      logger.warn("raw image mapping failed after sequential matching; retrying with exhaustive matching", {
        jobId: job.id,
        imageCount,
        error: error.message
      });
      await writeLogBanner(
        job.logPath,
        `>>> [matching] retrying_with_exhaustive image_count=${imageCount} reason=${JSON.stringify(error.message)}`
      );

      await runLoggedCommand(job, config.colmapBin, buildMatcherArgs(normalized.databasePath, "exhaustive", imageCount, gpuArgs), {
        phase: "matching",
        phaseMessage: "Retrying with exhaustive matching",
        progressPercent: PHASE_PROGRESS.matching
      });

      try {
        await runMapper("Retrying sparse COLMAP model");
      } catch (retryError) {
        throw new Error(
          `COLMAP could not build a sparse model from ${imageCount} images. Try uploading at least 8 overlapping photos or a prepared dataset. Last error: ${retryError.message}`
        );
      }
    }

    const bestModelDir = await chooseBestModel(job, normalized.sparseRoot);
    const normalizedDatasetRoot = path.join(job.jobDir, "normalized-dataset");
    await prepareDatasetRoot(normalizedDatasetRoot, normalized.imageRoot, normalized.imageFiles, bestModelDir);

    return {
      datasetRoot: normalizedDatasetRoot,
      datasetType: "colmap",
      sourceFormat: "raw_images",
      colmapWorkspacePath: normalized.workspace,
      colmapDatabasePath: normalized.databasePath,
      colmapModelPath: bestModelDir,
      phase: "selecting_model",
      phaseMessage: "COLMAP model selected"
    };
  }

  async function normalizeUploadedJob(job) {
    await setPhase(job.id, "validating", "Inspecting upload", 5);

    const inputRoot = job.inputKind === "raw_files" ? job.uploadPath : await stageArchive(job);
    const info = await inspectInput(inputRoot);
    logger.info("upload inspected", {
      jobId: job.id,
      inputKind: job.inputKind,
      sourceFormat: info.sourceFormat,
      datasetType: info.type,
      datasetRoot: info.datasetRoot
    });
    await writeLogBanner(
      job.logPath,
      `>>> [validating] detected source_format=${info.sourceFormat} dataset_type=${info.type} dataset_root=${info.datasetRoot}`
    );

    store.patchJob(job.id, {
      sourceFormat: info.sourceFormat,
      datasetType: info.type,
      inputKind: info.sourceFormat === "raw_images" ? (job.inputKind === "raw_files" ? "raw_files" : "raw_zip") : job.inputKind
    });
    await assertNotCancelled(job.id);

    if (info.sourceFormat === "colmap_bin" || info.sourceFormat === "nerfstudio" || info.sourceFormat === "polycam") {
      logger.info("prepared dataset queued", {
        jobId: job.id,
        sourceFormat: info.sourceFormat,
        datasetRoot: info.datasetRoot
      });
      store.markQueued(job.id, {
        datasetRoot: info.datasetRoot,
        datasetType: info.type,
        sourceFormat: info.sourceFormat,
        phase: "validating",
        phaseMessage: "Prepared dataset ready",
        progressPercent: PHASE_PROGRESS.validating
      });
      return;
    }

    if (info.sourceFormat === "colmap_txt") {
      const normalized = await convertTextModel(job, info);
      await assertNotCancelled(job.id);
      logger.info("COLMAP TXT dataset queued", {
        jobId: job.id,
        datasetRoot: normalized.datasetRoot,
        colmapModelPath: normalized.colmapModelPath
      });
      store.markQueued(job.id, {
        datasetRoot: normalized.datasetRoot,
        datasetType: normalized.datasetType,
        sourceFormat: normalized.sourceFormat,
        colmapWorkspacePath: normalized.colmapWorkspacePath,
        colmapDatabasePath: normalized.colmapDatabasePath,
        colmapModelPath: normalized.colmapModelPath,
        phase: normalized.phase,
        phaseMessage: normalized.phaseMessage,
        progressPercent: PHASE_PROGRESS.converting_text_model
      });
      return;
    }

    if (info.sourceFormat === "raw_images") {
      const normalized = await reconstructRawImages(job, info);
      await assertNotCancelled(job.id);
      logger.info("raw image dataset queued", {
        jobId: job.id,
        datasetRoot: normalized.datasetRoot,
        colmapMode: job.colmapMode || "sequential",
        colmapModelPath: normalized.colmapModelPath
      });
      store.markQueued(job.id, {
        datasetRoot: normalized.datasetRoot,
        datasetType: normalized.datasetType,
        sourceFormat: normalized.sourceFormat,
        colmapMode: job.colmapMode || "sequential",
        colmapWorkspacePath: normalized.colmapWorkspacePath,
        colmapDatabasePath: normalized.colmapDatabasePath,
        colmapModelPath: normalized.colmapModelPath,
        phase: normalized.phase,
        phaseMessage: normalized.phaseMessage,
        progressPercent: PHASE_PROGRESS.selecting_model
      });
      return;
    }

    throw new Error(`Unsupported source format: ${info.sourceFormat}`);
  }

  async function runTrainingJob(job) {
    logger.info("training job starting", {
      jobId: job.id,
      datasetRoot: job.datasetRoot,
      preset: job.preset
    });
    let lastPreviewStep = lastPreviewStepByJob.get(job.id) ?? 0;
    await runLoggedCommand(job, config.msplatBin, buildRunArgs(job), {
      phase: "training",
      phaseMessage: "Training splat with msplat",
      progressPercent: PHASE_PROGRESS.training,
      onInterval: async () => {
        const latestPreview = await getLatestPreview(job.previewsDir);
        if (!latestPreview) return;
        const preset = getPreset(job.preset);
        const progressPercent = clamp(70 + (latestPreview.step / preset.iterations) * 30, 70, 100);
        if (latestPreview.step !== lastPreviewStep) {
          lastPreviewStep = latestPreview.step;
          lastPreviewStepByJob.set(job.id, latestPreview.step);
          logger.info("training preview updated", {
            jobId: job.id,
            step: latestPreview.step,
            progressPercent: Number(progressPercent.toFixed(1)),
            previewPath: latestPreview.path
          });
          await writeLogBanner(
            job.logPath,
            `>>> [training] preview step=${latestPreview.step} progress_percent=${progressPercent.toFixed(1)}`
          );
        }
        store.markPhase(job.id, "training", "Training splat with msplat", progressPercent, latestPreview.step, {
          latestPreviewPath: latestPreview.path
        });
      }
    });

    const latestPreview = await getLatestPreview(job.previewsDir);
    if (latestPreview) {
      store.markPhase(job.id, "finalizing", "Collecting training metrics", 99, latestPreview.step, {
        latestPreviewPath: latestPreview.path
      });
    } else {
      store.markPhase(job.id, "finalizing", "Collecting training metrics", 99);
    }

    const metrics = parseFinalMetrics(await readTail(job.logPath, 200000));
    logger.info("training job finished", {
      jobId: job.id,
      psnr: metrics?.psnr ?? null,
      ssim: metrics?.ssim ?? null,
      l1: metrics?.l1 ?? null,
      gaussians: metrics?.gaussians ?? null
    });
    await writeLogBanner(
      job.logPath,
      `>>> [finalizing] metrics psnr=${metrics?.psnr ?? "n/a"} ssim=${metrics?.ssim ?? "n/a"} l1=${metrics?.l1 ?? "n/a"} gaussians=${metrics?.gaussians ?? "n/a"}`
    );
    store.markSucceeded(job.id, metrics ?? {});
    lastPreviewStepByJob.delete(job.id);
  }

  async function processUploadedJob(job) {
    try {
      await normalizeUploadedJob(job);
    } catch (error) {
      if (error instanceof CancelledJobError) {
        logger.warn("job cancelled during preprocessing", { jobId: job.id });
        store.markCancelled(job.id);
        lastPreviewStepByJob.delete(job.id);
        return;
      }
      logger.error("job preprocessing failed", {
        jobId: job.id,
        error: error.message
      });
      store.markFailed(job.id, error.message);
      lastPreviewStepByJob.delete(job.id);
    }
  }

  async function processQueuedJob(job) {
    try {
      await runTrainingJob(job);
    } catch (error) {
      if (error instanceof CancelledJobError) {
        logger.warn("job cancelled during training", { jobId: job.id });
        store.markCancelled(job.id);
        lastPreviewStepByJob.delete(job.id);
        return;
      }
      logger.error("job training failed", {
        jobId: job.id,
        error: error.message
      });
      store.markFailed(job.id, error.message);
      lastPreviewStepByJob.delete(job.id);
    }
  }

  async function tick() {
    const queuedJob = store.claimNextQueued(process.pid);
    if (queuedJob) {
      logger.info("claimed queued job", {
        jobId: queuedJob.id,
        status: queuedJob.status,
        preset: queuedJob.preset,
        datasetType: queuedJob.datasetType
      });
      await processQueuedJob(queuedJob);
      return;
    }

    const uploadedJob = store.claimNextUploaded(process.pid);
    if (uploadedJob) {
      logger.info("claimed uploaded job", {
        jobId: uploadedJob.id,
        inputKind: uploadedJob.inputKind,
        uploadName: uploadedJob.uploadName,
        preset: uploadedJob.preset
      });
      await processUploadedJob(uploadedJob);
      return;
    }

    await sleep(config.pollIntervalMs);
  }

  async function run() {
    try {
      logger.info("worker loop started", {
        jobsDir: config.jobsDir,
        databasePath: config.databasePath,
        msplatBin: config.msplatBin,
        colmapBin: config.colmapBin,
        colmapFlagStyle: config.colmapFlagStyle,
        pollIntervalMs: config.pollIntervalMs
      });
      while (!shuttingDown) {
        await tick();
      }
      if (activeChild) {
        killProcessGroup(activeChild);
      }
    } finally {
      logger.info("worker loop stopping");
      closeDbIfNeeded();
    }
  }

  async function close() {
    shuttingDown = true;
    logger.info("worker shutdown requested");
    if (activeChild) {
      killProcessGroup(activeChild);
    }
    if (!activeChild) {
      closeDbIfNeeded();
    }
  }

  return {
    config,
    db,
    store,
    run,
    async tickOnce() {
      await tick();
    },
    close
  };
}
