import fs from "node:fs";
import fsp from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { Readable, Writable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { Transform } from "node:stream";
import { buildArtifactList, getLatestPreview, resolveArtifact } from "./artifacts.mjs";
import { loadConfig } from "./config.mjs";
import { openDatabase, createJobStore } from "./db.mjs";
import { createLogger } from "./logger.mjs";
import { renderJobDetailPage, renderJobsPage, renderNewJobPage } from "./pages.mjs";
import { getPreset } from "./presets.mjs";
import { readTail } from "./logs.mjs";
import {
  createJobId,
  ensureDir,
  escapeHtml,
  fileExists,
  formatBytes,
  formatNumber,
  safeFileName
} from "./utils.mjs";

const VALID_COLMAP_MODES = new Set(["sequential", "exhaustive"]);
const RAW_IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".JPG"]);
const DELETABLE_JOB_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

function json(response, statusCode, payload) {
  response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function html(response, statusCode, body) {
  response.writeHead(statusCode, { "content-type": "text/html; charset=utf-8" });
  response.end(body);
}

function sendFile(response, filePath, contentType) {
  response.writeHead(200, { "content-type": contentType });
  fs.createReadStream(filePath).pipe(response);
}

function notFound(response, message = "Not found") {
  html(response, 404, `<!doctype html><title>Not found</title><p>${escapeHtml(message)}</p>`);
}

function buildJobPayload(job) {
  return {
    ...job,
    finalPsnrLabel: formatNumber(job.finalPsnr),
    finalSsimLabel: formatNumber(job.finalSsim, 3),
    finalL1Label: formatNumber(job.finalL1, 4),
    finalGaussiansLabel: job.finalGaussians != null ? Number(job.finalGaussians).toLocaleString() : "—"
  };
}

function validatePreset(rawPreset) {
  const preset = getPreset(rawPreset || "");
  if (!preset) {
    throw new Error("Invalid preset");
  }
  return preset;
}

function validateColmapMode(rawMode, fallback = null) {
  if (!rawMode) return fallback;
  if (!VALID_COLMAP_MODES.has(rawMode)) {
    throw new Error("Invalid COLMAP mode");
  }
  return rawMode;
}

function isPathInside(basePath, targetPath) {
  const relative = path.relative(path.resolve(basePath), path.resolve(targetPath));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function buildJobPaths(baseJobsDir, jobId) {
  const jobDir = path.join(baseJobsDir, jobId);
  const outputDir = path.join(jobDir, "output");
  return {
    jobDir,
    outputDir,
    uploadPath: path.join(jobDir, "upload.zip"),
    rawUploadDir: path.join(jobDir, "raw-upload"),
    outputPath: path.join(outputDir, "final.spl"),
    camerasPath: path.join(outputDir, "cameras.json"),
    previewsDir: path.join(jobDir, "previews"),
    logPath: path.join(jobDir, "train.log")
  };
}

async function readRequestBody(request, maxBytes) {
  const chunks = [];
  let bytes = 0;

  for await (const chunk of request) {
    bytes += chunk.length;
    if (bytes > maxBytes) {
      throw new Error(`Upload exceeds ${formatBytes(maxBytes)}`);
    }
    chunks.push(Buffer.from(chunk));
  }

  return Buffer.concat(chunks);
}

async function streamUploadToFile(request, destinationPath, maxBytes) {
  let bytes = 0;
  const limiter = new Transform({
    transform(chunk, _encoding, callback) {
      bytes += chunk.length;
      if (bytes > maxBytes) {
        callback(new Error(`Upload exceeds ${formatBytes(maxBytes)}`));
        return;
      }
      callback(null, chunk);
    }
  });

  await pipeline(request, limiter, fs.createWriteStream(destinationPath));
  return bytes;
}

async function parseMultipartForm(request, maxBytes) {
  const contentType = request.headers["content-type"] || "";
  if (!contentType.includes("multipart/form-data")) {
    throw new Error("Expected multipart/form-data upload");
  }

  const body = await readRequestBody(request, maxBytes);
  const parsed = await new Request("http://local/upload", {
    method: "POST",
    headers: { "content-type": contentType },
    body
  }).formData();

  return parsed;
}

async function saveRawFiles(formData, rawUploadDir) {
  const files = formData.getAll("images");
  if (files.length < 3) {
    throw new Error("Raw photo uploads need at least 3 images");
  }

  await ensureDir(rawUploadDir);

  let uploadSizeBytes = 0;
  const saved = [];

  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    if (typeof file?.arrayBuffer !== "function" || !file.name) {
      continue;
    }

    const extension = path.extname(file.name) || ".jpg";
    if (!RAW_IMAGE_EXTS.has(extension) && !(file.type || "").startsWith("image/")) {
      throw new Error(`Unsupported raw photo file: ${file.name}`);
    }
    const baseName = safeFileName(path.basename(file.name, extension)) || `image-${index + 1}`;
    const destinationName = `${String(index + 1).padStart(6, "0")}_${baseName}${extension}`;
    const destinationPath = path.join(rawUploadDir, destinationName);
    const buffer = Buffer.from(await file.arrayBuffer());
    uploadSizeBytes += buffer.length;
    await fsp.writeFile(destinationPath, buffer);
    saved.push(destinationPath);
  }

  if (saved.length < 3) {
    throw new Error("Raw photo uploads need at least 3 image files");
  }

  return {
    uploadSizeBytes,
    saved
  };
}

export async function createAppServer(overrides = {}) {
  const config = { ...loadConfig(), ...overrides };
  const logger = createLogger("server");
  await ensureDir(config.jobsDir);
  const db = overrides.db ?? await openDatabase(config.databasePath);
  const store = overrides.store ?? createJobStore(db);
  const cssPath = path.join(config.projectRoot, "web", "public", "app.css");

  const requestHandler = async (request, response) => {
    try {
      const url = new URL(request.url, `http://${request.headers.host || "localhost"}`);
      const parts = url.pathname.split("/").filter(Boolean);

      if (request.method === "GET" && url.pathname === "/") {
        response.writeHead(302, { location: "/jobs" });
        response.end();
        return;
      }

      if (request.method === "GET" && url.pathname === "/assets/app.css") {
        sendFile(response, cssPath, "text/css; charset=utf-8");
        return;
      }

      if (request.method === "GET" && url.pathname === "/jobs") {
        html(response, 200, await renderJobsPage(config, store.listJobs()));
        return;
      }

      if (request.method === "GET" && url.pathname === "/jobs/new") {
        html(response, 200, renderNewJobPage(config));
        return;
      }

      if (request.method === "GET" && parts[0] === "jobs" && parts.length === 2) {
        const job = store.getJob(parts[1]);
        if (!job) {
          notFound(response, "Job not found");
          return;
        }
        html(response, 200, await renderJobDetailPage(config, job));
        return;
      }

      if (parts[0] === "api" && parts[1] === "jobs" && request.method === "POST" && parts.length === 2) {
        const preset = validatePreset(url.searchParams.get("preset"));
        const colmapMode = validateColmapMode(url.searchParams.get("colmapMode"), null);
        const rawName = safeFileName(url.searchParams.get("name") || "");
        const uploadName = safeFileName(request.headers["x-file-name"] || "dataset.zip");
        if (!uploadName.toLowerCase().endsWith(".zip")) {
          json(response, 400, { error: "Upload must be a .zip file" });
          return;
        }

        const jobId = createJobId();
        const paths = buildJobPaths(config.jobsDir, jobId);
        const name = rawName || path.basename(uploadName, ".zip");

        await ensureDir(paths.jobDir);
        await ensureDir(paths.outputDir);
        await ensureDir(paths.previewsDir);

        try {
          const uploadSizeBytes = await streamUploadToFile(request, paths.uploadPath, config.maxUploadBytes);
          const job = store.createJob({
            id: jobId,
            name,
            preset: preset.key,
            inputKind: "prepared_zip",
            colmapMode,
            uploadName,
            uploadSizeBytes,
            uploadPath: paths.uploadPath,
            jobDir: paths.jobDir,
            outputPath: paths.outputPath,
            camerasPath: paths.camerasPath,
            logPath: paths.logPath,
            previewsDir: paths.previewsDir
          });
          logger.info("zip upload accepted", {
            jobId,
            name,
            preset: preset.key,
            colmapMode,
            uploadName,
            uploadSizeBytes
          });
          json(response, 201, { job: buildJobPayload(job) });
        } catch (error) {
          await fsp.rm(paths.jobDir, { recursive: true, force: true });
          logger.warn("zip upload rejected", {
            uploadName,
            error: error.message
          });
          json(response, 400, { error: error.message });
        }
        return;
      }

      if (parts[0] === "api" && parts[1] === "jobs" && request.method === "POST" && parts[2] === "raw") {
        const formData = await parseMultipartForm(request, config.maxUploadBytes);
        const preset = validatePreset(formData.get("preset"));
        const colmapMode = validateColmapMode(formData.get("colmapMode"), "sequential");
        const rawName = safeFileName(formData.get("name") || "");

        const jobId = createJobId();
        const paths = buildJobPaths(config.jobsDir, jobId);
        await ensureDir(paths.jobDir);
        await ensureDir(paths.outputDir);
        await ensureDir(paths.previewsDir);
        await ensureDir(paths.rawUploadDir);

        try {
          const { uploadSizeBytes, saved } = await saveRawFiles(formData, paths.rawUploadDir);
          const job = store.createJob({
            id: jobId,
            name: rawName || `Raw upload ${jobId}`,
            preset: preset.key,
            inputKind: "raw_files",
            colmapMode,
            sourceFormat: "raw_images",
            uploadName: `${saved.length} images`,
            uploadSizeBytes,
            uploadPath: paths.rawUploadDir,
            jobDir: paths.jobDir,
            outputPath: paths.outputPath,
            camerasPath: paths.camerasPath,
            logPath: paths.logPath,
            previewsDir: paths.previewsDir
          });
          logger.info("raw upload accepted", {
            jobId,
            name: rawName || `Raw upload ${jobId}`,
            preset: preset.key,
            colmapMode,
            imageCount: saved.length,
            uploadSizeBytes
          });
          json(response, 201, { job: buildJobPayload(job) });
        } catch (error) {
          await fsp.rm(paths.jobDir, { recursive: true, force: true });
          logger.warn("raw upload rejected", {
            error: error.message
          });
          json(response, 400, { error: error.message });
        }
        return;
      }

      if (parts[0] === "api" && parts[1] === "jobs" && request.method === "GET" && parts.length === 2) {
        json(response, 200, { jobs: store.listJobs().map(buildJobPayload) });
        return;
      }

      if (parts[0] === "api" && parts[1] === "jobs" && request.method === "GET" && parts.length === 3) {
        const job = store.getJob(parts[2]);
        if (!job) {
          json(response, 404, { error: "Job not found" });
          return;
        }

        const latestPreview = await getLatestPreview(job.previewsDir);
        const artifacts = await buildArtifactList(job);
        json(response, 200, {
          job: buildJobPayload(job),
          logTail: await readTail(job.logPath, config.tailBytes),
          latestPreviewUrl: latestPreview
            ? `/api/jobs/${job.id}/artifacts/${encodeURIComponent(latestPreview.fileName)}`
            : "",
          artifacts: artifacts.map((artifact) => ({
            name: artifact.name,
            label: artifact.label ?? artifact.name,
            kind: artifact.kind,
            path: artifact.path,
            sizeBytes: artifact.sizeBytes,
            sizeLabel: formatBytes(artifact.sizeBytes),
            url: `/api/jobs/${job.id}/artifacts/${encodeURIComponent(artifact.name)}`
          }))
        });
        return;
      }

      if (parts[0] === "api" && parts[1] === "jobs" && request.method === "POST" && parts.length === 4 && parts[3] === "cancel") {
        const job = store.requestCancel(parts[2]);
        if (!job) {
          json(response, 404, { error: "Job not found" });
          return;
        }
        logger.info("cancel requested", {
          jobId: parts[2],
          status: job.status,
          cancelRequested: job.cancelRequested
        });
        json(response, 200, { job: buildJobPayload(job) });
        return;
      }

      if (parts[0] === "api" && parts[1] === "jobs" && request.method === "POST" && parts.length === 4 && parts[3] === "delete") {
        const job = store.getJob(parts[2]);
        if (!job) {
          json(response, 404, { error: "Job not found" });
          return;
        }
        if (!DELETABLE_JOB_STATUSES.has(job.status)) {
          logger.warn("delete rejected for active job", {
            jobId: job.id,
            status: job.status
          });
          json(response, 409, { error: "Active jobs cannot be deleted. Cancel them first and wait for completion." });
          return;
        }
        if (!job.jobDir || !isPathInside(config.jobsDir, job.jobDir)) {
          logger.error("delete rejected for unmanaged job directory", {
            jobId: job.id,
            jobDir: job.jobDir
          });
          json(response, 500, { error: "Job directory is outside the managed jobs folder" });
          return;
        }

        await fsp.rm(job.jobDir, { recursive: true, force: true });
        store.deleteJob(job.id);
        logger.info("job deleted", {
          jobId: job.id,
          status: job.status,
          jobDir: job.jobDir
        });
        json(response, 200, { deleted: true, jobId: job.id });
        return;
      }

      if (parts[0] === "api" && parts[1] === "jobs" && request.method === "GET" && parts.length === 5 && parts[3] === "artifacts") {
        const job = store.getJob(parts[2]);
        if (!job) {
          json(response, 404, { error: "Job not found" });
          return;
        }
        const artifact = await resolveArtifact(job, decodeURIComponent(parts[4]));
        if (!artifact || !(await fileExists(artifact.path))) {
          json(response, 404, { error: "Artifact not found" });
          return;
        }
        const contentType = artifact.kind === "preview"
          ? "image/png"
          : artifact.kind === "log"
            ? "text/plain; charset=utf-8"
            : artifact.kind === "colmap" && artifact.name.endsWith(".db")
              ? "application/octet-stream"
              : "application/octet-stream";
        sendFile(response, artifact.path, contentType);
        return;
      }

      notFound(response);
    } catch (error) {
      logger.error("request failed", {
        method: request.method,
        url: request.url,
        error: error.message
      });
      json(response, 500, { error: error.message });
    }
  };

  const server = http.createServer(requestHandler);

  class InjectRequest extends Readable {
    constructor({ method, url, headers, body }) {
      super();
      this.method = method;
      this.url = url;
      this.headers = headers;
      this._body = body;
      this._sent = false;
    }

    _read() {
      if (this._sent) {
        this.push(null);
        return;
      }
      this._sent = true;
      if (this._body?.length) {
        this.push(this._body);
      }
      this.push(null);
    }
  }

  class InjectResponse extends Writable {
    constructor() {
      super();
      this.statusCode = 200;
      this.headers = {};
      this.chunks = [];
      this.done = new Promise((resolve, reject) => {
        this.once("finish", resolve);
        this.once("error", reject);
      });
    }

    writeHead(statusCode, headers = {}) {
      this.statusCode = statusCode;
      for (const [key, value] of Object.entries(headers)) {
        this.headers[String(key).toLowerCase()] = value;
      }
      return this;
    }

    setHeader(key, value) {
      this.headers[String(key).toLowerCase()] = value;
    }

    _write(chunk, _encoding, callback) {
      this.chunks.push(Buffer.from(chunk));
      callback();
    }

    end(chunk, encoding, callback) {
      if (chunk) {
        this.write(chunk, encoding);
      }
      return super.end(callback);
    }
  }

  return {
    config,
    db,
    store,
    server,
    async start() {
      await new Promise((resolve) => {
        server.listen(config.port, config.host, resolve);
      });
      logger.info("server listening", {
        host: config.host,
        port: config.port,
        jobsDir: config.jobsDir,
        databasePath: config.databasePath
      });
      return server.address();
    },
    async inject({ method = "GET", url = "/", headers = {}, body = null }) {
      const request = new InjectRequest({
        method,
        url,
        headers: { host: "inject.local", ...headers },
        body: body ? (Buffer.isBuffer(body) ? body : Buffer.from(body)) : null
      });
      const response = new InjectResponse();
      await requestHandler(request, response);
      await response.done;
      const responseBody = Buffer.concat(response.chunks);
      return {
        statusCode: response.statusCode,
        headers: response.headers,
        body: responseBody,
        text: responseBody.toString("utf8"),
        json() {
          return JSON.parse(responseBody.toString("utf8"));
        }
      };
    },
    async close() {
      logger.info("server shutdown requested");
      if (server.listening) {
        await new Promise((resolve, reject) => {
          server.close((error) => (error ? reject(error) : resolve()));
        });
      }
      if (!overrides.db) db.close();
    }
  };
}
