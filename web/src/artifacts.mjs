import fs from "node:fs/promises";
import path from "node:path";
import { fileExists, statIfExists } from "./utils.mjs";

async function listPreviewFiles(previewsDir) {
  try {
    const files = await fs.readdir(previewsDir);
    return files
      .filter((fileName) => fileName.endsWith(".png"))
      .sort((left, right) => Number(path.basename(left, ".png")) - Number(path.basename(right, ".png")));
  } catch {
    return [];
  }
}

async function addArtifact(artifacts, name, filePath, kind, label = name) {
  if (!(await fileExists(filePath))) return;
  const stat = await statIfExists(filePath);
  artifacts.push({
    name,
    label,
    path: filePath,
    sizeBytes: stat?.size ?? 0,
    kind
  });
}

export async function getLatestPreview(previewsDir) {
  const files = await listPreviewFiles(previewsDir);
  if (files.length === 0) return null;
  const fileName = files.at(-1);
  return {
    fileName,
    step: Number(path.basename(fileName, ".png")),
    path: path.join(previewsDir, fileName)
  };
}

export async function buildArtifactList(job) {
  const artifacts = [];
  const outputDir = path.dirname(job.outputPath);
  const primaryOutputName = path.basename(job.outputPath);

  await addArtifact(artifacts, primaryOutputName, job.outputPath, "output");
  await addArtifact(artifacts, "final.ply", path.join(outputDir, "final.ply"), "output");
  await addArtifact(artifacts, "final.spz", path.join(outputDir, "final.spz"), "output");
  await addArtifact(artifacts, "cameras.json", job.camerasPath, "metadata");
  await addArtifact(artifacts, "train.log", job.logPath, "log");

  if (job.colmapDatabasePath) {
    await addArtifact(artifacts, "colmap-database.db", job.colmapDatabasePath, "colmap", "colmap/database.db");
  }

  if (job.colmapModelPath && (await fileExists(job.colmapModelPath))) {
    const entries = await fs.readdir(job.colmapModelPath, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const fileName = entry.name;
      await addArtifact(
        artifacts,
        `colmap-model-${fileName}`,
        path.join(job.colmapModelPath, fileName),
        "colmap",
        `colmap/model/${fileName}`
      );
    }
  }

  const previewFiles = await listPreviewFiles(job.previewsDir);
  for (const fileName of previewFiles.slice(-8).reverse()) {
    await addArtifact(
      artifacts,
      fileName,
      path.join(job.previewsDir, fileName),
      "preview",
      `preview/${fileName}`
    );
  }

  return artifacts;
}

export async function resolveArtifact(job, name) {
  if (name === "latest-preview") {
    const latest = await getLatestPreview(job.previewsDir);
    return latest ? { name: latest.fileName, label: `preview/${latest.fileName}`, path: latest.path, kind: "preview" } : null;
  }

  const artifacts = await buildArtifactList(job);
  return artifacts.find((artifact) => artifact.name === name) ?? null;
}
