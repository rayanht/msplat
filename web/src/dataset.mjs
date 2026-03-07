import fs from "node:fs/promises";
import path from "node:path";
import { fileExists } from "./utils.mjs";

export const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".JPG"];

async function listFiles(dir) {
  try {
    return await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
}

async function listCandidateRoots(rootDir) {
  const entries = (await listFiles(rootDir)).filter((entry) => entry.name !== "__MACOSX");
  if (entries.length === 1 && entries[0].isDirectory()) {
    return [path.join(rootDir, entries[0].name), rootDir];
  }
  return [rootDir];
}

async function hasFile(filePath) {
  return fileExists(filePath);
}

async function findColmapBinaryModelDir(rootDir) {
  for (const candidate of [rootDir, path.join(rootDir, "sparse", "0")]) {
    if (
      (await hasFile(path.join(candidate, "cameras.bin"))) &&
      (await hasFile(path.join(candidate, "images.bin"))) &&
      ((await hasFile(path.join(candidate, "points3D.bin"))) || (await hasFile(path.join(candidate, "points3D.ply"))))
    ) {
      return candidate;
    }
  }
  return null;
}

async function findColmapTextModelDir(rootDir) {
  for (const candidate of [rootDir, path.join(rootDir, "sparse"), path.join(rootDir, "sparse", "0")]) {
    if (
      (await hasFile(path.join(candidate, "cameras.txt"))) &&
      (await hasFile(path.join(candidate, "images.txt"))) &&
      (await hasFile(path.join(candidate, "points3D.txt")))
    ) {
      return candidate;
    }
  }
  return null;
}

async function collectImages(rootDir, maxDepth = 3, currentDepth = 0) {
  if (currentDepth > maxDepth) return [];
  const entries = await listFiles(rootDir);
  const images = [];
  for (const entry of entries) {
    const entryPath = path.join(rootDir, entry.name);
    if (entry.isFile() && IMAGE_EXTS.includes(path.extname(entry.name))) {
      images.push(entryPath);
      continue;
    }
    if (entry.isDirectory()) {
      images.push(...await collectImages(entryPath, maxDepth, currentDepth + 1));
    }
  }
  return images.sort((left, right) => left.localeCompare(right));
}

async function resolveImageRoot(rootDir) {
  for (const candidate of [path.join(rootDir, "images"), rootDir]) {
    const images = await collectImages(candidate);
    if (images.length > 0) {
      return { imageRoot: candidate, imageFiles: images };
    }
  }
  return { imageRoot: null, imageFiles: [] };
}

async function resolveImagePath(baseDir, rawPath) {
  const candidate = path.isAbsolute(rawPath) ? rawPath : path.join(baseDir, rawPath);
  if (await fileExists(candidate)) return candidate;
  for (const ext of IMAGE_EXTS) {
    if (await fileExists(candidate + ext)) return candidate + ext;
  }
  return null;
}

async function validateColmapBinary(rootDir, modelDir) {
  const { imageFiles } = await resolveImageRoot(rootDir);
  if (imageFiles.length === 0) {
    throw new Error("COLMAP dataset is missing source images");
  }

  return {
    type: "colmap",
    sourceFormat: "colmap_bin",
    datasetRoot: rootDir,
    modelDir,
    imageRoot: path.join(rootDir, "images"),
    imageFiles
  };
}

async function validateColmapText(rootDir, modelDir) {
  const { imageFiles, imageRoot } = await resolveImageRoot(rootDir);
  if (imageFiles.length === 0) {
    throw new Error("COLMAP text export is missing source images");
  }

  return {
    type: "colmap",
    sourceFormat: "colmap_txt",
    datasetRoot: rootDir,
    modelDir,
    imageRoot,
    imageFiles
  };
}

async function validateNerfstudio(rootDir) {
  const transformsPath = path.join(rootDir, "transforms.json");
  const content = JSON.parse(await fs.readFile(transformsPath, "utf8"));
  const frames = Array.isArray(content.frames) ? content.frames : [];

  if (frames.length === 0) {
    throw new Error("Nerfstudio dataset has no frames");
  }

  for (const frame of frames) {
    if (!frame.file_path) {
      throw new Error("Nerfstudio dataset contains a frame without file_path");
    }
    const resolved = await resolveImagePath(rootDir, frame.file_path);
    if (!resolved) {
      throw new Error(`Nerfstudio image not found: ${frame.file_path}`);
    }
  }

  let pointCloudPath = null;
  if (content.ply_file_path) {
    pointCloudPath = path.isAbsolute(content.ply_file_path)
      ? content.ply_file_path
      : path.join(rootDir, content.ply_file_path);
  } else if (await fileExists(path.join(rootDir, "sparse", "0", "points3D.ply"))) {
    pointCloudPath = path.join(rootDir, "sparse", "0", "points3D.ply");
  } else if (await fileExists(path.join(rootDir, "points3D.ply"))) {
    pointCloudPath = path.join(rootDir, "points3D.ply");
  }

  if (!pointCloudPath || !(await fileExists(pointCloudPath))) {
    throw new Error("Nerfstudio dataset is missing a seed point cloud");
  }

  return {
    type: "nerfstudio",
    sourceFormat: "nerfstudio",
    datasetRoot: rootDir,
    pointCloudPath
  };
}

async function validatePolycam(rootDir) {
  const correctedCamerasDir = path.join(rootDir, "keyframes", "corrected_cameras");
  const correctedImagesDir = path.join(rootDir, "keyframes", "corrected_images");
  const pointCloudCandidates = [
    path.join(rootDir, "keyframes", "point_cloud.ply"),
    path.join(rootDir, "point_cloud.ply"),
    path.join(rootDir, "sparse.ply")
  ];

  let pointCloudFound = false;
  for (const candidate of pointCloudCandidates) {
    if (await fileExists(candidate)) {
      pointCloudFound = true;
      break;
    }
  }

  if (!pointCloudFound) {
    throw new Error("Polycam dataset is missing point_cloud.ply or sparse.ply");
  }

  if (await fileExists(correctedCamerasDir)) {
    const files = (await fs.readdir(correctedCamerasDir)).filter((name) => name.endsWith(".json")).sort();
    if (files.length === 0) {
      throw new Error("Polycam dataset has no corrected camera JSON files");
    }
    for (const fileName of files) {
      const stem = path.basename(fileName, ".json");
      const matched = await Promise.any(
        IMAGE_EXTS.map(async (ext) => {
          const candidate = path.join(correctedImagesDir, `${stem}${ext}`);
          if (await fileExists(candidate)) return candidate;
          return Promise.reject();
        })
      ).catch(() => null);
      if (!matched) {
        throw new Error(`Polycam image not found for ${fileName}`);
      }
    }

    return {
      type: "polycam",
      sourceFormat: "polycam",
      datasetRoot: rootDir
    };
  }

  const camerasJsonPath = path.join(rootDir, "cameras.json");
  if (!(await fileExists(camerasJsonPath))) {
    throw new Error("Polycam dataset is missing cameras.json");
  }

  const content = JSON.parse(await fs.readFile(camerasJsonPath, "utf8"));
  const frames = Array.isArray(content.frames) ? content.frames : Array.isArray(content) ? content : [];

  if (frames.length === 0) {
    throw new Error("Polycam cameras.json contains no frames");
  }

  for (const frame of frames) {
    if (!frame.file_path) {
      throw new Error("Polycam cameras.json contains a frame without file_path");
    }
    const resolved = await resolveImagePath(rootDir, frame.file_path);
    if (!resolved) {
      throw new Error(`Polycam image not found: ${frame.file_path}`);
    }
  }

  return {
    type: "polycam",
    sourceFormat: "polycam",
    datasetRoot: rootDir
  };
}

async function validateRawImages(rootDir) {
  const { imageFiles, imageRoot } = await resolveImageRoot(rootDir);
  if (imageFiles.length < 3) {
    throw new Error("Raw photo uploads need at least 3 images");
  }

  return {
    type: "colmap",
    sourceFormat: "raw_images",
    datasetRoot: rootDir,
    imageRoot,
    imageFiles
  };
}

export async function inspectInput(rootDir) {
  const candidates = await listCandidateRoots(rootDir);

  for (const candidate of candidates) {
    if (await fileExists(path.join(candidate, "transforms.json"))) {
      return validateNerfstudio(candidate);
    }

    const colmapBinaryDir = await findColmapBinaryModelDir(candidate);
    if (colmapBinaryDir) {
      return validateColmapBinary(candidate, colmapBinaryDir);
    }

    const colmapTextDir = await findColmapTextModelDir(candidate);
    if (colmapTextDir) {
      return validateColmapText(candidate, colmapTextDir);
    }

    if (
      (await fileExists(path.join(candidate, "keyframes", "corrected_cameras"))) ||
      (await fileExists(path.join(candidate, "cameras.json")))
    ) {
      return validatePolycam(candidate);
    }
  }

  for (const candidate of candidates) {
    const rawImages = await collectImages(candidate);
    if (rawImages.length > 0) {
      return validateRawImages(candidate);
    }
  }

  throw new Error("Dataset is not a supported COLMAP, Nerfstudio, or Polycam export");
}

export async function inspectDataset(rootDir) {
  return inspectInput(rootDir);
}
