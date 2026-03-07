import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { assertArchiveSafe, extractArchive } from "../../web/src/archive.mjs";
import { inspectDataset } from "../../web/src/dataset.mjs";
import {
  createColmapDataset,
  createColmapTextDataset,
  createNerfstudioDataset,
  createPolycamDataset,
  createRawImagesDataset,
  createUnsafeZipWithTraversal,
  createZipFromDirectory,
  createZipWithSymlink,
  makeTempDir
} from "./helpers.mjs";

test("detects supported dataset layouts including COLMAP TXT and raw image sets", async () => {
  const root = await makeTempDir();

  const colmapBinDir = path.join(root, "colmap-bin");
  await createColmapDataset(colmapBinDir);
  assert.equal((await inspectDataset(colmapBinDir)).sourceFormat, "colmap_bin");

  const colmapTextSparseDir = path.join(root, "colmap-text-sparse");
  await createColmapTextDataset(colmapTextSparseDir, { layout: "sparse" });
  assert.equal((await inspectDataset(colmapTextSparseDir)).sourceFormat, "colmap_txt");

  const colmapTextSparse0Dir = path.join(root, "colmap-text-sparse0");
  await createColmapTextDataset(colmapTextSparse0Dir, { layout: "sparse0" });
  assert.equal((await inspectDataset(colmapTextSparse0Dir)).sourceFormat, "colmap_txt");

  const colmapTextRootDir = path.join(root, "colmap-text-root");
  await createColmapTextDataset(colmapTextRootDir, { layout: "root" });
  assert.equal((await inspectDataset(colmapTextRootDir)).sourceFormat, "colmap_txt");

  const nerfstudioDir = path.join(root, "nerfstudio");
  await createNerfstudioDataset(nerfstudioDir);
  assert.equal((await inspectDataset(nerfstudioDir)).sourceFormat, "nerfstudio");

  const polycamDir = path.join(root, "polycam");
  await createPolycamDataset(polycamDir);
  assert.equal((await inspectDataset(polycamDir)).sourceFormat, "polycam");

  const rawImagesDir = path.join(root, "raw-images");
  await createRawImagesDataset(rawImagesDir);
  const rawInfo = await inspectDataset(rawImagesDir);
  assert.equal(rawInfo.sourceFormat, "raw_images");
  assert.equal(rawInfo.imageFiles.length, 3);
});

test("rejects incomplete raw photo sets", async () => {
  const root = await makeTempDir();
  const rawImagesDir = path.join(root, "too-few-images");
  await createRawImagesDataset(rawImagesDir, { fileNames: ["frame-1.jpg", "frame-2.jpg"] });

  await assert.rejects(() => inspectDataset(rawImagesDir), /at least 3 images/);
});

test("rejects archives with traversal paths", async () => {
  const root = await makeTempDir();
  const archivePath = path.join(root, "unsafe.zip");
  await createUnsafeZipWithTraversal(archivePath);
  await assert.rejects(() => assertArchiveSafe(archivePath), /escapes the extraction root/);
});

test("rejects archives with symlinks", async () => {
  const root = await makeTempDir();
  const sourceDir = path.join(root, "source");
  await fs.mkdir(sourceDir, { recursive: true });
  const archivePath = path.join(root, "symlink.zip");
  await createZipWithSymlink(sourceDir, archivePath);
  await assert.rejects(() => assertArchiveSafe(archivePath), /symbolic link/);
});

test("extracts a safe archive and preserves dataset detection", async () => {
  const root = await makeTempDir();
  const sourceDir = path.join(root, "dataset");
  await createColmapTextDataset(sourceDir, { layout: "sparse" });
  const archivePath = path.join(root, "dataset.zip");
  await createZipFromDirectory(sourceDir, archivePath);

  await assert.doesNotReject(() => assertArchiveSafe(archivePath));

  const extractedDir = path.join(root, "unzipped");
  await extractArchive(archivePath, extractedDir);
  const detected = await inspectDataset(extractedDir);
  assert.equal(detected.sourceFormat, "colmap_txt");
});
