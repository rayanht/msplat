#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";

const args = process.argv.slice(2);
const inputDir = args[0];

function readOption(name, fallback = "") {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const outputPath = readOption("-o");
const exportPlyPath = readOption("--export-ply");
const valRenderDir = readOption("--val-render");
const pngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5X7ioAAAAASUVORK5CYII=";

async function writePreview(step) {
  await fs.mkdir(valRenderDir, { recursive: true });
  await fs.writeFile(path.join(valRenderDir, `${step}.png`), Buffer.from(pngBase64, "base64"));
  console.log(`preview ${step}`);
}

let cancelled = false;
process.on("SIGTERM", () => {
  cancelled = true;
  console.error("cancelled");
});

const mode = await (async () => {
  for (const name of ["mode-fail.txt", "mode-slow.txt"]) {
    try {
      await fs.access(path.join(inputDir, name));
      return name.includes("fail") ? "fail" : "slow";
    } catch {
      continue;
    }
  }
  return "success";
})();

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await writePreview(10);

if (mode !== "fail") {
  await writePreview(20);
}

if (mode === "slow") {
  for (let index = 0; index < 20; index += 1) {
    if (cancelled) process.exit(143);
    await sleep(200);
  }
}

if (cancelled) {
  process.exit(143);
}

if (mode === "fail") {
  console.error("simulated failure");
  process.exit(2);
}

await fs.writeFile(outputPath, "fake splat");
if (exportPlyPath) {
  await fs.mkdir(path.dirname(exportPlyPath), { recursive: true });
  await fs.writeFile(exportPlyPath, "fake ply");
}
await fs.writeFile(path.join(path.dirname(outputPath), "cameras.json"), JSON.stringify([{ file_path: "frame.png" }], null, 2));
console.log("=== Validation (frame.png) ===");
console.log("  PSNR:  28.5  SSIM:  0.912  L1:  0.034  Gaussians:  12345");
