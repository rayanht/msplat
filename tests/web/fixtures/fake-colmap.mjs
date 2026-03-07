#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";

const [command, ...args] = process.argv.slice(2);
const flagStyle = process.env.FAKE_COLMAP_FLAG_STYLE || "modern";

function readOption(name, fallback = "") {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
}

function expectedGpuFlagNames() {
  if (flagStyle === "legacy") {
    return {
      featureExtraction: "--SiftExtraction.use_gpu",
      featureMatching: "--SiftMatching.use_gpu"
    };
  }
  if (flagStyle === "modern") {
    return {
      featureExtraction: "--FeatureExtraction.use_gpu",
      featureMatching: "--FeatureMatching.use_gpu"
    };
  }
  throw new Error(`Unsupported FAKE_COLMAP_FLAG_STYLE: ${flagStyle}`);
}

function assertExpectedFlag(expectedName, unexpectedName) {
  const expectedIndex = args.indexOf(expectedName);
  if (expectedIndex < 0 || args[expectedIndex + 1] !== "0") {
    console.error(`expected ${expectedName} 0`);
    process.exit(10);
  }
  const unexpectedIndex = args.indexOf(unexpectedName);
  if (unexpectedIndex >= 0) {
    console.error(`unexpected ${unexpectedName}`);
    process.exit(11);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let cancelled = false;
process.on("SIGTERM", () => {
  cancelled = true;
  console.error("cancelled");
});

async function maybeSlow(tokens) {
  if (!tokens.has("slow")) return;
  for (let index = 0; index < 30; index += 1) {
    if (cancelled) {
      process.exit(143);
    }
    await sleep(100);
  }
}

async function ensureModelFiles(modelDir, label) {
  await fs.mkdir(modelDir, { recursive: true });
  await fs.writeFile(path.join(modelDir, "cameras.bin"), `cameras-${label}`);
  await fs.writeFile(path.join(modelDir, "images.bin"), `images-${label}`);
  await fs.writeFile(path.join(modelDir, "points3D.bin"), `points-${label}`);
  await fs.writeFile(path.join(modelDir, `selected-model-${label}.txt`), label);
}

async function loadImageTokens(imageDir) {
  const entries = await fs.readdir(imageDir).catch(() => []);
  const tokens = new Set();
  for (const entry of entries) {
    const lower = entry.toLowerCase();
    if (lower.includes("slow")) tokens.add("slow");
    if (lower.includes("multi")) tokens.add("multi");
    if (lower.includes("tie")) tokens.add("tie");
    if (lower.includes("fallback")) tokens.add("fallback");
    if (lower.includes("extractfail")) tokens.add("extractfail");
    if (lower.includes("matchfail")) tokens.add("matchfail");
    if (lower.includes("mapfail")) tokens.add("mapfail");
  }
  return tokens;
}

async function readDbState(databasePath) {
  try {
    return JSON.parse(await fs.readFile(databasePath, "utf8"));
  } catch {
    return { tokens: [] };
  }
}

async function writeDbState(databasePath, state) {
  await fs.mkdir(path.dirname(databasePath), { recursive: true });
  await fs.writeFile(databasePath, JSON.stringify(state, null, 2));
}

async function handleModelConverter() {
  const inputPath = readOption("--input_path");
  const outputPath = readOption("--output_path");
  const failMarker = path.join(path.dirname(inputPath), "mode-convert-fail.txt");
  if (await fs.access(failMarker).then(() => true).catch(() => false)) {
    console.error("simulated model conversion failure");
    process.exit(2);
  }

  await fs.mkdir(outputPath, { recursive: true });
  await ensureModelFiles(outputPath, "converted");
  console.log(`Converted ${inputPath} -> ${outputPath}`);
}

async function handleFeatureExtractor() {
  const databasePath = readOption("--database_path");
  const imagePath = readOption("--image_path");
  const flags = expectedGpuFlagNames();
  assertExpectedFlag(
    flags.featureExtraction,
    flags.featureExtraction === "--FeatureExtraction.use_gpu" ? "--SiftExtraction.use_gpu" : "--FeatureExtraction.use_gpu"
  );
  const tokens = await loadImageTokens(imagePath);
  if (tokens.has("extractfail")) {
    console.error("simulated feature extraction failure");
    process.exit(3);
  }

  await maybeSlow(tokens);
  if (cancelled) process.exit(143);

  await writeDbState(databasePath, {
    imagePath,
    tokens: [...tokens]
  });
  console.log(`feature_extractor image_path=${imagePath}`);
}

async function handleMatcher() {
  const databasePath = readOption("--database_path");
  const flags = expectedGpuFlagNames();
  assertExpectedFlag(
    flags.featureMatching,
    flags.featureMatching === "--FeatureMatching.use_gpu" ? "--SiftMatching.use_gpu" : "--FeatureMatching.use_gpu"
  );
  const state = await readDbState(databasePath);
  const tokens = new Set(state.tokens || []);
  if (tokens.has("matchfail")) {
    console.error("simulated matcher failure");
    process.exit(4);
  }

  await maybeSlow(tokens);
  if (cancelled) process.exit(143);

  state.matcher = command;
  state.matcherArgs = args;
  await writeDbState(databasePath, state);
  console.log(command);
}

async function handleMapper() {
  const databasePath = readOption("--database_path");
  const outputPath = readOption("--output_path");
  const state = await readDbState(databasePath);
  const tokens = new Set(state.tokens || []);
  if (tokens.has("mapfail")) {
    console.error("simulated mapper failure");
    process.exit(5);
  }
  if (tokens.has("fallback") && state.matcher !== "exhaustive_matcher") {
    console.error("simulated no initial pair found");
    process.exit(6);
  }

  await maybeSlow(tokens);
  if (cancelled) process.exit(143);

  await fs.mkdir(outputPath, { recursive: true });

  if (tokens.has("multi") || tokens.has("tie")) {
    const models = tokens.has("tie")
      ? [
          { name: "0", registeredImages: 4, points: 100, label: "0" },
          { name: "1", registeredImages: 4, points: 250, label: "1" }
        ]
      : [
          { name: "0", registeredImages: 3, points: 90, label: "0" },
          { name: "1", registeredImages: 6, points: 180, label: "1" }
        ];

    for (const model of models) {
      const modelDir = path.join(outputPath, model.name);
      await ensureModelFiles(modelDir, model.label);
      await fs.writeFile(path.join(modelDir, "stats.json"), JSON.stringify(model));
    }
  } else {
    const modelDir = path.join(outputPath, "0");
    await ensureModelFiles(modelDir, "0");
    await fs.writeFile(path.join(modelDir, "stats.json"), JSON.stringify({
      registeredImages: 5,
      points: 120,
      label: "0"
    }));
  }

  console.log(`mapper output_path=${outputPath}`);
}

async function handleModelAnalyzer() {
  const modelPath = readOption("--path");
  const stats = JSON.parse(await fs.readFile(path.join(modelPath, "stats.json"), "utf8").catch(() => "{}"));
  console.log(`Registered images: ${stats.registeredImages ?? 0}`);
  console.log(`Points: ${stats.points ?? 0}`);
}

try {
  if (command === "model_converter") {
    await handleModelConverter();
  } else if (command === "feature_extractor") {
    await handleFeatureExtractor();
  } else if (command === "sequential_matcher" || command === "exhaustive_matcher") {
    await handleMatcher();
  } else if (command === "mapper") {
    await handleMapper();
  } else if (command === "model_analyzer") {
    await handleModelAnalyzer();
  } else {
    console.error(`unsupported command: ${command}`);
    process.exit(64);
  }
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
