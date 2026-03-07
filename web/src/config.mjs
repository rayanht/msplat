import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..");
const stateRoot = path.resolve(projectRoot, ".msplat-web");
const VALID_COLMAP_FLAG_STYLES = new Set(["modern", "legacy"]);

function resolveMaybeRelative(value, fallback) {
  const raw = value || fallback;
  return path.isAbsolute(raw) ? raw : path.resolve(projectRoot, raw);
}

function resolveCommand(value, fallback) {
  const raw = value || fallback;
  if (!raw.includes("/") && !raw.startsWith(".")) {
    return raw;
  }
  return path.isAbsolute(raw) ? raw : path.resolve(projectRoot, raw);
}

export function resolveColmapFlagStyle(value) {
  const style = value || "modern";
  if (!VALID_COLMAP_FLAG_STYLES.has(style)) {
    throw new Error(`COLMAP_FLAG_STYLE must be one of: ${[...VALID_COLMAP_FLAG_STYLES].join(", ")}`);
  }
  return style;
}

export function loadConfig(env = process.env) {
  const jobsDir = resolveMaybeRelative(env.MSPLAT_JOBS_DIR, path.join(stateRoot, "jobs"));
  const databasePath = resolveMaybeRelative(env.DATABASE_URL, path.join(stateRoot, "jobs.sqlite"));
  const maxUploadGb = Number(env.MAX_UPLOAD_GB || "10");

  if (!Number.isFinite(maxUploadGb) || maxUploadGb <= 0) {
    throw new Error("MAX_UPLOAD_GB must be a positive number");
  }

  return {
    host: env.HOST || "127.0.0.1",
    port: Number(env.PORT || "4321"),
    projectRoot,
    jobsDir,
    databasePath,
    maxUploadBytes: Math.floor(maxUploadGb * 1024 * 1024 * 1024),
    msplatBin: resolveCommand(env.MSPLAT_BIN, path.join(projectRoot, "build", "msplat")),
    colmapBin: resolveCommand(env.COLMAP_BIN, "colmap"),
    colmapFlagStyle: resolveColmapFlagStyle(env.COLMAP_FLAG_STYLE),
    pollIntervalMs: Number(env.MSPLAT_POLL_MS || "2000"),
    tailBytes: Number(env.MSPLAT_LOG_TAIL_BYTES || "24000"),
    title: env.MSPLAT_SITE_TITLE || "msplat Internal Trainer",
    machineName: env.MSPLAT_MACHINE_NAME || os.hostname()
  };
}
