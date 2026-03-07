import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

export function createJobId() {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 16);
}

export function nowIso() {
  return new Date().toISOString();
}

export async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

export async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function statIfExists(filePath) {
  try {
    return await fs.stat(filePath);
  } catch {
    return null;
  }
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = units[0];
  for (let index = 0; index < units.length; index += 1) {
    unit = units[index];
    if (value < 1024 || index === units.length - 1) break;
    value /= 1024;
  }
  const precision = value >= 10 || unit === "B" ? 0 : 1;
  return `${value.toFixed(precision)} ${unit}`;
}

export function formatDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

export function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

export function safeFileName(name) {
  return String(name || "")
    .trim()
    .replace(/[/\\]/g, "-")
    .replace(/\s+/g, " ")
    .slice(0, 120);
}

export function withLeadingSlash(relativePath) {
  return relativePath.startsWith("/") ? relativePath : `/${relativePath}`;
}

export function toPosixPath(filePath) {
  return filePath.split(path.sep).join(path.posix.sep);
}
