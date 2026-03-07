import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { ensureDir, fileExists, toPosixPath } from "./utils.mjs";

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });

    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(stderr.trim() || `${command} exited with code ${code}`));
    });
  });
}

function validateEntryName(entryName) {
  if (!entryName || entryName.includes("\0")) {
    throw new Error("Archive contains an invalid path");
  }

  const normalized = toPosixPath(entryName).replace(/^\.\/+/, "");
  const segments = normalized.split("/").filter(Boolean);

  if (normalized.startsWith("/") || /^[a-zA-Z]:\//.test(normalized)) {
    throw new Error(`Archive path is not relative: ${entryName}`);
  }

  if (segments.some((segment) => segment === "..")) {
    throw new Error(`Archive path escapes the extraction root: ${entryName}`);
  }
}

async function assertNoSymlinks(rootDir) {
  const stack = [rootDir];
  while (stack.length > 0) {
    const currentDir = stack.pop();
    const entries = await fs.readdir(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(currentDir, entry.name);
      const stat = await fs.lstat(entryPath);
      if (stat.isSymbolicLink()) {
        throw new Error(`Archive contains a symbolic link: ${entryPath}`);
      }
      if (entry.isDirectory()) {
        stack.push(entryPath);
      }
    }
  }
}

export async function listArchiveEntries(archivePath) {
  const { stdout } = await runCommand("/usr/bin/zipinfo", ["-1", archivePath]);
  return stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((entry) => {
      validateEntryName(entry);
      return entry;
    });
}

export async function assertArchiveSafe(archivePath) {
  const entries = await listArchiveEntries(archivePath);
  const { stdout } = await runCommand("/usr/bin/zipinfo", ["-l", archivePath]);
  const symlinkLine = stdout
    .split(/\r?\n/)
    .find((line) => line.trim().startsWith("l"));

  if (symlinkLine) {
    throw new Error("Archive contains a symbolic link entry");
  }

  if (entries.length === 0) {
    throw new Error("Archive is empty");
  }

  return entries;
}

export async function extractArchive(archivePath, destinationDir) {
  await ensureDir(destinationDir);
  await runCommand("/usr/bin/unzip", ["-qq", archivePath, "-d", destinationDir]);
  await assertNoSymlinks(destinationDir);
}

export async function detectDatasetRoot(extractionDir) {
  const visibleEntries = (await fs.readdir(extractionDir, { withFileTypes: true }))
    .filter((entry) => entry.name !== "__MACOSX");

  const candidates = [extractionDir];
  if (visibleEntries.length === 1 && visibleEntries[0].isDirectory()) {
    candidates.push(path.join(extractionDir, visibleEntries[0].name));
  }

  for (const candidate of candidates) {
    if (await fileExists(path.join(candidate, "transforms.json"))) return candidate;
    if (
      await fileExists(path.join(candidate, "cameras.bin")) ||
      await fileExists(path.join(candidate, "sparse", "0", "cameras.bin")) ||
      await fileExists(path.join(candidate, "keyframes", "corrected_cameras")) ||
      await fileExists(path.join(candidate, "cameras.json"))
    ) {
      return candidate;
    }
  }

  return extractionDir;
}
