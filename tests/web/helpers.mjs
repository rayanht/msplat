import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createAppServer } from "../../web/src/server-app.mjs";
import { createWorker } from "../../web/src/worker-app.mjs";

const execFileAsync = promisify(execFile);

async function chmodExec(filePath) {
  if (!filePath) return;
  await fs.chmod(filePath, 0o755);
}

export async function makeTempDir(prefix = "msplat-web-") {
  return fs.mkdtemp(path.join(os.tmpdir(), prefix));
}

export async function writeFile(filePath, content) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content);
}

export async function createZipFromDirectory(sourceDir, archivePath) {
  await execFileAsync("/usr/bin/zip", ["-qry", archivePath, "."], { cwd: sourceDir });
}

export async function createUnsafeZipWithTraversal(archivePath) {
  await execFileAsync("/usr/bin/python3", [
    "-c",
    "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1],'w'); z.writestr('../escape.txt','x'); z.close()",
    archivePath
  ]);
}

export async function createZipWithSymlink(sourceDir, archivePath) {
  await execFileAsync("/bin/ln", ["-sf", "/tmp/escape", path.join(sourceDir, "linked.txt")]);
  await execFileAsync("/usr/bin/zip", ["-qry", "-y", archivePath, "."], { cwd: sourceDir });
}

export async function createColmapDataset(dir, { modeFile } = {}) {
  await writeFile(path.join(dir, "images", "frame.jpg"), "jpg");
  await writeFile(path.join(dir, "sparse", "0", "cameras.bin"), "bin");
  await writeFile(path.join(dir, "sparse", "0", "images.bin"), "bin");
  await writeFile(path.join(dir, "sparse", "0", "points3D.bin"), "bin");
  if (modeFile) {
    await writeFile(path.join(dir, modeFile), modeFile);
  }
}

export async function createColmapTextDataset(dir, { layout = "sparse", includeDatabase = true, modeFile } = {}) {
  const modelDir = {
    root: dir,
    sparse: path.join(dir, "sparse"),
    sparse0: path.join(dir, "sparse", "0")
  }[layout];

  if (!modelDir) {
    throw new Error(`Unsupported COLMAP text layout: ${layout}`);
  }

  await writeFile(path.join(dir, "images", "frame.jpg"), "jpg");
  await writeFile(path.join(modelDir, "cameras.txt"), "# cameras");
  await writeFile(path.join(modelDir, "images.txt"), "# images");
  await writeFile(path.join(modelDir, "points3D.txt"), "# points");

  if (includeDatabase) {
    await writeFile(path.join(dir, "database.db"), "db");
  }

  if (modeFile) {
    await writeFile(path.join(dir, modeFile), modeFile);
  }
}

export async function createRawImagesDataset(dir, { fileNames = ["frame-1.jpg", "frame-2.jpg", "frame-3.jpg"] } = {}) {
  for (const fileName of fileNames) {
    await writeFile(path.join(dir, fileName), "raw-image");
  }
}

export async function createNerfstudioDataset(dir) {
  await writeFile(path.join(dir, "images", "frame.png"), "png");
  await writeFile(path.join(dir, "points3D.ply"), "ply");
  await writeFile(
    path.join(dir, "transforms.json"),
    JSON.stringify({
      frames: [{ file_path: "images/frame.png", transform_matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] }]
    })
  );
}

export async function createPolycamDataset(dir) {
  await writeFile(
    path.join(dir, "keyframes", "corrected_cameras", "0001.json"),
    JSON.stringify({ width: 1, height: 1, fx: 1, fy: 1, cx: 0.5, cy: 0.5 })
  );
  await writeFile(path.join(dir, "keyframes", "corrected_images", "0001.png"), "png");
  await writeFile(path.join(dir, "keyframes", "point_cloud.ply"), "ply");
}

async function createMultipartBody({ fields = {}, files = [] }) {
  const formData = new FormData();

  for (const [key, value] of Object.entries(fields)) {
    if (value != null) {
      formData.append(key, String(value));
    }
  }

  for (const file of files) {
    formData.append(
      file.fieldName ?? "images",
      new Blob([file.content ?? "raw-image"], { type: file.contentType ?? "image/jpeg" }),
      file.name
    );
  }

  const request = new Request("http://local/upload", {
    method: "POST",
    body: formData
  });

  return {
    body: Buffer.from(await request.arrayBuffer()),
    headers: Object.fromEntries(request.headers.entries())
  };
}

export async function startHarness({ msplatBin, colmapBin, colmapFlagStyle } = {}) {
  await chmodExec(msplatBin);
  await chmodExec(colmapBin);

  const tempRoot = await makeTempDir();
  const jobsDir = path.join(tempRoot, "jobs");
  const databasePath = path.join(tempRoot, "jobs.sqlite");
  const app = await createAppServer({
    port: 0,
    host: "127.0.0.1",
    jobsDir,
    databasePath
  });
  const worker = await createWorker({
    jobsDir,
    databasePath,
    msplatBin,
    colmapBin,
    colmapFlagStyle,
    pollIntervalMs: 20
  });

  return {
    tempRoot,
    app,
    worker,
    async close() {
      await worker.close();
      await app.close();
    }
  };
}

export async function waitFor(predicate, { timeoutMs = 8000, intervalMs = 50 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = await predicate();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error("Timed out waiting for condition");
}

export async function driveWorkerUntil(worker, checkDone, { timeoutMs = 8000 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    await worker.tickOnce();
    const result = await checkDone();
    if (result) return result;
  }
  throw new Error("Worker did not reach the expected state");
}

export async function requestJson(app, url, { method = "GET", headers = {}, body = null } = {}) {
  const response = await app.inject({ method, url, headers, body });
  return {
    response,
    payload: response.text ? response.json() : null
  };
}

export async function uploadJob(app, archivePath, { name = "Test Job", preset = "preview", colmapMode } = {}) {
  const body = await fs.readFile(archivePath);
  const params = new URLSearchParams({ name, preset });
  if (colmapMode) {
    params.set("colmapMode", colmapMode);
  }

  const response = await app.inject({
    url: `/api/jobs?${params.toString()}`,
    method: "POST",
    headers: {
      "content-type": "application/zip",
      "x-file-name": path.basename(archivePath)
    },
    body
  });
  return {
    response,
    payload: response.json()
  };
}

export async function uploadRawJob(app, { name = "Raw Job", preset = "preview", colmapMode = "sequential", files = [] } = {}) {
  const { body, headers } = await createMultipartBody({
    fields: { name, preset, colmapMode },
    files
  });

  const response = await app.inject({
    url: "/api/jobs/raw",
    method: "POST",
    headers,
    body
  });

  return {
    response,
    payload: response.json()
  };
}
