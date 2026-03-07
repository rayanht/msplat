import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { createAppServer } from "../../web/src/server-app.mjs";
import { createWorker } from "../../web/src/worker-app.mjs";
import { driveWorkerUntil, makeTempDir, requestJson, uploadJob } from "./helpers.mjs";

const smokeDataset = process.env.MSPLAT_SMOKE_DATASET;
const smokeBinary = process.env.MSPLAT_SMOKE_BIN;
const smokeColmapBin = process.env.MSPLAT_SMOKE_COLMAP_BIN;

test("optional Apple Silicon smoke run", { skip: !smokeDataset || !smokeBinary }, async () => {
  const tempRoot = await makeTempDir("msplat-smoke-");
  const jobsDir = path.join(tempRoot, "jobs");
  const databasePath = path.join(tempRoot, "jobs.sqlite");
  const app = await createAppServer({ port: 0, host: "127.0.0.1", jobsDir, databasePath });
  const worker = await createWorker({
    jobsDir,
    databasePath,
    msplatBin: smokeBinary,
    colmapBin: smokeColmapBin,
    pollIntervalMs: 50
  });

  try {
    const { payload } = await uploadJob(app, smokeDataset, { name: "Smoke", preset: "preview" });
    const finalDetail = await driveWorkerUntil(
      worker,
      async () => {
        const { payload: detail } = await requestJson(app, `/api/jobs/${payload.job.id}`);
        return ["succeeded", "failed"].includes(detail.job.status) ? detail : null;
      },
      { timeoutMs: 180000 }
    );

    assert.equal(finalDetail.job.status, "succeeded");
  } finally {
    await worker.close();
    await app.close();
  }
});
