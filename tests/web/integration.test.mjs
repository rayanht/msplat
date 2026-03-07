import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { createWorker } from "../../web/src/worker-app.mjs";
import {
  createColmapDataset,
  createColmapTextDataset,
  createRawImagesDataset,
  createZipFromDirectory,
  driveWorkerUntil,
  makeTempDir,
  requestJson,
  startHarness,
  uploadJob,
  uploadRawJob,
  waitFor
} from "./helpers.mjs";

const fakeMsplatBin = path.resolve("tests/web/fixtures/fake-msplat.mjs");
const fakeColmapBin = path.resolve("tests/web/fixtures/fake-colmap.mjs");

async function waitForTerminalDetail(app, worker, jobId, timeoutMs = 8000) {
  return driveWorkerUntil(
    worker,
    async () => {
      const { payload: detail } = await requestJson(app, `/api/jobs/${jobId}`);
      return ["succeeded", "failed", "cancelled"].includes(detail.job.status) ? detail : null;
    },
    { timeoutMs }
  );
}

test("shows a sample dataset link on the new job page", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const page = await harness.app.inject({ url: "/jobs/new" });
    assert.match(page.text, /COLMAP sample datasets/);
    assert.match(page.text, /https:\/\/demuc\.de\/colmap\/datasets\//);
  } finally {
    await harness.close();
  }
});

test("queues, runs, and exposes artifacts for a prepared COLMAP BIN upload", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const datasetDir = path.join(await makeTempDir(), "dataset");
    await createColmapDataset(datasetDir);
    const archivePath = path.join(await makeTempDir(), "dataset.zip");
    await createZipFromDirectory(datasetDir, archivePath);

    const { response, payload } = await uploadJob(harness.app, archivePath, { name: "Garden pass", preset: "preview" });
    assert.equal(response.statusCode, 201);

    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);
    assert.equal(detail.job.status, "succeeded");
    assert.equal(detail.job.datasetType, "colmap");
    assert.equal(detail.job.sourceFormat, "colmap_bin");
    assert.equal(detail.job.finalPsnr, 28.5);
    assert.ok(detail.latestPreviewUrl.includes(".png"));
    const primaryOutput = detail.artifacts.find((artifact) => artifact.name === "final.spl");
    assert.ok(primaryOutput);
    assert.match(primaryOutput.path, /output\/final\.spl$/);
    assert.ok(detail.artifacts.some((artifact) => artifact.name === "final.ply"));
    assert.ok(detail.artifacts.some((artifact) => artifact.name === "train.log"));

    const detailPage = await harness.app.inject({ url: `/jobs/${payload.job.id}` });
    assert.match(detailPage.text, /Stored on this machine/);
    assert.match(detailPage.text, /output\/final\.spl/);
    assert.match(detailPage.text, /Copy log/);
  } finally {
    await harness.close();
  }
});

test("deletes a completed job and removes its files", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const datasetDir = path.join(await makeTempDir(), "dataset");
    await createColmapDataset(datasetDir);
    const archivePath = path.join(await makeTempDir(), "dataset.zip");
    await createZipFromDirectory(datasetDir, archivePath);

    const { payload } = await uploadJob(harness.app, archivePath, { name: "Delete me", preset: "preview" });
    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);
    const jobDir = detail.job.jobDir;

    const detailPage = await harness.app.inject({ url: `/jobs/${payload.job.id}` });
    assert.match(detailPage.text, /Delete Job/);

    const jobsPage = await harness.app.inject({ url: "/jobs" });
    assert.match(jobsPage.text, new RegExp(`data-delete-job="${payload.job.id}"`));

    const { response, payload: deletePayload } = await requestJson(harness.app, `/api/jobs/${payload.job.id}/delete`, { method: "POST" });
    assert.equal(response.statusCode, 200);
    assert.equal(deletePayload.deleted, true);

    const { response: detailResponse, payload: detailPayload } = await requestJson(harness.app, `/api/jobs/${payload.job.id}`);
    assert.equal(detailResponse.statusCode, 404);
    assert.equal(detailPayload.error, "Job not found");

    const { payload: jobsPayload } = await requestJson(harness.app, "/api/jobs");
    assert.equal(jobsPayload.jobs.length, 0);

    await assert.rejects(fs.stat(jobDir));
  } finally {
    await harness.close();
  }
});

test("converts a COLMAP TXT upload and keeps COLMAP artifacts", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const datasetDir = path.join(await makeTempDir(), "dataset");
    await createColmapTextDataset(datasetDir, { layout: "sparse" });
    const archivePath = path.join(await makeTempDir(), "dataset.zip");
    await createZipFromDirectory(datasetDir, archivePath);

    const { payload } = await uploadJob(harness.app, archivePath, { name: "TXT upload", preset: "preview" });
    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);

    assert.equal(detail.job.status, "succeeded");
    assert.equal(detail.job.sourceFormat, "colmap_txt");
    assert.equal(detail.job.inputKind, "prepared_zip");
    assert.equal(detail.job.datasetType, "colmap");
    assert.ok(detail.artifacts.some((artifact) => artifact.name === "colmap-database.db"));
    assert.ok(detail.artifacts.some((artifact) => artifact.name === "colmap-model-cameras.bin"));
  } finally {
    await harness.close();
  }
});

test("reconstructs a raw image zip with exhaustive matching before training", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const rawDir = path.join(await makeTempDir(), "raw");
    await createRawImagesDataset(rawDir, {
      fileNames: ["multi-1.jpg", "multi-2.jpg", "multi-3.jpg", "multi-4.jpg"]
    });
    const archivePath = path.join(await makeTempDir(), "raw.zip");
    await createZipFromDirectory(rawDir, archivePath);

    const { payload } = await uploadJob(harness.app, archivePath, {
      name: "Raw zip",
      preset: "preview",
      colmapMode: "exhaustive"
    });
    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);

    assert.equal(detail.job.status, "succeeded");
    assert.equal(detail.job.sourceFormat, "raw_images");
    assert.equal(detail.job.inputKind, "raw_zip");
    assert.equal(detail.job.colmapMode, "exhaustive");
    assert.ok(detail.artifacts.some((artifact) => artifact.name === "colmap-database.db"));
    assert.ok(detail.artifacts.some((artifact) => artifact.name === "colmap-model-selected-model-1.txt"));
    assert.match(detail.logTail, /exhaustive_matcher/);
    assert.match(detail.logTail, /--FeatureExtraction\.use_gpu 0/);
    assert.match(detail.logTail, /--FeatureMatching\.use_gpu 0/);
    assert.match(detail.logTail, /--Mapper\.tri_ignore_two_view_tracks 0/);
  } finally {
    await harness.close();
  }
});

test("accepts direct raw photo uploads and runs sequential COLMAP reconstruction", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const { response, payload } = await uploadRawJob(harness.app, {
      name: "Raw files",
      preset: "preview",
      colmapMode: "sequential",
      files: [
        { name: "tie-1.jpg" },
        { name: "tie-2.jpg" },
        { name: "tie-3.jpg" },
        { name: "tie-4.jpg" }
      ]
    });

    assert.equal(response.statusCode, 201);

    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);
    assert.equal(detail.job.status, "succeeded");
    assert.equal(detail.job.inputKind, "raw_files");
    assert.equal(detail.job.sourceFormat, "raw_images");
    assert.equal(detail.job.colmapMode, "sequential");
    assert.ok(detail.artifacts.some((artifact) => artifact.name === "colmap-model-selected-model-1.txt"));
    assert.match(detail.logTail, /sequential_matcher/);
    assert.match(detail.logTail, /colmap_flag_style=modern/);
    assert.match(detail.logTail, /--FeatureExtraction\.use_gpu 0/);
    assert.match(detail.logTail, /--FeatureMatching\.use_gpu 0/);
    assert.match(detail.logTail, /--Mapper\.tri_ignore_two_view_tracks 0/);
  } finally {
    await harness.close();
  }
});

test("retries small sequential raw uploads with exhaustive matching when COLMAP cannot initialize", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const { payload } = await uploadRawJob(harness.app, {
      name: "Fallback raw files",
      preset: "preview",
      colmapMode: "sequential",
      files: [
        { name: "fallback-1.jpg" },
        { name: "fallback-2.jpg" },
        { name: "fallback-3.jpg" },
        { name: "fallback-4.jpg" }
      ]
    });

    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);
    assert.equal(detail.job.status, "succeeded");
    assert.match(detail.logTail, /sequential_matcher/);
    assert.match(detail.logTail, /retrying_with_exhaustive/);
    assert.match(detail.logTail, /exhaustive_matcher/);
    assert.match(detail.logTail, /<<< \[mapping\] FAILED/);
  } finally {
    await harness.close();
  }
});

test("supports legacy COLMAP option names when explicitly configured", async () => {
  const harness = await startHarness({
    msplatBin: fakeMsplatBin,
    colmapBin: fakeColmapBin,
    colmapFlagStyle: "legacy"
  });
  try {
    const { payload } = await uploadRawJob(harness.app, {
      name: "Legacy raw files",
      preset: "preview",
      colmapMode: "sequential",
      files: [
        { name: "legacy-1.jpg" },
        { name: "legacy-2.jpg" },
        { name: "legacy-3.jpg" }
      ]
    });

    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);
    assert.equal(detail.job.status, "succeeded");
    assert.match(detail.logTail, /colmap_flag_style=legacy/);
    assert.match(detail.logTail, /--SiftExtraction\.use_gpu 0/);
    assert.match(detail.logTail, /--SiftMatching\.use_gpu 0/);
  } finally {
    await harness.close();
  }
});

test("rejects invalid COLMAP_FLAG_STYLE at worker startup", async () => {
  const tempRoot = await makeTempDir();
  const jobsDir = path.join(tempRoot, "jobs");
  const databasePath = path.join(tempRoot, "jobs.sqlite");

  await assert.rejects(
    createWorker({
      jobsDir,
      databasePath,
      msplatBin: fakeMsplatBin,
      colmapBin: fakeColmapBin,
      colmapFlagStyle: "bad-style"
    }),
    /COLMAP_FLAG_STYLE must be one of: modern, legacy/
  );
});

test("cancels a job during COLMAP preprocessing", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const { payload } = await uploadRawJob(harness.app, {
      name: "Slow raw files",
      preset: "preview",
      colmapMode: "sequential",
      files: [
        { name: "slow-1.jpg" },
        { name: "slow-2.jpg" },
        { name: "slow-3.jpg" },
        { name: "slow-4.jpg" }
      ]
    });

    const jobId = payload.job.id;
    const runner = (async () => {
      while (true) {
        await harness.worker.tickOnce();
        const { payload: detail } = await requestJson(harness.app, `/api/jobs/${jobId}`);
        if (["cancelled", "failed", "succeeded"].includes(detail.job.status)) {
          return detail;
        }
      }
    })();

    await waitFor(async () => {
      const { payload: detail } = await requestJson(harness.app, `/api/jobs/${jobId}`);
      return ["extracting_features", "matching", "mapping"].includes(detail.job.phase) ? detail : null;
    });

    const { response: cancelResponse } = await requestJson(harness.app, `/api/jobs/${jobId}/cancel`, { method: "POST" });
    assert.equal(cancelResponse.statusCode, 200);

    const detail = await runner;
    assert.equal(detail.job.status, "cancelled");
  } finally {
    await harness.close();
  }
});

test("cancels a running training job", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const datasetDir = path.join(await makeTempDir(), "dataset");
    await createColmapDataset(datasetDir, { modeFile: "mode-slow.txt" });
    const archivePath = path.join(await makeTempDir(), "dataset.zip");
    await createZipFromDirectory(datasetDir, archivePath);

    const { payload } = await uploadJob(harness.app, archivePath, { name: "Slow job", preset: "preview" });
    const jobId = payload.job.id;

    const runner = (async () => {
      while (true) {
        await harness.worker.tickOnce();
        const { payload: detail } = await requestJson(harness.app, `/api/jobs/${jobId}`);
        if (["cancelled", "failed", "succeeded"].includes(detail.job.status)) {
          return detail.job.status;
        }
      }
    })();

    await waitFor(async () => {
      const { payload: detail } = await requestJson(harness.app, `/api/jobs/${jobId}`);
      return detail.job.status === "running" ? detail : null;
    });

    const { response: cancelResponse } = await requestJson(harness.app, `/api/jobs/${jobId}/cancel`, { method: "POST" });
    assert.equal(cancelResponse.statusCode, 200);

    const finalStatus = await runner;
    assert.equal(finalStatus, "cancelled");
  } finally {
    await harness.close();
  }
});

test("rejects deleting an active job", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const datasetDir = path.join(await makeTempDir(), "dataset");
    await createColmapDataset(datasetDir);
    const archivePath = path.join(await makeTempDir(), "dataset.zip");
    await createZipFromDirectory(datasetDir, archivePath);

    const { payload } = await uploadJob(harness.app, archivePath, { name: "Active job", preset: "preview" });
    const { response, payload: deletePayload } = await requestJson(harness.app, `/api/jobs/${payload.job.id}/delete`, { method: "POST" });

    assert.equal(response.statusCode, 409);
    assert.match(deletePayload.error, /Active jobs cannot be deleted/);

    const { payload: detail } = await requestJson(harness.app, `/api/jobs/${payload.job.id}`);
    assert.equal(detail.job.status, "uploaded");
  } finally {
    await harness.close();
  }
});

test("marks the job failed when COLMAP preprocessing fails", async () => {
  const harness = await startHarness({ msplatBin: fakeMsplatBin, colmapBin: fakeColmapBin });
  try {
    const rawDir = path.join(await makeTempDir(), "raw");
    await createRawImagesDataset(rawDir, {
      fileNames: ["mapfail-1.jpg", "mapfail-2.jpg", "mapfail-3.jpg"]
    });
    const archivePath = path.join(await makeTempDir(), "mapfail.zip");
    await createZipFromDirectory(rawDir, archivePath);

    const { payload } = await uploadJob(harness.app, archivePath, { name: "Failing raw zip", preset: "preview" });
    const detail = await waitForTerminalDetail(harness.app, harness.worker, payload.job.id);

    assert.equal(detail.job.status, "failed");
    assert.equal(detail.job.phase, "mapping");
    assert.match(detail.job.errorMessage, /exited with code 5/);
  } finally {
    await harness.close();
  }
});
