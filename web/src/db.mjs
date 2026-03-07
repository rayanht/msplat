import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { ensureDir, nowIso } from "./utils.mjs";
import { getPreset } from "./presets.mjs";

const BOOLEAN_COLUMNS = new Set(["cancel_requested"]);

const COLUMN_MAP = {
  name: "name",
  preset: "preset",
  status: "status",
  inputKind: "input_kind",
  phase: "phase",
  phaseMessage: "phase_message",
  colmapMode: "colmap_mode",
  sourceFormat: "source_format",
  datasetType: "dataset_type",
  uploadName: "upload_name",
  uploadSizeBytes: "upload_size_bytes",
  uploadPath: "upload_path",
  jobDir: "job_dir",
  datasetRoot: "dataset_root",
  outputPath: "output_path",
  camerasPath: "cameras_path",
  logPath: "log_path",
  previewsDir: "previews_dir",
  latestPreviewPath: "latest_preview_path",
  progressStep: "progress_step",
  progressPercent: "progress_percent",
  finalPsnr: "final_psnr",
  finalSsim: "final_ssim",
  finalL1: "final_l1",
  finalGaussians: "final_gaussians",
  errorMessage: "error_message",
  cancelRequested: "cancel_requested",
  workerPid: "worker_pid",
  runnerPid: "runner_pid",
  colmapWorkspacePath: "colmap_workspace_path",
  colmapDatabasePath: "colmap_database_path",
  colmapModelPath: "colmap_model_path",
  createdAt: "created_at",
  updatedAt: "updated_at",
  startedAt: "started_at",
  finishedAt: "finished_at"
};

const MIGRATIONS = [
  ["input_kind", "ALTER TABLE jobs ADD COLUMN input_kind TEXT NOT NULL DEFAULT 'prepared_zip'"],
  ["phase", "ALTER TABLE jobs ADD COLUMN phase TEXT NOT NULL DEFAULT 'validating'"],
  ["phase_message", "ALTER TABLE jobs ADD COLUMN phase_message TEXT NOT NULL DEFAULT 'Waiting for worker'"],
  ["colmap_mode", "ALTER TABLE jobs ADD COLUMN colmap_mode TEXT"],
  ["source_format", "ALTER TABLE jobs ADD COLUMN source_format TEXT"],
  ["colmap_workspace_path", "ALTER TABLE jobs ADD COLUMN colmap_workspace_path TEXT"],
  ["colmap_database_path", "ALTER TABLE jobs ADD COLUMN colmap_database_path TEXT"],
  ["colmap_model_path", "ALTER TABLE jobs ADD COLUMN colmap_model_path TEXT"]
];

function rowToJob(row) {
  if (!row) return null;
  return {
    id: row.id,
    name: row.name,
    preset: row.preset,
    presetLabel: getPreset(row.preset)?.label ?? row.preset,
    status: row.status,
    inputKind: row.input_kind,
    phase: row.phase,
    phaseMessage: row.phase_message,
    colmapMode: row.colmap_mode,
    sourceFormat: row.source_format,
    datasetType: row.dataset_type,
    uploadName: row.upload_name,
    uploadSizeBytes: row.upload_size_bytes,
    uploadPath: row.upload_path,
    jobDir: row.job_dir,
    datasetRoot: row.dataset_root,
    outputPath: row.output_path,
    camerasPath: row.cameras_path,
    logPath: row.log_path,
    previewsDir: row.previews_dir,
    latestPreviewPath: row.latest_preview_path,
    progressStep: row.progress_step,
    progressPercent: row.progress_percent,
    finalPsnr: row.final_psnr,
    finalSsim: row.final_ssim,
    finalL1: row.final_l1,
    finalGaussians: row.final_gaussians,
    errorMessage: row.error_message,
    cancelRequested: Boolean(row.cancel_requested),
    workerPid: row.worker_pid,
    runnerPid: row.runner_pid,
    colmapWorkspacePath: row.colmap_workspace_path,
    colmapDatabasePath: row.colmap_database_path,
    colmapModelPath: row.colmap_model_path,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    startedAt: row.started_at,
    finishedAt: row.finished_at
  };
}

function normalizePatchValue(column, value) {
  if (BOOLEAN_COLUMNS.has(column)) {
    return value ? 1 : 0;
  }
  return value;
}

function ensureColumns(db) {
  const columns = new Set(db.prepare("PRAGMA table_info(jobs)").all().map((row) => row.name));
  for (const [column, statement] of MIGRATIONS) {
    if (!columns.has(column)) {
      db.exec(statement);
    }
  }
}

export async function openDatabase(databasePath) {
  await ensureDir(path.dirname(databasePath));
  const db = new DatabaseSync(databasePath);
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA busy_timeout = 5000;
    CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      preset TEXT NOT NULL,
      status TEXT NOT NULL,
      input_kind TEXT NOT NULL DEFAULT 'prepared_zip',
      phase TEXT NOT NULL DEFAULT 'validating',
      phase_message TEXT NOT NULL DEFAULT 'Waiting for worker',
      colmap_mode TEXT,
      source_format TEXT,
      dataset_type TEXT,
      upload_name TEXT NOT NULL,
      upload_size_bytes INTEGER NOT NULL,
      upload_path TEXT NOT NULL,
      job_dir TEXT NOT NULL,
      dataset_root TEXT,
      output_path TEXT NOT NULL,
      cameras_path TEXT NOT NULL,
      log_path TEXT NOT NULL,
      previews_dir TEXT NOT NULL,
      latest_preview_path TEXT,
      progress_step INTEGER NOT NULL DEFAULT 0,
      progress_percent REAL NOT NULL DEFAULT 0,
      final_psnr REAL,
      final_ssim REAL,
      final_l1 REAL,
      final_gaussians INTEGER,
      error_message TEXT,
      cancel_requested INTEGER NOT NULL DEFAULT 0,
      worker_pid INTEGER,
      runner_pid INTEGER,
      colmap_workspace_path TEXT,
      colmap_database_path TEXT,
      colmap_model_path TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      started_at TEXT,
      finished_at TEXT
    );
    CREATE INDEX IF NOT EXISTS jobs_status_created_idx ON jobs(status, created_at);
  `);
  ensureColumns(db);
  return db;
}

export function createJobStore(db) {
  function patchJob(id, patch) {
    const sets = [];
    const values = [];

    for (const [key, value] of Object.entries(patch)) {
      if (value === undefined) continue;
      const column = COLUMN_MAP[key];
      if (!column) continue;
      sets.push(`${column} = ?`);
      values.push(normalizePatchValue(column, value));
    }

    sets.push("updated_at = ?");
    values.push(nowIso());
    values.push(id);

    db.prepare(`UPDATE jobs SET ${sets.join(", ")} WHERE id = ?`).run(...values);
    return getJob(id);
  }

  function getJob(id) {
    return rowToJob(db.prepare("SELECT * FROM jobs WHERE id = ?").get(id));
  }

  return {
    createJob(input) {
      const timestamp = nowIso();
      db.prepare(`
        INSERT INTO jobs (
          id, name, preset, status, input_kind, phase, phase_message, colmap_mode, source_format,
          upload_name, upload_size_bytes, upload_path, job_dir, output_path, cameras_path,
          log_path, previews_dir, created_at, updated_at
        ) VALUES (?, ?, ?, 'uploaded', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        input.id,
        input.name,
        input.preset,
        input.inputKind,
        input.phase ?? "validating",
        input.phaseMessage ?? "Waiting for worker",
        input.colmapMode ?? null,
        input.sourceFormat ?? null,
        input.uploadName,
        input.uploadSizeBytes,
        input.uploadPath,
        input.jobDir,
        input.outputPath,
        input.camerasPath,
        input.logPath,
        input.previewsDir,
        timestamp,
        timestamp
      );
      return getJob(input.id);
    },

    getJob,

    listJobs() {
      return db.prepare("SELECT * FROM jobs ORDER BY created_at DESC").all().map(rowToJob);
    },

    deleteJob(id) {
      const result = db.prepare("DELETE FROM jobs WHERE id = ?").run(id);
      return result.changes > 0;
    },

    patchJob,

    claimNextUploaded(workerPid) {
      const row = db.prepare(`
        UPDATE jobs
        SET status = 'validating',
            worker_pid = ?,
            updated_at = ?
        WHERE id = (
          SELECT id
          FROM jobs
          WHERE status = 'uploaded' AND cancel_requested = 0
          ORDER BY created_at ASC
          LIMIT 1
        )
        RETURNING *
      `).get(workerPid, nowIso());
      return rowToJob(row);
    },

    markQueued(id, fields = {}) {
      return patchJob(id, {
        status: "queued",
        cancelRequested: false,
        ...fields
      });
    },

    claimNextQueued(workerPid) {
      const timestamp = nowIso();
      const row = db.prepare(`
        UPDATE jobs
        SET status = 'running',
            worker_pid = ?,
            started_at = COALESCE(started_at, ?),
            updated_at = ?
        WHERE id = (
          SELECT id
          FROM jobs
          WHERE status = 'queued' AND cancel_requested = 0
          ORDER BY created_at ASC
          LIMIT 1
        )
        RETURNING *
      `).get(workerPid, timestamp, timestamp);
      return rowToJob(row);
    },

    requestCancel(id) {
      const job = getJob(id);
      if (!job) return null;

      if (job.status === "uploaded" || (job.status === "queued" && !job.runnerPid && !job.workerPid)) {
        return this.markCancelled(id);
      }

      if (job.status === "validating" || job.status === "queued" || job.status === "running") {
        return patchJob(id, { cancelRequested: true });
      }

      return job;
    },

    setRunnerPid(id, runnerPid) {
      return patchJob(id, { runnerPid });
    },

    markPhase(id, phase, phaseMessage, progressPercent, progressStep = 0, extra = {}) {
      return patchJob(id, {
        phase,
        phaseMessage,
        progressPercent,
        progressStep,
        ...extra
      });
    },

    markSucceeded(id, metrics = {}) {
      return patchJob(id, {
        status: "succeeded",
        phase: "finalizing",
        phaseMessage: "Training complete",
        progressPercent: 100,
        finalPsnr: metrics.psnr ?? null,
        finalSsim: metrics.ssim ?? null,
        finalL1: metrics.l1 ?? null,
        finalGaussians: metrics.gaussians ?? null,
        errorMessage: null,
        cancelRequested: false,
        runnerPid: null,
        workerPid: null,
        finishedAt: nowIso()
      });
    },

    markFailed(id, errorMessage) {
      return patchJob(id, {
        status: "failed",
        phaseMessage: errorMessage,
        errorMessage,
        cancelRequested: false,
        runnerPid: null,
        workerPid: null,
        finishedAt: nowIso()
      });
    },

    markCancelled(id, errorMessage = null) {
      return patchJob(id, {
        status: "cancelled",
        phaseMessage: "Cancelled",
        errorMessage,
        cancelRequested: false,
        runnerPid: null,
        workerPid: null,
        finishedAt: nowIso()
      });
    }
  };
}
