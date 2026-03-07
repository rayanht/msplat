import path from "node:path";
import { buildArtifactList } from "./artifacts.mjs";
import { escapeHtml, formatBytes, formatDate, formatNumber } from "./utils.mjs";
import { listPresets } from "./presets.mjs";

function layout(config, title, content, scripts = "") {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${escapeHtml(title)} · ${escapeHtml(config.title)}</title>
    <link rel="stylesheet" href="/assets/app.css">
  </head>
  <body>
    <div class="page-shell">
      <header class="site-header">
        <a class="site-mark" href="/jobs">
          <span class="site-mark__badge">M</span>
          <span>
            <strong>${escapeHtml(config.title)}</strong>
            <small>${escapeHtml(config.machineName)}</small>
          </span>
        </a>
        <nav class="site-nav">
          <a href="/jobs">Jobs</a>
          <a href="/jobs/new">New Job</a>
        </nav>
      </header>
      <main class="page-main">
        ${content}
      </main>
    </div>
    ${scripts}
  </body>
</html>`;
}

function statusPill(status, id = "") {
  const attr = id ? ` id="${id}"` : "";
  return `<span${attr} class="status status--${escapeHtml(status)}">${escapeHtml(status)}</span>`;
}

function progressBar(percent) {
  return `<div class="progress">
    <div class="progress__fill" style="width:${Math.max(0, Math.min(100, percent || 0))}%"></div>
  </div>`;
}

function labelForSourceFormat(sourceFormat) {
  return {
    colmap_bin: "COLMAP BIN",
    colmap_txt: "COLMAP TXT",
    nerfstudio: "Nerfstudio",
    polycam: "Polycam",
    raw_images: "Raw Images"
  }[sourceFormat] ?? "Pending";
}

function labelForInputKind(inputKind) {
  return {
    prepared_zip: "Prepared Zip",
    raw_zip: "Raw Image Zip",
    raw_files: "Raw Photos"
  }[inputKind] ?? "Pending";
}

function buildProgressLabel(job) {
  if (job.phaseMessage && job.phase === "training" && job.progressStep) {
    return `${job.phaseMessage} · step ${job.progressStep.toLocaleString()}`;
  }
  if (job.phaseMessage) {
    return job.phaseMessage;
  }
  if (job.progressStep) {
    return `Step ${job.progressStep.toLocaleString()}`;
  }
  return "Waiting for worker";
}

function isActiveStatus(status) {
  return ["uploaded", "validating", "queued", "running"].includes(status);
}

function canDeleteStatus(status) {
  return ["succeeded", "failed", "cancelled"].includes(status);
}

function renderPresetCards(groupName, checkedKey = "preview") {
  return listPresets().map((preset) => `
    <label class="preset-card">
      <input type="radio" name="${groupName}" value="${preset.key}" ${preset.key === checkedKey ? "checked" : ""}>
      <span>${escapeHtml(preset.label)}</span>
      <small>${preset.iterations.toLocaleString()} iterations</small>
    </label>
  `).join("");
}

function renderColmapModeSelect(name, required = false, blankLabel = "Use only for raw-image uploads") {
  return `
    <label>
      <span>COLMAP mode</span>
      <select name="${name}" ${required ? "required" : ""}>
        ${required ? "" : `<option value="">${escapeHtml(blankLabel)}</option>`}
        <option value="sequential" ${required ? "selected" : ""}>Sequential</option>
        <option value="exhaustive">Exhaustive</option>
      </select>
      <small class="field-help">${required ? "Photos will be reconstructed into COLMAP before training." : "Ignored for prepared datasets and COLMAP TXT exports."}</small>
    </label>
  `;
}

export async function renderJobsPage(config, jobs) {
  const rows = jobs.map((job) => {
    const latestPreview = job.latestPreviewPath
      ? `/api/jobs/${job.id}/artifacts/${encodeURIComponent(path.basename(job.latestPreviewPath))}`
      : "";

    return `<article class="job-card">
      <div class="job-card__meta">
        ${statusPill(job.status)}
        <span class="meta-chip">${escapeHtml(job.presetLabel)}</span>
        <span class="meta-chip">${escapeHtml(labelForSourceFormat(job.sourceFormat))}</span>
        <span class="meta-chip">${escapeHtml(labelForInputKind(job.inputKind))}</span>
      </div>
      <div class="job-card__body">
        <div>
          <h2><a href="/jobs/${job.id}">${escapeHtml(job.name)}</a></h2>
          <p>${escapeHtml(job.uploadName)}</p>
          <p class="job-card__phase">${escapeHtml(buildProgressLabel(job))}</p>
        </div>
        ${latestPreview
          ? `<img class="job-card__thumb" src="${latestPreview}" alt="Latest validation preview for ${escapeHtml(job.name)}">`
          : `<div class="job-card__thumb job-card__thumb--empty">Awaiting preview</div>`}
      </div>
      <div class="job-card__details">
        <span>${formatDate(job.createdAt)}</span>
        <span>${formatBytes(job.uploadSizeBytes)}</span>
        <span>${escapeHtml(job.colmapMode ? `COLMAP ${job.colmapMode}` : (job.phase || "pending"))}</span>
      </div>
      ${progressBar(job.progressPercent || 0)}
      <div class="job-card__actions">
        <a class="button button--ghost button--small" href="/jobs/${job.id}">Open</a>
        ${canDeleteStatus(job.status) ? `<button class="button button--danger button--small" type="button" data-delete-job="${job.id}">Delete</button>` : ""}
      </div>
    </article>`;
  }).join("");

  return layout(
    config,
    "Jobs",
    `<section class="hero">
      <p class="eyebrow">Internal training queue</p>
      <h1>Queue prepared datasets or raw photos and let the worker build the splat.</h1>
      <p class="lede">This site now accepts prepared zips, COLMAP TXT exports, raw-photo zips, and direct multi-image uploads. Raw photos are reconstructed with COLMAP on the same Apple Silicon host before <code>msplat</code> training starts.</p>
      <div class="hero__actions">
        <a class="button" href="/jobs/new">Create a Job</a>
      </div>
    </section>
    <section class="section-head">
      <h2>Recent jobs</h2>
      <p id="job-count-label">${jobs.length} total</p>
    </section>
    <section class="job-grid">
      ${rows || `<div class="empty-state"><h2>No jobs yet</h2><p>Upload a prepared dataset zip or start with raw photos.</p><a class="button" href="/jobs/new">Create the first job</a></div>`}
    </section>`,
    `<script type="module">
      for (const button of document.querySelectorAll("[data-delete-job]")) {
        button.addEventListener("click", async () => {
          if (!window.confirm("Delete this job and all of its artifacts?")) {
            return;
          }

          button.disabled = true;
          const response = await fetch("/api/jobs/" + button.dataset.deleteJob + "/delete", { method: "POST" });
          let payload = null;
          try {
            payload = await response.json();
          } catch {}

          if (!response.ok) {
            window.alert(payload?.error || "Delete failed");
            button.disabled = false;
            return;
          }

          window.location.reload();
        });
      }
    </script>`
  );
}

export function renderNewJobPage(config) {
  const presetCards = renderPresetCards("preset");

  return layout(
    config,
    "New Job",
    `<section class="hero hero--compact">
      <p class="eyebrow">New training run</p>
      <h1>Upload one input and queue the job.</h1>
      <p class="lede">Use a dataset zip for prepared inputs, COLMAP TXT exports, or raw-image zips. Use raw photos when you only have images and want the worker to build COLMAP before training.</p>
    </section>
    <section class="upload-stack">
      <article class="panel panel--upload">
        <div class="panel__header">
          <p class="eyebrow">Single Upload Flow</p>
          <h2>Choose the input type, then upload once</h2>
          <p class="panel__lede">Prepared datasets can train directly. Raw images and raw-image zips will run through COLMAP first.</p>
        </div>
        <form id="upload-form" class="upload-form">
          <label>
            <span>Job name</span>
            <input name="name" type="text" maxlength="120" placeholder="South building sweep" required>
          </label>
          <fieldset>
            <legend>Input type</legend>
            <div class="upload-mode-grid">
              <label class="preset-card upload-mode-card">
                <input type="radio" name="inputMode" value="zip" checked>
                <span>Dataset Zip</span>
                <small>Prepared COLMAP BIN, COLMAP TXT, Nerfstudio, Polycam, or raw-image zip.</small>
              </label>
              <label class="preset-card upload-mode-card">
                <input type="radio" name="inputMode" value="raw">
                <span>Raw Photos</span>
                <small>Multiple JPG or PNG files. The worker will build COLMAP before training.</small>
              </label>
            </div>
          </fieldset>
          <div id="zip-fields" class="upload-section">
            <label>
              <span>Dataset zip</span>
              <input name="file" type="file" accept=".zip,application/zip" required>
            </label>
          </div>
          <div id="raw-fields" class="upload-section" hidden>
            <label>
              <span>Photos</span>
              <input name="images" type="file" accept="image/*,.jpg,.jpeg,.png" multiple>
              <small class="field-help">Upload at least 3 images. In practice, 8 or more overlapping photos usually reconstruct much more reliably.</small>
            </label>
          </div>
          <label>
            <span>COLMAP mode</span>
            <select name="colmapMode">
              <option value="">Only needed for raw-photo inputs</option>
              <option value="sequential">Sequential</option>
              <option value="exhaustive">Exhaustive</option>
            </select>
            <small id="colmap-mode-help" class="field-help">Ignored for prepared datasets and COLMAP TXT exports. Use it for raw photos or raw-image zips when you want to control matching strategy.</small>
          </label>
          <fieldset>
            <legend>Preset</legend>
            <div class="preset-grid">${presetCards}</div>
          </fieldset>
          <div id="upload-notes" class="upload-form__notes">
            <p>Dataset zip mode accepts prepared datasets directly.</p>
            <p>COLMAP TXT zips will be converted to BIN automatically.</p>
            <p>Raw-image zips can optionally use a selected COLMAP matching mode before training.</p>
            <p>Example prepared zip datasets: <a href="https://demuc.de/colmap/datasets/" target="_blank" rel="noreferrer">COLMAP sample datasets</a>.</p>
          </div>
          <button id="upload-submit" class="button" type="submit">Queue Upload</button>
          <p id="upload-form-status" class="inline-status" aria-live="polite"></p>
        </form>
      </article>
    </section>`,
    `<script type="module">
      const form = document.getElementById("upload-form");
      const zipFields = document.getElementById("zip-fields");
      const rawFields = document.getElementById("raw-fields");
      const statusEl = document.getElementById("upload-form-status");
      const notesEl = document.getElementById("upload-notes");
      const submitButton = document.getElementById("upload-submit");
      const colmapModeHelp = document.getElementById("colmap-mode-help");

      function selectedMode() {
        return form.querySelector('input[name="inputMode"]:checked').value;
      }

      function syncMode() {
        const mode = selectedMode();
        const isRaw = mode === "raw";
        zipFields.hidden = isRaw;
        rawFields.hidden = !isRaw;
        form.file.required = !isRaw;
        form.images.required = isRaw;
        submitButton.textContent = isRaw ? "Queue Raw Photo Job" : "Queue Upload";
        statusEl.textContent = "";

        if (isRaw) {
          notesEl.innerHTML = [
            "<p><strong>Sequential</strong> is usually better for ordered walkthroughs.</p>",
            "<p><strong>Exhaustive</strong> is safer for unordered captures, but slower.</p>",
            "<p>The selected COLMAP mode is required for raw-photo jobs.</p>"
          ].join("");
          colmapModeHelp.textContent = "Required for raw photos. Choose sequential for ordered captures or exhaustive for unordered images.";
        } else {
          notesEl.innerHTML = [
            "<p>Dataset zip mode accepts prepared datasets directly.</p>",
            "<p>COLMAP TXT zips will be converted to BIN automatically.</p>",
            "<p>Raw-image zips can optionally use a selected COLMAP matching mode before training.</p>"
          ].join("");
          colmapModeHelp.textContent = "Ignored for prepared datasets and COLMAP TXT exports. Use it for raw photos or raw-image zips when you want to control matching strategy.";
        }
      }

      async function uploadZip() {
        const file = form.file.files[0];
        const preset = form.querySelector('input[name="preset"]:checked').value;
        const name = form.name.value.trim();
        const colmapMode = form.colmapMode.value;

        if (!file) {
          statusEl.textContent = "Choose a zip file first.";
          return;
        }

        statusEl.textContent = "Uploading archive...";

        try {
          const params = new URLSearchParams({ preset, name });
          if (colmapMode) params.set("colmapMode", colmapMode);

          const response = await fetch("/api/jobs?" + params.toString(), {
            method: "POST",
            headers: {
              "content-type": "application/zip",
              "x-file-name": file.name
            },
            body: file
          });

          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.error || "Upload failed");
          }

          statusEl.textContent = "Upload complete. Redirecting...";
          window.location.href = "/jobs/" + payload.job.id;
        } catch (error) {
          statusEl.textContent = error.message;
        }
      }

      async function uploadRawPhotos() {
        const files = Array.from(form.images.files || []);
        const colmapMode = form.colmapMode.value;
        if (files.length < 3) {
          statusEl.textContent = "Upload at least 3 images. 8 or more is recommended.";
          return;
        }
        if (!colmapMode) {
          statusEl.textContent = "Choose a COLMAP mode for raw photos.";
          return;
        }

        statusEl.textContent = "Uploading photos...";

        try {
          const formData = new FormData();
          formData.set("name", form.name.value.trim());
          formData.set("preset", form.querySelector('input[name="preset"]:checked').value);
          formData.set("colmapMode", colmapMode);
          for (const file of files) {
            formData.append("images", file, file.name);
          }

          const response = await fetch("/api/jobs/raw", {
            method: "POST",
            body: formData
          });

          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.error || "Upload failed");
          }

          statusEl.textContent = "Upload complete. Redirecting...";
          window.location.href = "/jobs/" + payload.job.id;
        } catch (error) {
          statusEl.textContent = error.message;
        }
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (selectedMode() === "raw") {
          await uploadRawPhotos();
          return;
        }
        await uploadZip();
      });

      for (const input of form.querySelectorAll('input[name="inputMode"]')) {
        input.addEventListener("change", syncMode);
      }

      syncMode();
    </script>`
  );
}

export async function renderJobDetailPage(config, job) {
  const artifacts = await buildArtifactList(job);
  const outputArtifacts = artifacts.filter((artifact) => artifact.kind === "output");
  const artifactLinks = artifacts
    .map((artifact) => `<li><a href="/api/jobs/${job.id}/artifacts/${encodeURIComponent(artifact.name)}">${escapeHtml(artifact.label || artifact.name)}</a> <small>${formatBytes(artifact.sizeBytes)}</small></li>`)
    .join("");
  const outputLinks = outputArtifacts
    .map((artifact) => `
      <li class="output-file">
        <div class="output-file__head">
          <div>
            <strong>${escapeHtml(artifact.name)}</strong>
            <small>${formatBytes(artifact.sizeBytes)}</small>
          </div>
          <a class="button button--ghost button--small" href="/api/jobs/${job.id}/artifacts/${encodeURIComponent(artifact.name)}" download>Download</a>
        </div>
        <p class="output-file__label">Stored on this machine</p>
        <code class="output-file__path">${escapeHtml(artifact.path)}</code>
      </li>
    `)
    .join("");
  const latestPreviewUrl = job.latestPreviewPath
    ? `/api/jobs/${job.id}/artifacts/${encodeURIComponent(path.basename(job.latestPreviewPath))}`
    : "";

  return layout(
    config,
    job.name,
    `<section class="hero hero--compact hero--job">
      <div class="job-hero">
        <div class="job-hero__main">
          <p class="eyebrow">Job detail</p>
          <h1 id="job-title">${escapeHtml(job.name)}</h1>
          <p id="job-progress-label" class="job-hero__lede">${escapeHtml(buildProgressLabel(job))}</p>
        </div>
        <div class="job-hero__meta">
          ${statusPill(job.status, "job-status-pill")}
          <span id="job-phase-pill" class="meta-chip">${escapeHtml(job.phase || "pending")}</span>
          <span class="meta-chip">${escapeHtml(job.presetLabel)}</span>
          <span class="meta-chip" id="job-source-chip">${escapeHtml(labelForSourceFormat(job.sourceFormat))}</span>
        </div>
        <div class="hero__actions hero__actions--job">
          <a class="button button--ghost" href="/jobs">Back to queue</a>
          <button id="cancel-job" class="button button--danger" type="button" ${isActiveStatus(job.status) ? "" : "hidden"}>Cancel</button>
          <button id="delete-job" class="button button--danger button--ghost-danger" type="button" ${canDeleteStatus(job.status) ? "" : "hidden"}>Delete Job</button>
        </div>
      </div>
    </section>
    <section class="job-board">
      <div class="job-lane">
        <div class="job-lane__head">
          <p class="eyebrow">Overview</p>
          <p id="job-progress-percent">${Math.round(job.progressPercent || 0)}%</p>
        </div>
        <article class="panel panel--board">
          <div class="section-head section-head--tight">
            <h2>Training progress</h2>
            <p>Current state</p>
          </div>
          ${progressBar(job.progressPercent || 0)}
          <dl class="stat-grid stat-grid--board">
            <div><dt>Status</dt><dd id="job-status">${escapeHtml(job.status)}</dd></div>
            <div><dt>Phase</dt><dd id="job-phase">${escapeHtml(job.phase || "pending")}</dd></div>
            <div><dt>Preset</dt><dd>${escapeHtml(job.presetLabel)}</dd></div>
            <div><dt>Dataset</dt><dd id="job-dataset">${escapeHtml(job.datasetType || "pending")}</dd></div>
            <div><dt>Source</dt><dd id="job-source">${escapeHtml(labelForSourceFormat(job.sourceFormat))}</dd></div>
            <div><dt>Input</dt><dd id="job-input-kind">${escapeHtml(labelForInputKind(job.inputKind))}</dd></div>
            <div><dt>COLMAP Mode</dt><dd id="job-colmap-mode">${escapeHtml(job.colmapMode || "—")}</dd></div>
            <div><dt>Uploaded</dt><dd>${formatDate(job.createdAt)}</dd></div>
            <div><dt>PSNR</dt><dd id="metric-psnr">${formatNumber(job.finalPsnr)}</dd></div>
            <div><dt>SSIM</dt><dd id="metric-ssim">${formatNumber(job.finalSsim, 3)}</dd></div>
            <div><dt>L1</dt><dd id="metric-l1">${formatNumber(job.finalL1, 4)}</dd></div>
            <div><dt>Gaussians</dt><dd id="metric-gaussians">${job.finalGaussians?.toLocaleString?.() ?? "—"}</dd></div>
          </dl>
          <p id="job-error" class="inline-status inline-status--error">${escapeHtml(job.errorMessage || "")}</p>
        </article>
      </div>
      <div class="job-lane">
        <div class="job-lane__head">
          <p class="eyebrow">Outputs</p>
          <p id="artifact-count-label">${artifacts.length} artifacts</p>
        </div>
        <article class="panel panel--board">
          <div class="section-head section-head--tight">
            <h2>Output files</h2>
            <p>Download or copy path</p>
          </div>
          <p class="panel-note">Click <strong>Download</strong> to save through the browser. The generated file also stays on this machine at the path shown below.</p>
          <ul id="output-file-list" class="output-file-list">${outputLinks || "<li class=\"output-file output-file--empty\">No output file yet.</li>"}</ul>
        </article>
        <article class="panel panel--board">
          <div class="section-head section-head--tight">
            <h2>Latest validation render</h2>
            <p>Updates during training</p>
          </div>
          ${latestPreviewUrl
            ? `<img id="job-preview" class="preview-image preview-image--board" src="${latestPreviewUrl}" alt="Latest validation preview">`
            : `<div id="job-preview-empty" class="preview-empty preview-empty--board">No preview yet</div><img id="job-preview" class="preview-image preview-image--board preview-image--hidden" alt="Latest validation preview">`}
        </article>
        <article class="panel panel--board panel--artifacts">
          <div class="section-head section-head--tight">
            <h2>Artifacts</h2>
            <p>Logs, previews, outputs</p>
          </div>
          <ul id="artifact-list" class="artifact-list artifact-list--board">${artifactLinks || "<li>Nothing available yet.</li>"}</ul>
        </article>
      </div>
      <div class="job-lane">
        <div class="job-lane__head">
          <p class="eyebrow">Diagnostics</p>
          <p>Polling every 5 seconds</p>
        </div>
        <article class="panel panel--board panel--log">
          <div class="section-head section-head--tight section-head--actions">
            <div>
              <h2>Log tail</h2>
              <p>Worker + trainer output</p>
            </div>
            <div class="section-head__actions">
              <button id="copy-log" class="button button--ghost button--small" type="button" disabled>Copy log</button>
              <p id="copy-log-status" class="inline-status" aria-live="polite"></p>
            </div>
          </div>
          <pre id="job-log" class="log-view log-view--board"></pre>
        </article>
      </div>
    </section>`,
    `<script type="module">
      const preview = document.getElementById("job-preview");
      const previewEmpty = document.getElementById("job-preview-empty");
      const progressLabel = document.getElementById("job-progress-label");
      const progressPercentEl = document.getElementById("job-progress-percent");
      const logEl = document.getElementById("job-log");
      const artifactList = document.getElementById("artifact-list");
      const outputFileList = document.getElementById("output-file-list");
      const artifactCountLabel = document.getElementById("artifact-count-label");
      const errorEl = document.getElementById("job-error");
      const cancelButton = document.getElementById("cancel-job");
      const deleteButton = document.getElementById("delete-job");
      const copyLogButton = document.getElementById("copy-log");
      const copyLogStatus = document.getElementById("copy-log-status");
      const statusPill = document.getElementById("job-status-pill");
      const phasePill = document.getElementById("job-phase-pill");
      const sourceChip = document.getElementById("job-source-chip");
      const jobId = ${JSON.stringify(job.id)};
      let copyLogStatusTimer = null;

      function escapeHtml(text) {
        return String(text ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function renderArtifacts(artifacts) {
        if (!artifacts.length) {
          artifactList.innerHTML = "<li>Nothing available yet.</li>";
          return;
        }
        artifactList.innerHTML = artifacts.map((artifact) =>
          '<li><a href="' + artifact.url + '">' + escapeHtml(artifact.label || artifact.name) + '</a> <small>' + escapeHtml(artifact.sizeLabel) + '</small></li>'
        ).join("");
      }

      function renderOutputFiles(artifacts) {
        const outputs = artifacts.filter((artifact) => artifact.kind === "output");
        if (!outputs.length) {
          outputFileList.innerHTML = '<li class="output-file output-file--empty">No output file yet.</li>';
          return;
        }
        outputFileList.innerHTML = outputs.map((artifact) =>
          '<li class="output-file">' +
            '<div class="output-file__head">' +
              '<div><strong>' + escapeHtml(artifact.name) + '</strong><small>' + escapeHtml(artifact.sizeLabel) + '</small></div>' +
              '<a class="button button--ghost button--small" href="' + artifact.url + '" download>Download</a>' +
            '</div>' +
            '<p class="output-file__label">Stored on this machine</p>' +
            '<code class="output-file__path">' + escapeHtml(artifact.path || "") + '</code>' +
          '</li>'
        ).join("");
      }

      function buildProgressLabel(job) {
        if (job.phaseMessage && job.phase === "training" && job.progressStep) {
          return job.phaseMessage + " · step " + job.progressStep.toLocaleString();
        }
        if (job.phaseMessage) {
          return job.phaseMessage;
        }
        if (job.progressStep) {
          return "Step " + job.progressStep.toLocaleString();
        }
        return "Waiting for worker";
      }

      function isActiveStatus(status) {
        return ["uploaded", "validating", "queued", "running"].includes(status);
      }

      function canDeleteStatus(status) {
        return ["succeeded", "failed", "cancelled"].includes(status);
      }

      function syncLogActions(logText) {
        if (copyLogButton) {
          copyLogButton.disabled = !String(logText || "").trim();
        }
      }

      function showCopyLogStatus(message, isError = false) {
        if (!copyLogStatus) return;
        copyLogStatus.textContent = message;
        copyLogStatus.dataset.state = isError ? "error" : "success";
        clearTimeout(copyLogStatusTimer);
        if (!message) return;
        copyLogStatusTimer = window.setTimeout(() => {
          copyLogStatus.textContent = "";
          copyLogStatus.dataset.state = "";
        }, 2200);
      }

      async function copyText(text) {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(text);
          return;
        }

        const helper = document.createElement("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "");
        helper.style.position = "absolute";
        helper.style.left = "-9999px";
        document.body.appendChild(helper);
        helper.select();
        const copied = document.execCommand("copy");
        document.body.removeChild(helper);
        if (!copied) {
          throw new Error("Copy failed");
        }
      }

      function sourceLabel(sourceFormat) {
        return {
          colmap_bin: "COLMAP BIN",
          colmap_txt: "COLMAP TXT",
          nerfstudio: "Nerfstudio",
          polycam: "Polycam",
          raw_images: "Raw Images"
        }[sourceFormat] || "Pending";
      }

      function inputLabel(inputKind) {
        return {
          prepared_zip: "Prepared Zip",
          raw_zip: "Raw Image Zip",
          raw_files: "Raw Photos"
        }[inputKind] || "Pending";
      }

      function syncActions(job) {
        if (cancelButton) {
          cancelButton.hidden = !isActiveStatus(job.status);
          if (cancelButton.hidden) {
            cancelButton.disabled = false;
          }
        }
        if (deleteButton) {
          deleteButton.hidden = !canDeleteStatus(job.status);
          if (deleteButton.hidden) {
            deleteButton.disabled = false;
          }
        }
      }

      async function refresh() {
        const response = await fetch("/api/jobs/" + jobId);
        if (response.status === 404) {
          clearInterval(timer);
          window.location.href = "/jobs";
          return;
        }
        if (!response.ok) return;
        const payload = await response.json();
        const job = payload.job;

        statusPill.className = "status status--" + job.status;
        statusPill.textContent = job.status;
        phasePill.textContent = job.phase || "pending";
        document.getElementById("job-status").textContent = job.status;
        document.getElementById("job-phase").textContent = job.phase || "pending";
        document.getElementById("job-dataset").textContent = job.datasetType || "pending";
        document.getElementById("job-source").textContent = sourceLabel(job.sourceFormat);
        sourceChip.textContent = sourceLabel(job.sourceFormat);
        document.getElementById("job-input-kind").textContent = inputLabel(job.inputKind);
        document.getElementById("job-colmap-mode").textContent = job.colmapMode || "—";
        document.getElementById("metric-psnr").textContent = job.finalPsnrLabel;
        document.getElementById("metric-ssim").textContent = job.finalSsimLabel;
        document.getElementById("metric-l1").textContent = job.finalL1Label;
        document.getElementById("metric-gaussians").textContent = job.finalGaussiansLabel;
        progressLabel.textContent = buildProgressLabel(job);
        progressPercentEl.textContent = Math.round(job.progressPercent || 0) + "%";
        document.querySelector(".progress__fill").style.width = job.progressPercent + "%";
        errorEl.textContent = job.errorMessage || "";
        logEl.textContent = payload.logTail || "";
        syncLogActions(payload.logTail || "");
        renderOutputFiles(payload.artifacts);
        renderArtifacts(payload.artifacts);
        artifactCountLabel.textContent = payload.artifacts.length + " artifacts";
        syncActions(job);

        if (payload.latestPreviewUrl) {
          preview.src = payload.latestPreviewUrl + "?t=" + Date.now();
          preview.classList.remove("preview-image--hidden");
          previewEmpty?.remove();
        }

        if (!["uploaded", "validating", "queued", "running"].includes(job.status)) {
          clearInterval(timer);
        }
      }

      const timer = setInterval(refresh, 5000);
      refresh();

      syncLogActions(logEl.textContent || "");

      if (copyLogButton) {
        copyLogButton.addEventListener("click", async () => {
          const logText = logEl.textContent || "";
          if (!logText.trim()) {
            showCopyLogStatus("No log yet.", true);
            return;
          }

          copyLogButton.disabled = true;
          try {
            await copyText(logText);
            showCopyLogStatus("Copied.");
          } catch {
            showCopyLogStatus("Copy failed.", true);
          } finally {
            syncLogActions(logEl.textContent || "");
          }
        });
      }

      if (cancelButton) {
        cancelButton.addEventListener("click", async () => {
          cancelButton.disabled = true;
          const response = await fetch("/api/jobs/" + jobId + "/cancel", { method: "POST" });
          if (!response.ok) {
            cancelButton.disabled = false;
            return;
          }
          refresh();
        });
      }

      if (deleteButton) {
        deleteButton.addEventListener("click", async () => {
          if (!window.confirm("Delete this job and all of its artifacts?")) {
            return;
          }

          deleteButton.disabled = true;
          const response = await fetch("/api/jobs/" + jobId + "/delete", { method: "POST" });
          let payload = null;
          try {
            payload = await response.json();
          } catch {}

          if (!response.ok) {
            errorEl.textContent = payload?.error || "Delete failed";
            deleteButton.disabled = false;
            return;
          }

          window.location.href = "/jobs";
        });
      }
    </script>`
  );
}
