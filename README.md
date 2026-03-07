# msplat

A 3D Gaussian Splatting training engine for Apple Silicon, built entirely on Metal. No external dependencies beyond system frameworks.

The entire training pipeline: projection, sorting, rasterization, SSIM loss, backward pass, Adam optimizer, and densification runs as fused Metal compute shaders.

The result is a self-contained engine that trains a full-resolution Mip-NeRF 360 scene in ~90 seconds and renders it at ~350 FPS on an M4 Max.

Python and Swift bindings are provided, as well as a standalone C++ CLI.

<div align="center">
  <video src="https://github.com/user-attachments/assets/cb942a38-cf6a-4b06-9899-675396550c57" />
</div>

## Why this exists

The original [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) implementation is CUDA-only. Ports to other frameworks (gsplat, taichi-3dgs, etc.) still depend on PyTorch for autograd, optimizer state, and tensor management. This means ~2GB of framework overhead, Python GIL contention, and no straightforward path to native macOS/iOS integration.

## Architecture

```
core/metal/msplat_metal.metal    ← Compute kernels
core/src/                        ← C++ training loop, dataset loaders, SSIM eval
core/include/                    ← MTensor (lightweight GPU tensor), Model, API headers
python/bindings.cpp              ← nanobind Python module
swift/Sources/Msplat/            ← Swift package (via C API bridge)
cli/msplat.cpp                   ← C++ CLI
```

### Training pipeline (single iteration)

Each training step dispatches all work into one Metal command encoder:

```
Forward:
  project_and_sh_forward     ← fused 3D→2D projection + spherical harmonics
  prefix_sum + scatter       ← gaussian→tile intersection mapping
  bitonic_sort_per_tile      ← tile-local depth sort + inline data packing
  nd_rasterize_forward       ← per-pixel alpha compositing (16x16 tiles)
  ssim_h_fwd + ssim_v_fwd   ← separable 11-tap SSIM + L1 loss

Backward:
  ssim_h_bwd + ssim_v_bwd   ← separable SSIM gradient
  rasterize_backward         ← per-pixel backward compositing
  project_and_sh_backward    ← fused projection + SH VJP
  fused_adam (×6 groups)     ← optimizer step (means, scales, quats, opacity, SH)
  accumulate_grad_stats      ← gradient norms for densification
```

### Key design decisions

**Tile-local bitonic sort** instead of global radix sort. Each 16x16 tile independently sorts its gaussians (up to 2048) in threadgroup shared memory. The sort kernel also packs per-gaussian data (xy, opacity, conic, color) inline, eliminating a separate scatter dispatch.

**GPU-resident densification.** The split/clone/cull cycle never leaves the GPU. Classification, growth, and compaction are all compute kernels operating on device buffers. No CPU readback of gradient statistics or gaussian counts.

**Fused kernels.** Projection and spherical harmonic evaluation share registers (avoid a device memory round-trip for world-space position). The backward pass recomputes 3D covariance from scales/quaternions on-the-fly rather than storing it. Adam optimizer updates all six parameter groups in fused dispatches.

**Separable SSIM.** The 11x11 Gaussian-weighted SSIM window decomposes into two 1D passes (horizontal then vertical), reducing per-pixel work from 121 to 22 multiply-adds. Forward and backward each take two kernels, using threadgroup shared memory for the intermediate statistics.

**Depth-chunked rasterization.** For tiles with extreme gaussian counts, the forward pass splits into 512-gaussian chunks with a merge kernel that reconstructs absolute transmittance. The backward pass uses precomputed prefix/suffix transmittance to avoid re-traversal.

## Installation & Usage

### Python

```bash
pip install msplat
```

```python
import msplat

dataset = msplat.load_dataset("path/to/colmap/", eval_mode=True)
config = msplat.TrainingConfig(iterations=7000, num_downscales=0)
trainer = msplat.GaussianTrainer(dataset, config)

trainer.train(lambda s: print(f"step={s.iteration} splats={s.splat_count:,}"),
              callback_every=100)

trainer.export_ply("output.ply")
trainer.save_checkpoint("checkpoint.msplat")  # save/resume training
metrics = trainer.evaluate()
print(f"PSNR: {metrics['psnr']:.2f}  SSIM: {metrics['ssim']:.3f}")

# Render from arbitrary viewpoints
pose = dataset.camera_pose(0)   # (4, 4) cam-to-world matrix
img = trainer.render_from_pose(pose)  # numpy (H, W, 3) float32
```

Supported dataset formats: COLMAP, Nerfstudio, Polycam.

Type stubs (`_core.pyi`) are included for IDE autocompletion.

#### CLI

```bash
pip install msplat[cli]
msplat-train path/to/dataset -n 7000 --eval
```

### Swift

Requires Xcode and CMake (`brew install cmake`).

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/rayanht/msplat.git", from: "1.1.0")
]
```

Build the XCFramework (one-time, from repo root):

```bash
./scripts/build-xcframework.sh
```

```swift
import Msplat

let dataset = GaussianDataset(path: "path/to/colmap/", downscaleFactor: 4.0)
let trainer = GaussianTrainer(dataset: dataset)

for _ in 0..<1000 {
    let stats = trainer.step()
    print("step=\(stats.iteration) splats=\(stats.splatCount)")
}

trainer.exportPly(to: "output.ply")

// Render from arbitrary viewpoints
let pose = dataset.cameraPose(at: 0)  // [Float] cam-to-world matrix
let img = trainer.renderFromPose(camToWorld: pose)
```

### C++ CLI

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/msplat path/to/dataset -n 7000 --eval
```

### Build from source

```bash
git clone https://github.com/rayanht/msplat.git && cd msplat

# Python
pip install -e .

# C++ CLI + static lib
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Swift XCFramework
./scripts/build-xcframework.sh
cd swift && swift build
```

Requires macOS 14+, Apple Silicon. No external dependencies.

## Run Modes

This repo now has two distinct ways to use `msplat` locally:

1. `CLI`: run `msplat` directly against a prepared dataset that already matches the native loader formats.
2. `Web GUI`: upload prepared datasets, COLMAP TXT exports, raw-image zips, or raw photos and let the worker run the full COLMAP-to-training pipeline.

### CLI Workflow

Use this path when you already have a dataset that `msplat` can read directly.

What the CLI expects:

- COLMAP with a binary sparse model such as `sparse/0/cameras.bin`, `sparse/0/images.bin`, and `sparse/0/points3D.bin`
- Nerfstudio with `transforms.json` and a seed point cloud
- Polycam with the supported camera/image layout and a seed point cloud

The CLI does **not** build COLMAP from raw photos for you. If you only have images, use the web GUI workflow below.

Build and run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/msplat path/to/dataset -n 7000 --val
```

Common examples:

```bash
# Preview
./build/msplat path/to/dataset -n 1500 -d 2 --num-downscales 2 --val

# Standard
./build/msplat path/to/dataset -n 7000 -d 1 --num-downscales 0 --val

# High
./build/msplat path/to/dataset -n 30000 -d 1 --num-downscales 0 --val
```

Add output flags as needed:

```bash
./build/msplat path/to/dataset -o output/final.spl --val --val-render output/previews
```

### Web GUI Workflow

Use this path when you want the full internal pipeline:

- upload a prepared dataset zip and train
- upload a COLMAP TXT export and convert it automatically
- upload a raw-image zip and reconstruct COLMAP first
- upload raw photo files directly and reconstruct COLMAP first

Requirements:

- built `msplat` CLI binary
- `colmap` installed and available on `PATH`, or passed explicitly through `COLMAP_BIN`
- Node.js with `npm`

Start the GUI and worker:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
MSPLAT_BIN=./build/msplat npm run web
MSPLAT_BIN=./build/msplat COLMAP_BIN=colmap npm run worker
```

Or run both together:

```bash
MSPLAT_BIN=./build/msplat COLMAP_BIN=colmap npm run dev
```

Then open:

```text
http://127.0.0.1:4321
```

The GUI exposes two upload flows:

- `Prepared Dataset / Zip`: prepared COLMAP BIN, COLMAP TXT, Nerfstudio, Polycam, or raw-image zip
- `Raw Photos`: direct multi-image upload with `Sequential` or `Exhaustive` COLMAP matching

What the worker does:

1. Validates the upload and rejects unsafe archives.
2. Converts COLMAP TXT to BIN when needed.
3. Runs COLMAP feature extraction, matching, mapper, and best-model selection for raw photos.
4. Queues the normalized dataset for `msplat` training.
5. Stores `final.spl`, `final.ply`, `cameras.json`, previews, logs, and generated COLMAP artifacts for download.

Useful environment variables:

- `MSPLAT_BIN`: path to the `msplat` CLI binary
- `COLMAP_BIN`: path to the `colmap` binary, default `colmap`
- `COLMAP_FLAG_STYLE`: COLMAP option family for CPU-only reconstruction, `modern` by default, or `legacy` for older COLMAP builds
- `MSPLAT_JOBS_DIR`: job storage root
- `DATABASE_URL`: SQLite file path
- `MAX_UPLOAD_GB`: upload limit, default `10`

Browser/API entry points:

- `POST /api/jobs`: prepared dataset zips, COLMAP TXT zips, and raw-image zips
- `POST /api/jobs/raw`: direct `multipart/form-data` raw photo uploads

Raw-photo jobs require at least 3 images. In practice, use at least 8 overlapping images whenever possible.

For very small raw-photo batches, the worker lowers COLMAP's mapper thresholds and will retry with exhaustive matching if sequential initialization cannot find a starting pair.

If raw-photo jobs fail with a COLMAP option parse error on an older install, restart the worker with:

```bash
COLMAP_FLAG_STYLE=legacy MSPLAT_BIN=./build/msplat COLMAP_BIN=colmap npm run worker
```

## Internal Website

This repo also includes a small internal training site for queued `msplat` runs.

Features:

- Upload a prepared dataset zip (COLMAP BIN, COLMAP TXT, Nerfstudio, or Polycam)
- Upload a raw-image zip or raw photo files directly and reconstruct COLMAP first
- Queue a single-worker training job on Apple Silicon
- Watch status, phase, log tail, validation thumbnails, and final metrics
- Download `final.spl`, `final.ply`, `cameras.json`, previews, the run log, and generated COLMAP artifacts

The website uses the existing CLI as its worker backend and stores job state in SQLite.

### Run it

Build the CLI first:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then start the app and worker in separate terminals:

```bash
MSPLAT_BIN=./build/msplat npm run web
MSPLAT_BIN=./build/msplat COLMAP_BIN=colmap npm run worker
```

Or run both together during development:

```bash
MSPLAT_BIN=./build/msplat COLMAP_BIN=colmap npm run dev
```

Homebrew COLMAP 3.13 works with the default `COLMAP_FLAG_STYLE=modern`. Use `COLMAP_FLAG_STYLE=legacy` only if your COLMAP build still expects `SiftExtraction.use_gpu` and `SiftMatching.use_gpu`.

Default web URL: `http://127.0.0.1:4321`

This section is the same web GUI workflow described above, kept here for feature-level reference.

### Website tests

```bash
npm test
```

There is also an opt-in smoke test that can run a real Apple Silicon training job:

```bash
MSPLAT_SMOKE_BIN=./build/msplat \
MSPLAT_SMOKE_COLMAP_BIN=colmap \
MSPLAT_SMOKE_DATASET=/absolute/path/to/dataset.zip \
npm test
```

## Benchmarks

mipnerf360, M4 Max. msplat runs 7K iterations with no downscales:

```bash
msplat-train path/to/scene -n 7000 --num-downscales 0 --eval
```

| Scene | msplat PSNR | msplat SSIM | msplat wall time | gsplat PSNR | gsplat SSIM | gsplat wall time
|-------|-------------|-------------|-----------|-------------|-------------|-------------|
| bicycle | 23.21 | 0.605 | 82s | 23.71 | 0.668 | ~335s
| counter | 27.44 | 0.881 | 91s | 27.14 | 0.878 | ~335s
| garden | 25.76 | 0.786 | 107s | 26.30 | 0.833 | ~335s
| room | 30.21 | 0.898 | 85s | 29.21 | 0.893 | ~335s

### 30K iterations (garden)

```bash
msplat-train path/to/garden -n 30000 --num-downscales 0 --eval
```

| | msplat | gsplat |
|---|---|---|
| PSNR | 27.17 | 27.32 |
| SSIM | 0.854 | 0.865 |
| Gaussians | 3.55M | — |
| Wall time | 1039s | ~2149s |

gsplat numbers from [docs.gsplat.studio](https://docs.gsplat.studio/main/tests/eval.html) (TITAN RTX). gsplat wall times are the reported average across *all* mipnerf360 scenes (per-scene times not published).

## License

Apache 2.0
