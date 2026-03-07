#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <CLI/CLI.hpp>
#include "model.hpp"
#include "input_data.hpp"
#include "msplat_c_api.h"
#include "random_iter.hpp"
#include "loaders.hpp"
#include "msplat.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    CLI::App app{"msplat — 3D Gaussian Splatting for Apple Silicon"};
    app.set_version_flag("--version", APP_VERSION);

    // Required
    std::string projectRoot;
    app.add_option("input", projectRoot, "Path to dataset (COLMAP, Nerfstudio, Polycam)")
        ->required()
        ->check(CLI::ExistingDirectory);

    // Output
    std::string outputScene = "splat.ply";
    app.add_option("-o,--output", outputScene, "Output scene path");
    std::string exportPly;
    app.add_option("--export-ply", exportPly, "Additional PLY export path");
    int saveEvery = -1;
    app.add_option("-s,--save-every", saveEvery, "Save every N steps (-1 to disable)");

    // Resume
    std::string resume;
    app.add_option("--resume", resume, "Resume training from PLY file")
        ->check(CLI::ExistingFile);

    // Validation
    bool validate = false;
    app.add_flag("--val", validate, "Withhold a camera for validation");
    std::string valImage = "random";
    app.add_option("--val-image", valImage, "Validation image filename");
    std::string valRender;
    app.add_option("--val-render", valRender, "Directory to render validation images");

    // Evaluation
    bool evalMode = false;
    app.add_flag("--eval", evalMode, "Evaluate on held-out test views");
    int testEvery = 8;
    app.add_option("--test-every", testEvery, "Hold out every Nth image for eval")
        ->check(CLI::Range(2, 100));

    // Training hyperparameters
    int numIters = 30000;
    app.add_option("-n,--num-iters", numIters, "Number of iterations")
        ->check(CLI::Range(1, 1000000));
    float downScaleFactor = 1.0f;
    app.add_option("-d,--downscale-factor", downScaleFactor, "Image downscale factor")
        ->check(CLI::Range(1.0f, 32.0f));
    int numDownscales = 2;
    app.add_option("--num-downscales", numDownscales, "Progressive downscale levels");
    int resolutionSchedule = 3000;
    app.add_option("--resolution-schedule", resolutionSchedule, "Double resolution every N steps");
    int shDegree = 3;
    app.add_option("--sh-degree", shDegree, "Max spherical harmonics degree")
        ->check(CLI::Range(0, 4));
    int shDegreeInterval = 1000;
    app.add_option("--sh-degree-interval", shDegreeInterval, "Increase SH degree every N steps");
    float ssimWeight = 0.2f;
    app.add_option("--ssim-weight", ssimWeight, "SSIM loss weight (0 = L1 only)")
        ->check(CLI::Range(0.0f, 1.0f));
    int refineEvery = 100;
    app.add_option("--refine-every", refineEvery, "Densify/prune every N steps");
    int warmupLength = 500;
    app.add_option("--warmup-length", warmupLength, "Steps before first densification");
    int resetAlphaEvery = 30;
    app.add_option("--reset-alpha-every", resetAlphaEvery, "Reset opacity every N refinements");
    float densifyGradThresh = 0.0002f;
    app.add_option("--densify-grad-thresh", densifyGradThresh, "Gradient threshold for split/dup");
    float densifySizeThresh = 0.01f;
    app.add_option("--densify-size-thresh", densifySizeThresh, "Size threshold (dup vs split)");
    int stopScreenSizeAt = 4000;
    app.add_option("--stop-screen-size-at", stopScreenSizeAt, "Stop splitting large gaussians after N steps");
    float splitScreenSize = 0.05f;
    app.add_option("--split-screen-size", splitScreenSize, "Screen-space split threshold");
    bool keepCrs = false;
    app.add_flag("--keep-crs", keepCrs, "Retain input coordinate reference system");
    std::vector<float> bgColor = {0.6130f, 0.0101f, 0.3984f};
    app.add_option("--bg-color", bgColor, "Background RGB (0-1), default magenta")
        ->expected(3);
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");

    CLI11_PARSE(app, argc, argv);

    if (validate || !valRender.empty()) validate = true;
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    downScaleFactor = std::max(downScaleFactor, 1.0f);

    try {
        fs::path executablePath = fs::absolute(argv[0]);
        fs::path metallibPath = executablePath.parent_path() / "default.metallib";
        if (fs::exists(metallibPath)) {
            msplat_set_metallib_path(metallibPath.string().c_str());
            std::cout << "[runtime] metallib=" << metallibPath << std::endl;
        } else {
            std::cout << "[runtime] warning: default.metallib not found next to executable: "
                      << metallibPath << std::endl;
        }

        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);

        auto formatGiB = [](double bytes) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0 * 1024.0));
            return ss.str();
        };

        double estimatedImageBytes = 0.0;
        for (const auto &cam : inputData.cameras) {
            double width = std::max(1.0, std::floor(static_cast<double>(cam.width) / static_cast<double>(downScaleFactor)));
            double height = std::max(1.0, std::floor(static_cast<double>(cam.height) / static_cast<double>(downScaleFactor)));
            estimatedImageBytes += width * height * 3.0 * sizeof(float);
        }

        std::cout << "[dataset] path=" << projectRoot
                  << " cameras=" << inputData.cameras.size()
                  << " seed_points=" << inputData.points.count
                  << " downscale=" << downScaleFactor
                  << " est_cpu_image_mem=" << formatGiB(estimatedImageBytes) << " GiB"
                  << std::endl;

        if (estimatedImageBytes > 4.0 * 1024.0 * 1024.0 * 1024.0) {
            std::cout << "[dataset] warning: high image memory estimate; consider Preview preset or -d 2 for faster startup"
                      << std::endl;
        }

        double loadedImageBytes = 0.0;
        const size_t loadLogEvery = std::max<size_t>(1, inputData.cameras.size() / 8);
        std::cout << "[dataset] loading images..." << std::endl;

        for (size_t index = 0; index < inputData.cameras.size(); index++) {
            auto &cam = inputData.cameras[index];
            cam.loadImage(downScaleFactor);
            loadedImageBytes += (double)cam.width * (double)cam.height * 3.0 * sizeof(float);

            if (index == 0 || (index + 1) % loadLogEvery == 0 || index + 1 == inputData.cameras.size()) {
                std::cout << "[dataset] loaded " << (index + 1) << "/" << inputData.cameras.size()
                          << " latest=" << fs::path(cam.filePath).filename().string()
                          << " size=" << cam.width << "x" << cam.height
                          << " approx_cpu_image_mem=" << formatGiB(loadedImageBytes) << " GiB"
                          << std::endl;
            }
        }

        std::vector<Camera> cams;
        std::vector<Camera> testCams;
        Camera *valCam = nullptr;

        if (evalMode) {
            auto [train, test] = inputData.splitTrainTest(testEvery);
            cams = train; testCams = test;
            std::cout << "Eval mode: " << cams.size() << " train, " << testCams.size() << " test" << std::endl;
        } else {
            auto [train, val] = inputData.getCameras(validate, valImage);
            cams = train; valCam = val;
        }

        std::cout << "[train] train_cameras=" << cams.size()
                  << " val_camera=" << (valCam ? fs::path(valCam->filePath).filename().string() : "none")
                  << " eval_cameras=" << testCams.size()
                  << std::endl;

        Model model(inputData, cams.size(),
                     numDownscales, resolutionSchedule, shDegree, shDegreeInterval,
                     refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh,
                     densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                     numIters, keepCrs,
                     bgColor.data());

        std::cout << "[train] starting iterations=" << numIters
                  << " preset_downscale=" << downScaleFactor
                  << " progressive_downscales=" << numDownscales
                  << " resolution_schedule=" << resolutionSchedule
                  << " output=" << outputScene
                  << std::endl;

        std::vector<size_t> camIndices(cams.size());
        std::iota(camIndices.begin(), camIndices.end(), 0);
        InfiniteRandomIterator<size_t> camsIter(camIndices);

        size_t step = 1;
        if (!resume.empty()) step = model.loadPly(resume) + 1;

        bool benchmarking = std::getenv("BENCHMARK") != nullptr;
        int bench_warmup = 50;
        std::vector<double> bench_iter_ms, bench_cpu_ms, bench_drain_ms;
        if (benchmarking) {
            bench_iter_ms.reserve(numIters);
            bench_cpu_ms.reserve(numIters);
            bench_drain_ms.reserve(numIters);
        }
        auto cpu_now = []() { return std::chrono::high_resolution_clock::now(); };

        auto bench_start = cpu_now();
        for (; step <= (size_t)numIters; step++) {
            Camera &cam = cams[camsIter.next()];

            auto iter_start = cpu_now();
            MTensor gt = cam.getGPUImage(model.getDownscaleFactor(step));
            model.fullIteration(cam, step, gt, ssimWeight);
            model.schedulersStep(step);
            model.afterTrain(step);
            msplat_commit();

            if (step <= 10 || step % 50 == 0 || step == (size_t)numIters) {
                std::cout << "[train] step=" << step
                          << "/" << numIters
                          << " ds=" << model.getDownscaleFactor((int)step)
                          << " gaussians=" << model.means.size(0)
                          << " camera=" << fs::path(cam.filePath).filename().string()
                          << std::endl;
            }

            if (benchmarking && step > (size_t)bench_warmup) {
                auto pre_sync = cpu_now();
                msplat_gpu_sync();
                auto iter_end = cpu_now();
                double iter_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
                double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(pre_sync - iter_start).count() / 1000.0;
                double drain_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - pre_sync).count() / 1000.0;
                bench_iter_ms.push_back(iter_ms);
                bench_cpu_ms.push_back(cpu_ms);
                bench_drain_ms.push_back(drain_ms);
            }

            if (saveEvery > 0 && step % saveEvery == 0) {
                fs::path p(outputScene);
                model.save(p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string(), step);
            }

            if (!valRender.empty() && step % 10 == 0) {
                MTensor rgb = model.render(*valCam, step);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                Image valImg;
                valImg.width = (int)rgb_cpu.size(1);
                valImg.height = (int)rgb_cpu.size(0);
                valImg.data.resize(valImg.width * valImg.height * 3);
                memcpy(valImg.ptr(), rgb_cpu.data_ptr(), valImg.data.size() * sizeof(float));
                std::string previewPath = (fs::path(valRender) / (std::to_string(step) + ".png")).string();
                imwriteRGB(previewPath, valImg);
                std::cout << "[train] wrote_preview step=" << step
                          << " path=" << previewPath
                          << std::endl;
            }
        }

        if (benchmarking && !bench_iter_ms.empty()) {
            auto bench_end = cpu_now();
            double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start).count() / 1000.0;
            size_t n = bench_iter_ms.size();
            std::vector<double> sorted = bench_iter_ms;
            std::sort(sorted.begin(), sorted.end());
            double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
            double mean = sum / n;
            double median = (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2];
            double sq_sum = 0;
            for (double v : sorted) sq_sum += (v - mean) * (v - mean);
            double stddev = std::sqrt(sq_sum / n);

            std::cout << "\n=== Benchmark (" << n << " iters, " << bench_warmup << " warmup, " << total_s << "s total) ===\n";
            std::cout << "  mean:   " << mean   << " ms/iter\n";
            std::cout << "  median: " << median  << " ms/iter\n";
            std::cout << "  stddev: " << stddev  << " ms/iter\n";
            std::cout << "  p5:     " << sorted[(size_t)(n * 0.05)] << " ms/iter\n";
            std::cout << "  p95:    " << sorted[(size_t)(n * 0.95)] << " ms/iter\n";
            std::cout << "  min:    " << sorted.front() << " ms/iter\n";
            std::cout << "  max:    " << sorted.back()  << " ms/iter\n";
            std::cout << "  wall:   " << total_s << "s for " << numIters << " iters\n";

            auto stats = [](std::vector<double> &v) {
                std::vector<double> s = v;
                std::sort(s.begin(), s.end());
                size_t n = s.size();
                double sum = std::accumulate(s.begin(), s.end(), 0.0);
                double med = (n % 2 == 0) ? (s[n/2-1] + s[n/2]) / 2.0 : s[n/2];
                return std::make_pair(sum / n, med);
            };
            auto [cpu_mean, cpu_med] = stats(bench_cpu_ms);
            auto [drain_mean, drain_med] = stats(bench_drain_ms);
            std::cout << "\n  --- CPU dispatch vs GPU drain ---\n";
            std::cout << "  cpu dispatch:  mean=" << cpu_mean << "  median=" << cpu_med << " ms\n";
            std::cout << "  gpu drain:     mean=" << drain_mean << "  median=" << drain_med << " ms\n";
            std::cout << "  gpu fraction:  " << (drain_med / median * 100) << "%\n\n";
        }

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene, numIters);
        if (!exportPly.empty() && fs::path(exportPly) != fs::path(outputScene)) {
            model.savePly(exportPly, numIters);
            std::cout << "[train] saved_output path=" << exportPly << std::endl;
        }
        std::cout << "[train] saved_output path=" << outputScene << std::endl;

        // Evaluation
        if (evalMode && !testCams.empty()) {
            double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
            int nTest = testCams.size();

            std::cout << "\n=== Evaluation (" << nTest << " test views) ===" << std::endl;
            for (int i = 0; i < nTest; i++) {
                MTensor rgb = model.render(testCams[i], numIters);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                MTensor gt_cpu = testCams[i].getGPUImage(model.getDownscaleFactor(numIters)).cpu();

                float p = psnr(rgb_cpu, gt_cpu);
                float s = ssim_eval(rgb_cpu, gt_cpu);
                float l = l1_loss(rgb_cpu, gt_cpu);
                sumPsnr += p; sumSsim += s; sumL1 += l;

                std::cout << "  [" << (i+1) << "/" << nTest << "] "
                          << fs::path(testCams[i].filePath).filename().string()
                          << "  PSNR=" << p << "  SSIM=" << s << "  L1=" << l << std::endl;
            }
            std::cout << "\n  PSNR:  " << (sumPsnr / nTest)
                      << "  SSIM:  " << (sumSsim / nTest)
                      << "  L1:  " << (sumL1 / nTest)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        // Validation
        if (valCam) {
            MTensor rgb = model.render(*valCam, numIters);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = valCam->getGPUImage(model.getDownscaleFactor(numIters)).cpu();

            std::cout << "\n=== Validation (" << valCam->filePath << ") ===" << std::endl;
            std::cout << "  PSNR:  " << psnr(rgb_cpu, gt_cpu)
                      << "  SSIM:  " << ssim_eval(rgb_cpu, gt_cpu)
                      << "  L1:  " << l1_loss(rgb_cpu, gt_cpu)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        cleanup_msplat_metal();
        msplat_gpu_sync();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        cleanup_msplat_metal();
        msplat_gpu_sync();
        return 1;
    }
}
