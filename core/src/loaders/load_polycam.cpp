#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string findImage(const fs::path &dir, const std::string &stem) {
    for (auto ext : {".png", ".jpg", ".jpeg", ".JPG"}) {
        auto p = dir / (stem + ext);
        if (fs::exists(p)) return p.string();
    }
    return (dir / (stem + ".png")).string();
}

InputData loaders::loadPolycam(const std::string &projectRoot) {
    InputData data;
    fs::path root(projectRoot);
    fs::path keyframesDir = root / "keyframes" / "corrected_cameras";
    fs::path imagesDir = root / "keyframes" / "corrected_images";

    if (fs::exists(keyframesDir)) {
        // Layout 1: individual camera JSONs
        std::vector<fs::path> jsonFiles;
        for (auto &entry : fs::directory_iterator(keyframesDir))
            if (entry.path().extension() == ".json") jsonFiles.push_back(entry.path());
        std::sort(jsonFiles.begin(), jsonFiles.end());

        for (auto &jp : jsonFiles) {
            std::ifstream f(jp);
            json j = json::parse(f);

            Camera cam;
            cam.width = j.value("width", 0);
            cam.height = j.value("height", 0);
            cam.fx = j.value("fx", 0.0f);  cam.fy = j.value("fy", 0.0f);
            cam.cx = j.value("cx", (float)cam.width / 2.0f);
            cam.cy = j.value("cy", (float)cam.height / 2.0f);

            float R[9], T[3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R[r*3+c] = j.value("R_" + std::to_string(r) + std::to_string(c), 0.0f);
            T[0] = j.value("t_20", 0.0f);
            T[1] = j.value("t_21", 0.0f);
            T[2] = j.value("t_22", 0.0f);

            // c2w with OpenGL Y/Z flip
            cam.camToWorld[0]  =  R[0]; cam.camToWorld[1]  =  R[1]; cam.camToWorld[2]  =  R[2]; cam.camToWorld[3]  =  T[0];
            cam.camToWorld[4]  = -R[3]; cam.camToWorld[5]  = -R[4]; cam.camToWorld[6]  = -R[5]; cam.camToWorld[7]  = -T[1];
            cam.camToWorld[8]  = -R[6]; cam.camToWorld[9]  = -R[7]; cam.camToWorld[10] = -R[8]; cam.camToWorld[11] = -T[2];
            cam.camToWorld[12] = 0;     cam.camToWorld[13] = 0;     cam.camToWorld[14] = 0;     cam.camToWorld[15] = 1;

            cam.filePath = findImage(imagesDir, jp.stem().string());
            data.cameras.push_back(cam);
        }
    } else {
        // Layout 2: single cameras.json (dispatcher already confirmed it exists)
        std::ifstream f(root / "cameras.json");
        json j = json::parse(f);

        auto &frames = j.contains("frames") ? j["frames"] : j;
        for (auto &frame : frames) {
            Camera cam;
            cam.width = frame.value("width", 0);
            cam.height = frame.value("height", 0);
            cam.fx = frame.value("fx", 0.0f);  cam.fy = frame.value("fy", 0.0f);
            cam.cx = frame.value("cx", (float)cam.width / 2.0f);
            cam.cy = frame.value("cy", (float)cam.height / 2.0f);

            if (frame.contains("transform_matrix")) {
                auto &tm = frame["transform_matrix"];
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        cam.camToWorld[r*4+c] = tm[r][c].get<float>();
            }

            cam.filePath = frame.value("file_path", "");
            if (!cam.filePath.empty() && cam.filePath[0] != '/')
                cam.filePath = (root / cam.filePath).string();
            data.cameras.push_back(cam);
        }
    }

    // Point cloud
    for (auto p : {"keyframes/point_cloud.ply", "point_cloud.ply", "sparse.ply"}) {
        auto path = (root / p).string();
        if (fs::exists(path)) { data.points = readPly(path); break; }
    }

    autoScaleAndCenter(data);
    return data;
}
