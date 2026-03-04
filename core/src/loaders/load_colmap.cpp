#include "loaders.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// Quaternion [w,x,y,z] → row-major 3x3 rotation matrix
static void quatToRotMat(const double q[4], float R[9]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    w /= n; x /= n; y /= n; z /= n;

    R[0] = (float)(1 - 2*(y*y + z*z));  R[1] = (float)(2*(x*y - w*z));      R[2] = (float)(2*(x*z + w*y));
    R[3] = (float)(2*(x*y + w*z));      R[4] = (float)(1 - 2*(x*x + z*z));  R[5] = (float)(2*(y*z - w*x));
    R[6] = (float)(2*(x*z - w*y));      R[7] = (float)(2*(y*z + w*x));      R[8] = (float)(1 - 2*(x*x + y*y));
}

enum ColmapModel { SIMPLE_PINHOLE=0, PINHOLE=1, SIMPLE_RADIAL=2, RADIAL=3, OPENCV=4 };

struct ColmapCamera {
    uint32_t id;
    int model;
    int width, height;
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;
};

struct ColmapImage {
    uint32_t camId;
    double quat[4]; // w, x, y, z
    double t[3];    // world-to-camera translation
    std::string filename;
};

static std::unordered_map<uint32_t, ColmapCamera> readCamerasBin(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), 8);

    std::unordered_map<uint32_t, ColmapCamera> cams;
    for (uint64_t i = 0; i < n; i++) {
        ColmapCamera c = {};
        uint32_t model;
        uint64_t w, h;
        f.read(reinterpret_cast<char*>(&c.id), 4);
        f.read(reinterpret_cast<char*>(&model), 4);
        f.read(reinterpret_cast<char*>(&w), 8);
        f.read(reinterpret_cast<char*>(&h), 8);
        c.model = (int)model;
        c.width = (int)w;
        c.height = (int)h;

        auto rd = [&]() -> double { double v; f.read(reinterpret_cast<char*>(&v), 8); return v; };

        switch (c.model) {
            case SIMPLE_PINHOLE: c.fx = c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); break;
            case PINHOLE:        c.fx = (float)rd(); c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); break;
            case SIMPLE_RADIAL:  c.fx = c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); c.k1 = (float)rd(); break;
            case RADIAL:         c.fx = c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); c.k1 = (float)rd(); c.k2 = (float)rd(); break;
            case OPENCV:         c.fx = (float)rd(); c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd();
                                 c.k1 = (float)rd(); c.k2 = (float)rd(); c.p1 = (float)rd(); c.p2 = (float)rd(); break;
            default: throw std::runtime_error("Unsupported COLMAP camera model: " + std::to_string(c.model));
        }
        cams[c.id] = c;
    }
    return cams;
}

static std::vector<ColmapImage> readImagesBin(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), 8);

    std::vector<ColmapImage> images;
    images.reserve(n);

    for (uint64_t i = 0; i < n; i++) {
        ColmapImage img;
        uint32_t imageId;
        f.read(reinterpret_cast<char*>(&imageId), 4);
        f.read(reinterpret_cast<char*>(img.quat), 32); // 4 doubles
        f.read(reinterpret_cast<char*>(img.t), 24);     // 3 doubles
        f.read(reinterpret_cast<char*>(&img.camId), 4);

        char ch;
        while (f.read(&ch, 1) && ch != '\0') img.filename += ch;

        uint64_t numPts2D;
        f.read(reinterpret_cast<char*>(&numPts2D), 8);
        f.seekg(numPts2D * 24, std::ios::cur);

        images.push_back(img);
    }
    return images;
}

// w2c rotation + translation → 4x4 c2w row-major with OpenGL Y/Z flip
static void w2cToCamToWorld(const double quat[4], const double t[3], float out[16]) {
    float R[9];
    quatToRotMat(quat, R);

    // R^T (transpose = inverse for rotation)
    float Ri[9] = { R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8] };

    // -R^T * t
    float Ti[3] = {
        -(Ri[0]*(float)t[0] + Ri[1]*(float)t[1] + Ri[2]*(float)t[2]),
        -(Ri[3]*(float)t[0] + Ri[4]*(float)t[1] + Ri[5]*(float)t[2]),
        -(Ri[6]*(float)t[0] + Ri[7]*(float)t[1] + Ri[8]*(float)t[2])
    };

    // OpenGL flip: negate columns 1,2 (camera Y-down→Y-up, Z-fwd→Z-back)
    out[0]  = Ri[0]; out[1]  = -Ri[1]; out[2]  = -Ri[2]; out[3]  = Ti[0];
    out[4]  = Ri[3]; out[5]  = -Ri[4]; out[6]  = -Ri[5]; out[7]  = Ti[1];
    out[8]  = Ri[6]; out[9]  = -Ri[7]; out[10] = -Ri[8]; out[11] = Ti[2];
    out[12] = 0;     out[13] = 0;      out[14] = 0;      out[15] = 1;
}

InputData loaders::loadColmap(const std::string &projectRoot, const std::string &imageSourcePath) {
    // Find sparse dir — dispatcher already confirmed cameras.bin exists
    fs::path root(projectRoot);
    std::string sparseDir = fs::exists(root / "cameras.bin")
        ? projectRoot
        : (root / "sparse" / "0").string();

    std::string imageDir = !imageSourcePath.empty() ? imageSourcePath
        : fs::exists(root / "images") ? (root / "images").string()
        : projectRoot;

    auto cameras = readCamerasBin(sparseDir + "/cameras.bin");
    auto images = readImagesBin(sparseDir + "/images.bin");

    std::sort(images.begin(), images.end(),
        [](const ColmapImage &a, const ColmapImage &b) { return a.filename < b.filename; });

    InputData data;
    data.cameras.reserve(images.size());

    for (auto &img : images) {
        auto it = cameras.find(img.camId);
        if (it == cameras.end()) continue;
        auto &cc = it->second;

        Camera cam;
        cam.width = cc.width; cam.height = cc.height;
        cam.fx = cc.fx; cam.fy = cc.fy; cam.cx = cc.cx; cam.cy = cc.cy;
        cam.k1 = cc.k1; cam.k2 = cc.k2; cam.p1 = cc.p1; cam.p2 = cc.p2;
        cam.filePath = imageDir + "/" + img.filename;
        w2cToCamToWorld(img.quat, img.t, cam.camToWorld);
        data.cameras.push_back(cam);
    }

    // Point cloud
    std::string ptsPath = sparseDir + "/points3D.bin";
    if (fs::exists(ptsPath))
        data.points = readColmapPoints(ptsPath);
    else if (fs::exists(sparseDir + "/points3D.ply"))
        data.points = readPly(sparseDir + "/points3D.ply");

    autoScaleAndCenter(data);
    return data;
}
