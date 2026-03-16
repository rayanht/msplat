#ifndef LOADERS_H
#define LOADERS_H

#include "input_data.hpp"

// Format-specific loaders
namespace loaders {
    InputData loadColmap(const std::string &projectRoot, const std::string &imageSourcePath = "");
    InputData loadNerfstudio(const std::string &projectRoot);
    InputData loadPolycam(const std::string &projectRoot);
}

// PLY point cloud reader
Points readPly(const std::string &path);

// COLMAP binary point cloud reader
Points readColmapPoints(const std::string &path);

// Image I/O
Image imreadRGB(const std::string &path);       // returns float32 [0,1] directly
Mask imreadMask(const std::string &path);        // returns single-channel float32 [0,1]
Image resizeArea(const Image &src, int dstW, int dstH);  // box-filter downscale
Mask resizeAreaMask(const Mask &src, int dstW, int dstH);  // box-filter downscale (single channel)
void imwriteRGB(const std::string &path, const Image &img);  // save as PNG

// Undistortion (Brown-Conrady model, alpha=0 crop)
struct UndistortResult {
    Image image;
    float fx, fy, cx, cy;  // updated intrinsics after crop
    int width, height;      // cropped dimensions
    int roiX, roiY;         // crop origin in undistorted full-size image
};
UndistortResult undistortImage(const Image &src,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3);
Mask undistortMask(const Mask &src,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3,
    int roiX, int roiY, int roiW, int roiH);

// Pose utilities
void autoScaleAndCenter(InputData &data);

// Gaussian PLY/splat I/O (trained scene export/import)
struct GaussianParams {
    MTensor &means, &scales, &quats, &featuresDc, &featuresRest, &opacities;
    float scale;          // CRS scale factor
    float translation[3]; // CRS translation
    bool keepCrs;
};

void saveGaussianPly(const std::string &path, GaussianParams &p, int step);
void saveGaussianSplat(const std::string &path, GaussianParams &p);

struct LoadedGaussians {
    MTensor means, scales, quats, featuresDc, featuresRest, opacities;
    int step;
};
LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs);

#endif
