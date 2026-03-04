#include "loaders.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

// ── Image loading (CoreGraphics) ─────────────────────────────────────────────

Image imreadRGB(const std::string &path) {
    CFStringRef cfPath = CFStringCreateWithCString(nullptr, path.c_str(), kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfPath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfPath);

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!cgImage) {
        throw std::runtime_error("Failed to decode image: " + path);
    }

    int w = (int)CGImageGetWidth(cgImage);
    int h = (int)CGImageGetHeight(cgImage);

    // Render into RGBA buffer, then extract RGB and convert to float32
    std::vector<uint8_t> rgba(w * h * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data(), w, h, 8, w * 4, colorSpace,
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault
    );
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cgImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);
    CGImageRelease(cgImage);

    // RGBA → float32 RGB [0,1]
    Image img;
    img.width = w;
    img.height = h;
    img.data.resize(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        img.data[i * 3 + 0] = rgba[i * 4 + 0] / 255.0f;
        img.data[i * 3 + 1] = rgba[i * 4 + 1] / 255.0f;
        img.data[i * 3 + 2] = rgba[i * 4 + 2] / 255.0f;
    }
    return img;
}

// ── Image writing (CoreGraphics PNG) ─────────────────────────────────────────

void imwriteRGB(const std::string &path, const Image &img) {
    int w = img.width, h = img.height;

    // float32 RGB → uint8 RGB
    std::vector<uint8_t> rgb8(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        float v = std::clamp(img.data[i] * 255.0f, 0.0f, 255.0f);
        rgb8[i] = (uint8_t)(v + 0.5f);
    }

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgb8.data(), w, h, 8, w * 3, colorSpace,
        kCGImageAlphaNone | kCGBitmapByteOrderDefault
    );
    CGImageRef cgImage = CGBitmapContextCreateImage(ctx);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);

    CFStringRef cfPath = CFStringCreateWithCString(nullptr, path.c_str(), kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfPath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfPath);

    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr);
    CFRelease(url);
    CGImageDestinationAddImage(dest, cgImage, nullptr);
    CGImageDestinationFinalize(dest);

    CFRelease(dest);
    CGImageRelease(cgImage);
}

// ── Area-based image resize (box filter) ─────────────────────────────────────

Image resizeArea(const Image &src, int dstW, int dstH) {
    Image dst;
    dst.width = dstW;
    dst.height = dstH;
    dst.data.resize(dstW * dstH * 3, 0.0f);

    float scaleX = (float)src.width / dstW;
    float scaleY = (float)src.height / dstH;

    for (int dy = 0; dy < dstH; dy++) {
        float srcY0 = dy * scaleY;
        float srcY1 = (dy + 1) * scaleY;

        for (int dx = 0; dx < dstW; dx++) {
            float srcX0 = dx * scaleX;
            float srcX1 = (dx + 1) * scaleX;

            float sum[3] = {};
            float totalArea = 0;

            int iy0 = (int)srcY0;
            int iy1 = std::min((int)std::ceil(srcY1), src.height);
            int ix0 = (int)srcX0;
            int ix1 = std::min((int)std::ceil(srcX1), src.width);

            for (int iy = iy0; iy < iy1; iy++) {
                float wy = std::min((float)(iy + 1), srcY1) - std::max((float)iy, srcY0);
                for (int ix = ix0; ix < ix1; ix++) {
                    float wx = std::min((float)(ix + 1), srcX1) - std::max((float)ix, srcX0);
                    float area = wx * wy;
                    const float *p = &src.data[(iy * src.width + ix) * 3];
                    sum[0] += p[0] * area;
                    sum[1] += p[1] * area;
                    sum[2] += p[2] * area;
                    totalArea += area;
                }
            }

            float *out = &dst.data[(dy * dstW + dx) * 3];
            float inv = 1.0f / totalArea;
            out[0] = sum[0] * inv;
            out[1] = sum[1] * inv;
            out[2] = sum[2] * inv;
        }
    }
    return dst;
}

// ── Undistortion (Brown-Conrady model) ───────────────────────────────────────

// Apply forward distortion: normalized undistorted → normalized distorted
static void distortPoint(float x, float y,
    float k1, float k2, float p1, float p2, float k3,
    float &xd, float &yd)
{
    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    xd = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
    yd = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
}

// Iteratively invert distortion: normalized distorted → normalized undistorted
static void undistortPoint(float xd, float yd,
    float k1, float k2, float p1, float p2, float k3,
    float &xu, float &yu)
{
    xu = xd;
    yu = yd;
    for (int i = 0; i < 20; i++) {
        float r2 = xu * xu + yu * yu;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
        float dx = 2.0f * p1 * xu * yu + p2 * (r2 + 2.0f * xu * xu);
        float dy = p1 * (r2 + 2.0f * yu * yu) + 2.0f * p2 * xu * yu;
        xu = (xd - dx) / radial;
        yu = (yd - dy) / radial;
    }
}

// Bilinear sample from float32 image, returns pixel value at (x, y)
static void bilinearSample(const Image &img, float x, float y, float out[3]) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to image bounds
    x0 = std::clamp(x0, 0, img.width - 1);
    x1 = std::clamp(x1, 0, img.width - 1);
    y0 = std::clamp(y0, 0, img.height - 1);
    y1 = std::clamp(y1, 0, img.height - 1);

    float fx = x - std::floor(x);
    float fy = y - std::floor(y);

    const float *p00 = &img.data[(y0 * img.width + x0) * 3];
    const float *p10 = &img.data[(y0 * img.width + x1) * 3];
    const float *p01 = &img.data[(y1 * img.width + x0) * 3];
    const float *p11 = &img.data[(y1 * img.width + x1) * 3];

    for (int c = 0; c < 3; c++) {
        float top    = p00[c] * (1.0f - fx) + p10[c] * fx;
        float bottom = p01[c] * (1.0f - fx) + p11[c] * fx;
        out[c] = top * (1.0f - fy) + bottom * fy;
    }
}

UndistortResult undistortImage(const Image &src,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3)
{
    int w = src.width, h = src.height;

    // Find valid region by undistorting boundary points of the source image.
    // For each point on the distorted boundary, find its undistorted position.
    // The inner rectangle of all undistorted boundary points = valid region (alpha=0).
    float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
    int nSamples = 200;
    for (int i = 0; i < nSamples; i++) {
        float t = (float)i / (nSamples - 1);
        // Four edges of the distorted image
        float edges[][2] = {
            {t * w, 0.0f},          // top
            {t * w, (float)(h-1)},  // bottom
            {0.0f, t * h},          // left
            {(float)(w-1), t * h},  // right
        };
        for (auto &pt : edges) {
            float xd = (pt[0] - cx) / fx;
            float yd = (pt[1] - cy) / fy;
            float xu, yu;
            undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
            // Back to pixel coords using original intrinsics as the "new" camera
            float pu = xu * fx + cx;
            float pv = yu * fy + cy;
            minX = std::min(minX, pu);
            maxX = std::max(maxX, pu);
            minY = std::min(minY, pv);
            maxY = std::max(maxY, pv);
        }
    }

    // Inner rectangle: clamp to image bounds and take the inner edges
    // For top/left edges: take the max (inner boundary)
    // For bottom/right edges: take the min (inner boundary)
    // But we need to separate inner from outer per edge...
    // Top edge gives us maxY from top → that's minY constraint
    // Bottom edge gives us minY from bottom → that's maxY constraint
    // Actually, let me resample per-edge:
    float topMax = -1e9f, bottomMin = 1e9f, leftMax = -1e9f, rightMin = 1e9f;
    for (int i = 0; i < nSamples; i++) {
        float t = (float)i / (nSamples - 1);

        // Top edge: all points along y=0
        float xd = (t * w - cx) / fx, yd = (0.0f - cy) / fy;
        float xu, yu;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        topMax = std::max(topMax, yu * fy + cy);

        // Bottom edge: all points along y=h-1
        xd = (t * w - cx) / fx; yd = ((float)(h-1) - cy) / fy;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        bottomMin = std::min(bottomMin, yu * fy + cy);

        // Left edge: all points along x=0
        xd = (0.0f - cx) / fx; yd = (t * h - cy) / fy;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        leftMax = std::max(leftMax, xu * fx + cx);

        // Right edge: all points along x=w-1
        xd = ((float)(w-1) - cx) / fx; yd = (t * h - cy) / fy;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        rightMin = std::min(rightMin, xu * fx + cx);
    }

    // Inner rectangle (alpha=0: no black borders)
    int roiX = std::max(0, (int)std::ceil(leftMax));
    int roiY = std::max(0, (int)std::ceil(topMax));
    int roiW = std::min(w, (int)std::floor(rightMin)) - roiX;
    int roiH = std::min(h, (int)std::floor(bottomMin)) - roiY;
    if (roiW <= 0 || roiH <= 0) { roiX = 0; roiY = 0; roiW = w; roiH = h; }

    // Undistort: for each pixel in the output (undistorted) image,
    // apply forward distortion to find source pixel in distorted input
    Image undist;
    undist.width = w;
    undist.height = h;
    undist.data.resize(w * h * 3);

    for (int oy = 0; oy < h; oy++) {
        for (int ox = 0; ox < w; ox++) {
            float x = ((float)ox - cx) / fx;
            float y = ((float)oy - cy) / fy;
            float xd_n, yd_n;
            distortPoint(x, y, k1, k2, p1, p2, k3, xd_n, yd_n);
            float srcX = xd_n * fx + cx;
            float srcY = yd_n * fy + cy;

            float pixel[3];
            bilinearSample(src, srcX, srcY, pixel);
            float *out = &undist.data[(oy * w + ox) * 3];
            out[0] = pixel[0];
            out[1] = pixel[1];
            out[2] = pixel[2];
        }
    }

    // Crop to ROI
    Image cropped;
    cropped.width = roiW;
    cropped.height = roiH;
    cropped.data.resize(roiW * roiH * 3);
    for (int y = 0; y < roiH; y++) {
        memcpy(&cropped.data[y * roiW * 3],
               &undist.data[((y + roiY) * w + roiX) * 3],
               roiW * 3 * sizeof(float));
    }

    UndistortResult result;
    result.image = std::move(cropped);
    result.fx = fx;
    result.fy = fy;
    result.cx = cx - roiX;
    result.cy = cy - roiY;
    result.width = roiW;
    result.height = roiH;
    return result;
}
