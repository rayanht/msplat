// SSIM — CPU evaluation and window creation
// Ported from https://github.com/Po-Hsun-Su/pytorch-ssim (MIT)

#include "ssim.hpp"
#include <cstring>

std::vector<float> createSSIMWindow(int windowSize, float sigma) {
    // 1D Gaussian
    std::vector<float> g(windowSize);
    float sum = 0;
    for (int i = 0; i < windowSize; i++) {
        float x = (float)(i - windowSize / 2);
        g[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += g[i];
    }
    for (int i = 0; i < windowSize; i++) g[i] /= sum;

    // 2D = outer product
    std::vector<float> w(windowSize * windowSize);
    for (int i = 0; i < windowSize; i++)
        for (int j = 0; j < windowSize; j++)
            w[i * windowSize + j] = g[i] * g[j];
    return w;
}

// Separable Gaussian blur on a single-channel (H, W) image.
// Writes result to `out`. `tmp` is scratch space (H * W floats).
static void gaussianBlur(const float* in, float* out, float* tmp,
                         int H, int W, const float* kernel, int kSize) {
    int pad = kSize / 2;

    // Horizontal pass: in → tmp
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0;
            for (int k = 0; k < kSize; k++) {
                int sx = x + k - pad;
                if (sx < 0) sx = 0;
                if (sx >= W) sx = W - 1;
                sum += in[y * W + sx] * kernel[k];
            }
            tmp[y * W + x] = sum;
        }
    }

    // Vertical pass: tmp → out
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0;
            for (int k = 0; k < kSize; k++) {
                int sy = y + k - pad;
                if (sy < 0) sy = 0;
                if (sy >= H) sy = H - 1;
                sum += tmp[sy * W + x] * kernel[k];
            }
            out[y * W + x] = sum;
        }
    }
}

float ssim_eval(const MTensor& rendered, const MTensor& gt,
                int windowSize, float sigma) {
    int H = rendered.size(0);
    int W = rendered.size(1);
    int C = rendered.size(2);
    int HW = H * W;

    // 1D Gaussian kernel
    std::vector<float> kernel(windowSize);
    float ksum = 0;
    for (int i = 0; i < windowSize; i++) {
        float x = (float)(i - windowSize / 2);
        kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        ksum += kernel[i];
    }
    for (int i = 0; i < windowSize; i++) kernel[i] /= ksum;

    const float* r = rendered.data<float>();
    const float* g = gt.data<float>();

    // Scratch buffers
    std::vector<float> r_ch(HW), g_ch(HW), rr(HW), gg(HW), rg(HW);
    std::vector<float> mu1(HW), mu2(HW), s_rr(HW), s_gg(HW), s_rg(HW);
    std::vector<float> tmp(HW);

    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;
    double ssim_sum = 0;
    int count = 0;

    for (int c = 0; c < C; c++) {
        // Extract channel (HWC → planar)
        for (int i = 0; i < HW; i++) {
            r_ch[i] = r[i * C + c];
            g_ch[i] = g[i * C + c];
            rr[i] = r_ch[i] * r_ch[i];
            gg[i] = g_ch[i] * g_ch[i];
            rg[i] = r_ch[i] * g_ch[i];
        }

        gaussianBlur(r_ch.data(), mu1.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(g_ch.data(), mu2.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(rr.data(), s_rr.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(gg.data(), s_gg.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(rg.data(), s_rg.data(), tmp.data(), H, W, kernel.data(), windowSize);

        for (int i = 0; i < HW; i++) {
            float m1sq = mu1[i] * mu1[i];
            float m2sq = mu2[i] * mu2[i];
            float m12  = mu1[i] * mu2[i];
            float sig1sq = s_rr[i] - m1sq;
            float sig2sq = s_gg[i] - m2sq;
            float sig12  = s_rg[i] - m12;

            float num = (2.0f * m12 + C1) * (2.0f * sig12 + C2);
            float den = (m1sq + m2sq + C1) * (sig1sq + sig2sq + C2);
            ssim_sum += num / den;
            count++;
        }
    }

    return (float)(ssim_sum / count);
}
