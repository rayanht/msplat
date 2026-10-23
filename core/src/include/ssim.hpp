#ifndef SSIM_H
#define SSIM_H

#include <vector>
#include <cmath>
#include "metal_tensor.hpp"

// SSIM window creation and CPU evaluation.
// Training uses Metal kernels (ssim_h/v_fwd/bwd) directly.

// Create 11x11 Gaussian window for Metal SSIM loss kernel.
// Returns flat float vector (121 elements).
std::vector<float> createSSIMWindow(int windowSize = 11, float sigma = 1.5f);

// CPU SSIM evaluation for metrics (separable Gaussian blur).
// Images are (H, W, 3) float32 in [0,1].
float ssim_eval(const MTensor& rendered, const MTensor& gt,
                int windowSize = 11, float sigma = 1.5f);

#endif
