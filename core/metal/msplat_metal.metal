#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8
#define RAST_BLOCK_SIZE (RAST_BLOCK_X * RAST_BLOCK_Y)
#define CHANNELS 3
#define MAX_REGISTER_CHANNELS 3

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
constant float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

inline uint num_sh_bases(const uint degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

inline float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5;
}

inline void get_bbox(
    const float2 center,
    const float2 dims,
    const int3 img_size,
    thread uint2 &bb_min,
    thread uint2 &bb_max
) {
    // Clamp axis-aligned bounding box to valid range [0, img_size).
    // Returns inclusive min, exclusive max.
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline void get_tile_bbox(
    const float2 pix_center,
    const float2 pix_radius,
    const int3 tile_bounds,
    thread uint2 &tile_min,
    thread uint2 &tile_max
) {
    // Convert pixel-space center/radius to tile coordinates and compute AABB.
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius.x / (float)BLOCK_X, pix_radius.y / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// Affine transform: mat (row-major 4x3) applied to point p.
inline float3 transform_4x3(constant float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// Full 4x4 row-major transform, returns homogeneous coordinates.
inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

// Normalized quaternion → 3x3 rotation matrix (column-major for Metal).
inline float3x3 quat_to_rotmat(const float4 quat) {
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    return float3x3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// Returns true if point is behind the near plane (should be culled).
inline bool clip_near_plane(
    const float3 p, 
    constant float *viewmat, 
    thread float3 &p_view, 
    float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline float3x3 scale_to_mat(const float3 scale, const float glob_scale) {
    float3x3 S = float3x3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// Build 3D covariance matrix from scale + quaternion: cov = R*S*S^T*R^T.
// Stores upper triangle (6 floats) since the matrix is symmetric.
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, device float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);

    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);

    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// Thread-local overload: writes cov3d to registers instead of device memory
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, thread float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// Project 3D covariance to 2D via EWA splatting.
// Takes pre-computed view-space position; exploits J sparsity (5/9 nonzero).
float3 project_cov3d_ewa(
    device float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 p_view
) {
    // Clamp view-space position to avoid extreme covariance at FOV edges
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    // T = J * W where J has only 5 nonzero entries.
    // Instead of full 3x3 matmul, compute T rows directly.
    // T_row0 = j00 * M_row0 + j20 * M_row2 (viewmat is row-major)
    // T_row1 = j11 * M_row1 + j21 * M_row2
    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2;
    float j21 = -fy * p_view.y * rz2;

    float3 mr0 = float3(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1 = float3(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2 = float3(viewmat[8], viewmat[9], viewmat[10]);

    float3 t0 = j00 * mr0 + j20 * mr2;  // T row 0
    float3 t1 = j11 * mr1 + j21 * mr2;  // T row 1

    // cov2d = T * V * T^T, upper-left 2x2 only (3 values)
    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    return float3(dot(tv0, t0) + 0.3f, dot(tv0, t1), dot(tv1, t1) + 0.3f);
}

// Thread-local overload: reads cov3d from registers
float3 project_cov3d_ewa(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2;
    float j21 = -fy * p_view.y * rz2;

    float3 mr0 = float3(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1 = float3(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2 = float3(viewmat[8], viewmat[9], viewmat[10]);

    float3 t0 = j00 * mr0 + j20 * mr2;
    float3 t1 = j11 * mr1 + j21 * mr2;

    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    return float3(dot(tv0, t0) + 0.3f, dot(tv0, t1), dot(tv1, t1) + 0.3f);
}

inline bool compute_cov2d_bounds(
    const float3 cov2d, 
    thread float3 &conic, 
    thread float &radius
) {
    // Invert 2x2 covariance (upper triangle in cov2d.xyz) to get the conic,
    // and compute the gaussian's screen-space radius from eigenvalues (3-sigma).
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // Conic = inverse of 2x2 covariance
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float disc = sqrt(max(0.1f, b * b - det));
    // 3-sigma radius from the larger eigenvalue
    radius = ceil(3.f * sqrt(b + disc));
    return true;
}

// Project 3D point to pixel coordinates via projection matrix.
inline float2 project_pix(
    constant float *mat, const float3 p, const uint2 img_size, const float2 pp
) {
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, (int)img_size.x, pp.x), ndc2pix(p_proj.y, (int)img_size.y, pp.y)
    };
}

// Metal pads vector types in arrays (e.g. float3 → 16 bytes). These helpers
// read/write contiguous packed data by indexing into the underlying scalar buffer.

inline int2 read_packed_int2(constant int* arr, int idx) {
    return int2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_int2(device int* arr, int idx, int2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline void write_packed_int2x(device int* arr, int idx, int x) {
    arr[2*idx] = x;
}

inline void write_packed_int2y(device int* arr, int idx, int y) {
    arr[2*idx+1] = y;
}

inline float2 read_packed_float2(constant float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline float2 read_packed_float2(device float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_float2(device float* arr, int idx, float2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline int3 read_packed_int3(constant int* arr, int idx) {
    return int3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_int3(device int* arr, int idx, int3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float3 read_packed_float3(constant float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline float3 read_packed_float3(device float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline float3 read_packed_float3(device const float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_float3(device float* arr, int idx, float3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float4 read_packed_float4(constant float* arr, int idx) {
    return float4(arr[4*idx], arr[4*idx+1], arr[4*idx+2], arr[4*idx+3]);
}

inline void write_packed_float4(device float* arr, int idx, float4 val) {
    arr[4*idx] = val.x;
    arr[4*idx+1] = val.y;
    arr[4*idx+2] = val.z;
    arr[4*idx+3] = val.w;
}

// Forward projection: one thread per gaussian. Computes 2D position, conic, radius.
kernel void project_gaussians_forward_kernel(
    constant int& num_points,
    constant float* means3d, // float3
    constant float* scales, // float3
    constant float& glob_scale,
    constant float* quats, // float4
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant uint3& tile_bounds,
    constant float& clip_thresh,
    device float* covs3d,
    device float* xys, // float2
    device float* depths,
    device int* radii,
    device float* conics, // float3
    device int32_t* num_tiles_hit,
    device float* aabb, // float2: per-axis pixel extents
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = read_packed_float3(means3d, idx);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        return;
    }

    // compute the projected covariance
    // scales are in log-space; exp() here to avoid a separate MPS dispatch
    float3 scale = exp(read_packed_float3(scales, idx));
    float4 quat = read_packed_float4(quats, idx);
    device float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5f * img_size.x / fx;
    float tan_fovy = 0.5f * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, p_view
    );

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) {
        return; // zero determinant
    }
    write_packed_float3(conics, idx, conic);

    float aabb_x = ceil(3.0f * sqrt(cov2d.x));
    float aabb_y = ceil(3.0f * sqrt(cov2d.z));

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, float2(aabb_x, aabb_y), (int3)tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    write_packed_float2(xys, idx, center);
    aabb[idx * 2] = aabb_x;
    aabb[idx * 2 + 1] = aabb_y;
}

kernel void nd_rasterize_forward_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int* tile_bins, // int2
    constant float* packed_xy_opac, // float3: (x, y, sigmoid(opacity))
    constant float* packed_conic,   // float3
    constant float* packed_rgb,     // float3: raw SH (NOT clamped)
    device float* final_Ts,
    device int* final_index,
    device float* out_img,
    constant float* background,
    constant uint2& blockDim,
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]]
) {
    // Threadgroup-batched forward rasterization: all threads in a tile
    // cooperatively load Gaussian data into shared memory, then read from it.
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * (int)img_size.x + j;

    const bool inside = (i < (int)img_size.y && j < (int)img_size.x);

    // which gaussians to look through in this tile
    int2 range = read_packed_int2(tile_bins, tile_id);
    const int num_batches = (range.y - range.x + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    // threadgroup shared memory for batch loading
    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];

    float T = 1.f;
    float3 pix_out = {0.f, 0.f, 0.f};
    int last_contributor = range.x - 1;
    bool done = false;

    for (int b = 0; b < num_batches; ++b) {
        // sync before loading next batch
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each thread loads one gaussian into shared memory
        int batch_start = range.x + RAST_BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            // Sequential reads from packed sorted-order buffers
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_float3(packed_conic, idx);
            // packed_rgb has raw SH output — clamp_min(raw + 0.5, 0)
            const float3 raw_c = read_packed_float3(packed_rgb, idx);
            rgbs_batch[tr] = max(raw_c + 0.5f, 0.0f);
        }
        // wait for all threads to finish loading
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (done || !inside) continue;

        int batch_size = min(RAST_BLOCK_SIZE, range.y - batch_start);

        // process gaussians in this batch
        for (int t = 0; t < batch_size; ++t) {
            const float3 conic_local = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};

            const float sigma = fma(0.5f,
                fma(conic_local.x, delta.x * delta.x, conic_local.z * delta.y * delta.y),
                conic_local.y * delta.x * delta.y);
            // Early-out: skip exp() when the result is guaranteed to be discarded.
            // alpha = min(0.999, opacity * exp(-sigma)), discarded when alpha < 1/255.
            // opacity = sigmoid(raw) ∈ [0,1], so alpha ≤ exp(-sigma).
            // exp(-5.55) ≈ 0.00389 < 1/255 ≈ 0.00392, so sigma ≥ 5.55 ⟹ alpha < 1/255.
            // Empirically 94% of evaluations have sigma ≥ 5.55 (garden scene, mipnerf360).
            if (sigma < 0.f || sigma >= 5.55f) {
                continue;
            }

            const float alpha = min(0.999f, xy_opac.z * exp(-sigma));
            if (alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) {
                last_contributor = batch_start + t - 1;
                done = true;
                break;
            }

            const float vis = alpha * T;
            const float3 rgb = rgbs_batch[t];
            pix_out = fma(rgb, vis, pix_out);
            T = next_T;
            last_contributor = batch_start + t;
        }
    }

    if (inside) {
        final_Ts[pix_id] = T;
        final_index[pix_id] = last_contributor;
        // Fused clamp_max(output, 1.0) — saturate clamps to [0,1]
        float3 bg = {background[0], background[1], background[2]};
        float3 final_rgb = saturate(fma(bg, T, pix_out));
        out_img[CHANNELS * pix_id + 0] = final_rgb.x;
        out_img[CHANNELS * pix_id + 1] = final_rgb.y;
        out_img[CHANNELS * pix_id + 2] = final_rgb.z;
    }
}

void sh_coeffs_to_color(
    const uint degree,
    const float3 viewdir,
    constant float *dc_coeffs,
    constant float *rest_coeffs,
    device float *colors
) {
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * dc_coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    // viewdir is already normalized by caller (normalize() in project_and_sh_forward_kernel etc.)
    float x = viewdir.x;
    float y = viewdir.y;
    float z = viewdir.z;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * rest_coeffs[0 * CHANNELS + c] +
                              z * rest_coeffs[1 * CHANNELS + c] -
                              x * rest_coeffs[2 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * rest_coeffs[3 * CHANNELS + c] +
             SH_C2[1] * yz * rest_coeffs[4 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * rest_coeffs[5 * CHANNELS + c] +
             SH_C2[3] * xz * rest_coeffs[6 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * rest_coeffs[7 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * rest_coeffs[8 * CHANNELS + c] +
             SH_C3[1] * xy * z * rest_coeffs[9 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * rest_coeffs[10 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 rest_coeffs[11 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * rest_coeffs[12 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * rest_coeffs[13 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * rest_coeffs[14 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * rest_coeffs[15 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * rest_coeffs[16 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * rest_coeffs[17 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * rest_coeffs[18 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 rest_coeffs[19 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * rest_coeffs[20 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 rest_coeffs[21 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * rest_coeffs[22 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 rest_coeffs[23 * CHANNELS + c]);
    }
}

void sh_coeffs_to_color_vjp(
    const uint degree,
    const float3 viewdir,
    constant float *v_colors,
    device float *v_dc_coeffs,
    device float *v_rest_coeffs
) {
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_dc_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    // viewdir is already normalized by caller
    float x = viewdir.x;
    float y = viewdir.y;
    float z = viewdir.z;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;

    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_rest_coeffs[0 * CHANNELS + c] = v1 * v_colors[c];
        v_rest_coeffs[1 * CHANNELS + c] = v2 * v_colors[c];
        v_rest_coeffs[2 * CHANNELS + c] = v3 * v_colors[c];
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_rest_coeffs[3 * CHANNELS + c] = v4 * v_colors[c];
        v_rest_coeffs[4 * CHANNELS + c] = v5 * v_colors[c];
        v_rest_coeffs[5 * CHANNELS + c] = v6 * v_colors[c];
        v_rest_coeffs[6 * CHANNELS + c] = v7 * v_colors[c];
        v_rest_coeffs[7 * CHANNELS + c] = v8 * v_colors[c];
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_rest_coeffs[8 * CHANNELS + c] = v9 * v_colors[c];
        v_rest_coeffs[9 * CHANNELS + c] = v10 * v_colors[c];
        v_rest_coeffs[10 * CHANNELS + c] = v11 * v_colors[c];
        v_rest_coeffs[11 * CHANNELS + c] = v12 * v_colors[c];
        v_rest_coeffs[12 * CHANNELS + c] = v13 * v_colors[c];
        v_rest_coeffs[13 * CHANNELS + c] = v14 * v_colors[c];
        v_rest_coeffs[14 * CHANNELS + c] = v15 * v_colors[c];
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_rest_coeffs[15 * CHANNELS + c] = v16 * v_colors[c];
        v_rest_coeffs[16 * CHANNELS + c] = v17 * v_colors[c];
        v_rest_coeffs[17 * CHANNELS + c] = v18 * v_colors[c];
        v_rest_coeffs[18 * CHANNELS + c] = v19 * v_colors[c];
        v_rest_coeffs[19 * CHANNELS + c] = v20 * v_colors[c];
        v_rest_coeffs[20 * CHANNELS + c] = v21 * v_colors[c];
        v_rest_coeffs[21 * CHANNELS + c] = v22 * v_colors[c];
        v_rest_coeffs[22 * CHANNELS + c] = v23 * v_colors[c];
        v_rest_coeffs[23 * CHANNELS + c] = v24 * v_colors[c];
    }
}

kernel void compute_sh_forward_kernel(
    constant uint& num_points,
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float* means3d, // float3
    constant float3& cam_pos,
    constant float* coeffs,
    device float* colors,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) {
        return;
    }
    // Compute view direction from means and camera position (fused, avoids 3 MPS ops)
    float3 viewdir = normalize(read_packed_float3(means3d, idx) - cam_pos);

    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint idx_sh = num_bases * num_channels * idx;
    uint idx_col = num_channels * idx;

    sh_coeffs_to_color(
        degrees_to_use, viewdir, &(coeffs[idx_sh]), &(coeffs[idx_sh + CHANNELS]), &(colors[idx_col])
    );
}

kernel void compute_sh_backward_kernel(
    constant uint& num_points,
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float* means3d, // float3
    constant float3& cam_pos,
    constant float* v_colors,
    device float* v_coeffs,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) {
        return;
    }
    // Recompute view direction (same as forward)
    float3 viewdir = normalize(read_packed_float3(means3d, idx) - cam_pos);

    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint idx_sh = num_bases * num_channels * idx;
    uint idx_col = num_channels * idx;

    sh_coeffs_to_color_vjp(
        degrees_to_use, viewdir, &(v_colors[idx_col]), &(v_coeffs[idx_sh]), &(v_coeffs[idx_sh + CHANNELS])
    );
}

// Build (tile_id, depth) pairs for each gaussian-tile intersection.
kernel void map_gaussian_to_intersects_kernel(
    constant int& num_points,
    constant float* xys, // float2
    constant float* depths,
    constant int* radii,
    constant int32_t* num_tiles_hit,
    constant uint3& tile_bounds,
    constant uint& capacity,
    device int64_t* isect_ids,
    device int32_t* gaussian_ids,
    constant float* aabb, // float2: per-axis pixel extents
    device atomic_uint* overflow_flag, // set to 1 if any intersection exceeds capacity
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian using AABB extents
    uint2 tile_min, tile_max;
    float2 center = read_packed_float2(xys, idx);
    get_tile_bbox(center, read_packed_float2(aabb, idx), (int3)tile_bounds, tile_min, tile_max);

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : num_tiles_hit[idx - 1];
    // Compressed sort key: tile_id in bits [16:31], upper 16 bits of float depth in bits [0:15].
    // Upper 16 bits of positive float preserve ordering (sign+exponent+7 mantissa bits).
    // Reduces effective key width from ~48 to ~28 bits → 4 radix passes instead of 6.
    int64_t depth_16 = ((int64_t) * (constant int32_t *)&(depths[idx])) >> 16;
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            if ((uint)cur_idx >= capacity) {
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                return;
            }
            int64_t tile_id = i * tile_bounds.x + j;
            isect_ids[cur_idx] = (tile_id << 16) | (depth_16 & 0xFFFF);
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
}

// Find start/end offsets for each tile in the sorted intersection array.
kernel void get_tile_bin_edges_kernel(
    constant uint& capacity,
    constant int64_t* isect_ids_sorted,
    device int* tile_bins, // int2
    device const int32_t* cum_tiles_hit,
    constant uint& num_points,
    uint idx [[thread_position_in_grid]]
) {
    // Read actual intersection count from GPU-resident prefix sum
    uint num_intersects = min(capacity, (uint)cum_tiles_hit[num_points - 1]);
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    // Extract tile_id from compressed key: tile_id is in bits [16:]
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 16);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            write_packed_int2x(tile_bins, cur_tile_idx, 0);
        if (idx == num_intersects - 1)
            write_packed_int2y(tile_bins, cur_tile_idx, num_intersects);
        return;
    }
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 16);
    if (prev_tile_idx != cur_tile_idx) {
        write_packed_int2y(tile_bins, prev_tile_idx, idx);
        write_packed_int2x(tile_bins, cur_tile_idx, idx);
        return;
    }
}

inline int warp_reduce_all_max(int val, const int warp_size) {
    return simd_max(val);
}

inline int warp_reduce_all_or(int val, const int warp_size) {
    return simd_or(val);
}

inline float3 warpSum3(float3 val, const int warp_size, const uint lane) {
    val.x = simd_sum(val.x);
    val.y = simd_sum(val.y);
    val.z = simd_sum(val.z);
    return val;
}

inline float2 warpSum2(float2 val, const int warp_size, const uint lane) {
    val.x = simd_sum(val.x);
    val.y = simd_sum(val.y);
    return val;
}

inline float warpSum(float val, const int warp_size, const uint lane) {
    return simd_sum(val);
}

kernel void rasterize_backward_kernel(
    constant uint3& tile_bounds,
    constant uint2& img_size,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins, // int2
    constant float* packed_xy_opac, // float3: (x, y, sigmoid(opacity))
    constant float* packed_conic,   // float3
    constant float* packed_rgb,     // float3: raw SH
    constant float* background, // single float3
    constant float* final_Ts,
    constant int* final_index,
    constant float* v_output, // float3
    device atomic_float* v_xy, // float2
    device atomic_float* v_conic, // float3
    device atomic_float* v_rgb, // float3
    device atomic_float* v_opacity,
    uint3 gp [[thread_position_in_grid]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint tr [[thread_index_in_threadgroup]],
    uint warp_size [[threads_per_simdgroup]],
    uint wr [[thread_index_in_simdgroup]]
) {
    uint i = gp.y;
    uint j = gp.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min((int32_t)(i * img_size.x + j), (int32_t)(img_size.x * img_size.y - 1));

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = read_packed_int2(tile_bins, tile_id);
    const int num_batches = (range.y - range.x + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    threadgroup int32_t id_batch[RAST_BLOCK_SIZE];
    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = read_packed_float3(v_output, pix_id);
    // Hoist loop-invariant background load and T_final * bg product
    const float3 bg = {background[0], background[1], background[2]};
    const float3 T_final_bg = T_final * bg;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int warp_bin_final = warp_reduce_all_max(bin_final, warp_size);

    // Subtile-level early exit: compute max bin_final across all warps,
    // skip leading batches where all gaussians are beyond any pixel's bin_final.
    const uint warp_id = tr / warp_size;
    constexpr uint NUM_WARPS = RAST_BLOCK_SIZE / 32;
    threadgroup int warp_max_finals[NUM_WARPS];
    if (wr == 0) warp_max_finals[warp_id] = warp_bin_final;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int tile_max_bin_final = warp_max_finals[0];
    for (uint w = 1; w < NUM_WARPS; w++)
        tile_max_bin_final = max(tile_max_bin_final, warp_max_finals[w]);
    int dead_count = max(0, (int)(range.y - 1) - tile_max_bin_final);
    int first_batch = dead_count / RAST_BLOCK_SIZE;

    for (int b = first_batch; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - RAST_BLOCK_SIZE * b;
        int batch_size = min(RAST_BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            id_batch[tr] = gaussian_ids_sorted[idx];
            // Sequential reads from packed sorted-order buffers
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_float3(packed_conic, idx);
            rgbs_batch[tr] = read_packed_float3(packed_rgb, idx);
        }
        // wait for other threads to collect the gaussians in batch
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            // Broadcast batch data from lane 0 → all lanes in SIMD group.
            // All threads read the same index t, so one threadgroup read +
            // simd_broadcast replaces 32 redundant threadgroup reads.
            float3 b_conic, b_xy_opac, b_rgb;
            int32_t b_id;
            if (wr == 0) {
                b_conic = conic_batch[t];
                b_xy_opac = xy_opacity_batch[t];
                b_rgb = rgbs_batch[t];
                b_id = id_batch[t];
            }
            b_conic = simd_broadcast(b_conic, 0);
            b_xy_opac = simd_broadcast(b_xy_opac, 0);
            b_rgb = simd_broadcast(b_rgb, 0);
            b_id = simd_broadcast(b_id, 0);

            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float vis;
            if(valid){
                opac = b_xy_opac.z;
                delta = {b_xy_opac.x - px, b_xy_opac.y - py};
                float sigma = fma(0.5f,
                    fma(b_conic.x, delta.x * delta.x, b_conic.z * delta.y * delta.y),
                    b_conic.y * delta.x * delta.y);
                if (sigma < 0.f || sigma >= 5.55f) {
                    valid = 0;
                } else {
                    vis = exp(-sigma);
                    alpha = min(0.999f, opac * vis);
                    if (alpha < 1.f / 255.f) {
                        valid = 0;
                    }
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if (!warp_reduce_all_or(valid, warp_size)) {
                continue;
            }

            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid && alpha<0.999f){
                // compute the current T for this gaussian
                // alpha = opac * vis (guaranteed since alpha < 0.99 = min cap)
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = fac * v_out;

                // b_rgb has raw SH output; clamp inline: max(raw + 0.5, 0)
                const float3 rgb = max(b_rgb + 0.5f, 0.f);
                // contribution from this pixel + background
                v_alpha += dot(fma(rgb, T, fma(-buffer, ra, -ra * T_final_bg)), v_out);
                // update the running sum
                buffer = fma(rgb, fac, buffer);

                // v_sigma = d(loss)/d(sigma) = -alpha * v_alpha
                const float v_sigma = -alpha * v_alpha;
                v_conic_local = (0.5f * v_sigma) * float3(delta.x * delta.x,
                                                           delta.x * delta.y,
                                                           delta.y * delta.y);
                v_xy_local = v_sigma * float2(
                    fma(b_conic.x, delta.x, b_conic.y * delta.y),
                    fma(b_conic.y, delta.x, b_conic.z * delta.y));
                // Fused sigmoid derivative: dL/d(logit) = -v_sigma * (1 - opac)
                v_opacity_local = -v_sigma * (1.f - opac);
            }

            v_rgb_local = warpSum3(v_rgb_local, warp_size, wr);
            v_conic_local = warpSum3(v_conic_local, warp_size, wr);
            v_xy_local = warpSum2(v_xy_local, warp_size, wr);
            v_opacity_local = warpSum(v_opacity_local, warp_size, wr);

            if (wr == 0) {
                // Fused clamp_min backward: zero gradient where raw_color + 0.5 < 0
                if (b_rgb.x + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 0, v_rgb_local.x, memory_order_relaxed);
                if (b_rgb.y + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 1, v_rgb_local.y, memory_order_relaxed);
                if (b_rgb.z + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 2, v_rgb_local.z, memory_order_relaxed);

                atomic_fetch_add_explicit(v_conic + 3*b_id + 0, v_conic_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*b_id + 1, v_conic_local.y, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*b_id + 2, v_conic_local.z, memory_order_relaxed);

                atomic_fetch_add_explicit(v_xy + 2*b_id + 0, v_xy_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_xy + 2*b_id + 1, v_xy_local.y, memory_order_relaxed);

                atomic_fetch_add_explicit(v_opacity + b_id, v_opacity_local, memory_order_relaxed);
            }
        }
    }
}

kernel void nd_rasterize_backward_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int32_t* gaussians_ids_sorted,
    constant int* tile_bins, // int2
    constant float* xys, // float2
    constant float* conics, // float3
    constant float* rgbs,
    constant float* opacities,
    constant float* background,
    constant float* final_Ts,
    constant int* final_index,
    constant float* v_output,
    device atomic_float* v_xy, // float2
    device atomic_float* v_conic, // float3
    device atomic_float* v_rgb,
    device atomic_float* v_opacity,
    device float* workspace,
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 threadIdx [[thread_position_in_threadgroup]]
) {
    if (channels > MAX_REGISTER_CHANNELS && workspace == nullptr) {
        return;
    }
    // Per-pixel backward pass (no shared-memory batching)
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians get gradients for this pixel
    int2 range = read_packed_int2(tile_bins, tile_id);
    // df/d_out for this pixel
    constant float *v_out = &(v_output[channels * pix_id]);
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    device float *S = &workspace[channels * pix_id];
    int bin_final = final_index[pix_id];

    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and
    // conic recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 -
    // alpha_j), and S_{n-1} from S_n, where S_j = sum_{i > j}(rgb_i * alpha_i *
    // T_i) df/dalpha_i = rgb_i * T_i - S_{i+1| / (1 - alpha_i)
    for (int idx = bin_final - 1; idx >= range.x; --idx) {
        const int32_t g = gaussians_ids_sorted[idx];
        const float3 conic = read_packed_float3(conics, g);
        const float2 center = read_packed_float2(xys, g);
        const float2 delta = {center.x - px, center.y - py};
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        // Early-out: skip exp() when the result is guaranteed to be discarded.
        // alpha = min(0.99, opacity * exp(-sigma)), discarded when alpha < 1/255.
        // opacity = sigmoid(raw) ∈ [0,1], so alpha ≤ exp(-sigma).
        // exp(-5.55) ≈ 0.00389 < 1/255 ≈ 0.00392, so sigma ≥ 5.55 ⟹ alpha < 1/255.
        if (sigma < 0.f || sigma >= 5.55f) {
            continue;
        }
        const float opac = opacities[g];
        const float vis = exp(-sigma);
        const float alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        // compute the current T for this gaussian
        const float ra = 1.f / (1.f - alpha);
        T *= ra;
        // rgb = rgbs[g];
        // update v_rgb for this gaussian
        const float fac = alpha * T;
        float v_alpha = 0.f;
        for (int c = 0; c < channels; ++c) {
            // gradient wrt rgb
            atomic_fetch_add_explicit(v_rgb + channels * g + c, fac * v_out[c], memory_order_relaxed);
            // contribution from this pixel
            v_alpha += (rgbs[channels * g + c] * T - S[c] * ra) * v_out[c];
            // contribution from background pixel
            v_alpha += -T_final * ra * background[c] * v_out[c];
            // update the running sum
            S[c] += rgbs[channels * g + c] * fac;
        }
        // update v_opacity for this gaussian
        atomic_fetch_add_explicit(v_opacity + g, vis * v_alpha, memory_order_relaxed);

        // compute vjps for conics and means
        // d_sigma / d_delta = conic * delta
        // d_sigma / d_conic = delta * delta.T
        const float v_sigma = -opac * vis * v_alpha;

        atomic_fetch_add_explicit(v_conic + 3*g + 0, 0.5f * v_sigma * delta.x * delta.x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conic + 3*g + 1, 0.5f * v_sigma * delta.x * delta.y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conic + 3*g + 2, 0.5f * v_sigma * delta.y * delta.y, memory_order_relaxed);
        atomic_fetch_add_explicit(
            v_xy + 2*g + 0, v_sigma * (conic.x * delta.x + conic.y * delta.y), memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            v_xy + 2*g + 1, v_sigma * (conic.y * delta.x + conic.z * delta.y), memory_order_relaxed
        );
    }
}

// given v_xy_pix, get v_xyz
inline float3 project_pix_vjp(
    constant float *mat, const float3 p, const uint2 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y, 0.0f};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

// compute vjp from df/d_conic to df/c_cov2d
inline void cov2d_to_conic_vjp(
    float3 conic, 
    float3 v_conic, 
    device float* v_cov2d // float3
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// Thread-local overload
inline void cov2d_to_conic_vjp(
    float3 conic,
    float3 v_conic,
    thread float* v_cov2d
) {
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// output space: 2D covariance, input space: cov3d
void project_cov3d_ewa_vjp(
    constant float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 v_cov2d,
    device float* v_mean3d,
    device float* v_cov3d,
    float3 p_view
) {
    // Apply same fov clipping as forward
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );

    float3x3 J = float3x3(
        fx * rz,                0.f,                0.f,
        0.f,                    fy * rz,            0.f,
        -fx * p_view.x * rz2,  -fy * p_view.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2;
    float fy_rz2 = fy * rz2;
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx_rz2 * v_J[2][0],
        -fy_rz2 * v_J[2][1],
        -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
            fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]
    );
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

// Thread-local overload: reads cov3d from registers, writes v_cov3d to registers
void project_cov3d_ewa_vjp(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 v_cov2d,
    device float* v_mean3d,
    thread float* v_cov3d,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );

    float3x3 J = float3x3(
        fx * rz,                0.f,                0.f,
        0.f,                    fy * rz,            0.f,
        -fx * p_view.x * rz2,  -fy * p_view.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2;
    float fy_rz2 = fy * rz2;
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx_rz2 * v_J[2][0],
        -fy_rz2 * v_J[2][1],
        -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
            fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]
    );
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

inline float4 quat_to_rotmat_vjp(const float4 quat, const float3x3 v_R) {
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const device float* v_cov3d,
    device float* v_scale, // float3
    device float* v_quat // float4
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}

// Thread-local overload: reads v_cov3d from registers
void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const thread float* v_cov3d,
    device float* v_scale,
    device float* v_quat
) {
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}

kernel void project_gaussians_backward_kernel(
    constant int& num_points,
    constant float* means3d, // float3
    constant float* scales, // float3
    constant float& glob_scale,
    constant float* quats, // float4
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant float* cov3d,
    constant int* radii,
    constant float* conics, // float3
    constant float* v_xy, // float2
    constant float* v_depth,
    constant float* v_conic, // float3
    device float* v_cov2d, // float3
    device float* v_cov3d,
    device float* v_mean3d, // float3
    device float* v_scale, // float3
    device float* v_quat, // float4
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = read_packed_float3(means3d, idx);
    float fx = intrins.x;
    float fy = intrins.y;
    // get v_mean3d from v_xy
    write_packed_float3(
        v_mean3d, idx, 
        project_pix_vjp(projmat, p_world, img_size, read_packed_float2(v_xy, idx))
    );

    // get z gradient contribution to mean3d gradient
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    float v_z = v_depth[idx];
    write_packed_float3(
        v_mean3d, idx, 
        read_packed_float3(v_mean3d, idx) + float3(viewmat[8], viewmat[9], viewmat[10]) * v_z
    );

    // get v_cov2d
    cov2d_to_conic_vjp(
        read_packed_float3(conics, idx), 
        read_packed_float3(v_conic, idx), 
        &(v_cov2d[3*idx])
    );
    // get v_cov3d (and v_mean3d contribution)
    float tan_fovx = 0.5f * (float)img_size.x / fx;
    float tan_fovy = 0.5f * (float)img_size.y / fy;
    float3 p_view = transform_4x3(viewmat, p_world);
    project_cov3d_ewa_vjp(
        &(cov3d[6 * idx]),
        viewmat,
        fx,
        fy,
        tan_fovx,
        tan_fovy,
        read_packed_float3(v_cov2d, idx),
        &(v_mean3d[3*idx]),
        &(v_cov3d[6 * idx]),
        p_view
    );
    // get v_scale and v_quat
    // scales are in log-space; exp() here and apply chain rule for dL/d(log_scale)
    float3 exp_scale = exp(read_packed_float3(scales, idx));
    scale_rot_to_cov3d_vjp(
        exp_scale,
        glob_scale,
        read_packed_float4(quats, idx),
        &(v_cov3d[6 * idx]),
        &(v_scale[3*idx]),
        &(v_quat[4*idx])
    );
    // chain rule: dL/d(log_s) = dL/d(exp_s) * exp(log_s)
    v_scale[3*idx + 0] *= exp_scale.x;
    v_scale[3*idx + 1] *= exp_scale.y;
    v_scale[3*idx + 2] *= exp_scale.z;
}

kernel void compute_cov2d_bounds_kernel(
    constant uint& num_pts, 
    constant float* covs2d, 
    device float* conics, 
    device float* radii,
    uint row [[thread_index_in_threadgroup]]
) {
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

// Fused Adam optimizer kernel: single-pass update for params, exp_avg, exp_avg_sq.
// Precomputed on CPU: step_size = lr / (1 - beta1^t), bc2_sqrt = sqrt(1 - beta2^t)
// Fused per-step gradient accumulation for densification.
// Replaces ~8 MPS dispatches (boolean mask, vector_norm, index_put_, max) with 1 kernel.
kernel void accumulate_grad_stats_kernel(
    constant int& num_points,
    constant int* radii [[buffer(1)]],
    constant float* xys_grad [[buffer(2)]],     // (N, 2) packed float2
    device float* vis_counts [[buffer(3)]],      // (N,) in-place
    device float* xys_grad_norm [[buffer(4)]],   // (N,) in-place
    device float* max_2d_size [[buffer(5)]],     // (N,) in-place
    constant float& inv_max_dim [[buffer(6)]],   // 1.0 / max(H, W)
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)num_points) return;
    if (radii[idx] <= 0) return;

    vis_counts[idx] += 1.0f;

    float gx = xys_grad[idx * 2];
    float gy = xys_grad[idx * 2 + 1];
    xys_grad_norm[idx] += sqrt(gx * gx + gy * gy);

    float r = (float)radii[idx] * inv_max_dim;
    max_2d_size[idx] = max(max_2d_size[idx], r);
}

kernel void fused_adam_kernel(
    device float * params [[buffer(0)]],
    device const float * grads [[buffer(1)]],
    device float * exp_avg [[buffer(2)]],
    device float * exp_avg_sq [[buffer(3)]],
    constant float & step_size [[buffer(4)]],
    constant float & beta1 [[buffer(5)]],
    constant float & beta2 [[buffer(6)]],
    constant float & bc2_sqrt [[buffer(7)]],
    constant float & eps [[buffer(8)]],
    constant uint & n [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float g = grads[tid];
    float m = fma(beta1, exp_avg[tid], (1.0f - beta1) * g);
    float v = fma(beta2, exp_avg_sq[tid], (1.0f - beta2) * g * g);

    params[tid] -= step_size * m / (sqrt(v) / bc2_sqrt + eps);

    exp_avg[tid] = m;
    exp_avg_sq[tid] = v;
}

// ===== Fused Projection + SH Kernels =====
// Combines project_gaussians_forward_kernel + compute_sh_forward_kernel into one dispatch.
// Saves 1 kernel dispatch + 1 read of means3d per direction. Skips SH for culled gaussians.

kernel void project_and_sh_forward_kernel(
    // Projection args
    constant int& num_points,
    constant float* means3d,
    constant float* scales,
    constant float& glob_scale,
    constant float* quats,
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant uint3& tile_bounds,
    constant float& clip_thresh,
    device float* xys,
    device float* depths,
    device int* radii,
    device float* conics,
    device int32_t* num_tiles_hit,
    // SH args
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float3& cam_pos,
    constant float* features_dc,
    constant float* features_rest,
    device float* colors,
    device float* aabb, // float2: per-axis pixel extents
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= (uint)num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = read_packed_float3(means3d, idx);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        return;
    }

    float3 scale = exp(read_packed_float3(scales, idx));
    float4 quat = read_packed_float4(quats, idx);
    // Compute cov3d in thread-local registers (no device memory round-trip)
    float local_cov3d[6];
    scale_rot_to_cov3d(scale, glob_scale, quat, local_cov3d);

    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5f * img_size.x / fx;
    float tan_fovy = 0.5f * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        local_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, p_view
    );

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) {
        return;
    }
    write_packed_float3(conics, idx, conic);

    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});

    float aabb_x = ceil(3.0f * sqrt(cov2d.x));
    float aabb_y = ceil(3.0f * sqrt(cov2d.z));
    uint2 tile_min, tile_max;
    get_tile_bbox(center, float2(aabb_x, aabb_y), (int3)tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);

    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    write_packed_float2(xys, idx, center);
    aabb[idx * 2] = aabb_x;
    aabb[idx * 2 + 1] = aabb_y;

    // SH: compute colors for non-culled gaussians (reuse p_world from registers)
    float3 viewdir = normalize(p_world - cam_pos);
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint dc_idx = num_channels * idx;
    uint rest_idx = (num_bases - 1) * num_channels * idx;
    uint idx_col = num_channels * idx;
    sh_coeffs_to_color(degrees_to_use, viewdir, &(features_dc[dc_idx]), &(features_rest[rest_idx]), &(colors[idx_col]));
}

// Adam update helper — applies one Adam step to a single element.
// Computes in registers, writes param/exp_avg/exp_avg_sq back to device memory.
inline void adam_update_element(
    device float& param, device float& ea, device float& eas,
    float grad, float step_size, float beta1, float beta2, float bc2_sqrt, float eps
) {
    float m = fma(beta1, ea, (1.0f - beta1) * grad);
    float v = fma(beta2, eas, (1.0f - beta2) * grad * grad);
    param -= step_size * m / (sqrt(v) / bc2_sqrt + eps);
    ea = m;
    eas = v;
}

// Packed Adam hyperparameters for SH groups (passed via setBytes)
struct SHAdamParams {
    float dc_step_size;
    float dc_bc2_sqrt;
    float rest_step_size;
    float rest_bc2_sqrt;
    float beta1;
    float beta2;
    float eps;
};

kernel void project_and_sh_backward_kernel(
    // Projection backward args
    constant int& num_points,
    constant float* means3d,
    constant float* scales,
    constant float& glob_scale,
    constant float* quats,
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant int* radii,
    constant float* conics,
    constant float* v_xy,
    constant float* v_depth,
    constant float* v_conic,
    device float* v_mean3d,
    device float* v_scale,
    device float* v_quat,
    // SH backward + fused Adam args
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float3& cam_pos,
    constant float* v_colors,
    device float* features_dc,         // params (read-write for Adam)
    device float* features_rest,       // params (read-write for Adam)
    device float* dc_exp_avg,          // Adam state
    device float* dc_exp_avg_sq,
    device float* rest_exp_avg,
    device float* rest_exp_avg_sq,
    constant SHAdamParams& adam_hp,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = read_packed_float3(means3d, idx);
    float fx = intrins.x;
    float fy = intrins.y;

    // Projection backward: v_mean3d from v_xy
    write_packed_float3(
        v_mean3d, idx,
        project_pix_vjp(projmat, p_world, img_size, read_packed_float2(v_xy, idx))
    );

    // z gradient contribution to mean3d
    float v_z = v_depth[idx];
    write_packed_float3(
        v_mean3d, idx,
        read_packed_float3(v_mean3d, idx) + float3(viewmat[8], viewmat[9], viewmat[10]) * v_z
    );

    // v_cov2d from v_conic (thread-local, no device memory round-trip)
    float local_v_cov2d[3];
    cov2d_to_conic_vjp(
        read_packed_float3(conics, idx),
        read_packed_float3(v_conic, idx),
        local_v_cov2d
    );

    // Recompute cov3d from scales+quats (avoids saving/reading 3.6MB tensor)
    float3 exp_scale = exp(read_packed_float3(scales, idx));
    float4 quat = read_packed_float4(quats, idx);
    float local_cov3d[6];
    scale_rot_to_cov3d(exp_scale, glob_scale, quat, local_cov3d);

    // v_cov3d (thread-local) and v_mean3d contribution
    float tan_fovx = 0.5f * (float)img_size.x / fx;
    float tan_fovy = 0.5f * (float)img_size.y / fy;
    float3 p_view = transform_4x3(viewmat, p_world);
    float local_v_cov3d[6];
    project_cov3d_ewa_vjp(
        local_cov3d,
        viewmat,
        fx,
        fy,
        tan_fovx,
        tan_fovy,
        float3(local_v_cov2d[0], local_v_cov2d[1], local_v_cov2d[2]),
        &(v_mean3d[3*idx]),
        local_v_cov3d,
        p_view
    );

    // v_scale and v_quat (reads v_cov3d from thread-local)
    scale_rot_to_cov3d_vjp(
        exp_scale,
        glob_scale,
        quat,
        local_v_cov3d,
        &(v_scale[3*idx]),
        &(v_quat[4*idx])
    );
    // Chain rule: dL/d(log_s) = dL/d(exp_s) * exp(log_s)
    v_scale[3*idx + 0] *= exp_scale.x;
    v_scale[3*idx + 1] *= exp_scale.y;
    v_scale[3*idx + 2] *= exp_scale.z;

    // ---- Fused SH backward + Adam ----
    // Compute SH gradients in registers and apply Adam inline.
    // Eliminates v_features_dc/v_features_rest write+read round-trip (~600 MB/iter at 1.6M gaussians).
    float3 viewdir = normalize(p_world - cam_pos);
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint dc_idx = num_channels * idx;
    uint rest_idx = (num_bases - 1) * num_channels * idx;
    uint idx_col = num_channels * idx;

    float vc[3] = { v_colors[idx_col], v_colors[idx_col + 1], v_colors[idx_col + 2] };

    // DC: grad = SH_C0 * v_colors[c]
    for (int c = 0; c < 3; c++) {
        float g = SH_C0 * vc[c];
        adam_update_element(features_dc[dc_idx + c], dc_exp_avg[dc_idx + c], dc_exp_avg_sq[dc_idx + c],
                           g, adam_hp.dc_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.dc_bc2_sqrt, adam_hp.eps);
    }

    if (degrees_to_use < 1) return;

    float x = viewdir.x, y = viewdir.y, z = viewdir.z;
    float xx = x*x, xy = x*y, xz = x*z, yy = y*y, yz = y*z, zz = z*z;

    // SH degree 1 (3 bases)
    float sh1[3] = { -SH_C1 * y, SH_C1 * z, -SH_C1 * x };
    for (int b = 0; b < 3; b++) {
        for (int c = 0; c < 3; c++) {
            uint i = rest_idx + b * 3 + c;
            float g = sh1[b] * vc[c];
            adam_update_element(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                               g, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.rest_bc2_sqrt, adam_hp.eps);
        }
    }

    if (degrees_to_use < 2) return;

    // SH degree 2 (5 bases)
    float sh2[5] = {
        SH_C2[0] * xy,
        SH_C2[1] * yz,
        SH_C2[2] * (2.f * zz - xx - yy),
        SH_C2[3] * xz,
        SH_C2[4] * (xx - yy)
    };
    for (int b = 0; b < 5; b++) {
        for (int c = 0; c < 3; c++) {
            uint i = rest_idx + (3 + b) * 3 + c;
            float g = sh2[b] * vc[c];
            adam_update_element(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                               g, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.rest_bc2_sqrt, adam_hp.eps);
        }
    }

    if (degrees_to_use < 3) return;

    // SH degree 3 (7 bases)
    float sh3[7] = {
        SH_C3[0] * y * (3.f * xx - yy),
        SH_C3[1] * xy * z,
        SH_C3[2] * y * (4.f * zz - xx - yy),
        SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy),
        SH_C3[4] * x * (4.f * zz - xx - yy),
        SH_C3[5] * z * (xx - yy),
        SH_C3[6] * x * (xx - 3.f * yy)
    };
    for (int b = 0; b < 7; b++) {
        for (int c = 0; c < 3; c++) {
            uint i = rest_idx + (8 + b) * 3 + c;
            float g = sh3[b] * vc[c];
            adam_update_element(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                               g, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.rest_bc2_sqrt, adam_hp.eps);
        }
    }
}

// ===== Pack Sorted Gaussians Kernel =====

kernel void pack_sorted_gaussians_kernel(
    constant int32_t* gaussian_ids_sorted [[buffer(0)]],
    constant float* xys              [[buffer(1)]],
    constant float* conics           [[buffer(2)]],
    constant float* colors           [[buffer(3)]],
    constant float* opacities        [[buffer(4)]],
    device float* packed_xy_opac     [[buffer(5)]],
    device float* packed_conic       [[buffer(6)]],
    device float* packed_rgb         [[buffer(7)]],
    constant uint& N                 [[buffer(8)]],
    constant int32_t* cum_tiles_hit  [[buffer(9)]],
    constant uint& num_points        [[buffer(10)]],
    uint idx [[thread_position_in_grid]]
) {
    uint actual_N = min(N, (uint)cum_tiles_hit[num_points - 1]);
    if (idx >= actual_N) return;
    int32_t g_id = gaussian_ids_sorted[idx];
    float2 xy = read_packed_float2(xys, g_id);
    float opac = 1.f / (1.f + exp(-opacities[g_id]));
    float3 conic = read_packed_float3(conics, g_id);
    float3 rgb = read_packed_float3(colors, g_id);
    write_packed_float3(packed_xy_opac, idx, {xy.x, xy.y, opac});
    write_packed_float3(packed_conic, idx, conic);
    write_packed_float3(packed_rgb, idx, rgb);
}

// ===== Tile-Local Sorting Kernels =====
// Replaces global radix sort with: count by tile → scatter to tile bins → per-tile bitonic sort.

#define SORT_TG_SIZE 256
#define MAX_TILE_ELEMS 2048

// Count how many intersections each tile has (for computing tile bin offsets).
kernel void count_intersections_per_tile_kernel(
    constant uint& num_points       [[buffer(0)]],
    constant float* xys             [[buffer(1)]],
    constant int* radii             [[buffer(2)]],
    constant float* aabb            [[buffer(3)]],
    constant uint3& tile_bounds     [[buffer(4)]],
    device atomic_uint* tile_counts [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) return;
    if (radii[idx] <= 0) return;
    float2 center = read_packed_float2(xys, idx);
    uint2 tile_min, tile_max;
    get_tile_bbox(center, read_packed_float2(aabb, idx), (int3)tile_bounds, tile_min, tile_max);
    for (uint i = tile_min.y; i < tile_max.y; i++) {
        for (uint j = tile_min.x; j < tile_max.x; j++) {
            uint tile_id = i * tile_bounds.x + j;
            atomic_fetch_add_explicit(&tile_counts[tile_id], 1u, memory_order_relaxed);
        }
    }
}

// Scatter each gaussian's intersections into per-tile bins.
// sort_pairs[tile_start + pos] = (depth_bits << 32) | gaussian_id
// Also writes tile_bins from tile_offsets/tile_counts (first num_tiles threads).
kernel void scatter_to_tiles_kernel(
    constant uint& num_points               [[buffer(0)]],
    constant float* xys                     [[buffer(1)]],
    constant float* depths                  [[buffer(2)]],
    constant int* radii                     [[buffer(3)]],
    constant float* aabb                    [[buffer(4)]],
    constant uint3& tile_bounds             [[buffer(5)]],
    constant int* tile_offsets              [[buffer(6)]],  // inclusive prefix sum
    constant int* tile_counts_in            [[buffer(7)]],
    device atomic_uint* scatter_counters    [[buffer(8)]],
    device uint64_t* sort_pairs             [[buffer(9)]],
    device int* tile_bins                   [[buffer(10)]],
    constant uint& capacity                 [[buffer(11)]],
    device atomic_uint* overflow_flag       [[buffer(12)]],
    constant uint& num_tiles                [[buffer(13)]],
    uint idx [[thread_position_in_grid]]
) {
    // First num_tiles threads write tile_bins
    if (idx < num_tiles) {
        int count = tile_counts_in[idx];
        int end = tile_offsets[idx];
        int start = end - count;
        write_packed_int2(tile_bins, idx, int2(start, end));
    }

    if (idx >= num_points) return;
    if (radii[idx] <= 0) return;

    float2 center = read_packed_float2(xys, idx);
    uint2 tile_min, tile_max;
    get_tile_bbox(center, read_packed_float2(aabb, idx), (int3)tile_bounds, tile_min, tile_max);

    uint depth_bits = as_type<uint>(depths[idx]);

    for (uint i = tile_min.y; i < tile_max.y; i++) {
        for (uint j = tile_min.x; j < tile_max.x; j++) {
            uint tile_id = i * tile_bounds.x + j;
            uint pos = atomic_fetch_add_explicit(&scatter_counters[tile_id], 1u, memory_order_relaxed);
            int tile_start = tile_offsets[tile_id] - tile_counts_in[tile_id];
            uint global_pos = (uint)tile_start + pos;
            if (global_pos >= capacity) {
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                return;
            }
            sort_pairs[global_pos] = ((uint64_t)depth_bits << 32) | (uint64_t)idx;
        }
    }
}

// Bitonic sort per tile in shared memory. Sorts by depth (upper 32 bits of uint64).
// Writes sorted gaussian_ids to output.
kernel void bitonic_sort_per_tile_kernel(
    constant int* tile_offsets          [[buffer(0)]],
    constant int* tile_counts_in        [[buffer(1)]],
    device uint64_t* sort_pairs         [[buffer(2)]],
    device int32_t* gaussian_ids_out    [[buffer(3)]],
    constant uint& num_tiles            [[buffer(4)]],
    // Pack buffers (fused sort+pack: eliminates separate pack dispatch)
    constant float* xys                 [[buffer(5)]],
    constant float* conics              [[buffer(6)]],
    constant float* colors              [[buffer(7)]],
    constant float* opacities           [[buffer(8)]],
    device float* packed_xy_opac        [[buffer(9)]],
    device float* packed_conic          [[buffer(10)]],
    device float* packed_rgb            [[buffer(11)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tg_id >= num_tiles) return;

    int count = tile_counts_in[tg_id];
    if (count == 0) return;
    int start = tile_offsets[tg_id] - count;

    int n_load = min(count, MAX_TILE_ELEMS);

    // Round up to next power of 2
    int n = 1;
    while (n < n_load) n <<= 1;

    threadgroup uint64_t data[MAX_TILE_ELEMS];

    // Load into shared memory
    for (int i = (int)tid; i < n; i += SORT_TG_SIZE) {
        data[i] = (i < n_load) ? sort_pairs[start + i] : 0xFFFFFFFFFFFFFFFFULL;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort (ascending on uint64 — depth in upper 32 bits)
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = (int)tid; i < (n >> 1); i += SORT_TG_SIZE) {
                int pos = 2 * i - (i & (j - 1));
                int partner = pos ^ j;
                bool ascending = ((pos & k) == 0);
                uint64_t a = data[pos];
                uint64_t b = data[partner];
                if ((a > b) == ascending) {
                    data[pos] = b;
                    data[partner] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Fused sort+pack: extract gaussian IDs, read per-gaussian data, write packed buffers
    for (int i = (int)tid; i < n_load; i += SORT_TG_SIZE) {
        int32_t g_id = (int32_t)(data[i] & 0xFFFFFFFF);
        int global_idx = start + i;
        gaussian_ids_out[global_idx] = g_id;
        float2 xy = read_packed_float2(xys, g_id);
        float opac = 1.f / (1.f + exp(-opacities[g_id]));
        write_packed_float3(packed_xy_opac, global_idx, {xy.x, xy.y, opac});
        write_packed_float3(packed_conic, global_idx, read_packed_float3(conics, g_id));
        write_packed_float3(packed_rgb, global_idx, read_packed_float3(colors, g_id));
    }
}

// ===== Radix Sort Kernels (legacy, kept for reference) =====
// 8-bit LSB radix sort for int64 keys + int32 values.
// 3 kernels per pass: histogram, scan, scatter.
// TG_SIZE = 256 = RS_RADIX (one element per thread, one histogram bin per thread).

#define RS_RADIX 256
#define RS_TG_SIZE 256

kernel void radix_sort_histogram_kernel(
    constant uint& capacity        [[buffer(0)]],
    device const int64_t* keys_in  [[buffer(1)]],
    device uint* counts            [[buffer(2)]],
    constant uint& shift           [[buffer(3)]],
    device const int32_t* cum_tiles_hit [[buffer(4)]],
    constant uint& num_points      [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    // Read actual element count from GPU-resident prefix sum
    uint N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);

    // Early exit for threadgroups entirely beyond N — write zero and return
    if (bid * RS_TG_SIZE >= N) {
        counts[bid * RS_RADIX + tid] = 0;
        return;
    }

    threadgroup atomic_uint local_hist[RS_RADIX];

    // Each thread zeros one histogram bin
    atomic_store_explicit(&local_hist[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count digit for this thread's element
    uint global_idx = bid * RS_TG_SIZE + tid;
    if (global_idx < N) {
        uint digit = extract_bits((uint64_t)keys_in[global_idx], shift, 8);
        atomic_fetch_add_explicit(&local_hist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write histogram to global memory (row-major: counts[block_id * 256 + digit])
    counts[bid * RS_RADIX + tid] = atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
}

kernel void radix_sort_scan_kernel(
    device uint* counts                     [[buffer(0)]],
    constant uint& num_blocks               [[buffer(1)]],
    device const int32_t* cum_tiles_hit     [[buffer(2)]],
    constant uint& capacity                 [[buffer(3)]],
    constant uint& num_points               [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    // Read actual element count and compute actual block count
    uint actual_N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);
    uint actual_num_blocks = (actual_N + RS_TG_SIZE - 1) / RS_TG_SIZE;

    // Each thread handles one digit value (tid = digit d)
    uint d = tid;

    // Phase 1: Exclusive prefix sum across actual blocks only
    uint running = 0;
    for (uint b = 0; b < actual_num_blocks; b++) {
        uint idx = b * RS_RADIX + d;
        uint count = counts[idx];
        counts[idx] = running;
        running += count;
    }

    // Phase 2: Cross-digit prefix sum to get global offsets
    threadgroup uint digit_totals[RS_RADIX];
    digit_totals[d] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 does sequential exclusive prefix sum of 256 digit totals
    if (d == 0) {
        uint accum = 0;
        for (uint i = 0; i < RS_RADIX; i++) {
            uint t = digit_totals[i];
            digit_totals[i] = accum;
            accum += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Add global digit offset to actual blocks only
    uint global_offset = digit_totals[d];
    for (uint b = 0; b < actual_num_blocks; b++) {
        counts[b * RS_RADIX + d] += global_offset;
    }
}

kernel void radix_sort_scatter_kernel(
    constant uint& capacity            [[buffer(0)]],
    device const int64_t* keys_in      [[buffer(1)]],
    device const int32_t* vals_in      [[buffer(2)]],
    device int64_t* keys_out           [[buffer(3)]],
    device int32_t* vals_out           [[buffer(4)]],
    device const uint* counts          [[buffer(5)]],
    constant uint& shift               [[buffer(6)]],
    device const int32_t* cum_tiles_hit [[buffer(7)]],
    constant uint& num_points          [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]]
) {
    // Read actual element count from GPU-resident prefix sum
    uint N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);

    // Early exit for threadgroups entirely beyond N
    if (bid * RS_TG_SIZE >= N) return;

    threadgroup uchar shared_digits[RS_TG_SIZE];

    uint global_idx = bid * RS_TG_SIZE + tid;

    // Load element
    int64_t my_key = 0;
    int32_t my_val = 0;
    uint my_digit = 0;
    bool valid = (global_idx < N);

    if (valid) {
        my_key = keys_in[global_idx];
        my_val = vals_in[global_idx];
        my_digit = extract_bits((uint64_t)my_key, shift, 8);
    }

    // Store digits for cross-simdgroup rank computation
    shared_digits[tid] = (uchar)my_digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hybrid rank: SIMD broadcast for intra-simdgroup, shared memory for cross-simdgroup.
    // Intra-simdgroup: all 32 lanes call simd_broadcast uniformly, each accumulates
    // only matches from lanes with lower index.
    uint sg_rank = 0;
    for (ushort l = 0; l < 32; l++) {
        uint other = simd_broadcast(my_digit, l);
        if (l < sg_lane && other == my_digit) sg_rank++;
    }

    // Cross-simdgroup: vectorized scan of preceding simdgroups via shared memory
    uint cross_rank = 0;
    uint preceding_end = sg_id * 32;
    uint full_words = preceding_end / 4;
    threadgroup uint* digits_u32 = (threadgroup uint*)shared_digits;
    for (uint i = 0; i < full_words; i++) {
        uint four = digits_u32[i];
        if (((four >>  0) & 0xFF) == my_digit) cross_rank++;
        if (((four >>  8) & 0xFF) == my_digit) cross_rank++;
        if (((four >> 16) & 0xFF) == my_digit) cross_rank++;
        if (((four >> 24) & 0xFF) == my_digit) cross_rank++;
    }
    uint rank = cross_rank + sg_rank;

    // Write to global output at computed position
    if (valid) {
        uint global_pos = counts[bid * RS_RADIX + my_digit] + rank;
        keys_out[global_pos] = my_key;
        vals_out[global_pos] = my_val;
    }
}

// ===== Prefix Sum Kernel =====
// Single-dispatch inclusive prefix sum (cumsum) for int32 arrays.
// Uses one threadgroup: each thread serially sums its chunk, thread 0 scans
// block totals, then all threads write inclusive prefix sums.
// Used for small N (≤ PS_TG_SIZE) only; large N uses multi-threadgroup path.

#define PS_TG_SIZE 1024

kernel void prefix_sum_kernel(
    constant uint& N,
    constant int* input,
    device int* output,
    uint tg_tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    // Phase 1: Each thread serially sums its chunk
    uint chunk = (N + tg_size - 1) / tg_size;
    uint start = tg_tid * chunk;
    uint end = min(start + chunk, N);

    int my_sum = 0;
    for (uint i = start; i < end; i++) {
        my_sum += input[i];
    }

    // Phase 2: Two-level parallel prefix sum using SIMD
    // Level 1: intra-simdgroup exclusive prefix sum (hardware-accelerated)
    int sg_prefix = simd_prefix_exclusive_sum(my_sum);
    int sg_total = simd_sum(my_sum);

    // Level 2: cross-simdgroup scan (max 32 simdgroups for 1024 threads)
    uint num_sg = (tg_size + sg_size - 1) / sg_size;
    threadgroup int sg_totals[PS_TG_SIZE / 32];
    if (sg_lane == 0) {
        sg_totals[sg_id] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 scans simdgroup totals (max 32 iterations)
    threadgroup int sg_offsets[PS_TG_SIZE / 32];
    if (tg_tid == 0) {
        sg_offsets[0] = 0;
        for (uint i = 1; i < num_sg; i++) {
            sg_offsets[i] = sg_offsets[i - 1] + sg_totals[i - 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final prefix = cross-simdgroup offset + intra-simdgroup prefix
    int my_prefix = sg_offsets[sg_id] + sg_prefix;

    // Phase 3: Each thread writes inclusive prefix sum for its chunk
    int running = my_prefix;
    for (uint i = start; i < end; i++) {
        running += input[i];
        output[i] = running;
    }
}

// Multi-threadgroup prefix sum, pass 1: each threadgroup reduces its block of
// 1024 elements to a single total. Coalesced reads, 1 write per threadgroup.
kernel void block_reduce_kernel(
    constant uint& N,
    constant int* input,
    device int* block_totals,
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]]
) {
    uint idx = tg_id * PS_TG_SIZE + tg_tid;
    int val = (idx < N) ? input[idx] : 0;

    // Two-level reduction: SIMD sum → cross-SIMD sum
    int sg_total = simd_sum(val);

    threadgroup int sg_sums[PS_TG_SIZE / 32];
    if (sg_lane == 0) sg_sums[sg_id] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_tid == 0) {
        int total = 0;
        for (uint i = 0; i < PS_TG_SIZE / 32; i++) total += sg_sums[i];
        block_totals[tg_id] = total;
    }
}

// Multi-threadgroup prefix sum, pass 2: each threadgroup computes its block
// offset from block_totals, then writes inclusive prefix sums with coalesced access.
kernel void block_scan_propagate_kernel(
    constant uint& N,
    constant int* input,
    device int* output,
    constant int* block_totals,
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    // Step 1: Compute block offset (sum of all preceding block totals)
    int block_offset = 0;
    for (uint i = 0; i < tg_id; i++) {
        block_offset += block_totals[i];
    }

    // Step 2: Load element (coalesced)
    uint idx = tg_id * PS_TG_SIZE + tg_tid;
    int val = (idx < N) ? input[idx] : 0;

    // Step 3: Intra-block inclusive prefix sum (SIMD + cross-SIMD)
    int sg_prefix = simd_prefix_exclusive_sum(val);
    int sg_total = simd_sum(val);

    uint num_sg = PS_TG_SIZE / 32;
    threadgroup int sg_totals[PS_TG_SIZE / 32];
    threadgroup int sg_offsets[PS_TG_SIZE / 32];
    if (sg_lane == 0) sg_totals[sg_id] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_tid == 0) {
        int acc = 0;
        for (uint i = 0; i < num_sg; i++) {
            sg_offsets[i] = acc;
            acc += sg_totals[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Write inclusive prefix sum (coalesced)
    int inclusive = block_offset + sg_offsets[sg_id] + sg_prefix + val;
    if (idx < N) output[idx] = inclusive;
}

// ===== Fused Loss Kernels =====
// Combines SSIM (with 11×11 Gaussian window) + L1 into a single forward pass,
// and computes the full backward (dL/d_rendered) in a single backward pass.
// Replaces ~15 MPS dispatches (5 conv2d + elementwise SSIM formula + conv2d_backward + L1 ops).

#define SSIM_WIN 11
#define SSIM_HALF_WIN 5
#define SSIM_C1 0.0001f
#define SSIM_C2 0.0009f

kernel void fused_loss_forward_kernel(
    constant float* rendered,       // (H, W, 3) HWC
    constant float* gt,             // (H, W, 3) HWC
    constant float* window,         // (121,) precomputed 2D Gaussian window
    constant uint2& img_size,       // (W, H)
    constant float& ssim_weight,
    device float* intermediates,    // (H, W, 15) — 5 values × 3 channels per pixel
    device atomic_float* loss_sum,  // scalar: atomic sum of all pixel losses
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    uint px = gid.x;
    uint py = gid.y;

    float pixel_loss = 0.0f;

    if (px < W && py < H) {
        float ssim_sum = 0.0f;
        float l1_sum = 0.0f;

        // Clamp loop bounds once to avoid per-iteration boundary checks
        int dy_min = max(-SSIM_HALF_WIN, -(int)py);
        int dy_max = min(SSIM_HALF_WIN, (int)(H - 1) - (int)py);
        int dx_min = max(-SSIM_HALF_WIN, -(int)px);
        int dx_max = min(SSIM_HALF_WIN, (int)(W - 1) - (int)px);

        for (uint c = 0; c < 3; c++) {
            float mu_x = 0, mu_y = 0, sq_x = 0, sq_y = 0, cross_xy = 0;

            for (int dy = dy_min; dy <= dy_max; dy++) {
                int ny = (int)py + dy;
                for (int dx = dx_min; dx <= dx_max; dx++) {
                    int nx = (int)px + dx;
                    float w = window[(dy + SSIM_HALF_WIN) * SSIM_WIN + (dx + SSIM_HALF_WIN)];
                    float x_val = gt[(ny * W + nx) * 3 + c];
                    float y_val = rendered[(ny * W + nx) * 3 + c];
                    mu_x += w * x_val;
                    mu_y += w * y_val;
                    sq_x += w * x_val * x_val;
                    sq_y += w * y_val * y_val;
                    cross_xy += w * x_val * y_val;
                }
            }

            float sigma_x_sq = sq_x - mu_x * mu_x;
            float sigma_y_sq = sq_y - mu_y * mu_y;
            float sigma_xy = cross_xy - mu_x * mu_y;

            uint iidx = (py * W + px) * 15 + c * 5;
            intermediates[iidx + 0] = mu_x;
            intermediates[iidx + 1] = mu_y;
            intermediates[iidx + 2] = sigma_x_sq;
            intermediates[iidx + 3] = sigma_y_sq;
            intermediates[iidx + 4] = sigma_xy;

            float A = 2.0f * mu_x * mu_y + SSIM_C1;
            float B = 2.0f * sigma_xy + SSIM_C2;
            float C_d = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
            float D = sigma_x_sq + sigma_y_sq + SSIM_C2;

            ssim_sum += (A * B) / (C_d * D);

            float gt_val = gt[(py * W + px) * 3 + c];
            float rend_val = rendered[(py * W + px) * 3 + c];
            l1_sum += fabs(gt_val - rend_val);
        }

        pixel_loss = ssim_weight * (1.0f - ssim_sum / 3.0f) + (1.0f - ssim_weight) * l1_sum / 3.0f;
    }

    // Threadgroup reduction then single atomic add to device memory
    threadgroup float tg_sum[256];
    tg_sum[tid] = pixel_loss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tg_total = tg_size.x * tg_size.y;
    for (uint s = tg_total / 2; s > 0; s >>= 1) {
        if (tid < s) tg_sum[tid] += tg_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        atomic_fetch_add_explicit(loss_sum, tg_sum[0], memory_order_relaxed);
    }
}

kernel void fused_loss_backward_kernel(
    constant float* rendered,       // (H, W, 3) HWC
    constant float* gt,             // (H, W, 3) HWC
    constant float* window,         // (121,) 2D Gaussian window
    constant uint2& img_size,       // (W, H)
    constant float* intermediates,  // (H, W, 15) from forward
    constant float& ssim_weight,
    constant float& inv_n,          // 1.0 / (H * W * 3)
    device float* v_rendered,       // (H, W, 3) output gradient
    uint2 gid [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    uint px = gid.x;
    uint py = gid.y;
    if (px >= W || py >= H) return;

    // Clamp loop bounds once to avoid per-iteration boundary checks
    int dy_min = max(-SSIM_HALF_WIN, (int)py - (int)(H - 1));
    int dy_max = min(SSIM_HALF_WIN, (int)py);
    int dx_min = max(-SSIM_HALF_WIN, (int)px - (int)(W - 1));
    int dx_max = min(SSIM_HALF_WIN, (int)px);

    for (uint c = 0; c < 3; c++) {
        float rend_val = rendered[(py * W + px) * 3 + c];
        float gt_val = gt[(py * W + px) * 3 + c];

        // L1 gradient: d|gt-rend|/d(rend) = -sign(gt-rend)
        float v_l1 = (gt_val > rend_val) ? -1.0f : ((gt_val < rend_val) ? 1.0f : 0.0f);

        // SSIM gradient: sum contributions from all windows containing this pixel
        float v_ssim = 0.0f;

        for (int dy = dy_min; dy <= dy_max; dy++) {
            int cy = (int)py - dy;  // center of neighboring window
            for (int dx = dx_min; dx <= dx_max; dx++) {
                int cx = (int)px - dx;

                float w = window[(dy + SSIM_HALF_WIN) * SSIM_WIN + (dx + SSIM_HALF_WIN)];

                // Read intermediates at window center (cy, cx)
                uint iidx = (cy * W + cx) * 15 + c * 5;
                float mu_x = intermediates[iidx + 0];
                float mu_y = intermediates[iidx + 1];
                float sigma_x_sq = intermediates[iidx + 2];
                float sigma_y_sq = intermediates[iidx + 3];
                float sigma_xy = intermediates[iidx + 4];

                float A = 2.0f * mu_x * mu_y + SSIM_C1;
                float B = 2.0f * sigma_xy + SSIM_C2;
                float C_d = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
                float D = sigma_x_sq + sigma_y_sq + SSIM_C2;

                // Partial derivatives of SSIM w.r.t. mu_y, sigma_y_sq, sigma_xy
                float inv_CD = 1.0f / (C_d * D);
                float dSSIM_dmu_y = 2.0f * B * (mu_x * C_d - A * mu_y) / (C_d * C_d * D);
                float dSSIM_dsigma_y_sq = -A * B * inv_CD / D;
                float dSSIM_dsigma_xy = 2.0f * A * inv_CD;

                // Chain: d(rendered[py,px])/d(mu_y) = w, etc.
                v_ssim += w * (dSSIM_dmu_y
                    + 2.0f * (rend_val - mu_y) * dSSIM_dsigma_y_sq
                    + (gt_val - mu_x) * dSSIM_dsigma_xy);
            }
        }

        // Combined: loss = ssim_weight*(1-mean(ssim)) + (1-ssim_weight)*mean(l1)
        // dL/d(rendered) = -ssim_weight * inv_n * v_ssim + (1-ssim_weight) * inv_n * v_l1
        v_rendered[(py * W + px) * 3 + c] = inv_n * (
            -ssim_weight * v_ssim + (1.0f - ssim_weight) * v_l1
        );
    }
}

// ============================================================================
// Depth-chunked rasterization kernels
// ============================================================================

#define CHUNK_SIZE 512

// Reduce max tile count: find max(bin_end - bin_start) across all tiles.
kernel void reduce_max_tile_count_kernel(
    constant uint& num_tiles,
    constant int* tile_bins, // int2 packed
    device atomic_uint* max_count,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_tiles) return;
    int2 range = read_packed_int2(tile_bins, (int)idx);
    uint count = (uint)max(0, range.y - range.x);
    atomic_fetch_max_explicit(max_count, count, memory_order_relaxed);
}

// Forward chunked rasterization: each threadgroup processes one (tile, chunk) pair.
// Grid: (tile_x, tile_y, K_max). blockIdx.z = chunk index k.
kernel void rasterize_forward_chunked_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int* tile_bins,
    constant float* packed_xy_opac,
    constant float* packed_conic,
    constant float* packed_rgb,
    device float* chunk_T,        // [K_max, H, W]
    device float* chunk_C,        // [K_max, H, W, 3]
    device int* chunk_final_idx,  // [K_max, H, W]
    constant uint& chunk_size,
    constant uint& K_max,
    constant uint2& blockDim,
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint tr [[thread_index_in_threadgroup]]
) {
    uint k = blockIdx.z; // chunk index
    // Reconstruct 2D thread position from 1D thread index
    uint threadIdx_x = tr % RAST_BLOCK_X;
    uint threadIdx_y = tr / RAST_BLOCK_X;
    int32_t i = blockIdx.y * blockDim.y + threadIdx_y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx_x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    uint num_pixels = img_size.x * img_size.y;
    int32_t pix_id = i * (int)img_size.x + j;
    const bool inside = (i < (int)img_size.y && j < (int)img_size.x);

    // Full tile range from tile_bins
    int2 full_range = read_packed_int2(tile_bins, tile_id);
    // Chunk sub-range
    int chunk_start = full_range.x + (int)(k * chunk_size);
    int chunk_end = min(full_range.x + (int)((k + 1) * chunk_size), full_range.y);

    // Output offset: k * num_pixels + pix_id
    uint out_offset = k * num_pixels + (uint)pix_id;

    if (chunk_start >= chunk_end) {
        // Empty chunk — write defaults
        if (inside) {
            chunk_T[out_offset] = 1.f;
            chunk_C[out_offset * 3 + 0] = 0.f;
            chunk_C[out_offset * 3 + 1] = 0.f;
            chunk_C[out_offset * 3 + 2] = 0.f;
            chunk_final_idx[out_offset] = -1;
        }
        return;
    }

    int num_batches = (chunk_end - chunk_start + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];

    float T = 1.f;
    float3 pix_out = {0.f, 0.f, 0.f};
    int last_contributor = chunk_start - 1;
    bool done = false;

    for (int b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int batch_start = chunk_start + RAST_BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < chunk_end) {
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_float3(packed_conic, idx);
            const float3 raw_c = read_packed_float3(packed_rgb, idx);
            rgbs_batch[tr] = max(raw_c + 0.5f, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (done || !inside) continue;

        int batch_size = min(RAST_BLOCK_SIZE, chunk_end - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            const float3 conic_local = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = fma(0.5f,
                fma(conic_local.x, delta.x * delta.x, conic_local.z * delta.y * delta.y),
                conic_local.y * delta.x * delta.y);
            if (sigma < 0.f || sigma >= 5.55f) continue;
            const float alpha = min(0.999f, xy_opac.z * exp(-sigma));
            if (alpha < 1.f / 255.f) continue;
            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) {
                last_contributor = batch_start + t - 1;
                done = true;
                break;
            }
            const float vis = alpha * T;
            pix_out = fma(rgbs_batch[t], vis, pix_out);
            T = next_T;
            last_contributor = batch_start + t;
        }
    }

    if (inside) {
        chunk_T[out_offset] = T;
        chunk_C[out_offset * 3 + 0] = pix_out.x;
        chunk_C[out_offset * 3 + 1] = pix_out.y;
        chunk_C[out_offset * 3 + 2] = pix_out.z;
        chunk_final_idx[out_offset] = last_contributor;
    }
}

// Forward merge: scan K chunks per pixel, produce final out_img/final_Ts/final_idx.
// Also applies absolute transmittance cutoff: when T_running drops below 1e-4,
// zeros out chunk_final_idx for remaining chunks so the backward skips them.
kernel void rasterize_forward_merge_kernel(
    constant uint& num_pixels, // H * W
    constant uint& K_max,
    constant float* chunk_T,        // [K_max, H, W]
    constant float* chunk_C,        // [K_max, H, W, 3]
    device int* chunk_final_idx,    // [K_max, H, W] — device (not constant) for cutoff fixup
    device float* final_Ts,         // [H, W]
    device int* final_index,        // [H, W]
    device float* out_img,          // [H, W, 3]
    constant float* background,
    constant uint2& img_size,       // (W, H)
    uint2 gp [[thread_position_in_grid]]
) {
    uint px = gp.x;
    uint py = gp.y;
    if (px >= img_size.x || py >= img_size.y) return;
    uint pix_id = py * img_size.x + px;

    float T_running = 1.f;
    float3 C_running = {0.f, 0.f, 0.f};
    int last_idx = -1;
    uint cutoff_k = K_max; // chunk index where absolute T cutoff triggered

    for (uint k = 0; k < K_max; ++k) {
        uint offset = k * num_pixels + pix_id;
        int cfidx = chunk_final_idx[offset];
        if (cfidx < 0 && k > 0) break; // empty chunk after first real one = done
        // Even if cfidx == chunk_start-1 (no contribution), chunk_T=1 and chunk_C=0
        float cT = chunk_T[offset];
        float3 cC = {chunk_C[offset * 3 + 0], chunk_C[offset * 3 + 1], chunk_C[offset * 3 + 2]};
        C_running = fma(cC, T_running, C_running);
        T_running *= cT;
        if (cfidx >= 0) last_idx = cfidx;
        // Absolute transmittance cutoff: stop when pixel is fully opaque
        if (T_running <= 1e-4f) {
            cutoff_k = k + 1;
            break;
        }
    }

    // Zero out chunk_final_idx for chunks past the absolute cutoff
    // so the backward kernel skips them (bin_final < chunk_start → return)
    for (uint k = cutoff_k; k < K_max; ++k) {
        chunk_final_idx[k * num_pixels + pix_id] = -1;
    }

    final_Ts[pix_id] = T_running;
    final_index[pix_id] = last_idx;
    float3 bg = {background[0], background[1], background[2]};
    float3 final_rgb = saturate(fma(bg, T_running, C_running));
    out_img[CHANNELS * pix_id + 0] = final_rgb.x;
    out_img[CHANNELS * pix_id + 1] = final_rgb.y;
    out_img[CHANNELS * pix_id + 2] = final_rgb.z;
}

// Compute prefix transmittance and suffix color for backward chunked rasterization.
// prefix_T[k] = product of chunk_T[0..k-1] (transmittance before chunk k)
// after_C[k] = sum_{j>k} prefix_T[j] * chunk_C[j] (color contribution after chunk k)
kernel void compute_chunk_prefix_suffix_kernel(
    constant uint& num_pixels,
    constant uint& K_max,
    constant float* chunk_T,        // [K_max, H, W]
    constant float* chunk_C,        // [K_max, H, W, 3]
    constant int* chunk_final_idx,  // [K_max, H, W]
    device float* prefix_T,         // [K_max, H, W]
    device float* after_C,          // [K_max, H, W, 3]
    constant uint2& img_size,
    uint2 gp [[thread_position_in_grid]]
) {
    uint px = gp.x;
    uint py = gp.y;
    if (px >= img_size.x || py >= img_size.y) return;
    uint pix_id = py * img_size.x + px;

    // Forward scan: compute prefix transmittance products
    float pT = 1.f;
    for (uint k = 0; k < K_max; ++k) {
        uint offset = k * num_pixels + pix_id;
        prefix_T[offset] = pT;
        pT *= chunk_T[offset];
    }

    // Backward scan: compute suffix color contribution
    // after_C[k] = sum_{j=k+1}^{K-1} prefix_T[j] * chunk_C[j]
    float3 aC = {0.f, 0.f, 0.f};
    for (int k = (int)K_max - 1; k >= 0; --k) {
        uint offset = (uint)k * num_pixels + pix_id;
        after_C[offset * 3 + 0] = aC.x;
        after_C[offset * 3 + 1] = aC.y;
        after_C[offset * 3 + 2] = aC.z;
        float pT_k = prefix_T[offset];
        float3 cC = {chunk_C[offset * 3 + 0], chunk_C[offset * 3 + 1], chunk_C[offset * 3 + 2]};
        aC += pT_k * cC;
    }
}

// Backward chunked rasterization: each threadgroup processes one (tile, chunk) pair.
// Grid: (tile_x, tile_y, K_max). blockIdx.z = chunk index k.
kernel void rasterize_backward_chunked_kernel(
    constant uint3& tile_bounds,
    constant uint2& img_size,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins,
    constant float* packed_xy_opac,
    constant float* packed_conic,
    constant float* packed_rgb,
    constant float* background,
    constant float* final_Ts,       // [H, W] — global final transmittance
    constant int* chunk_final_idx,  // [K_max, H, W]
    constant float* prefix_T_buf,   // [K_max, H, W]
    constant float* chunk_T_buf,    // [K_max, H, W]
    constant float* after_C_buf,    // [K_max, H, W, 3]
    constant float* v_output,
    device atomic_float* v_xy,
    device atomic_float* v_conic,
    device atomic_float* v_rgb,
    device atomic_float* v_opacity,
    constant uint& chunk_size,
    constant uint& K_max,
    uint3 gp [[thread_position_in_grid]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint tr [[thread_index_in_threadgroup]],
    uint warp_size [[threads_per_simdgroup]],
    uint wr [[thread_index_in_simdgroup]]
) {
    uint k = blockIdx.z;
    uint i = gp.y;
    uint j = gp.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = min((int32_t)(i * img_size.x + j), (int32_t)(img_size.x * img_size.y - 1));
    const bool inside = (i < img_size.y && j < img_size.x);
    uint num_pixels = img_size.x * img_size.y;
    uint chunk_offset = k * num_pixels + (uint)pix_id;

    // Read per-pixel, per-chunk data
    float pT_k = prefix_T_buf[chunk_offset];       // transmittance before chunk k
    float cT_k = chunk_T_buf[chunk_offset];         // local transmittance of chunk k
    float3 aC_k = {after_C_buf[chunk_offset * 3 + 0],
                    after_C_buf[chunk_offset * 3 + 1],
                    after_C_buf[chunk_offset * 3 + 2]};

    // Initialize T and buffer as if monolithic backward just finished all chunks > k
    // T = prefix_T[k] * chunk_T[k] = absolute transmittance after chunk k
    float T = pT_k * cT_k;
    // buffer = after_C[k] = weighted color contribution from all chunks after k
    float3 buffer = aC_k;

    float T_final = final_Ts[pix_id];
    const float3 bg = {background[0], background[1], background[2]};
    const float3 T_final_bg = T_final * bg;
    const float3 v_out = read_packed_float3(v_output, pix_id);

    const int bin_final = inside ? chunk_final_idx[chunk_offset] : -1;

    // Chunk sub-range
    int2 full_range = read_packed_int2(tile_bins, tile_id);
    int chunk_start = full_range.x + (int)(k * chunk_size);
    int chunk_end = min(full_range.x + (int)((k + 1) * chunk_size), full_range.y);

    if (chunk_start >= chunk_end || bin_final < chunk_start) {
        return; // empty chunk or no contributors
    }

    const int num_batches = (chunk_end - chunk_start + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    threadgroup int32_t id_batch[RAST_BLOCK_SIZE];
    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];

    // Warp-level early exit
    const int warp_bin_final = warp_reduce_all_max(bin_final, warp_size);

    // Subtile-level early exit: skip leading batches beyond any pixel's bin_final
    const uint warp_id = tr / warp_size;
    constexpr uint NUM_WARPS = RAST_BLOCK_SIZE / 32;
    threadgroup int warp_max_finals[NUM_WARPS];
    if (wr == 0) warp_max_finals[warp_id] = warp_bin_final;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int tile_max_bin_final = warp_max_finals[0];
    for (uint w = 1; w < NUM_WARPS; w++)
        tile_max_bin_final = max(tile_max_bin_final, warp_max_finals[w]);
    int dead_count = max(0, (int)(chunk_end - 1) - tile_max_bin_final);
    int first_batch = dead_count / RAST_BLOCK_SIZE;

    for (int b = first_batch; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load batch from back to front within this chunk
        const int batch_end = chunk_end - 1 - RAST_BLOCK_SIZE * b;
        int batch_size = min(RAST_BLOCK_SIZE, batch_end + 1 - chunk_start);
        const int idx = batch_end - tr;
        if (idx >= chunk_start) {
            id_batch[tr] = gaussian_ids_sorted[idx];
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_float3(packed_conic, idx);
            rgbs_batch[tr] = read_packed_float3(packed_rgb, idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            float3 b_conic, b_xy_opac, b_rgb;
            int32_t b_id;
            if (wr == 0) {
                b_conic = conic_batch[t];
                b_xy_opac = xy_opacity_batch[t];
                b_rgb = rgbs_batch[t];
                b_id = id_batch[t];
            }
            b_conic = simd_broadcast(b_conic, 0);
            b_xy_opac = simd_broadcast(b_xy_opac, 0);
            b_rgb = simd_broadcast(b_rgb, 0);
            b_id = simd_broadcast(b_id, 0);

            int valid = inside;
            if (batch_end - t > bin_final) valid = 0;

            float alpha;
            float opac;
            float2 delta;
            float vis;
            if (valid) {
                opac = b_xy_opac.z;
                delta = {b_xy_opac.x - px, b_xy_opac.y - py};
                float sigma = fma(0.5f,
                    fma(b_conic.x, delta.x * delta.x, b_conic.z * delta.y * delta.y),
                    b_conic.y * delta.x * delta.y);
                if (sigma < 0.f || sigma >= 5.55f) {
                    valid = 0;
                } else {
                    vis = exp(-sigma);
                    alpha = min(0.999f, opac * vis);
                    if (alpha < 1.f / 255.f) valid = 0;
                }
            }

            if (!warp_reduce_all_or(valid, warp_size)) continue;

            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;

            if (valid && alpha < 0.999f) {
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = fac * v_out;

                const float3 rgb = max(b_rgb + 0.5f, 0.f);
                v_alpha += dot(fma(rgb, T, fma(-buffer, ra, -ra * T_final_bg)), v_out);
                buffer = fma(rgb, fac, buffer);

                const float v_sigma = -alpha * v_alpha;
                v_conic_local = (0.5f * v_sigma) * float3(delta.x * delta.x,
                                                           delta.x * delta.y,
                                                           delta.y * delta.y);
                v_xy_local = v_sigma * float2(
                    fma(b_conic.x, delta.x, b_conic.y * delta.y),
                    fma(b_conic.y, delta.x, b_conic.z * delta.y));
                v_opacity_local = -v_sigma * (1.f - opac);
            }

            v_rgb_local = warpSum3(v_rgb_local, warp_size, wr);
            v_conic_local = warpSum3(v_conic_local, warp_size, wr);
            v_xy_local = warpSum2(v_xy_local, warp_size, wr);
            v_opacity_local = warpSum(v_opacity_local, warp_size, wr);

            if (wr == 0) {
                if (b_rgb.x + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 0, v_rgb_local.x, memory_order_relaxed);
                if (b_rgb.y + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 1, v_rgb_local.y, memory_order_relaxed);
                if (b_rgb.z + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 2, v_rgb_local.z, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*b_id + 0, v_conic_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*b_id + 1, v_conic_local.y, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*b_id + 2, v_conic_local.z, memory_order_relaxed);
                atomic_fetch_add_explicit(v_xy + 2*b_id + 0, v_xy_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_xy + 2*b_id + 1, v_xy_local.y, memory_order_relaxed);
                atomic_fetch_add_explicit(v_opacity + b_id, v_opacity_local, memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// Separable SSIM loss kernels (v29)
// Decompose 11×11 2D Gaussian convolution into 1D horizontal + vertical passes.
// Reduces per-pixel work from O(121) to O(22).
// ============================================================================

// 1D Gaussian window (sigma=1.5, size=11) — matches ssim.cpp:gaussian(1.5f)
// The 2D window is the outer product: w2d[i][j] = GAUSS_1D[i] * GAUSS_1D[j]
constant float GAUSS_1D[11] = {
    0.0010283801f, 0.0075987581f, 0.0360007721f, 0.1093606895f, 0.2130055377f,
    0.2660117249f,
    0.2130055377f, 0.1093606895f, 0.0360007721f, 0.0075987581f, 0.0010283801f
};

#define SSIM_TG 16   // threadgroup dimension (16×16 = 256 threads)

// Forward pass 1: horizontal convolution of rendered and gt.
// For each pixel, computes 5 horizontal partial sums per channel:
//   h_mu_x, h_mu_y, h_sq_x, h_sq_y, h_cross_xy
// Output: ssim_h_buf (H, W, 15) — 5 values × 3 channels
kernel void ssim_h_fwd_kernel(
    constant float* rendered,       // (H, W, 3) HWC
    constant float* gt,             // (H, W, 3) HWC
    constant uint2& img_size,       // (W, H)
    device float* ssim_h_buf,       // (H, W, 15)
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const int base_gx = (int)(tgid.x * SSIM_TG) - SSIM_HALF_WIN;
    const int base_gy = (int)(tgid.y * SSIM_TG);
    constexpr uint TILE_W = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = SSIM_TG * TILE_W;         // 416

    // Load all 3 channels at once: 6 × 16×26 × 4B = 9.75KB shared memory
    threadgroup float tg_gt[3][SSIM_TG][TILE_W];
    threadgroup float tg_rd[3][SSIM_TG][TILE_W];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / TILE_W;
            uint sx = i % TILE_W;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            float gv = 0.0f, rv = 0.0f;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint idx = (gy * W + gx) * 3 + c;
                gv = gt[idx];
                rv = rendered[idx];
            }
            tg_gt[c][sy][sx] = gv;
            tg_rd[c][sy][sx] = rv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (px < W && py < H) {
        for (uint c = 0; c < 3; c++) {
            float mu_x = 0, mu_y = 0, sq_x = 0, sq_y = 0, cross_xy = 0;
            for (uint dx = 0; dx < SSIM_WIN; dx++) {
                float w = GAUSS_1D[dx];
                float gv = tg_gt[c][lid.y][lid.x + dx];
                float rv = tg_rd[c][lid.y][lid.x + dx];
                mu_x += w * gv;
                mu_y += w * rv;
                sq_x += w * gv * gv;
                sq_y += w * rv * rv;
                cross_xy += w * gv * rv;
            }
            uint out = (py * W + px) * 15 + c * 5;
            ssim_h_buf[out + 0] = mu_x;
            ssim_h_buf[out + 1] = mu_y;
            ssim_h_buf[out + 2] = sq_x;
            ssim_h_buf[out + 3] = sq_y;
            ssim_h_buf[out + 4] = cross_xy;
        }
    }
}

// Forward pass 2: vertical convolution + SSIM/L1 computation + loss reduction.
// Reads ssim_h_buf, processes 1 channel at a time for occupancy.
// Output: intermediates (H, W, 15) — same format as fused_loss_forward_kernel
kernel void ssim_v_fwd_kernel(
    constant float* rendered,       // (H, W, 3) for L1
    constant float* gt,             // (H, W, 3) for L1
    constant float* ssim_h_buf,     // (H, W, 15)
    constant uint2& img_size,       // (W, H)
    constant float& ssim_weight,
    device float* intermediates,    // (H, W, 15)
    device atomic_float* loss_sum,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const int base_gx = (int)(tgid.x * SSIM_TG);
    const int base_gy = (int)(tgid.y * SSIM_TG) - SSIM_HALF_WIN;
    constexpr uint TILE_H = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = TILE_H * SSIM_TG;         // 416

    // Load all 3 channels at once: 3 × 26×16×5 × 4B = 24.96KB shared memory
    threadgroup float tg_hp[3][TILE_H][SSIM_TG][5];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / SSIM_TG;
            uint sx = i % SSIM_TG;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint hp = (gy * W + gx) * 15 + c * 5;
                for (uint f = 0; f < 5; f++) tg_hp[c][sy][sx][f] = ssim_h_buf[hp + f];
            } else {
                for (uint f = 0; f < 5; f++) tg_hp[c][sy][sx][f] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ssim_sum = 0.0f;
    float l1_sum = 0.0f;

    if (px < W && py < H) {
        for (uint c = 0; c < 3; c++) {
            float mu_x = 0, mu_y = 0, sq_x = 0, sq_y = 0, cross_xy = 0;
            for (uint dy = 0; dy < SSIM_WIN; dy++) {
                float w = GAUSS_1D[dy];
                mu_x     += w * tg_hp[c][lid.y + dy][lid.x][0];
                mu_y     += w * tg_hp[c][lid.y + dy][lid.x][1];
                sq_x     += w * tg_hp[c][lid.y + dy][lid.x][2];
                sq_y     += w * tg_hp[c][lid.y + dy][lid.x][3];
                cross_xy += w * tg_hp[c][lid.y + dy][lid.x][4];
            }

            float sigma_x_sq = sq_x - mu_x * mu_x;
            float sigma_y_sq = sq_y - mu_y * mu_y;
            float sigma_xy = cross_xy - mu_x * mu_y;

            // Store intermediates (same format as fused_loss_forward_kernel)
            uint iidx = (py * W + px) * 15 + c * 5;
            intermediates[iidx + 0] = mu_x;
            intermediates[iidx + 1] = mu_y;
            intermediates[iidx + 2] = sigma_x_sq;
            intermediates[iidx + 3] = sigma_y_sq;
            intermediates[iidx + 4] = sigma_xy;

            // SSIM for this channel
            float A  = 2.0f * mu_x * mu_y + SSIM_C1;
            float B  = 2.0f * sigma_xy + SSIM_C2;
            float Cd = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
            float D  = sigma_x_sq + sigma_y_sq + SSIM_C2;
            ssim_sum += (A * B) / (Cd * D);

            // L1 for this channel
            float gt_v  = gt[(py * W + px) * 3 + c];
            float rd_v  = rendered[(py * W + px) * 3 + c];
            l1_sum += fabs(gt_v - rd_v);
        }
    }

    // Pixel loss
    float pixel_loss = 0.0f;
    if (px < W && py < H) {
        pixel_loss = ssim_weight * (1.0f - ssim_sum / 3.0f) + (1.0f - ssim_weight) * l1_sum / 3.0f;
    }

    // Threadgroup reduction → atomic add to loss_sum
    threadgroup float tg_sum[256];
    tg_sum[tr] = pixel_loss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tg_total = tg_size.x * tg_size.y;
    for (uint s = tg_total / 2; s > 0; s >>= 1) {
        if (tr < s) tg_sum[tr] += tg_sum[tr + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tr == 0) {
        atomic_fetch_add_explicit(loss_sum, tg_sum[0], memory_order_relaxed);
    }
}

// Backward pass 1: compute derivative fields + horizontal convolution.
// For each pixel, computes F1, F2, F3 from intermediates, then convolves horizontally.
// Output: ssim_h_buf (H, W, 15) — 3 values per channel at stride 5
kernel void ssim_h_bwd_kernel(
    constant float* intermediates,  // (H, W, 15) from forward
    constant uint2& img_size,       // (W, H)
    device float* ssim_h_buf,       // (H, W, 15) — reused from forward
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const int base_gx = (int)(tgid.x * SSIM_TG) - SSIM_HALF_WIN;
    const int base_gy = (int)(tgid.y * SSIM_TG);
    constexpr uint TILE_W = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = SSIM_TG * TILE_W;         // 416

    // Compute derivative fields for all 3 channels: 9 × 16×26 × 4B = 14.6KB
    threadgroup float tg_f1[3][SSIM_TG][TILE_W];
    threadgroup float tg_f2[3][SSIM_TG][TILE_W];
    threadgroup float tg_f3[3][SSIM_TG][TILE_W];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / TILE_W;
            uint sx = i % TILE_W;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            float f1 = 0, f2 = 0, f3 = 0;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint ii = (gy * W + gx) * 15 + c * 5;
                float mu_x  = intermediates[ii + 0];
                float mu_y  = intermediates[ii + 1];
                float sx_sq = intermediates[ii + 2];
                float sy_sq = intermediates[ii + 3];
                float sxy   = intermediates[ii + 4];

                float A   = 2.0f * mu_x * mu_y + SSIM_C1;
                float B   = 2.0f * sxy + SSIM_C2;
                float Cd  = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
                float D   = sx_sq + sy_sq + SSIM_C2;
                float iCD = 1.0f / (Cd * D);

                float dmu  = 2.0f * B * (mu_x * Cd - A * mu_y) / (Cd * Cd * D);
                float dsyq = -A * B * iCD / D;
                float dsxy = 2.0f * A * iCD;

                f1 = dmu - 2.0f * mu_y * dsyq - mu_x * dsxy;
                f2 = 2.0f * dsyq;
                f3 = dsxy;
            }
            tg_f1[c][sy][sx] = f1;
            tg_f2[c][sy][sx] = f2;
            tg_f3[c][sy][sx] = f3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (px < W && py < H) {
        for (uint c = 0; c < 3; c++) {
            float h1 = 0, h2 = 0, h3 = 0;
            for (uint dx = 0; dx < SSIM_WIN; dx++) {
                float w = GAUSS_1D[SSIM_WIN - 1 - dx];
                h1 += w * tg_f1[c][lid.y][lid.x + dx];
                h2 += w * tg_f2[c][lid.y][lid.x + dx];
                h3 += w * tg_f3[c][lid.y][lid.x + dx];
            }
            uint out = (py * W + px) * 15 + c * 5;
            ssim_h_buf[out + 0] = h1;
            ssim_h_buf[out + 1] = h2;
            ssim_h_buf[out + 2] = h3;
        }
    }
}

// Backward pass 2: vertical convolution + combine to produce v_rendered.
// Output: v_rendered (H, W, 3)
kernel void ssim_v_bwd_kernel(
    constant float* rendered,       // (H, W, 3)
    constant float* gt,             // (H, W, 3)
    constant float* ssim_h_buf,     // (H, W, 15)
    constant uint2& img_size,       // (W, H)
    constant float& ssim_weight,
    constant float& inv_n,          // 1.0 / (H * W * 3)
    device float* v_rendered,       // (H, W, 3)
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const int base_gx = (int)(tgid.x * SSIM_TG);
    const int base_gy = (int)(tgid.y * SSIM_TG) - SSIM_HALF_WIN;
    constexpr uint TILE_H = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = TILE_H * SSIM_TG;         // 416

    // Load all 3 channels at once: 9 × 26×16 × 4B = 14.6KB
    threadgroup float tg_h1[3][TILE_H][SSIM_TG];
    threadgroup float tg_h2[3][TILE_H][SSIM_TG];
    threadgroup float tg_h3[3][TILE_H][SSIM_TG];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / SSIM_TG;
            uint sx = i % SSIM_TG;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            float v1 = 0, v2 = 0, v3 = 0;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint hp = (gy * W + gx) * 15 + c * 5;
                v1 = ssim_h_buf[hp + 0];
                v2 = ssim_h_buf[hp + 1];
                v3 = ssim_h_buf[hp + 2];
            }
            tg_h1[c][sy][sx] = v1;
            tg_h2[c][sy][sx] = v2;
            tg_h3[c][sy][sx] = v3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (px < W && py < H) {
        for (uint c = 0; c < 3; c++) {
            float conv_f1 = 0, conv_f2 = 0, conv_f3 = 0;
            for (uint dy = 0; dy < SSIM_WIN; dy++) {
                float w = GAUSS_1D[SSIM_WIN - 1 - dy];
                conv_f1 += w * tg_h1[c][lid.y + dy][lid.x];
                conv_f2 += w * tg_h2[c][lid.y + dy][lid.x];
                conv_f3 += w * tg_h3[c][lid.y + dy][lid.x];
            }

            float rend_val = rendered[(py * W + px) * 3 + c];
            float gt_val = gt[(py * W + px) * 3 + c];

            float v_ssim = conv_f1 + rend_val * conv_f2 + gt_val * conv_f3;
            float v_l1 = (gt_val > rend_val) ? -1.0f : ((gt_val < rend_val) ? 1.0f : 0.0f);

            v_rendered[(py * W + px) * 3 + c] = inv_n * (
                -ssim_weight * v_ssim + (1.0f - ssim_weight) * v_l1
            );
        }
    }
}

// ============================================================
// GPU Densification Kernels (Phase 3)
// ============================================================

#define DENSIFY_NOTHING 0
#define DENSIFY_SPLIT   1
#define DENSIFY_DUP     2

// Classify each gaussian as split, dup, or nothing based on gradient and scale thresholds.
kernel void densify_classify_kernel(
    constant int& N,
    constant float* xys_grad_norm    [[buffer(1)]],
    constant float* vis_counts       [[buffer(2)]],
    constant float* scales           [[buffer(3)]],  // [N,3] log-space
    constant float* max_2d_size      [[buffer(4)]],
    constant float& half_max_dim     [[buffer(5)]],  // 0.5 * max(W,H)
    constant float& grad_thresh      [[buffer(6)]],
    constant float& size_thresh      [[buffer(7)]],
    constant float& screen_thresh    [[buffer(8)]],
    constant int& check_screen       [[buffer(9)]],
    device int* split_flag           [[buffer(10)]],
    device int* dup_flag             [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N) return;
    float vc = vis_counts[idx];
    if (vc <= 0.0f) { split_flag[idx] = 0; dup_flag[idx] = 0; return; }

    float avg_grad = (xys_grad_norm[idx] / vc) * half_max_dim;
    bool high_grad = avg_grad > grad_thresh;

    float s0 = scales[idx*3], s1 = scales[idx*3+1], s2 = scales[idx*3+2];
    float max_scale = max(max(exp(s0), exp(s1)), exp(s2));
    bool is_large = max_scale > size_thresh;

    bool do_split = is_large;
    if (check_screen && max_2d_size[idx] > screen_thresh) do_split = true;
    do_split = do_split && high_grad;

    bool do_dup = !is_large && high_grad;

    split_flag[idx] = do_split ? 1 : 0;
    dup_flag[idx]   = do_dup   ? 1 : 0;
}

// Append split children into backing buffers. One thread per original gaussian.
// Each split gaussian produces 2 children at [N + 2*(ord)], [N + 2*(ord)+1].
// Also shrinks parent scale by 1/1.6 and zeros optimizer state for children.
kernel void densify_append_split_kernel(
    constant int& N,
    constant int* split_flag         [[buffer(1)]],
    constant int* split_prefix       [[buffer(2)]],  // inclusive prefix sum
    constant float* random_samples   [[buffer(3)]],  // [2*N, 3] randn
    constant float& log_size_fac     [[buffer(4)]],  // log(1.6)
    device float* means_buf          [[buffer(5)]],
    device float* scales_buf         [[buffer(6)]],
    device float* quats_buf          [[buffer(7)]],
    device float* featuresDc_buf     [[buffer(8)]],
    device float* featuresRest_buf   [[buffer(9)]],
    device float* opacities_buf      [[buffer(10)]],
    constant int& fr_stride          [[buffer(11)]],  // featuresRest stride (e.g. 45)
    device float* adam_ea0           [[buffer(12)]],  // adam_exp_avg_buf[0..5]
    device float* adam_ea1           [[buffer(13)]],
    device float* adam_ea2           [[buffer(14)]],
    device float* adam_ea3           [[buffer(15)]],
    device float* adam_ea4           [[buffer(16)]],
    device float* adam_ea5           [[buffer(17)]],
    device float* adam_es0           [[buffer(18)]],  // adam_exp_avg_sq_buf[0..5]
    device float* adam_es1           [[buffer(19)]],
    device float* adam_es2           [[buffer(20)]],
    device float* adam_es3           [[buffer(21)]],
    device float* adam_es4           [[buffer(22)]],
    device float* adam_es5           [[buffer(23)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N || split_flag[idx] == 0) return;

    int ord = split_prefix[idx] - 1;  // 0-based ordinal among splits
    int c0 = N + 2 * ord;             // child 0 position
    int c1 = c0 + 1;                  // child 1 position

    // Read parent quaternion and normalize
    float qw = quats_buf[idx*4], qx = quats_buf[idx*4+1];
    float qy = quats_buf[idx*4+2], qz = quats_buf[idx*4+3];
    float qlen = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen;

    // Parent scale (exp)
    float sx = exp(scales_buf[idx*3]), sy = exp(scales_buf[idx*3+1]), sz = exp(scales_buf[idx*3+2]);

    // For each of 2 children
    for (int k = 0; k < 2; k++) {
        int child = (k == 0) ? c0 : c1;
        int rand_idx = ord * 2 + k;

        // Scale random sample by parent scale
        float r0 = random_samples[rand_idx*3]   * sx;
        float r1 = random_samples[rand_idx*3+1] * sy;
        float r2 = random_samples[rand_idx*3+2] * sz;

        // Rotate by parent quaternion: v' = R @ v
        float v0 = (1-2*(qy*qy+qz*qz))*r0 + 2*(qx*qy-qw*qz)*r1 + 2*(qx*qz+qw*qy)*r2;
        float v1 = 2*(qx*qy+qw*qz)*r0 + (1-2*(qx*qx+qz*qz))*r1 + 2*(qy*qz-qw*qx)*r2;
        float v2 = 2*(qx*qz-qw*qy)*r0 + 2*(qy*qz+qw*qx)*r1 + (1-2*(qx*qx+qy*qy))*r2;

        // Child position = parent + rotated offset
        means_buf[child*3]   = means_buf[idx*3]   + v0;
        means_buf[child*3+1] = means_buf[idx*3+1] + v1;
        means_buf[child*3+2] = means_buf[idx*3+2] + v2;

        // Child scale = shrunk parent scale
        scales_buf[child*3]   = scales_buf[idx*3]   - log_size_fac;
        scales_buf[child*3+1] = scales_buf[idx*3+1] - log_size_fac;
        scales_buf[child*3+2] = scales_buf[idx*3+2] - log_size_fac;

        // Copy parent quaternion, featuresDc, opacities
        for (int j = 0; j < 4; j++) quats_buf[child*4+j] = quats_buf[idx*4+j];
        for (int j = 0; j < 3; j++) featuresDc_buf[child*3+j] = featuresDc_buf[idx*3+j];
        for (int j = 0; j < fr_stride; j++) featuresRest_buf[child*fr_stride+j] = featuresRest_buf[idx*fr_stride+j];
        opacities_buf[child] = opacities_buf[idx];

        // Zero optimizer state for children (strides: 3,3,4,3,fr_stride,1)
        for (int j = 0; j < 3; j++) { adam_ea0[child*3+j] = 0; adam_es0[child*3+j] = 0; }
        for (int j = 0; j < 3; j++) { adam_ea1[child*3+j] = 0; adam_es1[child*3+j] = 0; }
        for (int j = 0; j < 4; j++) { adam_ea2[child*4+j] = 0; adam_es2[child*4+j] = 0; }
        for (int j = 0; j < 3; j++) { adam_ea3[child*3+j] = 0; adam_es3[child*3+j] = 0; }
        for (int j = 0; j < fr_stride; j++) { adam_ea4[child*fr_stride+j] = 0; adam_es4[child*fr_stride+j] = 0; }
        adam_ea5[child] = 0; adam_es5[child] = 0;
    }

    // Shrink parent scale in-place
    scales_buf[idx*3]   -= log_size_fac;
    scales_buf[idx*3+1] -= log_size_fac;
    scales_buf[idx*3+2] -= log_size_fac;
}

// Append duplicate copies into backing buffers. One thread per original gaussian.
// Each dup produces 1 copy at [N + 2*nSplits + dup_ord].
kernel void densify_append_dup_kernel(
    constant int& N,
    constant int* dup_flag           [[buffer(1)]],
    constant int* dup_prefix         [[buffer(2)]],  // inclusive prefix sum
    constant int* split_prefix       [[buffer(3)]],  // to read nSplits = split_prefix[N-1]
    device float* means_buf          [[buffer(4)]],
    device float* scales_buf         [[buffer(5)]],
    device float* quats_buf          [[buffer(6)]],
    device float* featuresDc_buf     [[buffer(7)]],
    device float* featuresRest_buf   [[buffer(8)]],
    device float* opacities_buf      [[buffer(9)]],
    constant int& fr_stride          [[buffer(10)]],
    device float* adam_ea0           [[buffer(11)]],
    device float* adam_ea1           [[buffer(12)]],
    device float* adam_ea2           [[buffer(13)]],
    device float* adam_ea3           [[buffer(14)]],
    device float* adam_ea4           [[buffer(15)]],
    device float* adam_ea5           [[buffer(16)]],
    device float* adam_es0           [[buffer(17)]],
    device float* adam_es1           [[buffer(18)]],
    device float* adam_es2           [[buffer(19)]],
    device float* adam_es3           [[buffer(20)]],
    device float* adam_es4           [[buffer(21)]],
    device float* adam_es5           [[buffer(22)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N || dup_flag[idx] == 0) return;

    int nSplits = (N > 0) ? split_prefix[N - 1] : 0;
    int ord = dup_prefix[idx] - 1;
    int dst = N + 2 * nSplits + ord;

    // Copy all parent data
    for (int j = 0; j < 3; j++) means_buf[dst*3+j] = means_buf[idx*3+j];
    for (int j = 0; j < 3; j++) scales_buf[dst*3+j] = scales_buf[idx*3+j];
    for (int j = 0; j < 4; j++) quats_buf[dst*4+j] = quats_buf[idx*4+j];
    for (int j = 0; j < 3; j++) featuresDc_buf[dst*3+j] = featuresDc_buf[idx*3+j];
    for (int j = 0; j < fr_stride; j++) featuresRest_buf[dst*fr_stride+j] = featuresRest_buf[idx*fr_stride+j];
    opacities_buf[dst] = opacities_buf[idx];

    // Zero optimizer state
    for (int j = 0; j < 3; j++) { adam_ea0[dst*3+j] = 0; adam_es0[dst*3+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea1[dst*3+j] = 0; adam_es1[dst*3+j] = 0; }
    for (int j = 0; j < 4; j++) { adam_ea2[dst*4+j] = 0; adam_es2[dst*4+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea3[dst*3+j] = 0; adam_es3[dst*3+j] = 0; }
    for (int j = 0; j < fr_stride; j++) { adam_ea4[dst*fr_stride+j] = 0; adam_es4[dst*fr_stride+j] = 0; }
    adam_ea5[dst] = 0; adam_es5[dst] = 0;
}

// Classify each post-growth gaussian as keep or cull.
// N_old = pre-growth count. N_new = N_old + 2*nSplits + nDups (computed from prefix sums).
// Dispatch with grid_size = worst_case (e.g. 3*N_old).
kernel void densify_cull_classify_kernel(
    constant int& N_old,
    constant int* split_prefix       [[buffer(1)]],  // [N_old] inclusive
    constant int* dup_prefix         [[buffer(2)]],  // [N_old] inclusive
    constant int* split_flag         [[buffer(3)]],  // [N_old] — marks split parents
    constant float* opacities_buf    [[buffer(4)]],
    constant float* scales_buf       [[buffer(5)]],
    constant float* max_2d_size      [[buffer(6)]],  // [N_old] only valid for idx < N_old
    constant float& cull_alpha_thresh [[buffer(7)]],  // 0.1
    constant float& cull_scale_thresh [[buffer(8)]],  // 0.5
    constant float& cull_screen_size  [[buffer(9)]],  // 0.15
    constant int& check_huge         [[buffer(10)]],
    constant int& check_screen       [[buffer(11)]],
    device int* keep_flag            [[buffer(12)]],
    uint idx [[thread_position_in_grid]]
) {
    int nSplits = (N_old > 0) ? split_prefix[N_old - 1] : 0;
    int nDups   = (N_old > 0) ? dup_prefix[N_old - 1] : 0;
    int N_new = N_old + 2 * nSplits + nDups;

    if (idx >= (uint)N_new) { keep_flag[idx] = 0; return; }

    // Sigmoid of opacity
    float opacity_sigmoid = 1.0f / (1.0f + exp(-opacities_buf[idx]));
    bool cull = opacity_sigmoid < cull_alpha_thresh;

    // Split parents are always culled
    if (idx < (uint)N_old && split_flag[idx] != 0) cull = true;

    // Huge gaussians
    if (check_huge) {
        float s0 = scales_buf[idx*3], s1 = scales_buf[idx*3+1], s2 = scales_buf[idx*3+2];
        float max_s = max(max(exp(s0), exp(s1)), exp(s2));
        if (max_s > cull_scale_thresh) cull = true;
        if (check_screen && idx < (uint)N_old && max_2d_size[idx] > cull_screen_size) cull = true;
    }

    keep_flag[idx] = cull ? 0 : 1;
}

// Scatter kept elements from src to dst at compacted positions.
// One thread per float (elem * stride + sub).
kernel void compact_scatter_kernel(
    constant float* src              [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant int* keep_prefix        [[buffer(2)]],
    constant int* keep_flag          [[buffer(3)]],
    constant uint& N                 [[buffer(4)]],
    constant uint& stride            [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint elem = tid / stride;
    uint sub  = tid % stride;
    if (elem >= N || keep_flag[elem] == 0) return;
    int dst_elem = keep_prefix[elem] - 1;
    dst[dst_elem * stride + sub] = src[elem * stride + sub];
}

// Copy compacted data from scratch back to original buffer.
// Reads new_count from keep_prefix to determine bounds.
kernel void compact_copy_back_kernel(
    constant float* src              [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant int* keep_prefix        [[buffer(2)]],
    constant uint& last_prefix_idx   [[buffer(3)]],  // N_new - 1 (or worst_case - 1)
    constant uint& stride            [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int new_count = keep_prefix[last_prefix_idx];
    uint elem = tid / stride;
    uint sub  = tid % stride;
    if (elem >= (uint)new_count) return;
    dst[elem * stride + sub] = src[elem * stride + sub];
}

// ============================================================================
// Zero buffer kernel — replaces PyTorch .zero_() MPS dispatches.
// Writes 0 as uint32, which is the zero bit-pattern for float32, int32, etc.
// ============================================================================
kernel void zero_buffer_kernel(
    device uint* buf           [[buffer(0)]],
    constant uint& count       [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) buf[idx] = 0;
}
