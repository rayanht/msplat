#import "bindings.h"
#define BLOCK_X 16
#define BLOCK_Y 16

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <chrono>
#import <unordered_map>
#import <functional>
#import <array>

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    dispatch_queue_t d_queue;

    // Command buffer lifecycle using MPSCommandBuffer for commitAndContinue support.
    MPSCommandBuffer* _currentCB = nil;

    id<MTLCommandBuffer> getCommandBuffer() {
        if (!_currentCB) {
            _currentCB = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            [_currentCB retain];
        }
        return _currentCB;
    }
    void commitCB() {
        if (_currentCB) {
            [_currentCB commitAndContinue];
        }
    }
    void syncCB() {
        if (_currentCB) {
            [_currentCB commit];
            [_currentCB waitUntilCompleted];
            [_currentCB release];
            _currentCB = nil;
        }
    }

    // Forward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_forward_kernel_cpso;
    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso;
    id<MTLComputePipelineState> pack_sorted_gaussians_kernel_cpso;
    // Tile-local sorting
    id<MTLComputePipelineState> count_intersections_per_tile_kernel_cpso;
    id<MTLComputePipelineState> scatter_to_tiles_kernel_cpso;
    id<MTLComputePipelineState> bitonic_sort_per_tile_kernel_cpso;
    // Prefix sum
    id<MTLComputePipelineState> prefix_sum_kernel_cpso;
    id<MTLComputePipelineState> block_reduce_kernel_cpso;
    id<MTLComputePipelineState> block_scan_propagate_kernel_cpso;
    // Depth-chunked rasterization
    id<MTLComputePipelineState> rasterize_forward_chunked_kernel_cpso;
    id<MTLComputePipelineState> rasterize_forward_merge_kernel_cpso;
    id<MTLComputePipelineState> compute_chunk_prefix_suffix_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_chunked_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_kernel_cpso;
    // Separable SSIM loss kernels
    id<MTLComputePipelineState> ssim_h_fwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_v_fwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_h_bwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_v_bwd_kernel_cpso;
    // Backward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_backward_kernel_cpso;
    id<MTLComputePipelineState> fused_adam_kernel_cpso;
    id<MTLComputePipelineState> accumulate_grad_stats_kernel_cpso;
    // GPU densification kernels
    id<MTLComputePipelineState> densify_classify_kernel_cpso;
    id<MTLComputePipelineState> densify_append_split_kernel_cpso;
    id<MTLComputePipelineState> densify_append_dup_kernel_cpso;
    id<MTLComputePipelineState> densify_cull_classify_kernel_cpso;
    id<MTLComputePipelineState> compact_scatter_kernel_cpso;
    id<MTLComputePipelineState> compact_copy_back_kernel_cpso;
};

unsigned num_sh_bases(const unsigned degree) {
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

// Explicit metallib path (set by Swift/Python wrappers before first use)
static char* g_metallib_path = NULL;

extern "C" void msplat_set_metallib_path(const char* path) {
    free(g_metallib_path);
    g_metallib_path = path ? strdup(path) : NULL;
}

// Fallback: NSBundle lookup for when metallib is co-located with the binary
@interface DummyClassForPathHack : NSObject
@end
@implementation DummyClassForPathHack
@end

MetalContext* init_msplat_metal_context() {
    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    // Retrieve the default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Configure context
    ctx->device = device;
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->d_queue = dispatch_queue_create("com.msplat.metal", DISPATCH_QUEUE_SERIAL);

    NSError *error = nil;

    id<MTLLibrary> metal_library = nil;
    NSString *path_lib = nil;

    // Check explicit path first (set by XCFramework / Python wrapper)
    if (g_metallib_path) {
        path_lib = [NSString stringWithUTF8String:g_metallib_path];
    }
    // Fallback: NSBundle lookup (works when metallib is in the app/build bundle)
    if (!path_lib) {
        NSBundle *bundle = [NSBundle bundleForClass:[DummyClassForPathHack class]];
        path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
    }

    if (path_lib != nil) {
        // pre-compiled library found
        NSURL * libURL = [NSURL fileURLWithPath:path_lib];
        metal_library = [ctx->device newLibraryWithURL:libURL error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }
    } else {
        NSString * source_path = [[@ __FILE__ stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"msplat_metal.metal"];

        NSString * src = [NSString stringWithContentsOfFile:source_path encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }

        @autoreleasepool {
            // dictionary of preprocessor macros
            NSMutableDictionary * prep = [NSMutableDictionary dictionary];

            MTLCompileOptions* options = [MTLCompileOptions new];
            options.preprocessorMacros = prep;
            // Note: mathMode defaults to MTLMathModeFast, fastMathEnabled defaults to YES

            metal_library = [ctx->device newLibraryWithSource:src options:options error:&error];
            if (error) {
                fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
        }
    }

    auto load = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [metal_library newFunctionWithName:name];
        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
        [fn release];
        return pso;
    };

    // Forward pipeline
    ctx->project_and_sh_forward_kernel_cpso       = load(@"project_and_sh_forward_kernel");
    ctx->nd_rasterize_forward_kernel_cpso         = load(@"nd_rasterize_forward_kernel");
    ctx->pack_sorted_gaussians_kernel_cpso        = load(@"pack_sorted_gaussians_kernel");
    // Tile-local sorting
    ctx->count_intersections_per_tile_kernel_cpso  = load(@"count_intersections_per_tile_kernel");
    ctx->scatter_to_tiles_kernel_cpso             = load(@"scatter_to_tiles_kernel");
    ctx->bitonic_sort_per_tile_kernel_cpso        = load(@"bitonic_sort_per_tile_kernel");
    // Prefix sum
    ctx->prefix_sum_kernel_cpso                   = load(@"prefix_sum_kernel");
    ctx->block_reduce_kernel_cpso                 = load(@"block_reduce_kernel");
    ctx->block_scan_propagate_kernel_cpso         = load(@"block_scan_propagate_kernel");
    // Depth-chunked rasterization
    ctx->rasterize_forward_chunked_kernel_cpso    = load(@"rasterize_forward_chunked_kernel");
    ctx->rasterize_forward_merge_kernel_cpso      = load(@"rasterize_forward_merge_kernel");
    ctx->compute_chunk_prefix_suffix_kernel_cpso  = load(@"compute_chunk_prefix_suffix_kernel");
    ctx->rasterize_backward_chunked_kernel_cpso   = load(@"rasterize_backward_chunked_kernel");
    ctx->rasterize_backward_kernel_cpso           = load(@"rasterize_backward_kernel");
    // Separable SSIM loss
    ctx->ssim_h_fwd_kernel_cpso                   = load(@"ssim_h_fwd_kernel");
    ctx->ssim_v_fwd_kernel_cpso                   = load(@"ssim_v_fwd_kernel");
    ctx->ssim_h_bwd_kernel_cpso                   = load(@"ssim_h_bwd_kernel");
    ctx->ssim_v_bwd_kernel_cpso                   = load(@"ssim_v_bwd_kernel");
    // Backward pipeline
    ctx->project_and_sh_backward_kernel_cpso      = load(@"project_and_sh_backward_kernel");
    ctx->fused_adam_kernel_cpso                    = load(@"fused_adam_kernel");
    ctx->accumulate_grad_stats_kernel_cpso        = load(@"accumulate_grad_stats_kernel");
    // GPU densification
    ctx->densify_classify_kernel_cpso             = load(@"densify_classify_kernel");
    ctx->densify_append_split_kernel_cpso         = load(@"densify_append_split_kernel");
    ctx->densify_append_dup_kernel_cpso           = load(@"densify_append_dup_kernel");
    ctx->densify_cull_classify_kernel_cpso        = load(@"densify_cull_classify_kernel");
    ctx->compact_scatter_kernel_cpso              = load(@"compact_scatter_kernel");
    ctx->compact_copy_back_kernel_cpso            = load(@"compact_copy_back_kernel");

    if (error) {
        fprintf(stderr, "msplat: failed to load kernel: %s\n", [[error description] UTF8String]);
        [metal_library release];
        return NULL;
    }

    [metal_library release];

    return ctx;
}

MetalContext* get_global_context() {
    static MetalContext* ctx = NULL;
    if (ctx == NULL) {
        ctx = init_msplat_metal_context();
    }
    return ctx;
}



#define ENC_SCALAR(encoder, x, i) [encoder setBytes:&x length:sizeof(x) atIndex:i]
#define ENC_ARRAY(encoder, x, i) [encoder setBytes:x length:sizeof(x) atIndex:i]
#define ENC_BUF(encoder, x, i) [encoder setBuffer:x.buffer() offset:0 atIndex:i]

id<MTLDevice> msplat_device() {
    return get_global_context()->device;
}

MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype) {
    return mtensor_zeros(get_global_context()->device, std::move(shape), dtype);
}

MTensor gpu_empty(std::vector<int64_t> shape, DType dtype) {
    return mtensor_empty(get_global_context()->device, std::move(shape), dtype);
}

void msplat_commit() {
    get_global_context()->commitCB();
}

void msplat_gpu_sync() {
    get_global_context()->syncCB();
}

#define RS_TG_SIZE 256
#define RS_RADIX 256
#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8

static bool g_profile_stages = false;
static bool g_profile_stages_checked = false;

// Cached buffer pool — all intermediate GPU buffers are reused across iterations.
// Sizes only change at densification (every 100 steps); between densifications
// this eliminates all per-iteration GPU allocations.
struct FusedTensorCache {
    int fwd_num_points = 0, capacity = 0, img_height = 0, img_width = 0, num_tiles = 0;
    int bwd_num_points = 0, features_rest_bases = 0;

    // Forward intermediates
    MTensor xys, depths, radii_out, conics, num_tiles_hit, colors, aabb;
    MTensor cum_tiles_hit;
    MTensor isect_ids, gaussian_ids;
    MTensor packed_xy_opac, packed_conic, packed_rgb;
    MTensor out_img, final_Ts, final_idx;
    MTensor loss_intermediates;
    MTensor ssim_h_buf;
    MTensor tile_bins, loss_sum;

    // Sort temp buffers
    MTensor sort_keys_tmp, sort_vals_tmp, sort_counts;

    // Tile-local sorting buffers
    MTensor tile_counts, tile_offsets, tile_scatter_counters;

    // Multi-threadgroup prefix sum temp buffer
    MTensor block_totals;

    // Intersection overflow detection
    MTensor overflow_flag;
    int64_t capacity_multiplier = 16;

    // Depth-chunked rasterization buffers
    uint32_t current_K_max = 1;
    int chunk_K_max = 0;
    MTensor chunk_T, chunk_C, chunk_final_idx;
    MTensor prefix_T, after_C;

    // Backward gradient accumulators
    MTensor v_rendered;
    MTensor v_xy, v_conic, v_colors_rast, v_opacity, v_depth;
    MTensor v_mean3d, v_scale, v_quat, v_features_dc, v_features_rest;

    void ensure_forward(int np, int64_t cap, int ih, int iw, int nt,
                        id<MTLDevice> dev) {
        if (np != fwd_num_points) {
            fwd_num_points = np;
            xys = mtensor_empty(dev, {np, 2}, DType::Float32);
            depths = mtensor_empty(dev, {np}, DType::Float32);
            radii_out = mtensor_empty(dev, {np}, DType::Int32);
            conics = mtensor_empty(dev, {np, 3}, DType::Float32);
            num_tiles_hit = mtensor_empty(dev, {np}, DType::Int32);
            colors = mtensor_empty(dev, {np, 3}, DType::Float32);
            aabb = mtensor_empty(dev, {np, 2}, DType::Float32);
            cum_tiles_hit = mtensor_empty(dev, {np}, DType::Int32);
            block_totals = mtensor_empty(dev, {(np + 1023) / 1024}, DType::Int32);
        }
        if (cap != capacity) {
            capacity = cap;
            isect_ids = mtensor_empty(dev, {cap}, DType::Int64);
            gaussian_ids = mtensor_empty(dev, {cap}, DType::Int32);
            packed_xy_opac = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_conic = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_rgb = mtensor_empty(dev, {cap, 3}, DType::Float32);
            int64_t rs_num_blocks = (cap + RS_TG_SIZE - 1) / RS_TG_SIZE;
            sort_keys_tmp = mtensor_empty(dev, {cap}, DType::Int64);
            sort_vals_tmp = mtensor_empty(dev, {cap}, DType::Int32);
            sort_counts = mtensor_empty(dev, {rs_num_blocks * RS_RADIX}, DType::Int32);
        }
        if (ih != img_height || iw != img_width) {
            img_height = ih; img_width = iw;
            out_img = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
            final_Ts = mtensor_empty(dev, {ih, iw}, DType::Float32);
            final_idx = mtensor_empty(dev, {ih, iw}, DType::Int32);
            loss_intermediates = mtensor_empty(dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
            ssim_h_buf = mtensor_empty(dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
            v_rendered = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
        }
        if (nt != num_tiles) {
            num_tiles = nt;
            tile_bins = mtensor_empty(dev, {nt, 2}, DType::Int32);
            tile_counts = mtensor_empty(dev, {nt}, DType::Int32);
            tile_offsets = mtensor_empty(dev, {nt}, DType::Int32);
            tile_scatter_counters = mtensor_empty(dev, {nt}, DType::Int32);
        }
        if (!loss_sum.defined()) {
            loss_sum = mtensor_empty(dev, {1}, DType::Float32);
        }
        if (!overflow_flag.defined()) {
            overflow_flag = mtensor_empty(dev, {1}, DType::Int32);
        }
    }

    void ensure_chunks(int K, int ih, int iw, id<MTLDevice> dev) {
        if (K <= chunk_K_max && ih == img_height && iw == img_width) return;
        chunk_K_max = K;
        chunk_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        chunk_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
        chunk_final_idx = mtensor_empty(dev, {K, ih, iw}, DType::Int32);
        prefix_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        after_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
    }

    void ensure_backward(int np, int frb, id<MTLDevice> dev) {
        if (np != bwd_num_points || frb != features_rest_bases || !v_xy.defined()) {
            bwd_num_points = np;
            features_rest_bases = frb;
            v_xy = mtensor_empty(dev, {np, 2}, DType::Float32);
            v_conic = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_colors_rast = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_opacity = mtensor_empty(dev, {np, 1}, DType::Float32);
            v_depth = mtensor_empty(dev, {np}, DType::Float32);
            v_mean3d = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_scale = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_quat = mtensor_empty(dev, {np, 4}, DType::Float32);
            v_features_dc = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_features_rest = mtensor_empty(dev, {(int64_t)np, (int64_t)frb, 3}, DType::Float32);
        }
    }
};
static FusedTensorCache g_tcache;

void cleanup_msplat_metal() {
    g_tcache = FusedTensorCache{};
}

// Internal forward pipeline — used by both msplat_render and msplat_train_step.
// When compute_loss=false, gt/window2d/ssim_weight are ignored.
static void forward_pipeline(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    bool compute_loss
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    // --- Overflow check: check every 100 iters and after densification ---
    // Checking every iteration is too expensive (synchronize drains MPS pipeline).
    // Checking too rarely risks stale sort data causing GPU hangs.
    static int iter_count_oc = 0;
    iter_count_oc++;
    bool num_points_changed = (num_points != g_tcache.fwd_num_points && g_tcache.fwd_num_points > 0);
    if (g_tcache.overflow_flag.defined() && g_tcache.fwd_num_points > 0
        && (num_points_changed || (iter_count_oc % 100) == 1)) {
        ctx->syncCB();
        int32_t flag_val = *g_tcache.overflow_flag.data<int32_t>();
        if (flag_val > 0) {
            int64_t actual_count = g_tcache.cum_tiles_hit.data<int32_t>()[g_tcache.fwd_num_points - 1];
            int64_t new_mult = (actual_count * 3 / 2 + num_points - 1) / num_points;
            g_tcache.capacity_multiplier = std::max(g_tcache.capacity_multiplier, std::max(new_mult, (int64_t)3));
            fprintf(stderr, "WARNING: intersection overflow (actual=%lld > capacity=%lld). "
                    "Increasing multiplier to %lldx for future iterations.\n",
                    (long long)actual_count, (long long)g_tcache.capacity,
                    (long long)g_tcache.capacity_multiplier);
        }
    }
    int64_t capacity = (int64_t)num_points * g_tcache.capacity_multiplier;
    uint32_t channels = 3;

    // --- Cached buffer pool: only reallocate on dimension change (densification) ---
    g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles, ctx->device);
    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &cum_tiles_hit = g_tcache.cum_tiles_hit;
    MTensor &isect_ids = g_tcache.isect_ids;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &loss_intermediates = g_tcache.loss_intermediates;

    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});

    // --- Constants (heap-allocated for Obj-C block) ---
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 3>>(std::array<float, 3>{cam_pos[0], cam_pos[1], cam_pos[2]});
    uint32_t num_points_u32 = (uint32_t)num_points;
    uint32_t capacity_u32 = (uint32_t)capacity;
    uint32_t prefix_N = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});

    // Periodic diagnostic: print key dimensions for roofline analysis
    static int diag_count = 0;
    diag_count++;
    if (std::getenv("BENCHMARK") && (diag_count == 100 || diag_count == 500 || diag_count == 1500)) {
            fprintf(stderr, "\n=== Roofline Dimensions (iter %d) ===\n", diag_count);
            fprintf(stderr, "  num_points:     %d\n", num_points);
            fprintf(stderr, "  capacity:       %lld (= num_points * %lld)\n", (long long)capacity, (long long)g_tcache.capacity_multiplier);
            fprintf(stderr, "  img:            %u x %u = %u pixels\n", img_width, img_height, img_width * img_height);
            fprintf(stderr, "  tiles:          %d x %d = %d\n", tile_bounds_x, tile_bounds_y, num_tiles);
            fprintf(stderr, "  SH degree:      %u (bases: %u)\n", degree, (degree + 1) * (degree + 1));
            fprintf(stderr, "  features_rest:  [%lld x %lld x %lld]\n",
                (long long)features_rest.size(0), (long long)features_rest.size(1), (long long)features_rest.size(2));
            fprintf(stderr, "  sort:           tile-local (bitonic, max 2048/tile)\n");
            fprintf(stderr, "  sort buffer:    %.1f MB (sort_pairs)\n", (double)capacity * 8.0 / 1e6);
            fprintf(stderr, "  opacities:      [%lld]\n", (long long)opacities.size(0));
            fprintf(stderr, "===========================\n\n");
    }

    // Helper lambdas to encode each stage onto a given encoder
    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        // buffer 23 removed (was opacity-aware AABB, reverted)

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_prefix_map = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_tiles_u32 = (uint32_t)num_tiles;
        // 1. prefix_sum(num_tiles_hit -> cum_tiles_hit) — multi-threadgroup path
        //    Pass 1: block_reduce — each of K threadgroups sums 1024 elements
        {
            uint32_t K = (uint32_t)((num_points + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, prefix_N, 0); ENC_BUF(enc, num_tiles_hit, 1);
            ENC_BUF(enc, g_tcache.block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        //    Pass 2: block_scan_propagate — each threadgroup applies offset + local prefix sum
        {
            uint32_t K = (uint32_t)((num_points + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, prefix_N, 0); ENC_BUF(enc, num_tiles_hit, 1);
            ENC_BUF(enc, cum_tiles_hit, 2); ENC_BUF(enc, g_tcache.block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 2. count_intersections_per_tile
        {
            NSUInteger tpg = MIN(ctx->count_intersections_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->count_intersections_per_tile_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1); ENC_BUF(enc, radii_out, 2); ENC_BUF(enc, aabb, 3);
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:4];
            ENC_BUF(enc, g_tcache.tile_counts, 5);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 3. prefix_sum(tile_counts -> tile_offsets)
        {
            NSUInteger tg2 = MIN(ctx->prefix_sum_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
            [enc setComputePipelineState:ctx->prefix_sum_kernel_cpso];
            ENC_SCALAR(enc, num_tiles_u32, 0); ENC_BUF(enc, g_tcache.tile_counts, 1); ENC_BUF(enc, g_tcache.tile_offsets, 2);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 4. scatter_to_tiles
        {
            NSUInteger tpg = MIN(ctx->scatter_to_tiles_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->scatter_to_tiles_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1); ENC_BUF(enc, depths, 2);
            ENC_BUF(enc, radii_out, 3); ENC_BUF(enc, aabb, 4);
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:5];
            ENC_BUF(enc, g_tcache.tile_offsets, 6); ENC_BUF(enc, g_tcache.tile_counts, 7);
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 8);
            ENC_BUF(enc, isect_ids, 9);  // reused as sort_pairs (uint64)
            ENC_BUF(enc, tile_bins, 10);
            ENC_SCALAR(enc, capacity_u32, 11); ENC_BUF(enc, g_tcache.overflow_flag, 12);
            ENC_SCALAR(enc, num_tiles_u32, 13);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 5. bitonic_sort_per_tile (fused with pack: writes packed buffers inline)
        {
            [enc setComputePipelineState:ctx->bitonic_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0); ENC_BUF(enc, g_tcache.tile_counts, 1);
            ENC_BUF(enc, isect_ids, 2);  // sort_pairs (in-place)
            ENC_BUF(enc, gaussian_ids, 3);  // output sorted gaussian IDs
            ENC_SCALAR(enc, num_tiles_u32, 4);
            ENC_BUF(enc, xys, 5); ENC_BUF(enc, conics, 6);
            ENC_BUF(enc, colors, 7); ENC_BUF(enc, opacities, 8);
            ENC_BUF(enc, packed_xy_opac, 9); ENC_BUF(enc, packed_conic, 10); ENC_BUF(enc, packed_rgb, 11);
            [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    auto encode_pack = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->pack_sorted_gaussians_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)capacity);
        [enc setComputePipelineState:ctx->pack_sorted_gaussians_kernel_cpso];
        ENC_BUF(enc, gaussian_ids, 0);  // always gaussian_ids (no ping-pong)
        ENC_BUF(enc, xys, 1); ENC_BUF(enc, conics, 2);
        ENC_BUF(enc, colors, 3); ENC_BUF(enc, opacities, 4);
        ENC_BUF(enc, packed_xy_opac, 5); ENC_BUF(enc, packed_conic, 6); ENC_BUF(enc, packed_rgb, 7);
        ENC_SCALAR(enc, capacity_u32, 8);
        ENC_BUF(enc, cum_tiles_hit, 9); ENC_SCALAR(enc, num_points_u32, 10);
        [enc dispatchThreads:MTLSizeMake(capacity, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    // K_max for chunked rasterization — set after GPU readback
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;

    auto encode_rast_fwd_monolithic = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
        ENC_BUF(enc, background, 10);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:11];
        [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:tg_size];
    };

    auto encode_rast_fwd_chunked = [&](id<MTLComputeCommandEncoder> enc) {
        // Phase 1: dispatch forward chunked kernel — grid (tile_x, tile_y, K_max)
        uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
        uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
        uint32_t num_pix = img_width * img_height;
        auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
        MTLSize chunked_tg = MTLSizeMake(tile_x, tile_y, K_max);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
        ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
        [enc dispatchThreadgroups:chunked_tg threadsPerThreadgroup:tg_size];

        // Phase 2: merge kernel — one thread per pixel
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger merge_tpg = 256;
        [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
        ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
        ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
        ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
        ENC_BUF(enc, background, 8);
        [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
        [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            encode_rast_fwd_monolithic(enc);
        } else {
            encode_rast_fwd_chunked(enc);
        }
    };

    auto encode_loss_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        // Separable SSIM forward: H conv → barrier → V conv + SSIM + reduction
        MTLSize grid = MTLSizeMake(img_width, img_height, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);

        // Pass 1: horizontal convolution
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Pass 2: vertical convolution + SSIM/L1 + loss reduction
        [enc setComputePipelineState:ctx->ssim_v_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4);
        ENC_BUF(enc, loss_intermediates, 5); ENC_BUF(enc, loss_sum, 6);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    };

    // Zero cached buffers will be done inside the command encoder via blit.
    // (CPU memset would race with previous CB's GPU reads.)

    // Compute K_max conservatively from CPU-side data (no GPU sync needed).
    // avg_per_tile = capacity / num_tiles. Use 6x average to cover heavy-tailed
    // tile distributions. Overestimate is cheap: empty chunks early-exit immediately.
    // If the densest tile exceeds K_max * CHUNK_SIZE, those gaussians are silently
    // skipped — but 6x covers typical skew (measured max/avg ratio: ~1.5-2.5x).
    if (num_tiles >= 400) {
        // High-res: enough tiles for good GPU occupancy, skip chunking
        K_max = 1;
    } else {
        uint32_t avg_per_tile = (uint32_t)(capacity / std::max(1, num_tiles));
        uint32_t conservative_max = avg_per_tile * 6;  // 6x average — covers heavy-tailed distributions
        K_max = (conservative_max + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (K_max < 2) K_max = 2;
        uint32_t abs_max = (uint32_t)((capacity + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > abs_max) K_max = abs_max;
    }
    g_tcache.current_K_max = K_max;
    if (K_max > 1) {
        g_tcache.ensure_chunks(K_max, img_height, img_width, ctx->device);
    }

    if (!g_profile_stages_checked) { g_profile_stages = std::getenv("PROFILE_STAGES") != nullptr; g_profile_stages_checked = true; }
    if (g_profile_stages) {
        // Profiling mode: separate command buffers per stage with synchronize()
        // This gives accurate per-stage GPU time (but total is higher due to pipeline bubbles)
        auto stage_time = [ctx](const char* name, std::function<void()> fn) {
            ctx->syncCB();
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            ctx->syncCB();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            static std::unordered_map<std::string, std::vector<double>> stage_times;
            stage_times[name].push_back(ms);
            auto &v = stage_times[name];
            if (v.size() % 200 == 0 && v.size() >= 200) {
                auto sorted = v;
                std::sort(sorted.begin(), sorted.end());
                fprintf(stderr, "  %-20s median=%.3fms (n=%zu)\n", name, sorted[sorted.size()/2], sorted.size());
            }
        };

        stage_time("proj_sh", [&]() {
            id<MTLCommandBuffer> cb = ctx->getCommandBuffer();
            dispatch_sync(ctx->d_queue, ^(){
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode_proj_sh(enc);
                [enc endEncoding];
            });
            ctx->commitCB();
        });
        stage_time("prefix_sort_pack", [&]() {
            id<MTLCommandBuffer> cb = ctx->getCommandBuffer();
            dispatch_sync(ctx->d_queue, ^(){
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode_prefix_map(enc);
                [enc endEncoding];
            });
            ctx->commitCB();
        });
        stage_time("rast_fwd", [&]() {
            id<MTLCommandBuffer> cb = ctx->getCommandBuffer();
            dispatch_sync(ctx->d_queue, ^(){
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode_rast_fwd(enc);
                [enc endEncoding];
            });
            ctx->commitCB();
        });
        if (compute_loss) {
            stage_time("loss_fwd", [&]() {
                id<MTLCommandBuffer> cb = ctx->getCommandBuffer();
                dispatch_sync(ctx->d_queue, ^(){
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    encode_loss_fwd(enc);
                    [enc endEncoding];
                });
                ctx->commitCB();
            });
        }
    } else {
        // Single encoder for all forward stages
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        dispatch_sync(ctx->d_queue, ^(){
            // Blit-zero buffers that accumulate across gaussians (must be GPU-side
            // to avoid racing with previous CB's reads on pipelined execution)
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            [blit fillBuffer:tile_bins.buffer() range:NSMakeRange(0, tile_bins.nbytes()) value:0];
            [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
            [blit fillBuffer:g_tcache.overflow_flag.buffer() range:NSMakeRange(0, g_tcache.overflow_flag.nbytes()) value:0];
            [blit fillBuffer:g_tcache.tile_counts.buffer() range:NSMakeRange(0, g_tcache.tile_counts.nbytes()) value:0];
            [blit fillBuffer:g_tcache.tile_scatter_counters.buffer() range:NSMakeRange(0, g_tcache.tile_scatter_counters.nbytes()) value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            assert(encoder && "Failed to create compute command encoder");

            encode_proj_sh(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_prefix_map(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(encoder);
            if (compute_loss) {
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_loss_fwd(encoder);
            }

            [encoder endEncoding];
        });
    }

    // All outputs are in g_tcache — no return needed
}

MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
) {
    MTensor dummyGt, dummyWindow;
    forward_pipeline(num_points, means3d, scales, glob_scale,
        quats, viewmat, projmat, fx, fy, cx, cy,
        img_height, img_width, tile_bounds, clip_thresh,
        degree, degrees_to_use, cam_pos, features_dc, features_rest,
        opacities, background, dummyGt, dummyWindow, 0.0f, false);
    return g_tcache.out_img;
}

std::tuple<MTensor, float> msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    // --- Overflow check: check every 100 iters and after densification ---
    // Checking every iteration is too expensive (synchronize drains MPS pipeline).
    // Checking too rarely risks stale sort data causing GPU hangs.
    static int iter_count_oc = 0;
    iter_count_oc++;
    bool num_points_changed = (num_points != g_tcache.fwd_num_points && g_tcache.fwd_num_points > 0);
    if (g_tcache.overflow_flag.defined() && g_tcache.fwd_num_points > 0
        && (num_points_changed || (iter_count_oc % 100) == 1)) {
        ctx->syncCB();
        int32_t flag_val = *g_tcache.overflow_flag.data<int32_t>();
        if (flag_val > 0) {
            int64_t actual_count = g_tcache.cum_tiles_hit.data<int32_t>()[g_tcache.fwd_num_points - 1];
            int64_t new_mult = (actual_count * 3 / 2 + num_points - 1) / num_points;
            g_tcache.capacity_multiplier = std::max(g_tcache.capacity_multiplier, std::max(new_mult, (int64_t)3));
            fprintf(stderr, "WARNING: intersection overflow (actual=%lld > capacity=%lld). "
                    "Increasing multiplier to %lldx for future iterations.\n",
                    (long long)actual_count, (long long)g_tcache.capacity,
                    (long long)g_tcache.capacity_multiplier);
        }
    }
    int64_t capacity = (int64_t)num_points * g_tcache.capacity_multiplier;
    uint32_t channels = 3;

    // --- Cached buffer pool ---
    g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles, ctx->device);
    g_tcache.ensure_backward(num_points, features_rest_bases, ctx->device);

    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &cum_tiles_hit = g_tcache.cum_tiles_hit;
    MTensor &isect_ids = g_tcache.isect_ids;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &loss_intermediates = g_tcache.loss_intermediates;

    MTensor &v_rendered = g_tcache.v_rendered;
    MTensor &v_xy = g_tcache.v_xy;
    MTensor &v_conic = g_tcache.v_conic;
    MTensor &v_colors_rast = g_tcache.v_colors_rast;
    MTensor &v_opacity = g_tcache.v_opacity;
    MTensor &v_depth = g_tcache.v_depth;
    MTensor &v_mean3d = g_tcache.v_mean3d;
    MTensor &v_scale = g_tcache.v_scale;
    MTensor &v_quat = g_tcache.v_quat;
    MTensor &v_features_dc = g_tcache.v_features_dc;
    MTensor &v_features_rest = g_tcache.v_features_rest;

    // Wire backward outputs as Adam grads (MTensor references for gradient buffers)
    auto adam_grads = std::make_shared<std::array<MTensor, 6>>(
        std::array<MTensor, 6>{v_mean3d, v_scale, v_quat, v_features_dc, v_features_rest, v_opacity});

    // --- Constants (heap-allocated for Obj-C block capture) ---
    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 3>>(std::array<float, 3>{cam_pos[0], cam_pos[1], cam_pos[2]});
    uint32_t num_points_u32 = (uint32_t)num_points;
    uint32_t capacity_u32 = (uint32_t)capacity;
    uint32_t prefix_N = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});
    // tile_bounds for rasterize kernels must be 16x16 tile counts (tile_bins granularity)
    auto rast_tb = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (img_width + 15u) / 16u,
        (img_height + 15u) / 16u, 1, 0xDEAD});
    auto rast_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_bwd_intr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_bwd_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});

    // --- K_max for chunked rasterization ---
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;
    if (num_tiles >= 400) {
        K_max = 1;
    } else {
        uint32_t avg_per_tile = (uint32_t)(capacity / std::max(1, num_tiles));
        uint32_t conservative_max = avg_per_tile * 6;
        K_max = (conservative_max + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (K_max < 2) K_max = 2;
        uint32_t abs_max = (uint32_t)((capacity + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > abs_max) K_max = abs_max;
    }
    g_tcache.current_K_max = K_max;
    if (K_max > 1) {
        g_tcache.ensure_chunks(K_max, img_height, img_width, ctx->device);
    }

    uint32_t bwd_K_max = K_max;
    constexpr uint32_t BWD_CHUNK_SIZE = 512;

    // ========================== FORWARD ENCODE LAMBDAS ==========================

    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        // buffer 23 removed (was opacity-aware AABB, reverted)

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_prefix_map = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_tiles_u32 = (uint32_t)num_tiles;
        // 1. prefix_sum (multi-threadgroup)
        {
            uint32_t K = (uint32_t)((num_points + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, prefix_N, 0); ENC_BUF(enc, num_tiles_hit, 1);
            ENC_BUF(enc, g_tcache.block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            uint32_t K = (uint32_t)((num_points + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, prefix_N, 0); ENC_BUF(enc, num_tiles_hit, 1);
            ENC_BUF(enc, cum_tiles_hit, 2); ENC_BUF(enc, g_tcache.block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 2. count_intersections_per_tile
        {
            NSUInteger tpg = MIN(ctx->count_intersections_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->count_intersections_per_tile_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1); ENC_BUF(enc, radii_out, 2); ENC_BUF(enc, aabb, 3);
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:4];
            ENC_BUF(enc, g_tcache.tile_counts, 5);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 3. prefix_sum(tile_counts -> tile_offsets)
        {
            NSUInteger tg2 = MIN(ctx->prefix_sum_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
            [enc setComputePipelineState:ctx->prefix_sum_kernel_cpso];
            ENC_SCALAR(enc, num_tiles_u32, 0); ENC_BUF(enc, g_tcache.tile_counts, 1); ENC_BUF(enc, g_tcache.tile_offsets, 2);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 4. scatter_to_tiles
        {
            NSUInteger tpg = MIN(ctx->scatter_to_tiles_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->scatter_to_tiles_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1); ENC_BUF(enc, depths, 2);
            ENC_BUF(enc, radii_out, 3); ENC_BUF(enc, aabb, 4);
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:5];
            ENC_BUF(enc, g_tcache.tile_offsets, 6); ENC_BUF(enc, g_tcache.tile_counts, 7);
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 8);
            ENC_BUF(enc, isect_ids, 9);
            ENC_BUF(enc, tile_bins, 10);
            ENC_SCALAR(enc, capacity_u32, 11); ENC_BUF(enc, g_tcache.overflow_flag, 12);
            uint32_t num_tiles_u32_local = (uint32_t)num_tiles;
            ENC_SCALAR(enc, num_tiles_u32_local, 13);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 5. bitonic_sort_per_tile (fused with pack)
        {
            uint32_t num_tiles_u32_local = (uint32_t)num_tiles;
            [enc setComputePipelineState:ctx->bitonic_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0); ENC_BUF(enc, g_tcache.tile_counts, 1);
            ENC_BUF(enc, isect_ids, 2);
            ENC_BUF(enc, gaussian_ids, 3);
            ENC_SCALAR(enc, num_tiles_u32_local, 4);
            ENC_BUF(enc, xys, 5); ENC_BUF(enc, conics, 6);
            ENC_BUF(enc, colors, 7); ENC_BUF(enc, opacities, 8);
            ENC_BUF(enc, packed_xy_opac, 9); ENC_BUF(enc, packed_conic, 10); ENC_BUF(enc, packed_rgb, 11);
            [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
            ENC_BUF(enc, background, 10);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:11];
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Merge
            [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
            ENC_BUF(enc, background, 8);
            [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
    };

    auto encode_loss_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize grid = MTLSizeMake(img_width, img_height, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:ctx->ssim_v_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4);
        ENC_BUF(enc, loss_intermediates, 5); ENC_BUF(enc, loss_sum, 6);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    };

    // ========================== BACKWARD ENCODE LAMBDAS ==========================

    auto encode_loss_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize grid = MTLSizeMake(img_width, img_height, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        [enc setComputePipelineState:ctx->ssim_h_bwd_kernel_cpso];
        ENC_BUF(enc, loss_intermediates, 0);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:1];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:ctx->ssim_v_bwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        ENC_BUF(enc, v_rendered, 6);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    };

    auto encode_rast_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (bwd_K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width+RAST_BLOCK_X-1)/RAST_BLOCK_X, (img_height+RAST_BLOCK_Y-1)/RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->rasterize_backward_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, v_rendered, 10);
            ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_conic, 12);
            ENC_BUF(enc, v_colors_rast, 13); ENC_BUF(enc, v_opacity, 14);
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked backward
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto bwd_img_sz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            // Phase 1: prefix_T and after_C
            [enc setComputePipelineState:ctx->compute_chunk_prefix_suffix_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, bwd_K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, g_tcache.prefix_T, 5); ENC_BUF(enc, g_tcache.after_C, 6);
            [enc setBytes:bwd_img_sz->data() length:sizeof(*bwd_img_sz) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Phase 2: backward chunked
            [enc setComputePipelineState:ctx->rasterize_backward_chunked_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_BUF(enc, g_tcache.prefix_T, 10); ENC_BUF(enc, g_tcache.chunk_T, 11);
            ENC_BUF(enc, g_tcache.after_C, 12);
            ENC_BUF(enc, v_rendered, 13);
            ENC_BUF(enc, v_xy, 14); ENC_BUF(enc, v_conic, 15);
            ENC_BUF(enc, v_colors_rast, 16); ENC_BUF(enc, v_opacity, 17);
            ENC_SCALAR(enc, BWD_CHUNK_SIZE, 18); ENC_SCALAR(enc, bwd_K_max, 19);
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, bwd_K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        }
    };

    auto encode_proj_sh_bwd_adam = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_backward_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0); ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_bwd_intr->data() length:sizeof(*proj_bwd_intr) atIndex:7];
        [enc setBytes:proj_bwd_isz->data() length:sizeof(*proj_bwd_isz) atIndex:8];
        ENC_BUF(enc, radii_out, 9); ENC_BUF(enc, conics, 10);
        ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_depth, 12); ENC_BUF(enc, v_conic, 13);
        ENC_BUF(enc, v_mean3d, 14); ENC_BUF(enc, v_scale, 15); ENC_BUF(enc, v_quat, 16);
        ENC_SCALAR(enc, degree, 17); ENC_SCALAR(enc, degrees_to_use, 18);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:19];
        ENC_BUF(enc, v_colors_rast, 20); ENC_BUF(enc, v_features_dc, 21); ENC_BUF(enc, v_features_rest, 22);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        // Fused Adam step
        if (num_adam_groups > 0) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            for (int g = 0; g < num_adam_groups; ++g) {
                uint32_t n = adam_params[g].numel();
                if (n == 0) continue;
                NSUInteger atpg = MIN(ctx->fused_adam_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
                [enc setComputePipelineState:ctx->fused_adam_kernel_cpso];
                [enc setBuffer:adam_params[g].buffer() offset:0 atIndex:0];
                [enc setBuffer:(*adam_grads)[g].buffer() offset:0 atIndex:1];
                [enc setBuffer:adam_exp_avg[g].buffer() offset:0 atIndex:2];
                [enc setBuffer:adam_exp_avg_sq[g].buffer() offset:0 atIndex:3];
                ENC_SCALAR(enc, adam_step_sizes[g], 4);
                ENC_SCALAR(enc, adam_beta1, 5);
                ENC_SCALAR(enc, adam_beta2, 6);
                ENC_SCALAR(enc, adam_bc2_sqrts[g], 7);
                ENC_SCALAR(enc, adam_eps, 8);
                ENC_SCALAR(enc, n, 9);
                [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
            }
        }
    };

    // ========================== SINGLE ENCODER DISPATCH ==========================

    id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
    assert(command_buffer && "Failed to retrieve command buffer reference");

    dispatch_sync(ctx->d_queue, ^(){
        // Blit-zero all accumulation buffers (GPU-side to avoid racing with pipelined CBs)
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        [blit fillBuffer:tile_bins.buffer() range:NSMakeRange(0, tile_bins.nbytes()) value:0];
        [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
        [blit fillBuffer:g_tcache.overflow_flag.buffer() range:NSMakeRange(0, g_tcache.overflow_flag.nbytes()) value:0];
        [blit fillBuffer:g_tcache.tile_counts.buffer() range:NSMakeRange(0, g_tcache.tile_counts.nbytes()) value:0];
        [blit fillBuffer:g_tcache.tile_scatter_counters.buffer() range:NSMakeRange(0, g_tcache.tile_scatter_counters.nbytes()) value:0];
        [blit fillBuffer:v_xy.buffer() range:NSMakeRange(0, v_xy.nbytes()) value:0];
        [blit fillBuffer:v_conic.buffer() range:NSMakeRange(0, v_conic.nbytes()) value:0];
        [blit fillBuffer:v_colors_rast.buffer() range:NSMakeRange(0, v_colors_rast.nbytes()) value:0];
        [blit fillBuffer:v_opacity.buffer() range:NSMakeRange(0, v_opacity.nbytes()) value:0];
        [blit fillBuffer:v_depth.buffer() range:NSMakeRange(0, v_depth.nbytes()) value:0];
        [blit fillBuffer:v_mean3d.buffer() range:NSMakeRange(0, v_mean3d.nbytes()) value:0];
        [blit fillBuffer:v_scale.buffer() range:NSMakeRange(0, v_scale.nbytes()) value:0];
        [blit fillBuffer:v_quat.buffer() range:NSMakeRange(0, v_quat.nbytes()) value:0];
        [blit fillBuffer:v_features_dc.buffer() range:NSMakeRange(0, v_features_dc.nbytes()) value:0];
        [blit fillBuffer:v_features_rest.buffer() range:NSMakeRange(0, v_features_rest.nbytes()) value:0];
        [blit endEncoding];

        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        assert(enc && "Failed to create compute command encoder");

        // --- Forward: proj_sh → prefix_map → rast_fwd → loss_fwd ---
        encode_proj_sh(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_prefix_map(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_rast_fwd(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_loss_fwd(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // --- Backward: loss_bwd → rast_bwd → proj_sh_bwd + Adam ---
        encode_loss_bwd(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_rast_bwd(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_proj_sh_bwd_adam(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // --- Accumulate grad stats ---
        {
            NSUInteger tpg = MIN(ctx->accumulate_grad_stats_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->accumulate_grad_stats_kernel_cpso];
            ENC_SCALAR(enc, num_points, 0);
            ENC_BUF(enc, radii_out, 1);
            ENC_BUF(enc, v_xy, 2);
            ENC_BUF(enc, vis_counts, 3);
            ENC_BUF(enc, xys_grad_norm, 4);
            ENC_BUF(enc, max_2d_size, 5);
            ENC_SCALAR(enc, inv_max_dim, 6);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }

        [enc endEncoding];
    });

    float loss_val = *loss_sum.data<float>() / (float)(img_height * img_width);
    return std::make_tuple(radii_out, loss_val);
}

// ============================================================================
// GPU-native densification (v34 Phase 3)
// Entire classify → grow → cull → compact pipeline in one compute encoder.
// Returns new num_active after densification.
// ============================================================================
int msplat_densify(
    int N, int buf_capacity,
    float grad_thresh, float size_thresh, float screen_thresh, int check_screen,
    float cull_alpha_thresh, float cull_scale_thresh, float cull_screen_size, int check_huge,
    MTensor &xys_grad_norm, MTensor &vis_counts, MTensor &max_2d_size,
    float half_max_dim,
    MTensor &means_buf, MTensor &scales_buf, MTensor &quats_buf,
    MTensor &featuresDc_buf, MTensor &featuresRest_buf, MTensor &opacities_buf,
    int fr_stride,
    MTensor adam_exp_avg_buf[], MTensor adam_exp_avg_sq_buf[],
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &keep_flag, MTensor &keep_prefix,
    MTensor &block_totals, MTensor &compact_scratch,
    MTensor &random_samples
) {
    MetalContext* ctx = get_global_context();

    // Worst case: each of N gaussians splits (2 children) + dups (1 copy) = 3N
    int worst_case = 3 * N;
    assert(worst_case <= buf_capacity && "gpu_densify: 3*N exceeds buf_capacity");

    float log_size_fac = std::log(1.6f);

    // Strides for each of the 18 buffers (6 params + 12 optimizer states)
    // Order: means(3), scales(3), quats(4), featuresDc(3), featuresRest(fr_stride), opacities(1)
    int strides[6] = {3, 3, 4, 3, fr_stride, 1};
    int max_stride = fr_stride;  // featuresRest has the largest stride

    // Collect all 18 buffers in order for compact loops (std::array for block capture)
    std::array<MTensor*, 18> all_bufs = {{
        &means_buf, &scales_buf, &quats_buf, &featuresDc_buf, &featuresRest_buf, &opacities_buf,
        &adam_exp_avg_buf[0], &adam_exp_avg_buf[1], &adam_exp_avg_buf[2],
        &adam_exp_avg_buf[3], &adam_exp_avg_buf[4], &adam_exp_avg_buf[5],
        &adam_exp_avg_sq_buf[0], &adam_exp_avg_sq_buf[1], &adam_exp_avg_sq_buf[2],
        &adam_exp_avg_sq_buf[3], &adam_exp_avg_sq_buf[4], &adam_exp_avg_sq_buf[5]
    }};
    std::array<int, 18> all_strides = {{
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1
    }};

    uint32_t N_u32 = (uint32_t)N;
    uint32_t K = (uint32_t)((N + 1023) / 1024);  // threadgroups for prefix sum on N elements
    int check_screen_int = check_screen;
    int check_huge_int = check_huge;

    id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
    assert(command_buffer && "Failed to retrieve command buffer reference");

    dispatch_sync(ctx->d_queue, ^(){
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        assert(enc && "Failed to create compute command encoder");

        // ---- Stage 1: Classify (split/dup) ----
        {
            NSUInteger tpg = MIN(ctx->densify_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, xys_grad_norm, 1);
            ENC_BUF(enc, vis_counts, 2);
            ENC_BUF(enc, scales_buf, 3);
            ENC_BUF(enc, max_2d_size, 4);
            ENC_SCALAR(enc, half_max_dim, 5);
            ENC_SCALAR(enc, grad_thresh, 6);
            ENC_SCALAR(enc, size_thresh, 7);
            ENC_SCALAR(enc, screen_thresh, 8);
            ENC_SCALAR(enc, check_screen_int, 9);
            ENC_BUF(enc, split_flag, 10);
            ENC_BUF(enc, dup_flag, 11);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 2: Prefix sum on split_flag → split_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 3: Prefix sum on dup_flag → dup_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 4: Append split children ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_split_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_split_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2);
            ENC_BUF(enc, random_samples, 3);
            ENC_SCALAR(enc, log_size_fac, 4);
            ENC_BUF(enc, means_buf, 5);
            ENC_BUF(enc, scales_buf, 6);
            ENC_BUF(enc, quats_buf, 7);
            ENC_BUF(enc, featuresDc_buf, 8);
            ENC_BUF(enc, featuresRest_buf, 9);
            ENC_BUF(enc, opacities_buf, 10);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 11);
            ENC_BUF(enc, adam_exp_avg_buf[0], 12);
            ENC_BUF(enc, adam_exp_avg_buf[1], 13);
            ENC_BUF(enc, adam_exp_avg_buf[2], 14);
            ENC_BUF(enc, adam_exp_avg_buf[3], 15);
            ENC_BUF(enc, adam_exp_avg_buf[4], 16);
            ENC_BUF(enc, adam_exp_avg_buf[5], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 22);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 23);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 5: Append duplicates ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_dup_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_dup_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_prefix, 3);
            ENC_BUF(enc, means_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, quats_buf, 6);
            ENC_BUF(enc, featuresDc_buf, 7);
            ENC_BUF(enc, featuresRest_buf, 8);
            ENC_BUF(enc, opacities_buf, 9);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 10);
            ENC_BUF(enc, adam_exp_avg_buf[0], 11);
            ENC_BUF(enc, adam_exp_avg_buf[1], 12);
            ENC_BUF(enc, adam_exp_avg_buf[2], 13);
            ENC_BUF(enc, adam_exp_avg_buf[3], 14);
            ENC_BUF(enc, adam_exp_avg_buf[4], 15);
            ENC_BUF(enc, adam_exp_avg_buf[5], 16);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 22);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 6: Cull classify (on post-growth population) ----
        // Dispatch worst_case threads; kernel reads N_new from prefix sums
        {
            uint32_t wc = (uint32_t)worst_case;
            NSUInteger tpg = MIN(ctx->densify_cull_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)worst_case);
            [enc setComputePipelineState:ctx->densify_cull_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_prefix, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_flag, 3);
            ENC_BUF(enc, opacities_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, max_2d_size, 6);
            ENC_SCALAR(enc, cull_alpha_thresh, 7);
            ENC_SCALAR(enc, cull_scale_thresh, 8);
            ENC_SCALAR(enc, cull_screen_size, 9);
            ENC_SCALAR(enc, check_huge_int, 10);
            ENC_SCALAR(enc, check_screen_int, 11);
            ENC_BUF(enc, keep_flag, 12);
            [enc dispatchThreads:MTLSizeMake(worst_case, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 7: Prefix sum on keep_flag → keep_prefix ----
        // Over worst_case elements (includes padding zeros for unused slots)
        {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t K2 = (uint32_t)((worst_case + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t K2 = (uint32_t)((worst_case + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, keep_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 8: Compact scatter (18 buffers → scratch) ----
        // For each buffer: scatter kept elements into compact_scratch
        // Then copy back. We reuse compact_scratch at different offsets per stride.
        for (int b = 0; b < 18; b++) {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t stride_u32 = (uint32_t)all_strides[b];
            uint32_t total_threads = wc * stride_u32;
            NSUInteger tpg = MIN(ctx->compact_scatter_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)total_threads);
            [enc setComputePipelineState:ctx->compact_scatter_kernel_cpso];
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:0];
            ENC_BUF(enc, compact_scratch, 1);
            ENC_BUF(enc, keep_prefix, 2);
            ENC_BUF(enc, keep_flag, 3);
            ENC_SCALAR(enc, wc, 4);
            ENC_SCALAR(enc, stride_u32, 5);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Copy back from scratch to buffer
            uint32_t last_idx = wc - 1;
            [enc setComputePipelineState:ctx->compact_copy_back_kernel_cpso];
            ENC_BUF(enc, compact_scratch, 0);
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:1];
            ENC_BUF(enc, keep_prefix, 2);
            ENC_SCALAR(enc, last_idx, 3);
            ENC_SCALAR(enc, stride_u32, 4);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc endEncoding];
    });

    // Single GPU→CPU sync: read new_count from keep_prefix[worst_case - 1]
    ctx->syncCB();
    int new_count = keep_prefix.data<int32_t>()[worst_case - 1];
    return new_count;
}
