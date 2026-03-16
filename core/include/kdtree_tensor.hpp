#ifndef KDTREE_TENSOR_H
#define KDTREE_TENSOR_H

#include <nanoflann.hpp>
#include <vector>
#include <cmath>
#include <cassert>
#include <dispatch/dispatch.h>

// nanoflann adapter for a flat float array of 3D points.
// Used to compute per-point initial scales via KNN.
struct PointsTensor {
    const float *data;
    int64_t count;

    PointsTensor(const float *data, int64_t count) : data(data), count(count) {}

    // nanoflann interface
    size_t kdtree_get_point_count() const { return (size_t)count; }
    float kdtree_get_pt(size_t idx, size_t dim) const { return data[idx * 3 + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }

    using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointsTensor>,
        PointsTensor, 3, size_t>;

    // Compute mean distance to k nearest neighbors for each point.
    static constexpr int kMaxK = 16;
    std::vector<float> scales(int k = 4) const {
        assert(k <= kMaxK && "k exceeds kMaxK for stack-allocated KNN buffers");
        KdTree index(3, *this, {10});
        const KdTree *idx = &index;

        std::vector<float> result(count);
        float *rp = result.data();
        const float *dp = data;
        dispatch_apply((size_t)count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t i) {
                size_t indices[kMaxK];
                float dists[kMaxK];
                idx->knnSearch(&dp[i * 3], k, indices, dists);
                float sum = 0;
                for (int j = 1; j < k; j++) sum += std::sqrt(dists[j]);
                rp[i] = sum / (k - 1);
            });
        return result;
    }
};

#endif
