#ifndef KDTREE_TENSOR_H
#define KDTREE_TENSOR_H

#include <nanoflann.hpp>
#include <vector>
#include <cmath>

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
    std::vector<float> scales(int k = 4) const {
        KdTree index(3, *this, {10});

        std::vector<float> result(count);
        std::vector<size_t> indices(k);
        std::vector<float> dists(k);

        for (int64_t i = 0; i < count; i++) {
            index.knnSearch(&data[i * 3], k, indices.data(), dists.data());
            float sum = 0;
            for (int j = 1; j < k; j++) sum += std::sqrt(dists[j]);
            result[i] = sum / (k - 1);
        }
        return result;
    }
};

#endif
