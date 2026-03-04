#ifndef RANDOM_ITER_H
#define RANDOM_ITER_H

#include <vector>
#include <algorithm>
#include <random>

// Infinite iterator that shuffles and cycles through elements.
template <typename T>
class InfiniteRandomIterator {
public:
    InfiniteRandomIterator(const std::vector<T> &items, unsigned seed = 42)
        : items_(items), rng_(seed) { shuffle(); }

    T next() {
        T val = items_[pos_++];
        if (pos_ >= items_.size()) shuffle();
        return val;
    }

private:
    void shuffle() {
        std::shuffle(items_.begin(), items_.end(), rng_);
        pos_ = 0;
    }

    std::vector<T> items_;
    size_t pos_ = 0;
    std::mt19937 rng_;
};

#endif
