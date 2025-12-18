#pragma once

#include <cstddef> // for size_t
#include <vector>

namespace nanodb {

    // We use Squared Euclidean Distance (L2-Squared).
    // Why? Because sqrt() is expensive and strictly increasing (monotonic).
    // If A < B, then sqrt(A) < sqrt(B). So for sorting/ranking, we don't need the sqrt.
    float get_distance(const float* a, const float* b, size_t dim);

} // namespace nanodb