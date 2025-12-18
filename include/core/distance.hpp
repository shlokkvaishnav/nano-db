#pragma once

#include <cstddef>

namespace nanodb {

    // Calculates Squared Euclidean Distance (L2^2) between two vectors.
    // Optimized with AVX2 SIMD instructions in the implementation (.cpp).
    // Note: We skip sqrt() for performance since it preserves ranking order.
    float get_distance(const float* a, const float* b, size_t dim);

} // namespace nanodb