#include "../../include/core/distance.hpp"
#include <immintrin.h> // AVX2 intrinsics
#include <cmath>

namespace nanodb {

    float get_distance(const float* a, const float* b, size_t dim) {
        // Initialize 256-bit accumulator (stores 8 floats) to zeros
        __m256 sum = _mm256_setzero_ps();

        size_t i = 0;
        // Process 8 floats per cycle using AVX2
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);      // Load 8 floats from A
            __m256 vb = _mm256_loadu_ps(b + i);      // Load 8 floats from B
            __m256 diff = _mm256_sub_ps(va, vb);     // Subtract: a - b
            __m256 sq = _mm256_mul_ps(diff, diff);   // Square: diff * diff
            sum = _mm256_add_ps(sum, sq);            // Accumulate: sum += sq
        }

        // Horizontal sum: Reduce the 8 SIMD lanes into a single float result
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        
        float total_dist = 0.0f;
        for (int k = 0; k < 8; ++k) total_dist += temp[k];

        // Tail case: Handle remaining elements if dimension is not a multiple of 8
        for (; i < dim; ++i) {
            float d = a[i] - b[i];
            total_dist += d * d;
        }

        return total_dist;
    }

} // namespace nanodb