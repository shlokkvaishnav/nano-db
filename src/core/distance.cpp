#include "../../include/core/distance.hpp"

#include <immintrin.h> // The header for AVX intrinsics
#include <cmath>

namespace nanodb {

    float get_distance(const float* a, const float* b, size_t dim) {
        
        // ---------------------------------------------------------
        // AVX2 Implementation (Process 8 floats per cycle)
        // ---------------------------------------------------------
        
        // Accumulator vector [sum0, sum1, ..., sum7] initialized to zeros
        __m256 sum = _mm256_setzero_ps();

        size_t i = 0;
        // Loop stepping by 8
        for (; i + 8 <= dim; i += 8) {
            // 1. Load 8 floats from memory into 256-bit registers
            // We use loadu (unaligned) to be safe, though our Nodes are aligned.
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);

            // 2. Subtraction: diff = a - b
            __m256 diff = _mm256_sub_ps(va, vb);

            // 3. Square: sq = diff * diff
            __m256 sq = _mm256_mul_ps(diff, diff);

            // 4. Accumulate: sum += sq
            sum = _mm256_add_ps(sum, sq);
        }

        // ---------------------------------------------------------
        // Horizontal Sum (Reduce 8 lanes to 1 float)
        // ---------------------------------------------------------
        // There isn't a single instruction to sum all elements in a YMM register.
        // We store the register back to a temporary array and sum it up.
        float temp[8];
        _mm256_storeu_ps(temp, sum);

        float total_dist = 0.0f;
        for (int k = 0; k < 8; ++k) {
            total_dist += temp[k];
        }

        // ---------------------------------------------------------
        // Tail Case (Handle remaining dimensions < 8)
        // ---------------------------------------------------------
        // Even though we fixed DIM=128, this makes the code robust if we change config later.
        for (; i < dim; ++i) {
            float d = a[i] - b[i];
            total_dist += d * d;
        }

        return total_dist;
    }

} // namespace nanodb