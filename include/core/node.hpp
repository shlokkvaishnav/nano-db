#pragma once

#include "../common/types.hpp"
#include "../common/config.hpp"
#include <cstring> // For memset

namespace nanodb {

    // Maximum height of the HNSW graph.
    // 4 layers is usually enough for ~1 million vectors.
    constexpr int MAX_LAYERS = 4;

    // We use alignas(32) to ensure the vector array starts on a 32-byte boundary.
    // This allows AVX2 instructions to load data faster (aligned load vs unaligned load).
    struct alignas(32) Node {
        
        // ---------------------------------------------------------
        // Header Data
        // ---------------------------------------------------------
        id_t id;          // External ID (e.g., User ID, Image ID)
        int max_layer;    // The highest layer this node exists in (0 to MAX_LAYERS-1)

        // ---------------------------------------------------------
        // Vector Data (Inline)
        // ---------------------------------------------------------
        // The actual embedding.
        val_t vector[config::VECTOR_DIM];

        // ---------------------------------------------------------
        // Graph Connectivity (The HNSW Links)
        // ---------------------------------------------------------
        // A 2D array storing neighbor IDs for each layer.
        // dim 1: Layer Index (0 is bottom, MAX_LAYERS-1 is top)
        // dim 2: Neighbor Slot (up to M_MAX0 for layer 0, M for others)
        // We use M_MAX0 for all layers to keep the struct simple (slightly wasteful but fast).
        id_t neighbors[MAX_LAYERS][config::M_MAX0];

        // Tracks how many neighbors are actually stored in each layer
        int neighbor_counts[MAX_LAYERS];

        // ---------------------------------------------------------
        // Constructor
        // ---------------------------------------------------------
        Node(id_t external_id, int level, const std::vector<float>& vec_data) 
            : id(external_id), max_layer(level) {
            
            // 1. Copy Vector Data
            // We ensure we don't overflow the fixed buffer
            size_t copy_size = vec_data.size() > config::VECTOR_DIM ? config::VECTOR_DIM : vec_data.size();
            for(size_t i = 0; i < copy_size; ++i) {
                vector[i] = vec_data[i];
            }

            // 2. Initialize Neighbors
            // Set counts to 0
            std::memset(neighbor_counts, 0, sizeof(neighbor_counts));
            
            // Set all neighbor slots to -1 (indicating empty)
            std::memset(neighbors, -1, sizeof(neighbors));
        }

        // Default constructor for mmap casting
        Node() = default;
    };

} // namespace nanodb