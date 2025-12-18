#pragma once

#include <cstddef>

namespace nanodb {

    namespace config {

        // ---------------------------------------------------------
        // Data Dimensions
        // ---------------------------------------------------------
        
        // The number of dimensions in your vectors (e.g., 128 for FaceNet/BERT-tiny).
        // IMPORTANT: For AVX2 optimization, this should ideally be a multiple of 8.
        constexpr size_t VECTOR_DIM = 128; 

        // Maximum number of vectors we expect to hold (can be dynamic, but useful for pre-allocation)
        constexpr size_t MAX_ELEMENTS = 100000; 


        // ---------------------------------------------------------
        // HNSW Algorithm Hyperparameters
        // ---------------------------------------------------------

        // 'M': The number of bi-directional links created for every new element during construction.
        // Higher M = higher recall but higher memory usage.
        // Reasonable range: 12 to 48.
        constexpr int M = 16; 

        // 'M_MAX0': Max connections in the bottom layer (Layer 0).
        // Usually 2 * M. The bottom layer needs to be denser for precision.
        constexpr int M_MAX0 = M * 2; 

        // 'efConstruction': The size of the dynamic candidate list during insertion.
        // Higher = better graph quality, but slower indexing.
        constexpr int EF_CONSTRUCTION = 200; 

        // ---------------------------------------------------------
        // System Settings
        // ---------------------------------------------------------
        
        // Default path for the persistent storage file
        constexpr char DB_FILE_PATH[] = "data/index.ndb";
        
        // Memory page size (usually 4KB), used for aligning memory mappings
        constexpr size_t PAGE_SIZE = 4096;

    } // namespace config

} // namespace nanodb