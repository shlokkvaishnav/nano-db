#pragma once

#include <vector>
#include <cstdint>
#include <utility>

namespace nanodb {

    // ---------------------------------------------------------
    // Primitive Aliases
    // ---------------------------------------------------------

    // Unique identifier for a vector (Supports up to 4 Billion vectors)
    // We use 32-bit to save RAM/Disk space compared to 64-bit.
    using id_t = uint32_t;

    // File offset for Memory Mapping (Supports files > 4GB)
    // Must be 64-bit because database files can grow very large.
    using offset_t = uint64_t;

    // The data type for a single dimension (Standard embedding float)
    using val_t = float;

    // High-level container for inserting data (e.g., coming from Python/API)
    using Vector = std::vector<val_t>;


    // ---------------------------------------------------------
    // Search Structures
    // ---------------------------------------------------------

    // Represents a search result: A pair of (Vector ID, Distance Score)
    struct Result {
        id_t id;
        float distance;

        // Operator overload for Priority Queue comparisons.
        // Standard priority_queue is a Max-Heap.
        // For Nearest Neighbor, we often want the smallest distance.
        bool operator>(const Result& other) const {
            return distance > other.distance;
        }

        bool operator<(const Result& other) const {
            return distance < other.distance;
        }
    };

} // namespace nanodb