#pragma once

#include <vector>
#include <cstdint>
#include <utility>

namespace nanodb {

    // --- Primitive Aliases ---
    using id_t = uint32_t;       // Unique vector ID (32-bit supports ~4B items)
    using offset_t = uint64_t;   // File offset for MMap (64-bit for large DB files)
    using val_t = float;         // Single dimension data type
    using Vector = std::vector<val_t>; // High-level container for input data

    // --- Search Structures ---
    struct Result {
        id_t id;
        float distance;

        // Min-Heap support for Priority Queue (orders by smallest distance)
        bool operator>(const Result& other) const { return distance > other.distance; }
        bool operator<(const Result& other) const { return distance < other.distance; }
    };

} // namespace nanodb