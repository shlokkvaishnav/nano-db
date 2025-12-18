#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <utility>

namespace nanodb {

    using id_t = uint32_t;
    using offset_t = uint64_t;
    using val_t = float;
    using Vector = std::vector<val_t>;

    struct Result {
        id_t id;
        float distance;
        std::string metadata; // <--- NEW FIELD

        bool operator>(const Result& other) const { return distance > other.distance; }
        bool operator<(const Result& other) const { return distance < other.distance; }
    };

} // namespace nanodb