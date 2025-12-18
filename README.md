# ⚡ NanoDB: High-Performance Vector Search Engine

> A standalone, header-only C++17 vector database for Approximate Nearest Neighbor (ANN) search.

![C++](https://img.shields.io/badge/Language-C%2B%2B17-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

NanoDB is a lightweight vector search engine designed to handle high-dimensional embedding vectors (e.g., 128d, 768d). Unlike wrapper libraries, NanoDB implements a custom **HNSW (Hierarchical Navigable Small World)** graph from scratch with disk-backed persistence.

It bridges the gap between raw algorithms (like FAISS) and full-scale databases (like Milvus) by offering a persistent, mmap-based storage engine without external dependencies.

---

## 🚀 Key Features

* **HNSW Graph Indexing:** Logarithmic time complexity $O(\log N)$ for searching millions of vectors.
* **Disk-Backed Persistence:** Uses **Memory Mapped Files (mmap)** to handle datasets larger than physical RAM by leveraging the OS page cache.
* **Offset-Based Addressing:** Solves the "pointer invalidation" problem by storing graph links as file offsets, making the database portable and instantly loadable (Zero-Copy deserialization).
* **SIMD Acceleration:** Euclidean distance calculations are optimized using **AVX2 Intrinsics**, achieving 4x-8x speedups over standard loops.
* **Parallelism:** Thread-safe operations using **OpenMP** (planned for bulk insertion).

---

## 📊 Performance Benchmarks

*Hardware: Standard Consumer Laptop (Windows x64)*
*Dimensions: 128d (Float32)*

| Metric | Result |
| :--- | :--- |
| **Search Latency** | **~0.14 ms** (138 microseconds) |
| **Throughput (Single Thread)** | ~5,000 vectors/sec |
| **Distance Metric** | Euclidean (L2) with AVX2 |

---

## 🛠️ Installation & Build

NanoDB uses **CMake** for cross-platform build management.

### Prerequisites
* C++17 Compiler (MSVC, GCC, or Clang)
* CMake 3.10+

### Windows Build
```powershell
# 1. Clone the repository
git clone [https://github.com/shlokkvaishnav/nano-db.git](https://github.com/shlokkvaishnav/nano-db.git)
cd nano-db

# 2. Create build directory
mkdir build
cd build

# 3. Configure and Build (Release mode recommended for AVX2 speed)
cmake ..
cmake --build . --config Release

# 4. Run
.\Release\nano_db.exe

```

### Linux/Mac Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./nano_db

```

---

## 💻 Usage

NanoDB is designed to be simple. Here is a minimal example of how the core engine works:

```cpp
#include "nano_db.hpp"

int main() {
    // 1. Initialize DB (Creates/Loads 'data.ndb' file)
    NanoDB db("data/index.ndb");

    // 2. Insert Vectors (ID, Vector)
    std::vector<float> vec1 = {0.1, 0.5, ...}; // 128d
    db.insert(101, vec1);

    // 3. Search for nearest neighbors
    auto results = db.search(query_vector, 5); // Find top 5

    // 4. Print results
    for (const auto& res : results) {
        std::cout << "ID: " << res.id << " Distance: " << res.dist << "\n";
    }

    return 0;
}

```

---

## 🧠 System Architecture

### Storage Engine (The "MMap" Trick)

Instead of loading the entire graph into the Heap (RAM), NanoDB maps the physical file directly into the virtual address space.

* **Write:** Changes are written to virtual memory; the OS flushes dirty pages to disk asynchronously.
* **Read:** The OS lazy-loads pages only when traversed.

### The Index (HNSW)

The graph is constructed with layers. Search starts at the sparse top layer (Layer L) and zooms in to the dense bottom layer (Layer 0).

---

## 🔮 Roadmap

* [x] Core HNSW Implementation
* [x] Memory Mapped Storage
* [x] AVX2 SIMD Optimization
* [ ] OpenMP Parallel Bulk Insert
* [ ] Python Bindings (Pybind11)
* [ ] REST API Wrapper

---

## 📜 License

MIT License. Free to use and modify.

```

***