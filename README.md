# ⚡ NanoDB: High-Performance Vector Search Engine

> A high-throughput, persistent Vector Database built from scratch in C++17 with Python bindings.

![C++](https://img.shields.io/badge/Language-C%2B%2B17-blue)
![Python](https://img.shields.io/badge/Bindings-Python%203.11%2B-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/Status-Functional-brightgreen)

**NanoDB** is a lightweight vector search engine designed to handle high-dimensional embedding vectors (e.g., 128d, 768d). Unlike wrapper libraries, NanoDB implements a custom **HNSW (Hierarchical Navigable Small World)** graph from scratch with disk-backed persistence.

It bridges the gap between raw algorithms (like FAISS) and full-scale databases (like Milvus) by offering a persistent, mmap-based storage engine without external dependencies.

---

## 🚀 Key Engineering Features

### 1. Hybrid Storage Engine
* **Vector Storage:** Uses **Memory Mapped Files (mmap)** to handle datasets larger than physical RAM. The OS page cache manages memory, allowing instant load times (Zero-Copy).
* **Metadata Storage:** Implements an **Append-Only Log** with an in-memory offset index to store variable-length strings (filenames, JSON labels) alongside vectors.

### 2. High-Performance Indexing
* **HNSW Graph:** Logarithmic time complexity $O(\log N)$ for searching millions of vectors.
* **SIMD Acceleration:** Euclidean distance calculations are hand-optimized using **AVX2 Intrinsics**, achieving 4x-8x speedups over standard loops.

### 3. Concurrency & Locking
* **Fine-Grained Locking:** Replaces global mutexes with a **Stripe of Atomic SpinLocks**, minimizing contention.
* **Parallel Insertion:** Thread-safe architecture allows **6,500+ TPS** (Transactions Per Second) with 8+ concurrent threads.

---

## 📊 Performance Benchmarks

*Hardware: Standard Consumer Laptop (8-Core CPU)*
*Dimensions: 128d (Float32)*

| Metric | Single-Threaded | **Multi-Threaded (8 Threads)** |
| :--- | :--- | :--- |
| **Throughput (Insert)** | ~2,200 TPS | **~6,500 TPS** |
| **Speedup** | 1.0x | **2.88x** |
| **Search Latency** | ~0.15 ms | ~0.15 ms |
| **Distance Metric** | Euclidean (L2) | **AVX2 Optimized** |

---

## 🛠️ Installation & Build

NanoDB uses **CMake** for cross-platform build management.

### Prerequisites
* C++17 Compiler (MSVC, GCC, or Clang)
* CMake 3.10+
* Python 3.x (for bindings)

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

# 4. Artifacts
# The build produces 'nanodb.cp3xx-win_amd64.pyd' in the Release folder.

```

### Linux/Mac Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

```

---

## 💻 Usage (Python)

NanoDB provides native Python bindings (`pybind11`) for easy integration with AI pipelines.

```python
import nanodb
import random

# 1. Initialize DB (Persists to disk automatically)
# - 'index.ndb': Stores vectors & graph
# - 'meta.bin':  Stores filenames/labels
index = nanodb.HNSW(
    storage=nanodb.MMapHandler(),
    meta_path="data/meta.bin"
)
index.storage.open_file("data/index.ndb", 50 * 1024 * 1024) # 50MB Buffer

# 2. Insert Data (Vector + ID + Metadata)
# Simulate 128d embedding
vector = [random.random() for _ in range(128)]
index.insert(vector, id=1, metadata="cat_photo.jpg")

# 3. Search
# Returns top-k nearest neighbors
results = index.search(query=vector, k=1)

for res in results:
    print(f"Found ID: {res.id}")
    print(f"Distance: {res.distance:.4f}")
    print(f"Metadata: {res.metadata}")

```

---

## 🧠 System Architecture

### 1. The "MMap" Storage Engine (Zero-Copy)

Most databases read files using `fread`, which copies data from Disk → Kernel Buffer → User RAM. This is slow and consumes physical memory immediately.

**NanoDB uses Memory Mapped Files (mmap):**

* **Lazy Loading:** The OS maps the file into the process's virtual address space but only loads physical **Pages (4KB)** when they are actually accessed.
* **Huge Datasets:** This allows NanoDB to search a **100GB dataset on a machine with only 8GB of RAM**, relying on the OS page cache for memory management.

### 2. Offset-Based Addressing (The "Pointer" Solution)

A major challenge in C++ database design is that standard pointers (`Node*`) store **absolute memory addresses** (e.g., `0x7fff5b...`). If you save these to disk and reload them, the OS will likely load the file at a different address, making the pointers invalid (pointing to garbage).

**The Solution:**
Instead of absolute pointers, NanoDB uses **Relative Offsets** (e.g., "Node B is 1024 bytes from the start of the file").

* **Portability:** The database file is "relocatable." It works instantly regardless of where it is loaded in memory.
* **Zero Serialization:** We don't need to parse or convert data when opening the DB. We just map the file and start reading.

### 3. The Index (HNSW)

The graph is constructed with layers. Search starts at the sparse top layer (Layer L) and zooms in to the dense bottom layer (Layer 0), using the **Offset Manager** to traverse links between nodes.

---

## 📜 License

MIT License. Free to use and modify.

```

```