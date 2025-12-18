#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

#include "../include/common/config.hpp"
#include "../include/common/types.hpp"
#include "../include/storage/mmap_handler.hpp"
#include "../include/core/hnsw.hpp"

using namespace nanodb;

// ---------------------------------------------------------
// Helper: Generate Random Vector
// ---------------------------------------------------------
std::vector<float> generate_random_vector(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(config::VECTOR_DIM);
    for (size_t i = 0; i < config::VECTOR_DIM; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   NanoDB: High-Performance Vector Engine   " << std::endl;
    std::cout << "============================================" << std::endl;

    // 1. Initialize Storage Engine
    // We start with a small file size; the engine will expand it automatically.
    std::string db_path = "data/index.ndb";
    size_t initial_size = 1024 * 1024; // 1MB
    
    std::cout << "[Storage] Initializing MMap at: " << db_path << std::endl;
    MMapHandler storage;
    try {
        storage.open_file(db_path, initial_size);
    } catch (const std::exception& e) {
        std::cerr << "Error opening file: " << e.what() << std::endl;
        return 1;
    }

    // 2. Initialize Index Manager
    std::cout << "[Index] Initializing HNSW Graph..." << std::endl;
    HNSW index(storage);

    // 3. Generate Data
    const int NUM_VECTORS = 10000; // Let's insert 10k vectors
    std::cout << "[Data] Generating " << NUM_VECTORS << " random vectors (" << config::VECTOR_DIM << "d)..." << std::endl;

    std::vector<std::vector<float>> dataset(NUM_VECTORS);
    std::mt19937 rng(42); // Fixed seed for reproducibility

    for (int i = 0; i < NUM_VECTORS; ++i) {
        dataset[i] = generate_random_vector(rng);
    }

    // 4. Benchmark Insertion
    std::cout << "[Benchmark] Starting Insertion..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_VECTORS; ++i) {
        index.insert(dataset[i], i);
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "  - Inserted " << (i + 1) << " vectors..." << "\r" << std::flush;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "\n[Benchmark] Insertion Complete." << std::endl;
    std::cout << "  - Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "  - TPS:  " << (NUM_VECTORS / diff.count()) << " vectors/sec" << std::endl;

    // 5. Benchmark Search
    // We take a vector from the dataset, modify it slightly, and search for it.
    // The original vector (ID X) should be the top result (Distance ~0).
    std::cout << "\n[Benchmark] Running Search Query..." << std::endl;
    
    int target_id = 500; // Arbitrary ID to search for
    std::vector<float> query = dataset[target_id];
    
    // Add slight noise to make it interesting
    query[0] += 0.01f; 

    auto search_start = std::chrono::high_resolution_clock::now();
    std::vector<Result> results = index.search(query, 5); // Find top 5
    auto search_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> search_diff = search_end - search_start;

    // 6. Report Results
    std::cout << "  - Search Time: " << (search_diff.count() * 1000) << " ms" << std::endl;
    std::cout << "  - Results for modified Vector #" << target_id << ":" << std::endl;
    
    std::cout << std::left << std::setw(10) << "Rank" 
              << std::setw(10) << "ID" 
              << "Distance (L2)" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << std::left << std::setw(10) << (i + 1) 
                  << std::setw(10) << results[i].id 
                  << results[i].distance << std::endl;
    }

    // Cleanup
    storage.close_file();
    std::cout << "\n[System] Database closed safely." << std::endl;

    return 0;
}