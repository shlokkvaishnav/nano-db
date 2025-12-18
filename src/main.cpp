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
using namespace std;

// Generate random float vector
vector<float> generate_random_vector(mt19937& rng) {
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    vector<float> vec(config::VECTOR_DIM);
    for (size_t i = 0; i < config::VECTOR_DIM; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

int main() {
    cout << "============================================" << endl;
    cout << "   NanoDB: High-Performance Vector Engine   " << endl;
    cout << "============================================" << endl;

    // Initialize MMap Storage
    string db_path = "data/index.ndb";
    MMapHandler storage;
    try {
        storage.open_file(db_path, 1024 * 1024); // Start with 1MB
        cout << "[Storage] MMap initialized at: " << db_path << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    // Initialize Index
    cout << "[Index] Initializing HNSW Graph..." << endl;
    HNSW index(storage);

    // Generate Data
    const int NUM_VECTORS = 10000;
    cout << "[Data] Generating " << NUM_VECTORS << " vectors (" << config::VECTOR_DIM << "d)..." << endl;
    
    vector<vector<float>> dataset(NUM_VECTORS);
    mt19937 rng(42);
    for (int i = 0; i < NUM_VECTORS; ++i) {
        dataset[i] = generate_random_vector(rng);
    }

    // Benchmark Insertion
    cout << "[Benchmark] Inserting..." << endl;
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_VECTORS; ++i) {
        index.insert(dataset[i], i);
        if ((i + 1) % 1000 == 0) cout << "  - Inserted " << (i + 1) << "\r" << flush;
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    
    cout << "\n[Benchmark] Done. Time: " << diff.count() << "s | TPS: " << (NUM_VECTORS / diff.count()) << endl;

    // Benchmark Search
    cout << "\n[Benchmark] Searching..." << endl;
    int target_id = 500;
    vector<float> query = dataset[target_id];
    query[0] += 0.01f; // Add noise

    auto s_start = chrono::high_resolution_clock::now();
    auto results = index.search(query, 5);
    auto s_end = chrono::high_resolution_clock::now();
    
    cout << "  - Time: " << (chrono::duration<double>(s_end - s_start).count() * 1000) << " ms" << endl;
    
    // Results
    cout << left << setw(10) << "Rank" << setw(10) << "ID" << "Distance" << endl;
    cout << "--------------------------------" << endl;
    for (size_t i = 0; i < results.size(); ++i) {
        cout << left << setw(10) << (i + 1) << setw(10) << results[i].id << results[i].distance << endl;
    }

    storage.close_file();
    cout << "\n[System] Database closed." << endl;

    return 0;
}