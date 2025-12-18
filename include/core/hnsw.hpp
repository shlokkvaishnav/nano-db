#pragma once

#include "node.hpp"
#include "distance.hpp"
#include "../common/config.hpp"
#include "../storage/mmap_handler.hpp"
#include "../common/spinlock.hpp"
#include "../storage/metadata_handler.hpp" // <--- Handler
#include <queue>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mutex>
#include <memory>

namespace nanodb {

    class HNSW {
    public:
        // --- Constructor ---
        // NEW: Accepts meta_path
        HNSW(MMapHandler& storage, const std::string& meta_path = "data/metadata.bin") 
            : storage_(storage) {
            
            // Initialize Metadata Storage
            metadata_storage_.open_file(meta_path);

            std::random_device rd;
            rng_.seed(rd());
            
            size_t current_count = 0;

            if (storage_.get_size() == 0) {
                entry_point_id_ = -1;
                current_max_layer_ = -1;
                element_count_ = 0;
                current_count = 0;
            } else {
                entry_point_id_ = 0; 
                current_max_layer_ = 0; 
                element_count_ = storage_.get_size() / sizeof(Node);
                current_count = element_count_;
            }

            node_locks_.reserve(current_count + 10000);
            for (size_t i = 0; i < current_count + 10000; ++i) {
                node_locks_.push_back(std::make_unique<SpinLock>());
            }
        }

        // --- Public API ---

        // NEW: Accepts metadata string
        void insert(const std::vector<float>& vec_data, id_t id, const std::string& metadata = "") {
            // 1. Assign random level
            int level = get_random_level();
            Node new_node(id, level, vec_data);

            // 2. Expand storage
            size_t offset = (size_t)id * sizeof(Node);
            if (offset + sizeof(Node) > storage_.get_size()) {
                std::lock_guard<std::mutex> lock(global_resize_lock_); 
                if (offset + sizeof(Node) > storage_.get_size()) {
                    storage_.resize(storage_.get_size() + 10 * 1024 * 1024);
                    if (id >= node_locks_.size()) {
                        size_t target_size = id + 10000;
                        node_locks_.reserve(target_size);
                        for (size_t i = node_locks_.size(); i < target_size; ++i) {
                            node_locks_.push_back(std::make_unique<SpinLock>());
                        }
                    }
                }
            }

            // 3. Write node
            Node* node_ptr = get_node(id);
            *node_ptr = new_node;

            // 4. Handle first element
            if (entry_point_id_ == -1) {
                std::lock_guard<std::mutex> lock(init_lock_);
                if (entry_point_id_ == -1) {
                    entry_point_id_ = id;
                    current_max_layer_ = level;
                    #pragma omp atomic
                    element_count_++;
                    
                    // Save metadata for Genesis node
                    if (!metadata.empty()) metadata_storage_.save_metadata(id, metadata);
                    return; 
                }
            }

            // 5. Greedy Search
            id_t curr_obj = entry_point_id_;
            float dist = get_distance(node_ptr->vector, get_node(curr_obj)->vector, config::VECTOR_DIM);

            for (int l = current_max_layer_; l > level; l--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    Node* curr_node = get_node(curr_obj);
                    for (int i = 0; i < curr_node->neighbor_counts[l]; i++) {
                        id_t n_id = curr_node->neighbors[l][i];
                        float d = get_distance(node_ptr->vector, get_node(n_id)->vector, config::VECTOR_DIM);
                        if (d < dist) { dist = d; curr_obj = n_id; changed = true; }
                    }
                }
            }

            // 6. Connect Neighbors
            for (int l = std::min(level, current_max_layer_); l >= 0; l--) {
                std::priority_queue<Result> candidates = search_layer(curr_obj, node_ptr->vector, config::EF_CONSTRUCTION, l);
                
                std::vector<id_t> selected_neighbors;
                while (!candidates.empty() && selected_neighbors.size() < (size_t)config::M) {
                    selected_neighbors.push_back(candidates.top().id);
                    candidates.pop();
                }

                for (id_t neighbor_id : selected_neighbors) {
                    add_link(id, neighbor_id, l);
                    add_link(neighbor_id, id, l);
                }
                
                if (!selected_neighbors.empty()) curr_obj = selected_neighbors[0];
            }

            if (level > current_max_layer_) {
                entry_point_id_ = id;
                current_max_layer_ = level;
            }
            
            #pragma omp atomic
            element_count_++; 

            // --- SAVE METADATA ---
            if (!metadata.empty()) {
                metadata_storage_.save_metadata(id, metadata);
            }
        }

        std::vector<Result> search(const std::vector<float>& query, int k) {
            if (entry_point_id_ == -1) return {};

            id_t curr_obj = entry_point_id_;
            float dist = get_distance(query.data(), get_node(curr_obj)->vector, config::VECTOR_DIM);

            for (int l = current_max_layer_; l > 0; l--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    Node* curr_node = get_node(curr_obj);
                    for (int i = 0; i < curr_node->neighbor_counts[l]; i++) {
                        id_t n_id = curr_node->neighbors[l][i];
                        float d = get_distance(query.data(), get_node(n_id)->vector, config::VECTOR_DIM);
                        if (d < dist) { dist = d; curr_obj = n_id; changed = true; }
                    }
                }
            }

            int ef_search = std::max(100, k);
            std::priority_queue<Result> top_candidates = search_layer(curr_obj, query.data(), ef_search, 0);

            std::vector<Result> results;
            while (!top_candidates.empty()) {
                Result r = top_candidates.top();
                // --- LOAD METADATA ---
                r.metadata = metadata_storage_.get_metadata(r.id);
                results.push_back(r);
                top_candidates.pop();
            }
            std::reverse(results.begin(), results.end());
            if (results.size() > (size_t)k) results.resize(k);

            return results;
        }

        // Helper
        std::string get_metadata(id_t id) {
            return metadata_storage_.get_metadata(id);
        }

    private:
        MMapHandler& storage_;
        MetadataHandler metadata_storage_; // <--- The Handler
        id_t entry_point_id_ = -1;
        int current_max_layer_ = -1;
        size_t element_count_ = 0;
        std::mt19937 rng_;
        std::mutex init_lock_;
        
        std::vector<std::unique_ptr<SpinLock>> node_locks_;
        std::mutex global_resize_lock_;

        Node* get_node(id_t id) {
            return reinterpret_cast<Node*>((char*)storage_.get_data() + (size_t)id * sizeof(Node));
        }

        int get_random_level() {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double r = dist(rng_);
            int level = 0;
            while (r < 0.03 && level < config::M) { 
                level++;
                r = dist(rng_);
            }
            return level;
        }

        std::priority_queue<Result> search_layer(id_t entry_point, const float* query_vec, int ef, int layer) {
            std::vector<bool> visited(std::max((size_t)entry_point, element_count_) + 2000, false);
            std::priority_queue<Result, std::vector<Result>, std::greater<Result>> candidates; 
            std::priority_queue<Result> found_results; 

            float d = get_distance(query_vec, get_node(entry_point)->vector, config::VECTOR_DIM);
            Result start_node = {entry_point, d};
            candidates.push(start_node);
            found_results.push(start_node);
            if (entry_point < visited.size()) visited[entry_point] = true;

            while (!candidates.empty()) {
                Result curr = candidates.top();
                candidates.pop();

                if (curr.distance > found_results.top().distance && found_results.size() >= (size_t)ef) break;

                Node* curr_node = get_node(curr.id);
                for (int i = 0; i < curr_node->neighbor_counts[layer]; i++) {
                    id_t neighbor_id = curr_node->neighbors[layer][i];
                    if (neighbor_id >= visited.size() || visited[neighbor_id]) continue;
                    visited[neighbor_id] = true;

                    float dist = get_distance(query_vec, get_node(neighbor_id)->vector, config::VECTOR_DIM);
                    if (found_results.size() < (size_t)ef || dist < found_results.top().distance) {
                        candidates.push({neighbor_id, dist});
                        found_results.push({neighbor_id, dist});
                        if (found_results.size() > (size_t)ef) found_results.pop();
                    }
                }
            }
            return found_results;
        }

        void add_link(id_t src, id_t dest, int layer) {
            if (src >= node_locks_.size()) return; 
            node_locks_[src]->lock(); 

            Node* node = get_node(src);
            int count = node->neighbor_counts[layer];
            int max_conn = (layer == 0) ? config::M_MAX0 : config::M;

            if (count < max_conn) {
                node->neighbors[layer][count] = dest;
                node->neighbor_counts[layer]++;
            } else {
                float dest_dist = get_distance(node->vector, get_node(dest)->vector, config::VECTOR_DIM);
                float max_d = -1.0f;
                int max_idx = -1;

                for(int i=0; i<count; ++i) {
                    float d = get_distance(node->vector, get_node(node->neighbors[layer][i])->vector, config::VECTOR_DIM);
                    if(d > max_d) { max_d = d; max_idx = i; }
                }

                if(dest_dist < max_d && max_idx != -1) {
                    node->neighbors[layer][max_idx] = dest;
                }
            }
            
            node_locks_[src]->unlock(); 
        }
    };

} // namespace nanodb