#pragma once

#include "node.hpp"
#include "distance.hpp"
#include "../common/config.hpp"
#include "../storage/mmap_handler.hpp"
#include <queue>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h> // OpenMP for locks/parallelism

namespace nanodb {

    class HNSW {
    public:
        // ---------------------------------------------------------
        // Constructor / Destructor
        // ---------------------------------------------------------
        HNSW(MMapHandler& storage) : storage_(storage) {
            // Seed random number generator for layer assignment
            std::random_device rd;
            rng_.seed(rd());
            
            // If file is new/empty, initialize metadata
            if (storage_.get_size() == 0) {
                entry_point_id_ = -1; // -1 means empty graph
                current_max_layer_ = -1;
                element_count_ = 0;
            } else {
                // In a real DB, we would load metadata from a header. 
                // For this MVP, we will assume a fresh start or simple reload logic.
                // (Simplified for brevity)
                entry_point_id_ = 0; 
                current_max_layer_ = 0; 
                element_count_ = storage_.get_size() / sizeof(Node);
            }
        }

        // ---------------------------------------------------------
        // Public API
        // ---------------------------------------------------------

        // Insert a new vector into the index
        void insert(const std::vector<float>& vec_data, id_t id) {
            // 1. Determine the level for this new node
            // Randomly assign a level. Most nodes stay at 0, some go to 1, few to 2...
            int level = get_random_level();
            
            // 2. Prepare the new Node
            Node new_node(id, level, vec_data);

            // 3. Write Node to Disk (via mmap)
            // Calculate offset based on ID (assuming sequential IDs for now)
            size_t offset = (size_t)id * sizeof(Node);
            
            // Ensure file is large enough
            if (offset + sizeof(Node) > storage_.get_size()) {
                 // Expand by 10MB chunks to avoid frequent resizing
                storage_.resize(storage_.get_size() + 10 * 1024 * 1024);
            }

            // Copy node into memory-mapped region
            Node* node_ptr = get_node(id);
            *node_ptr = new_node;

            // 4. Critical Section: Update Graph
            // If graph is empty, this is the entry point
            if (entry_point_id_ == -1) {
                entry_point_id_ = id;
                current_max_layer_ = level;
                return;
            }

            // 5. Search for the nearest entry point at the top layer
            id_t curr_obj = entry_point_id_;
            float dist = get_distance(node_ptr->vector, get_node(curr_obj)->vector, config::VECTOR_DIM);

            // Zoom down from top layer to the node's specific level
            for (int l = current_max_layer_; l > level; l--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    Node* curr_node = get_node(curr_obj);
                    
                    // Greedily move to a closer neighbor
                    for (int i = 0; i < curr_node->neighbor_counts[l]; i++) {
                        id_t n_id = curr_node->neighbors[l][i];
                        Node* n_node = get_node(n_id);
                        float d = get_distance(node_ptr->vector, n_node->vector, config::VECTOR_DIM);
                        if (d < dist) {
                            dist = d;
                            curr_obj = n_id;
                            changed = true;
                        }
                    }
                }
            }

            // 6. Connect neighbors from 'level' down to 0
            for (int l = std::min(level, current_max_layer_); l >= 0; l--) {
                // Find K nearest candidates in this layer
                std::priority_queue<Result> candidates = search_layer(curr_obj, node_ptr->vector, config::EF_CONSTRUCTION, l);
                
                // Select best neighbors (heuristic: simple closest for MVP)
                std::vector<id_t> selected_neighbors;
                while (!candidates.empty() && selected_neighbors.size() < (size_t)config::M) {
                    selected_neighbors.push_back(candidates.top().id);
                    candidates.pop();
                }

                // Add connections (bidirectional)
                for (id_t neighbor_id : selected_neighbors) {
                    add_link(id, neighbor_id, l);
                    add_link(neighbor_id, id, l);
                }
                
                // Use the best candidate as the start for the next layer down
                if (!selected_neighbors.empty()) {
                    curr_obj = selected_neighbors[0]; 
                }
            }

            // Update global max layer if needed
            if (level > current_max_layer_) {
                entry_point_id_ = id;
                current_max_layer_ = level;
            }
        }

        // K-Nearest Neighbor Search
        std::vector<Result> search(const std::vector<float>& query, int k) {
            if (entry_point_id_ == -1) return {}; // Empty graph

            id_t curr_obj = entry_point_id_;
            float dist = get_distance(query.data(), get_node(curr_obj)->vector, config::VECTOR_DIM);

            // 1. Zoom down to Layer 0
            for (int l = current_max_layer_; l > 0; l--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    Node* curr_node = get_node(curr_obj);
                    for (int i = 0; i < curr_node->neighbor_counts[l]; i++) {
                        id_t n_id = curr_node->neighbors[l][i];
                        float d = get_distance(query.data(), get_node(n_id)->vector, config::VECTOR_DIM);
                        if (d < dist) {
                            dist = d;
                            curr_obj = n_id;
                            changed = true;
                        }
                    }
                }
            }

            // 2. Precision Search at Layer 0
            // EF_SEARCH should be >= K
            int ef_search = std::max(100, k);
            std::priority_queue<Result> top_candidates = search_layer(curr_obj, query.data(), ef_search, 0);

            // 3. Extract Top K
            std::vector<Result> results;
            while (!top_candidates.empty()) {
                results.push_back(top_candidates.top());
                top_candidates.pop();
            }
            
            // The priority queue pops worst first (Max Heap), so reverse to get best first
            std::reverse(results.begin(), results.end());
            if (results.size() > (size_t)k) {
                results.resize(k);
            }

            return results;
        }

    private:
        MMapHandler& storage_;
        id_t entry_point_id_ = -1;
        int current_max_layer_ = -1;
        size_t element_count_ = 0;
        std::mt19937 rng_;

        // Helper: Convert integer ID to Memory Pointer
        Node* get_node(id_t id) {
            return reinterpret_cast<Node*>((char*)storage_.get_data() + (size_t)id * sizeof(Node));
        }

        // Helper: Determine random layer height (like a skip-list)
        int get_random_level() {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double r = dist(rng_);
            int level = 0;
            // ~3% chance to go up a level (approx log scale)
            while (r < 0.03 && level < MAX_LAYERS - 1) { 
                level++;
                r = dist(rng_);
            }
            return level;
        }

        // Core Algorithm: Search within a specific layer
        // Returns a Max-Heap of the nearest elements found
        std::priority_queue<Result> search_layer(id_t entry_point, const float* query_vec, int ef, int layer) {
            
            // Visited set (using a simple vector for MVP, bitset is faster)
            // In production, use a dedicated "VisitedList" to avoid allocations
            std::vector<bool> visited(element_count_ + 1000, false); // +buffer
            
            // Candidates to explore (Min-Heap: closest first)
            std::priority_queue<Result, std::vector<Result>, std::greater<Result>> candidates;
            
            // Top results found so far (Max-Heap: farthest first)
            std::priority_queue<Result> found_results;

            float d = get_distance(query_vec, get_node(entry_point)->vector, config::VECTOR_DIM);
            Result start_node = {entry_point, d};
            
            candidates.push(start_node);
            found_results.push(start_node);
            if (entry_point < visited.size()) visited[entry_point] = true;

            while (!candidates.empty()) {
                Result curr = candidates.top();
                candidates.pop();

                // Optimization: If the closest candidate is worse than the worst found result, stop.
                if (curr.distance > found_results.top().distance && found_results.size() >= (size_t)ef) {
                    break;
                }

                Node* curr_node = get_node(curr.id);
                for (int i = 0; i < curr_node->neighbor_counts[layer]; i++) {
                    id_t neighbor_id = curr_node->neighbors[layer][i];
                    
                    if (neighbor_id >= visited.size()) continue; // Safety check
                    if (visited[neighbor_id]) continue;
                    visited[neighbor_id] = true;

                    float dist = get_distance(query_vec, get_node(neighbor_id)->vector, config::VECTOR_DIM);
                    Result neighbor_res = {neighbor_id, dist};

                    if (found_results.size() < (size_t)ef || dist < found_results.top().distance) {
                        candidates.push(neighbor_res);
                        found_results.push(neighbor_res);
                        if (found_results.size() > (size_t)ef) {
                            found_results.pop(); // Remove worst candidate
                        }
                    }
                }
            }
            return found_results;
        }

        // Helper: Add a connection between two nodes
        void add_link(id_t src, id_t dest, int layer) {
            Node* node = get_node(src);
            int count = node->neighbor_counts[layer];
            int max_conn = (layer == 0) ? config::M_MAX0 : config::M;

            if (count < max_conn) {
                // Space available, just add
                node->neighbors[layer][count] = dest;
                node->neighbor_counts[layer]++;
            } else {
                // List is full. Simple heuristic:
                // Find the farthest neighbor and replace it if 'dest' is closer.
                // (Production HNSW uses advanced heuristics here)
                float max_d = -1.0f;
                int max_idx = -1;
                
                // Calculate distance to new candidate
                float dest_dist = get_distance(node->vector, get_node(dest)->vector, config::VECTOR_DIM);

                for(int i=0; i<count; ++i) {
                    float d = get_distance(node->vector, get_node(node->neighbors[layer][i])->vector, config::VECTOR_DIM);
                    if(d > max_d) {
                        max_d = d;
                        max_idx = i;
                    }
                }

                if(dest_dist < max_d && max_idx != -1) {
                    node->neighbors[layer][max_idx] = dest;
                }
            }
        }
    };

} // namespace nanodb