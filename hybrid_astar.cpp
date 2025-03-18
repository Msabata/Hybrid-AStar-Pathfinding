#include "hybrid_astar.h"
#include "gpu_hash_tables.cuh"
#include <algorithm>
#include <cmath>
#include <omp.h> // Added
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernels for heuristic calculation and node filtering
// Note: These would typically be in a .cu file, but are included here for simplicity
extern "C" {
    void computeHeuristicWrapper(int* d_x, int* d_y, int size, int goal_x, int goal_y, float* d_h, int threads_per_block);
    void filterNodesWrapper(int* d_x, int* d_y, int* d_g, int size, int width, int height, int* d_weights, 
                           int* d_valid, int* d_closed_x, int* d_closed_y, int* d_closed_g, int closed_size, int threads_per_block);
}

HybridAStar::HybridAStar(const Grid& g, int queues, int threads) 
    : grid(g), num_queues(queues), threads_per_block(threads) {}

// Helper method to get index in grid array
int HybridAStar::getIndex(int x, int y) const {
    return y * grid.width + x;
}

// CPU version of heuristic calculation
float HybridAStar::calculateHeuristic(int x, int y, const Point& goal) const {
    // Diagonal distance heuristic
    int dx = abs(x - goal.x);
    int dy = abs(y - goal.y);
    int min_coord = std::min(dx, dy);
    int max_coord = std::max(dx, dy);
    return 10.0f * (dx + dy) + (14.0f - 2 * 10.0f) * min_coord;
}

// Reconstruct path from closed list
std::vector<Point> HybridAStar::reconstructPath(const std::vector<Node>& nodes, int goal_idx) const {
    std::vector<Point> path;
    int current = goal_idx;
    
    while (current != -1) {
        const Node& node = nodes[current];
        Point p = { node.x, node.y };
        path.push_back(p);
        current = node.parent_idx;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

// Main A* search function
std::vector<Point> HybridAStar::findPath(const Point& start, const Point& goal) {
    std::vector<Point> path;
    
    // Initialize multiple priority queues for open list
    std::vector<std::priority_queue<Node, std::vector<Node>, std::greater<Node>>> open_queues(num_queues);
    
    // Initialize closed list and nodes list
    std::vector<Node> all_nodes;
    std::unordered_map<Point, Node, PointHash> closed_map;
    
    // Add start node
    Node start_node(start.x, start.y, 0, calculateHeuristic(start.x, start.y, goal), -1);
    all_nodes.push_back(start_node);
    
    // Add to first queue
    open_queues[0].push(start_node);
    
    // Target node index (when found)
    int target_idx = -1;
    
    // Main search loop
    while (true) {
        bool all_empty = true;
        std::vector<Node> expanded_nodes;
        
        // Extract nodes from each queue in parallel
        #pragma omp parallel for
        for (int i = 0; i < num_queues; i++) {
            if (!open_queues[i].empty()) {
                #pragma omp critical
                {
                    all_empty = false;
                }
                
                Node current = open_queues[i].top();
                open_queues[i].pop();
                
                // Check if already processed
                Point p = { current.x, current.y };
                if (closed_map.find(p) != closed_map.end())
                    continue;
                
                // Add to closed list
                #pragma omp critical
                {
                    closed_map[p] = current;
                }
                
                // If target is reached
                if (current.x == goal.x && current.y == goal.y) {
                    #pragma omp critical
                    {
                        int node_idx = all_nodes.size() - 1;
                        if (target_idx == -1 || current.f < all_nodes[target_idx].f) {
                            target_idx = node_idx;
                        }
                    }
                    continue;
                }
                
                // Expand neighbors
                for (int dir = 0; dir < 8; dir++) {
                    int nx = current.x + dx[dir];
                    int ny = current.y + dy[dir];
                    
                    // Skip out-of-bounds cells
                    if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height)
                        continue;
                    
                    // Skip walls
                    int idx = getIndex(nx, ny);
                    if (grid.weights[idx] == -1)
                        continue;
                    
                    // Calculate g value
                    int ng = current.g + costs[dir];
                    
                    // Create point for lookup
                    Point np = { nx, ny };
                    
                    // Skip if already in closed list with better g
                    if (closed_map.find(np) != closed_map.end() && 
                        closed_map[np].g <= ng)
                        continue;
                    
                    // Calculate heuristic and create node
                    float h = calculateHeuristic(nx, ny, goal);
                    Node neighbor(nx, ny, ng, ng + h, all_nodes.size() - 1);
                    
                    #pragma omp critical
                    {
                        all_nodes.push_back(neighbor);
                        expanded_nodes.push_back(neighbor);
                    }
                }
            }
        }
        
        // If all queues are empty, break
        if (all_empty)
            break;
        
        // If goal is found and has best f-value, break
        if (target_idx != -1) {
            bool is_optimal = true;
            
            for (int i = 0; i < num_queues; i++) {
                if (!open_queues[i].empty() && 
                    open_queues[i].top().f < all_nodes[target_idx].f) {
                    is_optimal = false;
                    break;
                }
            }
            
            if (is_optimal) {
                return reconstructPath(all_nodes, target_idx);
            }
        }
        
        // Process expanded nodes with GPU
        if (!expanded_nodes.empty()) {
            processBatchGPU(expanded_nodes, goal, open_queues, closed_map);
        }
    }
    
    // If target was found, reconstruct path
    if (target_idx != -1) {
        return reconstructPath(all_nodes, target_idx);
    }
    
    // No path found
    return path;
}

// GPU batch processing of expanded nodes
void HybridAStar::processBatchGPU(const std::vector<Node>& expanded_nodes, 
                                 const Point& goal, 
                                 std::vector<std::priority_queue<Node, std::vector<Node>, std::greater<Node>>>& open_queues, const std::unordered_map<Point, Node, PointHash>& closed_map) {
    int batch_size = expanded_nodes.size();
    if (batch_size == 0) return;
    
    // Prepare data for GPU
    std::vector<int> h_x(batch_size);
    std::vector<int> h_y(batch_size);
    std::vector<int> h_g(batch_size);
    std::vector<int> h_parent(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        h_x[i] = expanded_nodes[i].x;
        h_y[i] = expanded_nodes[i].y;
        h_g[i] = expanded_nodes[i].g;
        h_parent[i] = expanded_nodes[i].parent_idx;
    }
    
    // Allocate device memory
    int *d_x, *d_y, *d_g, *d_parent, *d_valid;
    float *d_h;
    
    cudaMalloc(&d_x, batch_size * sizeof(int));
    cudaMalloc(&d_y, batch_size * sizeof(int));
    cudaMalloc(&d_g, batch_size * sizeof(int));
    cudaMalloc(&d_parent, batch_size * sizeof(int));
    cudaMalloc(&d_h, batch_size * sizeof(float));
    cudaMalloc(&d_valid, batch_size * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_x, h_x.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, h_parent.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Compute heuristics in parallel on GPU
    computeHeuristicWrapper(d_x, d_y, batch_size, goal.x, goal.y, d_h, threads_per_block);
    
    // Copy grid weights to device if needed
    int *d_weights;
    cudaMalloc(&d_weights, grid.width * grid.height * sizeof(int));
    cudaMemcpy(d_weights, grid.weights, grid.width * grid.height * sizeof(int), cudaMemcpyHostToDevice);
    
    // Prepare closed list for GPU
    std::vector<int> closed_x, closed_y, closed_g;
    for (const auto& pair : closed_map) {
        closed_x.push_back(pair.first.x);
        closed_y.push_back(pair.first.y);
        closed_g.push_back(pair.second.g);
    }
    
    int closed_size = closed_x.size();
    int *d_closed_x = nullptr, *d_closed_y = nullptr, *d_closed_g = nullptr;
    
    if (closed_size > 0) {
        cudaMalloc(&d_closed_x, closed_size * sizeof(int));
        cudaMalloc(&d_closed_y, closed_size * sizeof(int));
        cudaMalloc(&d_closed_g, closed_size * sizeof(int));
        
        cudaMemcpy(d_closed_x, closed_x.data(), closed_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_closed_y, closed_y.data(), closed_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_closed_g, closed_g.data(), closed_size * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    // Filter valid nodes on GPU
    filterNodesWrapper(d_x, d_y, d_g, batch_size, grid.width, grid.height, d_weights,
                     d_valid, d_closed_x, d_closed_y, d_closed_g, closed_size, threads_per_block);
    
    // Copy results back to host
    std::vector<float> h_h(batch_size);
    std::vector<int> h_valid(batch_size);
    
    cudaMemcpy(h_h.data(), d_h, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid.data(), d_valid, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Add valid nodes to open queues
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        if (h_valid[i]) {
            Node node(h_x[i], h_y[i], h_g[i], h_g[i] + h_h[i], h_parent[i]);
            
            // Distribute to queues (simplified hash distribution)
            int queue_idx = (h_x[i] * 31 + h_y[i]) % num_queues;
            
            #pragma omp critical
            {
                open_queues[queue_idx].push(node);
            }
        }
    }
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_g);
    cudaFree(d_parent);
    cudaFree(d_h);
    cudaFree(d_valid);
    cudaFree(d_weights);
    
    if (closed_size > 0) {
        cudaFree(d_closed_x);
        cudaFree(d_closed_y);
        cudaFree(d_closed_g);
    }
}

// Implementation of grid generator function
Grid grid_generator(int width, int height, int seed) {
    Grid grid;
    grid.width = width;
    grid.height = height;
    grid.weights = new int[width * height];
    
    // Initialize with random seed
    srand(seed);
    
    // Fill with random weights (1-10), -1 for walls
    for (int i = 0; i < width * height; i++) {
        // 20% chance of wall
        if (rand() % 100 < 20) {
            grid.weights[i] = -1;  // Wall
        } else {
            grid.weights[i] = 1 + (rand() % 10);  // Random weight 1-10
        }
    }
    
    // Ensure start and end are not walls (assuming corners)
    grid.weights[0] = 1;  // Top-left
    grid.weights[width * height - 1] = 1;  // Bottom-right
    
    return grid;
}