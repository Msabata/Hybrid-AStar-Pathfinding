#include "hybrid_astar.h"
#include "gpu_hash_tables.cuh"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel function declarations
extern "C" {
    void fusedHeuristicAndValidationWrapper(
        int* d_x, int* d_y, int* d_g, int size,
        int goal_x, int goal_y, int width, int height,
        int* d_weights, float* d_h, int* d_valid, int* d_is_goal,
        int threads_per_block, cudaStream_t stream = 0);
}

// Constructor: Initialize with search space and parallelization configuration
HybridAStar::HybridAStar(const Grid& g, int queues, int threads)
    : grid(g), num_queues(queues), threads_per_block(threads) {
    // GPU resources initialized on-demand
}

// Destructor: Release GPU resources
HybridAStar::~HybridAStar() {
    releaseGPUResources();
}

// Linear index calculation for 2D grid
int HybridAStar::getIndex(int x, int y) const {
    return y * grid.width + x;
}

// Diagonal distance heuristic (admissible for 8-directional grid)
float HybridAStar::calculateHeuristic(int x, int y, const Point& goal) const {
    int dx = abs(x - goal.x);
    int dy = abs(y - goal.y);
    int diag = std::min(dx, dy);
    int straight = dx + dy - 2 * diag;
    // Cost: 14 for diagonal, 10 for cardinal moves
    return 14.0f * diag + 10.0f * straight;
}

// Path reconstruction from closed set nodes
std::vector<Point> HybridAStar::reconstructPath(const std::vector<CompactNode>& nodes, int goal_idx) const {
    std::vector<Point> path;
    int current = goal_idx;

    // Trace back from goal to start using parent indices
    while (current != -1) {
        const CompactNode& node = nodes[current];
        Point p = { node.x, node.y };
        path.push_back(p);
        current = node.parent_idx;
    }

    // Reverse to get start-to-goal ordering
    std::reverse(path.begin(), path.end());
    return path;
}

// Initialize GPU resources with optimal memory allocation pattern
void HybridAStar::initializeGPUResources() {
    if (gpu_initialized) return;

    try {
        // Create CUDA streams for overlapped execution
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&transfer_stream);

        // Allocate device memory with initial capacity
        buffer_capacity = INITIAL_BUFFER_CAPACITY;
        cudaMalloc(&d_x_buffer, buffer_capacity * sizeof(int));
        cudaMalloc(&d_y_buffer, buffer_capacity * sizeof(int));
        cudaMalloc(&d_g_buffer, buffer_capacity * sizeof(int));
        cudaMalloc(&d_parent_buffer, buffer_capacity * sizeof(int));
        cudaMalloc(&d_h_buffer, buffer_capacity * sizeof(float));
        cudaMalloc(&d_valid_buffer, buffer_capacity * sizeof(int));
        cudaMalloc(&d_is_goal_buffer, buffer_capacity * sizeof(int));

        // Pre-cache grid terrain data for reduced transfers
        cudaMalloc(&d_grid_weights, grid.width * grid.height * sizeof(int));
        cudaMemcpyAsync(d_grid_weights, grid.weights,
            grid.width * grid.height * sizeof(int),
            cudaMemcpyHostToDevice, transfer_stream);
        cudaStreamSynchronize(transfer_stream);

        gpu_initialized = true;
    }
    catch (const std::exception& e) {
        // Clean up partial allocations on failure
        releaseGPUResources();
        throw;  // Re-throw for upstream handling
    }
}

// Dynamic buffer resizing with 1.5x growth pattern
void HybridAStar::ensureBufferCapacity(size_t required_size) {
    if (required_size <= buffer_capacity) return;

    // Calculate new capacity with room for growth
    size_t new_capacity = std::max(required_size,
        static_cast<size_t>(buffer_capacity * 1.5f));

    try {
        // Allocate new buffers
        int* new_x_buffer, * new_y_buffer, * new_g_buffer, * new_parent_buffer;
        float* new_h_buffer;
        int* new_valid_buffer, * new_is_goal_buffer;

        cudaMalloc(&new_x_buffer, new_capacity * sizeof(int));
        cudaMalloc(&new_y_buffer, new_capacity * sizeof(int));
        cudaMalloc(&new_g_buffer, new_capacity * sizeof(int));
        cudaMalloc(&new_parent_buffer, new_capacity * sizeof(int));
        cudaMalloc(&new_h_buffer, new_capacity * sizeof(float));
        cudaMalloc(&new_valid_buffer, new_capacity * sizeof(int));
        cudaMalloc(&new_is_goal_buffer, new_capacity * sizeof(int));

        // Copy existing data if present
        if (buffer_capacity > 0) {
            cudaMemcpyAsync(new_x_buffer, d_x_buffer,
                buffer_capacity * sizeof(int),
                cudaMemcpyDeviceToDevice, compute_stream);
            cudaMemcpyAsync(new_y_buffer, d_y_buffer,
                buffer_capacity * sizeof(int),
                cudaMemcpyDeviceToDevice, compute_stream);
            cudaMemcpyAsync(new_g_buffer, d_g_buffer,
                buffer_capacity * sizeof(int),
                cudaMemcpyDeviceToDevice, compute_stream);
            cudaMemcpyAsync(new_parent_buffer, d_parent_buffer,
                buffer_capacity * sizeof(int),
                cudaMemcpyDeviceToDevice, compute_stream);
            cudaMemcpyAsync(new_h_buffer, d_h_buffer,
                buffer_capacity * sizeof(float),
                cudaMemcpyDeviceToDevice, compute_stream);
            cudaMemcpyAsync(new_valid_buffer, d_valid_buffer,
                buffer_capacity * sizeof(int),
                cudaMemcpyDeviceToDevice, compute_stream);
            cudaMemcpyAsync(new_is_goal_buffer, d_is_goal_buffer,
                buffer_capacity * sizeof(int),
                cudaMemcpyDeviceToDevice, compute_stream);

            // Wait for transfers to complete
            cudaStreamSynchronize(compute_stream);

            // Free old buffers
            cudaFree(d_x_buffer);
            cudaFree(d_y_buffer);
            cudaFree(d_g_buffer);
            cudaFree(d_parent_buffer);
            cudaFree(d_h_buffer);
            cudaFree(d_valid_buffer);
            cudaFree(d_is_goal_buffer);
        }

        // Update pointers and capacity
        d_x_buffer = new_x_buffer;
        d_y_buffer = new_y_buffer;
        d_g_buffer = new_g_buffer;
        d_parent_buffer = new_parent_buffer;
        d_h_buffer = new_h_buffer;
        d_valid_buffer = new_valid_buffer;
        d_is_goal_buffer = new_is_goal_buffer;
        buffer_capacity = new_capacity;

    }
    catch (const std::exception& e) {
        // Maintain existing capacity on error
        std::cerr << "Buffer resize failed: " << e.what() << std::endl;
        throw;
    }
}

// Release all GPU resources
void HybridAStar::releaseGPUResources() {
    if (!gpu_initialized) return;

    // Free device memory
    if (d_x_buffer) cudaFree(d_x_buffer);
    if (d_y_buffer) cudaFree(d_y_buffer);
    if (d_g_buffer) cudaFree(d_g_buffer);
    if (d_parent_buffer) cudaFree(d_parent_buffer);
    if (d_h_buffer) cudaFree(d_h_buffer);
    if (d_valid_buffer) cudaFree(d_valid_buffer);
    if (d_is_goal_buffer) cudaFree(d_is_goal_buffer);
    if (d_grid_weights) cudaFree(d_grid_weights);

    // Reset pointers
    d_x_buffer = d_y_buffer = d_g_buffer = d_parent_buffer = nullptr;
    d_valid_buffer = d_is_goal_buffer = d_grid_weights = nullptr;
    d_h_buffer = nullptr;

    // Destroy streams
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);

    // Reset state
    gpu_initialized = false;
    buffer_capacity = 0;
}

// Core A* search algorithm with GPU acceleration
std::vector<Point> HybridAStar::findPath(const Point& start, const Point& goal) {
    nodes_expanded = 0;
    found_goal = false;
    best_goal_node_idx = -1;
    all_nodes_next_idx = 0;
    std::vector<Point> path;

    // Initialize GPU resources on first use
    if (!gpu_initialized) {
        try {
            initializeGPUResources();
        }
        catch (const std::exception& e) {
            std::cerr << "GPU initialization failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            return fallbackCPUSearch(start, goal);
        }
    }

    // Ensure buffer capacity for grid
    if (grid.width * grid.height > buffer_capacity) {
        size_t grid_size = static_cast<size_t>(grid.width) * static_cast<size_t>(grid.height);
        size_t max_size = static_cast<size_t>(1 << 24);  // 16M entries maximum
        try {
            ensureBufferCapacity(std::min(grid_size, max_size));
        }
        catch (const std::exception& e) {
            std::cerr << "Buffer allocation failed: " << e.what() << std::endl;
            return fallbackCPUSearch(start, goal);
        }
    }

    // Initialize data structures
    std::vector<std::priority_queue<CompactNode, std::vector<CompactNode>,
        std::greater<CompactNode>>> open_queues(num_queues);
    std::vector<CompactNode> all_nodes;
    std::unordered_map<Point, CompactNode, PointHash> closed_map;

    // Add start node
    CompactNode start_node(start.x, start.y, 0, calculateHeuristic(start.x, start.y, goal), -1);
    all_nodes.push_back(start_node);
    all_nodes_next_idx = all_nodes.size();

    // Insert to appropriate queue using spatial hash for distribution
    int queue_idx = (start.x * 73856093 + start.y * 19349663) % num_queues;
    if (queue_idx < 0) queue_idx += num_queues;  // Ensure positive index
    open_queues[queue_idx].push(start_node);

    // Main search loop with iteration limit for safety
    int iterations = 0;
    const int max_iterations = grid.width * grid.height;

    while (iterations++ < max_iterations) {
        // Find queue with minimum f-value globally
        float min_f = std::numeric_limits<float>::max();
        int min_queue = -1;

        for (int i = 0; i < num_queues; i++) {
            if (!open_queues[i].empty() && open_queues[i].top().f < min_f) {
                min_f = open_queues[i].top().f;
                min_queue = i;
            }
        }

        // Exit if open set exhausted
        if (min_queue == -1) {
            break;
        }

        // Check for goal state from GPU processing
        if (found_goal && best_goal_node_idx >= 0) {
            return reconstructPath(all_nodes, best_goal_node_idx);
        }

        // Extract node with minimum f-value
        CompactNode current = open_queues[min_queue].top();
        open_queues[min_queue].pop();

        // Create point for closed list lookup
        Point p = { current.x, current.y };
        auto it = closed_map.find(p);
        if (it != closed_map.end() && it->second.g <= current.g) {
            continue;  // Skip if better path already found
        }

        // Add to closed list and tracking structures
        int current_idx = all_nodes.size();
        all_nodes.push_back(current);
        closed_map[p] = current;
        nodes_expanded++;
        all_nodes_next_idx = all_nodes.size();  // Update for goal tracking

        // Direct goal check on CPU side
        if (current.x == goal.x && current.y == goal.y) {
            return reconstructPath(all_nodes, current_idx);
        }

        // Generate successor nodes
        std::vector<CompactNode> neighbors;
        neighbors.reserve(8);  // Pre-allocate for 8-directional movement

        for (int dir = 0; dir < 8; dir++) {
            int nx = current.x + dx[dir];
            int ny = current.y + dy[dir];

            // Skip invalid cells
            if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height)
                continue;

            // Skip walls/obstacles
            int idx = getIndex(nx, ny);
            if (idx < 0 || idx >= grid.width * grid.height || grid.weights[idx] == -1)
                continue;

            // Calculate path cost including terrain weight
            int cell_weight = std::max(1, grid.weights[idx]);
            int ng = current.g + costs[dir] * cell_weight;

            // Create neighbor point
            Point np = { nx, ny };

            // Skip if better path exists in closed list
            auto closed_it = closed_map.find(np);
            if (closed_it != closed_map.end() && closed_it->second.g <= ng)
                continue;

            // Create node and add to batch
            float h = calculateHeuristic(nx, ny, goal);
            CompactNode neighbor(nx, ny, ng, ng + h, current_idx);
            neighbors.push_back(neighbor);
        }

        // Process neighbors batch if non-empty
        if (!neighbors.empty()) {
            try {
                // Choose processing method based on batch characteristics
                if (shouldProcessOnGPU(neighbors.size())) {
                    processBatchGPU(neighbors, goal, open_queues, closed_map);
                }
                else {
                    processBatchCPU(neighbors, goal, open_queues, closed_map);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Batch processing error: " << e.what() << std::endl;
                // Fall back to CPU processing on error
                processBatchCPU(neighbors, goal, open_queues, closed_map);
            }
        }

        // Periodic diagnostic output
        if (nodes_expanded % 10000 == 0) {
            int total_open = std::accumulate(open_queues.begin(), open_queues.end(), 0,
                [](int sum, const auto& q) { return sum + static_cast<int>(q.size()); });

            std::cout << "Processed " << nodes_expanded << " nodes, open: "
                << total_open << ", closed: " << closed_map.size() << std::endl;
        }
    }

    // Return best path or empty if none found
    if (found_goal && best_goal_node_idx >= 0) {
        return reconstructPath(all_nodes, best_goal_node_idx);
    }

    return path;  // Empty path if no solution found
}

// GPU-based batch processing with goal detection
void HybridAStar::processBatchGPU(
    const std::vector<CompactNode>& neighbors,
    const Point& goal,
    std::vector<std::priority_queue<CompactNode, std::vector<CompactNode>,
    std::greater<CompactNode>>>& open_queues,
    const std::unordered_map<Point, CompactNode, PointHash>& closed_map) {

    int batch_size = neighbors.size();
    if (batch_size == 0) return;

    // Ensure adequate buffer capacity
    ensureBufferCapacity(batch_size);

    // Convert to Structure-of-Arrays format for GPU
    std::vector<int> h_x(batch_size);
    std::vector<int> h_y(batch_size);
    std::vector<int> h_g(batch_size);
    std::vector<int> h_parent(batch_size);

    for (int i = 0; i < batch_size; i++) {
        h_x[i] = neighbors[i].x;
        h_y[i] = neighbors[i].y;
        h_g[i] = neighbors[i].g;
        h_parent[i] = neighbors[i].parent_idx;
    }

    // Asynchronous host-to-device transfer 
    cudaMemcpyAsync(d_x_buffer, h_x.data(), batch_size * sizeof(int),
        cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_y_buffer, h_y.data(), batch_size * sizeof(int),
        cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_g_buffer, h_g.data(), batch_size * sizeof(int),
        cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_parent_buffer, h_parent.data(), batch_size * sizeof(int),
        cudaMemcpyHostToDevice, transfer_stream);

    // Wait for transfers to complete
    cudaStreamSynchronize(transfer_stream);

    // Initialize result buffers
    cudaMemsetAsync(d_valid_buffer, 0, batch_size * sizeof(int), compute_stream);
    cudaMemsetAsync(d_is_goal_buffer, 0, batch_size * sizeof(int), compute_stream);

    // Execute fused GPU kernel (heuristic + validation + goal detection)
    fusedHeuristicAndValidationWrapper(
        d_x_buffer, d_y_buffer, d_g_buffer, batch_size,
        goal.x, goal.y, grid.width, grid.height,
        d_grid_weights, d_h_buffer, d_valid_buffer, d_is_goal_buffer,
        threads_per_block, compute_stream);

    // Asynchronous device-to-host transfer of results
    std::vector<float> h_h(batch_size);
    std::vector<int> h_valid(batch_size);
    std::vector<int> h_is_goal(batch_size);

    cudaMemcpyAsync(h_h.data(), d_h_buffer, batch_size * sizeof(float),
        cudaMemcpyDeviceToHost, transfer_stream);
    cudaMemcpyAsync(h_valid.data(), d_valid_buffer, batch_size * sizeof(int),
        cudaMemcpyDeviceToHost, transfer_stream);
    cudaMemcpyAsync(h_is_goal.data(), d_is_goal_buffer, batch_size * sizeof(int),
        cudaMemcpyDeviceToHost, transfer_stream);

    // Wait for results
    cudaStreamSynchronize(transfer_stream);
    cudaStreamSynchronize(compute_stream);

    // Check for goal nodes first (higher priority)
    for (int i = 0; i < batch_size; i++) {
        if (h_valid[i] && h_is_goal[i]) {
            // Found goal state in GPU batch
            CompactNode goal_node(
                h_x[i], h_y[i], h_g[i],
                h_g[i] + h_h[i], h_parent[i]
            );

            // Atomic goal state update
#pragma omp critical
            {
                if (!found_goal || goal_node.f < best_goal_node.f) {
                    found_goal = true;
                    best_goal_node = goal_node;
                    best_goal_node_idx = all_nodes_next_idx + i;
                }
            }
        }
    }

    // Process remaining valid nodes in parallel
#pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        if (h_valid[i]) {
            // Create node with computed heuristic
            CompactNode node(
                h_x[i], h_y[i], h_g[i],
                h_g[i] + h_h[i], h_parent[i]
            );

            // Determine queue assignment with consistent spatial hash
            int queue_idx = (h_x[i] * 73856093 + h_y[i] * 19349663) % num_queues;
            if (queue_idx < 0) queue_idx += num_queues;  // Ensure positive index

            // Thread-safe insertion
#pragma omp critical
            {
                open_queues[queue_idx].push(node);
            }
        }
    }
}

// CPU-based batch processing (fallback path)
void HybridAStar::processBatchCPU(
    const std::vector<CompactNode>& neighbors,
    const Point& goal,
    std::vector<std::priority_queue<CompactNode, std::vector<CompactNode>,
    std::greater<CompactNode>>>& open_queues,
    const std::unordered_map<Point, CompactNode, PointHash>& closed_map) {

    // Process neighbors in parallel with OpenMP
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(neighbors.size()); i++) {
        const auto& neighbor = neighbors[i];

        // Skip if already in closed list with better g-value
        Point p = { neighbor.x, neighbor.y };

        bool skip = false;
#pragma omp critical
        {
            auto it = closed_map.find(p);
            if (it != closed_map.end() && it->second.g <= neighbor.g) {
                skip = true;
            }
        }

        if (skip) continue;

        // Check for goal state
        if (neighbor.x == goal.x && neighbor.y == goal.y) {
#pragma omp critical
            {
                if (!found_goal || neighbor.f < best_goal_node.f) {
                    found_goal = true;
                    best_goal_node = neighbor;
                    best_goal_node_idx = all_nodes_next_idx + i;
                }
            }
        }

        // Determine queue index with consistent spatial hash
        int queue_idx = (neighbor.x * 73856093 + neighbor.y * 19349663) % num_queues;
        if (queue_idx < 0) queue_idx += num_queues;

        // Add to appropriate queue
#pragma omp critical
        {
            open_queues[queue_idx].push(neighbor);
        }
    }
}

// CPU-only A* implementation for fallback
std::vector<Point> HybridAStar::fallbackCPUSearch(const Point& start, const Point& goal) {
    std::vector<Point> path;
    std::vector<CompactNode> all_nodes;
    std::unordered_map<Point, CompactNode, PointHash> closed_map;
    std::priority_queue<CompactNode, std::vector<CompactNode>, std::greater<CompactNode>> open_queue;

    // Initialize with start node
    CompactNode start_node(start.x, start.y, 0, calculateHeuristic(start.x, start.y, goal), -1);
    all_nodes.push_back(start_node);
    open_queue.push(start_node);

    // Standard A* search loop
    while (!open_queue.empty()) {
        CompactNode current = open_queue.top();
        open_queue.pop();

        Point p = { current.x, current.y };
        if (closed_map.find(p) != closed_map.end() && closed_map[p].g <= current.g) {
            continue;  // Skip if better path exists
        }

        int current_idx = all_nodes.size();
        all_nodes.push_back(current);
        closed_map[p] = current;
        nodes_expanded++;

        // Goal test
        if (current.x == goal.x && current.y == goal.y) {
            return reconstructPath(all_nodes, current_idx);
        }

        // Expand neighbors
        for (int dir = 0; dir < 8; dir++) {
            int nx = current.x + dx[dir];
            int ny = current.y + dy[dir];

            // Skip invalid cells
            if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height) continue;

            int idx = getIndex(nx, ny);
            if (grid.weights[idx] == -1) continue;  // Wall/obstacle

            // Calculate path cost
            int ng = current.g + costs[dir] * std::max(1, grid.weights[idx]);
            Point np = { nx, ny };

            // Skip if in closed list with better g
            if (closed_map.find(np) != closed_map.end() && closed_map[np].g <= ng) continue;

            // Create and add node
            float h = calculateHeuristic(nx, ny, goal);
            open_queue.push(CompactNode(nx, ny, ng, ng + h, current_idx));
        }
    }

    return path;  // Empty path if no solution
}

// Determine if batch should use GPU based on size and utilization
bool HybridAStar::shouldProcessOnGPU(size_t batch_size) const {
    // Small batches have high overhead on GPU
    if (batch_size < GPU_MIN_BATCH_THRESHOLD)  // Changed from GPU_MIN_BATCH_SIZE
        return false;

    // Large batches benefit from GPU parallelism
    if (batch_size > GPU_PREFERRED_BATCH_SIZE)
        return true;

    // Medium batches based on current GPU utilization
    return estimateGPUUtilization() < 0.8f;
}

// Estimate current GPU utilization (values from 0.0 to 1.0)
float HybridAStar::estimateGPUUtilization() const {
    return gpu_utilization_estimate;
}

// Update GPU utilization estimate with exponential moving average
void HybridAStar::updateGPUUtilizationEstimate(float new_estimate) {
    const float alpha = 0.2f;  // EMA coefficient for smoothing
    gpu_utilization_estimate = alpha * new_estimate + (1.0f - alpha) * gpu_utilization_estimate;
}