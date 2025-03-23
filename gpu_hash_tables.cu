#include "gpu_hash_tables.cuh"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <iostream>
#include <climits>

// Hash function constants for uniform distribution
const unsigned int HASH_SEED1 = 0x1234ABCD;
const unsigned int HASH_SEED2 = 0xA1B2C3D4;

// MurmurHash3-inspired hash function with improved distribution
__host__ __device__ unsigned int hashPoint(const Point& p, unsigned int seed, unsigned int table_size) {
    unsigned int hash = seed;

    // Mix point coordinates with multiplicative primes
    hash ^= (static_cast<unsigned int>(p.x) * 73856093u);
    hash = (hash ^ (hash >> 16)) * 0x85ebca6bu;

    hash ^= (static_cast<unsigned int>(p.y) * 19349663u);
    hash = (hash ^ (hash >> 13)) * 0xc2b2ae35u;

    // Final mixing
    hash = hash ^ (hash >> 16);

    // Modulo with table size
    return hash % table_size;
}

// Optimized Cuckoo hash insert kernel with atomic operations and conflict resolution
__global__ void optimizedInsertCuckooKernel(
    Point* d_points, int* d_g_values, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int max_iterations, int* d_success) {

    // Compute global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load point and g-value (coalesced memory access)
    Point p = d_points[idx];
    int g = d_g_values[idx];

    // Calculate hash positions for primary and secondary tables
    unsigned int pos1 = hashPoint(p, HASH_SEED1, table_size);
    unsigned int pos2 = hashPoint(p, HASH_SEED2, table_size);

    // Try inserting into primary table (atomically)
    if (atomicCAS(&d_table1[pos1].valid, 0, 1) == 0) {
        // Slot was empty, write values
        d_table1[pos1].x = p.x;
        d_table1[pos1].y = p.y;
        d_table1[pos1].g = g;
        d_success[idx] = 1;
        return;
    }

    // Check if already exists in primary table (update g if better)
    if (d_table1[pos1].x == p.x && d_table1[pos1].y == p.y) {
        atomicMin(&d_table1[pos1].g, g);
        d_success[idx] = 1;
        return;
    }

    // Try inserting into secondary table
    if (atomicCAS(&d_table2[pos2].valid, 0, 1) == 0) {
        d_table2[pos2].x = p.x;
        d_table2[pos2].y = p.y;
        d_table2[pos2].g = g;
        d_success[idx] = 1;
        return;
    }

    // Check if already exists in secondary table
    if (d_table2[pos2].x == p.x && d_table2[pos2].y == p.y) {
        atomicMin(&d_table2[pos2].g, g);
        d_success[idx] = 1;
        return;
    }

    // Start cuckoo hashing with eviction chain
    Point curr_p = p;
    int curr_g = g;
    unsigned int curr_pos;
    bool insert_table1 = true;  // Alternate between tables

    // Perform Cuckoo displacement iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        if (insert_table1) {
            // Insert into table 1, potentially displacing existing entry
            curr_pos = hashPoint(curr_p, HASH_SEED1, table_size);

            // Evict current entry
            Point evicted_p = { d_table1[curr_pos].x, d_table1[curr_pos].y };
            int evicted_g = d_table1[curr_pos].g;

            // Replace with new entry
            d_table1[curr_pos].x = curr_p.x;
            d_table1[curr_pos].y = curr_p.y;
            d_table1[curr_pos].g = curr_g;

            // Prepare evicted entry for next iteration
            curr_p = evicted_p;
            curr_g = evicted_g;
        }
        else {
            // Insert into table 2
            curr_pos = hashPoint(curr_p, HASH_SEED2, table_size);

            // Evict current entry
            Point evicted_p = { d_table2[curr_pos].x, d_table2[curr_pos].y };
            int evicted_g = d_table2[curr_pos].g;

            // Replace with new entry
            d_table2[curr_pos].x = curr_p.x;
            d_table2[curr_pos].y = curr_p.y;
            d_table2[curr_pos].g = curr_g;

            // Prepare evicted entry for next iteration
            curr_p = evicted_p;
            curr_g = evicted_g;
        }

        // Check if we can place the evicted entry in the other table without displacement
        if (insert_table1) {
            // Try table 2 for evicted entry
            unsigned int alt_pos = hashPoint(curr_p, HASH_SEED2, table_size);
            if (d_table2[alt_pos].valid == 0) {
                // Empty slot found in alternate table
                d_table2[alt_pos].x = curr_p.x;
                d_table2[alt_pos].y = curr_p.y;
                d_table2[alt_pos].g = curr_g;
                d_table2[alt_pos].valid = 1;
                d_success[idx] = 1;
                return;
            }
        }
        else {
            // Try table 1 for evicted entry
            unsigned int alt_pos = hashPoint(curr_p, HASH_SEED1, table_size);
            if (d_table1[alt_pos].valid == 0) {
                // Empty slot found in alternate table
                d_table1[alt_pos].x = curr_p.x;
                d_table1[alt_pos].y = curr_p.y;
                d_table1[alt_pos].g = curr_g;
                d_table1[alt_pos].valid = 1;
                d_success[idx] = 1;
                return;
            }
        }

        // Toggle between tables for next iteration
        insert_table1 = !insert_table1;
    }

    // Exceeded maximum iterations - insertion failed
    d_success[idx] = 0;
}

// Optimized lookup kernel with early exit
__global__ void optimizedFindCuckooKernel(
    Point* d_points, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int* d_found, int* d_g_values) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load point (coalesced memory access)
    Point p = d_points[idx];

    // Check primary table
    unsigned int pos1 = hashPoint(p, HASH_SEED1, table_size);
    if (d_table1[pos1].valid && d_table1[pos1].x == p.x && d_table1[pos1].y == p.y) {
        d_found[idx] = 1;
        d_g_values[idx] = d_table1[pos1].g;
        return;
    }

    // If not found, check secondary table
    unsigned int pos2 = hashPoint(p, HASH_SEED2, table_size);
    if (d_table2[pos2].valid && d_table2[pos2].x == p.x && d_table2[pos2].y == p.y) {
        d_found[idx] = 1;
        d_g_values[idx] = d_table2[pos2].g;
        return;
    }

    // Not found in either table
    d_found[idx] = 0;
    d_g_values[idx] = INT_MAX;
}

// Fused kernel for heuristic calculation and validation
// Combines multiple operations to reduce kernel launch overhead
__global__ void fusedHeuristicAndValidationKernel(
        int* d_x, int* d_y, int* d_g, int size,
        int goal_x, int goal_y, int width, int height,
        int* d_weights, float* d_h, int* d_valid, int* d_is_goal) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Load node coordinates (coalesced memory access)
    int x = d_x[idx];
    int y = d_y[idx];

    // Compute heuristic with optimal diagonal distance metric
    int dx = abs(x - goal_x);
    int dy = abs(y - goal_y);
    int diag = min(dx, dy);
    int straight = dx + dy - 2 * diag;
    d_h[idx] = 14.0f * diag + 10.0f * straight;

    // Initialize validity and goal flags
    d_valid[idx] = 0;
    d_is_goal[idx] = 0;

    // Validate node (boundary check)
    if (x < 0 || x >= width || y < 0 || y >= height)
        return;

    // Grid cell index with bounds verification
    int cell_idx = y * width + x;
    if (cell_idx < 0 || cell_idx >= width * height)
        return;

    // Check for obstacles/walls
    if (d_weights[cell_idx] == -1)
        return;

    // Node passed all validation checks
    d_valid[idx] = 1;

    // Explicit goal state detection
    if (x == goal_x && y == goal_y) {
        d_is_goal[idx] = 1;
    }
}
// Wrapper functions that can be called from C++ code

extern "C" {
    // Run fused heuristic and validation kernel
    void fusedHeuristicAndValidationWrapper(
        int* d_x, int* d_y, int* d_g, int size,
        int goal_x, int goal_y, int width, int height,
        int* d_weights, float* d_h, int* d_valid, int* d_is_goal,
        int threads_per_block, cudaStream_t stream) {

        int blocks = (size + threads_per_block - 1) / threads_per_block;
        fusedHeuristicAndValidationKernel << <blocks, threads_per_block, 0, stream >> > (
            d_x, d_y, d_g, size, goal_x, goal_y, width, height,
            d_weights, d_h, d_valid, d_is_goal);
    }
    // Compute heuristic only (for benchmarking comparison)
    void computeHeuristicWrapper(
        int* d_x, int* d_y, int size, int goal_x, int goal_y,
        float* d_h, int threads_per_block, cudaStream_t stream) {

        int blocks = (size + threads_per_block - 1) / threads_per_block;
        optimizedHeuristicKernel << <blocks, threads_per_block, 0, stream >> > (
            d_x, d_y, size, goal_x, goal_y, d_h);
    }
}

// Standalone heuristic kernel (for comparison with fused kernel)
__global__ void optimizedHeuristicKernel(
    int* d_x, int* d_y, int size, int goal_x, int goal_y, float* d_h) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Optimized diagonal distance heuristic
    int dx = abs(d_x[idx] - goal_x);
    int dy = abs(d_y[idx] - goal_y);
    int diag = min(dx, dy);
    int straight = dx + dy - 2 * diag;
    d_h[idx] = 14.0f * diag + 10.0f * straight;
}

// CuckooHashGPU implementation
CuckooHashGPU::CuckooHashGPU(unsigned int table_size, unsigned int max_iter, cudaStream_t cuda_stream)
    : size(table_size), max_iterations(max_iter), stream(cuda_stream), has_stream_ownership(false) {

    // Create stream if not provided
    if (stream == nullptr) {
        cudaStreamCreate(&stream);
        has_stream_ownership = true;
    }

    // Allocate GPU memory for hash tables
    cudaMalloc(&d_table1, size * sizeof(Entry));
    cudaMalloc(&d_table2, size * sizeof(Entry));

    // Initialize tables to empty state
    cudaMemsetAsync(d_table1, 0, size * sizeof(Entry), stream);
    cudaMemsetAsync(d_table2, 0, size * sizeof(Entry), stream);

    // Pre-allocate host buffers for batch operations
    prefetch_points.reserve(1024);
    prefetch_g_values.reserve(1024);
}

CuckooHashGPU::~CuckooHashGPU() {
    // Free GPU memory
    cudaFree(d_table1);
    cudaFree(d_table2);

    // Destroy stream if we created it
    if (has_stream_ownership) {
        cudaStreamDestroy(stream);
    }
}

bool CuckooHashGPU::insertBatch(const std::vector<Point>& points, const std::vector<int>& g_values,
    int threads_per_block) {
    int batch_size = points.size();
    if (batch_size == 0) return true;

    // Allocate device memory for batch data
    Point* d_points;
    int* d_g_values;
    int* d_success;

    cudaMalloc(&d_points, batch_size * sizeof(Point));
    cudaMalloc(&d_g_values, batch_size * sizeof(int));
    cudaMalloc(&d_success, batch_size * sizeof(int));

    // Copy batch data to device
    cudaMemcpyAsync(d_points, points.data(), batch_size * sizeof(Point),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_g_values, g_values.data(), batch_size * sizeof(int),
        cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_success, 0, batch_size * sizeof(int), stream);

    // Launch kernel
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    optimizedInsertCuckooKernel << <num_blocks, threads_per_block, 0, stream >> > (
        d_points, d_g_values, batch_size, d_table1, d_table2,
        size, max_iterations, d_success);

    // Copy success flags back to host
    std::vector<int> success(batch_size);
    cudaMemcpyAsync(success.data(), d_success, batch_size * sizeof(int),
        cudaMemcpyDeviceToHost, stream);

    // Synchronize to ensure completion
    cudaStreamSynchronize(stream);

    // Check if all insertions succeeded
    bool all_success = true;
    for (int s : success) {
        if (!s) {
            all_success = false;
            break;
        }
    }

    // Free temporary device memory
    cudaFree(d_points);
    cudaFree(d_g_values);
    cudaFree(d_success);

    return all_success;
}

void CuckooHashGPU::insertBatchAsync(const std::vector<Point>& points,
    const std::vector<int>& g_values,
    int threads_per_block) {
    // For async operation, we need to store the points and g-values
    // to ensure they remain valid until the operation completes
    prefetch_points = points;
    prefetch_g_values = g_values;

    int batch_size = points.size();
    if (batch_size == 0) return;

    // Allocate device memory (this will be freed on next batch operation)
    Point* d_points;
    int* d_g_values;
    int* d_success;

    cudaMalloc(&d_points, batch_size * sizeof(Point));
    cudaMalloc(&d_g_values, batch_size * sizeof(int));
    cudaMalloc(&d_success, batch_size * sizeof(int));

    // Copy batch data to device
    cudaMemcpyAsync(d_points, prefetch_points.data(), batch_size * sizeof(Point),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_g_values, prefetch_g_values.data(), batch_size * sizeof(int),
        cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_success, 0, batch_size * sizeof(int), stream);

    // Launch kernel
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    optimizedInsertCuckooKernel << <num_blocks, threads_per_block, 0, stream >> > (
        d_points, d_g_values, batch_size, d_table1, d_table2,
        size, max_iterations, d_success);

    // Note: No synchronization here for fully asynchronous operation

    // Schedule cleanup for after kernel completion
    cudaFree(d_points);
    cudaFree(d_g_values);
    cudaFree(d_success);
}

void CuckooHashGPU::findBatch(const std::vector<Point>& points, std::vector<bool>& found,
    std::vector<int>& g_values, int threads_per_block) {
    int batch_size = points.size();
    if (batch_size == 0) return;

    // Allocate device memory
    Point* d_points;
    int* d_found;
    int* d_g_values;

    cudaMalloc(&d_points, batch_size * sizeof(Point));
    cudaMalloc(&d_found, batch_size * sizeof(int));
    cudaMalloc(&d_g_values, batch_size * sizeof(int));

    // Copy points to device
    cudaMemcpyAsync(d_points, points.data(), batch_size * sizeof(Point),
        cudaMemcpyHostToDevice, stream);

    // Launch lookup kernel
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    optimizedFindCuckooKernel << <num_blocks, threads_per_block, 0, stream >> > (
        d_points, batch_size, d_table1, d_table2, size, d_found, d_g_values);

    // Prepare output vectors
    std::vector<int> found_int(batch_size);
    g_values.resize(batch_size);

    // Copy results back to host
    cudaMemcpyAsync(found_int.data(), d_found, batch_size * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(g_values.data(), d_g_values, batch_size * sizeof(int),
        cudaMemcpyDeviceToHost, stream);

    // Synchronize to ensure completion
    cudaStreamSynchronize(stream);

    // Convert int found flags to bool
    found.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        found[i] = (found_int[i] != 0);
    }

    // Free temporary device memory
    cudaFree(d_points);
    cudaFree(d_found);
    cudaFree(d_g_values);
}

void CuckooHashGPU::findBatchAsync(const std::vector<Point>& points, std::vector<bool>& found,
    std::vector<int>& g_values, int threads_per_block) {
    // Store points for async operation
    prefetch_points = points;

    int batch_size = points.size();
    if (batch_size == 0) return;

    // Pre-allocate output vectors
    found.resize(batch_size);
    g_values.resize(batch_size);

    // Allocate device memory
    Point* d_points;
    int* d_found;
    int* d_g_values;

    cudaMalloc(&d_points, batch_size * sizeof(Point));
    cudaMalloc(&d_found, batch_size * sizeof(int));
    cudaMalloc(&d_g_values, batch_size * sizeof(int));

    // Copy points to device
    cudaMemcpyAsync(d_points, prefetch_points.data(), batch_size * sizeof(Point),
        cudaMemcpyHostToDevice, stream);

    // Launch lookup kernel
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    optimizedFindCuckooKernel << <num_blocks, threads_per_block, 0, stream >> > (
        d_points, batch_size, d_table1, d_table2, size, d_found, d_g_values);

    // Note: Results will need to be retrieved with a subsequent call to synchronize()
    // and then manually reading the output

    // Schedule cleanup
    cudaFree(d_points);
    cudaFree(d_found);
    cudaFree(d_g_values);
}

void CuckooHashGPU::clear() {
    // Reset both hash tables to empty state
    cudaMemsetAsync(d_table1, 0, size * sizeof(Entry), stream);
    cudaMemsetAsync(d_table2, 0, size * sizeof(Entry), stream);
    cudaStreamSynchronize(stream);
}

void CuckooHashGPU::synchronize() {
    cudaStreamSynchronize(stream);
}

float CuckooHashGPU::getLoadFactor() const {
    // Count valid entries in both tables (approximate)
    Entry* h_sample1 = new Entry[1024];
    Entry* h_sample2 = new Entry[1024];

    // Sample portions of the tables
    size_t sample_size = std::min(size, 1024u);
    cudaMemcpy(h_sample1, d_table1, sample_size * sizeof(Entry), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sample2, d_table2, sample_size * sizeof(Entry), cudaMemcpyDeviceToHost);

    // Count valid entries in samples
    int valid_count = 0;
    for (size_t i = 0; i < sample_size; i++) {
        valid_count += h_sample1[i].valid;
        valid_count += h_sample2[i].valid;
    }

    // Extrapolate to full tables
    float estimated_load = (float)valid_count / (2.0f * sample_size);

    delete[] h_sample1;
    delete[] h_sample2;

    return estimated_load;
}