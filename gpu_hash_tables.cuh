#pragma once
#include "common.h"
#include <vector>
#include <cuda_runtime.h>

/**
 * GPU-optimized Cuckoo hash table implementation
 * Provides O(1) expected lookup and insertion operations
 * with conflict resolution via displacement
 */
class CuckooHashGPU {
public:
    /**
     * Hash table entry with memory-aligned layout
     * Optimized for 32-bit word alignment
     */
    struct Entry {
        int x, y;    // Coordinate values
        int g;       // Path cost
        int valid;   // Validity flag (0 = invalid, 1 = valid)

        __host__ __device__ Entry() : x(0), y(0), g(0), valid(0) {}
    };

private:
    Entry* d_table1;          // Primary hash table
    Entry* d_table2;          // Secondary hash table for collision resolution
    unsigned int size;        // Hash table capacity
    unsigned int max_iterations; // Maximum displacement iterations
    cudaStream_t stream;      // Associated CUDA stream

    // Prefetched host side buffers for efficient batch operations
    std::vector<Point> prefetch_points;
    std::vector<int> prefetch_g_values;
    bool has_stream_ownership;

public:
    /**
     * Constructor with capacity specification
     *
     * @param table_size Hash table size (should be prime for best distribution)
     * @param max_iter Maximum number of cuckoo displacement iterations
     * @param cuda_stream CUDA stream for operations (nullptr uses default stream)
     */
    CuckooHashGPU(unsigned int table_size, unsigned int max_iter = 20,
        cudaStream_t cuda_stream = nullptr);

    /**
     * Destructor for resource cleanup
     */
    ~CuckooHashGPU();

    /**
     * Batch insert operation with efficient batching
     *
     * @param points Vector of points to insert
     * @param g_values Corresponding g-values
     * @param threads_per_block Thread block size for CUDA execution
     * @return True if all insertions successful, false otherwise
     */
    bool insertBatch(const std::vector<Point>& points, const std::vector<int>& g_values,
        int threads_per_block = 256);

    /**
     * Asynchronous batch insert operation
     * Returns immediately without waiting for completion
     *
     * @param points Vector of points to insert
     * @param g_values Corresponding g-values
     * @param threads_per_block Thread block size
     */
    void insertBatchAsync(const std::vector<Point>& points, const std::vector<int>& g_values,
        int threads_per_block = 256);

    /**
     * Batch query operation
     *
     * @param points Vector of points to look up
     * @param found Output vector indicating if each point was found
     * @param g_values Output vector of g-values (INT_MAX if not found)
     * @param threads_per_block Thread block size
     */
    void findBatch(const std::vector<Point>& points, std::vector<bool>& found,
        std::vector<int>& g_values, int threads_per_block = 256);

    /**
     * Asynchronous batch query
     *
     * @param points Vector of points to look up
     * @param found Output vector indicating if each point was found
     * @param g_values Output vector of g-values (INT_MAX if not found)
     * @param threads_per_block Thread block size
     */
    void findBatchAsync(const std::vector<Point>& points, std::vector<bool>& found,
        std::vector<int>& g_values, int threads_per_block = 256);

    /**
     * Clear all entries from the hash table
     */
    void clear();

    /**
     * Wait for completion of asynchronous operations
     */
    void synchronize();

    /**
     * Get current load factor of the hash table
     *
     * @return Ratio of filled slots to total capacity
     */
    float getLoadFactor() const;
};

/**
 * Memory-efficient hash table using linear probing
 * Suitable for closed sets with less stringent collision requirements
 */
class LinearProbingHashGPU {
public:
    /**
     * Hash table entry with compact representation
     */
    struct Entry {
        int x, y;
        int g;
        int valid;
    };

private:
    Entry* d_table;           // Device hash table
    unsigned int size;        // Table capacity
    unsigned int probe_limit; // Maximum linear probing distance
    cudaStream_t stream;      // Associated CUDA stream
    bool has_stream_ownership;

public:
    /**
     * Constructor
     *
     * @param table_size Hash table capacity
     * @param max_probe Maximum probe distance (default: 20)
     * @param cuda_stream CUDA stream for operations
     */
    LinearProbingHashGPU(unsigned int table_size, unsigned int max_probe = 20,
        cudaStream_t cuda_stream = nullptr);

    /**
     * Destructor
     */
    ~LinearProbingHashGPU();

    /**
     * Batch insert with minimum-g replacement policy
     *
     * @param points Points to insert
     * @param g_values Corresponding g-values
     * @param threads_per_block Thread block size
     * @return True if all insertions successful
     */
    bool insertBatch(const std::vector<Point>& points, const std::vector<int>& g_values,
        int threads_per_block = 256);

    /**
     * Batch lookup operation
     *
     * @param points Points to query
     * @param found Output flags for found status
     * @param g_values Output g-values
     * @param threads_per_block Thread block size
     */
    void findBatch(const std::vector<Point>& points, std::vector<bool>& found,
        std::vector<int>& g_values, int threads_per_block = 256);

    /**
     * Clear all entries
     */
    void clear();

    /**
     * Wait for completion of operations
     */
    void synchronize();
};

// Fused GPU kernel for heuristic calculation and node validation
__global__ void fusedHeuristicAndValidationKernel(
    int* d_x, int* d_y, int* d_g, int size,
    int goal_x, int goal_y, int width, int height,
    int* d_weights, float* d_h, int* d_valid);

// Heuristic-only kernel for when validation is handled separately
__global__ void optimizedHeuristicKernel(
    int* d_x, int* d_y, int size, int goal_x, int goal_y, float* d_h);

// Optimized cuckoo hash insert kernel with atomic operations
__global__ void optimizedInsertCuckooKernel(
    Point* d_points, int* d_g_values, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int max_iterations, int* d_success);

// Efficient lookup kernel with early termination
__global__ void optimizedFindCuckooKernel(
    Point* d_points, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int* d_found, int* d_g_values);