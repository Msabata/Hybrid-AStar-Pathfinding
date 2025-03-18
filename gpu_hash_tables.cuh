#pragma once
#include "common.h"
#include <vector>
#include <cuda_runtime.h>

// Complete class declarations first
class CuckooHashGPU {
public:
    struct Entry {
        int x, y;
        int g;
        int valid;

        __host__ __device__ Entry() : x(0), y(0), g(0), valid(0) {}
    };

private:
    Entry* d_table1;
    Entry* d_table2;
    unsigned int size;
    unsigned int max_iterations;

public:
    CuckooHashGPU(unsigned int table_size, unsigned int max_iter = 20);
    ~CuckooHashGPU();

    bool insertBatch(const std::vector<Point>& points, const std::vector<int>& g_values,
        int threads_per_block = 256);

    void findBatch(const std::vector<Point>& points, std::vector<bool>& found,
        std::vector<int>& g_values, int threads_per_block = 256);
};

class HashWithReplacementGPU {
public:
    struct Entry {
        int x, y;
        int g;
        int valid;
    };

private:
    Entry* d_table;
    unsigned int size;

public:
    HashWithReplacementGPU(unsigned int table_size);
    ~HashWithReplacementGPU();

    bool insertBatch(const std::vector<Point>& points, const std::vector<int>& g_values,
        int threads_per_block = 256);

    void findBatch(const std::vector<Point>& points, std::vector<bool>& found,
        std::vector<int>& g_values, int threads_per_block = 256);
};

// Kernel declarations after class definitions
__global__ void insertCuckooKernel(Point* d_points, int* d_g_values, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int max_iterations, int* d_success);

__global__ void findCuckooKernel(Point* d_points, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int* d_found, int* d_g_values);