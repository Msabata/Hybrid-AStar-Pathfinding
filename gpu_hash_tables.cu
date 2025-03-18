#include "gpu_hash_tables.cuh"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <iostream>
#include <climits>

const unsigned int HASH_SEED1 = 0x1234ABCD;
const unsigned int HASH_SEED2 = 0xA1B2C3D4;

__host__ __device__ unsigned int hashPoint(const Point& p, unsigned int seed, unsigned int table_size) {
    unsigned int hash = seed;
    hash ^= (p.x * 73856093) ^ (p.y * 19349663);
    return hash % table_size;
}

__global__ void insertCuckooKernel(Point* d_points, int* d_g_values, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int max_iterations, int* d_success) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        Point p = d_points[idx];
        int g = d_g_values[idx];

        unsigned int pos1 = hashPoint(p, HASH_SEED1, table_size);
        if (atomicCAS(&d_table1[pos1].valid, 0, 1) == 0) {
            d_table1[pos1].x = p.x;
            d_table1[pos1].y = p.y;
            d_table1[pos1].g = g;
            d_success[idx] = 1;
            return;
        }
        else if (d_table1[pos1].x == p.x && d_table1[pos1].y == p.y) {
            atomicMin(&d_table1[pos1].g, g);
            d_success[idx] = 1;
            return;
        }

        unsigned int pos2 = hashPoint(p, HASH_SEED2, table_size);
        if (atomicCAS(&d_table2[pos2].valid, 0, 1) == 0) {
            d_table2[pos2].x = p.x;
            d_table2[pos2].y = p.y;
            d_table2[pos2].g = g;
            d_success[idx] = 1;
            return;
        }
        else if (d_table2[pos2].x == p.x && d_table2[pos2].y == p.y) {
            atomicMin(&d_table2[pos2].g, g);
            d_success[idx] = 1;
            return;
        }

        d_success[idx] = 0;
    }
}

__global__ void findCuckooKernel(Point* d_points, int batch_size,
    CuckooHashGPU::Entry* d_table1, CuckooHashGPU::Entry* d_table2,
    unsigned int table_size, int* d_found, int* d_g_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        Point p = d_points[idx];

        unsigned int pos1 = hashPoint(p, HASH_SEED1, table_size);
        if (d_table1[pos1].valid && d_table1[pos1].x == p.x && d_table1[pos1].y == p.y) {
            d_found[idx] = 1;
            d_g_values[idx] = d_table1[pos1].g;
            return;
        }

        unsigned int pos2 = hashPoint(p, HASH_SEED2, table_size);
        if (d_table2[pos2].valid && d_table2[pos2].x == p.x && d_table2[pos2].y == p.y) {
            d_found[idx] = 1;
            d_g_values[idx] = d_table2[pos2].g;
            return;
        }

        d_found[idx] = 0;
        d_g_values[idx] = INT_MAX;
    }
}

CuckooHashGPU::CuckooHashGPU(unsigned int table_size, unsigned int max_iter)
    : size(table_size), max_iterations(max_iter) {
    cudaMalloc(&d_table1, size * sizeof(Entry));
    cudaMalloc(&d_table2, size * sizeof(Entry));
    cudaMemset(d_table1, 0, size * sizeof(Entry));
    cudaMemset(d_table2, 0, size * sizeof(Entry));
}

CuckooHashGPU::~CuckooHashGPU() {
    cudaFree(d_table1);
    cudaFree(d_table2);
}

bool CuckooHashGPU::insertBatch(const std::vector<Point>& points, const std::vector<int>& g_values, int threads_per_block) {
    int batch_size = points.size();
    if (batch_size == 0) return true;

    Point* d_points;
    int* d_g_values;
    int* d_success;

    cudaMalloc(&d_points, batch_size * sizeof(Point));
    cudaMalloc(&d_g_values, batch_size * sizeof(int));
    cudaMalloc(&d_success, batch_size * sizeof(int));

    cudaMemcpy(d_points, points.data(), batch_size * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_values, g_values.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_success, 0, batch_size * sizeof(int));

    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    insertCuckooKernel << <num_blocks, threads_per_block >> > (d_points, d_g_values, batch_size, d_table1, d_table2, size, max_iterations, d_success);

    std::vector<int> success(batch_size);
    cudaMemcpy(success.data(), d_success, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    bool all_success = true;
    for (int s : success) {
        if (!s) {
            all_success = false;
            break;
        }
    }

    cudaFree(d_points);
    cudaFree(d_g_values);
    cudaFree(d_success);

    return all_success;
}

void CuckooHashGPU::findBatch(const std::vector<Point>& points, std::vector<bool>& found, std::vector<int>& g_values, int threads_per_block) {
    int batch_size = points.size();
    if (batch_size == 0) return;

    Point* d_points;
    int* d_found;
    int* d_g_values;

    cudaMalloc(&d_points, batch_size * sizeof(Point));
    cudaMalloc(&d_found, batch_size * sizeof(int));
    cudaMalloc(&d_g_values, batch_size * sizeof(int));

    cudaMemcpy(d_points, points.data(), batch_size * sizeof(Point), cudaMemcpyHostToDevice);

    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    findCuckooKernel << <num_blocks, threads_per_block >> > (d_points, batch_size, d_table1, d_table2, size, d_found, d_g_values);

    std::vector<int> found_int(batch_size);
    g_values.resize(batch_size);
    cudaMemcpy(found_int.data(), d_found, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g_values.data(), d_g_values, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    found.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        found[i] = (found_int[i] != 0);
    }

    cudaFree(d_points);
    cudaFree(d_found);
    cudaFree(d_g_values);
}

// CUDA kernels for heuristic calculation and node filtering
// Note: These would typically be in a .cu file, but are included here for simplicity
extern "C" {

    __global__ void computeHeuristicKernel(int* d_x, int* d_y, int size, int goal_x, int goal_y, float* d_h) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            int dx = abs(d_x[idx] - goal_x);
            int dy = abs(d_y[idx] - goal_y);
            // For example, using Euclidean distance
            d_h[idx] = sqrtf(float(dx * dx + dy * dy));
        }
    }

    void computeHeuristicWrapper(int* d_x, int* d_y, int size, int goal_x, int goal_y, float* d_h, int threads_per_block) {
        int blocks = (size + threads_per_block - 1) / threads_per_block;
        computeHeuristicKernel << <blocks, threads_per_block >> > (d_x, d_y, size, goal_x, goal_y, d_h);
        cudaDeviceSynchronize();
    }

    __global__ void filterNodesKernel(int* d_x, int* d_y, int* d_g, int size, int width, int height, int* d_weights, int* d_valid,
        int* d_closed_x, int* d_closed_y, int* d_closed_g, int closed_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // A simple filtering: mark valid if within bounds and not a wall
            int x = d_x[idx];
            int y = d_y[idx];
            if (x >= 0 && x < width && y >= 0 && y < height && d_weights[y * width + x] != -1)
                d_valid[idx] = 1;
            else
                d_valid[idx] = 0;
        }
    }

    void filterNodesWrapper(int* d_x, int* d_y, int* d_g, int size, int width, int height, int* d_weights,
        int* d_valid, int* d_closed_x, int* d_closed_y, int* d_closed_g, int closed_size, int threads_per_block) {
        int blocks = (size + threads_per_block - 1) / threads_per_block;
        filterNodesKernel << <blocks, threads_per_block >> > (d_x, d_y, d_g, size, width, height, d_weights, d_valid, d_closed_x, d_closed_y, d_closed_g, closed_size);
        cudaDeviceSynchronize();
    }

} // extern "C"

