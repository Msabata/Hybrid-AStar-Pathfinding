#pragma once
#include "common.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <cuda_runtime.h>
#include <limits>

// Optimization configuration constants
constexpr int GPU_MIN_BATCH_THRESHOLD = 64;    // Minimum batch size for GPU processing
constexpr int GPU_PREFERRED_BATCH_SIZE = 256;  // Ideal batch size for optimal GPU utilization
constexpr int INITIAL_BUFFER_CAPACITY = 4096;  // Initial GPU memory buffer allocation

// Memory-efficient node representation for A* search
struct CompactNode {
    int x, y;           // Spatial coordinates
    int g;              // Accumulated path cost
    float f;            // Total estimated cost (g + h)
    int parent_idx;     // Reference to parent node (-1 for root)

    CompactNode() : x(0), y(0), g(0), f(0.0f), parent_idx(-1) {}

    CompactNode(int _x, int _y, int _g, float _f, int _parent)
        : x(_x), y(_y), g(_g), f(_f), parent_idx(_parent) {}

    // Min-heap ordering for priority queue
    bool operator>(const CompactNode& other) const {
        return f > other.f;
    }
};

// GPU-accelerated A* implementation with CPU fallback
class HybridAStar {
private:
    // Search space representation
    Grid grid;

    // Parallelization parameters
    int num_queues;              // Number of parallel priority queues
    int threads_per_block;       // CUDA thread block size

    // Performance tracking
    size_t nodes_expanded = 0;   // Total nodes processed
    int all_nodes_next_idx = 0;  // Next available index in node array

    // Goal detection state
    bool found_goal = false;                // Flag for GPU-side goal detection
    CompactNode best_goal_node;             // Best goal node found so far
    int best_goal_node_idx = -1;            // Index of best goal node

    // GPU resource management
    bool gpu_initialized = false;
    cudaStream_t compute_stream;    // Stream for kernel execution
    cudaStream_t transfer_stream;   // Stream for memory transfers

    // Device memory buffers with persistent allocation
    int* d_x_buffer = nullptr;         // X-coordinates
    int* d_y_buffer = nullptr;         // Y-coordinates
    int* d_g_buffer = nullptr;         // G-values (path costs)
    int* d_parent_buffer = nullptr;    // Parent indices
    float* d_h_buffer = nullptr;       // Heuristic values
    int* d_valid_buffer = nullptr;     // Node validity flags
    int* d_is_goal_buffer = nullptr;   // Goal state flags
    int* d_grid_weights = nullptr;     // Grid terrain weights
    size_t buffer_capacity = 0;        // Current buffer size

    // GPU utilization tracking for adaptive processing
    float gpu_utilization_estimate = 0.0f;

    // Grid coordinate to linear index conversion
    int getIndex(int x, int y) const;

    // Admissible heuristic calculation (diagonal distance)
    float calculateHeuristic(int x, int y, const Point& goal) const;

    // Path reconstruction from closed set
    std::vector<Point> reconstructPath(const std::vector<CompactNode>& nodes, int goal_idx) const;

    // GPU memory management
    void initializeGPUResources();
    void ensureBufferCapacity(size_t required_size);
    void releaseGPUResources();

    // Batch processing methods
    void processBatchGPU(
        const std::vector<CompactNode>& neighbors,
        const Point& goal,
        std::vector<std::priority_queue<CompactNode, std::vector<CompactNode>,
        std::greater<CompactNode>>>& open_queues,
        const std::unordered_map<Point, CompactNode, PointHash>& closed_map);

    void processBatchCPU(
        const std::vector<CompactNode>& neighbors,
        const Point& goal,
        std::vector<std::priority_queue<CompactNode, std::vector<CompactNode>,
        std::greater<CompactNode>>>& open_queues,
        const std::unordered_map<Point, CompactNode, PointHash>& closed_map);

    // CPU fallback implementation
    std::vector<Point> fallbackCPUSearch(const Point& start, const Point& goal);

    // Adaptive processing strategy
    bool shouldProcessOnGPU(size_t batch_size) const;
    float estimateGPUUtilization() const;
    void updateGPUUtilizationEstimate(float new_estimate);

public:
    // Constructor with search space and parallelization parameters
    HybridAStar(const Grid& g, int queues = 32, int threads = 256);

    // Resource cleanup
    ~HybridAStar();

    // Main search algorithm
    std::vector<Point> findPath(const Point& start, const Point& goal);

    // Performance metric accessor
    size_t getNodesExpanded() const {
        return nodes_expanded;
    }
};