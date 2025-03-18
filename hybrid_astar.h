#pragma once
#include "common.h"
#include <vector>
#include <queue>
#include <unordered_map>

// A node in the search space
struct Node {
    int x, y;       // Coordinates
    int g;          // Cost from start to current node
    float f;        // Estimated total cost (g + h)
    int parent_idx; // Index of parent node (-1 for start node)

    Node() : x(0), y(0), g(0), f(0.0f), parent_idx(-1) {}
    
    Node(int x, int y, int g, float f, int parent) 
        : x(x), y(y), g(g), f(f), parent_idx(parent) {}
    
    // For priority queue comparison
    bool operator>(const Node& other) const {
        return f > other.f;
    }
};

// Class that implements the hybrid A* pathfinding
class HybridAStar {
private:
    Grid grid;
    int num_queues;  // Number of parallel priority queues
    int threads_per_block; // CUDA threads per block
    
    // Helper methods
    int getIndex(int x, int y) const;
    float calculateHeuristic(int x, int y, const Point& goal) const;
    std::vector<Point> reconstructPath(const std::vector<Node>& nodes, int goal_idx) const;
    void processBatchGPU(const std::vector<Node>& expanded_nodes, 
                         const Point& goal, 
                         std::vector<std::priority_queue<Node, std::vector<Node>, std::greater<Node>>>& open_queues,const std::unordered_map<Point, Node, PointHash>& closed_map);

public:
    HybridAStar(const Grid& g, int queues = 32, int threads = 256);
    std::vector<Point> findPath(const Point& start, const Point& goal);
};