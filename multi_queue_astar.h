#pragma once
#include "common.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <limits>

// CPU A* algorithm node structure
struct MQNode {
    Point pos;      // Current position coordinates
    int g;          // Accumulated path cost from start
    float f;        // Estimated total cost (g + h)
    Point parent;   // Parent node position for path reconstruction

    MQNode() : pos({ -1,-1 }), g(0), f(0.0f), parent({ -1,-1 }) {}

    MQNode(Point p, int g_val, float f_val, Point par)
        : pos(p), g(g_val), f(f_val), parent(par) {}

    // Comparison operator for priority queue (min-heap)
    bool operator>(const MQNode& other) const {
        return f > other.f;
    }
};

// Priority queue definition for the open set
typedef std::priority_queue<MQNode, std::vector<MQNode>, std::greater<MQNode>> PriorityQueue;

// Multi-Queue A* implementation with CPU parallelism
class MultiQueueAStar {
private:
    Grid grid;                  // Search space grid
    int num_queues;             // Number of parallel priority queues
    size_t nodes_expanded = 0;  // Performance tracking counter

    // Compute admissible heuristic estimate
    float heuristic(const Point& p, const Point& goal);

    // Retrieve cell weight with boundary checking
    int getWeight(const Point& p);

    // Find minimum f-value across all queues
    float getMinFValue(const std::vector<PriorityQueue>& open_queues);

    // Determine queue assignment via spatial hashing
    int getQueueIndex(const Point& p);

public:
    // Constructor with grid specification and parallelism level
    MultiQueueAStar(const Grid& g, int queues = 8);

    // Execute A* search algorithm
    std::vector<Point> findPath(const Point& start, const Point& goal);

    // Performance metric accessor
    size_t getNodesExpanded() const {
        return nodes_expanded;
    }
};

// Utility functions
float distance(const Point& a, const Point& b);
void printPath(const std::vector<Point>& path);