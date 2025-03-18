#pragma once
#include "common.h"
#include <vector>
#include <queue>
#include <unordered_map>

// Node in the A* search with necessary attributes
struct AStarNode {
    Point pos;
    int g;
    float f;
    Point parent;

    AStarNode() : pos({ -1,-1 }), g(0), f(0.0f), parent({ -1,-1 }) {} 

    AStarNode(Point p, int g_val, float f_val, Point par)  // Existing constructor
        : pos(p), g(g_val), f(f_val), parent(par) {}
    
    // For the priority queue to get min f value
    bool operator>(const AStarNode& other) const {
        return f > other.f;
    }
};

// A single priority queue for the open list
typedef std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> PriorityQueue;

// Class to implement multi-queue A* search with OpenMP parallelization
class MultiQueueAStar {
private:
    Grid grid;
    int num_queues;

    // Heuristic function (diagonal distance)
    float heuristic(const Point& p, const Point& goal);
    
    // Get weight of a cell
    int getWeight(const Point& p);
    
    // Get minimum f value among all queues
    float getMinFValue(const std::vector<PriorityQueue>& open_queues);
    
    // Determine which queue to use based on node coordinates
    int getQueueIndex(const Point& p);

public:
    MultiQueueAStar(const Grid& g, int queues = 8);
    
    // Find path from start to goal
    std::vector<Point> findPath(const Point& start, const Point& goal);
};

// Utility functions
float distance(const Point& a, const Point& b);
void printPath(const std::vector<Point>& path);