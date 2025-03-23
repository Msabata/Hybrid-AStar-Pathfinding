#pragma once
#include <cstddef>
#include <functional>

// Grid and Point structures as defined in the problem
typedef struct {
    int width;
    int height;
    int* weights;  // Weight of each cell, -1 indicates an impassable wall
} Grid;

// Point structure representing 2D coordinates
struct Point {
    int x;
    int y;

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// Hash function for Point to use in unordered_map
struct PointHash {
    std::size_t operator()(const Point& p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
};

// Direction vectors for 8-directional movement
extern const int dx[8];
extern const int dy[8];
extern const int costs[8];

// Grid generator function declaration
Grid grid_generator(int width, int height, int seed);