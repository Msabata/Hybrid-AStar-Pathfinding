#include "hybrid_astar.h"
#include "multi_queue_astar.h"
#include <iostream>
#include <string>
#include <chrono>
#include <functional> // Add this line
#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>

// Function to print the grid with the path overlay
void printGridWithPath(const Grid& grid, const std::vector<Point>& path, const Point& start, const Point& goal) {
    // Create a 2D vector to represent the grid characters
    std::vector<std::vector<char>> display(grid.height, std::vector<char>(grid.width, '.'));

    // Mark walls in the grid (assuming a weight of -1 represents a wall)
    for (int y = 0; y < grid.height; y++) {
        for (int x = 0; x < grid.width; x++) {
            if (grid.weights[y * grid.width + x] == -1)
                display[y][x] = '#';
        }
    }

    // Mark the path on the grid
    for (const auto& p : path) {
        display[p.y][p.x] = 'X';
    }

    // Mark start and goal positions explicitly
    display[start.y][start.x] = 'S';
    display[goal.y][goal.x] = 'G';

    // Print the grid row by row
    for (int y = 0; y < grid.height; y++) {
        for (int x = 0; x < grid.width; x++) {
            std::cout << display[y][x] << " ";
        }
        std::cout << "\n";
    }
}

// Example main() function using printGridWithPath
int main(int argc, char** argv) {
    // Default parameters
    int width = 100;
    int height = 100;
    int seed = 42;
    int num_queues = 32;
    bool use_hybrid = true;

    // Parse command line arguments if provided
    if (argc > 1) width = std::atoi(argv[1]);
    if (argc > 2) height = std::atoi(argv[2]);
    if (argc > 3) seed = std::atoi(argv[3]);
    if (argc > 4) num_queues = std::atoi(argv[4]);
    if (argc > 5) use_hybrid = (std::atoi(argv[5]) != 0);

    // Set number of OpenMP threads
    int max_threads = omp_get_max_threads();
    std::cout << "Using up to " << max_threads << " OpenMP threads" << std::endl;

    // Generate grid
    std::cout << "Generating " << width << "x" << height << " grid with seed " << seed << std::endl;
    Grid grid = grid_generator(width, height, seed);

    // Define start and goal
    Point start = { 0, 0 };
    Point goal = { width - 1, height - 1 };

    std::cout << "Finding path from (" << start.x << ", " << start.y << ") to ("
        << goal.x << ", " << goal.y << ") using " << num_queues << " queues" << std::endl;

    // Run pathfinding
    std::vector<Point> path;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (use_hybrid) {
        std::cout << "Using hybrid GPU-CPU A* implementation" << std::endl;
        HybridAStar astar(grid, num_queues);
        path = astar.findPath(start, goal);
    }
    else {
        std::cout << "Using multi-queue CPU A* implementation" << std::endl;
        MultiQueueAStar astar(grid, num_queues);
        path = astar.findPath(start, goal);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Print path information
    if (path.empty()) {
        std::cout << "No path found!" << std::endl;
    }
    else {
        std::cout << "Path found with " << path.size() << " steps in " << duration << " ms" << std::endl;

        // Print first few and last few steps
        const int max_print = 5;
        std::cout << "Path: ";
        for (int i = 0; i < std::min(max_print, (int)path.size()); i++) {
            std::cout << "(" << path[i].x << "," << path[i].y << ") ";
        }
        if (path.size() > max_print * 2) {
            std::cout << "... ";
        }
        if (path.size() > max_print) {
            for (int i = std::max(max_print, (int)path.size() - max_print); i < path.size(); i++) {
                std::cout << "(" << path[i].x << "," << path[i].y << ") ";
            }
        }
        std::cout << std::endl;

        // Print the entire grid with the path overlay
        printGridWithPath(grid, path, start, goal);
    }

    // Clean up
    delete[] grid.weights;

    return 0;
}
