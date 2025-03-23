#include "hybrid_astar.h"
#include "multi_queue_astar.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <functional>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <random>

// Structure to hold benchmark results
struct BenchmarkResult {
    std::chrono::milliseconds execution_time;
    size_t path_length;
    float path_cost;
    size_t nodes_expanded;
    bool path_found;
};

// Function to print the grid with path overlay
void printGridWithPath(const Grid& grid, const std::vector<Point>& path, const Point& start, const Point& goal) {
    // Only print for small grids (maximum size 50x50)
    if (grid.width > 50 || grid.height > 50) {
        std::cout << "Grid too large to display (" << grid.width << "x" << grid.height << ")" << std::endl;
        return;
    }

    // Create a 2D vector to represent the grid characters
    std::vector<std::vector<char>> display(grid.height, std::vector<char>(grid.width, '.'));

    // Safety check for grid weights
    if (!grid.weights) {
        std::cerr << "Error: Grid weights array is null" << std::endl;
        return;
    }

    // Mark walls in the grid (assuming a weight of -1 represents a wall)
    for (int y = 0; y < grid.height; y++) {
        for (int x = 0; x < grid.width; x++) {
            int index = y * grid.width + x;
            if (index >= 0 && index < grid.width * grid.height && grid.weights[index] == -1)
                display[y][x] = '#';
        }
    }

    // Mark the path on the grid
    for (const auto& p : path) {
        // Safety check for point coordinates
        if (p.x >= 0 && p.x < grid.width && p.y >= 0 && p.y < grid.height)
            display[p.y][p.x] = 'X';
    }

    // Mark start and goal positions explicitly
    if (start.x >= 0 && start.x < grid.width && start.y >= 0 && start.y < grid.height)
        display[start.y][start.x] = 'S';
    if (goal.x >= 0 && goal.x < grid.width && goal.y >= 0 && goal.y < grid.height)
        display[goal.y][goal.x] = 'G';

    // Print the grid row by row
    std::cout << "Grid visualization (S=start, G=goal, X=path, #=wall, .=open):" << std::endl;
    for (int y = 0; y < grid.height; y++) {
        for (int x = 0; x < grid.width; x++) {
            std::cout << display[y][x] << " ";
        }
        std::cout << "\n";
    }
}

// Calculate path cost considering movement and terrain costs
float calculatePathCost(const std::vector<Point>& path, const Grid& grid) {
    if (path.empty() || path.size() == 1) return 0.0f;

    float total_cost = 0.0f;

    for (size_t i = 0; i < path.size() - 1; i++) {
        const Point& p1 = path[i];
        const Point& p2 = path[i + 1];

        // Determine movement type (cardinal or diagonal)
        int dx = std::abs(p2.x - p1.x);
        int dy = std::abs(p2.y - p1.y);
        int movement_cost = (dx + dy == 2) ? 14 : 10; // 14 for diagonal, 10 for cardinal

        // Apply terrain cost
        int terrain_cost = 1; // Default
        if (p2.x >= 0 && p2.x < grid.width && p2.y >= 0 && p2.y < grid.height) {
            int idx = p2.y * grid.width + p2.x;
            terrain_cost = std::max(1, grid.weights[idx]);
        }

        total_cost += movement_cost * terrain_cost / 10.0f;
    }

    return total_cost;
}

// Function to run a single pathfinding test and return benchmark results
BenchmarkResult runPathfindingTest(const Grid& grid, const Point& start, const Point& goal, int num_queues, bool use_hybrid) {
    BenchmarkResult result;
    result.path_found = false;

    std::vector<Point> path;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (use_hybrid) {
        HybridAStar astar(grid, num_queues);
        path = astar.findPath(start, goal);
        result.nodes_expanded = astar.getNodesExpanded();
    }
    else {
        MultiQueueAStar astar(grid, num_queues);
        path = astar.findPath(start, goal);
        result.nodes_expanded = astar.getNodesExpanded();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Calculate metrics
    result.path_found = !path.empty();
    result.path_length = path.size();
    result.path_cost = calculatePathCost(path, grid);

    return result;
}

// Function to generate multiple test grids with varying parameters
std::vector<Grid> generateTestGrids(int count, int width, int height, int base_seed, int wall_density_min, int wall_density_max) {
    std::vector<Grid> grids;
    std::uniform_int_distribution<int> density_dist(wall_density_min, wall_density_max);
    std::mt19937 rng(base_seed);

    for (int i = 0; i < count; i++) {
        int seed = base_seed + i;
        int wall_density = density_dist(rng);

        Grid grid;
        grid.width = width;
        grid.height = height;
        grid.weights = new int[width * height];

        // Initialize random generator for this grid
        std::mt19937 grid_rng(seed);
        std::uniform_int_distribution<int> wall_dist(1, 100);
        std::uniform_int_distribution<int> weight_dist(1, 10);

        // Fill with weights and walls
        for (int j = 0; j < width * height; j++) {
            if (wall_dist(grid_rng) <= wall_density) {
                grid.weights[j] = -1;  // Wall
            }
            else {
                grid.weights[j] = weight_dist(grid_rng);  // Random weight 1-10
            }
        }

        // Ensure start and end are not walls
        grid.weights[0] = 1;  // Top-left
        grid.weights[width * height - 1] = 1;  // Bottom-right

        grids.push_back(grid);
    }

    return grids;
}

// Benchmark mode - run multiple tests and compare algorithms
void runBenchmark(int width, int height, int num_grids, int num_queues) {
    std::cout << "\n===== BENCHMARK MODE =====" << std::endl;
    std::cout << "Grid size: " << width << "x" << height << std::endl;
    std::cout << "Number of test grids: " << num_grids << std::endl;
    std::cout << "Priority queues: " << num_queues << std::endl;

    // Generate test grids with varying wall densities (10-30%)
    std::vector<Grid> test_grids = generateTestGrids(num_grids, width, height, 42, 10, 30);

    // Define start and goal for all tests
    Point start = { 0, 0 };
    Point goal = { width - 1, height - 1 };

    // Results storage
    std::vector<BenchmarkResult> hybrid_results;
    std::vector<BenchmarkResult> cpu_results;

    // Run tests for each grid
    for (int i = 0; i < num_grids; i++) {
        std::cout << "\nRunning test grid " << (i + 1) << "/" << num_grids << "..." << std::endl;

        // Run hybrid GPU algorithm
        std::cout << "Testing Hybrid GPU-CPU A*..." << std::endl;
        BenchmarkResult hybrid_result = runPathfindingTest(test_grids[i], start, goal, num_queues, true);
        hybrid_results.push_back(hybrid_result);

        // Run CPU-only algorithm
        std::cout << "Testing Multi-Queue CPU A*..." << std::endl;
        BenchmarkResult cpu_result = runPathfindingTest(test_grids[i], start, goal, num_queues, false);
        cpu_results.push_back(cpu_result);

        // Print individual test results
        std::cout << "Results for test grid " << (i + 1) << ":" << std::endl;
        std::cout << std::left << std::setw(15) << "Algorithm"
            << std::setw(15) << "Time (ms)"
            << std::setw(15) << "Path Length"
            << std::setw(15) << "Path Cost"
            << std::setw(15) << "Nodes Expanded"
            << std::setw(10) << "Success" << std::endl;

        std::cout << std::setw(15) << "Hybrid A*"
            << std::setw(15) << hybrid_result.execution_time.count()
            << std::setw(15) << hybrid_result.path_length
            << std::setw(15) << std::fixed << std::setprecision(2) << hybrid_result.path_cost
            << std::setw(15) << hybrid_result.nodes_expanded
            << std::setw(10) << (hybrid_result.path_found ? "Yes" : "No") << std::endl;

        std::cout << std::setw(15) << "CPU A*"
            << std::setw(15) << cpu_result.execution_time.count()
            << std::setw(15) << cpu_result.path_length
            << std::setw(15) << std::fixed << std::setprecision(2) << cpu_result.path_cost
            << std::setw(15) << cpu_result.nodes_expanded
            << std::setw(10) << (cpu_result.path_found ? "Yes" : "No") << std::endl;
    }

    // Calculate aggregate statistics
    size_t hybrid_success_count = 0;
    size_t cpu_success_count = 0;

    double hybrid_avg_time = 0.0;
    double cpu_avg_time = 0.0;

    double hybrid_avg_path_length = 0.0;
    double cpu_avg_path_length = 0.0;

    double hybrid_avg_path_cost = 0.0;
    double cpu_avg_path_cost = 0.0;

    double hybrid_avg_nodes = 0.0;
    double cpu_avg_nodes = 0.0;

    for (int i = 0; i < num_grids; i++) {
        if (hybrid_results[i].path_found) {
            hybrid_success_count++;
            hybrid_avg_time += hybrid_results[i].execution_time.count();
            hybrid_avg_path_length += hybrid_results[i].path_length;
            hybrid_avg_path_cost += hybrid_results[i].path_cost;
            hybrid_avg_nodes += hybrid_results[i].nodes_expanded;
        }

        if (cpu_results[i].path_found) {
            cpu_success_count++;
            cpu_avg_time += cpu_results[i].execution_time.count();
            cpu_avg_path_length += cpu_results[i].path_length;
            cpu_avg_path_cost += cpu_results[i].path_cost;
            cpu_avg_nodes += cpu_results[i].nodes_expanded;
        }
    }

    // Normalize averages
    if (hybrid_success_count > 0) {
        hybrid_avg_time /= hybrid_success_count;
        hybrid_avg_path_length /= hybrid_success_count;
        hybrid_avg_path_cost /= hybrid_success_count;
        hybrid_avg_nodes /= hybrid_success_count;
    }

    if (cpu_success_count > 0) {
        cpu_avg_time /= cpu_success_count;
        cpu_avg_path_length /= cpu_success_count;
        cpu_avg_path_cost /= cpu_success_count;
        cpu_avg_nodes /= cpu_success_count;
    }

    // Print aggregate results
    std::cout << "\n===== BENCHMARK SUMMARY =====" << std::endl;
    std::cout << std::left << std::setw(15) << "Algorithm"
        << std::setw(15) << "Success Rate"
        << std::setw(15) << "Avg Time (ms)"
        << std::setw(15) << "Avg Path Len"
        << std::setw(15) << "Avg Path Cost"
        << std::setw(15) << "Avg Nodes Exp" << std::endl;

    std::cout << std::setw(15) << "Hybrid A*"
        << std::setw(15) << std::fixed << std::setprecision(2) << (100.0 * hybrid_success_count / num_grids) << "%"
        << std::setw(15) << std::fixed << std::setprecision(2) << hybrid_avg_time
        << std::setw(15) << std::fixed << std::setprecision(2) << hybrid_avg_path_length
        << std::setw(15) << std::fixed << std::setprecision(2) << hybrid_avg_path_cost
        << std::setw(15) << std::fixed << std::setprecision(2) << hybrid_avg_nodes << std::endl;

    std::cout << std::setw(15) << "CPU A*"
        << std::setw(15) << std::fixed << std::setprecision(2) << (100.0 * cpu_success_count / num_grids) << "%"
        << std::setw(15) << std::fixed << std::setprecision(2) << cpu_avg_time
        << std::setw(15) << std::fixed << std::setprecision(2) << cpu_avg_path_length
        << std::setw(15) << std::fixed << std::setprecision(2) << cpu_avg_path_cost
        << std::setw(15) << std::fixed << std::setprecision(2) << cpu_avg_nodes << std::endl;

    // Calculate speedup if both algorithms succeeded on at least one test
    if (hybrid_success_count > 0 && cpu_success_count > 0) {
        double speedup = cpu_avg_time / hybrid_avg_time;
        std::cout << "\nHybrid A* is " << std::fixed << std::setprecision(2) << speedup << "x "
            << (speedup > 1.0 ? "faster than" : "slower than") << " CPU A*" << std::endl;
    }

    // Clean up memory
    for (auto& grid : test_grids) {
        delete[] grid.weights;
    }
}

// Main function
int main(int argc, char** argv) {
    try {
        // Default parameters
        int width = 1000;
        int height = 1000;
        int seed = 42;
        int num_queues = 32;
        bool use_hybrid = true;
        bool benchmark_mode = true;
        int num_benchmark_grids = 5;

        // Parse command line arguments if provided
        if (argc > 1) width = std::atoi(argv[1]);
        if (argc > 2) height = std::atoi(argv[2]);
        if (argc > 3) seed = std::atoi(argv[3]);
        if (argc > 4) num_queues = std::atoi(argv[4]);
        if (argc > 5) use_hybrid = (std::atoi(argv[5]) != 0);
        if (argc > 6) benchmark_mode = (std::atoi(argv[6]) != 0);
        if (argc > 7) num_benchmark_grids = std::atoi(argv[7]);

        // Input validation
        if (width <= 0 || height <= 0 || num_queues <= 0 || num_benchmark_grids <= 0) {
            std::cerr << "Error: Invalid parameters. Width, height, num_queues, and num_benchmark_grids must be positive." << std::endl;
            return 1;
        }

        // Set number of OpenMP threads
        int max_threads = omp_get_max_threads();
        std::cout << "Using up to " << max_threads << " OpenMP threads" << std::endl;

        // Run benchmark mode if selected
        if (benchmark_mode) {
            runBenchmark(width, height, num_benchmark_grids, num_queues);
            return 0;
        }

        // Regular mode: Run single test
        // Generate grid
        std::cout << "Generating " << width << "x" << height << " grid with seed " << seed << std::endl;
        Grid grid = grid_generator(width, height, seed);

        // Verify grid
        if (!grid.weights) {
            std::cerr << "Error: Failed to allocate grid" << std::endl;
            return 1;
        }

        // Define start and goal
        Point start = { 0, 0 };
        Point goal = { width - 1, height - 1 };

        std::cout << "Finding path from (" << start.x << ", " << start.y << ") to ("
            << goal.x << ", " << goal.y << ") using " << num_queues << " queues" << std::endl;

        // Run pathfinding
        std::vector<Point> path;
        size_t nodes_expanded = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        if (use_hybrid) {
            std::cout << "Using hybrid GPU-CPU A* implementation" << std::endl;
            HybridAStar astar(grid, num_queues);
            path = astar.findPath(start, goal);
            nodes_expanded = astar.getNodesExpanded();
        }
        else {
            std::cout << "Using multi-queue CPU A* implementation" << std::endl;
            MultiQueueAStar astar(grid, num_queues);
            path = astar.findPath(start, goal);
            nodes_expanded = astar.getNodesExpanded();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Print path information
        if (path.empty()) {
            std::cout << "No path found!" << std::endl;
        }
        else {
            std::cout << "Path found with " << path.size() << " steps in " << duration << " ms" << std::endl;
            std::cout << "Nodes expanded: " << nodes_expanded << std::endl;

            // Calculate actual path cost
            float path_cost = calculatePathCost(path, grid);
            std::cout << "Path cost: " << path_cost << std::endl;

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

            // Print the grid with path overlay (only for reasonably sized grids)
            printGridWithPath(grid, path, start, goal);
        }

        // Clean up
        delete[] grid.weights;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}