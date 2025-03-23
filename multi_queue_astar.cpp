#include "multi_queue_astar.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <limits>

// Optimized diagonal distance heuristic implementation
float MultiQueueAStar::heuristic(const Point& p, const Point& goal) {
    int dx = std::abs(p.x - goal.x);
    int dy = std::abs(p.y - goal.y);
    int diag = std::min(dx, dy);
    int straight = dx + dy - 2 * diag;
    return 14.0f * diag + 10.0f * straight;
}

// Boundary-checked grid weight accessor with O(1) complexity
int MultiQueueAStar::getWeight(const Point& p) {
    if (p.x < 0 || p.x >= grid.width || p.y < 0 || p.y >= grid.height)
        return -1; // Out of bounds or wall

    int index = p.y * grid.width + p.x;
    if (index < 0 || index >= grid.width * grid.height) {
        return -1; // Additional safety check for memory access
    }

    return grid.weights[index];
}

// Minimum f-value computation across queues in O(q) time where q = queue count
float MultiQueueAStar::getMinFValue(const std::vector<PriorityQueue>& open_queues) {
    float min_f = std::numeric_limits<float>::max();

    for (const auto& queue : open_queues) {
        if (!queue.empty() && queue.top().f < min_f) {
            min_f = queue.top().f;
        }
    }

    return min_f;
}

// Spatial hashing function for balanced queue distribution
int MultiQueueAStar::getQueueIndex(const Point& p) {
    // MurmurHash-inspired hash function for improved distribution
    unsigned int hash = static_cast<unsigned int>(p.x * 73856093) ^
        static_cast<unsigned int>(p.y * 19349663);
    hash = (hash ^ (hash >> 16)) * 0x85ebca6b;
    hash = (hash ^ (hash >> 13)) * 0xc2b2ae35;
    hash = hash ^ (hash >> 16);
    return static_cast<int>(hash % static_cast<unsigned int>(std::max(1, num_queues)));
}

// Constructor with parameter validation
MultiQueueAStar::MultiQueueAStar(const Grid& g, int queues) : grid(g), num_queues(queues) {
    if (num_queues < 1) {
        std::cout << "Warning: Invalid queue count, defaulting to 1" << std::endl;
        num_queues = 1;
    }
}

// Main A* search implementation with optimal priority enforcement
std::vector<Point> MultiQueueAStar::findPath(const Point& start, const Point& goal) {
    nodes_expanded = 0; // Reset performance counter
    std::vector<Point> path;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Validate start/goal positions
    if (getWeight(start) == -1 || getWeight(goal) == -1) {
        std::cout << "Start or goal position is invalid or unreachable" << std::endl;
        return path;
    }

    try {
        // Initialize priority queues for open set management
        std::vector<PriorityQueue> open_queues(num_queues);

        // Closed set for visited nodes
        std::unordered_map<Point, MQNode, PointHash> closed_list;

        // Lookup table for open set membership test in O(1) time
        std::unordered_map<Point, float, PointHash> open_f_values;

        // Initialize with start node
        float h = heuristic(start, goal);
        MQNode start_node(start, 0, h, Point{ -1, -1 });

        int queue_idx = getQueueIndex(start);
        if (queue_idx < 0 || queue_idx >= num_queues) {
            queue_idx = 0; // Failsafe for index out of bounds
        }

        open_queues[queue_idx].push(start_node);
        open_f_values[start] = h;

        // Target node tracking for optimal path verification
        MQNode* target_node = nullptr;

        // Main A* search loop - O(n log n) where n = search space size
        while (true) {
            // Find queue with globally minimum f-value node
            float min_f = std::numeric_limits<float>::max();
            int min_queue = -1;

            for (int i = 0; i < num_queues; ++i) {
                if (!open_queues[i].empty() && open_queues[i].top().f < min_f) {
                    min_f = open_queues[i].top().f;
                    min_queue = i;
                }
            }

            // Termination condition: open set exhausted
            if (min_queue == -1) {
                break;
            }

            // Early termination if optimal path found
            if (target_node != nullptr && target_node->f <= min_f) {
                break;
            }

            // Extract node with minimum f-value
            MQNode current = open_queues[min_queue].top();
            open_queues[min_queue].pop();
            open_f_values.erase(current.pos);

            // Skip if better path already found
            auto closed_it = closed_list.find(current.pos);
            if (closed_it != closed_list.end() && closed_it->second.g <= current.g) {
                continue;
            }

            // Add to closed set
            closed_list[current.pos] = current;
            nodes_expanded++; // Increment performance counter

            // Goal test
            if (current.pos.x == goal.x && current.pos.y == goal.y) {
                if (target_node == nullptr || current.f < target_node->f) {
                    delete target_node;
                    target_node = new MQNode(current);
                }
                continue; // Continue to ensure optimality
            }

            // Generate successor nodes
            std::vector<MQNode> new_nodes;

            // Parallel neighbor generation with OpenMP
#pragma omp parallel
            {
                std::vector<MQNode> thread_nodes;

#pragma omp for
                for (int dir = 0; dir < 8; dir++) {
                    Point neighbor = { current.pos.x + dx[dir], current.pos.y + dy[dir] };

                    // Skip invalid neighbors
                    int weight = getWeight(neighbor);
                    if (weight == -1) continue;

                    // Calculate g-cost with terrain factor
                    int new_g = current.g + costs[dir] * std::max(1, weight);

                    // Calculate f-value
                    float h = heuristic(neighbor, goal);
                    float new_f = new_g + h;

                    // Critical section checks
                    bool should_add = true;

#pragma omp critical
                    {
                        // Check against closed set
                        auto it = closed_list.find(neighbor);
                        if (it != closed_list.end() && it->second.g <= new_g) {
                            should_add = false;
                        }

                        // Check against open set
                        if (should_add) {
                            auto it = open_f_values.find(neighbor);
                            if (it != open_f_values.end() && it->second <= new_f) {
                                should_add = false;
                            }
                        }
                    }

                    if (should_add) {
                        MQNode new_node(neighbor, new_g, new_f, current.pos);
                        thread_nodes.push_back(new_node);
                    }
                }

                // Merge thread-local results
#pragma omp critical
                {
                    new_nodes.insert(new_nodes.end(), thread_nodes.begin(), thread_nodes.end());
                }
            }

            // Add new nodes to priority queues
            for (const auto& node : new_nodes) {
                int queue_idx = getQueueIndex(node.pos);
                if (queue_idx < 0 || queue_idx >= num_queues) {
                    queue_idx = 0; // Failsafe for index out of bounds
                }

                open_queues[queue_idx].push(node);
                open_f_values[node.pos] = node.f;
            }
        }

        // Path reconstruction phase - O(p) where p = path length
        if (target_node != nullptr) {
            // Start from goal
            Point current = target_node->pos;
            path.push_back(current);

            // Follow parent pointers to start
            int max_path_length = grid.width * grid.height; // Cycle detection
            int path_steps = 0;

            while (!(current.x == start.x && current.y == start.y) && path_steps < max_path_length) {
                auto it = closed_list.find(current);
                if (it == closed_list.end()) {
                    std::cerr << "Path reconstruction error at ("
                        << current.x << "," << current.y << ")" << std::endl;
                    path.clear();
                    break;
                }

                current = it->second.parent;
                path.push_back(current);
                path_steps++;
            }

            // Safety check for cycles in path
            if (path_steps >= max_path_length) {
                std::cerr << "Path reconstruction exceeded maximum length" << std::endl;
                path.clear();
            }
            else {
                // Reverse to get start-to-goal ordering
                std::reverse(path.begin(), path.end());
            }

            delete target_node;
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Exception during pathfinding: " << e.what() << std::endl;
        path.clear();
    }

    // Performance reporting
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "A* search completed in " << duration << " ms" << std::endl;
    std::cout << "Nodes expanded: " << nodes_expanded << std::endl;

    return path;
}

// Euclidean distance calculation between points
float distance(const Point& a, const Point& b) {
    return std::sqrt(static_cast<float>(
        (b.x - a.x) * (b.x - a.x) +
        (b.y - a.y) * (b.y - a.y)
        ));
}

// Path visualization and metrics calculation
void printPath(const std::vector<Point>& path) {
    if (path.empty()) {
        std::cout << "No path found" << std::endl;
        return;
    }

    std::cout << "Path found with " << path.size() << " steps:" << std::endl;

    float total_distance = 0.0f;
    const size_t display_limit = 10;

    // Display path start
    std::cout << "Start: ";
    for (size_t i = 0; i < std::min(display_limit, path.size()); i++) {
        std::cout << "(" << path[i].x << "," << path[i].y << ")";

        if (i < path.size() - 1) {
            float segment_distance = distance(path[i], path[i + 1]);
            total_distance += segment_distance;
            std::cout << " -> "; // ASCII arrow for compatibility
        }
    }

    // Indicate truncation
    if (path.size() > display_limit * 2) {
        std::cout << " ... ";
    }

    // Display path end
    if (path.size() > display_limit) {
        for (size_t i = std::max(display_limit, path.size() - display_limit); i < path.size(); i++) {
            std::cout << "(" << path[i].x << "," << path[i].y << ")";

            if (i < path.size() - 1) {
                std::cout << " -> "; // ASCII arrow for compatibility
            }
        }
    }

    std::cout << std::endl;
    std::cout << "Total path length: " << total_distance << " units" << std::endl;

    // Calculate path efficiency metrics
    if (path.size() >= 2) {
        Point start = path.front();
        Point goal = path.back();
        float direct_distance = distance(start, goal);
        float efficiency = direct_distance / total_distance * 100.0f;

        std::cout << "Direct distance: " << direct_distance << " units" << std::endl;
        std::cout << "Path efficiency: " << efficiency << "%" << std::endl;
    }
}