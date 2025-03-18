#include "multi_queue_astar.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>

// Heuristic function (diagonal distance)
float MultiQueueAStar::heuristic(const Point& p, const Point& goal) {
    int dx = std::abs(p.x - goal.x);
    int dy = std::abs(p.y - goal.y);
    int min_coord = std::min(dx, dy);
    int max_coord = std::max(dx, dy);
    return 10.0f * (dx + dy) + (14.0f - 2 * 10.0f) * min_coord;
}

// Get weight of a cell
int MultiQueueAStar::getWeight(const Point& p) {
    if (p.x < 0 || p.x >= grid.width || p.y < 0 || p.y >= grid.height)
        return -1; // Out of bounds

    return grid.weights[p.y * grid.width + p.x];
}

// Get minimum f value among all queues
float MultiQueueAStar::getMinFValue(const std::vector<PriorityQueue>& open_queues) {
    float min_f = std::numeric_limits<float>::max();
    
    for (const auto& queue : open_queues) {
        if (!queue.empty() && queue.top().f < min_f) {
            min_f = queue.top().f;
        }
    }
    
    return min_f;
}

// Determine which queue to use based on node coordinates
int MultiQueueAStar::getQueueIndex(const Point& p) {
    // Simple hash function to distribute nodes among queues
    return (p.x * 73856093 + p.y * 19349663) % num_queues;
}

MultiQueueAStar::MultiQueueAStar(const Grid& g, int queues) : grid(g), num_queues(queues) {
    // Ensure at least one queue
    if (num_queues < 1) num_queues = 1;
}

// Find path from start to goal
std::vector<Point> MultiQueueAStar::findPath(const Point& start, const Point& goal) {
    std::vector<Point> path;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check if start or goal is a wall
    if (getWeight(start) == -1 || getWeight(goal) == -1) {
        std::cout << "Start or goal position is invalid!" << std::endl;
        return path;
    }

    // Initialize multiple priority queues for the open list
    std::vector<PriorityQueue> open_queues(num_queues);
    
    // Shared closed list (visited nodes)
    std::unordered_map<Point, AStarNode, PointHash> closed_list;
    
    // Shared open list tracker to check if a node is in any open queue
    std::unordered_map<Point, float, PointHash> open_f_values;

    // Add start node to first queue
    float h = heuristic(start, goal);
    AStarNode start_node(start, 0, h, Point{ -1, -1 });
    
    int queue_idx = getQueueIndex(start);
    open_queues[queue_idx].push(start_node);
    open_f_values[start] = h;

    // Target node when found
    AStarNode* target_node = nullptr;

    // Main search loop
    while (true) {
        bool all_empty = true;
        std::vector<AStarNode> extracted_nodes;

        // Extract one node from each non-empty queue
        // multi_queue_astar.cpp
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(extracted_nodes.size()); i++) {
#pragma omp critical(queue_access)  // Unique name
            {
                if (!open_queues[i].empty()) {
                    all_empty = false;
                    AStarNode current = open_queues[i].top();
                    open_queues[i].pop();

#pragma omp critical(closed_list_update)  // Unique name
                    {
                        if (closed_list.find(current.pos) == closed_list.end()) {
                            extracted_nodes.push_back(current);
                            open_f_values.erase(current.pos);
                        }
                    }
                }
            }
        }

        // If all queues are empty, end search
        if (all_empty) {
            break;
        }

        // Process extracted nodes
        std::vector<AStarNode> new_nodes;
        
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(extracted_nodes.size()); i++) {
            AStarNode current = extracted_nodes[i];
            
            // Add to closed list
#pragma omp critical(closed_list_insert) 
            {
                closed_list[current.pos] = current;
            }
            
            // Check if this is the goal
            if (current.pos.x == goal.x && current.pos.y == goal.y) {
                #pragma omp critical
                {
                    if (target_node == nullptr || current.f < target_node->f) {
                        if (target_node != nullptr) {
                            delete target_node;
                        }
                        target_node = new AStarNode(current);
                    }
                }
                continue;
            }
            
            // Generate neighbors
            for (int dir = 0; dir < 8; dir++) {
                Point neighbor = {current.pos.x + dx[dir], current.pos.y + dy[dir]};
                
                // Skip invalid neighbors
                int weight = getWeight(neighbor);
                if (weight == -1) continue;
                
                // Calculate g score
                int new_g = current.g + costs[dir] * weight;
                
                // Check if already in closed list with better g
                bool skip = false;
                
                #pragma omp critical
                {
                    auto it = closed_list.find(neighbor);
                    if (it != closed_list.end() && it->second.g <= new_g) {
                        skip = true;
                    }
                }
                
                if (skip) continue;
                
                // Calculate f score
                float h = heuristic(neighbor, goal);
                float new_f = new_g + h;
                
                bool should_add = false;
                
                #pragma omp critical
                {
                    // Check if already in open list with better f
                    auto it = open_f_values.find(neighbor);
                    if (it == open_f_values.end() || new_f < it->second) {
                        open_f_values[neighbor] = new_f;
                        should_add = true;
                    }
                }
                
                if (should_add) {
                    AStarNode new_node(neighbor, new_g, new_f, current.pos);
                    
                    #pragma omp critical
                    {
                        new_nodes.push_back(new_node);
                    }
                }
            }
        }
        
        // If target found and has best f value among all open nodes, we found the optimal path
        if (target_node != nullptr) {
            float min_f = getMinFValue(open_queues);
            
            // Also consider nodes just expanded
            for (const auto& node : new_nodes) {
                min_f = std::min(min_f, node.f);
            }
            
            if (target_node->f <= min_f) {
                break;
            }
        }
        
        // Add new nodes to appropriate queues
        for (const auto& node : new_nodes) {
            int queue_idx = getQueueIndex(node.pos);
            open_queues[queue_idx].push(node);
        }
    }

    // Reconstruct path if target was found
    if (target_node != nullptr) {
        // Start from goal
        Point current = target_node->pos;
        path.push_back(current);
        
        // Follow parent pointers back to start
        while (!(current.x == start.x && current.y == start.y)) {
            AStarNode parent_node = closed_list[current];
            current = parent_node.parent;
            path.push_back(current);
        }
        
        // Reverse to get path from start to goal
        std::reverse(path.begin(), path.end());
        
        delete target_node;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "A* search completed in " << duration << " ms" << std::endl;
    std::cout << "Nodes expanded: " << closed_list.size() << std::endl;
    
    return path;
}

// Utility function to calculate distance between two points
float distance(const Point& a, const Point& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// Utility function to print a path
void printPath(const std::vector<Point>& path) {
    if (path.empty()) {
        std::cout << "No path found!" << std::endl;
        return;
    }
    
    std::cout << "Path found with " << path.size() << " steps:" << std::endl;
    
    float total_distance = 0.0f;
    
    for (size_t i = 0; i < path.size(); i++) {
        std::cout << "(" << path[i].x << ", " << path[i].y << ")";
        
        if (i < path.size() - 1) {
            float segment_distance = distance(path[i], path[i+1]);
            total_distance += segment_distance;
            std::cout << " -> ";
        }
    }
    
    std::cout << std::endl;
    std::cout << "Total path length: " << total_distance << std::endl;
}