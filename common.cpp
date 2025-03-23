#include "common.h"
#include <cstdlib>
#include <ctime>

// Direction vectors for 8-directional movement (declared as extern in common.h)
const int dx[8] = { 0, 1, 0, -1, 1, 1, -1, -1 };
const int dy[8] = { -1, 0, 1, 0, -1, 1, 1, -1 };
const int costs[8] = { 10, 10, 10, 10, 14, 14, 14, 14 };

// Implements grid generation with controlled randomization
Grid grid_generator(int width, int height, int seed) {
    Grid grid;
    grid.width = width;
    grid.height = height;
    grid.weights = new int[width * height];

    // Seed initialization for reproducible results
    srand(seed);

    // Fill grid with weighted costs and obstacles
    for (int i = 0; i < width * height; i++) {
        // 20% probability of generating a wall cell
        if (rand() % 100 < 20) {
            grid.weights[i] = -1;  // Wall/obstacle marker
        }
        else {
            grid.weights[i] = 1 + (rand() % 10);  // Variable terrain cost [1-10]
        }
    }

    // Ensure start and goal cells are traversable (top-left to bottom-right)
    grid.weights[0] = 1;  // Start position (top-left)
    grid.weights[width * height - 1] = 1;  // Goal position (bottom-right)

    return grid;
}