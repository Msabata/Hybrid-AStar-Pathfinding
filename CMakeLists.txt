cmake_minimum_required(VERSION 3.18)
project(HybridAStar LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA architectures based on your GPU
# For example: 
# - 61 for GTX 1080/1070
# - 75 for RTX 2080/2070
# - 86 for RTX 3080/3070
# Set appropriate value for your GPU
set(CMAKE_CUDA_ARCHITECTURES 61)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# With explicit CUDA source specification:
add_executable(hybrid_astar 
    main.cpp
    hybrid_astar.cpp
    multi_queue_astar.cpp
    common.cpp  # Added common implementation file
    $<TARGET_OBJECTS:hybrid_astar_cuda>
)

# Link libraries
target_link_libraries(hybrid_astar hybrid_astar_cuda ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

# Add CUDA objects as a separate target
add_library(hybrid_astar_cuda OBJECT 
    gpu_hash_tables.cu
)
set_target_properties(hybrid_astar_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Enable OpenMP
if(MSVC)
    target_compile_options(hybrid_astar PRIVATE /openmp)
else()
    target_compile_options(hybrid_astar PRIVATE -fopenmp)
endif()

# Set CUDA properties
set_target_properties(hybrid_astar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Print some info
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")