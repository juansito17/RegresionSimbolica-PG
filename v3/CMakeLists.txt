cmake_minimum_required(VERSION 3.18) # Raised for better CUDA support
cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0146 NEW)

project(SymbolicRegressionGP LANGUAGES CXX CUDA)

# Debug information
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "Looking for CUDA in standard locations...")

# Try to find CUDA in standard locations
if(EXISTS "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
elseif(EXISTS "/usr/lib/nvidia-cuda-toolkit")
    set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
endif()

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found. Please ensure CUDA toolkit is installed. Try: sudo apt install nvidia-cuda-toolkit")
endif()

# Set CUDA architectures - supporting modern GPUs
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

# Set C++ optimizations for Release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Increase CUDA heap and stack sizes for larger problems
add_compile_definitions(
    CUDA_MAX_HEAP_SIZE=2147483648  # 2GB
    CUDA_MAX_STACK_SIZE=131072     # 128KB
)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# Explicitly set the CUDA host compiler to match the C++ compiler
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Ensure proper CUDA device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA components
find_package(CUDAToolkit REQUIRED)

# Add source files (separate C++ and CUDA for clarity)
set(CPP_SOURCES
    src/main.cpp
    src/ExpressionTree.cpp
    src/Fitness.cpp
    src/GeneticOperators.cpp
    src/AdvancedFeatures.cpp
    src/GeneticAlgorithm.cpp
)

set(CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/cuda/fitness_kernels.cu # Use CMAKE_SOURCE_DIR
)

# Include directories
include_directories(
    src
    include
)

# Create executable from both C++ and CUDA sources
add_executable(SymbolicRegressionGP ${CPP_SOURCES} ${CUDA_SOURCES})

# Add preprocessor definition to enable CUDA code paths in C++
target_compile_definitions(SymbolicRegressionGP PRIVATE USE_CUDA)

# Link libraries
target_link_libraries(SymbolicRegressionGP
    PUBLIC
        OpenMP::OpenMP_CXX # OpenMP flags
    PRIVATE
        # Use modern CUDA target
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        # Math library (often needed, especially on Linux)
        m
)

# Add CUDA-specific compile definitions
target_compile_definitions(SymbolicRegressionGP 
    PRIVATE 
        USE_CUDA
        GPU_ACCELERATION
)

# Optional: Add optimization flags for release builds
# target_compile_options(SymbolicRegressionGP PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
# target_compile_options(SymbolicRegressionGP PRIVATE $<$<CONFIG:Debug>:-g>)