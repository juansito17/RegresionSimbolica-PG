cmake_minimum_required(VERSION 3.18)

# Habilitar CUDA
enable_language(CUDA)
project(SymbolicRegressionGP LANGUAGES CXX CUDA)

# Set C++ flags for Linux/Colab
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Adjusted flags for potentially better performance on T4
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math \
    -gencode arch=compute_75,code=sm_75 \
    --ptxas-options=-v \
    -Xcompiler -O3")
# Explicitly set architecture for clarity (matches gencode)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find required packages
find_package(OpenMP)

# Source files
set(CPP_SOURCES
    src/main.cpp
    src/ExpressionTree.cpp
    src/Fitness.cpp
    src/GeneticOperators.cpp
    src/AdvancedFeatures.cpp
    src/GeneticAlgorithm.cpp
)

set(CUDA_SOURCES
    cuda/fitness_kernels.cu
)

# Create executable
add_executable(SymbolicRegressionGP ${CPP_SOURCES} ${CUDA_SOURCES})

# Configure OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(SymbolicRegressionGP PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link CUDA libraries
target_link_libraries(SymbolicRegressionGP PRIVATE cuda cudart)

# Define USE_CUDA
target_compile_definitions(SymbolicRegressionGP PRIVATE USE_CUDA)