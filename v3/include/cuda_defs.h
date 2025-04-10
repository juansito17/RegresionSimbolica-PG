#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits> // Required for std::numeric_limits

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants for GPU configuration
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCKS = 65535;

// Large finite value to use as penalty within the kernel instead of INF
// Use double max divided by a factor to avoid immediate overflow when penalty is added later.
const double LARGE_FINITE_PENALTY = std::numeric_limits<double>::max() / 10.0;

// Utility function to calculate grid dimensions
inline dim3 calculateGrid(int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;
    return dim3(blocks);
}

inline dim3 calculateBlock() {
    return dim3(BLOCK_SIZE);
}
