#ifndef GRADIENT_OPTIMIZER_GPU_CUH
#define GRADIENT_OPTIMIZER_GPU_CUH

#include "ExpressionTree.h"
#include <vector>

#ifdef __CUDACC__
#define DEVICE_FUNC __device__
#else
#define DEVICE_FUNC
#endif

// Main entry point for GPU-based gradient optimization
// This function handles memory allocation/copying and kernel launches
void optimize_constants_gradient_gpu_impl(
    NodePtr& tree,
    const std::vector<double>& targets,
    const std::vector<std::vector<double>>& x_values,
    double learning_rate,
    int iterations
);

#endif // GRADIENT_OPTIMIZER_GPU_CUH
