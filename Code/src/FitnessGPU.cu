#include "FitnessGPU.cuh"
#include "Globals.h"
#include <cuda_runtime.h>
#include <math.h>

// Helper function to linearize the tree into a post-order array
void linearize_tree(const NodePtr& node, std::vector<LinearGpuNode>& linear_tree) {
    if (!node) {
        return;
    }
    linearize_tree(node->left, linear_tree);
    linearize_tree(node->right, linear_tree);
    linear_tree.push_back({node->type, node->value, node->op});
}

#if USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
// CUDA kernel to evaluate a linearized tree
__global__ void calculate_raw_fitness_kernel(const LinearGpuNode* d_linear_tree,
                                             int tree_size,
                                             const double* d_targets,
                                             const double* d_x_values,
                                             size_t num_points,
                                             double* d_raw_fitness_results) {
    // Shared memory optimization: Load tree into shared memory
    extern __shared__ LinearGpuNode s_linear_tree[];

    // Cooperative load
    for (int i = threadIdx.x; i < tree_size; i += blockDim.x) {
        s_linear_tree[i] = d_linear_tree[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        double x_val = d_x_values[idx];
        double stack[64]; // Max tree depth
        int stack_top = -1;

        for (int i = 0; i < tree_size; ++i) {
            LinearGpuNode node = s_linear_tree[i]; // Access from shared memory
            if (node.type == NodeType::Constant) {
                stack[++stack_top] = node.value;
            } else if (node.type == NodeType::Variable) {
                stack[++stack_top] = x_val;
            } else if (node.type == NodeType::Operator) {
                double right = stack[stack_top--];
                double left = stack[stack_top--];
                double result;
                switch (node.op) {
                    case '+': result = left + right; break;
                    case '-': result = left - right; break;
                    case '*': result = left * right; break;
                    case '/':
                        if (fabs(right) < 1e-9) { // Avoid division by zero
                            result = 1e308; // Use large finite double instead of HUGE_VAL
                        } else {
                            result = left / right;
                        }
                        break;
                    default: result = NAN; break;
                }
                stack[++stack_top] = result;
            }
        }

        double predicted_val = (stack_top == 0) ? stack[0] : NAN;

        if (isnan(predicted_val) || isinf(predicted_val)) {
            d_raw_fitness_results[idx] = 1e308; // Use large finite double
        } else {
            double diff = predicted_val - d_targets[idx];
            d_raw_fitness_results[idx] = diff * diff;
        }
    }
}

// CUDA kernel for parallel reduction (summation)
__global__ void reduce_sum_kernel(double* d_data, int N) {
    extern __shared__ double sdata[]; // Shared memory for reduction

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? d_data[i] : 0.0; // Load data into shared memory

    __syncthreads(); // Synchronize threads in block

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) { // Write result back to global memory (first element of block)
        d_data[blockIdx.x] = sdata[0];
    }
}

// Host-side wrapper function to launch the CUDA kernel
double evaluate_fitness_gpu(NodePtr tree,
                            const std::vector<double>& targets,
                            const std::vector<double>& x_values,
                            double* d_targets, double* d_x_values) {
    if (x_values.size() != targets.size() || x_values.empty()) return INF;

    // Linearize the tree
    std::vector<LinearGpuNode> h_linear_tree;
    linearize_tree(tree, h_linear_tree);
    int tree_size = h_linear_tree.size();

    if (tree_size == 0) {
        return INF;
    }

    size_t num_points = x_values.size();
    LinearGpuNode* d_linear_tree;
    double* d_raw_fitness_results; // This will hold individual errors and then the final sum

    cudaMalloc((void**)&d_linear_tree, tree_size * sizeof(LinearGpuNode));
    cudaMalloc((void**)&d_raw_fitness_results, num_points * sizeof(double));

    cudaMemcpy(d_linear_tree, h_linear_tree.data(), tree_size * sizeof(LinearGpuNode), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to calculate individual squared errors
    size_t shared_mem_size = tree_size * sizeof(LinearGpuNode);
    calculate_raw_fitness_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        d_linear_tree, tree_size, d_targets, d_x_values, num_points, d_raw_fitness_results
    );
    cudaDeviceSynchronize(); // Ensure kernel completes before reduction

    // --- Perform reduction on the GPU ---
    int current_size = num_points;
    while (current_size > 1) {
        int next_blocks_per_grid = (current_size + threadsPerBlock - 1) / threadsPerBlock;
        // Use shared memory for reduction, size is threadsPerBlock * sizeof(double)
        reduce_sum_kernel<<<next_blocks_per_grid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(
            d_raw_fitness_results, current_size
        );
        cudaDeviceSynchronize(); // Ensure reduction step completes
        current_size = next_blocks_per_grid; // The result is in the first `next_blocks_per_grid` elements
    }

    double sum_sq_error_gpu = 0.0;
    cudaMemcpy(&sum_sq_error_gpu, d_raw_fitness_results, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_linear_tree);
    cudaFree(d_raw_fitness_results);

    // Check for invalid results (propagated from kernel)
    if (isinf(sum_sq_error_gpu) || isnan(sum_sq_error_gpu)) {
        return INF;
    }

    double raw_fitness;
    if (USE_RMSE_FITNESS) {
        if (num_points == 0) return INF;
        double mse = sum_sq_error_gpu / num_points;
        raw_fitness = sqrt(mse);
    } else {
        raw_fitness = sum_sq_error_gpu;
    }

    double complexity = static_cast<double>(::tree_size(tree));
    double penalty = complexity * COMPLEXITY_PENALTY_FACTOR;
    double final_fitness = raw_fitness * (1.0 + penalty);

    if (isnan(final_fitness) || isinf(final_fitness) || final_fitness < 0) {
        return INF;
    }

    return final_fitness;
}
#endif // USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
