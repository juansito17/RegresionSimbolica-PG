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

// CUDA kernel to evaluate a linearized tree
__global__ void calculate_raw_fitness_kernel(const LinearGpuNode* d_linear_tree,
                                             int tree_size,
                                             const double* d_targets,
                                             const double* d_x_values,
                                             size_t num_points,
                                             double* d_raw_fitness_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        double x_val = d_x_values[idx];
        double stack[64]; // Max tree depth
        int stack_top = -1;

        for (int i = 0; i < tree_size; ++i) {
            LinearGpuNode node = d_linear_tree[i];
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
                        if (fabs(right) < 1e-9) {
                            result = HUGE_VAL;
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
            d_raw_fitness_results[idx] = HUGE_VAL;
        } else {
            double diff = predicted_val - d_targets[idx];
            d_raw_fitness_results[idx] = diff * diff;
        }
    }
}

// Host-side wrapper function to launch the CUDA kernel
double evaluate_fitness_gpu(NodePtr tree,
                            const std::vector<double>& targets,
                            const std::vector<double>& x_values) {
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
    double* d_x_values;
    double* d_targets;
    double* d_raw_fitness_results;

    cudaMalloc((void**)&d_linear_tree, tree_size * sizeof(LinearGpuNode));
    cudaMalloc((void**)&d_x_values, num_points * sizeof(double));
    cudaMalloc((void**)&d_targets, num_points * sizeof(double));
    cudaMalloc((void**)&d_raw_fitness_results, num_points * sizeof(double));

    cudaMemcpy(d_linear_tree, h_linear_tree.data(), tree_size * sizeof(LinearGpuNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_values, x_values.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    calculate_raw_fitness_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_linear_tree, tree_size, d_targets, d_x_values, num_points, d_raw_fitness_results
    );

    std::vector<double> h_raw_fitness_results(num_points);
    cudaMemcpy(h_raw_fitness_results.data(), d_raw_fitness_results, num_points * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_sq_error = 0.0;
    for (double val : h_raw_fitness_results) {
        if (isinf(val) || isnan(val)) {
            sum_sq_error = INF;
            break;
        }
        sum_sq_error += val;
    }

    cudaFree(d_linear_tree);
    cudaFree(d_x_values);
    cudaFree(d_targets);
    cudaFree(d_raw_fitness_results);

    if (isinf(sum_sq_error)) {
        return INF;
    }

    double raw_fitness;
    if (USE_RMSE_FITNESS) {
        if (num_points == 0) return INF;
        double mse = sum_sq_error / num_points;
        raw_fitness = sqrt(mse);
    } else {
        raw_fitness = sum_sq_error;
    }

    double complexity = static_cast<double>(::tree_size(tree));
    double penalty = complexity * COMPLEXITY_PENALTY_FACTOR;
    double final_fitness = raw_fitness * (1.0 + penalty);

    if (isnan(final_fitness) || isinf(final_fitness) || final_fitness < 0) {
        return INF;
    }

    return final_fitness;
}
