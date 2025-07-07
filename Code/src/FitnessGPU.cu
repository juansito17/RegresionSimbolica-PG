#include "FitnessGPU.cuh" // This will contain declarations for GPU functions
#include "Globals.h"
#include "ExpressionTree.h" // For Node and NodePtr structure, needs adaptation for GPU
#include <cuda_runtime.h> // For __int_as_double
#include <math.h> // For NAN, isinf

// __device__ function for evaluating a tree on the GPU
__device__ double evaluate_tree_gpu(GpuNode* tree_node_ptr, double x_val) {
    if (!tree_node_ptr) {
        return NAN;
    }

    switch (tree_node_ptr->type) {
        case NodeType::Constant:
            return tree_node_ptr->value;
        case NodeType::Variable:
            return x_val;
        case NodeType::Operator: {
            switch (tree_node_ptr->op) {
                case '+':
                    return evaluate_tree_gpu(tree_node_ptr->left, x_val) + evaluate_tree_gpu(tree_node_ptr->right, x_val);
                case '-':
                    return evaluate_tree_gpu(tree_node_ptr->left, x_val) - evaluate_tree_gpu(tree_node_ptr->right, x_val);
                case '*':
                    return evaluate_tree_gpu(tree_node_ptr->left, x_val) * evaluate_tree_gpu(tree_node_ptr->right, x_val);
                case '/': {
                    double right_val = evaluate_tree_gpu(tree_node_ptr->right, x_val);
                    if (fabs(right_val) < 1e-9) return HUGE_VAL; // Avoid division by zero
                    return evaluate_tree_gpu(tree_node_ptr->left, x_val) / right_val;
                }
                // Add other operations as needed
                default:
                    return NAN; // Unknown operator
            }
        }
        default:
            return NAN; // Unknown node type
    }
}

// CUDA Kernel to calculate raw fitness for a batch of individuals
__global__ void calculate_raw_fitness_kernel(GpuNode** d_trees, // Changed to GpuNode**
                                             const double* d_targets,
                                             const double* d_x_values,
                                             size_t num_points,
                                             double* d_raw_fitness_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Corrected kernel indexing

    if (idx < num_points) {
        GpuNode* tree_node = d_trees[idx]; // Changed to GpuNode*
        double target_val = d_targets[idx];
        double x_val = d_x_values[idx];

        double predicted_val = evaluate_tree_gpu(tree_node, x_val);

        if (isnan(predicted_val) || isinf(predicted_val)) {
            d_raw_fitness_results[idx] = HUGE_VAL; // Mark as failed
        } else {
            double diff = predicted_val - target_val;
            double abs_diff = fabs(diff);

            if (!USE_RMSE_FITNESS) {
                d_raw_fitness_results[idx] = diff * diff;
            } else {
                d_raw_fitness_results[idx] = diff * diff; // Using squared error for RMSE
            }
        }
    }
}

// Host-side wrapper function to launch the CUDA kernel
double evaluate_fitness_gpu(NodePtr tree,
                            const std::vector<double>& targets,
                            const std::vector<double>& x_values) {
    if (x_values.size() != targets.size() || x_values.empty()) return INF;

    size_t num_points = x_values.size();
    double* d_x_values;
    double* d_targets;
    double* d_raw_fitness_results;
    GpuNode** d_trees; // Array of device pointers to individual GpuNode trees

    // Allocate host memory for the device tree pointers
    // This is still a conceptual placeholder for a single tree.
    // A real implementation would involve converting the host-side Node tree
    // into a GPU-compatible GpuNode tree structure and copying it to device memory.
    // This typically involves a recursive traversal and allocation for each node.
    GpuNode* h_tree_root_gpu_format = new GpuNode(); // Placeholder for the converted tree

    // For demonstration, let's just create a very simple GpuNode structure
    // that mirrors a single constant node from the input tree.
    // THIS IS NOT A GENERAL SOLUTION FOR ARBITRARY TREES.
    // A complete solution would require a function to deep-copy a NodePtr tree
    // to a GpuNode tree in device memory.
    // For now, to make it compile and run, we'll assign the root's value if it's a constant.
    // This needs to be replaced by a proper tree serialization/deserialization for GPU.
    if (tree) {
        h_tree_root_gpu_format->type = tree->type;
        h_tree_root_gpu_format->value = tree->value;
        h_tree_root_gpu_format->op = tree->op;
        // Pointers left and right need to be set up to point to device memory
        // This is the hard part and is currently NOT handled here.
    } else {
        delete h_tree_root_gpu_format;
        return INF;
    }


    // Simplified: Copying only the root pointer. This will NOT work for complex trees unless
    // the entire tree structure is manually copied to device memory beforehand.
    // This is a conceptual example for the integration point.
    cudaMalloc((void**)&d_trees, sizeof(GpuNode*));
    // This line is problematic: it copies a host GpuNode* address, not the GpuNode content.
    // To copy the content, you'd need to allocate device memory for h_tree_root_gpu_format
    // and then copy its content.
    // For now, let's allocate and copy the content of the root GpuNode only.
    GpuNode* d_root_node;
    cudaMalloc((void**)&d_root_node, sizeof(GpuNode));
    cudaMemcpy(d_root_node, h_tree_root_gpu_format, sizeof(GpuNode), cudaMemcpyHostToDevice);
    
    // Now d_trees should point to d_root_node
    cudaMemcpy(d_trees, &d_root_node, sizeof(GpuNode*), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_x_values, num_points * sizeof(double));
    cudaMalloc((void**)&d_targets, num_points * sizeof(double));
    cudaMalloc((void**)&d_raw_fitness_results, num_points * sizeof(double));

    cudaMemcpy(d_x_values, x_values.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    calculate_raw_fitness_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_trees, d_targets, d_x_values, num_points, d_raw_fitness_results
    );

    // Summing results from GPU (reduction)
    // This is also a simplification. A proper reduction would be a separate CUDA kernel.
    std::vector<double> h_raw_fitness_results(num_points);
    cudaMemcpy(h_raw_fitness_results.data(), d_raw_fitness_results, num_points * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_sq_error = 0.0;
    for (double val : h_raw_fitness_results) {
        if (isnan(val) || isinf(val)) return INF; // If any point evaluation failed
        sum_sq_error += val;
    }

    cudaFree(d_x_values);
    cudaFree(d_targets);
    cudaFree(d_raw_fitness_results);
    cudaFree(d_trees);
    cudaFree(d_root_node); // Free the allocated GpuNode on device
    delete h_tree_root_gpu_format; // Free the temporary host GpuNode

    double raw_fitness;
    if (USE_RMSE_FITNESS) {
        if (num_points == 0) return INF;
        double mse = sum_sq_error / num_points;
        raw_fitness = sqrt(mse);
    } else {
        raw_fitness = sum_sq_error;
    }

    // Apply complexity penalty (same as CPU version)
    double complexity = static_cast<double>(tree_size(tree));
    double penalty = complexity * COMPLEXITY_PENALTY_FACTOR;
    double final_fitness = raw_fitness * (1.0 + penalty);

    if (isnan(final_fitness) || isinf(final_fitness) || final_fitness < 0) {
        return INF;
    }

    return final_fitness;
}
