#ifdef _WIN32
#define NOMINMAX // Evitar conflictos con std::min/max
#endif

#pragma nv_exec_check_disable
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <iostream>
#include "../src/ExpressionTree.h"
#include "../src/Globals.h" // Include Globals for INF

// Forward declarations of CPU-side types
class Node;
using NodePtr = std::shared_ptr<Node>;

// Estructura para representar un nodo en la GPU
struct GPUNode {
    enum class Type { Constant, Variable, Operator };
    Type type;
    double value;
    char op;
    int left_idx;
    int right_idx;
};

// Maximum tree depth for GPU evaluation
constexpr int MAX_TREE_DEPTH_GPU = 64; // Aumentado de 32 a 64 para permitir árboles más profundos
constexpr size_t MAX_NODES_PER_TREE = 10000; // Aumentado de 1000 a 10000 nodos por árbol
constexpr size_t MAX_TOTAL_NODES = 100000000; // Increased from 50M to 100M

// Structure for stack-based evaluation
struct EvalStackFrame {
    int node_idx;
    double value;
    bool needs_right;
    double left_value;
};

// Helper function to convert CPU tree to GPU format
__host__ void convertTreeToGPUFormat(const NodePtr& cpu_tree, std::vector<GPUNode>& gpu_nodes, size_t& current_tree_node_count, size_t& total_batch_nodes) {
    if (!cpu_tree) return;

    // Check limits before adding the current node
    if (current_tree_node_count >= MAX_NODES_PER_TREE) { // Check 1: Nodes added *for this specific tree*
         // Use a more specific error message
         throw std::runtime_error("Tree size limit exceeded (MAX_NODES_PER_TREE=" + std::to_string(MAX_NODES_PER_TREE) + ")");
    }
     if (total_batch_nodes >= MAX_TOTAL_NODES) { // Check 2: Total nodes added *across all trees so far in the batch*
         // Use a more specific error message
         throw std::runtime_error("Total batch node limit exceeded (MAX_TOTAL_NODES=" + std::to_string(MAX_TOTAL_NODES) + ")");
     }

    size_t current_idx_in_batch = gpu_nodes.size(); // Index within the batch's flattened vector
    gpu_nodes.push_back(GPUNode()); // Add placeholder for current node
    current_tree_node_count++; // Increment count for the current tree being processed
    total_batch_nodes++;     // Increment total count for the batch

    GPUNode& gpu_node = gpu_nodes.back(); // Get reference to the added node

    // Initialize indices to -1 (or another indicator for no child)
    gpu_node.left_idx = -1;
    gpu_node.right_idx = -1;

    switch (cpu_tree->type) {
        case NodeType::Constant:
            gpu_node.type = GPUNode::Type::Constant;
            gpu_node.value = cpu_tree->value;
            // No children, indices remain -1
            break;
        case NodeType::Variable:
            gpu_node.type = GPUNode::Type::Variable;
            // No children, indices remain -1
            break;
        case NodeType::Operator:
            gpu_node.type = GPUNode::Type::Operator;
            gpu_node.op = cpu_tree->op;

            // Recursively convert left child if it exists
            if (cpu_tree->left) {
                size_t left_child_start_idx = gpu_nodes.size(); // Index where left subtree will start
                convertTreeToGPUFormat(cpu_tree->left, gpu_nodes, current_tree_node_count, total_batch_nodes);
                // Calculate relative offset from current node to left child's start
                gpu_node.left_idx = static_cast<int>(left_child_start_idx - current_idx_in_batch);
            }

            // Recursively convert right child if it exists
            if (cpu_tree->right) {
                size_t right_child_start_idx = gpu_nodes.size(); // Index where right subtree will start
                convertTreeToGPUFormat(cpu_tree->right, gpu_nodes, current_tree_node_count, total_batch_nodes);
                // Calculate relative offset from current node to right child's start
                gpu_node.right_idx = static_cast<int>(right_child_start_idx - current_idx_in_batch);
            }
            break;
         default:
             // Should not happen with valid NodeType
             throw std::runtime_error("Unknown NodeType encountered during conversion");
    }
}

// Check if CUDA device is available and return device properties
__host__ bool checkCUDADevice() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: Failed to get device count - " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }

    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: Failed to get device properties - " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Verificar memoria disponible
    size_t free_mem, total_mem;
    error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: Failed to get memory info - " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Non-recursive GPU tree evaluation
__device__ double evaluateTreeGPU(const GPUNode* nodes, int start_idx, double x, int max_depth) {
    if (start_idx < 0) return INFINITY;

    EvalStackFrame stack[MAX_TREE_DEPTH_GPU];
    int stack_top = 0;

    stack[stack_top++] = {start_idx, 0.0, false, 0.0};
    double result = 0.0;
    int iterations = 0; // Safeguard counter
    const int max_iterations = MAX_NODES_PER_TREE * 5; // Heuristic limit

    while (stack_top > 0 && iterations < max_iterations) { // Added iteration limit
        iterations++;
        EvalStackFrame current_frame = stack[--stack_top];
        int current_node_idx = current_frame.node_idx;

        // Basic check: Is the index within reasonable bounds? (Requires knowing array size, hard here)
        // Relying on valid indices from conversion is primary.

        const GPUNode& node = nodes[current_node_idx];

        if (node.type == GPUNode::Type::Constant) {
            result = node.value;
        } else if (node.type == GPUNode::Type::Variable) {
            result = x;
        } else if (node.type == GPUNode::Type::Operator) {
            if (!current_frame.needs_right) {
                // First visit: Need to evaluate left child first
                current_frame.needs_right = true;
                if (stack_top >= MAX_TREE_DEPTH_GPU - 1) { result = INFINITY; break; } // Stack overflow check
                stack[stack_top++] = current_frame;

                if (node.left_idx != -1) { // Check if left child exists and index is valid
                     int left_child_node_idx = current_node_idx + node.left_idx;
                     // Add basic bounds check if possible, otherwise rely on conversion correctness
                     if (stack_top >= MAX_TREE_DEPTH_GPU) { result = INFINITY; break; } // Stack overflow check
                     stack[stack_top++] = {left_child_node_idx, 0.0, false, 0.0};
                } else {
                     result = INFINITY; // Operator missing left child
                     break;
                }
                continue;
            } else if (current_frame.left_value == 0.0 && !isnan(result)) {
                 // Second visit: Left child evaluated, need right child
                 current_frame.left_value = result;
                 if (stack_top >= MAX_TREE_DEPTH_GPU - 1) { result = INFINITY; break; } // Stack overflow check
                 stack[stack_top++] = current_frame;

                 if (node.right_idx != -1) { // Check if right child exists and index is valid
                      int right_child_node_idx = current_node_idx + node.right_idx;
                      // Add basic bounds check if possible
                      if (stack_top >= MAX_TREE_DEPTH_GPU) { result = INFINITY; break; } // Stack overflow check
                      stack[stack_top++] = {right_child_node_idx, 0.0, false, 0.0};
                 } else {
                      result = INFINITY; // Operator missing right child
                      break;
                 }
                 continue;
            } else {
                // Third visit: Both children evaluated
                double left_val = current_frame.left_value;
                double right_val = result;

                // ... (rest of operator evaluation logic remains the same) ...
                 if (isinf(left_val) || isnan(left_val) || isinf(right_val) || isnan(right_val)) {
                     result = INFINITY;
                 } else {
                     switch (node.op) {
                         // ... cases ...
                         case '/':
                             result = fabsf(right_val) < 1e-9 ? INFINITY : left_val / right_val;
                             break;
                         case '^': {
                             // ... existing pow checks ...
                             result = powf(left_val, right_val);
                             if (isinf(result) || isnan(result)) result = INFINITY;
                             // } // Removed extra brace
                             break; // Added break
                         }
                         default: result = INFINITY;
                     }
                 }
                 if (isinf(result) || isnan(result)) {
                     result = INFINITY;
                 }
            }
        } else {
             result = INFINITY; // Unknown node type
        }

        // Check result validity after each step inside the loop? Maybe too much overhead.
        if (isinf(result) || isnan(result)) {
             result = INFINITY;
             break; // Exit loop early if invalid state reached
        }

    } // End while loop

    if (iterations >= max_iterations) {
        // Exceeded iteration limit, likely an issue
        printf("Warning: Max iterations reached in evaluateTreeGPU for start_idx %d\n", start_idx); // Use printf carefully in kernels
        return INFINITY;
    }


    return (isinf(result) || isnan(result)) ? INFINITY : result;
}


// CUDA kernel for evaluating trees
extern "C" __global__ void evaluateTreesKernel(
    const GPUNode* nodes,
    const int* tree_starts,
    const int* tree_sizes,
    const double* x_values,
    const double* targets,
    double* fitness_results,
    int num_trees,
    int num_points
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_trees) return;

    double total_error = 0.0;
    bool valid_evaluation = true;
    int tree_start = tree_starts[tid];
    // int tree_node_count = tree_sizes[tid]; // Potentially use for checks

    for (int i = 0; i < num_points; ++i) { // Removed valid_evaluation check from loop condition for now
        double x = x_values[i];
        double predicted = evaluateTreeGPU(nodes, tree_start, x, MAX_TREE_DEPTH_GPU); // Pass start index relative to nodes array

        if (isinf(predicted) || isnan(predicted)) {
            valid_evaluation = false;
            total_error = INFINITY;
            break; // Stop evaluating points for this tree
        }

        double error = fabs(predicted - targets[i]);
        // Check for massive individual errors
        if (error > 1e50) { // Reduced threshold
            valid_evaluation = false;
            total_error = INFINITY;
            break;
        }

        // --- Use powf for error calculation ---
        // total_error += error; // Old absolute error
        total_error += powf(error, 1.3f); // Use powf for consistency with CPU

        // Check for massive cumulative error
        if (total_error > 1e50) { // Reduced threshold
             valid_evaluation = false;
             total_error = INFINITY;
             break;
        }
    }

    // Final assignment based on validity
    fitness_results[tid] = valid_evaluation ? total_error : INFINITY;
}


// Wrapper function to evaluate multiple trees on GPU
extern "C" __host__ void evaluatePopulationGPU(
    const std::vector<NodePtr>& trees,
    const std::vector<double>& x_values,
    const std::vector<double>& targets,
    std::vector<double>& fitness_results // Note: This should be pre-sized by the caller
) {
    if (!checkCUDADevice()) {
        throw std::runtime_error("No suitable CUDA device available");
    }

    if (trees.empty()) { // Handle empty input gracefully
        fitness_results.assign(fitness_results.size(), INF); // Assign INF if caller expects results
        return;
    }

    if (x_values.empty() || targets.empty()) {
        throw std::runtime_error("Empty input data");
    }

    // Preparar datos para GPU
    std::vector<GPUNode> all_nodes;
    std::vector<int> tree_starts;
    std::vector<int> tree_sizes_gpu; // Renamed to avoid confusion with CPU tree_size()
    size_t total_batch_nodes = 0; // Counter for nodes added in this batch
    std::vector<int> original_indices; // Track original index for skipped trees

    // Reserve memory to reduce reallocations
    all_nodes.reserve(trees.size() * 50); // Estimate average size
    tree_starts.reserve(trees.size());
    tree_sizes_gpu.reserve(trees.size());
    original_indices.reserve(trees.size());

    // Convert trees to GPU format, skipping oversized ones
    for (size_t i = 0; i < trees.size(); ++i) {
        const auto& tree = trees[i];

        // Pre-check: Skip null or excessively large trees before conversion attempt
        if (!tree) {
            // std::cerr << "Warning: Skipping null tree at index " << i << std::endl;
            // Fitness should be set by caller or handled later based on index mapping
            continue; // Skip this tree
        }

        size_t cpu_tree_node_count = tree_size(tree); // Use CPU tree_size for pre-check
        if (cpu_tree_node_count >= MAX_NODES_PER_TREE) {
             std::cerr << "Warning: Skipping tree at index " << i << " due to excessive size ("
                       << cpu_tree_node_count << " >= " << MAX_NODES_PER_TREE << ")" << std::endl;
             // Mark this index for INF fitness later
             // We need a way to map results back. Let's assume fitness_results corresponds
             // to the input 'trees' vector and set INF here.
             if (i < fitness_results.size()) {
                 fitness_results[i] = INF;
             }
             continue; // Skip this tree
        }

        // If pre-check passes, attempt conversion
        tree_starts.push_back(all_nodes.size()); // Start index for this tree in the flattened vector
        size_t current_tree_node_count_gpu = 0; // Nodes added by converter for *this* tree
        size_t nodes_before_conversion = all_nodes.size();

        try {
            convertTreeToGPUFormat(tree, all_nodes, current_tree_node_count_gpu, total_batch_nodes);
            // Store the actual number of nodes added by the conversion
            tree_sizes_gpu.push_back(static_cast<int>(all_nodes.size() - nodes_before_conversion));
            original_indices.push_back(i); // Store original index of successfully converted tree
        } catch (const std::exception& e) {
            // Handle conversion error (e.g., total batch limit exceeded during this tree's conversion)
            std::cerr << "Error converting tree at original index " << i << ": " << e.what() << std::endl;
            // Roll back nodes added for this tree (if any) - tricky, maybe just stop batch?
            // For simplicity, let's re-throw and let the caller handle batch failure.
            throw; // Re-throw to signal batch failure
        }
    }

    // Check if any trees were actually converted
    if (tree_starts.empty()) {
         return; // Nothing to do on GPU
    }

    // --- GPU Execution ---
    // Calculate memory requirements based on *actual* nodes added
    size_t required_memory = sizeof(GPUNode) * total_batch_nodes +
                           sizeof(int) * tree_starts.size() * 2 +
                           sizeof(double) * (x_values.size() + targets.size() + tree_starts.size());

    size_t free_mem, total_mem;
    cudaError_t mem_error = cudaMemGetInfo(&free_mem, &total_mem); // Check error here too
    if (mem_error != cudaSuccess) {
         std::cerr << "CUDA error getting memory info: " << cudaGetErrorString(mem_error) << std::endl;
         throw std::runtime_error("Failed to get GPU memory info");
    }

    if (required_memory > free_mem * 0.95) {
        throw std::runtime_error("Insufficient GPU memory for converted trees");
    }

    // Device vectors should be sized based on converted trees
    thrust::device_vector<GPUNode> d_nodes;
    thrust::device_vector<int> d_tree_starts;
    thrust::device_vector<int> d_tree_sizes; // Use the GPU sizes vector
    thrust::device_vector<double> d_x_values;
    thrust::device_vector<double> d_targets;
    thrust::device_vector<double> d_batch_fitness_results; // Results for the converted trees only

    try {
        d_nodes.assign(all_nodes.begin(), all_nodes.end()); // Copy actual nodes
        d_tree_starts.assign(tree_starts.begin(), tree_starts.end());
        d_tree_sizes.assign(tree_sizes_gpu.begin(), tree_sizes_gpu.end()); // Copy actual GPU sizes
        d_x_values.assign(x_values.begin(), x_values.end());
        d_targets.assign(targets.begin(), targets.end());
        d_batch_fitness_results.resize(tree_starts.size()); // Size for results of converted trees

        // Increased threads per block, common for compute-heavy tasks on newer GPUs
        const int threadsPerBlock = 512;
        const int numBlocks = (tree_starts.size() + threadsPerBlock - 1) / threadsPerBlock;

        evaluateTreesKernel<<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_tree_starts.data()),
            thrust::raw_pointer_cast(d_tree_sizes.data()),
            thrust::raw_pointer_cast(d_x_values.data()),
            thrust::raw_pointer_cast(d_targets.data()),
            thrust::raw_pointer_cast(d_batch_fitness_results.data()),
            tree_starts.size(),
            x_values.size()
        );

        // --- ADDED: Check for errors immediately after launch ---
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::string error_msg = "CUDA error after kernel launch: ";
            error_msg += cudaGetErrorString(error);
            // Clean up device memory before throwing? Maybe not necessary if error is severe.
            throw std::runtime_error(error_msg);
        }
        // --- END ADDED ---


        // Synchronize device to ensure kernel completion
        error = cudaDeviceSynchronize(); // Check error from sync
        if (error != cudaSuccess) {
            std::string error_msg = "CUDA error during device synchronization: ";
            error_msg += cudaGetErrorString(error);
            throw std::runtime_error(error_msg);
        }

        // Copy results from d_batch_fitness_results back to the correct positions
        // in the original fitness_results vector using original_indices mapping
        thrust::host_vector<double> h_batch_results = d_batch_fitness_results;
        for(size_t i = 0; i < h_batch_results.size(); ++i) {
            int original_idx = original_indices[i];
            if (original_idx < fitness_results.size()) {
                fitness_results[original_idx] = h_batch_results[i];
            } else {
                 std::cerr << "Error: Original index " << original_idx << " out of bounds for fitness results." << std::endl;
            }
        }

    } catch (const thrust::system_error& e) {
        std::cerr << "Thrust error: " << e.what() << std::endl;
        throw std::runtime_error(std::string("GPU error: ") + e.what());
    } catch (const std::runtime_error& e) { // Catch other runtime errors (like CUDA errors)
         std::cerr << "Runtime error during GPU execution: " << e.what() << std::endl;
         throw;
    }
}