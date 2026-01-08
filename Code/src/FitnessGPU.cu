#include "FitnessGPU.cuh"
#include "Globals.h"
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <iostream>

// Helper function to linearize the tree into a post-order array
void linearize_tree(const NodePtr& node, std::vector<LinearGpuNode>& linear_tree) {
    if (!node) {
        return;
    }
    linearize_tree(node->left, linear_tree);
    linearize_tree(node->right, linear_tree);
    // Include var_index in the struct initialization
    linear_tree.push_back({node->type, node->value, node->var_index, node->op});
}

#if USE_GPU_ACCELERATION_DEFINED_BY_CMAKE

// Constant for large finite value
#define GPU_MAX_DOUBLE 1e308

// --- WEIGHTED FITNESS: Constantes para CUDA ---
// Estas deben coincidir con los valores en Globals.h
// CUDA device code no puede acceder a const C++, así que usamos #define
#define GPU_USE_WEIGHTED_FITNESS true
#define GPU_WEIGHTED_FITNESS_EXPONENT 0.25

// Single Tree Evaluation Kernel (Updated for Multivariable)
__global__ void calculate_raw_fitness_kernel(const LinearGpuNode* d_linear_tree,
                                             int tree_size,
                                             const double* d_targets,
                                             const double* d_x_values, // Flattened [num_points * num_vars]
                                             size_t num_points,
                                             int num_vars,
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
        double stack[64]; // Max tree depth
        int stack_top = -1;

        for (int i = 0; i < tree_size; ++i) {
            LinearGpuNode node = s_linear_tree[i]; // Access from shared memory
            if (node.type == NodeType::Constant) {
                stack[++stack_top] = node.value;
            } else if (node.type == NodeType::Variable) {
                // Access correct variable for this sample
                int var_idx = node.var_index;
                if (var_idx >= num_vars) var_idx = 0; // Safety fallback
                stack[++stack_top] = d_x_values[idx * num_vars + var_idx];
            } else if (node.type == NodeType::Operator) {
                bool is_unary = (node.op == 's' || node.op == 'c' || node.op == 'l' || node.op == 'e' || node.op == '!' || node.op == '_' || node.op == 'g');
                double result = 0.0;
                
                if (is_unary) {
                     if (stack_top < 0) {
                         result = GPU_MAX_DOUBLE;
                     } else {
                         double val = stack[stack_top--];
                         switch (node.op) {
                            case 's': result = sin(val); break;
                            case 'c': result = cos(val); break;
                            case 'l': result = (val <= 1e-9) ? GPU_MAX_DOUBLE : log(val); break;
                            case 'e': result = (val > 700.0) ? GPU_MAX_DOUBLE : exp(val); break;
                            case '!': result = (val < 0 || val > 170.0) ? GPU_MAX_DOUBLE : tgamma(val + 1.0); break;
                            case '_': result = floor(val); break;
                            case 'g': result = (val <= -1.0) ? GPU_MAX_DOUBLE : lgamma(val + 1.0); break;
                            default: result = NAN; break;
                         }
                     }
                     stack[++stack_top] = result;
                } else {
                    if (stack_top < 1) { 
                        result = GPU_MAX_DOUBLE;
                        stack[++stack_top] = result; // Push error
                    } else {
                        double right = stack[stack_top--];
                        double left = stack[stack_top--];
                        switch (node.op) {
                            case '+': result = left + right; break;
                            case '-': result = left - right; break;
                            case '*': result = left * right; break;
                            case '/':
                                if (fabs(right) < 1e-9) { // Avoid division by zero
                                    result = GPU_MAX_DOUBLE; 
                                } else {
                                    result = left / right;
                                }
                                break;
                            case '^': result = pow(left, right); break;
                            case '%':
                                if (fabs(right) < 1e-9) result = GPU_MAX_DOUBLE;
                                else result = fmod(left, right);
                                break;
                            default: result = NAN; break;
                        }
                        stack[++stack_top] = result;
                    }
                }
            }
        }

        double predicted_val = (stack_top == 0) ? stack[0] : NAN;

        if (isnan(predicted_val) || isinf(predicted_val)) {
            d_raw_fitness_results[idx] = GPU_MAX_DOUBLE; 
        } else {
            double diff = predicted_val - d_targets[idx];
            double sq_error = diff * diff;
            // --- WEIGHTED FITNESS: Apply exponential weight ---
            if (GPU_USE_WEIGHTED_FITNESS) {
                double weight = exp((double)idx * GPU_WEIGHTED_FITNESS_EXPONENT);
                sq_error *= weight;
            }
            d_raw_fitness_results[idx] = sq_error;
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


// --- New Batch Kernel (Updated for Multivariable) ---
__global__ void evaluate_population_kernel(const LinearGpuNode* d_all_nodes,
                                           const int* d_offsets,
                                           const int* d_sizes,
                                           int pop_size,
                                           const double* d_targets,
                                           const double* d_x_values,
                                           int num_points,
                                           int num_vars,
                                           double* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pop_size) {
        int offset = d_offsets[idx];
        int size = d_sizes[idx];
        double sum_sq_error = 0.0;
        double total_weight = 0.0; // Para normalizar fitness ponderado
        bool valid = true;

        for (int p = 0; p < num_points; ++p) {
            double stack[64]; 
            int stack_top = -1;

            // Simple interpreter
            for (int i = 0; i < size; ++i) {
                LinearGpuNode node = d_all_nodes[offset + i];
                if (node.type == NodeType::Constant) {
                    stack[++stack_top] = node.value;
                } else if (node.type == NodeType::Variable) {
                    int var_idx = node.var_index;
                    if (var_idx >= num_vars) var_idx = 0;
                    stack[++stack_top] = d_x_values[p * num_vars + var_idx];
                } else if (node.type == NodeType::Operator) {
                    bool is_unary = (node.op == 's' || node.op == 'c' || node.op == 'l' || node.op == 'e' || node.op == '!' || node.op == '_' || node.op == 'g');
                    
                    if (is_unary) {
                        if (stack_top < 0) { valid = false; break; }
                        double val = stack[stack_top--];
                        double result = 0.0;
                         switch (node.op) {
                            case 's': result = sin(val); break;
                            case 'c': result = cos(val); break;
                            case 'l': result = (val <= 1e-9) ? GPU_MAX_DOUBLE : log(val); break;
                            case 'e': result = (val > 700.0) ? GPU_MAX_DOUBLE : exp(val); break;
                            case '!': result = (val < 0 || val > 170.0) ? GPU_MAX_DOUBLE : tgamma(val + 1.0); break;
                            case '_': result = floor(val); break;
                            case 'g': result = (val <= -1.0) ? GPU_MAX_DOUBLE : lgamma(val + 1.0); break;
                             default: result = NAN; break;
                        }
                        stack[++stack_top] = result;
                    } else {
                        // Safety check index
                        if (stack_top < 1) { valid = false; break; }

                        double right = stack[stack_top--];
                        double left = stack[stack_top--];
                        double result;
                        switch (node.op) {
                            case '+': result = left + right; break;
                            case '-': result = left - right; break;
                            case '*': result = left * right; break;
                            case '/':
                                if (fabs(right) < 1e-9) { 
                                    result = GPU_MAX_DOUBLE; 
                                } else {
                                    result = left / right;
                                }
                                break;
                            case '^': result = pow(left, right); break;
                            case '%':
                                if (fabs(right) < 1e-9) result = GPU_MAX_DOUBLE;
                                else result = fmod(left, right);
                                break;
                            default: result = NAN; break;
                        }
                        stack[++stack_top] = result;
                    }
                }
            }

            if (!valid || stack_top != 0) {
                sum_sq_error = GPU_MAX_DOUBLE;
                break;
            }

            double predicted_val = stack[0];
            if (isnan(predicted_val) || isinf(predicted_val)) {
                sum_sq_error = GPU_MAX_DOUBLE;
                break;
            }

            double diff = predicted_val - d_targets[p];
            double sq_error = diff * diff;
            
            // --- WEIGHTED FITNESS: Peso exponencial ---
            double weight = 1.0;
            if (GPU_USE_WEIGHTED_FITNESS) {
                weight = exp((double)p * GPU_WEIGHTED_FITNESS_EXPONENT);
            }
            total_weight += weight;
            sum_sq_error += sq_error * weight;
        }

        // Normalizar por suma de pesos para obtener MSE ponderado
        if (GPU_USE_WEIGHTED_FITNESS && total_weight > 0.0) {
            sum_sq_error = sum_sq_error / total_weight * num_points; // Escalar de vuelta
        }
        d_results[idx] = sum_sq_error;
    }
}


// Host-side wrapper function to launch the CUDA kernel
double evaluate_fitness_gpu(NodePtr tree,
                            const std::vector<double>& targets,
                            const std::vector<std::vector<double>>& x_values,
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
    int num_vars = (num_points > 0) ? x_values[0].size() : 0;
    
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
        d_linear_tree, tree_size, d_targets, d_x_values, num_points, num_vars, d_raw_fitness_results
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

void evaluate_population_gpu(const std::vector<LinearGpuNode>& all_nodes,
                             const std::vector<int>& tree_offsets,
                             const std::vector<int>& tree_sizes,
                             const std::vector<double>& targets,
                             const std::vector<std::vector<double>>& x_values,
                             std::vector<double>& results,
                             double* d_targets, double* d_x_values,
                             void*& d_nodes_ptr, size_t& d_nodes_cap,
                             void*& d_offsets_ptr, void*& d_sizes_ptr, void*& d_results_ptr, size_t& d_pop_cap) {
    
    int pop_size = tree_offsets.size();
    if (pop_size == 0) return;

    size_t total_nodes = all_nodes.size();
    int num_points = x_values.size();
    int num_vars = (num_points > 0) ? x_values[0].size() : 0;

    // Buffer Management for Nodes
    if (total_nodes > d_nodes_cap) {
        if (d_nodes_ptr) cudaFree(d_nodes_ptr);
        size_t new_cap = total_nodes * 1.5; // Growth factor
        cudaMalloc(&d_nodes_ptr, new_cap * sizeof(LinearGpuNode));
        d_nodes_cap = new_cap;
    }

    // Buffer Management for Population Arrays
    if (pop_size > d_pop_cap) {
        if (d_offsets_ptr) cudaFree(d_offsets_ptr);
        if (d_sizes_ptr) cudaFree(d_sizes_ptr);
        if (d_results_ptr) cudaFree(d_results_ptr);
        
        size_t new_cap = pop_size * 1.5;
        cudaMalloc(&d_offsets_ptr, new_cap * sizeof(int));
        cudaMalloc(&d_sizes_ptr, new_cap * sizeof(int));
        cudaMalloc(&d_results_ptr, new_cap * sizeof(double));
        d_pop_cap = new_cap;
    }

    LinearGpuNode* d_all_nodes = (LinearGpuNode*)d_nodes_ptr;
    int* d_offsets = (int*)d_offsets_ptr;
    int* d_sizes = (int*)d_sizes_ptr;
    double* d_results = (double*)d_results_ptr;

    cudaMemcpy(d_all_nodes, all_nodes.data(), total_nodes * sizeof(LinearGpuNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, tree_offsets.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, tree_sizes.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    evaluate_population_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_all_nodes, d_offsets, d_sizes, pop_size, d_targets, d_x_values, num_points, num_vars, d_results
    );

    // Synchronize and copy back
    cudaDeviceSynchronize();
    
    cudaMemcpy(results.data(), d_results, pop_size * sizeof(double), cudaMemcpyDeviceToHost);
}

// ============================================================
// GLOBAL BATCH EVALUATION - Maximum GPU Utilization
// ============================================================

void init_global_gpu_buffers(GlobalGpuBuffers& buffers) {
    // Create CUDA stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    buffers.cuda_stream = (void*)stream;
    
    // Pre-allocate initial buffers (will grow as needed)
    // Initial capacity for 50,000 trees with ~30 nodes each
    buffers.d_nodes_capacity = 1500000;
    buffers.d_pop_capacity = 60000;
    
    cudaMalloc(&buffers.d_nodes, buffers.d_nodes_capacity * sizeof(LinearGpuNode));
    cudaMalloc(&buffers.d_offsets, buffers.d_pop_capacity * sizeof(int));
    cudaMalloc(&buffers.d_sizes, buffers.d_pop_capacity * sizeof(int));
    cudaMalloc(&buffers.d_results, buffers.d_pop_capacity * sizeof(double));
}

void cleanup_global_gpu_buffers(GlobalGpuBuffers& buffers) {
    if (buffers.cuda_stream) {
        cudaStreamDestroy((cudaStream_t)buffers.cuda_stream);
        buffers.cuda_stream = nullptr;
    }
    if (buffers.d_nodes) { cudaFree(buffers.d_nodes); buffers.d_nodes = nullptr; }
    if (buffers.d_offsets) { cudaFree(buffers.d_offsets); buffers.d_offsets = nullptr; }
    if (buffers.d_sizes) { cudaFree(buffers.d_sizes); buffers.d_sizes = nullptr; }
    if (buffers.d_results) { cudaFree(buffers.d_results); buffers.d_results = nullptr; }
    buffers.d_nodes_capacity = 0;
    buffers.d_pop_capacity = 0;
}

// Optimized kernel: Process one tree per thread
// REMOVED SHARED MEMORY FOR DATA to support arbitrary dataset sizes and multivariable
__global__ void evaluate_all_populations_kernel(
    const LinearGpuNode* __restrict__ d_all_nodes,
    const int* __restrict__ d_offsets,
    const int* __restrict__ d_sizes,
    int total_trees,
    const double* __restrict__ d_targets,
    const double* __restrict__ d_x_values,
    int num_points,
    int num_vars,
    double* __restrict__ d_results,
    double complexity_penalty_factor,
    bool use_rmse) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_trees) {
        int offset = d_offsets[idx];
        int size = d_sizes[idx];
        double sum_sq_error = 0.0;
        double total_weight = 0.0;
        bool valid = true;

        // Evaluate tree on all data points
        for (int p = 0; p < num_points && valid; ++p) {
            double stack[64]; 
            int stack_top = -1;

            // Interpret the linearized tree
            for (int i = 0; i < size && valid; ++i) {
                LinearGpuNode node = d_all_nodes[offset + i];
                
                if (node.type == NodeType::Constant) {
                    stack[++stack_top] = node.value;
                } else if (node.type == NodeType::Variable) {
                    // Access global memory directly
                    int var_idx = node.var_index;
                    if (var_idx >= num_vars) var_idx = 0;
                    stack[++stack_top] = d_x_values[p * num_vars + var_idx];
                } else if (node.type == NodeType::Operator) {
                    bool is_unary = (node.op == 's' || node.op == 'c' || node.op == 'l' || 
                                    node.op == 'e' || node.op == '!' || node.op == '_' || node.op == 'g');
                    
                    if (is_unary) {
                        if (stack_top < 0) { valid = false; break; }
                        double val = stack[stack_top--];
                        double result = 0.0;
                        switch (node.op) {
                            case 's': result = sin(val); break;
                            case 'c': result = cos(val); break;
                            case 'l': result = (val <= 1e-9) ? GPU_MAX_DOUBLE : log(val); break;
                            case 'e': result = (val > 700.0) ? GPU_MAX_DOUBLE : exp(val); break;
                            case '!': result = (val < 0 || val > 170.0) ? GPU_MAX_DOUBLE : tgamma(val + 1.0); break;
                            case '_': result = floor(val); break;
                            case 'g': result = (val <= -1.0) ? GPU_MAX_DOUBLE : lgamma(val + 1.0); break;
                            default: result = NAN; break;
                        }
                        stack[++stack_top] = result;
                    } else {
                        if (stack_top < 1) { valid = false; break; }
                        double right = stack[stack_top--];
                        double left = stack[stack_top--];
                        double result;
                        switch (node.op) {
                            case '+': result = left + right; break;
                            case '-': result = left - right; break;
                            case '*': result = left * right; break;
                            case '/': result = (fabs(right) < 1e-9) ? GPU_MAX_DOUBLE : left / right; break;
                            case '^': result = pow(left, right); break;
                            case '%': result = (fabs(right) < 1e-9) ? GPU_MAX_DOUBLE : fmod(left, right); break;
                            default: result = NAN; break;
                        }
                        stack[++stack_top] = result;
                    }
                }
            }

            if (!valid || stack_top != 0) {
                sum_sq_error = GPU_MAX_DOUBLE;
                valid = false;
                break;
            }

            double predicted_val = stack[0];
            if (isnan(predicted_val) || isinf(predicted_val)) {
                sum_sq_error = GPU_MAX_DOUBLE;
                valid = false;
                break;
            }

            double diff = predicted_val - d_targets[p];
            double sq_error = diff * diff;
            
            // Weighted fitness
            double weight = 1.0;
            if (GPU_USE_WEIGHTED_FITNESS) {
                weight = exp((double)p * GPU_WEIGHTED_FITNESS_EXPONENT);
            }
            total_weight += weight;
            sum_sq_error += sq_error * weight;
        }

        // Calculate final fitness with complexity penalty ON GPU
        double raw_fitness = GPU_MAX_DOUBLE;
        if (valid && sum_sq_error < 1e300) {
            if (GPU_USE_WEIGHTED_FITNESS && total_weight > 0.0) {
                sum_sq_error = sum_sq_error / total_weight * num_points;
            }
            
            if (use_rmse && num_points > 0) {
                double mse = sum_sq_error / num_points;
                raw_fitness = sqrt(mse);
            } else {
                raw_fitness = sum_sq_error;
            }
            
            // Apply complexity penalty (size is same as tree size in linearized form)
            double complexity = (double)size;
            double penalty = complexity * complexity_penalty_factor;
            raw_fitness = raw_fitness * (1.0 + penalty);
        }
        
        d_results[idx] = raw_fitness;
    }
}

void evaluate_all_populations_gpu(
    const std::vector<LinearGpuNode>& all_nodes,
    const std::vector<int>& tree_offsets,
    const std::vector<int>& tree_sizes,
    const std::vector<int>& tree_complexities,
    int total_trees,
    const std::vector<double>& targets,
    const std::vector<std::vector<double>>& x_values,
    std::vector<double>& results,
    double* d_targets, double* d_x_values,
    GlobalGpuBuffers& buffers)
{
    if (total_trees == 0) return;
    
    cudaStream_t stream = (cudaStream_t)buffers.cuda_stream;
    size_t total_nodes = all_nodes.size();
    int num_points = x_values.size();
    int num_vars = (num_points > 0) ? x_values[0].size() : 0;

    // Dynamic buffer resizing with growth factor
    if (total_nodes > buffers.d_nodes_capacity) {
        if (buffers.d_nodes) cudaFree(buffers.d_nodes);
        size_t new_cap = total_nodes * 1.5;
        cudaMalloc(&buffers.d_nodes, new_cap * sizeof(LinearGpuNode));
        buffers.d_nodes_capacity = new_cap;
    }

    if ((size_t)total_trees > buffers.d_pop_capacity) {
        if (buffers.d_offsets) cudaFree(buffers.d_offsets);
        if (buffers.d_sizes) cudaFree(buffers.d_sizes);
        if (buffers.d_results) cudaFree(buffers.d_results);
        
        size_t new_cap = total_trees * 1.5;
        cudaMalloc(&buffers.d_offsets, new_cap * sizeof(int));
        cudaMalloc(&buffers.d_sizes, new_cap * sizeof(int));
        cudaMalloc(&buffers.d_results, new_cap * sizeof(double));
        buffers.d_pop_capacity = new_cap;
    }

    LinearGpuNode* d_all_nodes = (LinearGpuNode*)buffers.d_nodes;
    int* d_offsets = (int*)buffers.d_offsets;
    int* d_sizes = (int*)buffers.d_sizes;
    double* d_results = (double*)buffers.d_results;

    // Async memory transfers using CUDA stream
    cudaMemcpyAsync(d_all_nodes, all_nodes.data(), total_nodes * sizeof(LinearGpuNode), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_offsets, tree_offsets.data(), total_trees * sizeof(int), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sizes, tree_sizes.data(), total_trees * sizeof(int), 
                    cudaMemcpyHostToDevice, stream);

    // Optimized kernel launch configuration for RTX 3050
    // RTX 3050 has 20 SMs, each can handle 2048 threads max
    // For 50k trees, we want maximum occupancy
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_trees + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel on stream
    evaluate_all_populations_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_all_nodes, d_offsets, d_sizes, total_trees,
        d_targets, d_x_values, num_points, num_vars, d_results,
        COMPLEXITY_PENALTY_FACTOR, USE_RMSE_FITNESS
    );

    // Async copy results back
    cudaMemcpyAsync(results.data(), d_results, total_trees * sizeof(double), 
                    cudaMemcpyDeviceToHost, stream);
    
    // Synchronize stream
    cudaStreamSynchronize(stream);
}

// ============================================================
// DOUBLE-BUFFERED GPU IMPLEMENTATION
// ============================================================

void init_double_buffered_gpu(DoubleBufferedGpu& db) {
    // Create two streams for overlapped execution
    cudaStreamCreate((cudaStream_t*)&db.streams[0]);
    cudaStreamCreate((cudaStream_t*)&db.streams[1]);
    
    // Pre-allocate buffers for both ping and pong
    size_t initial_nodes_cap = 1500000;  // 50k trees * 30 nodes avg
    size_t initial_pop_cap = 60000;      // Slightly more than 50k
    
    for (int i = 0; i < 2; ++i) {
        db.d_nodes_capacity[i] = initial_nodes_cap;
        db.d_pop_capacity[i] = initial_pop_cap;
        
        cudaMalloc(&db.d_nodes[i], initial_nodes_cap * sizeof(LinearGpuNode));
        cudaMalloc(&db.d_offsets[i], initial_pop_cap * sizeof(int));
        cudaMalloc(&db.d_sizes[i], initial_pop_cap * sizeof(int));
        cudaMalloc(&db.d_results[i], initial_pop_cap * sizeof(double));
    }
    
    // Allocate pinned host memory for faster H2D/D2H transfers
    db.h_pinned_capacity = initial_pop_cap;
    cudaMallocHost(&db.h_pinned_results, initial_pop_cap * sizeof(double));
    
    db.current_buffer = 0;
}

void cleanup_double_buffered_gpu(DoubleBufferedGpu& db) {
    for (int i = 0; i < 2; ++i) {
        if (db.streams[i]) {
            cudaStreamDestroy((cudaStream_t)db.streams[i]);
            db.streams[i] = nullptr;
        }
        if (db.d_nodes[i]) { cudaFree(db.d_nodes[i]); db.d_nodes[i] = nullptr; }
        if (db.d_offsets[i]) { cudaFree(db.d_offsets[i]); db.d_offsets[i] = nullptr; }
        if (db.d_sizes[i]) { cudaFree(db.d_sizes[i]); db.d_sizes[i] = nullptr; }
        if (db.d_results[i]) { cudaFree(db.d_results[i]); db.d_results[i] = nullptr; }
    }
    
    if (db.h_pinned_results) {
        cudaFreeHost(db.h_pinned_results);
        db.h_pinned_results = nullptr;
    }
}

void launch_evaluation_async(
    const std::vector<LinearGpuNode>& all_nodes,
    const std::vector<int>& tree_offsets,
    const std::vector<int>& tree_sizes,
    int total_trees,
    double* d_targets, double* d_x_values,
    int num_points,
    int num_vars,
    DoubleBufferedGpu& db)
{
    if (total_trees == 0) return;
    
    int buf = db.current_buffer;
    cudaStream_t stream = (cudaStream_t)db.streams[buf];
    size_t total_nodes = all_nodes.size();
    
    // Ensure buffers are large enough
    if (total_nodes > db.d_nodes_capacity[buf]) {
        cudaFree(db.d_nodes[buf]);
        size_t new_cap = total_nodes * 1.5;
        cudaMalloc(&db.d_nodes[buf], new_cap * sizeof(LinearGpuNode));
        db.d_nodes_capacity[buf] = new_cap;
    }
    
    if ((size_t)total_trees > db.d_pop_capacity[buf]) {
        cudaFree(db.d_offsets[buf]);
        cudaFree(db.d_sizes[buf]);
        cudaFree(db.d_results[buf]);
        
        size_t new_cap = total_trees * 1.5;
        cudaMalloc(&db.d_offsets[buf], new_cap * sizeof(int));
        cudaMalloc(&db.d_sizes[buf], new_cap * sizeof(int));
        cudaMalloc(&db.d_results[buf], new_cap * sizeof(double));
        db.d_pop_capacity[buf] = new_cap;
    }
    
    // Ensure pinned results buffer is large enough
    if ((size_t)total_trees > db.h_pinned_capacity) {
        cudaFreeHost(db.h_pinned_results);
        db.h_pinned_capacity = total_trees * 1.5;
        cudaMallocHost(&db.h_pinned_results, db.h_pinned_capacity * sizeof(double));
    }
    
    LinearGpuNode* d_nodes = (LinearGpuNode*)db.d_nodes[buf];
    int* d_offsets = (int*)db.d_offsets[buf];
    int* d_sizes = (int*)db.d_sizes[buf];
    double* d_results = (double*)db.d_results[buf];
    
    // Async transfers
    cudaMemcpyAsync(d_nodes, all_nodes.data(), total_nodes * sizeof(LinearGpuNode), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_offsets, tree_offsets.data(), total_trees * sizeof(int), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sizes, tree_sizes.data(), total_trees * sizeof(int), 
                    cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_trees + threadsPerBlock - 1) / threadsPerBlock;
    
    evaluate_all_populations_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_nodes, d_offsets, d_sizes, total_trees,
        d_targets, d_x_values, num_points, num_vars, d_results,
        COMPLEXITY_PENALTY_FACTOR, USE_RMSE_FITNESS
    );
    
    // Async copy results to pinned memory
    cudaMemcpyAsync(db.h_pinned_results, d_results, total_trees * sizeof(double), 
                    cudaMemcpyDeviceToHost, stream);
    
    // DO NOT SYNC HERE - let CPU do other work
}

void retrieve_results_sync(
    std::vector<double>& results,
    int total_trees,
    DoubleBufferedGpu& db)
{
    int buf = db.current_buffer;
    cudaStream_t stream = (cudaStream_t)db.streams[buf];
    
    // Wait for this stream to complete
    cudaStreamSynchronize(stream);
    
    // Copy from pinned memory to results vector (this is very fast - memory to memory)
    results.resize(total_trees);
    memcpy(results.data(), db.h_pinned_results, total_trees * sizeof(double));
    
    // Switch to other buffer for next generation
    db.current_buffer = 1 - buf;
}

#endif // USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
