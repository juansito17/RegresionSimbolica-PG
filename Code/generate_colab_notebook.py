import json
import os

source_dir = 'src'
files = [
    'AdvancedFeatures.cpp', 'AdvancedFeatures.h',
    'ExpressionTree.cpp', 'ExpressionTree.h',
    'Fitness.cpp', 'Fitness.h',
    'FitnessGPU.cu', 'FitnessGPU.cuh',
    'GeneticAlgorithm.cpp', 'GeneticAlgorithm.h',
    'GeneticOperators.cpp', 'GeneticOperators.h',
    'main.cpp'
]

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 1. Header
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Symbolic Regression GP (CUDA Enabled)\n",
        "## Instructions\n",
        "1. Go to **Runtime -> Change runtime type** and select **T4 GPU** (or any available GPU).\n",
        "2. Run all cells to compile and execute the project.\n"
    ]
})

# 2. Check GPU
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!nvidia-smi"]
})

# 3. Create directories
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "os.makedirs('src', exist_ok=True)\n",
        "os.makedirs('build', exist_ok=True)"
    ]
})

# 4. Create Files (One Cell for All)
files_data = {}

# Read CMakeLists.txt
try:
    content = open('CMakeLists.txt', 'r', encoding='utf-8').read()
    files_data['CMakeLists.txt'] = content
except Exception as e:
    print(f"Error reading CMakeLists.txt: {e}")

# Read Source Files
for filename in files:
    path = os.path.join(source_dir, filename)
    if os.path.exists(path):
        try:
            content = open(path, 'r', encoding='utf-8').read()
            files_data[f"src/{filename}"] = content
        except Exception as e:
             print(f"Error reading {filename}: {e}")
    else:
        print(f"Warning: {filename} not found.")

# Apply specific content change for src/FitnessGPU.cu
files_data["src/FitnessGPU.cu"] = """#include "FitnessGPU.cuh"
#include "Globals.h"
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <cstdio>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

// Constant for large finite value
#define GPU_MAX_DOUBLE 1e308

// Single Tree Evaluation Kernel (Legacy/Single Use)
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
                            result = GPU_MAX_DOUBLE; 
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
            d_raw_fitness_results[idx] = GPU_MAX_DOUBLE; 
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
    }\n
    if (tid == 0) { // Write result back to global memory (first element of block)
        d_data[blockIdx.x] = sdata[0];
    }
}


// --- New Batch Kernel ---
// Evaluates one tree per thread across all data points
__global__ void evaluate_population_kernel(const LinearGpuNode* d_all_nodes,
                                           const int* d_offsets,
                                           const int* d_sizes,
                                           int pop_size,
                                           const double* d_targets,
                                           const double* d_x_values,
                                           int num_points,
                                           double* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pop_size) {
        int offset = d_offsets[idx];
        int size = d_sizes[idx];
        double sum_sq_error = 0.0;
        bool valid = true;

        for (int p = 0; p < num_points; ++p) {
            double x_val = d_x_values[p];
            double stack[64]; 
            int stack_top = -1;

            // Simple interpreter
            for (int i = 0; i < size; ++i) {
                LinearGpuNode node = d_all_nodes[offset + i];
                if (node.type == NodeType::Constant) {
                    stack[++stack_top] = node.value;
                } else if (node.type == NodeType::Variable) {
                    stack[++stack_top] = x_val;
                } else if (node.type == NodeType::Operator) {
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
                        default: result = NAN; break;
                    }
                    stack[++stack_top] = result;
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
            sum_sq_error += diff * diff;
        }

        d_results[idx] = sum_sq_error;
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

    gpuErrchk(cudaMalloc((void**)&d_linear_tree, tree_size * sizeof(LinearGpuNode)));
    gpuErrchk(cudaMalloc((void**)&d_raw_fitness_results, num_points * sizeof(double)));

    gpuErrchk(cudaMemcpy(d_linear_tree, h_linear_tree.data(), tree_size * sizeof(LinearGpuNode), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to calculate individual squared errors
    size_t shared_mem_size = tree_size * sizeof(LinearGpuNode);
    calculate_raw_fitness_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        d_linear_tree, tree_size, d_targets, d_x_values, num_points, d_raw_fitness_results
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); // Ensure kernel completes before reduction

    // --- Perform reduction on the GPU ---
    int current_size = num_points;
    while (current_size > 1) {
        int next_blocks_per_grid = (current_size + threadsPerBlock - 1) / threadsPerBlock;
        // Use shared memory for reduction, size is threadsPerBlock * sizeof(double)
        reduce_sum_kernel<<<next_blocks_per_grid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(
            d_raw_fitness_results, current_size
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize()); // Ensure reduction step completes
        current_size = next_blocks_per_grid; // The result is in the first `next_blocks_per_grid` elements
    }

    double sum_sq_error_gpu = 0.0;
    gpuErrchk(cudaMemcpy(&sum_sq_error_gpu, d_raw_fitness_results, sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_linear_tree));
    gpuErrchk(cudaFree(d_raw_fitness_results));

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
                             const std::vector<double>& x_values,
                             std::vector<double>& results,
                             double* d_targets, double* d_x_values,
                             void*& d_nodes_ptr, size_t& d_nodes_cap,
                             void*& d_offsets_ptr, void*& d_sizes_ptr, void*& d_results_ptr, size_t& d_pop_cap) {
    
    int pop_size = tree_offsets.size();
    if (pop_size == 0) return;

    size_t total_nodes = all_nodes.size();
    int num_points = x_values.size();

    // Buffer Management for Nodes
    if (total_nodes > d_nodes_cap) {
        if (d_nodes_ptr) gpuErrchk(cudaFree(d_nodes_ptr));
        size_t new_cap = total_nodes * 1.5; // Growth factor
        gpuErrchk(cudaMalloc(&d_nodes_ptr, new_cap * sizeof(LinearGpuNode)));
        d_nodes_cap = new_cap;
    }

    // Buffer Management for Population Arrays
    if (pop_size > d_pop_cap) {
        if (d_offsets_ptr) gpuErrchk(cudaFree(d_offsets_ptr));
        if (d_sizes_ptr) gpuErrchk(cudaFree(d_sizes_ptr));
        if (d_results_ptr) gpuErrchk(cudaFree(d_results_ptr));
        
        size_t new_cap = pop_size * 1.5;
        gpuErrchk(cudaMalloc(&d_offsets_ptr, new_cap * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_sizes_ptr, new_cap * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_results_ptr, new_cap * sizeof(double)));
        d_pop_cap = new_cap;
    }

    LinearGpuNode* d_all_nodes = (LinearGpuNode*)d_nodes_ptr;
    int* d_offsets = (int*)d_offsets_ptr;
    int* d_sizes = (int*)d_sizes_ptr;
    double* d_results = (double*)d_results_ptr;

    gpuErrchk(cudaMemcpy(d_all_nodes, all_nodes.data(), total_nodes * sizeof(LinearGpuNode), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_offsets, tree_offsets.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_sizes, tree_sizes.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    evaluate_population_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_all_nodes, d_offsets, d_sizes, pop_size, d_targets, d_x_values, num_points, d_results
    );
    gpuErrchk(cudaPeekAtLastError());

    // Synchronize and copy back
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(results.data(), d_results, pop_size * sizeof(double), cudaMemcpyDeviceToHost));
}

#endif // USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
"""

# Create the Python code for the cell
# We use json.dumps to handle escaping of strings safely
file_creation_code = [
    "import os",
    "",
    "files_to_create = " + json.dumps(files_data, indent=2),
    "",
    "for filepath, content in files_to_create.items():",
    "    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None",
    "    with open(filepath, 'w', encoding='utf-8') as f:",
    "        f.write(content)",
    "    print(f'Created: {filepath}')"
]

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["\n".join(file_creation_code)]
})

# 5. Build
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Configuration & Compilation"]
})

# Cell to define parameters and write Globals.h
globals_code = [
    "# @title Algorithm Parameters",
    "# @markdown Modify these values to tune the genetic algorithm.",
    "",
    "import os",
    "",
    "# Problem Data",
    "RAW_TARGETS_STR = \"2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884\" # @param {type:\"string\"}",
    "X_VALUES_STR = \"4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20\" # @param {type:\"string\"}",
    "USE_LOG_TRANSFORMATION = True # @param {type:\"boolean\"}",
    "",
    "# Algorithm Settings",
    "USE_GPU_ACCELERATION = True # @param {type:\"boolean\"}",
    "TOTAL_POPULATION_SIZE = 50000 # @param {type:\"integer\"}",
    "GENERATIONS = 500000 # @param {type:\"integer\"}",
    "NUM_ISLANDS = 10 # @param {type:\"integer\"}",
    "MIGRATION_INTERVAL = 100 # @param {type:\"integer\"}",
    "MIGRATION_SIZE = 50 # @param {type:\"integer\"}",
    "",
    "# Genetic Operators",
    "BASE_MUTATION_RATE = 0.30 # @param {type:\"number\"}",
    "BASE_ELITE_PERCENTAGE = 0.15 # @param {type:\"number\"}",
    "",
    "# Operator Selection",
    "USE_OP_PLUS = True # @param {type:\"boolean\"}",
    "USE_OP_MINUS = True # @param {type:\"boolean\"}",
    "USE_OP_MULT = True # @param {type:\"boolean\"}",
    "USE_OP_DIV = True # @param {type:\"boolean\"}",
    "USE_OP_POW = True # @param {type:\"boolean\"}",
    "USE_OP_MOD = True # @param {type:\"boolean\"}",
    "USE_OP_SIN = True # @param {type:\"boolean\"}",
    "USE_OP_COS = True # @param {type:\"boolean\"}",
    "USE_OP_LOG = True # @param {type:\"boolean\"}",
    "USE_OP_EXP = True # @param {type:\"boolean\"}",
    "USE_OP_FACT = True # @param {type:\"boolean\"}",
    "USE_OP_FLOOR = True # @param {type:\"boolean\"}",
    "USE_OP_GAMMA = True # @param {type:\"boolean\"}",
    "",
    "# Initial Formula Injection",
    "USE_INITIAL_FORMULA = True # @param {type:\"boolean\"}",
    "INITIAL_FORMULA_STRING = \"l(g(x+1))-(x*0.92)\" # @param {type:\"string\"}",
    "",
    "# Optimization Settings",
    "FORCE_INTEGER_CONSTANTS = False # @param {type:\"boolean\"}",
    "USE_SIMPLIFICATION = True # @param {type:\"boolean\"}",
    "USE_ISLAND_CATACLYSM = True # @param {type:\"boolean\"}",
    "",
    "# Weighted Fitness",
    "USE_WEIGHTED_FITNESS = True # @param {type:\"boolean\"}",
    "WEIGHTED_FITNESS_EXPONENT = 0.25 # @param {type:\"number\"}",
    "",
    "# Construct Globals.h content",
    "globals_content = f\"\"\"",
    "#ifndef GLOBALS_H",
    "#define GLOBALS_H",
    "",
    "#include <vector>",
    "#include <random>",
    "#include <string>",
    "#include <limits>",
    "#include <cmath>",
    "",
    "// Data",
    "const std::vector<double> RAW_TARGETS = {{ {RAW_TARGETS_STR} }};",
    "const std::vector<double> X_VALUES = {{ {X_VALUES_STR} }};",
    "const bool USE_LOG_TRANSFORMATION = {'true' if USE_LOG_TRANSFORMATION else 'false'};",
    "",
    "// GPU Settings",
    "#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE",
    "const bool USE_GPU_ACCELERATION = {'true' if USE_GPU_ACCELERATION else 'false'};",
    "#else",
    "const bool USE_GPU_ACCELERATION = false;",
    "#endif",
    "",
    "// Algorithm Parameters",
    "const int TOTAL_POPULATION_SIZE = {TOTAL_POPULATION_SIZE};",
    "const int GENERATIONS = {GENERATIONS};",
    "const int NUM_ISLANDS = {NUM_ISLANDS};",
    "const int MIN_POP_PER_ISLAND = 10;",
    "",
    "// Migration",
    "const int MIGRATION_INTERVAL = {MIGRATION_INTERVAL};",
    "const int MIGRATION_SIZE = {MIGRATION_SIZE};",
    "",
    "// Initial Formula",
    "const bool USE_INITIAL_FORMULA = {'true' if USE_INITIAL_FORMULA else 'false'};",
    "const std::string INITIAL_FORMULA_STRING = \"{INITIAL_FORMULA_STRING}\";",
    "",
    "// Genetic Operators",
    "const double BASE_MUTATION_RATE = {BASE_MUTATION_RATE};",
    "const double BASE_ELITE_PERCENTAGE = {BASE_ELITE_PERCENTAGE};",
    "const double DEFAULT_CROSSOVER_RATE = 0.85;",
    "const int DEFAULT_TOURNAMENT_SIZE = 30;",
    "const int MAX_TREE_DEPTH_MUTATION = 8;",
    "",
    "// Operator Configuration",
    "const bool USE_OP_PLUS     = {'true' if USE_OP_PLUS else 'false'};",
    "const bool USE_OP_MINUS    = {'true' if USE_OP_MINUS else 'false'};",
    "const bool USE_OP_MULT     = {'true' if USE_OP_MULT else 'false'};",
    "const bool USE_OP_DIV      = {'true' if USE_OP_DIV else 'false'};",
    "const bool USE_OP_POW      = {'true' if USE_OP_POW else 'false'};",
    "const bool USE_OP_MOD      = {'true' if USE_OP_MOD else 'false'};",
    "const bool USE_OP_SIN      = {'true' if USE_OP_SIN else 'false'};",
    "const bool USE_OP_COS      = {'true' if USE_OP_COS else 'false'};",
    "const bool USE_OP_LOG      = {'true' if USE_OP_LOG else 'false'};",
    "const bool USE_OP_EXP      = {'true' if USE_OP_EXP else 'false'};",
    "const bool USE_OP_FACT     = {'true' if USE_OP_FACT else 'false'};",
    "const bool USE_OP_FLOOR    = {'true' if USE_OP_FLOOR else 'false'};",
    "const bool USE_OP_GAMMA    = {'true' if USE_OP_GAMMA else 'false'};",
    "",
    "// Tree Generation",
    "const int MAX_TREE_DEPTH_INITIAL = 8;",
    "const double TERMINAL_VS_VARIABLE_PROB = 0.75;",
    "const double CONSTANT_MIN_VALUE = -10.0;",
    "const double CONSTANT_MAX_VALUE = 10.0;",
    "const int CONSTANT_INT_MIN_VALUE = -10;",
    "const int CONSTANT_INT_MAX_VALUE = 10;",
    "// Order: +, -, *, /, ^, %, s, c, l, e, !, _, g",
    "const std::vector<double> OPERATOR_WEIGHTS = {{",
    "    0.10 * (USE_OP_PLUS  ? 1.0 : 0.0),",
    "    0.15 * (USE_OP_MINUS ? 1.0 : 0.0),",
    "    0.10 * (USE_OP_MULT  ? 1.0 : 0.0),",
    "    0.10 * (USE_OP_DIV   ? 1.0 : 0.0),",
    "    0.05 * (USE_OP_POW   ? 1.0 : 0.0),",
    "    0.01 * (USE_OP_MOD   ? 1.0 : 0.0),",
    "    0.01 * (USE_OP_SIN   ? 1.0 : 0.0),",
    "    0.01 * (USE_OP_COS   ? 1.0 : 0.0),",
    "    0.15 * (USE_OP_LOG   ? 1.0 : 0.0),",
    "    0.02 * (USE_OP_EXP   ? 1.0 : 0.0),",
    "    0.05 * (USE_OP_FACT  ? 1.0 : 0.0),",
    "    0.05 * (USE_OP_FLOOR ? 1.0 : 0.0),",
    "    0.20 * (USE_OP_GAMMA ? 1.0 : 0.0)",
    "}};",
    "",
    "// Fitness & Other",
    "const double COMPLEXITY_PENALTY_FACTOR = 0.05;",
    "const bool USE_RMSE_FITNESS = true;",
    "const double FITNESS_ORIGINAL_POWER = 1.3;",
    "const double FITNESS_PRECISION_THRESHOLD = 0.001;",
    "const double FITNESS_PRECISION_BONUS = 0.0001;",
    "const double FITNESS_EQUALITY_TOLERANCE = 1e-9;",

    "const double EXACT_SOLUTION_THRESHOLD = 1e-8;",
    "",
    "// Weighted Fitness",
    "const bool USE_WEIGHTED_FITNESS = {'true' if USE_WEIGHTED_FITNESS else 'false'};",
    "const double WEIGHTED_FITNESS_EXPONENT = {WEIGHTED_FITNESS_EXPONENT};",
    "",
    "// Advanced Features",
    "const int STAGNATION_LIMIT_ISLAND = 50;",
    "const int GLOBAL_STAGNATION_LIMIT = 5000;",
    "const double STAGNATION_RANDOM_INJECT_PERCENT = 0.1;",
    "const int PARAM_MUTATE_INTERVAL = 50;",
    "const double PATTERN_RECORD_FITNESS_THRESHOLD = 10.0;",
    "const int PATTERN_MEM_MIN_USES = 3;",
    "const int PATTERN_INJECT_INTERVAL = 10;",
    "const double PATTERN_INJECT_PERCENT = 0.05;",

    "const size_t PARETO_MAX_FRONT_SIZE = 50;",
    "const double SIMPLIFY_NEAR_ZERO_TOLERANCE = 1e-9;",
    "const double SIMPLIFY_NEAR_ONE_TOLERANCE = 1e-9;",
    "const int LOCAL_SEARCH_ATTEMPTS = 30;",
    "const bool USE_SIMPLIFICATION = {'true' if USE_SIMPLIFICATION else 'false'};",
    "const bool USE_ISLAND_CATACLYSM = {'true' if USE_ISLAND_CATACLYSM else 'false'};",
    "const int PROGRESS_REPORT_INTERVAL = 100;",
    "const bool FORCE_INTEGER_CONSTANTS = {'true' if FORCE_INTEGER_CONSTANTS else 'false'};",
    "",
    "#include <random>",
    "const double MUTATE_INSERT_CONST_PROB = 0.6;",
    "const int MUTATE_INSERT_CONST_INT_MIN = 1;",
    "const int MUTATE_INSERT_CONST_INT_MAX = 5;",
    "const double MUTATE_INSERT_CONST_FLOAT_MIN = 0.5;",
    "const double MUTATE_INSERT_CONST_FLOAT_MAX = 5.0;",
    "",
    "std::mt19937& get_rng();",
    "const double INF = std::numeric_limits<double>::infinity();",
    "",
    "#endif // GLOBALS_H",
    "\"\"\"",
    "",
    "with open('src/Globals.h', 'w') as f:",
    "    f.write(globals_content)",
    "print(\"Globals.h updated with new parameters.\")"
]

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["\n".join(globals_code)]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!cmake -B build -S . -DCMAKE_BUILD_TYPE=Release\n",
        "!cmake --build build -j $(nproc)"
    ]
})

# 7. Run
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Execution"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!./build/SymbolicRegressionGP"]
})

output_file = 'GoogleColab_Project.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created successfully: {output_file}")
