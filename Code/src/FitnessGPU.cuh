#ifndef FITNESS_GPU_CUH
#define FITNESS_GPU_CUH

#include <vector>
#include <memory> // For NodePtr in the host-side wrapper
#include "ExpressionTree.h" // For NodeType enum and original Node structure (host-side)
#include "Globals.h" // For INF, USE_RMSE_FITNESS, COMPLEXITY_PENALTY_FACTOR etc.

// Forward declaration for host-side NodePtr
struct Node;
using NodePtr = std::shared_ptr<Node>;

// A simplified node structure for the linearized tree on the GPU
struct LinearGpuNode {
    NodeType type;
    double value;
    int var_index;
    char op;
};

// Helper function to linearize the tree into a post-order array
void linearize_tree(const NodePtr& node, std::vector<LinearGpuNode>& linear_tree);

// Host-side wrapper for launching CUDA kernel
#if USE_GPU_ACCELERATION_DEFINED_BY_CMAKE

// Persistent GPU buffers for global batch (managed by GeneticAlgorithm)
struct GlobalGpuBuffers {
    void* d_nodes = nullptr;
    void* d_offsets = nullptr;
    void* d_sizes = nullptr;
    void* d_results = nullptr;
    size_t d_nodes_capacity = 0;
    size_t d_pop_capacity = 0;
    void* cuda_stream = nullptr; // cudaStream_t
};

// Two sets of device buffers for ping-pong operation
struct DoubleBufferedGpu {
    void* d_nodes[2] = {nullptr, nullptr};
    void* d_offsets[2] = {nullptr, nullptr};
    void* d_sizes[2] = {nullptr, nullptr};
    void* d_results[2] = {nullptr, nullptr};
    size_t d_nodes_capacity[2] = {0, 0};
    size_t d_pop_capacity[2] = {0, 0};
    
    void* streams[2] = {nullptr, nullptr};
    int current_buffer = 0;
    
    void* h_pinned_results = nullptr;
    size_t h_pinned_capacity = 0;
};

double evaluate_fitness_gpu(NodePtr tree,
                            const std::vector<double>& targets,
                            const std::vector<std::vector<double>>& x_values,
                            double* d_targets, double* d_x_values);

// Batch evaluation function with persistent buffers
void evaluate_population_gpu(const std::vector<LinearGpuNode>& all_nodes,
                             const std::vector<int>& tree_offsets,
                             const std::vector<int>& tree_sizes,
                             const std::vector<double>& targets,
                             const std::vector<std::vector<double>>& x_values,
                             std::vector<double>& results,
                             double* d_targets, double* d_x_values,
                             void*& d_nodes_ptr, size_t& d_nodes_cap,
                             void*& d_offsets_ptr, void*& d_sizes_ptr, void*& d_results_ptr, size_t& d_pop_cap);

// Retrieves the full error matrix (PopSize x NumPoints) for Lexicase Selection
// Output: flat vector [PopSize * NumPoints]
void get_population_errors_gpu(
    const std::vector<LinearGpuNode>& all_nodes,
    const std::vector<int>& tree_offsets,
    const std::vector<int>& tree_sizes,
    const std::vector<double>& targets,
    const std::vector<std::vector<double>>& x_values,
    std::vector<double>& flat_errors, 
    double* d_targets, double* d_x_values,
    GlobalGpuBuffers& buffers);

// Initialize global GPU buffers and CUDA stream
void init_global_gpu_buffers(GlobalGpuBuffers& buffers);

// Cleanup global GPU buffers
void cleanup_global_gpu_buffers(GlobalGpuBuffers& buffers);

// Evaluate ALL trees from ALL islands in a single GPU batch call (maximum GPU utilization)
void evaluate_all_populations_gpu(
    const std::vector<LinearGpuNode>& all_nodes,
    const std::vector<int>& tree_offsets,
    const std::vector<int>& tree_sizes,
    const std::vector<int>& tree_complexities, // For complexity penalty
    int total_trees,
    const std::vector<double>& targets,
    const std::vector<std::vector<double>>& x_values,
    std::vector<double>& results,
    double* d_targets, double* d_x_values,
    GlobalGpuBuffers& buffers);

// Initialize double-buffered GPU resources
void init_double_buffered_gpu(DoubleBufferedGpu& db);

// Cleanup double-buffered GPU resources
void cleanup_double_buffered_gpu(DoubleBufferedGpu& db);

// Async launch - starts GPU work without waiting (CPU can do other work)
void launch_evaluation_async(
    const std::vector<LinearGpuNode>& all_nodes,
    const std::vector<int>& tree_offsets,
    const std::vector<int>& tree_sizes,
    int total_trees,
    double* d_targets, double* d_x_values,
    int num_points,
    int num_vars,
    DoubleBufferedGpu& db);

// Wait for GPU work to complete and retrieve results
void retrieve_results_sync(
    std::vector<double>& results,
    int total_trees,
    DoubleBufferedGpu& db);

#endif

#endif // FITNESS_GPU_CUH
