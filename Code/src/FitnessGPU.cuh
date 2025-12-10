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
    char op;
};

// Helper function to linearize the tree into a post-order array
void linearize_tree(const NodePtr& node, std::vector<LinearGpuNode>& linear_tree);

// Host-side wrapper for launching CUDA kernel
#if USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
double evaluate_fitness_gpu(NodePtr tree,
                            const std::vector<double>& targets,
                            const std::vector<double>& x_values,
                            double* d_targets, double* d_x_values);

// Batch evaluation function with persistent buffers
void evaluate_population_gpu(const std::vector<LinearGpuNode>& all_nodes,
                             const std::vector<int>& tree_offsets,
                             const std::vector<int>& tree_sizes,
                             const std::vector<double>& targets,
                             const std::vector<double>& x_values,
                             std::vector<double>& results,
                             double* d_targets, double* d_x_values,
                             void*& d_nodes_ptr, size_t& d_nodes_cap,
                             void*& d_offsets_ptr, void*& d_sizes_ptr, void*& d_results_ptr, size_t& d_pop_cap);
#endif

#endif // FITNESS_GPU_CUH
