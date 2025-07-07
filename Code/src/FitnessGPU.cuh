#ifndef FITNESS_GPU_CUH
#define FITNESS_GPU_CUH

#include <vector>
#include <memory> // For NodePtr in the host-side wrapper
#include "ExpressionTree.h" // For NodeType enum and original Node structure (host-side)
#include "Globals.h" // For INF, USE_RMSE_FITNESS, COMPLEXITY_PENALTY_FACTOR etc.

// Forward declaration for host-side NodePtr
struct Node;
using NodePtr = std::shared_ptr<Node>;

// Define a GPU-compatible Node struct
// This struct will be used for computations on the device.
// It uses raw pointers for children, as shared_ptr is not supported on device.
struct GpuNode {
    NodeType type;
    double value;
    char op;
    GpuNode* left;
    GpuNode* right;

#ifdef __CUDACC__
    // Default constructor for __device__ usage
    __host__ __device__ GpuNode(NodeType t = NodeType::Constant, double val = 0.0, char op_char = 0)
        : type(t), value(val), op(op_char), left(nullptr), right(nullptr) {}
#endif
};


#ifdef __CUDACC__
// __device__ function for evaluating a tree on the GPU
__device__ double evaluate_tree_gpu(GpuNode* tree_node_ptr, double x_val);
#endif

// Host-side wrapper for launching CUDA kernel
double evaluate_fitness_gpu(NodePtr tree,
                            const std::vector<double>& targets,
                            const std::vector<double>& x_values);

#endif // FITNESS_GPU_CUH
