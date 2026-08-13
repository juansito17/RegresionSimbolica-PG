#include "GradientOptimizerGPU.cuh"
#include "FitnessGPU.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <map>

// --- Helper Functions and Structs ---

struct GradientNode {
    NodeType type;
    double value;
    int var_index;
    char op;
    int left_idx; // Index in the linearized array
    int right_idx;
};

// Helper to fill Indices
void linearize_for_gpu_grad(NodePtr node, std::vector<GradientNode>& linear_nodes, std::map<Node*, int>& ptr_to_idx) {
    if (!node) return;
    
    // Post-order implies children first
    if (node->left) linearize_for_gpu_grad(node->left, linear_nodes, ptr_to_idx);
    if (node->right) linearize_for_gpu_grad(node->right, linear_nodes, ptr_to_idx);
    
    GradientNode gn;
    gn.type = node->type;
    gn.value = node->value;
    gn.var_index = node->var_index;
    gn.op = node->op;
    gn.left_idx = (node->left) ? ptr_to_idx[node->left.get()] : -1;
    gn.right_idx = (node->right) ? ptr_to_idx[node->right.get()] : -1;
    
    ptr_to_idx[node.get()] = linear_nodes.size();
    linear_nodes.push_back(gn);
}

// --- CUDA KERNELS ---

#define GPU_MAX_DOUBLE 1e308

__global__ void compute_gradients_kernel(
    const GradientNode* __restrict__ d_nodes,
    int tree_size,
    const double* __restrict__ d_targets,
    const double* __restrict__ d_x_values, // Flattened
    int num_points,
    int num_vars,
    double* d_activations, // [num_points * tree_size]
    double* d_node_grads,  // [num_points * tree_size] - Temporary gradients per sample
    double* d_const_grads_accum // [tree_size] - Accumulated gradients for constants
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    int act_offset = idx * tree_size;
    
    // 1. FORWARD PASS (Post-Order: Children guaranteed to be processed before parents)
    for (int i = 0; i < tree_size; ++i) {
        GradientNode n = d_nodes[i];
        double val = 0.0;
        
        if (n.type == NodeType::Constant) {
            val = n.value;
        } else if (n.type == NodeType::Variable) {
             int v_idx = n.var_index;
             if (v_idx >= num_vars) v_idx = 0;
             val = d_x_values[idx * num_vars + v_idx];
        } else if (n.type == NodeType::Operator) {
            double left = (n.left_idx >= 0) ? d_activations[act_offset + n.left_idx] : 0.0;
            double right = (n.right_idx >= 0) ? d_activations[act_offset + n.right_idx] : 0.0;
            
            switch(n.op) {
                case '+': val = left + right; break;
                case '-': val = left - right; break;
                case '*': val = left * right; break;
                case '/': val = (fabs(right) > 1e-9) ? left / right : 0.0; break;
                case '^': val = pow(left, right); break;
                case 's': val = sin(left); break;
                case 'c': val = cos(left); break;
                case 'e': val = exp(min(left, 20.0)); break;
                case 'l': val = (left > 1e-9) ? log(left) : -20.0; break;
                default: val = 0.0; break;
            }
        }
        
        if (isnan(val) || isinf(val)) val = 0.0; // Safety
        d_activations[act_offset + i] = val;
    }
    
    // 2. BACKWARD PASS (Reverse Post-Order: Parents before children)
    // Initialize gradients for this sample to 0
    // d_node_grads is reused, need to clear it? No, we overwrite/add carefully.
    // Actually simpler to zero out first.
    for (int i = 0; i < tree_size; ++i) d_node_grads[act_offset + i] = 0.0;
    
    // Loss Output Gradient
    double pred = d_activations[act_offset + tree_size - 1]; // Root is at end
    double diff = pred - d_targets[idx];
    d_node_grads[act_offset + tree_size - 1] = 2.0 * diff; // MSE Derivative
    
    for (int i = tree_size - 1; i >= 0; --i) {
        GradientNode n = d_nodes[i];
        double grad = d_node_grads[act_offset + i];
        
        if (fabs(grad) < 1e-15) continue; // Skip small
        
        if (n.type == NodeType::Constant) {
            // Accumulate to global constant gradient
             atomicAdd(&d_const_grads_accum[i], grad);
        } else if (n.type == NodeType::Operator) {
             double left = (n.left_idx >= 0) ? d_activations[act_offset + n.left_idx] : 0.0;
             double right = (n.right_idx >= 0) ? d_activations[act_offset + n.right_idx] : 0.0;
             
             double d_left = 0.0, d_right = 0.0;
             
             switch(n.op) {
                case '+': d_left = 1.0; d_right = 1.0; break;
                case '-': d_left = 1.0; d_right = -1.0; break;
                case '*': d_left = right; d_right = left; break;
                case '/': 
                    if (fabs(right) > 1e-9) { 
                        d_left = 1.0/right; 
                        d_right = -left/(right*right); 
                    } 
                    break;
                case '^':
                    if (left > 1e-9) {
                        d_left = right * pow(left, right - 1.0);
                        d_right = pow(left, right) * log(left);
                    }
                    break;
                case 's': d_left = cos(left); break;
                case 'c': d_left = -sin(left); break;
                case 'e': d_left = exp(min(left, 20.0)); break;
                case 'l': if (left > 1e-9) d_left = 1.0/left; break;
             }
             
             // Propagate
             if (n.left_idx >= 0) atomicAdd(&d_node_grads[act_offset + n.left_idx], grad * d_left);
             if (n.right_idx >= 0) atomicAdd(&d_node_grads[act_offset + n.right_idx], grad * d_right);
        }
    }
}

// --- HOST IMPLEMENTATION ---

void optimize_constants_gradient_gpu_impl(
    NodePtr& tree,
    const std::vector<double>& targets,
    const std::vector<std::vector<double>>& x_values,
    double learning_rate,
    int iterations
) {
    if (!tree || targets.empty()) return;
    
    // 1. Linearize Tree
    std::vector<GradientNode> host_nodes;
    std::map<Node*, int> ptr_to_idx;
    linearize_for_gpu_grad(tree, host_nodes, ptr_to_idx);
    
    int tree_size = host_nodes.size();
    int num_points = targets.size();
    int num_vars = (x_values.empty()) ? 0 : x_values[0].size();
    
    // Identify constants (map linear index -> host pointer)
    std::vector<int> constant_indices;
    std::vector<Node*> constant_ptrs;
    for(int i=0; i<tree_size; ++i) {
        if (host_nodes[i].type == NodeType::Constant) {
            constant_indices.push_back(i);
            // Need to reverse map index to NodePtr to update the original tree
            // Since we built in post-order, and we have ptr_to_idx, we can just rebuild 
            // the association or traverse.
            // ptr_to_idx maps PTR -> IDX. We iterate IDX 0..N.
            // We need IDX -> PTR.
        }
    }
    // Reconstruct index -> ptr map
    std::vector<Node*> idx_to_ptr(tree_size);
    for(auto const& [ptr, idx] : ptr_to_idx) {
        idx_to_ptr[idx] = ptr;
    }
    
    // 2. Allocate GPU Memory
    GradientNode* d_nodes;
    double *d_targets, *d_x, *d_activations, *d_node_grads, *d_const_grads_accum;
    
    cudaMalloc(&d_nodes, tree_size * sizeof(GradientNode));
    cudaMalloc(&d_targets, num_points * sizeof(double));
    cudaMalloc(&d_x, num_points * num_vars * sizeof(double));
    
    // Flatten X
    std::vector<double> flat_x;
    flat_x.reserve(num_points * num_vars);
    for(const auto& row : x_values) flat_x.insert(flat_x.end(), row.begin(), row.end());
    
    cudaMemcpy(d_nodes, host_nodes.data(), tree_size * sizeof(GradientNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, flat_x.data(), flat_x.size() * sizeof(double), cudaMemcpyHostToDevice);
    
    // Scratchpad
    cudaMalloc(&d_activations, num_points * tree_size * sizeof(double));
    cudaMalloc(&d_node_grads, num_points * tree_size * sizeof(double));
    cudaMalloc(&d_const_grads_accum, tree_size * sizeof(double)); // Accumulator
    
    // Adam Params (Host)
    std::vector<double> m(tree_size, 0.0);
    std::vector<double> v(tree_size, 0.0);
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int iter = 1; iter <= iterations; ++iter) {
        // Zero accumulators
        cudaMemset(d_const_grads_accum, 0, tree_size * sizeof(double));
        
        // Launch Kernel
        compute_gradients_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_nodes, tree_size, d_targets, d_x, num_points, num_vars,
            d_activations, d_node_grads, d_const_grads_accum
        );
        cudaDeviceSynchronize();
        
        // Copy Gradients back
        std::vector<double> host_grads(tree_size);
        cudaMemcpy(host_grads.data(), d_const_grads_accum, tree_size * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Update Constants (Adam) on Host
        for (int i : constant_indices) {
            double grad = host_grads[i] / num_points; // Average
             
             // Clip
            if (grad > 10.0) grad = 10.0;
            if (grad < -10.0) grad = -10.0;
            
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
            
            double m_hat = m[i] / (1.0 - pow(beta1, iter));
            double v_hat = v[i] / (1.0 - pow(beta2, iter));
            double step = learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            
            // Update Host Node
            idx_to_ptr[i]->value -= step;
            host_nodes[i].value = idx_to_ptr[i]->value; // Update linear copy for next iter
        }
        
        // Upload updated constants to GPU for next iteration (Forward pass needs new values)
        cudaMemcpy(d_nodes, host_nodes.data(), tree_size * sizeof(GradientNode), cudaMemcpyHostToDevice);
    }
    
    // Cleanup
    cudaFree(d_nodes);
    cudaFree(d_targets);
    cudaFree(d_x);
    cudaFree(d_activations);
    cudaFree(d_node_grads);
    cudaFree(d_const_grads_accum);
}
