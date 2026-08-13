#include "GradientOptimizer.h"
#include "GradientOptimizerGPU.cuh" // Include GPU interface
#include <cmath>
#include <map>
#include <iostream>
#include <algorithm>
#include <random>

// Check globally defined macro from CMake
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    static bool USE_GPU_GRADIENT = true;
#else
    static bool USE_GPU_GRADIENT = false;
#endif

// Estructura auxiliar para manejar el árbol linearizado y sus valores
struct LinearNode {
    Node* node_ptr;
    std::vector<double> values;     // Forward pass values (per batch item)
    std::vector<double> gradients;  // Backward pass gradients (per batch item)
    
    // Indices de los hijos en el arreglo linearizado
    int left_idx = -1;
    int right_idx = -1;
};

// Convierte el árbol a una estructura plana (Topological Sort / Post-Order implícito si se itera al revés)
void linearize_for_gradient(NodePtr node, std::vector<LinearNode>& linear_nodes, std::map<Node*, int>& ptr_to_idx) {
    if (!node) return;
    
    LinearNode ln;
    ln.node_ptr = node.get();
    
    // Procesar hijos primero (si queremos orden bottom-up natural, pero aquí construiremos top-down y enlazaremos indices)
    // Mejor: Insertamos el nodo, luego procesamos hijos y actualizamos indices.
    
    int current_idx = linear_nodes.size();
    ptr_to_idx[node.get()] = current_idx;
    linear_nodes.push_back(ln);
    
    if (node->left) {
        linearize_for_gradient(node->left, linear_nodes, ptr_to_idx);
        linear_nodes[current_idx].left_idx = ptr_to_idx[node->left.get()];
    }
    if (node->right) {
        linearize_for_gradient(node->right, linear_nodes, ptr_to_idx);
        linear_nodes[current_idx].right_idx = ptr_to_idx[node->right.get()];
    }
}

// Forward Pass: Calcula valores para un lote de datos
void forward_pass(std::vector<LinearNode>& nodes, const std::vector<std::vector<double>>& batch_x, int batch_size) {
    // Los nodos están en orden topológico (Padre antes que hijos) según nuestra linearización recursiva DFS pre-order?
    // NO, DFS pre-order pone padre antes. Para forward necesitamos hijos antees si calculamos desde 0?
    // No, evaluate es recursivo. Si usamos iterativo, necesitamos post-order (hijos primero).
    // O podemos hacerlo recursivo simple usando los punteros, pero queremos vectorización/cache.
    
    // Vamos a usar la estructura recursiva implícita pero iterando en orden inverso (Post-Order) 
    // Si la lista se construyó en Pre-Order (Padre, Izq, Der), el reverso es (Der, Izq, Padre)? No exactamente.
    
    // Mejor reconstruyamos la lista en Post-Order puro.
    
    // ... Replanteando linearización para simplificar forward/backward ...
}

void build_post_order(NodePtr node, std::vector<Node*>& order) {
    if (!node) return;
    build_post_order(node->left, order);
    build_post_order(node->right, order);
    order.push_back(node.get());
}

// --------------------------------------------------------------------------

void optimize_constants_gradient(NodePtr& tree, 
                                 const std::vector<double>& targets, 
                                 const std::vector<std::vector<double>>& x_values,
                                 double learning_rate,
                                 int iterations) {
    if (!tree || targets.empty()) return;

#if defined(USE_GPU_ACCELERATION_DEFINED_BY_CMAKE)
    // Use GPU implementation if available and problem size warrants it (overhead > benefit for tiny problems)
    // For very small datasets, CPU might be faster, but for SOTA/Large Data -> GPU.
    optimize_constants_gradient_gpu_impl(tree, targets, x_values, learning_rate, iterations);
    return;
#endif

    // 1. Identificar constantes a optimizar (CPU Fallback)
    std::vector<Node*> constants;
    std::vector<Node*> all_nodes_post_order;
    build_post_order(tree, all_nodes_post_order);
    
    for(Node* n : all_nodes_post_order) {
        if (n->type == NodeType::Constant) {
            constants.push_back(n);
        }
    }

    if (constants.empty()) return;

    // Adam Parameters
    std::vector<double> m(constants.size(), 0.0);
    std::vector<double> v(constants.size(), 0.0);
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    
    // Batch configuration
    size_t num_samples = targets.size();
    size_t batch_size = std::min(num_samples, (size_t)64); // Mini-batch size
    size_t num_vars = (x_values.empty()) ? 0 : x_values[0].size();

    // Buffers para forward/backward (reutilizables)
    // Mapa de Nodo -> vector de valores (tamaño batch)
    // Para eficiencia, usamos índices fijos basados en 'all_nodes_post_order'
    // Map Node* -> index en all_nodes_post_order
    std::map<Node*, int> node_to_idx;
    for(size_t i=0; i<all_nodes_post_order.size(); ++i) {
        node_to_idx[all_nodes_post_order[i]] = i;
    }

    // Storage: [num_nodes][batch_size]
    std::vector<std::vector<double>> node_values(all_nodes_post_order.size(), std::vector<double>(batch_size));
    std::vector<std::vector<double>> node_gradients(all_nodes_post_order.size(), std::vector<double>(batch_size));
    
    std::vector<int> indices(num_samples);
    for(size_t i=0; i<num_samples; ++i) indices[i] = i;
    
    // RNG for shuffling
    std::random_device rd;
    std::mt19937 g(rd());

    for (int iter = 1; iter <= iterations; ++iter) {
        // Shuffle batch
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Reset gradients accumulator for constants (sum over batch)
        std::vector<double> constant_grads(constants.size(), 0.0);

        // Process mini-batch
        for (size_t b = 0; b < batch_size; ++b) {
            int sample_idx = indices[b];
            
            // --- FORWARD PASS (Post-Order) ---
            for (size_t i = 0; i < all_nodes_post_order.size(); ++i) {
                Node* n = all_nodes_post_order[i];
                double val = 0.0;
                
                if (n->type == NodeType::Constant) {
                    val = n->value;
                } else if (n->type == NodeType::Variable) {
                    int v_idx = n->var_index;
                    if (v_idx < num_vars) val = x_values[sample_idx][v_idx];
                } else if (n->type == NodeType::Operator) {
                    double left_val = (n->left) ? node_values[node_to_idx[n->left.get()]][b] : 0.0;
                    double right_val = (n->right) ? node_values[node_to_idx[n->right.get()]][b] : 0.0;
                    
                    switch(n->op) {
                        case '+': val = left_val + right_val; break;
                        case '-': val = left_val - right_val; break;
                        case '*': val = left_val * right_val; break;
                        case '/': val = (std::abs(right_val) > 1e-9) ? left_val / right_val : 0.0; break; // Safe div
                        case '^': val = std::pow(left_val, right_val); break;
                        case 's': val = std::sin(left_val); break;
                        case 'c': val = std::cos(left_val); break;
                        case 'e': val = std::exp(std::min(left_val, 20.0)); break; // Clamp exp
                        case 'l': val = (left_val > 1e-9) ? std::log(left_val) : -20.0; break;
                        default: val = 0.0; break;
                    }
                }
                if (std::isnan(val) || std::isinf(val)) val = 0.0; // Safety clamp
                node_values[i][b] = val;
            }
            
            // --- BACKWARD PASS (Pre-Order / Reverse Post-Order) ---
            // Calcular gradiente inicial dLoss/dOutput = 2 * (pred - target)
            Node* root = all_nodes_post_order.back();
            int root_idx = node_to_idx[root];
            double pred = node_values[root_idx][b];
            double diff = pred - targets[sample_idx];
            
            // Inicializar gradientes a 0
            for(auto& grad_vec : node_gradients) grad_vec[b] = 0.0;
            
            node_gradients[root_idx][b] = 2.0 * diff; // dL/dy
            
            // Iterar desde el final hacia el principio (Root -> Leaves)
            for (int i = (int)all_nodes_post_order.size() - 1; i >= 0; --i) {
                Node* n = all_nodes_post_order[i];
                double incoming_grad = node_gradients[i][b];
                
                if (std::abs(incoming_grad) < 1e-15) continue; // Skip small gradients
                
                if (n->type == NodeType::Operator) {
                    double left_val = (n->left) ? node_values[node_to_idx[n->left.get()]][b] : 0.0;
                    double right_val = (n->right) ? node_values[node_to_idx[n->right.get()]][b] : 0.0;
                    
                    double d_left = 0.0;
                    double d_right = 0.0;
                    
                    switch(n->op) {
                        case '+': 
                            d_left = 1.0; d_right = 1.0; 
                            break;
                        case '-': 
                            d_left = 1.0; d_right = -1.0; 
                            break;
                        case '*': 
                            d_left = right_val; d_right = left_val; 
                            break;
                        case '/': 
                            if (std::abs(right_val) > 1e-9) {
                                d_left = 1.0 / right_val;
                                d_right = -left_val / (right_val * right_val);
                            }
                            break;
                        case '^':
                            // y = u^v -> dy/du = v*u^(v-1), dy/dv = u^v * ln(u)
                            if (left_val > 1e-9) { // Solo soportamos bases positivas para gradiente estable por ahora
                                d_left = right_val * std::pow(left_val, right_val - 1.0);
                                d_right = std::pow(left_val, right_val) * std::log(left_val);
                            }
                            break;
                        case 's': // sin(u) -> cos(u)
                            d_left = std::cos(left_val);
                            break;
                        case 'c': // cos(u) -> -sin(u)
                            d_left = -std::sin(left_val);
                            break;
                        case 'e': // exp(u) -> exp(u)
                            d_left = std::exp(std::min(left_val, 20.0));
                            break;
                        case 'l': // log(u) -> 1/u
                            if (left_val > 1e-9) d_left = 1.0 / left_val;
                            break;
                    }
                    
                    if (n->left) {
                        int child_idx = node_to_idx[n->left.get()];
                        node_gradients[child_idx][b] += incoming_grad * d_left;
                    }
                    if (n->right) {
                        int child_idx = node_to_idx[n->right.get()];
                        node_gradients[child_idx][b] += incoming_grad * d_right;
                    }
                }
            }
        } // End mini-batch loop
        
        // --- UPDATE CONSTANTS (Adam) ---
        for (size_t c = 0; c < constants.size(); ++c) {
            Node* const_node = constants[c];
            int node_idx = node_to_idx[const_node];
            
            // Sumar gradientes del batch
            double sum_grad = 0.0;
            for(size_t b=0; b<batch_size; ++b) {
                sum_grad += node_gradients[node_idx][b];
            }
            double avg_grad = sum_grad / batch_size;
            
            // Clip gradient
            if (avg_grad > 10.0) avg_grad = 10.0;
            if (avg_grad < -10.0) avg_grad = -10.0;
            
            // Adam Update
            m[c] = beta1 * m[c] + (1.0 - beta1) * avg_grad;
            v[c] = beta2 * v[c] + (1.0 - beta2) * avg_grad * avg_grad;
            
            double m_hat = m[c] / (1.0 - std::pow(beta1, iter));
            double v_hat = v[c] / (1.0 - std::pow(beta2, iter));
            
            double step = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            
            const_node->value -= step;
        }
    }
}
