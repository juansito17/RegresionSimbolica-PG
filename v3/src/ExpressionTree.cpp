#include "ExpressionTree.h"
#include "Globals.h" // Include for MAX constants if needed elsewhere
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <stack>
#include <vector>
#include <map> // For operator mapping

double evaluate_tree(const NodePtr& node, double x) {
    if (!node) {
        // Consider how to handle null nodes - return 0, NaN, or throw?
        // Returning 0 might silently hide issues. NaN propagates.
        return std::nan(""); // Or 0.0, depending on desired behavior
    }

    switch (node->type) {
        case NodeType::Constant:
            return node->value;
        case NodeType::Variable:
            return x;
        case NodeType::Operator: {
            double leftVal = evaluate_tree(node->left, x);
            double rightVal = evaluate_tree(node->right, x);

            // Check for NaN propagation
            if (std::isnan(leftVal) || std::isnan(rightVal)) {
                return std::nan("");
            }

            switch (node->op) {
                case '+': return leftVal + rightVal;
                case '-': return leftVal - rightVal;
                case '*': return leftVal * rightVal;
                case '/':
                    // Check for division by zero or near-zero
                    if (std::fabs(rightVal) < 1e-9) {
                        return INF; // Return controlled infinity
                    }
                    return leftVal / rightVal;
                case '^':
                    // Handle potential issues with pow:
                    // - pow(0,0) is often 1, but can be undefined
                    // - pow(negative, non-integer) -> NaN
                    // - large exponents -> overflow
                    if (leftVal == 0.0 && rightVal == 0.0) return 1.0; // Common convention
                    if (leftVal < 0.0 && std::floor(rightVal) != rightVal) return std::nan(""); // Imaginary result
                    // Add check for large results if needed, potentially cap exponent
                    // double result = std::pow(leftVal, rightVal);
                    // return std::isinf(result) ? INF : result; // Control overflow
                     return std::pow(leftVal, rightVal); // Be careful with overflow/underflow -> INF

                default:
                    // Should not happen with valid operators
                    return std::nan(""); // Indicate error
            }
        }
        default: // Should not happen
             return std::nan("");
    }
}

std::string tree_to_string(const NodePtr& node) {
     if (!node) return "";

     switch (node->type) {
        case NodeType::Constant:
            // Improve constant formatting slightly
            {
                double val = node->value;
                if (std::fabs(val - std::round(val)) < 1e-6) {
                    return std::to_string(static_cast<long long>(std::round(val)));
                } else {
                    // Could format with fixed precision if needed
                    return std::to_string(val);
                }
            }
        case NodeType::Variable:
            return "x";
        case NodeType::Operator:
            // Add parentheses only if necessary based on precedence (more complex)
            // Simple approach: always add parentheses for clarity
            return "(" + tree_to_string(node->left) + node->op + tree_to_string(node->right) + ")";
        default:
            return "?"; // Should not happen
    }
}

int tree_size(const NodePtr& node) {
    if (!node) return 0;
    if (node->type != NodeType::Operator) return 1;
    return 1 + tree_size(node->left) + tree_size(node->right);
}

int tree_depth(const NodePtr& root) {
    if (!root) return 0;
    
    // Leaf nodes (variable or constant)
    if (!root->left && !root->right) return 1;
    
    // Get depth of both subtrees
    int left_depth = tree_depth(root->left);
    int right_depth = tree_depth(root->right);
    
    // Return the larger depth plus one for current node
    return std::max(left_depth, right_depth) + 1;
}

NodePtr clone_tree(const NodePtr& node) {
    if (!node) return nullptr;

    auto new_node = std::make_shared<Node>();
    new_node->type = node->type;
    new_node->value = node->value;
    new_node->op = node->op;

    // Recursively clone children only if they exist
    if (node->left) {
        new_node->left = clone_tree(node->left);
    }
    if (node->right) {
        new_node->right = clone_tree(node->right);
    }

    return new_node;
}

// Helper to collect pointers to NodePtr members within the tree structure
// This allows mutation/crossover to replace subtrees directly.
void collect_node_ptrs(NodePtr& node, std::vector<NodePtr*>& vec) {
    if (!node) return; // Stop if the current node is null

    vec.push_back(&node); // Add the pointer *to* the shared_ptr we are looking at

    // Only recurse if it's an operator node (terminals have no children pointers to modify)
    if (node->type == NodeType::Operator) {
        // Pass the actual member shared_ptr variables by reference
        if (node->left) {
             collect_node_ptrs(node->left, vec);
        }
         if (node->right) {
             collect_node_ptrs(node->right, vec);
        }
    }
}

// Define the global RNG instance (implementation detail)
namespace {
    std::mt19937 global_rng(std::random_device{}());
}

std::mt19937& get_rng() {
    return global_rng;
}

// --- Flattening Implementation ---

// Operator mapping for flattening
const std::map<char, int> op_to_int = {
    {'+', 0}, {'-', 1}, {'*', 2}, {'/', 3}, {'^', 4}
};
const int VAR_CODE = -1; // Code for variable 'x'
const int CONST_CODE_OFFSET = 1000; // Constants will be encoded as 1000, 1001, ...

void flatten_tree_recursive(const NodePtr& node,
                            std::vector<int>& structure,
                            std::vector<double>& constants,
                            int& const_idx,
                            int max_struct_size,
                            int max_consts,
                            bool& success)
{
    if (!success || !node) {
        success = false;
        return;
    }
    if (structure.size() >= max_struct_size) {
        success = false;
        return;
    }

    // Declare iterator outside switch to fix cross-initialization error
    std::map<char, int>::const_iterator op_it;

    switch (node->type) {
        case NodeType::Constant:
            if (const_idx >= max_consts) {
                success = false;
                return;
            }
            structure.push_back(CONST_CODE_OFFSET + const_idx);
            constants[const_idx++] = node->value;
            break;

        case NodeType::Variable:
            structure.push_back(VAR_CODE);
            break;

        case NodeType::Operator:
            flatten_tree_recursive(node->left, structure, constants, const_idx, max_struct_size, max_consts, success);
            if (!success) return;

            flatten_tree_recursive(node->right, structure, constants, const_idx, max_struct_size, max_consts, success);
            if (!success) return;

            if (structure.size() >= max_struct_size) {
                success = false;
                return;
            }
            op_it = op_to_int.find(node->op);
            if (op_it != op_to_int.end()) {
                structure.push_back(op_it->second);
            } else {
                success = false;
            }
            break;

        default:
            success = false;
            break;
    }
}

bool flatten_tree(const NodePtr& tree,
                  std::vector<int>& structure,
                  std::vector<double>& constants,
                  int max_struct_size,
                  int max_consts)
{
    structure.clear();
    structure.reserve(max_struct_size);
    // Initialize constants vector with a placeholder (e.g., 0 or NaN)
    // We only fill up to const_idx, but the vector passed to CUDA needs the full size.
    constants.assign(max_consts, 0.0);

    int const_idx = 0;
    bool success = true;
    flatten_tree_recursive(tree, structure, constants, const_idx, max_struct_size, max_consts, success);

    // Pad structure vector to full size if needed (CUDA kernel expects fixed size)
    if (success) {
         // Use a special code (e.g., -99) for padding? Or ensure kernel handles shorter structures.
         // Let's assume the kernel can handle the actual size based on structure content.
         // No padding needed if kernel is robust.
    } else {
        // Clear vectors on failure to indicate invalid state
        structure.clear();
        constants.clear();
    }

    return success;
}