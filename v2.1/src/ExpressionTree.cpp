#include "ExpressionTree.h"
#include "Globals.h" // For INF
#include <cmath>
#include <limits>
#include <stdexcept> // For potential errors
#include <vector>

double evaluate_tree(const NodePtr& node, double x) {
    if (!node) {
        return std::nan("");
    }

    switch (node->type) {
        case NodeType::Constant:
            return USE_INTEGER_MODE ? std::round(node->value) : node->value;
        case NodeType::Variable:
            return USE_INTEGER_MODE ? std::round(x) : x;
        case NodeType::Operator: {
            double leftVal = evaluate_tree(node->left, x);
            double rightVal = evaluate_tree(node->right, x);

            if (std::isnan(leftVal) || std::isnan(rightVal)) {
                return std::nan("");
            }

            double result;
            switch (node->op) {
                case '+': result = leftVal + rightVal; break;
                case '-': result = leftVal - rightVal; break;
                case '*': result = leftVal * rightVal; break;
                case '/':
                    if (std::fabs(rightVal) < 1e-9) {
                        return INF;
                    }
                    result = leftVal / rightVal; break;
                case '^':
                    if (leftVal == 0.0 && rightVal == 0.0) return 1.0;
                    if (leftVal < 0.0 && std::floor(rightVal) != rightVal) return std::nan("");
                    result = std::pow(leftVal, rightVal); break;
                default:
                    return std::nan("");
            }
            return USE_INTEGER_MODE ? std::round(result) : result;
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
                if (USE_INTEGER_MODE)
                    val = std::round(val);
                if (USE_INTEGER_MODE || std::fabs(val - std::round(val)) < 1e-6) {
                    return std::to_string(static_cast<long long>(val));
                } else {
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

// Hash estructural recursivo para árboles
size_t tree_structural_hash(const NodePtr& node) {
    if (!node) return 0x9e3779b9; // Valor especial para null
    size_t h = std::hash<int>()(static_cast<int>(node->type));
    switch (node->type) {
        case NodeType::Constant:
            h ^= std::hash<double>()(node->value) + 0x9e3779b9 + (h << 6) + (h >> 2);
            break;
        case NodeType::Variable:
            h ^= 0x12345678 + 0x9e3779b9 + (h << 6) + (h >> 2);
            break;
        case NodeType::Operator:
            h ^= std::hash<char>()(node->op) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= tree_structural_hash(node->left) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= tree_structural_hash(node->right) + 0x9e3779b9 + (h << 6) + (h >> 2);
            break;
    }
    return h;
}

// Define the global RNG instance (implementation detail)
namespace {
    std::mt19937 global_rng(std::random_device{}());
}

std::mt19937& get_rng() {
    return global_rng;
}