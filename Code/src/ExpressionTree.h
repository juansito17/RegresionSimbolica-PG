#ifndef EXPRESSIONTREE_H
#define EXPRESSIONTREE_H

#include <memory>
#include <string>
#include <vector>
#include <stdexcept> // Para std::runtime_error

// Forward declaration
struct Node;

// Use shared_ptr for automatic memory management
using NodePtr = std::shared_ptr<Node>;

enum class NodeType { Constant, Variable, Operator };

struct Node {
    NodeType type;
    double value = 0.0;             // If type == Constant
    int var_index = 0;              // If type == Variable: index of the variable (0 for x0, 1 for x1...)
    char op = 0;                    // If type == Operator: '+', '-', '*', '/', '^'
    NodePtr left = nullptr;         // Children (for Operators)
    NodePtr right = nullptr;

    // Constructor for convenience
    Node(NodeType t = NodeType::Constant) : type(t) {}
};

// Core Tree Functions
// MODIFIED: Takes a vector of variables instead of a single double
double evaluate_tree(const NodePtr& node, const std::vector<double>& vars);

// Convenience overload for single variable case
double evaluate_tree(const NodePtr& node, double val);
std::string tree_to_string(const NodePtr& node);
int tree_size(const NodePtr& node);
NodePtr clone_tree(const NodePtr& node);
int get_tree_depth(const NodePtr& node);
void trim_tree(NodePtr& node, int max_depth);

// Helper for mutation/crossover
void collect_node_ptrs(NodePtr& node, std::vector<NodePtr*>& vec);

// --- NUEVO: Función para parsear una fórmula desde string ---
// Parsea una fórmula en notación infija simple (con paréntesis).
// Lanza std::runtime_error si hay error de sintaxis.
NodePtr parse_formula_string(const std::string& formula);


#endif // EXPRESSIONTREE_H
