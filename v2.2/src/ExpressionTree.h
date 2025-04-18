#ifndef EXPRESSIONTREE_H
#define EXPRESSIONTREE_H

#include <memory>
#include <string>
#include <vector>

// Forward declaration
struct Node;

// Use shared_ptr for automatic memory management
using NodePtr = std::shared_ptr<Node>;

enum class NodeType { Constant, Variable, Operator };

struct Node {
    NodeType type;
    double value = 0.0;             // If type == Constant
    char op = 0;                    // If type == Operator: '+', '-', '*', '/', '^'
    NodePtr left = nullptr;         // Children (for Operators)
    NodePtr right = nullptr;

    // Constructor for convenience
    Node(NodeType t = NodeType::Constant) : type(t) {}
};

// Core Tree Functions
double evaluate_tree(const NodePtr& node, double x);
std::string tree_to_string(const NodePtr& node);
int tree_size(const NodePtr& node);
NodePtr clone_tree(const NodePtr& node);

// Helper for mutation/crossover
void collect_node_ptrs(NodePtr& node, std::vector<NodePtr*>& vec);


#endif // EXPRESSIONTREE_H