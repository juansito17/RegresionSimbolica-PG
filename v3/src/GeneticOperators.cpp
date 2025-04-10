#include "GeneticOperators.h"
#include "Globals.h"
#include "Fitness.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <iostream> // For potential debug output

// Generates a random tree with strict max_depth enforcement.
NodePtr generate_random_tree(int max_depth, int current_depth) { // Default current_depth to 0
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    // Force terminal node if max depth is reached
    if (current_depth >= max_depth) {
        // Create a terminal node (Constant or Variable)
        if (prob_dist(rng) < 0.75) { // 75% chance of Variable 'x'
            return std::make_shared<Node>(NodeType::Variable);
        } else { // 25% chance of Constant
            auto node = std::make_shared<Node>(NodeType::Constant);
            std::uniform_int_distribution<int> const_dist(1, 10);
            node->value = const_dist(rng);
            return node;
        }
    }

    // Decide between operator or terminal based on depth (grow method often uses this)
    // Simple approach: 50/50 chance unless at max_depth-1
    bool force_terminal_children = (current_depth == max_depth - 1);
    bool create_operator = !force_terminal_children && (prob_dist(rng) < 0.5); // 50% chance for operator if not at penultimate level

    if (create_operator) {
        // Create an operator node
        auto node = std::make_shared<Node>(NodeType::Operator);

        const std::vector<char> ops = {'+', '-', '*', '/', '^'};
        const std::vector<double> weights = {0.3, 0.3, 0.25, 0.1, 0.05};
        std::discrete_distribution<int> op_dist(weights.begin(), weights.end());
        node->op = ops[op_dist(rng)];

        // Generate children recursively
        node->left = generate_random_tree(max_depth, current_depth + 1);
        node->right = generate_random_tree(max_depth, current_depth + 1);

        // Ensure children are not null (shouldn't happen with strict depth)
        if (!node->left) { // Fallback just in case
             node->left = std::make_shared<Node>(NodeType::Variable);
        }
        if (!node->right) { // Fallback just in case
             // Special case for power: ensure right child is suitable if regenerated
             if (node->op == '^') {
                 auto right = std::make_shared<Node>(NodeType::Constant);
                 std::uniform_int_distribution<int> exp_dist(2, 4);
                 right->value = exp_dist(rng);
                 node->right = right;
             } else {
                 node->right = std::make_shared<Node>(NodeType::Variable);
             }
        }

         // Special handling for power operator's right child (exponent) - ensure it's simple constant
         if (node->op == '^') {
             // Force right child to be a small integer constant if it wasn't already generated as terminal
             if (node->right->type != NodeType::Constant) {
                 auto right_const = std::make_shared<Node>(NodeType::Constant);
                 std::uniform_int_distribution<int> exp_dist(2, 4); // Exponents 2, 3, 4
                 right_const->value = exp_dist(rng);
                 node->right = right_const;
             } else {
                 // Ensure existing constant is a small integer
                 node->right->value = std::round(std::clamp(node->right->value, 2.0, 4.0));
             }
         }


        return node;
    } else {
        // Create a terminal node (Constant or Variable) - same logic as max depth case
        if (prob_dist(rng) < 0.75) {
            return std::make_shared<Node>(NodeType::Variable);
        } else {
            auto node = std::make_shared<Node>(NodeType::Constant);
            std::uniform_int_distribution<int> const_dist(1, 10);
            node->value = const_dist(rng);
            return node;
        }
    }
}

std::vector<Individual> create_initial_population(int population_size) {
    std::vector<Individual> population;
    population.reserve(population_size);
    std::uniform_int_distribution<int> depth_dist(3, MAX_TREE_DEPTH_INITIAL); // Use global constant
    auto& rng = get_rng();

    for (int i = 0; i < population_size; ++i) {
        population.emplace_back(generate_random_tree(depth_dist(rng)));
    }
    return population;
}

// Tournament Selection - returns reference to avoid copy
const Individual& tournament_selection(const std::vector<Individual>& population, int tournament_size) {
    if (population.empty()) {
        throw std::runtime_error("Cannot perform tournament selection on empty population.");
    }
    if (tournament_size <= 0) tournament_size = 1; // Ensure valid size
    if (static_cast<size_t>(tournament_size) > population.size()) tournament_size = static_cast<int>(population.size()); // Cap size

    std::uniform_int_distribution<int> dist(0, population.size() - 1);
    auto& rng = get_rng();

    const Individual* best_in_tournament = &population[dist(rng)];

    // Perform tournament rounds
    for (int i = 1; i < tournament_size; ++i) {
        const Individual& contender = population[dist(rng)];
        // Compare based on cached fitness (must be valid)
        if (!best_in_tournament->fitness_valid || (contender.fitness_valid && contender.fitness < best_in_tournament->fitness)) {
             if (!contender.fitness_valid) {
                 // Usa std::runtime_error (ahora incluido)
                 throw std::runtime_error("Tournament selection encountered individual with invalid fitness.");
             }
            best_in_tournament = &contender;
        }
    }
    // Ensure the selected individual has valid fitness before returning
     if (!best_in_tournament->fitness_valid) {
         // Usa std::runtime_error (ahora incluido)
         throw std::runtime_error("Tournament selection failed to find individual with valid fitness.");
     }


    return *best_in_tournament;
}


// Enhanced Mutation Helper Function
enum class MutationType {
    ConstantChange,
    OperatorChange,
    SubtreeReplace,
    NodeInsertion,
    NodeDeletion, // Harder to implement safely without breaking structure
    Simplification
};

NodePtr mutate_tree(const NodePtr& tree, double mutation_rate, int max_mutation_depth) { // Renamed max_depth param
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    // Clone the tree first to avoid modifying the original
    auto new_tree = clone_tree(tree);

    if (prob(rng) >= mutation_rate) {
        return new_tree; // No mutation applied
    }

    std::vector<NodePtr*> nodes;
    collect_node_ptrs(new_tree, nodes);

    if (nodes.empty()) {
        return new_tree;
    }

    std::uniform_int_distribution<int> node_dist(0, nodes.size() - 1);
    int node_idx = node_dist(rng);
    NodePtr* node_to_mutate_ptr = nodes[node_idx];

    const std::vector<MutationType> mutation_types = {
        MutationType::ConstantChange,
        MutationType::OperatorChange,
        MutationType::SubtreeReplace,
        MutationType::NodeInsertion,
    };
    std::uniform_int_distribution<int> type_dist(0, mutation_types.size() - 1);
    MutationType mut_type = mutation_types[type_dist(rng)];

    NodePtr& current_node_ptr = *node_to_mutate_ptr;
    if (!current_node_ptr) return new_tree;

    Node& current_node = *current_node_ptr;

    // Calculate remaining depth budget if needed (more complex)
    // For now, just use the global max_mutation_depth for replacements

    switch (mut_type) {
        case MutationType::ConstantChange:
            if (current_node.type == NodeType::Constant) {
                std::uniform_real_distribution<double> change_factor(0.8, 1.2); // Multiplicative change
                std::uniform_real_distribution<double> additive_change(-2.0, 2.0); // Additive change
                if(prob(rng) < 0.5) {
                    current_node.value *= change_factor(rng);
                } else {
                    current_node.value += additive_change(rng);
                }
                // Ensure constants don't become excessively large or small, or exactly zero if unwanted
                 current_node.value = std::clamp(current_node.value, -10000.0, 10000.0);
                 if (std::fabs(current_node.value) < 1e-7) current_node.value = 0.0; // Snap to zero
                 // If 0 is problematic (e.g., denominator), maybe change to 1 or -1? Requires context.

            } else {
                 // Fallback: Replace non-constant with a small random tree
                 *node_to_mutate_ptr = generate_random_tree(max_mutation_depth); // Use mutation depth limit
            }
            break;

        case MutationType::OperatorChange:
            if (current_node.type == NodeType::Operator) {
                const std::vector<char> ops = {'+', '-', '*', '/', '^'};
                std::vector<char> possible_ops;
                for (char op : ops) {
                    if (op != current_node.op) {
                        possible_ops.push_back(op);
                    }
                }
                if (!possible_ops.empty()) {
                    std::uniform_int_distribution<int> op_choice(0, possible_ops.size() - 1);
                    current_node.op = possible_ops[op_choice(rng)];

                    // Special handling if changed to power operator
                    if (current_node.op == '^') {
                        // Ensure right child is a small integer constant
                        if (!current_node.right || current_node.right->type != NodeType::Constant) {
                             auto right = std::make_shared<Node>(NodeType::Constant);
                             std::uniform_int_distribution<int> exp_dist(2, 4);
                             right->value = exp_dist(rng);
                             current_node.right = right;
                        } else {
                            // Ensure existing constant is a small integer
                             current_node.right->value = std::round(std::clamp(current_node.right->value, 2.0, 4.0));
                        }
                    }
                }
            } else {
                // Fallback: Replace non-operator with a small random tree
                 *node_to_mutate_ptr = generate_random_tree(max_mutation_depth); // Use mutation depth limit
            }
            break;

        case MutationType::SubtreeReplace:
            // Replace the entire subtree pointed to by node_to_mutate_ptr
            *node_to_mutate_ptr = generate_random_tree(max_mutation_depth); // Use mutation depth limit
            break;

        case MutationType::NodeInsertion:
            {
                // Insert a new operator node above the selected node
                auto new_op_node = std::make_shared<Node>(NodeType::Operator);
                const std::vector<char> insert_ops = {'+', '-'}; // Simple ops for insertion
                std::uniform_int_distribution<int> op_dist(0, insert_ops.size() - 1);
                new_op_node->op = insert_ops[op_dist(rng)];

                // The original node becomes the left child
                new_op_node->left = current_node_ptr;

                // Create a new simple right child (e.g., small constant or 'x')
                 NodePtr right_child;
                 if (prob(rng) < 0.5) {
                     auto right_const = std::make_shared<Node>(NodeType::Constant);
                     std::uniform_int_distribution<int> const_val(1, 3);
                     right_const->value = const_val(rng);
                     right_child = right_const;
                 } else {
                     right_child = std::make_shared<Node>(NodeType::Variable);
                 }
                 new_op_node->right = right_child; // Assign generated right child

                // Replace the original node pointer with the new operator node pointer
                *node_to_mutate_ptr = new_op_node;
            }
            break;

         default: // Fallback to subtree replacement
             *node_to_mutate_ptr = generate_random_tree(max_mutation_depth); // Use mutation depth limit
            break;
    }

    // Optional: Check overall tree size/depth after mutation and simplify/reject if too large?
    // int final_size = tree_size(new_tree);
    // if (final_size > MAX_ALLOWED_MUTATED_SIZE) { /* handle */ }

    return new_tree;
}


// Crossover: Swaps randomly selected subtrees between two parents
// Modifies the input trees directly.
void crossover_trees(NodePtr& tree1, NodePtr& tree2) {
    // Collect potential crossover points (pointers to NodePtr members)
    std::vector<NodePtr*> nodes1, nodes2;
    collect_node_ptrs(tree1, nodes1);
    collect_node_ptrs(tree2, nodes2);

    // Need at least one potential point in each tree
    if (nodes1.empty() || nodes2.empty()) {
        return; // Cannot perform crossover
    }

    auto& rng = get_rng();
    std::uniform_int_distribution<int> dist1(0, nodes1.size() - 1);
    std::uniform_int_distribution<int> dist2(0, nodes2.size() - 1);

    // Select random crossover points (pointers to the NodePtrs)
    NodePtr* point1 = nodes1[dist1(rng)];
    NodePtr* point2 = nodes2[dist2(rng)];

    // Swap the shared_ptrs themselves
    // This effectively swaps the subtrees rooted at these points
    std::swap(*point1, *point2);
}