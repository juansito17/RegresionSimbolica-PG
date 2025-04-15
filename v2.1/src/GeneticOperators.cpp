#include "AdvancedFeatures.h"
#include "GeneticOperators.h"
#include "Globals.h"
#include "Fitness.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept> // <--- AÑADIDO: Incluir para std::runtime_error

// Generates a random tree. A higher probability of terminals at greater depths.
NodePtr generate_random_tree(int max_depth, int current_depth) {
    int max_attempts = 5;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        auto& rng = get_rng();

        // Increase probability of terminal node as depth increases
        double terminal_prob = 0.2 + 0.8 * (static_cast<double>(current_depth) / max_depth);

        NodePtr node;
        if (current_depth >= max_depth || prob_dist(rng) < terminal_prob) {
            // Create a terminal node (Constant or Variable)
            if (prob_dist(rng) < 0.75) { // 75% chance of Variable 'x'
                node = std::make_shared<Node>(NodeType::Variable);
            } else { // 25% chance of Constant
                node = std::make_shared<Node>(NodeType::Constant);
                 // Generate constants within a reasonable integer range initially
                std::uniform_int_distribution<int> const_dist(1, 10); // Smaller range for initial constants
                if (USE_INTEGER_MODE) {
                    node->value = const_dist(rng);
                } else {
                    node->value = const_dist(rng); // Puedes cambiar a real si quieres decimales
                }
            }
        } else {
            // Create an operator node
            node = std::make_shared<Node>(NodeType::Operator);

            // Define operators and their probabilities
            // Favor simpler operators slightly more
            const std::vector<char> ops = {'+', '-', '*', '/', '^'};
            const std::vector<double> weights = {0.3, 0.3, 0.25, 0.1, 0.05}; // Sum must match usage in discrete_distribution
            std::discrete_distribution<int> op_dist(weights.begin(), weights.end());
            node->op = ops[op_dist(rng)];

            // Handle children generation
            if (node->op == '^') {
                // For power, ensure the exponent is a small integer constant for stability
                node->left = generate_random_tree(max_depth, current_depth + 1);
                auto right = std::make_shared<Node>(NodeType::Constant);
                std::uniform_int_distribution<int> exp_dist(2, 4); // Exponents 2, 3, 4
                if (USE_INTEGER_MODE) {
                    right->value = exp_dist(rng);
                } else {
                    right->value = exp_dist(rng); // Puedes cambiar a real si quieres decimales
                }
                node->right = right;
            } else {
                // For other operators, generate children recursively
                node->left = generate_random_tree(max_depth, current_depth + 1);
                node->right = generate_random_tree(max_depth, current_depth + 1);
            }
             // Ensure generated nodes are not null (handle potential nulls from recursion if needed)
            if (!node->left) node->left = generate_random_tree(max_depth, current_depth + 1); // Re-generate if null
            if (!node->right) node->right = generate_random_tree(max_depth, current_depth + 1); // Re-generate if null
        }
        if (USE_INTEGER_MODE) {
            node->value = std::round(node->value);
        }
        // --- Poda dinámica: validar y simplificar ---
        node = DomainConstraints::fix_or_simplify(node);
        // Poda extra: descartar árboles triviales o sin variable
        bool has_variable = false;
        std::function<void(const NodePtr&)> check_var = [&](const NodePtr& n) {
            if (!n) return;
            if (n->type == NodeType::Variable) has_variable = true;
            if (n->left) check_var(n->left);
            if (n->right) check_var(n->right);
        };
        check_var(node);
        // Permitir árboles pequeños o sin variable con baja probabilidad (exploración)
        double relax_prob = 0.15 + 0.25 * (1.0 - (double)current_depth / (double)max_depth); // Más permisivo en profundidad baja
        std::uniform_real_distribution<double> relax_dist(0.0, 1.0);
        if ((DomainConstraints::is_valid(node) && has_variable && tree_size(node) >= 2) ||
            (DomainConstraints::is_valid(node) && tree_size(node) >= 2 && relax_dist(rng) < relax_prob)) {
            return node;
        }
        // Si no es válido, reintentar
    }
    // Si no se logró un árbol válido tras varios intentos, devolver un terminal simple
    return std::make_shared<Node>(NodeType::Variable);
}

std::vector<Individual> create_initial_population(int population_size) {
    std::vector<Individual> population(population_size);
    std::uniform_int_distribution<int> depth_dist(3, MAX_TREE_DEPTH_INITIAL); // Profundidad máxima inicial reducida a 5
    auto& rng = get_rng();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < population_size; ++i) {
        population[i] = Individual(generate_random_tree(depth_dist(rng)));
    }
    return population;
}

// Tournament Selection - returns reference to avoid copy
const Individual& tournament_selection(const std::vector<Individual>& population, int tournament_size) {
    if (population.empty()) {
        // Usa std::runtime_error (ahora incluido)
        throw std::runtime_error("Cannot perform tournament selection on empty population.");
    }
    if (tournament_size <= 0) tournament_size = 1; // Ensure valid size
    if (tournament_size > population.size()) tournament_size = population.size(); // Cap size

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

NodePtr mutate_tree(const NodePtr& tree, double mutation_rate, int max_depth) {
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    // Clone the tree first to avoid modifying the original
    auto new_tree = clone_tree(tree);

    // Apply mutation recursively with a probability at each node? Or just once per tree?
    // Original code applied mutation once per tree based on rate. Let's stick to that.
    if (prob(rng) >= mutation_rate) {
        return new_tree; // No mutation applied
    }

    std::vector<NodePtr*> nodes;
    collect_node_ptrs(new_tree, nodes); // Collect pointers to the NodePtrs in the new tree

    if (nodes.empty()) {
        return new_tree; // Cannot mutate an empty tree
    }

    // Select a random node to mutate
    std::uniform_int_distribution<int> node_dist(0, nodes.size() - 1);
    int node_idx = node_dist(rng);
    NodePtr* node_to_mutate_ptr = nodes[node_idx]; // Pointer to the NodePtr to be modified

    // Select mutation type
    // NodeDeletion is tricky, let's skip for now. Simplification can be part of constraints.
    std::vector<MutationType> mutation_types = {
        MutationType::ConstantChange,
        MutationType::OperatorChange,
        MutationType::SubtreeReplace,
        MutationType::NodeInsertion,
    };
    // Aumentar peso de mutaciones expansivas
    std::discrete_distribution<int> mut_dist = std::discrete_distribution<int>({2,2,4,4});
    MutationType mut_type = mutation_types[mut_dist(rng)];

    // Get the actual NodePtr and the Node object
    NodePtr& current_node_ptr = *node_to_mutate_ptr;
    if (!current_node_ptr) return new_tree; // Should not happen if collect_node_ptrs is correct

    Node& current_node = *current_node_ptr;


    switch (mut_type) {
        case MutationType::ConstantChange:
            if (current_node.type == NodeType::Constant) {
                if (USE_INTEGER_MODE) {
                    std::uniform_int_distribution<int> int_change(-2, 2);
                    current_node.value += int_change(rng);
                } else {
                    std::uniform_real_distribution<double> change_factor(0.8, 1.2);
                    std::uniform_real_distribution<double> additive_change(-2.0, 2.0);
                    if(prob(rng) < 0.5) {
                        current_node.value *= change_factor(rng);
                    } else {
                        current_node.value += additive_change(rng);
                    }
                }
                current_node.value = std::clamp(current_node.value, -10000.0, 10000.0);
                if (std::fabs(current_node.value) < 1e-7) current_node.value = 0.0;
                if (USE_INTEGER_MODE) current_node.value = std::round(current_node.value);
            } else {
                *node_to_mutate_ptr = generate_random_tree(max_depth);
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
                            // If it was already constant, ensure it's a small integer
                            if (USE_INTEGER_MODE) {
                                std::uniform_int_distribution<int> exp_dist(2, 4);
                                current_node.right->value = exp_dist(rng);
                            } else {
                                current_node.right->value = std::round(std::clamp(current_node.right->value, 2.0, 4.0));
                            }
                        }
                    }
                }
            } else {
                *node_to_mutate_ptr = generate_random_tree(max_depth);
            }
            break;

        case MutationType::SubtreeReplace:
            // Replace the entire subtree pointed to by node_to_mutate_ptr
            *node_to_mutate_ptr = generate_random_tree(max_depth);
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
                if (prob(rng) < 0.5) {
                    auto right_const = std::make_shared<Node>(NodeType::Constant);
                    if (USE_INTEGER_MODE) {
                        std::uniform_int_distribution<int> const_val(1, 3);
                        right_const->value = const_val(rng);
                    } else {
                        std::uniform_int_distribution<int> const_val(1, 3);
                        right_const->value = const_val(rng);
                    }
                    new_op_node->right = right_const;
                } else {
                    new_op_node->right = std::make_shared<Node>(NodeType::Variable);
                }


                // Replace the original node pointer with the new operator node pointer
                *node_to_mutate_ptr = new_op_node;
            }
            break;

         // case MutationType::Simplification:
             // Handled better by a dedicated simplification function / domain constraints
             // break;
         default: // Fallback to subtree replacement
             *node_to_mutate_ptr = generate_random_tree(max_depth);
            break;

    }

    // --- Poda dinámica: validar y simplificar el árbol mutado ---
    new_tree = DomainConstraints::fix_or_simplify(new_tree);
    // Poda extra: descartar árboles triviales o sin variable tras mutación
    bool has_variable = false;
    std::function<void(const NodePtr&)> check_var = [&](const NodePtr& n) {
        if (!n) return;
        if (n->type == NodeType::Variable) has_variable = true;
        if (n->left) check_var(n->left);
        if (n->right) check_var(n->right);
    };
    check_var(new_tree);
    // Permitir árboles sin variable o pequeños con baja probabilidad (exploración)
    std::uniform_real_distribution<double> relax_dist(0.0, 1.0);
    double relax_prob = 0.10; // 10% de las veces permitimos árboles "no ideales"
    if (!DomainConstraints::is_valid(new_tree) || tree_size(new_tree) < 2) {
        if (relax_dist(rng) < relax_prob) return new_tree;
        return clone_tree(tree);
    }
    if (!has_variable && relax_dist(rng) >= relax_prob) {
        return clone_tree(tree);
    }
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