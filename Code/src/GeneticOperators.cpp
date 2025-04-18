#include "GeneticOperators.h"
#include "Globals.h"
#include "Fitness.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <set>

// Genera un árbol aleatorio (CON TODOS LOS OPERADORES, EXPONENTES SIN RESTRICCIÓN)
NodePtr generate_random_tree(int max_depth, int current_depth) {
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    auto& rng = get_rng();
    double terminal_prob = 0.2 + 0.8 * (static_cast<double>(current_depth) / max_depth);

    if (current_depth >= max_depth || prob_dist(rng) < terminal_prob) {
        // Crear terminal
        if (prob_dist(rng) < TERMINAL_VS_VARIABLE_PROB) { return std::make_shared<Node>(NodeType::Variable); }
        else {
            auto node = std::make_shared<Node>(NodeType::Constant);
            if (FORCE_INTEGER_CONSTANTS) { std::uniform_int_distribution<int> cd(CONSTANT_INT_MIN_VALUE, CONSTANT_INT_MAX_VALUE); node->value = static_cast<double>(cd(rng)); }
            else { std::uniform_real_distribution<double> cd(CONSTANT_MIN_VALUE, CONSTANT_MAX_VALUE); node->value = cd(rng); }
            if (std::fabs(node->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) node->value = 0.0;
            return node;
        }
    } else {
        // Crear operador (+, -, *, /, ^)
        auto node = std::make_shared<Node>(NodeType::Operator);
        const std::vector<char> ops = {'+', '-', '*', '/', '^'};
        std::discrete_distribution<int> op_dist(OPERATOR_WEIGHTS.begin(), OPERATOR_WEIGHTS.end());
        node->op = ops[op_dist(rng)];

        // Generar hijos recursivamente (sin caso especial para '^')
        node->left = generate_random_tree(max_depth, current_depth + 1);
        node->right = generate_random_tree(max_depth, current_depth + 1);

        // Fallback
        if (!node->left) node->left = generate_random_tree(max_depth, current_depth + 1);
        if (!node->right) node->right = generate_random_tree(max_depth, current_depth + 1);
        return node;
    }
    // return nullptr; // Safeguard - Should be unreachable due to if/else structure
}

// --- Crea la población inicial (CUERPO COMPLETO) ---
std::vector<Individual> create_initial_population(int population_size) {
    std::vector<Individual> population;
    population.reserve(population_size); // Reserva memoria
    std::uniform_int_distribution<int> depth_dist(3, MAX_TREE_DEPTH_INITIAL); // Usa constante global
    auto& rng = get_rng();
    for (int i = 0; i < population_size; ++i) {
        // Crea un individuo con un árbol generado aleatoriamente
        population.emplace_back(generate_random_tree(depth_dist(rng)));
    }
    return population; // <-- Return
}

// --- Selección por torneo con parsimonia (CUERPO COMPLETO) ---
const Individual& tournament_selection(const std::vector<Individual>& population, int tournament_size) {
    if (population.empty()) throw std::runtime_error("Cannot perform tournament selection on empty population.");
    if (tournament_size <= 0) tournament_size = 1;
    tournament_size = std::min(tournament_size, (int)population.size()); // Ajustar tamaño

    std::uniform_int_distribution<int> dist(0, population.size() - 1);
    auto& rng = get_rng();
    const Individual* best_in_tournament = nullptr;

    // Encontrar el primer contendiente válido
    int attempts = 0; const int max_attempts = std::min((int)population.size() * 2, 100);
    do { best_in_tournament = &population[dist(rng)]; attempts++; }
    while (!best_in_tournament->fitness_valid && attempts < max_attempts);
    if (!best_in_tournament || !best_in_tournament->fitness_valid) throw std::runtime_error("Tournament selection couldn't find any valid individual.");

    // Rondas restantes
    for (int i = 1; i < tournament_size; ++i) {
        const Individual& contender = population[dist(rng)];
        if (!contender.fitness_valid) continue; // Saltar inválidos
        // Comparar fitness
        if (contender.fitness < best_in_tournament->fitness) {
            best_in_tournament = &contender;
        }
        // Desempate por tamaño (parsimonia)
        else if (std::fabs(contender.fitness - best_in_tournament->fitness) < FITNESS_EQUALITY_TOLERANCE) {
            int contender_size = tree_size(contender.tree);
            int best_size = tree_size(best_in_tournament->tree);
            if (contender_size < best_size) best_in_tournament = &contender;
        }
    }
    return *best_in_tournament; // <-- Return
}


// Aplica una mutación al árbol (EXPONENTES SIN RESTRICCIÓN en OperatorChange)
NodePtr mutate_tree(const NodePtr& tree, double mutation_rate, int max_depth) {
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    auto new_tree = clone_tree(tree);
    if (prob(rng) >= mutation_rate) return new_tree;

    std::vector<NodePtr*> nodes; collect_node_ptrs(new_tree, nodes);
    if (nodes.empty()) return new_tree;

    std::uniform_int_distribution<int> node_dist(0, nodes.size() - 1);
    int node_idx = node_dist(rng);
    NodePtr* node_to_mutate_ptr = nodes[node_idx];

    const std::vector<MutationType> mutation_types = {
        MutationType::ConstantChange, MutationType::OperatorChange,
        MutationType::SubtreeReplace, MutationType::NodeInsertion,
        MutationType::NodeDeletion
    };
    std::uniform_int_distribution<int> type_dist(0, mutation_types.size() - 1);
    MutationType mut_type = mutation_types[type_dist(rng)];

    NodePtr& current_node_ptr_ref = *node_to_mutate_ptr;
    if (!current_node_ptr_ref) return new_tree;
    Node& current_node = *current_node_ptr_ref;

    switch (mut_type) {
        case MutationType::ConstantChange:
             if (current_node.type == NodeType::Constant) { /* ... */ }
             else { *node_to_mutate_ptr = generate_random_tree(max_depth); }
            break;
        case MutationType::OperatorChange:
             if (current_node.type == NodeType::Operator) {
                 const std::vector<char> ops = {'+', '-', '*', '/', '^'}; // Lista completa
                 std::vector<char> possible_ops;
                 for (char op : ops) if (op != current_node.op) possible_ops.push_back(op);
                 if (!possible_ops.empty()) {
                     std::uniform_int_distribution<int> op_choice(0, possible_ops.size() - 1);
                     current_node.op = possible_ops[op_choice(rng)]; // Cambiar operador (sin manejo especial para ^)
                 }
             } else { *node_to_mutate_ptr = generate_random_tree(max_depth); }
            break;
        case MutationType::SubtreeReplace:
            *node_to_mutate_ptr = generate_random_tree(max_depth);
            break;
        case MutationType::NodeInsertion:
            {
                auto new_op_node = std::make_shared<Node>(NodeType::Operator);
                const std::vector<char> insert_ops = {'+', '-'}; std::uniform_int_distribution<int> op_dist(0, insert_ops.size() - 1); new_op_node->op = insert_ops[op_dist(rng)];
                new_op_node->left = current_node_ptr_ref;
                 if (prob(rng) < MUTATE_INSERT_CONST_PROB) {
                     auto right_child = std::make_shared<Node>(NodeType::Constant);
                     if (FORCE_INTEGER_CONSTANTS) { std::uniform_int_distribution<int> cv(MUTATE_INSERT_CONST_INT_MIN, MUTATE_INSERT_CONST_INT_MAX); right_child->value = static_cast<double>(cv(rng)); }
                     else { std::uniform_real_distribution<double> cv(MUTATE_INSERT_CONST_FLOAT_MIN, MUTATE_INSERT_CONST_FLOAT_MAX); right_child->value = cv(rng); }
                     if (std::fabs(right_child->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) right_child->value = 0.0;
                     new_op_node->right = right_child;
                 } else { new_op_node->right = std::make_shared<Node>(NodeType::Variable); }
                 if (!new_op_node->right) new_op_node->right = std::make_shared<Node>(NodeType::Variable);
                *node_to_mutate_ptr = new_op_node;
            }
            break;
        case MutationType::NodeDeletion:
            {
                bool is_root = (node_to_mutate_ptr == &new_tree);
                if (current_node.type == NodeType::Operator) {
                    NodePtr replacement = nullptr; bool prefer_left = (prob(rng) < 0.5);
                    if (prefer_left && current_node.left) replacement = current_node.left;
                    else if (current_node.right) replacement = current_node.right;
                    else if (current_node.left) replacement = current_node.left;
                    if (!replacement) replacement = generate_random_tree(0);
                    *node_to_mutate_ptr = replacement;
                } else { // Terminal
                    if (!is_root) *node_to_mutate_ptr = generate_random_tree(0);
                }
            }
            break;
         default:
             *node_to_mutate_ptr = generate_random_tree(max_depth);
            break;
    }
    return new_tree;
}

// Cruce
void crossover_trees(NodePtr& tree1, NodePtr& tree2) {
    std::vector<NodePtr*> nodes1, nodes2; collect_node_ptrs(tree1, nodes1); collect_node_ptrs(tree2, nodes2);
    if (nodes1.empty() || nodes2.empty()) return;
    auto& rng = get_rng(); std::uniform_int_distribution<int> d1(0, nodes1.size()-1), d2(0, nodes2.size()-1);
    std::swap(*nodes1[d1(rng)], *nodes2[d2(rng)]);
}
