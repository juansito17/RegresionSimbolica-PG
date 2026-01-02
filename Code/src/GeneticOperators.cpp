#include "GeneticOperators.h"
#include "Globals.h"
#include "Fitness.h"
#include "AdvancedFeatures.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>  // For std::iota
#include <map>
#include <stdexcept>
#include <set>
#include <iostream> // Para mensajes de error/info

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

        // Crear operador
        auto node = std::make_shared<Node>(NodeType::Operator);
        // Match the weights in Globals.h: +, -, *, /, ^, %, s, c, l, e, !, _, g
        const std::vector<char> ops = {'+', '-', '*', '/', '^', '%', 's', 'c', 'l', 'e', '!', '_', 'g'};
        std::discrete_distribution<int> op_dist(OPERATOR_WEIGHTS.begin(), OPERATOR_WEIGHTS.end());
        node->op = ops[op_dist(rng)];

        bool is_unary = (node->op == 's' || node->op == 'c' || node->op == 'l' || node->op == 'e' || node->op == '!' || node->op == '_' || node->op == 'g');

        // Generar hijos recursivamente
        node->left = generate_random_tree(max_depth, current_depth + 1);
        if (!is_unary) {
            node->right = generate_random_tree(max_depth, current_depth + 1);
        } else {
            node->right = nullptr;
        }

        // Fallback para hijos nulos
        auto generate_random_terminal = [&]() -> NodePtr {
            if (prob_dist(rng) < TERMINAL_VS_VARIABLE_PROB) { return std::make_shared<Node>(NodeType::Variable); }
            else {
                auto const_node = std::make_shared<Node>(NodeType::Constant);
                if (FORCE_INTEGER_CONSTANTS) { std::uniform_int_distribution<int> cd(CONSTANT_INT_MIN_VALUE, CONSTANT_INT_MAX_VALUE); const_node->value = static_cast<double>(cd(rng)); }
                else { std::uniform_real_distribution<double> cd(CONSTANT_MIN_VALUE, CONSTANT_MAX_VALUE); const_node->value = cd(rng); }
                if (std::fabs(const_node->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) const_node->value = 0.0;
                return const_node;
            }
        };

        if (!node->left) node->left = generate_random_terminal();
        if (!is_unary && !node->right) node->right = generate_random_terminal();

        // --- Manejo especial para el operador de potencia '^' ---
        if (node->op == '^') {
            // Regla 1: Evitar 0^0 o 0^negativo
            if (node->left->type == NodeType::Constant && std::fabs(node->left->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) {
                if (node->right->type == NodeType::Constant && node->right->value <= SIMPLIFY_NEAR_ZERO_TOLERANCE) {
                    const std::vector<char> safe_ops = {'+', '-', '*'};
                    std::uniform_int_distribution<int> safe_op_dist(0, safe_ops.size() - 1);
                    node->op = safe_ops[safe_op_dist(rng)];
                }
            }
            // Regla 2: Evitar base negativa con exponente no entero
            else if (node->left->type == NodeType::Constant && node->left->value < 0.0) {
                if (node->right->type == NodeType::Constant && std::fabs(node->right->value - std::round(node->right->value)) > SIMPLIFY_NEAR_ZERO_TOLERANCE) {
                     // Change exponent to int
                     std::uniform_int_distribution<int> int_exp_dist(-3, 3);
                     node->right = std::make_shared<Node>(NodeType::Constant);
                     node->right->value = static_cast<double>(int_exp_dist(rng));
                }
            }
        }
        return node;
    }
}

// --- Crea la población inicial (MODIFICADO para inyectar fórmula) ---
// === OPTIMIZACIÓN: Paralelizado con OpenMP ===
std::vector<Individual> create_initial_population(int population_size) {
    std::vector<Individual> population;
    population.resize(population_size); // Pre-allocate all slots for parallel access

    // --- NUEVO: Inyección de Fórmula Inicial ---
    int start_index = 0;
    if (USE_INITIAL_FORMULA && !INITIAL_FORMULA_STRING.empty() && population_size > 0) {
        try {
            NodePtr initial_tree = parse_formula_string(INITIAL_FORMULA_STRING);
            if (initial_tree) {
                population[0] = Individual(std::move(initial_tree));
                start_index = 1;
                std::cout << "[INFO] Injected initial formula: " << INITIAL_FORMULA_STRING << std::endl;
            } else {
                 std::cerr << "[WARNING] Parsing initial formula returned null. Skipping injection." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to parse initial formula '" << INITIAL_FORMULA_STRING
                      << "': " << e.what() << ". Skipping injection." << std::endl;
        }
    }
    // -----------------------------------------

    // === OPTIMIZACIÓN: Loop paralelo para generar árboles ===
    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = start_index; i < population_size; ++i) {
        // Cada hilo tiene su propio RNG (thread_local en get_rng)
        auto& rng = get_rng();
        std::uniform_int_distribution<int> depth_dist(3, MAX_TREE_DEPTH_INITIAL);
        
        NodePtr random_tree = nullptr;
        int attempts = 0;
        const int max_attempts = 10;
        while (!random_tree && attempts < max_attempts) {
            random_tree = generate_random_tree(depth_dist(rng));
            attempts++;
        }
        if (random_tree) {
            population[i] = Individual(std::move(random_tree));
        } else {
            // Fallback: crear constante simple
            auto fallback_node = std::make_shared<Node>(NodeType::Constant);
            fallback_node->value = 0.0;
            population[i] = Individual(std::move(fallback_node));
        }
    }
    return population;
}

// --- Selección por torneo con parsimonia ---
Individual tournament_selection(const std::vector<Individual>& population, int tournament_size) {
    if (population.empty()) throw std::runtime_error("Cannot perform tournament selection on empty population.");
    if (tournament_size <= 0) tournament_size = 1;
    tournament_size = std::min(tournament_size, (int)population.size());

    std::uniform_int_distribution<int> dist(0, population.size() - 1);
    auto& rng = get_rng();
    const Individual* best_in_tournament = nullptr;

    int attempts = 0; const int max_attempts = std::min((int)population.size() * 2, 100);
    do {
        best_in_tournament = &population[dist(rng)];
        attempts++;
    } while ((!best_in_tournament || !best_in_tournament->tree || !best_in_tournament->fitness_valid) && attempts < max_attempts);

    if (!best_in_tournament || !best_in_tournament->tree || !best_in_tournament->fitness_valid) {
         if (!population.empty()) return population[0];
         else throw std::runtime_error("Tournament selection couldn't find any valid individual in a non-empty population.");
    }

    for (int i = 1; i < tournament_size; ++i) {
        const Individual& contender = population[dist(rng)];
        if (!contender.tree || !contender.fitness_valid) continue;

        if (contender.fitness < best_in_tournament->fitness) {
            best_in_tournament = &contender;
        }
        else if (std::fabs(contender.fitness - best_in_tournament->fitness) < FITNESS_EQUALITY_TOLERANCE) {
            int contender_size = tree_size(contender.tree);
            int best_size = tree_size(best_in_tournament->tree);
            if (contender_size < best_size) best_in_tournament = &contender;
        }
    }
    return *best_in_tournament;
}

// --- Epsilon-Lexicase Selection Implementation ---
// Calculates residuals on demand if not present (Lazy Eval)
void ensure_errors_computed(Individual& ind, const std::vector<double>& targets, const std::vector<double>& x_values) {
    if (!ind.errors.empty()) return; // Already computed
    if (!ind.tree) return;
    
    ind.errors.reserve(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        double val = evaluate_tree(ind.tree, x_values[i]);
        if (std::isnan(val) || std::isinf(val)) {
            ind.errors.push_back(INF);
        } else {
            // Use ABSOLUTE error for lexicase
            ind.errors.push_back(std::fabs(val - targets[i]));
        }
    }
}

Individual lexicase_selection(std::vector<Individual>& population, const std::vector<double>& targets, const std::vector<double>& x_values) {
    auto& rng = get_rng();
    
    // 1. Initial Candidates: Random subset (Tournament Size * 2) or Full Population?
    // Efficiency: Using a subset is "Tournament Lexicase". Using Full is "Standard Lexicase".
    // For 50k pop, full lexicase is slow. Let's use a large pool (e.g. 50-100).
    int pool_size = 100; 
    std::vector<Individual*> candidates;
    candidates.reserve(pool_size);
    
    std::uniform_int_distribution<int> dist(0, population.size() - 1);
    for(int i=0; i<pool_size; ++i) {
        Individual& ind = population[dist(rng)];
        if(ind.tree && ind.fitness_valid) candidates.push_back(&ind);
    }
    
    if (candidates.empty()) return population[0]; // Should not happen
    
    // 2. Shuffle test cases
    std::vector<int> cases(targets.size());
    std::iota(cases.begin(), cases.end(), 0);
    std::shuffle(cases.begin(), cases.end(), rng);
    
    // 3. Filter loop
    for (int case_idx : cases) {
        // Compute errors for this case for all candidates (Lazy)
        double min_error = INF;
        for (Individual* cand : candidates) {
            ensure_errors_computed(*cand, targets, x_values);
            if (case_idx < cand->errors.size()) {
                 if (cand->errors[case_idx] < min_error) min_error = cand->errors[case_idx];
            }
        }
        
        // Define epsilon (MAD or simple threshold)
        // Here we use a simple dynamic epsilon based on min_error
        double epsilon = std::max(min_error * 0.1, 1e-5); 
        // Or if min_error is 0, epsilon is 1e-5. Epsilon-Lexicase implies "close enough".
        
        // Filter
        std::vector<Individual*> next_candidates;
        next_candidates.reserve(candidates.size());
        for (Individual* cand : candidates) {
             if (case_idx < cand->errors.size()) {
                 if (cand->errors[case_idx] <= min_error + epsilon) {
                     next_candidates.push_back(cand);
                 }
             }
        }
        
        candidates = std::move(next_candidates);
        if (candidates.empty()) break; // Should not happen given min_error logic
        if (candidates.size() == 1) return *candidates[0];
    }
    
    // If multiple remain, pick random
    if (candidates.empty()) return population[dist(rng)];
    std::uniform_int_distribution<int> pick(0, candidates.size() - 1);
    return *candidates[pick(rng)];
}

// Implementación de crossover
Individual crossover(const Individual& parent1, const Individual& parent2) {
    NodePtr tree1_clone = clone_tree(parent1.tree);
    NodePtr tree2_clone = clone_tree(parent2.tree);
    crossover_trees(tree1_clone, tree2_clone);
    if (USE_HARD_DEPTH_LIMIT) trim_tree(tree1_clone, MAX_TREE_DEPTH_HARD_LIMIT); // Enforce hard limit
    return Individual(tree1_clone); // Devolver uno de los hijos, el otro se descarta
}

// Implementación de mutate
void mutate(Individual& individual, double mutation_rate) {
    individual.tree = mutate_tree(individual.tree, mutation_rate, MAX_TREE_DEPTH_MUTATION);
    individual.fitness_valid = false; // El fitness se invalida al mutar el árbol
}

// Mutata un árbol (EXPONENTES SIN RESTRICCIÓN en OperatorChange)
NodePtr mutate_tree(const NodePtr& tree, double mutation_rate, int max_depth) {
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    auto new_tree = clone_tree(tree); // Siempre clonar primero
    if (!new_tree) return nullptr; // Si el árbol original era nulo, el clon también

    if (prob(rng) >= mutation_rate) return new_tree; // No mutar

    std::vector<NodePtr*> nodes; collect_node_ptrs(new_tree, nodes);
    if (nodes.empty()) return new_tree; // No hay nodos para mutar (árbol vacío?)

    std::uniform_int_distribution<int> node_dist(0, nodes.size() - 1);
    int node_idx = node_dist(rng);
    NodePtr* node_to_mutate_ptr = nodes[node_idx];
    if (!node_to_mutate_ptr || !(*node_to_mutate_ptr)) return new_tree; // Puntero o nodo nulo inesperado

    const std::vector<MutationType> mutation_types = {
        MutationType::ConstantChange, MutationType::OperatorChange,
        MutationType::SubtreeReplace, MutationType::NodeInsertion,
        MutationType::NodeDeletion
    };
    std::uniform_int_distribution<int> type_dist(0, mutation_types.size() - 1);
    MutationType mut_type = mutation_types[type_dist(rng)];

    NodePtr& current_node_ptr_ref = *node_to_mutate_ptr;
    Node& current_node = *current_node_ptr_ref;

    // Generar reemplazo aleatorio (usado en varios casos)
    auto generate_replacement = [&](int depth) -> NodePtr {
        NodePtr replacement = generate_random_tree(depth);
        if (!replacement) { // Fallback si la generación falla
            replacement = std::make_shared<Node>(NodeType::Constant);
            replacement->value = 1.0; // Usar 1.0 como fallback simple
        }
        return replacement;
    };


    switch (mut_type) {
        case MutationType::ConstantChange:
             if (current_node.type == NodeType::Constant) {
                 // Cambiar valor de la constante
                 double change_factor = std::uniform_real_distribution<double>(0.8, 1.2)(rng);
                 double add_factor = std::uniform_real_distribution<double>(-1.0, 1.0)(rng);
                 current_node.value = current_node.value * change_factor + add_factor;
                 if (FORCE_INTEGER_CONSTANTS) current_node.value = std::round(current_node.value);
                 if (std::fabs(current_node.value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) current_node.value = 0.0;
             } else {
                 // Si no es constante, reemplazar por un subárbol aleatorio pequeño
                 *node_to_mutate_ptr = generate_replacement(1); // Profundidad 1 (terminal)
             }
            break;
        case MutationType::OperatorChange:
             if (current_node.type == NodeType::Operator) {
                 // Usar la distribución ponderada global para elegir el nuevo operador.
                 // Esto asegura que si 'g' tiene peso alto, sea elegido frecuentemente.
                 // Asumimos que OPERATOR_WEIGHTS tiene 0.0 para operadores deshabilitados.
                 
                 const std::vector<char> all_ops = {'+', '-', '*', '/', '^', '%', 's', 'c', 'l', 'e', '!', '_', 'g'};
                 std::discrete_distribution<int> op_dist(OPERATOR_WEIGHTS.begin(), OPERATOR_WEIGHTS.end());
                 
                 // Verificar si hay al menos 2 operadores habilitados para evitar bucle infinito
                 int enabled_count = 0;
                 for (double w : OPERATOR_WEIGHTS) if (w > 0.0) enabled_count++;
                 
                 if (enabled_count > 1) {
                     int attempts = 0;
                     int new_op_idx = -1;
                     do {
                         new_op_idx = op_dist(rng);
                         attempts++;
                     } while (all_ops[new_op_idx] == current_node.op && attempts < 20); // Intentar cambiar
                     
                     if (attempts < 20) { // Si logramos encontrar uno diferente
                         char old_op = current_node.op;
                         char new_op = all_ops[new_op_idx];
                         
                         bool was_unary = (old_op == 's' || old_op == 'c' || old_op == 'l' || old_op == 'e' || old_op == '!' || old_op == '_' || old_op == 'g');
                         bool is_unary = (new_op == 's' || new_op == 'c' || new_op == 'l' || new_op == 'e' || new_op == '!' || new_op == '_' || new_op == 'g');

                         if (was_unary && !is_unary) {
                             current_node.right = generate_replacement(1);
                         } else if (!was_unary && is_unary) {
                             current_node.right = nullptr;
                         }
                         current_node.op = new_op;
                     }
                 }
             } else {
                 // Si no es operador, reemplazar por un subárbol aleatorio
                  *node_to_mutate_ptr = generate_replacement(max_depth);
             }
            break;
        case MutationType::SubtreeReplace:
            *node_to_mutate_ptr = generate_replacement(max_depth);
            break;
        case MutationType::NodeInsertion:
            {
                auto new_op_node = std::make_shared<Node>(NodeType::Operator);
                // Usar distribución ponderada
                const std::vector<char> all_ops = {'+', '-', '*', '/', '^', '%', 's', 'c', 'l', 'e', '!', '_', 'g'};
                std::discrete_distribution<int> op_dist(OPERATOR_WEIGHTS.begin(), OPERATOR_WEIGHTS.end());
                
                int new_op_idx = op_dist(rng); // Siempre elegirá uno habilitado
                new_op_node->op = all_ops[new_op_idx];

                bool is_unary = (new_op_node->op == 's' || new_op_node->op == 'c' || new_op_node->op == 'l' || new_op_node->op == 'e' || new_op_node->op == '!' || new_op_node->op == '_' || new_op_node->op == 'g');

                new_op_node->left = current_node_ptr_ref;

                if (!is_unary) {
                     if (prob(rng) < MUTATE_INSERT_CONST_PROB) {
                         auto right_child = std::make_shared<Node>(NodeType::Constant);
                         if (FORCE_INTEGER_CONSTANTS) { std::uniform_int_distribution<int> cv(MUTATE_INSERT_CONST_INT_MIN, MUTATE_INSERT_CONST_INT_MAX); right_child->value = static_cast<double>(cv(rng)); }
                         else { std::uniform_real_distribution<double> cv(MUTATE_INSERT_CONST_FLOAT_MIN, MUTATE_INSERT_CONST_FLOAT_MAX); right_child->value = cv(rng); }
                         if (std::fabs(right_child->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) right_child->value = 0.0;
                         new_op_node->right = right_child;
                     } else {
                         new_op_node->right = std::make_shared<Node>(NodeType::Variable);
                     }
                     if (!new_op_node->right) new_op_node->right = std::make_shared<Node>(NodeType::Variable);
                } else {
                    new_op_node->right = nullptr;
                }

                *node_to_mutate_ptr = new_op_node;
            }
            break;
        case MutationType::NodeDeletion:
            {
                // No eliminar la raíz directamente si es la única opción
                if (node_to_mutate_ptr == &new_tree && nodes.size() == 1) return new_tree;

                if (current_node.type == NodeType::Operator) {
                    // Si es operador, reemplazarlo por uno de sus hijos (aleatorio)
                    NodePtr replacement = nullptr;
                    bool has_left = (current_node.left != nullptr);
                    bool has_right = (current_node.right != nullptr);

                    if (has_left && has_right) {
                        replacement = (prob(rng) < 0.5) ? current_node.left : current_node.right;
                    } else if (has_left) {
                        replacement = current_node.left;
                    } else if (has_right) {
                        replacement = current_node.right;
                    }
                    // Si no tiene hijos válidos (¿cómo?), reemplazar por terminal
                    if (!replacement) replacement = generate_replacement(0); // Profundidad 0 (terminal)

                    *node_to_mutate_ptr = replacement;
                } else {
                    // Si es terminal, reemplazar por otro terminal aleatorio
                    // (Evitar eliminar si es la raíz y no hay más nodos)
                     if (node_to_mutate_ptr != &new_tree || nodes.size() > 1) {
                          *node_to_mutate_ptr = generate_replacement(0); // Profundidad 0 (terminal)
                     }
                }
            }
            break;
         default: // Caso inesperado, reemplazar por seguridad
             *node_to_mutate_ptr = generate_replacement(max_depth);
            break;
    }
    if (USE_HARD_DEPTH_LIMIT) trim_tree(new_tree, MAX_TREE_DEPTH_HARD_LIMIT); // Enforce hard limit after mutation
    return new_tree;
}

// Cruce
void crossover_trees(NodePtr& tree1, NodePtr& tree2) {
    if (!tree1 || !tree2) return; // No cruzar si alguno es nulo

    std::vector<NodePtr*> nodes1, nodes2;
    collect_node_ptrs(tree1, nodes1);
    collect_node_ptrs(tree2, nodes2);

    // No cruzar si alguno no tiene nodos (árbol vacío o solo raíz nula?)
    if (nodes1.empty() || nodes2.empty()) return;

    auto& rng = get_rng();
    std::uniform_int_distribution<int> d1(0, nodes1.size()-1);
    std::uniform_int_distribution<int> d2(0, nodes2.size()-1);

    // Seleccionar puntos de cruce
    NodePtr* crossover_point1 = nodes1[d1(rng)];
    NodePtr* crossover_point2 = nodes2[d2(rng)];

    // Intercambiar los subárboles (los NodePtr)
    std::swap(*crossover_point1, *crossover_point2);
}

// Implementación de simplify_tree
void simplify_tree(NodePtr& tree) {
    if (USE_SIMPLIFICATION) {
        tree = DomainConstraints::fix_or_simplify(tree);
    }
}
