#include "GeneticOperators.h"
#include "Globals.h"
#include "Fitness.h"
#include <vector>
#include <cmath>
#include <algorithm>
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
        // Crear operador (+, -, *, /, ^)
        auto node = std::make_shared<Node>(NodeType::Operator);
        const std::vector<char> ops = {'+', '-', '*', '/', '^'};
        std::discrete_distribution<int> op_dist(OPERATOR_WEIGHTS.begin(), OPERATOR_WEIGHTS.end());
        node->op = ops[op_dist(rng)];

        // Generar hijos recursivamente (sin caso especial para '^')
        node->left = generate_random_tree(max_depth, current_depth + 1);
        node->right = generate_random_tree(max_depth, current_depth + 1);

        // Fallback simple: si un hijo es nulo, reemplazarlo por una constante 1.0
        // Esto evita errores en evaluación si la generación falla, aunque puede
        // no ser lo ideal genéticamente. Una mejor opción sería reintentar la generación.
        if (!node->left) {
            node->left = std::make_shared<Node>(NodeType::Constant);
            node->left->value = 1.0;
             // std::cerr << "Warning: Null left child generated, replaced with 1.0" << std::endl;
        }
        if (!node->right) {
             node->right = std::make_shared<Node>(NodeType::Constant);
             node->right->value = 1.0;
             // std::cerr << "Warning: Null right child generated, replaced with 1.0" << std::endl;
        }
        return node;
    }
}

// --- Crea la población inicial (MODIFICADO para inyectar fórmula) ---
std::vector<Individual> create_initial_population(int population_size) {
    std::vector<Individual> population;
    population.reserve(population_size); // Reserva memoria
    std::uniform_int_distribution<int> depth_dist(3, MAX_TREE_DEPTH_INITIAL); // Usa constante global
    auto& rng = get_rng();

    // --- NUEVO: Inyección de Fórmula Inicial ---
    bool formula_injected = false;
    if (USE_INITIAL_FORMULA && !INITIAL_FORMULA_STRING.empty() && population_size > 0) {
        try {
            NodePtr initial_tree = parse_formula_string(INITIAL_FORMULA_STRING);
            if (initial_tree) {
                population.emplace_back(std::move(initial_tree)); // Mover el árbol parseado
                formula_injected = true;
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

    // Rellenar el resto de la población con árboles aleatorios
    int start_index = formula_injected ? 1 : 0; // Empezar desde 1 si se inyectó
    for (int i = start_index; i < population_size; ++i) {
        // Crea un individuo con un árbol generado aleatoriamente
         NodePtr random_tree = nullptr;
         int attempts = 0;
         const int max_attempts = 10; // Intentar generar un árbol válido varias veces
         while (!random_tree && attempts < max_attempts) {
             random_tree = generate_random_tree(depth_dist(rng));
             attempts++;
         }
         if (random_tree) {
             population.emplace_back(std::move(random_tree));
         } else {
              std::cerr << "[ERROR] Failed to generate a valid random tree after " << max_attempts << " attempts for individual " << i << "." << std::endl;
              // ¿Qué hacer aquí? Podríamos lanzar una excepción o añadir un individuo inválido simple.
              // Añadir un individuo simple (constante 0) para evitar parar todo.
              auto fallback_node = std::make_shared<Node>(NodeType::Constant);
              fallback_node->value = 0.0;
              population.emplace_back(std::move(fallback_node));
         }
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
    do {
        best_in_tournament = &population[dist(rng)];
        attempts++;
    } while ((!best_in_tournament || !best_in_tournament->tree || !best_in_tournament->fitness_valid) && attempts < max_attempts); // Asegurar que el árbol exista y el fitness sea válido

    // Si no se encontró un individuo válido inicial, lanzar error o devolver el primero (aunque sea inválido)
    if (!best_in_tournament || !best_in_tournament->tree || !best_in_tournament->fitness_valid) {
         // Devolver el primer individuo como fallback si no se encontró ninguno válido
         if (!population.empty()) return population[0];
         else throw std::runtime_error("Tournament selection couldn't find any valid individual in a non-empty population.");
    }


    // Rondas restantes
    for (int i = 1; i < tournament_size; ++i) {
        const Individual& contender = population[dist(rng)];
        // Saltar inválidos (sin árbol o fitness inválido)
        if (!contender.tree || !contender.fitness_valid) continue;

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
                 const std::vector<char> ops = {'+', '-', '*', '/', '^'}; // Lista completa
                 std::vector<char> possible_ops;
                 for (char op : ops) if (op != current_node.op) possible_ops.push_back(op);
                 if (!possible_ops.empty()) {
                     std::uniform_int_distribution<int> op_choice(0, possible_ops.size() - 1);
                     current_node.op = possible_ops[op_choice(rng)]; // Cambiar operador
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
                const std::vector<char> insert_ops = {'+', '-', '*'}; // Operadores comunes para inserción
                std::uniform_int_distribution<int> op_dist(0, insert_ops.size() - 1);
                new_op_node->op = insert_ops[op_dist(rng)];

                // El nodo original se convierte en el hijo izquierdo
                new_op_node->left = current_node_ptr_ref;

                // Generar un hijo derecho (terminal)
                 if (prob(rng) < MUTATE_INSERT_CONST_PROB) {
                     auto right_child = std::make_shared<Node>(NodeType::Constant);
                     if (FORCE_INTEGER_CONSTANTS) { std::uniform_int_distribution<int> cv(MUTATE_INSERT_CONST_INT_MIN, MUTATE_INSERT_CONST_INT_MAX); right_child->value = static_cast<double>(cv(rng)); }
                     else { std::uniform_real_distribution<double> cv(MUTATE_INSERT_CONST_FLOAT_MIN, MUTATE_INSERT_CONST_FLOAT_MAX); right_child->value = cv(rng); }
                     if (std::fabs(right_child->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) right_child->value = 0.0;
                     new_op_node->right = right_child;
                 } else {
                     new_op_node->right = std::make_shared<Node>(NodeType::Variable);
                 }
                 // Fallback si el hijo derecho es nulo (no debería pasar aquí)
                 if (!new_op_node->right) new_op_node->right = std::make_shared<Node>(NodeType::Variable);

                // Reemplazar el puntero original con el nuevo nodo operador
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
