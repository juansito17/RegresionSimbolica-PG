#include "AdvancedFeatures.h"
#include "Globals.h"
#include "GeneticOperators.h"
#include "Fitness.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <unordered_map> // Asegurarse de que esté incluido también aquí por si acaso

//---------------------------------
// EvolutionParameters
//---------------------------------
EvolutionParameters EvolutionParameters::create_default() {
    return {MUTATION_RATE, ELITE_PERCENTAGE, 20, 0.8}; // Default crossover rate 80%
}

void EvolutionParameters::mutate() {
    auto& rng = get_rng();
    std::uniform_real_distribution<double> rate_change(-0.05, 0.05);
    std::uniform_int_distribution<int> tourney_change(-2, 2);

    mutation_rate = std::clamp(mutation_rate + rate_change(rng), 0.05, 0.5);
    elite_percentage = std::clamp(elite_percentage + rate_change(rng), 0.02, 0.25);
    tournament_size = std::clamp(tournament_size + tourney_change(rng), 3, 30);
    crossover_rate = std::clamp(crossover_rate + rate_change(rng), 0.5, 0.95);
}

//---------------------------------
// PatternMemory
//---------------------------------
// Definición del constructor de ParetoSolution
ParetoSolution::ParetoSolution(NodePtr t, double acc, double complexity_val) // <-- CAMBIADO compl a complexity_val
    : tree(std::move(t)),
      accuracy(acc),
      complexity(complexity_val), // <-- CAMBIADO compl a complexity_val
      dominated(false)
{}
// <------------------------------------------------------------->

// NUEVO: registra éxito o fracaso y actualiza memoria negativa
void PatternMemory::record_success(const NodePtr& tree, double fitness, bool is_success) {
    std::string pattern = extract_pattern(tree);
    if (pattern.length() > 50 || pattern.length() < 3) return;
    auto& info = patterns[pattern];
    info.pattern_str = pattern;
    info.uses++;
    if (is_success) {
        double improvement = (fitness < info.best_fitness && info.best_fitness < INF) ? 1.0 : 0.0;
        info.success_rate = ((info.success_rate * (info.uses - 1)) + improvement) / info.uses;
        info.best_fitness = std::min(info.best_fitness, fitness);
    } else {
        info.failures++;
    }
    // Actualizar fitness promedio
    if (info.avg_fitness >= INF) info.avg_fitness = fitness;
    else info.avg_fitness = (info.avg_fitness * (info.uses - 1) + fitness) / info.uses;
    // Si demasiados fracasos, añadir a memoria negativa
    if (info.failures >= min_failures_for_negative) {
        negative_patterns.insert(pattern);
    }
}

void PatternMemory::record_failure(const NodePtr& tree, double fitness) {
    record_success(tree, fitness, false);
}

// NUEVO: importar patrones exitosos y negativos de otra memoria
void PatternMemory::import_from(const PatternMemory& other) {
    for (const auto& [pat, info] : other.patterns) {
        auto& myinfo = patterns[pat];
        if (myinfo.uses < info.uses) myinfo = info;
    }
    for (const auto& pat : other.negative_patterns) {
        negative_patterns.insert(pat);
    }
}

// NUEVO: exportar patrones exitosos y negativos a otra memoria
void PatternMemory::export_to(PatternMemory& other) const {
    for (const auto& [pat, info] : patterns) {
        auto& oinfo = other.patterns[pat];
        if (oinfo.uses < info.uses) oinfo = info;
    }
    for (const auto& pat : negative_patterns) {
        other.negative_patterns.insert(pat);
    }
}

NodePtr PatternMemory::suggest_pattern_based_tree(int max_depth) {
    if (patterns.empty()) return nullptr; // patterns es miembro de la clase

    std::vector<std::pair<std::string, double>> candidates;
    // patterns es miembro de la clase
    for (const auto& [pattern_str, info] : patterns) {
        // Penalizar patrones negativos: peso 0 o muy bajo
        if (negative_patterns.count(pattern_str)) continue;
        if (info.uses >= min_uses_for_suggestion && (info.success_rate > 0.1 || info.best_fitness < 1.0)) {
             double weight = info.success_rate + (1.0 / (1.0 + info.best_fitness));
             // Usar emplace_back en lugar de push_back con {}
             candidates.emplace_back(pattern_str, weight); // <--- CAMBIADO
        }
    }

    if (candidates.empty()) return nullptr;

    std::vector<double> weights;
    std::transform(candidates.begin(), candidates.end(), std::back_inserter(weights),
                   [](const auto& p){ return p.second; });

    std::discrete_distribution<> dist(weights.begin(), weights.end());
    auto& rng = get_rng();
    int selected_idx = dist(rng);

    return parse_pattern(candidates[selected_idx].first, max_depth);
}

std::string PatternMemory::extract_pattern(const NodePtr& node) {
    if (!node) return "N"; // Null
    switch (node->type) {
        case NodeType::Constant:
            if (USE_INTEGER_MODE) {
                node->value = std::round(node->value);
            }
            return "#"; // Constant
        case NodeType::Variable: return "x"; // Variable
        case NodeType::Operator:
            return "(" + extract_pattern(node->left) + node->op + extract_pattern(node->right) + ")";
        default: return "?";
    }
}

// Basic pattern parser - needs refinement for robustness
NodePtr PatternMemory::parse_pattern(const std::string& pattern, int max_depth) {
    // This is a simplified placeholder. A real parser is complex.
    // For now, we just generate a random tree if parsing is too hard.
     // std::cout << "Trying pattern: " << pattern << std::endl; // Debug
    if (pattern == "#") {
        auto node = std::make_shared<Node>(NodeType::Constant);
        node->value = USE_INTEGER_MODE ? 1 : 1.0; // Valor fijo por defecto
        return node;
    }
    if (pattern == "x") {
        return std::make_shared<Node>(NodeType::Variable);
    }
     if (pattern == "N") {
        return nullptr;
    }
     if (pattern.length() > 3 && pattern.front() == '(' && pattern.back() == ')') {
          // Very basic parsing for (L op R) structure
          // This needs a proper recursive descent parser for real use
          // Find the main operator (tricky with nested parentheses)
          // For now, let's just generate a random tree as a fallback
          return generate_random_tree(max_depth);
     }


    // Fallback: generate a random tree if pattern is unrecognized or complex
    return generate_random_tree(max_depth);
}


//---------------------------------
// Pareto Optimizer
//---------------------------------
bool ParetoSolution::dominates(const ParetoSolution& other) const {
    // Check if this solution is strictly better in at least one objective
    // and not worse in any objective.
    bool better_in_one = (accuracy < other.accuracy) || (complexity < other.complexity);
    bool not_worse_in_any = (accuracy <= other.accuracy) && (complexity <= other.complexity);
    return better_in_one && not_worse_in_any;
}

// --- NSGA-II: crowding distance ---
static void compute_crowding_distance(std::vector<ParetoSolution>& front) {
    size_t n = front.size();
    if (n == 0) return;
    std::vector<double> crowding(n, 0.0);
    // Para cada objetivo (accuracy y complexity)
    for (int obj = 0; obj < 2; ++obj) {
        // Ordenar por el objetivo
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return (obj == 0) ? (front[a].accuracy < front[b].accuracy)
                              : (front[a].complexity < front[b].complexity);
        });
        // Extremos: infinito
        crowding[idx[0]] = crowding[idx[n-1]] = 1e9;
        double min_val = (obj == 0) ? front[idx[0]].accuracy : front[idx[0]].complexity;
        double max_val = (obj == 0) ? front[idx[n-1]].accuracy : front[idx[n-1]].complexity;
        if (max_val - min_val < 1e-12) continue; // Evitar división por cero
        // Interiores
        for (size_t i = 1; i < n-1; ++i) {
            double prev = (obj == 0) ? front[idx[i-1]].accuracy : front[idx[i-1]].complexity;
            double next = (obj == 0) ? front[idx[i+1]].accuracy : front[idx[i+1]].complexity;
            crowding[idx[i]] += (next - prev) / (max_val - min_val);
        }
    }
    // Guardar crowding en el campo 'dominated' (no se usa para esto)
    for (size_t i = 0; i < n; ++i) front[i].dominated = false; // Reset
    for (size_t i = 0; i < n; ++i) front[i].complexity = crowding[i]; // Reutilizo complexity para crowding solo para el recorte
}

void ParetoOptimizer::update(const std::vector<Individual>& population,
                           const std::vector<double>& targets,
                           const std::vector<double>& x_values) {
    // Add current front members to the candidate pool
    std::vector<ParetoSolution> candidates = pareto_front;

    // Add potentially new non-dominated solutions from the population
    for (const auto& ind : population) {
        if (ind.fitness_valid && ind.fitness < INF) { // Only consider valid, finite fitness individuals
            candidates.emplace_back(ind.tree, ind.fitness, static_cast<double>(tree_size(ind.tree)));
        }
    }

    // Determine dominance relationships among all candidates
    for (auto& sol1 : candidates) {
        sol1.dominated = false; // Reset dominance flag
        for (const auto& sol2 : candidates) {
            if (&sol1 == &sol2) continue; // Don't compare with self
            if (sol2.dominates(sol1)) {
                sol1.dominated = true;
                break; // Stop checking if dominated by someone
            }
        }
    }

    // Filter out dominated solutions
    pareto_front.clear();
    std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(pareto_front),
                 [](const auto& sol) { return !sol.dominated; });

    // --- NSGA-II: crowding distance y recorte por diversidad ---
    if (pareto_front.size() > max_front_size) {
        compute_crowding_distance(pareto_front);
        // Ordenar por crowding distance descendente (más diversidad primero)
        std::sort(pareto_front.begin(), pareto_front.end(), [](const ParetoSolution& a, const ParetoSolution& b) {
            return a.complexity > b.complexity; // Aquí complexity es crowding temporalmente
        });
        pareto_front.resize(max_front_size);
        // Restaurar complexity real si se requiere después
    }
}


std::vector<NodePtr> ParetoOptimizer::get_pareto_solutions() {
    std::vector<NodePtr> result;
    result.reserve(pareto_front.size());
    std::transform(pareto_front.begin(), pareto_front.end(), std::back_inserter(result),
                   [](const auto& sol) { return sol.tree; }); // Return clones? Or shared_ptrs? Depends on usage.
    return result;
}


//---------------------------------
// Domain Constraints
//---------------------------------
bool DomainConstraints::is_valid_recursive(const NodePtr& node) {
     if (!node) return true; // Null node is considered valid base case

     if (node->type == NodeType::Operator) {
         // Rule 1: Avoid division by constant zero (or very close)
         if (node->op == '/' && node->right && node->right->type == NodeType::Constant) {
             if (std::fabs(node->right->value) < 1e-9) {
                 // std::cerr << "Warning: Division by near zero constant found." << std::endl;
                 return false;
             }
         }
         // Rule 2: Avoid large constant exponents in power op
         if (node->op == '^' && node->right && node->right->type == NodeType::Constant) {
              if (node->right->value > 5 || node->right->value < -5) { // Allow small negative exponents?
                 // std::cerr << "Warning: Large exponent found." << std::endl;
                 return false;
              }
              // Also check if base is constant zero and exponent is non-positive
              if (node->left && node->left->type == NodeType::Constant &&
                  std::fabs(node->left->value) < 1e-9 && node->right->value <= 0) {
                   // std::cerr << "Warning: Power 0^non-positive found." << std::endl;
                  return false; // 0^0 or 0^-ve
              }
         }
         // Rule 3: Avoid redundant operations (e.g., * 1, / 1, + 0, - 0)
         if ((node->op == '*' || node->op == '/') && node->right && node->right->type == NodeType::Constant) {
             if (std::fabs(node->right->value - 1.0) < 1e-9) return false;
         }
         if ((node->op == '+' || node->op == '-') && node->right && node->right->type == NodeType::Constant) {
              if (std::fabs(node->right->value) < 1e-9) return false;
         }
         // Add symmetric checks for left where appropriate (e.g., 1 * tree)
         if (node->op == '*' && node->left && node->left->type == NodeType::Constant) {
              if (std::fabs(node->left->value - 1.0) < 1e-9) return false;
         }
         if (node->op == '+' && node->left && node->left->type == NodeType::Constant) {
              if (std::fabs(node->left->value) < 1e-9) return false;
         }


         // Recursively check children
         if (!is_valid_recursive(node->left) || !is_valid_recursive(node->right)) {
             return false;
         }
     }
     return true;
 }

bool DomainConstraints::is_valid(const NodePtr& tree) {
    return is_valid_recursive(tree);
}


NodePtr DomainConstraints::simplify_recursive(NodePtr node) {
    if (!node || node->type != NodeType::Operator) {
        return node; // Return terminals or null directly
    }

    // Recursively simplify children first (bottom-up)
    node->left = simplify_recursive(node->left);
    node->right = simplify_recursive(node->right);

    // --- Apply simplification rules ---

    // Rule: Constant folding (e.g., 3 + 5 -> 8)
    if (node->left && node->left->type == NodeType::Constant &&
        node->right && node->right->type == NodeType::Constant)
    {
        double result = evaluate_tree(node, 0.0); // Evaluate the constant subexpression
        if (!std::isnan(result) && !std::isinf(result)) {
            auto constant_node = std::make_shared<Node>(NodeType::Constant);
            if (USE_INTEGER_MODE) {
                constant_node->value = std::round(result);
            } else {
                constant_node->value = result;
            }
            return constant_node; // Replace operator node with the constant result
        }
        // else: couldn't evaluate (e.g., division by zero), leave as is for now
    }

     // Rule: Identity simplifications (A + 0 -> A, A * 1 -> A, etc.)
     if ((node->op == '+' || node->op == '-') && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value) < 1e-9) {
         return node->left; // A + 0 -> A, A - 0 -> A
     }
     if (node->op == '+' && node->left && node->left->type == NodeType::Constant && std::fabs(node->left->value) < 1e-9) {
         return node->right; // 0 + A -> A
     }
      if ((node->op == '*' || node->op == '/') && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value - 1.0) < 1e-9) {
         return node->left; // A * 1 -> A, A / 1 -> A
     }
     if (node->op == '*' && node->left && node->left->type == NodeType::Constant && std::fabs(node->left->value - 1.0) < 1e-9) {
         return node->right; // 1 * A -> A
     }
     // Rule: A * 0 -> 0, 0 * A -> 0
     if (node->op == '*' && ((node->left && node->left->type == NodeType::Constant && std::fabs(node->left->value) < 1e-9) ||
                             (node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value) < 1e-9))) {
         auto zero_node = std::make_shared<Node>(NodeType::Constant);
         zero_node->value = 0.0;
         return zero_node;
     }
     // Rule: A^1 -> A
      if (node->op == '^' && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value - 1.0) < 1e-9) {
          return node->left;
      }
     // Rule: A^0 -> 1 (unless A is 0, handled by evaluator)
       if (node->op == '^' && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value) < 1e-9) {
           auto one_node = std::make_shared<Node>(NodeType::Constant);
           one_node->value = 1.0;
           return one_node;
       }

    // --- Nuevas reglas simbólicas avanzadas ---
    // Simplificar doble negación: --A -> A
    if (node->op == '-' && node->left && node->left->type == NodeType::Operator && node->left->op == '-') {
        return node->left->left;
    }
    // Simplificar suma/resta de opuestos: A + (-A) -> 0, A - A -> 0
    if ((node->op == '+' || node->op == '-') && node->left && node->right) {
        if (tree_to_string(node->left) == tree_to_string(node->right)) {
            auto zero_node = std::make_shared<Node>(NodeType::Constant);
            zero_node->value = 0.0;
            return zero_node;
        }
        // A + (-A) o A - (-A)
        if (node->right->type == NodeType::Operator && node->right->op == '-' && tree_to_string(node->left) == tree_to_string(node->right->left)) {
            if (node->op == '+') {
                auto zero_node = std::make_shared<Node>(NodeType::Constant);
                zero_node->value = 0.0;
                return zero_node;
            } else if (node->op == '-') {
                return node->right->left; // A - (-A) = A + A
            }
        }
    }
    // Simplificar multiplicación por opuestos: A * (-1) -> -A
    if (node->op == '*' && node->right && node->right->type == NodeType::Constant && node->right->value == -1.0) {
        auto neg_node = std::make_shared<Node>(NodeType::Operator);
        neg_node->op = '-';
        neg_node->left = node->left;
        return neg_node;
    }
    // Simplificar potencia de potencia: (A^b)^c -> A^(b*c)
    if (node->op == '^' && node->left && node->left->type == NodeType::Operator && node->left->op == '^') {
        if (node->right && node->left->right && node->right->type == NodeType::Constant && node->left->right->type == NodeType::Constant) {
            auto new_exp = std::make_shared<Node>(NodeType::Constant);
            new_exp->value = node->right->value * node->left->right->value;
            auto new_pow = std::make_shared<Node>(NodeType::Operator);
            new_pow->op = '^';
            new_pow->left = node->left->left;
            new_pow->right = new_exp;
            return new_pow;
        }
    }

    // --- Fix invalid structures found during simplification ---
    // Fix division by constant zero -> make it division by 1 or return INF? Let's change to 1.
    if (node->op == '/' && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value) < 1e-9) {
        node->right->value = 1.0; // Change 0 to 1 in denominator
    }
    // Fix large constant exponents -> clamp them
    if (node->op == '^' && node->right && node->right->type == NodeType::Constant) {
        node->right->value = std::round(std::clamp(node->right->value, -4.0, 4.0)); // Clamp exponent to [-4, 4]
        if (std::fabs(node->right->value) < 1e-9) node->right->value = 0.0; // Ensure zero is exact if clamped near it
    }

    return node; // Return the potentially modified node
}


NodePtr DomainConstraints::fix_or_simplify(NodePtr tree) {
    // Clone first to avoid modifying the original shared tree structure if it came from population
    NodePtr simplified_tree = simplify_recursive(clone_tree(tree));
    // Maybe run simplify multiple times?
    // simplified_tree = simplify_recursive(simplified_tree);

    // Final check if it's now valid - if not, maybe return null or original?
    // Or just return the simplified version, even if still technically "invalid" by some rule
    // if (!is_valid(simplified_tree)) {
    //     std::cerr << "Warning: Tree still invalid after simplification: " << tree_to_string(simplified_tree) << std::endl;
         // Decide what to do: return original? return simplified anyway? return null?
         // Returning simplified is often best as it might be closer to valid.
    // }
    return simplified_tree;
}

//---------------------------------
// Local Improvement
//---------------------------------
std::pair<NodePtr, double> try_local_improvement(const NodePtr& tree,
                                                  double current_fitness,
                                                  const std::vector<double>& targets,
                                                  const std::vector<double>& x_values,
                                                  int attempts)
{
    NodePtr best_neighbor = tree; // Start with the original tree
    double best_neighbor_fitness = current_fitness;

    if (current_fitness >= INF) return {best_neighbor, best_neighbor_fitness}; // Don't try if fitness is infinite

    for (int i = 0; i < attempts; ++i) {
        // Generate a neighbor using a small mutation
        NodePtr neighbor = mutate_tree(tree, 1.0, 2); // Use high rate, small depth for local change
        neighbor = DomainConstraints::fix_or_simplify(neighbor); // Simplify/fix neighbor

        if (!neighbor) continue; // Skip if simplification failed badly

        double neighbor_fitness = evaluate_fitness(neighbor, targets, x_values);

        if (neighbor_fitness < best_neighbor_fitness) {
            best_neighbor = neighbor;
            best_neighbor_fitness = neighbor_fitness;
        }
    }

    // Hibridación ML: intentar reemplazo por regresión lineal en sub-árboles grandes
    std::vector<NodePtr*> nodes;
    collect_node_ptrs(best_neighbor, nodes);
    for (NodePtr* sub : nodes) {
        if (*sub && tree_size(*sub) >= 4) { // Solo sub-árboles de tamaño >= 4
            NodePtr lin = try_linear_regression_replacement(*sub, x_values, targets);
            if (lin) {
                *sub = lin;
                double fit_lin = evaluate_fitness(best_neighbor, targets, x_values);
                if (fit_lin < best_neighbor_fitness) {
                    best_neighbor_fitness = fit_lin;
                }
            }
        }
    }

    // Return the best neighbor found (could be the original tree)
    return {best_neighbor, best_neighbor_fitness};
}


//---------------------------------
// Target Pattern Detection
//---------------------------------
std::pair<std::string, double> detect_target_pattern(const std::vector<double>& targets) {
    if (targets.size() < 3) return {"none", 0.0};

    // Check for Arithmetic Series (constant difference)
    bool is_arithmetic = true;
    double diff = targets[1] - targets[0];
    for (size_t i = 2; i < targets.size(); ++i) {
        if (std::fabs((targets[i] - targets[i-1]) - diff) > 1e-6) { // Tolerance for floating point
            is_arithmetic = false;
            break;
        }
    }
    if (is_arithmetic) return {"arithmetic", diff};

    // Check for Geometric Series (constant ratio) - Avoid division by zero
    bool is_geometric = true;
    if (std::fabs(targets[0]) < 1e-9) {
        is_geometric = false; // Cannot start with zero for geometric check easily
        // Check if all are zero?
        bool all_zero = true;
        for(double t : targets) if (std::fabs(t) > 1e-9) all_zero = false;
        if(all_zero) return {"constant_zero", 0.0};

    } else {
        double ratio = targets[1] / targets[0];
        for (size_t i = 2; i < targets.size(); ++i) {
             if (std::fabs(targets[i-1]) < 1e-9) { // Avoid division by zero mid-sequence
                 if (std::fabs(targets[i]) > 1e-9) { // If current is non-zero, ratio breaks
                     is_geometric = false;
                     break;
                 } // else: Both are zero, ratio holds (ambiguously)
             } else {
                 if (std::fabs((targets[i] / targets[i-1]) - ratio) > 1e-6) {
                     is_geometric = false;
                     break;
                 }
             }
        }
        if (is_geometric) return {"geometric", ratio};
    }


    return {"none", 0.0};
}

// Generate tree based on simple patterns (assumes x = 1, 2, 3...)
NodePtr generate_pattern_based_tree(const std::string& pattern_type, double pattern_value) {
    // This is highly heuristic and assumes a simple relationship with 'x'
    // Needs adjustment based on actual X_VALUES if they aren't 1, 2, 3...

    if (pattern_type == "arithmetic") {
        // Pattern: a, a+d, a+2d ... -> f(x) = a + d*(x-1) ?
        // If targets[0] corresponds to x=X_VALUES[0]
        double a = TARGETS[0];
        double d = pattern_value;
        double x0 = X_VALUES[0];

        // f(x) = a + d * (x - x0) = (a - d*x0) + d*x
        auto root = std::make_shared<Node>(NodeType::Operator);
        root->op = '+';

        auto const_part = std::make_shared<Node>(NodeType::Constant);
        const_part->value = a - d * x0;

        auto var_part = std::make_shared<Node>(NodeType::Operator);
        var_part->op = '*';
        auto d_const = std::make_shared<Node>(NodeType::Constant);
        d_const->value = d;
        auto x_var = std::make_shared<Node>(NodeType::Variable);
        var_part->left = d_const;
        var_part->right = x_var;

        root->left = const_part;
        root->right = var_part;
        return DomainConstraints::fix_or_simplify(root);

    } else if (pattern_type == "geometric") {
        // Pattern: a, a*r, a*r^2 ... -> f(x) = a * r^(x-1) ?
        double a = TARGETS[0];
        double r = pattern_value;
         double x0 = X_VALUES[0];

        // f(x) = a * r^(x - x0) = (a / r^x0) * r^x
        if (std::fabs(r) < 1e-9) return nullptr; // Avoid r=0 case

        auto root = std::make_shared<Node>(NodeType::Operator);
        root->op = '*';

        auto const_part = std::make_shared<Node>(NodeType::Constant);
        const_part->value = a / std::pow(r, x0); // Potential issue if r^x0 is huge/tiny

        auto var_part = std::make_shared<Node>(NodeType::Operator);
        var_part->op = '^';
        auto r_const = std::make_shared<Node>(NodeType::Constant);
        r_const->value = r;
        auto x_var = std::make_shared<Node>(NodeType::Variable);
        var_part->left = r_const;
        var_part->right = x_var;

        root->left = const_part;
        root->right = var_part;
        return DomainConstraints::fix_or_simplify(root);
    }
     else if (pattern_type == "constant_zero") {
         auto node = std::make_shared<Node>(NodeType::Constant);
         node->value = 0.0;
         return node;
     }

    return nullptr; // No simple pattern tree generated
}

// Hibridación ML: Ajuste de regresión lineal sobre un sub-árbol
NodePtr try_linear_regression_replacement(const NodePtr& subtree,
                                          const std::vector<double>& x_values,
                                          const std::vector<double>& targets) {
    if (!subtree) return nullptr;
    size_t n = x_values.size();
    if (n < 2) return nullptr;
    // Generar los valores de salida del sub-árbol
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        y[i] = evaluate_tree(subtree, x_values[i]);
        if (std::isnan(y[i]) || std::isinf(y[i])) return nullptr;
    }
    // Ajuste de regresión lineal y = a + b*x
    double sum_x = std::accumulate(x_values.begin(), x_values.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double sum_xx = 0.0, sum_xy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_xx += x_values[i] * x_values[i];
        sum_xy += x_values[i] * y[i];
    }
    double denom = n * sum_xx - sum_x * sum_x;
    if (std::fabs(denom) < 1e-12) return nullptr;
    double b = (n * sum_xy - sum_x * sum_y) / denom;
    double a = (sum_y - b * sum_x) / n;
    // Construir árbol equivalente a la regresión lineal
    auto plus = std::make_shared<Node>(NodeType::Operator);
    plus->op = '+';
    auto a_node = std::make_shared<Node>(NodeType::Constant);
    a_node->value = a;
    auto mult = std::make_shared<Node>(NodeType::Operator);
    mult->op = '*';
    auto b_node = std::make_shared<Node>(NodeType::Constant);
    b_node->value = b;
    auto x_node = std::make_shared<Node>(NodeType::Variable);
    mult->left = b_node;
    mult->right = x_node;
    plus->left = a_node;
    plus->right = mult;
    // Evaluar error de la regresión
    double err_reg = 0.0, err_orig = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double y_reg = evaluate_tree(plus, x_values[i]);
        err_reg += std::pow(y_reg - y[i], 2);
        err_orig += std::pow(y[i] - targets[i], 2);
    }
    // Solo reemplazar si la regresión es igual o mejor y el árbol es más simple
    if (err_reg <= err_orig && tree_size(plus) < tree_size(subtree)) {
        return plus;
    }
    return nullptr;
}