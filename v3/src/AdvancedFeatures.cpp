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

void PatternMemory::record_success(const NodePtr& tree, double fitness) {
    std::string pattern = extract_pattern(tree);
    if (pattern.length() > 50 || pattern.length() < 3) return;

    auto it = patterns.find(pattern); // patterns es miembro de la clase
    if (it == patterns.end()) {
        patterns[pattern] = {pattern, fitness, 1, (fitness < INF ? 1.0 : 0.0)};
    } else {
        auto& p = it->second;
        p.uses++;
        double improvement = (fitness < p.best_fitness && p.best_fitness < INF) ? 1.0 : 0.0;
        p.success_rate = ((p.success_rate * (p.uses - 1)) + improvement) / p.uses;
        p.best_fitness = std::min(p.best_fitness, fitness);
    }
}

NodePtr PatternMemory::suggest_pattern_based_tree(int max_depth) {
    if (patterns.empty()) return nullptr; // patterns es miembro de la clase

    std::vector<std::pair<std::string, double>> candidates;
    // patterns es miembro de la clase
    for (const auto& [pattern_str, info] : patterns) {
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
        case NodeType::Constant: return "#"; // Constant
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
         std::uniform_int_distribution<int> const_dist(1, 10);
         node->value = const_dist(get_rng());
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

void ParetoOptimizer::update(const std::vector<Individual>& population,
                           const std::vector<double>& /*targets*/,    // Marked as unused
                           const std::vector<double>& /*x_values*/) { // Marked as unused
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

    // Optional: Prune the front if it gets too large
    if (pareto_front.size() > max_front_size) {
        // Simple pruning: sort by one objective (e.g., accuracy) and keep the best
        std::sort(pareto_front.begin(), pareto_front.end(), [](const auto& a, const auto& b){
            return a.accuracy < b.accuracy;
        });
        pareto_front.resize(max_front_size);
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
            constant_node->value = result;
            // std::cout << "Simplified: " << tree_to_string(node) << " -> " << result << std::endl; // Debug
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