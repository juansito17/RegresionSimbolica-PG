#include "AdvancedFeatures.h"
#include "Globals.h"
#include "GeneticOperators.h"
#include "Fitness.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

//---------------------------------
// EvolutionParameters
//---------------------------------
EvolutionParameters EvolutionParameters::create_default() {
    // Usa constantes globales para los valores por defecto
    return {BASE_MUTATION_RATE, BASE_ELITE_PERCENTAGE, DEFAULT_TOURNAMENT_SIZE, DEFAULT_CROSSOVER_RATE};
}

void EvolutionParameters::mutate(int stagnation_counter) {
    auto& rng = get_rng();
    double aggression_factor = 1.0;
    // Ajuste del factor de agresión basado en el estancamiento
    if (stagnation_counter > STAGNATION_LIMIT_ISLAND / 2) {
        // Aumenta la agresión si hay estancamiento significativo
        aggression_factor = 1.0 + (static_cast<double>(stagnation_counter - STAGNATION_LIMIT_ISLAND / 2) / (STAGNATION_LIMIT_ISLAND / 2.0)) * 0.5; // Escala de 1.0 a 1.5
        aggression_factor = std::min(aggression_factor, 2.0); // Limitar la agresión máxima
    } else if (stagnation_counter < STAGNATION_LIMIT_ISLAND / 4 && stagnation_counter > 0) {
        // Reduce la agresión si no hay mucho estancamiento, pero no es 0
        aggression_factor = 1.0 - (static_cast<double>(STAGNATION_LIMIT_ISLAND / 4 - stagnation_counter) / (STAGNATION_LIMIT_ISLAND / 4.0)) * 0.5; // Escala de 0.5 a 1.0
        aggression_factor = std::max(aggression_factor, 0.5); // Limitar la agresión mínima
    } else if (stagnation_counter == 0) {
        // Muy poco estancamiento, cambios muy pequeños
        aggression_factor = 0.2; // Cambios muy conservadores
    }

    std::uniform_real_distribution<double> base_rate_change(-0.05, 0.05);
    std::uniform_int_distribution<int> base_tourney_change(-2, 2);

    double rate_change_val = base_rate_change(rng) * aggression_factor;
    int tourney_change_val = static_cast<int>(std::round(base_tourney_change(rng) * aggression_factor));
    
    // Asegurar que haya algún cambio si la agresión es alta y el cambio base es 0
    if (aggression_factor > 1.0 && tourney_change_val == 0 && base_tourney_change(rng) != 0) {
         tourney_change_val = (base_tourney_change(rng) > 0) ? 1 : -1;
    }

    // Definir límites dinámicos para los parámetros
    double min_mutation = 0.05;
    double max_mutation_base = 0.5;
    double max_mutation = min_mutation + (max_mutation_base - min_mutation) * (1.0 + aggression_factor / 2.0);

    double min_elite = 0.02;
    double max_elite_base = 0.25;
    double max_elite = min_elite + (max_elite_base - min_elite) * (1.0 + aggression_factor / 2.0);

    int min_tournament = 3;
    int max_tournament_base = 30;
    int max_tournament = min_tournament + static_cast<int>((max_tournament_base - min_tournament) * (1.0 + aggression_factor / 2.0));

    // Aplicar los cambios y asegurar que estén dentro de los límites
    mutation_rate = std::clamp(mutation_rate + rate_change_val, min_mutation, max_mutation);
    elite_percentage = std::clamp(elite_percentage + rate_change_val, min_elite, max_elite);
    tournament_size = std::clamp(tournament_size + tourney_change_val, min_tournament, max_tournament);
    crossover_rate = std::clamp(crossover_rate + rate_change_val, 0.5, 0.95);
}

//---------------------------------
// PatternMemory
//---------------------------------
void PatternMemory::record_success(const NodePtr& tree, double fitness) {
    std::string pattern = extract_pattern(tree);
    if (pattern.empty() || pattern.length() > 50 || pattern.length() < 3 || pattern == "N") return;
    auto it = patterns.find(pattern);
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
    if (patterns.empty()) return nullptr;
    std::vector<std::pair<std::string, double>> candidates;
    for (const auto& [pattern_str, info] : patterns) {
        if (info.uses >= PATTERN_MEM_MIN_USES && (info.success_rate > 0.1 || info.best_fitness < PATTERN_RECORD_FITNESS_THRESHOLD)) {
             double weight = info.success_rate + (1.0 / (1.0 + info.best_fitness));
             candidates.emplace_back(pattern_str, weight);
        }
    }
    if (candidates.empty()) return nullptr;
    std::vector<double> weights;
    std::transform(candidates.begin(), candidates.end(), std::back_inserter(weights), [](const auto& p){ return p.second; });
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    auto& rng = get_rng();
    int selected_idx = dist(rng);
    return parse_pattern(candidates[selected_idx].first, max_depth);
}

std::string PatternMemory::extract_pattern(const NodePtr& node) {
    if (!node) return "N";
    switch (node->type) {
        case NodeType::Constant: return "#";
        case NodeType::Variable: return "x";
        case NodeType::Operator:
            return "(" + extract_pattern(node->left) + node->op + extract_pattern(node->right) + ")";
        default: return "?";
    }
}

NodePtr PatternMemory::parse_pattern(const std::string& pattern, int max_depth) {
    // Placeholder implementation
    if (pattern == "#") {
        auto node = std::make_shared<Node>(NodeType::Constant);
        if (FORCE_INTEGER_CONSTANTS) { std::uniform_int_distribution<int> cd(CONSTANT_INT_MIN_VALUE, CONSTANT_INT_MAX_VALUE); node->value = static_cast<double>(cd(get_rng())); }
        else { std::uniform_real_distribution<double> cd(CONSTANT_MIN_VALUE, CONSTANT_MAX_VALUE); node->value = cd(get_rng()); }
        if(std::fabs(node->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) node->value = 0.0;
        return node;
    }
    if (pattern == "x") return std::make_shared<Node>(NodeType::Variable);
    if (pattern == "N") return nullptr;
    if (pattern.length() > 3 && pattern.front() == '(' && pattern.back() == ')') {
          return generate_random_tree(max_depth); // Fallback
     }
    return generate_random_tree(max_depth); // Fallback
}

//---------------------------------
// Pareto Optimizer
//---------------------------------
ParetoSolution::ParetoSolution(NodePtr t, double acc, double complexity_val) : tree(std::move(t)), accuracy(acc), complexity(complexity_val), dominated(false) {}

bool ParetoSolution::dominates(const ParetoSolution& other) const {
    bool better_in_one = (accuracy < other.accuracy) || (complexity < other.complexity);
    bool not_worse_in_any = (accuracy <= other.accuracy) && (complexity <= other.complexity);
    return better_in_one && not_worse_in_any;
}

void ParetoOptimizer::update(const std::vector<Individual>& population, const std::vector<double>& targets, const std::vector<double>& x_values) {
    std::vector<ParetoSolution> candidates = pareto_front;
    for (const auto& ind : population) {
        if (ind.tree && ind.fitness_valid && ind.fitness < INF) {
            candidates.emplace_back(ind.tree, ind.fitness, static_cast<double>(tree_size(ind.tree)));
        }
    }
    for (auto& sol1 : candidates) {
        sol1.dominated = false;
        for (const auto& sol2 : candidates) {
            if (&sol1 == &sol2) continue;
            if (sol2.dominates(sol1)) { sol1.dominated = true; break; }
        }
    }
    pareto_front.clear();
    std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(pareto_front),
                 [](const auto& sol) { return !sol.dominated; });
    if (pareto_front.size() > PARETO_MAX_FRONT_SIZE) {
        std::sort(pareto_front.begin(), pareto_front.end(), [](const auto& a, const auto& b){ return a.accuracy < b.accuracy; });
        pareto_front.resize(PARETO_MAX_FRONT_SIZE);
    }
}

std::vector<NodePtr> ParetoOptimizer::get_pareto_solutions() {
    std::vector<NodePtr> result;
    result.reserve(pareto_front.size());
    std::transform(pareto_front.begin(), pareto_front.end(), std::back_inserter(result),
                   [](const auto& sol) { return sol.tree; });
    return result;
}

//---------------------------------
// Domain Constraints
//---------------------------------
bool DomainConstraints::is_valid_recursive(const NodePtr& node) {
     if (!node) return true;
     if (node->type == NodeType::Operator) {
         if (node->op == '/' && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return false;
         if (node->op == '^') { // Solo chequear 0^negativo/0
              if (node->left && node->left->type == NodeType::Constant && std::fabs(node->left->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE &&
                  node->right && node->right->type == NodeType::Constant && node->right->value <= SIMPLIFY_NEAR_ZERO_TOLERANCE) {
                      return false;
              }
         }
         if ((node->op == '*' || node->op == '/') && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value - 1.0) < SIMPLIFY_NEAR_ONE_TOLERANCE) return false;
         if ((node->op == '+' || node->op == '-') && node->right && node->right->type == NodeType::Constant && std::fabs(node->right->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return false;
         if (node->op == '*' && node->left && node->left->type == NodeType::Constant && std::fabs(node->left->value - 1.0) < SIMPLIFY_NEAR_ONE_TOLERANCE) return false;
         if (node->op == '+' && node->left && node->left->type == NodeType::Constant && std::fabs(node->left->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return false;
         if (!is_valid_recursive(node->left) || !is_valid_recursive(node->right)) return false;
     }
     return true;
 }

bool DomainConstraints::is_valid(const NodePtr& tree) {
    return is_valid_recursive(tree);
}

NodePtr DomainConstraints::simplify_recursive(NodePtr node) {
    if (!node || node->type != NodeType::Operator) return node;
    node->left = simplify_recursive(node->left);
    node->right = simplify_recursive(node->right);

    // Manejo de hijos nulos
    if (node->left && !node->right) return node->left;
    if (!node->left && node->right) return node->right;
    if (!node->left && !node->right) { auto cn = std::make_shared<Node>(NodeType::Constant); cn->value = 1.0; return cn; }

    // Constant Folding
    if (node->left->type == NodeType::Constant && node->right->type == NodeType::Constant) {
        try {
            double result = evaluate_tree(node, 0.0);
            if (!std::isnan(result) && !std::isinf(result)) {
                auto cn = std::make_shared<Node>(NodeType::Constant);
                if (FORCE_INTEGER_CONSTANTS) cn->value = std::round(result); else cn->value = result;
                if (std::fabs(cn->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) cn->value = 0.0; return cn;
            }
        } catch (const std::exception&) {}
    }
    // Identity Simplifications & Fixes
     if ((node->op == '+' || node->op == '-') && node->right->type == NodeType::Constant && std::fabs(node->right->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return node->left;
     if (node->op == '+' && node->left->type == NodeType::Constant && std::fabs(node->left->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return node->right;
     if ((node->op == '*' || node->op == '/') && node->right->type == NodeType::Constant && std::fabs(node->right->value - 1.0) < SIMPLIFY_NEAR_ONE_TOLERANCE) return node->left;
     if (node->op == '*' && node->left && node->left->type == NodeType::Constant && std::fabs(node->left->value - 1.0) < SIMPLIFY_NEAR_ONE_TOLERANCE) return node->right;
     if (node->op == '*' && ((node->left->type == NodeType::Constant && std::fabs(node->left->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) || (node->right->type == NodeType::Constant && std::fabs(node->right->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE))) { auto z = std::make_shared<Node>(NodeType::Constant); z->value = 0.0; return z; }
     if (node->op == '^' && node->right->type == NodeType::Constant && std::fabs(node->right->value - 1.0) < SIMPLIFY_NEAR_ONE_TOLERANCE) return node->left; // A^1 -> A
     if (node->op == '^' && node->right->type == NodeType::Constant && std::fabs(node->right->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) { auto o = std::make_shared<Node>(NodeType::Constant); o->value = 1.0; return o; } // A^0 -> 1
    // Fix div by zero (constante)
    if (node->op == '/' && node->right->type == NodeType::Constant && std::fabs(node->right->value) < SIMPLIFY_NEAR_ZERO_TOLERANCE) node->right->value = 1.0;
    // Ya no se hace clamp de exponente constante aquí, se quitó la restricción

    return node;
}

NodePtr DomainConstraints::fix_or_simplify(NodePtr tree) {
    if (!tree) return nullptr;
    NodePtr cloned_tree = clone_tree(tree);
    NodePtr simplified_tree = simplify_recursive(cloned_tree);
    return simplified_tree;
}

//---------------------------------
// Local Improvement
//---------------------------------
#if USE_GPU_ACCELERATION
std::pair<NodePtr, double> try_local_improvement(const NodePtr& tree, double current_fitness, const std::vector<double>& targets, const std::vector<double>& x_values, int attempts, double* d_targets, double* d_x_values) {
    NodePtr best_neighbor = tree;
    double best_neighbor_fitness = current_fitness;
    if (current_fitness >= INF) return {best_neighbor, best_neighbor_fitness};
    for (int i = 0; i < attempts; ++i) {
        NodePtr neighbor = mutate_tree(tree, 1.0, 2);
        neighbor = DomainConstraints::fix_or_simplify(neighbor);
        if (!neighbor) continue;
        double neighbor_fitness = evaluate_fitness(neighbor, targets, x_values, d_targets, d_x_values);
        if (neighbor_fitness < best_neighbor_fitness) {
            best_neighbor = neighbor;
            best_neighbor_fitness = neighbor_fitness;
        }
    }
    return {best_neighbor, best_neighbor_fitness};
}
#else
std::pair<NodePtr, double> try_local_improvement(const NodePtr& tree, double current_fitness, const std::vector<double>& targets, const std::vector<double>& x_values, int attempts) {
    NodePtr best_neighbor = tree;
    double best_neighbor_fitness = current_fitness;
    if (current_fitness >= INF) return {best_neighbor, best_neighbor_fitness};
    for (int i = 0; i < attempts; ++i) {
        NodePtr neighbor = mutate_tree(tree, 1.0, 2);
        neighbor = DomainConstraints::fix_or_simplify(neighbor);
        if (!neighbor) continue;
        double neighbor_fitness = evaluate_fitness(neighbor, targets, x_values);
        if (neighbor_fitness < best_neighbor_fitness) {
            best_neighbor = neighbor;
            best_neighbor_fitness = neighbor_fitness;
        }
    }
    return {best_neighbor, best_neighbor_fitness};
}
#endif

//---------------------------------
// Target Pattern Detection
//---------------------------------
std::pair<std::string, double> detect_target_pattern(const std::vector<double>& targets) {
    if (targets.size() < 3) return {"none", 0.0};
    bool is_arithmetic = true; double diff = targets[1] - targets[0];
    for (size_t i = 2; i < targets.size(); ++i) if (std::fabs((targets[i] - targets[i-1]) - diff) > 1e-6) { is_arithmetic = false; break; }
    if (is_arithmetic) return {"arithmetic", diff};
    bool is_geometric = true;
    if (std::fabs(targets[0]) < 1e-9) {
        bool all_zero = true; for(double t : targets) if (std::fabs(t) > 1e-9) { all_zero = false; break; }
        if(all_zero) return {"constant_zero", 0.0}; else is_geometric = false;
    }
    if (is_geometric && std::fabs(targets[0]) >= 1e-9) {
        double ratio = targets[1] / targets[0];
        for (size_t i = 2; i < targets.size(); ++i) {
             if (std::fabs(targets[i-1]) < 1e-9) { if (std::fabs(targets[i]) > 1e-9) { is_geometric = false; break; } }
             else { if (std::fabs((targets[i] / targets[i-1]) - ratio) > 1e-6) { is_geometric = false; break; } }
        }
        if (is_geometric) return {"geometric", ratio};
    }
    return {"none", 0.0};
}

//---------------------------------
// Generate Pattern Based Tree
//---------------------------------
NodePtr generate_pattern_based_tree(const std::string& pattern_type, double pattern_value) {
    if (X_VALUES.empty() || TARGETS.empty()) return nullptr;
    double a = TARGETS[0]; double x0 = X_VALUES[0];
    if (pattern_type == "arithmetic") {
        double d = pattern_value; auto root = std::make_shared<Node>(NodeType::Operator); root->op = '+';
        auto cp = std::make_shared<Node>(NodeType::Constant); double cv = a - d * x0; if (FORCE_INTEGER_CONSTANTS) cv = std::round(cv); cp->value = (std::fabs(cv) < SIMPLIFY_NEAR_ZERO_TOLERANCE) ? 0.0 : cv;
        auto vp = std::make_shared<Node>(NodeType::Operator); vp->op = '*';
        auto dc = std::make_shared<Node>(NodeType::Constant); double dv = d; if (FORCE_INTEGER_CONSTANTS) dv = std::round(dv); dc->value = (std::fabs(dv) < SIMPLIFY_NEAR_ZERO_TOLERANCE) ? 0.0 : dv;
        auto xv = std::make_shared<Node>(NodeType::Variable); vp->left = dc; vp->right = xv;
        root->left = cp; root->right = vp; return DomainConstraints::fix_or_simplify(root);
    } else if (pattern_type == "geometric") {
        double r = pattern_value; if (std::fabs(r) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return nullptr;
        auto root = std::make_shared<Node>(NodeType::Operator); root->op = '*';
        auto cp = std::make_shared<Node>(NodeType::Constant); double rpx0 = std::pow(r, x0); if (std::fabs(rpx0) < 1e-100) return nullptr;
        double cv = a / rpx0; if (FORCE_INTEGER_CONSTANTS) cv = std::round(cv); cp->value = (std::fabs(cv) < SIMPLIFY_NEAR_ZERO_TOLERANCE) ? 0.0 : cv;
        auto vp = std::make_shared<Node>(NodeType::Operator); vp->op = '^';
        auto rc = std::make_shared<Node>(NodeType::Constant); double rv = r; if (FORCE_INTEGER_CONSTANTS) rv = std::round(rv); rc->value = (std::fabs(rv) < SIMPLIFY_NEAR_ZERO_TOLERANCE) ? 0.0 : rv;
        auto xv = std::make_shared<Node>(NodeType::Variable); vp->left = rc; vp->right = xv;
        root->left = cp; root->right = vp; return DomainConstraints::fix_or_simplify(root);
    } else if (pattern_type == "constant_zero") {
         auto node = std::make_shared<Node>(NodeType::Constant); node->value = 0.0; return node;
     }
    return nullptr; // No pattern tree generated
}
