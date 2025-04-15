#include "Fitness.h"
#include "Globals.h" // For INF, COMPLEXITY_PENALTY_FACTOR
#include <cmath>
#include <limits>
#include <vector>
#include <unordered_map>
#include <mutex>

// --- Fitness cache global ---
namespace {
    std::unordered_map<size_t, double> fitness_cache;
    std::mutex fitness_cache_mutex;
}

double COMPLEXITY_PENALTY_FACTOR = 0.04;

double calculate_raw_fitness(const NodePtr& tree,
                             const std::vector<double>& targets,
                             const std::vector<double>& x_values) {
    double error_sum = 0.0;
    bool all_precise = true;

    if (x_values.size() != targets.size() || x_values.empty()) {
        // Handle error: mismatched sizes or empty input
        return INF;
    }

    for (size_t i = 0; i < x_values.size(); ++i) {
        double val = evaluate_tree(tree, x_values[i]);

        // Check for invalid results from evaluation
        if (std::isnan(val) || std::isinf(val)) {
            return INF; // Penalize invalid expressions heavily
        }

        double diff = std::fabs(val - targets[i]);

        if (diff >= 0.001) {
            all_precise = false;
        }

        // Use pow(diff, 1.3) as in the original code
        error_sum += std::pow(diff, 1.3);

        // Optional: Add check for massive errors to prevent overflow in error_sum
        if (error_sum > 1e100) { // Adjust threshold as needed
           return INF;
        }
    }

     // Bonus for finding an exact solution
    if (all_precise) {
        error_sum *= 0.0001; // Significant bonus
    }

    // Check if the final sum itself is problematic
    if (std::isnan(error_sum) || std::isinf(error_sum)) {
        return INF;
    }

    return error_sum;
}

double evaluate_fitness(const NodePtr& tree,
                        const std::vector<double>& targets,
                        const std::vector<double>& x_values) {
    // --- Poda avanzada: evitar árboles inútiles ---
    if (!tree || tree_size(tree) < 2) return INF; // Árbol demasiado pequeño
    // Si el árbol es constante (no depende de la variable), es inútil
    bool has_variable = false;
    std::function<void(const NodePtr&)> check_var = [&](const NodePtr& n) {
        if (!n) return;
        if (n->type == NodeType::Variable) has_variable = true;
        if (n->left) check_var(n->left);
        if (n->right) check_var(n->right);
    };
    check_var(tree);
    if (!has_variable) return INF;
    // Si el árbol es una constante pura
    if (tree->type == NodeType::Constant) return INF;

    // --- Fitness cache: buscar primero en el caché ---
    size_t h = tree_structural_hash(tree);
    {
        std::lock_guard<std::mutex> lock(fitness_cache_mutex);
        auto it = fitness_cache.find(h);
        if (it != fitness_cache.end()) {
            return it->second;
        }
    }

    double raw_fitness = calculate_raw_fitness(tree, targets, x_values);

    if (std::isinf(raw_fitness)) {
        // Guardar también los infinitos para evitar recalcular
        std::lock_guard<std::mutex> lock(fitness_cache_mutex);
        fitness_cache[h] = INF;
        return INF;
    }

    // Add complexity penalty
    // Penalty increases with tree size
    double penalty = static_cast<double>(tree_size(tree)) * COMPLEXITY_PENALTY_FACTOR;

    // Apply penalty multiplicatively
    // Ensure penalty doesn't make fitness zero or negative if raw_fitness is very small
    double result = raw_fitness * (1.0 + penalty);

    {
        std::lock_guard<std::mutex> lock(fitness_cache_mutex);
        fitness_cache[h] = result;
    }

    return result;
}