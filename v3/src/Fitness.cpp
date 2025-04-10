#include "Fitness.h"
#include "Globals.h"
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <sstream>
#include <mutex>

// Cache para evitar re-evaluaciones innecesarias
struct TreeCacheEntry {
    double fitness;
    std::string tree_str;
    bool valid;
};

std::unordered_map<std::string, TreeCacheEntry> evaluation_cache;
std::mutex cache_mutex;

std::string get_tree_hash(const NodePtr& tree) {
    if (!tree) return "null";
    std::stringstream ss;
    ss << tree_to_string(tree);
    return ss.str();
}

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

    double raw_fitness = calculate_raw_fitness(tree, targets, x_values);

    if (std::isinf(raw_fitness)) {
        return INF; // Propagate infinity
    }

    // Add complexity penalty
    // Penalty increases with tree size
    double penalty = static_cast<double>(tree_size(tree)) * COMPLEXITY_PENALTY_FACTOR;

    // Apply penalty multiplicatively
    // Ensure penalty doesn't make fitness zero or negative if raw_fitness is very small
    return raw_fitness * (1.0 + penalty);
}

void evaluate_population_fitness(
    const std::vector<NodePtr>& trees,
    const std::vector<double>& targets,
    const std::vector<double>& x_values,
    std::vector<double>& fitness_results) {

    std::vector<NodePtr> uncached_trees_all;
    std::vector<size_t> uncached_indices_all;

    // Prepare all trees for GPU evaluation
    for (size_t i = 0; i < trees.size(); ++i) {
        uncached_trees_all.push_back(trees[i]);
        uncached_indices_all.push_back(i);
    }

    size_t total_uncached = uncached_trees_all.size();
    size_t processed_count = 0;

    if (!uncached_trees_all.empty()) {
        // Evaluate all trees in a single GPU batch
        size_t current_batch_size = total_uncached - processed_count;
        std::vector<NodePtr> batch_trees(uncached_trees_all.begin() + processed_count,
                                         uncached_trees_all.begin() + processed_count + current_batch_size);
        std::vector<size_t> batch_indices(uncached_indices_all.begin() + processed_count,
                                          uncached_indices_all.begin() + processed_count + current_batch_size);
        std::vector<double> batch_results(current_batch_size);

        // GPU evaluation
        evaluatePopulationGPU(batch_trees, x_values, targets, batch_results);

        // Update results vector
        for (size_t i = 0; i < current_batch_size; ++i) {
            size_t orig_idx = batch_indices[i];
            fitness_results[orig_idx] = batch_results[i];
        }
    }
}