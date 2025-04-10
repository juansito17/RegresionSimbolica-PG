#include "Fitness.h"
#include "Globals.h" // For INF, COMPLEXITY_PENALTY_FACTOR
#include <cmath>
#include <limits>
#include <vector>

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