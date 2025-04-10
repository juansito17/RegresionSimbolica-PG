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

    // Resize results vector
    fitness_results.resize(trees.size());

    // Collect trees that need evaluation (not in cache)
    std::vector<NodePtr> uncached_trees_all;
    std::vector<size_t> uncached_indices_all;
    std::vector<std::string> tree_hashes(trees.size());
    std::vector<bool> needs_evaluation(trees.size(), false);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (size_t i = 0; i < trees.size(); i++) {
            if (!trees[i]) { // Handle null trees immediately
                 fitness_results[i] = INF;
                 continue;
            }
            tree_hashes[i] = get_tree_hash(trees[i]);
            auto it = evaluation_cache.find(tree_hashes[i]);
            if (it != evaluation_cache.end() && it->second.valid) {
                fitness_results[i] = it->second.fitness;
            } else {
                uncached_trees_all.push_back(trees[i]);
                uncached_indices_all.push_back(i);
                needs_evaluation[i] = true; // Mark original index as needing evaluation
            }
        }
    }

    if (!uncached_trees_all.empty()) {
        const size_t total_uncached = uncached_trees_all.size();
        // Increased batch size significantly
        const size_t gpu_batch_size = 10000; // Increased from 2000
        size_t processed_count = 0;

        while(processed_count < total_uncached) {
            size_t current_batch_size = std::min(gpu_batch_size, total_uncached - processed_count);
            std::vector<NodePtr> batch_trees(uncached_trees_all.begin() + processed_count,
                                             uncached_trees_all.begin() + processed_count + current_batch_size);
            std::vector<size_t> batch_indices(uncached_indices_all.begin() + processed_count,
                                              uncached_indices_all.begin() + processed_count + current_batch_size);
            std::vector<double> batch_results(current_batch_size);

            // Try GPU evaluation for the current batch
            bool gpu_success = true;
            try {
                evaluatePopulationGPU(batch_trees, x_values, targets, batch_results);
            } catch (const std::exception& e) {
                // Keep error message for actual failures
                std::cerr << "GPU evaluation failed for batch: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU evaluation for this batch..." << std::endl;
                gpu_success = false;
            }

            // If GPU failed for the batch, fallback to CPU
            if (!gpu_success) {
                batch_results.resize(current_batch_size); // Ensure size
                #pragma omp parallel for
                for (size_t i = 0; i < current_batch_size; ++i) {
                    if (batch_trees[i]) {
                        // Calculate raw fitness first for CPU path
                        batch_results[i] = calculate_raw_fitness(batch_trees[i], targets, x_values);
                    } else {
                        batch_results[i] = INF;
                    }
                }
            }

            // Apply complexity penalty (always done on CPU after getting raw fitness)
            // and update cache and final results vector
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                for (size_t i = 0; i < current_batch_size; ++i) {
                    size_t orig_idx = batch_indices[i];
                    double fitness = batch_results[i]; // Raw fitness from GPU or CPU fallback

                    if (!std::isinf(fitness) && !std::isnan(fitness) && batch_trees[i]) {
                        double penalty = static_cast<double>(tree_size(batch_trees[i])) * COMPLEXITY_PENALTY_FACTOR;
                        fitness *= (1.0 + penalty);
                         // Final check for NaN/Inf after penalty application
                         if (std::isnan(fitness) || std::isinf(fitness)) {
                            fitness = INF;
                         }
                    } else {
                         fitness = INF; // Ensure invalid results are INF
                    }

                    fitness_results[orig_idx] = fitness;

                    // Update cache only if fitness is valid
                    if (!std::isinf(fitness) && !std::isnan(fitness)) {
                         evaluation_cache[tree_hashes[orig_idx]] = {
                            fitness,
                            tree_to_string(batch_trees[i]), // Store string representation if needed
                            true
                        };
                    } else {
                         // Optionally cache invalid results too, or just skip
                         evaluation_cache[tree_hashes[orig_idx]] = { INF, "", false };
                    }
                }
            }
             processed_count += current_batch_size;
        } // End of batch processing loop

    } // End if !uncached_trees_all.empty()

    // Optional: Clean cache if it gets too large
    if (evaluation_cache.size() > 100000) { // Adjust this number as needed
        std::lock_guard<std::mutex> lock(cache_mutex);
        // Consider a more sophisticated cache eviction strategy than clear() if needed
        evaluation_cache.clear();
    }
}