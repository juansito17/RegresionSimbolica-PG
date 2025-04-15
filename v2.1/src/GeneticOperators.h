// ============================================================
// Archivo: src/GeneticOperators.h
// ============================================================
#ifndef GENETICOPERATORS_H
#define GENETICOPERATORS_H

#include "ExpressionTree.h"
#include "Globals.h" // <--- AÑADIDO: Incluir Globals.h para INF
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_set>

// Structure to hold an individual and its cached fitness
struct Individual {
    NodePtr tree;
    double fitness = INF; // Usa INF de Globals.h
    bool fitness_valid = false;

    // Default constructor
    Individual() = default;
    // Constructor from tree
    explicit Individual(NodePtr t) : tree(std::move(t)) {}

    // Comparison operator for sorting (lower fitness is better)
    bool operator<(const Individual& other) const {
        return fitness < other.fitness;
    }
};

namespace Statistics {
// Calcula media, desviación estándar, mínimo y máximo de fitness
struct FitnessStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
};

inline FitnessStats compute_fitness_stats(const std::vector<Individual>& pop) {
    FitnessStats stats;
    if (pop.empty()) return stats;
    std::vector<double> fitnesses;
    for (const auto& ind : pop) if (ind.fitness_valid) fitnesses.push_back(ind.fitness);
    if (fitnesses.empty()) return stats;
    stats.min = *std::min_element(fitnesses.begin(), fitnesses.end());
    stats.max = *std::max_element(fitnesses.begin(), fitnesses.end());
    stats.mean = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0) / fitnesses.size();
    double sq_sum = std::inner_product(fitnesses.begin(), fitnesses.end(), fitnesses.begin(), 0.0);
    stats.stddev = std::sqrt(sq_sum / fitnesses.size() - stats.mean * stats.mean);
    return stats;
}

// Calcula diversidad estructural (proporción de árboles únicos)
inline double compute_structural_diversity(const std::vector<Individual>& pop) {
    std::unordered_set<size_t> hashes;
    for (const auto& ind : pop) if (ind.tree) hashes.insert(tree_structural_hash(ind.tree));
    return hashes.empty() ? 0.0 : double(hashes.size()) / double(pop.size());
}
}

// Generates a random tree up to a max depth
NodePtr generate_random_tree(int max_depth, int current_depth = 0);

// Creates the initial population
std::vector<Individual> create_initial_population(int population_size);

// Selects an individual using tournament selection
const Individual& tournament_selection(const std::vector<Individual>& population, int tournament_size);

// Mutates a tree (returns a new, potentially modified tree)
// Uses enhanced mutation types
NodePtr mutate_tree(const NodePtr& tree, double mutation_rate, int max_depth);

// Performs crossover between two parent trees (modifies them in place)
// Uses enhanced crossover
void crossover_trees(NodePtr& tree1, NodePtr& tree2);


#endif // GENETICOPERATORS_H