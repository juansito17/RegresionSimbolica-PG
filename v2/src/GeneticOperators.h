// ============================================================
// Archivo: src/GeneticOperators.h
// ============================================================
#ifndef GENETICOPERATORS_H
#define GENETICOPERATORS_H

#include "ExpressionTree.h"
#include "Globals.h" // <--- AÑADIDO: Incluir Globals.h para INF
#include <vector>

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