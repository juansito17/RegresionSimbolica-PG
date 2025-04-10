#ifndef FITNESS_H
#define FITNESS_H

#include "ExpressionTree.h"
#include <vector>

// Forward declaration of GPU evaluation function
// Nota: La función ahora es declarada como función C pura
extern "C" {
    void evaluatePopulationGPU(
        const std::vector<NodePtr>& trees,
        const std::vector<double>& x_values,
        const std::vector<double>& targets,
        std::vector<double>& fitness_results
    );
}

// Calculates raw fitness based on target matching
// Lower is better. Returns INF if evaluation results in NaN/Inf.
double calculate_raw_fitness(const NodePtr& tree,
                             const std::vector<double>& targets,
                             const std::vector<double>& x_values);

// Calculates final fitness including complexity penalty
double evaluate_fitness(const NodePtr& tree,
                        const std::vector<double>& targets,
                        const std::vector<double>& x_values);

// Evaluates multiple trees in parallel using GPU
void evaluate_population_fitness(
    const std::vector<NodePtr>& trees,
    const std::vector<double>& targets,
    const std::vector<double>& x_values,
    std::vector<double>& fitness_results);

#endif // FITNESS_H