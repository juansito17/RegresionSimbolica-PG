#ifndef FITNESS_H
#define FITNESS_H

#include "ExpressionTree.h"
#include <vector>

// Forward declaration of GPU evaluation function
extern "C" {
    void evaluatePopulationGPU(
        const std::vector<NodePtr>& trees,
        const std::vector<double>& x_values,
        const std::vector<double>& targets,
        std::vector<double>& fitness_results
    );
}

// Evaluates multiple trees in parallel using GPU
void evaluate_population_fitness(
    const std::vector<NodePtr>& trees,
    const std::vector<double>& targets,
    const std::vector<double>& x_values,
    std::vector<double>& fitness_results);

#endif