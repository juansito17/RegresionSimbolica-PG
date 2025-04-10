#ifndef FITNESS_H
#define FITNESS_H

#include "ExpressionTree.h"
#include "GeneticOperators.h"
#include <vector>
#include <fstream>
#include <algorithm>

// Evaluates fitness for the entire population using CUDA
// Updates fitness and fitness_valid flags in the population vector directly.
void evaluate_population_fitness_cuda(std::vector<Individual>& population,
                                      const std::vector<double>& targets,
                                      const std::vector<double>& x_values);

void plot_predictions(const NodePtr& tree, 
                     const std::vector<double>& targets,
                     const std::vector<double>& x_values);

#endif // FITNESS_H