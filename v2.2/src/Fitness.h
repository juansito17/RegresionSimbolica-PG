#ifndef FITNESS_H
#define FITNESS_H

#include "ExpressionTree.h"
#include <vector>

// Calculates raw fitness based on target matching
// Lower is better. Returns INF if evaluation results in NaN/Inf.
double calculate_raw_fitness(const NodePtr& tree,
                             const std::vector<double>& targets,
                             const std::vector<double>& x_values);

// Calculates final fitness including complexity penalty
double evaluate_fitness(const NodePtr& tree,
                        const std::vector<double>& targets,
                        const std::vector<double>& x_values);

#endif // FITNESS_H