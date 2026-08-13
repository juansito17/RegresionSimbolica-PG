#ifndef FITNESS_H
#define FITNESS_H

#include "ExpressionTree.h"
#include <vector>

// Calculates raw fitness based on target matching
// Lower is better. Returns INF if evaluation results in NaN/Inf.
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
double calculate_raw_fitness(const NodePtr& tree,
                             const std::vector<double>& targets,
                             const std::vector<std::vector<double>>& x_values,
                             double* d_targets, double* d_x_values);
#else
double calculate_raw_fitness(const NodePtr& tree,
                             const std::vector<double>& targets,
                             const std::vector<std::vector<double>>& x_values);
#endif

// Calculates final fitness including complexity penalty
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
double evaluate_fitness(const NodePtr& tree,
                        const std::vector<double>& targets,
                        const std::vector<std::vector<double>>& x_values,
                        double* d_targets, double* d_x_values);
#else
double evaluate_fitness(const NodePtr& tree,
                        const std::vector<double>& targets,
                        const std::vector<std::vector<double>>& x_values);
#endif

#endif // FITNESS_H
