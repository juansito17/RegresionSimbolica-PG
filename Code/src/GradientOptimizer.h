#ifndef GRADIENT_OPTIMIZER_H
#define GRADIENT_OPTIMIZER_H

#include "ExpressionTree.h"
#include <vector>

// Optimizes the constants of a given tree using Gradient Descent (Adam optimizer).
// This modifies the tree in-place.
//
// @param tree: The expression tree to optimize.
// @param targets: The target y values.
// @param x_values: The input x values (multivariable).
// @param learning_rate: The step size for the optimizer (default 0.01).
// @param iterations: Number of optimization steps (default 50).
void optimize_constants_gradient(NodePtr& tree, 
                                 const std::vector<double>& targets, 
                                 const std::vector<std::vector<double>>& x_values,
                                 double learning_rate = 0.05,
                                 int iterations = 30);

#endif // GRADIENT_OPTIMIZER_H
