#ifndef FITNESS_CUDA_H
#define FITNESS_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Launches the CUDA kernel to evaluate fitness for the entire population
// - d_constants: device pointer to flattened constants (pop_size * max_constants_per_tree)
// - d_structures: device pointer to flattened structures (pop_size * max_flat_size)
// - d_x_values: device pointer to input x values
// - d_y_values: device pointer to target y values (targets)
// - d_fitness_results: device pointer for output fitness values (pop_size)
// - pop_size: number of individuals in the population
// - data_size: number of data points (size of x_values/y_values)
// - max_flat_size: maximum size of the structure array for one individual
// - max_constants_per_tree: maximum number of constants for one individual
void launch_fitness_evaluation(const double* d_constants,
                             const int* d_structures,
                             const double* d_x_values,
                             const double* d_y_values,
                             double* d_fitness_results,
                             int pop_size,
                             int data_size,
                             int max_flat_size,
                             int max_constants_per_tree); // Updated parameter name

#ifdef __cplusplus
}
#endif

#endif // FITNESS_CUDA_H
