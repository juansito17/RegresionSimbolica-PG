#include "Fitness.h"
#include "Globals.h" // For INF, COMPLEXITY_PENALTY_FACTOR, CUDA constants
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept> // For runtime_error
#include <iostream> // Include for std::cout
#include <iomanip> // For std::fixed, std::setprecision
#include <fstream> // For std::ofstream
#include <algorithm> // For std::min_element, std::max_element

// Include CUDA specific headers only when needed
#ifdef USE_CUDA // Add a preprocessor flag USE_CUDA to your build system
#include <cuda_runtime.h>
#include "../cuda/FitnessCuda.h" // Path to the C wrapper header
#include "../include/cuda_defs.h" // Include for LARGE_FINITE_PENALTY
#endif // USE_CUDA

// --- CUDA Population Fitness Evaluation ---
#ifdef USE_CUDA
void evaluate_population_fitness_cuda(std::vector<Individual>& population,
                                      const std::vector<double>& targets,
                                      const std::vector<double>& x_values)
{
    // Add this line for verification
    // std::cout << "--- Evaluating population fitness using CUDA ---" << std::endl; // Keep this if helpful

    if (population.empty() || targets.empty() || x_values.empty()) {
        // Nothing to evaluate
        return;
    }
    if (targets.size() != x_values.size()) {
         throw std::runtime_error("CUDA Fitness: Target and X_Values size mismatch.");
    }

    int pop_size = population.size();
    int data_size = targets.size();

    // Add diagnostic counters
    int flattening_fails = 0;
    int kernel_fails = 0;
    int penalty_fails = 0;

    // --- 1. Prepare Host Data ---
    std::vector<int> h_structures(pop_size * MAX_FLAT_SIZE);
    std::vector<double> h_constants(pop_size * MAX_CONSTANTS_PER_TREE);
    std::vector<double> h_fitness_results(pop_size);
    std::vector<int> h_tree_sizes(pop_size); // Store tree sizes for penalty calculation

    bool any_flatten_failed = false;
    int flatten_fail_count = 0; // Debug: Count flattening failures
    for (int i = 0; i < pop_size; ++i) {
        std::vector<int> current_structure;
        std::vector<double> current_constants; // Will be resized by flatten_tree

        // Flatten the tree
        bool success = flatten_tree(population[i].tree, current_structure, current_constants,
                                    MAX_FLAT_SIZE, MAX_CONSTANTS_PER_TREE);

        if (success && !current_structure.empty()) {
            // Copy flattened data into the large host vectors at the correct offset
            size_t struct_offset = i * MAX_FLAT_SIZE;
            std::copy(current_structure.begin(), current_structure.end(), h_structures.begin() + struct_offset);
            // Pad the rest of the structure for this individual if necessary (using a code CUDA ignores or handles)
            // For now, assume kernel handles non-full structures based on op-codes.

            size_t const_offset = i * MAX_CONSTANTS_PER_TREE;
            std::copy(current_constants.begin(), current_constants.end(), h_constants.begin() + const_offset);

            h_tree_sizes[i] = tree_size(population[i].tree); // Store size for penalty
            h_fitness_results[i] = 0.0;
            // Debug: Log successful flattening (optional, can be verbose)
            // if (i < 5) { // Log only first few
            //     std::cout << "  Debug: Flatten success for individual " << i << ", Tree Size: " << h_tree_sizes[i] << std::endl;
            // }
        } else {
            flattening_fails++;
            std::cerr << "\nTree Flattening Failed:"
                      << "\n  Size: " << tree_size(population[i].tree)
                      << "\n  Structure: " << tree_to_string(population[i].tree)
                      << "\n  Depth: " << tree_depth(population[i].tree) << std::endl;

            // Handle flattening failure (tree too large, invalid structure, etc.)
            // Mark this individual for INF fitness using the penalty constant
            h_fitness_results[i] = LARGE_FINITE_PENALTY; // Pre-assign penalty on host
            h_tree_sizes[i] = MAX_FLAT_SIZE + 1; // Mark with large size for penalty if needed
            population[i].fitness = INF; // Set final fitness to INF immediately
            population[i].fitness_valid = true; // Mark as evaluated (to INF)
            any_flatten_failed = true;
            flatten_fail_count++; // Increment failure count
            // Debug: Log flattening failure
            // std::cerr << "  Warning: Flatten failed for individual " << i << ". Tree: " << tree_to_string(population[i].tree) << std::endl;
        }
    }
    // Debug: Log total flattening failures for this population
    if (any_flatten_failed) {
        std::cerr << "  Warning: Flattening failed for " << flatten_fail_count << " out of " << pop_size << " individuals." << std::endl;
    }

    // --- Moved declaration here ---
    int inf_count = 0; // Debug: Count INF results

    // --- 2. Allocate and Copy Data to Device ---
    double* d_constants = nullptr;
    int* d_structures = nullptr;
    double* d_x_values = nullptr;
    double* d_y_values = nullptr;
    double* d_fitness_results = nullptr;
    cudaError_t err;

    // Allocate
    err = cudaMalloc(&d_constants, h_constants.size() * sizeof(double));
    if (err != cudaSuccess) goto CudaError;
    err = cudaMalloc(&d_structures, h_structures.size() * sizeof(int));
    if (err != cudaSuccess) goto CudaError;
    err = cudaMalloc(&d_x_values, x_values.size() * sizeof(double));
    if (err != cudaSuccess) goto CudaError;
    err = cudaMalloc(&d_y_values, targets.size() * sizeof(double));
    if (err != cudaSuccess) goto CudaError;
    err = cudaMalloc(&d_fitness_results, h_fitness_results.size() * sizeof(double));
    if (err != cudaSuccess) goto CudaError;

    // Copy Host -> Device
    err = cudaMemcpy(d_constants, h_constants.data(), h_constants.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto CudaError;
    err = cudaMemcpy(d_structures, h_structures.data(), h_structures.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto CudaError;
    err = cudaMemcpy(d_x_values, x_values.data(), x_values.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto CudaError;
    err = cudaMemcpy(d_y_values, targets.data(), targets.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto CudaError;
    // Copy initial fitness results (contains LARGE_FINITE_PENALTY for failed flattening)
    err = cudaMemcpy(d_fitness_results, h_fitness_results.data(), h_fitness_results.size() * sizeof(double), cudaMemcpyHostToDevice);
     if (err != cudaSuccess) goto CudaError;

    // --- 3. Launch Kernel ---
    launch_fitness_evaluation(d_constants, d_structures, d_x_values, d_y_values,
                              d_fitness_results, pop_size, data_size,
                              MAX_FLAT_SIZE, MAX_CONSTANTS_PER_TREE);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle kernel launch error specifically
        cudaFree(d_constants); cudaFree(d_structures); cudaFree(d_x_values);
        cudaFree(d_y_values); cudaFree(d_fitness_results);
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // --- 4. Copy Results Device -> Host ---
    err = cudaMemcpy(h_fitness_results.data(), d_fitness_results, h_fitness_results.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto CudaError; // Error during copy back

    // --- 5. Clean Up Device Memory ---
    cudaFree(d_constants);
    cudaFree(d_structures);
    cudaFree(d_x_values);
    cudaFree(d_y_values);
    cudaFree(d_fitness_results);

    // --- 6. Update Population Fitness on Host ---
    for (int i = 0; i < pop_size; ++i) {
        if (population[i].fitness_valid && population[i].fitness >= INF) {
            inf_count++;
            continue;
        }

        double raw_fitness = h_fitness_results[i];

        // Usar valores más permisivos antes de considerar un fitness como inválido
        if (raw_fitness >= LARGE_FINITE_PENALTY * 0.9) {
            kernel_fails++;
            population[i].fitness = INF;
            inf_count++;
            continue;
        }

        // Aplicar penalización por complejidad como en v2
        double penalty = static_cast<double>(h_tree_sizes[i]) * COMPLEXITY_PENALTY_FACTOR;
        double final_fitness = raw_fitness * (1.0 + penalty);

        // Check for overflow/NaN after penalty
        if (std::isinf(final_fitness) || std::isnan(final_fitness) || final_fitness >= LARGE_FINITE_PENALTY) {
            penalty_fails++;
            population[i].fitness = INF;
            inf_count++;
        } else {
            population[i].fitness = final_fitness;
        }
        population[i].fitness_valid = true; // Mark as evaluated
    }

    // Debug: Log total INF count for this population
    // std::cout << "  Debug: Total INF fitness values in population: " << inf_count << "/" << pop_size << std::endl;

    // Print diagnostic summary
    std::cout << "\nFitness Evaluation Summary:"
              << "\n  Flattening Failures: " << flattening_fails
              << "\n  Kernel Evaluation Failures: " << kernel_fails
              << "\n  Penalty Application Failures: " << penalty_fails
              << "\n  Total Population Size: " << pop_size << std::endl;

    return; // Success

CudaError:
    // Cleanup allocated memory in case of error
    cudaFree(d_constants);
    cudaFree(d_structures);
    cudaFree(d_x_values);
    cudaFree(d_y_values);
    cudaFree(d_fitness_results);
    // Throw exception
    throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
}

void plot_predictions(const NodePtr& tree, 
                     const std::vector<double>& targets,
                     const std::vector<double>& x_values) {
    // Generar puntos para la curva
    std::vector<double> x_curve, y_curve;
    double x_min = *std::min_element(x_values.begin(), x_values.end());
    double x_max = *std::max_element(x_values.begin(), x_values.end());
    
    for (double x = x_min; x <= x_max; x += (x_max - x_min) / 100.0) {
        x_curve.push_back(x);
        y_curve.push_back(evaluate_tree(tree, x));
    }

    // Crear archivo temporal con los datos
    std::ofstream plot_data("plot_data.txt");
    for (size_t i = 0; i < x_curve.size(); ++i) {
        plot_data << x_curve[i] << " " << y_curve[i] << "\n";
    }
    plot_data.close();

    // Crear archivo temporal con los puntos objetivo
    std::ofstream target_data("target_data.txt");
    for (size_t i = 0; i < targets.size(); ++i) {
        target_data << x_values[i] << " " << targets[i] << "\n";
    }
    target_data.close();

    // Crear script de gnuplot
    std::ofstream plot_script("plot_script.gp");
    plot_script << "set terminal png\n";
    plot_script << "set output 'prediction_plot.png'\n";
    plot_script << "set grid\n";
    plot_script << "plot 'plot_data.txt' with lines title 'Predicción', ";
    plot_script << "'target_data.txt' with points pt 7 title 'Objetivos'\n";
    plot_script.close();

    // Ejecutar gnuplot
    system("gnuplot plot_script.gp");
}

#else
// This part should not be compiled if USE_CUDA is defined and required.
// If you need a CPU version for testing without CUDA, it should be implemented here.
// Since the request was GPU-only, we assume this won't be hit in the target build.
void evaluate_population_fitness_cuda(std::vector<Individual>& population,
                                      const std::vector<double>& targets,
                                      const std::vector<double>& x_values)
{
     throw std::runtime_error("CUDA is required but USE_CUDA is not defined during compilation.");
     // Or provide a minimal CPU implementation if absolutely necessary for non-CUDA builds
}
#endif // USE_CUDA