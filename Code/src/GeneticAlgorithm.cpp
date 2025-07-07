#include "GeneticAlgorithm.h"
#include "Fitness.h"
#include "GeneticOperators.h"
#include "AdvancedFeatures.h"
#include "Globals.h"
#include "ExpressionTree.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>
#include <iomanip> // Para std::fixed/scientific
#if USE_GPU_ACCELERATION
#include <cuda_runtime.h>
#endif

// Constructor
GeneticAlgorithm::GeneticAlgorithm(const std::vector<double>& targets_ref,
                                   const std::vector<double>& x_values_ref,
                                   int total_pop, int gens, int n_islands)
    : targets(targets_ref), x_values(x_values_ref),
      total_population_size(total_pop), generations(gens), num_islands(n_islands) {

    if (num_islands <= 0) num_islands = 1;
    pop_per_island = total_population_size / num_islands;

    for (int i = 0; i < num_islands; ++i) {
        islands.push_back(std::make_unique<Island>(i, pop_per_island));
    }

#if USE_GPU_ACCELERATION
    size_t data_size = targets.size() * sizeof(double);
    cudaMalloc((void**)&d_targets, data_size);
    cudaMalloc((void**)&d_x_values, data_size);
    cudaMemcpy(d_targets, targets.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_values, x_values.data(), data_size, cudaMemcpyHostToDevice);
#endif
}

// Destructor
GeneticAlgorithm::~GeneticAlgorithm() {
#if USE_GPU_ACCELERATION
    cudaFree(d_targets);
    cudaFree(d_x_values);
#endif
}

// Ejecuta el algoritmo genético
NodePtr GeneticAlgorithm::run() {
    // Evaluación inicial de la población
    std::cout << "Evaluating initial population (simplifying all)..." << std::endl;
    for (auto& island : islands) {
        for (auto& ind : island->population) {
            simplify_tree(ind.tree); // Simplificar antes de evaluar
        }
        evaluate_population(*island);
        update_overall_best(*island);
    }
    std::cout << "Initial best fitness: " << std::scientific << overall_best_fitness << std::endl;
    if (overall_best_fitness >= INF) {
        std::cout << "No valid initial solution found (all fitness INF?)." << std::endl;
    } else {
        std::cout << "Initial best formula size: " << tree_size(overall_best_tree) << std::endl;
        std::cout << "Initial best formula: " << tree_to_string(overall_best_tree) << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Starting Genetic Algorithm..." << std::endl;

    // Bucle principal de generaciones
    for (int gen = 0; gen < generations; ++gen) {
        for (auto& island : islands) {
            evolve_island(*island, gen);
        }

        if (gen % MIGRATION_INTERVAL == 0 && gen > 0) {
            migrate();
        }

        // Reporte de progreso
        if ((gen + 1) % PROGRESS_REPORT_INTERVAL == 0 || gen == 0) {
            std::cout << "\n--- Generation " << gen + 1 << "/" << generations
                      << " | Current Best Fitness: " << std::scientific << overall_best_fitness
                      << " | Formula Size: " << tree_size(overall_best_tree) << " ---" << std::endl;
        }

        // Condición de parada si se encuentra una solución suficientemente buena
        if (overall_best_fitness < EXACT_SOLUTION_THRESHOLD) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Solution found meeting criteria at Generation " << gen + 1 << "!" << std::endl;
            std::cout << "Final Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
            std::cout << "Final Formula Size: " << tree_size(overall_best_tree) << std::endl;
            std::cout << "Final Formula: " << tree_to_string(overall_best_tree) << std::endl;
            std::cout << "========================================" << std::endl;
            return overall_best_tree;
        }
    }
    return overall_best_tree;
}

// Evalúa la población de una isla
void GeneticAlgorithm::evaluate_population(Island& island) {
    for (auto& individual : island.population) {
#if USE_GPU_ACCELERATION
        individual.fitness = evaluate_fitness(individual.tree, targets, x_values, d_targets, d_x_values);
#else
        individual.fitness = evaluate_fitness(individual.tree, targets, x_values);
#endif
    }
}

// Evoluciona una isla por una generación
void GeneticAlgorithm::evolve_island(Island& island, int current_generation) {
    std::vector<Individual> new_population;

    // Elitismo
    int elite_size = static_cast<int>(island.population.size() * BASE_ELITE_PERCENTAGE);
    std::sort(island.population.begin(), island.population.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness < b.fitness;
              });
    for (int i = 0; i < elite_size; ++i) {
        new_population.push_back(island.population[i]);
    }

    // Creación de nueva población
    while (new_population.size() < island.population.size()) {
        Individual parent1 = tournament_selection(island.population, 5);
        Individual parent2 = tournament_selection(island.population, 5);
        Individual offspring = crossover(parent1, parent2);
        mutate(offspring, BASE_MUTATION_RATE);
        simplify_tree(offspring.tree);
        new_population.push_back(offspring);
    }

    island.population = new_population;
    evaluate_population(island);
    update_overall_best(island);
}

// Realiza la migración entre islas
void GeneticAlgorithm::migrate() {
    for (size_t i = 0; i < islands.size(); ++i) {
        Island& source_island = *islands[i];
        Island& dest_island = *islands[(i + 1) % islands.size()];

        std::sort(source_island.population.begin(), source_island.population.end(),
                  [](const Individual& a, const Individual& b) {
                      return a.fitness < b.fitness;
                  });

        for (int j = 0; j < MIGRATION_SIZE && j < source_island.population.size(); ++j) {
            if (!dest_island.population.empty()) {
                auto worst_it = std::max_element(dest_island.population.begin(), dest_island.population.end(),
                                                 [](const Individual& a, const Individual& b) {
                                                     return a.fitness < b.fitness;
                                                 });
                *worst_it = source_island.population[j];
            }
        }
    }
}

// Actualiza el mejor global
void GeneticAlgorithm::update_overall_best(const Island& island) {
    for (const auto& individual : island.population) {
        if (individual.fitness < overall_best_fitness) {
            overall_best_fitness = individual.fitness;
            overall_best_tree = clone_tree(individual.tree);
        }
    }
}
