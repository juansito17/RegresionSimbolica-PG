// ============================================================
// Archivo: src/GeneticAlgorithm.h
// ============================================================
#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include "ExpressionTree.h"
#include "GeneticOperators.h"
#include "AdvancedFeatures.h"
#include "Globals.h" // <--- AÑADIDO: Incluir Globals.h para INF y NUM_ISLANDS
#include "Fitness.h" // Include Fitness.h to use plot_predictions
#include <vector>
#include <string>
#include <memory>

class GeneticAlgorithm {
public:
    GeneticAlgorithm(const std::vector<double>& targets_ref,
                       const std::vector<double>& x_values_ref,
                       int total_pop,
                       int gens,
                       // Usa NUM_ISLANDS de Globals.h como valor por defecto
                       int n_islands = NUM_ISLANDS);

    NodePtr run();

private:
    struct Island {
        int id;
        std::vector<Individual> population;
        EvolutionParameters params;
        NodePtr best_individual_tree = nullptr;
        double best_individual_fitness = INF;
        int stagnation_counter = 0;
        std::vector<double> fitness_history;
        double best_fitness = INF;
        ParetoOptimizer pareto_optimizer;
        PatternMemory pattern_memory;

        explicit Island(int i) : id(i) {
            params = EvolutionParameters::create_default();
        }
    };

    const std::vector<double>& targets;
    const std::vector<double>& x_values;
    int total_population_size_;
    int generations_;
    int num_islands_;
    int migration_interval_;
    std::vector<std::unique_ptr<Island>> islands;
    NodePtr overall_best_individual;
    double overall_best_fitness;

    void evaluate_population(Island& island);
    void evolve_island(Island& island, int current_generation, NodePtr& overall_best_tree, double& overall_best_fitness);
    void migrate();
    void update_overall_best(const Island& island);
    void initialize_islands(); // <--- ADDED
    void update_best_individual(const Island& island, NodePtr& overall_best_tree, double& overall_best_fitness); // <--- ADDED

    // Helper function to update the overall best individual
    void update_best_individual(const Island& island);
};


#endif // GENETICALGORITHM_H