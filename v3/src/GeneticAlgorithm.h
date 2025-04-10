// ============================================================
// Archivo: src/GeneticAlgorithm.h
// ============================================================
#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include "ExpressionTree.h"
#include "GeneticOperators.h"
#include "AdvancedFeatures.h"
#include "Globals.h" // <--- AÑADIDO: Incluir Globals.h para INF y NUM_ISLANDS
#include <vector>
#include <string>
#include <memory>

class GeneticAlgorithm {
    struct Island {
        std::vector<Individual> population;
        EvolutionParameters params;
        PatternMemory pattern_memory;
        ParetoOptimizer pareto_optimizer;
        int stagnation_counter = 0;
        double best_fitness = INF; // Usa INF de Globals.h
        std::vector<double> fitness_history;
        int id;

        explicit Island(int island_id, int pop_size) : id(island_id) {
             population = create_initial_population(pop_size);
             params = EvolutionParameters::create_default();
        }
    };

    std::vector<std::unique_ptr<Island>> islands;
    const std::vector<double>& targets;
    const std::vector<double>& x_values;
    int total_population_size;
    int generations;
    int num_islands;

    NodePtr overall_best_tree = nullptr;
    double overall_best_fitness = INF; // Usa INF de Globals.h

    int pop_per_island;

public:
    GeneticAlgorithm(const std::vector<double>& targets_ref,
                       const std::vector<double>& x_values_ref,
                       int total_pop,
                       int gens,
                       // Usa NUM_ISLANDS de Globals.h como valor por defecto
                       int n_islands = NUM_ISLANDS);

    NodePtr run();

private:
    void evaluate_population(Island& island);
    void evolve_island(Island& island, int current_generation);
    void migrate();
    void update_overall_best(const Island& island);
};


#endif // GENETICALGORITHM_H