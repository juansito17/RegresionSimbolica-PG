#include "GeneticAlgorithm.h"
#include "Globals.h"
#include "Fitness.h"
#include <iostream>
#include <algorithm> // For sort, min_element, sample
#include <vector>
#include <cmath>
#include <omp.h> // For OpenMP parallelization
#include <iomanip> // For std::setprecision


GeneticAlgorithm::GeneticAlgorithm(const std::vector<double>& targets_ref,
                                     const std::vector<double>& x_values_ref,
                                     int total_pop,
                                     int gens,
                                     int n_islands)
    : targets(targets_ref),
      x_values(x_values_ref),
      total_population_size(total_pop),
      generations(gens),
      num_islands(n_islands)
{
    if (num_islands <= 0) num_islands = 1;
    pop_per_island = total_population_size / num_islands;
    if (pop_per_island < 10) { // Ensure reasonable population per island
        pop_per_island = 10;
        num_islands = total_population_size / pop_per_island;
        if (num_islands == 0) num_islands = 1;
        std::cerr << "Warning: Adjusted number of islands to " << num_islands
                  << " for minimum population size." << std::endl;
    }

    islands.reserve(num_islands);
    for (int i = 0; i < num_islands; ++i) {
        islands.push_back(std::make_unique<Island>(i, pop_per_island));
    }

     // --- Initial Population Enhancement ---
     // 1. Detect patterns in target data
     auto [pattern_type, pattern_value] = detect_target_pattern(targets);
     if (pattern_type != "none") {
         std::cout << "Detected target pattern: " << pattern_type
                   << " (Value: " << pattern_value << ")" << std::endl;
         NodePtr pattern_tree = generate_pattern_based_tree(pattern_type, pattern_value);
         if (pattern_tree) {
             std::cout << "Injecting pattern-based seed: " << tree_to_string(pattern_tree) << std::endl;
             // Add this seed tree to each island's initial population
             for (auto& island_ptr : islands) {
                 // Replace a random individual or just add? Let's replace the last one.
                 if (!island_ptr->population.empty()) {
                      island_ptr->population.back() = Individual(pattern_tree);
                 } else {
                      island_ptr->population.emplace_back(pattern_tree);
                 }
             }
         }
     }

     // 2. Initial fitness evaluation and best tracking
     std::cout << "Evaluating initial population..." << std::endl;
     for (auto& island_ptr : islands) {
         evaluate_population(*island_ptr);
         update_overall_best(*island_ptr);
     }
      std::cout << "Initial best fitness: " << std::fixed << std::setprecision(6) << overall_best_fitness << std::endl;
      if (overall_best_tree) {
          std::cout << "Initial best formula: " << tree_to_string(overall_best_tree) << std::endl;
      }
       std::cout << "----------------------------------------" << std::endl;

}

// Evaluate fitness for all individuals in an island's population
// Uses GPU for parallel evaluation.
void GeneticAlgorithm::evaluate_population(Island& island) {
    int pop_size = island.population.size();
    std::vector<NodePtr> trees;
    std::vector<double> fitness_results;
    std::vector<int> indices;
    
    // Recolectar árboles que necesitan evaluación
    for (int i = 0; i < pop_size; ++i) {
        Individual& ind = island.population[i];
        if (!ind.fitness_valid && ind.tree) {
            ind.tree = DomainConstraints::fix_or_simplify(ind.tree);
            if (ind.tree) {
                trees.push_back(ind.tree);
                indices.push_back(i);
            } else {
                ind.fitness = INF;
                ind.fitness_valid = true;
            }
        }
    }

    // Si hay árboles para evaluar, usar GPU
    if (!trees.empty()) {
        fitness_results.resize(trees.size());
        evaluate_population_fitness(trees, targets, x_values, fitness_results);
        
        // Actualizar fitness de los individuos
        for (size_t i = 0; i < trees.size(); ++i) {
            Individual& ind = island.population[indices[i]];
            ind.fitness = fitness_results[i];
            ind.fitness_valid = true;
        }
    }
}

void GeneticAlgorithm::update_overall_best(const Island& island) {
     for (const auto& ind : island.population) {
         if (ind.fitness_valid && ind.fitness < overall_best_fitness) {
             overall_best_fitness = ind.fitness;
             overall_best_tree = clone_tree(ind.tree); // Clone to store the best safely

             // --- Output Improvement ---
             std::cout << "\n========================================" << std::endl;
             std::cout << "New Best Found (Island " << island.id << ")" << std::endl;
             std::cout << "Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
             std::cout << "Size: " << tree_size(overall_best_tree) << std::endl;
             std::cout << "Formula: " << tree_to_string(overall_best_tree) << std::endl;
              std::cout << "Predictions vs Targets:" << std::endl;
              std::cout << std::fixed << std::setprecision(4);
              for (size_t j = 0; j < x_values.size(); ++j) {
                    double val = evaluate_tree(overall_best_tree, x_values[j]);
                    double diff = std::fabs(val - targets[j]);
                    std::cout << "  x=" << std::setw(2) << static_cast<int>(x_values[j])
                              << ": Pred=" << std::setw(10) << val
                              << ", Target=" << std::setw(10) << targets[j]
                              << ", Diff=" << std::setw(10) << diff << std::endl;
               }
               std::cout << "========================================" << std::endl;
             // --- End Output ---
         }
     }
 }

void GeneticAlgorithm::evolve_island(Island& island, int current_generation) {
    int current_pop_size = island.population.size();
    if (current_pop_size == 0) return; // Skip empty island

    // --- 1. Evaluation & Statistics ---
    // Fitness should already be evaluated by evaluate_population before this step
    // Sort population by fitness (best first)
    std::sort(island.population.begin(), island.population.end());

    double current_best_fitness = island.population[0].fitness;
    island.fitness_history.push_back(current_best_fitness);

    // Update stagnation counter
    if (current_best_fitness < island.best_fitness - 1e-9) { // Improved (with tolerance)
        island.best_fitness = current_best_fitness;
        island.stagnation_counter = 0;

        // Try local improvement on the very best individual of the island
         auto local_search_result = try_local_improvement(
             island.population[0].tree,
             island.population[0].fitness,
             targets, x_values, 15); // More attempts for local search

         if (local_search_result.second < island.population[0].fitness) {
            // std::cout << "Local search improved island " << island.id << " best." << std::endl;
             island.population[0].tree = local_search_result.first;
             island.population[0].fitness = local_search_result.second;
             // fitness_valid remains true
             island.best_fitness = local_search_result.second; // Update island best fitness too
         }

    } else {
        island.stagnation_counter++;
    }

    // Update overall best across all islands
    update_overall_best(island);


    // Update Pareto Front and Pattern Memory for this island
    island.pareto_optimizer.update(island.population, targets, x_values);
    for(const auto& ind : island.population) {
        // Record success for good solutions (e.g., fitness < 10 or significantly better than average)
        if(ind.fitness_valid && ind.fitness < 10.0) {
            island.pattern_memory.record_success(ind.tree, ind.fitness);
        }
    }


    // --- 2. Selection & Reproduction ---
    std::vector<Individual> next_generation;
    next_generation.reserve(current_pop_size);

    // Elitism: Copy the best individuals directly
    int elite_count = std::max(1, static_cast<int>(current_pop_size * island.params.elite_percentage));
    for (int i = 0; i < elite_count && i < current_pop_size; ++i) {
        next_generation.push_back(island.population[i]); // Copy elite individuals
         // Keep fitness valid flag true for elite individuals
    }

    // Add new random individuals if stagnated
    int random_injection_count = 0;
    if (island.stagnation_counter > STAGNATION_LIMIT / 2) {
         random_injection_count = static_cast<int>(current_pop_size * 0.1); // Inject 10% random
         for(int i = 0; i < random_injection_count; ++i) {
              next_generation.emplace_back(generate_random_tree(MAX_TREE_DEPTH_INITIAL));
              // Fitness will be evaluated later
         }
    }
    // Add pattern-based individuals occasionally
     int pattern_injection_count = 0;
     if (current_generation % 10 == 0) { // Every 10 generations
         pattern_injection_count = static_cast<int>(current_pop_size * 0.05); // Inject 5% from patterns
         for (int i = 0; i < pattern_injection_count; ++i) {
             NodePtr pattern_tree = island.pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_INITIAL);
             if (pattern_tree) {
                 next_generation.emplace_back(pattern_tree);
             } else {
                  // Fallback if no pattern suggested
                  next_generation.emplace_back(generate_random_tree(MAX_TREE_DEPTH_INITIAL));
             }
         }
     }


    // Fill the rest of the population using selection, crossover, mutation
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    int remaining_slots = current_pop_size - next_generation.size();

    for (int i = 0; i < remaining_slots; ++i) {
        // Select parents using tournament
        const Individual& parent1 = tournament_selection(island.population, island.params.tournament_size);
        Individual child; // Create a new individual

        if (prob_dist(rng) < island.params.crossover_rate) {
            const Individual& parent2 = tournament_selection(island.population, island.params.tournament_size);
            // Clone parents for crossover to avoid modifying originals in the population
            NodePtr p1_clone = clone_tree(parent1.tree);
            NodePtr p2_clone = clone_tree(parent2.tree);
            crossover_trees(p1_clone, p2_clone);
            // Choose one of the results as the child (e.g., p1_clone)
            child.tree = p1_clone;
            // Invalidate fitness as tree changed
            child.fitness_valid = false;
        } else {
            // No crossover, just clone parent1
            child.tree = clone_tree(parent1.tree);
             // Invalidate fitness as tree changed (will be mutated next)
            child.fitness_valid = false;
        }

        // Mutate the child
        child.tree = mutate_tree(child.tree, island.params.mutation_rate, MAX_TREE_DEPTH_MUTATION);
        child.fitness_valid = false; // Ensure fitness is marked invalid after mutation


        next_generation.push_back(std::move(child)); // Move child into next generation
    }

    // --- 3. Replace Population ---
    island.population = std::move(next_generation); // Replace old population

    // --- 4. Meta-Evolution ---
    // Adapt island's parameters periodically or based on stagnation
    if (current_generation > 0 && current_generation % 50 == 0) { // Every 50 generations
         island.params.mutate();
         // std::cout << "Island " << island.id << " params mutated." << std::endl; // Debug
    }

     // Ensure all individuals have fitness marked as invalid for the next round's evaluation
     // (Except maybe elites if we are sure they weren't modified - but safer to re-eval)
     // Evaluation happens at the start of the next generation or before selection.
     // Let's ensure evaluate_population() handles the fitness_valid flag correctly.
}

void GeneticAlgorithm::migrate() {
    if (num_islands <= 1) return; // No migration needed for single island

    int num_migrants = std::min(MIGRATION_SIZE, pop_per_island / 5); // Don't migrate too many
    if (num_migrants == 0) return;

    std::vector<std::vector<Individual>> outgoing_migrants(num_islands);

    // Select migrants from each island (best individuals)
    for (int i = 0; i < num_islands; ++i) {
        Island& source_island = *islands[i];
        // Sort population to easily get the best
        std::sort(source_island.population.begin(), source_island.population.end());
        
        // Fix signed/unsigned comparison
        size_t max_migrants = std::min(static_cast<size_t>(num_migrants), source_island.population.size());
        for (size_t j = 0; j < max_migrants; ++j) {
            // Clone the migrant's tree for sending
            outgoing_migrants[i].emplace_back(clone_tree(source_island.population[j].tree));
            outgoing_migrants[i].back().fitness = source_island.population[j].fitness;
            outgoing_migrants[i].back().fitness_valid = source_island.population[j].fitness_valid;
        }
    }

    // Receive migrants in each island (replace worst individuals)
     // Circular migration: island i sends to (i+1)%N
    for (int dest_idx = 0; dest_idx < num_islands; ++dest_idx) {
        int source_idx = (dest_idx + num_islands - 1) % num_islands; // Get migrants from previous island
        Island& dest_island = *islands[dest_idx];
        const auto& migrants = outgoing_migrants[source_idx];

        if (migrants.empty()) continue;

        // Sort destination population by fitness (worst first) for replacement
        std::sort(dest_island.population.begin(), dest_island.population.end(),
                  [](const Individual& a, const Individual& b) {
                      // Put invalid fitness individuals last (worst)
                      if (!a.fitness_valid) return false;
                      if (!b.fitness_valid) return true;
                      return a.fitness > b.fitness; // Sort descending by fitness (worst first)
                  });

        // Fix signed/unsigned comparison
        size_t max_replacements = std::min(migrants.size(), dest_island.population.size());
        for (size_t i = 0; i < max_replacements; ++i) {
            dest_island.population[i] = migrants[i];
        }
    }
}


NodePtr GeneticAlgorithm::run() {
    std::cout << "Starting Genetic Algorithm..." << std::endl;
    std::cout << "Islands: " << num_islands << ", Pop/Island: " << pop_per_island
              << ", Generations: " << generations << std::endl;
    std::cout << "----------------------------------------" << std::endl;


    for (int gen = 0; gen < generations; ++gen) {

        // Evolve each island
        for (auto& island_ptr : islands) {
             // 1. Evaluate fitness for the current population
             evaluate_population(*island_ptr); // Ensures fitness is up-to-date before evolution steps

             // 2. Perform evolutionary steps (selection, crossover, mutation, etc.)
             evolve_island(*island_ptr, gen); // Creates the *next* generation

        }

        // Periodic Migration
        if ((gen + 1) % MIGRATION_INTERVAL == 0 && num_islands > 1) {
             std::cout << "\n--- Generation " << gen + 1 << ": Performing Migration ---" << std::endl;
             migrate();
             // Fitness of migrants needs re-evaluation in their new islands?
             // evaluate_population() at the start of the next gen handles this.
        }

        // Check for termination condition (e.g., perfect fitness)
        if (overall_best_fitness < 1e-6) { // Threshold for "exact" solution
            std::cout << "\n========================================" << std::endl;
            std::cout << "Solution found meeting criteria at Generation " << gen + 1 << "!" << std::endl;
            std::cout << "========================================" << std::endl;
            break;
        }

        // Progress Report (less frequent)
        if ((gen + 1) % 100 == 0 || gen == generations - 1) {
             std::cout << "\n--- Generation " << gen + 1 << " ---" << std::endl;
             std::cout << "Overall Best Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
              if(overall_best_tree) {
                   std::cout << "Best Formula Size: " << tree_size(overall_best_tree) << std::endl;
                  // std::cout << "Best Formula: " << tree_to_string(overall_best_tree) << std::endl; // Can be long
              }
              // Report average island fitness or diversity?
        }

    } // End of generations loop


    std::cout << "\n========================================" << std::endl;
    std::cout << "Evolution Finished!" << std::endl;
    std::cout << "Final Best Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
     if (overall_best_tree) {
         std::cout << "Final Best Formula Size: " << tree_size(overall_best_tree) << std::endl;
         std::cout << "Final Best Formula: " << tree_to_string(overall_best_tree) << std::endl;
          std::cout << "--- Verification ---" << std::endl;
          std::cout << std::fixed << std::setprecision(4);
          for (size_t j = 0; j < x_values.size(); ++j) {
                double val = evaluate_tree(overall_best_tree, x_values[j]);
                double diff = std::fabs(val - targets[j]);
                 std::cout << "  x=" << std::setw(2) << static_cast<int>(x_values[j])
                          << ": Pred=" << std::setw(10) << val
                          << ", Target=" << std::setw(10) << targets[j]
                          << ", Diff=" << std::setw(10) << diff << std::endl;
           }
     } else {
         std::cout << "No solution found." << std::endl;
     }
      std::cout << "========================================" << std::endl;


    return overall_best_tree; // Return the best tree found
}