#include "GeneticAlgorithm.h"
#include "GeneticOperators.h"
#include "Fitness.h"
#include "ExpressionTree.h"
#include "AdvancedFeatures.h" // Include for advanced features
#include "Globals.h"          // Include Globals for constants like FITNESS_THRESHOLD
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <iomanip> // For std::setprecision, std::fixed, std::setw
#include <cmath>   // For std::isinf, std::isnan, std::fabs
#include <random>  // For shuffling islands
#include <thread>  // For potential parallel island evolution (optional)
#include <future>  // For std::async (optional)
#include <memory>  // For std::make_unique
#include "Fitness.h" // Make sure Fitness.h is included for plot_predictions

// Constructor
GeneticAlgorithm::GeneticAlgorithm(const std::vector<double>& targets,
                                   const std::vector<double>& x_values,
                                   int total_population_size,
                                   int generations,
                                   int num_islands)
    : targets(targets),
      x_values(x_values),
      total_population_size_(total_population_size),
      generations_(generations),
      num_islands_(std::max(1, num_islands)), // Ensure at least one island
      migration_interval_(MIGRATION_INTERVAL) // Initialize from global or parameter
{
    if (num_islands_ <= 0) {
        throw std::invalid_argument("Number of islands must be positive.");
    }
    if (total_population_size_ < num_islands_) {
         throw std::invalid_argument("Total population size must be at least the number of islands.");
    }
}

// --- Private Helper Functions ---

// Initializes the islands and their populations
void GeneticAlgorithm::initialize_islands() {
    islands.clear();
    int base_pop_size = total_population_size_ / num_islands_;
    int remainder = total_population_size_ % num_islands_;

    for (int i = 0; i < num_islands_; ++i) {
        int island_pop_size = base_pop_size + (i < remainder ? 1 : 0);
        auto island_ptr = std::make_unique<Island>(i);
        island_ptr->population = create_initial_population(island_pop_size);
        islands.push_back(std::move(island_ptr));
    }
    std::cout << "Initialized " << num_islands_ << " islands." << std::endl;
}

// Updates the overall best individual found so far and prints details if improved
void GeneticAlgorithm::update_best_individual(const Island& island) {
    for (const auto& ind : island.population) {
        if (ind.fitness_valid && ind.fitness < overall_best_fitness) {
            overall_best_fitness = ind.fitness;
            overall_best_individual = clone_tree(ind.tree); // Clone for safety

            // --- Output Improvement in v2 Format ---
            std::cout << "\n========================================" << std::endl;
            std::cout << "New Best Found (Island " << island.id << ")" << std::endl;
            std::cout << "Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
            int size = tree_size(overall_best_individual);
            std::cout << "Size: " << size << std::endl;
            std::cout << "Formula: " << tree_to_string(overall_best_individual) << std::endl;
            std::cout << "Predictions vs Targets:" << std::endl;
            std::cout << std::fixed << std::setprecision(4); // Set precision for predictions
            for (size_t i = 0; i < x_values.size(); ++i) {
                double pred = evaluate_tree(overall_best_individual, x_values[i]);
                double diff = std::fabs(pred - targets[i]);
                std::cout << "  x=" << std::setw(2) << x_values[i] // Adjust setw as needed
                          << ": Pred=" << std::setw(10) << pred
                          << ", Target=" << std::setw(10) << targets[i]
                          << ", Diff=" << std::setw(10) << diff << std::endl;
            }
            std::cout << "========================================" << std::endl;
            // --- End Output ---

            // Plot the new best solution
            plot_predictions(overall_best_individual, targets, x_values);
        }
    }
}

// Evaluates the fitness of the entire population of an island
void GeneticAlgorithm::evaluate_population(Island& island) {
    // Invalidate fitness for all individuals before evaluation
    for (auto& ind : island.population) {
        ind.fitness_valid = false;
    }

    // Always use the function that handles CUDA/CPU internally (or just CUDA now)
    try {
        // This function now MUST be implemented, either via CUDA or potentially
        // a CPU implementation if USE_CUDA is off (though the request was to remove CPU path).
        // If USE_CUDA is mandatory, this call assumes it's defined during compilation.
        evaluate_population_fitness_cuda(island.population, targets, x_values);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during fitness evaluation: " << e.what() << std::endl;
        // Handle error appropriately - mark all as INF since evaluation failed
        for (auto& ind : island.population) {
            ind.fitness = INF;
            ind.fitness_valid = true;
        }
    }

    // Post-evaluation checks (optional but recommended)
    for (auto& ind : island.population) {
        if (!ind.fitness_valid) {
             std::cerr << "Warning: Individual fitness not validated after evaluation." << std::endl;
             ind.fitness = INF; // Assign penalty if validation failed
             ind.fitness_valid = true;
        }
         // Ensure fitness is not NaN, handle potential issues from evaluation
         if (std::isnan(ind.fitness)) {
             // std::cerr << "Warning: NaN fitness detected, setting to INF." << std::endl;
             ind.fitness = INF;
         }
    }
}

// Evolves a single island for one generation
// Note: Renamed 'current_generation' parameter to 'generation' to match call site
void GeneticAlgorithm::evolve_island(Island& island, int generation, NodePtr& overall_best_tree, double& overall_best_fitness) {
    int current_pop_size = island.population.size();
    if (current_pop_size == 0) return; // Skip empty island

    // --- 1. Evaluation & Statistics ---
    // Fitness should already be evaluated by evaluate_population before this step
    // Sort population by fitness (best first)
    std::sort(island.population.begin(), island.population.end());

    double current_best_fitness = island.population[0].fitness;

    // Track fitness history
    island.fitness_history.push_back(current_best_fitness);

    // Update island's best
    if (current_best_fitness < island.best_fitness - 1e-9) { // Improved (with tolerance)
        island.best_fitness = current_best_fitness;
        island.stagnation_counter = 0;
    } else {
        island.stagnation_counter++;
    }

    // Update overall best
    update_best_individual(island, overall_best_tree, overall_best_fitness);

    // Update advanced features
    island.pareto_optimizer.update(island.population, targets, x_values);

    // Record successful patterns
    for (const auto& ind : island.population) {
        if (ind.fitness_valid && ind.fitness < island.best_fitness) {
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
     if (generation % 10 == 0) { // Every 10 generations
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
        child.tree = mutate_tree(child.tree, island.params.mutation_rate, MAX_TREE_DEPTH_EVOLVE);
        child.fitness_valid = false; // Ensure fitness is marked invalid after mutation


        next_generation.push_back(std::move(child)); // Move child into next generation
    }

    // --- 3. Replace Population ---
    island.population = std::move(next_generation); // Replace old population

    // --- 4. Meta-Evolution ---
    // Adapt island's parameters periodically or based on stagnation
    if (generation > 0 && generation % 50 == 0) { // Every 50 generations
         island.params.mutate();
         // std::cout << "Island " << island.id << " params mutated." << std::endl; // Debug
    }

     // Ensure all individuals have fitness marked as invalid for the next round's evaluation
     // (Except maybe elites if we are sure they weren't modified - but safer to re-eval)
     // Evaluation happens at the start of the next generation or before selection.
     // Let's ensure evaluate_population() handles the fitness_valid flag correctly.
}

// Performs migration between islands
void GeneticAlgorithm::migrate() {
    if (num_islands_ <= 1) return; // No migration needed for a single island

    auto& rng = get_rng();
    std::vector<int> island_indices(num_islands_);
    std::iota(island_indices.begin(), island_indices.end(), 0); // 0, 1, 2, ...
    std::shuffle(island_indices.begin(), island_indices.end(), rng); // Shuffle for random ring topology

    for (int i = 0; i < num_islands_; ++i) {
        int source_island_idx = island_indices[i];
        int dest_island_idx = island_indices[(i + 1) % num_islands_]; // Next island in the shuffled ring

        Island& source_island = *islands[source_island_idx];
        Island& dest_island = *islands[dest_island_idx];

        if (source_island.population.empty() || dest_island.population.empty()) continue;

        // Sort source island by fitness to select best individuals for migration
        std::sort(source_island.population.begin(), source_island.population.end());

        // Sort destination island by fitness (worst first) to select individuals to replace
        std::sort(dest_island.population.rbegin(), dest_island.population.rend()); // Sort descending

        int num_to_migrate = std::min({MIGRATION_SIZE, (int)source_island.population.size(), (int)dest_island.population.size()});

        // std::cout << "  Migrating " << num_to_migrate << " from Island " << source_island.id << " to Island " << dest_island.id << std::endl;

        for (int j = 0; j < num_to_migrate; ++j) {
            if (source_island.population[j].fitness < INF) { // Only migrate valid individuals
                // Replace the worst individual in the destination island with a clone of the migrant
                dest_island.population[j] = source_island.population[j]; // Direct copy (includes tree shared_ptr and fitness)
                dest_island.population[j].tree = clone_tree(source_island.population[j].tree); // Give dest its own copy
                dest_island.population[j].fitness_valid = false; // Mark fitness as invalid after migration
            }
        }
        // Invalidate fitness for the entire destination island population after receiving migrants?
        // Or just rely on the next generation's evaluation step. Let's rely on the next eval.
    }
}

// --- Public run() method ---

NodePtr GeneticAlgorithm::run() {
    initialize_islands(); // <--- Use the helper function

    NodePtr overall_best_individual = nullptr;
    double overall_best_fitness = INF;

    // Initial evaluation
    std::cout << "--- Initial Population Evaluation ---" << std::endl;
    for (auto& island_ptr : islands) { // Iterate over unique_ptrs
        Island& island = *island_ptr; // Dereference the pointer
        evaluate_population(island);
        update_best_individual(island, overall_best_individual, overall_best_fitness); // <--- Use the helper function
    }
     std::cout << "Initial Best Fitness: " << (overall_best_fitness >= INF ? "inf" : std::to_string(overall_best_fitness)) << std::endl;
     if(overall_best_individual) {
         std::cout << "Initial Best Tree: " << tree_to_string(overall_best_individual) << std::endl;
     }


    // Main generational loop
    for (int gen = 1; gen <= generations_; ++gen) { // Use member variable generations_
        // --- Evolution Step ---
        // Evolve islands (potentially in parallel)
        // Simple sequential evolution for now:
        for (auto& island_ptr : islands) { // Iterate over unique_ptrs
            evolve_island(*island_ptr, gen, overall_best_individual, overall_best_fitness); // Dereference the pointer
        }

        // --- Evaluation Step ---
        // std::cout << "--- Evaluating Generation " << gen << " ---" << std::endl; // Can be verbose
        for (auto& island_ptr : islands) { // Iterate over unique_ptrs
            evaluate_population(*island_ptr); // Dereference the pointer
        }

        // --- Update Overall Best ---
        double current_gen_best_fitness = INF; // Track best in *this* generation
        for (auto& island_ptr : islands) { // Iterate over unique_ptrs
             Island& island = *island_ptr; // Dereference the pointer
             update_best_individual(island, overall_best_individual, overall_best_fitness); // <--- Use the helper function
             // Find the best fitness in the current generation across all islands
             for(const auto& ind : island.population) { // Access population via dereferenced pointer
                 if(ind.fitness_valid && ind.fitness < current_gen_best_fitness) {
                     current_gen_best_fitness = ind.fitness;
                 }
             }
        }

        // --- Log Best Fitness for the Generation ---
        std::cout << "Generation " << gen << " - Best: " 
                  << std::fixed << std::setprecision(8) 
                  << (current_gen_best_fitness >= INF ? "inf" : std::to_string(current_gen_best_fitness))
                  << std::endl;


        // --- Migration Step ---
        // Use member variable migration_interval_
        if (gen % migration_interval_ == 0 && gen < generations_) { // Don't migrate on the very last generation
            std::cout << "\n--- Generation " << gen << ": Performing Migration ---" << std::endl;
            migrate();
             // Re-evaluate fitness after migration? Optional, depends on strategy.
             // If migrants replace worst individuals, re-evaluation might be good.
             // If they replace random ones, maybe wait until next generation's eval.
             // Let's skip re-evaluation for now.
        }

        // --- Advanced Features Update ---
        // ... (code for Pareto, PatternMemory - ensure correct population access if needed) ...

        // Termination condition check (e.g., fitness threshold)
        if (overall_best_fitness < FITNESS_THRESHOLD) { // Use constant from Globals.h
            std::cout << "\n--- Target Fitness Threshold Reached! ---" << std::endl;
            break;
        }
         if (gen % 100 == 0) { // Periodic status update
             std::cout << "Overall Best Fitness after " << gen << " gens: "
                       << (overall_best_fitness >= INF ? "inf" : std::to_string(overall_best_fitness)) << std::endl;
             if (overall_best_individual) {
                 std::cout << "  Best Tree Structure: " << tree_to_string(overall_best_individual) << std::endl;
             }
         }

    }

    // Final results
    std::cout << "\n--- Evolution Finished ---" << std::endl;
    std::cout << "Overall Best Fitness Achieved: " << (overall_best_fitness >= INF ? "inf" : std::to_string(overall_best_fitness)) << std::endl;
    if (overall_best_individual) {
        std::cout << "Best Solution Tree: " << tree_to_string(overall_best_individual) << std::endl;
        // Optionally, evaluate and print points for the best solution
        std::cout << "Evaluating Best Solution:" << std::endl;
        double final_raw_fitness = 0; // Needs recalculation if only final fitness stored
        bool possible_precise = true;
        for(size_t i=0; i < x_values.size(); ++i) {
            double eval_val = evaluate_tree(overall_best_individual, x_values[i]);
            std::cout << "  f(" << x_values[i] << ") = " << eval_val << " (Target: " << targets[i] << ")" << std::endl;
             if (std::isnan(eval_val) || std::isinf(eval_val)) {
                 final_raw_fitness = INF;
                 possible_precise = false;
                 break;
             }
             double diff = std::fabs(eval_val - targets[i]);
             if (diff >= 0.001) possible_precise = false;
             if (final_raw_fitness < INF) {
                final_raw_fitness += std::pow(diff, 1.3);
             }
        }
         if (possible_precise && final_raw_fitness < INF) final_raw_fitness *= 0.0001; // Apply bonus if applicable
         std::cout << "Final Raw Fitness (recalculated): " << (final_raw_fitness >= INF ? "inf" : std::to_string(final_raw_fitness)) << std::endl;

    } else {
        std::cout << "No valid solution found." << std::endl;
    }

    // Optional: Print Pareto front solutions
    // auto pareto_solutions = pareto_optimizer.get_pareto_solutions();
    // std::cout << "\nPareto Front Solutions (" << pareto_solutions.size() << "):" << std::endl;
    // for(const auto& sol_tree : pareto_solutions) {
    //     // Need to recalculate fitness/complexity or store them in ParetoSolution
    //     std::cout << "  Tree: " << tree_to_string(sol_tree) << " (Size: " << tree_size(sol_tree) << ")" << std::endl;
    // }


    return overall_best_individual; // Return the best tree found
}