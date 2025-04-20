#include "GeneticAlgorithm.h"
#include "Globals.h"
#include "Fitness.h"
#include "AdvancedFeatures.h" // Incluir este para DomainConstraints::
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <iterator>
#include <chrono>

// --- Constructor (Modificado para que evaluate_population procese todo) ---
GeneticAlgorithm::GeneticAlgorithm(const std::vector<double>& targets_ref,
                                     const std::vector<double>& x_values_ref,
                                     int total_pop,
                                     int gens,
                                     int n_islands)
    : targets(targets_ref),
      x_values(x_values_ref),
      total_population_size(total_pop),
      generations(gens),
      num_islands(n_islands),
      overall_best_fitness(INF),
      last_overall_best_fitness(INF),
      generation_last_improvement(0)
{
    // Validar y ajustar número de islas y población por isla
    if (this->num_islands <= 0) this->num_islands = 1;
    pop_per_island = this->total_population_size / this->num_islands;
    if (pop_per_island < MIN_POP_PER_ISLAND) {
        pop_per_island = MIN_POP_PER_ISLAND;
        this->num_islands = this->total_population_size / pop_per_island;
        if (this->num_islands == 0) this->num_islands = 1;
        std::cerr << "Warning: Adjusted number of islands to " << this->num_islands
                  << " for minimum population size per island (" << pop_per_island <<")." << std::endl;
    }
    this->total_population_size = this->num_islands * pop_per_island;
    std::cout << "Info: Running with " << this->num_islands << " islands, "
              << pop_per_island << " individuals per island." << std::endl;

    // Crear las islas
    islands.reserve(this->num_islands);
    for (int i = 0; i < this->num_islands; ++i) {
        try {
            islands.push_back(std::make_unique<Island>(i, pop_per_island));
        }
        catch (const std::exception& e) { std::cerr << "[ERROR] Creating Island " << i << ": " << e.what() << std::endl; throw; }
        catch (...) { std::cerr << "[ERROR] Unknown exception creating island " << i << std::endl; throw; }
    }

    // --- ELIMINADO: Bloque de evaluación especial para fórmula inyectada ---
    // if (USE_INITIAL_FORMULA) { ... }

     // Evaluación inicial de TODA la población (incluyendo la inyectada)
     // La función evaluate_population ahora simplificará y evaluará a todos.
     std::cout << "Evaluating initial population (simplifying all)..." << std::endl;
     #pragma omp parallel for
     for (int i = 0; i < islands.size(); ++i) {
        evaluate_population(*islands[i]);
     }

     // Actualizar el mejor global inicial (en serie)
     overall_best_fitness = INF;
     overall_best_tree = nullptr;
     int initial_best_island = -1;
     int initial_best_idx = -1;

     for (int i = 0; i < islands.size(); ++i) {
        for(int j=0; j < islands[i]->population.size(); ++j) {
            const auto& ind = islands[i]->population[j];
            if (ind.tree && ind.fitness_valid && ind.fitness < overall_best_fitness) {
                overall_best_fitness = ind.fitness;
                initial_best_island = i;
                initial_best_idx = j;
            }
        }
     }
     if(initial_best_island != -1 && initial_best_idx != -1) {
         overall_best_tree = clone_tree(islands[initial_best_island]->population[initial_best_idx].tree);
     }

     last_overall_best_fitness = overall_best_fitness;
     generation_last_improvement = 0;
     std::cout << "Initial best fitness: " << std::scientific << overall_best_fitness << std::fixed << std::endl;
     if (overall_best_tree) {
          std::cout << "Initial best formula size: " << tree_size(overall_best_tree) << std::endl;
          std::cout << "Initial best formula: " << tree_to_string(overall_best_tree) << std::endl;
          // Nota para saber si el mejor inicial fue la fórmula inyectada (ahora simplificada)
          if (USE_INITIAL_FORMULA && initial_best_island != -1 && initial_best_idx == 0) {
               std::cout << "   (Note: Initial best is the (simplified) injected formula from Island " << initial_best_island << ")" << std::endl;
          } else if (initial_best_island != -1) {
               std::cout << "   (Note: Initial best found in Island " << initial_best_island << ", Index " << initial_best_idx << ")" << std::endl;
          }
      } else { std::cout << "No valid initial solution found (all fitness INF?)." << std::endl; }
     std::cout << "----------------------------------------" << std::endl;
}

// --- evaluate_population (Modificado para procesar TODOS los individuos) ---
void GeneticAlgorithm::evaluate_population(Island& island) {
    int pop_size = island.population.size();
    if (pop_size == 0) return;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pop_size; ++i) {
        Individual& ind = island.population[i];

        // *** ELIMINADO: Ya no se salta individuos ***
        // if (!ind.fitness_valid) { ... }

        // Procesar siempre (simplificar y evaluar)
        if (ind.tree) {
            // Simplificar ANTES de evaluar
            ind.tree = DomainConstraints::fix_or_simplify(ind.tree);

            if (ind.tree) {
                 // Calcular fitness del árbol (posiblemente simplificado)
                 ind.fitness = evaluate_fitness(ind.tree, targets, x_values);
                 ind.fitness_valid = true;
            } else {
                 // Simplificación resultó en árbol nulo
                 ind.fitness = INF;
                 ind.fitness_valid = true;
            }
        } else {
             // Árbol era nulo inicialmente
             ind.fitness = INF;
             ind.fitness_valid = true;
        }
        // --- Fin del procesamiento ---
    } // --- Fin for loop ---
}


// --- evolve_island ---
// (Sin cambios)
void GeneticAlgorithm::evolve_island(Island& island, int current_generation) {
    int current_pop_size = island.population.size(); if (current_pop_size == 0) return;
    auto best_it = std::min_element(island.population.begin(), island.population.end(),
        [](const Individual& a, const Individual& b) {
            if (!a.tree || !a.fitness_valid) return false;
            if (!b.tree || !b.fitness_valid) return true;
            return a.fitness < b.fitness;
        });
    double current_best_fitness = INF;
    int best_idx = -1;
    if (best_it != island.population.end() && best_it->tree && best_it->fitness_valid) {
        best_idx = std::distance(island.population.begin(), best_it);
        current_best_fitness = best_it->fitness;
    }
    island.fitness_history.push_back(current_best_fitness);
    if (best_idx != -1 && current_best_fitness < INF) {
         auto local_search_result = try_local_improvement(island.population[best_idx].tree, island.population[best_idx].fitness, targets, x_values, LOCAL_SEARCH_ATTEMPTS);
         if (local_search_result.first && local_search_result.second < island.population[best_idx].fitness) {
             island.population[best_idx].tree = local_search_result.first;
             island.population[best_idx].fitness = local_search_result.second;
             island.population[best_idx].fitness_valid = true;
             current_best_fitness = local_search_result.second;
         }
    }
    if (current_best_fitness < island.best_fitness - FITNESS_EQUALITY_TOLERANCE) {
        island.best_fitness = current_best_fitness;
        island.stagnation_counter = 0;
    } else if (current_best_fitness < INF) {
        island.stagnation_counter++;
    }
    island.pareto_optimizer.update(island.population, targets, x_values);
    for(const auto& ind : island.population) {
        if(ind.tree && ind.fitness_valid && ind.fitness < PATTERN_RECORD_FITNESS_THRESHOLD) {
            island.pattern_memory.record_success(ind.tree, ind.fitness);
        }
    }
    std::vector<Individual> next_generation;
    next_generation.reserve(current_pop_size);
    int elite_count = std::max(1, static_cast<int>(current_pop_size * island.params.elite_percentage));
    if (elite_count > 0 && elite_count <= current_pop_size) {
        std::partial_sort(island.population.begin(), island.population.begin() + elite_count, island.population.end());
        int added_elites = 0;
        for (int i = 0; i < elite_count && i < island.population.size(); ++i) {
             if (island.population[i].tree && island.population[i].fitness_valid) {
                 next_generation.emplace_back(clone_tree(island.population[i].tree));
                 next_generation.back().fitness = island.population[i].fitness;
                 next_generation.back().fitness_valid = true;
                 added_elites++;
             }
        }
        elite_count = added_elites;
    } else { elite_count = 0; }
    int random_injection_count = 0;
    if (island.stagnation_counter > STAGNATION_LIMIT_ISLAND / 2) {
        random_injection_count = static_cast<int>(current_pop_size * STAGNATION_RANDOM_INJECT_PERCENT);
        for(int i = 0; i < random_injection_count && next_generation.size() < current_pop_size; ++i) {
             NodePtr random_tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL);
             if (random_tree) next_generation.emplace_back(std::move(random_tree));
        }
    }
    int pattern_injection_count = 0;
    if (random_injection_count == 0 && current_generation % PATTERN_INJECT_INTERVAL == 0) {
        pattern_injection_count = static_cast<int>(current_pop_size * PATTERN_INJECT_PERCENT);
        for (int i = 0; i < pattern_injection_count && next_generation.size() < current_pop_size; ++i) {
            NodePtr pt = island.pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_INITIAL);
            if (pt) { next_generation.emplace_back(std::move(pt)); }
            else {
                 NodePtr random_tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL);
                 if (random_tree) next_generation.emplace_back(std::move(random_tree));
            }
        }
    }
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    int remaining_slots = current_pop_size - next_generation.size();
    if (remaining_slots > 0 && !island.population.empty()) {
         long long valid_parent_count = std::count_if(island.population.begin(), island.population.end(),
                                             [](const Individual& ind){ return ind.tree && ind.fitness_valid; });
         if (valid_parent_count > 0) {
             for (int i = 0; i < remaining_slots; ++i) {
                 const Individual& p1 = tournament_selection(island.population, island.params.tournament_size);
                 Individual child;
                 if (prob_dist(rng) < island.params.crossover_rate && valid_parent_count >= 2) {
                     const Individual& p2 = tournament_selection(island.population, island.params.tournament_size);
                     if (p1.tree && p2.tree) {
                         NodePtr t1 = clone_tree(p1.tree); NodePtr t2 = clone_tree(p2.tree);
                         crossover_trees(t1, t2); child.tree = t1;
                     } else if (p1.tree) { child.tree = clone_tree(p1.tree); }
                 } else { if (p1.tree) child.tree = clone_tree(p1.tree); }
                 if (child.tree) child.tree = mutate_tree(child.tree, island.params.mutation_rate, MAX_TREE_DEPTH_MUTATION);
                 child.fitness_valid = false; next_generation.push_back(std::move(child));
             }
         } else {
              for (int i = 0; i < remaining_slots; ++i) {
                   NodePtr random_tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL);
                   if (random_tree) next_generation.emplace_back(std::move(random_tree));
              }
         }
    }
    if (next_generation.size() < current_pop_size) {
         int gap = current_pop_size - next_generation.size();
         for (int i = 0; i < gap; ++i) {
              NodePtr random_tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL);
              if (random_tree) next_generation.emplace_back(std::move(random_tree));
         }
    }
     if (next_generation.size() > current_pop_size) next_generation.resize(current_pop_size);
    island.population = std::move(next_generation);
    if (current_generation > 0 && current_generation % PARAM_MUTATE_INTERVAL == 0) island.params.mutate(island.stagnation_counter);
}

// --- migrate ---
// (Sin cambios)
void GeneticAlgorithm::migrate() {
    if (num_islands <= 1) return;
    int current_pop_per_island = islands.empty() ? 0 : islands[0]->population.size();
    if (current_pop_per_island == 0) return;
    int num_migrants = std::min(MIGRATION_SIZE, current_pop_per_island / 5);
    if (num_migrants <= 0) return;
    std::vector<std::vector<Individual>> outgoing_migrants(num_islands);
    #pragma omp parallel for
    for (int i = 0; i < num_islands; ++i) {
        Island& src = *islands[i];
        if (src.population.size() < num_migrants) continue;
        std::partial_sort(src.population.begin(), src.population.begin() + num_migrants, src.population.end());
        outgoing_migrants[i].reserve(num_migrants);
        int migrants_selected = 0;
        for (int j = 0; j < src.population.size() && migrants_selected < num_migrants; ++j) {
             if (src.population[j].tree && src.population[j].fitness_valid) {
                 Individual migrant_copy;
                 migrant_copy.tree = clone_tree(src.population[j].tree);
                 migrant_copy.fitness = src.population[j].fitness;
                 migrant_copy.fitness_valid = true;
                 outgoing_migrants[i].push_back(std::move(migrant_copy));
                 migrants_selected++;
             }
        }
    }
    for (int dest_idx = 0; dest_idx < num_islands; ++dest_idx) {
        int src_idx = (dest_idx + num_islands - 1) % num_islands;
        Island& dest = *islands[dest_idx];
        const auto& migrants_to_receive = outgoing_migrants[src_idx];
        if (migrants_to_receive.empty() || dest.population.empty()) continue;
        int replace_count = std::min((int)migrants_to_receive.size(), (int)dest.population.size());
        if (replace_count <= 0) continue;
        std::partial_sort(dest.population.begin(), dest.population.end() - replace_count, dest.population.end());
        int migrant_idx = 0;
        for (int i = 0; i < replace_count; ++i) {
            int replace_idx = dest.population.size() - 1 - i;
            if (migrant_idx < migrants_to_receive.size()) {
                 dest.population[replace_idx] = std::move(migrants_to_receive[migrant_idx++]);
                 dest.population[replace_idx].fitness_valid = false; // Marcar para reevaluar
            }
        }
    }
}


// --- run ---
// (Sin cambios)
NodePtr GeneticAlgorithm::run() {
    std::cout << "Starting Genetic Algorithm..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        #pragma omp parallel for
        for (int i = 0; i < islands.size(); ++i) {
             evaluate_population(*islands[i]);
             evolve_island(*islands[i], gen);
        }

        double current_gen_best_fitness = INF;
        int best_island_idx = -1;
        int best_ind_idx = -1;
        for (int i = 0; i < islands.size(); ++i) {
             for (int j = 0; j < islands[i]->population.size(); ++j) {
                 const auto& ind = islands[i]->population[j];
                 if (ind.tree && ind.fitness_valid && ind.fitness < current_gen_best_fitness) {
                     current_gen_best_fitness = ind.fitness;
                     best_island_idx = i; best_ind_idx = j;
                 }
             }
        }

        if (best_island_idx != -1 && current_gen_best_fitness < overall_best_fitness) {
             if (current_gen_best_fitness < overall_best_fitness) {
                  overall_best_fitness = current_gen_best_fitness;
                  overall_best_tree = clone_tree(islands[best_island_idx]->population[best_ind_idx].tree);
                  std::cout << "\n========================================" << std::endl;
                  std::cout << "New Global Best Found (Gen " << gen + 1 << ", Island " << best_island_idx << ")" << std::endl;
                  std::cout << "Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
                  std::cout << "Size: " << tree_size(overall_best_tree) << std::endl;
                  std::cout << "Formula: " << tree_to_string(overall_best_tree) << std::endl;
                  std::cout << "Predictions vs Targets:" << std::endl;
                  std::cout << std::fixed << std::setprecision(4);
                  if (overall_best_tree && !x_values.empty()) {
                      for (size_t j = 0; j < x_values.size(); ++j) {
                          double val = evaluate_tree(overall_best_tree, x_values[j]);
                          double target_val = (j < targets.size()) ? targets[j] : std::nan("");
                          double diff = (!std::isnan(val) && !std::isnan(target_val)) ? std::fabs(val - target_val) : std::nan("");
                          std::cout << "  x=" << std::setw(8) << x_values[j]
                                    << ": Pred=" << std::setw(12) << val
                                    << ", Target=" << std::setw(12) << target_val
                                    << ", Diff=" << std::setw(12) << diff << std::endl;
                      }
                  } else { std::cout << "  (No data or no valid tree to show predictions)" << std::endl; }
                  std::cout << "========================================" << std::endl;
                  last_overall_best_fitness = overall_best_fitness;
                  generation_last_improvement = gen;
             }
        } else {
             if (overall_best_fitness < INF && (gen - generation_last_improvement) >= GLOBAL_STAGNATION_LIMIT) {
                  std::cout << "\n========================================" << std::endl;
                  std::cout << "TERMINATION: Global best fitness hasn't improved for " << GLOBAL_STAGNATION_LIMIT << " generations." << std::endl;
                  std::cout << "Stopping at Generation " << gen + 1 << "." << std::endl;
                  std::cout << "========================================" << std::endl;
                  break;
             }
        }

        if ((gen + 1) % MIGRATION_INTERVAL == 0 && num_islands > 1) {
             migrate();
             #pragma omp parallel for
             for (int i = 0; i < islands.size(); ++i) { evaluate_population(*islands[i]); }
        }

        if (overall_best_fitness < EXACT_SOLUTION_THRESHOLD) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Solution found meeting criteria at Generation " << gen + 1 << "!" << std::endl;
            std::cout << "Final Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
            if(overall_best_tree) {
                 std::cout << "Final Formula Size: " << tree_size(overall_best_tree) << std::endl;
                 std::cout << "Final Formula: " << tree_to_string(overall_best_tree) << std::endl;
            }
            std::cout << "========================================" << std::endl;
            break;
        }

        if ((gen + 1) % PROGRESS_REPORT_INTERVAL == 0 || gen == generations - 1) {
             auto current_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double> elapsed = current_time - start_time;
             std::cout << "\n--- Generation " << gen + 1 << "/" << generations
                       << " (Elapsed: " << std::fixed << std::setprecision(2) << elapsed.count() << "s) ---" << std::endl;
             std::cout << "Overall Best Fitness: " << std::scientific << overall_best_fitness << std::fixed << std::endl;
              if(overall_best_tree) { std::cout << "Best Formula Size: " << tree_size(overall_best_tree) << std::endl; }
              else { std::cout << "Best Formula Size: N/A" << std::endl; }
              std::cout << "(Last improvement at gen: " << generation_last_improvement + 1 << ")" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end_time - start_time;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Evolution Finished!" << std::endl;
    std::cout << "Total Time: " << std::fixed << std::setprecision(2) << total_elapsed.count() << " seconds" << std::endl;
    std::cout << "Final Best Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
     if (overall_best_tree) {
         std::cout << "Final Best Formula Size: " << tree_size(overall_best_tree) << std::endl;
         std::cout << "Final Best Formula: " << tree_to_string(overall_best_tree) << std::endl;
          std::cout << "--- Final Verification ---" << std::endl;
          double final_check_fitness = evaluate_fitness(overall_best_tree, targets, x_values);
          std::cout << "Recalculated Fitness: " << std::fixed << std::setprecision(8) << final_check_fitness << std::endl;
          std::cout << std::fixed << std::setprecision(4);
          for (size_t j = 0; j < x_values.size(); ++j) {
                double val = evaluate_tree(overall_best_tree, x_values[j]);
                 double target_val = (j < targets.size()) ? targets[j] : std::nan("");
                 double diff = (!std::isnan(val) && !std::isnan(target_val)) ? std::fabs(val - target_val) : std::nan("");
                 std::cout << "  x=" << std::setw(8) << x_values[j]
                          << ": Pred=" << std::setw(12) << val
                          << ", Target=" << std::setw(12) << target_val
                          << ", Diff=" << std::setw(12) << diff << std::endl;
           }
     } else { std::cout << "No valid solution found." << std::endl; }
      std::cout << "========================================" << std::endl;
    return overall_best_tree;
}

