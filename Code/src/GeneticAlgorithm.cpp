#include "GeneticAlgorithm.h"
#include "Globals.h"
#include "Fitness.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <iterator>
#include <chrono>

// --- Constructor CORREGIDO (Firma completa + inicialización) ---
GeneticAlgorithm::GeneticAlgorithm(const std::vector<double>& targets_ref,
                                     const std::vector<double>& x_values_ref,
                                     int total_pop,
                                     int gens,
                                     int n_islands) // <-- Firma completa
    : targets(targets_ref),           // <-- Inicializar referencia targets
      x_values(x_values_ref),         // <-- Inicializar referencia x_values
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
        try { islands.push_back(std::make_unique<Island>(i, pop_per_island)); }
        catch (const std::exception& e) { std::cerr << "[ERROR] Island " << i << ": " << e.what() << std::endl; throw; }
        catch (...) { std::cerr << "[ERROR] Unknown exception island " << i << std::endl; throw; }
    }

     // Mejora de Población Inicial
     auto [pattern_type, pattern_value] = detect_target_pattern(this->targets);
     if (pattern_type != "none") {
         NodePtr pattern_tree = generate_pattern_based_tree(pattern_type, pattern_value);
         if (pattern_tree) {
             std::cout << "Injecting pattern-based seed: " << tree_to_string(pattern_tree) << std::endl;
             for (auto& island_ptr : islands) { /* ... inyectar ... */ }
         }
     }

     // Evaluación inicial
     std::cout << "Evaluating initial population..." << std::endl;
     for (int i = 0; i < islands.size(); ++i) {
        evaluate_population(*islands[i]);
        update_overall_best(*islands[i]);
     }
     last_overall_best_fitness = overall_best_fitness;
     generation_last_improvement = 0;
     std::cout << "Initial best fitness: " << std::fixed << std::setprecision(6) << overall_best_fitness << std::endl;
     if (overall_best_tree) {
          std::cout << "Initial best formula size: " << tree_size(overall_best_tree) << std::endl;
          std::cout << "Initial best formula: " << tree_to_string(overall_best_tree) << std::endl;
      } else { std::cout << "No valid initial solution found." << std::endl; }
     std::cout << "----------------------------------------" << std::endl;
}

// Evalúa el fitness (CON DEBUG DE SIMPLIFICACIÓN)
void GeneticAlgorithm::evaluate_population(Island& island) {
    int pop_size = island.population.size();
    if (pop_size == 0) return;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pop_size; ++i) {
        Individual& ind = island.population[i];
        if (!ind.fitness_valid) {
            if (ind.tree) {
                // NodePtr original_tree_ptr = ind.tree; // Para debug
                ind.tree = DomainConstraints::fix_or_simplify(ind.tree);

                if (ind.tree) {
                     ind.fitness = evaluate_fitness(ind.tree, targets, x_values);
                     ind.fitness_valid = true;
                } else {
                     // ¡Simplificación falló!
                     std::cout << "[DEBUG EvalPop] fix_or_simplify returned null! Assigning INF fitness." << std::endl;
                     // std::cout << "  Original tree was: " << tree_to_string(original_tree_ptr) << std::endl;
                     ind.fitness = INF;
                     ind.fitness_valid = true;
                }
            } else {
                 ind.fitness = INF;
                 ind.fitness_valid = true;
            }
        }
    }
}

// Actualiza mejor global
void GeneticAlgorithm::update_overall_best(const Island& island) {
     for (const auto& ind : island.population) {
         if (ind.fitness_valid && ind.fitness < overall_best_fitness) {
             overall_best_fitness = ind.fitness;
             overall_best_tree = clone_tree(ind.tree);
             // Imprimir mejora
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
                 std::cout << "  x=" << std::setw(5) << x_values[j]
                          << ": Pred=" << std::setw(10) << val
                          << ", Target=" << std::setw(10) << targets[j]
                          << ", Diff=" << std::setw(10) << diff << std::endl;
            }
             std::cout << "========================================" << std::endl;
         }
     }
 }

// Evoluciona una isla
void GeneticAlgorithm::evolve_island(Island& island, int current_generation) {
    int current_pop_size = island.population.size(); if (current_pop_size == 0) return;

    // Estadísticas y Búsqueda Local
    auto best_it = std::min_element(island.population.begin(), island.population.end());
    int best_idx = (best_it == island.population.end()) ? -1 : std::distance(island.population.begin(), best_it);
    double current_best_fitness = (best_idx != -1 && best_it->fitness_valid) ? best_it->fitness : INF;
    island.fitness_history.push_back(current_best_fitness);
    if (best_idx != -1 && current_best_fitness < INF) {
         auto local_search_result = try_local_improvement(island.population[best_idx].tree, island.population[best_idx].fitness, targets, x_values, LOCAL_SEARCH_ATTEMPTS);
         if (local_search_result.second < island.population[best_idx].fitness) {
             island.population[best_idx].tree = local_search_result.first;
             island.population[best_idx].fitness = local_search_result.second;
             current_best_fitness = local_search_result.second;
         }
    }
    // Estancamiento local
    if (current_best_fitness < island.best_fitness - FITNESS_EQUALITY_TOLERANCE) {
        island.best_fitness = current_best_fitness; island.stagnation_counter = 0;
    } else { island.stagnation_counter++; }
    // Actualizar Pareto y Memoria
    island.pareto_optimizer.update(island.population, targets, x_values);
    for(const auto& ind : island.population) { if(ind.fitness_valid && ind.fitness < PATTERN_RECORD_FITNESS_THRESHOLD) { island.pattern_memory.record_success(ind.tree, ind.fitness); } }

    // Selección y Reproducción
    std::vector<Individual> next_generation; next_generation.reserve(current_pop_size);
    // Elitismo
    int elite_count = std::max(1, static_cast<int>(current_pop_size * island.params.elite_percentage));
    if (elite_count > 0 && elite_count <= current_pop_size) { std::partial_sort(island.population.begin(), island.population.begin() + elite_count, island.population.end()); for (int i = 0; i < elite_count; ++i) next_generation.push_back(island.population[i]); }
    // Inyección Aleatoria y por Patrones
    int random_injection_count = 0; if (island.stagnation_counter > STAGNATION_LIMIT_ISLAND / 2) { random_injection_count = static_cast<int>(current_pop_size * STAGNATION_RANDOM_INJECT_PERCENT); for(int i = 0; i < random_injection_count && next_generation.size() < current_pop_size; ++i) next_generation.emplace_back(generate_random_tree(MAX_TREE_DEPTH_INITIAL)); }
    int pattern_injection_count = 0; if (current_generation % PATTERN_INJECT_INTERVAL == 0 && random_injection_count == 0) { pattern_injection_count = static_cast<int>(current_pop_size * PATTERN_INJECT_PERCENT); for (int i = 0; i < pattern_injection_count && next_generation.size() < current_pop_size; ++i) { NodePtr pt = island.pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_INITIAL); if (pt) next_generation.emplace_back(pt); else next_generation.emplace_back(generate_random_tree(MAX_TREE_DEPTH_INITIAL)); } }
    // Rellenar con hijos
    auto& rng = get_rng(); std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    int remaining_slots = current_pop_size - next_generation.size();
    for (int i = 0; i < remaining_slots; ++i) {
        const Individual& p1 = tournament_selection(island.population, island.params.tournament_size); Individual child;
        if (prob_dist(rng) < island.params.crossover_rate) { const Individual& p2 = tournament_selection(island.population, island.params.tournament_size); NodePtr t1 = clone_tree(p1.tree), t2 = clone_tree(p2.tree); crossover_trees(t1, t2); child.tree = t1; }
        else { child.tree = clone_tree(p1.tree); }
        child.tree = mutate_tree(child.tree, island.params.mutation_rate, MAX_TREE_DEPTH_MUTATION); child.fitness_valid = false;
        next_generation.push_back(std::move(child));
    }

    // Reemplazar Población
    island.population = std::move(next_generation);

    // Meta-Evolución
    if (current_generation > 0 && current_generation % PARAM_MUTATE_INTERVAL == 0) { island.params.mutate(island.stagnation_counter); }
}

// Migración
void GeneticAlgorithm::migrate() {
    if (num_islands <= 1) return;
    int num_migrants = std::min(MIGRATION_SIZE, pop_per_island / 5); if (num_migrants <= 0) return;
    std::vector<std::vector<Individual>> outgoing_migrants(num_islands);
    // Selección
    for (int i = 0; i < num_islands; ++i) { Island& src = *islands[i]; if (src.population.size() < num_migrants) continue; std::partial_sort(src.population.begin(), src.population.begin() + num_migrants, src.population.end()); outgoing_migrants[i].reserve(num_migrants); for (int j = 0; j < num_migrants; ++j) { Individual mc; mc.tree = clone_tree(src.population[j].tree); mc.fitness = src.population[j].fitness; mc.fitness_valid = src.population[j].fitness_valid; outgoing_migrants[i].push_back(std::move(mc)); } }
    // Recepción
    for (int dest_idx = 0; dest_idx < num_islands; ++dest_idx) { int src_idx = (dest_idx + num_islands - 1) % num_islands; Island& dest = *islands[dest_idx]; const auto& migrants = outgoing_migrants[src_idx]; if (migrants.empty() || dest.population.empty()) continue; int replace_count = std::min((int)migrants.size(), (int)dest.population.size()); if (replace_count <= 0) continue; std::partial_sort(dest.population.begin(), dest.population.end() - replace_count, dest.population.end(), [](const Individual& a, const Individual& b) { if (!a.fitness_valid && b.fitness_valid) return false; if (a.fitness_valid && !b.fitness_valid) return true; if (!a.fitness_valid && !b.fitness_valid) return false; return a.fitness < b.fitness; }); int migrant_idx = 0; for (int i = 0; i < replace_count; ++i) { int replace_idx = dest.population.size() - 1 - i; dest.population[replace_idx] = std::move(migrants[migrant_idx++]); dest.population[replace_idx].fitness_valid = false; } }
}

// --- Ejecuta el algoritmo (CUERPO RESTAURADO) ---
NodePtr GeneticAlgorithm::run() {
    std::cout << "Starting Genetic Algorithm..." << std::endl;

    // Bucle principal de generaciones
    for (int gen = 0; gen < generations; ++gen) {

        // Evolucionar islas
        for (auto& island_ptr : islands) {
            evaluate_population(*island_ptr);
            evolve_island(*island_ptr, gen);
        }

        // Actualizar Mejor Global y Chequear Estancamiento Global
        // double previous_overall_best = overall_best_fitness; // No es necesario si usamos last_overall_best_fitness
        for(const auto& island_ptr : islands) {
            // Re-chequear el mejor de cada isla contra el global
            // (update_overall_best actualiza e imprime si mejora)
            update_overall_best(*island_ptr);
        }

        // Comprobar si el mejor global REALMENTE mejoró
        if (overall_best_fitness < last_overall_best_fitness - FITNESS_EQUALITY_TOLERANCE) {
            last_overall_best_fitness = overall_best_fitness;
            generation_last_improvement = gen;
        } else {
            // Si no hubo mejora, comprobar límite de estancamiento global
            if ((gen - generation_last_improvement) >= GLOBAL_STAGNATION_LIMIT) {
                 std::cout << "\n========================================" << std::endl;
                 std::cout << "TERMINATION: Global best fitness hasn't improved for "
                           << GLOBAL_STAGNATION_LIMIT << " generations." << std::endl;
                 std::cout << "Stopping at Generation " << gen + 1 << "." << std::endl;
                 std::cout << "========================================" << std::endl;
                 break; // Salir del bucle
            }
        }

        // Migración
        if ((gen + 1) % MIGRATION_INTERVAL == 0 && num_islands > 1) {
             std::cout << "\n--- Generation " << gen + 1 << ": Performing Migration ---" << std::endl;
             migrate();
        }

        // Condición de terminación por fitness perfecto
        if (overall_best_fitness < EXACT_SOLUTION_THRESHOLD) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Solution found meeting criteria at Generation " << gen + 1 << "!" << std::endl;
            std::cout << "========================================" << std::endl;
            break;
        }

        // Informe de progreso
        if ((gen + 1) % PROGRESS_REPORT_INTERVAL == 0 || gen == generations - 1) {
             std::cout << "\n--- Generation " << gen + 1 << " ---" << std::endl;
             std::cout << "Overall Best Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
              if(overall_best_tree) {
                   std::cout << "Best Formula Size: " << tree_size(overall_best_tree) << std::endl;
              }
              std::cout << "(Last improvement at gen: " << generation_last_improvement + 1 << ")" << std::endl;
        }
    } // Fin del bucle de generaciones

    // --- Resultados Finales ---
    std::cout << "\n========================================" << std::endl;
    std::cout << "Evolution Finished!" << std::endl;
    std::cout << "Final Best Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
     if (overall_best_tree) {
         std::cout << "Final Best Formula Size: " << tree_size(overall_best_tree) << std::endl;
         std::cout << "Final Best Formula: " << tree_to_string(overall_best_tree) << std::endl;
          // Verificación final
          std::cout << "--- Verification ---" << std::endl;
          std::cout << std::fixed << std::setprecision(4);
          for (size_t j = 0; j < x_values.size(); ++j) {
                double val = evaluate_tree(overall_best_tree, x_values[j]);
                double diff = std::fabs(val - targets[j]);
                 std::cout << "  x=" << std::setw(5) << x_values[j]
                          << ": Pred=" << std::setw(10) << val
                          << ", Target=" << std::setw(10) << targets[j]
                          << ", Diff=" << std::setw(10) << diff << std::endl;
           }
     } else {
         std::cout << "No valid solution found." << std::endl;
     }
      std::cout << "========================================" << std::endl;

    // Asegurarse de devolver el mejor árbol
    return overall_best_tree; // <-- Return restaurado
}
