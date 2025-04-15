#include "GeneticAlgorithm.h"
#include "Globals.h"
#include "Fitness.h"
#include <iostream>
#include <algorithm> // For sort, min_element, sample
#include <vector>
#include <cmath>
#include <omp.h> // For OpenMP parallelization
#include <iomanip> // For std::setprecision
#include <fstream>
#include <unordered_set>


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
// Uses OpenMP for parallel evaluation.
void GeneticAlgorithm::evaluate_population(Island& island) {
    int pop_size = island.population.size();

    // Use OpenMP to parallelize the fitness calculation loop
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pop_size; ++i) {
        Individual& ind = island.population[i];
        // Only evaluate if fitness is not valid or tree pointer is null (shouldn't happen)
        if (!ind.fitness_valid && ind.tree) {
            ind.tree = DomainConstraints::fix_or_simplify(ind.tree); // Simplify before eval
            if (ind.tree) { // Check if simplification resulted in null
                 ind.fitness = evaluate_fitness(ind.tree, targets, x_values);
                 ind.fitness_valid = true; // Mark as valid after evaluation
            } else {
                 ind.fitness = INF; // Penalize if simplification failed
                 ind.fitness_valid = true;
            }

        } else if (!ind.tree) {
             ind.fitness = INF;
             ind.fitness_valid = true; // Mark invalid tree as evaluated (with Inf fitness)
        }
        // If fitness_valid is already true, do nothing.
    }

    // Optional: Handle cases where all evaluations resulted in INF?
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

    // --- Estadísticas automáticas ---
    auto stats = Statistics::compute_fitness_stats(island.population);
    double diversity = Statistics::compute_structural_diversity(island.population);
    // Solo imprimir si se desea debug, comentar para producción
    // if (current_generation % 10 == 0) {
    //     std::cout << "[Isla " << island.id << "] Gen " << current_generation
    //               << ": Fitness (min/mean/max/std): " << stats.min << "/" << stats.mean << "/" << stats.max << "/" << stats.stddev
    //               << ", Diversidad estructural: " << diversity << std::endl;
    // }

    // --- Ajuste automático de parámetros ---
    // Si la diversidad estructural es baja, aumentar mutación y agregar aleatorios
    if (diversity < 0.25) { // Más agresivo
        island.params.mutation_rate = std::min(0.7, island.params.mutation_rate * 1.7);
        int extra_random = static_cast<int>(island.population.size() * 0.20); // 20% aleatorios
        for (int i = 0; i < extra_random; ++i) {
            island.population.push_back(Individual(generate_random_tree(MAX_TREE_DEPTH_INITIAL)));
        }
        evaluate_population(island);
    }
    // Si la desviación estándar de fitness es baja (convergencia), aumentar presión de selección
    if (stats.stddev < 1e-2) {
        island.params.tournament_size = std::min(40, island.params.tournament_size + 4); // Más presión
        island.params.elite_percentage = std::min(0.30, island.params.elite_percentage + 0.02);
    } else {
        // Si hay mucha dispersión, reducir presión de selección
        island.params.tournament_size = std::max(5, island.params.tournament_size - 1);
        island.params.elite_percentage = std::max(0.02, island.params.elite_percentage - 0.01);
    }

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

    // --- Reemplazo de fórmulas inútiles (INF) ---
    int inf_count = 0;
    for (const auto& ind : island.population) {
        if (ind.fitness >= INF/2) inf_count++;
    }
    if (inf_count > current_pop_size / 2) { // Si más de la mitad son inútiles
        for (auto& ind : island.population) {
            if (ind.fitness >= INF/2) {
                ind.tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL);
                ind.fitness_valid = false;
            }
        }
        evaluate_population(island);
    }

    // Update overall best across all islands
    update_overall_best(island);


    // Update Pareto Front and Pattern Memory para esta isla
    island.pareto_optimizer.update(island.population, targets, x_values);
    for(const auto& ind : island.population) {
        // Record success for good solutions (e.g., fitness < 10 or significantly better than average)
        if(ind.fitness_valid && ind.fitness < 10.0) {
            island.pattern_memory.record_success(ind.tree, ind.fitness);
            global_pattern_memory.record_success(ind.tree, ind.fitness); // También registrar en la memoria global
        }
    }

    // --- Diversidad estructural: medir y actuar si es baja ---
    std::unordered_set<size_t> unique_hashes;
    for (const auto& ind : island.population) {
        if (ind.tree) unique_hashes.insert(tree_structural_hash(ind.tree));
    }
    double structural_diversity = static_cast<double>(unique_hashes.size()) / std::max(1, (int)island.population.size());
    if (structural_diversity < 0.25) { // Umbral configurable
        island.params.mutation_rate = std::min(0.5, island.params.mutation_rate * 1.5);
        int extra_random = static_cast<int>(island.population.size() * 0.1);
        for (int i = 0; i < extra_random; ++i) {
            island.population.push_back(Individual(generate_random_tree(MAX_TREE_DEPTH_INITIAL)));
        }
        // Evaluar fitness de toda la población tras inyección de aleatorios
        evaluate_population(island);
        // (Ya no imprimir mensaje de baja diversidad estructural)
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

    // --- Inyección de diversidad aleatoria: 15% de la población ---
    int random_diversity_count = static_cast<int>(current_pop_size * 0.15);
    for (int i = 0; i < random_diversity_count; ++i) {
        next_generation.emplace_back(generate_random_tree(MAX_TREE_DEPTH_INITIAL));
        // Fitness se evaluará después
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
             NodePtr pattern_tree = nullptr;
             // 50% probabilidad de usar memoria global o local
             if (get_rng()() % 2 == 0) {
                 pattern_tree = global_pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_INITIAL);
             } else {
                 pattern_tree = island.pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_INITIAL);
             }
             if (pattern_tree) {
                 next_generation.emplace_back(pattern_tree);
             } else {
                  // Fallback if no pattern suggested
                  next_generation.emplace_back(generate_random_tree(MAX_TREE_DEPTH_INITIAL));
             }
         }
     }

    // --- Transferencia de Conocimiento: Transfer learning de sub-árboles ---
    if (current_generation > 0 && current_generation % 20 == 0) { // Cada 20 generaciones
        NodePtr transfer_pattern = global_pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_MUTATION);
        if (transfer_pattern) {
            // Seleccionar un individuo aleatorio (no elite) para insertar el sub-árbol
            int idx = elite_count + (get_rng()() % std::max(1, current_pop_size - elite_count));
            if (idx < island.population.size()) {
                // Reemplazar un sub-árbol aleatorio del individuo seleccionado
                NodePtr& target_tree = island.population[idx].tree;
                std::vector<NodePtr*> nodes;
                collect_node_ptrs(target_tree, nodes);
                if (!nodes.empty()) {
                    int sub_idx = get_rng()() % nodes.size();
                    // Insertar el patrón transferido, con ligera mutación para adaptación
                    *nodes[sub_idx] = mutate_tree(transfer_pattern, 0.2, MAX_TREE_DEPTH_MUTATION);
                    island.population[idx].fitness_valid = false; // Marcar para reevaluar
                }
            }
        }
    }

    // Búsqueda local inteligente periódica sobre los mejores individuos
    if (current_generation % 5 == 0) { // Más frecuente
        int local_search_top = std::min(6, current_pop_size); // Top 6 individuos
        for (int i = 0; i < local_search_top; ++i) {
            auto local_result = try_local_improvement(
                island.population[i].tree,
                island.population[i].fitness,
                targets, x_values, 20 // Más intentos
            );
            if (local_result.second < island.population[i].fitness) {
                island.population[i].tree = local_result.first;
                island.population[i].fitness = local_result.second;
                island.population[i].fitness_valid = true;
                if (i == 0 && local_result.second < island.best_fitness) {
                    island.best_fitness = local_result.second;
                }
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

    // Registrar desempeño de los parámetros evolutivos actuales
    island.record_param_performance(current_best_fitness);

    // Meta-evolución: Si la isla está estancada, adopta parámetros de otra isla exitosa
    if (island.stagnation_counter > STAGNATION_LIMIT) {
        // Buscar la isla con mejor fitness promedio reciente
        int best_island_idx = -1;
        double best_fitness = INF;
        for (size_t i = 0; i < islands.size(); ++i) {
            if (islands[i].get() == &island) continue; // Saltar la propia
            if (!islands[i]->fitness_history.empty()) {
                double avg_fit = islands[i]->fitness_history.back();
                if (avg_fit < best_fitness) {
                    best_fitness = avg_fit;
                    best_island_idx = static_cast<int>(i);
                }
            }
        }
        if (best_island_idx >= 0) {
            // Copiar parámetros de la isla exitosa y mutar ligeramente
            island.params = islands[best_island_idx]->params;
            island.params.mutate();
        }
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
        for (int j = 0; j < num_migrants && j < source_island.population.size(); ++j) {
            // Clone the migrant's tree for sending
            outgoing_migrants[i].emplace_back(clone_tree(source_island.population[j].tree));
            outgoing_migrants[i].back().fitness = source_island.population[j].fitness; // Copy fitness
            outgoing_migrants[i].back().fitness_valid = source_island.population[j].fitness_valid; // Copy validity
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

        // Replace the worst individuals with the incoming migrants
        int replaced_count = 0;
        for (int i = 0; i < migrants.size() && i < dest_island.population.size(); ++i) {
             // Check for duplicates? Maybe not essential.
            dest_island.population[i] = migrants[i]; // Replace (copies the Individual struct)
            replaced_count++;
        }
         // if (replaced_count > 0) {
         //     std::cout << "Island " << dest_idx << " received " << replaced_count << " migrants." << std::endl;
         // }
    }

    // --- Sincronización automática de memorias de patrones ---
    // 1. Cada isla exporta sus patrones a la memoria global
    for (auto& island_ptr : islands) {
        global_pattern_memory.import_from(island_ptr->pattern_memory);
    }
    // 2. Cada isla importa los patrones globales
    for (auto& island_ptr : islands) {
        island_ptr->pattern_memory.import_from(global_pattern_memory);
    }
}


NodePtr GeneticAlgorithm::run() {
    std::cout << "Starting Genetic Algorithm..." << std::endl;
    std::cout << "Islands: " << num_islands << ", Pop/Island: " << pop_per_island
              << ", Generations: " << generations << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- Logging inteligente ---
    std::ofstream log_fitness("evolution_fitness.csv");
    std::ofstream log_pareto("pareto_front.csv");
    std::ofstream log_best("best_formulas.txt");
    log_fitness << "Generation,OverallBest";
    for (int i = 0; i < num_islands; ++i) log_fitness << ",Island" << i;
    log_fitness << std::endl;

    for (int gen = 0; gen < generations; ++gen) {

        // Evaluar todas las islas en paralelo (batch)
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
            evaluate_population(*islands[i]);
        }

        // Evolucionar cada isla (puede ser en serie para evitar condiciones de carrera en migración)
        for (auto& island_ptr : islands) {
            evolve_island(*island_ptr, gen);
        }

        // Periodic Migration
        if ((gen + 1) % MIGRATION_INTERVAL == 0 && num_islands > 1) {
             std::cout << "\n--- Generation " << gen + 1 << ": Performing Migration ---" << std::endl;
             migrate();
             // Fitness of migrants needs re-evaluation in their new islands?
             // evaluate_population() at the start of the next gen handles this.
        }

        // Penalización adaptativa de complejidad cada 50 generaciones
        if ((gen + 1) % 50 == 0) {
            double avg_best = 0.0;
            double min_best = INF, max_best = -INF;
            for (const auto& island_ptr : islands) {
                if (!island_ptr->fitness_history.empty()) {
                    double fit = island_ptr->fitness_history.back();
                    avg_best += fit;
                    min_best = std::min(min_best, fit);
                    max_best = std::max(max_best, fit);
                }
            }
            avg_best /= islands.size();
            double diversity = max_best - min_best;
            // Si la diversidad es baja o el progreso es lento, *reducir* penalización para permitir exploración
            if (diversity < 1e-2) { // Low diversity -> decrease penalty to allow bigger trees
                COMPLEXITY_PENALTY_FACTOR = std::max(COMPLEXITY_PENALTY_FACTOR * 0.80, 0.0005); // Más rápido
                std::cout << "[Penalización Adaptativa] Diversidad baja. Reduzco penalización de complejidad a " << COMPLEXITY_PENALTY_FACTOR << std::endl;
            } else if (diversity > 1.0) { // High diversity -> increase penalty to focus search
                COMPLEXITY_PENALTY_FACTOR = std::min(COMPLEXITY_PENALTY_FACTOR * 1.20, 7.0); // Más fuerte
                std::cout << "[Penalización Adaptativa] Diversidad alta. Aumento penalización de complejidad a " << COMPLEXITY_PENALTY_FACTOR << std::endl;
            }

            // Logging cada 50 generaciones
            log_fitness << (gen + 1) << "," << overall_best_fitness;
            for (const auto& island_ptr : islands) {
                double fit = island_ptr->fitness_history.empty() ? -1 : island_ptr->fitness_history.back();
                log_fitness << "," << fit;
            }
            log_fitness << std::endl;
            // Logging del Pareto Front
            log_pareto << "Generation: " << (gen + 1) << std::endl;
            for (const auto& sol : islands[0]->pareto_optimizer.get_pareto_front()) {
                log_pareto << sol.accuracy << "," << sol.complexity << "," << tree_to_string(sol.tree) << std::endl;
            }
            log_pareto << std::endl;
            // Logging de la mejor fórmula
            if (overall_best_tree) {
                log_best << "Gen " << (gen + 1) << ": " << tree_to_string(overall_best_tree) << " | Fitness: " << overall_best_fitness << std::endl;
            }
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

    log_fitness.close();
    log_pareto.close();
    log_best.close();

    return overall_best_tree; // Return the best tree found
}