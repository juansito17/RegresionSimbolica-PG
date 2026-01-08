#include "GeneticAlgorithm.h"
#include "Globals.h"
#include "Fitness.h"
#include "AdvancedFeatures.h" // Incluir este para DomainConstraints::
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
#include "FitnessGPU.cuh"     // Para funciones de GPU
#include <cuda_runtime.h>     // Para CUDA runtime
#endif
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <iterator>
#include <chrono>
#include <unordered_set>

// --- Constructor (Modificado para que evaluate_population procese todo) ---
// --- Constructor (Modificado para que evaluate_population procese todo) ---
GeneticAlgorithm::GeneticAlgorithm(const std::vector<double>& targets_ref,
                                     const std::vector<std::vector<double>>& x_values_ref,
                                     int total_pop,
                                     int gens,
                                     const std::vector<std::string>& seeds,
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

    // --- INJECT SEEDS ---
    if (!seeds.empty()) {
        std::cout << "Info: Injecting " << seeds.size() << " seed formulas into population..." << std::endl;
        int seed_idx = 0;
        
        // Distribute seeds cyclically across islands to promote diversity
        for (int i = 0; i < this->num_islands && seed_idx < seeds.size(); ++i) {
            for(size_t j = 0; j < islands[i]->population.size(); ++j) {
                if (seed_idx >= seeds.size()) break;

                try {
                    NodePtr parsed_tree = parse_formula_string(seeds[seed_idx]);
                    if (parsed_tree) {
                        islands[i]->population[j].tree = std::move(parsed_tree);
                    }
                    seed_idx++; 
                } catch (const std::exception& e) {
                    std::cerr << "[Warning] Failed to parse seed formula: " << seeds[seed_idx] << " | Error: " << e.what() << std::endl;
                    seed_idx++; // Skip this seed
                }
            }
        }
    }

#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    bool gpu_init_failed = false;
    if (!FORCE_CPU_MODE) {
        // Asignar memoria en la GPU y copiar datos
        size_t targets_size = targets.size() * sizeof(double);
        // Multivariable: flatten X values [NUM_SAMPLES * NUM_FEATURES]
        size_t n_samples = x_values.size();
        size_t n_features = (n_samples > 0) ? x_values[0].size() : 0;
        size_t x_values_size = n_samples * n_features * sizeof(double);
        
        // Linearize
        std::vector<double> flattened_x;
        flattened_x.reserve(n_samples * n_features);
        for(const auto& row : x_values) {
            flattened_x.insert(flattened_x.end(), row.begin(), row.end());
        }

        cudaError_t err_t = cudaMalloc(&d_targets, targets_size);
        cudaError_t err_x = cudaMalloc(&d_x_values, x_values_size);

        if (err_t != cudaSuccess || err_x != cudaSuccess) {
            std::cerr << "[WARNING] CUDA memory allocation failed: "
                      << cudaGetErrorString(err_t) << " | " << cudaGetErrorString(err_x) << std::endl;
            std::cerr << "[INFO] Falling back to CPU mode." << std::endl;
            gpu_init_failed = true;
            // Clean up any partial allocation
            if (d_targets) { cudaFree(d_targets); d_targets = nullptr; }
            if (d_x_values) { cudaFree(d_x_values); d_x_values = nullptr; }
        } else {
            cudaMemcpy(d_targets, targets.data(), targets_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_x_values, flattened_x.data(), x_values_size, cudaMemcpyHostToDevice);
            
            // Initialize global GPU buffers for batch evaluation of ALL islands
            init_global_gpu_buffers(global_gpu_buffers);
            
            // Initialize double-buffered GPU for async pipelining
            init_double_buffered_gpu(double_buffer_gpu);
            
            std::cout << "GPU buffers initialized for global batch evaluation (max " 
                      << total_population_size << " trees in single kernel call)" << std::endl;
            std::cout << "Double-buffered GPU enabled for async CPU/GPU overlap" << std::endl;
        }
    }
    
    if (FORCE_CPU_MODE || gpu_init_failed) {
        std::cout << "Using CPU for all evaluations" << std::endl;
    }
#endif

     // Evaluación inicial de TODA la población (incluyendo la inyectada)
     // La función evaluate_population ahora simplificará y evaluará a todos.
     std::cout << "Evaluating initial population (simplifying all)..." << std::endl;
     evaluate_all_islands(); // Use new global batch evaluation

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

GeneticAlgorithm::~GeneticAlgorithm() {
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    if (!FORCE_CPU_MODE) {
        // Cleanup double-buffered GPU
        cleanup_double_buffered_gpu(double_buffer_gpu);
        
        // Cleanup global GPU buffers
        cleanup_global_gpu_buffers(global_gpu_buffers);
        
        if (d_targets) {
            cudaFree(d_targets);
            d_targets = nullptr;
        }
        if (d_x_values) {
            cudaFree(d_x_values);
            d_x_values = nullptr;
        }
    }
#endif
    // El destructor de std::unique_ptr en 'islands' se encarga de liberar la memoria de las islas.
    // 'overall_best_tree' es un NodePtr. Si es un smart pointer (como std::unique_ptr<Node>),
    // su memoria se liberará automáticamente. Si es un puntero crudo, necesitaría una función delete_tree.
    // Asumiendo que NodePtr es un smart pointer o que la liberación se maneja en otro lugar,
    // o que un árbol nulo al final no causa fugas si no fue asignado con 'new'.
    // Si NodePtr es un puntero crudo y se asigna con 'new' en clone_tree, entonces
    // delete_tree(overall_best_tree) sería necesario aquí.
    // Por ahora, se deja vacío, asumiendo manejo automático o externo.
}

void GeneticAlgorithm::evaluate_population(Island& island) {
    int pop_size = island.population.size();
    if (pop_size == 0) return;

    // 1. Simplify trees (CPU Parallel)
    // We do this first so we only send simplified trees to GPU
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pop_size; ++i) {
        Individual& ind = island.population[i];
        if (ind.tree) {
            ind.tree = DomainConstraints::fix_or_simplify(ind.tree);
        }
    }

#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    // 2. Prepare for Batch GPU Evaluation
    std::vector<LinearGpuNode> all_nodes;
    std::vector<int> tree_offsets;
    std::vector<int> tree_sizes;
    
    // Reserve memory to avoid reallocations (Optimization)
    // Assuming average tree size is around 20-30 nodes. 
    // This dramatically reduces CPU overhead during linearization.
    all_nodes.reserve(pop_size * 30); 
    tree_offsets.reserve(pop_size);
    tree_sizes.reserve(pop_size);
    
    // We need map back to original index because some trees might be null
    std::vector<int> valid_indices; 
    valid_indices.reserve(pop_size);

    for (int i = 0; i < pop_size; ++i) {
        if (island.population[i].tree) {
            int start_offset = all_nodes.size();
            linearize_tree(island.population[i].tree, all_nodes);
            int size = all_nodes.size() - start_offset;
            
            if (size > 0) {
                tree_offsets.push_back(start_offset);
                tree_sizes.push_back(size);
                valid_indices.push_back(i);
            } else {
                 island.population[i].fitness = INF;
                 island.population[i].fitness_valid = true;
            }
        } else {
             island.population[i].fitness = INF;
             island.population[i].fitness_valid = true;
        }
    }

    if (valid_indices.empty()) return;

    // 3. call GPU Batch (d_targets and d_x_values already exist)
    std::vector<double> raw_results(valid_indices.size());
    evaluate_population_gpu(all_nodes, tree_offsets, tree_sizes, targets, x_values, raw_results, d_targets, d_x_values,
                            island.d_nodes, island.d_nodes_capacity,
                            island.d_offsets, island.d_sizes, island.d_results, island.d_pop_capacity);

    // 4. Process results
    for (size_t k = 0; k < valid_indices.size(); ++k) {
        int idx = valid_indices[k];
        double sum_sq_error = raw_results[k];
        double raw_fitness = INF;

        // Check for validity
        if (!std::isnan(sum_sq_error) && !std::isinf(sum_sq_error) && sum_sq_error < 1e300) { // 1e300 as safety threshold
             if (USE_RMSE_FITNESS) {
                 if (x_values.size() > 0) {
                     double mse = sum_sq_error / x_values.size();
                     raw_fitness = sqrt(mse);
                 }
             } else {
                 raw_fitness = sum_sq_error;
             }
        }

        if (raw_fitness >= INF/2) {
             island.population[idx].fitness = INF;
        } else {
             // Complexity Penalty
             double complexity = static_cast<double>(tree_sizes[k]); // We already have the linear size
             double penalty = complexity * COMPLEXITY_PENALTY_FACTOR;
             island.population[idx].fitness = raw_fitness * (1.0 + penalty);
        }
        island.population[idx].fitness_valid = true;
    }

#else
    // CPU Fallback (Parallel)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pop_size; ++i) {
        Individual& ind = island.population[i];
        if (ind.tree) {
             ind.fitness = evaluate_fitness(ind.tree, targets, x_values);
             ind.fitness_valid = true;
        } else {
             ind.fitness = INF;
             ind.fitness_valid = true;
        }
    }
#endif
}


// ============================================================
// GLOBAL BATCH EVALUATION - Evaluates ALL islands in ONE GPU kernel call
// ============================================================
void GeneticAlgorithm::evaluate_all_islands() {
    int total_trees = 0;
    for (const auto& island : islands) {
        total_trees += island->population.size();
    }
    if (total_trees == 0) return;

    // Step 1: Simplify ALL trees in parallel (CPU)
    // Note: collapse(2) not supported by MSVC OpenMP 2.0, using nested parallel for
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
        for (int j = 0; j < static_cast<int>(islands[i]->population.size()); ++j) {
            Individual& ind = islands[i]->population[j];
            if (ind.tree) {
                ind.tree = DomainConstraints::fix_or_simplify(ind.tree);
            }
        }
    }

#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    // Runtime check: if FORCE_CPU_MODE is true or GPU init failed (d_targets == nullptr), use CPU
    if (!FORCE_CPU_MODE && d_targets != nullptr) {
    // Step 2: Linearize ALL trees from ALL islands into single buffer
    // OPTIMIZATION: Parallel linearization using OpenMP
    
    // First pass: count valid trees and compute per-tree sizes in parallel
    std::vector<int> tree_sizes_temp(total_trees, 0);
    std::vector<std::pair<int, int>> index_mapping(total_trees); // (island, individual)
    std::vector<bool> tree_valid(total_trees, false);
    
    int tree_idx = 0;
    for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
        for (int j = 0; j < static_cast<int>(islands[i]->population.size()); ++j) {
            index_mapping[tree_idx] = {i, j};
            tree_idx++;
        }
    }
    
    // Parallel linearization into per-thread buffers
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<LinearGpuNode>> thread_nodes(num_threads);
    std::vector<std::vector<int>> thread_offsets(num_threads);
    std::vector<std::vector<int>> thread_sizes(num_threads);
    std::vector<std::vector<std::pair<int, int>>> thread_mappings(num_threads);
    
    // Pre-allocate per-thread buffers
    int trees_per_thread = (total_trees + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        thread_nodes[t].reserve(trees_per_thread * 30);
        thread_offsets[t].reserve(trees_per_thread);
        thread_sizes[t].reserve(trees_per_thread);
        thread_mappings[t].reserve(trees_per_thread);
    }
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_nodes = thread_nodes[tid];
        auto& local_offsets = thread_offsets[tid];
        auto& local_sizes = thread_sizes[tid];
        auto& local_mappings = thread_mappings[tid];
        
        #pragma omp for schedule(static)
        for (int t = 0; t < total_trees; ++t) {
            int i = index_mapping[t].first;
            int j = index_mapping[t].second;
            Individual& ind = islands[i]->population[j];
            
            if (ind.tree) {
                int start_offset = local_nodes.size();
                linearize_tree(ind.tree, local_nodes);
                int size = local_nodes.size() - start_offset;
                
                if (size > 0) {
                    local_offsets.push_back(start_offset);
                    local_sizes.push_back(size);
                    local_mappings.push_back({i, j});
                } else {
                    ind.fitness = INF;
                    ind.fitness_valid = true;
                }
            } else {
                ind.fitness = INF;
                ind.fitness_valid = true;
            }
        }
    }
    
    // Merge thread-local buffers into global buffers
    std::vector<LinearGpuNode> all_nodes;
    std::vector<int> tree_offsets;
    std::vector<int> tree_sizes;
    std::vector<std::pair<int, int>> result_mapping;
    
    size_t total_node_count = 0;
    size_t total_valid_trees = 0;
    for (int t = 0; t < num_threads; ++t) {
        total_node_count += thread_nodes[t].size();
        total_valid_trees += thread_mappings[t].size();
    }
    
    all_nodes.reserve(total_node_count);
    tree_offsets.reserve(total_valid_trees);
    tree_sizes.reserve(total_valid_trees);
    result_mapping.reserve(total_valid_trees);
    
    for (int t = 0; t < num_threads; ++t) {
        int offset_adjustment = all_nodes.size();
        
        // Copy nodes
        all_nodes.insert(all_nodes.end(), thread_nodes[t].begin(), thread_nodes[t].end());
        
        // Adjust offsets and copy
        for (size_t k = 0; k < thread_offsets[t].size(); ++k) {
            tree_offsets.push_back(thread_offsets[t][k] + offset_adjustment);
            tree_sizes.push_back(thread_sizes[t][k]);
            result_mapping.push_back(thread_mappings[t][k]);
        }
    }
    
    std::vector<int> tree_complexities = tree_sizes; // Same as sizes for now

    if (result_mapping.empty()) return;

    int valid_trees = result_mapping.size();
    int num_points = x_values.size();
    int num_vars = (num_points > 0) ? x_values[0].size() : 0;
    
    // Step 3: Launch GPU evaluation ASYNC (no blocking!)
    // GPU will work while CPU continues with other tasks
    launch_evaluation_async(
        all_nodes, tree_offsets, tree_sizes,
        valid_trees, d_targets, d_x_values, num_points,
        num_vars,
        double_buffer_gpu
    );
    
    // Step 4: Wait for GPU results (this is where we sync)
    std::vector<double> results;
    retrieve_results_sync(results, valid_trees, double_buffer_gpu);

    // Step 5: Distribute results back to islands
    for (size_t k = 0; k < static_cast<size_t>(valid_trees); ++k) {
        int island_idx = result_mapping[k].first;
        int ind_idx = result_mapping[k].second;
        double fitness = results[k];
        
        // Validate result
        if (std::isnan(fitness) || std::isinf(fitness) || fitness >= 1e300) {
            fitness = INF;
        }
        
        islands[island_idx]->population[ind_idx].fitness = fitness;
        islands[island_idx]->population[ind_idx].fitness_valid = true;
    }
    
    } else {
        // FORCE_CPU_MODE is true OR GPU init failed: Use CPU
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
            for (int j = 0; j < static_cast<int>(islands[i]->population.size()); ++j) {
                Individual& ind = islands[i]->population[j];
                if (ind.tree) {
                    // Pass GPU pointers (even if null) to match signature
                    ind.fitness = evaluate_fitness(ind.tree, targets, x_values, d_targets, d_x_values);
                    ind.fitness_valid = true;
                } else {
                    ind.fitness = INF;
                    ind.fitness_valid = true;
                }
            }
        }
    }

#else
    // CPU Fallback: CUDA not available, use parallel CPU evaluation
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
        for (int j = 0; j < static_cast<int>(islands[i]->population.size()); ++j) {
            Individual& ind = islands[i]->population[j];
            if (ind.tree) {
                ind.fitness = evaluate_fitness(ind.tree, targets, x_values);
                ind.fitness_valid = true;
            } else {
                ind.fitness = INF;
                ind.fitness_valid = true;
            }
        }
    }
#endif
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
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
         auto local_search_result = try_local_improvement(island.population[best_idx].tree, island.population[best_idx].fitness, targets, x_values, LOCAL_SEARCH_ATTEMPTS, d_targets, d_x_values);
#else
         auto local_search_result = try_local_improvement(island.population[best_idx].tree, island.population[best_idx].fitness, targets, x_values, LOCAL_SEARCH_ATTEMPTS);
#endif
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
    
    // --- ISLAND CATACLYSM ---
    // If enabled, triggers a hard reset if stagnation persists.
    if (USE_ISLAND_CATACLYSM && island.stagnation_counter >= STAGNATION_LIMIT_ISLAND) {
        // Keep only top 1 elite (already in next_generation[0] if elite_count > 0)
        // Or if we need to enforce better elitism during cataclysm:
        
        int survivors = 1; // Only the absolute best one survives
        // Resize to survivors
        if (next_generation.size() > survivors) next_generation.resize(survivors);
        
        // Fill the rest with completely random trees
        int to_fill = current_pop_size - next_generation.size();
        for(int i=0; i<to_fill; ++i) {
             NodePtr random_tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL);
             if (random_tree) next_generation.emplace_back(std::move(random_tree));
        }
        
        island.stagnation_counter = 0; // Reset counter
        // Optional: Pattern injection could also happen here, but random is better for total diversity.
    }
    // Only do standard injections if we didn't just nuke everything
    else {
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
    }
    auto& rng = get_rng();
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    // >>> Parallel Parent Selection Loop with Uniqueness Check <<<
    
    // 1. Initialize uniqueness set with survivors (elites/injected)
    std::unordered_set<std::string> unique_signatures;
    if (PREVENT_DUPLICATES) {
        for (const auto& ind : next_generation) {
            if (ind.tree) {
                unique_signatures.insert(tree_to_string(ind.tree));
            }
        }
    }

    // 2. Fill the rest of the population
    int fail_safe_counter = 0;
    while (next_generation.size() < current_pop_size) {
        int needed = current_pop_size - next_generation.size();
        
        // Generate candidates in parallel
        std::vector<Individual> candidates(needed);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < needed; ++i) {
            // Thread-local RNG
            auto& rng = get_rng(); 
            
            Individual offspring;
            // Use distribution defined outside or create new one? 
            // Better create local to avoid shared state issues if not const
            std::uniform_real_distribution<double> local_prob_dist(0.0, 1.0);

            if (local_prob_dist(rng) < island.params.crossover_rate) {
                Individual p1, p2;
                if (USE_LEXICASE_SELECTION) {
                    p1 = lexicase_selection(island.population, targets, x_values);
                    p2 = lexicase_selection(island.population, targets, x_values);
                } else {
                    p1 = tournament_selection(island.population, island.params.tournament_size);
                    p2 = tournament_selection(island.population, island.params.tournament_size);
                }
                offspring = crossover(p1, p2);
            } else {
                Individual p1;
                if (USE_LEXICASE_SELECTION) {
                    p1 = lexicase_selection(island.population, targets, x_values);
                } else {
                    p1 = tournament_selection(island.population, island.params.tournament_size);
                }
                if (p1.tree) p1.tree = clone_tree(p1.tree); 
                mutate(p1, island.params.mutation_rate);
                offspring = std::move(p1);
            }
            
            candidates[i] = std::move(offspring);
        }
        
        // Filter and add unique candidates (Serial)
        int added_this_round = 0;
        for (auto& cand : candidates) {
            if (next_generation.size() >= current_pop_size) break;
            
            bool is_valid_to_add = true;
            if (PREVENT_DUPLICATES && cand.tree) {
                std::string sig = tree_to_string(cand.tree);
                if (unique_signatures.find(sig) != unique_signatures.end()) {
                    is_valid_to_add = false; 
                } else {
                    unique_signatures.insert(sig);
                }
            }
            
            if (is_valid_to_add) {
                next_generation.emplace_back(std::move(cand));
                added_this_round++;
            }
        }
        
        // Deadlock prevention
        if (added_this_round == 0) {
            fail_safe_counter++;
            if (fail_safe_counter > DUPLICATE_RETRIES) {
                // Fill remaining with random trees
                int remaining = current_pop_size - next_generation.size();
                for (int k = 0; k < remaining; ++k) {
                    NodePtr random_tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL);
                    if (random_tree) next_generation.emplace_back(std::move(random_tree));
                }
                break; // Exit loop
            }
        } else {
            fail_safe_counter = 0; // Reset if we made progress
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
        // [DEBUG] Trace execution
        if (gen == 0) { std::cout << "[DEBUG] Entering main loop, gen=0" << std::endl; std::cout.flush(); }
        
        // 1. Evaluate ALL islands in ONE GPU kernel call (maximum GPU utilization)
        evaluate_all_islands();

        // 2. Evolve Islands (Parallel Island Loop)
        // Genetic operators (crossover, mutation) are CPU-bound and independent per island.
        #pragma omp parallel for
        for (int i = 0; i < islands.size(); ++i) {
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
                  std::cout.flush(); // Ensure Formula: line is captured
                  std::cout << "Predictions vs Targets:" << std::endl;
                  std::cout << std::fixed << std::setprecision(4);
                  if (overall_best_tree && !x_values.empty()) {
                      for (size_t j = 0; j < x_values.size(); ++j) {
                          double val = evaluate_tree(overall_best_tree, x_values[j]);
                          double target_val = (j < targets.size()) ? targets[j] : std::nan("");
                          double diff = (!std::isnan(val) && !std::isnan(target_val)) ? std::fabs(val - target_val) : std::nan("");
                          std::cout << "  x=(";
                          for(size_t v=0; v<x_values[j].size(); ++v) std::cout << (v>0?",":"") << x_values[j][v];
                          std::cout << "): Pred=" << std::setw(12) << val
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
             // Re-evaluate after migration using global batch
             evaluate_all_islands();
        }

        if (overall_best_fitness < EXACT_SOLUTION_THRESHOLD) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Solution found meeting criteria at Generation " << gen + 1 << "!" << std::endl;
            std::cout << "Final Fitness: " << std::fixed << std::setprecision(8) << overall_best_fitness << std::endl;
            if(overall_best_tree) {
                 std::cout << "Final Formula Size: " << tree_size(overall_best_tree) << std::endl;
                 std::cout << "Final Formula: " << tree_to_string(overall_best_tree) << std::endl;
                 std::cout.flush(); // Ensure Final Formula: line is captured
            }
            std::cout << "========================================" << std::endl;
            std::cout.flush(); // Ensure flush
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
         std::cout << "Final Formula: " << tree_to_string(overall_best_tree) << std::endl;
         std::cout.flush(); // Ensure Final Formula: line is captured
          std::cout << "--- Final Verification ---" << std::endl;
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
          double final_check_fitness = evaluate_fitness(overall_best_tree, targets, x_values, d_targets, d_x_values);
#else
          double final_check_fitness = evaluate_fitness(overall_best_tree, targets, x_values);
#endif
          std::cout << "Recalculated Fitness: " << std::fixed << std::setprecision(8) << final_check_fitness << std::endl;
          std::cout << std::fixed << std::setprecision(4);
          for (size_t j = 0; j < x_values.size(); ++j) {
                double val = evaluate_tree(overall_best_tree, x_values[j]);
                 double target_val = (j < targets.size()) ? targets[j] : std::nan("");
                 double diff = (!std::isnan(val) && !std::isnan(target_val)) ? std::fabs(val - target_val) : std::nan("");
                 std::cout << "  x=(";
                 for(size_t v=0; v<x_values[j].size(); ++v) std::cout << (v>0?",":"") << x_values[j][v];
                 std::cout << "): Pred=" << std::setw(12) << val
                          << ", Target=" << std::setw(12) << target_val
                          << ", Diff=" << std::setw(12) << diff << std::endl;
           }
     } else { std::cout << "No valid solution found." << std::endl; }
      std::cout << "========================================" << std::endl;
    return overall_best_tree;
}
