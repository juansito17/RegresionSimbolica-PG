// ============================================================
// Archivo: src/GeneticAlgorithm.h
// ============================================================
#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include "ExpressionTree.h"
#include "GeneticOperators.h"
#include "AdvancedFeatures.h"
#include "Globals.h" // Incluir Globals.h para INF, NUM_ISLANDS, etc.
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
#include "FitnessGPU.cuh" // For GlobalGpuBuffers definition
#endif
#include <vector>
#include <string>
#include <memory> // Para std::unique_ptr

class GeneticAlgorithm {
    // Estructura interna para representar una isla
    struct Island {
        std::vector<Individual> population; // Población de la isla
        EvolutionParameters params;         // Parámetros evolutivos propios de la isla
        PatternMemory pattern_memory;       // Memoria de patrones de la isla
        ParetoOptimizer pareto_optimizer;   // Optimizador Pareto de la isla
        int stagnation_counter = 0;         // Contador de estancamiento local de la isla
        double best_fitness = INF;          // Mejor fitness histórico de la isla
        std::vector<double> fitness_history;// Historial de fitness (opcional)
        int id;                             // Identificador de la isla

        // Constructor de la isla
        explicit Island(int island_id, int pop_size) : id(island_id) {
             population = create_initial_population(pop_size); // Crear población inicial
             params = EvolutionParameters::create_default();   // Usar parámetros por defecto
        }

#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
        // Persistent GPU buffers
        void* d_nodes = nullptr;
        void* d_offsets = nullptr;
        void* d_sizes = nullptr;
        void* d_results = nullptr;
        size_t d_nodes_capacity = 0;
        size_t d_pop_capacity = 0;

        ~Island() {
            // We cannot easily call cudaFree here because this header might be included
            // where cuda_runtime.h is not. However, we can trust the OS/driver to clean up
            // or we should add a cleanup function.
            // For now, we will rely on GeneticAlgorithm destructor or explict cleanup if possible.
            // But since Island is unique_ptr, we can't easily add a destructor that calls cudaFree 
            // without including cuda_runtime.
            // Optimization: Let's rely on OS cleanup at exit, OR add a cleanup method called by GA.
        }
#endif
    };

    // Miembros principales de la clase GeneticAlgorithm
    std::vector<std::unique_ptr<Island>> islands; // Vector de punteros únicos a las islas
    const std::vector<double>& targets;           // Referencia a los datos objetivo
    const std::vector<double>& x_values;          // Referencia a los valores de x
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    double* d_targets = nullptr;                  // Puntero a los datos objetivo en la GPU
    double* d_x_values = nullptr;                 // Puntero a los valores de x en la GPU
    GlobalGpuBuffers global_gpu_buffers;          // Global buffers for batch evaluation of ALL islands
    DoubleBufferedGpu double_buffer_gpu;          // Double-buffered GPU for async overlap
#endif
    int total_population_size;                    // Tamaño total de la población
    int generations;                              // Número máximo de generaciones
    int num_islands;                              // Número de islas

    // Seguimiento del mejor global
    NodePtr overall_best_tree = nullptr;          // Mejor árbol encontrado globalmente
    double overall_best_fitness = INF;            // Mejor fitness encontrado globalmente

    // --- NUEVO: Seguimiento de Estancamiento Global ---
    int generation_last_improvement = 0;          // Generación en la que mejoró el overall_best_fitness
    double last_overall_best_fitness = INF;       // Valor del overall_best_fitness en la última mejora
    // -------------------------------------------------

    int pop_per_island;                           // Población calculada por isla

public:
    // Constructor
    GeneticAlgorithm(const std::vector<double>& targets_ref,
                       const std::vector<double>& x_values_ref,
                       int total_pop,
                       int gens,
                       int n_islands = NUM_ISLANDS); // Usar valor de Globals.h por defecto
    ~GeneticAlgorithm(); // Destructor para liberar memoria de la GPU

    // Ejecuta el algoritmo genético
    NodePtr run();

private:
    // Funciones auxiliares internas
    void evaluate_population(Island& island); // Evalúa fitness de una isla (legacy)
    void evaluate_all_islands(); // Evalúa ALL islands in ONE GPU batch call (optimized)
    void evolve_island(Island& island, int current_generation); // Evoluciona una isla por una generación
    void migrate(); // Realiza la migración entre islas
    void update_overall_best(const Island& island); // Actualiza el mejor global
};


#endif // GENETICALGORITHM_H
