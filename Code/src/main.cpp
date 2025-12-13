#include "Globals.h" // Necesario para las constantes globales
#include "GeneticAlgorithm.h"
#include "Fitness.h" // Para evaluate_fitness
#include "ExpressionTree.h" // Para tree_to_string si se necesita aquí
#include <iostream>
#include <vector>
#include <memory> // Para shared_ptr
#include <iomanip> // Para std::setprecision
#include <omp.h>   // Para configuración de OpenMP

int main() {
    // === OPTIMIZACIÓN: Configuración explícita de hilos OpenMP ===
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "[OpenMP] Using " << num_threads << " threads" << std::endl;
    
    // Configurar precisión de salida para números flotantes
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Symbolic Regression using Genetic Programming (Island Model)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::vector<double> targets;
    std::vector<double> final_x_values;

    if (USE_LOG_TRANSFORMATION) {
        std::cout << "Info: Log Transformation is ON (Target = ln(Q(N)))." << std::endl;
        for (size_t i = 0; i < RAW_TARGETS.size(); ++i) {
            if (RAW_TARGETS[i] > 0) {
                targets.push_back(std::log(RAW_TARGETS[i]));
                final_x_values.push_back(X_VALUES[i]);
            }
        }
    } else {
        std::cout << "Info: Log Transformation is OFF." << std::endl;
        targets = RAW_TARGETS;
        final_x_values = X_VALUES;
    }

    std::cout << "Target Function Points (Effective):" << std::endl;
    // Imprimir los puntos objetivo
    for (size_t i = 0; i < targets.size(); ++i) {
        std::cout << "  f(" << final_x_values[i] << ") = " << targets[i] << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    std::cout << "Info: Running with GPU acceleration." << std::endl;
#else
    std::cout << "Info: Running with CPU acceleration." << std::endl;
#endif
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Parameters:" << std::endl;
    // Imprimir los parámetros globales definidos en Globals.h
    std::cout << "  Total Population: " << TOTAL_POPULATION_SIZE << std::endl;
    std::cout << "  Generations: " << GENERATIONS << std::endl;
    std::cout << "  Islands: " << NUM_ISLANDS << std::endl;
    std::cout << "  Migration Interval: " << MIGRATION_INTERVAL << std::endl;
    std::cout << "  Migration Size: " << MIGRATION_SIZE << std::endl;
    // --- NOMBRES CORREGIDOS ---
    std::cout << "  Mutation Rate (Initial): " << BASE_MUTATION_RATE << std::endl; // <-- Nombre corregido
    std::cout << "  Elite Percentage (Initial): " << BASE_ELITE_PERCENTAGE << std::endl; // <-- Nombre corregido
    // --------------------------
    std::cout << "----------------------------------------" << std::endl;


    // Crear la instancia del Algoritmo Genético
    // Pasa las referencias a los vectores de datos y los parámetros principales
    GeneticAlgorithm ga(targets, final_x_values, TOTAL_POPULATION_SIZE, GENERATIONS, NUM_ISLANDS);

    // Ejecutar el algoritmo
    // La función run() contiene el bucle principal de generaciones y devuelve el mejor árbol encontrado
    NodePtr best_solution_tree = ga.run();

    // La función run() ya imprime el resumen final y la verificación.
    // Comprobar si se encontró alguna solución válida al final
    if (!best_solution_tree) {
        std::cerr << "\nFailed to find any valid solution." << std::endl;
        return 1; // Salir con código de error si no se encontró solución
    }

    return 0; // Salir con éxito
}
