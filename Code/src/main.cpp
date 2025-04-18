#include "Globals.h" // Necesario para las constantes globales
#include "GeneticAlgorithm.h"
#include "ExpressionTree.h" // Para tree_to_string si se necesita aquí
#include <iostream>
#include <vector>
#include <memory> // Para shared_ptr
#include <iomanip> // Para std::setprecision

int main() {
    // Configurar precisión de salida para números flotantes
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Symbolic Regression using Genetic Programming (Island Model)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Target Function Points:" << std::endl;
    // Imprimir los puntos objetivo definidos en Globals.h
    for (size_t i = 0; i < TARGETS.size(); ++i) {
        std::cout << "  f(" << X_VALUES[i] << ") = " << TARGETS[i] << std::endl;
    }
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
    GeneticAlgorithm ga(TARGETS, X_VALUES, TOTAL_POPULATION_SIZE, GENERATIONS, NUM_ISLANDS);

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
