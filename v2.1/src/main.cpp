#include "Globals.h"
#include "GeneticAlgorithm.h"
#include "ExpressionTree.h" // For tree_to_string if needed
#include <iostream>
#include <vector>
#include <memory> // For shared_ptr
#include <iomanip> // For std::setprecision

bool USE_INTEGER_MODE = false; // Por defecto, modo decimal

int main() {
    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Symbolic Regression using Genetic Programming (Island Model)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Target Function Points:" << std::endl;
    for (size_t i = 0; i < TARGETS.size(); ++i) {
        std::cout << "  f(" << X_VALUES[i] << ") = " << TARGETS[i] << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Total Population: " << TOTAL_POPULATION_SIZE << std::endl;
    std::cout << "  Generations: " << GENERATIONS << std::endl;
    std::cout << "  Islands: " << NUM_ISLANDS << std::endl;
    std::cout << "  Migration Interval: " << MIGRATION_INTERVAL << std::endl;
    std::cout << "  Migration Size: " << MIGRATION_SIZE << std::endl;
    std::cout << "  Mutation Rate (Initial): " << MUTATION_RATE << std::endl;
    std::cout << "  Elite Percentage (Initial): " << ELITE_PERCENTAGE << std::endl;
    std::cout << "----------------------------------------" << std::endl;


    // Create the Genetic Algorithm instance
    GeneticAlgorithm ga(TARGETS, X_VALUES, TOTAL_POPULATION_SIZE, GENERATIONS, NUM_ISLANDS);

    // Run the algorithm
    NodePtr best_solution_tree = ga.run();

    // Final result summary is printed within ga.run()

    if (!best_solution_tree) {
        std::cerr << "\nFailed to find any valid solution." << std::endl;
        return 1;
    }

    return 0;
}