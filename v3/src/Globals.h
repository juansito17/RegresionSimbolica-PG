#ifndef GLOBALS_H
#define GLOBALS_H

#include <vector>
#include <limits>
#include <random> // For std::mt19937

// --- Target Function Data ---
// Example: Target function y = x^2 + 2x + 1
const std::vector<double> X_VALUES = {8.0, 9.0, 10.0};
const std::vector<double> TARGETS = {92.0, 352.0, 724.0};

// --- Genetic Algorithm Parameters ---
const int TOTAL_POPULATION_SIZE = 1000;     // Total individuals across all islands
const int GENERATIONS = 500;                // Max number of generations
const int NUM_ISLANDS = 5;                  // Number of islands
const double MUTATION_RATE = 0.15;          // Base mutation rate
const double ELITE_PERCENTAGE = 0.10;       // Base elite percentage
const double CROSSOVER_RATE = 0.80;         // Base crossover rate
const int TOURNAMENT_SIZE = 15;             // Base tournament size
const int MIGRATION_INTERVAL = 50;          // Base migration interval
const int MIGRATION_SIZE = 5;              // Base migration size (Corrected from 30 based on later definition)
const int PARAM_ADAPT_INTERVAL = 20;        // How often island parameters adapt

// --- Fitness Evaluation ---
const double INF = std::numeric_limits<double>::infinity();
const double COMPLEXITY_PENALTY_FACTOR = 0.005; // Penalty per node in the tree
const double FITNESS_THRESHOLD = 1e-5;      // Target fitness to stop evolution early // <--- KEPT THIS DEFINITION

// --- Tree Constraints ---
const int MAX_TREE_DEPTH_INITIAL = 6;       // Max depth for initial random trees
const int MAX_TREE_DEPTH_EVOLVE = 10;       // Max depth during evolution (mutation/crossover)

// --- Evolution Parameters ---
const int STAGNATION_LIMIT = 20;           // Generations without improvement before considering stagnation
const int MAX_TREE_DEPTH_MUTATION = MAX_TREE_DEPTH_EVOLVE; // Use same depth as evolution

// --- CUDA Specific Constants ---
#ifdef USE_CUDA
const int MAX_FLAT_SIZE = 128;              // Max elements in flattened structure array (adjust based on MAX_TREE_DEPTH_EVOLVE)
const int MAX_CONSTANTS_PER_TREE = 32;      // Max constants allowed per tree (adjust as needed)
#endif

// --- Advanced Features ---
const double LOCAL_IMPROVEMENT_RATE = 0.0; // Probability of applying local search (Set to 0 as it was removed)
const int LOCAL_IMPROVEMENT_ATTEMPTS = 5;   // Attempts per local search

// --- Random Number Generation ---
// Provides access to a global random number generator instance
std::mt19937& get_rng();

#endif // GLOBALS_H