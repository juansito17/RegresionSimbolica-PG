#ifndef GLOBALS_H
#define GLOBALS_H

#include <vector>
#include <random>
#include <string>
#include <limits>

// ================================
// Global Parameters
// ================================
const std::vector<double> TARGETS = {92, 352, 724};
const std::vector<double> X_VALUES = {8, 9, 10};

//const std::vector<double> TARGETS = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528};
//const std::vector<double> X_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

//const std::vector<double> TARGETS = {380, 336, 324, 308, 301, 313, 271, 268, 251, 231};
//const std::vector<double> X_VALUES = {76.5, 67.9, 67.7, 62, 60.9, 60.5, 55.8, 51.7, 50.6, 46.4};

const int TOTAL_POPULATION_SIZE = 50000;
const int GENERATIONS = 100000;

const double MUTATION_RATE = 0.20; // Base mutation rate
const int STAGNATION_LIMIT = 20;
const double ELITE_PERCENTAGE = 0.10; // Base elite percentage

const int NUM_ISLANDS = 7;
const int MIGRATION_INTERVAL = 50;
const int MIGRATION_SIZE = 30;

// Maximum depth for newly generated trees (mutation/initial)
const int MAX_TREE_DEPTH_INITIAL = 7; // Increased from 5 to 7
const int MAX_TREE_DEPTH_MUTATION = 7; // Increased from 5 to 7

// Penalty factor for complexity in fitness
extern double COMPLEXITY_PENALTY_FACTOR; // (Was 1/25.0)

// Control integer/decimal mode
extern bool USE_INTEGER_MODE; // Si es true, solo se usan enteros en las fórmulas

// ================================
// Global RNG
// ================================
// Use a function to access the RNG to avoid static initialization order issues
// if used across different translation units directly.
std::mt19937& get_rng();

// Utility constant
const double INF = std::numeric_limits<double>::infinity();

#endif // GLOBALS_H