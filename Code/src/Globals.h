#ifndef GLOBALS_H
#define GLOBALS_H

#include <vector>
#include <random>
#include <string>
#include <limits>
#include <cmath>

// ============================================================
//                  PARÁMETROS GLOBALES
// ============================================================

// ----------------------------------------
// Datos del Problema (Regresión Simbólica)
// ----------------------------------------
// Asegúrate de que estos sean los datos correctos para la fórmula que quieres probar

const std::vector<double> TARGETS = {92, 352, 724}; // O los que correspondan
const std::vector<double> X_VALUES = {8, 9, 10};    // O los que correspondan

//const std::vector<double> TARGETS = {380, 336, 324, 308, 301, 313, 271, 268, 251, 231};
//const std::vector<double> X_VALUES = {76.5, 67.9, 67.7, 62, 60.9, 60.5, 55.8, 51.7, 50.6, 46.4};

//const std::vector<double> TARGETS = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528};
//const std::vector<double> X_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

// ----------------------------------------
// Configuración General del Algoritmo Genético
// ----------------------------------------
const bool USE_GPU_ACCELERATION = true; // Set to true to enable GPU acceleration
const int TOTAL_POPULATION_SIZE = 1000000;
const int GENERATIONS = 100000;
const int NUM_ISLANDS = 7;
const int MIN_POP_PER_ISLAND = 20;

// --- Fórmula Inicial ---
const bool USE_INITIAL_FORMULA = false; // Poner en 'true' para inyectar la fórmula
const std::string INITIAL_FORMULA_STRING = "(((0.62358531/(((x+(0.62358531/(((-(3599.19050242/(((52.38107258-x)-0.87618851)-6.34160259)))+1.86961523)-((x-1.74732397)*((x-0.49109558)+(67.17975136/(60.76860957-x)))))))-0.49109558)-(((x+0.00294194)-0.31272581)+((5.00877511/(-60.04757248+((x+0.00398195)+0.00398195)))/(67.17975136-(x-0.49109558))))))/(67.17975136-(x+(0.33459658/(67.17975136-(x+(0.33459658/(67.17975136-((((x+0.00263118)+0.00263118)-0.47205995)+0.62358531)))))))))+((0.33459658/(67.17975136-(x-0.49109558)))+((2.68328288/(49.94971888-((x-0.00114565)+(0.33459658/(67.17975136-((x-0.00134655)-0.47205995))))))+((0.74346506/(60.76860957-x))+(((5.43909022/(-60.04757248+x))-3.3156463)+((10.02892889/(52.38107258-x))+(5.00877511*((x+(0.62358531/((385.84981692/(60.76860957-x))-(((x+0.62358531)+(0.33459658/(67.17975136-(x+(0.33459658/(67.17975136-(((x+(0.62358531/((((5.36601249*x)+5.36601249)/(60.76860957-x))-60.76860957)))-0.47205995)+0.62358531)))))))*x))))+(0.62358531/((72.88834001/(52.38107258-(x+0.00294194)))-((((5.74354339*(x+1.21141146))+x)/(67.17975136-((((x+0.00398195)+0.00398195)+0.00263118)-0.31272581)))+(x+((52.38107258/(60.76860957-(x+(0.62358531/((((5.36601249*x)+5.36601249)/(60.76860957-x))-60.76860957)))))+((-9.16958904+(((10.02892889*(52.38107258-((x+0.00263118)+0.00263118)))+(5.00877511*x))+3.25767525))-x))))))))))))))";
// ---------------------------------------------------------

// ----------------------------------------
// Parámetros del Modelo de Islas
// ----------------------------------------
const int MIGRATION_INTERVAL = 50;
const int MIGRATION_SIZE = 30;

// ----------------------------------------
// Parámetros de Generación Inicial de Árboles
// ----------------------------------------
const int MAX_TREE_DEPTH_INITIAL = 7;
const double TERMINAL_VS_VARIABLE_PROB = 0.75;
const double CONSTANT_MIN_VALUE = -10.0;
const double CONSTANT_MAX_VALUE = 10.0;
const int CONSTANT_INT_MIN_VALUE = -10;
const int CONSTANT_INT_MAX_VALUE = 10;
const std::vector<double> OPERATOR_WEIGHTS = {0.3, 0.3, 0.25, 0.1, 0.05};

// ----------------------------------------
// Parámetros de Operadores Genéticos (Mutación, Cruce, Selección)
// ----------------------------------------
const double BASE_MUTATION_RATE = 0.20;
const double BASE_ELITE_PERCENTAGE = 0.10;
const double DEFAULT_CROSSOVER_RATE = 0.8;
const int DEFAULT_TOURNAMENT_SIZE = 25;
const int MAX_TREE_DEPTH_MUTATION = 5;
const double MUTATE_INSERT_CONST_PROB = 0.6;
const int MUTATE_INSERT_CONST_INT_MIN = 1;
const int MUTATE_INSERT_CONST_INT_MAX = 5;
const double MUTATE_INSERT_CONST_FLOAT_MIN = 0.5;
const double MUTATE_INSERT_CONST_FLOAT_MAX = 5.0;

// ----------------------------------------
// Parámetros de Fitness y Evaluación
// ----------------------------------------
const double COMPLEXITY_PENALTY_FACTOR = 0.01;
const bool USE_RMSE_FITNESS = true;
const double FITNESS_ORIGINAL_POWER = 1.3;
const double FITNESS_PRECISION_THRESHOLD = 0.001;
const double FITNESS_PRECISION_BONUS = 0.0001;
const double FITNESS_EQUALITY_TOLERANCE = 1e-9;
const double EXACT_SOLUTION_THRESHOLD = 1e-8;

// ----------------------------------------
// Parámetros de Características Avanzadas
// ----------------------------------------
const int STAGNATION_LIMIT_ISLAND = 50;
const int GLOBAL_STAGNATION_LIMIT = 5000;
const double STAGNATION_RANDOM_INJECT_PERCENT = 0.1;
const int PARAM_MUTATE_INTERVAL = 50;
const double PATTERN_RECORD_FITNESS_THRESHOLD = 10.0;
const int PATTERN_MEM_MIN_USES = 3;
const int PATTERN_INJECT_INTERVAL = 10;
const double PATTERN_INJECT_PERCENT = 0.05;
const size_t PARETO_MAX_FRONT_SIZE = 50;
const double SIMPLIFY_NEAR_ZERO_TOLERANCE = 1e-9;
const double SIMPLIFY_NEAR_ONE_TOLERANCE = 1e-9;
const int LOCAL_SEARCH_ATTEMPTS = 30;

// ----------------------------------------
// Otros Parámetros
// ----------------------------------------
const int PROGRESS_REPORT_INTERVAL = 100;
const bool FORCE_INTEGER_CONSTANTS = true;

// ============================================================
//                  UTILIDADES GLOBALES
// ============================================================
std::mt19937& get_rng();
const double INF = std::numeric_limits<double>::infinity();

#endif // GLOBALS_H
