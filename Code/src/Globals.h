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
//const std::vector<double> TARGETS = {92, 352, 724};
//const std::vector<double> X_VALUES = {8, 9, 10};
const std::vector<double> TARGETS = {380, 336, 324, 308, 301, 313, 271, 268, 251, 231};
const std::vector<double> X_VALUES = {76.5, 67.9, 67.7, 62, 60.9, 60.5, 55.8, 51.7, 50.6, 46.4};

//const std::vector<double> TARGETS = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528};
//const std::vector<double> X_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

// ----------------------------------------
// Configuración General del Algoritmo Genético
// ----------------------------------------
const int TOTAL_POPULATION_SIZE = 50000; // Tamaño total de la población
const int GENERATIONS = 100000;          // Número máximo de generaciones
const int NUM_ISLANDS = 7;               // Número de islas
const int MIN_POP_PER_ISLAND = 20;       // Población mínima requerida por isla

// ----------------------------------------
// Parámetros del Modelo de Islas
// ----------------------------------------
const int MIGRATION_INTERVAL = 50; // Generaciones entre migraciones
const int MIGRATION_SIZE = 30;     // Número de individuos que migran

// ----------------------------------------
// Parámetros de Generación Inicial de Árboles
// ----------------------------------------
const int MAX_TREE_DEPTH_INITIAL = 7;           // Profundidad máxima inicial
const double TERMINAL_VS_VARIABLE_PROB = 0.75;  // Probabilidad de que un nodo terminal sea 'x'
const double CONSTANT_MIN_VALUE = -10.0;        // Valor mínimo para constantes aleatorias (flotantes)
const double CONSTANT_MAX_VALUE = 10.0;         // Valor máximo para constantes aleatorias (flotantes)
const int CONSTANT_INT_MIN_VALUE = -10;         // Valor mínimo para constantes aleatorias (enteras)
const int CONSTANT_INT_MAX_VALUE = 10;          // Valor máximo para constantes aleatorias (enteras)
const std::vector<double> OPERATOR_WEIGHTS = {0.3, 0.3, 0.25, 0.1, 0.05}; // Pesos para +, -, *, /, ^
// Ya no hay POWER_EXPONENT_MIN/MAX porque se quitaron las restricciones

// ----------------------------------------
// Parámetros de Operadores Genéticos (Mutación, Cruce, Selección)
// ----------------------------------------
const double BASE_MUTATION_RATE = 0.20;         // Tasa de mutación base
const double BASE_ELITE_PERCENTAGE = 0.10;      // Porcentaje de élite base
const double DEFAULT_CROSSOVER_RATE = 0.8;      // Tasa de cruce por defecto
const int DEFAULT_TOURNAMENT_SIZE = 25;         // Tamaño de torneo por defecto (más presión)
const int MAX_TREE_DEPTH_MUTATION = 5;          // Profundidad máx. para subárboles de mutación
const double MUTATE_INSERT_CONST_PROB = 0.6;    // Prob. de insertar constante en NodeInsertion
const int MUTATE_INSERT_CONST_INT_MIN = 1;      // Rango para constante entera insertada
const int MUTATE_INSERT_CONST_INT_MAX = 5;
const double MUTATE_INSERT_CONST_FLOAT_MIN = 0.5;// Rango para constante flotante insertada
const double MUTATE_INSERT_CONST_FLOAT_MAX = 5.0;

// ----------------------------------------
// Parámetros de Fitness y Evaluación
// ----------------------------------------
// Reducir mucho la penalización para priorizar fitness=0
const double COMPLEXITY_PENALTY_FACTOR = 0.01; // <-- MUY BAJO (era 1.0)
// Usar RMSE para un gradiente más claro hacia cero
const bool USE_RMSE_FITNESS = true;             // <-- CAMBIADO a true (era false)
const double FITNESS_ORIGINAL_POWER = 1.3;      // Exponente si USE_RMSE_FITNESS = false
const double FITNESS_PRECISION_THRESHOLD = 0.001; // Umbral para bonus
const double FITNESS_PRECISION_BONUS = 0.0001;    // Factor de bonus
const double FITNESS_EQUALITY_TOLERANCE = 1e-9;   // Tolerancia para empate en torneo/mejora
const double EXACT_SOLUTION_THRESHOLD = 1e-6;     // Umbral para considerar fitness "perfecto"

// ----------------------------------------
// Parámetros de Características Avanzadas
// ----------------------------------------
// --- Estancamiento ---
// Aumentar límites para dar más tiempo
const int STAGNATION_LIMIT_ISLAND = 50;         // <-- AUMENTADO (era 30)
const int GLOBAL_STAGNATION_LIMIT = 5000;       // <-- AUMENTADO (era 2000)
const double STAGNATION_RANDOM_INJECT_PERCENT = 0.1; // % inyección aleatoria
// --- Adaptación de Parámetros ---
const int PARAM_MUTATE_INTERVAL = 50;           // Frecuencia adaptación
// --- Memoria de Patrones ---
const double PATTERN_RECORD_FITNESS_THRESHOLD = 10.0; // Umbral registro
const int PATTERN_MEM_MIN_USES = 3;             // Usos mínimos para sugerir
const int PATTERN_INJECT_INTERVAL = 10;         // Frecuencia inyección patrón
const double PATTERN_INJECT_PERCENT = 0.05;     // % inyección patrón
// --- Optimización Pareto ---
const size_t PARETO_MAX_FRONT_SIZE = 50;        // Tamaño máx frente Pareto
// --- Simplificación y Restricciones ---
const double SIMPLIFY_NEAR_ZERO_TOLERANCE = 1e-9; // Tolerancia cero
const double SIMPLIFY_NEAR_ONE_TOLERANCE = 1e-9;  // Tolerancia uno
// Ya no hay SIMPLIFY_EXPONENT_CLAMP porque se quitaron restricciones
// --- Búsqueda Local ---
// Aumentar intentos para refinar más
const int LOCAL_SEARCH_ATTEMPTS = 30;           // <-- AUMENTADO (era 10)

// ----------------------------------------
// Otros Parámetros
// ----------------------------------------
const int PROGRESS_REPORT_INTERVAL = 100;       // Frecuencia informe progreso
const bool FORCE_INTEGER_CONSTANTS = true;     // Forzar constantes enteras

// ============================================================
//                  UTILIDADES GLOBALES
// ============================================================
std::mt19937& get_rng();
const double INF = std::numeric_limits<double>::infinity();

#endif // GLOBALS_H
