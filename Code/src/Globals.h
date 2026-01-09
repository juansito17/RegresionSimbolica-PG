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
// MODIFICADO: Usamos log(TARGETS) para aplanar el crecimiento exponencial.
// X representa N. TARGETS_LOG es ln(Q(N)).
// Se han filtrado valores N<4 donde Q(N) es 0 o pequeño irrelevante.

// MODIFICADO: RAW_TARGETS contiene los datos crudos. TARGETS se generará en runtime.
const std::vector<double> RAW_TARGETS = {2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884};
// MODIFICADO: X_VALUES ahora es vector<vector<double>> para soporte multivariable.
// Inicializador por defecto para problema univariable.
const std::vector<std::vector<double>> X_VALUES = {
    {1, 1, 1},   // 1
    {2, 2, 0},   // 2
    {3, 3, 1},   // 3
    {4, 4, 0},   // 4
    {5, 5, 1},   // 5
    {6, 0, 0},   // 6
    {7, 1, 1},   // 7
    {8, 2, 0},   // 8
    {9, 3, 1},   // 9
    {10, 4, 0},  // 10
    {11, 5, 1},  // 11
    {12, 0, 0},  // 12
    {13, 1, 1},  // 13
    {14, 2, 0},  // 14
    {15, 3, 1},  // 15
    {16, 4, 0},  // 16
    {17, 5, 1},  // 17
    {18, 0, 0},  // 18
    {19, 1, 1},  // 19
    {20, 2, 0},  // 20
    {21, 3, 1},  // 21
    {22, 4, 0},  // 22
    {23, 5, 1},  // 23
    {24, 0, 0},  // 24
    {25, 1, 1},  // 25
    {26, 2, 0}   // 26
};extern int NUM_VARIABLES; // Definido en Globals.cpp o main.cpp

// Flag para activar la transformación logarítmica automática
const bool USE_LOG_TRANSFORMATION = false;

// ----------------------------------------
// Configuración General del Algoritmo Genético
// ----------------------------------------
// Controla si se utiliza la aceleración por GPU.
// FORCE_CPU_MODE: Si es true, usa CPU aunque CUDA esté disponible (útil para comparar rendimiento)
const bool FORCE_CPU_MODE = true;  // Cambiar a 'true' para forzar modo CPU

// USE_GPU_ACCELERATION se define automáticamente por CMake si CUDA está disponible
// Pero si FORCE_CPU_MODE es true, se ignora y usa CPU
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
const bool USE_GPU_ACCELERATION = !FORCE_CPU_MODE;
#else
const bool USE_GPU_ACCELERATION = false;
#endif
// Aumentamos el tamaño de la población y el número de generaciones para maximizar la utilización de la GPU,
// ya que la GPU puede procesar un gran número de individuos en paralelo.
// Ajustamos el tamaño de la población para una GPU con 4GB de VRAM (RTX 3050),
// buscando un equilibrio entre el aprovechamiento de la GPU y el uso de memoria.
// Para hacer un uso aún más intensivo de la GPU y acelerar el algoritmo,
// OPTIMIZADO para Hybrid Search: Población más pequeña = convergencia más rápida en timeouts cortos
const int TOTAL_POPULATION_SIZE = 5000; // Reducido de 50000 para convergencia rápida
const int GENERATIONS = 50000;           // Reducido (timeout domina de todas formas)
const int NUM_ISLANDS = 5;               // Menos islas = más foco por isla
const int MIN_POP_PER_ISLAND = 10;        

// --- Fórmula Inicial ---
const bool USE_INITIAL_FORMULA = false; // Poner en 'true' para inyectar la fórmula
const std::string INITIAL_FORMULA_STRING = "log((x1+exp((((((1.28237193+((x0+2.59195138)+8.54688985))*x0)+(log((((x2/-0.99681346)-(x0-8.00219939))/(0.35461932-x2)))+(x0+(88.95319019/((x0+x0)+x0)))))-x1)/((exp(exp(((exp(x2)*(1.39925709/x0))^exp(x0))))+0.76703064)*6.05423753)))))";

// ----------------------------------------
// Parámetros del Modelo de Islas
// ----------------------------------------
// Aumentamos el intervalo y tamaño de migración para permitir que las islas realicen más trabajo en paralelo
// antes de intercambiar individuos, reduciendo la sobrecarga de comunicación y maximizando el procesamiento GPU.
const int MIGRATION_INTERVAL = 100; // Incrementado para permitir más trabajo por isla entre migraciones
const int MIGRATION_SIZE = 50;      // Incrementado para una migración más sustancial

// ----------------------------------------
// Parámetros de Generación Inicial de Árboles
// ----------------------------------------
const int MAX_TREE_DEPTH_INITIAL = 8; // Reducido para fórmulas iniciales más simples y rápidas
const double TERMINAL_VS_VARIABLE_PROB = 0.75;
const double CONSTANT_MIN_VALUE = -10.0;
const double CONSTANT_MAX_VALUE = 10.0;
const int CONSTANT_INT_MIN_VALUE = -10;
const int CONSTANT_INT_MAX_VALUE = 10;
const bool USE_HARD_DEPTH_LIMIT = true; // Toggle for hard depth limit
const int MAX_TREE_DEPTH_HARD_LIMIT = 12; // Hard limit to prevent bloat
// Order: +, -, *, /, ^, %, s, c, l, e, !, _, g
// ----------------------------------------
// Parámetros de Operadores Genéticos (Configuración de Operadores)
// ----------------------------------------
const bool USE_OP_PLUS     = true; // +
const bool USE_OP_MINUS    = true; // -
const bool USE_OP_MULT     = true; // *
const bool USE_OP_DIV      = true; // /
const bool USE_OP_POW      = true; // ^
const bool USE_OP_MOD      = false; // % (DISABLED)
const bool USE_OP_SIN      = false; // s (DISABLED)
const bool USE_OP_COS      = false; // c (DISABLED)
const bool USE_OP_LOG      = true; // l
const bool USE_OP_EXP      = true; // e
const bool USE_OP_FACT     = false; // ! (DISABLED - using lgamma instead)
const bool USE_OP_FLOOR    = false; // _ (DISABLED)
const bool USE_OP_GAMMA    = true; // g
const bool USE_OP_ASIN     = false; // S (DISABLED)
const bool USE_OP_ACOS     = false; // C (DISABLED)
const bool USE_OP_ATAN     = false; // T (DISABLED)

// Order: +, -, *, /, ^, %, s, c, l, e, !, _, g
// Los pesos se multiplican por el flag (0 o 1) para habilitar/deshabilitar.
const std::vector<double> OPERATOR_WEIGHTS = {
    0.20 * (USE_OP_PLUS  ? 1.0 : 0.0), // +
    0.20 * (USE_OP_MINUS ? 1.0 : 0.0), // -
    0.20 * (USE_OP_MULT  ? 1.0 : 0.0), // *
    0.15 * (USE_OP_DIV   ? 1.0 : 0.0), // /
    0.10 * (USE_OP_POW   ? 1.0 : 0.0), // ^
    0.02 * (USE_OP_MOD   ? 1.0 : 0.0), // %
    0.10 * (USE_OP_SIN   ? 1.0 : 0.0), // s
    0.10 * (USE_OP_COS   ? 1.0 : 0.0), // c
    0.05 * (USE_OP_LOG   ? 1.0 : 0.0), // l
    0.05 * (USE_OP_EXP   ? 1.0 : 0.0), // e
    0.01 * (USE_OP_FACT  ? 1.0 : 0.0), // !
    0.01 * (USE_OP_FLOOR ? 1.0 : 0.0), // _
    0.01 * (USE_OP_GAMMA ? 1.0 : 0.0), // g
    0.01 * (USE_OP_ASIN  ? 1.0 : 0.0), // S
    0.01 * (USE_OP_ACOS  ? 1.0 : 0.0), // C
    0.01 * (USE_OP_ATAN  ? 1.0 : 0.0)  // T
};

// ----------------------------------------
// Parámetros de Operadores Genéticos (Mutación, Cruce, Selección)
// ----------------------------------------
const double BASE_MUTATION_RATE = 0.30;
const double BASE_ELITE_PERCENTAGE = 0.15;
const double DEFAULT_CROSSOVER_RATE = 0.85;
const int DEFAULT_TOURNAMENT_SIZE = 30;
const int MAX_TREE_DEPTH_MUTATION = 8; // Slight increase to allow complexity
const double MUTATE_INSERT_CONST_PROB = 0.6;
const int MUTATE_INSERT_CONST_INT_MIN = 1;
const int MUTATE_INSERT_CONST_INT_MAX = 5;
const double MUTATE_INSERT_CONST_FLOAT_MIN = 0.5;
const double MUTATE_INSERT_CONST_FLOAT_MAX = 5.0;

// ----------------------------------------
// Parámetros de Fitness y Evaluación
// ----------------------------------------
// Reducimos ligeramente la penalización por complejidad para permitir que fórmulas más complejas
// (y computacionalmente más intensivas para la GPU) sean favorecidas por el algoritmo.
// MODIFICADO: Ajustado para ser menos agresivo y permitir multivariable.
const double COMPLEXITY_PENALTY_FACTOR = 0.01; // Was 0.05. Reduced to 0.01.
const bool USE_RMSE_FITNESS = true;
const double FITNESS_ORIGINAL_POWER = 1.3;
const double FITNESS_PRECISION_THRESHOLD = 0.001;
const double FITNESS_PRECISION_BONUS = 0.0001;
const double FITNESS_EQUALITY_TOLERANCE = 1e-9;
const double EXACT_SOLUTION_THRESHOLD = 1e-8;

// ----------------------------------------
// Fitness Ponderado (Weighted Fitness)
// ----------------------------------------
// Activa el fitness ponderado para penalizar fuertemente errores en valores altos de N.
// Esto destruye a las parábolas que fallan en N=20 pero dan buen promedio general.
const bool USE_WEIGHTED_FITNESS = false;
// Tipo de peso: "quadratic" usa i*i, "exponential" usa exp(i*WEIGHTED_FITNESS_EXPONENT)
// Exponente para peso exponencial (más agresivo). Usar 0.2-0.3 para datasets pequeños.
const double WEIGHTED_FITNESS_EXPONENT = 0.25;

// ----------------------------------------
// Parámetros de Características Avanzadas
// ----------------------------------------
const int STAGNATION_LIMIT_ISLAND = 50;
// Lowered from 5000 to allow faster early termination in Hybrid Search mode.
// If best fitness doesn't improve for N generations, terminate early.
const int GLOBAL_STAGNATION_LIMIT = 100; // Reducido para terminar más rápido si no mejora
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
// Simplification Toggle
const bool USE_SIMPLIFICATION = true;
// Anti-Stagnation: Island Cataclysm (Hard Reset)
const bool USE_ISLAND_CATACLYSM = true;
// Selection Strategy: Epsilon-Lexicase Selection (Replaces Tournament)
const bool USE_LEXICASE_SELECTION = true;

// ----------------------------------------
// Otros Parámetros
// ----------------------------------------
const int PROGRESS_REPORT_INTERVAL = 100;
// Optimizaciones adicionales:
// Deshabilitamos las constantes enteras forzadas para permitir una mayor flexibilidad
// en las constantes generadas y mutadas, lo que podría conducir a mejores soluciones
// y mantener la GPU ocupada con un rango más amplio de valores.
const bool FORCE_INTEGER_CONSTANTS = false; // Mantenemos false para mayor flexibilidad

// ----------------------------------------
// Control de Duplicados
// ----------------------------------------
const bool PREVENT_DUPLICATES = true; // Activa la verificación de unicidad
const int DUPLICATE_RETRIES = 10;     // Intentos para generar un individuo único antes de rendirse


// ============================================================
//                  UTILIDADES GLOBALES
// ============================================================
std::mt19937& get_rng();
const double INF = std::numeric_limits<double>::infinity();

#endif // GLOBALS_H
