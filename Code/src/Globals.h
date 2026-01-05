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
const std::vector<double> X_VALUES = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

// Flag para activar la transformación logarítmica automática
const bool USE_LOG_TRANSFORMATION = true;

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
// aumentamos el número de islas para fomentar más paralelismo, manteniendo la población total.
// Esto distribuye la carga de trabajo de evaluación de fitness en más unidades de procesamiento concurrentes.
const int TOTAL_POPULATION_SIZE = 50000; // Mantenemos este tamaño, ajustado para 4GB VRAM
const int GENERATIONS = 500000;           // Mantenemos las generaciones altas
const int NUM_ISLANDS = 10;               // Aumentado para mayor paralelismo
const int MIN_POP_PER_ISLAND = 10;        // Ajustado para permitir más islas con población mínima

// --- Fórmula Inicial ---
const bool USE_INITIAL_FORMULA = true; // Poner en 'true' para inyectar la fórmula
const std::string INITIAL_FORMULA_STRING = "(g(x)-(x*0.912079)+0.146743+(3.78968/x))";

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
const bool USE_OP_MOD      = true; // %
const bool USE_OP_SIN      = true; // s
const bool USE_OP_COS      = true; // c
const bool USE_OP_LOG      = true; // l
const bool USE_OP_EXP      = true; // e
const bool USE_OP_FACT     = true; // !
const bool USE_OP_FLOOR    = true; // _
const bool USE_OP_GAMMA    = true; // g

// Order: +, -, *, /, ^, %, s, c, l, e, !, _, g
// Los pesos se multiplican por el flag (0 o 1) para habilitar/deshabilitar.
const std::vector<double> OPERATOR_WEIGHTS = {
    0.10 * (USE_OP_PLUS  ? 1.0 : 0.0), // +
    0.15 * (USE_OP_MINUS ? 1.0 : 0.0), // -
    0.10 * (USE_OP_MULT  ? 1.0 : 0.0), // *
    0.10 * (USE_OP_DIV   ? 1.0 : 0.0), // /
    0.05 * (USE_OP_POW   ? 1.0 : 0.0), // ^
    0.01 * (USE_OP_MOD   ? 1.0 : 0.0), // %
    0.01 * (USE_OP_SIN   ? 1.0 : 0.0), // s
    0.01 * (USE_OP_COS   ? 1.0 : 0.0), // c
    0.15 * (USE_OP_LOG   ? 1.0 : 0.0), // l
    0.02 * (USE_OP_EXP   ? 1.0 : 0.0), // e
    0.05 * (USE_OP_FACT  ? 1.0 : 0.0), // !
    0.05 * (USE_OP_FLOOR ? 1.0 : 0.0), // _
    0.20 * (USE_OP_GAMMA ? 1.0 : 0.0)  // g
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
// MODIFICADO: Aumentado para penalizar bloat (Strategy 3).
const double COMPLEXITY_PENALTY_FACTOR = 0.05; // Was 0.005. Increased significantly to fight bloat.
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
const bool USE_WEIGHTED_FITNESS = true;
// Tipo de peso: "quadratic" usa i*i, "exponential" usa exp(i*WEIGHTED_FITNESS_EXPONENT)
// Exponente para peso exponencial (más agresivo). Usar 0.2-0.3 para datasets pequeños.
const double WEIGHTED_FITNESS_EXPONENT = 0.25;

// ----------------------------------------
// Parámetros de Características Avanzadas
// ----------------------------------------
const int STAGNATION_LIMIT_ISLAND = 50;
// Lowered from 5000 to allow faster early termination in Hybrid Search mode.
// If best fitness doesn't improve for N generations, terminate early.
const int GLOBAL_STAGNATION_LIMIT = 200;
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
