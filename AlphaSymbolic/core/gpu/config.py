import math

class GpuGlobals:
    # ============================================================
    #                  PARÁMETROS GLOBALES
    # ============================================================

    # ----------------------------------------
    # Datos del Problema (Regresión Simbólica)
    # ----------------------------------------
    USE_LOG_TRANSFORMATION = True

    # ----------------------------------------
    # Configuración General del Algoritmo Genético
    # ----------------------------------------
    FORCE_CPU_MODE = False # Si es True, usa CPU aunque CUDA esté disponible
    
    # Tamaño de población
    POP_SIZE = 5000 
    GENERATIONS = 50000
    NUM_ISLANDS = 5
    MIN_POP_PER_ISLAND = 10

    # --- Fórmula Inicial ---
    USE_INITIAL_FORMULA = False
    INITIAL_FORMULA_STRING = "log((x1+exp((((((1.28237193+((x0+2.59195138)+8.54688985))*x0)+(log((((x2/-0.99681346)-(x0-8.00219939))/(0.35461932-x2)))+(x0+(88.95319019/((x0+x0)+x0)))))-x1)/((exp(exp(((exp(x2)*(1.39925709/x0))^exp(x0))))+0.76703064)*6.05423753)))))"

    # ----------------------------------------
    # Parámetros del Modelo de Islas
    # ----------------------------------------
    MIGRATION_INTERVAL = 100
    MIGRATION_SIZE = 50

    # ----------------------------------------
    # Parámetros de Generación Inicial de Árboles
    # ----------------------------------------
    MAX_TREE_DEPTH_INITIAL = 8
    TERMINAL_VS_VARIABLE_PROB = 0.75
    CONSTANT_MIN_VALUE = -10.0
    CONSTANT_MAX_VALUE = 10.0
    CONSTANT_INT_MIN_VALUE = -10
    CONSTANT_INT_MAX_VALUE = 10
    USE_HARD_DEPTH_LIMIT = True
    MAX_TREE_DEPTH_HARD_LIMIT = 12

    # ----------------------------------------
    # Parámetros de Operadores Genéticos (Configuración de Operadores)
    # ----------------------------------------
    USE_OP_PLUS     = True
    USE_OP_MINUS    = True
    USE_OP_MULT     = True
    USE_OP_DIV      = True
    USE_OP_POW      = True
    USE_OP_MOD      = False
    USE_OP_SIN      = False
    USE_OP_COS      = False
    USE_OP_LOG      = True
    USE_OP_EXP      = True
    USE_OP_FACT     = False
    USE_OP_FLOOR    = False
    USE_OP_GAMMA    = True
    USE_OP_ASIN     = False
    USE_OP_ACOS     = False
    USE_OP_ATAN     = False

    # Pesos de Operadores (Order: +, -, *, /, ^, %, s, c, l, e, !, _, g, S, C, T)
    OPERATOR_WEIGHTS = [
        0.20 * (1.0 if USE_OP_PLUS else 0.0),
        0.20 * (1.0 if USE_OP_MINUS else 0.0),
        0.20 * (1.0 if USE_OP_MULT else 0.0),
        0.15 * (1.0 if USE_OP_DIV else 0.0),
        0.10 * (1.0 if USE_OP_POW else 0.0),
        0.02 * (1.0 if USE_OP_MOD else 0.0),
        0.10 * (1.0 if USE_OP_SIN else 0.0),
        0.10 * (1.0 if USE_OP_COS else 0.0),
        0.05 * (1.0 if USE_OP_LOG else 0.0),
        0.05 * (1.0 if USE_OP_EXP else 0.0),
        0.01 * (1.0 if USE_OP_FACT else 0.0),
        0.01 * (1.0 if USE_OP_FLOOR else 0.0),
        0.01 * (1.0 if USE_OP_GAMMA else 0.0),
        0.01 * (1.0 if USE_OP_ASIN else 0.0),
        0.01 * (1.0 if USE_OP_ACOS else 0.0),
        0.01 * (1.0 if USE_OP_ATAN else 0.0)
    ]

    # ----------------------------------------
    # Parámetros de Operadores Genéticos (Mutación, Cruce, Selección)
    # ----------------------------------------
    BASE_MUTATION_RATE = 0.30
    BASE_ELITE_PERCENTAGE = 0.15
    DEFAULT_CROSSOVER_RATE = 0.85
    DEFAULT_TOURNAMENT_SIZE = 30
    MAX_TREE_DEPTH_MUTATION = 8
    MUTATE_INSERT_CONST_PROB = 0.6
    MUTATE_INSERT_CONST_INT_MIN = 1
    MUTATE_INSERT_CONST_INT_MAX = 5
    MUTATE_INSERT_CONST_FLOAT_MIN = 0.5
    MUTATE_INSERT_CONST_FLOAT_MAX = 5.0

    # ----------------------------------------
    # Parámetros de Fitness y Evaluación
    # ----------------------------------------
    COMPLEXITY_PENALTY = 0.01
    USE_RMSE_FITNESS = True
    FITNESS_ORIGINAL_POWER = 1.3
    FITNESS_PRECISION_THRESHOLD = 0.001
    FITNESS_PRECISION_BONUS = 0.0001
    FITNESS_EQUALITY_TOLERANCE = 1e-9
    EXACT_SOLUTION_THRESHOLD = 1e-8

    # ----------------------------------------
    # Fitness Ponderado (Weighted Fitness)
    # ----------------------------------------
    USE_WEIGHTED_FITNESS = False
    WEIGHTED_FITNESS_EXPONENT = 0.25

    # ----------------------------------------
    # Parámetros de Características Avanzadas
    # ----------------------------------------
    STAGNATION_LIMIT = 50
    GLOBAL_STAGNATION_LIMIT = 100
    STAGNATION_RANDOM_INJECT_PERCENT = 0.1
    PARAM_MUTATE_INTERVAL = 50
    PATTERN_RECORD_FITNESS_THRESHOLD = 10.0
    PATTERN_MEM_MIN_USES = 3
    PATTERN_INJECT_INTERVAL = 10
    PATTERN_INJECT_PERCENT = 0.05
    PARETO_MAX_FRONT_SIZE = 50
    
    SIMPLIFY_NEAR_ZERO_TOLERANCE = 1e-9
    SIMPLIFY_NEAR_ONE_TOLERANCE = 1e-9
    LOCAL_SEARCH_ATTEMPTS = 30
    
    USE_SIMPLIFICATION = True
    USE_ISLAND_CATACLYSM = True
    USE_LEXICASE_SELECTION = True
    USE_PARETO_SELECTION = True  # NSGA-II multi-objective (error vs complexity)

    # ----------------------------------------
    # Otros Parámetros
    # ----------------------------------------
    PROGRESS_REPORT_INTERVAL = 100
    FORCE_INTEGER_CONSTANTS = False
    
    # Control de Duplicados
    PREVENT_DUPLICATES = True
    DUPLICATE_RETRIES = 10
    INF = float('inf')
