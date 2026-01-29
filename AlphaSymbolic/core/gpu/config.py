import math
import numpy as np

class GpuGlobals:
    # ============================================================
    #                  PARÁMETROS GLOBALES
    # ============================================================

    # ----------------------------------------
    # Datos del Problema (Regresión Simbólica)
    # ----------------------------------------
    USE_FLOAT32 = False # Optimización: Float32 (10x velocidad)
    USE_LOG_TRANSFORMATION = True # Default False for general usage (User can enable it)

    # DATASET CENTRALIZADO (N-Reinas)
    # x0 = n
    # x1 = n % 6
    # x2 = n % 2
    # Targets: OEIS A000170
    _PROBLEM_Y_RAW = np.array([1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848,39029188884,314666222712,2691008701644,24233937684440,227514171973736,2207893435808352], dtype=np.float64)
    # _PROBLEM_Y_RAW = np.array([1, 2, 6, 24, 120, 144, 28, 1408, 2025, 86400, 1782, 1092096, 4186, 31360, 241920000, 23953408, 140692, 114108912, 1092690], dtype=np.float64)
    
    # Coefficients optimized for n=8..27 (User Note).
    # We must skip N < 8 because the (8/n)^26 term explodes.
    PROBLEM_X_START = 8
    # PROBLEM_X_END = 25 # Inclusive
    PROBLEM_X_END = 24 # Inclusive
    
    DATA_FILTER_TYPE = "ALL" # Options: "ALL", "ODD", "EVEN"

    # Filter Logic
    _indices_raw = np.arange(PROBLEM_X_START, PROBLEM_X_END + 1)
    
    if DATA_FILTER_TYPE == "ODD":
        _mask = _indices_raw % 2 != 0
    elif DATA_FILTER_TYPE == "EVEN":
        _mask = _indices_raw % 2 == 0
    else:
        _mask = np.ones(len(_indices_raw), dtype=bool)

    PROBLEM_X_FILTERED = _indices_raw[_mask]
    # Slice Y to match the X range (N=1 at index 0)
    PROBLEM_Y_FILTERED = _PROBLEM_Y_RAW[(PROBLEM_X_START - 1) : PROBLEM_X_END][_mask]
    
    # Legacy/Direct Access Alias (pointing to FILTERED data to ensure usage)
    PROBLEM_Y_FULL = PROBLEM_Y_FILTERED 
    
    VAR_MOD_X1 = 6 # n % 6
    VAR_MOD_X2 = 2 # n % 2 (Paridad)

    # ----------------------------------------
    # Configuración General del Algoritmo Genético
    # ----------------------------------------

    # ----------------------------------------
    # Configuración General del Algoritmo Genético
    # ----------------------------------------
    FORCE_CPU_MODE = False # Si es True, usa CPU aunque CUDA esté disponible
    
    # Tamaño de población - MÁXIMO para RTX 3050 (4GB VRAM)
    # Analysis (Optimized Engine 2025-01-12):
    # - With Chunked Reproduction: 4.0M Population is STABLE.
    # - Peak VRAM: ~3.65 GB (Cycle) / 2.75 GB (Eval).
    # - Island Migration limit hit at 5.0M.
    # Recommended: 100,000 (General) | 4,000,000 (Hard Benchmarks)
    POP_SIZE = 1_000_000
    GENERATIONS = 500  
    NUM_ISLANDS = 50 # 1M / 50 = 100k pop per island
    MIN_POP_PER_ISLAND = 20

    # --- Fórmula Inicial ---
    USE_INITIAL_FORMULA = True
    #INITIAL_FORMULA_STRING = "(cos(sqrt(abs(((((5 + floor((x1 + x0))) / (lgamma(x0) - x0)) - (1.09359063 * x0)) - 5.31499599)))) + (lgamma((-0.09963219 + x0)) + (5 - x0)))"
    # Evolved Gen 16 seed (Verified < 1% error)
    INITIAL_FORMULA_STRING = "((lgamma(x0) + sqrt(((x0 + (cos(((x0 + 5) + x1)) / x0)) + (cos(gamma((7.00021942 - x0))) % (5 + log(log(5))))))) - (x0 - 0.45850736))"

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
    MAX_TREE_DEPTH_HARD_LIMIT = 30  # MÁXIMO - expresiones muy complejas
    MAX_CONSTANTS = 50 # Increased to 50 to guarantee ANY generated formula (size < 30) fits as a seed without truncation.

    # ----------------------------------------
    # Parámetros de Operadores Genéticos (Configuración de Operadores)
    # ----------------------------------------
    USE_OP_PLUS     = True
    USE_OP_MINUS    = True
    USE_OP_MULT     = True
    USE_OP_DIV      = True
    USE_OP_POW      = True
    USE_OP_MOD      = True
    USE_OP_SIN      = True
    USE_OP_COS      = True
    USE_OP_LOG      = True
    USE_OP_EXP      = True
    USE_OP_FACT     = True
    USE_OP_FLOOR    = True
    USE_OP_GAMMA    = True
    USE_OP_ASIN     = True
    USE_OP_ACOS     = True
    USE_OP_ATAN     = True
    USE_OP_CEIL     = True
    USE_OP_SIGN     = True

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
        0.01 * (1.0 if USE_OP_ATAN else 0.0),
        0.005 * (1.0 if USE_OP_CEIL else 0.0),
        0.005 * (1.0 if USE_OP_SIGN else 0.0)
    ]

    # ----------------------------------------
    # Parámetros de Operadores Genéticos (Mutación, Cruce, Selección)
    # ----------------------------------------
    BASE_MUTATION_RATE = 0.40
    BASE_ELITE_PERCENTAGE = 0.10
    DEFAULT_CROSSOVER_RATE = 0.50
    DEFAULT_TOURNAMENT_SIZE = 8
    MAX_TREE_DEPTH_MUTATION = 12
    MUTATE_INSERT_CONST_PROB = 0.5
    MUTATE_INSERT_CONST_INT_MIN = 1
    MUTATE_INSERT_CONST_INT_MAX = 5
    MUTATE_INSERT_CONST_FLOAT_MIN = 0.5
    MUTATE_INSERT_CONST_FLOAT_MAX = 5.0

    # ----------------------------------------
    # Parámetros de Fitness y Evaluación
    # ----------------------------------------
    COMPLEXITY_PENALTY = 0.0001 # Reduced to allow growth
    LOSS_FUNCTION = 'RMSE' # Changed from RMSLE because USE_LOG_TRANSFORMATION=True. 
                           # RMSE on Log(y) == RMSLE on y. Avoids double log.
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
    K_SIMPLIFY = 20                # Number of top formulas to simplify per island
    SIMPLIFICATION_INTERVAL = 20    # Simplify every N generations
    USE_ISLAND_CATACLYSM = True
    USE_LEXICASE_SELECTION = True
    USE_PARETO_SELECTION = True  # Disabled for stronger fitness pressure on simple problems
    USE_WEIGHTED_FITNESS = False  # Enable to weight fitness cases (e.g., by difficulty)
    USE_NANO_PSO = True # Enable Particle Swarm Optimization for constants
    
    USE_SNIPER = True             # Enable Linear/Geometric/Log-Linear detection
    USE_RESIDUAL_BOOSTING = True  # Enable finding f(x)+g(x) using Sniper on residuals
    USE_NEURAL_FLASH = False      # Enable Neural Inspiration (Beam Search injection)
    USE_ALPHA_MCTS = False        # Enable Alpha Mode (MCTS Refinement)
    USE_PATTERN_MEMORY = True     # Optimized: GPU-Based Pattern Extraction

    # ----------------------------------------
    # Otros Parámetros
    # ----------------------------------------
    PROGRESS_REPORT_INTERVAL = 100
    FORCE_INTEGER_CONSTANTS = False
    
    # Control de Duplicados
    PREVENT_DUPLICATES = True
    DUPLICATE_RETRIES = 10
    INF = float('inf')
