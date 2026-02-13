import math
import numpy as np

class GpuGlobals:
    # ============================================================
    #                  PARÁMETROS GLOBALES
    # ============================================================

    # ----------------------------------------
    # Datos del Problema (Regresión Simbólica)
    # ----------------------------------------
    USE_FLOAT32 = True # Optimización: Float32 (10x velocidad)
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
    FORCE_CPU_MODE = False # Si es True, usa CPU aunque CUDA esté disponible
    
    # Tamaño de población - MÁXIMO para RTX 3050 (4GB VRAM)
    # Analysis (Optimized Engine 2025-01-12):
    # - With Chunked Reproduction: 4.0M Population is STABLE.
    # - Peak VRAM: ~3.65 GB (Cycle) / 2.75 GB (Eval).
    # - Island Migration limit hit at 5.0M.
    # Recommended: 100,000 (General) | 4,000,000 (Hard Benchmarks)
    POP_SIZE = 1_000_000
    GENERATIONS = 1_000_000  
    NUM_ISLANDS = 25 # 500k / 25 = 20k pop per island
    MIN_POP_PER_ISLAND = 20

    # --- Fórmula Inicial ---
    # --- Fórmula Inicial ---
    USE_INITIAL_FORMULA = False
    #INITIAL_FORMULA_STRING = "(cos(sqrt(abs(((((5 + floor((x1 + x0))) / (lgamma(x0) - x0)) - (1.09359063 * x0)) - 5.31499599)))) + (lgamma((-0.09963219 + x0)) + (5 - x0)))"
    # Evolved Gen 16 seed (Verified < 1% error)
    INITIAL_FORMULA_STRING = "((atan(fact(sin(log((3 + x0))))) ^ (3.09679054 % sqrt(x0))) + (lgamma((atan(((x0 - atan(gamma(cos(x0)))) - (x0 % ceil(pi)))) + x0)) - x0))"

    # ----------------------------------------
    # Parámetros del Modelo de Islas
    # ----------------------------------------
    MIGRATION_INTERVAL = 10         # Intervalo normal de migración (generaciones)
    MIGRATION_INTERVAL_STAGNATION = 20  # Intervalo durante estancamiento (mayor = preserva diversidad)
    MIGRATION_STAGNATION_THRESHOLD = 10 # Gens de estancamiento para cambiar intervalo
    MIGRATION_SIZE = 50

    # ----------------------------------------
    # Parámetros de Generación Inicial de Árboles
    # ----------------------------------------
    MAX_TREE_DEPTH_INITIAL = 5
    CONSTANT_MIN_VALUE = -10.0
    CONSTANT_MAX_VALUE = 10.0
    CONSTANT_INT_MIN_VALUE = -10
    CONSTANT_INT_MAX_VALUE = 10
    MAX_CONSTANTS = 15
    USE_HARD_DEPTH_LIMIT = True
    MAX_TREE_DEPTH_HARD_LIMIT = 60   # Límite duro de profundidad de arboles
    MAX_TREE_DEPTH_MUTATION = 6      # Profundidad máxima de subtrees generados en mutación (más profundo = más exploración estructural)

    # ----------------------------------------
    # Parámetros de Operadores Genéticos (Configuración de Operadores)
    # ----------------------------------------
    USE_OP_PLUS     = True
    USE_OP_MINUS    = True
    USE_OP_MULT     = True
    USE_OP_DIV      = True
    USE_OP_POW      = True
    USE_OP_MOD      = False
    USE_OP_SIN      = False  # Desactivado: causa bloat sin beneficio para regresión
    USE_OP_COS      = False
    USE_OP_TAN      = False
    USE_OP_LOG      = True
    USE_OP_EXP      = True
    USE_OP_FACT     = True   # Desactivado: causa bloat sin beneficio para regresión
    USE_OP_FLOOR    = False
    USE_OP_GAMMA    = True    # Desactivado: lgamma/gamma causan explosión numérica y bloat
    USE_OP_ASIN     = False
    USE_OP_ACOS     = False
    USE_OP_ATAN     = False
    USE_OP_CEIL     = False
    USE_OP_SIGN     = False
    USE_OP_SQRT     = True
    USE_OP_ABS      = True    # Activado: PySR lo usa frecuentemente

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
        0.10 * (1.0 if USE_OP_TAN else 0.0),
        0.05 * (1.0 if USE_OP_LOG else 0.0),
        0.05 * (1.0 if USE_OP_EXP else 0.0),
        0.01 * (1.0 if USE_OP_FACT else 0.0),
        0.01 * (1.0 if USE_OP_FLOOR else 0.0),
        0.01 * (1.0 if USE_OP_GAMMA else 0.0),
        0.01 * (1.0 if USE_OP_ASIN else 0.0),
        0.01 * (1.0 if USE_OP_ACOS else 0.0),
        0.01 * (1.0 if USE_OP_ATAN else 0.0),
        0.005 * (1.0 if USE_OP_CEIL else 0.0),
        0.005 * (1.0 if USE_OP_SIGN else 0.0),
        0.10 * (1.0 if USE_OP_SQRT else 0.0),
        0.05 * (1.0 if USE_OP_ABS else 0.0)
    ]

    # ----------------------------------------
    # Parámetros de Operadores Genéticos (Mutación, Cruce, Selección)
    # ----------------------------------------
    BASE_MUTATION_RATE = 0.15
    MUTATION_RATE_CAP = 0.70         # Techo de mutación adaptativa (más bajo = preserva mejores soluciones)
    MUTATION_RAMP_PER_GEN = 0.015    # Incremento por gen de estancamiento (más suave)
    MUTATION_STAGNATION_TRIGGER = 5  # Gens estancadas para empezar a subir mutación
    BASE_ELITE_PERCENTAGE = 0.12     # Más elites preservadas = convergencia más estable
    DEFAULT_CROSSOVER_RATE = 0.60    # Más crossover = más recombinación de building blocks
    DEFAULT_TOURNAMENT_SIZE = 5      # Torneo más pequeño = más diversidad en selección
    TOURNAMENT_SIZE_FLOOR = 3        # Piso del torneo adaptativo
    TOURNAMENT_ADAPTIVE_DIVISOR = 6  # Cada N gens estancadas baja 1 el torneo
    TERMINAL_VS_VARIABLE_PROB = 0.50 # Prob de terminal vs variable en generación aleatoria (0.5 = neutro)

    # ----------------------------------------
    # Parámetros de Fitness y Evaluación
    # ----------------------------------------
    COMPLEXITY_PENALTY = 0.02
    VAR_DIVERSITY_PENALTY = 0.0    # Desactivado: usar seed injection + PSO en lugar de penalidad
    VAR_FORCE_SEED_PERCENT = 0.0   # Desactivado: seeds estructurales reemplazan fuerza bruta
    LOSS_FUNCTION = 'RMSE'
    EXACT_SOLUTION_THRESHOLD = 1e-6
    FITNESS_EQUALITY_TOLERANCE = 1e-9  # Tolerancia para considerar dos fitness iguales
    USE_WEIGHTED_FITNESS = False       # Ponderar casos de fitness por dificultad
    WEIGHTED_FITNESS_EXPONENT = 0.25   # Exponente para ponderación de fitness

    # ----------------------------------------
    # Parámetros de Estancamiento y Cataclismo
    # ----------------------------------------
    STAGNATION_LIMIT = 30            # Gens sin mejora para disparar cataclismo (reaccionar más rápido)
    GLOBAL_STAGNATION_LIMIT = 150     # Gens sin mejora global para reinicio completo (menos desperdicio)
    CATACLYSM_ELITE_PERCENT = 0.08   # % de élites que sobreviven el cataclismo (más = mejores building blocks preservados)
    STAGNATION_RANDOM_INJECT_PERCENT = 0.0  # Desactivado: overhead sin beneficio (peores nunca ganan selección)
    USE_ISLAND_CATACLYSM = True      # Activar/desactivar cataclismo
    
    # --- Soft Restart (Global) ---
    SOFT_RESTART_ENABLED = True      # If True, global stagnation preserves elite % instead of full reset
    SOFT_RESTART_ELITE_RATIO = 0.10  # % of population to keep during global restart (10% = 800 indivs for 8k pop)
    
    # ----------------------------------------
    # Parámetros de PSO (Particle Swarm Optimization)
    # ----------------------------------------
    USE_NANO_PSO = True
    PSO_INTERVAL = 2                 # Ejecutar PSO cada 2 generaciones
    PSO_K_NORMAL = 300               # Top-300 optimizados por PSO (más individuos benefician de constantes optimizadas)
    PSO_K_STAGNATION = 600           # Más individuos durante estancamiento
    PSO_STEPS_NORMAL = 20            # Pasos de PSO normales (más pasos = constantes más precisas)
    PSO_STEPS_STAGNATION = 30        # Pasos agresivos durante estancamiento
    PSO_STAGNATION_THRESHOLD = 10    # Gens estancadas para usar parámetros agresivos (reaccionar antes)
    PSO_PARTICLES = 30               # Partículas por individuo (más = mejor exploración)
    
    # ----------------------------------------
    # Parámetros de Pattern Memory
    # ----------------------------------------
    USE_PATTERN_MEMORY = True
    PATTERN_RECORD_FITNESS_THRESHOLD = 10.0
    PATTERN_MEM_MIN_USES = 3
    PATTERN_RECORD_INTERVAL = 20     # Cada cuántas gens grabar patrones
    PATTERN_INJECT_INTERVAL = 10
    PATTERN_INJECT_PERCENT = 0.05
    PATTERN_MIN_SIZE = 3             # Tamaño mínimo de subtree para patrón
    PATTERN_MAX_SIZE = 12            # Tamaño máximo de subtree para patrón
    PATTERN_MAX_PATTERNS = 100       # Patrones máximos en memoria
    
    # ----------------------------------------
    # Parámetros de Simplificación
    # ----------------------------------------
    USE_SIMPLIFICATION = True
    K_SIMPLIFY = 50
    SIMPLIFICATION_INTERVAL = 10
    USE_CONSOLE_BEST_SIMPLIFICATION = False
    SIMPLIFY_NEAR_ZERO_TOLERANCE = 1e-9  # Tolerancia para colapsar valores cercanos a 0
    SIMPLIFY_NEAR_ONE_TOLERANCE = 1e-9   # Tolerancia para colapsar valores cercanos a 1
    
    # ----------------------------------------
    # Parámetros de Deduplicación
    # ----------------------------------------
    DEDUPLICATION_INTERVAL = 50      # Cada cuántas gens eliminar clones
    PREVENT_DUPLICATES = True        # Activar/desactivar deduplicación
    
    # ----------------------------------------
    # Parámetros de Selección Pareto
    # ----------------------------------------
    USE_PARETO_SELECTION = True       # Pareto liviano: protege fórmulas cortas+buenas como élites
    PARETO_INTERVAL = 5              # Cada cuántas gens actualizar el frente de Pareto
    PARETO_MAX_FRONT_SIZE = 30       # Tamaño máximo del frente de Pareto
    
    # ----------------------------------------
    # Parámetros de Residual Boosting
    # ----------------------------------------
    USE_RESIDUAL_BOOSTING = True
    RESIDUAL_BOOST_INTERVAL = 20     # Cada cuántas gens de estancamiento intentar boost

    # ----------------------------------------
    # Early Exit "Good Enough"
    # ----------------------------------------
    GOOD_ENOUGH_RMSE = -1.0          # Desactivado globalmente (condición nunca se cumple)
    GOOD_ENOUGH_R2 = 0.995           # Umbral de R² mínimo para salida temprana
    GOOD_ENOUGH_MIN_SECONDS = 4.0    # Tiempo mínimo antes de permitir salida temprana

    # ----------------------------------------
    # Global Restart Injection
    # ----------------------------------------
    USE_STRUCTURAL_RESTART_INJECTION = False
    STRUCTURAL_RESTART_INJECTION_RATIO = 0.25  # Porcentaje de población inyectada con bases estructurales
    HARD_RESTART_ELITE_RATIO = 0.12            # % élites preservadas en hard restart (sin plantillas)
    
    # ----------------------------------------
    # Parámetros de Mutation Bank
    # ----------------------------------------
    MUTATION_BANK_SIZE = 2000        # Subtrees aleatorios para ingredientes de mutación
    MUTATION_BANK_REFRESH_INTERVAL = 50  # Cada cuántas gens renovar el banco
    
    # ----------------------------------------
    # Parámetros de Neural Flash y MCTS
    # ----------------------------------------
    USE_NEURAL_FLASH = False
    NEURAL_FLASH_INTERVAL = 50       # Cada cuántas gens inyectar
    NEURAL_FLASH_INJECT_PERCENT = 0.10  # % de la población a reemplazar
    USE_ALPHA_MCTS = False
    ALPHA_MCTS_INTERVAL = 100        # Cada cuántas gens ejecutar MCTS
    ALPHA_MCTS_N_SIMULATIONS = 50    # Simulaciones por ejecución de MCTS
    
    # ----------------------------------------
    # Parámetros de Selección
    # ----------------------------------------
    USE_LEXICASE_SELECTION = True
    USE_SNIPER = False
    USE_STRUCTURAL_SEEDS = False
    
    # ----------------------------------------
    # Otros Parámetros
    # ----------------------------------------
    PROGRESS_REPORT_INTERVAL = 100
    USE_CUDA_ORCHESTRATOR = True
    USE_INITIAL_POP_CACHE = True
    USE_SYMPY = False
    FORCE_INTEGER_CONSTANTS = False   # Forzar constantes enteras en PSO/generación

    # ----------------------------------------
    # Anti-trivial collapse controls
    # ----------------------------------------
    TRIVIAL_FORMULA_MAX_TOKENS = 2      # e.g., x0 or constant-only
    TRIVIAL_FORMULA_PENALTY = 1.5       # additive penalty in selection metric
    TRIVIAL_FORMULA_ALLOW_RMSE = 1e-3   # allow trivial elites only if genuinely good
    NO_VARIABLE_PENALTY = 2.5           # penalize formulas that do not use input variables
    
    INF = float('inf')
