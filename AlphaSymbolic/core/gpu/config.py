import math
import numpy as np

class GpuGlobals:
    # ============================================================
    #                  1. SYSTEM & HARDWARE
    # ============================================================
    USE_FLOAT32 = True             # Optimization: Float32 (10x speedup vs Float64)
    FORCE_CPU_MODE = False         # Force CPU even if CUDA is available
    USE_CUDA_ORCHESTRATOR = True   # Use C++ Orchestrator for evolution loop
    INF = float('inf')

    # ============================================================
    #                  2. DATASET CONFIGURATION
    # ============================================================
    # Target Sequence: OEIS A000170 (N-Queens)
    _PROBLEM_Y_RAW = np.array([
        1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,
        14772512,95815104,666090624,4968057848,39029188884,314666222712,
        2691008701644,24233937684440,227514171973736,2207893435808352
    ], dtype=np.float64)

    # Range and Filter
    PROBLEM_X_START = 8
    PROBLEM_X_END = 24            # Inclusive
    DATA_FILTER_TYPE = "ALL"      # Options: "ALL", "ODD", "EVEN"

    # Filter Logic (Computed)
    _indices_raw = np.arange(PROBLEM_X_START, PROBLEM_X_END + 1)
    if DATA_FILTER_TYPE == "ODD":
        _mask = _indices_raw % 2 != 0
    elif DATA_FILTER_TYPE == "EVEN":
        _mask = _indices_raw % 2 == 0
    else:
        _mask = np.ones(len(_indices_raw), dtype=bool)

    PROBLEM_X_FILTERED = _indices_raw[_mask]
    # Slice Y to match the X range (N=1 at index 0 corresponds to raw index 0)
    PROBLEM_Y_FILTERED = _PROBLEM_Y_RAW[(PROBLEM_X_START - 1) : PROBLEM_X_END][_mask]
    PROBLEM_Y_FULL = PROBLEM_Y_FILTERED  # Alias

    # Input Transformations
    USE_LOG_TRANSFORMATION = True  # Transform Y to log(Y) for high-magnitude data
    VAR_MOD_X1 = 6                 # x1 = n % 6
    VAR_MOD_X2 = 2                 # x2 = n % 2

    # ============================================================
    #                  3. SEARCH STRATEGY (ISLAND MODEL)
    # ============================================================
    # Population Size
    # Recommended: 100k (Fast) | 1M (Standard) | 4M (Hard/RTX 3050 limit)
    POP_SIZE = 1_000_000
    GENERATIONS = 1_000_000
    
    # Islands
    NUM_ISLANDS = 25              # 1M / 25 = 40k per island
    MIN_POP_PER_ISLAND = 20
    
    # Migration
    MIGRATION_INTERVAL = 10            # Standard migration interval
    MIGRATION_INTERVAL_STAGNATION = 20 # Slower migration during stagnation (preserve diversity)
    MIGRATION_STAGNATION_THRESHOLD = 10
    MIGRATION_SIZE = 50

    # Stagnation & Restarts
    STAGNATION_LIMIT = 30              # Gens without local improvement before cataclysm
    GLOBAL_STAGNATION_LIMIT = 150      # Gens without global improvement before restart
    
    USE_ISLAND_CATACLYSM = True        # Local restart of island
    CATACLYSM_ELITE_PERCENT = 0.08     # Elites survived in cataclysm
    
    SOFT_RESTART_ENABLED = True        # Global soft restart
    SOFT_RESTART_ELITE_RATIO = 0.10    # Elites survived in global restart
    
    USE_STRUCTURAL_RESTART_INJECTION = False
    STRUCTURAL_RESTART_INJECTION_RATIO = 0.25
    HARD_RESTART_ELITE_RATIO = 0.12

    # ============================================================
    #                  4. GRAMMAR & INITIALIZATION
    # ============================================================
    # Initial Population
    USE_INITIAL_POP_CACHE = True
    USE_INITIAL_FORMULA = True
    # Evolved Gen 16 seed (Verified < 1% error)
    INITIAL_FORMULA_STRING = "(lgamma(x0) - (x0 - sqrt((5 + (x0 + (1 / ((10 + (x2 - x0)) - sqrt(3))))))))"

    USE_STRUCTURAL_SEEDS = False       # Generate polynomial/trig basis seeds

    # Tree Constraints
    MAX_TREE_DEPTH_INITIAL = 5
    USE_HARD_DEPTH_LIMIT = True
    MAX_TREE_DEPTH_HARD_LIMIT = 60
    MAX_TREE_DEPTH_MUTATION = 6

    # Constants
    MAX_CONSTANTS = 15
    CONSTANT_MIN_VALUE = -10.0
    CONSTANT_MAX_VALUE = 10.0
    CONSTANT_INT_MIN_VALUE = -10
    CONSTANT_INT_MAX_VALUE = 10
    FORCE_INTEGER_CONSTANTS = False # Force int constants in PSO

    # Operators Config
    USE_OP_PLUS     = True
    USE_OP_MINUS    = True
    USE_OP_MULT     = True
    USE_OP_DIV      = True
    USE_OP_POW      = True
    USE_OP_MOD      = False
    USE_OP_SIN      = False
    USE_OP_COS      = False
    USE_OP_TAN      = False
    USE_OP_LOG      = True
    USE_OP_EXP      = True
    USE_OP_FACT     = True
    USE_OP_FLOOR    = False
    USE_OP_GAMMA    = True
    USE_OP_ASIN     = False
    USE_OP_ACOS     = False
    USE_OP_ATAN     = False
    USE_OP_CEIL     = False
    USE_OP_SIGN     = False
    USE_OP_SQRT     = True
    USE_OP_ABS      = True

    # Operator Weights (Probabilities)
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

    # ============================================================
    #                  5. GENETIC OPERATORS
    # ============================================================
    # Rates
    BASE_MUTATION_RATE = 0.15
    DEFAULT_CROSSOVER_RATE = 0.60
    
    # Adaptive Mutation
    MUTATION_RATE_CAP = 0.70
    MUTATION_RAMP_PER_GEN = 0.015
    MUTATION_STAGNATION_TRIGGER = 5
    
    # Selection
    DEFAULT_TOURNAMENT_SIZE = 5
    TOURNAMENT_SIZE_FLOOR = 3
    TOURNAMENT_ADAPTIVE_DIVISOR = 6
    BASE_ELITE_PERCENTAGE = 0.12
    
    # Generation
    TERMINAL_VS_VARIABLE_PROB = 0.50
    DEDUPLICATION_INTERVAL = 50
    PREVENT_DUPLICATES = True
    
    # Mutation Bank
    MUTATION_BANK_SIZE = 2000
    MUTATION_BANK_REFRESH_INTERVAL = 50

    # ============================================================
    #                  6. EVALUATION & FITNESS
    # ============================================================
    LOSS_FUNCTION = 'RMSE'
    
    # Validation
    FORCE_STRICT_VALIDATION = True     # Strict math Mode (No protected operators)
    
    # Penalties
    COMPLEXITY_PENALTY = 0.02
    TRIVIAL_FORMULA_PENALTY = 1.5
    NO_VARIABLE_PENALTY = 2.5
    TRIVIAL_FORMULA_MAX_TOKENS = 2
    TRIVIAL_FORMULA_ALLOW_RMSE = 1e-3
    
    # Diversity (Disabled)
    VAR_DIVERSITY_PENALTY = 0.0
    VAR_FORCE_SEED_PERCENT = 0.0
    
    # Weighted Fitness
    USE_WEIGHTED_FITNESS = False
    WEIGHTED_FITNESS_EXPONENT = 0.25

    # ============================================================
    #                  7. OPTIMIZATION (PSO & SIMPLIFICATION)
    # ============================================================
    # Particle Swarm Optimization (PSO)
    USE_NANO_PSO = True
    PSO_INTERVAL = 2
    PSO_PARTICLES = 30
    PSO_STEPS_NORMAL = 20
    PSO_STEPS_STAGNATION = 30
    PSO_K_NORMAL = 300            # Top K individuals to optimize
    PSO_K_STAGNATION = 600
    PSO_STAGNATION_THRESHOLD = 10
    
    # Simplification
    USE_SIMPLIFICATION = True
    USE_SYMPY = False             # Heavy symbolic simplification (Slow)
    USE_CONSOLE_BEST_SIMPLIFICATION = False
    SIMPLIFICATION_INTERVAL = 10
    K_SIMPLIFY = 50
    SIMPLIFY_NEAR_ZERO_TOLERANCE = 1e-9
    SIMPLIFY_NEAR_ONE_TOLERANCE = 1e-9
    
    # Residual Boosting
    USE_RESIDUAL_BOOSTING = True
    RESIDUAL_BOOST_INTERVAL = 20

    # ============================================================
    #                  8. ADVANCED FEATURES
    # ============================================================
    # Selection Strategies
    USE_LEXICASE_SELECTION = True
    USE_PARETO_SELECTION = True
    PARETO_INTERVAL = 5
    PARETO_MAX_FRONT_SIZE = 30
    
    # Pattern Memory
    USE_PATTERN_MEMORY = True
    PATTERN_RECORD_INTERVAL = 20
    PATTERN_INJECT_INTERVAL = 10
    PATTERN_INJECT_PERCENT = 0.05
    PATTERN_RECORD_FITNESS_THRESHOLD = 10.0
    PATTERN_MEM_MIN_USES = 3
    PATTERN_MIN_SIZE = 3
    PATTERN_MAX_SIZE = 12
    PATTERN_MAX_PATTERNS = 100
    
    # Neural / MCTS / Sniper (Disabled/Experimental)
    USE_SNIPER = False
    USE_NEURAL_FLASH = False
    NEURAL_FLASH_INTERVAL = 50
    NEURAL_FLASH_INJECT_PERCENT = 0.10
    USE_ALPHA_MCTS = False
    ALPHA_MCTS_INTERVAL = 100
    ALPHA_MCTS_N_SIMULATIONS = 50

    # ============================================================
    #                  9. REPORTING & EXIT
    # ============================================================
    PROGRESS_REPORT_INTERVAL = 100
    
    # Early Exit
    EXACT_SOLUTION_THRESHOLD = 1e-6
    FITNESS_EQUALITY_TOLERANCE = 1e-9
    ALLOW_WARMUP_EARLY_EXIT = False
    
    # Good Enough Exit
    GOOD_ENOUGH_RMSE = -1.0        # Disabled
    GOOD_ENOUGH_R2 = 0.995
    GOOD_ENOUGH_MIN_SECONDS = 4.0
