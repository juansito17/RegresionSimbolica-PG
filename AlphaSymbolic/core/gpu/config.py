import math
import numpy as np

class GpuGlobals:
    # ============================================================
    #                  1. SYSTEM & HARDWARE
    # ============================================================
    # RTX 3050 Laptop: ~40 TFLOPS FP32 vs ~2.5 TFLOPS FP64 (ratio 1/16).
    # Float32 gives 4-8x speedup and activates the fused PSO kernel.
    USE_FLOAT32 = True             # OPTIMIZED: Float32 (4-8x speedup on consumer GPUs)
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
    NUM_ISLANDS = 50              # OPTIMIZED: 50 islas de 80k c/u → más diversidad (era 25)
    MIN_POP_PER_ISLAND = 80
    
    # Migration
    MIGRATION_INTERVAL = 30                    # OPTIMIZED: más aislamiento (era 15)
    MIGRATION_INTERVAL_STAGNATION = 25         # Slower migration during island stagnation
    MIGRATION_INTERVAL_GLOBAL_STAGNATION = 300 # NEW: durante global stagnation > 20 → casi no migrar para mantener islas aisladas
    MIGRATION_STAGNATION_THRESHOLD = 10
    MIGRATION_SIZE = 80                # OPTIMIZED: más individuos por migración (era 50)

    # Stagnation & Restarts
    STAGNATION_LIMIT = 15              # OPTIMIZED: balance entre exploración y escape (was 20→15)
    GLOBAL_STAGNATION_LIMIT = 40       # ANTI-STAG: 80→40. Escalar restarts más rápido para escapar antes
    STAGNATION_RANDOM_INJECT_PERCENT = 0.05  # FIX: 5% para no destruir elites durante cooldown (era 0.50)
    
    USE_ISLAND_CATACLYSM = True        # Local restart of island
    CATACLYSM_ELITE_PERCENT = 0.06     # FIX: cataclismos más agresivos — menos elites = más diversidad (era 0.12)
    
    SOFT_RESTART_ENABLED = True        # Global soft restart
    SOFT_RESTART_ELITE_RATIO = 0.04    # FIX: muy pocos elites en restart → fuerza diversidad estructural (era 0.15)
    ESCALATE_RESTART_LIMIT = 1         # ANTI-STAG: 2→1. TRUE HARD restart después del 1er soft restart sin mejora
    
    USE_STRUCTURAL_RESTART_INJECTION = False  # Sin bias de estructura — búsqueda completamente aleatoria
    STRUCTURAL_RESTART_INJECTION_RATIO = 0.25
    HARD_RESTART_ELITE_RATIO = 0.12
    
    # Elitismo dinámico durante estancamiento global
    # Cuando global_stagnation > GLOBAL_STAGNATION_LIMIT/2 (plateau profundo),
    # reducir elitismo para permitir que estructuras nuevas compitan sin ser aplastadas por el super-elite.
    ELITE_PCT_STAGNATION = 0.005       # ANTI-STAG: 0.5% elites durante estancamiento profundo (solo ~20k)

    # ============================================================
    #                  4. GRAMMAR & INITIALIZATION
    # ============================================================
    # Initial Population
    USE_INITIAL_POP_CACHE = False
    USE_INITIAL_FORMULA = False
    # Evolved Gen 16 seed (Verified < 1% error)
    INITIAL_FORMULA_STRING = "log(((((lgamma(x0) * pi) + ((1 + 1.10762465)**(e + lgamma((x0 - 3.24382305))))) - lgamma(fact(x1))) + (sqrt(x0)**((exp(2) - x1)**x2))))"

    USE_STRUCTURAL_SEEDS = False       # Generate polynomial/trig basis seeds

    # Tree Constraints
    MAX_TREE_DEPTH_INITIAL = 5
    USE_HARD_DEPTH_LIMIT = True
    MAX_TREE_DEPTH_HARD_LIMIT = 60
    MAX_TREE_DEPTH_MUTATION = 10   # ANTI-STAG: 6→10. Subtrees más grandes compiten con el elite inyectado

    # Constants — reduced from 15 to 8: typical formulas use 3-5 constants.
    # Smaller K = faster PSO (47% fewer dimensions to optimize).
    MAX_CONSTANTS = 8
    CONSTANT_MIN_VALUE = -25.0
    CONSTANT_MAX_VALUE = 25.0
    CONSTANT_INT_MIN_VALUE = -25
    CONSTANT_INT_MAX_VALUE = 25
    FORCE_INTEGER_CONSTANTS = False # Force int constants in PSO
    CONSTANT_PRECISION = 8          # Number of decimal places to show in formulas (Console)

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
    USE_OP_ABS      = False

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
        # OPTIMIZED: fact/lgamma/pow aumentados para N-Queens (fórmula objetivo usa lgamma)
        0.08 * (1.0 if USE_OP_FACT else 0.0),
        0.01 * (1.0 if USE_OP_FLOOR else 0.0),
        0.08 * (1.0 if USE_OP_GAMMA else 0.0),
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
    BASE_MUTATION_RATE = 0.18      # OPTIMIZED: más exploración estructural (era 0.15)
    DEFAULT_CROSSOVER_RATE = 0.60
    
    # Adaptive Mutation
    MUTATION_RATE_CAP = 0.50       # Reduced from 0.70 to prevent destroying elites during long plateaus
    MUTATION_RAMP_PER_GEN = 0.015
    MUTATION_STAGNATION_TRIGGER = 5
    
    # Selection
    DEFAULT_TOURNAMENT_SIZE = 5
    TOURNAMENT_SIZE_FLOOR = 3
    TOURNAMENT_ADAPTIVE_DIVISOR = 6
    BASE_ELITE_PERCENTAGE = 0.12
    
    # Generation
    TERMINAL_VS_VARIABLE_PROB = 0.40   # OPTIMIZED: más tokens C en árboles aleatorios (was 0.50→0.40)
    DEDUPLICATION_INTERVAL = 50
    
    # --- SOTA P0: Headless Chicken Crossover ---
    # Con esta probabilidad, uno de los padres se reemplaza con un individuo 100% aleatorio.
    # Fuerza exploración estructural radical cuando la población converge hacia un super-elite.
    # LaSR, PySR y Operon usan variantes de este mecanismo como escape de mínimos locales.
    HEADLESS_CHICKEN_RATE = 0.15       # 15% de crossovers usan un padre aleatorio
    
    # --- SOTA P1: Depth-Fair Crossover ---
    # Standard crossover picks a random NODE as swap point → large subtrees dominate selection.
    # Depth-Fair: pick a random DEPTH first, then a random node at that depth.
    # This gives small subtrees equal probability of being selected, reducing bloat.
    DEPTH_FAIR_CROSSOVER = True        # Enable depth-fair subtree crossover

    # --- SOTA P2: ALPS (Age-Layered Population Structure) ---
    # ALPS prevents elite stagnation by tracking the age of each individual.
    # Older individuals in early layers are penalized during selection,
    # giving younger, freshly produced individuals a better chance.
    # Layer 0 is periodically reseeded with fresh random individuals.
    # Based on: Hornby (2006) — no single algorithm dominates ALPS.
    USE_ALPS = True                    # Enable age-layered selection pressure
    ALPS_AGE_GAP = 10                  # Gens between layer boundaries (layer_idx = age // gap)
    ALPS_MAX_LAYER = 5                 # Max layer index (beyond = capped, treated as =max)
    ALPS_AGE_PENALTY_WEIGHT = 0.05    # Weight of age penalty in selection_metric
    ALPS_LAYER0_RESEED_INTERVAL = 50  # Every N gens, reseed layer-0 individuals from scratch
    ALPS_LAYER0_RESEED_FRACTION = 0.01 # Fraction of pop_size to reseed as fresh individuals


    
    # --- SOTA P0: Constant Perturbation Mutation ---
    # Perturba constantes con ruido gaussiano multiplicativo. Complementa PSO:
    # PSO explora globalmente en el espacio de constantes, perturbación explora localmente.
    # Especialmente efectivo para escapar mínimos de constantes donde PSO se estancó.
    CONSTANT_PERTURBATION_RATE = 0.05  # 5% de individuos perturbados por generación
    CONSTANT_PERTURBATION_SIGMA = 0.01 # 1% ruido relativo (|c| * sigma)
    PREVENT_DUPLICATES = True
    
    # Mutation Bank
    MUTATION_BANK_SIZE = 2000
    MUTATION_BANK_REFRESH_INTERVAL = 100  # OPTIMIZED: less overhead (was 50)

    # --- SOTA P2: Library Learning ---
    # Extracts frequently-found, high-fitness subtrees and reuses them
    # as building blocks — injection of proven structural patterns.
    # Inspired by LaSR (Li et al., 2024).
    USE_LIBRARY_LEARNING = True           # Enable library learning
    LIBRARY_MAX_BLOCK_LEN = 8             # Max token length of stored subtrees
    LIBRARY_TOP_K_FRACTION = 0.05        # Top-% of pop to scan for subtrees
    LIBRARY_UPDATE_INTERVAL = 10          # Update library every N generations
    LIBRARY_INJECT_FRACTION = 0.05       # Fraction of mutation bank to fill with library blocks
    LIBRARY_CAPACITY = 512               # Number of slots in the library hash table

    # --- SOTA P1: Pareto Multi-Objective Selection (NSGA-II style) ---
    # Balances RMSE (accuracy) vs tree complexity (parsimony) in selection.
    # PySR uses this natively; AlphaSymbolic now has it too.
    # Non-dominated sort is O(N²) — we run it on a sampled subset (PARETO_SAMPLE_K)
    # and blend the resulting rank into the selection metric.
    USE_PARETO_SELECTION = True        # Enable NSGA-II Pareto blending
    PARETO_SAMPLE_K = 2000             # Subset size for Pareto sort (per island sample)
    PARETO_RANK_WEIGHT = 0.15          # How much Pareto rank adds to selection_metric
    PARETO_INTERVAL = 5                # Run Pareto sort every N generations (amortize cost)


    # ============================================================
    #                  6. EVALUATION & FITNESS
    # ============================================================
    LOSS_FUNCTION = 'RMSE'
    
    # Validations & Penalties
    FORCE_STRICT_VALIDATION = True     # Strict math Mode (No protected operators)
    
    # Penalties
    COMPLEXITY_PENALTY = 0.04          # Raised from 0.02 to combat formula bloat at log-scale errors
    TRIVIAL_FORMULA_PENALTY = 1.5
    NO_VARIABLE_PENALTY = 2.5
    TRIVIAL_FORMULA_MAX_TOKENS = 2
    TRIVIAL_FORMULA_ALLOW_RMSE = 1e-3
    
    # Diversity — ya no necesario: el algoritmo encontró fórmulas con x1/x2 por sí solo.
    # FIX: 0.0 elimina el overhead de escanear 1M fórmulas por var token cada gen,
    # y evita penalizar sub-fórmulas en crossover que momentáneamente no tienen todas las vars.
    # El best tracking ahora usa raw RMSE → acepta cualquier mejora estructural.
    VAR_DIVERSITY_PENALTY = 0.0
    VAR_FORCE_SEED_PERCENT = 0.0    # FIX Bug4: con VAR_DIVERSITY_PENALTY=0 este forced seeding es innecesario y sesga la búsqueda (era 0.25)
    
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
    PSO_STEPS_NORMAL = 40          # OPTIMIZED: más pasos por individuo (was 25→40)
    PSO_STEPS_STAGNATION = 80      # FIX: más pasos para escapar mínimo local (was 40→60→80)
    PSO_K_NORMAL = 400             # OPTIMIZED: menos individuos, más profundidad (was 800→400)
    PSO_K_STAGNATION = 6000        # FIX: más candidatos en plateau real (era 2500; elite fix permite refinamiento más profundo)
    PSO_STAGNATION_THRESHOLD = 10
    # ANTI-STAG: PSO adaptativo. Si el best_rpn no cambió desde el último PSO run,
    # saltar PSO_SKIP_IF_NO_STRUCT_CHANGE generaciones para liberar GPU a exploración.
    PSO_ADAPTIVE = True
    PSO_SKIP_GENS = 6              # ANTI-STAG: saltar PSO cada N gens si no hubo cambio estructural

    # L-BFGS-B Constant Optimizer
    USE_BFGS_OPTIMIZER = True      # Re-enabled to polish constants the PSO can't refine
    BFGS_INTERVAL = 30             # Run L-BFGS-B every 30 gens
    BFGS_TOP_K = 15                # Only refine the absolute top 15 elites island-wide
    BFGS_MAX_ITER = 3
    
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
    # LEXICASE: Disabled — computes [B×D] error matrix every gen = 68MB/gen overhead.
    # Tournament selection with complexity penalty is sufficient.
    USE_LEXICASE_SELECTION = False
    USE_PARETO_SELECTION = True
    PARETO_INTERVAL = 15           # OPTIMIZED: less overhead (was 5)
    PARETO_MAX_FRONT_SIZE = 30
    
    # Pattern Memory
    USE_PATTERN_MEMORY = True
    PATTERN_RECORD_INTERVAL = 30   # OPTIMIZED: less overhead (was 20)
    PATTERN_INJECT_INTERVAL = 25   # OPTIMIZED: less overhead (was 10)
    PATTERN_INJECT_PERCENT = 0.05
    PATTERN_RECORD_FITNESS_THRESHOLD = 10.0
    PATTERN_MEM_MIN_USES = 3
    PATTERN_MIN_SIZE = 3
    PATTERN_MAX_SIZE = 12
    PATTERN_MAX_PATTERNS = 200     # OPTIMIZED: más memoria genética (era 100)
    
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
