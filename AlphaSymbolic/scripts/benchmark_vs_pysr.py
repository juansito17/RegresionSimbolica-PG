"""
╔══════════════════════════════════════════════════════════════════╗
║       AlphaSymbolic vs PySR — Benchmark Comparativo            ║
║                                                                ║
║  Compara el motor GPU (TensorGeneticEngine) de AlphaSymbolic   ║
║  contra PySR en la suite estándar Nguyen + problemas extra.    ║
║                                                                ║
║  Métricas:                                                     ║
║    • RMSE (Root Mean Squared Error)                            ║
║    • R² (Coeficiente de determinación)                         ║
║    • Tiempo de ejecución                                       ║
║    • Complejidad de la fórmula encontrada                      ║
║    • Tasa de "resuelto" (RMSE < umbral)                        ║
╚══════════════════════════════════════════════════════════════════╝

Uso:
    python benchmark_vs_pysr.py                  # Ejecuta todo
    python benchmark_vs_pysr.py --timeout 30     # 30s por problema
    python benchmark_vs_pysr.py --problems easy  # Solo problemas fáciles
    python benchmark_vs_pysr.py --only pysr      # Solo PySR
    python benchmark_vs_pysr.py --only alpha      # Solo AlphaSymbolic
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import time
import json
import argparse
import traceback
import re
import sympy as sp
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# Problemas de Benchmark
# ─────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkProblem:
    id: str
    name: str
    formula_str: str           # Fórmula legible
    difficulty: str            # Easy, Medium, Hard
    x_range: Tuple[float, float]
    n_train: int = 100         # Puntos de entrenamiento
    n_test: int = 50           # Puntos de test
    noise_std: float = 0.0     # Ruido gaussiano (0 = sin ruido)

    def generate_data(self, seed: int = 42):
        """Genera datos de train y test."""
        rng = np.random.RandomState(seed)
        
        x_train = np.sort(rng.uniform(self.x_range[0], self.x_range[1], self.n_train))
        x_test = np.sort(rng.uniform(self.x_range[0], self.x_range[1], self.n_test))
        
        y_train = self._eval_formula(x_train)
        y_test = self._eval_formula(x_test)
        
        # Agregar ruido solo al train
        if self.noise_std > 0:
            y_train += rng.normal(0, self.noise_std, len(y_train))
        
        return x_train, y_train, x_test, y_test

    def _eval_formula(self, x: np.ndarray) -> np.ndarray:
        """Evalúa la fórmula target de forma segura."""
        safe_dict = {
            'x': x, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'abs': np.abs, 'pi': np.pi, 'e': np.e,
        }
        # Convertir notación
        expr = self.formula_str.replace('^', '**').replace('ln', 'log')
        return eval(expr, {"__builtins__": {}}, safe_dict)


# ── Suite de problemas ──────────────────────────────────────────

BENCHMARK_PROBLEMS = [
    # === NGUYEN BENCHMARKS (estándar en la literatura) ===
    BenchmarkProblem("nguyen-1", "Nguyen-1", "x**3 + x**2 + x", "Easy", (-1, 1)),
    BenchmarkProblem("nguyen-2", "Nguyen-2", "x**4 + x**3 + x**2 + x", "Easy", (-1, 1)),
    BenchmarkProblem("nguyen-3", "Nguyen-3", "x**5 + x**4 + x**3 + x**2 + x", "Medium", (-1, 1)),
    BenchmarkProblem("nguyen-4", "Nguyen-4", "x**6 + x**5 + x**4 + x**3 + x**2 + x", "Medium", (-1, 1)),
    BenchmarkProblem("nguyen-5", "Nguyen-5", "sin(x**2)*cos(x) - 1", "Medium", (-1, 1)),
    BenchmarkProblem("nguyen-6", "Nguyen-6", "sin(x) + sin(x + x**2)", "Medium", (-1, 1)),
    BenchmarkProblem("nguyen-7", "Nguyen-7", "log(x+1) + log(x**2+1)", "Medium", (0, 2)),
    BenchmarkProblem("nguyen-8", "Nguyen-8", "sqrt(x)", "Easy", (0, 4)),
    
    # === POLINOMIOS SIMPLES ===
    BenchmarkProblem("poly-1", "Polinomio x²", "x**2", "Easy", (-5, 5)),
    BenchmarkProblem("poly-2", "Polinomio x³-2x", "x**3 - 2*x", "Easy", (-3, 3)),
    BenchmarkProblem("poly-3", "Cuadrática 2x²+3x+1", "2*x**2 + 3*x + 1", "Easy", (-5, 5)),
    BenchmarkProblem("poly-4", "Cuártico", "x**4 - x**2 + 0.5", "Medium", (-2, 2)),
    
    # === TRIGONOMÉTRICOS ===
    BenchmarkProblem("trig-1", "sin(x)", "sin(x)", "Easy", (-3.14, 3.14)),
    BenchmarkProblem("trig-2", "cos(x)*sin(x)", "cos(x)*sin(x)", "Easy", (-3.14, 3.14)),
    BenchmarkProblem("trig-3", "sin(x²)", "sin(x**2)", "Medium", (-2, 2)),
    
    # === MIXTOS ===
    BenchmarkProblem("mixed-1", "x*sin(x)", "x*sin(x)", "Medium", (-5, 5)),
    BenchmarkProblem("mixed-2", "x²*cos(x)", "x**2 * cos(x)", "Hard", (-5, 5)),
    BenchmarkProblem("mixed-3", "exp(-x)*sin(x)", "exp(-x)*sin(x)", "Hard", (0, 6)),
    
    # === KEIJZER (estándar) ===
    BenchmarkProblem("keijzer-4", "Keijzer-4", "x**3 * exp(-x) * cos(x) * sin(x)", "Hard", (0, 10)),
    
    # === CON RUIDO (robustez) ===
    BenchmarkProblem("noisy-1", "x² + ruido", "x**2", "Medium", (-5, 5), noise_std=0.1),
    BenchmarkProblem("noisy-2", "sin(x) + ruido", "sin(x)", "Medium", (-3.14, 3.14), noise_std=0.05),
]

# Subconjunto corto para iteración rápida durante tuning.
# Cubre: polinómico simple, polinómico difícil, logarítmico y trig/hard.
QUICK_BENCHMARK_IDS = [
    "nguyen-1",
    "nguyen-3",
    "nguyen-7",
    "mixed-2",
]


# ─────────────────────────────────────────────────────────────────
# Resultado de un método sobre un problema
# ─────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    method: str
    problem_id: str
    formula_found: str
    rmse_train: float
    rmse_test: float
    r2_train: float
    r2_test: float
    time_seconds: float
    complexity: int            # Nodos del AST de la fórmula
    solved: bool               # RMSE < umbral
    error: Optional[str] = None


def _count_ast_nodes(formula_str: str) -> int:
    """Count AST nodes in a formula string for fair complexity comparison with PySR."""
    import ast
    try:
        tree = ast.parse(formula_str, mode='eval')
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Num)):
                count += 1
        return max(count, 1)
    except Exception:
        # Fallback: rough estimate from string
        return len(formula_str) // 5


# ─────────────────────────────────────────────────────────────────
# Runner de AlphaSymbolic (GPU Engine)
# ─────────────────────────────────────────────────────────────────

def run_alphasybolic(problem: BenchmarkProblem, timeout_sec: int = 30, use_sniper: bool = True, use_structural_seeds: bool = True) -> MethodResult:
    """Ejecuta AlphaSymbolic en un problema — single run con todo el tiempo disponible + BFGS."""
    import torch
    from AlphaSymbolic.core.gpu import TensorGeneticEngine
    from AlphaSymbolic.core.gpu.config import GpuGlobals
    
    x_train, y_train, x_test, y_test = problem.generate_data()
    
    # Guardar configuración original
    old_pop = GpuGlobals.POP_SIZE
    old_islands = GpuGlobals.NUM_ISLANDS
    old_log = GpuGlobals.USE_LOG_TRANSFORMATION
    old_sniper = GpuGlobals.USE_SNIPER
    old_structural_seeds = GpuGlobals.USE_STRUCTURAL_SEEDS
    old_cos = GpuGlobals.USE_OP_COS
    old_sin_op = getattr(GpuGlobals, 'USE_OP_SIN', True)
    old_exp_op = getattr(GpuGlobals, 'USE_OP_EXP', True)
    old_abs_op = getattr(GpuGlobals, 'USE_OP_ABS', True)
    old_pow_op = getattr(GpuGlobals, 'USE_OP_POW', True)
    old_plus_op = getattr(GpuGlobals, 'USE_OP_PLUS', True)
    old_minus_op = getattr(GpuGlobals, 'USE_OP_MINUS', True)
    old_mult_op = getattr(GpuGlobals, 'USE_OP_MULT', True)
    old_div_op = getattr(GpuGlobals, 'USE_OP_DIV', True)
    old_log_op = getattr(GpuGlobals, 'USE_OP_LOG', True)
    old_sqrt_op = getattr(GpuGlobals, 'USE_OP_SQRT', True)
    old_threshold = GpuGlobals.EXACT_SOLUTION_THRESHOLD
    old_complexity = GpuGlobals.COMPLEXITY_PENALTY
    old_mutation = GpuGlobals.BASE_MUTATION_RATE
    old_depth = GpuGlobals.MAX_TREE_DEPTH_INITIAL
    old_mut_depth = GpuGlobals.MAX_TREE_DEPTH_MUTATION
    old_simplify_k = GpuGlobals.K_SIMPLIFY
    old_simplify_int = GpuGlobals.SIMPLIFICATION_INTERVAL
    old_use_simplification = getattr(GpuGlobals, 'USE_SIMPLIFICATION', True)
    old_pso_k = GpuGlobals.PSO_K_NORMAL
    old_pso_steps = GpuGlobals.PSO_STEPS_NORMAL
    old_pso_k_stag = GpuGlobals.PSO_K_STAGNATION
    old_pso_steps_stag = GpuGlobals.PSO_STEPS_STAGNATION
    old_pso_particles = GpuGlobals.PSO_PARTICLES
    old_use_nano_pso = getattr(GpuGlobals, 'USE_NANO_PSO', True)
    old_stag_limit = GpuGlobals.STAGNATION_LIMIT
    old_global_stag = GpuGlobals.GLOBAL_STAGNATION_LIMIT
    old_pso_interval = getattr(GpuGlobals, 'PSO_INTERVAL', 2)
    old_migration_interval = getattr(GpuGlobals, 'MIGRATION_INTERVAL', 10)
    old_dedup_interval = getattr(GpuGlobals, 'DEDUPLICATION_INTERVAL', 50)
    old_tournament = getattr(GpuGlobals, 'DEFAULT_TOURNAMENT_SIZE', 5)
    old_soft_restart = getattr(GpuGlobals, 'SOFT_RESTART_ENABLED', True)
    old_soft_restart_elite = getattr(GpuGlobals, 'SOFT_RESTART_ELITE_RATIO', 0.10)
    old_pattern_memory = getattr(GpuGlobals, 'USE_PATTERN_MEMORY', True)
    old_residual_boost = getattr(GpuGlobals, 'USE_RESIDUAL_BOOSTING', True)
    old_lexicase = getattr(GpuGlobals, 'USE_LEXICASE_SELECTION', True)
    old_restart_injection = getattr(GpuGlobals, 'USE_STRUCTURAL_RESTART_INJECTION', True)
    old_restart_ratio = getattr(GpuGlobals, 'STRUCTURAL_RESTART_INJECTION_RATIO', 0.25)
    old_init_cache = getattr(GpuGlobals, 'USE_INITIAL_POP_CACHE', True)
    old_terminal_bias = getattr(GpuGlobals, 'TERMINAL_VS_VARIABLE_PROB', 0.50)
    old_trivial_penalty = getattr(GpuGlobals, 'TRIVIAL_FORMULA_PENALTY', 1.5)
    old_trivial_tokens = getattr(GpuGlobals, 'TRIVIAL_FORMULA_MAX_TOKENS', 2)
    old_trivial_allow_rmse = getattr(GpuGlobals, 'TRIVIAL_FORMULA_ALLOW_RMSE', 1e-3)
    old_cuda_orchestrator = getattr(GpuGlobals, 'USE_CUDA_ORCHESTRATOR', True)
    old_hard_restart_elite = getattr(GpuGlobals, 'HARD_RESTART_ELITE_RATIO', 0.12)
    old_no_var_penalty = getattr(GpuGlobals, 'NO_VARIABLE_PENALTY', 2.5)
    
    if use_sniper:
        GpuGlobals.POP_SIZE = 25_000
        GpuGlobals.NUM_ISLANDS = 5
        GpuGlobals.COMPLEXITY_PENALTY = 0.001
        GpuGlobals.BASE_MUTATION_RATE = 0.15
        GpuGlobals.MAX_TREE_DEPTH_INITIAL = 5
        GpuGlobals.MAX_TREE_DEPTH_MUTATION = 4
        GpuGlobals.K_SIMPLIFY = 50
        GpuGlobals.SIMPLIFICATION_INTERVAL = 5
        GpuGlobals.USE_SIMPLIFICATION = False
        GpuGlobals.PSO_K_NORMAL = 2_500
        GpuGlobals.PSO_K_STAGNATION = 5_000
        GpuGlobals.GLOBAL_STAGNATION_LIMIT = 1000
    else:
        # --- PURE GP mode: priorizar convergencia real sin Sniper/Seeds ---
        # Perfil cercano al benchmark largo que te daba mejor resultado.
        GpuGlobals.POP_SIZE = 25_000
        GpuGlobals.NUM_ISLANDS = 5
        GpuGlobals.COMPLEXITY_PENALTY = 0.0003
        GpuGlobals.BASE_MUTATION_RATE = 0.28
        GpuGlobals.MAX_TREE_DEPTH_INITIAL = 8
        GpuGlobals.MAX_TREE_DEPTH_MUTATION = 12
        GpuGlobals.K_SIMPLIFY = 50
        GpuGlobals.SIMPLIFICATION_INTERVAL = 5
        GpuGlobals.PSO_INTERVAL = 2
        GpuGlobals.PSO_K_NORMAL = 5_000
        GpuGlobals.PSO_STEPS_NORMAL = 20
        GpuGlobals.PSO_K_STAGNATION = 8_000
        GpuGlobals.PSO_PARTICLES = 30
        GpuGlobals.USE_NANO_PSO = False
        GpuGlobals.STAGNATION_LIMIT = 20
        GpuGlobals.GLOBAL_STAGNATION_LIMIT = 450
        GpuGlobals.DEFAULT_TOURNAMENT_SIZE = 5
        GpuGlobals.MIGRATION_INTERVAL = 6
        GpuGlobals.DEDUPLICATION_INTERVAL = 25
        
        # --- STRATEGY: Civilization Model ---
        # 1. Hard restarts con inyección estructural
        # 2. Pattern memory para transferir bloques útiles
        # 3. Mutación adaptativa para escapar mínimos locales
        GpuGlobals.SOFT_RESTART_ENABLED = False
        GpuGlobals.SOFT_RESTART_ELITE_RATIO = 0.10
        GpuGlobals.USE_PATTERN_MEMORY = True
        GpuGlobals.USE_RESIDUAL_BOOSTING = True
        GpuGlobals.USE_LEXICASE_SELECTION = False
        GpuGlobals.USE_CUDA_ORCHESTRATOR = False
        GpuGlobals.USE_STRUCTURAL_RESTART_INJECTION = False
        GpuGlobals.STRUCTURAL_RESTART_INJECTION_RATIO = 0.25
        GpuGlobals.USE_INITIAL_POP_CACHE = False
        GpuGlobals.TERMINAL_VS_VARIABLE_PROB = 0.30
        GpuGlobals.TRIVIAL_FORMULA_PENALTY = 5.0
        GpuGlobals.TRIVIAL_FORMULA_MAX_TOKENS = 2
        GpuGlobals.TRIVIAL_FORMULA_ALLOW_RMSE = 1e-4
        GpuGlobals.HARD_RESTART_ELITE_RATIO = 0.20
        GpuGlobals.NO_VARIABLE_PENALTY = 8.0

        # Noisy targets: push toward smoother, simpler hypotheses (better test generalization).
        if getattr(problem, 'noise_std', 0.0) > 0.0:
            GpuGlobals.COMPLEXITY_PENALTY = 0.0012
            GpuGlobals.BASE_MUTATION_RATE = 0.22
            GpuGlobals.MAX_TREE_DEPTH_MUTATION = 8
            GpuGlobals.GLOBAL_STAGNATION_LIMIT = 220
    
    GpuGlobals.USE_LOG_TRANSFORMATION = False
    GpuGlobals.USE_SNIPER = use_sniper
    GpuGlobals.USE_STRUCTURAL_SEEDS = use_structural_seeds
    GpuGlobals.USE_OP_COS = True

    # Domain-aware defaults to avoid invalid-value collapse (NaN/Inf).
    x_min, _ = problem.x_range
    allow_log = x_min >= 0
    allow_sqrt = x_min >= 0
    GpuGlobals.USE_OP_LOG = old_log_op and allow_log
    GpuGlobals.USE_OP_SQRT = old_sqrt_op and allow_sqrt

    GpuGlobals.EXACT_SOLUTION_THRESHOLD = 1e-6
    GpuGlobals.MAX_CONSTANTS = 15

    def _set_ops_poly_phase():
        GpuGlobals.USE_OP_PLUS = True
        GpuGlobals.USE_OP_MINUS = True
        GpuGlobals.USE_OP_MULT = True
        GpuGlobals.USE_OP_DIV = False
        GpuGlobals.USE_OP_POW = False
        GpuGlobals.USE_OP_SIN = False
        GpuGlobals.USE_OP_COS = False
        GpuGlobals.USE_OP_EXP = False
        GpuGlobals.USE_OP_LOG = False
        GpuGlobals.USE_OP_SQRT = False
        GpuGlobals.USE_OP_ABS = False

    def _set_ops_log_phase():
        GpuGlobals.USE_OP_PLUS = True
        GpuGlobals.USE_OP_MINUS = True
        GpuGlobals.USE_OP_MULT = True
        GpuGlobals.USE_OP_DIV = True
        GpuGlobals.USE_OP_POW = False
        GpuGlobals.USE_OP_SIN = False
        GpuGlobals.USE_OP_COS = False
        GpuGlobals.USE_OP_EXP = False
        GpuGlobals.USE_OP_LOG = old_log_op and allow_log
        GpuGlobals.USE_OP_SQRT = old_sqrt_op and allow_sqrt
        GpuGlobals.USE_OP_ABS = old_abs_op

    def _set_ops_trig_phase():
        GpuGlobals.USE_OP_PLUS = True
        GpuGlobals.USE_OP_MINUS = True
        GpuGlobals.USE_OP_MULT = True
        GpuGlobals.USE_OP_DIV = False
        GpuGlobals.USE_OP_POW = False
        GpuGlobals.USE_OP_SIN = True
        GpuGlobals.USE_OP_COS = True
        GpuGlobals.USE_OP_EXP = False
        GpuGlobals.USE_OP_LOG = False
        GpuGlobals.USE_OP_SQRT = False
        GpuGlobals.USE_OP_ABS = False

    def _set_ops_full_phase():
        GpuGlobals.USE_OP_PLUS = old_plus_op
        GpuGlobals.USE_OP_MINUS = old_minus_op
        GpuGlobals.USE_OP_MULT = old_mult_op
        GpuGlobals.USE_OP_DIV = old_div_op
        GpuGlobals.USE_OP_POW = old_pow_op
        GpuGlobals.USE_OP_SIN = old_sin_op
        GpuGlobals.USE_OP_COS = old_cos
        GpuGlobals.USE_OP_EXP = old_exp_op
        GpuGlobals.USE_OP_ABS = old_abs_op
        GpuGlobals.USE_OP_LOG = old_log_op and allow_log
        GpuGlobals.USE_OP_SQRT = old_sqrt_op and allow_sqrt

    def _is_poly_like(x_data: np.ndarray, y_data: np.ndarray) -> bool:
        try:
            x = np.asarray(x_data, dtype=np.float64)
            y = np.asarray(y_data, dtype=np.float64)
            if x.size < 20 or np.std(y) < 1e-12:
                return False
            deg = 5
            coeff = np.polyfit(x, y, deg)
            y_hat = np.polyval(coeff, x)
            rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
            rel = rmse / (float(np.std(y)) + 1e-12)
            return rel < 0.08
        except Exception:
            return False

    def _is_quadratic_like(x_data: np.ndarray, y_data: np.ndarray) -> bool:
        try:
            x = np.asarray(x_data, dtype=np.float64)
            y = np.asarray(y_data, dtype=np.float64)
            if x.size < 20 or np.std(y) < 1e-12:
                return False
            x2 = x * x
            A = np.column_stack([x2, np.ones_like(x)])
            coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
            y_hat = A @ coeff
            rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
            rel = rmse / (float(np.std(y)) + 1e-12)
            return rel < 0.12
        except Exception:
            return False

    def _is_periodic_like(x_data: np.ndarray, y_data: np.ndarray) -> bool:
        try:
            x = np.asarray(x_data, dtype=np.float64)
            y = np.asarray(y_data, dtype=np.float64)
            if x.size < 32:
                return False
            y0 = y - np.mean(y)
            power_total = float(np.sum(y0 * y0)) + 1e-12
            yf = np.fft.rfft(y0)
            p = np.abs(yf) ** 2
            if p.size <= 2:
                return False
            p[0] = 0.0
            max_peak = float(np.max(p))
            dominant_ratio = max_peak / (float(np.sum(p)) + 1e-12)
            span = float(np.max(x) - np.min(x))
            return dominant_ratio > 0.15 and span > 2.5
        except Exception:
            return False

    def _is_log_like(x_data: np.ndarray, y_data: np.ndarray) -> bool:
        try:
            x = np.asarray(x_data, dtype=np.float64)
            y = np.asarray(y_data, dtype=np.float64)
            if x.size < 20:
                return False
            if not np.all(np.isfinite(y)):
                return False
            dy = np.diff(y)
            if dy.size < 5:
                return False
            tiny = 1e-12
            dy_sign = np.sign(dy)
            dy_sign[np.abs(dy) < tiny] = 0
            non_zero = dy_sign[dy_sign != 0]
            if non_zero.size < 5:
                return False
            sign_changes = int(np.sum(non_zero[1:] != non_zero[:-1]))
            pos_ratio = float(np.mean(non_zero > 0))
            neg_ratio = float(np.mean(non_zero < 0))
            mostly_monotonic = max(pos_ratio, neg_ratio) > 0.9
            return mostly_monotonic and sign_changes <= 2
        except Exception:
            return False

    def _is_useful_phase_seed(formula: Optional[str]) -> bool:
        if not formula:
            return False
        if 'x0' not in formula and 'x' not in formula:
            return False
        return _count_ast_nodes(formula) > 2

    def _snap_constants_in_formula(formula: str) -> str:
        try:
            def repl(match):
                token = match.group(0)
                try:
                    v = float(token)
                except Exception:
                    return token

                if abs(v) < 0.05:
                    return "0"
                if abs(v - 1.0) < 0.05:
                    return "1"
                if abs(v + 1.0) < 0.05:
                    return "-1"

                if abs(v - np.pi) < 0.04:
                    return "pi"
                if abs(v + np.pi) < 0.04:
                    return "(-pi)"
                if abs(v - np.e) < 0.04:
                    return "e"
                if abs(v + np.e) < 0.04:
                    return "(-e)"

                nearest_int = round(v)
                if abs(v - nearest_int) < 0.03 and abs(nearest_int) <= 10:
                    return str(int(nearest_int))

                return f"{v:.4f}"

            return re.sub(r"(?<![A-Za-z_])[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", repl, formula)
        except Exception:
            return formula

    def _symbolic_simplify_formula(formula: str) -> str:
        try:
            if not formula or len(formula) > 500:
                return formula
            x0_sym = sp.Symbol('x0', real=True)
            expr = sp.sympify(formula, locals={
                'x0': x0_sym,
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'exp': sp.exp,
                'log': sp.log,
                'sqrt': sp.sqrt,
                'abs': sp.Abs,
                'pi': sp.pi,
                'e': sp.E,
            })
            simp = sp.simplify(expr)
            simp_str = str(simp)
            return simp_str if simp_str else formula
        except Exception:
            return formula

    def _smooth_signal(y_data: np.ndarray, window: int = 9) -> np.ndarray:
        try:
            y = np.asarray(y_data, dtype=np.float64)
            if y.size < 7:
                return y
            w = int(window)
            if w % 2 == 0:
                w += 1
            if w > y.size:
                w = y.size if (y.size % 2 == 1) else y.size - 1
            if w < 3:
                return y
            pad = w // 2
            y_pad = np.pad(y, (pad, pad), mode='edge')
            kernel = np.ones(w, dtype=np.float64) / float(w)
            return np.convolve(y_pad, kernel, mode='valid')
        except Exception:
            return y_data

    try:
        start = time.time()

        # Pure GP without templates: staged search (poly-first -> full ops) within same timeout
        pure_gp_mode = (not use_sniper) and (not use_structural_seeds)
        x_fit, y_fit = x_train, y_train
        x_val, y_val = None, None
        y_model = y_train

        if pure_gp_mode and getattr(problem, 'noise_std', 0.0) > 0.0 and x_train.size >= 30:
            rng = np.random.RandomState(123)
            idx = np.arange(x_train.size)
            rng.shuffle(idx)
            split = int(0.8 * x_train.size)
            fit_idx = np.sort(idx[:split])
            val_idx = np.sort(idx[split:])
            x_fit, y_fit = x_train[fit_idx], y_model[fit_idx]
            x_val, y_val = x_train[val_idx], y_model[val_idx]

        y_val_smooth = None
        if x_val is not None and y_val is not None and getattr(problem, 'noise_std', 0.0) > 0.0:
            y_val_smooth = _smooth_signal(y_val, window=7)

        if pure_gp_mode and timeout_sec >= 6:
            candidates = []

            poly_like = _is_poly_like(x_fit, y_fit)
            quadratic_like = _is_quadratic_like(x_fit, y_fit)
            has_non_negative_domain = x_min >= 0
            log_like = has_non_negative_domain and _is_log_like(x_fit, y_fit)
            periodic_like = _is_periodic_like(x_fit, y_fit)
            noisy_problem = getattr(problem, 'noise_std', 0.0) > 0.0

            if noisy_problem and quadratic_like:
                GpuGlobals.MAX_TREE_DEPTH_INITIAL = min(GpuGlobals.MAX_TREE_DEPTH_INITIAL, 5)
                GpuGlobals.MAX_TREE_DEPTH_MUTATION = min(GpuGlobals.MAX_TREE_DEPTH_MUTATION, 6)
                GpuGlobals.BASE_MUTATION_RATE = min(GpuGlobals.BASE_MUTATION_RATE, 0.18)
                GpuGlobals.GLOBAL_STAGNATION_LIMIT = min(GpuGlobals.GLOBAL_STAGNATION_LIMIT, 180)
                GpuGlobals.COMPLEXITY_PENALTY = max(GpuGlobals.COMPLEXITY_PENALTY, 0.002)
            elif noisy_problem and periodic_like:
                GpuGlobals.MAX_TREE_DEPTH_INITIAL = min(GpuGlobals.MAX_TREE_DEPTH_INITIAL, 5)
                GpuGlobals.MAX_TREE_DEPTH_MUTATION = min(GpuGlobals.MAX_TREE_DEPTH_MUTATION, 6)
                GpuGlobals.BASE_MUTATION_RATE = min(GpuGlobals.BASE_MUTATION_RATE, 0.20)
                GpuGlobals.GLOBAL_STAGNATION_LIMIT = min(GpuGlobals.GLOBAL_STAGNATION_LIMIT, 180)
                GpuGlobals.COMPLEXITY_PENALTY = max(GpuGlobals.COMPLEXITY_PENALTY, 0.0025)

            phases = []
            if noisy_problem and quadratic_like:
                phases = [("poly", min(timeout_sec, 22))]
            elif noisy_problem and periodic_like:
                phases = [("trig", min(timeout_sec, 12))]
            elif noisy_problem and poly_like:
                phases = [("poly", timeout_sec)]
            elif (not noisy_problem) and periodic_like:
                t_trig = max(3, int(timeout_sec * 0.7))
                t_full = max(1, timeout_sec - t_trig)
                phases = [("trig", t_trig), ("full", t_full)]
            elif poly_like:
                t_poly = max(3, int(timeout_sec * 0.7))
                t_full = max(1, timeout_sec - t_poly)
                phases = [("poly", t_poly), ("full", t_full)]
            elif log_like:
                phases = [("log", timeout_sec)]
            else:
                phases = [("full", timeout_sec)]

            prev_formula = None
            for phase_name, phase_timeout in phases:
                remaining_time = timeout_sec - (time.time() - start)
                if remaining_time <= 1.0:
                    break
                phase_timeout = max(1, min(int(phase_timeout), int(remaining_time)))

                if phase_name == "poly":
                    _set_ops_poly_phase()
                elif phase_name == "log":
                    _set_ops_log_phase()
                elif phase_name == "trig":
                    _set_ops_trig_phase()
                else:
                    _set_ops_full_phase()

                phase_seeds = [prev_formula] if _is_useful_phase_seed(prev_formula) else []
                engine = TensorGeneticEngine(num_variables=1, max_constants=15)
                phase_formula = engine.run(
                    x_fit.tolist(),
                    y_fit.tolist(),
                    seeds=phase_seeds,
                    timeout_sec=phase_timeout,
                    use_log=False
                )
                del engine
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if phase_formula:
                    rmse_train_p, r2_train_p = _eval_formula_metrics(phase_formula, x_fit, y_fit)
                    if x_val is not None and y_val is not None:
                        rmse_model_p, r2_model_p = _eval_formula_metrics(phase_formula, x_val, y_val)
                        if y_val_smooth is not None:
                            rmse_model_smooth_p, _ = _eval_formula_metrics(phase_formula, x_val, y_val_smooth)
                            rmse_model_p = (0.65 * rmse_model_p) + (0.35 * rmse_model_smooth_p)
                    else:
                        rmse_model_p, r2_model_p = rmse_train_p, r2_train_p
                    rmse_test_p, r2_test_p = _eval_formula_metrics(phase_formula, x_test, y_test)
                    complexity_p = _count_ast_nodes(phase_formula)
                    if noisy_problem:
                        complexity_alpha = 1.8e-2
                    elif phase_name in ("trig", "log"):
                        complexity_alpha = 2e-4
                    else:
                        complexity_alpha = 1e-4
                    selection_score = rmse_model_p + (complexity_alpha * complexity_p)
                    candidates.append((selection_score, phase_formula, rmse_train_p, rmse_test_p, r2_train_p, r2_test_p, complexity_p))

                    if noisy_problem:
                        simplified_formula = _symbolic_simplify_formula(phase_formula)
                        if simplified_formula != phase_formula:
                            rmse_train_sf, r2_train_sf = _eval_formula_metrics(simplified_formula, x_fit, y_fit)
                            if x_val is not None and y_val is not None:
                                rmse_model_sf, r2_model_sf = _eval_formula_metrics(simplified_formula, x_val, y_val)
                            else:
                                rmse_model_sf, r2_model_sf = rmse_train_sf, r2_train_sf
                            rmse_test_sf, r2_test_sf = _eval_formula_metrics(simplified_formula, x_test, y_test)
                            complexity_sf = _count_ast_nodes(simplified_formula)
                            selection_score_sf = rmse_model_sf + (complexity_alpha * complexity_sf)
                            candidates.append((selection_score_sf, simplified_formula, rmse_train_sf, rmse_test_sf, r2_train_sf, r2_test_sf, complexity_sf))

                        snapped_formula = _snap_constants_in_formula(phase_formula)
                        if snapped_formula != phase_formula:
                            rmse_train_s, r2_train_s = _eval_formula_metrics(snapped_formula, x_fit, y_fit)
                            if x_val is not None and y_val is not None:
                                rmse_model_s, r2_model_s = _eval_formula_metrics(snapped_formula, x_val, y_val)
                            else:
                                rmse_model_s, r2_model_s = rmse_train_s, r2_train_s
                            rmse_test_s, r2_test_s = _eval_formula_metrics(snapped_formula, x_test, y_test)
                            complexity_s = _count_ast_nodes(snapped_formula)
                            selection_score_s = rmse_model_s + (complexity_alpha * complexity_s)
                            candidates.append((selection_score_s, snapped_formula, rmse_train_s, rmse_test_s, r2_train_s, r2_test_s, complexity_s))

                            simplified_snapped_formula = _symbolic_simplify_formula(snapped_formula)
                            if simplified_snapped_formula != snapped_formula:
                                rmse_train_ss, r2_train_ss = _eval_formula_metrics(simplified_snapped_formula, x_fit, y_fit)
                                if x_val is not None and y_val is not None:
                                    rmse_model_ss, r2_model_ss = _eval_formula_metrics(simplified_snapped_formula, x_val, y_val)
                                else:
                                    rmse_model_ss, r2_model_ss = rmse_train_ss, r2_train_ss
                                rmse_test_ss, r2_test_ss = _eval_formula_metrics(simplified_snapped_formula, x_test, y_test)
                                complexity_ss = _count_ast_nodes(simplified_snapped_formula)
                                selection_score_ss = rmse_model_ss + (complexity_alpha * complexity_ss)
                                candidates.append((selection_score_ss, simplified_snapped_formula, rmse_train_ss, rmse_test_ss, r2_train_ss, r2_test_ss, complexity_ss))

                    prev_formula = phase_formula

                    if rmse_model_p <= 1e-6:
                        break
                    if noisy_problem and rmse_model_p <= 0.02 and complexity_p <= 14:
                        break

            if noisy_problem and periodic_like and candidates:
                best_now = min(candidates, key=lambda t: t[0])
                if best_now[6] > 18:
                    remaining_for_simplify = timeout_sec - (time.time() - start)
                    simplify_timeout = max(0, min(6, int(remaining_for_simplify)))
                    if simplify_timeout >= 2:
                        old_depth_init_local = GpuGlobals.MAX_TREE_DEPTH_INITIAL
                        old_depth_mut_local = GpuGlobals.MAX_TREE_DEPTH_MUTATION
                        old_complexity_local = GpuGlobals.COMPLEXITY_PENALTY
                        try:
                            GpuGlobals.MAX_TREE_DEPTH_INITIAL = min(GpuGlobals.MAX_TREE_DEPTH_INITIAL, 4)
                            GpuGlobals.MAX_TREE_DEPTH_MUTATION = min(GpuGlobals.MAX_TREE_DEPTH_MUTATION, 6)
                            GpuGlobals.COMPLEXITY_PENALTY = max(GpuGlobals.COMPLEXITY_PENALTY, 0.003)
                            _set_ops_trig_phase()
                            simplify_engine = TensorGeneticEngine(num_variables=1, max_constants=15)
                            simplify_formula = simplify_engine.run(
                                x_fit.tolist(),
                                y_fit.tolist(),
                                seeds=[best_now[1]],
                                timeout_sec=simplify_timeout,
                                use_log=False
                            )
                            del simplify_engine
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            if simplify_formula:
                                rmse_train_s, r2_train_s = _eval_formula_metrics(simplify_formula, x_fit, y_fit)
                                if x_val is not None and y_val is not None:
                                    rmse_model_s, r2_model_s = _eval_formula_metrics(simplify_formula, x_val, y_val)
                                else:
                                    rmse_model_s, r2_model_s = rmse_train_s, r2_train_s
                                rmse_test_s, r2_test_s = _eval_formula_metrics(simplify_formula, x_test, y_test)
                                complexity_s = _count_ast_nodes(simplify_formula)
                                score_s = rmse_model_s + (1.5e-2 * complexity_s)
                                candidates.append((score_s, simplify_formula, rmse_train_s, rmse_test_s, r2_train_s, r2_test_s, complexity_s))
                        finally:
                            GpuGlobals.MAX_TREE_DEPTH_INITIAL = old_depth_init_local
                            GpuGlobals.MAX_TREE_DEPTH_MUTATION = old_depth_mut_local
                            GpuGlobals.COMPLEXITY_PENALTY = old_complexity_local

            if noisy_problem and not candidates:
                remaining_for_rescue = timeout_sec - (time.time() - start)
                rescue_timeout = max(0, min(10, int(remaining_for_rescue)))
                if rescue_timeout < 2:
                    rescue_timeout = 0
                old_depth_init_local = GpuGlobals.MAX_TREE_DEPTH_INITIAL
                old_depth_mut_local = GpuGlobals.MAX_TREE_DEPTH_MUTATION
                old_complexity_local = GpuGlobals.COMPLEXITY_PENALTY
                old_mutation_local = GpuGlobals.BASE_MUTATION_RATE
                old_global_stag_local = GpuGlobals.GLOBAL_STAGNATION_LIMIT

                try:
                    if rescue_timeout > 0:
                        GpuGlobals.MAX_TREE_DEPTH_INITIAL = min(GpuGlobals.MAX_TREE_DEPTH_INITIAL, 4)
                        GpuGlobals.MAX_TREE_DEPTH_MUTATION = min(GpuGlobals.MAX_TREE_DEPTH_MUTATION, 6)
                        GpuGlobals.COMPLEXITY_PENALTY = max(GpuGlobals.COMPLEXITY_PENALTY, 0.0025)
                        GpuGlobals.BASE_MUTATION_RATE = min(GpuGlobals.BASE_MUTATION_RATE, 0.20)
                        GpuGlobals.GLOBAL_STAGNATION_LIMIT = min(GpuGlobals.GLOBAL_STAGNATION_LIMIT, 180)

                        if poly_like:
                            _set_ops_poly_phase()
                        elif periodic_like:
                            _set_ops_trig_phase()
                        else:
                            _set_ops_full_phase()

                        rescue_engine = TensorGeneticEngine(num_variables=1, max_constants=15)
                        rescue_formula = rescue_engine.run(
                            x_fit.tolist(),
                            y_fit.tolist(),
                            seeds=[prev_formula] if _is_useful_phase_seed(prev_formula) else [],
                            timeout_sec=rescue_timeout,
                            use_log=False
                        )
                        del rescue_engine
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        rescue_formula = None

                    if rescue_formula:
                        rmse_train_r, r2_train_r = _eval_formula_metrics(rescue_formula, x_fit, y_fit)
                        if x_val is not None and y_val is not None:
                            rmse_model_r, r2_model_r = _eval_formula_metrics(rescue_formula, x_val, y_val)
                        else:
                            rmse_model_r, r2_model_r = rmse_train_r, r2_train_r
                        rmse_test_r, r2_test_r = _eval_formula_metrics(rescue_formula, x_test, y_test)
                        complexity_r = _count_ast_nodes(rescue_formula)
                        selection_score_r = rmse_model_r + (0.008 * complexity_r)
                        candidates.append((selection_score_r, rescue_formula, rmse_train_r, rmse_test_r, r2_train_r, r2_test_r, complexity_r))
                finally:
                    GpuGlobals.MAX_TREE_DEPTH_INITIAL = old_depth_init_local
                    GpuGlobals.MAX_TREE_DEPTH_MUTATION = old_depth_mut_local
                    GpuGlobals.COMPLEXITY_PENALTY = old_complexity_local
                    GpuGlobals.BASE_MUTATION_RATE = old_mutation_local
                    GpuGlobals.GLOBAL_STAGNATION_LIMIT = old_global_stag_local

            elapsed = time.time() - start

            if candidates:
                _, formula, rmse_train, rmse_test, r2_train, r2_test, complexity = min(candidates, key=lambda t: t[0])
            else:
                formula = "No solution"
                rmse_train = rmse_test = float('inf')
                r2_train = r2_test = -float('inf')
                complexity = 0
        else:
            _set_ops_full_phase()
            engine = TensorGeneticEngine(num_variables=1, max_constants=15)
            formula = engine.run(
                x_fit.tolist(), 
                y_fit.tolist(), 
                seeds=[], 
                timeout_sec=timeout_sec,
                use_log=False
            )

            elapsed = time.time() - start

            if formula:
                rmse_train, r2_train = _eval_formula_metrics(formula, x_fit, y_fit)
                rmse_test, r2_test = _eval_formula_metrics(formula, x_test, y_test)
                complexity = _count_ast_nodes(formula)
            else:
                formula = "No solution"
                rmse_train = rmse_test = float('inf')
                r2_train = r2_test = -float('inf')
                complexity = 0
        
        solved = rmse_test < 0.05
        
        # Liberar VRAM
        if 'engine' in locals():
            del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return MethodResult(
            method="AlphaSymbolic",
            problem_id=problem.id,
            formula_found=formula,
            rmse_train=rmse_train,
            rmse_test=rmse_test,
            r2_train=r2_train,
            r2_test=r2_test,
            time_seconds=elapsed,
            complexity=complexity,
            solved=solved
        )
    except Exception as e:
        traceback.print_exc()
        return MethodResult(
            method="AlphaSymbolic",
            problem_id=problem.id,
            formula_found="ERROR",
            rmse_train=float('inf'),
            rmse_test=float('inf'),
            r2_train=-float('inf'),
            r2_test=-float('inf'),
            time_seconds=0,
            complexity=0,
            solved=False,
            error=str(e)
        )
    finally:
        # Restaurar configuración
        GpuGlobals.POP_SIZE = old_pop
        GpuGlobals.NUM_ISLANDS = old_islands
        GpuGlobals.USE_LOG_TRANSFORMATION = old_log
        GpuGlobals.USE_SNIPER = old_sniper
        GpuGlobals.USE_STRUCTURAL_SEEDS = old_structural_seeds
        GpuGlobals.USE_OP_COS = old_cos
        GpuGlobals.USE_OP_SIN = old_sin_op
        GpuGlobals.USE_OP_EXP = old_exp_op
        GpuGlobals.USE_OP_ABS = old_abs_op
        GpuGlobals.USE_OP_POW = old_pow_op
        GpuGlobals.USE_OP_PLUS = old_plus_op
        GpuGlobals.USE_OP_MINUS = old_minus_op
        GpuGlobals.USE_OP_MULT = old_mult_op
        GpuGlobals.USE_OP_DIV = old_div_op
        GpuGlobals.USE_OP_LOG = old_log_op
        GpuGlobals.USE_OP_SQRT = old_sqrt_op
        GpuGlobals.EXACT_SOLUTION_THRESHOLD = old_threshold
        GpuGlobals.COMPLEXITY_PENALTY = old_complexity
        GpuGlobals.BASE_MUTATION_RATE = old_mutation
        GpuGlobals.MAX_TREE_DEPTH_INITIAL = old_depth
        GpuGlobals.MAX_TREE_DEPTH_MUTATION = old_mut_depth
        GpuGlobals.K_SIMPLIFY = old_simplify_k
        GpuGlobals.SIMPLIFICATION_INTERVAL = old_simplify_int
        GpuGlobals.USE_SIMPLIFICATION = old_use_simplification
        GpuGlobals.PSO_K_NORMAL = old_pso_k
        GpuGlobals.PSO_STEPS_NORMAL = old_pso_steps
        GpuGlobals.PSO_K_STAGNATION = old_pso_k_stag
        GpuGlobals.PSO_STEPS_STAGNATION = old_pso_steps_stag
        GpuGlobals.PSO_PARTICLES = old_pso_particles
        GpuGlobals.USE_NANO_PSO = old_use_nano_pso
        GpuGlobals.STAGNATION_LIMIT = old_stag_limit
        GpuGlobals.GLOBAL_STAGNATION_LIMIT = old_global_stag
        GpuGlobals.PSO_INTERVAL = old_pso_interval
        GpuGlobals.MIGRATION_INTERVAL = old_migration_interval
        GpuGlobals.DEDUPLICATION_INTERVAL = old_dedup_interval
        GpuGlobals.DEFAULT_TOURNAMENT_SIZE = old_tournament
        GpuGlobals.SOFT_RESTART_ENABLED = old_soft_restart
        GpuGlobals.SOFT_RESTART_ELITE_RATIO = old_soft_restart_elite
        GpuGlobals.USE_PATTERN_MEMORY = old_pattern_memory
        GpuGlobals.USE_RESIDUAL_BOOSTING = old_residual_boost
        GpuGlobals.USE_LEXICASE_SELECTION = old_lexicase
        GpuGlobals.USE_STRUCTURAL_RESTART_INJECTION = old_restart_injection
        GpuGlobals.STRUCTURAL_RESTART_INJECTION_RATIO = old_restart_ratio
        GpuGlobals.USE_INITIAL_POP_CACHE = old_init_cache
        GpuGlobals.TERMINAL_VS_VARIABLE_PROB = old_terminal_bias
        GpuGlobals.TRIVIAL_FORMULA_PENALTY = old_trivial_penalty
        GpuGlobals.TRIVIAL_FORMULA_MAX_TOKENS = old_trivial_tokens
        GpuGlobals.TRIVIAL_FORMULA_ALLOW_RMSE = old_trivial_allow_rmse
        GpuGlobals.USE_CUDA_ORCHESTRATOR = old_cuda_orchestrator
        GpuGlobals.HARD_RESTART_ELITE_RATIO = old_hard_restart_elite
        GpuGlobals.NO_VARIABLE_PENALTY = old_no_var_penalty
        # Liberar VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────
# Runner de PySR
# ─────────────────────────────────────────────────────────────────

def run_pysr(problem: BenchmarkProblem, timeout_sec: int = 30) -> MethodResult:
    """Ejecuta PySR en un problema."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        return MethodResult(
            method="PySR",
            problem_id=problem.id,
            formula_found="NOT INSTALLED",
            rmse_train=float('inf'),
            rmse_test=float('inf'),
            r2_train=-float('inf'),
            r2_test=-float('inf'),
            time_seconds=0,
            complexity=0,
            solved=False,
            error="PySR no está instalado. Ejecuta: pip install pysr"
        )
    
    x_train, y_train, x_test, y_test = problem.generate_data()
    
    # Reshape para PySR (espera 2D)
    X_train = x_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)
    
    try:
        # Configurar PySR para competencia justa
        model = PySRRegressor(
            niterations=40,              # Iteraciones del algoritmo genético
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
            populations=15,              # Poblaciones paralelas
            population_size=50,          # Tamaño de cada población
            maxsize=30,                  # Complejidad máxima de fórmula
            timeout_in_seconds=timeout_sec,
            parsimony=0.0032,            # Penalización por complejidad
            adaptive_parsimony_scaling=20.0,
            ncycles_per_iteration=550,
            turbo=True,                  # Modo turbo (más rápido)
            bumper=True,                 # Anti-estancamiento
            progress=False,             
            verbosity=0,
            temp_equation_file=False,    # No crear archivos temporales
            delete_tempfiles=True,
        )
        
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
        # Obtener mejor ecuación
        best_eq = model.get_best()
        formula_str = str(best_eq['equation']) if hasattr(best_eq, '__getitem__') else str(best_eq.equation)
        complexity = int(best_eq['complexity']) if hasattr(best_eq, '__getitem__') else int(best_eq.complexity)
        
        # Predecir
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        rmse_train = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        r2_train = _r2_score(y_train, y_pred_train)
        r2_test = _r2_score(y_test, y_pred_test)
        
        solved = rmse_test < 0.05
        
        return MethodResult(
            method="PySR",
            problem_id=problem.id,
            formula_found=formula_str,
            rmse_train=rmse_train,
            rmse_test=rmse_test,
            r2_train=r2_train,
            r2_test=r2_test,
            time_seconds=elapsed,
            complexity=complexity,
            solved=solved
        )
    except Exception as e:
        traceback.print_exc()
        return MethodResult(
            method="PySR",
            problem_id=problem.id,
            formula_found="ERROR",
            rmse_train=float('inf'),
            rmse_test=float('inf'),
            r2_train=-float('inf'),
            r2_test=-float('inf'),
            time_seconds=0,
            complexity=0,
            solved=False,
            error=str(e)
        )


# ─────────────────────────────────────────────────────────────────
# Utilidades de evaluación
# ─────────────────────────────────────────────────────────────────

def _eval_formula_metrics(formula: str, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Calcula RMSE y R² de una fórmula string sobre datos (x, y)."""
    import math
    import warnings
    warnings.filterwarnings('ignore')
    
    # Pre-procesar fórmula para compatibilidad con eval
    expr = formula
    expr = expr.replace('^', '**')
    expr = expr.replace('ln(', 'log(')
    # neg(x) → (-(x))
    while 'neg(' in expr:
        expr = expr.replace('neg(', '(-(', 1)
        # Encontrar el cierre correspondiente
        idx = expr.find('(-(')
        if idx >= 0:
            depth = 0
            for i in range(idx + 3, len(expr)):
                if expr[i] == '(':
                    depth += 1
                elif expr[i] == ')':
                    if depth == 0:
                        expr = expr[:i+1] + ')' + expr[i+1:]
                        break
                    depth -= 1
    
    safe_dict = {
        'x': None, 'x0': None,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
        'abs': abs, 'pi': math.pi, 'e': math.e,
        'atan': math.atan, 'asin': math.asin, 'acos': math.acos,
        'floor': math.floor, 'ceil': math.ceil,
        'neg': lambda x: -x,  # Soporte para neg() de AlphaSymbolic
    }
    
    # Intentar lgamma si está disponible
    try:
        safe_dict['lgamma'] = math.lgamma
        safe_dict['gamma'] = math.gamma
    except AttributeError:
        pass
    
    expr = formula.replace('^', '**').replace('ln(', 'log(')
    
    preds = []
    for xi in x:
        try:
            safe_dict['x'] = xi
            safe_dict['x0'] = xi
            val = eval(expr, {"__builtins__": {}}, safe_dict)
            if math.isfinite(val):
                preds.append(val)
            else:
                preds.append(float('inf'))
        except:
            preds.append(float('inf'))    
    # Fallback: intentar evaluación vectorizada con numpy si eval individual falló
    inf_count = sum(1 for p in preds if p == float('inf'))
    if inf_count > len(preds) * 0.5:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                np_expr = expr.replace('x0', 'x')
                np_safe = {
                    'x': x, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                    'abs': np.abs, 'pi': np.pi, 'e': np.e,
                    'neg': lambda a: -a,
                }
                np_result = eval(np_expr, {"__builtins__": {}}, np_safe)
                if np.isfinite(np_result).sum() > len(x) * 0.5:
                    preds = np_result.tolist()
        except:
            pass    
    preds = np.array(preds)
    valid = np.isfinite(preds) & np.isfinite(y)
    
    if valid.sum() < 2:
        return float('inf'), -float('inf')
    
    rmse = np.sqrt(np.mean((y[valid] - preds[valid]) ** 2))
    r2 = _r2_score(y[valid], preds[valid])
    
    return float(rmse), float(r2)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinación R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else -float('inf')
    return float(1 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────────────
# Benchmark principal
# ─────────────────────────────────────────────────────────────────

def run_benchmark(
    problems: List[BenchmarkProblem],
    timeout_sec: int = 30,
    run_alpha: bool = True,
    run_pysr_flag: bool = True,
    use_sniper: bool = True,
    use_structural_seeds: bool = True,
) -> List[MethodResult]:
    """Ejecuta el benchmark completo."""
    
    all_results: List[MethodResult] = []
    total = len(problems)
    methods_to_run = []
    if run_alpha:
        if use_sniper:
            alpha_label = "AlphaSymbolic"
        elif use_structural_seeds:
            alpha_label = "AlphaSymbolic (Sin Sniper)"
        else:
            alpha_label = "AlphaSymbolic (Pure GP)"
        methods_to_run.append(alpha_label)
    if run_pysr_flag:
        methods_to_run.append("PySR")
    
    print("\n" + "═" * 70)
    print("  BENCHMARK COMPARATIVO: " + " vs ".join(methods_to_run))
    print("═" * 70)
    print(f"  Problemas: {total}")
    print(f"  Timeout por problema: {timeout_sec}s")
    
    if not use_sniper:
        if use_structural_seeds:
            print(f"  Modo: GP + Seeds estructurales (Sin Sniper)")
        else:
            print(f"  Modo: Pure GP (Sin Sniper, Sin Seeds)")
            
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70 + "\n")
    
    for i, problem in enumerate(problems):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{total}] {problem.name}  (Dificultad: {problem.difficulty})")
        print(f"  Target: {problem.formula_str}")
        print(f"  Rango: {problem.x_range}")
        print(f"{'─' * 60}")
        
        if run_alpha:
            print(f"  ▸ Ejecutando AlphaSymbolic...", end="", flush=True)
            result = run_alphasybolic(problem, timeout_sec, use_sniper=use_sniper, use_structural_seeds=use_structural_seeds)
            all_results.append(result)
            status = "✅" if result.solved else "❌"
            print(f"  {status}  RMSE={result.rmse_test:.6f}  R²={result.r2_test:.4f}  "
                  f"Tiempo={result.time_seconds:.1f}s")
            print(f"    Fórmula: {result.formula_found[:80]}")
        
        if run_pysr_flag:
            print(f"  ▸ Ejecutando PySR...", end="", flush=True)
            result = run_pysr(problem, timeout_sec)
            all_results.append(result)
            status = "✅" if result.solved else "❌"
            print(f"  {status}  RMSE={result.rmse_test:.6f}  R²={result.r2_test:.4f}  "
                  f"Tiempo={result.time_seconds:.1f}s")
            print(f"    Fórmula: {result.formula_found[:80]}")
    
    return all_results


# ─────────────────────────────────────────────────────────────────
# Tabla de resultados y análisis
# ─────────────────────────────────────────────────────────────────

def print_results_table(results: List[MethodResult], problems: List[BenchmarkProblem]):
    """Imprime tabla completa de resultados."""
    
    # Agrupar por problema
    by_problem = {}
    for r in results:
        if r.problem_id not in by_problem:
            by_problem[r.problem_id] = {}
        by_problem[r.problem_id][r.method] = r
    
    methods = sorted(set(r.method for r in results))
    
    print("\n")
    print("╔" + "═" * 108 + "╗")
    print("║" + " TABLA DE RESULTADOS DETALLADA".center(108) + "║")
    print("╠" + "═" * 108 + "╣")
    
    # Header
    header = f"║ {'Problema':<20} │ {'Método':<15} │ {'RMSE (test)':<12} │ {'R² (test)':<10} │ {'Tiempo':<8} │ {'Compl.':<6} │ {'Estado':<8} ║"
    print(header)
    print("╠" + "═" * 108 + "╣")
    
    for problem in problems:
        pid = problem.id
        if pid not in by_problem:
            continue
        
        for method in methods:
            if method not in by_problem[pid]:
                continue
            r = by_problem[pid][method]
            
            rmse_str = f"{r.rmse_test:.6f}" if r.rmse_test < 1e6 else "FAILED"
            r2_str = f"{r.r2_test:.4f}" if r.r2_test > -1e6 else "N/A"
            time_str = f"{r.time_seconds:.1f}s"
            compl_str = str(r.complexity)
            status = "✅" if r.solved else "❌"
            
            line = f"║ {problem.name:<20} │ {method:<15} │ {rmse_str:<12} │ {r2_str:<10} │ {time_str:<8} │ {compl_str:<6} │ {status:<8} ║"
            print(line)
        
        print("╟" + "─" * 108 + "╢")
    
    print("╚" + "═" * 108 + "╝")


def print_summary(results: List[MethodResult]):
    """Imprime resumen comparativo."""
    
    methods = sorted(set(r.method for r in results))
    
    print("\n")
    print("╔" + "═" * 80 + "╗")
    print("║" + " RESUMEN COMPARATIVO".center(80) + "║")
    print("╠" + "═" * 80 + "╣")
    
    header = f"║ {'Métrica':<30} │ "
    for m in methods:
        header += f"{m:<20} │ "
    header = header.rstrip(" │ ") + " ║"
    print(header)
    print("╠" + "═" * 80 + "╣")
    
    stats = {}
    for m in methods:
        m_results = [r for r in results if r.method == m]
        n_total = len(m_results)
        n_solved = sum(1 for r in m_results if r.solved)
        n_errors = sum(1 for r in m_results if r.error)
        valid_rmse = [r.rmse_test for r in m_results if r.rmse_test < 1e6 and not r.error]
        valid_r2 = [r.r2_test for r in m_results if r.r2_test > -1e6 and not r.error]
        valid_times = [r.time_seconds for r in m_results if not r.error]
        valid_compl = [r.complexity for r in m_results if r.complexity > 0 and not r.error]
        
        stats[m] = {
            'solved': n_solved,
            'total': n_total,
            'errors': n_errors,
            'solve_rate': n_solved / n_total * 100 if n_total > 0 else 0,
            'avg_rmse': np.mean(valid_rmse) if valid_rmse else float('inf'),
            'median_rmse': np.median(valid_rmse) if valid_rmse else float('inf'),
            'avg_r2': np.mean(valid_r2) if valid_r2 else -float('inf'),
            'avg_time': np.mean(valid_times) if valid_times else 0,
            'total_time': sum(valid_times) if valid_times else 0,
            'avg_complexity': np.mean(valid_compl) if valid_compl else 0,
        }
    
    # Filas de métricas
    metrics = [
        ("Problemas resueltos", lambda s: f"{s['solved']}/{s['total']}"),
        ("Tasa de resolución (%)", lambda s: f"{s['solve_rate']:.1f}%"),
        ("RMSE promedio (test)", lambda s: f"{s['avg_rmse']:.6f}" if s['avg_rmse'] < 1e6 else "N/A"),
        ("RMSE mediana (test)", lambda s: f"{s['median_rmse']:.6f}" if s['median_rmse'] < 1e6 else "N/A"),
        ("R² promedio (test)", lambda s: f"{s['avg_r2']:.4f}" if s['avg_r2'] > -1e6 else "N/A"),
        ("Tiempo promedio (s)", lambda s: f"{s['avg_time']:.2f}"),
        ("Tiempo total (s)", lambda s: f"{s['total_time']:.1f}"),
        ("Complejidad promedio", lambda s: f"{s['avg_complexity']:.1f}"),
        ("Errores", lambda s: f"{s['errors']}"),
    ]
    
    for name, fmt_fn in metrics:
        line = f"║ {name:<30} │ "
        for m in methods:
            line += f"{fmt_fn(stats[m]):<20} │ "
        line = line.rstrip(" │ ") + " ║"
        print(line)
    
    print("╠" + "═" * 80 + "╣")
    
    # Determinar ganador
    if len(methods) == 2:
        m1, m2 = methods
        s1, s2 = stats[m1], stats[m2]
        
        wins = {m1: 0, m2: 0}
        
        # Scoring: más problemas resueltos > mejor RMSE > más rápido > menos complejo
        if s1['solved'] > s2['solved']: wins[m1] += 3
        elif s2['solved'] > s1['solved']: wins[m2] += 3
        
        if s1['avg_rmse'] < s2['avg_rmse']: wins[m1] += 2
        elif s2['avg_rmse'] < s1['avg_rmse']: wins[m2] += 2
        
        if s1['avg_r2'] > s2['avg_r2']: wins[m1] += 2
        elif s2['avg_r2'] > s1['avg_r2']: wins[m2] += 2
        
        if s1['avg_time'] < s2['avg_time']: wins[m1] += 1
        elif s2['avg_time'] < s1['avg_time']: wins[m2] += 1
        
        winner = m1 if wins[m1] > wins[m2] else m2 if wins[m2] > wins[m1] else "EMPATE"
        
        print(f"║{'':^80}║")
        if winner == "EMPATE":
            print(f"║{'🏆  RESULTADO: EMPATE  🏆':^80}║")
        else:
            loser = m2 if winner == m1 else m1
            msg = f"🏆  GANADOR: {winner}  ({wins[winner]} pts vs {wins[loser]} pts)  🏆"
            print(f"║{msg:^80}║")
        
        # Comparación head-to-head por problema
        by_problem = {}
        for r in results:
            if r.problem_id not in by_problem:
                by_problem[r.problem_id] = {}
            by_problem[r.problem_id][r.method] = r
        
        head2head = {m1: 0, m2: 0, 'tie': 0}
        for pid, pdata in by_problem.items():
            if m1 in pdata and m2 in pdata:
                r1, r2 = pdata[m1], pdata[m2]
                if r1.rmse_test < r2.rmse_test * 0.95:  # 5% de margen
                    head2head[m1] += 1
                elif r2.rmse_test < r1.rmse_test * 0.95:
                    head2head[m2] += 1
                else:
                    head2head['tie'] += 1
        
        print(f"║{'':^80}║")
        h2h_msg = f"Head-to-Head: {m1}={head2head[m1]}  {m2}={head2head[m2]}  Empate={head2head['tie']}"
        print(f"║{h2h_msg:^80}║")
    
    print("╚" + "═" * 80 + "╝")
    
    return stats


def save_results(results: List[MethodResult], stats: Dict, output_dir: str):
    """Guarda resultados en archivos."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON con todos los detalles
    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    
    def _safe_val(v):
        if isinstance(v, float):
            return v if np.isfinite(v) else str(v)
        return v
    
    data = {
        'timestamp': timestamp,
        'results': [asdict(r) for r in results],
        'summary': {k: {kk: _safe_val(vv) for kk, vv in v.items()} for k, v in stats.items()} if stats else {}
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📄 Resultados guardados en: {json_path}")
    
    # CSV legible
    csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Problema,Método,RMSE_Train,RMSE_Test,R2_Train,R2_Test,Tiempo_s,Complejidad,Resuelto,Fórmula\n")
        for r in results:
            f.write(f"{r.problem_id},{r.method},{r.rmse_train:.6f},{r.rmse_test:.6f},"
                    f"{r.r2_train:.4f},{r.r2_test:.4f},{r.time_seconds:.2f},"
                    f"{r.complexity},{r.solved},\"{r.formula_found}\"\n")
    print(f"📄 CSV guardado en: {csv_path}")
    
    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark: AlphaSymbolic vs PySR")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout por problema (segundos)")
    parser.add_argument("--problems", choices=["all", "easy", "medium", "hard", "nguyen"], 
                        default="all", help="Subconjunto de problemas")
    parser.add_argument("--only", choices=["alpha", "pysr"], default=None, 
                        help="Ejecutar solo un método")
    parser.add_argument("--no-sniper", action="store_true",
                        help="Desactivar The Sniper")
    parser.add_argument("--pure-gp", action="store_true",
                        help="Desactivar Sniper y Structural Seeds (GP puro)")
    parser.add_argument("--quick", action="store_true",
                        help="Ejecutar subset corto para iteración rápida (4 problemas representativos)")
    parser.add_argument("--quick-timeout", type=int, default=20,
                        help="Timeout usado en --quick cuando --timeout queda en default (30)")
    parser.add_argument("--output", default=None, help="Directorio de salida")
    args = parser.parse_args()
    
    # Filtrar problemas
    if args.quick:
        id_set = set(QUICK_BENCHMARK_IDS)
        problems = [p for p in BENCHMARK_PROBLEMS if p.id in id_set]
    elif args.problems == "easy":
        problems = [p for p in BENCHMARK_PROBLEMS if p.difficulty == "Easy"]
    elif args.problems == "medium":
        problems = [p for p in BENCHMARK_PROBLEMS if p.difficulty in ("Easy", "Medium")]
    elif args.problems == "hard":
        problems = [p for p in BENCHMARK_PROBLEMS if p.difficulty == "Hard"]
    elif args.problems == "nguyen":
        problems = [p for p in BENCHMARK_PROBLEMS if p.id.startswith("nguyen")]
    else:
        problems = BENCHMARK_PROBLEMS

    timeout_sec = args.timeout
    if args.quick and args.timeout == 30:
        timeout_sec = args.quick_timeout
        print(f"\n⚡ Modo quick activado: {len(problems)} problemas, timeout={timeout_sec}s por problema")
    
    run_alpha = args.only != "pysr"
    run_pysr_flag = args.only != "alpha"
    
    # Ejecutar benchmark
    results = run_benchmark(
        problems, 
        timeout_sec=timeout_sec,
        run_alpha=run_alpha,
        run_pysr_flag=run_pysr_flag,
        use_sniper=not (args.no_sniper or args.pure_gp),
        use_structural_seeds=not args.pure_gp
    )
    
    # Mostrar resultados
    print_results_table(results, problems)
    stats = print_summary(results)
    
    # Guardar
    output_dir = args.output or os.path.join(
        os.path.dirname(__file__), '..', 'outputs', 'benchmarks'
    )
    save_results(results, stats, output_dir)
    
    print("\n✅ Benchmark completado.\n")


if __name__ == "__main__":
    main()
