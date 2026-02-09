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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import json
import argparse
import traceback
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

def run_alphasybolic(problem: BenchmarkProblem, timeout_sec: int = 30) -> MethodResult:
    """Ejecuta AlphaSymbolic en un problema."""
    import torch
    from core.gpu import TensorGeneticEngine
    from core.gpu.config import GpuGlobals
    
    x_train, y_train, x_test, y_test = problem.generate_data()
    
    # Configurar engine para benchmark (poblacion moderada, rápido)
    old_pop = GpuGlobals.POP_SIZE
    old_islands = GpuGlobals.NUM_ISLANDS
    old_log = GpuGlobals.USE_LOG_TRANSFORMATION
    old_sniper = GpuGlobals.USE_SNIPER
    old_cos = GpuGlobals.USE_OP_COS
    old_threshold = GpuGlobals.EXACT_SOLUTION_THRESHOLD
    old_complexity = GpuGlobals.COMPLEXITY_PENALTY
    old_mutation = GpuGlobals.BASE_MUTATION_RATE
    old_depth = GpuGlobals.MAX_TREE_DEPTH_INITIAL
    old_mut_depth = GpuGlobals.MAX_TREE_DEPTH_MUTATION
    old_simplify_k = GpuGlobals.K_SIMPLIFY
    old_simplify_int = GpuGlobals.SIMPLIFICATION_INTERVAL
    
    GpuGlobals.POP_SIZE = 50_000       # Población moderada para benchmark
    GpuGlobals.NUM_ISLANDS = 10
    GpuGlobals.USE_LOG_TRANSFORMATION = False
    GpuGlobals.USE_SNIPER = True
    GpuGlobals.USE_OP_COS = True       # NECESARIO para muchos benchmarks
    GpuGlobals.EXACT_SOLUTION_THRESHOLD = 1e-6  # Early termination
    GpuGlobals.COMPLEXITY_PENALTY = 0.02   # Parsimonia fuerte
    GpuGlobals.BASE_MUTATION_RATE = 0.15   # Menos destructivo
    GpuGlobals.MAX_TREE_DEPTH_INITIAL = 5  # Árboles iniciales pequeños
    GpuGlobals.MAX_TREE_DEPTH_MUTATION = 4
    GpuGlobals.K_SIMPLIFY = 50
    GpuGlobals.SIMPLIFICATION_INTERVAL = 10
    
    try:
        engine = TensorGeneticEngine(num_variables=1, max_constants=10)
        
        start = time.time()
        formula = engine.run(
            x_train.tolist(), 
            y_train.tolist(), 
            seeds=[], 
            timeout_sec=timeout_sec,
            use_log=False
        )
        elapsed = time.time() - start
        
        # Evaluar resultado
        if formula:
            rmse_train, r2_train = _eval_formula_metrics(formula, x_train, y_train)
            rmse_test, r2_test = _eval_formula_metrics(formula, x_test, y_test)
            # Count AST nodes for fair comparison with PySR
            complexity = _count_ast_nodes(formula)
        else:
            formula = "No solution"
            rmse_train = rmse_test = float('inf')
            r2_train = r2_test = -float('inf')
            complexity = 0
        
        solved = rmse_test < 0.05
        
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
        GpuGlobals.USE_OP_COS = old_cos
        GpuGlobals.EXACT_SOLUTION_THRESHOLD = old_threshold
        GpuGlobals.COMPLEXITY_PENALTY = old_complexity
        GpuGlobals.BASE_MUTATION_RATE = old_mutation
        GpuGlobals.MAX_TREE_DEPTH_INITIAL = old_depth
        GpuGlobals.MAX_TREE_DEPTH_MUTATION = old_mut_depth
        GpuGlobals.K_SIMPLIFY = old_simplify_k
        GpuGlobals.SIMPLIFICATION_INTERVAL = old_simplify_int
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
) -> List[MethodResult]:
    """Ejecuta el benchmark completo."""
    
    all_results: List[MethodResult] = []
    total = len(problems)
    methods_to_run = []
    if run_alpha:
        methods_to_run.append("AlphaSymbolic")
    if run_pysr_flag:
        methods_to_run.append("PySR")
    
    print("\n" + "═" * 70)
    print("  BENCHMARK COMPARATIVO: " + " vs ".join(methods_to_run))
    print("═" * 70)
    print(f"  Problemas: {total}")
    print(f"  Timeout por problema: {timeout_sec}s")
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
            result = run_alphasybolic(problem, timeout_sec)
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
    parser.add_argument("--output", default=None, help="Directorio de salida")
    args = parser.parse_args()
    
    # Filtrar problemas
    if args.problems == "easy":
        problems = [p for p in BENCHMARK_PROBLEMS if p.difficulty == "Easy"]
    elif args.problems == "medium":
        problems = [p for p in BENCHMARK_PROBLEMS if p.difficulty in ("Easy", "Medium")]
    elif args.problems == "hard":
        problems = [p for p in BENCHMARK_PROBLEMS if p.difficulty == "Hard"]
    elif args.problems == "nguyen":
        problems = [p for p in BENCHMARK_PROBLEMS if p.id.startswith("nguyen")]
    else:
        problems = BENCHMARK_PROBLEMS
    
    run_alpha = args.only != "pysr"
    run_pysr_flag = args.only != "alpha"
    
    # Ejecutar benchmark
    results = run_benchmark(
        problems, 
        timeout_sec=args.timeout,
        run_alpha=run_alpha,
        run_pysr_flag=run_pysr_flag
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
