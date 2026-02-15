"""
╔══════════════════════════════════════════════════════════════════╗
║       Test Individual — Prueba rápida de un problema            ║
║                                                                 ║
║  Ejecuta Alpha y/o PySR en UN solo problema del benchmark.      ║
║  Mucho más rápido que correr los 21 problemas completos.        ║
║                                                                 ║
║  Uso:                                                           ║
║    python test_individual.py poly-4             # Solo Alpha    ║
║    python test_individual.py keijzer-4 --pysr   # Alpha + PySR  ║
║    python test_individual.py poly-4 -n 5        # 5 repeticiones║
║    python test_individual.py --list              # Ver problemas ║
║    python test_individual.py poly-4 -t 60       # 60s timeout   ║
║    python test_individual.py --custom "x**3+1"  # Fórmula custom║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import argparse
import traceback
from typing import Tuple

# ── Importar problemas y utilidades del benchmark ──
from benchmark_vs_pysr import (
    BENCHMARK_PROBLEMS, BenchmarkProblem, MethodResult,
    _eval_formula_metrics, _count_ast_nodes,
    run_alphasybolic, run_pysr
)


def find_problem(query: str) -> BenchmarkProblem:
    """Busca un problema por ID o nombre parcial."""
    q = query.lower().strip()
    # Búsqueda exacta por ID
    for p in BENCHMARK_PROBLEMS:
        if p.id == q:
            return p
    # Búsqueda parcial por nombre o ID
    for p in BENCHMARK_PROBLEMS:
        if q in p.id.lower() or q in p.name.lower():
            return p
    return None


def print_result(result: MethodResult, problem: BenchmarkProblem):
    """Imprime resultado de forma clara."""
    status = "SOLVED" if result.solved else "FAILED"
    icon = "+" if result.solved else "X"
    print(f"  [{icon}] {result.method:15s}  RMSE={result.rmse_test:.6f}  "
          f"R2={result.r2_test:.4f}  T={result.time_seconds:.1f}s  "
          f"C={result.complexity}  {status}")
    # Truncar fórmula larga
    formula = result.formula_found
    if len(formula) > 100:
        formula = formula[:97] + "..."
    print(f"      Formula: {formula}")


def run_test(problem: BenchmarkProblem, n_runs: int = 1, timeout: int = 30,
             run_pysr_too: bool = False, use_sniper: bool = False):
    """Ejecuta test individual con estadísticas."""
    
    print(f"\n{'='*65}")
    print(f"  Problema: {problem.name} ({problem.id})")
    print(f"  Target:   {problem.formula_str}")
    print(f"  Rango:    {problem.x_range}  Dificultad: {problem.difficulty}")
    mode_label = "Sniper" if use_sniper else "Pure GP (sin seeds estructurales)"
    print(f"  Timeout:  {timeout}s  Runs: {n_runs}  Modo: {mode_label}")
    print(f"{'='*65}")
    
    alpha_results = []
    pysr_results = []
    
    for i in range(n_runs):
        if n_runs > 1:
            print(f"\n--- Run {i+1}/{n_runs} ---")
        
        # AlphaSymbolic
        print(f"\n  Running AlphaSymbolic...")
        try:
            result = run_alphasybolic(
                problem,
                timeout_sec=timeout,
                use_sniper=use_sniper,
                use_structural_seeds=use_sniper
            )
            alpha_results.append(result)
            print_result(result, problem)
        except Exception as e:
            print(f"  [!] Error: {e}")
            traceback.print_exc()
        
        # PySR
        if run_pysr_too:
            print(f"  Running PySR...")
            try:
                result = run_pysr(problem, timeout_sec=timeout)
                pysr_results.append(result)
                print_result(result, problem)
            except Exception as e:
                print(f"  [!] PySR Error: {e}")
    
    # Estadísticas
    if len(alpha_results) > 0:
        print(f"\n{'='*65}")
        print(f"  ESTADISTICAS AlphaSymbolic ({len(alpha_results)} runs)")
        print(f"{'='*65}")
        
        rmses = [r.rmse_test for r in alpha_results]
        times = [r.time_seconds for r in alpha_results]
        solved = sum(1 for r in alpha_results if r.solved)
        
        print(f"  Solved:     {solved}/{len(alpha_results)} ({100*solved/len(alpha_results):.0f}%)")
        print(f"  RMSE best:  {min(rmses):.6f}")
        print(f"  RMSE worst: {max(rmses):.6f}")
        print(f"  RMSE avg:   {np.mean(rmses):.6f}")
        print(f"  RMSE med:   {np.median(rmses):.6f}")
        print(f"  Time avg:   {np.mean(times):.1f}s")
        
        best_idx = np.argmin(rmses)
        print(f"  Best formula: {alpha_results[best_idx].formula_found}")
    
    if len(pysr_results) > 0:
        print(f"\n  ESTADISTICAS PySR ({len(pysr_results)} runs)")
        print(f"  {'-'*50}")
        rmses = [r.rmse_test for r in pysr_results]
        times = [r.time_seconds for r in pysr_results]
        solved = sum(1 for r in pysr_results if r.solved)
        print(f"  Solved:     {solved}/{len(pysr_results)}")
        print(f"  RMSE best:  {min(rmses):.6f}")
        print(f"  RMSE avg:   {np.mean(rmses):.6f}")
        print(f"  Time avg:   {np.mean(times):.1f}s")
    
    return alpha_results, pysr_results


def main():
    parser = argparse.ArgumentParser(description="Test individual de problemas del benchmark")
    parser.add_argument("problem", nargs="?", help="ID o nombre del problema (ej: poly-4, keijzer-4, nguyen-3)")
    parser.add_argument("--list", action="store_true", help="Listar todos los problemas disponibles")
    parser.add_argument("-n", "--runs", type=int, default=1, help="Numero de repeticiones (default: 1)")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="Timeout en segundos (default: 30)")
    parser.add_argument("--pysr", action="store_true", help="Tambien ejecutar PySR para comparar")
    parser.add_argument("--sniper", action="store_true", help="Usar Sniper (default: pure GP)")
    parser.add_argument("--custom", type=str, help="Formula custom (ej: 'x**3+1')")
    parser.add_argument("--range", type=str, default="-5,5", help="Rango para formula custom (ej: '-3,3')")
    
    args = parser.parse_args()
    
    # Listar problemas
    if args.list:
        print("\nProblemas disponibles:")
        print(f"  {'ID':<15} {'Nombre':<25} {'Dificultad':<10} {'Formula'}")
        print(f"  {'-'*15} {'-'*25} {'-'*10} {'-'*30}")
        for p in BENCHMARK_PROBLEMS:
            print(f"  {p.id:<15} {p.name:<25} {p.difficulty:<10} {p.formula_str}")
        print(f"\n  Total: {len(BENCHMARK_PROBLEMS)} problemas")
        print(f"\n  Ejemplo: python test_individual.py poly-4 -n 3 --pysr")
        return
    
    # Formula custom
    if args.custom:
        r = [float(x) for x in args.range.split(",")]
        problem = BenchmarkProblem("custom", "Custom", args.custom, "Custom", (r[0], r[1]))
        run_test(problem, n_runs=args.runs, timeout=args.timeout,
                 run_pysr_too=args.pysr, use_sniper=args.sniper)
        return
    
    # Necesitamos un problema
    if not args.problem:
        parser.print_help()
        print("\n  Usa --list para ver problemas disponibles")
        return
    
    # Buscar problema
    problem = find_problem(args.problem)
    if problem is None:
        print(f"\n  Error: Problema '{args.problem}' no encontrado.")
        print(f"  Usa --list para ver problemas disponibles.")
        return
    
    run_test(problem, n_runs=args.runs, timeout=args.timeout,
             run_pysr_too=args.pysr, use_sniper=args.sniper)


if __name__ == "__main__":
    main()
