"""
Comparative Benchmark: Beam Search vs MCTS vs Alpha-GP Hybrid
Runs all three search methods on the standard benchmark suite and compares performance.
"""
import torch
import numpy as np
import time
import traceback
from typing import List, Dict, Callable, Optional

from search.mcts import MCTS
from search.beam_search import BeamSearch
from search.hybrid_search import hybrid_solve
from data.benchmark_data import BENCHMARK_SUITE, get_benchmark_data
from core.grammar import ExpressionTree
from utils.optimize_constants import optimize_constants


def run_single_problem(
    x: np.ndarray, 
    y: np.ndarray, 
    method: str, 
    model, 
    device,
    timeout_sec: int = 30,
    beam_width: int = 50
) -> Dict:
    """
    Runs a single search method on a single problem.
    
    Returns:
        dict with keys: formula, rmse, time, success
    """
    start_time = time.time()
    
    try:
        if method == "beam":
            searcher = BeamSearch(model, device, beam_width=beam_width)
            # BeamSearch expects list-like input and returns a list of results sorted by RMSE
            results_list = searcher.search(x.tolist(), y.tolist())
            elapsed = time.time() - start_time
            if results_list and len(results_list) > 0:
                result = results_list[0]  # Best result (sorted by RMSE)
                return {
                    'formula': result.get('formula', 'N/A'),
                    'rmse': result.get('rmse', 1e9),
                    'time': elapsed,
                    'success': result.get('rmse', 1e9) < 0.05
                }
            else:
                return {'formula': 'No Result', 'rmse': 1e9, 'time': elapsed, 'success': False}
            
        elif method == "mcts":
            mcts = MCTS(model, device, max_simulations=500, batch_size=32)
            # MCTS expects list-like input 
            result = mcts.search(x.tolist(), y.tolist())
            elapsed = time.time() - start_time
            return {
                'formula': result.get('formula', 'N/A'),
                'rmse': result.get('rmse', 1e9),
                'time': elapsed,
                'success': result.get('rmse', 1e9) < 0.05
            }
            
        elif method == "hybrid":
            result = hybrid_solve(
                model=model,
                device=device,
                x_values=x.tolist(),
                y_values=y.tolist(),
                beam_width=beam_width,
                gp_timeout=timeout_sec
            )
            elapsed = time.time() - start_time
            
            if result['formula']:
                # Evaluate RMSE for hybrid result
                try:
                    tree = ExpressionTree.from_infix(result['formula'])
                    if tree.is_valid:
                        preds = tree.evaluate(x)
                        rmse = np.sqrt(np.mean((preds - y) ** 2))
                    else:
                        rmse = 1e9
                except:
                    rmse = 1e9
            else:
                rmse = 1e9
                
            return {
                'formula': result.get('formula', 'N/A') or 'Failed',
                'rmse': rmse,
                'time': elapsed,
                'success': rmse < 0.05
            }
        else:
            return {'formula': 'Unknown Method', 'rmse': 1e9, 'time': 0, 'success': False}
            
    except Exception as e:
        print(f"[ERROR] Method {method} failed: {e}")
        traceback.print_exc()
        return {'formula': 'Error', 'rmse': 1e9, 'time': time.time() - start_time, 'success': False}


def run_comparison_benchmark(
    model, 
    device, 
    methods: List[str] = ["beam", "mcts", "hybrid"],
    gp_timeout: int = 30,
    beam_width: int = 50,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Runs all methods on all benchmark problems.
    
    Returns:
        Dict with 'results' (per-problem-per-method) and 'summary' (aggregated stats)
    """
    results = []
    method_stats = {m: {'solved': 0, 'total_time': 0, 'total_rmse': 0} for m in methods}
    
    total_steps = len(BENCHMARK_SUITE) * len(methods)
    current_step = 0
    
    for problem in BENCHMARK_SUITE:
        x, y, _ = get_benchmark_data(problem['id'])
        
        for method in methods:
            current_step += 1
            
            if progress_callback:
                progress_callback(
                    current_step / total_steps, 
                    f"[{method.upper()}] {problem['name']}..."
                )
            
            result = run_single_problem(x, y, method, model, device, gp_timeout, beam_width)
            
            results.append({
                'problem_id': problem['id'],
                'problem_name': problem['name'],
                'level': problem['level'],
                'method': method,
                'formula': result['formula'],
                'rmse': result['rmse'],
                'time': result['time'],
                'success': result['success']
            })
            
            # Update stats
            method_stats[method]['total_time'] += result['time']
            method_stats[method]['total_rmse'] += result['rmse'] if result['rmse'] < 1e6 else 0
            if result['success']:
                method_stats[method]['solved'] += 1
    
    # Compute summary
    num_problems = len(BENCHMARK_SUITE)
    summary = {}
    for method in methods:
        stats = method_stats[method]
        summary[method] = {
            'solved': stats['solved'],
            'total': num_problems,
            'score': (stats['solved'] / num_problems) * 100,
            'avg_time': stats['total_time'] / num_problems,
            'avg_rmse': stats['total_rmse'] / num_problems
        }
    
    if progress_callback:
        progress_callback(1.0, "Benchmark Complete!")
    
    return {'results': results, 'summary': summary}


def format_comparison_table(results: List[Dict]) -> str:
    """
    Formats the results as a human-readable table.
    """
    # Group by problem
    problems = {}
    for r in results:
        pid = r['problem_id']
        if pid not in problems:
            problems[pid] = {'name': r['problem_name'], 'level': r['level'], 'methods': {}}
        problems[pid]['methods'][r['method']] = {
            'rmse': r['rmse'],
            'time': r['time'],
            'success': r['success'],
            'formula': r['formula']
        }
    
    output = []
    output.append("=" * 100)
    output.append(f"{'Problem':<25} | {'Method':<8} | {'RMSE':<12} | {'Time':<8} | {'Status':<10} | Formula")
    output.append("=" * 100)
    
    for pid, pdata in problems.items():
        name = pdata['name'][:24]
        for method, mdata in pdata['methods'].items():
            rmse_str = f"{mdata['rmse']:.6f}" if mdata['rmse'] < 1e6 else "FAILED"
            time_str = f"{mdata['time']:.2f}s"
            status = "[OK]" if mdata['success'] else "[FAIL]"
            formula = mdata['formula'][:40] if mdata['formula'] else "N/A"
            output.append(f"{name:<25} | {method:<8} | {rmse_str:<12} | {time_str:<8} | {status:<10} | {formula}")
        output.append("-" * 100)
    
    return "\n".join(output)


def print_summary(summary: Dict):
    """
    Prints a formatted summary comparison.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY - Method Comparison")
    print("=" * 60)
    print(f"{'Method':<12} | {'Solved':<10} | {'Score':<10} | {'Avg Time':<10} | {'Avg RMSE':<12}")
    print("-" * 60)
    
    for method, stats in summary.items():
        solved_str = f"{stats['solved']}/{stats['total']}"
        score_str = f"{stats['score']:.1f}%"
        time_str = f"{stats['avg_time']:.2f}s"
        rmse_str = f"{stats['avg_rmse']:.6f}"
        print(f"{method.upper():<12} | {solved_str:<10} | {score_str:<10} | {time_str:<10} | {rmse_str:<12}")
    
    print("=" * 60)
    
    # Determine winner
    best_method = max(summary.items(), key=lambda x: (x[1]['solved'], -x[1]['avg_rmse']))
    print(f"\n*** WINNER: {best_method[0].upper()} with {best_method[1]['solved']}/{best_method[1]['total']} problems solved! ***")


if __name__ == "__main__":
    # Standalone test
    import sys
    sys.path.insert(0, '.')
    
    from ui.app_core import load_model, get_model
    
    print("Loading model...")
    load_model()
    model, device = get_model()
    
    if model is None:
        print("Error: No model loaded!")
        exit(1)
    
    print("Running comparison benchmark...")
    result = run_comparison_benchmark(
        model, 
        device, 
        methods=["beam", "mcts", "hybrid"],
        gp_timeout=30,
        beam_width=50
    )
    
    print(format_comparison_table(result['results']))
    print_summary(result['summary'])
