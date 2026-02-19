"""
Performance Benchmarking Suite for GPU GP Engine.

Standard benchmark problems for evaluating symbolic regression performance.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    problem_name: str
    target_formula: str
    found_formula: Optional[str]
    rmse: float
    exact_match: bool
    time_seconds: float
    generations: int


class BenchmarkSuite:
    """
    Standard symbolic regression benchmark problems.
    
    Includes problems from:
    - Nguyen benchmark suite
    - Keijzer benchmarks
    - Custom problems
    """
    
    # Standard benchmark problems: (name, formula, x_range, n_points)
    PROBLEMS = {
        # Nguyen benchmarks
        'nguyen-1': ('x^3 + x^2 + x', (-1, 1), 20),
        'nguyen-2': ('x^4 + x^3 + x^2 + x', (-1, 1), 20),
        'nguyen-3': ('x^5 + x^4 + x^3 + x^2 + x', (-1, 1), 20),
        'nguyen-4': ('x^6 + x^5 + x^4 + x^3 + x^2 + x', (-1, 1), 20),
        'nguyen-5': ('sin(x^2)*cos(x) - 1', (-1, 1), 20),
        'nguyen-6': ('sin(x) + sin(x + x^2)', (-1, 1), 20),
        'nguyen-7': ('log(x+1) + log(x^2+1)', (0, 2), 20),
        'nguyen-8': ('sqrt(x)', (0, 4), 20),
        
        # Keijzer benchmarks (simpler)
        'keijzer-1': ('x^3/5 + x^2/2 - x', (-3, 3), 20),
        'keijzer-4': ('x^3 * exp(-x) * cos(x) * sin(x)', (0, 10), 20),
        
        # Simple polynomials
        'poly-1': ('x^2', (-5, 5), 20),
        'poly-2': ('x^3 - 2*x', (-3, 3), 20),
        'poly-3': ('2*x^2 + 3*x + 1', (-5, 5), 20),
        
        # Trigonometric
        'trig-1': ('sin(x)', (-3.14, 3.14), 20),
        'trig-2': ('cos(x)*sin(x)', (-3.14, 3.14), 20),
        
        # Mixed
        'mixed-1': ('x*sin(x)', (-5, 5), 20),
        'mixed-2': ('sqrt(x)*log(x+1)', (0.1, 10), 20),
    }
    
    def __init__(self, engine_factory):
        """
        Args:
            engine_factory: Function that creates a TensorGeneticEngine instance
        """
        self.engine_factory = engine_factory
        self.results: List[BenchmarkResult] = []
    
    def generate_data(self, formula: str, x_range: Tuple[float, float], n_points: int) -> Tuple[List[float], List[float]]:
        """Generate x,y data from a formula string."""
        import math
        
        x_vals = np.linspace(x_range[0], x_range[1], n_points).tolist()
        y_vals = []
        
        for x in x_vals:
            try:
                # Safe eval with math functions
                y = eval(formula.replace('^', '**'), {"x": x, "sin": math.sin, "cos": math.cos, 
                                                        "tan": math.tan, "exp": math.exp, 
                                                        "log": math.log, "sqrt": math.sqrt,
                                                        "pi": math.pi, "e": math.e})
                y_vals.append(float(y))
            except:
                y_vals.append(0.0)
        
        return x_vals, y_vals
    
    def run_benchmark(self, problem_name: str, timeout_sec: float = 10) -> BenchmarkResult:
        """Run a single benchmark problem."""
        if problem_name not in self.PROBLEMS:
            raise ValueError(f"Unknown problem: {problem_name}")
        
        formula, x_range, n_points = self.PROBLEMS[problem_name]
        x_vals, y_vals = self.generate_data(formula, x_range, n_points)
        
        engine = self.engine_factory()
        
        start_time = time.time()
        result = engine.run(x_vals, y_vals, [], timeout_sec=timeout_sec)
        elapsed = time.time() - start_time
        
        # Calculate RMSE of found solution
        rmse = float('inf')
        if result:
            try:
                # Evaluate found formula
                import math
                found_y = []
                for x in x_vals:
                    try:
                        y = eval(result.replace('^', '**'), 
                                {"x": x, "x0": x, "sin": math.sin, "cos": math.cos,
                                 "tan": math.tan, "exp": math.exp, "log": math.log, 
                                 "sqrt": math.sqrt, "abs": abs, "pi": math.pi, "e": math.e})
                        found_y.append(float(y))
                    except:
                        found_y.append(float('inf'))
                
                mse = sum((a-b)**2 for a,b in zip(y_vals, found_y)) / len(y_vals)
                rmse = mse ** 0.5
            except:
                pass
        
        # Check exact match (simplified comparison)
        exact_match = rmse < 1e-6
        
        bench_result = BenchmarkResult(
            problem_name=problem_name,
            target_formula=formula,
            found_formula=result,
            rmse=rmse,
            exact_match=exact_match,
            time_seconds=elapsed,
            generations=0  # Would need to track in engine
        )
        
        self.results.append(bench_result)
        return bench_result
    
    def run_suite(self, problem_names: List[str] = None, timeout_sec: float = 10, 
                  callback=None) -> Dict[str, BenchmarkResult]:
        """
        Run a suite of benchmark problems.
        
        Args:
            problem_names: List of problems to run (default: all)
            timeout_sec: Timeout per problem
            callback: Optional progress callback
            
        Returns:
            Dict mapping problem name to result
        """
        if problem_names is None:
            problem_names = list(self.PROBLEMS.keys())
        
        results = {}
        for i, name in enumerate(problem_names):
            if callback:
                callback(f"Running {name} ({i+1}/{len(problem_names)})")
            
            results[name] = self.run_benchmark(name, timeout_sec)
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary statistics of benchmark results."""
        if not self.results:
            return {}
        
        n_exact = sum(1 for r in self.results if r.exact_match)
        avg_rmse = np.mean([r.rmse for r in self.results if r.rmse < float('inf')])
        avg_time = np.mean([r.time_seconds for r in self.results])
        
        return {
            'n_problems': len(self.results),
            'n_exact_matches': n_exact,
            'success_rate': n_exact / len(self.results) * 100,
            'avg_rmse': avg_rmse,
            'avg_time_seconds': avg_time,
        }
    
    def print_report(self):
        """Print a formatted benchmark report."""
        print("\n" + "="*70)
        print("GPU GP ENGINE BENCHMARK REPORT")
        print("="*70)
        
        for result in self.results:
            status = "✓" if result.exact_match else "✗"
            print(f"\n{status} {result.problem_name}")
            print(f"  Target: {result.target_formula}")
            print(f"  Found:  {result.found_formula or 'None'}")
            print(f"  RMSE:   {result.rmse:.6e}")
            print(f"  Time:   {result.time_seconds:.2f}s")
        
        summary = self.get_summary()
        print("\n" + "-"*70)
        print(f"SUMMARY: {summary.get('n_exact_matches', 0)}/{summary.get('n_problems', 0)} exact matches")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"Average RMSE: {summary.get('avg_rmse', 0):.6e}")
        print(f"Average Time: {summary.get('avg_time_seconds', 0):.2f}s")
        print("="*70)


def create_benchmark_suite(device=None, pop_size=1000):
    """Factory function to create a benchmark suite."""
    from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
    
    def factory():
        return TensorGeneticEngine(device=device, pop_size=pop_size, n_islands=4)
    
    return BenchmarkSuite(factory)
