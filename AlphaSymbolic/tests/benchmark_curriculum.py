import torch
import numpy as np
import time
from search.hybrid_search import hybrid_solve

def run_benchmark():
    print("=== AlphaSymbolic Curriculum Benchmark (10 Formulas) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define 10 formulas of increasing difficulty
    curriculum = [
        {"name": "Linear", "func": lambda x: 2*x + 3, "use_log": False, "pop": 50_000},
        {"name": "Quadratic", "func": lambda x: x**2 + x + 1, "use_log": False, "pop": 100_000},
        {"name": "Cubic", "func": lambda x: x**3 - 2*x, "use_log": False, "pop": 200_000},
        {"name": "Sinusoidal", "func": lambda x: np.sin(x), "use_log": False, "pop": 200_000},
        {"name": "Exponential", "func": lambda x: np.exp(0.5 * x), "use_log": True, "pop": 100_000},
        {"name": "Logarithmic", "func": lambda x: np.log(x + 1), "use_log": False, "pop": 100_000},
        {"name": "Rational", "func": lambda x: 1 / (x + 1), "use_log": False, "pop": 200_000},
        {"name": "Trig Mix", "func": lambda x: np.sin(x) + np.cos(2*x), "use_log": False, "pop": 500_000},
        {"name": "Complex Mix", "func": lambda x: x**2 + np.exp(x/10), "use_log": True, "pop": 500_000},
        {"name": "The Boss", "func": lambda x: (x**2 - 1) / (x**2 + 1), "use_log": False, "pop": 1_000_000},
    ]
    
    stats = []
    
    for i, step in enumerate(curriculum):
        print(f"\n[Step {i+1}/10] Solving {step['name']}...")
        
        # Prepare Data
        x = np.linspace(1, 10, 20)
        y = step['func'](x)
        
        start_time = time.time()
        
        # Run Search
        result = hybrid_solve(
            x, y, 
            model=None, 
            device=device,
            beam_width=10,
            gp_timeout=15, # Max 15s per formula
            pop_size=step['pop'],
            use_log=step['use_log'],
            num_variables=1
        )
        
        elapsed = time.time() - start_time
        success = result['rmse'] < 1e-4 if result['formula'] else False
        
        print(f"   Result: {result['formula']}")
        print(f"   RMSE: {result['rmse']:.6f} | Time: {elapsed:.2f}s | Success: {success}")
        
        stats.append({
            "name": step['name'],
            "success": success,
            "time": elapsed,
            "rmse": result['rmse'],
            "formula": result['formula']
        })

    print("\n\n=== BENCHMARK SUMMARY ===")
    print(f"{'Name':<15} | {'Success':<8} | {'Time':<8} | {'Formula'}")
    print("-" * 60)
    for s in stats:
        print(f"{s['name']:<15} | {str(s['success']):<8} | {s['time']:>6.2f}s | {s['formula']}")
    
    total_success = sum(1 for s in stats if s['success'])
    print(f"\nScore: {total_success}/10")

if __name__ == "__main__":
    run_benchmark()
