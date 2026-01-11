
import torch
import numpy as np
import time
from core.gpu.engine import TensorGeneticEngine
from core.gpu.config import GpuGlobals

def run_test():
    print("Running Quick GPU Benchmark...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    
    # Override settings for speed
    # Override settings for speed/rigor
    GpuGlobals.POP_SIZE = 20000
    GpuGlobals.GENERATIONS = 50
    GpuGlobals.USE_PARETO_SELECTION = False # Disable Pareto to test Lexicase pure or standard
    GpuGlobals.USE_LEXICASE_SELECTION = True
    GpuGlobals.USE_NANO_PSO = True
    
    # Problems
    problems = [
        ("Poly", lambda x: x**2 + x + 1, 1),
        ("Trig", lambda x: np.sin(x) * x**2, 1), # The one failing in benchmark
        ("Multi", lambda x: x[:,0]*x[:,1] + np.sin(x[:,2]), 3) # The one failing in benchmark
    ]
    
    for name, func, vars_count in problems:
        print(f"\n--- Problem: {name} ---")
        try:
            # Data
            if vars_count == 1:
                x_np = np.linspace(-5, 5, 100).reshape(1, 100)
                y_np = func(x_np[0])
                active_vars = ['x0']
            else:
                x_np = np.random.uniform(-5, 5, (vars_count, 100))
                y_np = func(x_np.T)
                active_vars = [f'x{i}' for i in range(vars_count)]
                
            x_torch = torch.tensor(x_np, dtype=torch.float64, device=device)
            y_torch = torch.tensor(y_np, dtype=torch.float64, device=device)
            
            # Engine
            GpuGlobals.MAX_GENERATIONS = GpuGlobals.GENERATIONS 
            
            engine = TensorGeneticEngine(
                pop_size=GpuGlobals.POP_SIZE,
                max_len=30,
                num_variables=vars_count,
                device=device
            )
            
            t0 = time.time()
            
             # Callback to print progress
            def test_callback(gen, rmse, rpn, consts, improved, _):
                  if improved:
                      print(f"  [Gen {gen}] New Best: {rmse:.4e}")
                      
            infix_res = engine.run(x_torch, y_torch, timeout_sec=120, callback=test_callback)
                
            best_rmse = getattr(engine, 'best_global_rmse', float('inf'))
            
            print(f"Time: {time.time() - t0:.2f}s")
            print(f"Final RMSE: {best_rmse}")
            print(f"Formula: {infix_res}")
                
        except Exception:
                import traceback
                print(f"CRITICAL FAILURE in Problem {name}")
                traceback.print_exc()
        
if __name__ == "__main__":
    run_test()
