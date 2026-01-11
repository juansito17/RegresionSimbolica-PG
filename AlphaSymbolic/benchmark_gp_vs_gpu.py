
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import warnings
from typing import List, Dict, Any

# Ensure project root is in path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file)) 
sys.path.append(project_root)

from core.gp_bridge import GPEngine
from core.gpu.engine import TensorGeneticEngine
from core.grammar import ExpressionTree, OPERATORS
from core.gpu.config import GpuGlobals

# Suppress warnings
warnings.filterwarnings("ignore")

# Define Benchmark Problems
# Format: (Name, Formula, Input_Ranges, Points, Variables, Difficulty)
BENCHMARKS = [
    {
        "name": "Polynomial Simple",
        "formula": lambda x: x[:,0]**2 + 2*x[:,0] + 1,
        "seed_formula": "x0 * x0 + 2 * x0 + 1",
        "lower": -5.0, "upper": 5.0, "points": 50, "vars": 1,
        "difficulty": 1,
        "target_mse": 1e-6
    },
    {
        "name": "Trig Mix",
        "formula": lambda x: np.sin(x[:,0]) * x[:,0]**2,
        "seed_formula": "sin(x0) * (x0 * x0)",
        "lower": -3.0, "upper": 3.0, "points": 100, "vars": 1,
        "difficulty": 2,
        "target_mse": 1e-4
    },
    {
        "name": "Multivariable Interaction",
        "formula": lambda x: x[:,0] * x[:,1] + np.sin(x[:,2]),
        "seed_formula": "x0 * x1 + sin(x2)",
        "lower": -2.0, "upper": 2.0, "points": 100, "vars": 3,
        "difficulty": 3,
        "target_mse": 1e-4
    }
]

# Config
POPULATION = 5000
GENERATIONS = 500
TIMEOUT_SEC = 30
RUNS_PER_BENCHMARK = 3

class BenchmarkRunner:
    def __init__(self):
        self.cpp_engine = GPEngine()
        self.gpu_engine = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Benchmark Device: {self.device}")

    def generate_data(self, prob):
        # Generate X
        if prob["vars"] == 1:
            X = np.linspace(prob["lower"], prob["upper"], prob["points"]).reshape(-1, 1)
        else:
            X = np.random.uniform(prob["lower"], prob["upper"], (prob["points"], prob["vars"]))
        
        # Generate Y
        Y = prob["formula"](X)
        return X, Y

    def run_cpp_gp(self, X, Y, prob):
        start_t = time.time()
        
        # Format X for bridge (list of lists or matrix)
        # GUB expects features as rows usually or handles transposition
        # GPEngine.run handles numpy arrays efficiently now
        
        # Flatten Y
        Y_list = Y.tolist()
        
        try:
            best_formula = self.cpp_engine.run(
                x_values=X.T, # Features as rows often safer for this legacy bridge? Check implementation
                # Actually bridge says: checks shape. If samples > features, transposed.
                # So passing X directly (Samples, Features) should work and be auto-transposed by bridge logic.
                y_values=Y_list,
                seeds=[],
                timeout_sec=TIMEOUT_SEC
            )
            # The bridge doesn't accept pop/gens args directly in 'run', it uses internal defaults or build config.
            # This is a limitation for fair comparison if we can't set POP/GENS dynamically.
            # However, we can re-compile or assume default.
            # Wait, the bridge interface run() command line args logic is:
            # cmd = [binary, --seed, ..., --data, ...]
            # It DOES NOT pass pop/gens in the python method signature!
            # The C++ engine uses Globals.h defaults (POP=5000, GENS=50000).
            # We must adhere to C++ strict defaults unless we modify bridge or pass extra args.
            
            # For this test, effectively C++ runs "Fast enough" or until timeout.
            # We measure time to return.
            
            elapsed = time.time() - start_t
            return best_formula, elapsed, True
            
        except Exception as e:
            print(f"C++ GP Error: {e}")
            return None, 0, False

    def run_gpu_gp(self, X, Y, prob):
        # Initialize Fresh Engine
        if not self.gpu_engine:
             self.gpu_engine = TensorGeneticEngine(
                 device=self.device, 
                 pop_size=POPULATION, 
                 num_variables=prob["vars"]
             )
        else:
            # Re-init for fresh population or just clean it
             self.gpu_engine = TensorGeneticEngine(
                 device=self.device, 
                 pop_size=POPULATION, 
                 num_variables=prob["vars"]
             )

        start_t = time.time()
        
        # Convert Data
        X_torch = torch.tensor(X, dtype=torch.float64, device=self.device)
        Y_torch = torch.tensor(Y, dtype=torch.float64, device=self.device)
        
        # Init Population
        population = self.gpu_engine.initialize_population()
        constants = torch.randn(POPULATION, self.gpu_engine.max_constants, device=self.device, dtype=torch.float64)
        
        best_rmse = float('inf')
        best_formula_str = ""
        
        try:
            # Initialize Globals if needed (important for grammar construction)
            # GpuGlobals are static but some might need setup? 
            # Actually, let's just ensure we print the error.
            
            for gen in range(GENERATIONS):
                # Evaluate
                fitness = self.gpu_engine.evaluate_batch(population, X_torch, Y_torch, constants)
                
                # Hybrid Search: Gradient Descent on Sub-population (Elites)
                # This gives GPU engine the ability to fine-tune constants like C++ but faster
                # Reduced intensity per generation to allow more structural evolution
                k_opt = min(5, population.shape[0])
                top_vals, top_indices = torch.topk(fitness, k_opt, largest=False)
                
                sub_pop = population[top_indices]
                sub_const = constants[top_indices]
                
                # 5 steps of Adam (Faster per generation)
                opt_const, new_fit = self.gpu_engine.optimize_constants(
                    sub_pop, sub_const, X_torch, Y_torch, steps=5, lr=0.5
                )
                
                # Update back to main strings
                constants[top_indices] = opt_const
                fitness[top_indices] = new_fit
                
                # Check Global Best
                min_fit, min_idx = torch.min(fitness, dim=0)
                if min_fit.item() < best_rmse:
                    best_rmse = min_fit.item()
                    best_ind = population[min_idx]
                    best_const = constants[min_idx]
                    best_formula_str = self.gpu_engine.rpn_to_infix(best_ind, best_const)
                    
                    # Target check
                    if best_rmse**2 < prob["target_mse"]:
                        break

                
                # Check timeout
                if time.time() - start_t > TIMEOUT_SEC:
                    break
                    
                # Evolve - Island Model with Crowding
                # 1. Select Parents (Island Aware)
                # Use 10 islands (Pop 1000 -> 100 per island)
                n_islands = 10
                parent_indices = self.gpu_engine.tournament_selection_island(population, fitness, n_islands)
                parents = population[parent_indices]
                parent_fitness = fitness[parent_indices]
                
                # 2. Crossover & Mutate
                offspring = self.gpu_engine.crossover_population(parents, 0.8)
                offspring = self.gpu_engine.mutate_population(offspring, 0.2)
                
                # 3. Evaluate Offspring
                # Inherit constants from parents and apply a small mutation
                offspring_const = constants[parent_indices].clone()
                # 20% chance of small Gaussian noise to constants
                c_noise = torch.randn_like(offspring_const) * 0.1
                # Only apply noise to some entries to maintain stability
                noise_mask = (torch.rand_like(offspring_const) < 0.2).to(torch.float64)
                offspring_const = offspring_const + (c_noise * noise_mask)
                
                off_fitness = self.gpu_engine.evaluate_batch(offspring, X_torch, Y_torch, offspring_const)
                
                # 4. Deterministic Crowding
                # Compare Offspring vs Parents (preserves diversity better than replacing worst)
                new_pop, new_fitness_batch = self.gpu_engine.deterministic_crowding(parents, offspring, parent_fitness, off_fitness)
                
                # Update Constants based on winners
                mask = off_fitness < parent_fitness
                # new_pop has Offspring where mask=True, Parent where mask=False.
                # So we take OffspringConstants where mask=True, SelectedParentConstants where mask=False.
                new_const = torch.where(mask.unsqueeze(1), offspring_const, constants[parent_indices])
                
                population = new_pop
                constants = new_const
                fitness = new_fitness_batch

                # 5. Migration (Every 5 gens)
                if gen % 5 == 0 and n_islands > 1:
                     # Shift islands to mix gene pools
                     B, L = population.shape
                     island_size = B // n_islands
                     
                     pop_islands = population.view(n_islands, island_size, L)
                     pop_islands = torch.roll(pop_islands, shifts=1, dims=0)
                     population = pop_islands.reshape(B, L)
                     
                     fit_islands = fitness.view(n_islands, island_size)
                     fit_islands = torch.roll(fit_islands, shifts=1, dims=0)
                     fitness = fit_islands.reshape(-1)
                     
                     const_islands = constants.view(n_islands, island_size, -1)
                     const_islands = torch.roll(const_islands, shifts=1, dims=0)
                     constants = const_islands.reshape(B, -1)

            # --- FINAL DEEP REFINEMENT ---
            # Now that evolution is done, take the absolute best and squeeze out more MSE accuracy
            if best_ind is not None:
                # Run 100 steps of Adam on the single best candidate
                best_ind_batch = best_ind.unsqueeze(0)
                best_const_batch = best_const.unsqueeze(0)
                
                opt_const_final, final_fit_batch = self.gpu_engine.optimize_constants(
                    best_ind_batch, best_const_batch, X_torch, Y_torch, steps=100, lr=0.2
                )
                
                best_const = opt_const_final[0]
                best_rmse = final_fit_batch[0].item()
                best_formula_str = self.gpu_engine.rpn_to_infix(best_ind, best_const)
                
            elapsed = time.time() - start_t
            return best_formula_str, elapsed, True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"GPU GP Error: {e}")
            return None, 0, False

    def evaluate_result(self, formula_str, X, Y):
        if not formula_str: return float('inf')
        
        # Method 1: ExpressionTree
        try:
            tree = ExpressionTree.from_infix(formula_str)
            if tree.is_valid:
                # ExpressionTree expects (Samples, Vars) and extracts columns via keys 'x0', 'x1'...
                # It handles column extraction internally if passed numpy array (Samples, Vars).
                y_pred = tree.evaluate(X)
                
                mse = np.mean((y_pred - Y)**2)
                if not np.isnan(mse) and not np.isinf(mse):
                    return mse
        except:
            pass

        # Method 2: Fallback Python Eval (for robustness against ExpressionTree bugs)
        try:
             # Create context
             ctx = {
                 "np": np, 
                 "sin": np.sin, "cos": np.cos, "tan": np.tan, 
                 "sqrt": np.sqrt, "log": np.log, "exp": np.exp, 
                 "abs": np.abs, "floor": np.floor, "gamma": lambda x: 1e300 # Ignore gamma in fallback or implement
             }
             # Variables
             for i in range(X.shape[1]):
                 # x0 corresponds to column 0
                 ctx[f"x{i}"] = X[:, i]
             
             # Engine might output 'pow(a, b)' or 'a^b' which is XOR in python.
             ctx["pow"] = np.power
             
             # Eval
             y_pred = eval(formula_str, {}, ctx)
             
             # Scalar result check
             if not isinstance(y_pred, np.ndarray):
                 y_pred = np.full_like(Y, y_pred)
             
             if y_pred.shape != Y.shape:
                 return float('inf')
                 
             mse = np.mean((y_pred - Y)**2)
             if np.isnan(mse): return float('inf')
             return mse
        except:
             return float('inf')

    def run(self):
        results = []
        
        print(f"Starting Benchmark: {len(BENCHMARKS)} problems, {RUNS_PER_BENCHMARK} runs each.")
        print(f"Comparing: C++ (Subprocess) vs Python (GPU/PyTorch)")
        print("-" * 60)
        
        scores = {"CPP": 0, "GPU": 0}
        
        for prob in BENCHMARKS:
            print(f"\nProblem: {prob['name']} (Vars: {prob['vars']})")
            X, Y = self.generate_data(prob)
            
            # --- UDPATE ENGINE ARGS for C++ ---
            # We can't easily change C++ args without recompiling or specialized flags.
            # We accept it runs with its defaults (often stronger/longer).
            
            for i in range(RUNS_PER_BENCHMARK):
                # 1. C++ Run
                res_cpp, time_cpp, success_cpp = self.run_cpp_gp(X, Y, prob)
                mse_cpp = self.evaluate_result(res_cpp, X, Y)
                
                # 2. GPU Run
                res_gpu, time_gpu, success_gpu = self.run_gpu_gp(X, Y, prob)
                mse_gpu = self.evaluate_result(res_gpu, X, Y)
                
                print(f" Run {i+1}:")
                if res_cpp:
                    print(f"  CPP -> Time: {time_cpp:.2f}s | MSE: {mse_cpp:.2e} | Formula: {res_cpp[:30]}...")
                else:
                    print(f"  CPP -> Failed")

                if res_gpu:
                    print(f"  GPU -> Time: {time_gpu:.2f}s | MSE: {mse_gpu:.2e} | Formula: {res_gpu[:30]}...")
                else:
                    print(f"  GPU -> Failed")
                
                # Scoring
                # Win = Better MSE (significantly) or Faster Time if MSE equal
                winner = "DRAW"
                if abs(mse_cpp - mse_gpu) < 1e-5:
                    if time_cpp < time_gpu: 
                        scores["CPP"] += 1
                        winner = "CPP (Faster)"
                    else: 
                        scores["GPU"] += 1
                        winner = "GPU (Faster)"
                elif mse_cpp < mse_gpu:
                    scores["CPP"] += 1
                    winner = "CPP (Accuracy)"
                else:
                    scores["GPU"] += 1
                    winner = "GPU (Accuracy)"
                
                results.append({
                    "Problem": prob["name"],
                    "Run": i,
                    "CPP_Time": time_cpp, "CPP_MSE": mse_cpp,
                    "GPU_Time": time_gpu, "GPU_MSE": mse_gpu,
                    "Winner": winner
                })
        
        print("\n" + "="*30)
        print(" FINAL RESULTS ")
        print("="*30)
        print(f"C++ Wins: {scores['CPP']}")
        print(f"GPU Wins: {scores['GPU']}")
        
        if scores['CPP'] > scores['GPU']:
            print("\n🏆 WINNER: C++ ENGINE")
        elif scores['GPU'] > scores['CPP']:
            print("\n🏆 WINNER: GPU ENGINE")
        else:
            print("\n🤝 DRAW")
            
        # Export
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results.csv", index=False)
        print("\nDetailed results saved to benchmark_results.csv")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
