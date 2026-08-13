import torch
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ALPHASYMBOLIC_DIR = os.path.join(_REPO_ROOT, "WarpSymbolic")
if _ALPHASYMBOLIC_DIR not in sys.path:
    sys.path.insert(0, _ALPHASYMBOLIC_DIR)
    sys.path.insert(0, _REPO_ROOT)

from warpsymbolic.gpu.pareto import ParetoOptimizer

def test_pareto_crowding_boundary():
    print("\nChecking Pareto Crowding Boundary Value Logic...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = ParetoOptimizer(device=device, max_front_size=50)
    
    # Create individuals where several share the absolute minimum fitness
    # AND are in the same front. Best way is to give them all the same fitness and complexity.
    fitness = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float32)
    complexity = torch.tensor([5.0, 5.0, 5.0, 3.0, 2.0], device=device, dtype=torch.float32)
    
    print("Fitness values:", fitness.tolist())
    print("Complexity values:", complexity.tolist())
    
    ranks, crowding = optimizer.compute_ranks_and_crowding(fitness, complexity)
    
    print("Ranks:", ranks.tolist())
    print("Crowding Distances:", crowding.tolist())
    
    # The individuals at index 0, 1, and 2 all share the minimum fitness (1.0).
    # In a proper NSGA-II implementation for boundary preservation, ALL of them 
    # should get infinite crowding distance because they describe the boundary.
    # The bug in B2 is that `args_sort` makes only the *first* one (index 0) infinite,
    # and maybe the *last* one if it's the maximum.
    
    # We expect indices 0, 1, 2 to have inf crowding because they are at the min boundary of fitness.
    is_inf = torch.isinf(crowding)
    bug_present = False
    
    if not is_inf[0] or not is_inf[1] or not is_inf[2]:
         print("CONFIRMED BUG B2: Not all individuals sharing the boundary minimum received 'inf' crowding distance.")
         print("Specifically:")
         for i in range(3):
             print(f"Index {i} (fitness {fitness[i].item()}): is_inf = {is_inf[i].item()}, value = {crowding[i].item()}")
         bug_present = True
    else:
         print("SUCCESS: Bug B2 is NOT present. All boundary value duplicates received 'inf'.")
         
    if bug_present:
        sys.exit(1)
        
if __name__ == "__main__":
    test_pareto_crowding_boundary()
