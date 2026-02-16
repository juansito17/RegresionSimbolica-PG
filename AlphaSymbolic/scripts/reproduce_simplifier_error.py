
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.gpu.grammar import GPUGrammar, PAD_ID
from core.gpu.gpu_simplifier import GPUSymbolicSimplifier

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Init Grammar and Simplifier
    grammar = GPUGrammar(num_variables=1)
    simplifier = GPUSymbolicSimplifier(grammar, device, dtype=torch.float32)
    
    # Create random population
    B = 100
    L = 30
    population = torch.randint(0, len(grammar.id_to_token), (B, L), device=device, dtype=torch.uint8)
    # Ensure some PADs at end
    population[:, -5:] = PAD_ID
    
    constants = torch.randn(B, 5, device=device, dtype=torch.float32)
    
    print("Starting simplification...")
    try:
        new_pop, new_consts, n_simplified = simplifier.simplify_batch(population, constants)
        print(f"Simplified {n_simplified} formulas.")
    except Exception as e:
        print(f"Caught expected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
