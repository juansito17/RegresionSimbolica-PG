import torch
import torch.nn as nn
import sys
import os

# Add parent to path to allow AlphaSymbolic import if run from outside
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AlphaSymbolic.core.gpu.evaluation import GPUEvaluator
from AlphaSymbolic.core.grammar import ExpressionTree
from AlphaSymbolic.core.gpu.grammar import GPUGrammar

def debug_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Setup
    grammar = GPUGrammar(num_variables=1)
    evaluator = GPUEvaluator(grammar, device=device)
    
    # 2. Create Bad Population (All PADs -> sp=0)
    # PAD_ID is 0.
    pop_size = 5
    L = 30
    population = torch.zeros((pop_size, L), dtype=torch.long, device=device)
    
    # Make ind 0 totally empty (PADs) -> Should be Invalid, RMSE ~ 1e75
    
    # Make ind 1 valid: "x0" (assuming x0 is token 1?)
    # grammar.active_variables is ['x0'] usually (if num_variables=1)
    # Check grammar ids
    x0_id = grammar.token_to_id['x0']
    print(f"x0 ID: {x0_id}")
    population[1, 0] = x0_id 
    # RPN: x0 -> sp=1. Valid.
    
    # 3. Targets
    x_val = torch.linspace(1, 5, 5, device=device).unsqueeze(1) # [5, 1]
    y_tgt = x_val.squeeze(1).clone() # y = x
    
    x_flat = x_val.view(-1)
    
    print(f"Population: {population.shape}")
    print(f"X: {x_flat.shape}")
    
    # 4. Run Step-by-Step
    # Mimic evaluate_batch logic manually to trace
    
    # Check VM
    out_preds, out_sp, out_err = evaluator._run_vm(population, x_flat, None)
    
    print("\n--- VM OUTPUT ---")
    print(f"Out SP: {out_sp.view(pop_size, -1)}")
    print(f"Out Err: {out_err.view(pop_size, -1)}")
    
    sp = out_sp
    has_error = out_err
    
    is_valid = (sp == 1) & (~has_error)
    print(f"\nIs Valid: {is_valid.view(pop_size, -1)}")
    
    final_preds = out_preds
    
    # Masking
    masked_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                              final_preds, 
                              torch.tensor(1e300, device=device, dtype=torch.float64))
    
    print(f"\nMasked Preds Sample (Ind 0): {masked_preds.view(pop_size, -1)[0]}")
    print(f"Masked Preds Sample (Ind 1): {masked_preds.view(pop_size, -1)[1]}")
    
    # RMSE Calc
    preds_matrix = masked_preds.view(pop_size, -1)
    target_matrix = y_tgt.unsqueeze(0).expand(pop_size, -1)
    
    diff = preds_matrix - target_matrix
    sq_diff = diff**2
    mse = torch.mean(sq_diff, dim=1)
    
    print(f"\nMSE: {mse}")
    
    rmse = torch.sqrt(torch.where(torch.isnan(mse) | torch.isinf(mse), 
                                  torch.tensor(1e150, device=device, dtype=torch.float64), 
                                  mse))
                                  
    print(f"\nRMSE: {rmse}")

if __name__ == "__main__":
    debug_run()
