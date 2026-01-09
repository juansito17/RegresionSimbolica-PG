
import torch
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
from AlphaSymbolic.core.gpu.engine import GPUGrammar

def test_gpu_variable_mapping():
    print("Testing GPU Variable Mapping (x0 fix)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CASE 1: 1 Variable
    grammar = GPUGrammar(num_variables=1)
    engine = TensorGeneticEngine(num_variables=1, device=device)
    
    x = torch.linspace(-1, 1, 10, device=device, dtype=torch.float64) # [10]
    y_target = x + 1.0 # Target: x0 + 1
    
    # Formula: x0 + 1 -> RPN: [x0, 1, +]
    id_x0 = grammar.token_to_id['x0']
    id_1 = grammar.token_to_id['1']
    id_add = grammar.token_to_id['+']
    pop = torch.tensor([[id_x0, id_1, id_add, -1, -1]], device=device) # [1, 5]
    
    rmse = engine.evaluate_batch(pop, x, y_target)
    print(f"CASE 1 (1 Var) RMSE for 'x0 + 1': {rmse.item()}")
    assert rmse.item() < 1e-7, f"Expected near-zero RMSE, got {rmse.item()}"

    # CASE 2: 3 Variables
    grammar3 = GPUGrammar(num_variables=3)
    engine3 = TensorGeneticEngine(num_variables=3, device=device)
    
    # x: [10, 3]
    x3 = torch.stack([
        torch.linspace(-1, 1, 10),
        torch.linspace(0, 10, 10),
        torch.linspace(-5, 5, 10)
    ], dim=1).to(device, dtype=torch.float64)
    
    y_target3 = x3[:, 2] # Target: x2
    
    # Formula: x2 -> RPN: [x2]
    id_x2 = grammar3.token_to_id['x2']
    pop3 = torch.tensor([[id_x2, -1, -1, -1, -1]], device=device)
    
    rmse3 = engine3.evaluate_batch(pop3, x3, y_target3)
    print(f"CASE 2 (3 Vars) RMSE for 'x2': {rmse3.item()}")
    assert rmse3.item() < 1e-7, f"Expected near-zero RMSE, got {rmse3.item()}"

def test_gpu_inf_handling():
    print("\nTesting GPU Inf Handling (Stability fix)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grammar = GPUGrammar(num_variables=1)
    engine = TensorGeneticEngine(num_variables=1, device=device)
    
    x = torch.linspace(1, 10, 10, device=device, dtype=torch.float64)
    y_target = x
    
    # Formula 1: 1 / 0 -> Penalty
    id_1 = grammar.token_to_id['1']
    id_div = grammar.token_to_id['/']
    pop_div0 = torch.tensor([[id_1, grammar.token_to_id.get('0', -1), id_div, -1, -1]], device=device) 
    # Wait, '0' might not be in grammar literals. Using x - x = 0
    id_sub = grammar.token_to_id['-']
    id_x0 = grammar.token_to_id['x0']
    # RPN: [1, x0, x0, -, /]
    pop_div0 = torch.tensor([[id_1, id_x0, id_x0, id_sub, id_div]], device=device)
    
    rmse = engine.evaluate_batch(pop_div0, x, y_target)
    print(f"RMSE for '1 / (x0 - x0)': {rmse.item()}")
    assert rmse.item() >= 1e140, f"Expected high penalty for division by zero, got {rmse.item()}"
    
    # Formula 2: log(-1) -> Penalty
    id_neg = grammar.token_to_id['neg']
    id_log = grammar.token_to_id['log']
    # RPN: [1, neg, log]
    pop_log_neg = torch.tensor([[id_1, id_neg, id_log, -1, -1]], device=device)
    
    rmse_log = engine.evaluate_batch(pop_log_neg, x, y_target)
    print(f"RMSE for 'log(-1)': {rmse_log.item()}")
    assert rmse_log.item() >= 1e140, f"Expected high penalty for log of negative, got {rmse_log.item()}"

if __name__ == "__main__":
    try:
        test_gpu_variable_mapping()
        test_gpu_inf_handling()
        print("\nAll GPU stability tests PASSED!")
    except Exception as e:
        print(f"\nTests FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
