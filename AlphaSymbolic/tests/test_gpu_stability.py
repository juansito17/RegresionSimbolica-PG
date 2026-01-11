
import torch
import numpy as np
# import pytest
import sys
import os

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.gpu.engine import TensorGeneticEngine
from core.gpu.grammar import GPUGrammar, PAD_ID

def test_protected_operators():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")
    
    # Simple grammar
    grammar = GPUGrammar(num_variables=1)
    
    # Engine helper to get ids
    def get_id(token):
        return grammar.token_to_id.get(token, PAD_ID)
        
    # Helper to run manual RPN
    from core.gpu.evaluation import GPUEvaluator
    evaluator = GPUEvaluator(grammar, device)
    
    # 1. Test Division by Zero
    # RPN: x0, 0, /
    # Note: 0 is not a token usually, we use C=0.0
    # Or 1, 1, -, / (1/(1-1))
    
    # Let's construct RPN tensor
    # "1 1 - " -> 0. "1" -> 1. "1 1 1 - /" -> 1/0
    rpn_div_zero = [
        get_id('1'), get_id('1'), get_id('1'), get_id('-'), get_id('/')
    ]
    
    # 2. Test Log Negative
    # "1 1 - 1 -" -> -1. "log"
    rpn_log_neg = [
         get_id('1'), get_id('1'), get_id('-'), get_id('1'), get_id('-'), # -1
         get_id('log')
    ]
    
    # 3. Test Sqrt Negative
    rpn_sqrt_neg = [
        get_id('1'), get_id('neg'), get_id('sqrt')
    ]
    
    # 4. Test Exp Explosion (Check if 'exp' exists, else use pow high)
    if 'exp' in grammar.token_to_id:
        rpn_exp_inf = [
           get_id('5'), get_id('5'), get_id('*'), get_id('5'), get_id('*'), get_id('exp') 
        ]
    else:
        # User e^x via 'e' 'x' 'pow'? No, maybe just big pow
        # 10 ^ 50
        rpn_exp_inf = [
           get_id('5'), get_id('5'), get_id('*'), get_id('2'), get_id('*'), # 50
           get_id('5'), get_id('2'), get_id('*'), # 10
           get_id('pow') 
        ]

    # 5. Test Pow Negative Base ( -1 ^ 0.5 ) -> NaN
    # "1 1 - 1 -" -> -1. "0.5" -> C.
    rpn_pow_nan = [
        get_id('1'), get_id('1'), get_id('-'), get_id('1'), get_id('-'), # -1
        get_id('C'), # We need constant 0.5. Constants are passed separately.
        get_id('pow')
    ]
    
    # Batch them
    max_len = 30
    
    def pad(l):
        return l + [PAD_ID] * (max_len - len(l))
    
    population = torch.tensor([
        pad(rpn_div_zero),
        pad(rpn_log_neg),
        pad(rpn_sqrt_neg),
        pad(rpn_exp_inf),
        pad(rpn_pow_nan)
    ], dtype=torch.long, device=device)
    
    x = torch.tensor([[0.5]], dtype=torch.float64, device=device) 
    y = torch.tensor([1.0], dtype=torch.float64, device=device)
    
    # Constants: Row 4 needs 0.5 at index 0.
    constants = torch.randn(5, 5, dtype=torch.float64, device=device)
    constants[4, 0] = 0.5
    
    print("\nEvaluating protected operators...")
    # Using evaluate_batch_full to see raw values
    # evaluate_batch returns RMSE, we want raw logic.
    # We can use _run_vm from evaluator directly if we access protected method
    
    preds, sp, err = evaluator._run_vm(population, x, constants)
    
    print(f"Preds: {preds}")
    print(f"Error Flags: {err}")
    print(f"Stack Ptr: {sp}")
    
    # Assertions
    # Div Zero -> Should be 1e30 (Protected) or 1
    # Check cupy_vm logic: if abs(b) < 1e-9: res = 1e30
    assert preds[0].item() > 1e10, f"DivZero failed: {preds[0]}"
    
    # Log Neg -> Should be -1e30
    # Check logic: if a <= 1e-9: res = -1e30
    assert preds[1].item() < -1e10, f"LogNeg failed: {preds[1]}"
    
    # Sqrt Neg -> Should operate on abs(a) -> sqrt(1) = 1
    # Check logic: sqrt(abs(a))
    assert abs(preds[2].item() - 1.0) < 1e-5, f"SqrtNeg failed: {preds[2]}"
    
    # Exp Inf -> Should be 1e30
    assert preds[3].item() > 1e10, f"ExpInf failed: {preds[3]}"
    
    # Pow NaN -> Should be 1e30
    assert preds[4].item() > 1e10, f"PowNaN failed: {preds[4]}"
    
    print("Protected Operators Test Passed!")

def test_benchmark_crash_repro():
    # Attempt to reproduce "Invalid" generation
    # The benchmark logs "Formula: Invalid..." or "Formula: x0..." (High MSE)
    # This might be valid tokens but producing NaN/Inf which propagate to MSE.
    pass

if __name__ == "__main__":
    test_protected_operators()
