import sys
sys.path.append(r"c:\Users\juanu\Documents\Repositorios\Algoritmo-Genetico-de-Formulas")

import torch
from warpsymbolic.gpu.config import GpuGlobals
from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.evaluation import GPUEvaluator
from warpsymbolic.gpu.operators import GPUOperators
from warpsymbolic.gpu.optimization import GPUOptimizer

def test_autograd_jacobian():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Test skipped, no CUDA")
        return
        
    # Enable necessary ops for this test formula: C0 * sin(C1*x0 + x1)
    GpuGlobals.USE_OP_SIN = True
    GpuGlobals.USE_OP_MULT = True
    GpuGlobals.USE_OP_PLUS = True
    
    # Simple grammar
    # Vocab and Arities are built automatically by GPUGrammar
    # We use num_variables=2 for x0, x1
    grammar = GPUGrammar(num_variables=2)
    evaluator = GPUEvaluator(grammar, device=device)
    operators = GPUOperators(grammar, device=device, pop_size=1)
    optimizer = GPUOptimizer(evaluator, operators, device=device)
    
    print("Testing Native Autograd...")
    if not optimizer._has_fused_pso:
        print("ERROR: Native CUDA module not built with fused_pso/autograd")
        return
        
    PAD = grammar.token_to_id["<PAD>"]
    
    # Formula: C0 * sin(C1 * x0 + x1)
    # RPN: C0 C1 x0 * x1 + sin *
    form = [
        grammar.token_to_id["C"],
        grammar.token_to_id["C"],
        grammar.token_to_id["x0"],
        grammar.token_to_id["*"],
        grammar.token_to_id["x1"],
        grammar.token_to_id["+"],
        grammar.token_to_id["sin"],
        grammar.token_to_id["*"]
    ]
    
    # Pad to Length 16
    pad_len = 16 - len(form)
    form += [PAD] * pad_len
    
    population = torch.tensor([form], dtype=torch.uint8, device=device)
    
    # Variables: D=5
    x = torch.linspace(0.1, 1.0, 5, dtype=torch.float64, device=device).unsqueeze(0).repeat(2, 1) # [2, 5]
    
    # Target (e.g., C0=2.0, C1=3.0)
    # y = 2.0 * sin(3.0 * x0 + x1)
    y_target = 2.0 * torch.sin(3.0 * x[0] + x[1])
    
    # Constants initialization: C0 = 1.0, C1 = 1.0
    constants = torch.tensor([[1.0, 1.0]], dtype=torch.float64, device=device)
    
    print(f"Initial MS E: {evaluator.evaluate_batch(population, x, y_target, constants)[0].item():.6f}")
    
    # Test L-BFGS Optimization!
    opt_c, opt_mse = optimizer.lbfgs_optimize_top_k(population, constants, x, y_target, top_k=1, max_iter=20)
    
    print(f"Optimized C0: {opt_c[0, 0].item():.6f}")
    print(f"Optimized C1: {opt_c[0, 1].item():.6f}")
    print(f"Optimized MSE: {opt_mse[0].item():.6f}")

if __name__ == "__main__":
    test_autograd_jacobian()
