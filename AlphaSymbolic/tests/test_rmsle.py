
import torch
import numpy as np
import sys
import os



# Adjust path to find core
# sys.path.append(os.getcwd())
print(f"SYS PATH: {sys.path}")
print(f"CWD: {os.getcwd()}")



from core.gpu.evaluation import GPUEvaluator
from core.gpu.grammar import GPUGrammar
from core.gpu.config import GpuGlobals

def test_rmsle():
    print("Testing RMSLE Implementation...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # Fallback to CPU if no CUDA
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Setup Grammar and Evaluator
    # GPUGrammar takes num_variables, not a core Grammar object
    gpu_grammar = GPUGrammar(num_variables=1)
    evaluator = GPUEvaluator(gpu_grammar, device)

    # Fake Data
    # Target: [1, 10, 100]
    # Prediction 1: [1, 10, 100] (Perfect) -> RMSLE = 0
    # Prediction 2: [2, 11, 101] (Small error)
    # Prediction 3: [1, 10, 1000] (Large error on large value)
    # Prediction 4: [10, 10, 100] (Large error on small value)
    
    y_target = torch.tensor([1.0, 10.0, 100.0], device=device, dtype=torch.float64) # [Samples]
    x_dummy = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=torch.float64) # [Vars, Samples]

    # Create dummy population (not actually used by VM if we mock calls, but we need structure)
    # Actually, constructing RPN for literal values is easier.
    # We want to test the evaluate_batch logic, particularly the loss function part.
    # But evaluating fully requires valid RPN.
    # Instead, let's unit test the logic if possible, or construct RPNs that yield specific values.
    
    # Let's construct RPNs that output specific values.
    # Easier: Mock the _run_vm return.
    
    class MockEvaluator(GPUEvaluator):
        def _run_vm(self, population, x, constants):
            # Return predetermined values based on population index
            # Pop 0: Perfect
            # Pop 1: +1 error everywhere
            # Pop 2: Scale error on large value (100 -> 1000)
            # Pop 3: Scale error on small value (1 -> 10)
            
            B, N = population.shape[0], x.shape[0] # N=Samples (x is [Samples, Vars])
            
            preds = torch.zeros((B * N), device=self.device, dtype=torch.float64)
            
            # Map indices
            # Pop 0 -> [1, 10, 100]
            # Pop 1 -> [2, 11, 101]
            # Pop 2 -> [1, 10, 1000]
            # Pop 3 -> [10, 10, 100]
            
            # Fill
            for i in range(B):
                start = i * N
                if i == 0:
                    vals = [1.0, 10.0, 100.0]
                elif i == 1:
                    vals = [2.0, 11.0, 101.0]
                elif i == 2:
                    vals = [1.0, 10.0, 1000.0]
                elif i == 3:
                    vals = [10.0, 10.0, 100.0]
                else:
                    vals = [0.0, 0.0, 0.0]
                    
                preds[start:start+N] = torch.tensor(vals, device=self.device, dtype=torch.float64)
                
            sp = torch.ones((B * N), device=self.device, dtype=torch.long)
            err = torch.zeros((B * N), device=self.device, dtype=torch.bool)
            
            return preds, sp, err

    mock_evaluator = MockEvaluator(gpu_grammar, device)
    
    # Create dummy population tensor
    pop_dummy = torch.zeros((4, 10), device=device, dtype=torch.long)
    
    # 1. Test RMSE (Standard)
    print("\n--- Testing RMSE ---")
    GpuGlobals.LOSS_FUNCTION = 'RMSE'
    rmse_scores = mock_evaluator.evaluate_batch(pop_dummy, x_dummy, y_target, constants=None)
    
    print(f"Perfect (Pop 0): {rmse_scores[0]:.4f}")
    print(f"Offset +1 (Pop 1): {rmse_scores[1]:.4f}")
    print(f"Large Val Error 100->1000 (Pop 2): {rmse_scores[2]:.4f}")
    print(f"Small Val Error 1->10 (Pop 3): {rmse_scores[3]:.4f}")
    
    # RMSE Analysis:
    # Pop 2 error is (900)^2 / 3 -> sqrt(270000) ~= 519
    # Pop 3 error is (9)^2 / 3 -> sqrt(27) ~= 5.2
    # RMSE punishes Pop 2 massively more.
    
    # 2. Test RMSLE
    print("\n--- Testing RMSLE ---")
    GpuGlobals.LOSS_FUNCTION = 'RMSLE'
    rmsle_scores = mock_evaluator.evaluate_batch(pop_dummy, x_dummy, y_target, constants=None)
    
    print(f"Perfect (Pop 0): {rmsle_scores[0]:.4f}")
    print(f"Offset +1 (Pop 1): {rmsle_scores[1]:.4f}")
    print(f"Large Val Error 100->1000 (Pop 2): {rmsle_scores[2]:.4f}")
    print(f"Small Val Error 1->10 (Pop 3): {rmsle_scores[3]:.4f}")

    # RMSLE Analysis:
    # Pop 2: log(1001) - log(101) ~= 6.9 - 4.6 = 2.3
    # Pop 3: log(11) - log(2) ~= 2.4 - 0.7 = 1.7
    # They should be much closer in magnitude than RMSE.
    # Actually:
    # Pop 2 (100 vs 1000): log10(100)=2, log10(1000)=3. Diff 1.
    # Pop 3 (1 vs 10): log10(1)=0, log10(10)=1. Diff 1.
    # RMSLE should treat 1->10 and 100->1000 similarly (order of magnitude errors).
    
    # Check if Pop 2 and Pop 3 are comparable in RMSLE
    print("\nVerdict:")
    if rmsle_scores[2] < rmse_scores[2] and rmsle_scores[3] > 1.0: 
         print("SUCCESS: RMSLE reduced penalty for large values and kept sensitivity for relative errors.")
    else:
         print("OBSERVATION: Check values above.")

if __name__ == "__main__":
    test_rmsle()
