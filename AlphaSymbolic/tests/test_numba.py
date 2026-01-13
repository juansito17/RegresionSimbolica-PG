
import torch
from core.gpu.numba_vm import run_vm_numba

def test_numba():
    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return
        
    device = 'cuda'
    
    # Dummy setup
    B = 2
    L = 5
    D = 3
    
    pop = torch.zeros((B, L), dtype=torch.long, device=device)
    # x + 1
    # x=1, 1=10, +=20, PAD=0
    pop[0, 0] = 1
    pop[0, 1] = 10
    pop[0, 2] = 20
    
    # x * 2 (x, 2, *)
    # x=1, 2=11, *=21
    pop[1, 0] = 1
    pop[1, 1] = 11
    pop[1, 2] = 21
    
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
    constants = torch.zeros((B, 1), dtype=torch.float64, device=device)
    
    print("Calling Numba VM...")
    try:
        preds, sp, err = run_vm_numba(
            pop, x, constants,
            0, 1, # PAD, x
            99, 99, 99, # C, pi, e
            10, 11, 12, 13, # 1, 2, 3, 5
            20, 22, 21, 23, 24, 25, # +, -, *, /, ^, % (mix IDs)
            30, 31, 32, # sin, cos, tan
            33, 34, # log, exp
            35, 36, 37, # sqrt, abs, neg
            38, 39, 40, # fact, floor, gamma
            41, 42, 43, # asin, acos, atan
            3.14, 2.71
        )
        print("Completed.")
        print("Preds:", preds)
        
        # formulas: x+1, x*2
        # x=[1,2,3]
        # x+1 -> [2,3,4]
        # x*2 -> [2,4,6]
        # Total: [2,3,4, 2,4,6]
        expected = torch.tensor([2., 3., 4., 2., 4., 6.], device=device, dtype=torch.float64)
        if torch.allclose(preds, expected):
            print("SUCCESS: Numba output matches.")
        else:
            print("FAILURE: Mismatch.")
            print("Expected:", expected)
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_numba()
