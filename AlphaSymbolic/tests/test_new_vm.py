
import torch
import torch.jit
from core.gpu.jit_vm import run_vm_jit

def test_new_vm():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device}")
    
    # 1. Setup Dummy Data
    B = 2
    L = 5
    D = 3
    
    population = torch.zeros((B, L), dtype=torch.long, device=device)
    # Formula 1: x + 1 (RPN: x, 1, +)
    # IDs: PAD=0, x=1, 1=10, +=20
    # Let's define simple IDs
    PAD_ID=0
    id_x=1
    id_1=10
    op_add=20
    
    population[0, 0] = id_x
    population[0, 1] = id_1
    population[0, 2] = op_add
    
    # Formula 2: x * x (RPN: x, x, *)
    op_mul=21
    population[1, 0] = id_x
    population[1, 1] = id_x
    population[1, 2] = op_mul
    
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
    # Expected results:
    # F1(x): [2, 3, 4]
    # F2(x): [1, 4, 9]
    # Output shape: [B*D] = [6]
    # [2, 3, 4, 1, 4, 9]
    
    constants = torch.zeros((B, 1), dtype=torch.float64, device=device)
    var_ids = [id_x]
    
    # Call JIT VM
    print("Calling JIT VM...")
    try:
        res, sp, err = run_vm_jit(
            population, x, constants,
            PAD_ID, id_x, var_ids,
            999, 999, 999,
            [id_1], [1.0],
            op_add, 999, op_mul, 999, 999, 999,
            999, 999, 999,
            999, 999,
            999, 999, 999,
            999, 999, 999,
            999, 999, 999,
            3.14, 2.71
        )
    except Exception as e:
        print(f"CRITICAL JIT ERROR: {e}")
        return
    
    print("Result shape:", res.shape)
    print("Result values:", res)
    
    expected = torch.tensor([2., 3., 4., 1., 4., 9.], device=device, dtype=torch.float64)
    if torch.allclose(res, expected):
        print("SUCCESS: JIT VM produced correct results.")
    else:
        print("FAILURE: Indirect results.")
        print("Expected:", expected)

if __name__ == "__main__":
    test_new_vm()
