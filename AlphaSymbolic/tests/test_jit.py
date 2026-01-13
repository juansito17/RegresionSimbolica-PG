
import torch
import time

# Constants (Must be inline or passed for JIT)
PAD_ID = 0

@torch.jit.script
def run_vm_jit(population: torch.Tensor, x: torch.Tensor, 
               op_add: int, op_sub: int, op_mul: int, op_div: int, 
               op_sin: int, op_cos: int, op_log: int, op_exp: int,
               op_abs: int, op_neg: int, 
               stack_size: int = 10) -> torch.Tensor:
    
    B, L = population.shape
    D = x.shape[0]
    eff_B = B * D
    
    # Expand X: [D] -> [1, D] -> [B, D] -> [B*D]
    # To avoid complex expansion inside JIT, assume x is already expanded or handle simple broadcasting
    # Let's assume evaluation logic:
    # We process [B, D] in parallel?
    # Original _run_vm processes [B * D] logic implicitly by expanding population?
    # Original:
    # x_expanded = x.unsqueeze(0).expand(B, D).reshape(-1)
    # pop_expanded = population.unsqueeze(1).expand(B, D, L).reshape(-1, L)
    
    # We'll do expansion inside for correctness matching original
    x_expanded = x.unsqueeze(0).expand(B, D).reshape(-1)
    pop_expanded = population.unsqueeze(1).expand(B, D, L).reshape(-1, L)
    
    # Stack: [B*D, stack_size]
    stack = torch.zeros((B*D, stack_size), device=population.device, dtype=torch.float64)
    sp = torch.zeros(B*D, device=population.device, dtype=torch.long)
    
    for i in range(L):
        token = pop_expanded[:, i]
        
        # Mask for padding
        active = (token != PAD_ID)
        
        # 1. Terminal (Active Variables & Constants)
        # Simplified: Assume x0 is ID 1 for test
        is_x = (token == 1)
        if is_x.any():
            # Push x
            idx = torch.clamp(sp, 0, stack_size - 1)
            # stack[active & is_x, idx] = ... indexing is tricky in JIT
            # Use scatter style
            # If is_x, val = x_expanded. Else val = 0 (ignored)
            # We need to update stack at sp
            
            # Vectorized update:
            # New value to push?
            # It's either x or a constant.
            # For this benchmark, let's just do operations.
            pass

    # Returning dummy for compilation check
    return stack[:, 0]

def test_jit():
    print("Testing TorchScript compilation on Windows...")
    try:
        # Create dummy inputs
        pop = torch.randint(0, 10, (100, 10), dtype=torch.long).cuda()
        x = torch.randn(50, dtype=torch.float64).cuda()
        
        # Warmup
        run_vm_jit(pop, x, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        print("JIT Compilation Successful!")
        return True
    except Exception as e:
        print(f"JIT Failed: {e}")
        return False

if __name__ == "__main__":
    test_jit()
