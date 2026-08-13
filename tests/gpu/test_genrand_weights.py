import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU is unavailable in this environment", allow_module_level=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from warpsymbolic.gpu.cuda_loader import load_rpn_cuda_native

rpn_cuda_native = load_rpn_cuda_native()

print("rpn_cuda_native loaded OK")
print("Functions:", [f for f in dir(rpn_cuda_native) if not f.startswith("_")])

# Quick smoke-test generate_random_rpn with new weight params
device = torch.device("cuda")
B, L = 1000, 30
# FIX: todos los tensores deben ser uint8 (coincide con pop_dtype del kernel)
population   = torch.zeros(B, L, dtype=torch.uint8, device=device)
terminal_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8, device=device)
unary_ids    = torch.tensor([10, 11],         dtype=torch.uint8, device=device)
binary_ids   = torch.tensor([20, 21, 22],     dtype=torch.uint8, device=device)
seed = 42

try:
    rpn_cuda_native.generate_random_rpn(
        population, terminal_ids, unary_ids, binary_ids, seed,
        0.40,   # term_weight
        0.18,   # unary_weight
        0.42    # bin_weight
    )
    print(f"generate_random_rpn OK — non-zero tokens: {(population > 0).sum().item()}")
except Exception as e:
    print(f"ERROR: {e}")
