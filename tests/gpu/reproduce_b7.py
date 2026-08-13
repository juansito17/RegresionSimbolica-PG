
import torch
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Try adding the specific directory where the .pyd might be
_GPU_DIR = os.path.join(_REPO_ROOT, "src", "warpsymbolic", "gpu")
if _GPU_DIR not in sys.path:
    sys.path.insert(0, _GPU_DIR)

try:
    import rpn_cuda_native
    print("SUCCESS: rpn_cuda_native imported.")
except ImportError:
    # Try different search path
    try:
        from warpsymbolic.gpu import rpn_cuda_native
        print("SUCCESS: warpsymbolic.gpu.rpn_cuda_native imported.")
    except ImportError as e:
        print(f"ERROR: rpn_cuda_native not found. {e}")
        print(f"sys.path: {sys.path}")
        sys.exit(1)

def test_tournament_selection_signature():
    print("\nChecking tournament_selection signature...")
    
    B = 16
    tournament_size = 3
    if torch.cuda.is_available():
        fitness = torch.randn(B).cuda().float()
        rand_idx = torch.randint(0, B, (B, tournament_size)).cuda().long()
        selected_idx = torch.zeros(B).cuda().long()
        
        # We need more arguments based on bindings.cpp
        # py::arg("fitness"), py::arg("errors"), py::arg("rand_idx"), py::arg("rand_cases"), py::arg("selected_idx"), py::arg("lengths")
        
        print(f"Calling with 6 arguments...")
        try:
            dummy_errs = torch.empty((0, 0), dtype=torch.float32, device='cuda')
            dummy_cases = torch.empty(0, dtype=torch.int32, device='cuda')
            dummy_lengths = torch.empty(0, dtype=torch.int32, device='cuda')
            
            rpn_cuda_native.tournament_selection(fitness, dummy_errs, rand_idx, dummy_cases, selected_idx, dummy_lengths)
            print("SUCCESS: Call with 6 arguments worked!")
        except TypeError as e:
            print(f"CONFIRMED BUG B7: TypeError: {e}")
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
    else:
        print("CUDA not available. Cannot run CUDA tests.")

if __name__ == "__main__":
    test_tournament_selection_signature()
