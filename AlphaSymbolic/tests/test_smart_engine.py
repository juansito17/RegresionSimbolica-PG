
import torch
import numpy as np
import time
from core.gpu import TensorGeneticEngine
from core.gpu.config import GpuGlobals

def test_smart_engine():
    print("=== Testing Smart Engine Upgrades ===")
    
    # 1. Setup Linear Data (y = 2x + 5) for Sniper
    # Note: Engine expects x to be float
    x = np.linspace(1, 10, 10).astype(np.float32)
    y_lin = 2.0 * x + 5.0
    
    # Force GPU mode if available, but config might force float32/64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Test Sniper Integration (Should find it instantly)
    # We use a small population to speed up init
    # But wait, Sniper runs BEFORE init.
    print("\n--- Test 1: Sniper Linear Detection ---")
    engine = TensorGeneticEngine(pop_size=1000, max_len=20, device=device)
    
    # Capture output? Just checking return
    # We can pass callback to see if it finishes instantly
    def callback(gen, rmse, rpn, consts, is_new, island):
        if is_new:
            print(f"Callback: Gen {gen}, RMSE {rmse:.6f}")
    
    # Run
    # It should print "[Engine] The Sniper found..."
    best_formula = engine.run(x, y_lin, timeout_sec=2, callback=callback)
    print(f"Result: {best_formula}")
    
    if "2" in str(best_formula) and "5" in str(best_formula):
        print("SUCCESS: Sniper detected linear pattern.")
    else:
        print("WARNING: Sniper might have failed or Random search found something else.")

    # 3. Test Residual Boosting
    # We need a problem where it might get stuck.
    # y = 2x + sin(x)
    # If we disable sin(x) in grammar it fails.
    # But let's assume grammar has sin.
    # To force stagnation, we can set Stagnation Limit very low.
    
    print("\n--- Test 2: Residual Boosting ---")
    # Temporarily lower stagnation limit
    ORIG_LIMIT = GpuGlobals.STAGNATION_LIMIT
    GpuGlobals.STAGNATION_LIMIT = 5 # Trigger very fast
    
    y_boost = 2.0 * x + np.sin(x)
    
    # We want it to find '2x' then boost with 'sin(x)'?
    # Or 'sin(x)' is not linear/geometric.
    # Sniper assumes linear/geometric residuals.
    # Residue of 2x+sin(x) vs 2x is sin(x). Sniper check_linear/geometric on sin(x) will fail.
    # Residue of 2x+sin(x) vs sin(x) is 2x. Sniper handles 2x!
    # So if GP finds sin(x) (unlikely as first guess) then it works.
    
    # Let's try y = x + 10.
    # Residue of x is 10. Sniper finds 10.
    # Residue of 10 is x. Sniper finds x.
    # This checks Additive Boosting.
    
    y_add = x + 10.0
    
    # We want to ensure it doesn't solve it by pure luck in gen 1.
    # We can seed it with "x" and see if it boosts?
    # Passing seeds=["x"]
    
    engine_boost = TensorGeneticEngine(pop_size=1000, max_len=20, device=device)
    best_boost = engine_boost.run(x, y_add, seeds=["x"], timeout_sec=5, callback=callback)
    print(f"Result Boost: {best_boost}")
    
    GpuGlobals.STAGNATION_LIMIT = ORIG_LIMIT
    
    print("\nDone.")

if __name__ == "__main__":
    test_smart_engine()
