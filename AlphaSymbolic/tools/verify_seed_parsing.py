
import sys
import os
import time

# Add path
sys.path.append(os.path.join(os.getcwd(), 'AlphaSymbolic'))

from core.gp_bridge import GPEngine

def verify():
    engine = GPEngine()
    if not engine.binary_path or not os.path.exists(engine.binary_path):
        print(f"Binary not found: {engine.binary_path}")
        return

    print(f"Using Binary: {engine.binary_path}")

    # Data: y = asin(x). x in [-0.5, 0.5]
    x = [0.1, 0.2, 0.3] # minimal data
    y = [0.100167, 0.201357, 0.304692] # approx asin
    
    # Pass 'asin(x)' as seed.
    # If parsing fails, engine might crash or ignore it.
    seeds = ["asin(x)"]
    
    print(f"Sending seed: {seeds[0]}")
    res = engine.run(x, y, seeds, timeout_sec=5)
    
    # Check output
    if res:
        print(f"Engine Result: {res}")
        if "asin" in res or "S" in res: # 'S' is internal char, but output usually converts back if tree_to_string is correct
            print("✅ Success: Engine returned a formula (likely containing asin).")
        else:
            print("⚠️ Engine returned a formula but maybe not asin(x).")
    else:
        print("❌ Engine failed or returned nothing.")

if __name__ == "__main__":
    verify()
