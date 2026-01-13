import torch
import numpy as np
from ui.app_core import load_model
from search.beam_search import beam_solve

def test_diversity():
    print("--- Testing Beam Search Diversity ---")
    
    # 1. Load Model
    print("Loading Model...")
    status, dev = load_model(preset_name='pro') # or 'lite'
    from ui.app_core import MODEL, DEVICE
    
    if MODEL is None:
        print("Error: Model not loaded.")
        return

    # 2. Create Dummy Problem (Simple x^2 + 1)
    print("Generating problem: y = x^2 + 1")
    x = np.linspace(-3, 3, 10)
    y = x**2 + 1
    
    # 3. Run Beam Search
    beam_width = 50
    print(f"Running Beam Search (Width={beam_width})...")
    results = beam_solve(x, y, MODEL, DEVICE, beam_width=beam_width)
    
    # 4. Analyze Diversity
    if not results:
        print("No results found.")
        return
        
    formulas = [r['formula'] for r in results]
    unique_formulas = []
    seen = set()
    
    for f in formulas:
        if f not in seen and not f.startswith("Partial"):
            unique_formulas.append(f)
            seen.add(f)
            
    print(f"\nTotal Candidates: {len(formulas)}")
    print(f"Unique Valid Formulas: {len(unique_formulas)}")
    
    print("\n--- Top 10 Unique Candidates ---")
    for i, f in enumerate(unique_formulas[:10]):
        print(f"{i+1}. {f}")
        
    if len(unique_formulas) >= 6:
        print("\n[SUCCESS] Enough diversity for 6-core Sniper Mode.")
    else:
        print(f"\n[WARNING] Only {len(unique_formulas)} unique seeds found. Some workers will duplicate effort.")

if __name__ == "__main__":
    test_diversity()
