"""
Test script to verify data balancing in DataGenerator.
Checks the ratio of 'x' vs constants and the presence of 'x' in formulas.
"""
import random
from data.synthetic_data import DataGenerator
from core.grammar import CONSTANTS

def test_balance(n_samples=500):
    gen = DataGenerator(max_depth=3)
    
    x_count = 0
    const_count = 0
    formula_with_x = 0
    total_terminals = 0
    
    print(f"Generating {n_samples} samples via generate_batch...")
    
    batch = gen.generate_batch(n_samples)
    
    for item in batch:
        tokens = item['tokens']
        
        has_x = 'x' in tokens
        if has_x:
            formula_with_x += 1
            
        for t in tokens:
            if t == 'x':
                x_count += 1
                total_terminals += 1
            elif t in CONSTANTS:
                const_count += 1
                total_terminals += 1
                
    print("\n=== DATA BALANCE RESULTS ===")
    print(f"Total terminals analyzed: {total_terminals}")
    print(f"  'x' count: {x_count} ({100*x_count/total_terminals:.1f}%)")
    print(f"  Constants: {const_count} ({100*const_count/total_terminals:.1f}%)")
    print(f"\nFormulas containing 'x': {formula_with_x}/{n_samples} ({100*formula_with_x/n_samples:.1f}%)")
    print("-" * 30)
    
    # Success criteria
    balance_ok = 40 <= (100*x_count/total_terminals) <= 60
    presence_ok = (100*formula_with_x/n_samples) >= 85
    
    if balance_ok and presence_ok:
        print("OK - TEST PASSED: Data is balanced and contains enough 'x'.")
    else:
        print("FAIL - TEST FAILED: Bias detected.")
        if not balance_ok: print(f"  Low balance: {100*x_count/total_terminals:.1f}% x")
        if not presence_ok: print(f"  Low presence: {100*formula_with_x/n_samples:.1f}% formulas with x")

if __name__ == "__main__":
    test_balance()
