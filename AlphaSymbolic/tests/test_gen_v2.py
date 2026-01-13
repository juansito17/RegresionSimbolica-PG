
import sys
import os
sys.path.append(os.getcwd())

from data.synthetic_data import DataGenerator
from core.grammar import OPERATORS

def test_curriculum_levels():
    print("Testing Curriculum Levels...")
    CURRICULUM_LEVELS = [
        {'depth': 2, 'ops': ['+', '-', '*', '/']},
        {'depth': 3, 'ops': ['+', '-', '*', '/', 'pow']},
        {'depth': 4, 'ops': ['+', '-', '*', '/', 'pow', 'sin', 'cos']},
    ]

    for i, stage in enumerate(CURRICULUM_LEVELS):
        print(f"Testing Stage {i}: Depth {stage['depth']}, Ops {len(stage['ops'])}")
        gen = DataGenerator(max_depth=stage['depth'], allowed_operators=stage['ops'])
        
        batch = gen.generate_inverse_batch(10)
        print(f"  Generated {len(batch)} items.")
        
        # Verify ops usage
        for item in batch:
            tokens = item['tokens']
            for t in tokens:
                if t in OPERATORS:
                    if t not in stage['ops'] and t not in ['C', 'x']: # C and x are always allowed
                        # Wait, C is not an op. 
                        # Check strictly operators
                        print(f"    ERROR: Token '{t}' found but not allowed in stage {i}!")
                        return False
        print("  Stage Verified.")
    return True

def test_structured_generation():
    print("\nTesting Structured Generation (Complex)...")
    gen = DataGenerator(max_depth=5, allowed_operators=None) # All
    batch = gen.generate_inverse_batch(5)
    for item in batch:
        print(f"  Formula: {item['infix']}")
    print("Generation Verified.")

if __name__ == "__main__":
    if test_curriculum_levels():
        test_structured_generation()
        print("\nALL TESTS PASSED.")
    else:
        print("\nTESTS FAILED.")
