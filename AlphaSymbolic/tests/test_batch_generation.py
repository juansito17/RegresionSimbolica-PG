
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.synthetic_data import DataGenerator

def test_batch_generation():
    print("=== Testing Batch Generation ===")
    
    for num_vars in [1, 2, 3]:
        print(f"\n--- Testing with {num_vars} variables ---")
        data_gen = DataGenerator(max_depth=2, num_variables=num_vars)
        
        batch_size = 128
        batch = data_gen.generate_batch(batch_size, point_count=10)
        
        print(f"Requested: {batch_size}, Generated: {len(batch)}")
        
        if len(batch) > 0:
            print(f"Sample formula: {batch[0]['infix']}")
            print(f"Sample x shape: {batch[0]['x'].shape}")
            print(f"Sample y shape: {batch[0]['y'].shape}")
        
        if len(batch) < batch_size * 0.5:
            print(f"⚠️ WARNING: Only {len(batch)}/{batch_size} formulas generated!")

if __name__ == "__main__":
    test_batch_generation()
