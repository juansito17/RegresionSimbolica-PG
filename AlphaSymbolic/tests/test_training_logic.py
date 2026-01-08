
import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.app_training import get_allowed_token_mask
from core.grammar import VOCABULARY, TOKEN_TO_ID, VARIABLES

class TestTrainingLogic(unittest.TestCase):
    def test_mask_includes_variables(self):
        """Verify that get_allowed_token_mask includes x0..x9 variables."""
        print("\n=== Testing Token Masking Logic ===")
        device = torch.device('cpu')
        vocab_size = len(VOCABULARY)
        
        # Test stage 0 (Arithmetic)
        mask = get_allowed_token_mask(0, vocab_size, device)
        
        print(f"Vocab Size: {vocab_size}")
        print("Checking Variables:", VARIABLES)
        
        for var in VARIABLES:
            if var in TOKEN_TO_ID:
                idx = TOKEN_TO_ID[var]
                self.assertEqual(mask[idx].item(), 1.0, f"Variable {var} should be allowed (1.0), but got {mask[idx].item()}")
                print(f"✓ Variable {var} is allowed.")
            else:
                print(f"Warning: {var} not in VOCABULARY")

        # Test random stage
        mask_adv = get_allowed_token_mask(4, vocab_size, device)
        for var in VARIABLES:
             if var in TOKEN_TO_ID:
                idx = TOKEN_TO_ID[var]
                self.assertEqual(mask_adv[idx].item(), 1.0, f"Variable {var} should be allowed in adv stage")

if __name__ == '__main__':
    unittest.main()
