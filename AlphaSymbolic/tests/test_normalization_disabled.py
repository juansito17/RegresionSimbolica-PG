
import unittest
import numpy as np
from utils.data_utils import normalize_batch

class TestNormalizationDisabled(unittest.TestCase):
    def test_no_scaling(self):
        """Verify that normalize_batch returns raw values without scaling."""
        
        # Create data with large range
        x_raw = [np.array([[-100.0], [0.0], [5000.0]])]
        y_raw = [np.array([0.0, 1.0, 25.0])]
        
        x_norm, y_norm = normalize_batch(x_raw, y_raw)
        
        # Check X
        # Should be exactly the same (floating point tolerance)
        np.testing.assert_array_almost_equal(x_norm[0], x_raw[0], decimal=5, 
            err_msg="X values should NOT be normalized")
            
        # Check Y
        np.testing.assert_array_almost_equal(y_norm[0], y_raw[0], decimal=5,
            err_msg="Y values should NOT be normalized")
            
        print("Success: Values preserved (Not normalized)")

if __name__ == '__main__':
    unittest.main()
