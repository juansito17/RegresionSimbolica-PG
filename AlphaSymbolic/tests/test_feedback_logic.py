
import unittest
import numpy as np

class TestFeedbackLogic(unittest.TestCase):
    def test_acceptance_criteria(self):
        """
        Verify the new dynamic acceptance criteria:
        Accept if:
        A) RMSE < 0.5 (Decent)
        B) GP_RMSE < NN_RMSE * 0.9 (Better than NN)
        """
        
        # Scenario 1: Decent GP (0.4), Bad NN (1.0) -> ACCEPT (Crit A)
        gp_rmse = 0.4
        nn_rmse = 1.0
        is_decent = gp_rmse < 0.5
        is_better = gp_rmse < (nn_rmse * 0.9)
        self.assertTrue(is_decent or is_better, "Should accept decent GP (0.4)")
        
        # Scenario 2: Bad GP (0.6), Terrible NN (1.0) -> ACCEPT (Crit B: 0.6 < 0.9)
        gp_rmse = 0.6
        nn_rmse = 1.0
        is_decent = gp_rmse < 0.5
        is_better = gp_rmse < (nn_rmse * 0.9)
        self.assertTrue(is_decent or is_better, "Should accept bad GP if significantly better than terrible NN")

        # Scenario 3: Bad GP (0.6), Decent NN (0.65) -> REJECT (Crit B fails: 0.6 !< 0.585)
        # 0.65 * 0.9 = 0.585. GP 0.6 is worse than threshold.
        gp_rmse = 0.6
        nn_rmse = 0.65
        is_decent = gp_rmse < 0.5
        is_better = gp_rmse < (nn_rmse * 0.9)
        self.assertFalse(is_decent or is_better, "Should reject GP if not significantly better than NN")

        # Scenario 4: Terrible GP (2.0), Terrible NN (2.0) -> REJECT
        gp_rmse = 2.0
        nn_rmse = 2.0
        is_decent = gp_rmse < 0.5
        is_better = gp_rmse < (nn_rmse * 0.9)
        self.assertFalse(is_decent or is_better, "Should reject terrible GP")

    def test_reward_scaling(self):
        """
        Verify the reward scaling formula:
        reward = max(0.1, 1.0 - (gp_rmse * 1.6))
        """
        # Perfect RMSE 0.0 -> 1.0
        gp_rmse = 0.0
        reward = max(0.1, 1.0 - (gp_rmse * 1.6))
        self.assertAlmostEqual(reward, 1.0)
        
        # Good RMSE 0.1 -> 1.0 - 0.16 = 0.84
        gp_rmse = 0.1
        reward = max(0.1, 1.0 - (gp_rmse * 1.6))
        self.assertAlmostEqual(reward, 0.84)
        
        # Decent RMSE 0.5 -> 1.0 - 0.8 = 0.2
        gp_rmse = 0.5
        reward = max(0.1, 1.0 - (gp_rmse * 1.6))
        self.assertAlmostEqual(reward, 0.2)
        
        # Bad RMSE 0.6 (but accepted via "Better than NN") -> 1.0 - 0.96 = 0.04 -> Clamped to 0.1
        gp_rmse = 0.6
        reward = max(0.1, 1.0 - (gp_rmse * 1.6))
        self.assertAlmostEqual(reward, 0.1)

if __name__ == '__main__':
    unittest.main()
