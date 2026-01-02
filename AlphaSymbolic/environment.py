import gymnasium as gym
from gymnasium import spaces
import numpy as np
from grammar import VOCABULARY, OPERATORS, TOKEN_TO_ID, ExpressionTree
from synthetic_data import DataGenerator

class SymbolicEnv(gym.Env):
    def __init__(self, max_length=50):
        super(SymbolicEnv, self).__init__()
        
        self.vocab_size = len(VOCABULARY)
        self.max_length = max_length
        self.vocab = VOCABULARY
        
        # Action space: Choose a token from the vocabulary
        self.action_space = spaces.Discrete(self.vocab_size)
        
        # Observation space: 
        # 1. Current token sequence (padded)
        # 2. X values (fixed size for simplicity)
        # 3. Y values
        # For this prototype we will expose a dictionary observation
        self.observation_space = spaces.Dict({
            "sequence": spaces.Box(low=0, high=self.vocab_size, shape=(max_length,), dtype=np.int32),
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            "y": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        })
        
        self.data_gen = DataGenerator(max_depth=4)
        self.current_problem = None
        self.current_sequence = []
        self.open_branches = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate a new problem (X, Y)
        # In a real scenario, this could be sampled from a fixed dataset
        batch = self.data_gen.generate_batch(1, point_count=10)
        self.current_problem = batch[0]
        
        self.current_sequence = []
        self.open_branches = 1 # Start expecting a root node
        
        return self._get_obs(), {}

    def step(self, action_id):
        token = self.vocab[action_id]
        self.current_sequence.append(token)
        
        # Update open branches
        if token in OPERATORS:
            arity = OPERATORS[token]
            self.open_branches += (arity - 1)
        else:
            self.open_branches -= 1
            
        term = False
        trunc = False
        reward = 0.0
        
        # Check completion
        if self.open_branches == 0:
            term = True
            # Tree is complete, evaluate
            reward = self._calculate_reward()
        elif self.open_branches < 0:
            # Should not happen if we mask actions, but for safety
            term = True
            reward = -100.0 # Syntax error penalty
        elif len(self.current_sequence) >= self.max_length:
            trunc = True
            reward = -10.0 # Incomplete penalty
            
        return self._get_obs(), reward, term, trunc, {}

    def _get_obs(self):
        # Convert sequence to IDs and pad
        seq_ids = [TOKEN_TO_ID[t] for t in self.current_sequence]
        padded_seq = np.zeros(self.max_length, dtype=np.int32)
        padded_seq[:len(seq_ids)] = seq_ids
        
        return {
            "sequence": padded_seq,
            "x": self.current_problem['x'].astype(np.float32),
            "y": self.current_problem['y'].astype(np.float32)
        }

    def _calculate_reward(self):
        try:
            tree = ExpressionTree(self.current_sequence)
            if not tree.is_valid:
                return -100.0
            
            y_pred = tree.evaluate(self.current_problem['x'])
            
            # Root Mean Squared Error (RMSE)
            mse = np.mean((y_pred - self.current_problem['y'])**2)
            rmse = np.sqrt(mse)
            
            if np.isnan(rmse) or np.isinf(rmse):
                return -1000.0
                
            # Reward is negative RMSE
            # We want to maximize reward -> minimize RMSE
            # Normalize or scale? simpler is just -RMSE
            return -rmse
            
        except Exception:
            return -100.0

if __name__ == "__main__":
    env = SymbolicEnv()
    obs, _ = env.reset()
    print("Initial Observation Keys:", obs.keys())
    
    # Simulate a few steps for x + x
    # Prefix: + x x
    actions = ['+', 'x', 'x']
    tot_reward = 0
    for tok in actions:
        aid = TOKEN_TO_ID[tok]
        obs, reward, term, trunc, _ = env.step(aid)
        print(f"Action: {tok}, Reward: {reward}, Term: {term}, Branches: {env.open_branches}")
        tot_reward += reward
        if term: break
    
    print(f"Total Reward: {tot_reward}")
