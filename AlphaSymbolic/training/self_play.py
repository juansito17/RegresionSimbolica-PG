"""
Self-Play AlphaZero Loop for AlphaSymbolic.
The model improves by learning from its own search results.

Process:
1. Generate problems (synthetic or from memory)
2. Use MCTS to find best formulas
3. Store successful (state, action, value) tuples with priority
4. Train network on this experience using weighted sampling
5. Repeat
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from collections import deque
import random
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import AlphaSymbolicModel
from core.grammar import VOCABULARY, TOKEN_TO_ID
from data.synthetic_data import DataGenerator
from search.mcts import MCTS
from data.pattern_memory import PatternMemory
from core.loss import QuantileLoss


class ReplayBuffer:
    """Experience replay buffer for storing (state, policy, value) tuples with priority."""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, x_data, y_data, tokens, policy, value):
        """
        Add an experience tuple (state, policy, value).
        """
        self.buffer.append({
            'x': x_data,
            'y': y_data, 
            'tokens': tokens,
            'policy': policy,
            'value': value
        })
        # Priority: could be based on error, but for now uniform/FIFO
        # We'll use value magnitude + small noise for diversity if needed
        # For now, just append 1.0
        self.priorities.append(1.0)
    
    def sample(self, batch_size):
        """Sample a batch."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Uniform sampling for now (simplify)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class AlphaZeroLoop:
    def __init__(self, model_path="alpha_symbolic_model.pth", fresh_start=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = len(VOCABULARY)
        self.model_path = model_path

        # Handle fresh start
        if fresh_start and os.path.exists(self.model_path):
            os.remove(self.model_path)
            print("Previous model deleted. Starting fresh.")
        
        # Model
        self.model = AlphaSymbolicModel(
            vocab_size=self.vocab_size + 1,
            d_model=128,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3
        ).to(self.device)
        
        self.model_path = model_path
        self.load_model()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Replay buffer
        self.replay = ReplayBuffer(capacity=100000)
        
        # Data generator for new problems
        self.data_gen = DataGenerator(max_depth=5)
        
        # Pattern memory
        self.memory = PatternMemory()
        
        # Search (MCTS)
        # TPSR params: complexity_lambda=0.1
        self.searcher = MCTS(self.model, self.device, max_simulations=50, max_depth=25, complexity_lambda=0.1)
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'best_rmse': float('inf'),
            'avg_rmse': deque(maxlen=100)
        }
    
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                print(f"Loaded model from {self.model_path}")
            except:
                print("Could not load model, using fresh weights")
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
    
    def self_play_episode(self, num_problems=10):
        """
        Generate problems, solve them with MCTS, store experiences (AlphaZero style).
        """
        self.model.eval()
        
        experiences = []
        
        # Generate problems
        problems = self.data_gen.generate_batch(num_problems)
        
        for prob in problems:
            x_data = prob['x'].astype(np.float64)
            y_data = prob['y'].astype(np.float64)
            
            # Search for solution via MCTS
            result = self.searcher.search(x_data, y_data)
            
            # Extract training examples from the tree (Polcy, Value)
            # result['root'] is now available
            if 'root' in result:
                examples = self.searcher.get_training_examples(result['root'])
                for (tokens, policy, value) in examples:
                    self.replay.add(x_data, y_data, tokens, policy, value)
            
            if result['tokens']:
                experiences.append(result['rmse'])
                
                # Update pattern memory
                if result['formula'] and result['rmse'] < 1.0: # Only good ones
                     self.memory.record(result['tokens'], result['rmse'], result['formula'])
                
                # Track statistics
                if result['rmse'] < self.stats['best_rmse']:
                    self.stats['best_rmse'] = result['rmse']
                self.stats['avg_rmse'].append(result['rmse'])
        
        return experiences
    
    def train_step(self, batch_size=32):
        """
        Train on experiences from replay buffer.
        """
        if len(self.replay) < batch_size:
            return None
        
        self.model.train()
        
        batch = self.replay.sample(batch_size)
        
        # Prepare batch
        SOS_ID = self.vocab_size
        
        x_list = [exp['x'] for exp in batch]
        y_list = [exp['y'] for exp in batch]
        token_lists = [exp['tokens'] for exp in batch]
        policy_targets = [exp['policy'] for exp in batch]
        value_targets_list = [exp['value'] for exp in batch]
        
        # Pad sequences
        max_len = max(len(t) for t in token_lists)
        
        decoder_input = torch.full((len(batch), max_len + 1), SOS_ID, dtype=torch.long)
        # Policy target is distribution [batch, vocab]
        # But wait, MCTS returns policy for the NEXT token from current state.
        # Our model predicts sequence. 
        # Standard AZ: (State) -> (Policy, Value).
        # Wrapper: (Input X, Y, Partial Formula) -> (Next Token Dist, Value).
        # So we align 'decoder_input' = tokens. 
        # Target for Policy head at the LAST step is 'policy_targets'.
        
        # The 'tokens' in replay buffer IS the state (partial sequence).
        # So decoder_input should be exactly that.
        
        for i, tokens in enumerate(token_lists):
            ids = [TOKEN_TO_ID[t] for t in tokens] # Assuming tokens are strings? 
            # Wait, MCTSNode.tokens are IDs or Strings?
            # Grammar says VOCABULARY list. MCTS seems to store tokens as strings or IDs?
            # MCTS code uses TOKEN_TO_ID when calling model. So MCTSNode.tokens likely Strings?
            # Let's verify MCTSNode usage.
            # MCTSNode init: tokens=[]. expand: names from VOCABULARY.
            # So they are strings.
            
            l = len(ids)
            decoder_input[i, 1:l+1] = torch.tensor(ids, dtype=torch.long)
            
        # Targets
        # Policy: [batch, vocab_size]
        policy_target_tensor = torch.tensor(np.array(policy_targets), dtype=torch.float32).to(self.device)
        value_target_tensor = torch.tensor(np.array(value_targets_list), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # To device
        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(self.device)
        decoder_input = decoder_input.to(self.device)
        
        # Forward
        logits, value_pred = self.model(x_tensor, y_tensor, decoder_input)
        
        # Losses
        # Policy Loss: KLDiv between predicted distribution (logits of last token) and target distribution
        # logits: [batch, seq_len, vocab]
        # We only care about the LAST token prediction because that matches the MCTS state
        # The 'decoder_input' has length L+1 (SOS + tokens).
        # The prediction for the 'next' token is at the last position.
        
        # Gather last step logits
        # We need to pick the logits corresponding to the end of each sequence?
        # Since we padded, we must be careful.
        # 'decoder_input' is padded with SOS? No, standard padding?
        # I initialized with SOS_ID.
        # The effective length for user i is len(tokens)+1.
        # So we take logits[i, len(tokens), :] ??
        # Or did I pad with something else?
        
        last_logits = []
        for i, tokens in enumerate(token_lists):
            idx = len(tokens) # index of last real token (SOS is at 0)
            # sequences: SOS, T1, T2 ...
            # Input: SOS, T1
            # Output at pos 1: Pred for T2.
            # State was [T1]. 
            # Correct.
            last_logits.append(logits[i, idx, :self.vocab_size])
            
        last_logits = torch.stack(last_logits)
        
        # Log Softmax for KLDiv
        log_probs = torch.log_softmax(last_logits, dim=1)
        
        # KLDivLoss(input, target) expects log_probs as input
        policy_loss = nn.KLDivLoss(reduction='batchmean')(log_probs, policy_target_tensor)
        
        # Value Loss (Quantile Regression)
        # value_pred: [batch, 3], value_target_tensor: [batch, 1]
        value_loss = QuantileLoss()(value_pred, value_target_tensor)
        
        total_loss = policy_loss + value_loss # Weighting? 1.0 each for now
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item()
        }
    
    def run(self, iterations=100, problems_per_iter=20, train_steps_per_iter=10, 
            save_interval=10, verbose=True):
        """
        Main AlphaZero loop.
        """
        if verbose:
            print("="*60)
            print("AlphaZero Self-Play Loop (MCTS Enhanced)")
            print("="*60)
            print(f"Device: {self.device}")
            print(f"Iterations: {iterations}")
            print(f"Problems per iteration: {problems_per_iter}")
        
        start_time = time.time()
        
        for i in range(iterations):
            self.stats['iterations'] = i + 1
            
            # Self-play phase
            rmses = self.self_play_episode(problems_per_iter)
            
            # Training phase
            losses = []
            for _ in range(train_steps_per_iter):
                loss = self.train_step()
                if loss:
                    losses.append(loss)
            
            # Logging
            if verbose and (i + 1) % 5 == 0:
                avg_rmse = np.mean(list(self.stats['avg_rmse'])) if self.stats['avg_rmse'] else 0
                avg_loss = np.mean([l['total'] for l in losses]) if losses else 0
                elapsed = time.time() - start_time
                
                print(f"Iter {i+1:4d} | Buffer: {len(self.replay):5d} | "
                      f"Avg RMSE: {avg_rmse:.4f} | Best: {self.stats['best_rmse']:.4f} | "
                      f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
            
            # Save model
            if (i + 1) % save_interval == 0:
                self.save_model()
                self.memory.save()
                if verbose:
                    print(f"  -> Checkpoint saved")
        
        # Final save
        self.save_model()
        self.memory.save()
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nSelf-play complete! Total time: {total_time:.1f}s")
            print(f"Final buffer size: {len(self.replay)}")
            print(f"Best RMSE achieved: {self.stats['best_rmse']:.6f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=5)
    args = parser.parse_args()
    
    loop = AlphaZeroLoop()
    loop.run(
        iterations=args.iterations,
        problems_per_iter=args.problems,
        train_steps_per_iter=args.train_steps
    )
