import math
import numpy as np
import torch
import copy
from core.grammar import VOCABULARY, TOKEN_TO_ID, OPERATORS, ExpressionTree, VARIABLES
from utils.optimize_constants import optimize_constants

class MCTSNode:
    def __init__(self, tokens, parent=None, prior=0.0):
        self.tokens = tokens # List of tokens
        self.parent = parent
        self.children = {} # {token: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        
    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.0):
        # UCB = Q(s,a) + U(s,a)
        # U(s,a) = c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
        if self.parent is None:
            return 0.0
            
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + u

class MCTS:
    def __init__(self, model, device, max_simulations=100, max_depth=30, c_puct=1.0, lambda_mix=0.5):
        self.model = model
        self.device = device
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.lambda_mix = lambda_mix  # Mix ratio: (1-λ)*Value + λ*Rollout
        self.vocab_size = len(VOCABULARY)
        self.sos_id = self.vocab_size
        
    def search(self, x_values, y_values, num_simulations=None):
        """
        Run MCTS to find the best formula.
        Returns the best node (or sequence of tokens).
        """
        # Root node
        root = MCTSNode(tokens=[]) # Empty start, will prepend SOS for model
        
        # Expand root immediately
        self._expand(root, x_values, y_values)
        
        best_rmse = float('inf')
        best_formula = None
        best_tokens = None
        
        limit = num_simulations if num_simulations is not None else self.max_simulations
        for _ in range(limit):
            node = root
            
            # 1. Selection
            depth = 0
            while node.is_expanded and node.children and depth < self.max_depth:
                # Select best child by UCB
                node = max(node.children.values(), key=lambda n: n.ucb_score(self.c_puct))
                depth += 1
            
            # 2. Expansion & Evaluation
            # If not terminal and not expanded/new
            if depth < self.max_depth:
                value = self._expand(node, x_values, y_values)
            else:
                value = 0.0 # Too deep or terminal without result
            
            # Check if this node represents a valid complete formula (terminal)
            # Or if we just expanded it, we might want to check its validity?
            # Actually, standard MCTS expands one step. 
            # We can check if `tokens` form a valid tree.
            
            # To evaluate "validity" or "score" of a partial formula is hard.
            # But the Value Head predicts the *future* score. 
            
            # Let's check if it's a valid complete tree
            if self._is_complete_tree(node.tokens):
                # Calculate real RMSE
                rmse = self._evaluate_formula(node.tokens, x_values, y_values)
                # Value for MCTS is -RMSE (or similar). 
                # Model predicts predicted_neg_rmse. 
                # We can mix Model Value with Real Value if terminal?
                value = -rmse
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_tokens = node.tokens
                    best_formula = ExpressionTree(node.tokens).get_infix()
            
            # 3. Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                
        return {
            'tokens': best_tokens,
            'formula': best_formula,
            'rmse': best_rmse
        }

    def _expand(self, node, x_values, y_values):
        """
        Expands the node using the policy head and returns the hybrid value estimate.
        V = (1-λ)*v_θ + λ*z  where z is the rollout value
        """
        if node.is_expanded:
            return node.value # Should not happen in standard flow unless re-visiting leaf
            
        # Prepare input
        # [SOS, t1, t2, ...]
        seq = [self.sos_id] + [TOKEN_TO_ID[t] for t in node.tokens]
        input_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
        x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value_pred = self.model(x_tensor, y_tensor, input_tensor)
            
        # Logits: [1, seq_len, vocab_size] -> Last token logits
        last_logits = logits[0, -1, :self.vocab_size]
        probs = torch.softmax(last_logits, dim=0).cpu().numpy()
        network_value = value_pred.item()
        
        # Fast rollout: complete the formula greedily and evaluate
        rollout_value = self._fast_rollout(node.tokens, x_values, y_values, logits)
        
        # Hybrid value: (1-λ)*network + λ*rollout
        value = (1 - self.lambda_mix) * network_value + self.lambda_mix * rollout_value
        
        # Create children
        # We can mask invalid tokens here (grammar constraints)
        valid_next_tokens = self._get_valid_next_tokens(node.tokens)
        
        for idx in valid_next_tokens:
            token = VOCABULARY[idx]
            prior = probs[idx]
            child = MCTSNode(tokens=node.tokens + [token], parent=node, prior=prior)
            node.children[token] = child
            
        node.is_expanded = True
        return value

    def _fast_rollout(self, tokens, x_values, y_values, initial_logits=None):
        """
        Fast greedy rollout: complete the formula by always picking the most likely token.
        Returns a value (negative RMSE, normalized).
        """
        current_tokens = tokens.copy()
        max_rollout_steps = 20
        
        for _ in range(max_rollout_steps):
            # Check if complete
            if self._is_complete_tree(current_tokens):
                rmse = self._evaluate_formula(current_tokens, x_values, y_values)
                # Normalize RMSE to value-like range [-1, 1]
                # Lower RMSE = higher value
                return max(-1.0, -rmse / 10.0)  # Simple normalization
            
            # Get next token (greedy)
            seq = [self.sos_id] + [TOKEN_TO_ID[t] for t in current_tokens]
            input_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
            x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0).to(self.device)
            y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits, _ = self.model(x_tensor, y_tensor, input_tensor)
            
            # Greedy selection
            last_logits = logits[0, -1, :self.vocab_size]
            
            # Grammar mask
            valid_tokens = self._get_valid_next_tokens(current_tokens)
            if not valid_tokens:
                break
            
            # Mask invalid tokens
            mask = torch.full((self.vocab_size,), float('-inf'))
            for idx in valid_tokens:
                mask[idx] = 0
            masked_logits = last_logits + mask.to(self.device)
            
            next_token_id = torch.argmax(masked_logits).item()
            current_tokens.append(VOCABULARY[next_token_id])
        
        # Did not reach valid formula
        return -1.0

    def _get_valid_next_tokens(self, tokens):
        """
        Simple grammar check.
        If we want to enforce structure:
        - Open branches count (arity)
        """
        # Determine open branches
        open_slots = 1 # Root
        for t in tokens:
            if t in OPERATORS:
                open_slots += OPERATORS[t] - 1
            else:
                open_slots -= 1
        
        if open_slots <= 0:
            return [] # No more tokens allowed (fully formed)
            
        # Identify valid indices
        # If open_slots > 0, we can basically place anything? 
        # Except if we want to limit depth or balance. 
        # For now, allow all.
        return list(range(self.vocab_size))

    def _is_complete_tree(self, tokens):
        if not tokens: return False
        try:
            tree = ExpressionTree(tokens)
            return tree.is_valid
        except:
            return False

    def _evaluate_formula(self, tokens, x, y):
        try:
            tree = ExpressionTree(tokens)
            _, rmse = optimize_constants(tree, x, y)
            return rmse
        except:
            return 1e9
