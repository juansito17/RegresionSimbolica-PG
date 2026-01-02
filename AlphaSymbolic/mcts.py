import math
import numpy as np
import torch
from grammar import VOCABULARY, OPERATORS, TOKEN_TO_ID, ExpressionTree

class MCTSNode:
    def __init__(self, sequence, parent=None, prior=0.0):
        self.sequence = sequence
        self.parent = parent
        self.children = {} # map action_id -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior # P(s, a) from policy head
        self.is_expanded = False
        
        # Calculate validity/terminal status
        self.open_branches = self._calculate_open_branches(sequence)
        self.is_terminal = (self.open_branches == 0) and (len(sequence) > 0)
        
    def _calculate_open_branches(self, seq):
        if not seq: return 1 # Expect root
        open_b = 1
        for token in seq:
            if token in OPERATORS:
                open_b += (OPERATORS[token] - 1)
            else:
                open_b -= 1
        return open_b

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, total_visits, c_puct=1.0):
        # Q + U
        q_value = self.value()
        # U = c * P * sqrt(N_parent) / (1 + N_child)
        u_value = c_puct * self.prior * math.sqrt(total_visits) / (1 + self.visits)
        return q_value + u_value

class MCTS:
    def __init__(self, model, device, max_len=50):
        self.model = model
        self.device = device
        self.max_len = max_len
        self.cpuct = 1.41
        
    def search(self, x_values, y_values, num_simulations=50):
        """
        Runs MCTS to find the best formula for (X, Y).
        Returns the best found formula string.
        """
        # Root state: empty specific sequence? Or do we assume we need to start?
        # Our model expects at least a restart token implicitly or we manage it.
        # Let's start with empty sequence [] representing state before first move.
        # Ideally, we loop MCTS step-by-step to build the formula, 
        # BUT standard AlphaZero MCTS builds a whole tree for one "move", then picks move, then discards/reuses tree.
        # For formula generation, the "Game" is short (10-50 moves).
        # We can just run one big MCTS from root if we want, or step-by-step.
        # Step-by-step is more robust.
        
        root = MCTSNode(sequence=[]) 
        
        # Prepare context tensors once
        x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        current_node = root
        
        # Build formula token by token
        for _ in range(self.max_len):
            if current_node.is_terminal:
                break
                
            # Run simulations from current state
            for _ in range(num_simulations):
                self._simulate(current_node, x_tensor, y_tensor)
            
            # Select best move (most visited)
            if not current_node.children:
                break # Should not happen if sims ran
                
            best_action = max(current_node.children.items(), key=lambda item: item[1].visits)[0]
            current_node = current_node.children[best_action]
            
        # Reconstruct formula
        return current_node.sequence

    def _simulate(self, node, x_tensor, y_tensor):
        path = []
        curr = node
        
        # 1. Selection
        while curr.is_expanded and not curr.is_terminal:
            path.append(curr)
            # Pick child with highest UCB
            total_parent_visits = curr.visits
            
            best_score = -float('inf')
            best_child = None
            
            for action, child in curr.children.items():
                score = child.ucb_score(total_parent_visits, self.cpuct)
                if score > best_score:
                    best_score = score
                    best_child = child
                    
            if best_child:
                curr = best_child
            else:
                # Should not happen if expanded
                break

        path.append(curr)
        leaf = curr
        
        # 2. Expansion & Evaluation
        value = 0.0
        
        if leaf.is_terminal:
            # Calculate true reward
            value = self._evaluate_formula(leaf.sequence, x_tensor, y_tensor)
        else:
            # Expand using NN
            # Prepare input for model
            # Sequence tokens to IDs
            seq_ids = [TOKEN_TO_ID[t] for t in leaf.sequence]
            # Add SOS (using len(VOCAB) as SOS ID from train.py logic)
            # Actually train.py used VOCAB_SIZE as SOS. Let's align.
            sos_id = len(VOCABULARY) 
            input_seq = [sos_id] + seq_ids
            
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                logits, v_pred = self.model(x_tensor, y_tensor, input_tensor)
            
            # Logits are [1, seq_len, vocab]. We want the LAST prediction.
            last_logits = logits[0, -1, :]
            # Filter invalid moves? (Optional, helps convergence)
            # e.g. if leaf.open_branches == 0 (though processed above) 
            
            probs = torch.softmax(last_logits, dim=0).cpu().numpy()
            value = v_pred.item()
            
            leaf.is_expanded = True
            
            # Create children
            # We treat all tokens as possible actions
            # Optimization: Mask invalid syntax (e.g. if open branches is high, maybe limit?)
            # For now, simplistic.
            for action_id, prob in enumerate(probs):
                if action_id >= len(VOCABULARY): continue # Skip special tokens if any
                
                token = VOCABULARY[action_id]
                new_seq = leaf.sequence + [token]
                
                # Check rudimentary validity to prune tree
                # E.g. don't allow too many branches or negative branches
                try:
                    child_node = MCTSNode(new_seq, parent=leaf, prior=prob)
                    if child_node.open_branches >= 0: # Valid prefix
                        leaf.children[action_id] = child_node
                except:
                    pass
        
        # 3. Backup
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            # Standard AlphaZero backup: value is from perspective of current player.
            # Here single player, so maximize positive reward.
            
    def _evaluate_formula(self, sequence, x_tensor, y_tensor):
        # Calculate RMSE/Reward
        try:
            tree = ExpressionTree(sequence)
            if not tree.is_valid:
                return -10.0 # Penalty
            
            x_np = x_tensor.cpu().numpy()[0]
            y_target = y_tensor.cpu().numpy()[0]
            
            y_pred = tree.evaluate(x_np)
            
            mse = np.mean((y_pred - y_target)**2)
            rmse = np.sqrt(mse)
            
            if np.isnan(rmse) or np.isinf(rmse):
                return -10.0
                
            # Reward: usually we want normalized.
            # Bound it? 1 / (1 + rmse) is good [0, 1]
            return 1.0 / (1.0 + rmse)
        except:
            return -10.0
