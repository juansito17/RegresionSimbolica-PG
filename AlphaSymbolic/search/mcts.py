import math
import numpy as np
import torch
import copy
from core.grammar import VOCABULARY, TOKEN_TO_ID, OPERATORS, ExpressionTree, VARIABLES
from utils.optimize_constants import optimize_constants

class MCTSNode:
    def __init__(self, tokens, parent=None, prior=0.0):
        self.tokens = tokens
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        
        # for parallel search
        self.virtual_loss = 0.0
        self.virtual_visits = 0

    @property
    def value(self):
        count = self.visit_count + self.virtual_visits
        if count == 0:
            return 0.0
        # Combine real value and virtual loss
        # Virtual loss is SUBTRACTED to discourage visits
        return (self.value_sum - self.virtual_loss) / count

    def ucb_score(self, c_puct=1.0):
        count = self.visit_count + self.virtual_visits
        parent_count = self.parent.visit_count + self.parent.virtual_visits if self.parent else 1
        
        if self.parent is None:
            return 0.0
            
        u = c_puct * self.prior * math.sqrt(parent_count) / (1 + count)
        return self.value + u

class MCTS:
    def __init__(self, model, device, max_simulations=100, max_depth=30, c_puct=1.0, lambda_mix=0.5, batch_size=8):
        self.model = model
        self.device = device
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.lambda_mix = lambda_mix 
        self.batch_size = batch_size
        self.vocab_size = len(VOCABULARY)
        self.sos_id = self.vocab_size
        
        # Virtual loss constant usually 1-3
        self.v_loss_const = 3.0
        
    def search(self, x_values, y_values, num_simulations=None):
        """
        Run MCTS (Parallel/Batched) to find the best formula.
        """
        root = MCTSNode(tokens=[])
        
        # Initial expansion (single)
        self._expand_batch([root], x_values, y_values)
        
        best_rmse = float('inf')
        best_formula = None
        best_tokens = None
        
        limit = num_simulations if num_simulations is not None else self.max_simulations
        
        # Loop in batches
        # Ensure we do at least 1 batch
        num_batches = max(1, (limit + self.batch_size - 1) // self.batch_size)
        
        for _ in range(num_batches): 
            leaves = []
            
            # 1. Selection (find N leaves)
            for _ in range(self.batch_size):
                node = root
                depth = 0
                
                # Selection loop
                while node.is_expanded and node.children and depth < self.max_depth:
                    node = max(node.children.values(), key=lambda n: n.ucb_score(self.c_puct))
                    
                    # Apply virtual loss to discourage re-selection in same batch
                    node.virtual_loss += self.v_loss_const
                    node.virtual_visits += 1
                    depth += 1
                
                # Check if valid leaf to expand
                if depth < self.max_depth and not node.is_expanded:
                    # Avoid duplicates in batch (simple check)
                    if node not in leaves:
                        leaves.append(node)
                else:
                    pass
            
            if not leaves:
                # If no leaves found (tree fully explored or locked), standard MCTS usually continues or stops.
                # We can just break or continue backprop of terminals.
                if root.visit_count > limit: break 
                continue
                
            # 2. Batch Expansion & Evaluation
            values = self._expand_batch(leaves, x_values, y_values)
            
            # 3. Backpropagation
            for node, val in zip(leaves, values):
                # Check for best solution found
                if self._is_complete_tree(node.tokens):
                    # For completed formulas, we calculate REAL RMSE
                    real_rmse = self._evaluate_formula(node.tokens, x_values, y_values)
                    
                    # Use real RMSE as value if valid
                    final_val = 1.0 / (1.0 + real_rmse) # [0, 1] range
                    
                    if real_rmse < best_rmse:
                        best_rmse = real_rmse
                        best_tokens = node.tokens
                        best_formula = ExpressionTree(node.tokens).get_infix()
                else:
                    final_val = val
                
                # Backpropagate
                curr = node
                while curr is not None:
                    curr.visit_count += 1
                    curr.value_sum += final_val
                    
                    # Revert virtual loss for parent and above
                    # Since we added to PARENT's child (which is curr), 
                    # and we traverse Up...
                    # Wait, logic: We selected CHILD. Virtual loss was added TO CHILD (curr).
                    # So we must remove it from curr.
                    if curr.virtual_visits > 0:
                        curr.virtual_loss -= self.v_loss_const
                        curr.virtual_visits -= 1
                            
                    curr = curr.parent
        
        # After search, force cleanup of any residual virtual loss (safety)
        # (Not strictly needed if logic is perfect, but good practice in complex async MCTS)
        
        return {
            'tokens': best_tokens,
            'formula': best_formula,
            'rmse': best_rmse
        }

    def _expand_batch(self, nodes, x_values, y_values):
        """
        Batched expansion. Returns list of values.
        """
        if not nodes:
            return []
            
        # Prepare inputs
        x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Repeat X/Y for batch
        batch_size = len(nodes)
        x_batch = x_tensor.repeat(batch_size, 1, 1).squeeze(1) # [batch, points]
        y_batch = y_tensor.repeat(batch_size, 1, 1).squeeze(1) # [batch, points]
        
        # Prepare sequences
        # Find max len
        max_len = 0
        seqs = []
        for n in nodes:
            s = [self.sos_id] + [TOKEN_TO_ID[t] for t in n.tokens]
            seqs.append(s)
            max_len = max(max_len, len(s))
            
        # Pad and stack
        input_tensor = torch.full((batch_size, max_len), self.sos_id, dtype=torch.long).to(self.device)
        for i, s in enumerate(seqs):
            input_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            
        # Inference
        with torch.no_grad():
            logits, value_preds = self.model(x_batch, y_batch, input_tensor)
            
        # Process results
        values = []
        
        # To CPU numpy for probability processing
        probs_batch = torch.softmax(logits[:, -1, :self.vocab_size], dim=1).cpu().numpy()
        value_preds = value_preds.cpu().numpy().flatten()
        
        for i, node in enumerate(nodes):
            # 1. Store Value
            val = float(np.clip(value_preds[i], 0.0, 1.0))
            values.append(val)
            
            # 2. Expand children
            node_probs = probs_batch[i]
            valid_next = self._get_valid_next_tokens(node.tokens)
            
            for idx in valid_next:
                token = VOCABULARY[idx]
                prior = node_probs[idx]
                child = MCTSNode(tokens=node.tokens + [token], parent=node, prior=prior)
                node.children[token] = child
            
            node.is_expanded = True
            
        return values

    def _get_valid_next_tokens(self, tokens):
        """Simple grammar check."""
        open_slots = 1
        for t in tokens:
            if t in OPERATORS:
                open_slots += OPERATORS[t] - 1
            else:
                open_slots -= 1
        
        if open_slots <= 0:
            return []
        return list(range(self.vocab_size))

    def _is_complete_tree(self, tokens):
        if not tokens: return False
        try:
            tree = ExpressionTree(tokens)
            # Basic validation
            if len(tokens) > self.max_depth * 2: return False
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
