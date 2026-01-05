import math
import numpy as np
import torch
import copy
from core.grammar import VOCABULARY, TOKEN_TO_ID, OPERATORS, ExpressionTree, VARIABLES, OPERATOR_STAGES
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

    @property
    def complexity(self):
        """Estimate complexity (length of formula)."""
        return len(self.tokens)

class MCTS:
    def __init__(self, model, device, grammar=None, c_puct=1.0, n_simulations=100, max_simulations=None, max_depth=50, complexity_lambda=0.1, max_len=200, batch_size=8, curriculum_stage=None):
        self.model = model
        self.device = device
        self.grammar = grammar
        self.c_puct = c_puct
        
        # Handle backwards compatibility for max_simulations
        if max_simulations is not None:
            self.n_simulations = max_simulations
        else:
            self.n_simulations = n_simulations
            
        self.max_depth = max_depth
        self.complexity_lambda = complexity_lambda
        self.max_len = max_len
        self.min_value = -float('inf')
        self.max_value = float('inf')
        self.vocab_size = len(VOCABULARY)
        self.sos_id = self.vocab_size
        self.batch_size = batch_size
        
        # Curriculum stage for operator filtering
        self.curriculum_stage = curriculum_stage
        self._build_allowed_tokens()
        
        # Pareto Front: List of {'tokens':, 'rmse':, 'complexity':, 'formula':}
        self.pareto_front = []
        
        # Virtual loss constant usually 1-3
        self.v_loss_const = 3.0
    
    def _build_allowed_tokens(self):
        """Build set of allowed token indices based on curriculum stage."""
        # Terminals are always allowed
        allowed = {'x', 'C', '0', '1', '2', '3', '5', '10', 'pi', 'e'}
        
        # Add operators based on curriculum stage
        if self.curriculum_stage is not None and self.curriculum_stage in OPERATOR_STAGES:
            allowed.update(OPERATOR_STAGES[self.curriculum_stage])
        else:
            # No stage = all operators allowed
            allowed.update(OPERATORS.keys())
        
        # Convert to indices
        self.allowed_token_indices = set()
        for token in allowed:
            if token in TOKEN_TO_ID:
                self.allowed_token_indices.add(TOKEN_TO_ID[token])
        
    def search(self, x_values, y_values, num_simulations=None):
        """
        Run MCTS (Parallel/Batched) to find the best formula.
        """
        self.pareto_front = [] # Reset Pareto Front for new search
        root = MCTSNode(tokens=[])
        
        # Initial expansion (single)
        self._expand_batch([root], x_values, y_values)
        
        best_rmse = float('inf')
        best_formula = None
        best_tokens = None
        
        limit = num_simulations if num_simulations is not None else self.n_simulations
        
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
                    try:
                        # Evaluar
                        # Importar aquí para evitar circular imports si es necesario
                        from utils.optimize_constants import optimize_constants
                        
                        # 1. Optimizar constants (Crucial para Accuracy)
                        # Esto es "Phase 1" de TPSR (constantes en las hojas)
                        # Por simplicidad en esta iteración, asumimos que 'evaluate_formula' ya hace algo o usamos el string directo.
                        # Idealmente llamaríamos a BFGS aquí.
                        
                        # Use existing _evaluate_formula to get RMSE and optimized constants
                        tree = ExpressionTree(node.tokens)
                        optimized_constants, real_rmse = optimize_constants(tree, x_values, y_values)
                        
                        # Get y_pred using the optimized constants
                        y_pred = tree.evaluate(x_values, constants=optimized_constants)
                        
                        # Check dimensions
                        if y_pred.shape != y_values.shape:
                            # If shapes don't match, it's an invalid evaluation
                            final_val = 0.0
                        else:
                            # 2. Calcular Reward TPSR (Hybrid Accuracy + Complexity)
                            # R = 1 / (1 + NMSE) + lambda * exp(-len/L)
                            
                            mse = np.mean((y_pred - y_values)**2)
                            var_y = np.var(y_values)
                            if var_y < 1e-9: var_y = 1.0 # Avoid division by zero
                            
                            nmse = mse / var_y
                            
                            # Evitar NMSE gigantes
                            if np.isnan(nmse) or np.isinf(nmse):
                                nmse = 1e9
                            
                            r_acc = 1.0 / (1.0 + nmse)
                            
                            # Penalización por complejidad
                            token_len = len(node.tokens)
                            L = self.max_len # Max length del modelo
                            
                            r_cplx = self.complexity_lambda * np.exp(-token_len / L)
                            
                            # Suma y Normalización (para mantener rango 0-1)
                            # El máximo teórico es (1.0 + lambda). Dividimos por eso.
                            raw_reward = r_acc + r_cplx
                            final_val = raw_reward / (1.0 + self.complexity_lambda)

                        # Update best formula based on RMSE (for reporting, not for MCTS value)
                        if real_rmse < best_rmse:
                            best_rmse = real_rmse
                            best_tokens = node.tokens
                            best_formula = ExpressionTree(node.tokens).get_infix()
                        
                        # Update Pareto Front
                        # Complexity = len(tokens) (or could use count_constants + nodes)
                        complexity = len(node.tokens)
                        self._update_pareto_front(node.tokens, real_rmse, complexity, ExpressionTree(node.tokens).get_infix())

                    except Exception as e:
                        # print(f"Error evaluating formula: {e}")
                        final_val = 0.0 # Invalid formula gets 0 reward
                else:
                    final_val = val
                
                # The following lines were part of the user's instruction but contained syntax errors and undefined variables.
                # They are commented out to maintain a syntactically correct and functional document.
                # If these lines were intended to be added, please provide a complete and correct snippet.
                #
                # # Construir vector de probabilidades
                # probs = np.zeros(self.vocab_size, dtype=np.float32)
                # for token_id, count in counts.items():
                #     probs[token_id] = count / total_visits_count += 1
                
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
            'rmse': best_rmse,
            'root': root,
            'pareto_front': self.pareto_front
        }

    def _update_pareto_front(self, tokens, rmse, complexity, formula_str):
        """
        Update the Pareto Front with a new solution.
        Keep solutions that are not dominated by any other solution.
        Solution A dominates B if:
        A.rmse <= B.rmse AND A.complexity <= B.complexity AND (A.rmse < B.rmse OR A.complexity < B.complexity)
        """
        # Create candidate
        candidate = {'tokens': tokens, 'rmse': rmse, 'complexity': complexity, 'formula': formula_str}
        
        # Check if dominated by existing
        is_dominated = False
        to_remove = []
        
        for existing in self.pareto_front:
            # Check if existing dominates candidate
            if (existing['rmse'] <= candidate['rmse'] and 
                existing['complexity'] <= candidate['complexity'] and 
                (existing['rmse'] < candidate['rmse'] or existing['complexity'] < candidate['complexity'])):
                is_dominated = True
                break
                
            # Check if candidate dominates existing
            if (candidate['rmse'] <= existing['rmse'] and 
                candidate['complexity'] <= existing['complexity'] and 
                (candidate['rmse'] < existing['rmse'] or candidate['complexity'] < existing['complexity'])):
                to_remove.append(existing)
        
        if not is_dominated:
            # Remove dominated existing solutions
            for item in to_remove:
                self.pareto_front.remove(item)
            
            # Add candidate
            self.pareto_front.append(candidate)
            # Sort by RMSE for easier viewing
            self.pareto_front.sort(key=lambda x: x['rmse'])

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
        value_preds = value_preds.cpu().numpy() # [batch, 3]
        
        for i, node in enumerate(nodes):
            # 1. Store Value (Median for now)
            # value_preds is [batch, 3] -> (Pessimistic, Median, Optimistic)
            # We use Median (index 1) for standard UCB.
            val_pred = value_preds[i, 1] 
            val = float(np.clip(val_pred, 0.0, 1.0))
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
        """Grammar check + curriculum filtering."""
        open_slots = 1
        for t in tokens:
            if t in OPERATORS:
                open_slots += OPERATORS[t] - 1
            else:
                open_slots -= 1
        
        if open_slots <= 0:
            return []
        
        # Filter by curriculum-allowed tokens
        return list(self.allowed_token_indices)

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

    def get_training_examples(self, root):
        """
        Extrae ejemplos de entrenamiento del árbol generado.
        Retorna: lista de (state_tokens, policy_probs, value_target)
        """
        examples = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            if node.visit_count < 1: 
                continue
            
            # Policy Target (Pi)
            # Distribución de visitas de los hijos
            counts = {}
            total_visits = 0
            has_children = False
            
            for token_id, child in node.children.items():
                # child key is token STRING or ID?
                # In _expand_batch: node.children[token] = child.
                # token = VOCABULARY[idx] (String).
                # So keys are strings.
                # But we need ID for probabilities array index.
                if token_id in TOKEN_TO_ID:
                    tid = TOKEN_TO_ID[token_id]
                    counts[tid] = child.visit_count
                    total_visits += child.visit_count
                    queue.append(child)
                    has_children = True
            
            if not has_children or total_visits == 0:
                continue
                
            # Construir vector de probabilidades
            probs = np.zeros(self.vocab_size, dtype=np.float32)
            for tid, count in counts.items():
                probs[tid] = count / total_visits
            
            # Value Target (V)
            # Usamos el Q-value (valor esperado) del nodo como target para el Value Head.
            # Q = value_sum / visit_count
            v = node.value_sum / node.visit_count
            
            # State: node.tokens (lista de ids?)
            # node.tokens is list of strings (from VOCABULARY).
            # self_play.py expects tokens as strings in ReplayBuffer.add.
            examples.append((node.tokens, probs, v))
            
        return examples
