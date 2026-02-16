"""
Beam Search for AlphaSymbolic.
Explores multiple formula candidates in parallel, keeping top-K at each step.
"""
import torch
import numpy as np
from core.grammar import VOCABULARY, OPERATORS, TOKEN_TO_ID, ExpressionTree, OPERATOR_STAGES
from utils.optimize_constants import optimize_constants
from utils.data_utils import normalize_batch

class BeamSearch:
    def __init__(self, model, device, beam_width=10, max_length=30, curriculum_stage=None, num_variables=1):
        self.model = model
        self.device = device
        self.beam_width = beam_width
        self.max_length = max_length
        self.num_variables = num_variables
        self.vocab_size = len(VOCABULARY)
        self.sos_id = self.vocab_size  # SOS token ID
        
        # Build token mask
        # 1. Start with everything allowed (mask = 0)
        # 2. disallow variables outside num_variables range
        mask = torch.zeros(self.vocab_size, device=device)
        
        # Determine allowed variables
        from core.grammar import VARIABLES
        if num_variables == 1:
            allowed_vars = set(['x', 'x0'])
        else:
            allowed_vars = set(VARIABLES[:num_variables])
            
        disallowed_vars = [v for v in VARIABLES if v not in allowed_vars]
        if 'x' not in allowed_vars: disallowed_vars.append('x')

        for v in disallowed_vars:
            if v in TOKEN_TO_ID:
                mask[TOKEN_TO_ID[v]] = float('-inf')

        # 3. Apply curriculum limits if set
        if curriculum_stage is not None:
            allowed_ops = set(OPERATOR_STAGES.get(curriculum_stage, list(OPERATORS.keys())))
            # Disallow operators not in the current stage
            for op in OPERATORS:
                if op not in allowed_ops:
                    mask[TOKEN_TO_ID[op]] = float('-inf')
        
        # Only set self.token_mask if there are actually restricted tokens
        if torch.any(mask != 0):
            self.token_mask = mask
        else:
            self.token_mask = None
        
    def search(self, x_values, y_values, return_partial=False):
        """
        Beam Search to find the best formula structure.
        """
        # Prepare data once
        # Normalize data for model inference
        # normalize_batch input: list of arrays. We wrap x_values/y_values in list.
        norm_x_list, norm_y_list = normalize_batch([x_values], [y_values])
        norm_x = norm_x_list[0]
        norm_y = norm_y_list[0]
        
        x_tensor = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0).to(self.device) 
        y_tensor = torch.tensor(norm_y, dtype=torch.float32).unsqueeze(0).to(self.device) 
        
        # STABILITY FIX: Match training logic (Clamp + Normalize)
        # 1. Clamp to training range
        x_tensor = torch.clamp(x_tensor, -100, 100)
        y_tensor = torch.clamp(y_tensor, -100, 100)
        
        # 2. Per-sample normalization to [-1, 1]
        y_min = y_tensor.min(dim=1, keepdim=True)[0]
        y_max = y_tensor.max(dim=1, keepdim=True)[0]
        y_range = (y_max - y_min).clamp(min=1e-6)
        y_tensor = 2 * (y_tensor - y_min) / y_range - 1
        
        # Each element in beams is just the sequence of tokens (list of strings)
        # We track scores and open branches in parallel lists or a list of dicts
        beams = [{'seq': [], 'log_prob': 0.0, 'open': 1}]
        
        completed = []
        
        for step in range(self.max_length):
            if not beams:
                break
                
            # Filter valid beams just in case
            active_beams = [b for b in beams if b['open'] > 0]
            if not active_beams:
                break
                
            # Prepare batch for model
            # Batch size = number of active beams
            batch_size = len(active_beams)
            
            # Expand X and Y to match batch size [batch, points, vars]
            # Use expand with *shape to handle arbitrary dimensions (1D or Multi-Var)
            # x_tensor shape is [1, points, vars] or [1, points]
            # We want [batch_size, points, vars]
            
            # Use repeat or expand. Expand is strictly view, safer:
            x_batch = x_tensor.expand(batch_size, *x_tensor.shape[1:])
            y_batch = y_tensor.expand(batch_size, *y_tensor.shape[1:])
            
            # Prepare input sequences [batch, current_seq_len]
            # Must prepend SOS token
            seqs = [[self.sos_id] + [TOKEN_TO_ID[t] for t in b['seq']] for b in active_beams]
            input_tensor = torch.tensor(seqs, dtype=torch.long).to(self.device)
            
            # Single model call for all beams
            with torch.no_grad():
                logits, _ = self.model(x_batch, y_batch, input_tensor)
            
            # Logits shape: [batch, seq_len, vocab_size]
            # We want the last token's probabilities
            last_token_logits = logits[:, -1, :self.vocab_size]
            
            # Apply curriculum mask if set
            if self.token_mask is not None:
                last_token_logits = last_token_logits + self.token_mask


            
            log_probs = torch.log_softmax(last_token_logits, dim=-1) # [batch, vocab]
            
            # --- Repetition Penalty (Simple) ---
            # If the same token was generated recently, penalize it slightly.
            # This prevents 10 ////////// loops.
            penalty_factor = 2.0  # Reduce log_prob (which is negative) by absolute amount or multiplier?
            # Log probs are negative (e.g. -0.1). Making them MORE negative penalizes.
            # If we multiply by 1.2, -0.1 becomes -0.12 (lower probability).
            
            for i, beam in enumerate(active_beams):
                if beam['seq']:
                     # Get last token ID
                    last_token = beam['seq'][-1]
                    if last_token in TOKEN_TO_ID:
                        last_id = TOKEN_TO_ID[last_token]
                        # Penalize current step logits for this token
                        # If log_prob is close to 0 (high prob), e.g. -0.01 -> -0.012
                        # If log_prob is -10 (low prob), -> -12
                        # Check bounds to avoid NaN if -inf
                        if log_probs[i, last_id] > -1e9:
                             log_probs[i, last_id] *= 1.5 
            # -----------------------------------
            
            # We need to find the top-K candidates ACROSS current beams?
            # Standard beam search: expand all, then prune to K
            
            all_candidates = []
            
            # Get top-K for EACH beam to avoid explosion (e.g. top 2*width)
            k_per_beam = min(self.beam_width, self.vocab_size)
            beam_topk_scores, beam_topk_indices = torch.topk(log_probs, k_per_beam, dim=-1)
            
            # Move to CPU for processing logic
            beam_topk_scores = beam_topk_scores.cpu().numpy()
            beam_topk_indices = beam_topk_indices.cpu().numpy()
            
            for i, beam in enumerate(active_beams):
                for score, idx in zip(beam_topk_scores[i], beam_topk_indices[i]):
                    token = VOCABULARY[idx]
                    new_seq = beam['seq'] + [token]
                    
                    # Calculate new open branches
                    if token in OPERATORS:
                        new_open = beam['open'] + OPERATORS[token] - 1
                    else:
                        new_open = beam['open'] - 1
                    
                    if new_open < 0:
                        continue
                        
                    all_candidates.append({
                        'seq': new_seq,
                        'log_prob': beam['log_prob'] + score,
                        'open': new_open
                    })
            
            # Global prune: keep top beam_width
            all_candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            beams = all_candidates[:self.beam_width]
            
            # Check for completions
            still_active = []
            for b in beams:
                if b['open'] == 0:
                    completed.append(b)
                else:
                    still_active.append(b)
            
            beams = still_active
            # If we filled up on completions, we might still want to explore? 
            # Usually we keep exploring until all beams are done or max length
            if len(completed) >= self.beam_width:
                 # Optional: early exit if we found enough good candidates
                 pass

        # Evaluate results
        scored_results = []
        for beam in completed:
            tree = ExpressionTree(beam['seq'])
            if tree.is_valid:
                constants, rmse = optimize_constants(tree, x_values, y_values)
                scored_results.append({
                    'tokens': beam['seq'],
                    'log_prob': beam['log_prob'],
                    'rmse': rmse,
                    'constants': constants,
                    'formula': tree.get_infix()
                })
        
        scored_results.sort(key=lambda x: x['rmse'])
        
        # If no results and return_partial is requested, return the best incomplete beam
        if not scored_results and return_partial and beams:
            # Take the beam with highest probability
            best_beam = beams[0] 
            # Construct a partial result
            # We can't optimize constants or get a valid infix easily, but we can show tokens
            scored_results.append({
                'tokens': best_beam['seq'],
                'log_prob': best_beam['log_prob'],
                'rmse': float('inf'),
                'constants': {},
                'formula': "Partial: " + " ".join(best_beam['seq']) + "..."
            })
            
        return scored_results


def beam_solve(target_x, target_y, model, device, beam_width=20, max_length=25, num_variables=1):
    """
    Solve symbolic regression using beam search.
    """
    searcher = BeamSearch(model, device, beam_width=beam_width, max_length=max_length, num_variables=num_variables)
    results = searcher.search(target_x, target_y)
    
    if not results:
        return None
        
    return results  # Return all results for Pareto analysis


if __name__ == "__main__":
    from core.model import AlphaSymbolicModel
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = len(VOCABULARY)
    
    model = AlphaSymbolicModel(vocab_size=VOCAB_SIZE + 1, d_model=64).to(DEVICE)
    try:
        model.load_state_dict(torch.load("alpha_symbolic_model.pth", map_location=DEVICE, weights_only=True))
    except:
        print("Model not found, using random weights")
    model.eval()
    
    # Test
    x_test = np.linspace(-5, 5, 20).astype(np.float64)
    y_test = 2 * x_test + 3
    
    print("Running Beam Search...")
    results = beam_solve(x_test, y_test, model, DEVICE, beam_width=10)
    
    print(f"\nFound {len(results)} valid formulas:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. {r['formula']} (RMSE: {r['rmse']:.4f})")
