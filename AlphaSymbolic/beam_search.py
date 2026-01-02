"""
Beam Search for AlphaSymbolic.
Explores multiple formula candidates in parallel, keeping top-K at each step.
"""
import torch
import numpy as np
from grammar import VOCABULARY, OPERATORS, TOKEN_TO_ID, ExpressionTree
from optimize_constants import optimize_constants

class BeamSearch:
    def __init__(self, model, device, beam_width=10, max_length=30):
        self.model = model
        self.device = device
        self.beam_width = beam_width
        self.max_length = max_length
        self.vocab_size = len(VOCABULARY)
        self.sos_id = self.vocab_size  # SOS token ID
        
    def search(self, x_values, y_values):
        """
        Beam Search to find the best formula structure.
        Returns list of (sequence, score) tuples sorted by score.
        """
        x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Each beam: (sequence, log_prob, open_branches)
        # open_branches: number of arguments still needed to complete the tree
        beams = [
            {'seq': [], 'log_prob': 0.0, 'open': 1}
        ]
        
        completed = []
        
        for step in range(self.max_length):
            if not beams:
                break
                
            all_candidates = []
            
            for beam in beams:
                if beam['open'] == 0:
                    # This beam is complete
                    completed.append(beam)
                    continue
                    
                # Get model predictions
                seq_ids = [self.sos_id] + [TOKEN_TO_ID[t] for t in beam['seq']]
                input_tensor = torch.tensor([seq_ids], dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    logits, _ = self.model(x_tensor, y_tensor, input_tensor)
                
                # Get probabilities for next token
                log_probs = torch.log_softmax(logits[0, -1, :self.vocab_size], dim=0)
                
                # Get top-K candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, min(self.beam_width * 2, self.vocab_size))
                
                for log_p, idx in zip(topk_log_probs.cpu().numpy(), topk_indices.cpu().numpy()):
                    token = VOCABULARY[idx]
                    new_seq = beam['seq'] + [token]
                    
                    # Calculate new open branches
                    if token in OPERATORS:
                        new_open = beam['open'] + OPERATORS[token] - 1
                    else:
                        new_open = beam['open'] - 1
                    
                    # Prune invalid states
                    if new_open < 0:
                        continue
                        
                    all_candidates.append({
                        'seq': new_seq,
                        'log_prob': beam['log_prob'] + log_p,
                        'open': new_open
                    })
            
            # Keep top beam_width candidates
            all_candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            beams = all_candidates[:self.beam_width]
        
        # Add any remaining incomplete beams that are close to complete
        for beam in beams:
            if beam['open'] == 0:
                completed.append(beam)
        
        # Score completed sequences by evaluating them
        scored_results = []
        for beam in completed:
            tree = ExpressionTree(beam['seq'])
            if tree.is_valid:
                # Optimize constants and get RMSE
                constants, rmse = optimize_constants(tree, x_values, y_values)
                scored_results.append({
                    'tokens': beam['seq'],
                    'log_prob': beam['log_prob'],
                    'rmse': rmse,
                    'constants': constants,
                    'formula': tree.get_infix()
                })
        
        # Sort by RMSE (lower is better)
        scored_results.sort(key=lambda x: x['rmse'])
        
        return scored_results


def beam_solve(target_x, target_y, model, device, beam_width=20, max_length=25):
    """
    Solve symbolic regression using beam search.
    """
    searcher = BeamSearch(model, device, beam_width=beam_width, max_length=max_length)
    results = searcher.search(target_x, target_y)
    
    if not results:
        return None
        
    return results  # Return all results for Pareto analysis


if __name__ == "__main__":
    from model import AlphaSymbolicModel
    
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
