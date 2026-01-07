"""
Diagnostic test for Token Masking in BeamSearch.
Checks if the mask correctly allows/disallows tokens per curriculum stage.
"""
import torch
from core.grammar import VOCABULARY, TOKEN_TO_ID, OPERATORS, OPERATOR_STAGES

def test_mask_generation():
    """Test that the mask is generated correctly for each stage."""
    device = torch.device("cpu")
    vocab_size = len(VOCABULARY)
    
    print("=== VOCABULARY ===")
    print(f"Total tokens: {vocab_size}")
    print(f"Tokens: {VOCABULARY}")
    print(f"\nTOKEN_TO_ID: {TOKEN_TO_ID}")
    print(f"\nOPERATOR_STAGES: {OPERATOR_STAGES}")
    
    print("\n=== MASK TEST PER STAGE ===")
    
    for stage in range(5):
        allowed_ops = OPERATOR_STAGES.get(stage, list(OPERATORS.keys()))
        allowed_tokens = set(['x', 'C', '0', '1', '2', '3', '5', '10', 'pi', 'e'])
        allowed_tokens.update(allowed_ops)
        
        # Create mask: 0 for allowed, -inf for disallowed
        mask = torch.full((vocab_size,), float('-inf'), device=device)
        for token in allowed_tokens:
            if token in TOKEN_TO_ID:
                mask[TOKEN_TO_ID[token]] = 0.0
        
        # Count allowed
        allowed_count = (mask == 0.0).sum().item()
        allowed_list = [VOCABULARY[i] for i in range(vocab_size) if mask[i] == 0.0]
        
        print(f"\nStage {stage}:")
        print(f"  Allowed ops: {allowed_ops}")
        print(f"  Total allowed tokens: {allowed_count}")
        print(f"  Allowed list: {allowed_list}")
        
        # Sanity check: Are basic operators allowed in stage 0?
        if stage == 0:
            for op in ['+', '-', '*', '/']:
                idx = TOKEN_TO_ID.get(op, -1)
                if idx >= 0:
                    is_allowed = mask[idx] == 0.0
                    print(f"  Operator '{op}' (id={idx}) allowed: {is_allowed}")

def test_beam_search_mask():
    """Test that BeamSearch applies the mask correctly."""
    from search.beam_search import BeamSearch
    
    device = torch.device("cpu")
    
    # Create a dummy model-like object for testing
    class DummyModel:
        def __call__(self, x, y, dec):
            # Return uniform logits
            batch = dec.shape[0]
            seq_len = dec.shape[1]
            vocab_size = len(VOCABULARY)
            logits = torch.zeros(batch, seq_len, vocab_size + 1)
            value = torch.zeros(batch, 1)
            return logits, value
        
        def eval(self):
            pass
            
        def to(self, device):
            return self
    
    dummy_model = DummyModel()
    
    print("\n=== BEAM SEARCH MASK TEST ===")
    
    for stage in [0, 1, 2]:
        bs = BeamSearch(dummy_model, device, beam_width=5, max_length=5, curriculum_stage=stage)
        
        print(f"\nStage {stage}:")
        print(f"  Token mask is set: {bs.token_mask is not None}")
        
        if bs.token_mask is not None:
            allowed_count = (bs.token_mask == 0.0).sum().item()
            print(f"  Allowed token count: {allowed_count}")
            
            # Show which tokens are allowed
            allowed = [VOCABULARY[i] for i in range(len(VOCABULARY)) if bs.token_mask[i] == 0.0]
            print(f"  Allowed tokens: {allowed}")

if __name__ == "__main__":
    test_mask_generation()
    test_beam_search_mask()
    print("\n=== TESTS COMPLETE ===")
