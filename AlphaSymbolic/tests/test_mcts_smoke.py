
import sys
import os
import torch
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import AlphaSymbolicModel
from core.grammar import VOCABULARY
from search.mcts import MCTS, MCTSNode

def test_mcts_pareto():
    print("Testing MCTS Pareto Front...")
    
    # Mock Model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 128
        def forward(self, x, y, seq):
            bs = seq.size(0)
            vocab = len(VOCABULARY)
            # Logits: [bs, len, vocab]
            logits = torch.randn(bs, seq.size(1), vocab)
            # Value: [bs, 1]
            value = torch.rand(bs, 1)
            return logits, value

    model = MockModel() 
    
    # Grammar mock
    class MockGrammar:
        n_tokens = len(VOCABULARY)
        
    grammar = MockGrammar()

    mcts = MCTS(model, grammar, complexity_lambda=0.1)
    
    # Manually populate Pareto Front
    mcts._update_pareto_front(['x'], 0.5, 1, "x")
    mcts._update_pareto_front(['x', '+', '1'], 0.1, 3, "x+1")
    
    # 1. Dominated solution (High error, High complexity) -> Should not assume
    mcts._update_pareto_front(['x', '+', '1', '+', '1'], 0.6, 5, "x+1+1")
    assert len(mcts.pareto_front) == 2, "Dominated solution should be rejected"
    
    # 2. Dominating solution (Low error, Low complexity) -> Should replace
    mcts._update_pareto_front(['C'], 0.05, 1, "C")
    # "C" (RMSE 0.05, Len 1) dominates "x" (RMSE 0.5, Len 1)
    # It also dominates "x+1"? 0.05 < 0.1 AND 1 < 3. Yes.
    
    assert len(mcts.pareto_front) == 1, "Dominating solution should replace others"
    assert mcts.pareto_front[0]['formula'] == "C"
    
    print("Pareto Front logic passed!")
    
    # Test Node Properties
    node = MCTSNode(tokens=['x'])
    assert node.complexity == 1
    
    print("Node properties passed!")

if __name__ == "__main__":
    test_mcts_pareto()
