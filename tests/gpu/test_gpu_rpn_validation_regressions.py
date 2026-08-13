import os
import sys

import torch


_test_dir = os.path.dirname(os.path.abspath(__file__))
_alpha_symbolic_dir = os.path.dirname(_test_dir)
_project_root = os.path.dirname(_alpha_symbolic_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _alpha_symbolic_dir)


def test_validate_rpn_rejects_prefix_underflow_sequence():
    from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
    from warpsymbolic.gpu.operators import GPUOperators

    grammar = GPUGrammar(num_variables=1, use_globals=False)
    ops = GPUOperators(grammar, "cpu", pop_size=4, max_len=4, num_variables=1)

    plus_id = grammar.token_to_id["+"]
    x0_id = grammar.token_to_id["x0"]

    # Invalid RPN: starts with binary operator (+), causing stack underflow.
    # Deltas are [-1, +1, +1] -> final stack = 1, so old validator accepted it.
    pop = torch.tensor([[plus_id, x0_id, x0_id, PAD_ID]], dtype=grammar.dtype)

    valid = ops._validate_rpn_batch(pop)
    assert not valid.item(), "RPN validator must reject prefix-underflow expressions"


def test_validate_rpn_custom_rejects_prefix_underflow_sequence():
    from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
    from warpsymbolic.gpu.operators import GPUOperators

    grammar = GPUGrammar(num_variables=1, use_globals=False)
    ops = GPUOperators(grammar, "cpu", pop_size=4, max_len=4, num_variables=1)

    sin_id = grammar.token_to_id["sin"]
    x0_id = grammar.token_to_id["x0"]

    # Invalid RPN: unary op without prior argument at first position.
    pop = torch.tensor([[sin_id, x0_id, PAD_ID, PAD_ID]], dtype=grammar.dtype)

    valid = ops._validate_rpn_batch_custom(pop, max_len=4)
    assert not valid.item(), "Custom RPN validator must reject unary-prefix underflow"
