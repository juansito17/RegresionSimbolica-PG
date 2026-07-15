import torch

from AlphaSymbolic.core.gpu import simplification
from AlphaSymbolic.core.gpu.grammar import GPUGrammar
from AlphaSymbolic.core.gpu.simplification import GPUSimplifier


class _InvalidNativeDecoder:
    @staticmethod
    def decode_rpn(*_args, **_kwargs):
        return ["Invalid"]


def test_native_invalid_decode_falls_back_to_python(monkeypatch):
    grammar = GPUGrammar(num_variables=1, use_globals=False)
    rpn = torch.tensor([grammar.token_to_id["x0"], 0, 0], dtype=torch.uint8)

    monkeypatch.setattr(simplification, "RPN_CUDA_AVAILABLE", True)
    monkeypatch.setattr(simplification, "rpn_cuda_native", _InvalidNativeDecoder())

    formula = GPUSimplifier.rpn_to_infix_static(rpn, torch.zeros(1), grammar)

    assert formula == "x0"
