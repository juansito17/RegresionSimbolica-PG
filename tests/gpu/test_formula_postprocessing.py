import sympy

from warpsymbolic.gpu.engine import TensorGeneticEngine


def test_formula_canonicalization_avoids_unbounded_sympy_search(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("unbounded SymPy simplification must not run")

    monkeypatch.setattr(sympy, "simplify", fail_if_called)
    monkeypatch.setattr(sympy, "nsimplify", fail_if_called)

    # This exact expression previously spent more than ten minutes inside
    # sympy.simplify -> trigsimp after a 60-second engine budget.
    formula = (
        "(x0 + cos((((sqrt(pi) / ((5 / x0) + "
        "((x0 + ((0 - pi) / ((-0.298541 + cos((x0 - 5))) / e))) / 5))) "
        "- x0) - 5)))"
    )

    result = TensorGeneticEngine._simplify_with_sympy(formula)

    assert result
    assert "x0" in result
    assert "cos" in result


def test_formula_canonicalization_keeps_basic_algebra_cleanup():
    result = TensorGeneticEngine._simplify_with_sympy("(x0 + 0)")

    assert result == "x0"
