import math

import numpy as np

from warpsymbolic.symbolic.grammar import ExpressionTree
from warpsymbolic.gpu.ensemble import EnsembleRunner


def test_caret_is_power_with_power_precedence():
    tree = ExpressionTree.from_infix("x0^2+1")

    assert tree.is_valid
    np.testing.assert_allclose(tree.evaluate(np.array([2.0, 3.0])), [5.0, 10.0])


def test_caret_remains_right_associative_like_exponentiation():
    tree = ExpressionTree.from_infix("2^3^2")

    assert tree.is_valid
    np.testing.assert_allclose(tree.evaluate(np.array([0.0])), [512.0])


def test_ensemble_reads_the_run_fitness_that_matches_the_formula():
    class FakeEngine:
        last_run_best_rmse = math.inf

        def run(self, *_args, **_kwargs):
            self.last_run_best_rmse = 0.125
            return "x0"

    runner = EnsembleRunner(lambda: FakeEngine(), n_runs=1)
    formula, rmse, run_id = runner.run_single(
        FakeEngine(), [0.0], [0.0], [], timeout_sec=0.1, run_id=7
    )

    assert formula == "x0"
    assert rmse == 0.125
    assert run_id == 7
