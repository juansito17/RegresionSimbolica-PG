import math

import numpy as np
import pytest

from AlphaSymbolic.utils.benchmark_comparison import (
    _compute_rmse,
    _normalize_power_syntax,
    run_comparison_benchmark,
)


def test_polynomial_baseline_uses_a_real_holdout_protocol():
    result = run_comparison_benchmark(
        methods=["polynomial"], n_problems=1, seed=123
    )

    row = result["results"][0]
    assert row["method"] == "polynomial"
    assert row["train_rmse"] < 1e-10
    assert row["test_rmse"] < 1e-10
    assert result["protocol"]["holdout"] == "independent_random_stream"
    assert result["summary"]["polynomial"]["valid_runs"] == 1
    assert result["summary"]["polynomial"]["failed"] == 0
    assert math.isfinite(result["summary"]["polynomial"]["avg_rmse"])


def test_unimplemented_labels_are_rejected_instead_of_aliasing_gpu_gp():
    with pytest.raises(ValueError, match="no implementados"):
        run_comparison_benchmark(methods=["beam"], n_problems=1)


def test_power_normalization_preserves_existing_double_asterisks():
    formula = "x0^2 + x0**3"

    normalized = _normalize_power_syntax(formula)

    assert normalized == "x0**2 + x0**3"
    assert "****" not in normalized


@pytest.mark.parametrize("formula", ["x0^2", "x0**2"])
def test_rmse_accepts_engine_and_python_power_syntax(formula):
    x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = x**2

    assert _compute_rmse(formula, x, y) == pytest.approx(0.0)
