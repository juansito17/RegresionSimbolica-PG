from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def test_production_runner_has_no_dataset_identity_branch():
    source = (ROOT / "AlphaSymbolic" / "scripts" / "srbench_runner.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = ("dataset_group", "a000170", "nqueens", "n-queens", "target_formula")
    assert not any(token in source for token in forbidden)


def test_engine_core_has_no_embedded_evaluation_sequence():
    core = ROOT / "AlphaSymbolic" / "core" / "gpu"
    source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in core.rglob("*")
        if path.suffix in {".py", ".cu", ".cpp", ".h", ".hpp"}
    )
    forbidden = ("a000170", "nqueens", "n-queens", "problem_y_full", "var_mod_x1")
    assert not any(token in source for token in forbidden)


def test_runner_parameters_ignore_dataset_group_at_execution():
    from AlphaSymbolic.scripts.srbench_runner import _effective_params

    base = dict(
        runner_params={},
        population_size=50_000,
        generations=150,
        fit_time_limit_sec=3600.0,
        random_state=7,
    )
    first = _effective_params(SimpleNamespace(**base, dataset_group="blackbox"), 20)
    second = _effective_params(
        SimpleNamespace(**base, dataset_group="firstprinciples"), 20
    )
    assert first == second
