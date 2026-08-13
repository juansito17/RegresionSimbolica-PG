import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from AlphaSymbolic.scripts.benchmark_srbench import (
    DEFAULT_MANIFEST,
    DatasetSpec,
    ExecutionPlan,
    ProtocolError,
    RunnerHandle,
    build_ranking,
    download_verified,
    execute_plan,
    load_manifest,
    prepare_dataset,
    read_jsonl,
    resolve_plan,
)
from AlphaSymbolic.scripts import srbench_runner


def test_pinned_manifest_and_profiles_are_complete():
    manifest = load_manifest()
    assert manifest["sources"]["srbench"]["commit"] == (
        "dc3f6daa93bf10955df8775256a6f8644f38fd93"
    )
    assert manifest["sources"]["pmlb"]["commit"] == (
        "7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68"
    )
    assert len(manifest["datasets"]) == 24
    assert len(manifest["seeds"]) == 30
    assert all(len(item["sha256"]) == 64 for item in manifest["datasets"])
    assert "raw.githubusercontent.com" not in (
        manifest["sources"]["pmlb"]["dataset_url_template"]
    )

    quick = resolve_plan(manifest, "smoke")
    full = resolve_plan(manifest, "full")
    official = resolve_plan(manifest, "official")
    assert quick.profile == "quick"
    assert quick.task_count == 2
    assert (quick.generations, quick.population_size) == (20, 10_000)
    assert full.task_count == 72
    assert (full.generations, full.population_size) == (100, 50_000)
    assert official.task_count == 720
    assert (official.generations, official.population_size) == (150, 50_000)
    assert official.official_protocol is False
    assert "fixed_runner_skips_upstream_hyperparameter_tuning" in (
        official.override_reasons
    )


def test_verified_download_is_atomic_reused_and_rejects_corruption(tmp_path):
    payload = b"pinned bytes\n"
    expected_hash = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "artifact.bin"
    calls = []

    def opener(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    path, downloaded = download_verified(
        "https://example.invalid/artifact",
        destination,
        expected_hash,
        len(payload),
        opener=opener,
    )
    assert downloaded is True
    assert path.read_bytes() == payload

    _, downloaded = download_verified(
        "https://example.invalid/artifact",
        destination,
        expected_hash,
        len(payload),
        opener=opener,
    )
    assert downloaded is False
    assert len(calls) == 1

    destination.write_bytes(b"corrupt")
    with pytest.raises(ProtocolError, match="size mismatch"):
        download_verified(
            "https://example.invalid/artifact",
            destination,
            expected_hash,
            len(payload),
            offline=True,
        )


def test_split_cap_and_scaling_are_deterministic_and_train_only():
    X = np.column_stack(
        (
            np.arange(100, dtype=np.float64),
            np.linspace(-3.0, 7.0, 100),
        )
    )
    y = 4.0 * X[:, 0] - 2.0
    spec = DatasetSpec("fixture", "blackbox", "0" * 64, 1, 100, 2)

    first = prepare_dataset(X, y, ("x0", "x1"), spec, 23654, max_train_samples=20)
    second = prepare_dataset(X, y, ("x0", "x1"), spec, 23654, max_train_samples=20)
    other = prepare_dataset(X, y, ("x0", "x1"), spec, 15795, max_train_samples=20)

    assert first.n_train_before_cap == 75
    assert len(first.y_train) == 20
    assert len(first.y_test_raw) == 25
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    assert not np.array_equal(first.train_indices, other.train_indices)
    np.testing.assert_allclose(first.X_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(first.X_train.std(axis=0), 1.0, atol=1e-12)
    assert first.y_train.mean() == pytest.approx(0.0, abs=1e-12)
    assert first.y_train.std() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        first.inverse_target(first.y_train), first.y_train_raw, atol=1e-12
    )


def _write_tiny_cached_dataset(cache_dir):
    text = "x0\tx1\ttarget\n" + "\n".join(
        f"{value}\t{value * 3}\t{value}" for value in range(12)
    ) + "\n"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    payload = buffer.getvalue()
    destination = cache_dir / "datasets" / "tiny.tsv.gz"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    return payload


def test_incremental_jsonl_resume_skips_a_terminal_run(tmp_path):
    cache_dir = tmp_path / "cache"
    payload = _write_tiny_cached_dataset(cache_dir)
    spec = DatasetSpec(
        "tiny",
        "blackbox",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        12,
        2,
    )
    manifest = {
        "sources": {
            "srbench": {"commit": "a" * 40},
            "pmlb": {
                "commit": "b" * 40,
                "dataset_url_template": "https://example.invalid/{dataset}",
            },
        },
        "seeds": [7],
    }
    plan = ExecutionPlan(
        profile="quick",
        datasets=(spec,),
        seeds=(7,),
        fit_time_limit_sec=1.0,
        max_train_samples=40_000,
        generations=2,
        population_size=10,
        official_protocol=False,
        override_reasons=("test",),
    )
    calls = []

    def fake_runner(prepared, context):
        calls.append((prepared.dataset, context.random_state))
        # target == x0, so independently standardized x0 and y are identical.
        return {
            "train_predictions_scaled": prepared.X_train[:, 0],
            "test_predictions_scaled": prepared.X_test[:, 0],
            "model_size": 1,
            "symbolic_model": "x0",
            "params": {"fixture": True},
        }

    runner = RunnerHandle(fake_runner, "tests:fake", None, "c" * 64)
    output = tmp_path / "runs.jsonl"
    source = {
        "alphasymbolic_commit": "d" * 40,
        "alphasymbolic_tree_dirty": False,
        "alphasymbolic_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }
    first = execute_plan(
        manifest,
        hashlib.sha256(b"manifest").hexdigest(),
        plan,
        runner,
        "Fixture",
        cache_dir,
        output,
        offline=True,
        source_metadata=source,
    )
    second = execute_plan(
        manifest,
        hashlib.sha256(b"manifest").hexdigest(),
        plan,
        runner,
        "Fixture",
        cache_dir,
        output,
        offline=True,
        source_metadata=source,
    )
    assert first == {"planned": 1, "completed": 1, "failed": 0, "skipped": 0}
    assert second == {"planned": 1, "completed": 0, "failed": 0, "skipped": 1}
    assert calls == [("tiny", 7)]
    records = read_jsonl(output)
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["metrics"]["test"]["r2"] == pytest.approx(1.0)


def test_ranking_is_explicitly_asymmetric_and_uses_correct_directions():
    official = pd.DataFrame(
        {
            "dataset": ["d1", "d2"],
            "algorithm": ["Reference", "Reference"],
            "random_state": [1, 1],
            "r2_test": [0.8, 0.7],
            "mse_test": [0.2, 0.3],
            "mae_test": [0.1, 0.2],
            "model_size": [4, 4],
            "training time (s)": [2.0, 2.0],
        }
    )
    local = [
        {
            "record_type": "srbench_run",
            "run_id": f"run-{dataset}",
            "status": "ok",
            "official_protocol": False,
            "algorithm": "Candidate",
            "dataset": dataset,
            "random_state": 1,
            "training_time_sec": 1.0,
            "model_size": 2,
            "metrics": {"test": {"r2": r2, "mse": 0.1, "mae": 0.05}},
        }
        for dataset, r2 in (("d1", 0.9), ("d2", 0.85))
    ]
    ranking = build_ranking(
        official,
        local,
        expected_official_datasets=["d1", "d2"],
        expected_official_seeds=[1],
    )
    assert ranking["official_leaderboard"] is False
    assert ranking["comparable_to_official"] is False
    assert ranking["comparison_basis"] == (
        "local_fixed_config_vs_upstream_tuned_reference"
    )
    assert ranking["algorithm_ranking"][0]["algorithm"] == "Candidate [local]"
    assert ranking["algorithm_ranking"][0][
        "estimated_reference_r2_rank_position"
    ] == 1


def test_truncated_jsonl_tail_is_ignored_but_interior_corruption_is_rejected(tmp_path):
    output = tmp_path / "runs.jsonl"
    output.write_text('{"run_id":"ok"}\n{"run_id":', encoding="utf-8")
    assert read_jsonl(output) == [{"run_id": "ok"}]
    output.write_text('{"run_id":\n{"run_id":"ok"}\n', encoding="utf-8")
    with pytest.raises(ProtocolError, match="invalid JSONL"):
        read_jsonl(output)


def test_manifest_is_adjacent_to_module_and_tracked_payload_is_valid_json():
    assert DEFAULT_MANIFEST.name == "srbench_2025_manifest.json"
    assert json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["schema_version"] == 1


def test_default_runner_applies_context_budgets_and_returns_cached_model(monkeypatch):
    captured = {}

    class FakeEstimator:
        def __init__(self, **params):
            captured.update(params)

        def fit(self, X, y):
            self.symbolic_complexity_ = 3
            self.sympy_formula_ = "feature_a"
            self.fit_status_ = "engine"
            self.selection_reason_ = "engine_validation_rmse"
            self.selected_feature_indices_ = np.array([0])
            self.selected_feature_names_ = np.array(["feature_a"])
            self.validation_size_ = 2
            self.validation_candidates_ = [
                {"name": "engine", "degree": None, "rmse": 0.1, "complexity": 3}
            ]
            self.polynomial_degree_ = None
            self.engine_validation_rmse_ = 0.1
            self.polynomial_validation_rmse_ = np.inf
            self.fallback_validation_rmse_ = 0.2
            self.engine_error_ = None
            self.n_gpu_samples_ = len(X)
            return self

        def predict(self, X):
            return X.iloc[:, 0].to_numpy()

    monkeypatch.setattr(srbench_runner, "AlphaSymbolicRegressor", FakeEstimator)
    prepared = SimpleNamespace(
        X_train=np.array([[0.0, 2.0], [1.0, 3.0]]),
        X_test=np.array([[2.0, 4.0]]),
        y_train=np.array([0.0, 1.0]),
        feature_names=("feature_a", "feature_b"),
    )
    context = SimpleNamespace(
        runner_params={"polynomial_degree": 2},
        population_size=12_345,
        generations=17,
        fit_time_limit_sec=9.0,
        random_state=123,
        dataset_group="blackbox",
    )

    result = srbench_runner.run(prepared, context)

    assert captured["pop_size"] == 12_345
    assert captured["generations"] == 17
    assert captured["max_time"] == 9.0
    assert captured["random_state"] == 123
    assert captured["polynomial_degree"] == 2
    assert result["symbolic_model"] == "feature_a"
    np.testing.assert_allclose(result["test_predictions_scaled"], [2.0])

    context.runner_params = {"pop_size": 1}
    with pytest.raises(ValueError, match="context-controlled"):
        srbench_runner.run(prepared, context)


def test_default_runner_adapts_tiny_budget_from_training_rows_only(monkeypatch):
    captured = {}

    class FakeEstimator:
        def __init__(self, **params):
            captured.update(params)

        def fit(self, X, y):
            self.symbolic_complexity_ = 1
            self.sympy_formula_ = "0"
            self.fit_status_ = "fallback"
            self.selection_reason_ = "fallback_validation_rmse"
            self.selected_feature_indices_ = np.array([0])
            self.selected_feature_names_ = np.array(["x0"])
            self.validation_size_ = 1
            self.validation_candidates_ = []
            self.polynomial_degree_ = None
            self.engine_validation_rmse_ = np.inf
            self.polynomial_validation_rmse_ = np.inf
            self.fallback_validation_rmse_ = 0.0
            self.engine_error_ = None
            self.n_gpu_samples_ = len(X)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(srbench_runner, "AlphaSymbolicRegressor", FakeEstimator)
    prepared = SimpleNamespace(
        X_train=np.zeros((6, 1)),
        X_test=np.zeros((2, 1)),
        y_train=np.zeros(6),
        feature_names=("x0",),
    )
    context = SimpleNamespace(
        runner_params={},
        population_size=50_000,
        generations=150,
        fit_time_limit_sec=3600.0,
        random_state=7,
        dataset_group="firstprinciples",
    )

    result = srbench_runner.run(prepared, context)

    assert captured["pop_size"] == 50_000
    assert captured["generations"] == 150
    assert captured["max_time"] == 60.0
    assert result["metadata"]["budget_policy"] == "universal_frozen"
