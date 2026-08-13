"""Pinned, resumable SRBench 2025 evaluation harness.

The harness owns data provenance, the official split/scaling protocol,
out-of-sample metrics, incremental persistence, and comparison with the
published SRBench Feather artifacts.  The estimator itself is supplied by a
small pluggable runner (``module:function``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(__file__).with_name("srbench_2025_manifest.json")
DEFAULT_CACHE = Path(".local/cache/srbench_2025")
DEFAULT_OUTPUT = Path("benchmarks/raw/srbench_2025.jsonl")
DEFAULT_RUNNER = "warpsymbolic.cli.srbench_runner:run"
PROFILE_ALIASES = {"smoke": "quick"}
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class ProtocolError(RuntimeError):
    """Raised when an input would make the benchmark non-reproducible."""


class _GpuEnergyMeter:
    """Best-effort GPU joule meter using the vendor-neutral CLI boundary.

    Missing power telemetry produces ``None`` instead of silently reporting
    zero.  Samples and method are persisted so energy ranks remain auditable.
    """

    def __init__(self, interval_sec: float = 0.2):
        self.interval_sec = float(interval_sec)
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _power_watts() -> Optional[float]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return None
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=True,
            )
            values = [
                float(line.strip())
                for line in completed.stdout.splitlines()
                if line.strip() and line.strip().lower() not in {"n/a", "[n/a]"}
            ]
            return float(sum(values)) if values else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _sample(self) -> bool:
        power = self._power_watts()
        if power is None:
            return False
        self.samples.append((time.perf_counter(), power))
        return True

    def start(self) -> None:
        if not self._sample():
            return

        def monitor() -> None:
            while not self._stop.wait(self.interval_sec):
                self._sample()

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()
        if len(self.samples) < 2:
            return None
        return float(
            sum(
                0.5 * (left[1] + right[1]) * (right[0] - left[0])
                for left, right in zip(self.samples, self.samples[1:])
            )
        )


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_group: str
    sha256: str
    size_bytes: int
    rows: int
    features: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetSpec":
        return cls(
            name=str(value["name"]),
            dataset_group=str(value["dataset_group"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            rows=int(value["rows"]),
            features=int(value["features"]),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    profile: str
    datasets: Tuple[DatasetSpec, ...]
    seeds: Tuple[int, ...]
    fit_time_limit_sec: float
    max_train_samples: int
    generations: int
    population_size: int
    official_protocol: bool
    override_reasons: Tuple[str, ...]
    runner_params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_count(self) -> int:
        return len(self.datasets) * len(self.seeds)


@dataclass(frozen=True)
class PreparedDataset:
    """Train/test arrays passed to an estimator runner.

    ``X_*`` and ``y_train`` are standardized with scalers fitted only on the
    capped training split.  The runner must return predictions in this scaled
    target space; this harness inverse-transforms them before scoring.
    """

    dataset: str
    dataset_group: str
    random_state: int
    feature_names: Tuple[str, ...]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    n_train_before_cap: int
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    def inverse_target(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return array * self.y_scale + self.y_mean


@dataclass(frozen=True)
class RunContext:
    profile: str
    algorithm: str
    dataset: str
    dataset_group: str
    random_state: int
    trial: int
    fit_time_limit_sec: float
    max_train_samples: int
    generations: int
    population_size: int
    official_protocol: bool
    protocol_sha256: str
    runner_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerHandle:
    function: Callable[[PreparedDataset, RunContext], Mapping[str, Any]]
    specification: str
    source_file: Optional[str]
    source_sha256: Optional[str]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read SRBench manifest {path}: {exc}") from exc
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ProtocolError("unsupported SRBench manifest schema")
    sources = manifest.get("sources", {})
    for source in ("srbench", "pmlb"):
        commit = str(sources.get(source, {}).get("commit", ""))
        if not HEX_40.fullmatch(commit):
            raise ProtocolError(f"{source} commit must be a full 40-character SHA")
    template = str(sources["pmlb"].get("dataset_url_template", ""))
    required_prefix = "https://github.com/EpistasisLab/pmlb/raw/"
    if not template.startswith(required_prefix) or "raw.githubusercontent.com" in template:
        raise ProtocolError("PMLB downloads must use the pinned github.com/.../raw URL")

    seeds = manifest.get("seeds", [])
    if len(seeds) != 30 or len(set(seeds)) != 30 or not all(isinstance(v, int) for v in seeds):
        raise ProtocolError("the manifest must contain the 30 unique official integer seeds")

    datasets = [DatasetSpec.from_dict(item) for item in manifest.get("datasets", [])]
    if len(datasets) != 24 or len({item.name for item in datasets}) != 24:
        raise ProtocolError("the pinned suite must contain 24 unique datasets")
    groups = {group: sum(item.dataset_group == group for item in datasets)
              for group in ("blackbox", "firstprinciples")}
    if groups != {"blackbox": 12, "firstprinciples": 12}:
        raise ProtocolError(f"expected 12 datasets per track, found {groups}")
    for item in datasets:
        if not HEX_64.fullmatch(item.sha256):
            raise ProtocolError(f"invalid SHA-256 for dataset {item.name}")
        if item.size_bytes <= 0 or item.rows <= 0 or item.features <= 0:
            raise ProtocolError(f"invalid dimensions for dataset {item.name}")

    result_groups = set()
    srbench_commit = sources["srbench"]["commit"]
    for result in manifest.get("official_results", []):
        group = str(result.get("dataset_group", ""))
        result_groups.add(group)
        if not HEX_64.fullmatch(str(result.get("sha256", ""))):
            raise ProtocolError(f"invalid official Feather SHA-256 for {group}")
        if int(result.get("size_bytes", 0)) <= 0:
            raise ProtocolError(f"invalid official Feather size for {group}")
        if srbench_commit not in str(result.get("url", "")):
            raise ProtocolError(f"official Feather URL is not pinned for {group}")
    if result_groups != {"blackbox", "firstprinciples"}:
        raise ProtocolError("both official SRBench result tracks must be pinned")

    protocol = manifest.get("protocol", {})
    expected = {
        "train_fraction": 0.75,
        "test_fraction": 0.25,
        "scale_x": True,
        "scale_y": True,
        "max_train_samples": 40000,
        "official_trials": 30,
        "official_fit_time_limit_sec": 3600,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ProtocolError(f"manifest protocol {key!r} must equal {value!r}")


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ProtocolError("a CSV override cannot be empty")
    return parsed


def _parse_runner_params(values: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in values:
        key, separator, raw = item.partition("=")
        if not separator or not key.strip():
            raise ProtocolError(f"runner parameter must be KEY=JSON, got {item!r}")
        try:
            result[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            result[key.strip()] = raw
    return result


def resolve_plan(
    manifest: Mapping[str, Any],
    profile: str,
    track: str = "all",
    dataset_names: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    fit_time_limit_sec: Optional[float] = None,
    max_train_samples: Optional[int] = None,
    generations: Optional[int] = None,
    population_size: Optional[int] = None,
    runner_params: Optional[Mapping[str, Any]] = None,
) -> ExecutionPlan:
    canonical_profile = PROFILE_ALIASES.get(profile, profile)
    profiles = manifest["profiles"]
    if canonical_profile not in profiles:
        raise ProtocolError(f"unknown profile {profile!r}")
    if track not in {"all", "blackbox", "firstprinciples"}:
        raise ProtocolError(f"unknown dataset track {track!r}")
    config = profiles[canonical_profile]
    all_specs = [DatasetSpec.from_dict(item) for item in manifest["datasets"]]
    by_name = {item.name: item for item in all_specs}

    if dataset_names is not None:
        if len(set(dataset_names)) != len(dataset_names):
            raise ProtocolError("dataset override contains duplicates")
        unknown = sorted(set(dataset_names) - set(by_name))
        if unknown:
            raise ProtocolError(f"unknown datasets: {unknown}")
        selected = [by_name[name] for name in dataset_names]
    elif "datasets" in config:
        selected = [by_name[name] for name in config["datasets"]]
    else:
        groups = set(config["dataset_groups"])
        selected = [item for item in all_specs if item.dataset_group in groups]
    if track != "all":
        selected = [item for item in selected if item.dataset_group == track]
    if not selected:
        raise ProtocolError("profile/track selection produced zero datasets")

    official_seeds = tuple(int(value) for value in manifest["seeds"])
    selected_seeds = (
        tuple(int(value) for value in seeds)
        if seeds is not None
        else official_seeds[: int(config["seed_count"])]
    )
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ProtocolError("seeds must be a non-empty unique sequence")

    effective_time = float(
        config["fit_time_limit_sec"] if fit_time_limit_sec is None else fit_time_limit_sec
    )
    effective_max = int(
        manifest["protocol"]["max_train_samples"]
        if max_train_samples is None
        else max_train_samples
    )
    effective_generations = int(
        config["generations"] if generations is None else generations
    )
    effective_population = int(
        config["population_size"] if population_size is None else population_size
    )
    if effective_time <= 0 or effective_max <= 0:
        raise ProtocolError("time limit and max train samples must be positive")
    if effective_generations <= 0 or effective_population <= 0:
        raise ProtocolError("generations and population size must be positive")

    # Coverage can reproduce 24 datasets x 30 trials, but this local runner
    # deliberately does not reproduce SRBench's upstream optimize_model/grid-CV
    # stage.  Never label a fixed-config local run as the official protocol.
    reasons: List[str] = ["fixed_runner_skips_upstream_hyperparameter_tuning"]
    official_config = profiles["official"]
    if canonical_profile != "official":
        reasons.append(f"profile={canonical_profile}")
    if tuple(item.name for item in selected) != tuple(item.name for item in all_specs):
        reasons.append("dataset selection differs from the 24-dataset official suite")
    if selected_seeds != official_seeds:
        reasons.append("seed selection differs from the 30 official seeds")
    if effective_time != float(manifest["protocol"]["official_fit_time_limit_sec"]):
        reasons.append("fit time limit differs from 3600 seconds")
    if effective_max != int(manifest["protocol"]["max_train_samples"]):
        reasons.append("training cap differs from 40000")
    if effective_generations != int(official_config["generations"]):
        reasons.append("generation budget differs from the pinned official profile")
    if effective_population != int(official_config["population_size"]):
        reasons.append("population differs from the pinned official profile")
    if track != "all":
        reasons.append(f"track={track}")
    if runner_params:
        reasons.append("custom runner parameters")

    return ExecutionPlan(
        profile=canonical_profile,
        datasets=tuple(selected),
        seeds=selected_seeds,
        fit_time_limit_sec=effective_time,
        max_train_samples=effective_max,
        generations=effective_generations,
        population_size=effective_population,
        official_protocol=not reasons,
        override_reasons=tuple(dict.fromkeys(reasons)),
        runner_params=dict(runner_params or {}),
    )


def verify_file(path: Path, expected_sha256: str, expected_size: int) -> None:
    if not path.is_file():
        raise ProtocolError(f"missing required file: {path}")
    actual_size = path.stat().st_size
    if actual_size != int(expected_size):
        raise ProtocolError(
            f"size mismatch for {path}: expected {expected_size}, found {actual_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ProtocolError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, found {actual_sha256}"
        )


def download_verified(
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int,
    offline: bool = False,
    timeout_sec: float = 120.0,
    opener: Optional[Callable[[str, float], Any]] = None,
) -> Tuple[Path, bool]:
    """Return ``(path, downloaded)`` after exact size/hash verification."""
    try:
        verify_file(destination, expected_sha256, expected_size)
        return destination, False
    except ProtocolError:
        if offline:
            raise

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=str(destination.parent)
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        if opener is None:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "WarpSymbolic-SRBench/1.0"},
            )
            response = urllib.request.urlopen(request, timeout=timeout_sec)
        else:
            response = opener(url, timeout_sec)
        with contextlib.closing(response), temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        verify_file(temporary, expected_sha256, expected_size)
        os.replace(temporary, destination)
        verify_file(destination, expected_sha256, expected_size)
        return destination, True
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def dataset_url(manifest: Mapping[str, Any], dataset: DatasetSpec) -> str:
    pmlb = manifest["sources"]["pmlb"]
    return str(pmlb["dataset_url_template"]).format(
        commit=pmlb["commit"], dataset=dataset.name
    )


def ensure_dataset(
    manifest: Mapping[str, Any],
    dataset: DatasetSpec,
    cache_dir: Path,
    offline: bool = False,
    timeout_sec: float = 120.0,
) -> Tuple[Path, bool]:
    destination = cache_dir / "datasets" / f"{dataset.name}.tsv.gz"
    return download_verified(
        dataset_url(manifest, dataset),
        destination,
        dataset.sha256,
        dataset.size_bytes,
        offline=offline,
        timeout_sec=timeout_sec,
    )


def read_dataset(
    path: Path, spec: DatasetSpec
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    try:
        frame = pd.read_csv(path, sep="\t", compression="gzip")
    except Exception as exc:
        raise ProtocolError(f"cannot parse pinned dataset {spec.name}: {exc}") from exc
    if "target" not in frame.columns:
        raise ProtocolError(f"dataset {spec.name} has no target column")
    feature_frame = frame.drop(columns=["target"])
    if len(frame) != spec.rows or feature_frame.shape[1] != spec.features:
        raise ProtocolError(
            f"dataset {spec.name} dimensions changed: expected "
            f"{spec.rows}x{spec.features}, found {len(frame)}x{feature_frame.shape[1]}"
        )
    try:
        X = feature_frame.to_numpy(dtype=np.float64)
        y = frame["target"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"dataset {spec.name} is not numeric: {exc}") from exc
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ProtocolError(f"dataset {spec.name} contains non-finite values")
    return X, y, tuple(str(name) for name in feature_frame.columns)


def prepare_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    spec: DatasetSpec,
    random_state: int,
    max_train_samples: int = 40000,
) -> PreparedDataset:
    """Apply the pinned SRBench split, cap, and train-only standardization."""
    indices = np.arange(len(y), dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        train_size=0.75,
        test_size=0.25,
        random_state=int(random_state),
    )
    n_train_before_cap = int(len(train_indices))
    if len(train_indices) > max_train_samples:
        # This matches np.random.seed(random_state) followed by np.random.choice
        # in the pinned SRBench evaluator. train_test_split uses its own seeded
        # RandomState, so it does not advance this generator.
        generator = np.random.RandomState(int(random_state))
        positions = generator.choice(
            np.arange(len(train_indices)),
            size=int(max_train_samples),
            replace=False,
        )
        train_indices = train_indices[positions]

    X_train_raw = np.asarray(X[train_indices], dtype=np.float64)
    X_test_raw = np.asarray(X[test_indices], dtype=np.float64)
    y_train_raw = np.asarray(y[train_indices], dtype=np.float64).reshape(-1)
    y_test_raw = np.asarray(y[test_indices], dtype=np.float64).reshape(-1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)

    return PreparedDataset(
        dataset=spec.name,
        dataset_group=spec.dataset_group,
        random_state=int(random_state),
        feature_names=tuple(feature_names),
        X_train=np.ascontiguousarray(X_train, dtype=np.float64),
        X_test=np.ascontiguousarray(X_test, dtype=np.float64),
        y_train=np.ascontiguousarray(y_train, dtype=np.float64),
        y_train_raw=np.ascontiguousarray(y_train_raw, dtype=np.float64),
        y_test_raw=np.ascontiguousarray(y_test_raw, dtype=np.float64),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        n_train_before_cap=n_train_before_cap,
        x_mean=np.asarray(x_scaler.mean_, dtype=np.float64),
        x_scale=np.asarray(x_scaler.scale_, dtype=np.float64),
        y_mean=float(y_scaler.mean_[0]),
        y_scale=float(y_scaler.scale_[0]),
    )


def _metrics(y_true: np.ndarray, y_prediction: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_prediction, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape:
        raise ProtocolError(
            f"prediction shape {prediction.shape} does not match target {truth.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ProtocolError("runner returned non-finite predictions")
    return {
        "r2": float(r2_score(truth, prediction)),
        "mse": float(mean_squared_error(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
    }


def score_runner_result(
    prepared: PreparedDataset, result: Mapping[str, Any]
) -> Dict[str, Dict[str, float]]:
    if not isinstance(result, Mapping):
        raise ProtocolError("runner must return a mapping")
    try:
        train_scaled = result["train_predictions_scaled"]
        test_scaled = result["test_predictions_scaled"]
    except KeyError as exc:
        raise ProtocolError(
            "runner result requires train_predictions_scaled and "
            "test_predictions_scaled"
        ) from exc
    return {
        "train": _metrics(
            prepared.y_train_raw, prepared.inverse_target(train_scaled)
        ),
        "test": _metrics(
            prepared.y_test_raw, prepared.inverse_target(test_scaled)
        ),
    }


def load_runner(specification: str) -> RunnerHandle:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ProtocolError("runner must use module:function syntax")
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise ProtocolError(f"cannot load runner {specification!r}: {exc}") from exc
    if not callable(function):
        raise ProtocolError(f"runner {specification!r} is not callable")
    source = inspect.getsourcefile(function)
    source_path = Path(source).resolve() if source else None
    source_sha = _sha256_file(source_path) if source_path and source_path.is_file() else None
    return RunnerHandle(
        function=function,
        specification=specification,
        source_file=str(source_path) if source_path else None,
        source_sha256=source_sha,
    )


def _git_metadata() -> Dict[str, Any]:
    def run(arguments: Sequence[str], binary: bool = False) -> Any:
        try:
            return subprocess.check_output(
                ["git", *arguments],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=not binary,
            )
        except Exception:
            return b"" if binary else "unknown"

    commit = str(run(["rev-parse", "HEAD"])).strip()
    status = str(run(["status", "--porcelain"])).strip()
    diff = run(["diff", "--binary", "HEAD"], binary=True)
    listed = run(
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        binary=True,
    )
    source_digest = hashlib.sha256()
    source_file_count = 0
    if isinstance(listed, bytes):
        for raw_path in sorted(path for path in listed.split(b"\0") if path):
            relative = raw_path.decode("utf-8", errors="surrogateescape")
            top_level = Path(relative).parts[0] if Path(relative).parts else ""
            if top_level.lower() in {"benchmarks", "cache", "outputs", "results"}:
                continue
            candidate = (REPO_ROOT / relative).resolve()
            try:
                candidate.relative_to(REPO_ROOT.resolve())
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            content = candidate.read_bytes()
            source_digest.update(len(raw_path).to_bytes(8, "little"))
            source_digest.update(raw_path)
            source_digest.update(len(content).to_bytes(8, "little"))
            source_digest.update(content)
            source_file_count += 1
    metadata = {
        "warpsymbolic_commit": commit,
        "warpsymbolic_tree_dirty": status not in {"", "unknown"},
        "warpsymbolic_diff_sha256": (
            hashlib.sha256(diff).hexdigest() if isinstance(diff, bytes) else None
        ),
        # Unlike ``git diff``, this also covers new untracked source files.
        "warpsymbolic_source_sha256": (
            source_digest.hexdigest() if source_file_count else None
        ),
        "warpsymbolic_source_file_count": source_file_count,
    }
    # Keep the old field names for one schema transition so resumptions and
    # downstream readers can identify records generated before the rename.
    metadata.update(
        {
            f"alphasymbolic_{key.removeprefix('warpsymbolic_')}": value
            for key, value in metadata.items()
        }
    )
    return metadata


def _environment_metadata() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(record)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    records: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            is_unterminated_tail = index == len(lines) - 1 and not text.endswith(("\n", "\r"))
            if is_unterminated_tail:
                print(f"warning: ignoring interrupted trailing JSONL record in {path}", file=sys.stderr)
                break
            raise ProtocolError(f"invalid JSONL at {path}:{index + 1}: {exc}") from exc
        if not isinstance(value, dict):
            raise ProtocolError(f"JSONL record at {path}:{index + 1} is not an object")
        records.append(value)
    return records


def _latest_by_run_id(records: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    latest: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("record_type") == "srbench_run" and record.get("run_id"):
            latest[str(record["run_id"])] = record
    return latest


def _trial_number(manifest: Mapping[str, Any], seed: int, fallback: int) -> int:
    try:
        return list(manifest["seeds"]).index(seed) + 1
    except ValueError:
        return fallback


def _protocol_payload(
    manifest: Mapping[str, Any], manifest_sha256: str, plan: ExecutionPlan
) -> Dict[str, Any]:
    return {
        "manifest_sha256": manifest_sha256,
        "srbench_commit": manifest["sources"]["srbench"]["commit"],
        "pmlb_commit": manifest["sources"]["pmlb"]["commit"],
        "profile": plan.profile,
        "datasets": [item.name for item in plan.datasets],
        "seeds": list(plan.seeds),
        "train_fraction": 0.75,
        "test_fraction": 0.25,
        "scale_x": True,
        "scale_y": True,
        "max_train_samples": plan.max_train_samples,
        "fit_time_limit_sec": plan.fit_time_limit_sec,
        "generations": plan.generations,
        "population_size": plan.population_size,
        "runner_params": plan.runner_params,
    }


def execute_plan(
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    plan: ExecutionPlan,
    runner: RunnerHandle,
    algorithm: str,
    cache_dir: Path,
    output: Path,
    resume: bool = True,
    retry_failed: bool = False,
    offline: bool = False,
    download_timeout_sec: float = 120.0,
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    existing = read_jsonl(output)
    if existing and not resume:
        raise ProtocolError(
            f"{output} already has records; --no-resume requires a new output path"
        )
    source = dict(source_metadata or _git_metadata())
    protocol_sha256 = _digest_json(_protocol_payload(manifest, manifest_sha256, plan))
    namespace_payload = {
        "protocol_sha256": protocol_sha256,
        "algorithm": algorithm,
        "runner": runner.specification,
        "runner_source_sha256": runner.source_sha256,
        **source,
    }
    namespace_sha256 = _digest_json(namespace_payload)
    old_namespaces = {
        str(row["run_namespace_sha256"])
        for row in existing
        if row.get("record_type") == "srbench_run" and row.get("run_namespace_sha256")
    }
    if old_namespaces and old_namespaces != {namespace_sha256}:
        raise ProtocolError(
            "output contains a different source/protocol namespace; choose a new JSONL"
        )
    latest = _latest_by_run_id(existing)
    stats = {"planned": plan.task_count, "completed": 0, "failed": 0, "skipped": 0}
    environment = _environment_metadata()

    for dataset in plan.datasets:
        dataset_path, downloaded = ensure_dataset(
            manifest,
            dataset,
            cache_dir,
            offline=offline,
            timeout_sec=download_timeout_sec,
        )
        print(
            f"dataset={dataset.name} cache={'downloaded' if downloaded else 'verified'}"
        )
        X, y, feature_names = read_dataset(dataset_path, dataset)
        for fallback_trial, seed in enumerate(plan.seeds, start=1):
            trial = _trial_number(manifest, seed, fallback_trial)
            run_id = _digest_json(
                {
                    "run_namespace_sha256": namespace_sha256,
                    "dataset": dataset.name,
                    "random_state": seed,
                }
            )
            previous = latest.get(run_id)
            if previous is not None and (
                previous.get("status") == "ok" or not retry_failed
            ):
                stats["skipped"] += 1
                print(f"skip dataset={dataset.name} seed={seed} status={previous.get('status')}")
                continue

            prepared = prepare_dataset(
                X,
                y,
                feature_names,
                dataset,
                random_state=seed,
                max_train_samples=plan.max_train_samples,
            )
            context = RunContext(
                profile=plan.profile,
                algorithm=algorithm,
                dataset=dataset.name,
                dataset_group=dataset.dataset_group,
                random_state=seed,
                trial=trial,
                fit_time_limit_sec=plan.fit_time_limit_sec,
                max_train_samples=plan.max_train_samples,
                generations=plan.generations,
                population_size=plan.population_size,
                official_protocol=plan.official_protocol,
                protocol_sha256=protocol_sha256,
                runner_params=plan.runner_params,
            )
            base_record: Dict[str, Any] = {
                "schema_version": 1,
                "record_type": "srbench_run",
                "run_id": run_id,
                "run_namespace_sha256": namespace_sha256,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "profile": plan.profile,
                "official_protocol": plan.official_protocol,
                "override_reasons": list(plan.override_reasons),
                "algorithm": algorithm,
                "runner": runner.specification,
                "runner_source_file": runner.source_file,
                "runner_source_sha256": runner.source_sha256,
                "dataset": dataset.name,
                "dataset_group": dataset.dataset_group,
                "dataset_sha256": dataset.sha256,
                "random_state": seed,
                "seed": seed,
                "trial": trial,
                "srbench_commit": manifest["sources"]["srbench"]["commit"],
                "pmlb_commit": manifest["sources"]["pmlb"]["commit"],
                "manifest_sha256": manifest_sha256,
                "protocol_sha256": protocol_sha256,
                "fit_time_limit_sec": plan.fit_time_limit_sec,
                "max_train_samples": plan.max_train_samples,
                "generations": plan.generations,
                "population_size": plan.population_size,
                "runner_params": plan.runner_params,
                "split": {
                    "train_fraction": 0.75,
                    "test_fraction": 0.25,
                    "n_rows": int(len(y)),
                    "n_train_before_cap": prepared.n_train_before_cap,
                    "n_train": int(len(prepared.y_train)),
                    "n_test": int(len(prepared.y_test_raw)),
                    "train_indices_sha256": _indices_sha256(prepared.train_indices),
                    "test_indices_sha256": _indices_sha256(prepared.test_indices),
                },
                "scaling": {
                    "x": True,
                    "y": True,
                    "fit_on": "capped_training_split_only",
                    "x_mean": prepared.x_mean,
                    "x_scale": prepared.x_scale,
                    "y_mean": prepared.y_mean,
                    "y_scale": prepared.y_scale,
                },
                "environment": environment,
                **source,
            }
            energy_meter = _GpuEnergyMeter()
            energy_meter.start()
            started = time.perf_counter()
            try:
                result = runner.function(prepared, context)
                elapsed = time.perf_counter() - started
                energy_joules = energy_meter.stop()
                metrics = score_runner_result(prepared, result)
                model_size = result.get("model_size")
                if model_size is not None:
                    model_size = int(model_size)
                    if model_size < 0:
                        raise ProtocolError("model_size cannot be negative")
                record = {
                    **base_record,
                    "status": "ok",
                    "error": None,
                    "training_time_sec": elapsed,
                    "reported_training_time_sec": result.get("training_time_sec"),
                    "energy_joules": energy_joules,
                    "energy_measurement": "nvidia-smi-power-trapezoid" if energy_joules is not None else None,
                    "energy_samples": len(energy_meter.samples),
                    "metrics": metrics,
                    "model_size": model_size,
                    "symbolic_model": result.get("symbolic_model"),
                    "params": result.get("params", {}),
                    "runner_metadata": result.get("metadata", {}),
                }
                stats["completed"] += 1
                print(
                    f"ok dataset={dataset.name} seed={seed} "
                    f"r2_test={metrics['test']['r2']:.6g} elapsed={elapsed:.3f}s"
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                energy_joules = energy_meter.stop()
                record = {
                    **base_record,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_traceback": traceback.format_exc(limit=20),
                    "training_time_sec": elapsed,
                    "reported_training_time_sec": None,
                    "energy_joules": energy_joules,
                    "energy_measurement": "nvidia-smi-power-trapezoid" if energy_joules is not None else None,
                    "energy_samples": len(energy_meter.samples),
                    "metrics": None,
                    "model_size": None,
                    "symbolic_model": None,
                    "params": {},
                    "runner_metadata": {},
                }
                stats["failed"] += 1
                print(
                    f"error dataset={dataset.name} seed={seed}: {record['error']}",
                    file=sys.stderr,
                )
            append_jsonl(output, record)
            latest[run_id] = record
    return stats


def ensure_official_results(
    manifest: Mapping[str, Any],
    groups: Iterable[str],
    cache_dir: Path,
    offline: bool = False,
    timeout_sec: float = 120.0,
) -> List[Path]:
    selected = set(groups)
    paths: List[Path] = []
    for result in manifest["official_results"]:
        group = str(result["dataset_group"])
        if group not in selected:
            continue
        destination = cache_dir / "official_results" / f"{group}.feather"
        path, downloaded = download_verified(
            str(result["url"]),
            destination,
            str(result["sha256"]),
            int(result["size_bytes"]),
            offline=offline,
            timeout_sec=timeout_sec,
        )
        print(
            f"official_results={group} cache={'downloaded' if downloaded else 'verified'}"
        )
        paths.append(path)
    return paths


def _find_column(
    frame: pd.DataFrame, candidates: Sequence[str], required: bool = True
) -> Optional[str]:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ProtocolError(
            f"official results lack any of columns {list(candidates)}; "
            f"found {list(frame.columns)}"
        )
    return None


def normalize_official_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dataset_column = _find_column(frame, ["dataset"])
    algorithm_column = _find_column(frame, ["algorithm"])
    r2_column = _find_column(frame, ["r2_test", "test_r2", "r2"])
    mse_column = _find_column(frame, ["mse_test", "test_mse", "mse"], required=False)
    mae_column = _find_column(frame, ["mae_test", "test_mae", "mae"], required=False)
    size_column = _find_column(frame, ["model_size", "complexity"], required=False)
    time_column = _find_column(
        frame,
        ["training time (s)", "time_time", "training_time_sec", "time_train"],
        required=False,
    )
    energy_column = _find_column(
        frame,
        [
            "energy_joules",
            "energy (j)",
            "energy_j",
            "energy",
            "power_consumption(kWh)",
        ],
        required=False,
    )
    energy_multiplier = 3_600_000.0 if energy_column == "power_consumption(kWh)" else 1.0
    recovery_column = _find_column(
        frame,
        ["symbolic_recovery", "symbolic_solution", "symbolic_equivalence", "recovered"],
        required=False,
    )
    normalized = pd.DataFrame(
        {
            "dataset": frame[dataset_column].astype(str),
            "algorithm": frame[algorithm_column].astype(str),
            "r2_test": pd.to_numeric(frame[r2_column], errors="coerce"),
            "mse_test": (
                pd.to_numeric(frame[mse_column], errors="coerce")
                if mse_column
                else np.nan
            ),
            "mae_test": (
                pd.to_numeric(frame[mae_column], errors="coerce")
                if mae_column
                else np.nan
            ),
            "model_size": (
                pd.to_numeric(frame[size_column], errors="coerce")
                if size_column
                else np.nan
            ),
            "training_time_sec": (
                pd.to_numeric(frame[time_column], errors="coerce")
                if time_column
                else np.nan
            ),
            "energy_joules": (
                pd.to_numeric(frame[energy_column], errors="coerce") * energy_multiplier
                if energy_column
                else np.nan
            ),
            "symbolic_recovery": (
                pd.to_numeric(frame[recovery_column], errors="coerce")
                if recovery_column
                else np.nan
            ),
            "source": "srbench_official",
        }
    )
    return normalized


def _deduplicate_local_records(
    records: Iterable[Mapping[str, Any]]
) -> List[Mapping[str, Any]]:
    return list(_latest_by_run_id(records).values())


def local_results_frame(
    records: Iterable[Mapping[str, Any]]
) -> Tuple[pd.DataFrame, List[Mapping[str, Any]]]:
    latest = _deduplicate_local_records(records)
    rows: List[Dict[str, Any]] = []
    for record in latest:
        metrics = record.get("metrics") or {}
        test_metrics = metrics.get("test") or {}
        failed = record.get("status") != "ok"
        fit_limit = float(record.get("fit_time_limit_sec") or 0.0)
        algorithm = f"{record.get('algorithm', 'local')} [local]"
        rows.append(
            {
                "dataset": str(record.get("dataset")),
                "algorithm": algorithm,
                "r2_test": 0.0 if failed else test_metrics.get("r2"),
                "mse_test": np.inf if failed else test_metrics.get("mse"),
                "mae_test": np.inf if failed else test_metrics.get("mae"),
                "model_size": np.inf if failed else record.get("model_size"),
                "training_time_sec": (
                    fit_limit
                    if failed
                    else record.get("training_time_sec")
                ),
                "energy_joules": np.inf if failed else record.get("energy_joules"),
                "symbolic_recovery": (
                    np.nan
                    if failed
                    else (record.get("runner_metadata") or {}).get("symbolic_recovery")
                ),
                "source": "local",
            }
        )
    return pd.DataFrame(rows), latest


def _bootstrap_mean_ci(values: Iterable[float], seed: int = 20260726) -> list[float]:
    sample = np.asarray(list(values), dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return [math.nan, math.nan]
    if sample.size == 1:
        return [float(sample[0]), float(sample[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(2000, sample.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def build_ranking(
    official_frame: pd.DataFrame,
    local_records: Iterable[Mapping[str, Any]],
    expected_official_datasets: Optional[Sequence[str]] = None,
    expected_official_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    official = normalize_official_frame(official_frame)
    local, latest = local_results_frame(local_records)
    if local.empty:
        raise ProtocolError("no local SRBench records are available to rank")
    official_dataset_names = set(official["dataset"].astype(str))
    local_dataset_names = set(local["dataset"].astype(str))
    requested_datasets = (
        set(str(name) for name in expected_official_datasets)
        if expected_official_datasets
        else official_dataset_names
    )
    datasets = sorted(
        official_dataset_names.intersection(local_dataset_names, requested_datasets)
    )
    official = official[official["dataset"].isin(datasets)]
    local = local[local["dataset"].isin(datasets)]
    latest = [
        row for row in latest if str(row.get("dataset")) in set(datasets)
    ]
    if official.empty:
        raise ProtocolError("official Feather has no datasets in common with local records")
    combined = pd.concat([official, local], ignore_index=True, sort=False)
    for column in (
        "r2_test",
        "mse_test",
        "mae_test",
        "model_size",
        "training_time_sec",
        "energy_joules",
        "symbolic_recovery",
    ):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined["r2_zero_test"] = combined["r2_test"].clip(lower=0).fillna(0.0)
    combined["r2_over_0999"] = (combined["r2_test"] > 0.999).astype(float)
    medians = (
        combined.groupby(["algorithm", "dataset"], as_index=False)[
            [
                "r2_zero_test",
                "r2_test",
                "mse_test",
                "mae_test",
                "model_size",
                "training_time_sec",
                "energy_joules",
                "symbolic_recovery",
                "r2_over_0999",
            ]
        ]
        .median()
    )
    medians["r2_rank"] = medians.groupby("dataset")["r2_zero_test"].rank(
        ascending=False, method="average", na_option="bottom"
    )
    medians["recovery_rank"] = medians["symbolic_recovery"].groupby(
        medians["dataset"]
    ).rank(ascending=False, method="average", na_option="bottom")
    medians["r2_over_0999_rank"] = medians["r2_over_0999"].groupby(
        medians["dataset"]
    ).rank(ascending=False, method="average", na_option="bottom")
    for metric, rank_name in (
        ("model_size", "model_size_rank"),
        ("training_time_sec", "training_time_rank"),
        ("energy_joules", "energy_rank"),
    ):
        rank_values = medians[metric].fillna(np.inf)
        medians[rank_name] = rank_values.groupby(medians["dataset"]).rank(
            ascending=True, method="average"
        )

    summaries: List[Dict[str, Any]] = []
    dataset_count = len(datasets)
    for algorithm, group in medians.groupby("algorithm"):
        summaries.append(
            {
                "algorithm": str(algorithm),
                "datasets_present": int(group["dataset"].nunique()),
                "eligible_full_scope": int(group["dataset"].nunique()) == dataset_count,
                "mean_r2_rank": float(group["r2_rank"].mean()),
                "mean_model_size_rank": float(group["model_size_rank"].mean()),
                "mean_training_time_rank": float(group["training_time_rank"].mean()),
                "mean_energy_rank": float(group["energy_rank"].mean()),
                "mean_symbolic_recovery_rank": float(group["recovery_rank"].mean()),
                "mean_r2_over_0999_rank": float(group["r2_over_0999_rank"].mean()),
                "median_r2_test": float(group["r2_test"].median()),
                "median_model_size": float(group["model_size"].median()),
                "median_training_time_sec": float(group["training_time_sec"].median()),
                "median_energy_joules": float(group["energy_joules"].median()),
                "mean_symbolic_recovery": float(group["symbolic_recovery"].mean()),
                "r2_over_0999_rate": float(group["r2_over_0999"].mean()),
                "r2_rank_bootstrap_95ci": _bootstrap_mean_ci(group["r2_rank"]),
                "model_size_rank_bootstrap_95ci": _bootstrap_mean_ci(group["model_size_rank"]),
                "training_time_rank_bootstrap_95ci": _bootstrap_mean_ci(group["training_time_rank"]),
                "energy_rank_bootstrap_95ci": _bootstrap_mean_ci(group["energy_rank"]),
                "recovery_rank_bootstrap_95ci": _bootstrap_mean_ci(group["recovery_rank"]),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["mean_r2_rank"],
            row["mean_model_size_rank"],
        )
    )
    for position, summary in enumerate(summaries, start=1):
        summary["estimated_reference_r2_rank_position"] = position
    for position, summary in enumerate(
        (row for row in summaries if row["eligible_full_scope"]), start=1
    ):
        summary["full_scope_r2_rank_position"] = position

    expected_datasets = set(expected_official_datasets or [])
    expected_seeds = set(int(seed) for seed in (expected_official_seeds or []))
    local_datasets = {str(row.get("dataset")) for row in latest}
    seed_sets = {
        dataset: {
            int(row.get("random_state", row.get("seed")))
            for row in latest
            if str(row.get("dataset")) == dataset
        }
        for dataset in local_datasets
    }
    comparable = bool(expected_datasets and expected_seeds)
    comparable = comparable and local_datasets == expected_datasets
    comparable = comparable and all(seed_sets.get(name) == expected_seeds for name in expected_datasets)
    comparable = comparable and all(bool(row.get("official_protocol")) for row in latest)

    return {
        "schema_version": 1,
        "record_type": "srbench_ranking",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_basis": "local_fixed_config_vs_upstream_tuned_reference",
        "official_leaderboard": False,
        "comparable_to_official": comparable,
        "comparison_scope": {
            "datasets": datasets,
            "dataset_count": len(datasets),
            "local_run_count": len(latest),
            "official_method_count": int(official["algorithm"].nunique()),
        },
        "ranking_method": {
            "trial_aggregation": "median per algorithm and dataset",
            "r2_transform": "max(r2_test, 0)",
            "dataset_rank": "average ties; R2 descending, size/time ascending",
            "global_rank": (
                "mean dataset rank, matching the pinned summary script; "
                "missing datasets are reported and not imputed"
            ),
            "failed_local_runs": "r2=0, error metrics/model size=inf, time=fit limit",
            "reference_implementation": (
                "https://github.com/cavalab/srbench/blob/"
                "dc3f6daa93bf10955df8775256a6f8644f38fd93/"
                "postprocessing/scripts/collate_experiments_results.py"
            ),
        },
        "algorithm_ranking": summaries,
        "dataset_medians": medians.to_dict(orient="records"),
    }


def rank_from_files(
    manifest: Mapping[str, Any],
    result_paths: Sequence[Path],
    output_jsonl: Path,
    ranking_output: Optional[Path] = None,
    expected_datasets: Optional[Sequence[str]] = None,
    expected_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    try:
        official = pd.concat(
            [pd.read_feather(path) for path in result_paths],
            ignore_index=True,
            sort=False,
        )
    except ImportError as exc:
        raise ProtocolError(
            "reading official Feather results requires pyarrow>=12"
        ) from exc
    except Exception as exc:
        raise ProtocolError(f"cannot read official Feather results: {exc}") from exc
    ranking = build_ranking(
        official,
        read_jsonl(output_jsonl),
        expected_official_datasets=(
            list(expected_datasets)
            if expected_datasets is not None
            else [item["name"] for item in manifest["datasets"]]
        ),
        expected_official_seeds=(
            list(expected_seeds)
            if expected_seeds is not None
            else manifest["seeds"]
        ),
    )
    if ranking_output is not None:
        ranking_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = ranking_output.with_suffix(ranking_output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_safe(ranking), indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, ranking_output)
        print(f"ranking={ranking_output.resolve()}")
    return ranking


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["quick", "full", "official", "smoke"],
        default="quick",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runner", default=DEFAULT_RUNNER, help="module:function")
    parser.add_argument("--algorithm", default="WarpSymbolic")
    parser.add_argument(
        "--track",
        choices=["all", "blackbox", "firstprinciples"],
        default="all",
    )
    parser.add_argument("--datasets", help="comma-separated manifest dataset names")
    parser.add_argument("--seeds", help="comma-separated integer random states")
    parser.add_argument("--fit-time-limit-sec", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument(
        "--runner-param",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="repeatable estimator-specific parameter; changes the protocol hash",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="on resume, rerun terminal error records (successful records are always skipped)",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-timeout-sec", type=float, default=120.0)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--prepare-only", action="store_true")
    action_group.add_argument("--dry-run", action="store_true")
    action_group.add_argument("--rank-only", action="store_true")
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--ranking-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)
        manifest_sha256 = _sha256_file(manifest_path)
        dataset_names = _parse_csv(args.datasets)
        seed_strings = _parse_csv(args.seeds)
        seed_values = [int(value) for value in seed_strings] if seed_strings else None
        runner_params = _parse_runner_params(args.runner_param)
        plan = resolve_plan(
            manifest,
            args.profile,
            track=args.track,
            dataset_names=dataset_names,
            seeds=seed_values,
            fit_time_limit_sec=args.fit_time_limit_sec,
            max_train_samples=args.max_train_samples,
            generations=args.generations,
            population_size=args.population_size,
            runner_params=runner_params,
        )
        plan_summary = {
            "profile": plan.profile,
            "official_protocol": plan.official_protocol,
            "override_reasons": list(plan.override_reasons),
            "datasets": [item.name for item in plan.datasets],
            "seeds": list(plan.seeds),
            "tasks": plan.task_count,
            "fit_time_limit_sec": plan.fit_time_limit_sec,
            "max_train_samples": plan.max_train_samples,
            "generations": plan.generations,
            "population_size": plan.population_size,
            "manifest_sha256": manifest_sha256,
        }
        print(json.dumps(plan_summary, indent=2, ensure_ascii=False))
        if args.dry_run:
            return 0

        cache_dir = args.cache_dir.resolve()
        output = args.output.resolve()
        groups = {item.dataset_group for item in plan.datasets}
        if args.rank_only:
            result_paths = ensure_official_results(
                manifest,
                groups,
                cache_dir,
                offline=args.offline,
                timeout_sec=args.download_timeout_sec,
            )
            ranking = rank_from_files(
                manifest,
                result_paths,
                output,
                args.ranking_output,
                expected_datasets=[item.name for item in plan.datasets],
                expected_seeds=plan.seeds,
            )
            print(json.dumps(_json_safe(ranking["algorithm_ranking"]), indent=2))
            return 0

        if args.prepare_only:
            for dataset in plan.datasets:
                path, downloaded = ensure_dataset(
                    manifest,
                    dataset,
                    cache_dir,
                    offline=args.offline,
                    timeout_sec=args.download_timeout_sec,
                )
                # Parsing also catches a valid compressed blob with an unexpected schema.
                read_dataset(path, dataset)
                print(
                    f"prepared={dataset.name} path={path} "
                    f"downloaded={str(downloaded).lower()}"
                )
            if args.rank:
                ensure_official_results(
                    manifest,
                    groups,
                    cache_dir,
                    offline=args.offline,
                    timeout_sec=args.download_timeout_sec,
                )
            return 0

        runner = load_runner(args.runner)
        stats = execute_plan(
            manifest,
            manifest_sha256,
            plan,
            runner,
            args.algorithm,
            cache_dir,
            output,
            resume=args.resume,
            retry_failed=args.retry_failed,
            offline=args.offline,
            download_timeout_sec=args.download_timeout_sec,
        )
        print(f"results={output}")
        print(json.dumps(stats, sort_keys=True))
        if args.rank:
            result_paths = ensure_official_results(
                manifest,
                groups,
                cache_dir,
                offline=args.offline,
                timeout_sec=args.download_timeout_sec,
            )
            ranking = rank_from_files(
                manifest,
                result_paths,
                output,
                args.ranking_output,
                expected_datasets=[item.name for item in plan.datasets],
                expected_seeds=plan.seeds,
            )
            print(json.dumps(_json_safe(ranking["algorithm_ranking"]), indent=2))
        return 1 if stats["failed"] else 0
    except (ProtocolError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
