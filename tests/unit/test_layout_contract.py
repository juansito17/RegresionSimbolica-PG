"""Static contracts that keep the repository organized around the GPU core."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_single_root_package_configuration():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "warp-symbolic"' in pyproject
    assert not (ROOT / "AlphaSymbolic" / "pyproject.toml").exists()
    assert not (ROOT / "src" / "AlphaSymbolic" / "pyproject.toml").exists()


def test_gpu_package_does_not_import_experimental_search():
    gpu_root = ROOT / "src" / "warpsymbolic" / "gpu"
    for path in gpu_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("AlphaSymbolic.experimental")
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("AlphaSymbolic.experimental")
                    for alias in node.names
                )


def test_primary_repository_directories_exist():
    for relative in (
        "src/warpsymbolic/api",
        "src/warpsymbolic/symbolic",
        "src/warpsymbolic/gpu/cuda",
        "src/warpsymbolic/cli",
        "src/AlphaSymbolic/ui",
        "src/AlphaSymbolic/experimental",
        "tests/unit",
        "tests/gpu",
        "tests/integration",
        "tests/e2e",
        "research",
        "legacy/cpp_engine",
    ):
        assert (ROOT / relative).is_dir(), relative


def test_core_package_contains_only_production_layers():
    core = ROOT / "src" / "warpsymbolic"
    directories = {path.name for path in core.iterdir() if path.is_dir()}
    assert directories <= {"api", "cli", "gpu", "symbolic", "__pycache__"}
    for optional_name in ("ui", "benchmarking", "data", "experimental"):
        assert not (core / optional_name).exists()
    assert (core / "cli").is_dir()
