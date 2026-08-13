"""Shared test configuration for the WarpSymbolic repository layout."""

from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA-capable GPU")
    config.addinivalue_line("markers", "integration: cross-module integration test")
    config.addinivalue_line("markers", "e2e: application or harness end-to-end test")
    config.addinivalue_line("markers", "ui: Gradio UI and callback test")


def pytest_collection_modifyitems(config, items):
    """Mark tests under ``tests/gpu`` without editing every historical test."""

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    for item in items:
        path = Path(str(item.fspath))
        if "gpu" in path.parts:
            item.add_marker(pytest.mark.gpu)
            if not cuda_available:
                item.add_marker(
                    pytest.mark.skip(reason="CUDA GPU is unavailable in this environment")
                )
