"""Collection rules for the versioned UI test suite.

The repository has legacy/debug tests on some local machines that are intentionally
not tracked. Keep default collection focused on the committed UI/E2E tests.
"""

from pathlib import Path


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    parts = set(path.parts)
    if "ui" in parts or "e2e" in parts:
        return False
    if path.suffix == ".py" and "tests" in parts:
        return True
    return False
