import os
from pathlib import Path
import pytest


def _is_torch_test_file(path: Path) -> bool:
    return "pytest.mark.requires_torch" in path.read_text(encoding="utf-8")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    run_torch = os.getenv("RUN_TORCH_TESTS", "1") == "1"
    if run_torch:
        return False
    if collection_path.suffix != ".py":
        return False
    if not collection_path.name.startswith("test_"):
        return False
    return _is_torch_test_file(collection_path)

def pytest_collection_modifyitems(config, items):
    run_torch = os.getenv("RUN_TORCH_TESTS", "1") == "1"
    if run_torch:
        return
    skip_torch = pytest.mark.skip(reason="Torch tests disabled on this CI job")
    for item in items:
        if "requires_torch" in item.keywords:
            item.add_marker(skip_torch)