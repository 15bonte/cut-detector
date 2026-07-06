import os
import pytest

def pytest_collection_modifyitems(config, items):
    run_torch = os.getenv("RUN_TORCH_TESTS", "1") == "1"
    if run_torch:
        return
    skip_torch = pytest.mark.skip(reason="Torch tests disabled on this CI job")
    for item in items:
        if "requires_torch" in item.keywords:
            item.add_marker(skip_torch)