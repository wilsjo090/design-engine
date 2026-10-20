import os
import tempfile

import pytest

# Tests must not fill (or read stale data from) the real SSD cache in %LOCALAPPDATA%.
os.environ["DESIGN_ENGINE_CACHE"] = tempfile.mkdtemp(prefix="design-engine-test-cache-")
# The suite runs on the CPU path; tests/test_gpu.py switches the GPU on explicitly.
os.environ["DESIGN_ENGINE_GPU"] = "0"


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true",
                     help="also run tests/test_gpu.py (uses the graphics card; not while the editor is busy)")


def pytest_collection_modifyitems(config, items):
    # GPU tests only on request: a full test run once started them while the editor was exporting,
    # GPU memory overflowed into RAM, Windows paged to a hard disk and the PC froze.
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="GPU test: run with --gpu")
    for item in items:
        if item.fspath.basename == "test_gpu.py":
            item.add_marker(skip)
