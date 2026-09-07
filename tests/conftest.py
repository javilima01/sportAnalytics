import os
import tempfile

# Keep third-party caches isolated, including during test collection.
_cache = tempfile.TemporaryDirectory(prefix="tagging-tests-")
os.environ["YOLO_CONFIG_DIR"] = _cache.name
os.environ["MPLCONFIGDIR"] = _cache.name
os.environ["MPLBACKEND"] = "Agg"
os.environ["YOLO_AUTOINSTALL"] = "false"


def pytest_unconfigure(config):
    _cache.cleanup()
