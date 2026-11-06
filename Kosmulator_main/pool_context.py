"""Upstream-style per-group argument transport for local/MPI workers."""
from functools import wraps
from pathlib import Path
import os
import pickle
import tempfile

_CACHE = {}
_FILES = set()


class PoolLogProbability:
    def __init__(self, function, path):
        self.function = function
        self.path = path

    def __call__(self, theta):
        if self.path not in _CACHE:
            with open(self.path, "rb") as handle:
                arguments = pickle.load(handle)
            _CACHE.clear()
            _CACHE[self.path] = arguments
        return self.function(theta, *_CACHE[self.path])


def prepare(function, arguments, pool, chain_path):
    if pool is None:
        return function, arguments
    folder = Path(chain_path).resolve().parent if chain_path else Path(tempfile.gettempdir())
    folder.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=".kosm_pool_args_", suffix=".pkl", dir=folder)
    _FILES.add(path)
    with os.fdopen(fd, "wb") as handle:
        pickle.dump(arguments, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return PoolLogProbability(function, path), ()


def cleanup_after_run(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = set(_FILES)
        try:
            return function(*args, **kwargs)
        finally:
            for path in _FILES - previous:
                Path(path).unlink(missing_ok=True)
                _CACHE.pop(path, None)
                _FILES.discard(path)
    return wrapped
