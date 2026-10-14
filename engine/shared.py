"""Big read-only arrays that worker processes need again and again: CAD meshes, the Voronoi foam.

A field is pickled into every task sent to a worker, and a field that reads a 2.4M-triangle mesh
carried ~90 MB of vertices and triangles each time (with 2 tasks queued per worker, over 1 GB in
flight). A Shared array pickles as its fingerprint instead: the first pickling writes the array to
a folder next to the SSD cache, and every process reads it from there once and keeps it while it
is used (most tasks never touch the array at all: the distance field is cached by then).
"""
import hashlib
import os
import threading
import time
from collections import OrderedDict

import numpy as np

from .cache import default_cache_dir

FOLDER = "shared"
INLINE_BYTES = 1 << 20      # smaller arrays travel inside the task as before
KEEP_BYTES = 256 * 2**20    # arrays each process keeps after reading them (oldest dropped first)
MAX_BYTES = 4 * 2**30       # files kept on the SSD; older files are deleted ...
MIN_AGE = 3600.0            # ... unless written or read in the last hour (a run may be using them)

_kept = OrderedDict()       # key -> array, this process
_kept_bytes = 0
_lock = threading.Lock()


def _folder():
    return default_cache_dir() / FOLDER


class Shared:
    """A read-only array that pickles as a fingerprint. `get()` returns the array."""

    __slots__ = ("key", "_array")

    def __init__(self, array, key=None):
        array = np.ascontiguousarray(array)
        if key is None:
            h = hashlib.sha1(repr((array.dtype.str, array.shape)).encode())
            h.update(memoryview(array).cast("B"))
            key = h.hexdigest()[:32]
        self.key = f"{key}-{array.dtype.str.strip('<>|=')}-{'x'.join(map(str, array.shape))}"
        self._array = array

    def get(self):
        if self._array is None:
            self._array = _load(self.key)
        return self._array

    def __reduce__(self):
        array = self.get()
        if array.nbytes < INLINE_BYTES:
            return _inline, (self.key, array)
        _store(self.key, array)
        return _attach, (self.key,)


def _inline(key, array):
    s = Shared.__new__(Shared)
    s.key, s._array = key, array
    return s


def _attach(key):
    s = Shared.__new__(Shared)
    s.key, s._array = key, None  # read on first use
    return s


def _store(key, array):
    path = _folder() / f"{key}.npy"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:  # a file object: np.save would append ".npy" to a name
            np.save(f, array, allow_pickle=False)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        if not path.exists():
            raise
    _trim(path.parent)


def _load(key):
    global _kept_bytes
    with _lock:
        if key in _kept:
            _kept.move_to_end(key)
            return _kept[key]
    path = _folder() / f"{key}.npy"
    try:
        array = np.load(path, allow_pickle=False)
        os.utime(path)  # recently used: not trimmed while a run needs it
    except (OSError, ValueError) as e:
        raise RuntimeError(f"shared data {key} is missing from the SSD cache folder ({e}); run again") from None
    array.flags.writeable = False
    with _lock:
        _kept[key] = array
        _kept_bytes += array.nbytes
        while _kept_bytes > KEEP_BYTES and len(_kept) > 1:
            _, old = _kept.popitem(last=False)
            _kept_bytes -= old.nbytes
    return array


def forget():
    """Drop the arrays this process keeps (read again from the SSD when needed)."""
    global _kept_bytes
    with _lock:
        _kept.clear()
        _kept_bytes = 0


def _trim(folder):
    try:
        files = sorted(((f.stat().st_mtime, f.stat().st_size, f) for f in folder.glob("*.npy")), reverse=True)
    except OSError:
        return
    total, now = 0, time.time()
    for mtime, size, f in files:  # newest first
        total += size
        if total > MAX_BYTES and now - mtime > MIN_AGE:
            try:
                f.unlink()
            except OSError:
                pass  # open in another process
