"""Big read-only arrays that worker processes need again and again: CAD meshes, the Voronoi foam.

A field is pickled into every task sent to a worker, and a field that reads a 2.4M-triangle mesh
carried ~90 MB of vertices and triangles each time (with 2 tasks queued per worker, over 1 GB in
flight). A Shared array pickles as its fingerprint instead: the first pickling writes the array to
a folder next to the SSD cache, and every process maps it from there (cache.read_npy: one copy in RAM
for all of them, the file's own pages) while it is used (most tasks never touch the array at all: the
distance field is cached by then).

`prepared` keeps arrays worked out from such data (a CAD mesh's normals and cracks, its split for the
GPU) in the same folder: built once by the first process that needs them, mapped by every other one.
"""
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict

import numpy as np

from . import trace
from .cache import default_cache_dir, psutil, read_npy

FOLDER = "shared"
INLINE_BYTES = 1 << 20      # smaller arrays travel inside the task as before
KEEP_BYTES = 256 * 2**20    # arrays each process keeps after reading them (oldest dropped first)
MAX_BYTES = 4 * 2**30       # files kept on the SSD; older files are deleted ...
MIN_AGE = 3600.0            # ... unless written or read in the last hour (a run may be using them)
PREPARED_VERSION = 1        # bump when what a `prepared` build returns changes (sdf, sdf_gpu)
PREPARED_WAIT = 600.0       # s a process waits for another one building the same prepared arrays

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


def _write(path, array):
    """Write an .npy file atomically (readers never see half a file); nothing when it is there already
    (same name, same content: it may be mapped by another process, and then cannot be replaced)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:  # a file object: np.save would append ".npy" to a name
            np.save(f, np.ascontiguousarray(array), allow_pickle=False)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        if not path.exists():
            raise


def _store(key, array):
    path = _folder() / f"{key}.npy"
    if path.exists():
        return
    _write(path, array)
    _trim(path.parent)


def _load(key):
    global _kept_bytes
    with _lock:
        if key in _kept:
            _kept.move_to_end(key)
            return _kept[key]
    path = _folder() / f"{key}.npy"
    try:
        with trace.span("shared read", key=key):
            array = read_npy(path)  # mapped: one copy in RAM for every process
        os.utime(path)  # recently used: not trimmed while a run needs it
    except (OSError, ValueError) as e:
        raise RuntimeError(f"shared data {key} is missing from the SSD cache folder ({e}); run again") from None
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


# --- arrays worked out once for all processes ---------------------------------------------------
def prepared(kind, key, build):
    """`build()`'s dict of arrays (and plain numbers) for `key`, worked out once for all processes: the first
    process that needs them builds them and writes them to the shared folder, every process maps them from
    there (read-only). Processes that need them while one builds them wait for its files instead of building
    their own (8 GPU workers each split and analysed a big part for themselves, ~290 B per triangle each at
    the same time: MEMORY_AUDIT C6, C8). The build must give the same result in every process."""
    stem = hashlib.sha1(repr((PREPARED_VERSION, kind, key)).encode()).hexdigest()[:32]
    folder = _folder()
    found = _read_prepared(folder, stem)
    if found is not None:
        return found
    folder.mkdir(parents=True, exist_ok=True)
    lock, deadline = folder / f"{stem}.lock", time.time() + PREPARED_WAIT
    while True:
        if _take(lock):
            try:
                found = _read_prepared(folder, stem)  # written while this one waited for the lock
                if found is None:
                    with trace.span("prepare", kind=kind):
                        built = build()
                    try:
                        _write_prepared(folder, stem, built)
                    except OSError:
                        return built  # a full disk: this process keeps its own
                    found = _read_prepared(folder, stem)
                    if found is None:
                        return built
            finally:
                lock.unlink(missing_ok=True)
            _trim(folder)
            return found
        with trace.span("prepare wait", kind=kind):
            while lock.exists() and not _stale(lock) and time.time() < deadline:
                time.sleep(0.02)
        found = _read_prepared(folder, stem)
        if found is not None:
            return found
        if time.time() >= deadline:
            return build()  # the other one takes too long: build without writing
        if _stale(lock):
            lock.unlink(missing_ok=True)


def _take(lock):
    """Create the lock file; False when another process has it."""
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError:  # e.g. being deleted right now
        return False
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    return True


def _stale(lock):
    """A lock left by a process that is gone (stopped by the watchdog, killed)."""
    try:
        pid = int(lock.read_text() or 0)
        age = time.time() - lock.stat().st_mtime
    except (OSError, ValueError):
        return False  # being written, or gone (the loop sees that)
    if psutil is not None and pid:
        return not psutil.pid_exists(pid)
    return age > PREPARED_WAIT


def _write_prepared(folder, stem, built):
    """Each array as `<stem>.<name>.npy`, then the list `<stem>.json` last: a list on the SSD means its arrays
    are complete."""
    arrays = {name: v for name, v in built.items() if isinstance(v, np.ndarray)}
    values = {name: v for name, v in built.items() if not isinstance(v, np.ndarray)}
    for name, array in arrays.items():
        _write(folder / f"{stem}.{name}.npy", array)
    path = folder / f"{stem}.json"
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"arrays": sorted(arrays), "values": values}))
    os.replace(tmp, path)


def _read_prepared(folder, stem):
    path = folder / f"{stem}.json"
    try:
        listed = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    out = dict(listed.get("values", {}))
    try:
        for name in listed["arrays"]:
            file = folder / f"{stem}.{name}.npy"
            out[name] = read_npy(file)
            os.utime(file)  # recently used: not trimmed while a run needs it
    except (OSError, ValueError, KeyError):  # an array trimmed away: build again
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return out


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
