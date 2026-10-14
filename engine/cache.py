"""Caching of evaluated field arrays, plus the engine's memory budget.

Keys are (field fingerprint, grid fingerprint), so an unchanged sub-expression on the same grid
(e.g. the signed distance of a CAD body) is reused when only a downstream parameter changes.

Two tiers: a small RAM cache (LRU) and an SSD cache behind it. Arrays pushed out of RAM are
written to disk and read back on the next hit (~10 ms for 16 MB, versus 0.2-3 s to recompute a
distance field). File names are content fingerprints, so every worker process can use what
another one wrote, and the cache survives restarts.
"""
import ctypes
import hashlib
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # optional: only needed for live memory checks
    psutil = None

# --- memory budget ----------------------------------------------------------------------------
LIMIT_OF_TOTAL = 0.6      # never plan to use more than this share of all RAM ...
LIMIT_OF_AVAILABLE = 0.75 # ... or of the RAM that was free when the engine started
CACHE_SHARE = 0.2         # part of the limit used by RAM caches; the SSD cache takes the rest
MAIN_SHARE_WITH_WORKERS = 0.3  # of the cache part, while worker processes are running


def physical_memory():
    """Total RAM in bytes (falls back to 16 GB if it cannot be determined)."""
    if psutil is not None:
        return int(psutil.virtual_memory().total)
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 16 * 1024 ** 3


def available_memory():
    """RAM not used by any program right now, in bytes."""
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    return physical_memory() // 2


_LIMIT = None


def memory_limit():
    """Bytes the whole engine (all its processes) should stay under. Fixed at first use, from
    total RAM and what was free then, so other programs keep their share."""
    global _LIMIT
    if _LIMIT is None:
        own = psutil.Process().memory_info().rss if psutil is not None else 0
        _LIMIT = int(min(LIMIT_OF_TOTAL * physical_memory(), LIMIT_OF_AVAILABLE * available_memory() + own))
    return _LIMIT


def ram_cache_budget(workers=0):
    """(main process, each worker) RAM cache sizes in bytes."""
    total = CACHE_SHARE * memory_limit()
    if workers <= 0:
        return int(total), 0
    main = MAIN_SHARE_WITH_WORKERS * total
    return int(main), int((total - main) / workers)


# --- SSD tier -----------------------------------------------------------------------------------
CACHE_VERSION = 1  # bump when distance-field or noise code changes, so old files are ignored


def default_cache_dir():
    base = os.environ.get("DESIGN_ENGINE_CACHE")
    if base:
        return Path(base)
    root = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".cache")
    return Path(root) / "DesignEngine" / "cache"


class DiskCache:
    """Arrays as .npy files in one folder, shared by all processes, oldest deleted first."""

    MIN_BYTES = 1 << 20  # smaller arrays are cheaper to recompute than to keep as files

    def __init__(self, folder=None, max_bytes=None):
        self.folder = Path(folder) if folder else default_cache_dir()
        self.folder.mkdir(parents=True, exist_ok=True)
        if max_bytes is None:
            max_bytes = min(20 * 2**30, 0.3 * shutil.disk_usage(self.folder).free)
        self.max_bytes = int(max_bytes)
        self._written = 0
        self._size = (0.0, 0)  # (time measured, bytes)

    def _path(self, key):
        name = hashlib.sha1(repr((CACHE_VERSION, key)).encode()).hexdigest()[:32]
        return self.folder / f"{name}.npy"

    def save(self, key, array):
        if not isinstance(array, np.ndarray) or array.nbytes < self.MIN_BYTES:
            return
        path = self._path(key)
        if path.exists():
            return
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(tmp, "wb") as f:  # a file object: np.save would append ".npy" to a name
                np.save(f, np.ascontiguousarray(array), allow_pickle=False)
            os.replace(tmp, path)  # atomic: readers never see half a file
        except OSError:
            tmp.unlink(missing_ok=True)
            return
        self._written += array.nbytes
        # every process counts only its own writes: with 8 workers a tenth of the budget each let
        # the folder grow to 25 GB of 20, so trim more often
        if self._written > self.max_bytes / 40:
            self._written = 0
            self.trim()

    def load(self, key):
        path = self._path(key)
        try:
            array = np.load(path, allow_pickle=False)
            os.utime(path)  # mark as recently used
        except (OSError, ValueError, EOFError):
            return None
        array.flags.writeable = False
        return array

    def trim(self):
        """Delete the least recently used files until under 90% of the size limit."""
        try:
            files = [(f.stat().st_mtime, f.stat().st_size, f) for f in self.folder.glob("*.npy")]
        except OSError:
            return
        total = sum(size for _, size, _ in files)
        for _, size, f in sorted(files):
            if total <= 0.9 * self.max_bytes:
                break
            try:
                f.unlink()
                total -= size
            except OSError:
                pass  # in use by another process

    def size(self, max_age=30.0):
        """Bytes in the folder. Measured at most every `max_age` seconds: the editor asks every few
        seconds, and stat-ing thousands of files on a busy disk once tied up the whole server."""
        measured, value = self._size
        if time.time() - measured > max_age:
            try:
                value = sum(f.stat().st_size for f in self.folder.glob("*.npy"))
            except OSError:
                pass
            self._size = (time.time(), value)
        return value

    def clear(self):
        for f in [*self.folder.glob("*.npy"), *(self.folder / STEP_FOLDER).glob("*.npz"),
                  *(self.folder / "shared").glob("*.npy")]:  # engine/shared.py
            try:
                f.unlink()
            except OSError:
                pass


# --- imported STEP files ------------------------------------------------------------------------
# Reading a STEP file and triangulating it takes 15-70 s for the user's parts; the result only
# depends on the file and two settings, so it is kept on the SSD next to the field arrays.
STEP_FOLDER = "step"
STEP_VERSION = 1            # bump when cad_import.tessellate's output changes
STEP_MAX_BYTES = 4 * 2**30  # oldest imports are deleted beyond this


def _step_file(path, tolerance, angle):
    """Cache file name: the file's full path, size and modification time, plus the settings."""
    st = os.stat(path)
    key = repr((STEP_VERSION, os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns,
                float(tolerance), float(angle)))
    return default_cache_dir() / STEP_FOLDER / (hashlib.sha1(key.encode()).hexdigest()[:32] + ".npz")


def load_step_meshes(path, tolerance, angle):
    """The meshes an earlier import of this file (same size and date) with the same settings made,
    or None."""
    import trimesh
    file = _step_file(path, tolerance, angle)
    try:
        with np.load(file, allow_pickle=False) as data:
            parts = [(data[f"v{i}"], data[f"f{i}"], data[f"c{i}"]) for i in range(int(data["count"]))]
        os.utime(file)  # mark as recently used
    except (OSError, KeyError, ValueError, EOFError):
        return None
    meshes = []
    for vertices, faces, cad_face in parts:
        mesh = trimesh.Trimesh(vertices, faces.astype(np.int64), process=False)
        mesh.face_attributes["cad_face"] = cad_face
        meshes.append(mesh)
    return meshes


def save_step_meshes(path, tolerance, angle, meshes):
    file = _step_file(path, tolerance, angle)
    file.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"count": np.array(len(meshes))}
    for i, mesh in enumerate(meshes):
        arrays[f"v{i}"] = np.asarray(mesh.vertices, dtype=np.float64)
        arrays[f"f{i}"] = np.asarray(mesh.faces, dtype=np.int32 if len(mesh.vertices) < 2**31 else np.int64)
        arrays[f"c{i}"] = mesh.face_attributes.get("cad_face", np.zeros(len(mesh.faces), dtype=np.int32))
    tmp = file.with_suffix(f".{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:  # a file object: savez would append ".npz" to the name
            np.savez(f, **arrays)
        os.replace(tmp, file)
    except OSError:
        tmp.unlink(missing_ok=True)
        return
    try:
        files = sorted(((f.stat().st_mtime, f.stat().st_size, f) for f in file.parent.glob("*.npz")), reverse=True)
    except OSError:
        return
    total = 0
    for _, size, f in files:  # newest first: keep until the budget is used up
        total += size
        if total > STEP_MAX_BYTES and f != file:
            f.unlink(missing_ok=True)


# --- RAM tier -----------------------------------------------------------------------------------
class ArrayCache:
    def __init__(self, max_bytes=1e9, disk=None):
        self.max_bytes = max_bytes
        self.disk = disk
        self._items = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = self.misses = self.disk_hits = 0

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
                self.hits += 1
                return value
        if self.disk is not None:
            value = self.disk.load(key)
            if value is not None:
                self.disk_hits += 1
                self.put(key, value, spill=False)
                return value
        self.misses += 1
        return None

    def put(self, key, value, spill=True):
        size = np.asarray(value).nbytes
        if size > self.max_bytes:
            if spill and self.disk is not None:
                self.disk.save(key, value)
            return
        with self._lock:
            if key in self._items:
                return
            self._items[key] = value
            self._bytes += size
            evicted = self._evict()
        self._spill(evicted)

    def set_limit(self, max_bytes):
        """Change the size limit; least recently used arrays move to disk if now over it."""
        with self._lock:
            self.max_bytes = max_bytes
            evicted = self._evict()
        self._spill(evicted)

    def _evict(self):
        evicted = []
        while self._bytes > self.max_bytes and self._items:
            item = self._items.popitem(last=False)
            self._bytes -= np.asarray(item[1]).nbytes
            evicted.append(item)
        return evicted

    def _spill(self, evicted):
        if self.disk is not None:
            for key, value in evicted:  # outside the lock: disk writes are slow
                self.disk.save(key, value)

    @property
    def nbytes(self):
        return self._bytes

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0
