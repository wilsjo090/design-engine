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
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # optional: only needed for live memory checks
    psutil = None

# --- memory budget ----------------------------------------------------------------------------
LIMIT_OF_TOTAL = 0.6      # never plan to use more than this share of all RAM (more, and the engine's own data is paged)
CACHE_SHARE = 0.2         # part of the limit (at most of the RAM free) used by RAM caches; the SSD cache takes the rest
MAIN_SHARE_WITH_WORKERS = 0.3  # of the cache part, while worker processes are running

# Commit: the memory Windows has promised to programs, at most RAM + page file (the commit limit). Free RAM
# is not the limit any more: with the page file on the NVMe SSD, a full RAM only pages other programs' idle
# memory out, and the PC stays responsive (FINDINGS, "The PC freeze"). Running out of commit is what breaks:
# then any program's next allocation fails, SolidWorks' too. (On 2026-10-31 an export reached 77.7 of the
# 77.8 GB with the engine holding 4 GB: other programs committed ~53 GB, its 7 GPU workers 17 GB.)
COMMIT_TIGHT = int(6 * 2**30)   # left for other programs when planning; under it caches are freed
COMMIT_STOP = int(3 * 2**30)    # the run stops
COMMIT_LAST_RESORT = int(1.5 * 2**30)  # after a stop: the process ends if it stays this low (core)


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


class _PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
                ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def commit_room():
    """Bytes the PC can still commit: its commit limit (RAM + page file, as large as the page file is now)
    less what all programs have committed. Elsewhere than on Windows: free RAM plus free swap."""
    if sys.platform == "win32":
        info = _PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return int(info.CommitLimit - info.CommitTotal) * int(info.PageSize)
    if psutil is not None:
        return int(psutil.virtual_memory().available + psutil.swap_memory().free)
    return physical_memory() // 2


def memory_room():
    """Bytes a step may still take as far as the PC goes: the commit room less COMMIT_TIGHT, which is left
    for other programs."""
    return commit_room() - COMMIT_TIGHT


_LIMIT = _CACHES = None


def memory_limit():
    """Bytes the whole engine (all its processes) should stay under in this run: set when each run starts
    (refresh_memory_limit; before the first run, from this process alone)."""
    if _LIMIT is None:
        refresh_memory_limit(psutil.Process().memory_info().rss if psutil is not None else 0)
    return _LIMIT


def refresh_memory_limit(in_use):
    """Set the limit for a run from total RAM and the commit room now (where the run would be stopped
    anyway, COMMIT_STOP); `in_use`: what the engine's processes hold already. Not from the RAM free: that
    stopped a 0.05 mm export at 7.4 GB with 4 GB still free, while Windows could have paged other programs'
    idle memory out. Fixed during the run: recomputed on every check, it would grow as the engine's own use
    took the room and never be reached. (It used to be fixed at start-up: an editor started with 8 GB free
    kept a 6.2 GB limit, and ran blocks 2 at a time, with 10.9 GB free later.)"""
    global _LIMIT, _CACHES
    _LIMIT = int(min(LIMIT_OF_TOTAL * physical_memory(), in_use + max(commit_room() - COMMIT_STOP, 0)))
    # the RAM caches only take RAM that is free: they would page other programs out for arrays the SSD
    # cache holds anyway, and an idle editor keeps them
    _CACHES = int(CACHE_SHARE * min(_LIMIT, in_use + available_memory()))
    return _LIMIT


def ram_cache_budget(workers=0):
    """(main process, each worker) RAM cache sizes in bytes."""
    memory_limit()
    total = _CACHES
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
                  *(self.folder / "shared").glob("*.npy"),  # engine/shared.py
                  *(self.folder / TOPOLOGY_FOLDER).glob("*.npz"), *(self.folder / FRAME_FOLDER).glob("*.npz")]:
            try:
                f.unlink()
            except OSError:
                pass


# --- imported STEP files ------------------------------------------------------------------------
# Reading a STEP file and triangulating it takes 15-70 s for the user's parts; the result only
# depends on the file and two settings, so it is kept on the SSD next to the field arrays.
STEP_FOLDER = "step"
STEP_VERSION = 2            # bump when cad_import.tessellate's output changes (2: UVs, surface types)
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
            parts = [(data[f"v{i}"], data[f"f{i}"], data[f"c{i}"], data[f"u{i}"], data[f"s{i}"])
                     for i in range(int(data["count"]))]
        os.utime(file)  # mark as recently used
    except (OSError, KeyError, ValueError, EOFError):
        return None
    meshes = []
    for vertices, faces, cad_face, cad_uv, surfaces in parts:
        mesh = trimesh.Trimesh(vertices, faces.astype(np.int64), process=False)
        mesh.face_attributes["cad_face"] = cad_face
        mesh.face_attributes["cad_uv"] = cad_uv
        mesh.metadata["cad_surfaces"] = surfaces
        meshes.append(mesh)
    return meshes


def load_step_names(path, tolerance, angle):
    """The bodies' names in the file (one per solid, "" for none) kept beside its meshes, or None. A file of
    their own (`.names.json`), so the imports cached before names were kept stay valid."""
    try:
        names = json.loads(_step_file(path, tolerance, angle).with_suffix(".names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return [str(n) for n in names] if isinstance(names, list) else None


def save_step_names(path, tolerance, angle, names):
    file = _step_file(path, tolerance, angle).with_suffix(".names.json")
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps([str(n) for n in names]), encoding="utf-8")
    except OSError:
        pass


def save_step_meshes(path, tolerance, angle, meshes):
    file = _step_file(path, tolerance, angle)
    file.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"count": np.array(len(meshes))}
    for i, mesh in enumerate(meshes):
        arrays[f"v{i}"] = np.asarray(mesh.vertices, dtype=np.float64)
        arrays[f"f{i}"] = np.asarray(mesh.faces, dtype=np.int32 if len(mesh.vertices) < 2**31 else np.int64)
        arrays[f"c{i}"] = mesh.face_attributes.get("cad_face", np.zeros(len(mesh.faces), dtype=np.int32))
        arrays[f"u{i}"] = mesh.face_attributes.get("cad_uv", np.full((len(mesh.faces), 3, 2), np.nan, np.float32))
        arrays[f"s{i}"] = mesh.metadata.get("cad_surfaces", np.zeros((0, 3)))
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
            f.with_suffix(".names.json").unlink(missing_ok=True)


# --- topology optimization states ---------------------------------------------------------------
# A finished or stopped optimization (minutes of work), and checkpoints of a running one, keyed by a
# fingerprint of all its inputs: previews read them, Optimize continues from them.
TOPOLOGY_FOLDER = "topology"
TOPOLOGY_MAX_BYTES = 2 * 2**30
FRAME_FOLDER = "frame"     # frame optimizations (small: nodes, bars, sizes)
FRAME_MAX_BYTES = 2**28


def _state_file(folder, key):
    return default_cache_dir() / folder / f"{key}.npz"


def _load_state(folder, key):
    file = _state_file(folder, key)
    try:
        with np.load(file, allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        os.utime(file)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):  # (a damaged file: the run is simply not there)
        return None
    return state


def load_topology(key):
    """The saved state {name: array} of the optimization with this key, or None."""
    return _load_state(TOPOLOGY_FOLDER, key)


def save_topology(key, **arrays):
    _save_state(TOPOLOGY_FOLDER, key, arrays, TOPOLOGY_MAX_BYTES)


def load_frame(key):
    """The saved result {name: array} of the frame optimization with this key, or None."""
    return _load_state(FRAME_FOLDER, key)


def save_frame(key, **arrays):
    _save_state(FRAME_FOLDER, key, arrays, FRAME_MAX_BYTES)


def _save_state(folder, key, arrays, max_bytes):
    file = _state_file(folder, key)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, file)  # atomic: a preview never reads half a state
    except OSError:
        tmp.unlink(missing_ok=True)
        return
    try:
        files = sorted(((f.stat().st_mtime, f.stat().st_size, f) for f in file.parent.glob("*.npz")), reverse=True)
    except OSError:
        return
    total = 0
    for _, size, f in files:
        total += size
        if total > max_bytes and f != file:
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
