"""Shared arrays read memory-mapped, and arrays prepared once for all processes (engine/shared.py)."""
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from engine import cache, sdf, shared
from engine.cache import DiskCache, is_mapped, read_npy, resident_memory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("DESIGN_ENGINE_CACHE", str(tmp_path))
    shared.forget()
    yield tmp_path / shared.FOLDER
    shared.forget()


def test_big_files_are_mapped_and_small_ones_read(tmp_path):
    big, small = tmp_path / "big.npy", tmp_path / "small.npy"
    np.save(big, np.arange(cache.MAP_BYTES // 8 + 10, dtype=np.float64))
    np.save(small, np.arange(100, dtype=np.float64))
    a, b = read_npy(big), read_npy(small)
    assert type(a) is np.ndarray and is_mapped(a) and not a.flags.writeable
    assert not is_mapped(b) and not b.flags.writeable
    assert a[-1] == cache.MAP_BYTES // 8 + 9 and b[-1] == 99


def test_shared_arrays_are_mapped_where_they_are_read(folder):
    array = np.random.default_rng(0).random((200_000, 3))   # 4.8 MB: goes by its fingerprint
    copy = pickle.loads(pickle.dumps(shared.Shared(array)))
    shared.forget()
    got = copy.get()
    assert is_mapped(got) and np.array_equal(got, array)


def test_big_disk_cache_hits_are_mapped_and_all_are_read_only(tmp_path):
    disk = DiskCache(tmp_path)
    disk.save("big", np.ones((DiskCache.MAP_BYTES // 4,), dtype=np.float32))
    disk.save("small", np.ones((512, 512), dtype=np.float32))  # 1 MB: read, not mapped
    big, small = disk.load("big"), disk.load("small")
    assert is_mapped(big) and not is_mapped(small)
    assert not big.flags.writeable and not small.flags.writeable
    disk.trim()       # a mapped file cannot be deleted on Windows: trimming and clearing skip it
    disk.clear()


def test_prepared_arrays_are_built_once_for_everyone_waiting(folder):
    built, results = [], []

    def build():
        built.append(1)
        time.sleep(0.3)  # the others wait for its files meanwhile
        return {"big": np.arange(400_000, dtype=np.float64), "small": np.arange(5), "count": 3}

    threads = [threading.Thread(target=lambda: results.append(shared.prepared("test", "k", build))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1 and len(results) == 4
    for r in results:
        assert r["count"] == 3 and np.array_equal(r["big"], np.arange(400_000)) and list(r["small"]) == list(range(5))
        assert is_mapped(r["big"]) and not r["big"].flags.writeable
    assert not list(folder.glob("*.lock"))
    shared.prepared("test", "k", build)
    assert len(built) == 1   # found on the SSD


def test_a_lock_left_by_a_stopped_process_does_not_block(folder):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    stem = __import__("hashlib").sha1(repr((shared.PREPARED_VERSION, "test", "k")).encode()).hexdigest()[:32]
    folder.mkdir(parents=True)
    (folder / f"{stem}.lock").write_text(str(gone.pid))
    start = time.time()
    got = shared.prepared("test", "k", lambda: {"a": np.ones(3)})
    assert time.time() - start < 5 and list(got["a"]) == [1, 1, 1]


def test_prepared_arrays_trimmed_away_are_built_again(folder):
    built = []

    def build():
        built.append(1)
        return {"a": np.arange(10), "b": np.arange(20)}

    shared.prepared("test", "k", build)
    next(folder.glob("*.b.npy")).unlink()
    got = shared.prepared("test", "k", build)
    assert len(built) == 2 and list(got["b"]) == list(range(20))


@pytest.fixture(scope="module")
def cad_part():
    from engine import load_step, tessellate
    return tessellate(load_step(ROOT / "samples" / "test_part.step")[0], tolerance=0.01)  # has cracks


def test_mesh_query_from_prepared_arrays_is_the_same(folder, cad_part, monkeypatch):
    monkeypatch.setattr(sdf, "PREPARE_TRIANGLES", 0)
    monkeypatch.setattr(sdf, "_QUERIES", type(sdf._QUERIES)())
    V = shared.Shared(np.ascontiguousarray(cad_part.vertices, dtype=np.float64), key="part")
    F = shared.Shared(np.ascontiguousarray(cad_part.faces, dtype=np.int64), key="part")
    V, F = pickle.loads(pickle.dumps(V)), pickle.loads(pickle.dumps(F))
    query = sdf.mesh_query(V, F, key="part")
    assert list(folder.glob("*.json"))          # prepared on the SSD
    assert query.cracks.count > 0
    rng = np.random.default_rng(1)
    lo, hi = cad_part.bounds
    points = rng.uniform(lo - 2, hi + 2, size=(50_000, 3))
    plain = sdf.MeshQuery(cad_part.vertices, cad_part.faces)
    assert np.array_equal(query.signed(points), plain.signed(points))
    for name in ("FN", "VN", "EN", "EMAP"):   # (degenerate triangles give a few NaN normals)
        assert np.array_equal(getattr(query, name), getattr(plain, name), equal_nan=True)


@pytest.mark.skipif(sys.platform != "win32", reason="the private working set is a Windows measure")
def test_resident_memory_leaves_out_mapped_files(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.ones(64 * 2**20 // 8))   # 64 MB
    before = resident_memory()
    mapped = read_npy(path)
    assert mapped.sum() == 64 * 2**20 // 8    # every page touched
    assert resident_memory() - before < 16 * 2**20
    del mapped
