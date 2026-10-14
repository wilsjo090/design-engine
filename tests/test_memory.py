import time

import numpy as np
import pytest
import trimesh

from engine import fields as F
from engine.cache import ArrayCache, DiskCache
from engine.graph import Evaluator, Graph, GraphError
from engine.grid import Grid


def counting(field, calls):
    """Wrap a field so each evaluation is counted."""
    inner = field.fn

    def fn(ctx):
        calls[field.name] = calls.get(field.name, 0) + 1
        return inner(ctx)
    field.fn = fn
    return field


def test_each_step_runs_once_and_intermediates_are_released():
    calls = {}
    x = counting(F.X * 2.0, calls)
    shared = counting(F.cos(x), calls)          # used three times below
    expr = shared * shared + F.sin(shared)
    ctx = F.Context(Grid((0, 0, 0), (4, 4, 4), 0.5))
    values = ctx.eval_full(expr)
    xs = np.arange(9, dtype=np.float32) * 0.5 * 2
    expected = np.cos(xs) ** 2 + np.sin(np.cos(xs))
    assert np.allclose(values[:, 0, 0], expected, atol=1e-6)
    assert all(n == 1 for n in calls.values())  # nothing computed twice
    assert ctx._memo == {} and ctx._uses == {}  # nothing held after the evaluation


def test_only_expensive_fields_go_into_the_shared_cache():
    cache = ArrayCache(1e9)
    box = trimesh.creation.box((4, 4, 4))
    expr = F.offset(F.body_from_mesh(box, 1.0), F.noise(5, 1, 0.5, 1) * 0.2)
    F.Context(Grid.from_bounds(box.bounds, 0.25, 1.0), cache).eval_full(expr)
    assert len(cache._items) == 2  # the distance field and the noise, not the math in between


def test_ram_cache_spills_to_disk_and_reads_back(tmp_path):
    disk = DiskCache(tmp_path, max_bytes=1e9)
    cache = ArrayCache(max_bytes=5e6, disk=disk)       # room for one 4 MB array in RAM
    a, b = (np.full((1024, 1024), v, np.float32) for v in (1.0, 2.0))
    cache.put("a", a)
    cache.put("b", b)                                   # pushes "a" out of RAM -> SSD
    assert "a" not in cache._items and len(list(tmp_path.glob("*.npy"))) == 1
    back = cache.get("a")
    assert back is not None and np.array_equal(back, a) and cache.disk_hits == 1
    assert not back.flags.writeable


def test_disk_cache_stays_under_its_size(tmp_path):
    disk = DiskCache(tmp_path, max_bytes=10e6)
    for i in range(6):
        disk.save(("k", i), np.full((1024, 1024), i, np.float32))  # 4 MB each
    disk.trim()
    assert disk.size() <= 10e6
    assert disk.load(("k", 5)) is not None                         # newest kept


def test_too_large_job_stops_before_it_starts(monkeypatch):
    # (bigger grids than this are meshed block by block, which needs little memory)
    import engine.graph.nodes as nodes
    monkeypatch.setattr(nodes, "available_memory", lambda: 2**30)       # pretend: 1 GB free
    box = trimesh.creation.box((25, 25, 25))                            # 0.1 mm grid: ~21M voxels, ~1 GB
    with pytest.raises(GraphError, match="needs about .* GB of memory"):
        nodes._mesh(F.body_from_mesh(box, 1.0), box.bounds, 0.1, 1.0, 1.0)


def test_big_grid_is_meshed_in_blocks_with_little_memory(monkeypatch):
    import engine.graph.nodes as nodes
    monkeypatch.setattr(nodes, "available_memory", lambda: 2**30)       # tiled path ignores it
    box = trimesh.creation.box((40, 40, 40))                            # 0.1 mm grid: ~77M voxels
    mesh = nodes._mesh(F.body_from_mesh(box, 1.0), box.bounds, 0.1, 1.0, 1.0)
    assert mesh.is_watertight and mesh.volume == pytest.approx(40 ** 3, rel=1e-3)


def test_parallel_slots_follow_the_memory_estimate(monkeypatch):
    import engine.graph.core as core
    from engine.graph.core import REGISTRY
    ev = Evaluator(Graph(), workers=8, log=None)
    monkeypatch.setattr(core, "memory_limit", lambda: 10 * 2**30)
    monkeypatch.setattr(core, "available_memory", lambda: 64 * 2**30)
    monkeypatch.setattr(ev, "memory_in_use", lambda: 2 * 2**30)        # 8 GB of room
    ntype = REGISTRY["mesh.from_implicit"]
    box = trimesh.creation.box((10, 10, 10))
    small = ntype.estimate(body=F.body_from_mesh(box, 1.0), voxel=0.4, padding=1.0, domain=None)
    big = ntype.estimate(body=F.body_from_mesh(box, 1.0), voxel=0.04, padding=1.0, domain=None)
    assert ev._parallel_slots(small) == 8                               # tiny grids: all workers
    assert 1 < ev._parallel_slots(big) == int(8 * 2**30 // big) < 8     # only as many as fit
    assert ev._parallel_slots(64 * 2**30) == 1                          # never fewer than one


def test_cache_size_is_remembered(tmp_path):
    from engine.cache import DiskCache
    disk = DiskCache(tmp_path, max_bytes=2**30)
    disk.save(("a",), np.ones(2**18, dtype=np.float32))       # 1 MB
    first = disk.size()
    disk.save(("b",), np.ones(2**18, dtype=np.float32))
    assert disk.size() == first                               # not re-counted within 30 s
    assert disk.size(max_age=0) > first


def test_release_memory_drops_meshes_but_keeps_imports(tmp_path):
    from pathlib import Path
    from engine.graph import link
    g = Graph()
    g.add("import", "input.import_step", path=str(Path(__file__).resolve().parents[1] / "samples" / "test_part.step"))
    g.add("sdf", "implicit.from_cad", cad=link("import.bodies"), band=1.0)
    g.add("mesh", "mesh.from_implicit", body=link("sdf.out"), voxel=0.8, padding=1.0)
    ev = Evaluator(g, workers=1, log=None)
    ev.run()
    ev.release_memory()
    assert "mesh" not in ev._cache and "import" in ev._cache
    ev.run()
    assert ev.computed == ["mesh"]  # only the dropped mesh is rebuilt


def test_idle_editor_gives_memory_back_when_the_pc_runs_low(monkeypatch):
    import threading
    import engine.server as server
    session = server.Session(server.Hub())
    calls = []
    monkeypatch.setattr(session.evaluator, "release_memory", lambda: calls.append(1))
    monkeypatch.setattr(server, "available_memory", lambda: 2**30)   # 1 GB free: low
    stop = threading.Event()
    watcher = threading.Thread(target=session.watch_idle_memory, args=(stop, 0.02), daemon=True)
    watcher.start()
    try:
        time.sleep(0.3)
    finally:
        stop.set()
        watcher.join(5)
    assert calls == [1]  # released once, not over and over while memory stays low


def test_big_arrays_travel_to_workers_once(tmp_path, monkeypatch):
    # a field reading a big mesh is pickled into every task: the mesh goes as a fingerprint (the
    # array is written to the SSD once) and the unpickled field reads it back and gives the same
    import cloudpickle
    from engine import shared
    monkeypatch.setenv("DESIGN_ENGINE_CACHE", str(tmp_path))
    mesh = trimesh.creation.icosphere(subdivisions=7, radius=10)      # 330k triangles, ~12 MB
    body = F.body_from_mesh(mesh, 1.0)
    data = cloudpickle.dumps(body)
    assert len(data) < 100_000
    assert len(list((tmp_path / shared.FOLDER).glob("*.npy"))) == 2  # vertices and triangles
    shared.forget()
    from engine import sdf
    sdf._QUERIES.clear()  # as in a fresh worker: rebuild from the arrays read back
    grid = Grid((-11, -11, -11), (11, 11, 11), 1.0)
    copy = cloudpickle.loads(data)
    assert np.array_equal(F.Context(grid).eval_full(copy), F.Context(grid).eval_full(body))
    assert len(cloudpickle.dumps(body)) == len(data)                  # written once, then reused
