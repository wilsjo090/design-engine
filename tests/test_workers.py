import os
import sys
import time
from pathlib import Path

import cloudpickle
import pytest

from engine.graph.workers import StickyPool, WorkerError

# Worker processes cannot import this test module, so ship its functions by value.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def whoami(i):
    return i, os.getpid()


def fail_on_three(i):
    if i == 3:
        raise ValueError("bad item 3")
    return i


def slow(i):
    time.sleep(5)
    return i


def chatty(i):
    from engine.graph.core import report_message
    report_message(f"item {i % 2}", "warning")
    return i


def commit(i):
    import psutil
    return os.environ.get("OPENBLAS_NUM_THREADS"), psutil.Process().memory_info().private


# --- two stages (engine/stages.py) ---------------------------------------------------------------------
def first_stage(i, delay=0.0, fail=None, folder=None, say=False):
    """A GPU worker's part: hands the rest to a CPU worker (in a map that splits)."""
    from engine.graph.core import report_message
    from engine.stages import cpu_stage
    if fail == "first" and i == 1:
        raise ValueError("first stage of item 1 failed")
    if say:
        report_message(f"first {i}", "warning")
    return cpu_stage(second_stage, i=i, first=os.getpid(), delay=delay, fail=fail, folder=folder, say=say)


def second_stage(i, first, delay, fail, folder, say):
    import sys
    from engine.stages import report  # (stage functions report through engine.stages, not the graph)
    if folder:
        (Path(folder) / f"started {i}").touch()
    time.sleep(delay)
    if fail == "second" and i == 3:
        raise ValueError("second stage of item 3 failed")
    if say:
        report(f"second {i}", "warning")
    return dict(i=i, first=first, second=os.getpid(), gpu=os.environ.get("DESIGN_ENGINE_GPU"),
                blas=os.environ.get("OPENBLAS_NUM_THREADS"), igl=os.environ.get("IGL_NUM_THREADS"),
                cuda="cupy" in sys.modules or "warp" in sys.modules,
                heavy=[m for m in ("OCP", "igl", "engine.graph", "engine.fields", "engine.sdf") if m in sys.modules])


def big_first_stage(i):
    import numpy as np
    from engine.stages import cpu_stage
    rng = np.random.default_rng(i)
    return cpu_stage(big_second_stage, verts=rng.random((40_000, 3)), faces=rng.integers(0, 40_000, (80_000, 3)),
                     fortran=np.asfortranarray(rng.random((300, 200))), small=np.arange(5), plain=[i, "x"])


def big_second_stage(verts, faces, fortran, small, plain):
    import numpy as np
    out = (verts.copy(), faces.copy(), fortran.copy(), small.copy(), plain)
    verts += 1.0  # (handed-over arrays are this process's own, writable)
    return out + (verts.flags.writeable and faces.flags.writeable and fortran.flags.f_contiguous,
                  np.shares_memory(verts, faces))


def no_split(i):
    from engine.stages import cpu_stage, splitting
    return cpu_stage(lambda: (i, splitting()))


def test_split_hands_the_cpu_stage_to_cpu_workers(pool, monkeypatch):
    monkeypatch.setenv("DESIGN_ENGINE_SPLIT", "1")   # (every stage handed over, also while the CPU queue is full)
    got = []
    out = pool.map(first_stage, [{"i": i} for i in range(6)], split=True, cpu_workers=2,
                   on_item=lambda i, r: got.append((i, r["i"])))
    assert [r["i"] for r in out] == list(range(6))
    assert sorted(got) == [(i, i) for i in range(6)]                 # each item once, as it came back
    gpu_side, cpu_side = {r["first"] for r in out}, {r["second"] for r in out}
    assert not gpu_side & cpu_side and pool.cpu_started() == 2
    assert gpu_side | cpu_side <= set(pool.pids())                   # (counted in the engine's memory)
    # CPU workers: no CUDA, one BLAS and libigl thread (each BLAS thread commits ~23 MB), and lean: not the
    # node library, OpenCascade or libigl (0.09 GB instead of 0.3 GB each)
    assert all(r["gpu"] == "0" and r["blas"] == "1" and r["igl"] == "1" and not r["cuda"] for r in out)
    assert all(r["heavy"] == [] for r in out)


def test_big_stages_go_through_shared_memory_exactly(pool):
    # arrays over SHARED_MIN go in a shared memory block, not the pipe: the same bits, writable, order kept
    import numpy as np
    from engine import stages
    out = pool.map(big_first_stage, [{"i": i} for i in range(4)], split=True, cpu_workers=2)
    for i, (verts, faces, fortran, small, plain, ok, shared) in enumerate(out):
        rng = np.random.default_rng(i)
        assert np.array_equal(verts, rng.random((40_000, 3))) and verts.nbytes > stages.SHARED_MIN
        assert np.array_equal(faces, rng.integers(0, 40_000, (80_000, 3)))
        assert np.array_equal(fortran, rng.random((300, 200)))
        assert np.array_equal(small, np.arange(5)) and plain == [i, "x"] and ok and not shared
    out = pool.map(big_first_stage, [{"i": i} for i in range(4)], split=True, cpu_workers=2)   # (blocks closed: again)
    assert len(out) == 4


def test_no_cpu_workers_without_a_split(pool):
    assert pool.map(no_split, [{"i": i} for i in range(3)]) == [(i, False) for i in range(3)]
    assert pool.cpu_started() == 0
    pool.map(whoami, [{"i": i} for i in range(3)], split=True, cpu_workers=4)   # (nothing handed over)
    assert pool.cpu_started() == 0


def test_split_results_keep_their_order(pool):
    # later items' CPU stages finish first: results still in item order, every item to on_item once
    got = []
    out = pool.map(first_stage, [{"i": i, "delay": 0.1 * (8 - i)} for i in range(8)], split=True, cpu_workers=4,
                   on_item=lambda i, r: got.append(i))
    assert [r["i"] for r in out] == list(range(8)) and sorted(got) == list(range(8))
    assert got != sorted(got)


def test_split_reports_lines_of_both_stages(pool):
    lines = []
    pool.map(first_stage, [{"i": i, "say": True} for i in range(3)], split=True, cpu_workers=2, lines=lines)
    assert sorted(lines) == sorted([(f"first {i}", "warning") for i in range(3)] +
                                   [(f"second {i}", "warning") for i in range(3)])


@pytest.mark.parametrize("stage", ["first", "second"])
def test_split_errors_of_either_stage(pool, stage):
    with pytest.raises(WorkerError, match=f"{stage} stage of item"):
        pool.map(first_stage, [{"i": i, "fail": stage} for i in range(6)], split=True, cpu_workers=2)
    assert [r["i"] for r in pool.map(first_stage, [{"i": i} for i in range(2)], split=True, cpu_workers=2)] == [0, 1]


def test_stop_skips_queued_cpu_stages(pool, tmp_path, monkeypatch):
    # one CPU worker, slow CPU stages: stopped once the first has started, the queued ones never run
    monkeypatch.setenv("DESIGN_ENGINE_SPLIT", "1")   # (every stage handed over, also while the CPU queue is full)
    started = lambda: any(tmp_path.glob("started *"))
    items = [{"i": i, "delay": 3.0, "folder": str(tmp_path)} for i in range(6)]
    assert pool.map(first_stage, items, split=True, cpu_workers=1, should_stop=started) is None
    t = time.perf_counter()
    out = pool.map(first_stage, [{"i": i} for i in range(2)], split=True, cpu_workers=1)
    assert [r["i"] for r in out] == [0, 1]
    assert time.perf_counter() - t < 8                               # waited for the one running stage only
    assert len(list(tmp_path.glob("started *"))) == 1


def test_gpu_workers_do_stages_themselves_while_the_cpu_queue_is_full(pool, tmp_path):
    # one CPU worker, slow CPU stages: a GPU worker hands a stage over only while none waits in the CPU queue,
    # else it runs it itself (in the editor 2 CPU workers were 96% busy while 8 GPU workers idled 80%)
    items = [{"i": i, "delay": 1.0, "folder": str(tmp_path)} for i in range(9)]
    t = time.perf_counter()
    out = pool.map(first_stage, items, split=True, cpu_workers=1)
    took = time.perf_counter() - t
    assert [r["i"] for r in out] == list(range(9))
    here = [r for r in out if r["second"] == r["first"]]       # CPU stage run by the GPU worker itself
    away = [r for r in out if r["second"] != r["first"]]
    assert here and away and pool.cpu_started() == 1
    assert took < 9 * 1.0                                        # (one CPU worker alone: at least 9 s)


def test_terminate_stops_the_cpu_workers_too():
    pool = StickyPool(2)
    try:
        pool.map(first_stage, [{"i": i} for i in range(4)], split=True, cpu_workers=2)
        procs = pool.started()
        assert len(procs) == 4
    finally:
        pool.terminate()
    assert not any(p.is_alive() for p in procs)


@pytest.fixture
def pool():
    p = StickyPool(3)
    yield p
    p.close()


def test_memory_budget_is_shared():
    from engine.cache import memory_limit, ram_cache_budget
    from engine.graph.nodes import ARRAY_CACHE
    alone, _ = ram_cache_budget()
    main, each = ram_cache_budget(2)
    assert main + 2 * each <= alone + 1 < memory_limit()   # caches are a part of the limit
    assert ARRAY_CACHE.max_bytes == alone
    p = StickyPool(2)
    try:
        assert ARRAY_CACHE.max_bytes == main
        p.shrink_caches()
        assert p._cache_cap.value == each // 2
    finally:
        p.close()
    assert ARRAY_CACHE.max_bytes == alone


def test_results_in_order_and_sticky(pool):
    first = pool.map(whoami, [{"i": i} for i in range(9)])
    second = pool.map(whoami, [{"i": i} for i in range(9)])
    assert [r[0] for r in first] == list(range(9))
    assert first == second                       # item i ran in the same process both times
    assert len({pid for _, pid in first}) == 3   # and the work was spread over all workers


def test_workers_start_only_when_given_work(pool):
    assert pool.pids() == []
    one = pool.map(whoami, [{"i": i} for i in range(4)], active=1)   # e.g. limited by free GPU memory
    assert len(pool.pids()) == 1 and {pid for _, pid in one} == set(pool.pids())
    assert pool.map(whoami, [{"i": 0}, {"i": 1}])[0] == one[0]       # worker 0 kept; 1 joins
    assert len(pool.pids()) == 2 and pool.alive()


@pytest.mark.skipif(sys.platform != "win32", reason="private bytes are a Windows measure")
def test_workers_load_blas_with_their_share_of_the_cores():
    # numpy's and scipy's OpenBLAS commit ~23 MB per thread when they load, one thread per core: a
    # worker committed 1.5 GB more than it ever used, and 7 GPU workers took 17 GB of the PC's commit
    before = os.environ.get("OPENBLAS_NUM_THREADS")
    pool = StickyPool(8)
    try:
        (threads, private), = pool.map(commit, [{"i": 0}])   # (only worker 0 starts)
    finally:
        pool.close()
    assert threads == str(max(1, (os.cpu_count() or 4) // 8))
    assert private < 0.6 * 2**30                              # engine imports included (1.6 GB on 32 cores before)
    assert os.environ.get("OPENBLAS_NUM_THREADS") == before   # this process's own setting is left as it was


def test_report_lines_come_back_from_the_workers(pool):
    lines = []
    assert pool.map(chatty, [{"i": i} for i in range(4)], lines=lines) == [0, 1, 2, 3]
    assert sorted(lines) == [("item 0", "warning")] * 2 + [("item 1", "warning")] * 2


def test_errors_are_reported_and_pool_survives(pool):
    with pytest.raises(WorkerError, match="bad item 3"):
        pool.map(fail_on_three, [{"i": i} for i in range(6)])
    assert pool.map(fail_on_three, [{"i": 1}]) == [1]


def test_stop_abandons_queued_work_but_keeps_workers(pool):
    first = pool.map(whoami, [{"i": i} for i in range(3)])
    t = time.perf_counter()
    stop_at = t + 0.5
    # 9 slow items = 3 per worker; stopping should skip the queued ones
    assert pool.map(slow, [{"i": i} for i in range(9)], should_stop=lambda: time.perf_counter() > stop_at) is None
    assert time.perf_counter() - t < 2
    assert pool.alive()
    t = time.perf_counter()
    assert pool.map(whoami, [{"i": i} for i in range(3)]) == first   # same processes, warm
    assert time.perf_counter() - t < 8   # only waits for the one running slow item per worker
