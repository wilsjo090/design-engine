import os
import sys
import time

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
