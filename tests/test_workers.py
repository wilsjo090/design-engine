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
