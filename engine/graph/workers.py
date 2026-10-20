"""Worker processes for running a node over a list of bodies in parallel.

Unlike concurrent.futures' pool, each worker has its own task queue and item i of a list always
goes to worker i % n. Workers keep caches between runs (evaluated field arrays, mesh search
trees), so when you tweak a parameter, each worker finds its own bodies' expensive results
still warm instead of someone else's.
"""
import collections
import gc
import multiprocessing as mp
import os
import queue
import time
import traceback

import cloudpickle

from ..cache import ram_cache_budget

QUEUED_PER_WORKER = 2  # items waiting in each worker's queue (one running, one ready)


def _worker_main(tasks, results, threads, cache_cap, current):
    # libigl starts one thread per core in every worker unless told otherwise; it reads this on
    # first use, so setting it here (in the worker only) leaves the main process at full speed.
    os.environ["IGL_NUM_THREADS"] = str(threads)
    from .core import _current
    from .nodes import ARRAY_CACHE
    last_gc = time.time()
    while True:
        task = tasks.get()
        if task is None:
            return
        token, index, payload = task
        if token != current.value:  # from a run that was cancelled: skip, keep caches warm
            continue
        if ARRAY_CACHE.max_bytes != cache_cap.value:  # the main process may lower it when memory is tight
            ARRAY_CACHE.set_limit(cache_cap.value)
        _current.sink = lines = []  # report_message lines of this item, sent back with its result
        try:
            fn, kwargs = cloudpickle.loads(payload)
            results.put((token, index, True, fn(**kwargs), lines))
        except BaseException as err:  # report, don't die: the pool stays usable
            msg = f"{type(err).__name__}: {err}"
            if not str(err):
                msg += "\n" + traceback.format_exc(limit=3)
            results.put((token, index, False, msg, lines))
        finally:
            fn = kwargs = _current.sink = None
            # Mesh objects hold reference cycles, so their arrays are only freed by the cycle
            # collector. Run it when idle, or every 1.5 s during long batches (~0.1 s each time).
            if tasks.empty() or time.time() - last_gc > 1.5:
                gc.collect()
                last_gc = time.time()


class WorkerError(Exception):
    pass


class StickyPool:
    def __init__(self, n):
        self.n = n
        ctx = mp.get_context("spawn")
        threads = max(1, (os.cpu_count() or 4) // n)
        # Memory budget: while workers exist, the main process keeps a small share of the RAM
        # cache and the workers split the rest (everything else spills to the SSD cache).
        from .nodes import ARRAY_CACHE
        main_bytes, worker_bytes = ram_cache_budget(n)
        ARRAY_CACHE.set_limit(main_bytes)
        self._cache_cap = ctx.Value("q", worker_bytes)  # per worker; lowered when memory is tight
        self._current = ctx.Value("i", 0)  # token of the run workers should be doing
        self._tasks = [ctx.Queue() for _ in range(n)]
        self._results = ctx.Queue()
        self._procs = [ctx.Process(target=_worker_main, daemon=True,
                                   args=(q, self._results, threads, self._cache_cap, self._current))
                       for q in self._tasks]
        for p in self._procs:
            p.start()
        self._token = 0
        self.busy = False  # True while map() is running
        self.gpu = False   # set by the evaluator: whether the workers were started with the GPU on
        self.gpu_contexts = 0  # workers 0..n-1 have used the GPU (each keeps a CUDA context)

    def alive(self):
        return all(p.is_alive() for p in self._procs)

    def pids(self):
        return [p.pid for p in self._procs if p.is_alive()]

    def shrink_caches(self):
        """Halve the workers' RAM caches (applied before their next item)."""
        self._cache_cap.value = max(self._cache_cap.value // 2, 16 << 20)

    def map(self, fn, arg_list, on_progress=None, should_stop=None, active=None, lines=None):
        """Run fn(**args) for each item; returns results in order. Raises WorkerError if an item
        fails. `active` limits how many workers are used (item i goes to worker i % active).
        `should_stop()` is polled while waiting; when it returns True the remaining items are
        abandoned (workers stay alive, with their caches) and None is returned. The (text, level)
        lines the items report (core.report_message) are added to `lines`."""
        self.busy = True
        try:
            return self._map(fn, arg_list, on_progress, should_stop, active, lines)
        finally:
            self.busy = False

    def _map(self, fn, arg_list, on_progress, should_stop, active, lines):
        active = max(1, min(self.n, active or self.n))
        self._token += 1
        token = self._token
        self._current.value = token
        # Items are pickled and queued a few per worker at a time, not all at once: a queued item
        # stays in this process's memory until its worker takes it, and items can be big (a body
        # with millions of triangles travels inside each of its blocks' tasks).
        waiting = [collections.deque(range(w, len(arg_list), active)) for w in range(active)]

        def feed(w):
            if waiting[w]:
                i = waiting[w].popleft()
                self._tasks[w].put((token, i, cloudpickle.dumps((fn, arg_list[i]))))

        for w in range(active):
            for _ in range(QUEUED_PER_WORKER):
                feed(w)
        results, done, failure = [None] * len(arg_list), 0, None
        while done < len(arg_list):
            if should_stop and should_stop():
                self._current.value = -token  # queued items of this run get skipped
                return None
            try:
                got_token, index, ok, value, said = self._results.get(timeout=0.2)
            except queue.Empty:
                if not self.alive():
                    self.terminate()
                    raise WorkerError("a worker process died (out of memory?)")
                continue
            if got_token != token:
                continue  # left over from an earlier, abandoned run
            feed(index % active)
            done += 1
            if lines is not None:
                lines.extend(said)
            if ok:
                results[index] = value
            elif failure is None:
                failure = value
            if on_progress:
                on_progress(done, len(arg_list))
        if failure:
            raise WorkerError(failure)
        return results

    def terminate(self):
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        deadline = time.time() + 2
        for p in self._procs:
            p.join(max(0.0, deadline - time.time()))
        # Items still waiting for a worker sit in a feeder thread that blocks forever once the
        # worker is gone (this process holds the pipe's other end too, so it never breaks): their
        # memory was never freed and the process could not exit. Read them out here.
        for q in self._tasks:
            q.cancel_join_thread()
            stop = time.time() + 5
            while time.time() < stop:
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    if not q._buffer:  # the feeder has nothing left to send
                        break
            q.close()
        from .nodes import ARRAY_CACHE
        ARRAY_CACHE.set_limit(ram_cache_budget()[0])  # no workers: the main process gets it all

    def close(self):
        for q in self._tasks:
            try:
                q.put(None)
            except Exception:
                pass
        deadline = time.time() + 2
        for p in self._procs:
            p.join(max(0.0, deadline - time.time()))
        self.terminate()
