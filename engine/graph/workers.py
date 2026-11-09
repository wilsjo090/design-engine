"""Worker processes for running a node over a list of bodies in parallel.

Unlike concurrent.futures' pool, each worker has its own task queue and item i of a list always
goes to worker i % n. Workers keep caches between runs (evaluated field arrays, mesh search
trees), so when you tweak a parameter, each worker finds its own bodies' expensive results
still warm instead of someone else's.

Two stages (engine/stages.py): in a map with split=True, an item whose function returns a CpuStage (a
body's decimation, a block's reduction, dual contouring's extraction) has that stage run by a CPU worker
(stages.cpu_worker_main): a lean process of its own without CUDA (DESIGN_ENGINE_GPU=0, one BLAS and libigl
thread, only meshing's modules loaded), started when the first item is handed over and kept like the
others. The GPU worker puts the stage straight on the CPU workers' shared queue, its arrays in a shared
memory block (stages.HandOver; no trip through the main process), tells the main process it is free for its
next item, and any idle CPU worker takes it; the CPU worker's result is the item's result, with the report
lines of both stages. CPU workers keep no caches, so they need no stickiness.
"""
import collections
import gc
import multiprocessing as mp
import os
import queue
import time

import cloudpickle

from .. import stages, trace
from ..cache import ram_cache_budget
from ..stages import CPU_DONE, CPU_FAILED

QUEUED_PER_WORKER = 2  # items waiting in each worker's queue (one running, one ready)

# result kinds (the third field of what the workers put on the results queue; CPU workers': stages)
DONE, FAILED = "done", "failed"  # an item finished in its (GPU) worker
HANDED = "handed"                # its GPU stage is done, its CPU stage is on the CPU queue


def _freeze_imports():
    """Import what the worker's tasks use, then gc.freeze(): the modules' ~350k objects never become
    garbage, and every gc.collect() between items walked them all (the lamp's Final: ~150 ms a collect,
    10.9 s of worker time). Only at start, before any task: an object frozen while alive is never collected
    as part of a reference cycle, so freezing later (caches holding trimesh meshes) could leak them.
    CuPy and Warp only when this pool may use the GPU (importing CuPy adds ~0.3 s and starts no CUDA)."""
    try:
        import igl, pyfqmr, scipy.ndimage, skimage.measure, trimesh  # noqa: F401,E401
        if os.environ.get("DESIGN_ENGINE_GPU") != "0":
            import cupy, warp  # noqa: F401,E401
    except Exception:  # (a missing optional module only means less is frozen)
        pass
    gc.collect()
    gc.freeze()


def _worker_main(tasks, results, threads, cache_cap, current, number=None, cpu_tasks=None, released=None):
    # libigl starts one thread per core in every worker unless told otherwise; it reads this on
    # first use, so setting it here (in the worker only) leaves the main process at full speed.
    os.environ["IGL_NUM_THREADS"] = str(threads)
    from .nodes import ARRAY_CACHE
    _freeze_imports()
    trace.set_role(f"worker {number}")
    if cpu_tasks is not None:  # (a stage still in this process's feeder when it ends is not waited for)
        cpu_tasks.cancel_join_thread()
    hand = stages.HandOver(released, number)
    last_gc = time.time()
    while True:
        try:  # (while blocks are handed over, look every 0.5 s whether they can be closed)
            task = tasks.get(timeout=0.5 if hand.held else None)
        except queue.Empty:
            hand.drain()
            continue
        hand.drain()
        if task is None:
            hand.close()
            return
        token, index, payload, split, meta = task  # (meta: the trace's run, only while one is traced)
        if token != current.value:  # from a run that was cancelled: skip, keep caches warm
            continue
        if ARRAY_CACHE.max_bytes != cache_cap.value:  # the main process may lower it when memory is tight
            ARRAY_CACHE.set_limit(cache_cap.value)
        stages.collect(lines := [])  # report_message lines of this item, sent back with its result
        node = trace.worker_task(meta) if meta or trace.ACTIVE else None
        handed = fn = kwargs = value = None
        try:
            with trace.span("task", node=node, item=index):
                with trace.span("load args"):
                    fn, kwargs = cloudpickle.loads(payload)
                with stages.allow(split and cpu_tasks is not None):
                    value = fn(**kwargs)
                if isinstance(value, stages.CpuStage) and _cpu_queue_full(cpu_tasks, split):
                    # every CPU worker has a stage waiting already: do this one here instead of waiting for
                    # them (lamp in the editor: 2 CPU workers 96% busy while 8 GPU workers idled 80%, 21.9 s)
                    with trace.span("cpu stage here"):
                        value = value.run()
                if isinstance(value, stages.CpuStage):
                    # pickled here, where an error is reported (in the queue's feeder thread it would be
                    # lost, and the item never done); the CPU worker gets this item's lines so far with it
                    with trace.span("hand over") as span:
                        stage, shared, handed = hand.pack(value)
                        value = None
                        cpu_tasks.put((token, index, stage, shared, lines, meta))
                        span.set(bytes=handed, shared=shared is not None)
                    stage = None
            if handed is not None:
                results.put((token, index, HANDED, handed, []))
            else:
                results.put((token, index, DONE, value, lines))
        except BaseException as err:  # report, don't die: the pool stays usable
            results.put((token, index, FAILED, stages.failure(err), lines))
        finally:
            fn = kwargs = value = None
            stages.collect(None)
            # Mesh objects hold reference cycles, so their arrays are only freed by the cycle
            # collector. Run it when idle, or every 1.5 s during long batches (~0.1 s each time).
            if tasks.empty() or time.time() - last_gc > 1.5:
                with trace.span("gc", node=node, item=index):
                    gc.collect()
                last_gc = time.time()


def _cpu_queue_full(cpu_tasks, cpu_workers):
    """Whether as many CPU stages wait in the queue as there are CPU workers (`cpu_workers`: the task's split, the
    map's number of CPU workers), so a stage handed over now would wait while this GPU worker idles.
    DESIGN_ENGINE_SPLIT=1 (always split: tests) also always hands over."""
    if os.environ.get("DESIGN_ENGINE_SPLIT") == "1":
        return False
    try:
        return cpu_tasks.qsize() >= max(1, int(cpu_workers))
    except (NotImplementedError, OSError):  # (qsize is not available everywhere: then always hand over)
        return False


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
        # CPU stages, put here by the GPU workers themselves (a queue can only be given to a process when
        # it starts, so it exists from the start; it costs nothing while no map splits)
        self._cpu_tasks = ctx.Queue()
        self._released = [ctx.Queue() for _ in range(n)]  # per GPU worker: its shared blocks a CPU worker copied
        # each worker starts when it is first given work: runs limited to fewer (by free GPU memory, or
        # memory) left the rest idle, ~0.3 GB each (the lamp: 4 of 8, 1.2 GB)
        self._procs = [None] * n
        self._new_worker = lambda w: ctx.Process(target=_worker_main, daemon=True, args=(
            self._tasks[w], self._results, threads, self._cache_cap, self._current, w, self._cpu_tasks,
            self._released[w]))
        self._cpu_procs = []  # CPU workers, started when a map hands over its first CPU stage
        self._new_cpu_worker = lambda w: ctx.Process(target=stages.cpu_worker_main, daemon=True, args=(
            self._cpu_tasks, self._results, self._current, w, self._released))
        self._cpu_wanted = 0  # how many CPU workers the running map may use
        self._threads = threads
        self._token = 0
        self.busy = False  # True while map() is running
        self.gpu = False   # set by the evaluator: whether the workers were started with the GPU on
        self.gpu_contexts = 0  # workers 0..n-1 have used the GPU (each keeps a CUDA context)

    def _start(self, proc, threads=None, gpu=None):
        # numpy and scipy each load their own OpenBLAS, which commits ~23 MB per thread at once, one
        # thread per logical core: 1.5 GB of commit per worker that is never touched (a GPU worker
        # committed 2.5 GB while 0.7 GB was resident, and 7 of them took 17 GB of the PC's commit
        # limit). A worker gets its share of the cores, as libigl does. It is read when the worker
        # loads numpy, so it goes in this process's environment while the worker starts (as does
        # DESIGN_ENGINE_GPU=0 for a CPU worker: it then never loads CuPy or Warp).
        env = {"OPENBLAS_NUM_THREADS": str(threads or self._threads)}
        if gpu is not None:
            env["DESIGN_ENGINE_GPU"] = "1" if gpu else "0"
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            proc.start()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _start_cpu_workers(self):
        """Start CPU workers up to what the running map may use (when it hands over its first stage)."""
        while len(self._cpu_procs) < self._cpu_wanted:
            p = self._new_cpu_worker(len(self._cpu_procs))
            self._start(p, threads=1, gpu=False)
            self._cpu_procs.append(p)

    def started(self):
        return [p for p in self._procs if p is not None] + self._cpu_procs

    def cpu_started(self):
        """How many CPU workers are running."""
        return len(self._cpu_procs)

    def alive(self):
        return all(p.is_alive() for p in self.started())

    def pids(self):
        return [p.pid for p in self.started() if p.is_alive()]

    def shrink_caches(self):
        """Halve the workers' RAM caches (applied before their next item)."""
        self._cache_cap.value = max(self._cache_cap.value // 2, 16 << 20)

    def map(self, fn, arg_list, on_progress=None, should_stop=None, active=None, lines=None, on_item=None,
            split=False, cpu_workers=0):
        """Run fn(**args) for each item; returns results in order. Raises WorkerError if an item
        fails. `active` limits how many workers are used (item i goes to worker i % active).
        `should_stop()` is polled while waiting; when it returns True the remaining items are
        abandoned (workers stay alive, with their caches) and None is returned. The (text, level)
        lines the items report (core.report_message) are added to `lines`. `on_item(index, result)`
        gets each item's result as it comes back. `split`: an item may hand its CPU stage to the CPU
        workers (stages.cpu_stage), of which up to `cpu_workers` run (at least one)."""
        self.busy = True
        active = max(1, min(self.n, active or self.n))
        try:
            with trace.span("map", items=len(arg_list), active=active, cpu=cpu_workers if split else 0):
                return self._map(fn, arg_list, on_progress, should_stop, active, lines, on_item, split,
                                 max(1, cpu_workers) if split else 0)
        finally:
            self.busy = False

    def _map(self, fn, arg_list, on_progress, should_stop, active, lines, on_item, split, cpu_workers):
        for w in range(min(active, len(arg_list))):
            if self._procs[w] is None:
                self._procs[w] = self._new_worker(w)
                self._start(self._procs[w])
        self._cpu_wanted = cpu_workers
        self._token += 1
        token = self._token
        self._current.value = token
        # Items are pickled and queued a few per worker at a time, not all at once: a queued item
        # stays in this process's memory until its worker takes it, and items can be big (a body
        # with millions of triangles travels inside each of its blocks' tasks).
        count = len(arg_list)
        waiting = [collections.deque(range(w, count, active)) for w in range(active)]
        meta = trace.task_meta()  # (None unless this run is traced: then the workers trace their tasks too)

        def feed(w):
            if waiting[w]:
                i = waiting[w].popleft()
                with trace.span("send", item=i, worker=w):
                    payload = cloudpickle.dumps((fn, arg_list[i]))
                    # (split: the number of CPU workers, 0 = one stage: a GPU worker hands a stage over only
                    # while fewer than that wait, else it does it itself, _cpu_queue_full)
                    self._tasks[w].put((token, i, payload, int(cpu_workers) if split else 0, meta))

        for w in range(active):
            for _ in range(QUEUED_PER_WORKER):
                feed(w)
        results, done, failure = [None] * count, 0, None
        # Stages handed over and not done yet: each holds a raw mesh (or a block's samples) in a GPU
        # worker's queue or in a CPU worker. A GPU worker is given its next item only while there are
        # fewer than `backlog` (enough to keep every CPU worker busy), else it waits in `held`.
        backlog, in_cpu, handed, early, held = cpu_workers + active, 0, set(), set(), collections.deque()
        while done < count:
            if should_stop and should_stop():
                self._current.value = -token  # queued items of this run get skipped, CPU stages too
                return None
            try:
                got_token, index, kind, value, said = self._results.get(timeout=0.2)
            except queue.Empty:
                if not self.alive():
                    self.terminate()
                    raise WorkerError("a worker process died (out of memory?)")
                continue
            if got_token != token:
                continue  # left over from an earlier, abandoned run
            if kind == HANDED:
                trace.mark("handed", item=index, worker=index % active, bytes=value)
                self._start_cpu_workers()
                if index in early:  # (its CPU stage was quicker than this notice)
                    early.discard(index)
                else:
                    handed.add(index)
                    in_cpu += 1
                if in_cpu < backlog:
                    feed(index % active)
                else:
                    held.append(index % active)
                continue
            from_cpu = kind in (CPU_DONE, CPU_FAILED)
            trace.mark("recv", item=index, worker=index % active, **({"cpu": 1} if from_cpu else {}))
            if not from_cpu:
                feed(index % active)
            elif index in handed:
                handed.discard(index)
                in_cpu -= 1
            else:
                early.add(index)
            while held and in_cpu < backlog:
                feed(held.popleft())
            done += 1
            if lines is not None:
                lines.extend(said)
            if kind in (DONE, CPU_DONE):
                results[index] = value
                if on_item and failure is None:
                    on_item(index, value)
            elif failure is None:
                failure = value
            if on_progress:
                on_progress(done, count)
        if failure:
            raise WorkerError(failure)
        return results

    def terminate(self):
        for p in self.started():
            if p.is_alive():
                p.terminate()
        deadline = time.time() + 2
        for p in self.started():
            p.join(max(0.0, deadline - time.time()))
        # Items still waiting for a worker sit in a feeder thread that blocks forever once the
        # worker is gone (this process holds the pipe's other end too, so it never breaks): their
        # memory was never freed and the process could not exit. Read them out here. (The CPU queue's
        # stages are fed by the GPU workers, gone now; only close() puts anything on it from here.)
        for q in self._tasks + [self._cpu_tasks] + self._released:
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
        for q, p in zip(self._tasks, self._procs):
            try:
                if p is not None:
                    q.put(None)
            except Exception:
                pass
        for _ in self._cpu_procs:  # (each CPU worker takes one and ends)
            try:
                self._cpu_tasks.put(None)
            except Exception:
                pass
        deadline = time.time() + 2
        for p in self.started():
            p.join(max(0.0, deadline - time.time()))
        self.terminate()
