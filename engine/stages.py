"""Two-stage worker items: the part of an item that needs the GPU, and the CPU-only part after it.

A GPU worker's item (a body meshed, a few blocks) can end in CPU-only work (decimation, reduction, dual
contouring's extraction) that keeps the GPU worker, and with it the card, from its next item: the lamp's
petals spent 0.3 s decimating for every 0.15 s on the GPU. An item that ends that way returns
`cpu_stage(fn, **kwargs)`: in a worker of a map that splits (StickyPool.map(split=True), chosen by the
evaluator when the map runs on the GPU), that is a CpuStage, which the worker hands to the pool's CPU-only
workers (cpu_worker_main: processes without CUDA, as many as the cores and memory allow), and fn(**kwargs)
there is the item's result. Anywhere else (the main process, a CPU worker, a map that does not split, a call
nested in the item) it calls fn(**kwargs) at once. The same function on the same arrays either way (the
arrays are copied exactly), so the results are the same bit for bit.

Only the item's own function may return it: code that calls a function which may return a CpuStage and
uses its result must run it with `allow(False)` (core.parallel_map does, for a list meshed inside a worker).

A CPU worker imports this module and what its stages need (meshing, dualcontour, blockmesh: numpy, trimesh,
scikit-image, pyfqmr), not the node library: 0.09 GB instead of 0.3 GB per process. So stage functions live
in those modules and report through `report` (core.report_message where the graph is loaded).
"""
import os
import queue
import threading
import time
import traceback
from contextlib import contextmanager

_local = threading.local()

SHARED_MIN = 256 << 10  # a CPU stage whose arrays take more is handed over in shared memory (HandOver)

# result kinds of a CPU worker (the third field of what it puts on the pool's results queue)
CPU_DONE, CPU_FAILED = "cpu done", "cpu failed"


class CpuStage:
    """The rest of an item, to run in a CPU-only worker: fn(**kwargs) is the item's result. fn must be a
    module-level function (it is pickled by reference) and kwargs plain data and numpy arrays (no CuPy:
    a CPU worker has no CUDA)."""
    __slots__ = ("fn", "kwargs")

    def __init__(self, fn, kwargs):
        self.fn, self.kwargs = fn, kwargs

    def run(self):
        return self.fn(**self.kwargs)


def splitting():
    """Whether a cpu_stage now would be handed to a CPU worker (code that prepares its arguments
    differently for that, e.g. keeping the blocks' samples instead of meshing each at once, asks this)."""
    return getattr(_local, "split", False)


def cpu_stage(fn, **kwargs):
    """The end of an item: fn(**kwargs), or, in a worker of a map that splits, a CpuStage that a CPU worker
    runs. Returned by the item's function as its result."""
    if splitting():
        _local.split = False  # one hand-over per item (the item returns it)
        return CpuStage(fn, kwargs)
    return fn(**kwargs)


@contextmanager
def allow(on=True):
    """Lets cpu_stage hand over inside the block (on) or not (off); the worker loop sets it per item."""
    old = getattr(_local, "split", False)
    _local.split = bool(on)
    try:
        yield
    finally:
        _local.split = old


# --- report lines --------------------------------------------------------------------------------
reporter = None  # core.report_message, set when the graph is loaded (not in a CPU worker)


def report(text, level="info"):
    """core.report_message for stage functions: in a CPU worker the line goes back with the item's result."""
    if reporter is not None:
        reporter(text, level)
    else:
        to_sink(text, level)


def collect(lines):
    """In a worker, around each item: the list its report lines go to (None: none)."""
    _local.sink = lines


def to_sink(text, level):
    sink = getattr(_local, "sink", None)
    if sink is not None:
        sink.append((text, level))


# --- the hand-over -------------------------------------------------------------------------------
class HandOver:
    """A GPU worker's side of handing CPU stages over. The stage is pickled with its arrays out of band
    (protocol 5): those go into a shared memory block, and only the rest (a few kB) through the CPU queue's
    pipe. Through the pipe they cost two more pickled copies and the one pipe all CPU workers read in turn:
    hex studs' blocks (21 MB a task) took 58 ms each to hand over, and waited 0.8 s on average for a CPU
    worker while those were 40% idle. On Windows a block lives while a process has it open, so this worker
    keeps each one until the CPU worker that took it says it has copied it (on `released`)."""

    def __init__(self, released, number):
        self.released, self.number, self.held = released, number, {}

    def pack(self, stage):
        """(pickled stage, shared block or None, bytes of its arrays) for the CPU queue."""
        import cloudpickle
        from multiprocessing import shared_memory
        buffers = []
        head = cloudpickle.dumps((stage.fn, stage.kwargs), protocol=5, buffer_callback=buffers.append)
        raws = [b.raw() for b in buffers]
        sizes = [r.nbytes for r in raws]
        total = sum(sizes)
        if total < SHARED_MIN or self.released is None:  # small: all of it through the pipe
            return cloudpickle.dumps((stage.fn, stage.kwargs)), None, total
        block = shared_memory.SharedMemory(create=True, size=total)
        at = 0
        for raw, n in zip(raws, sizes):
            block.buf[at:at + n] = raw
            at += n
        self.held[block.name] = block
        return head, (block.name, sizes, self.number), total

    def drain(self):
        """Close the blocks the CPU workers have copied."""
        while self.held:
            try:
                name = self.released.get_nowait()
            except queue.Empty:
                return
            block = self.held.pop(name, None)
            if block is not None:
                block.close()

    def close(self):
        for block in self.held.values():
            block.close()
        self.held.clear()


def unpack(head, shared, released):
    """A CPU worker's side: the stage (fn, kwargs) from the CPU queue; its arrays are copied out of their
    shared block (into memory of this process, writable), and the GPU worker told it may close the block."""
    import cloudpickle
    if shared is None:
        return cloudpickle.loads(head)
    import numpy as np
    from multiprocessing import shared_memory
    name, sizes, owner = shared
    total = sum(sizes)
    data = np.empty(total, dtype=np.uint8)
    block = shared_memory.SharedMemory(name=name)
    try:
        view = np.frombuffer(block.buf, dtype=np.uint8, count=total)
        data[:] = view
        del view  # (a block with views on it cannot be closed)
    finally:
        block.close()
        released[owner].put(name)
    whole, buffers, at = memoryview(data), [], 0
    for n in sizes:
        buffers.append(whole[at:at + n])
        at += n
    return cloudpickle.loads(head, buffers=buffers)


def failure(err):
    """An error as the line a worker sends back."""
    msg = f"{type(err).__name__}: {err}"
    if not str(err):
        msg += "\n" + traceback.format_exc(limit=3)
    return msg


# --- the CPU worker ------------------------------------------------------------------------------
def _freeze_imports():
    """Import what CPU stages use, then gc.freeze() (as the GPU workers do: their modules' objects are not
    walked by every gc.collect between items)."""
    import gc
    try:
        import pyfqmr, scipy.sparse, scipy.spatial, skimage.measure, trimesh  # noqa: F401,E401
        from . import blockmesh, dualcontour, meshing  # noqa: F401
    except Exception:  # (a missing optional module only means less is frozen)
        pass
    gc.collect()
    gc.freeze()


def cpu_worker_main(cpu_tasks, results, current, number=None, released=None):
    """A CPU worker: runs the CPU stages GPU workers hand over (any one takes the next), no GPU, no caches.
    Started with DESIGN_ENGINE_GPU=0 and one BLAS thread (workers.StickyPool._start_cpu_workers)."""
    import gc
    os.environ["IGL_NUM_THREADS"] = "1"
    from . import trace
    _freeze_imports()
    trace.set_role(f"cpu worker {number}")
    for q in released:  # (a note to a GPU worker that has ended is not waited for when this one ends)
        q.cancel_join_thread()
    last_gc = time.time()
    while True:
        task = cpu_tasks.get()
        if task is None:
            return
        token, index, payload, shared, said, meta = task
        if token != current.value:  # from a run that was cancelled: skip (its block can be closed)
            if shared is not None:
                released[shared[2]].put(shared[0])
            continue
        collect(lines := list(said))  # the GPU stage's lines first, as when one process did both
        node = trace.worker_task(meta) if meta or trace.ACTIVE else None
        fn = kwargs = value = None
        try:
            with trace.span("task", node=node, item=index):
                with trace.span("load args", bytes=len(payload) + (sum(shared[1]) if shared else 0)):
                    fn, kwargs = unpack(payload, shared, released)
                task = payload = None
                value = fn(**kwargs)
            results.put((token, index, CPU_DONE, value, lines))
        except BaseException as err:
            results.put((token, index, CPU_FAILED, failure(err), lines))
        finally:
            fn = kwargs = value = task = payload = None
            collect(None)
            if cpu_tasks.empty() or time.time() - last_gc > 1.5:
                with trace.span("gc", node=node, item=index):
                    gc.collect()
                last_gc = time.time()
