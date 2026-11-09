"""Opt-in timeline trace: when each process works on what, and how busy the GPU and the cores are.

Off unless the environment variable DESIGN_ENGINE_TRACE=1 (`app.py --trace` and tools/trace_run.py set
it). Then every run (a server job, or Evaluator.run outside the server: `begin_run` / `end_run`) gets a
folder <trace dir>/<run id>/ (trace dir: DESIGN_ENGINE_TRACE_DIR, default %LOCALAPPDATA%\\DesignEngine\\trace):

- run.json: the run's id, label, start and end, the clock offset, the PC's logical cores.
- <pid>.jsonl: one JSON line per span of that process: pid, role ("main", "worker 3"), stage, node,
  item, device ("gpu" / "cpu" where it matters), t0, t1, depth (nesting on its thread), thread (when not
  the run's own thread), and whatever else the hook adds. Marks (`mark`) have t0 == t1 and "mark": 1.
- samples.jsonl: every SAMPLE_EVERY s the GPU's use (%) and memory (NVML), the PC's CPU and commit room,
  and for every process of the engine (the main process and its children) its CPU (in cores: 1.0 = one
  core busy all the time) and resident memory. Written by a small sampler process (this file run as a
  script: no numpy, no engine), started by the first traced run and kept for the later ones: a thread of
  the main process went 0.5 s without a sample while a C extension held the GIL (a CPU distance field),
  just when the main process's one-core steps are what the samples should show. It ends with the engine
  (its stdin closes).

Times are seconds on one clock in every process: time.perf_counter() (QueryPerformanceCounter, the same
counter for all processes on Windows) plus the run's offset to the wall clock, taken by the main process
where time.time() ticks over (Python 3.12's time.time() moves in steps of 1-16 ms on Windows). Worker
processes get the run id, folder and offset with each task they are sent while a run is traced
(`task_meta` / `worker_task`): nothing is added to tasks when tracing is off.

Hooks cost next to nothing when off: `span()` and `mark()` check ACTIVE first and return at once (a
shared no-op context manager); no file is opened and psutil / NVML are not touched.

    with trace.span("decimate", item=i):
        ...
    with trace.span("field", xp=xp):   # device from the array module; on the GPU it waits for the GPU
        ...                            # at the end, so the time is the work's, not the next step's

tools/trace_report.py draws a run folder as one HTML page.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SAMPLE_EVERY = 0.1   # seconds between samples
CHILDREN_EVERY = 10  # samples between looking for new processes of the engine (workers start late)
ANSWER_WAIT = 3.0    # s to wait for the sampler process to start or stop a run's samples


def _flag(value):
    return (value or "").strip().lower() not in ("", "0", "false", "off", "no")


ENABLED = _flag(os.environ.get("DESIGN_ENGINE_TRACE"))  # runs begun in this process are traced
ACTIVE = False  # a run is being traced in this process now: spans are written

last_folder = None  # the folder of the last run begun in this process (tools/trace_run.py reports it)

_role = "main"
_run = None        # the traced run: {"id", "folder", "offset", "thread", "info"}
_file = None       # this process's span file of that run
_lock = threading.Lock()
_local = threading.local()  # per thread: depth, node and item of the open spans
_sampler = None    # the sampler process (_Sampler), kept between runs


def enable(on=True, folder=None):
    """Trace the runs begun from now on in this process, and in the worker processes it starts (sets
    DESIGN_ENGINE_TRACE, and DESIGN_ENGINE_TRACE_DIR when `folder` is given, which they inherit)."""
    global ENABLED
    ENABLED = bool(on)
    os.environ["DESIGN_ENGINE_TRACE"] = "1" if on else "0"
    if folder is not None:
        os.environ["DESIGN_ENGINE_TRACE_DIR"] = str(folder)


def trace_dir():
    base = os.environ.get("DESIGN_ENGINE_TRACE_DIR")
    if base:
        return Path(base)
    root = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".cache")
    return Path(root) / "DesignEngine" / "trace"


def set_role(role):
    """What this process is in the trace ("worker 3"); the main process is "main"."""
    global _role
    _role = role


def _wall_offset():
    """time.time() - time.perf_counter(), read just as time.time() ticks over (it moves in steps of 1-16 ms
    on Windows with Python 3.12), so the offset is good to a few microseconds."""
    a = time.time()
    deadline = time.perf_counter() + 0.05
    while True:
        p = time.perf_counter()
        b = time.time()
        if b != a or p > deadline:
            return b - p


# --- runs (main process) -------------------------------------------------------------------------
def begin_run(label=None):
    """Start tracing a run in this process: its folder is made and the sampler told to write there. Returns
    the run id (the folder's name), or None when tracing is off or a run is traced already (a server job
    around its Evaluator.run: the outer one covers it). Pass the id to end_run."""
    global ACTIVE, _run, _sampler, last_folder
    if not ENABLED:
        return None
    with _lock:
        if _run is not None:
            return None
        offset = _wall_offset()
        start = time.perf_counter() + offset
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(start)) + f"-{int(start * 1000) % 1000:03d}"
        name = f"{stamp}-{re.sub(r'[^A-Za-z0-9_-]+', '-', label or 'run').strip('-')[:40] or 'run'}"
        folder, n = trace_dir() / name, 1
        info = {"label": label, "pid": os.getpid(), "start": start, "offset": offset, "cpus": os.cpu_count(),
                "sample_every": SAMPLE_EVERY}
        try:
            while folder.exists():
                n += 1
                folder = trace_dir() / f"{name}-{n}"
            folder.mkdir(parents=True)
            info["run"] = folder.name
            (folder / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        except OSError as err:  # (a trace must never break a run)
            print(f"trace: cannot write to {folder}: {err}", flush=True)
            return None
        _run = {"id": folder.name, "folder": folder, "offset": offset, "thread": threading.get_ident(), "info": info}
        ACTIVE = True
        last_folder = folder
        if _sampler is None or not _sampler.alive():
            _sampler = _Sampler()
        sampler = _sampler
    sampler.command({"cmd": "run", "folder": str(folder), "offset": offset})
    return folder.name


def end_run(run_id):
    """Stop tracing the run begun as `run_id` (None: nothing to do). Its files are complete when this returns."""
    global ACTIVE, _run
    if run_id is None:
        return
    with _lock:
        run = _run
        if run is None or run["id"] != run_id:
            return
        sampler = _sampler
    if sampler is not None:
        sampler.command({"cmd": "stop"}, wait=True)
    with _lock:
        info = dict(run["info"], end=time.perf_counter() + run["offset"])
        try:
            (run["folder"] / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        except OSError:
            pass
        _close()
        _run = None
        ACTIVE = False


@contextmanager
def run(label=None):
    """begin_run / end_run around a block; yields the run id (None when not traced)."""
    run_id = begin_run(label)
    try:
        yield run_id
    finally:
        end_run(run_id)


# --- worker processes ----------------------------------------------------------------------------
def task_meta():
    """What a task sent to a worker process carries while this run is traced, else None: the run's id,
    folder and clock offset, and the node it is for."""
    run = _run
    if run is None:
        return None
    return run["id"], str(run["folder"]), run["offset"], getattr(_local, "node", None)


def worker_task(meta):
    """In a worker process, before each task: trace into the run `meta` (from task_meta) names, or stop
    tracing (None). Returns the node the task is for."""
    global ACTIVE, _run
    with _lock:
        if meta is None:
            _close()
            _run, ACTIVE = None, False
            return None
        run_id, folder, offset, node = meta
        if _run is None or _run["id"] != run_id:
            _close()
            _run = {"id": run_id, "folder": Path(folder), "offset": offset, "thread": threading.get_ident()}
            ACTIVE = True
    return node


# --- spans ---------------------------------------------------------------------------------------
class _Nothing:
    """The span of a process that is not tracing."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set(self, **fields):
        pass


_NOTHING = _Nothing()


class _Span:
    __slots__ = ("rec", "sync", "saved", "t0")

    def __init__(self, rec, sync):
        self.rec, self.sync = rec, sync

    def set(self, **fields):
        """Add fields known only inside the span (e.g. how many triangles came out)."""
        self.rec.update(fields)

    def __enter__(self):
        loc, rec = _local, self.rec
        depth, node, item = getattr(loc, "depth", 0), getattr(loc, "node", None), getattr(loc, "item", None)
        self.saved = depth, node, item
        # nested spans belong to the node and item of the span around them
        loc.depth, loc.node, loc.item = depth + 1, rec.setdefault("node", node), rec.setdefault("item", item)
        rec["depth"] = depth
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, kind, err, tb):
        if self.sync is not None:  # (on the GPU: the work queued in the span is done before it ends)
            try:
                self.sync.cuda.Device().synchronize()
            except Exception:
                pass
        t1 = time.perf_counter()
        _local.depth, _local.node, _local.item = self.saved
        if kind is not None:
            self.rec["error"] = kind.__name__
        _write(self.rec, self.t0, t1)
        return False


def span(stage, node=None, item=None, device=None, xp=None, **fields):
    """A context manager timing what runs inside it as `stage`. node and item default to the enclosing
    span's. `xp`: the array module the work runs on (cupy: device "gpu", and the span waits for the GPU at
    its end; numpy: "cpu"). Other keyword arguments are written with the span."""
    if not ACTIVE:
        return _NOTHING
    sync = None
    if xp is not None:
        on_gpu = hasattr(xp, "cuda")  # (cupy has it, numpy not)
        device = device or ("gpu" if on_gpu else "cpu")
        sync = xp if on_gpu else None
    rec = {"stage": stage, **fields}
    if node is not None:
        rec["node"] = node
    if item is not None:
        rec["item"] = item
    if device is not None:
        rec["device"] = device
    return _Span(rec, sync)


def mark(stage, **fields):
    """A moment (t0 == t1): an item's result received, a decision taken (how many workers a map uses)."""
    if not ACTIVE:
        return
    t = time.perf_counter()
    rec = {"stage": stage, "mark": 1, "depth": getattr(_local, "depth", 0), **fields}
    rec.setdefault("node", getattr(_local, "node", None))
    rec.setdefault("item", getattr(_local, "item", None))
    _write(rec, t, t)


def _write(rec, t0, t1):
    global _file
    with _lock:
        run = _run
        if run is None:
            return
        offset = run["offset"]
        rec = {"pid": os.getpid(), "role": _role, **{k: v for k, v in rec.items() if v is not None},
               "t0": round(t0 + offset, 6), "t1": round(t1 + offset, 6)}
        if threading.get_ident() != run["thread"]:
            rec["thread"] = threading.current_thread().name
        try:
            if _file is None:
                _file = open(run["folder"] / f"{os.getpid()}.jsonl", "a", encoding="utf-8", buffering=1)
            _file.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
        except OSError:
            pass  # (a trace must never break a run)


def _close():
    """Close this process's span file (call with _lock held)."""
    global _file
    if _file is not None:
        try:
            _file.close()
        except OSError:
            pass
        _file = None


# --- the sampler process -------------------------------------------------------------------------
class _Sampler:
    """The main process's handle on the sampler process: one JSON command per line on its stdin ("run" with a
    folder and the clock offset, "stop"), each answered with a line once done."""

    def __init__(self):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
        # -I: only this file, not the folder it is in (engine/ would shadow the standard library's trace)
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), str(os.getpid())], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.answers = queue.Queue()
        self.sent = self.answered = 0
        threading.Thread(target=self._read, name="trace sampler answers", daemon=True).start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.answers.put(line.strip())
        except (OSError, ValueError):
            pass
        self.answers.put(None)  # (it has ended)

    def alive(self):
        return self.proc.poll() is None

    def command(self, cmd, wait=False):
        """Send a command; with `wait`, return once it and every one before it is done (a run's samples
        are complete). A run does not wait for its start: the first one would wait for the process to
        start (0.15 s). A missing sampler only costs the samples."""
        try:
            self.proc.stdin.write(json.dumps(cmd) + "\n")
            self.proc.stdin.flush()
            self.sent += 1
            while wait and self.answered < self.sent:
                if self.answers.get(timeout=ANSWER_WAIT) is None:
                    return  # (it has ended)
                self.answered += 1
        except (OSError, ValueError, queue.Empty):
            pass


def _commit_room(psutil):
    """As engine.cache.commit_room (which this process does not import: it loads numpy)."""
    if sys.platform == "win32":
        import ctypes

        class Info(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                        ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                        ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                        ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong),
                        ("ThreadCount", ctypes.c_ulong)]
        info = Info()
        info.cb = ctypes.sizeof(info)
        if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return int(info.CommitLimit - info.CommitTotal) * int(info.PageSize)
    return int(psutil.virtual_memory().available + psutil.swap_memory().free)


def _sampler_main(parent):
    """The sampler process: samples while a run is on, sleeps between runs, ends with its stdin."""
    import psutil
    gpu = None
    try:  # (as engine.backend.gpu_memory reads the card: NVML, same numbers as nvidia-smi)
        import pynvml
        pynvml.nvmlInit()
        gpu = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        pass
    commands = queue.Queue()

    def read():
        for line in sys.stdin:
            commands.put(line)
        commands.put(None)

    threading.Thread(target=read, daemon=True).start()
    engine, me = psutil.Process(parent), os.getpid()
    out, offset, procs, count, tick = None, 0.0, {}, 0, 0.0
    while True:
        try:
            line = commands.get(timeout=max(0.0, tick - time.perf_counter())) if out is not None else commands.get()
        except queue.Empty:
            line = ""
        if line is None or not engine.is_running():
            break  # the engine has ended
        cmd = json.loads(line) if line.strip() else {}
        if cmd.get("cmd") == "run":
            if out is not None:
                out.close()
            offset, procs, count = float(cmd["offset"]), {}, 0
            out = open(Path(cmd["folder"]) / "samples.jsonl", "a", encoding="utf-8", buffering=1)
            head = {"info": 1, "cpus": os.cpu_count(), "pid": parent, "sampler": me}
            if gpu is not None:
                try:
                    name = gpu[0].nvmlDeviceGetName(gpu[1])
                    head.update(gpu_name=name.decode() if isinstance(name, bytes) else str(name),
                                gpu_total=int(gpu[0].nvmlDeviceGetMemoryInfo(gpu[1]).total))
                except Exception:
                    pass
            out.write(json.dumps(head) + "\n")
            psutil.cpu_percent(None)  # (the first call only starts the count)
            tick = time.perf_counter()
        elif cmd.get("cmd") == "stop":
            if out is not None:
                _sample(out, offset, gpu, psutil, engine, me, procs, count)
                out.close()
                out = None
        elif out is not None:  # time for a sample
            _sample(out, offset, gpu, psutil, engine, me, procs, count)
            count += 1
            tick = max(tick + SAMPLE_EVERY, time.perf_counter())
            continue
        print("ok", flush=True)


def _sample(out, offset, gpu, psutil, engine, me, procs, count):
    rec = {"t": round(time.perf_counter() + offset, 4)}
    if gpu is not None:
        try:
            lib, handle = gpu
            rec["gpu"] = int(lib.nvmlDeviceGetUtilizationRates(handle).gpu)
            rec["gpu_mem"] = int(lib.nvmlDeviceGetMemoryInfo(handle).used)
        except Exception:
            pass
    if count % CHILDREN_EVERY == 0:
        try:
            found = [engine] + engine.children(recursive=True)
            mine = {me} | {p.pid for p in psutil.Process(me).children(recursive=True)}  # (its console host)
        except psutil.Error:
            found, mine = [engine], {me}
        for p in found:
            if p.pid not in procs and p.pid not in mine:
                procs[p.pid] = [p, None, None]
    rec["sys_cpu"] = round(psutil.cpu_percent(None) / 100 * (os.cpu_count() or 1), 2)
    rows = {}
    for pid, entry in list(procs.items()):
        try:
            with entry[0].oneshot():
                times = entry[0].cpu_times()
                rss = entry[0].memory_info().rss
        except psutil.Error:
            del procs[pid]
            continue
        now, busy = time.perf_counter(), times.user + times.system
        if entry[1] is not None and now > entry[2]:
            rows[str(pid)] = [round((busy - entry[1]) / (now - entry[2]), 3), int(rss)]
        entry[1], entry[2] = busy, now
    rec["procs"] = rows
    try:
        rec["commit_room"] = _commit_room(psutil)
    except Exception:
        pass
    out.write(json.dumps(rec, separators=(",", ":")) + "\n")


if __name__ == "__main__":  # the sampler process (see _Sampler)
    _sampler_main(int(sys.argv[1]))
