"""Node graph: typed ports, a node registry, the graph model (JSON) and a caching evaluator.

Graph JSON:

    {
      "version": 1,
      "nodes": {
        "part":   {"type": "implicit.from_cad", "inputs": {"cad": {"link": "import.bodies"}}},
        "offset": {"type": "implicit.offset",
                   "inputs": {"body": {"link": "part.out"}, "distance": 0.5}},
        ...
      }
    }

An input is either a literal value or {"link": "<node id>.<output port>"}. Unset inputs use the
port default. Nodes may carry extra UI data (e.g. "pos") which is preserved but ignored here.

Groups (own nodes made from other nodes) and named face selections are resolved when a graph is
read (`Graph.from_dict`), so the evaluator only ever sees plain nodes:

    "groups": {"noise_wave": {
        "nodes": {...},                                   # inner nodes; "$in.<name>" = a group input
        "inputs": [{"name": "body", "type": "implicit"}],
        "outputs": [{"name": "out", "type": "implicit", "link": "offset.out"}]}},
    "selections": {"top": "10, 11"}

A node {"type": "group:noise_wave", "inputs": {...}} becomes a copy of the group's nodes with ids
"<node>__<inner id>". Its inputs are the group inputs plus one per Number node inside the group
(named by the Number node's id: the group's settings). A faces input may name selections.

Lists: a node output can be a `Many` (e.g. all bodies of a STEP file). When a `Many` reaches a
port that expects a single item, the node runs once per item and its outputs become `Many` too,
so one graph handles one part or a whole assembly. Ports declared `many=True` receive the
whole list instead (e.g. export, which merges everything into one file).
"""
import collections
import copy
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import backend
from ..cache import available_memory, memory_limit, psutil
from .workers import StickyPool

REQUIRED = object()

# Free RAM (whole PC) at which the memory watchdog acts. Generous on purpose: this PC's page file
# is on a mechanical hard disk, and once Windows starts paging to it the whole machine freezes
# (that happened with ~1 GB still "free"), long before memory is actually exhausted.
FREE_TIGHT = 3 * 2**30        # free caches, stop idle worker processes
FREE_STOP = int(1.5 * 2**30)  # stop the run
FREE_LAST_RESORT = int(0.75 * 2**30)  # after a stop: end the process if it stays this low


_current = threading.local()  # (evaluator, node id) of the node running on this thread


def report_progress(done, total):
    """Called by slow node functions to report progress (and to let a cancel take effect)."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        return
    evaluator, node_id = ctx
    evaluator._check_cancel()
    evaluator._emit(type="progress", node=node_id, done=done, total=total)


def check_cancel():
    """Called inside long node functions: raises Cancelled once the run is cancelled."""
    ctx = getattr(_current, "ctx", None)
    if ctx is not None:
        ctx[0]._check_cancel()


def parallel_map(fn, arg_list, need=0, gpu=False, title="Blocks", gpu_need=0):
    """Run fn(**args) for every item, in the evaluator's worker processes when called by a node
    running in the main process (one big body split into pieces); one by one inside a worker or
    outside a run. `need`: bytes one item needs (limits how many run at once); `gpu`: the items
    use the GPU (limits how many run at once by free GPU memory, `gpu_need` = extra GPU bytes per
    worker)."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        return [fn(**a) for a in arg_list]
    evaluator, node_id = ctx
    return evaluator._map_in_workers(node_id, fn, arg_list, need, gpu, title, gpu_need)


def report_view(mesh):
    """Called by long node functions to show an intermediate shape in the 3D view (the editor shows
    it as the node's view; the node's real result replaces it when it is done)."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        return
    evaluator, node_id = ctx
    evaluator._emit(type="live_view", node=node_id, mesh=mesh)


def report_message(text, level="info"):
    """Called by node functions to put a line in the editor's log (level: info, warning or error).
    Warnings and errors also stay on the node's card, and come again with every later run that takes
    the node's result from the cache. In a worker process the lines go back with the item's result."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        sink = getattr(_current, "sink", None)  # set by the worker process around each item
        if sink is not None:
            sink.append((text, level))
        return
    evaluator, node_id = ctx
    evaluator._message(node_id, text, level)

# Which output types may feed which input types. Scalar fields and implicit bodies are the
# same thing underneath (a body can drive math too), so they connect both ways.
_COMPATIBLE = {
    "scalar": {"scalar", "implicit", "number"},
    "implicit": {"implicit", "scalar", "number"},
    "number": {"number"},
    "int": {"int", "number"},
    "cad": {"cad"},
    "mesh": {"mesh"},
    "path": {"path", "text"},
    "text": {"text", "path"},
    "faces": {"faces", "text"},  # CAD face numbers, picked in the editor's 3D view
    "part": {"cad", "implicit", "scalar"},  # a CAD body or an implicit body (Static Analysis)
    "material": {"material"},
    "conditions": {"conditions"},  # supports and loads (Fixed Faces, Load)
    "frame": {"frame", "cad"},  # a frame's bars (Frame Optimization), or a CAD body to connect or join
}


class Many(list):
    """A list of values flowing along one link (e.g. every body in a STEP file)."""


class GraphError(Exception):
    """A problem with the graph. `problems` lists (node id or None, message) pairs so a UI can
    mark the offending nodes."""

    def __init__(self, message, problems=None, node_id=None):
        super().__init__(message)
        self.problems = problems or ([(node_id, message)] if node_id else [])


class Cancelled(GraphError):
    pass


@dataclass
class Port:
    name: str
    type: str
    default: Any = REQUIRED
    label: str = None
    doc: str = ""
    choices: list = None   # for enum-like text inputs
    many: bool = False     # receive a whole Many instead of mapping over it
    optional: bool = False # may be left unconnected (value None)
    hidden: bool = False   # set by the server, not shown in the editor (Topology Optimization's compute)

    def __post_init__(self):
        self.label = self.label or self.name.replace("_", " ").title()

    def describe(self):
        d = {"name": self.name, "type": self.type, "label": self.label}
        if self.default is not REQUIRED:
            d["default"] = self.default
        for k in ("doc", "choices", "many", "optional", "hidden"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d


@dataclass
class NodeType:
    type_id: str
    title: str
    category: str
    inputs: list
    outputs: list
    fn: Callable
    doc: str = ""
    parallel: bool = False     # when mapping over a Many, run items in worker processes
    side_effect: bool = False  # always re-run (writes files)
    hidden: bool = False       # internal (used by the editor), not listed in the catalog
    estimate: Callable = None  # kwargs -> bytes one call needs (limits how many run in parallel)
    paint: str = None          # output (a field) that colours the node's own body in the 3D view
    preview: bool = False      # evaluated in every preview once its inputs are linked (shows markers)

    def input(self, name):
        for p in self.inputs:
            if p.name == name:
                return p
        raise GraphError(f"{self.type_id} has no input '{name}'")

    def output(self, name):
        for p in self.outputs:
            if p.name == name:
                return p
        raise GraphError(f"{self.type_id} has no output '{name}'")

    def describe(self):
        return {"type": self.type_id, "title": self.title, "category": self.category, "doc": self.doc,
                "inputs": [p.describe() for p in self.inputs],
                "outputs": [p.describe() for p in self.outputs]}


REGISTRY: dict = {}


def node(type_id, title, category, inputs, outputs, doc="", parallel=False, side_effect=False, hidden=False,
         estimate=None, paint=None, preview=False):
    """Register a function as a node type. The function receives inputs as keyword arguments
    and returns a dict of outputs, or a single value when the node has one output."""
    def register(fn):
        REGISTRY[type_id] = NodeType(type_id, title, category, inputs, outputs, fn,
                                     doc or (fn.__doc__ or "").strip(), parallel, side_effect, hidden, estimate,
                                     paint, preview)
        return fn
    return register


def node_catalog():
    """All node types as plain data, for UIs and documentation."""
    return [t.describe() for t in sorted(REGISTRY.values(), key=lambda t: (t.category, t.title))
            if not t.hidden]


def compatibility():
    """Input type -> list of output types that may connect to it."""
    return {k: sorted(v) for k, v in _COMPATIBLE.items()}


# --- graph model ---------------------------------------------------------------------------
def link(ref):
    return {"link": ref}


def _is_link(value):
    return isinstance(value, dict) and set(value) == {"link"}


def _split(ref):
    node_id, _, port = ref.partition(".")
    if not port:
        raise GraphError(f"link '{ref}' must look like 'node.port'")
    return node_id, port


GROUP_PREFIX = "group:"
GROUP_SEP = "__"              # node ids inside an expanded group: "<group node>__<inner id>"
GROUP_SETTING = "input.number"  # Number nodes inside a group are its settings


def _expand_groups(nodes, groups, depth=0):
    """`nodes` with every group node replaced by a copy of the group's nodes. Returns (nodes,
    aliases, outputs): aliases maps "<group node>.<output>" to the inner "<node>.<port>" giving it
    (it may itself be an alias when groups are nested); outputs lists each group node's outputs
    as (name, type)."""
    if depth > 16:
        raise GraphError("groups are nested more than 16 deep (does a group contain itself?)")
    out, aliases, outputs = {}, {}, {}
    for node_id, spec in nodes.items():
        type_id = spec.get("type", "")
        group = groups.get(type_id[len(GROUP_PREFIX):]) if type_id.startswith(GROUP_PREFIX) else None
        if group is None:
            out[node_id] = spec
            continue
        given = spec.get("inputs", {})
        inner = {}
        for inner_id, inner_spec in group.get("nodes", {}).items():
            inputs = {}
            for port, value in inner_spec.get("inputs", {}).items():
                if _is_link(value):
                    src, src_port = _split(value["link"])
                    if src == "$in":  # a group input: whatever the group node gets (or the default)
                        if src_port in given:
                            inputs[port] = given[src_port]
                        continue
                    value = link(f"{node_id}{GROUP_SEP}{src}.{src_port}")
                inputs[port] = value
            if inner_spec.get("type") == GROUP_SETTING and inner_id in given:
                inputs["value"] = given[inner_id]
            inner[f"{node_id}{GROUP_SEP}{inner_id}"] = {**inner_spec, "inputs": inputs}
        inner, inner_aliases, _ = _expand_groups(inner, groups, depth + 1)
        out.update(inner)
        aliases.update(inner_aliases)
        outputs[node_id] = []
        for o in group.get("outputs", []):
            src, src_port = _split(o["link"])
            aliases[f"{node_id}.{o['name']}"] = f"{node_id}{GROUP_SEP}{src}.{src_port}"
            outputs[node_id].append((o["name"], o.get("type", "")))
    return out, aliases, outputs


def _resolve_alias(ref, aliases):
    for _ in range(64):
        if ref not in aliases:
            return ref
        ref = aliases[ref]
    raise GraphError(f"group outputs refer to each other in a loop ({ref})")


_SELECTION_NAME = re.compile(r"[A-Za-z_][\w-]*")


def _resolve_selections(nodes, selections):
    """Names in faces inputs replaced by the face numbers they stand for (unknown names stay, and
    the node reports them)."""
    for spec in nodes.values():
        ntype = REGISTRY.get(spec.get("type"))
        if ntype is None:
            continue
        inputs = spec.get("inputs", {})
        for port in ntype.inputs:
            value = inputs.get(port.name)
            if port.type == "faces" and isinstance(value, str) and _SELECTION_NAME.search(value):
                inputs[port.name] = _SELECTION_NAME.sub(lambda m: str(selections.get(m.group(0), m.group(0))), value)


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)  # "<group node>.<output>" -> "<inner node>.<port>"
    groups: dict = field(default_factory=dict)   # group node id -> [(output name, type)]
    owners: dict = field(default_factory=dict)   # node id inside a group -> the (top level) group node

    # construction ----------------------------------------------------------------------
    def add(self, node_id, type_id, **inputs):
        if node_id in self.nodes:
            raise GraphError(f"duplicate node id '{node_id}'")
        self.nodes[node_id] = {"type": type_id, "inputs": dict(inputs)}
        return node_id

    def set(self, node_id, port, value):
        if node_id not in self.nodes:
            raise GraphError(f"no node '{node_id}'")
        self.nodes[node_id].setdefault("inputs", {})[port] = value

    # serialisation ---------------------------------------------------------------------
    def to_dict(self):
        return {"version": 1, "nodes": self.nodes}

    @classmethod
    def from_dict(cls, data):
        """A graph from recipe JSON, with groups expanded and face selection names resolved."""
        if data.get("version", 1) != 1:
            raise GraphError(f"unsupported graph version {data.get('version')}")
        data = copy.deepcopy(data)
        top = data["nodes"]
        nodes, aliases, outputs = _expand_groups(top, data.get("groups") or {})
        for spec in nodes.values():
            for port, value in spec.get("inputs", {}).items():
                if _is_link(value) and value["link"] in aliases:
                    spec["inputs"][port] = link(_resolve_alias(value["link"], aliases))
        _resolve_selections(nodes, data.get("selections") or {})
        owners = {n: n.split(GROUP_SEP, 1)[0] for n in nodes if n not in top and n.split(GROUP_SEP, 1)[0] in outputs}
        aliases = {ref: _resolve_alias(ref, aliases) for ref in aliases if ref.split(".", 1)[0] in outputs}
        return cls(nodes, aliases, {n: o for n, o in outputs.items() if n in top}, owners)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # structure -------------------------------------------------------------------------
    def upstream(self, node_id):
        return [_split(v["link"])[0] for v in self.nodes[node_id].get("inputs", {}).values() if _is_link(v)]

    def sinks(self):
        used = {u for n in self.nodes for u in self.upstream(n)}
        return [n for n in self.nodes if n not in used]

    def order(self, targets=None):
        """Topological order of the nodes needed for `targets` (default: all sinks)."""
        out, state = [], {}

        def visit(n, path):
            if state.get(n) == "done":
                return
            if state.get(n) == "visiting":
                raise GraphError("cycle: " + " -> ".join(path + [n]))
            if n not in self.nodes:
                raise GraphError(f"link to missing node '{n}'")
            state[n] = "visiting"
            for u in self.upstream(n):
                visit(u, path + [n])
            state[n] = "done"
            out.append(n)

        for t in targets or self.sinks():
            visit(t, [])
        return out

    def problems(self, node_ids=None):
        """Every problem found, as (node id or None, message) pairs. Checks all nodes, or only
        `node_ids` (so a half-wired new node does not block previewing the rest)."""
        problems = []
        for node_id in (list(self.nodes) if node_ids is None else node_ids):
            spec = self.nodes[node_id]
            ntype = REGISTRY.get(spec.get("type"))
            if ntype is None:
                problems.append((node_id, f"unknown node type '{spec.get('type')}'"))
                continue
            inputs = spec.get("inputs", {})
            for name, value in inputs.items():
                try:
                    port = ntype.input(name)
                except GraphError as err:
                    problems.append((node_id, str(err)))
                    continue
                if _is_link(value):
                    src_id, src_port = _split(value["link"])
                    src = self.nodes.get(src_id)
                    if src is None or src.get("type") not in REGISTRY:
                        problems.append((node_id, f"{name}: links to missing node '{src_id}'"))
                        continue
                    try:
                        out = REGISTRY[src["type"]].output(src_port)
                    except GraphError as err:
                        problems.append((node_id, f"{name}: {err}"))
                        continue
                    if out.type not in _COMPATIBLE.get(port.type, {port.type}):
                        problems.append((node_id, f"{name}: cannot connect {out.type} output "
                                                  f"'{value['link']}' to {port.type} input"))
            for port in ntype.inputs:
                if port.name not in inputs and port.default is REQUIRED and not port.optional:
                    problems.append((node_id, f"required input '{port.name}' is not set"))
        try:
            self.order(list(self.nodes) if node_ids is None else node_ids)
        except GraphError as err:
            problems.append((None, str(err)))
        return problems

    def validate(self, node_ids=None):
        """Raise GraphError listing every problem found."""
        problems = self.problems(node_ids)
        if problems:
            lines = [f"{n}: {m}" if n else m for n, m in problems]
            raise GraphError("invalid graph:\n  " + "\n  ".join(lines), problems=problems)


# --- evaluation ----------------------------------------------------------------------------
def _output_range(outputs):
    """[low, high] of a node's first output when it follows from the formulas (fields carry it,
    numbers are their own range); None when unknown. Shown on the node's card."""
    if not outputs:
        return None
    value = next(iter(outputs.values()))
    lo = hi = None
    for v in (value if isinstance(value, list) else [value]):
        r = getattr(v, "range", None)
        if r is None and isinstance(v, (int, float)) and not isinstance(v, bool):
            r = (float(v), float(v))
        if r is None:
            return None
        lo, hi = (r[0], r[1]) if lo is None else (min(lo, r[0]), max(hi, r[1]))
    return None if lo is None else [lo, hi]


def _output_extras(outputs):
    """Extra node_done fields from a node's first output: `markers` (supports and loads to draw in
    the 3D view; a list for a list of values), `note` (a line shown on the node's card) and `frames`
    (a frame's bars for the editor's frame view)."""
    if not outputs:
        return {}
    value = next(iter(outputs.values()))
    items = value if isinstance(value, list) else [value]
    found = [m for v in items if (m := getattr(v, "markers", None))]
    notes = [n for v in items if (n := getattr(v, "note", None))]
    frames = [f for v in items if (f := getattr(v, "frame_view", None))]
    return {"markers": found or None, "note": notes[0] if notes else None, "frames": frames or None}


def _literal_signature(port, value):
    if port.type == "path" and isinstance(value, str) and os.path.exists(value):
        return (value, os.path.getmtime(value))  # re-run when the file on disk changes
    return value


class Evaluator:
    """Runs a graph, caching each node's outputs. Re-running after a change only recomputes the
    nodes whose inputs (or upstream nodes) changed. Keep one Evaluator alive across edits.

    `on_event` receives dicts while running: node_start, node_done (with time and cached),
    node_error, and progress (done/total while mapping over a list). `cancel()` may be called
    from another thread; the run stops at the next node or list item with `Cancelled`.
    """

    def __init__(self, graph, workers=4, log=print, on_event=None):
        self.graph = graph
        self.workers = workers
        self.log = log or (lambda *_: None)
        self.on_event = on_event
        self._cache = {}  # node id -> (signature, outputs, warnings)
        self._warnings = None  # warnings and errors reported by the node being computed
        self._collect = None   # lines of a list's items, gathered to report each line once
        self._cancel = False
        self._stop_reason = None  # set when the memory watchdog stops a run
        self._pool = None  # worker processes, kept alive so later runs start instantly
        self.computed = []  # node ids actually computed in the last run
        self.memory_peak = 0

    def cancel(self):
        self._cancel = True

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def release_memory(self):
        """Between runs, when the PC needs memory back: stop the worker processes and forget the
        cached meshes (the biggest results; they are rebuilt, mostly from the SSD cache, on the
        next run). Imported bodies and fields are kept."""
        self.close()
        for node_id in list(self._cache):
            spec = self.graph.nodes.get(node_id, {})
            if str(spec.get("type", "")).startswith(("mesh.", "output.")):
                self._cache.pop(node_id, None)

    def _emit(self, **event):
        if self.on_event:
            self.on_event(event)

    MAX_WARNINGS = 20  # kept per node for its card

    def _message(self, node_id, text, level):
        if self._collect is not None:
            self._collect.append((text, level))
            return
        self.log(text)
        self._emit(type="message", node=node_id, text=text, level=level)
        if level != "info" and self._warnings is not None and len(self._warnings) < self.MAX_WARNINGS:
            if [text, level] not in self._warnings:
                self._warnings.append([text, level])

    def _gathered(self, node_id, lines):
        """Report the lines that the items of a list reported, each line once, with how many items
        reported it (144 bodies saying the same thing make one line)."""
        for (text, level), n in collections.Counter(lines).items():
            self._message(node_id, text if n == 1 else f"{text} (x{n})", level)

    def _check_cancel(self):
        if self._cancel:
            if self._stop_reason:
                raise GraphError(self._stop_reason)
            raise Cancelled("cancelled")

    # --- memory ------------------------------------------------------------------------------
    def memory_in_use(self):
        """Resident bytes of this process plus its worker processes (0 if psutil is missing).

        When RAM runs short, Windows pages memory out and this drops while a process keeps
        growing (a runaway import showed 5 GB resident with 23 GB committed). Committed memory is
        no better a measure: loading the engine commits 1.3 GB it never touches, starting CUDA
        another 0.4 GB. That case is caught by the free-memory checks in _watch_memory instead."""
        if psutil is None:
            return 0
        pids = [os.getpid()] + (self._pool.pids() if self._pool is not None else [])
        total = 0
        for pid in pids:
            try:
                total += psutil.Process(pid).memory_info().rss
            except psutil.Error:
                pass
        return total

    def _watch_memory(self, stop):
        """Runs during a run: frees caches when memory gets tight, and as a last resort stops
        the run (killing the workers frees their memory at once) rather than let the PC run
        out of memory."""
        from .nodes import ARRAY_CACHE
        limit, gb = memory_limit(), 2 ** 30
        shrunk = False
        while not stop.wait(0.5):
            used, free = self.memory_in_use(), available_memory()
            self.memory_peak = max(self.memory_peak, used)
            pool = self._pool
            tight = used > 0.85 * limit or free < FREE_TIGHT
            if tight and pool is not None and not pool.busy:
                # Idle workers hold ~0.6-1 GB each; their cached results are on the SSD anyway,
                # so stop them (they restart on the next parallel step) instead of failing the run.
                self._pool = None
                pool.terminate()
                self.log(f"memory {used / gb:.1f} GB of {limit / gb:.1f} GB: stopped idle worker processes")
                continue
            if tight:
                ARRAY_CACHE.set_limit(ARRAY_CACHE.max_bytes // 2)  # moves to the SSD cache
                if self._pool is not None:
                    self._pool.shrink_caches()
                if not shrunk:
                    self.log(f"memory {used / gb:.1f} GB of {limit / gb:.1f} GB: freeing caches")
                shrunk = True
            if used > limit or free < FREE_STOP:
                self._stop_reason = (
                    f"stopped to protect the PC: memory use reached {used / gb:.1f} GB "
                    f"(limit {limit / gb:.1f} GB, {free / gb:.1f} GB left free). Use a larger voxel "
                    f"size or Draft quality, or close other programs.")
                self._cancel = True
                pool, self._pool = self._pool, None
                if pool is not None:
                    pool.terminate()
                self._last_resort(limit, stop)
                return

    def _last_resort(self, limit, stop):
        """The run has been told to stop, but a step that cannot be interrupted may keep growing.
        If memory stays far over the limit, end the whole process rather than freeze the PC (the
        editor's graph lives in the browser, so nothing is lost)."""
        over_since = None
        while not stop.wait(1.0):
            if self.memory_in_use() > 1.25 * limit or available_memory() < FREE_LAST_RESORT:
                over_since = over_since or time.time()
                if time.time() - over_since > 10:
                    print("Design Engine: memory is still far over the limit after stopping the run; "
                          "closing to protect the PC. Start it again; your graph is still in the browser.",
                          flush=True)
                    os._exit(3)
            else:
                over_since = None

    def run(self, targets=None):
        self._cancel = False
        self._stop_reason = None
        self.memory_peak = 0
        stop = threading.Event()
        watchdog = threading.Thread(target=self._watch_memory, args=(stop,), daemon=True)
        watchdog.start()
        try:
            return self._run(targets)
        finally:
            stop.set()
            watchdog.join()

    def _run(self, targets):
        order = self.graph.order(targets)
        self.graph.validate(order)
        t0 = time.perf_counter()
        sigs, results = {}, {}
        self.computed = []
        for node_id in order:
            self._check_cancel()
            spec = self.graph.nodes[node_id]
            ntype = REGISTRY[spec["type"]]
            kwargs, sig_parts = {}, [spec["type"]]
            for port in ntype.inputs:
                value = spec.get("inputs", {}).get(port.name, port.default)
                if value is REQUIRED:
                    value = None
                if _is_link(value):
                    src_id, src_port = _split(value["link"])
                    kwargs[port.name] = results[src_id][src_port]
                    sig_parts.append((port.name, "link", sigs[src_id], src_port))
                else:
                    kwargs[port.name] = value
                    sig_parts.append((port.name, _literal_signature(port, value)))
            sig = hashlib.sha1(repr(sig_parts).encode()).hexdigest()
            sigs[node_id] = sig

            cached = self._cache.get(node_id)
            if cached and cached[0] == sig and not ntype.side_effect:
                results[node_id] = cached[1]
                self._emit(type="node_done", node=node_id, time=0.0, cached=True, range=_output_range(cached[1]),
                           messages=cached[2], **_output_extras(cached[1]))
                continue
            start = time.perf_counter()
            self._emit(type="node_start", node=node_id)
            _current.ctx = (self, node_id)
            self._warnings, self._collect = [], None
            try:
                outputs = self._call(node_id, ntype, kwargs)
            except Cancelled:
                raise
            except Exception as err:
                if self._stop_reason:  # the memory watchdog stopped it
                    err = GraphError(self._stop_reason)
                self._emit(type="node_error", node=node_id, message=str(err))
                raise GraphError(f"node '{node_id}' ({ntype.title}) failed: {err}",
                                 problems=[(node_id, str(err))]) from err
            finally:
                _current.ctx = None
                warnings, self._warnings, self._collect = self._warnings, None, None
            results[node_id] = outputs
            self._cache[node_id] = (sig, outputs, warnings)
            self.computed.append(node_id)
            took = time.perf_counter() - start
            self._emit(type="node_done", node=node_id, time=took, cached=False, range=_output_range(outputs),
                       messages=warnings, **_output_extras(outputs))
            if took > 0.05 or ntype.side_effect:
                self.log(f"[{time.perf_counter() - t0:6.1f}s] {node_id} ({ntype.title}) {took:.1f}s")
        return results

    def _call(self, node_id, ntype, kwargs):
        mapped = [p.name for p in ntype.inputs if not p.many and isinstance(kwargs.get(p.name), Many)]
        if not mapped:
            return self._normalise(ntype, ntype.fn(**kwargs))

        lengths = {len(kwargs[n]) for n in mapped}
        if len(lengths) > 1:
            raise GraphError(f"{ntype.title}: list inputs have different lengths {sorted(lengths)}")
        count = lengths.pop()

        def args_for(i):
            return {k: (v[i] if k in mapped else v) for k, v in kwargs.items()}

        if ntype.parallel and self.workers > 1 and count > 1:
            items = self._call_parallel(node_id, ntype, [args_for(i) for i in range(count)])
        else:
            items, outer, self._collect = [], self._collect, []
            try:
                for i in range(count):
                    self._check_cancel()
                    items.append(self._normalise(ntype, ntype.fn(**args_for(i))))
                    self._emit(type="progress", node=node_id, done=i + 1, total=count)
            finally:  # also the lines of the items done before one failed
                lines, self._collect = self._collect, outer
                self._gathered(node_id, lines)
        return {p.name: Many(item[p.name] for item in items) for p in ntype.outputs}

    def _call_parallel(self, node_id, ntype, arg_list):
        from .nodes import grid_memory
        need = 0
        if ntype.estimate is not None:
            try:
                need = max(ntype.estimate(**a) for a in arg_list)
            except Exception:
                need = 0
        uses_gpu = need >= grid_memory(backend.GPU_MIN_VOXELS)  # big enough to run on the GPU
        items = self._map_in_workers(node_id, ntype.fn, arg_list, need, uses_gpu, ntype.title)
        return [self._normalise(ntype, r) for r in items]

    def _map_in_workers(self, node_id, fn, arg_list, need, uses_gpu, title, gpu_need=0):
        # Separate processes: most of the work holds the GIL, so threads barely help. Item i
        # always goes to the same worker, whose caches then stay warm across edits.
        if self.workers <= 1 or len(arg_list) < 2:
            out, outer, self._collect = [], self._collect, []
            try:
                for i, args in enumerate(arg_list):
                    self._check_cancel()
                    out.append(fn(**args))
                    self._emit(type="progress", node=node_id, done=i + 1, total=len(arg_list))
            finally:  # also the lines of the items done before one failed
                lines, self._collect = self._collect, outer
                self._gathered(node_id, lines)
            return out
        gpu = backend.gpu_enabled()
        if self._pool is not None and (not self._pool.alive() or self._pool.gpu != gpu):
            self._pool.close()  # dead, or the GPU was switched on/off since it started
            self._pool = None
        if self._pool is None:
            os.environ["DESIGN_ENGINE_GPU"] = "1" if gpu else "0"  # inherited by the workers
            self._pool = StickyPool(self.workers)
            self._pool.gpu = gpu
        active = min(self._parallel_slots(need), self._pool.n)
        if active < self._pool.n:
            self.log(f"{title}: running {active} at a time to stay within the memory limit")
        if gpu and uses_gpu:
            # Each worker that touches the GPU keeps a CUDA context (~0.8 GB of GPU memory with
            # its work): only as many as the card's free memory allows (workers 0..n-1, so the
            # same ones are reused).
            n_gpu = backend.gpu_worker_count(active, running=self._pool.gpu_contexts, extra=gpu_need)
            if n_gpu < active:
                self.log(f"{title}: {n_gpu} on the GPU at a time (limited by free GPU memory)")
            active = n_gpu
            self._pool.gpu_contexts = max(self._pool.gpu_contexts, active)
        lines = []
        try:
            items = self._pool.map(
                fn, arg_list, active=active,
                on_progress=lambda done, total: self._emit(type="progress", node=node_id, done=done, total=total),
                should_stop=lambda: self._cancel, lines=lines)
        finally:
            self._gathered(node_id, lines)  # also the lines of the items done before one failed
        if items is None:  # cancelled; unless the watchdog killed them, workers keep warm caches
            self._check_cancel()
            raise Cancelled("cancelled")
        return items

    def _parallel_slots(self, need):
        """How many items may run at once: as many workers as fit in the memory still free
        under the limit, given what one item needs."""
        if need <= 0:
            return self.workers
        room = min(memory_limit() - self.memory_in_use(), available_memory() - FREE_TIGHT)
        return max(1, min(self.workers, int(room // need)))

    @staticmethod
    def _normalise(ntype, result):
        if len(ntype.outputs) == 1 and not (isinstance(result, dict) and ntype.outputs[0].name in result):
            return {ntype.outputs[0].name: result}
        return result
