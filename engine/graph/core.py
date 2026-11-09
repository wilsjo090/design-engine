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
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import backend, stages, trace
from ..cache import (COMMIT_LAST_RESORT, COMMIT_STOP, COMMIT_TIGHT, available_memory, commit_room, memory_limit,
                     memory_room, psutil, ram_cache_budget, refresh_memory_limit, resident_memory)
from .workers import StickyPool

REQUIRED = object()

# Two-stage maps (engine/stages.py): a map on the GPU hands its items' CPU stages (decimation, reduction,
# dual contouring) to CPU workers. DESIGN_ENGINE_SPLIT=0 never splits, =1 always (also on the CPU: for
# tests and measuring); otherwise maps that use the GPU do (Evaluator.split overrides it).
SPLIT = {"0": False, "off": False, "no": False, "1": True, "on": True, "yes": True}.get(
    os.environ.get("DESIGN_ENGINE_SPLIT", "").strip().lower(), "auto")
CPU_WORKERS_MAX = int(os.environ.get("DESIGN_ENGINE_CPU_WORKERS") or 10)  # CPU workers at most (Evaluator._cpu_slots)
CPU_WORKER_BYTES = int(0.12 * 2**30)  # RAM of a CPU worker itself (meshing's libraries only: 0.09-0.14 GB resident)

# Free RAM (whole PC) at which the memory watchdog acts. With the page file on a mechanical hard disk
# the PC froze long before memory ran out (with ~1 GB still "free"), so runs stopped at 1.5 GB free.
# Since 2026-10-24 it is on the NVMe SSD: 13 minutes with ~0 GB free stayed responsive, and other
# programs' idle memory went to it at 0.6-1.3 GB/s without slowing anything (FINDINGS, "The PC freeze").
# So low free RAM only frees caches; runs stop on the commit room (cache.COMMIT_*) and the limit, which
# keeps the engine's own data in RAM (31.5 GB read back from the page file took 4-9 minutes). Free RAM
# that stays near 0 means Windows cannot page out fast enough for what is asked of it: then it stops.
FREE_TIGHT = int(1.5 * 2**30)  # free caches, stop idle worker processes
FREE_STOP = int(0.25 * 2**30)  # stop the run when free RAM stays under this ...
STARVED_SECONDS = 30           # ... for this long
LAST_RESORT_SECONDS = 30       # after a stop: end the process if commit stays under COMMIT_LAST_RESORT this long


_current = threading.local()  # (evaluator, node id) of the node running on this thread


def report_progress(done, total, step=None):
    """Called by slow node functions to report progress (and to let a cancel take effect). `step`:
    what is being counted (e.g. "iteration"), shown with the numbers."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        return
    evaluator, node_id = ctx
    evaluator._check_cancel()
    evaluator._emit(type="progress", node=node_id, done=done, total=total, step=step)


def report_finished(text):
    """Called by an optimizing node whose saved result has nothing left to do with these settings, as soon as
    it knows: the card says why at once, not when the result is built (minutes), and offers no Optimize.
    Its first output's `finished` says the same for runs that read the cached node."""
    ctx = getattr(_current, "ctx", None)
    if ctx is not None:
        ctx[0]._emit(type="finished", node=ctx[1], text=text)


def check_cancel():
    """Called inside long node functions: raises Cancelled once the run is cancelled."""
    ctx = getattr(_current, "ctx", None)
    if ctx is not None:
        ctx[0]._check_cancel()


def parallel_map(fn, arg_list, need=0, gpu=False, title="Blocks", gpu_need=0, parts=None, cpu_need=None):
    """Run fn(**args) for every item, in the evaluator's worker processes when called by a node
    running in the main process (one big body split into pieces); one by one inside a worker or
    outside a run. `need`: bytes one item needs (limits how many run at once); `gpu`: the items
    use the GPU (limits how many run at once by free GPU memory, `gpu_need` = extra GPU bytes per
    worker). `parts`: the node's output (a port name) is the list of the results, in order: each is
    passed on as it comes (Evaluator.on_item; the 3D view shows them before the last is done).
    `cpu_need`: bytes an item's CPU stage needs (stages.cpu_stage; sizes the CPU workers), default need."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        with stages.allow(False):  # (inside a worker's item: the results are used here, not handed over)
            return [fn(**a) for a in arg_list]
    evaluator, node_id = ctx
    return evaluator._map_in_workers(node_id, fn, arg_list, need, gpu, title, gpu_need, step=title,
                                     parts=(lambda value: {parts: value}) if parts else None, cpu_need=cpu_need)


def report_view(mesh):
    """Called by long node functions to show an intermediate shape in the 3D view (the editor shows
    it as the node's view; the node's real result replaces it when it is done)."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        return
    evaluator, node_id = ctx
    evaluator._emit(type="live_view", node=node_id, mesh=mesh)


def report_frame(view):
    """Called by Frame Optimization while it works: the frame as it is now (`nodes_frame.frame_view`
    plus a `stage`), drawn in the 3D view in place of the node's bars until its result comes."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        return
    evaluator, node_id = ctx
    evaluator._emit(type="live_frame", node=node_id, frame=view)


def report_message(text, level="info"):
    """Called by node functions to put a line in the editor's log (level: info, warning or error).
    Warnings and errors also stay on the node's card, and come again with every later run that takes
    the node's result from the cache. In a worker process the lines go back with the item's result."""
    ctx = getattr(_current, "ctx", None)
    if ctx is None:
        stages.to_sink(text, level)  # (set by the worker process around each item)
        return
    evaluator, node_id = ctx
    evaluator._message(node_id, text, level)


stages.reporter = report_message  # (stage functions report through engine.stages, which a CPU worker has alone)


def stop_idle_workers():
    """Called by a node running in the main process that needs memory the run's idle worker processes
    hold (RAM, and on the GPU a CUDA context and cached meshes each): stops them, as the memory
    watchdog does when memory is tight; they start again on the next parallel step, their results are
    in the SSD cache. Returns whether any were stopped (False in a worker or outside a run)."""
    ctx = getattr(_current, "ctx", None)
    return ctx is not None and ctx[0]._stop_idle_workers()

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
    # The editor shows the input only while it applies: {other input: [values]} (every one must hold), or
    # a list of such dicts (any one may hold). A linked input, or one whose other input is linked, shows.
    when: Any = None
    advanced: bool = False # tucked under the card's "more" row while it has its default and no link
    # Shown in the editor's number field: "mm", or {other input: {its value: unit}} (a Load's Amount).
    unit: Any = None
    zero: str = None       # what 0 means, shown in place of it ("auto", "no limit", "sharp")
    slider: tuple = None   # (low, high): the usual values, a slider under the field (typing may go beyond)
    # A list in the editor to choose the value from: "body" (an int: which item of the node's first cad input,
    # /api/bodies), "font" (a font file in the Windows Fonts folder, /api/fonts). (Int inputs named seed get a
    # dice button in the editor without it.)
    pick: str = None
    switch: bool = False   # an int that is 1 or 0: shown as an on/off switch

    def __post_init__(self):
        self.label = self.label or self.name.replace("_", " ").title()

    def describe(self):
        d = {"name": self.name, "type": self.type, "label": self.label}
        if self.default is not REQUIRED:
            d["default"] = self.default
        for k in ("doc", "choices", "many", "optional", "hidden", "when", "advanced", "unit", "zero", "slider", "pick",
                  "switch"):
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
    numeric: bool = False      # Math: its scalar output is a plain number when every scalar input is one

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
        d = {"type": self.type_id, "title": self.title, "category": self.category, "doc": self.doc,
             "inputs": [p.describe() for p in self.inputs],
             "outputs": [p.describe() for p in self.outputs]}
        if self.numeric:
            d["numeric"] = True
        return d


REGISTRY: dict = {}


def node(type_id, title, category, inputs, outputs, doc="", parallel=False, side_effect=False, hidden=False,
         estimate=None, paint=None, preview=False, numeric=False):
    """Register a function as a node type. The function receives inputs as keyword arguments
    and returns a dict of outputs, or a single value when the node has one output."""
    def register(fn):
        REGISTRY[type_id] = NodeType(type_id, title, category, inputs, outputs, fn,
                                     doc or (fn.__doc__ or "").strip(), parallel, side_effect, hidden, estimate,
                                     paint, preview, numeric)
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


def bypass_input(spec):
    """The input a bypassed node passes on: its first linked input of the same type as its first
    output (an Offset Body's body, a Multiply's A), as (input name, output name), or None."""
    ntype = REGISTRY.get(spec.get("type"))
    if ntype is None or not ntype.outputs:
        return None
    out = ntype.outputs[0]
    for port in ntype.inputs:
        if port.type == out.type and _is_link(spec.get("inputs", {}).get(port.name)):
            return port.name, out.name
    return None


def _apply_bypass(nodes):
    """Nodes marked "bypass": true pass that input straight on (see bypass_input): every link to
    their first output is replaced by the link into that input, following chains of bypassed nodes.
    The nodes stay (their other outputs still work). Returns {"node.output": "source.port"}."""
    through = {}
    for node_id, spec in nodes.items():
        found = bypass_input(spec) if spec.get("bypass") else None
        if found:
            through[f"{node_id}.{found[1]}"] = spec["inputs"][found[0]]["link"]

    def resolve(ref):
        seen = set()
        while ref in through and ref not in seen:
            seen.add(ref)
            ref = through[ref]
        return ref

    for spec in nodes.values():
        for port, value in spec.get("inputs", {}).items():
            if _is_link(value) and value["link"] in through:
                spec["inputs"][port] = link(resolve(value["link"]))
    return {ref: resolve(ref) for ref in through}


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)  # "<group node>.<output>" -> "<inner node>.<port>"
    groups: dict = field(default_factory=dict)   # group node id -> [(output name, type)]
    owners: dict = field(default_factory=dict)   # node id inside a group -> the (top level) group node
    bypassed: dict = field(default_factory=dict)  # "<bypassed node>.<output>" -> "<source>.<port>" it passes on

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
        bypassed = _apply_bypass(nodes)
        owners = {n: n.split(GROUP_SEP, 1)[0] for n in nodes if n not in top and n.split(GROUP_SEP, 1)[0] in outputs}
        aliases = {ref: bypassed.get(r, r) for ref in aliases if ref.split(".", 1)[0] in outputs
                   for r in [_resolve_alias(ref, aliases)]}
        return cls(nodes, aliases, {n: o for n, o in outputs.items() if n in top}, owners, bypassed)

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
                        if out.type == "scalar" and "number" in _COMPATIBLE.get(port.type, ()) \
                                and REGISTRY[src["type"]].numeric:
                            if not self.gives_number(src_id):
                                problems.append((node_id, f"{port.label} needs a plain number, but '{src_id}' "
                                                          f"gives a field (one of its inputs varies with position)"))
                            continue
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

    def gives_number(self, node_id, _seen=None):
        """Whether a Math node's output is a plain number: every scalar input a literal or a number
        (a Number node, or Math that gives one). Such a node can drive number inputs (a Voxel size)."""
        path = _seen if _seen is not None else set()  # the nodes being checked further down (a cycle)
        spec = self.nodes.get(node_id) or {}
        ntype = REGISTRY.get(spec.get("type"))
        if node_id in path or ntype is None or not ntype.numeric:
            return False
        path.add(node_id)
        try:
            for port in ntype.inputs:
                value = spec.get("inputs", {}).get(port.name)
                if port.type != "scalar" or not _is_link(value):
                    continue
                src_id, src_port = _split(value["link"])
                src = REGISTRY.get((self.nodes.get(src_id) or {}).get("type"))
                out = next((p.type for p in src.outputs if p.name == src_port), None) if src else None
                if out not in ("number", "int") and not (out == "scalar" and self.gives_number(src_id, path)):
                    return False
            return True
        finally:
            path.discard(node_id)

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
    the 3D view; a list for a list of values), `note` (a line shown on the node's card: a list's
    own, else its first item's), `frames`
    (a frame's bars for the editor's frame view), `ground` (a frame's candidate nodes, Show:
    candidate nodes) and `finished` (see report_finished)."""
    if not outputs:
        return {}
    value = next(iter(outputs.values()))
    items = value if isinstance(value, list) else [value]
    found = [m for v in items if (m := getattr(v, "markers", None))]
    notes = [n for v in ([value] if isinstance(value, list) else []) + items if (n := getattr(v, "note", None))]
    frames = [f for v in items if (f := getattr(v, "frame_view", None))]
    grounds = [g for v in items if (g := getattr(v, "ground_view", None))]
    finished = [f for v in items if (f := getattr(v, "finished", None))]
    return {"markers": found or None, "note": notes[0] if notes else None, "frames": frames or None,
            "ground": grounds[0] if grounds else None, "finished": finished[0] if finished else None}


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
    `on_result(node_id, outputs)` gets each node's outputs as soon as they are there (also cached
    ones), `on_item(node_id, index, total, outputs)` each item of a node's output list as it comes
    (a node mapped over a list, or `parallel_map(parts=)`): the server shows them before the run ends.
    """

    def __init__(self, graph, workers=4, log=print, on_event=None):
        self.graph = graph
        self.workers = workers
        self.log = log or (lambda *_: None)
        self.on_event = on_event
        self.on_result = None
        self.on_item = None
        self._cache = {}  # node id -> (signature, outputs, warnings, node type)
        self._warnings = None  # warnings and errors reported by the node being computed
        self._collect = None   # lines of a list's items, gathered to report each line once
        self._cancel = False
        self._stop_reason = None  # set when the memory watchdog stops a run
        self._pool = None  # worker processes, kept alive so later runs start instantly
        self._pool_lock = threading.Lock()  # (the watchdog's thread and a node may both stop them)
        self.computed = []  # node ids actually computed in the last run
        self.failed = []    # run(keep_going=True): a GraphError per node that failed or has a problem
        self.memory_peak = 0
        self._notices = set()  # lines already sent this run (_notice)
        self.split = None  # two-stage maps: None = as SPLIT says (on the GPU), True = always, False = never

    def cancel(self):
        self._cancel = True

    def clear_stop(self):
        """Forget a stop asked for. A server does this before it starts a run's thread, not the run itself
        when it gets going: a stop in between was wiped out there, and whoever asked waited for the whole
        run (Optimize after a changed setting, behind a preview of a topology result: 3 minutes)."""
        self._cancel = False
        self._stop_reason = None

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _stop_idle_workers(self):
        """Stop the worker processes if none is working (see stop_idle_workers); whether it did."""
        with self._pool_lock:
            pool = self._pool
            if pool is None or pool.busy:
                return False
            self._pool = None
        pool.terminate()
        return True

    def release_memory(self):
        """Between runs, when the PC needs memory back: stop the worker processes and forget the
        cached meshes (the biggest results; they are rebuilt, mostly from the SSD cache, on the
        next run). Imported bodies and fields are kept."""
        self.close()
        for node_id, cached in list(self._cache.items()):
            if cached[3].startswith(("mesh.", "output.")):  # (the type as cached: the last run may have used another graph)
                self._cache.pop(node_id, None)

    def forget(self, keep):
        """Drop the cached results of nodes not in `keep` (node ids): nodes deleted from the recipe, or
        of a recipe closed since. Node ids are names recipes share (`import`, `mesh`), so the rest of a
        closed recipe's results otherwise stayed in memory for the life of the server."""
        for node_id in [n for n in self._cache if n not in keep]:
            self._cache.pop(node_id, None)

    def _emit(self, **event):
        if self.on_event:
            self.on_event(event)

    def _result(self, node_id, outputs):
        if self.on_result:
            try:
                self.on_result(node_id, outputs)
            except Exception:  # (showing it early failed: the run goes on, its end shows it)
                traceback.print_exc()

    def _item(self, node_id, index, total, outputs):
        if self.on_item:
            try:
                self.on_item(node_id, index, total, outputs)
            except Exception:
                traceback.print_exc()

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

    def _notice(self, text, node_id=None):
        """A line about how the run goes (slowed down to stay within memory) for the log, not the
        node's card. Each line once per run; safe from the memory watchdog's thread."""
        if text in self._notices:
            return
        self._notices.add(text)
        self.log(text)
        self._emit(type="message", text=text, level="info", **({"node": node_id} if node_id else {}))

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
        """Resident bytes of this process plus its worker processes (0 if psutil is missing): each one's
        private working set (cache.resident_memory), so files they all map (CAD arrays, SSD cache hits) are
        not counted once per process.

        When RAM runs short, Windows pages memory out and this drops while a process keeps
        growing (a runaway import showed 5 GB resident with 23 GB committed). Committed memory is
        no better a measure: loading the engine commits 1.5 GB it never touches (OpenBLAS's buffers,
        in the main process), starting CUDA another 0.6 GB. That case is caught by the commit room
        checks in _watch_memory instead."""
        if psutil is None:
            return 0
        pids = [os.getpid()] + (self._pool.pids() if self._pool is not None else [])
        return sum(resident_memory(pid) for pid in pids)

    def _watch_memory(self, stop):
        """Runs during a run: frees caches when memory gets tight, and as a last resort stops
        the run (killing the workers frees their memory at once) rather than let the PC run
        out of memory."""
        from .nodes import ARRAY_CACHE
        limit, gb = memory_limit(), 2 ** 30
        shrunk, starved_since = False, None
        while not stop.wait(0.5):
            used, free, room = self.memory_in_use(), available_memory(), commit_room()
            self.memory_peak = max(self.memory_peak, used)
            tight = used > 0.85 * limit or free < FREE_TIGHT or room < COMMIT_TIGHT
            # Idle workers hold ~0.6-1 GB each; their cached results are on the SSD anyway,
            # so stop them (they restart on the next parallel step) instead of failing the run.
            if tight and self._stop_idle_workers():
                self._notice(f"Memory {used / gb:.1f} GB of {limit / gb:.1f} GB: stopped idle worker processes")
                continue
            if tight:
                ARRAY_CACHE.set_limit(ARRAY_CACHE.max_bytes // 2)  # moves to the SSD cache
                if self._pool is not None:
                    self._pool.shrink_caches()
                if not shrunk:
                    self._notice(f"Memory {used / gb:.1f} GB of {limit / gb:.1f} GB: freeing caches")
                shrunk = True
            starved_since = (starved_since or time.time()) if free < FREE_STOP else None
            starved = starved_since is not None and time.time() - starved_since > STARVED_SECONDS
            if used > limit or room < COMMIT_STOP or starved:
                why = (f"memory use reached {used / gb:.1f} GB (limit {limit / gb:.1f} GB)" if used > limit else
                       f"the PC can give out only {room / gb:.1f} GB more memory (RAM and page file)"
                       if room < COMMIT_STOP else
                       f"the PC has had under {FREE_STOP / gb:.2f} GB of RAM free for {STARVED_SECONDS} s")
                self._stop_reason = (f"stopped to protect the PC: {why}. Use a larger voxel size or Draft "
                                     f"quality, or close other programs.")
                self._cancel = True
                pool, self._pool = self._pool, None
                if pool is not None:
                    pool.terminate()
                self._last_resort(limit, stop)
                return

    def _reset_cache_budgets(self):
        """Give the RAM caches their share of this run's limit (main process and each worker). The
        watchdog halves them when memory gets tight; before, they then stayed small for the rest of the
        session."""
        from .nodes import ARRAY_CACHE
        pool = self._pool
        main, each = ram_cache_budget(pool.n if pool is not None else 0)
        ARRAY_CACHE.set_limit(main)
        if pool is not None:
            pool._cache_cap.value = each

    def _last_resort(self, limit, stop):
        """The run has been told to stop, but a step that cannot be interrupted may keep growing.
        If memory stays far over the limit, or the PC has almost nothing left to commit, end the whole
        process rather than let other programs' allocations fail (the editor's graph lives in the
        browser, so nothing is lost)."""
        over_since = None
        while not stop.wait(1.0):
            if self.memory_in_use() > 1.25 * limit or commit_room() < COMMIT_LAST_RESORT:
                over_since = over_since or time.time()
                if time.time() - over_since > LAST_RESORT_SECONDS:
                    print("Design Engine: memory is still far over the limit after stopping the run; "
                          "closing to protect the PC. Start it again; your graph is still in the browser.",
                          flush=True)
                    os._exit(3)
            else:
                over_since = None

    def run(self, targets=None, keep_going=False):
        """Outputs of `targets` (default: all sinks) and what they need, by node id. A node that
        fails stops the run with GraphError, unless `keep_going` (previews): then only the nodes
        that need it are left out of the results, and `failed` says why (a support whose faces were
        picked on the previous STEP file must not keep the new part from being shown)."""
        self.memory_peak = 0
        self._notices = set()
        refresh_memory_limit(self.memory_in_use())  # from the RAM free now, fixed for this run
        self._reset_cache_budgets()
        stop = threading.Event()
        watchdog = threading.Thread(target=self._watch_memory, args=(stop,), daemon=True)
        watchdog.start()
        traced = trace.begin_run("run")  # (None when off, or inside a server job, which traces its whole run)
        try:
            return self._run(targets, keep_going)
        finally:
            trace.end_run(traced)
            stop.set()
            watchdog.join()
            self.clear_stop()

    def _run(self, targets, keep_going=False):
        order = self.graph.order(targets)
        self.failed = []
        if keep_going:
            self.failed = [GraphError(f"{n}: {m}", problems=[(n, m)]) for n, m in self.graph.problems(order)]
        else:
            self.graph.validate(order)
        left_out = {n for err in self.failed for n, _ in err.problems}
        t0 = time.perf_counter()
        sigs, results = {}, {}
        self.computed = []
        for node_id in order:
            self._check_cancel()
            spec = self.graph.nodes[node_id]
            if node_id in left_out or any(u not in results for u in self.graph.upstream(node_id)):
                continue  # it has a problem, or needs a node that failed (keep_going)
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
                trace.mark("cached", node=node_id)
                self._emit(type="node_done", node=node_id, time=0.0, cached=True, range=_output_range(cached[1]),
                           messages=cached[2], **_output_extras(cached[1]))
                self._result(node_id, cached[1])
                continue
            start = time.perf_counter()
            self._emit(type="node_start", node=node_id)
            _current.ctx = (self, node_id)
            self._warnings, self._collect = [], None
            try:
                with trace.span("node", node=node_id, type=spec["type"]):
                    outputs = self._call(node_id, ntype, kwargs)
            except Cancelled:
                raise
            except Exception as err:
                if self._stop_reason:  # the memory watchdog stopped it
                    err = GraphError(self._stop_reason)
                self._emit(type="node_error", node=node_id, message=str(err))
                failure = GraphError(f"node '{node_id}' ({ntype.title}) failed: {err}", problems=[(node_id, str(err))])
                if not keep_going or self._stop_reason:
                    raise failure from err
                self.failed.append(failure)
                continue
            finally:
                _current.ctx = None
                warnings, self._warnings, self._collect = self._warnings, None, None
            results[node_id] = outputs
            self._cache[node_id] = (sig, outputs, warnings, spec["type"])
            self.computed.append(node_id)
            took = time.perf_counter() - start
            self._emit(type="node_done", node=node_id, time=took, cached=False, range=_output_range(outputs),
                       messages=warnings, **_output_extras(outputs))
            self._result(node_id, outputs)
            if took > 0.05 or ntype.side_effect:
                self.log(f"[{time.perf_counter() - t0:6.1f}s] {node_id} ({ntype.title}) {took:.1f}s")
        return results

    def _call(self, node_id, ntype, kwargs):
        if ntype.estimate is not None and not ntype.parallel:  # runs here, all at once (Merge Meshes' fuse)
            self._make_room(node_id, ntype, kwargs)
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
                    with trace.span("item", item=i):
                        items.append(self._normalise(ntype, ntype.fn(**args_for(i))))
                    self._item(node_id, i, count, items[-1])
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
        items = self._map_in_workers(node_id, ntype.fn, arg_list, need, uses_gpu, ntype.title,
                                     parts=lambda result: self._normalise(ntype, result))
        return [self._normalise(ntype, r) for r in items]

    def _splits(self, on_gpu):
        """Whether a map hands its items' CPU stages to CPU workers: when it runs on the GPU (unless
        DESIGN_ENGINE_SPLIT or `split` say otherwise). On the CPU every worker does both stages anyway."""
        split = self.split if self.split is not None else SPLIT
        return bool(on_gpu) if split == "auto" else bool(split)

    def _cpu_slots(self, need, cpu_need, active, items):
        """How many CPU workers a splitting map may use: the cores the GPU workers and this process leave,
        CPU_WORKERS_MAX at most (16 cores are full at ~14 working processes, and then each stage just takes
        longer: hex studs' block reductions took 49 ms each with 8 workers, 56 with 8 GPU + 6 CPU workers, 104
        with 8 + 12, meshing 17.4 and 17.2 s; the lamp's petals with 4 GPU workers 10.4 s with 10 CPU workers
        and with 12), and as many as fit in the memory still free under the limit beside the map's GPU workers
        (`need` each): a new one CPU_WORKER_BYTES plus a stage (`cpu_need`), one already running a stage."""
        pool = self._pool
        cores = (os.cpu_count() or 4) - active - 1
        have = pool.cpu_started()
        room = min(memory_limit() - self.memory_in_use(), memory_room()) - active * need - have * cpu_need
        more = max(0, int(room // (CPU_WORKER_BYTES + cpu_need)))
        # ... and new ones only in the RAM that is free now: CPU workers only speed a map up, and the limits above
        # allow paging out (commit room). The lamp's Final in the editor beside the user's other programs: 8 GPU
        # + 10 CPU workers took the PC under 2 GB free (2026-11-09), where paging costs more than they gain
        ram = available_memory() - FREE_TIGHT - active * need
        more = max(0, min(more, int(ram // (CPU_WORKER_BYTES + cpu_need))))
        return max(0, min(CPU_WORKERS_MAX, cores, items, have + more))

    def _map_in_workers(self, node_id, fn, arg_list, need, uses_gpu, title, gpu_need=0, step=None, parts=None,
                        cpu_need=None):
        # Separate processes: most of the work holds the GIL, so threads barely help. Item i
        # always goes to the same worker, whose caches then stay warm across edits.
        # `parts(result)`: the node's outputs of one item, when the results are its output list
        # (passed on to on_item as each comes)
        if self.workers <= 1 or len(arg_list) < 2:
            out, outer, self._collect = [], self._collect, []
            try:
                for i, args in enumerate(arg_list):
                    self._check_cancel()
                    with trace.span("item", item=i):
                        out.append(fn(**args))
                    if parts and self.on_item:
                        self._item(node_id, i, len(arg_list), parts(out[-1]))
                    self._emit(type="progress", node=node_id, done=i + 1, total=len(arg_list), step=step)
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
        active = slots = min(self._parallel_slots(need), self._pool.n)
        if active < self._pool.n:
            self._notice(f"{title}: running {active} at a time to stay within the memory limit", node_id)
        if gpu and uses_gpu:
            # Each worker that touches the GPU keeps a CUDA context (~0.8 GB of GPU memory with
            # its work): only as many as the card's free memory allows (workers 0..n-1, so the
            # same ones are reused).
            n_gpu = backend.gpu_worker_count(active, running=self._pool.gpu_contexts, extra=gpu_need)
            if n_gpu < active:
                self._notice(f"{title}: {n_gpu} on the GPU at a time (limited by free GPU memory)", node_id)
            active = n_gpu
            self._pool.gpu_contexts = max(self._pool.gpu_contexts, active)
        # Items whose function ends in a CPU stage (stages.cpu_stage) hand it to CPU workers, so the few
        # GPU workers go on to their next item at once (started when the first stage is handed over).
        cpu = 0
        if self._splits(gpu and uses_gpu):
            cpu = self._cpu_slots(need, need if cpu_need is None else cpu_need, active, len(arg_list))
        # (memory: how many the memory limit allows; active: how many run, fewer when the GPU limits them;
        # cpu: CPU workers for the items' CPU stages, 0 = no split)
        trace.mark("map plan", items=len(arg_list), workers=self._pool.n, memory=slots, active=active,
                   gpu=bool(gpu and uses_gpu), cpu=cpu, step=title)
        lines = []
        try:
            items = self._pool.map(
                fn, arg_list, active=active, split=cpu > 0, cpu_workers=cpu,
                on_progress=lambda done, total: self._emit(type="progress", node=node_id, done=done, total=total,
                                                           step=step),
                should_stop=lambda: self._cancel, lines=lines,
                on_item=(lambda i, result: self._item(node_id, i, len(arg_list), parts(result)))
                if parts and self.on_item else None)
        finally:
            self._gathered(node_id, lines)  # also the lines of the items done before one failed
        if items is None:  # cancelled; unless the watchdog killed them, workers keep warm caches
            self._check_cancel()
            raise Cancelled("cancelled")
        return items

    def _make_room(self, node_id, ntype, kwargs):
        """Before a node that runs in this process and needs much memory at once (its estimate): stop the
        run's idle worker processes when it does not fit beside them (they held ~3 GB while the lamp's
        fuse needed 5-10 GB), and stop with a readable error when it does not fit even then, rather than
        the watchdog stopping the run halfway or Windows paging to its hard disk."""
        try:
            need = int(ntype.estimate(**kwargs))
        except Exception:
            return
        room = lambda: min(memory_limit() - self.memory_in_use(), memory_room())
        if need <= 0 or need <= room():
            return
        gb = 2 ** 30
        if self._stop_idle_workers():
            self._notice(f"{ntype.title}: stopped idle worker processes to make room ({need / gb:.1f} GB needed)",
                         node_id)
        left = room()
        if need > left:
            raise GraphError(f"{ntype.title} needs about {need / gb:.1f} GB of memory at once but only "
                             f"{max(left, 0) / gb:.1f} GB is left (limit {memory_limit() / gb:.1f} GB, the PC "
                             f"can give out {commit_room() / gb:.1f} GB more). Close other programs, or use "
                             f"fewer triangles (Keep or Tolerance in Mesh from Implicit Body).")

    def _parallel_slots(self, need):
        """How many items may run at once: as many workers as fit in the memory still free
        under the limit, given what one item needs."""
        if need <= 0:
            return self.workers
        room = min(memory_limit() - self.memory_in_use(), memory_room())
        return max(1, min(self.workers, int(room // need)))

    @staticmethod
    def _normalise(ntype, result):
        if len(ntype.outputs) == 1 and not (isinstance(result, dict) and ntype.outputs[0].name in result):
            return {ntype.outputs[0].name: result}
        return result
