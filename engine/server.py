"""Local web server behind the node editor (frontend/).

The browser owns the graph and sends it with every run request. The server keeps one Evaluator
alive, so unchanged nodes (and cached field arrays) are reused between edits, and streams run
events over a WebSocket. Meshes for the 3D view are fetched as compact binary blobs.

Binary view format (little endian):  uint32 vertex_count, uint32 triangle_count,
float32 positions[vertex_count * 3], uint32 indices[triangle_count * 3].
Surface colours ("paint:<node>"): float32 value per vertex of that view (NaN = no value).

Exports queue up: an export never cancels another, and preview requests that arrive during an
export wait until the queue is done (only the newest one runs).
"""
import asyncio
import copy
import gc
import hashlib
import itertools
import json
import os
import re
import string
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .graph import REGISTRY, Cancelled, Evaluator, Graph, GraphError, compatibility, node_catalog
from .graph.core import FREE_TIGHT, GROUP_SEP
from . import backend
from .cache import available_memory, default_cache_dir, memory_limit
from .slice import field_slice

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
PRESETS = ROOT / "presets"
DIST = ROOT / "frontend" / "dist"

# quality -> (multiplier for mesh-node voxel sizes, cells along the longest side for body previews)
QUALITY = {"draft": (4.0, 128), "medium": (2.0, 256), "final": (1.0, 512)}
LEVELS = list(QUALITY)  # coarse to fine
# node types build_run_graph changes with the quality: what needs one shows more detail at a finer quality
QUALITY_TYPES = ("mesh.from_implicit", "mesh.merge")
DISPLAYABLE = ("mesh", "cad", "implicit")
# simple shapes the editor draws itself (Viewport's guide shapes), so they follow a drag at once
DRAWN_BY_VIEWPORT = ("implicit.ball", "implicit.bar", "implicit.box")
# The 3D view gets a decimated copy above this. The browser draws ~12M triangles fine on this
# PC; the careful decimation used for files reached only 10.6M of a 2.5M target on the lamp and
# took 24 s, so display copies use the fast setting and only when really needed.
DISPLAY_MAX_FACES = 15_000_000
# A field painted on surfaces is sampled on a grid with this many cells along the longest side
# of the bodies (at most PAINT_MAX_VOXELS in all) and interpolated at their vertices: smooth
# fields (masks, noise) come out right, sharp creases finer than ~2 cells are softened.
PAINT_CELLS = {"draft": 128, "medium": 192, "final": 256}
PAINT_MAX_VOXELS = 12_000_000
PRESET_NAME = re.compile(r"^[\w\- ]{1,60}$")
LONG_JOBS = ("export", "optimize")  # queued, never cancelled by previews (which wait for them)


class Hub:
    """Fans events out to every connected WebSocket. publish() is safe from any thread."""

    def __init__(self):
        self.clients = set()
        self.loop = None

    def publish(self, event):
        if self.loop is None:
            return
        for queue in list(self.clients):
            self.loop.call_soon_threadsafe(queue.put_nowait, event)


def _is_link(value):
    return isinstance(value, dict) and set(value) == {"link"}


def _wired(graph, node_id):
    """The node and everything it needs are connected (a half-wired node is left out of previews)."""
    try:
        return not graph.problems(graph.order([node_id]))
    except GraphError:
        return False


def _set_compute(graph, mode, optimize, stamp):
    """Hidden inputs of nodes that run long jobs only on request (Topology Optimization): compute
    for the node being optimized (and in exports), stamp so their results are read again after a job."""
    for node_id, spec in graph.nodes.items():
        ntype = REGISTRY.get(spec.get("type"))
        if ntype is not None and any(p.name == "compute" and p.hidden for p in ntype.inputs):
            inputs = spec.setdefault("inputs", {})
            inputs["compute"] = 1 if (mode == "export" or node_id == optimize) else 0
            inputs["stamp"] = int(stamp)


class ViewSpec(NamedTuple):
    """A visible node's view in a run: the node id and port holding its displayable output, the
    output's type, the (node, output) colouring it (Static Analysis) or None, what it is made of
    (`key`, the same at every quality), whether it looks the same at every quality (`exact`), and
    whether the editor already shows it at this quality or finer, so the run leaves it out (`kept`)."""
    src: str
    port: str
    type: str
    paint: tuple | None
    key: str
    exact: bool
    kept: bool = False


def _content_key(graph, inputs, src, port, extra=""):
    """What the output src.port is made of: type and inputs of every node it needs (`inputs`: a node's
    inputs before the quality changed them), not positions or other editor data. Equal keys give
    equal results at one quality."""
    data = [[n, graph.nodes[n].get("type"), inputs.get(n, graph.nodes[n].get("inputs", {}))]
            for n in graph.order([src])]
    blob = json.dumps([data, src, port, extra], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def build_run_graph(graph_dict, mode, quality, visible, optimize=None, stamp=0, have=None):
    """The graph actually evaluated: preview quality applied to mesh nodes, plus hidden
    `mesh.view` nodes for visible implicit bodies. Returns (graph, targets, views, slices) where
    views maps a visible node id to its ViewSpec and slices maps a visible field node to the
    (node id, port) of the field. A visible group node shows its first displayable output (or its
    field). `have`: {node: {key, level}} of the views the editor shows; one with the same key at
    this quality or finer is kept (not computed again)."""
    multiplier, cells = QUALITY[quality]
    graph = Graph.from_dict(copy.deepcopy(graph_dict))
    voxel_default = REGISTRY["mesh.from_implicit"].input("voxel").default
    before = {n: copy.deepcopy(s.get("inputs", {})) for n, s in graph.nodes.items() if s.get("type") in QUALITY_TYPES}
    for spec in graph.nodes.values():
        if spec.get("type") == "mesh.from_implicit":
            inputs = spec.setdefault("inputs", {})
            voxel = inputs.get("voxel", voxel_default)
            if not _is_link(voxel):
                inputs["voxel"] = float(voxel) * multiplier
            if quality != "final":
                inputs["keep"] = 1.0  # decimation only matters for the files you export
                inputs["tolerance"] = 0.0
        elif spec.get("type") == "mesh.merge" and quality != "final":
            # looks the same unfused, and coarse draft meshes may not overlap enough to join
            spec.setdefault("inputs", {})["fuse"] = "no"

    _set_compute(graph, mode, optimize, stamp)
    if mode == "optimize":
        if optimize not in graph.nodes:
            raise GraphError(f"no node '{optimize}' to optimize")
        return graph, [optimize], {}, {}
    targets, views, slices = [], {}, {}
    if mode == "export":
        targets = [n for n, s in graph.nodes.items() if s.get("type") in REGISTRY and REGISTRY[s["type"]].side_effect]
    else:  # supports and loads: their markers are drawn whether or not anything uses them yet
        targets = [n for n, s in graph.nodes.items() if s.get("type") in REGISTRY and REGISTRY[s["type"]].preview
                   and _wired(graph, n)]
    for node_id in visible:
        own_paint = None  # (node, output) colouring this node's body (Static Analysis)
        if node_id in graph.groups:  # a group node: its outputs are inner nodes' outputs
            outputs = [(name, typ, *graph.aliases[f"{node_id}.{name}"].split(".", 1)) for name, typ in graph.groups[node_id]]
        else:
            spec = graph.nodes.get(node_id)
            if spec is None or spec.get("type") not in REGISTRY:
                continue
            ntype = REGISTRY[spec["type"]]
            if spec["type"] in DRAWN_BY_VIEWPORT:  # (exactly, and live while dragged: only checked here)
                targets.append(node_id)
                continue
            outputs = [(p.name, p.type, node_id, p.name) for p in ntype.outputs]
            if ntype.paint:
                own_paint = (node_id, ntype.paint)
        out = next((o for o in outputs if o[1] in DISPLAYABLE), None)
        if out is None:
            field_out = next((o for o in outputs if o[1] == "scalar"), None)
            if field_out is not None:  # a plain field: shown as a coloured slice plane
                slices[node_id] = (field_out[2], field_out[3])
                targets.append(field_out[2])
            continue
        _, out_type, src, port = out
        key = _content_key(graph, before, src, port, out_type)
        # a body meshed for the view, or coloured on a grid, gets finer with the quality
        exact = out_type != "implicit" and own_paint is None and not any(
            graph.nodes[n].get("type") in QUALITY_TYPES for n in graph.order([src]))
        shown = (have or {}).get(node_id) or {}
        kept = shown.get("key") == key and shown.get("level") in LEVELS and \
            LEVELS.index(shown["level"]) >= LEVELS.index(quality) and mode == "preview"
        if out_type == "implicit":
            view_id = f"__view__{node_id}"
            graph.nodes[view_id] = {"type": "mesh.view", "inputs": {"body": {"link": f"{src}.{port}"}, "cells": cells}}
            views[node_id] = ViewSpec(view_id, "mesh", out_type, own_paint, key, exact, kept)
            if not kept:
                targets.append(view_id)
        else:
            views[node_id] = ViewSpec(src, port, out_type, own_paint, key, exact, kept)
            if not kept:
                targets.append(src)
    return graph, list(dict.fromkeys(targets)), views, slices


SLICE_RES = {"draft": 600, "medium": 1000, "final": 1600}  # pixels along the longer side; slices are cheap (2D)


def pack_slice(image):
    """uint32 width, uint32 height, float32 values row by row (NaN = not covered)."""
    h, w = image.shape
    return np.array([w, h], dtype=np.uint32).tobytes() + np.ascontiguousarray(image, dtype=np.float32).tobytes()


def display_arrays(value):
    """(vertices float32, triangles uint32) of a mesh or a list of meshes, as shown in the 3D view
    (decimated if too big to display)."""
    items = list(value) if isinstance(value, list) else [value]
    total = sum(len(m.faces) for m in items)
    keep = min(1.0, DISPLAY_MAX_FACES / total) if total else 1.0
    verts, faces, offset = [], [], 0
    for m in items:
        if keep < 1.0:
            m = m.simplify_quadric_decimation(percent=1.0 - keep, aggression=7)  # display only
        verts.append(np.asarray(m.vertices, dtype=np.float32))
        faces.append(np.asarray(m.faces, dtype=np.uint32) + offset)
        offset += len(m.vertices)
    V = np.concatenate(verts) if verts else np.zeros((0, 3), np.float32)
    F = np.concatenate(faces) if faces else np.zeros((0, 3), np.uint32)
    return V, F


def pack_meshes(value):
    """One binary blob for a mesh or a list of meshes. Returns (blob, vertices, triangles)."""
    V, F = display_arrays(value)
    header = np.array([len(V), len(F)], dtype=np.uint32)
    return header.tobytes() + V.tobytes() + F.tobytes(), len(V), len(F)


def unpack_meshes(blob):
    """(vertices, triangles) of a packed view, as views into the blob (no copy)."""
    nv, nf = (int(x) for x in np.frombuffer(blob, dtype=np.uint32, count=2))
    V = np.frombuffer(blob, dtype=np.float32, count=nv * 3, offset=8).reshape(-1, 3)
    F = np.frombuffer(blob, dtype=np.uint32, count=nf * 3, offset=8 + nv * 12).reshape(-1, 3)
    return V, F


def pack_faces(value, shown):
    """See /api/faces. b"" when the view's triangles are not the bodies' own (e.g. decimated
    for display) or the bodies carry no CAD faces."""
    from .facemask import cad_body, cad_faces, same_part, tangent_groups
    items = list(value) if isinstance(value, list) else [value]
    ids = [cad_faces(m) for m in items]
    if not items or any(i is None for i in ids) or sum(len(i) for i in ids) != shown:
        return b""
    # copies of one part (the petals, also when scaled: same faces, same triangles per face, same
    # share of the surface per face) share face numbers, so a plain number picks that face on all
    # of them; different parts need "body:face"
    same = same_part(items, ids)
    bodies = [cad_body(m, k) for k, m in enumerate(items)]
    groups = [tangent_groups(items[0], ids[0])] if same else [tangent_groups(m, i) for m, i in zip(items, ids)]
    out = [np.array([shown], dtype=np.uint32), *[i.astype(np.uint32) for i in ids],
           *[np.full(len(i), b, dtype=np.uint32) for i, b in zip(ids, bodies)],
           np.array([int(same), len(groups)], dtype=np.uint32)]
    for g in groups:
        out += [np.array([len(g)], dtype=np.uint32), g.astype(np.uint32)]
    return b"".join(a.tobytes() for a in out)


def view_voxel(value):
    """The voxel a shown mesh (or list) was made with, if marching cubes made it (None after
    decimation or merging)."""
    items = list(value) if isinstance(value, list) else [value]
    found = [getattr(m, "metadata", {}).get("voxel") for m in items]
    return float(min(found)) if found and all(found) else None


def detail_floor(graph, src, quality):
    """The finest voxel the editor's detail of a view made by `src` may use, or None when the view
    gets no detail: a body meshed for the view goes down to DETAIL_MIN_VOXEL, a Mesh from Implicit
    Body to its own voxel (finer would show what its file will not have)."""
    from .graph.nodes import DETAIL_MIN_VOXEL
    spec = graph.nodes.get(src) or {}
    inputs = spec.get("inputs", {})
    if not _is_link(inputs.get("body")):
        return None
    if spec.get("type") == "mesh.view":
        return DETAIL_MIN_VOXEL
    if spec.get("type") == "mesh.from_implicit" and inputs.get("domain") is None:
        voxel = inputs.get("voxel", REGISTRY["mesh.from_implicit"].input("voxel").default)
        return None if _is_link(voxel) else max(float(voxel) / QUALITY[quality][0], DETAIL_MIN_VOXEL)
    return None


def cad_sources(graph, node_id, port):
    """The CAD bodies the output `node_id.port` is built from, as [(node id, output port)]: walks up
    the input links and stops on each path at the first output of type cad (for a textured body:
    Pick Body's `body`, not the Import STEP behind it). Nodes inside a group are left out: the editor
    cannot link to them."""
    found, seen, todo = [], set(), [(node_id, port)]
    while todo:
        node, out = todo.pop(0)
        spec = graph.nodes.get(node) or {}
        ntype = REGISTRY.get(spec.get("type"))
        port_type = next((p.type for p in ntype.outputs if p.name == out), None) if ntype else None
        if port_type == "cad":
            if (node, out) not in found and node not in graph.owners:
                found.append((node, out))
        elif ntype is not None and node not in seen:
            seen.add(node)
            todo += [tuple(v["link"].partition(".")[::2]) for v in spec.get("inputs", {}).values() if _is_link(v)]
    return found


def _range(values):
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)


class Session:
    def __init__(self, hub):
        self.hub = hub
        workers = max(2, min(8, (os.cpu_count() or 4) // 2))
        self.evaluator = Evaluator(Graph(), workers=workers, log=None, on_event=self._on_event)
        self.thread = None
        self.current = None   # the job running (or last run)
        self.queue = []       # exports waiting
        self.pending = None   # the newest preview asked for during an export
        self.run_id = 0
        self.views = {}  # node id -> (version, bytes); also "slice:<node>" and "paint:<node>"
        self.view_faces = {}  # node id -> (version, CAD bodies shown, triangles shown, packed faces or None)
        self.face_infos = {}  # (node id, view version, body, face) -> /api/faceinfo answer (newest few)
        self.view_sources = {}  # node id -> (version, graph of that run, node id, port, quality): what its view shows
        self.view_version = 0
        self._job_ids = itertools.count(1)
        self._owners = {}     # node id inside a group -> group node (events are shown on the group)
        self._group_out = set()  # inner nodes that give a group's outputs
        self._group_warnings = {}  # group node -> warnings of its inner nodes in this run
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._requests = 0
        self.stamp = 0  # bumped after optimize/export jobs: previews then read optimization results again

    def _on_event(self, event):
        node = event.get("node", "")
        if event.get("type") == "live_view":  # a shape while optimizing, shown as the node's view
            node = self._owners.get(node, node)
            data, nv, nf = pack_meshes(event["mesh"])
            with self._lock:
                self.view_version += 1
                self.views[node] = (self.view_version, data)
                self.view_faces.pop(node, None)
                version = self.view_version
            self.hub.publish({"type": "view_ready", "node": node, "version": version, "verts": nv, "faces": nf,
                              "kind": "mesh", "live": True, "run": self.run_id})
            return
        detail = (self.current or {}).get("mode") == "detail"
        if detail and event.get("type") == "node_done" and event.get("cached"):
            return  # a detail reads what the view's run made: the nodes' cards stay as they are
        if node.startswith(("__view__", "__detail__")):  # report hidden preview nodes as the node they show
            event = dict(event, node=node.split("__", 2)[2], view=True)
        elif node in self._owners:
            # inner steps of a group are shown on the group node; it is done when its output is,
            # and then carries the warnings of all its steps so far
            group, inner = self._owners[node], node.split(GROUP_SEP, 1)[1]
            if event.get("type") == "node_done":
                found = self._group_warnings.setdefault(group, [])
                for text, level in event.get("messages") or []:
                    if [f"{inner}: {text}", level] not in found:
                        found.append([f"{inner}: {text}", level])
                if node not in self._group_out:
                    return
                event = dict(event, messages=list(found))
            event = dict(event, node=group, inner=inner)
        self.hub.publish(dict(event, run=self.run_id))

    def _left_out(self, graph, targets, results):
        """Nodes a preview left out (they failed, or need a node that did) as the editor knows them:
        it stops showing their last results (an old part's shape and loads on a new STEP file)."""
        out = set()
        for n in graph.order(targets):
            if n in results:
                continue
            if n.startswith("__view__"):
                out.add(n[len("__view__"):])
            elif n not in self._owners:
                out.add(n)
            elif n in self._group_out:
                out.add(self._owners[n])
        return sorted(out)

    def _problems(self, problems):
        """Problems of nodes inside groups are shown on the group node."""
        out = []
        for n, m in problems:
            if n in self._owners:
                out.append((self._owners[n], f"{n.split(GROUP_SEP, 1)[1]}: {m}"))
            elif not (n or "").startswith("__view__"):
                out.append((n, m))
        return out

    # --- runs and the export queue ------------------------------------------------------
    def start(self, graph_dict, mode, quality, visible, slice_settings=None, label=None, optimize=None, have=None,
              tag=None):
        """Run (or queue) a job. `have`: {node: {key, level, version}} of the views the editor shows (a
        preview keeps those it would compute the same at this quality or coarser); `tag`: anything the
        editor wants back with the job's run_start (it sends its graph version)."""
        if quality not in QUALITY:
            raise HTTPException(400, f"unknown quality '{quality}'")
        # only views the editor and this server still agree on (the server may have restarted)
        have = {n: h for n, h in (have or {}).items()
                if isinstance(h, dict) and self.views.get(n, (None,))[0] == h.get("version")}
        try:
            graph, targets, views, slices = build_run_graph(graph_dict, mode, quality, visible, optimize, self.stamp,
                                                            have)
        except Exception as err:
            raise HTTPException(400, f"bad graph: {err}")
        job = {"id": next(self._job_ids), "graph": graph, "targets": targets, "views": views, "slices": slices,
               "slice": slice_settings or {}, "mode": mode, "quality": quality, "label": label or mode,
               "queued": time.time(), "tag": tag, "have": have}
        with self._lock:
            self._requests += 1
            ticket = self._requests
        # One run at a time: requests that arrive while a run is still stopping wait here instead
        # of each starting a run (two heavy runs side by side once filled the PC's memory), and
        # only the newest preview of them runs. Exports are never dropped or cancelled by others.
        with self._start_lock:
            if self.exporting():
                if mode in LONG_JOBS:
                    self.queue.append(job)
                else:
                    self.pending = job  # runs when the exports are done
                self._publish_queue()
                return self.run_id
            if mode not in LONG_JOBS and ticket != self._requests:
                return self.run_id
            self.cancel()
            self._launch(job)
            return self.run_id

    def detail(self, node_id, version, lo, hi, voxel, tag=None):
        """Mesh the part lo..hi of the view of `node_id` (its `version`) finer, at `voxel` mm (not finer
        than its detail_floor), from the results of the run that made the view. Only while nothing else
        runs or waits: a detail never stops a preview, an export or an optimization (it returns busy;
        the editor asks again when they are done), but a new preview stops it."""
        with self._lock:
            item = self.view_sources.get(node_id)
        if item is None or item[0] != version:
            raise HTTPException(409, f"the view of {node_id} has changed")
        _, graph, src, _, quality = item
        floor = detail_floor(graph, src, quality)
        if floor is None:
            raise HTTPException(409, f"the view of {node_id} cannot be shown finer")
        try:
            lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
            box = lo.shape == (3,) and hi.shape == (3,) and np.all(np.isfinite([*lo, *hi])) and np.all(hi > lo)
        except (TypeError, ValueError):
            box = False
        if not box:
            raise HTTPException(422, "lo and hi must be the corners of a box")
        detail = dict(graph.nodes)
        detail_id = f"__detail__{node_id}"
        detail[detail_id] = {"type": "mesh.detail", "inputs": {
            "body": graph.nodes[src]["inputs"]["body"], "lo": lo.tolist(), "hi": hi.tolist(),
            "voxel": max(float(voxel), floor)}}
        detail_graph = copy.copy(graph)
        detail_graph.nodes = detail
        job = {"id": next(self._job_ids), "graph": detail_graph, "targets": [detail_id], "views": {}, "slices": {},
               "slice": {}, "mode": "detail", "quality": quality, "label": "detail", "queued": time.time(),
               "tag": tag, "have": {}, "detail": {"node": node_id, "view_version": version, "lo": lo.tolist(),
                                                  "hi": hi.tolist(), "id": detail_id}}
        with self._start_lock:
            current = self.current or {}
            running = self.thread is not None and self.thread.is_alive() and not current.get("done")
            if self.queue or self.pending is not None or (running and current.get("mode") != "detail"):
                return {"run": None, "busy": True}
            self.cancel()  # an older detail
            self._launch(job)
            return {"run": self.run_id}

    def exporting(self):
        job = self.current or {}
        return self.thread is not None and self.thread.is_alive() and job.get("mode") in LONG_JOBS and not job.get("done")

    def _launch(self, job):
        graph = job["graph"]
        with self._lock:
            self.run_id += 1
            run_id = self.run_id
            job["run"], job["started"] = run_id, time.time()
            self.current = job
            if job["mode"] != "export":
                self.pending = None  # an older waiting preview is out of date now
            self._owners = dict(graph.owners)
            self._group_out = {graph.aliases[f"{g}.{name}"].split(".", 1)[0]
                               for g, outs in graph.groups.items() for name, _ in outs}
            self._group_warnings = {}
        self.thread = threading.Thread(target=self._run, args=(run_id, job), daemon=True)
        self.thread.start()
        self._publish_queue()

    def _advance(self, finished):
        """After a run: start the next queued export, or the preview that waited for them."""
        finished.join()
        with self._start_lock:
            if self.thread is not finished and self.thread is not None and self.thread.is_alive():
                return  # something else has started meanwhile
            job = self.queue.pop(0) if self.queue else self.pending
            if job is self.pending:
                self.pending = None
            if job is not None:
                self._launch(job)
            else:
                self._publish_queue()

    def remove_queued(self, job_id):
        with self._start_lock:
            self.queue = [j for j in self.queue if j["id"] != job_id]
            self._publish_queue()

    def queue_state(self):
        current = self.current if self.exporting() else None
        job = self.current or {}
        alive = self.thread is not None and self.thread.is_alive() and not job.get("done")
        return {"type": "queue",
                "current": {"id": current["id"], "label": current["label"]} if current else None,
                "exports": [{"id": j["id"], "label": j["label"], "queued": j["queued"]} for j in self.queue],
                "preview_waiting": self.pending is not None,
                # the run going on now (any kind): an editor opened during it shows it in its status bar
                "running": {"run": job["run"], "mode": job["mode"], "quality": job["quality"], "label": job["label"],
                            "tag": job.get("tag"), "elapsed": time.time() - job["started"]} if alive else None}

    def _publish_queue(self):
        self.hub.publish(self.queue_state())

    def cancel(self):
        if self.thread is not None and self.thread.is_alive():
            self.evaluator.cancel()
            self.thread.join()

    def watch_idle_memory(self, stop, interval=2.0):
        """While no run is going on, give memory back when the PC runs low because another
        program needs it: an idle editor once held ~9 GB while a test run filled the rest, and
        the PC froze paging to a hard disk. (During runs the evaluator's watchdog does this.)"""
        from .graph.nodes import ARRAY_CACHE
        released = False
        while not stop.wait(interval):
            low = available_memory() < FREE_TIGHT
            if not low:
                released = False
                continue
            if released or not self._start_lock.acquire(blocking=False):
                continue
            try:
                if self.thread is not None and self.thread.is_alive():
                    continue
                self.evaluator.release_memory()
                ARRAY_CACHE.clear()  # RAM tier only; the arrays stay in the SSD cache
                gc.collect()
                released = True
                self.hub.publish({"type": "message", "level": "info",
                                  "text": "The PC is low on memory: the engine released its worker processes and "
                                          "cached meshes (they are rebuilt on the next update)."})
            finally:
                self._start_lock.release()

    def _run(self, run_id, job):
        try:
            self._run_job(run_id, job)
        finally:
            if job["mode"] in LONG_JOBS:
                self.stamp += 1
                if job["mode"] == "optimize":
                    self.hub.publish({"type": "optimized", "run": run_id})  # the editor updates its preview
            job["done"] = True  # the thread is still alive for a moment: the queue must not show it
            if self.queue or self.pending is not None:
                threading.Thread(target=self._advance, args=(threading.current_thread(),), daemon=True).start()
            elif job["mode"] == "export":
                self._publish_queue()

    def _shown_id(self, node_id):
        """A node as the editor knows it: hidden view meshers as the node they show, nodes inside a
        group as the group."""
        if node_id.startswith(("__view__", "__detail__")):
            return node_id.split("__", 2)[2]
        return self._owners.get(node_id, node_id)

    def _run_job(self, run_id, job):
        graph, targets, views, mode, quality = job["graph"], job["targets"], job["views"], job["mode"], job["quality"]
        t0 = time.perf_counter()
        publish = lambda **e: self.hub.publish(dict(e, run=run_id))
        # nodes behind views the editor keeps from a finer run, and not needed now: their cards keep
        # what they showed
        kept = [v.src for v in views.values() if v.kept]
        needed = set(graph.order(targets)) if targets else set()
        keep_status = sorted({self._shown_id(n) for n in graph.order(kept) if n not in needed}) if kept else []
        publish(type="run_start", mode=mode, quality=quality, label=job["label"], tag=job.get("tag"),
                keep_status=keep_status)
        if mode != "detail":  # (a detail runs the graph of the run that made its view, maybe not the newest)
            publish(type="problems", problems=self._problems(graph.problems()))
        # the coarsest quality of what the editor shows after this run (bodies meshed or coloured for the
        # view, field slices and colours; None: nothing that changes with the quality): finer runs up to
        # it would change nothing, so refining goes on from there
        levels = [LEVELS.index(job["have"][n]["level"]) if v.kept else LEVELS.index(quality)
                  for n, v in views.items() if not v.exact]
        if job["slices"]:
            levels.append(LEVELS.index(quality))
        shown = LEVELS[min(levels)] if levels else None
        sharper = shown is not None and shown != LEVELS[-1]  # would a finer quality show more?
        if not targets:
            publish(type="run_done", ok=True, time=0.0, files={}, sharper=sharper, shown=shown)
            return
        self.evaluator.graph = graph
        try:  # previews show what still works when a node fails (exports and optimizations stop)
            results = self.evaluator.run(targets, keep_going=mode == "preview")
        except Cancelled:
            publish(type="run_done", ok=False, cancelled=True, time=time.perf_counter() - t0)
            return
        except GraphError as err:
            publish(type="run_done", ok=False, error=str(err), problems=self._problems(err.problems or []),
                    time=time.perf_counter() - t0)
            return
        except Exception as err:
            traceback.print_exc()
            publish(type="run_done", ok=False, error=f"{type(err).__name__}: {err}",
                    time=time.perf_counter() - t0)
            return

        if job.get("detail"):
            d = job["detail"]
            mesh = results[d["id"]]["mesh"]
            data, nv, nf = pack_meshes(mesh)
            with self._lock:
                self.view_version += 1
                self.views[f"detail:{d['node']}"] = (self.view_version, data)
                version = self.view_version
            publish(type="detail_ready", node=d["node"], version=version, view_version=d["view_version"],
                    lo=d["lo"], hi=d["hi"], voxel=float(mesh.metadata.get("voxel", 0.0)), verts=nv, faces=nf)
            publish(type="run_done", ok=True, time=time.perf_counter() - t0, files={}, label=job["label"],
                    voxel=float(mesh.metadata.get("voxel", 0.0)))
            return

        settings = job["slice"]
        surface = settings.get("mode") == "surface"
        paint = None  # (field node, field value): coloured onto every visible body
        paint_key = None
        if surface and job["slices"]:
            field_node, (src, port) = next(iter(job["slices"].items()))
            paint = (field_node, results.get(src, {}).get(port))
            paint_key = _content_key(graph, {}, src, port, "paint")

        for node_id, view in views.items():
            if run_id != self.run_id:
                return  # a newer run has started; its views will follow
            if view.kept:  # the editor shows it at this quality or finer; only a changed colouring is new
                item = self.views.get(node_id)
                coloured = (job["have"].get(node_id) or {}).get("paint") or {}
                same = coloured.get("key") == paint_key and coloured.get("level") in LEVELS and \
                    LEVELS.index(coloured["level"]) >= LEVELS.index(quality)
                if item is not None and paint is not None and paint[1] is not None and view.type != "cad" and not same:
                    publish(type="phase", text=f"colouring {node_id}")
                    self._paint(publish, node_id, item[0], item[1], paint, quality, key=paint_key)
                continue
            value = results.get(view.src, {}).get(view.port)
            if value is None:
                continue
            publish(type="phase", text=f"preparing the view of {node_id}")
            data, nv, nf = pack_meshes(value)
            with self._lock:
                self.view_version += 1
                self.views[node_id] = (self.view_version, data)
                version = self.view_version
                # for /api/face_at, /api/node_at and /api/detail
                self.view_sources[node_id] = (version, graph, view.src, view.port, quality)
                # CAD bodies: kept so the editor can ask which face each triangle is (face picking)
                if view.type == "cad":
                    self.view_faces[node_id] = (version, value, nf, None)
                else:
                    self.view_faces.pop(node_id, None)
            # voxel: what the view was meshed with; floor: how fine a detail of it may go (None: no detail)
            publish(type="view_ready", node=node_id, version=version, verts=nv, faces=nf, kind=view.type,
                    key=view.key, level=LEVELS[-1] if view.exact else quality, exact=view.exact,
                    voxel=None if view.exact else view_voxel(value),
                    floor=None if view.exact else detail_floor(graph, view.src, quality))
            own = results.get(view.paint[0], {}).get(view.paint[1]) if view.paint else None
            if own is not None:  # a result coloured onto the node's own body (Static Analysis)
                publish(type="phase", text=f"colouring {node_id}")
                self._paint(publish, node_id, version, data, (node_id, own), quality, source="result")
            # CAD views are not coloured: a flat CAD face is only its corner points, so a field
            # sampled there (e.g. a mask that is 0 at a face's edge) shows nothing of the face
            elif paint is not None and paint[1] is not None and view.type != "cad":
                publish(type="phase", text=f"colouring {node_id}")
                self._paint(publish, node_id, version, data, paint, quality, key=paint_key)

        for node_id, (src_id, port) in job["slices"].items():
            if run_id != self.run_id or surface:
                break
            value = results.get(src_id, {}).get(port)
            if value is None:
                continue
            publish(type="phase", text=f"slicing {node_id}")
            axis = settings.get("axis", "z")
            lo = np.asarray(settings.get("lo", (-50, -50, -50)), dtype=float)
            hi = np.asarray(settings.get("hi", (50, 50, 50)), dtype=float)
            position = float(settings.get("position", (lo["xyz".index(axis)] + hi["xyz".index(axis)]) / 2))
            try:
                image, rect = field_slice(value, axis, position, lo, hi, SLICE_RES[quality])
            except Exception as err:
                publish(type="node_error", node=node_id, message=f"cannot show this field: {err}")
                continue
            with self._lock:
                self.view_version += 1
                self.views[f"slice:{node_id}"] = (self.view_version, pack_slice(image))
                version = self.view_version
            lo_value, hi_value = _range(image)
            publish(type="slice_ready", node=node_id, version=version, axis=axis, position=position, rect=rect,
                    min=lo_value, max=hi_value)

        files = {}
        for node_id in targets:
            spec = graph.nodes.get(node_id, {})
            if spec.get("type") in REGISTRY and REGISTRY[spec["type"]].side_effect and node_id in results:
                files[node_id] = results[node_id].get("path")
        failed = self.evaluator.failed if mode == "preview" else []
        if failed:
            publish(type="run_done", ok=False, error="\n".join(str(err) for err in failed),
                    problems=self._problems([p for err in failed for p in err.problems]),
                    left_out=self._left_out(graph, targets, results), time=time.perf_counter() - t0,
                    label=job["label"], sharper=sharper, shown=shown)
        else:
            publish(type="run_done", ok=True, time=time.perf_counter() - t0, files=files, label=job["label"],
                    sharper=sharper, shown=shown)
        # Replaced meshes hold reference cycles; free them now (after the result is out, so
        # nobody waits for it) instead of letting them pile up until Python gets round to it.
        results = None
        gc.collect()

    def _paint(self, publish, node_id, view_version, data, paint, quality, source="field", key=None):
        """Colour a view by a field: the field's value at every vertex. source "result": the node's
        own result (labelled with the field's name and unit; reversed colours for a safety factor,
        where low is bad). `key`: what the field is made of, sent back by the editor (`have`) so a
        kept view is not coloured again with the same field."""
        from .graph.nodes import ARRAY_CACHE, array_module
        from .paint import field_at_points
        field_node, field = paint
        V, _ = unpack_meshes(data)
        try:
            values = field_at_points(field, V, PAINT_CELLS[quality], ARRAY_CACHE, array_module, PAINT_MAX_VOXELS)
        except Exception as err:
            publish(type="node_error", node=field_node, message=f"cannot colour the surfaces with this field: {err}")
            return
        with self._lock:
            self.view_version += 1
            self.views[f"paint:{node_id}"] = (self.view_version, values.astype(np.float32).tobytes())
            version = self.view_version
        lo, hi = _range(values)
        clipped = False
        shown = getattr(field, "display_range", None) if source == "result" else None
        if shown is not None:  # e.g. stress without the artificial peak at a support
            clipped = hi > shown[1] + 1e-9 or lo < shown[0] - 1e-9
            lo, hi = float(shown[0]), float(shown[1])
        unit = getattr(field, "unit", "") if source == "result" else ""
        label = getattr(field, "name", field_node) if source == "result" else field_node
        publish(type="paint_ready", node=node_id, field=field_node, version=version, view_version=view_version,
                min=lo, max=hi, label=label, unit=unit, source=source, clipped=clipped,
                reverse=source == "result" and label == "safety factor", key=key, level=quality)

    # --- print checks -------------------------------------------------------------------
    def check(self, node_id, density=1.24, thickness=False, min_wall=0.8):
        item = self.views.get(node_id)
        if item is None:
            raise HTTPException(404, f"{node_id} is not shown in the 3D view")
        from . import checks
        view_version, data = item
        V, F = unpack_meshes(data)
        report = checks.mesh_report(V, F, density)
        report["view_version"] = view_version
        if thickness and len(F):
            values = checks.wall_thickness(V, F)
            report["thickness"] = checks.thickness_report(values, min_wall)
            with self._lock:
                self.view_version += 1
                self.views[f"paint:{node_id}"] = (self.view_version, values.tobytes())
                version = self.view_version
            lo, hi = _range(values)
            report["paint"] = {"version": version, "view_version": view_version, "min": lo, "max": hi}
        return report

    def face_at(self, node_id, point):
        """See /api/face_at: the CAD face nearest to `point` among the bodies the view of `node_id`
        is built from (their results from the last run, kept by the evaluator)."""
        from .facemask import cad_faces, face_info, nearest_face, same_part
        with self._lock:
            item = self.view_sources.get(node_id)
        if item is None:
            raise HTTPException(404, f"{node_id} is not shown in the 3D view")
        _, graph, src_id, port, _ = item
        best, within = None, np.inf
        for source, out in cad_sources(graph, src_id, port):
            cached = self.evaluator._cache.get(source)
            value = cached[1].get(out) if cached else None
            if value is None:
                continue
            items = list(value) if isinstance(value, list) else [value]
            hit = nearest_face(items, point, within)
            if hit is not None:
                best, within = (source, out, items, hit), hit[2]
        if best is None:
            raise HTTPException(404, f"no CAD body with faces behind {node_id}")
        source, out, items, (face, body, distance) = best
        ids = [cad_faces(m) for m in items]
        return {"source": source, "port": out, "face": face, "body": body, "distance": distance,
                "same": len(items) == 1 or (all(i is not None for i in ids) and same_part(items, ids)),
                "bodies": len(items), "info": face_info(items, body, face)}

    def node_at(self, node_id, point, tol):
        """See /api/node_at: the node that made the surface at `point` of the view of `node_id`. Reads
        the results of the last run (kept by the evaluator); the walk itself runs outside the lock."""
        from .creator import find_creator
        with self._lock:
            item = self.view_sources.get(node_id)
        if item is None:
            raise HTTPException(404, f"{node_id} is not shown in the 3D view")
        _, graph, src_id, port, _ = item
        cache = self.evaluator._cache
        creator, visited = find_creator(graph, lambda n: (cache.get(n) or (None, None))[1], src_id, port, point, tol)
        shown = lambda n: graph.owners.get(n, n)   # a node inside a group is the group, for the editor
        path = []
        for n, value in visited:
            if not path or path[-1]["node"] != shown(n):  # a group's inner steps make one entry
                path.append({"node": shown(n), "value": value})
        inner = creator.split(GROUP_SEP, 1)[1] if creator in graph.owners else None
        return {"node": shown(creator), "inner": inner, "type": (graph.nodes.get(creator) or {}).get("type"),
                "path": path}


def _drives():
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


def _autosave_file(path):
    """Where unsaved changes to the recipe `path` (None: a new recipe) are kept."""
    key = os.path.normcase(os.path.abspath(path)) if path else "untitled"
    return default_cache_dir().parent / "autosave" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")


def _preset_file(name):
    if not PRESET_NAME.match(name or ""):
        raise HTTPException(400, "preset names may use letters, digits, spaces, - and _")
    return PRESETS / f"{name}.json"


def create_app():
    hub = Hub()
    session = Session(hub)

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        from .graph.nodes import ARRAY_CACHE
        if ARRAY_CACHE.disk is not None:  # bring the SSD cache back under its size limit
            threading.Thread(target=ARRAY_CACHE.disk.trim, daemon=True).start()
        stop_watch = threading.Event()
        threading.Thread(target=session.watch_idle_memory, args=(stop_watch,), daemon=True).start()
        yield
        stop_watch.set()
        session.queue.clear()
        session.pending = None
        session.evaluator.cancel()
        session.evaluator.close()

    app = FastAPI(title="Design Engine", lifespan=lifespan)

    @app.get("/api/catalog")
    def catalog():
        return {"nodes": node_catalog(), "compatibility": compatibility(),
                "qualities": list(QUALITY), "root": str(ROOT)}

    @app.get("/api/recipes")
    def recipes():
        return [{"name": p.stem, "path": str(p)} for p in sorted(RECIPES.glob("*.json"))]

    @app.get("/api/file")
    def read_file(path: str):
        p = Path(path)
        if p.suffix.lower() != ".json" or not p.is_file():
            raise HTTPException(404, f"not a recipe file: {path}")
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise HTTPException(400, f"invalid JSON in {path}: {err}")

    @app.post("/api/file")
    def write_file(path: str, graph: dict = Body(...)):
        p = Path(path)
        if p.suffix.lower() != ".json":
            raise HTTPException(400, "recipes must be saved as .json")
        try:
            Graph.from_dict(graph)
        except Exception as err:
            raise HTTPException(400, f"not a valid graph: {err}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        return {"path": str(p)}

    # unsaved changes, kept by the server so they survive a crash of the browser or the PC
    @app.get("/api/autosave")
    def get_autosave(path: str = ""):
        f = _autosave_file(path or None)
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @app.post("/api/autosave")
    def put_autosave(body: dict = Body(...)):
        f = _autosave_file(body.get("path") or None)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"path": body.get("path"), "time": time.time(), "recipe": body.get("recipe")}),
                       encoding="utf-8")
        os.replace(tmp, f)
        return {"ok": True}

    @app.delete("/api/autosave")
    def delete_autosave(path: str = ""):
        _autosave_file(path or None).unlink(missing_ok=True)
        return {"ok": True}

    # presets: saved node selections (with the groups they use), inserted from the node palette
    @app.get("/api/presets")
    def list_presets():
        out = []
        for p in sorted(PRESETS.glob("*.json")):
            try:
                clip = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({"name": p.stem, "nodes": len(clip.get("nodes", [])), "doc": clip.get("doc", "")})
        return out

    @app.get("/api/presets/{name}")
    def get_preset(name: str):
        f = _preset_file(name)
        if not f.is_file():
            raise HTTPException(404, f"no preset '{name}'")
        return json.loads(f.read_text(encoding="utf-8"))

    @app.post("/api/presets/{name}")
    def put_preset(name: str, clip: dict = Body(...)):
        f = _preset_file(name)
        if not isinstance(clip.get("nodes"), list) or not clip["nodes"]:
            raise HTTPException(400, "a preset needs at least one node")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(clip, indent=2), encoding="utf-8")
        return {"name": name}

    @app.delete("/api/presets/{name}")
    def delete_preset(name: str):
        _preset_file(name).unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/browse")
    def browse(path: str = "", exts: str = ""):
        wanted = {e.strip().lower() for e in exts.split(",") if e.strip()}
        if not path:
            return {"path": "", "parent": None,
                    "entries": [{"name": d, "path": d, "dir": True} for d in _drives()]}
        p = Path(path)
        if p.is_file():
            p = p.parent
        if not p.is_dir():
            raise HTTPException(404, f"folder not found: {path}")
        entries = []
        try:
            for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
                if child.name.startswith(("$", ".")):
                    continue
                is_dir = child.is_dir()
                if is_dir or not wanted or child.suffix.lower() in wanted:
                    entries.append({"name": child.name, "path": str(child), "dir": is_dir})
        except PermissionError:
            raise HTTPException(403, f"no access to {p}")
        parent = str(p.parent) if p.parent != p else ""
        return {"path": str(p), "parent": parent, "entries": entries}

    @app.post("/api/run")
    def run(body: dict = Body(...)):
        run_id = session.start(body.get("graph") or {"version": 1, "nodes": {}}, body.get("mode", "preview"),
                               body.get("quality", "draft"), body.get("visible", []), body.get("slice"),
                               body.get("label"), have=body.get("have"), tag=body.get("tag"))
        return {"run": run_id}

    @app.post("/api/optimize")
    def optimize(body: dict = Body(...)):
        node = body.get("node")
        if not node:
            raise HTTPException(400, "which node to optimize?")
        run_id = session.start(body.get("graph") or {"version": 1, "nodes": {}}, "optimize",
                               body.get("quality", "draft"), body.get("visible", []), body.get("slice"),
                               f"optimize {node}", optimize=node, tag=body.get("tag"))
        return {"run": run_id}

    @app.post("/api/detail")
    def detail(body: dict = Body(...)):
        """The part lo..hi of a shown body finer than its view (see Session.detail): {node, version (the
        view's), lo, hi, voxel, tag}. Answers {run} or {busy: true} while something else runs; the result
        comes as a detail_ready event, the mesh from /api/view/detail:<node>."""
        try:
            voxel = float(body.get("voxel", 0))
        except (TypeError, ValueError):
            raise HTTPException(422, "voxel must be a number")
        return session.detail(str(body.get("node", "")), body.get("version"), body.get("lo"), body.get("hi"),
                              voxel, body.get("tag"))

    @app.get("/api/queue")
    def queue():
        return session.queue_state()

    @app.post("/api/queue/remove")
    def remove_from_queue(body: dict = Body(...)):
        session.remove_queued(int(body.get("id", -1)))
        return session.queue_state()

    @app.post("/api/check")
    def check(body: dict = Body(...)):
        return session.check(str(body.get("node", "")), float(body.get("density", 1.24)),
                             bool(body.get("thickness")), float(body.get("min_wall", 0.8)))

    @app.get("/api/memory")
    def memory():
        from .graph.nodes import ARRAY_CACHE
        disk = ARRAY_CACHE.disk
        gpu_free, gpu_total = backend.gpu_memory() if backend.gpu_enabled() else (0, 0)
        return {"used": session.evaluator.memory_in_use(), "limit": memory_limit(), "free": available_memory(),
                "ssd_cache": disk.size() if disk is not None else 0,
                "gpu_free": gpu_free, "gpu_total": gpu_total}

    @app.get("/api/settings")
    def get_settings():
        return {"gpu": backend.gpu_enabled(), "gpu_available": backend.gpu_available()}

    @app.post("/api/settings")
    def set_settings(body: dict = Body(...)):
        if "gpu" in body:
            session.cancel()
            backend.set_gpu(bool(body["gpu"]) and backend.gpu_available())
        return get_settings()

    @app.post("/api/cancel")
    def cancel():
        session.cancel()
        return {"ok": True}

    @app.get("/api/view/{node_id}")
    def view(node_id: str):
        item = session.views.get(node_id)
        if item is None:
            raise HTTPException(404, f"no view for {node_id}")
        return Response(item[1], media_type="application/octet-stream",
                        headers={"X-View-Version": str(item[0])})

    @app.get("/api/faces/{node_id}")
    def faces(node_id: str):
        """For face picking: which CAD face and body each triangle of a CAD view is, and which
        faces join smoothly. uint32 triangle_count n, uint32 face[n], uint32 body[n], uint32 same
        (1: the bodies are copies of one part), uint32 group_lists, then per list (one if same,
        else one per body) uint32 face_count, uint32 group[face_count]."""
        item = session.view_faces.get(node_id)
        if item is None:
            raise HTTPException(404, f"no CAD view for {node_id}")
        version, value, shown, packed = item
        if packed is None:
            packed = pack_faces(value, shown)
            with session._lock:
                if session.view_faces.get(node_id, (None,))[0] == version:
                    session.view_faces[node_id] = (version, value, shown, packed)
        if not packed:
            raise HTTPException(409, "faces cannot be picked here (not CAD faces, or too many triangles to show)")
        return Response(packed, media_type="application/octet-stream", headers={"X-View-Version": str(version)})

    @app.get("/api/face_at/{node_id}")
    def face_at(node_id: str, x: float, y: float, z: float):
        """For the face menu on bodies whose triangles carry no faces (a textured, implicit body, a
        topology result): the CAD face nearest to the point (x, y, z) among the CAD bodies the view of
        the node is built from. {source, port}: the CAD output to link new nodes to (the first cad
        output upstream of each path, not inside a group); {face, body}: as /api/faces reports them;
        distance: mm from the point to that face; same: the source's bodies are copies of one part;
        bodies: how many it has; info: facemask.face_info of the face (or null)."""
        if not all(np.isfinite([x, y, z])):
            raise HTTPException(422, "x, y and z must be finite numbers")
        return session.face_at(node_id, (x, y, z))

    @app.get("/api/node_at/{node_id}")
    def node_at(node_id: str, x: float, y: float, z: float, tol: float = 0.05):
        """For selecting by clicking in the 3D view: the node that made the surface at the point
        (x, y, z) of the node's view, found by walking upstream from what the view shows (see
        engine/creator.py). tol: how far the point can be from the true surface, mm (the shown mesh's
        resolution there: the longest edge of the clicked triangle); at least 0.001.
        {node, inner, type, path}: node is the creator (a node inside a group is reported as its group,
        with `inner` the inner node's id, else null; `type` is the creator's own node type); path lists
        the nodes visited from the shown one to the creator (the same ids, a group's inner steps as one
        entry) with each one's field value at the point (null for CAD and mesh steps)."""
        if not all(np.isfinite([x, y, z, tol])):
            raise HTTPException(422, "x, y, z and tol must be finite numbers")
        return session.node_at(node_id, (x, y, z), max(tol, 1e-3))

    @app.get("/api/faceinfo/{node_id}")
    def faceinfo(node_id: str, body: int, face: int):
        """What one CAD face of a CAD view is (for the face menu): see facemask.face_info. `body` and
        `face` are the numbers /api/faces reports."""
        item = session.view_faces.get(node_id)
        if item is None:
            raise HTTPException(404, f"no CAD view for {node_id}")
        version, value = item[:2]
        key = (node_id, version, body, face)
        info = session.face_infos.get(key)
        if info is None:
            from .facemask import face_info
            info = face_info(value, body, face)
            if info is None:
                raise HTTPException(404, f"face {face} of body {body} is not in the view of {node_id}")
            with session._lock:
                session.face_infos[key] = info
                while len(session.face_infos) > 64:
                    session.face_infos.pop(next(iter(session.face_infos)))
        return info

    @app.websocket("/ws")
    async def events(ws: WebSocket):
        await ws.accept()
        queue = asyncio.Queue()
        hub.clients.add(queue)
        try:
            while True:
                await ws.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.clients.discard(queue)

    if DIST.exists():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
    return app
