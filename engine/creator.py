"""Which node made the surface at a point of a shown body (for /api/node_at: click a bump of a
texture and the Offset Body that made it is selected, click a hole's wall and it is the Hole Pattern).

The walk starts at what the view shows and goes upstream, one node at a time, reading each node's
result from the last run. At a node it reads the node's own field and that of every body it was
made from (inputs of port type `implicit`) at the point: the bodies that still have their surface
there are where the node did not change it, so the walk goes on into the one closest to the point;
when none has, this node is the creator."""
import math

import numpy as np

from .fields import sample
from .graph import REGISTRY
from .graph.core import _is_link

MAX_STEPS = 64     # more nodes than this in a row cannot happen (graphs have no cycles); stop anyway
MAX_SAMPLES = 64   # items of one list read per step at most: their bounds rule out nearly all of them
HIDDEN = "__view__"  # the hidden mesher the server puts behind a shown implicit body (build_run_graph)


def _output_type(ntype, port):
    return next((p.type for p in ntype.outputs if p.name == port), None)


def _linked(graph, node_id, port_type):
    """[(source node, source output)] of the node's linked inputs whose port type is `port_type`
    (the input's own type: an Offset's Distance is a scalar, not a body), in port order."""
    spec = graph.nodes.get(node_id) or {}
    ntype = REGISTRY.get(spec.get("type"))
    inputs = spec.get("inputs", {})
    return [tuple(inputs[p.name]["link"].partition(".")[::2]) for p in (ntype.inputs if ntype else [])
            if p.type == port_type and _is_link(inputs.get(p.name))]


def _near(item, point, tol):
    """False when the item's bounds, grown by `tol`, leave the point out (no bounds: could be)."""
    box = getattr(item, "bounds", None)
    return box is None or bool(np.all(point >= box[0] - tol) and np.all(point <= box[1] + tol))


def _read(item, point):
    """The field's value at the point, None when it has none there (not finite, or the field
    cannot be read at a single point: a lookup should never break the click)."""
    try:
        value = sample(item, point)   # a one-voxel grid and no array cache: nothing is stored
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _probe(value, point, tol, pick):
    """(signed value, pick) of a field, or of the item of a list of fields (one per body) that has its
    surface nearest the point; (None, no pick) when none reaches it. `pick` = (index, length) of the
    item the node downstream settled on: a list of the same length is read at that item only,
    anything else is read at the items whose bounds reach the point (at most MAX_SAMPLES)."""
    items = list(value) if isinstance(value, list) else [value]
    index, count = pick
    if isinstance(value, list) and index is not None and count == len(items):
        candidates = [index]
    else:
        candidates = [i for i, f in enumerate(items) if _near(f, point, tol)][:MAX_SAMPLES]
    best = None
    for i in candidates:
        v = _read(items[i], point)
        if v is not None and (best is None or abs(v) < abs(best[0])):
            best = (v, i)
    if best is None:
        return None, (None, None)
    return best[0], ((best[1], len(items)) if isinstance(value, list) else (None, None))


def find_creator(graph, outputs, node, port, point, tol):
    """The node that made the surface at `point` of the output `node.port`, and the nodes visited on
    the way there: (creator, [(node, field value there or None)]). Ids are those of the expanded graph
    (inside groups too, `hidden` meshers left out). `outputs(node)`: the node's outputs from the last
    run, or None. `tol`: how far the point can be off the true surface (the shown mesh's resolution)."""
    point = np.asarray(point, dtype=np.float64)
    path, pick, known = [], (None, None), None   # known: (value, pick) the step before already read here
    for _ in range(MAX_STEPS):
        spec = graph.nodes.get(node) or {}
        ntype = REGISTRY.get(spec.get("type"))
        entry = [node, known[0] if known else None]
        if not node.startswith(HIDDEN):
            path.append(entry)
        if ntype is None:
            break
        kind = _output_type(ntype, port)
        if kind == "cad":
            break
        below = None
        if kind == "mesh":   # the surface of the one body it was made from (a mesh node without one, or
            below = _linked(graph, node, "implicit")   # with several, is a creator: Merge Meshes, an export)
            if len(below) != 1:
                break
        elif spec["type"] == "implicit.from_cad":   # the surface of that CAD body, which is the creator
            below = _linked(graph, node, "cad")[:1]
            if not below:
                break
        if below:
            (node, port), known = below[0], None
            continue
        value = (outputs(node) or {}).get(port)
        if value is None:
            break
        own, here = known if known else _probe(value, point, tol, pick)
        entry[1] = own
        reach = max(tol, 1.5 * abs(own or 0.0))   # off the surface by about as much as the node's own value
        found = []
        for src, src_port in _linked(graph, node, "implicit"):
            given = (outputs(src) or {}).get(src_port)
            if given is not None:
                v, there = _probe(given, point, tol, here)
                if v is not None and abs(v) <= reach:
                    found.append((abs(v), len(found), src, src_port, v, there))
        if not found:
            break
        _, _, node, port, v, pick = min(found)   # the body closest to the point (the first of equals)
        known = (v, pick)
    else:   # MAX_STEPS nodes: the last one is the creator
        if not node.startswith(HIDDEN):
            path.append([node, known[0] if known else None])
    return (node[len(HIDDEN):] if node.startswith(HIDDEN) else node), [tuple(e) for e in path]
