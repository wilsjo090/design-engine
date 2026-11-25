"""Simulation nodes: materials, supports and loads picked as CAD faces, static stress analysis.

Units: mm, N, MPa (N/mm^2), g/cm^3. See engine/fea and docs/TOPOLOGY_PLAN.md.
"""
import dataclasses
import math
import re
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from .. import backend, facemask, trace
from .. import fields as F
from ..cache import load_topology, memory_limit, memory_room, save_topology
from ..fea import analysis as FA
from ..fea import topology as T
from ..fea.materials import PRESETS, custom
from ..fea.model import (DETAIL, DIRECTIONS, LOAD_KINDS, OFFSET_KINDS, Condition, ConditionError, Conditions,
                         marker_samples, parse_vector)
from ..fea import solver as FS
from ..grid import Grid
from ..fea.solver import SLAB_BYTES_GPU, NotConverged
from .core import (Cancelled, GraphError, Port, check_cancel, node, report_finished, report_message, report_progress,
                   report_view, stop_idle_workers)
from .nodes import ARRAY_CACHE, S

SIM = "Simulation"
SHOW = {"stress": "stress", "safety factor": "safety", "displacement": "displacement"}
GPU_MIN_ELEMENTS = 60_000   # smaller analyses are as fast on the CPU (and need no CUDA context)
# Memory of an analysis per element of the grid box (not just the part), with a margin. Measured on
# the L bracket (17% of its box filled): CPU 480-620 bytes, GPU 540 (float64 CG vectors over the
# whole box are most of it).
CPU_BYTES_PER_ELEMENT = 800
# GPU: GPU_FIXED + bytes per box element, ~15-20% over the most measured (2026-10-23, the card's free
# memory sampled while it ran, fused kernels and CuPy code alike): Static Analysis 412-491 B in all
# (bolted bracket space at fine 1.3M elements 0.55-0.62 GB, key box 5.3M 2.1-2.3 GB, a fine topology
# result 6.3M 2.5 GB), topology ~65 MB + 550-630 B (715 B in all with an FFT filter). The old 600 B +
# 768 MB of product batches sent that 2.5 GB check to the CPU unless 5.9 GB were free.
GPU_FIXED = 128 * 2**20
GPU_BYTES_PER_ELEMENT = 470
TOPOLOGY_GPU_BYTES_PER_ELEMENT = 720


# --- materials ---------------------------------------------------------------------------------
@node("sim.material", "Material", SIM,
      [Port("preset", "text", "PLA", choices=list(PRESETS),
            doc="Typical values: " + "; ".join(m.describe() for m in PRESETS.values()) + ". Printed plastic "
                "parts are weaker between layers than along them. Other values: Custom Material.")],
      [Port("material", "material", doc="For Static Analysis.")])
def material(preset):
    """A material from the list: its stiffness, Poisson's ratio, density and strength."""
    if preset not in PRESETS:
        raise GraphError(f"unknown material '{preset}': choose one of {', '.join(PRESETS)}")
    return PRESETS[preset]


@node("sim.custom_material", "Custom Material", SIM,
      [Port("stiffness", "number", 3500.0, label="Stiffness E", unit="MPa",
            doc="Young's modulus, MPa (N/mm²): how hard it is to stretch. PLA ~3500, PETG ~2000, aluminium 69000, "
                "steel 210000. The datasheet's tensile modulus."),
       Port("poisson", "number", 0.35, label="Poisson's ratio",
            slider=(0, 0.5), doc="How much it gets thinner when stretched: plastics 0.35-0.4, metals ~0.3."),
       Port("density", "number", 1.24, unit="g/cm³", doc="g/cm³, for the mass in the report."),
       Port("strength", "number", 50.0, unit="MPa", doc="MPa: the stress at which it yields (metals) or breaks "
                                                        "(plastics, tensile strength). The safety factor is strength / "
                                                        "stress.")],
      [Port("material", "material", doc="For Static Analysis.")])
def custom_material(stiffness, poisson, density, strength):
    """A material with your own values (from a datasheet)."""
    try:
        return custom(stiffness, poisson, density, strength)
    except ValueError as err:
        raise GraphError(str(err)) from err


# --- supports and loads ------------------------------------------------------------------------
def _chosen_faces(cad, faces):
    """(corner positions (k, 3, 3), CAD face number (k,), body number (k,)) of the triangles of the
    chosen CAD faces on one body or a list. The body numbers are those the face view reports."""
    unknown = re.findall(r"[A-Za-z_][\w-]*", str(faces or ""))
    if unknown:
        raise GraphError(f"no face selection named '{unknown[0]}' in this recipe")
    out, face_ids, body_ids = [], [], []
    listed = list(cad) if isinstance(cad, list) else None
    for k, body in enumerate(listed if listed is not None else [cad]):
        ids = facemask.cad_faces(body)
        if ids is None:
            raise GraphError("this body has no CAD faces: connect a body from Import STEP")
        chosen = facemask.faces_on(faces, body, among=listed)
        wrong = [f for f in chosen if f > ids.max()]
        if wrong:
            raise GraphError(f"face {wrong[0]} does not exist (the body has faces 0-{ids.max()})")
        on = np.isin(ids, chosen)
        out.append(np.asarray(body.triangles)[on])
        face_ids.append(ids[on].astype(np.int32))
        body_ids.append(np.full(int(on.sum()), facemask.cad_body(body, k), dtype=np.int32))
    if not out:
        return np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.concatenate(out), np.concatenate(face_ids), np.concatenate(body_ids)


def _condition(kind, cad, faces, **settings):
    """A support or load on the chosen faces of `cad`, remembering which CAD face and body each
    triangle belongs to."""
    triangles, face_ids, body_ids = _chosen_faces(cad, faces)
    return Condition(kind, triangles, faces=face_ids, bodies=body_ids, **settings)


def _with(add_to, condition):
    """`add_to` plus the new condition, carrying the new one's markers for the 3D view."""
    if add_to is not None and not isinstance(add_to, Conditions):
        raise GraphError("Add to takes supports and loads (from Fixed Faces or Load)")
    out = (add_to or Conditions()).add(condition)
    if len(condition.triangles):
        pts, dirs, sizes, tri = marker_samples(condition)
        push = (np.einsum("ij,ij->i", dirs, _normals_at(condition, pts)) < 0).tolist()
        out.markers = {"kind": condition.kind, "label": condition.label,
                       "points": np.round(pts, 4).tolist(), "dirs": np.round(dirs, 4).tolist(), "push": push,
                       "sizes": np.round(sizes, 3).tolist()}
        if condition.faces is not None and condition.bodies is not None:   # the CAD face under each point
            out.markers.update(faces=condition.faces[tri].tolist(), bodies=condition.bodies[tri].tolist())
        if condition.kind == "load":  # for the big arrow: the faces' centre and the force (torque axis)
            arrow = condition.arrow()
            out.markers.update(load=condition.load, centre=np.round(condition.geometry[1], 4).tolist(),
                               arrow=None if arrow is None else np.round(arrow, 4).tolist())
    else:
        out.markers = {"kind": condition.kind, "label": condition.label, "points": [], "dirs": [], "push": [],
                       "faces": [], "bodies": []}
    return out


def _normals_at(condition, points):
    """Outward normal of the chosen triangle nearest to each point."""
    t = condition.triangles
    centres = t.mean(axis=1)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    from scipy.spatial import cKDTree
    return n[cKDTree(centres).query(points)[1]]


_DIRECTED = [k for k in LOAD_KINDS if k != "pressure"]   # loads with a direction
LOAD_UNITS = {"force": "N", "pressure": "MPa", "bearing": "N", "torque": "N·m"}   # of Amount, by kind


def _faces_label(faces):
    text = str(faces or "").strip()
    return text if len(text) <= 24 else text[:22] + "…"


@node("sim.fixed", "Fixed Faces", SIM,
      [Port("cad", "cad", many=True, label="Body",
            doc="The CAD body the faces are picked on: the part you analyse (from Import STEP or Pick Body)."),
       Port("faces", "faces", "", doc="The faces that are held still: bolted, glued or clamped (held in every "
                                      "direction). Click Pick, then click the faces in the 3D view."),
       Port("add_to", "conditions", optional=True, label="Add to",
            doc="Optional: supports and loads from another Fixed Faces or Load node, so one link carries them all.")],
      [Port("conditions", "conditions", label="Supports & loads",
            doc="These faces held still (plus what came in on Add to). Connect to Static Analysis, or to the next "
                "Load's Add to. Shown as green cones in the 3D view.")],
      preview=True)
def fixed_faces(cad, faces, add_to):
    """Faces that are held still (bolted, glued, clamped): they do not move at all."""
    return _with(add_to, _condition("fixed", cad, faces, label=f"Fixed faces {_faces_label(faces)}"))


@node("sim.load", "Load", SIM,
      [Port("cad", "cad", many=True, label="Body",
            doc="The CAD body the faces are picked on: the part you analyse."),
       Port("faces", "faces", "", doc="The faces the load acts on. Click Pick, then click them in the 3D view."),
       Port("kind", "text", "force", choices=list(LOAD_KINDS),
            doc="force = a total force in N, spread evenly over the faces. pressure = MPa (N/mm²) pushing on every "
                "bit of the faces, always square to them (Direction is not used). bearing = a pin, bolt or shaft "
                "in a hole pressing on it: only the side of the hole the force points to carries it, most at the "
                "middle of that side, none at the far side (as SolidWorks' bearing load). Needs a direction "
                "such as -Z. torque = twisting, N·m, about the Direction axis (right-hand rule: thumb along the "
                "axis, the fingers show the turn), as through a rigid connection: the further from the centre "
                "of the faces, the more each bit carries (a shaft's torque on its hole, bolts on a circle)."),
       Port("amount", "number", 100.0, unit={"kind": LOAD_UNITS},
            doc="Force and bearing: N in total (100 N is about the weight of 10 kg). Pressure: MPa (0.1 MPa = 1 bar). "
                "Torque: N·m (10 N on a 10 cm lever = 1 N·m). Negative turns it round."),
       Port("direction", "text", "into the face", choices=list(DIRECTIONS), when={"kind": _DIRECTED},
            doc="Force and bearing. into the face = pushing on it, square to it everywhere (on a curved face the "
                "pushes partly cancel). out of the face = pulling. +X ... -Z = along an axis (-Z: downwards; a "
                "bearing load -Z presses on the bottom of the hole). custom = along Vector. Torque: the axis it "
                "turns about (into the face on a flat face = turning like driving a screw into it). Easiest: "
                "select this node and drag the head of the yellow arrow in the 3D view (it snaps to the axes "
                "and 45°)."),
       Port("vector", "text", "0, 0, -1", when={"kind": _DIRECTED, "direction": ["custom"]},
            doc="Custom direction only: x, y, z (any length), e.g. 1, 0, -1 for 45° "
                                              "down in the X-Z plane."),
       Port("offset", "text", "0, 0, 0", when={"kind": list(OFFSET_KINDS)},
            doc="Force and bearing: where the force really acts, x, y, z in mm from the centre of the faces (the "
                "end of a lever or handle, a weight hanging on an arm). The faces then also carry its turning "
                "effect (force x distance), spread as through a rigid connection. 0, 0, 0 = on the faces. "
                "Easiest: drag the ball at the tail of the yellow arrow."),
       Port("add_to", "conditions", optional=True, label="Add to",
            doc="Optional: supports and loads from a Fixed Faces or another Load, so one link carries them all.")],
      [Port("conditions", "conditions", label="Supports & loads",
            doc="This load (plus what came in on Add to). Connect to Static Analysis, or to the next node's Add to. "
                "Shown as red arrows in the 3D view.")],
      preview=True)
def load(cad, faces, kind, amount, direction, vector, add_to, offset="0, 0, 0"):
    """A force, a pressure, a bearing load (a pin pressing on one side of a hole) or a torque on faces
    of the part; a force can act away from the faces (Offset)."""
    if kind not in LOAD_KINDS:
        raise GraphError(f"kind must be one of: {', '.join(LOAD_KINDS)}")
    if direction not in DIRECTIONS:
        raise GraphError(f"direction must be one of: {', '.join(DIRECTIONS)}")
    directed = kind != "pressure"
    unit = {"pressure": "MPa", "torque": "N·m"}.get(kind, "N")
    try:
        vec = parse_vector(vector) if (directed and direction == "custom") else (0.0, 0.0, -1.0)
        if directed and direction == "custom" and np.linalg.norm(vec) == 0:
            raise ConditionError("the vector must not be 0, 0, 0")
        shift = parse_vector(offset or "0, 0, 0") if kind in OFFSET_KINDS else (0.0, 0.0, 0.0)
        condition = _condition("load", cad, faces, load=kind, amount=float(amount),
                               direction=direction, vector=vec, offset=shift,
                               label=f"{float(amount):g} {unit} {kind + ' ' if kind in ('bearing', 'torque') else ''}"
                                     f"on {_faces_label(faces) or 'no faces'}")
        condition.check()  # the direction (a bearing load: that something faces it), now
    except ConditionError as err:
        raise GraphError(str(err)) from err
    return _with(add_to, condition)


@node("sim.load_case", "Load Case", SIM,
      [Port("loads", "conditions", label="Loads",
            doc="The loads of this case: the last of a chain of Load nodes (their Add to). They act together, "
                "at the same time; other cases' loads do not act with them."),
       Port("name", "text", "Case 1",
            doc="What the case is called in the log and the notes. Its colour (on its loads' cards and arrows "
                "in the 3D view) goes with the name. Two Load Case nodes with the same name are one case."),
       Port("add_to", "conditions", optional=True, label="Add to",
            doc="The other cases and the supports (Fixed Faces), so one link carries them all to Static "
                "Analysis or Topology Optimization. Supports, and loads in no case, act in every case.")],
      [Port("conditions", "conditions", label="Supports & loads",
            doc="This case's loads (plus what came in on Add to). Connect to Static Analysis or Topology "
                "Optimization, or to the next Load Case's Add to.")])
def load_case(loads, name, add_to):
    """Loads that act together, apart from other cases': e.g. a shelf loaded on its front edge, and
    separately pushed from the side. Static Analysis and the optimizers make the part stand every case
    (the worst of them counts); supports act in every case. Copy and paste a Load Case with its loads to
    make another case (the copy gets the next free name)."""
    name = str(name or "").strip()
    if not name:
        raise GraphError("the load case needs a name")
    for item, port in ((loads, "Loads"), (add_to, "Add to")):
        if item is not None and not isinstance(item, Conditions):
            raise GraphError(f"{port} takes supports and loads (from Fixed Faces, Load or Load Case)")
    if not loads or not any(c.kind == "load" for c in loads):
        raise GraphError("connect this case's loads (the last Load node of its chain) to Loads")
    out = list(add_to or ())
    seen = {(c.key, c.case) for c in out}
    for c in loads:   # (supports and loads already in a case keep theirs; anything already there once)
        if c.kind == "load" and c.case is None:
            c = dataclasses.replace(c, case=name)
        if (c.key, c.case) not in seen:
            seen.add((c.key, c.case))
            out.append(c)
    return Conditions(out)


# --- analysis ----------------------------------------------------------------------------------
def _part(part):
    """(implicit body, volume or None, bounds) of a CAD body, an implicit body or a list of them;
    several bodies are analysed as one (bonded where they touch)."""
    items = list(part) if isinstance(part, list) else [part]
    if not items:
        raise GraphError("connect the part to analyse")
    bodies, volumes = [], []
    for item in items:
        if isinstance(item, F.Field):
            bodies.append(item)
            volumes.append(None)
        elif hasattr(item, "faces") and hasattr(item, "vertices"):
            bodies.append(F.body_from_mesh(item, 2.0))
            # (not item.volume / is_watertight: trimesh keeps per-triangle arrays on the CAD body for them)
            from ..meshing import signed_volume
            volumes.append(abs(float(signed_volume(item.vertices, item.faces))) if _closed(item.faces) else None)
        else:
            raise GraphError("Part takes a CAD body or an implicit body")
    body = bodies[0]
    for other in bodies[1:]:
        body = F.union(body, other)
    if body.bounds is None:
        raise GraphError("this part has no size of its own (e.g. a lattice): fill a body with it first")
    volume = sum(volumes) if all(v is not None for v in volumes) else None
    return body, volume, len(items)


def _hollow(part, walls, keep_body, conditions, skin, min_thickness):
    """Walls: the body Topology Optimization keeps empty so that the design space is walls of `walls` mm along
    its faces: everything deeper than that inside the part, less Keep and the solid skin round the faces of
    supports and loads (a boss round a hole stays whole where the walls are thinner than it). The part's
    distance has to be exact that deep: CAD bodies are measured again with a band of the walls (their usual
    2 mm band is interpolated beyond it); an implicit body is taken as it is. Where two walls meet (a part
    thinner than two of them) nothing is hollow. A hole's own faces are faces like any other: a blind hole
    that ends in the hollow has walls round its end, and a Skin round it stands in the hollow as a ball (the
    hanger's screw holes, held by their walls: held by their countersinks the skin stays in the plate)."""
    items = list(part) if isinstance(part, list) else [part]
    deep = None
    for item in items:
        one = item if isinstance(item, F.Field) else F.body_from_mesh(item, float(walls) + 1.0)
        deep = one if deep is None else F.union(deep, one)
    core = F.offset(deep, -float(walls))
    if keep_body is not None:
        core = F.subtract(core, keep_body)
    if skin > 0 and len(conditions):
        around = T.skin_body(conditions, T.skin_reach(skin, min_thickness, 0.0))
        if around is not None:
            core = F.subtract(core, around)
    return core


def _closed(faces):
    """Whether every edge of a mesh has exactly two triangles (its volume is then its own)."""
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return False
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges[:, 0] * (int(faces.max()) + 1) + edges[:, 1], return_counts=True)
    return bool((counts == 2).all())


def gpu_bytes(elements, per_element=GPU_BYTES_PER_ELEMENT):
    """GPU memory an analysis (or, per_element=TOPOLOGY_GPU_BYTES_PER_ELEMENT, an optimization) of this
    many box elements takes. Without the fused kernels a product's CuPy temporaries come on top."""
    batches = 0 if FS.GPU_KERNELS else 3 * SLAB_BYTES_GPU
    return GPU_FIXED + elements * per_element + batches


def _gpu_fits(elements, per_element, wait=0.0):
    """Whether the analysis fits on the GPU now (waiting up to `wait` s for memory being given back)."""
    return _gpu_has(gpu_bytes(elements, per_element), wait)


def _gpu_has(need, wait=0.0):
    """Whether `need` bytes fit on the GPU beside GPU_HEADROOM now (waiting up to `wait` s for memory being
    given back)."""
    deadline = time.perf_counter() + wait
    while True:
        # blocks this process's CuPy pool keeps for reuse look used to the card: right after a fine
        # optimization ~1 GB of them sent its result's check to the CPU
        backend.release_gpu_memory()
        free, _ = backend.gpu_memory()
        if need < free - backend.GPU_HEADROOM:
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(0.1)


def _room_for_smoothing(shape, blur):
    """Before Topology Optimization smooths its result (T.density_mesh, which runs on the GPU when it fits and
    takes ~10x as long on the CPU): when its arrays do not fit there as things are, room is made as for an
    analysis (`_array_module`), cheapest first: this process's GPU search structures, then the run's idle worker
    processes. What still does not fit says so in the log. The card's other programs held 5.5 GB of its 12."""
    samples = T.smoothing_samples(shape, blur)
    if not backend.use_gpu_for(samples):
        return
    need = samples * T.SMOOTH_GPU_BYTES
    if _gpu_has(need) or (backend.release_gpu_caches() and _gpu_has(need, wait=0.5)):
        return
    if stop_idle_workers():
        report_message("stopped the idle worker processes to make room on the GPU for smoothing the shape (they "
                       "start again on the next update)")
        if _gpu_has(need, wait=2.0):
            return
    free, _ = backend.gpu_memory()
    report_message(f"smoothing the shape runs on the CPU (slower): it needs about "
                   f"{(need + backend.GPU_HEADROOM) / 2**30:.1f} GB free on the GPU (with "
                   f"{backend.GPU_HEADROOM / 2**30:g} GB left for other programs), {free / 2**30:.1f} GB are")


def _array_module(elements, gpu_min=GPU_MIN_ELEMENTS, per_element=GPU_BYTES_PER_ELEMENT):
    """CuPy for big analyses when the GPU is on and has the memory, else numpy (if the PC has the
    memory: else an error rather than paging to a hard disk). When it does not fit on the GPU (or not
    in RAM) as things are, room is made, cheapest first, and it is decided again: this process's GPU
    search structures go (Warp meshes and BVHs of earlier runs), then the run's idle worker processes
    (after a Final preview five GPU workers held 3.2 GB of RAM and a CUDA context each, and an optimized
    part's check, 6.4M elements, fit neither on the GPU nor in the RAM left). An analysis the GPU cannot
    take after all says so in the log: on the CPU a result's check once ran 10 minutes."""
    gpu = elements >= gpu_min and backend.gpu_enabled()
    need = elements * CPU_BYTES_PER_ELEMENT
    if gpu and (_gpu_fits(elements, per_element) or
                (backend.release_gpu_caches() and _gpu_fits(elements, per_element, wait=0.5))):
        return backend.cupy()
    room = min(memory_limit(), memory_room())
    if (gpu or need > room) and stop_idle_workers():
        report_message(f"stopped the idle worker processes to make room for an analysis of {elements:,} elements "
                       f"(they start again on the next update)")
        if gpu and _gpu_fits(elements, per_element, wait=2.0):
            return backend.cupy()
        room = min(memory_limit(), memory_room())
    if need > room:
        raise GraphError(f"this analysis needs about {need / 2**30:.1f} GB of memory ({elements:,} elements) but "
                         f"only {max(room, 0) / 2**30:.1f} GB can be used: choose a lower detail")
    if gpu:
        free, _ = backend.gpu_memory()
        report_message(f"an analysis of {elements:,} elements runs on the CPU (slower): it needs about "
                       f"{(gpu_bytes(elements, per_element) + backend.GPU_HEADROOM) / 2**30:.1f} GB free on the GPU "
                       f"(with {backend.GPU_HEADROOM / 2**30:g} GB left for other programs), "
                       f"{free / 2**30:.1f} GB are")
    return np


@node("sim.static", "Static Analysis", SIM,
      [Port("part", "part", many=True,
            doc="The part: a CAD body (Import STEP, Pick Body) or an implicit body (e.g. a textured one). Several "
                "bodies are analysed as one piece, joined where they touch."),
       Port("material", "material", doc="From Material or Custom Material."),
       Port("conditions", "conditions", label="Supports & loads",
            doc="From Fixed Faces and Load (chained with their Add to): it needs at least one of each."),
       Port("detail", "text", "medium", choices=list(DETAIL),
            doc="How finely the part is divided: draft ~40,000 elements (about a second), medium ~250,000 (seconds), "
                "fine ~1 million (needs the GPU to be quick). The element size is in the log. Results are "
                "approximate at sharp inner corners and fillets: compare two details to see whether a peak is real."),
       Port("show", "text", "stress", choices=list(SHOW),
            doc="What colours the part in the 3D view (its eye button): stress (von Mises, MPa), safety factor "
                "(strength / stress: below 1 it breaks or bends for good, red) or displacement (how far it moves, mm).")],
      [Port("result", S, label="Shown result",
            doc="The field chosen by Show, e.g. for Remap -> a lattice's thickness."),
       Port("body", "implicit", doc="The part as analysed. Shown coloured by Show when the node's eye is on."),
       Port("stress", S, doc="von Mises stress, MPa: the usual single number for how hard the material works."),
       Port("safety", S, label="Safety factor",
            doc="Material strength / stress (shown up to 10). Below 1: it yields or breaks."),
       Port("displacement", S, doc="How far each point moves under the loads, mm.")],
      paint="result")
def static_analysis(part, material, conditions, detail, show):
    """How the part deforms under its loads, and how hard the material works (stress) everywhere.
    Linear: small deformations, the material springs back."""
    if material is None:
        raise GraphError("connect a Material")
    if not isinstance(conditions, Conditions) or not len(conditions):
        raise GraphError("connect Fixed Faces and a Load (Supports & loads)")
    if show not in SHOW:
        raise GraphError(f"show must be one of: {', '.join(SHOW)}")
    for cond in conditions:
        if not len(cond.triangles):
            raise GraphError(f"{'Fixed Faces' if cond.kind == 'fixed' else 'Load'}: no faces chosen yet (Pick)")
    body, volume, count = _part(part)

    def progress(iteration, residual):  # the residual falls from 1 to 1e-6: show that on a log scale
        report_progress(solved_share(residual), 100)

    try:
        result = FA.static_analysis(body, conditions, material, detail, xp=_array_module, cache=ARRAY_CACHE,
                                    store=ARRAY_CACHE.disk, callback=progress, volume=volume, cancel=check_cancel)
    except ConditionError as err:
        raise GraphError(str(err)) from err
    except NotConverged as err:
        raise GraphError(f"{err}: is the part held well enough? A part that can turn about its supports has no "
                         f"answer") from err
    finally:
        backend.release_gpu_memory()  # CuPy keeps its pool otherwise (3.6 GB after a fine analysis)
    lines = result.summary().splitlines()
    if count > 1:  # (not into the report: it is kept, and every run added the line again)
        lines.append(f"note: {count} bodies analysed as one piece, joined where they touch")
    for line in lines:
        report_message(line)
    # the numbers that matter on the card too (they were only in the log)
    text, warnings = judge(result.report, body)
    for line in warnings:
        report_message(line, "warning")
    f = result.fields
    shown = f[SHOW[show]]
    shown.note = (f"{text}; stress {result.report['max_away']:.3g} MPa away from the supports; mass "
                  f"{result.report['mass']:.3g} g")
    return {"result": shown, "body": body, "stress": f["stress"], "safety": f["safety"],
            "displacement": f["displacement"]}


# --- checking an optimizer's result ------------------------------------------------------------
# Deflections above this share of the part's size are too large for the linear (small-deformation)
# analysis to be exact.
LARGE_DEFLECTION = 0.02


def safety_text(sf):
    return f">{FA.SAFETY_CAP:g}" if sf >= FA.SAFETY_CAP else f"{sf:.3g}"


def check_result(part, conditions, material, detail):
    """Static Analysis of an optimizer's result, the body it shows: (report or None, note text,
    warnings). The optimizers judge their designs with their own models (densities, beams), which
    see neither the smoothed surface nor joints and bosses, and only relative stiffness."""
    result, text, warnings = _check(part, conditions, material, detail)
    return (result.report if result is not None else None), text, warnings


def solved_share(residual, tol=FA.TOLERANCE):
    """How far a solve has got, 1..99 (%): the residual falls from 1 to `tol` on a log scale."""
    return min(99, max(1, int(100 * math.log(max(residual, 1e-30)) / math.log(tol))))


PROGRESS_EVERY = 0.5   # s between the solver's progress events on a card (a GPU iteration takes milliseconds)


def _solver_progress(step):
    """progress(iteration, residual) for _check: the card's bar and step follow the analysis' solver (a
    result's check once ran 10 minutes on the CPU with the card saying nothing but "analysing the shape")."""
    shown = {"share": 0, "at": -math.inf}

    def progress(iteration, residual):
        share = max(shown["share"], solved_share(residual))   # (conjugate gradients' residual goes up and down)
        now = time.perf_counter()
        if share > shown["share"] or now - shown["at"] >= PROGRESS_EVERY:
            shown.update(share=share, at=now)
            report_progress(share, 100, f"{step} (solver iteration {iteration}), solved")
    return progress


def _check(part, conditions, material, detail, h=None, progress=None):
    """check_result with the whole analysis Result (its fields colour the body: Show). `h`: the element
    size (default: from `detail` and the part's volume). `progress(iteration, residual)` follows the solver."""
    def callback(iteration, residual):
        check_cancel()   # (Stop)
        if progress is not None:
            progress(iteration, residual)
    warnings = []
    try:
        result = FA.static_analysis(part, conditions, material, detail, xp=_array_module, cache=ARRAY_CACHE,
                                    store=ARRAY_CACHE.disk, callback=callback, cancel=check_cancel, h=h)
    except Cancelled:
        raise
    except (ConditionError, NotConverged, GraphError) as err:
        return None, "", [f"the result could not be checked with Static Analysis: {err}"]
    finally:
        backend.release_gpu_memory()
    text, warnings = judge(result.report, part)
    return result, text, warnings


def judge(r, part):
    """(note text, warnings) for an analysis report of `part`: its safety factor and deflection, and
    what is wrong with them (below 1; too large for a linear analysis)."""
    warnings = []
    lo, hi = part.bounds
    size = float(np.max(np.asarray(hi) - np.asarray(lo)))
    if r["min_safety"] < 1:
        warnings.append(f"safety factor {safety_text(r['min_safety'])}: the part would yield or break under these "
                        f"loads ({r['max_away']:.3g} MPa away from the supports, strength {r['strength']:g} MPa). "
                        f"More material, a stronger material or more supports help.")
    if r["max_displacement"] > LARGE_DEFLECTION * size:
        warnings.append(f"max deflection {r['max_displacement']:.3g} mm on a {size:.0f} mm part: too large for this "
                        f"(small-deformation) analysis to be exact")
    text = f"safety factor {safety_text(r['min_safety'])}, max deflection {r['max_displacement']:.3g} mm"
    if r.get("cases"):
        worst = min(r["cases"], key=lambda c: c["min_safety"])
        text += f" (the worst of {len(r['cases'])} load cases: {worst['name']})"
    return text, warnings


# --- topology optimization ---------------------------------------------------------------------
LIVE_EVERY = 3          # iterations between shapes shown in the 3D view while optimizing
# An optimization is ~70 solves: the GPU pays off sooner than for one analysis (3D cantilever,
# 47k elements: 0.3 s per iteration on the GPU, 1.0 s on the CPU)
TOPOLOGY_GPU_MIN = 10_000
CHECKPOINT_EVERY = 5    # iterations between saved states (Stop, a crash or a restart lose at most this)
FILLET_SIZE = 3.0 * (2.0 - math.sqrt(2.0))   # a G2 blend this x a radius wide is as deep as a round fillet of it
_TOPO_RESULTS = OrderedDict()  # (key, iteration, done, ...) -> (body, density, check): building them takes a while


def _validate(material, conditions):
    if material is None:
        raise GraphError("connect a Material")
    if not isinstance(conditions, Conditions) or not len(conditions):
        raise GraphError("connect Fixed Faces and a Load (Supports & loads)")
    for cond in conditions:
        if not len(cond.triangles):
            raise GraphError(f"{'Fixed Faces' if cond.kind == 'fixed' else 'Load'}: no faces chosen yet (Pick)")


def _optimize(key, body, volume, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, detail,
              state, max_thickness=0.0, members="plates and struts", gap=0.0, smoothness=0.0, symmetry="none",
              coarse=False, prefer_body=None, prefer_strength=0.0, prune=False, max_iterations=T.MAX_ITERATIONS,
              max_section=0.0, extrude="none"):
    """Run (or continue) the optimization, saving its state as it goes; with `coarse`, on the coarser
    details first (`T.levels`), each handing its design over to the next; with `prune`, dead ends kept
    empty as they show; at most `max_iterations` in all. Returns the final state."""
    levels = T.levels(body, detail, coarse, volume, min_thickness, max_thickness,
                      max_section if extrude == "none" else 0.0)   # (Extrude along turns Max section off)
    if state is None:
        first = 0
    else:   # where it stopped (a state without a level is on the chosen detail)
        saved = str(state["level"]) if "level" in state else detail
        first = levels.index(saved) if saved in levels else len(levels) - 1
    if len(levels) > 1 and state is None:
        report_message(f"Topology Optimization starts coarse: {', then '.join(levels)} elements (Start coarse)")
    t0 = time.perf_counter()
    handover = None
    for i in range(first, len(levels)):
        level, last = levels[i], i == len(levels) - 1
        try:
            problem = T.prepare(body, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, level,
                                xp=lambda n: _array_module(n, TOPOLOGY_GPU_MIN, TOPOLOGY_GPU_BYTES_PER_ELEMENT),
                                cache=ARRAY_CACHE, volume=volume,
                                max_thickness=max_thickness, members=members, gap=gap, smoothness=smoothness,
                                symmetry=symmetry, prefer_body=prefer_body, prefer_strength=prefer_strength,
                                prune=prune, max_section=max_section, extrude=extrude)
        except ConditionError as err:
            raise GraphError(str(err)) from err
        if handover is None:
            resume = state is not None and tuple(state["x"].shape) == tuple(problem.grid.shape)
            check_cancel()   # (preparing a fine problem takes ~30 s)
            opt = T.Optimizer(problem, state if resume else None, max_iterations,
                              callback=lambda it, r: check_cancel())
            how = f"continuing at iteration {opt.iteration}" if resume else "starting"
        else:
            check_cancel()
            opt = T.Optimizer(problem, max_iterations=max_iterations,
                              callback=lambda it, r: check_cancel()).take_over(handover)
            how = f"continuing at iteration {opt.iteration} on finer elements"
        if len(levels) > 1:
            opt.level = level
        device = "GPU" if problem.xp is not np else "CPU"
        free = int(problem.xp.sum(problem.free))
        report_message(f"Topology Optimization ({level}{'' if last else f' on the way to {detail}'}, "
                       f"{problem.grid.describe()}, {free:,} design elements, {device}): {how}")
        if last:   # (a coarser level's notes are mostly about thicknesses its elements cannot hold)
            for n in problem.notes:
                report_message(f"note: {n}")
        live = T.LiveSurface(backend.to_host(problem.phi), problem.grid.origin, problem.grid.h, problem.space)

        def on_iteration(o, compliance, kept, change, seconds):
            report_progress(o.iteration, max_iterations, "iteration")  # also where Stop takes effect
            if o.iteration % LIVE_EVERY == 0:
                try:
                    with trace.span("live view", item=o.iteration):
                        report_view(live(o.density()))
                except (ConditionError, ValueError):
                    pass  # no surface yet (all densities on one side of 0.5)
            if o.iteration % CHECKPOINT_EVERY == 0:
                with trace.span("checkpoint", item=o.iteration):
                    save_topology(key, **o.state(stopped=True))
            if o.iteration % 10 == 0:
                local = (f", fullest neighbourhood {100 * o.local_max:.0f}% (max {100 * T.LOCAL_SHARE:.0f}%)"
                         if o.local_max is not None else "")
                if o.section_max is not None:
                    area = problem.section_area
                    local += (f", fullest spot ~{area * o.section_max / problem.section_share:.0f} mm2 of "
                              f"section (max {area:g}, joints hold more)")
                dead = f", {o.dead_ends} dead ends kept empty" if o.dead_ends else ""
                report_message(f"iteration {o.iteration}: {100 * kept:.1f}% kept, stiffness "
                               f"{100 * o.full_compliance / o.design_compliance():.0f}% of the full design space"
                               f"{local}{dead} ({seconds:.1f} s per iteration)")

        try:
            opt.run(on_iteration, solver_callback=lambda it, r: check_cancel(),
                    hand_over=None if last else T.HAND_OVER[level])
            if not last:
                save_topology(key, **opt.state(stopped=True))   # (a Stop now continues on the next level)
                report_message(f"{level} done at iteration {opt.iteration} ({time.perf_counter() - t0:.0f} s, "
                               f"stiffness {100 * opt.full_compliance / opt.design_compliance():.0f}%): moving on "
                               f"to {levels[i + 1]} elements")
                handover = opt.handover()
                del opt, problem, live
        except (Cancelled, GraphError, NotConverged) as err:  # Stop, the memory watchdog, the solver: keep what is done
            # (the stiffness for the card; not when memory is short, nor on the CPU, where this solve kept a
            # fine run "Stopping…" for 70 s)
            if type(err) is Cancelled and problem.xp is not np:
                opt.design_compliance()
            save_topology(key, **opt.state(stopped=True))
            on = "" if last else f" on {level} elements"
            report_message(f"Topology Optimization stopped at iteration {opt.iteration}{on}: the shape so far is "
                           f"shown; Optimize continues from there")
            raise
        finally:
            backend.release_gpu_memory()
    opt.design_compliance()
    final = opt.state()
    save_topology(key, **final)
    report_message(f"Topology Optimization done in {time.perf_counter() - t0:.0f} s: {opt.summary()}")
    if opt.history and opt.history[-1][1] < keep / 100 - 0.01:
        short = f"note: {100 * opt.history[-1][1]:.1f}% kept instead of {keep:g}%: "
        if problem.section_sigma or problem.local_sigma:
            raised = (f", raised from {max_thickness:g} for these elements"
                      if problem.max_member > float(max_thickness) + 1e-9 else "")
            limit = (f"Max section ({problem.section_area:g} mm²)" if problem.section_sigma
                     else f"Max thickness ({problem.max_member:g} mm{raised})")
            # (a larger section only trades many thin members for fewer thick ones: the share that fits hardly moves)
            cure = ("A lower Keep % avoids this; a larger Max section adds little room" if problem.section_sigma
                    else "A larger limit or a lower Keep % avoids this")
            report_message(f"{short}no more material fits without a member over {limit}. {cure}."
                           + (" Smoothness works against these limits (it wants fewer, thicker members): a lower "
                              "Smoothness lets more fit." if smoothness else ""))
        elif prefer_body is not None and prefer_strength < 0:
            report_message(f"{short}Prefer's negative strength makes the rest dearer than it is worth. A smaller one "
                           f"or a smaller Prefer body lets it stay.")
        else:
            report_message(f"{short}the loads have no use for more material.")
    return final


def _nothing_left(state):
    """Why Optimize has nothing to do with this saved state (`T.finished`), for the card."""
    at = int(state["iteration"])
    return (f"Settled at iteration {at}, so there is nothing left to optimize. Change Keep, a thickness, Detail, or the "
            f"supports and loads to run it again (Smoothing and Show need no new run)." if T.settled(state) else
            f"Stopped at Max iterations ({at}): raise it to go on.")


BAR_COVER = 1.0   # mm: in walls, a bar's outer face lies this far (and the edge's size) outside the design space's
BAR_FILLET = 0.4  # x a bar's width: its fillets' radius where it joins others, when that is more than Fillet
BAR_FILLET_MOST = 10.0   # mm: ... and at most (the bosses' field is worked out that far past them)
BAR_FLAT = 0.94   # the cosine within which a member's pieces' normals are one (20 degrees)
BAR_MOSTLY = 0.6  # ... and the share of its length that has to look one way for all of it to
BAR_BENT = 0.1    # x a bar's width: a member's line bent more than this (rms from straight) has a plane of its own
BAR_KNOT = 1.0    # x half the widest bar's smaller side: the radius of the ball where bars of different planes meet
BAR_KNOT_NEAR = 0.8    # x a bar's larger side: members' ends this near each other are one joint
BAR_KNOT_REACH = 1.5   # x the larger side: the planes are blended in space this far from such a joint, fading to twice
BAR_LOFT = True   # a free part's joints are lofts: every bar cut short and lofted into each bar it meets (`bars.lofts`)
BAR_FILL = True   # a lofted joint's dimples and creases are filled (`bars.fills`) ...
BAR_PATCH = False # ... or rather its surface fitted again where its lofts overlap (`bars.patches`, as Rhino's Patch; off until the user chooses: FINDINGS "Patched joints")
BAR_BY_JOINTS = False  # (an experiment, off: a free part's bars laid by their joints' planes; FINDINGS "Bars laid by their joints")
BAR_JOINT_LOOK = 2.5   # x a bar's side: the way a member leaves a joint is read this far along it
BAR_JOINT_OUT = 0.3    # a member leaving a joint's plane by more than this (sine, 17 degrees) is not laid in it
BAR_SQUARE = (0.7, 1.6)    # a bar laid by its joints is this many times as wide as thick, at least and at most
BAR_TWIST = 2.5    # degrees per mm: a member turns from one joint's plane to the other's at most this fast, over half its length
BAR_TURN = (0.15, 0.4)     # of a member's length: where it turns from one joint's plane to the other's, from each end
BAR_LIFT = 0.35   # x a bar's width: a member's line that leaves its plane by less than this is laid in it
BAR_PAST = 0.3   # mm: ... and its inner face this far past the hollow's, which cuts it (short of it by a hair, a slot
                  # between the two was left under Keep; in one plane with it, an edged cut made stripes)


def _bars(made, space, avoid, fixed, size, least, walls, fillet, edge, edge_shape, mirror=(), follow=True):
    """Members as bars (`fea.bars`): the smoothed members `made` (`T.density_field`) traced into straight bars
    and drawn again: (the body, a note's numbers), or (None, why not): "none" when there is no member to
    trace, "loose" when the bars leave the part in more pieces than the members did (plates, which have no
    centre line, and members shorter than a bar is wide: a frame of bars is not what was optimized). `space`: the design space, `avoid` and `fixed`
    (the bosses and Keep) as bodies. A bar's faces lie as the design space's nearest face does (its normal is
    the design space's gradient at its middle), its width is `size`, or with 0 what its member's material
    needs: a square of the member's section, in walls a strip of it as thick as the walls. `follow`: the bars
    keep their members' form, the centre line's curve and the width along it; else straight bars of one width
    each. `fillet`: mm, the least radius of the fillets where bars meet (0 = none); a bar's own are BAR_FILLET x
    its width where that is more, so wide bars flow into each other and narrow ones keep a small radius (the
    note's numbers say which: `about.fillet`, the middle one, is the radius the bosses and Keep are joined
    with). In walls the bars
    are thicker than the walls, so that the design space and the hollow cut their two faces. `mirror`: (axis,
    mm) of Symmetry's planes: the bars are made mirror images across them (`bars.mirrored`)."""
    from ..fea import bars as B
    values = made.values[::2, ::2, ::2]
    voxel = 2.0 * float(made.voxel)
    grid = Grid.block(np.asarray(made.start, dtype=np.float64), voxel, (0, 0, 0), values.shape)
    xp = backend.cupy() if backend.use_gpu_for(values.size) else np
    read = lambda field: backend.to_host(F.Context(grid, xp=xp).eval_full(field))
    try:
        inside = read(space).astype(np.float32)      # mm, negative in the design space
        members = (values < 0) & (inside < 0)
        if avoid is not None:
            members &= read(avoid) >= 0
        depth = read(fixed).astype(np.float32) if fixed is not None else np.ones(values.shape, dtype=np.float32)
    finally:
        if xp is not np:
            backend.release_gpu_memory()
    if not members.any():
        return None, "none"
    held = depth < 0

    def at(array, points):   # (between the samples)
        return ndimage.map_coordinates(array, ((np.asarray(points) - np.asarray(made.start)) / voxel).T, order=1,
                                       mode="nearest")

    def slope(array, points):
        """The unit direction in which `array` grows at each point (zeros where it does not)."""
        g = np.stack([at(d, points) for d in np.gradient(array, voxel)], axis=1)
        size_ = np.linalg.norm(g, axis=1, keepdims=True)
        return np.where(size_ > 1e-6, g / np.maximum(size_, 1e-6), 0.0)

    # a joint held by a boss or Keep comes to rest in the body's middle nearest by: up the distance from the
    # body's surface (inside it) to its ridge, a boss's axis or the middle of a plate
    core = (ndimage.distance_transform_edt(held) * voxel).astype(np.float32)

    def settle(points, reach):
        points, gone = np.array(points, dtype=np.float64), np.zeros(len(points))
        height, step = at(core, points), 0.5 * voxel
        for _ in range(int(float(np.max(reach)) / step) + 1):
            there = points + step * slope(core, points)
            higher = at(core, there)
            go = (higher > height + 1e-3 * voxel) & (gone + step <= reach)
            if not go.any():
                break
            points[go], height[go] = there[go], higher[go]
            gone[go] += step
        return points

    net = B.trace(members | held, held, made.start, voxel, size=float(size), least=float(least), cancel=check_cancel,
                  settle=settle, thick=float(walls), follow=bool(follow), normal=lambda points: slope(inside, points))
    for axis, plane in mirror:
        net = B.mirrored(net, axis, plane)
    if not len(net.a):
        return None, "none"
    a, b = net.a.copy(), net.b.copy()
    normal = B.normals(a, b, slope(inside, 0.5 * (a + b)))
    # a member's pieces lie in one plane where they nearly do (each piece's own normal and depth left the
    # pieces of a curved bar a hair askew of each other: shards at every joint between them)
    flat, facing = {}, {}
    long = np.linalg.norm(b - a, axis=1)
    by_joints = BAR_BY_JOINTS and walls <= 0     # a free part: the bars are laid by their joints (below)
    lines = (a.copy(), b.copy())
    for m in np.unique(net.branch):
        of = np.flatnonzero(net.branch == m)
        # the way most of the member looks (by length): near an edge of the design space single pieces look
        # the neighbouring face's way
        ways, votes = np.linalg.eigh((normal[of] * long[of, None]).T @ normal[of])[1][:, -1], normal[of] @ normal[of].T
        ways = ways if float((normal[of] @ ways) @ long[of]) >= 0 else -ways
        agree = normal[of] @ ways > BAR_FLAT
        flat[m] = bool(long[of][agree].sum() > BAR_MOSTLY * long[of].sum())
        del votes
        if not flat[m]:
            # a member deep in the part, as far from several faces: the design space's nearest face is another
            # one every few pieces, and pieces turned 30-80 degrees against each other stood out of each other's
            # faces as crumpled shards. One direction for the whole member instead: across the plane its line
            # bends in, or, for a straight one, the way most of it looked, square to its line. A line that
            # leaves that plane by less than BAR_LIFT x its width is laid in it (pieces 4-10 degrees askew
            # still showed as ribs across the bar); one that leaves it farther gets each piece's normal made
            # square to the piece, so its faces turn along it
            ends = np.concatenate([a[of], b[of]])
            spread, planes_ = np.linalg.svd((ends - ends.mean(axis=0)) * np.tile(np.sqrt(long[of]), 2)[:, None],
                                            full_matrices=False)[1:]
            wide_ = float(np.median(np.concatenate([net.width_a[of], net.width_b[of]])))
            if len(of) > 1 and spread[1] > BAR_BENT * wide_ * math.sqrt(2.0 * long[of].sum()):
                ways = planes_[2] if planes_[2] @ ways >= 0 else -planes_[2]
            else:
                square = ways - (ways @ planes_[0]) * planes_[0]
                if np.linalg.norm(square) > 0.2:
                    ways = square / np.linalg.norm(square)
            off = (ends - ends.mean(axis=0)) @ ways
            normal[of] = B.normals(a[of], b[of], np.tile(ways, (len(of), 1)))
            flat[m] = bool(np.abs(off).max() < BAR_LIFT * wide_ and (normal[of] @ ways).min() > BAR_FLAT)
        facing[m] = ways
        if flat[m]:
            # ... and in one plane: a joint that lies inside a boss lies at the boss's depth, not the member's
            deep = np.concatenate([a[of] @ ways, b[of] @ ways])
            level = float(np.median(deep))
            a[of] += (level - a[of] @ ways)[:, None] * ways
            b[of] += (level - b[of] @ ways)[:, None] * ways
            normal[of] = B.normals(a[of], b[of], np.tile(ways, (len(of), 1)))
    if by_joints:
        a, b = lines
    width, width_b = net.width_a.copy(), net.width_b.copy()
    # as thick as the member is along its normal (one thickness along a member), or as it is wide
    thick = np.where(net.span > 0, net.span, 0.5 * (width + width_b)) if size <= 0 else 0.5 * (width + width_b)
    for m in np.unique(net.branch):
        thick[net.branch == m] = thick[net.branch == m].mean()
    if walls > 0:
        # strips of the walls: every bar's inner face a wall's thickness in, its outer face past the design
        # space's (which cuts it, and breaks that rim), wherever the centre line ran in the wall
        # (how deep a bar lies is measured from its middle straight out along its normal, as far as the
        # design space's face there: the distance to the nearest face is another face's near an edge, and a
        # bar's ends lie in the plate or a boss)
        cover = float(edge) + BAR_COVER
        inner = float(walls) + BAR_PAST
        thick = np.full(len(width), inner + cover)
        middle = 0.5 * (a + b)
        steps = np.arange(0.0, 1.5 * walls + voxel, 0.5 * voxel)
        along = np.stack([at(inside, middle + t * normal) for t in steps], axis=1)      # (bars, steps)
        out = along >= 0
        first = np.argmax(out, axis=1)
        rows = np.arange(len(a))
        before = along[rows, np.maximum(first - 1, 0)]
        deep = steps[first] - np.where(first > 0, along[rows, first] / np.maximum(along[rows, first] - before, 1e-6),
                                       0.0) * (0.5 * voxel)
        found = out.any(axis=1) & (along[:, 0] < 0)
        move = np.where(found, deep - 0.5 * (inner - cover), 0.0)
        for m, same in flat.items():
            of = np.flatnonzero((net.branch == m) & found)
            if same and len(of):
                move[net.branch == m] = np.median(move[of])
        a += move[:, None] * normal
        b += move[:, None] * normal
    member, thick_b, twist = net.branch, None, None
    if by_joints:
        # A free part's bars are laid by their joints. Where members meet they mostly lie in one plane (9 of the
        # bolted bracket's 12 joints), and bars in one plane with one thickness join as a wall's do, by fillets
        # in that plane. Each bar on its own (its faces as the part's nearest face lies, as thick as its member)
        # put 27 bars in 21 planes: at every joint strips turned against each other, of other thicknesses. So a
        # joint's bars share its plane and thickness, and a bar whose two joints lie otherwise turns and tapers
        # from one to the other along its middle
        from scipy.spatial import cKDTree
        order = {int(m): np.flatnonzero(net.branch == m) for m in np.unique(net.branch)}
        chained = {m: bool(len(of) == 1 or np.allclose(b[of[:-1]], a[of[1:]], atol=1e-6)) for m, of in order.items()}
        section = 0.5 * (width + width_b) * thick
        side = {m: math.sqrt(float(np.median(section[of]))) for m, of in order.items()}
        ends = []      # (member, which end, the point, the way the member leaves it)
        for m, of in order.items():
            if not chained[m]:
                continue
            points, run = np.concatenate([a[of[:1]], b[of]]), np.concatenate([[0.0], np.cumsum(long[of])])
            look = min(BAR_JOINT_LOOK * side[m], float(run[-1]))
            for which, at_ in ((0, look), (1, float(run[-1]) - look)):
                way = np.array([np.interp(at_, run, points[:, d]) for d in range(3)]) - points[-which]
                ends.append((m, which, points[-which], way / max(float(np.linalg.norm(way)), 1e-9)))
        group = np.arange(len(ends))
        if len(ends) > 1:
            where = np.array([e[2] for e in ends])
            for i, j in sorted(cKDTree(where).query_pairs(BAR_KNOT_NEAR * max(side.values()))):
                if np.linalg.norm(where[i] - where[j]) < BAR_KNOT_NEAR * max(side[ends[i][0]], side[ends[j][0]]):
                    group[group == group[j]] = group[i]
        lies = {}      # (member, which end): (the joint's plane, its layer, its thickness)
        for g in np.unique(group):
            of = [ends[i] for i in np.flatnonzero(group == g)]
            if len(of) < 2:
                continue
            ways = np.array([e[3] for e in of])
            if len(of) == 2 and abs(float(ways[0] @ ways[1])) > BAR_FLAT:
                continue       # (two bars in one line have no plane)
            across_ = np.linalg.eigh((ways * np.array([side[e[0]] ** 2 for e in of])[:, None]).T @ ways)[1][:, 0]
            inside_ = [e for e, way in zip(of, ways) if abs(float(way @ across_)) < BAR_JOINT_OUT]
            if len(inside_) < 2:
                continue
            tall = float(np.median([side[e[0]] for e in inside_]))
            centre = np.mean([e[2] for e in inside_], axis=0)
            for e in inside_:
                lies[(e[0], e[1])] = (across_, ("joint", int(g)), tall, centre)
        tokens = {}

        def token(t):
            while tokens.setdefault(t, t) != t:
                t = tokens[t]
            return t
        normal, thick, thick_b, twist = normal.copy(), thick.copy(), thick.copy(), np.zeros(len(a))
        member, halves = net.branch.copy(), {}
        ease = lambda v: np.clip(v, 0.0, 1.0) ** 2 * (3.0 - 2.0 * np.clip(v, 0.0, 1.0))
        for m, of in order.items():
            if not chained[m]:
                halves[m] = (("member", m), ("member", m), None)
                continue
            first, last = lies.get((m, 0)), lies.get((m, 1))
            # an end in a boss or Keep lies as the part's face does there (as the boss's own faces: in its
            # joint's plane at the other end, a wide bar stood 5 mm out of the ring it ends in)
            for which, piece in ((0, of[0]), (1, of[-1])):
                end_ = (b if which else a)[piece]
                if (first, last)[which] is None and float(at(depth, end_[None])[0]) < 0 and np.any(normal[piece]):
                    here = (normal[piece].copy(), ("end", m, which), side[m], None)
                    first, last = (here, last) if which == 0 else (first, here)
            first, last = first or last, last or first
            if first is None:
                first = last = (facing[m], ("member", m), side[m], None)
            (p0, t0, tall0, _), (p1, t1, tall1, _) = first, last
            p1 = p1 if float(p0 @ p1) >= 0 else -p1
            run = np.concatenate([[0.0], np.cumsum(long[of])])
            share = run / max(float(run[-1]), 1e-9)
            turned = float(p0 @ p1) < BAR_FLAT
            laid = [True, True]
            if turned and math.degrees(math.acos(min(float(p0 @ p1), 1.0))) > BAR_TWIST * 0.5 * float(run[-1]):
                # too short to turn that far (a bar turned 40-100 degrees in 10 mm was crumpled): it keeps one
                # joint's plane all along and meets the other joint as a bar of another plane does
                keep_ = 0 if lies.get((m, 0)) is not None or lies.get((m, 1)) is None else 1
                (p0, t0, tall0), (p1, t1, tall1) = ((p0, t0, tall0),) * 2 if keep_ == 0 else ((p1, t1, tall1),) * 2
                turned, laid[1 - keep_] = False, False
            if not turned:
                tokens[token(t1)] = token(t0)
            points = np.concatenate([a[of[:1]], b[of]])
            # in its middle a bent member lies in the plane it bends in (a straight one between its joints'
            # planes); laid whole in a joint's plane its line moved up to 9.5 mm, 22% of the bars' material
            # came to lie outside the part and the part's faces cut it to shards
            mid = p0 + p1
            mid = mid / float(np.linalg.norm(mid))
            rest = (points - points.mean(axis=0)) * np.sqrt(np.gradient(run))[:, None]
            spread, ways_ = np.linalg.svd(rest, full_matrices=False)[1:]
            if len(of) > 1 and spread[1] > BAR_BENT * side[m] * math.sqrt(float(run[-1])):
                mid = ways_[2] if float(ways_[2] @ mid) >= 0 else -ways_[2]
                off = (points - points.mean(axis=0)) @ mid
                if float(np.abs(off).max()) < BAR_LIFT * side[m]:
                    points = points - off[:, None] * mid
            # near a joint it lies in the joint's plane, its end at the joint's middle (ends a few mm apart
            # left a hole in the joint between them)
            for which, near, across_ in ((0, run, p0), (1, run[-1] - run, p1)):
                held_ = lies.get((m, which))
                if held_ is not None and laid[which]:
                    pull = ease(1.0 - near / (BAR_JOINT_LOOK * side[m]))[:, None]
                    points = points + pull * (held_[3] - points[-which])
                    flat_ = ease(2.0 * (1.0 - near / (BAR_JOINT_LOOK * side[m])))[:, None]      # (all of it for half that way)
                    points = points - flat_ * ((points - held_[3]) @ across_)[:, None] * across_
            # the faces turn from one plane to the next in two stretches; the ends and the middle fifth do not
            # turn (a turned member's halves join there)
            first_, second_ = (ease((share - at_) / (BAR_TURN[1] - BAR_TURN[0])) for at_ in (BAR_TURN[0], 1.0 - BAR_TURN[1]))
            target = (1.0 - first_)[:, None] * p0 + (first_ - second_)[:, None] * mid + second_[:, None] * p1
            target /= np.linalg.norm(target, axis=1, keepdims=True)
            a[of], b[of] = points[:-1], points[1:]
            long[of] = np.linalg.norm(b[of] - a[of], axis=1)
            n_a, n_b = B.normals(a[of], b[of], target[:-1]), B.normals(a[of], b[of], target[1:])
            axis_ = (b[of] - a[of]) / long[of, None]
            normal[of] = n_a
            twist[of] = np.arctan2((n_b * np.cross(n_a, axis_)).sum(axis=1), (n_b * n_a).sum(axis=1))
            tall = tall0 + (tall1 - tall0) * share
            for ends_, tall_ in ((width, tall[:-1]), (width_b, tall[1:])):
                ends_[of] = np.clip(ends_[of] * thick[of] / tall_, BAR_SQUARE[0] * tall_, BAR_SQUARE[1] * tall_)
            thick[of], thick_b[of] = tall[:-1], tall[1:]
            halves[m] = (t0, t1, 0.5 * (share[:-1] + share[1:]) >= 0.5 if turned else None)
        names = {}
        layer = np.zeros(len(a), dtype=np.int64)
        for m, of in order.items():
            t0, t1, second = halves[m]
            which = np.zeros(len(of), dtype=bool) if second is None else second
            for half, t in ((False, t0), (True, t1)):
                layer[of[which == half]] = names.setdefault(token(t), len(names))
            # (a turned member's halves are members of their own, each of its joint's layer)
            member[of] = 2 * m + (which if second is not None else 0)
    else:
        # the members that lie in one plane are one layer: fillets are for bars that meet in their plane (a
        # blend between a bar of the front wall and one of the back wall is a web from wall to wall)
        layer, planes = np.zeros(len(a), dtype=np.int64), []
        for m in np.unique(net.branch):
            of = net.branch == m
            height = float(np.median(np.concatenate([a[of] @ facing[m], b[of] @ facing[m]])))
            found = next((k for k, (ways, at_, deep_) in enumerate(planes) if flat[m] and deep_ is not None
                          and ways @ facing[m] > BAR_FLAT and abs(at_ - height) < 0.5 * deep_), None)
            if found is None:
                planes.append((facing[m], height, float(thick[of].mean()) if flat[m] else None))
                found = len(planes) - 1
            layer[of] = found
    # fillets: a bar's own radius where it joins others, by its width
    radius = np.zeros(len(a))
    if fillet > 0:
        for m in np.unique(net.branch):
            of = net.branch == m
            radius[of] = max(float(fillet), min(BAR_FILLET * float(np.median(np.concatenate([width[of], width_b[of]]))),
                                                BAR_FILLET_MOST))
    blend = FILLET_SIZE * radius
    # where bars of different planes meet (a frame in 3D): a ball the bars' ends lie in, and round it the planes
    # are blended in space (`B.body` knots). A joint is where members' ends lie together
    from scipy.spatial import cKDTree
    tips, whose = [], []
    for m in np.unique(net.branch):
        of = np.flatnonzero(net.branch == m)
        points = np.concatenate([a[of], b[of]])
        alone = np.array([len(n) for n in cKDTree(points).query_ball_point(points, 1e-6)]) == 1
        tips += list(points[alone])
        whose += [int(of[i % len(of)]) for i in np.flatnonzero(alone)]      # (the piece that ends there)
    knots, knot_size, knot_blend, knot_reach = [], [], [], []
    lofted = []      # the joints that are lofts: the member ends, (member, 0 its start or 1 its end), that meet there
    if len(tips) > 1:
        tips, whose = np.array(tips), np.array(whose)
        size_ = np.maximum(0.5 * (width + width_b), thick)       # per piece: its member's larger side
        for m in np.unique(net.branch):
            size_[net.branch == m] = np.median(size_[net.branch == m])
        group = np.arange(len(tips))
        for i, j in sorted(cKDTree(tips).query_pairs(BAR_KNOT_NEAR * float(size_.max()))):
            if np.linalg.norm(tips[i] - tips[j]) < BAR_KNOT_NEAR * max(size_[whose[i]], size_[whose[j]]):
                group[group == group[j]] = group[i]
        starts = {int(m): a[np.flatnonzero(net.branch == m)[0]] for m in np.unique(net.branch)}
        for g in np.unique(group):
            of = whose[group == g]
            centre = tips[group == g].mean(axis=0)
            if BAR_LOFT and walls <= 0 and not by_joints and len(of) > 1 and float(at(depth, centre[None])[0]) >= 0:
                # a free part's joint: no bar runs through it or ends in it, each is lofted into the others
                # (`B.lofts`); no fillets in a plane, no ball
                lofted.append([(int(net.branch[i]), 0 if np.linalg.norm(tip - starts[int(net.branch[i])]) < 1e-6 else 1)
                               for i, tip in zip(of, tips[group == g])])
                continue
            if len(set(layer[of].tolist())) < 2:
                continue
            lean = np.minimum(0.5 * (width + width_b), thick)[of]
            knots.append(centre)
            # (in a boss or Keep the joint needs no ball of its own)
            knot_size.append(0.0 if float(at(depth, centre[None])[0]) < 0 else BAR_KNOT * 0.5 * float(lean.max()))
            knot_blend.append(float(blend[of].min()))
            knot_reach.append(BAR_KNOT_REACH * float(size_[of].max()))
    flat_a = flat_b = None
    curves = ()
    if lofted:
        made_ = B.lofts(a, b, normal, width, width_b, thick, net.branch, lofted, edge)
        member, layer, blend = np.asarray(member)[made_.piece], layer[made_.piece], blend[made_.piece]
        thick_b = None if thick_b is None else thick_b[made_.piece]
        twist = None if twist is None else twist[made_.piece]
        a, b, normal, width, width_b, thick = made_.a, made_.b, made_.normal, made_.width, made_.width_b, made_.thick
        flat_a, flat_b, curves = made_.flat_a, made_.flat_b, made_.curves
    drawn = lambda fills=(), seams=False: B.body(
        a, b, normal, width, thick, blend, edge, edge_shape, reach=max(float(blend.max()), edge, 1.0) + 1.0,
        width_b=width_b, member=member, layer=layer, thick_b=thick_b, twist=twist,
        chain=net.branch if by_joints else None, knots=np.array(knots).reshape(-1, 3),
        knot_size=knot_size, knot_blend=knot_blend, knot_reach=knot_reach,
        flat_a=flat_a, flat_b=flat_b, curves=curves, fills=fills, seams=seams)
    field = drawn()
    if lofted and BAR_FILL and made_.joints:
        # a joint's lofts cover its middle each with its own faces, with grooves and steps between them, and leave
        # a dimple on the outside of a corner in space: each joint's surface flows out there (`B.fills`), read off
        # the bars as drawn
        def read(local, bars=field):
            on = backend.cupy() if backend.use_gpu_for(int(np.prod(local.shape))) else np
            try:
                return backend.to_host(F.Context(local, xp=on).eval_full(bars))
            finally:
                if on is not np:
                    backend.release_gpu_memory()
        # (and where two lofts meet, so that their sharp edges are rounded and the bars' own are not)
        seams_ = lambda local, seams=drawn(seams=True): read(local, seams)
        if BAR_PATCH:     # (each joint's surface fitted again where its lofts overlap, as Rhino's Patch: `B.patches`)
            filled = B.patches(read, made_.joints, seams_, cancel=check_cancel)
        else:
            filled = B.fills(read, made_.joints, cancel=check_cancel, seams=seams_)
        if filled:
            field = drawn(filled)
    pieces = lambda solid: ndimage.label(solid, structure=np.ones((3, 3, 3), dtype=bool))[1]
    if pieces((F.Context(grid).eval_full(field) < 0) & (inside < 0) | held) > pieces(members | held):
        return None, "loose"
    return field, SimpleNamespace(bars=net.members, pieces=len(net.a), fillet=float(np.median(radius)),
                                  width=(float(min(width.min(), width_b.min())), float(max(width.max(), width_b.max()))))


def _result(key, state, body, keep_body, smoothing=T.SMOOTHING, avoid_body=None, min_thickness=0.0, check=None,
            skin=2.0, trim=True, fillet=0.0, members=None, even=0.0, axis=None, edge=0.0, edge_shape="round",
            members_as="optimized", bar_size=0.0, walls=0.0, mirror=(), bar_form="follow"):
    """(body, density field, (note text, warnings) of its Static Analysis, that analysis' fields or
    None) of a saved state, kept for a few states. Members are at least the min thickness (automatic:
    MIN_MEMBER elements) after the smoothing, whatever it is. `check`: (conditions, material, detail)
    to analyse the body with. `trim`: dead ends that analysis finds (`T.dead_ends`) are emptied in the
    densities and the body made and analysed again. `fillet`: mm, the inside corners where members meet
    the skin's bosses and Keep. `members(state)`: gives the members' field of a state saved without one
    (`T.members_field`), or None when it cannot. `even`: Even out (`T.even_out`, of the members only: not for a
    state without their field), `axis`: Extrude along's. `edge`: mm, the edges where the shape meets the
    design space's faces rounded, or chamfered (`edge_shape`; `F.intersect_edged`)."""
    # (the material's strength is not in the topology key, but the safety factor needs it)
    base_key = (key, int(state["iteration"]), bool(state["done"]), str(state.get("level", "")), smoothing,
                check[1].key if check else None, bool(trim) and check is not None, float(skin))
    memo_key = (*base_key, float(fillet), float(even), float(edge), str(edge_shape), str(members_as), float(bar_size),
                str(bar_form))
    if memo_key in _TOPO_RESULTS:
        _TOPO_RESULTS.move_to_end(memo_key)
        return _TOPO_RESULTS[memo_key]
    g = state["grid"]
    origin, h = g[:3], float(g[3])
    density = state["density"]
    # the body is the members' own field, smoothed, joined with the exact bodies of what the optimizer kept
    # solid: the skin at supports and loads, and Keep (`T.members_field`). States saved before 2026-11-12 have no
    # such field: worked out once from their design variables and saved with them; when that fails, the
    # densities with those regions in them as elements, as before (no fillet then)
    smooth = T.saved_members(state)
    if smooth is None and members is not None:
        smooth = members(state)
    if smooth is not None and np.shape(smooth) != density.shape:
        smooth = None
    conds = check[0] if check is not None else ()
    reach = T.skin_reach(skin, min_thickness, h)
    # (a blend of the frames' kind, curvature continuous, as deep in its middle as a round fillet of that radius:
    # the Booleans' rolling ball, `union_edged`, reads how each body's values curve, and pitted on these)
    blend = FILLET_SIZE * float(fillet) if smooth is not None else 0.0
    # (bars have fillets by their width, up to BAR_FILLET_MOST: the bosses' field has to be right that far out)
    beyond = FILLET_SIZE * max(float(fillet), BAR_FILLET_MOST) if members_as == "bars" and fillet > 0 else blend
    around = T.skin_body(conds, reach, beyond=beyond) if smooth is not None and skin > 0 and len(conds) else None
    fixed = None
    if smooth is not None:
        for part in (around, keep_body):
            if part is not None:
                fixed = part if fixed is None else F.union(fixed, part)
    thinnest = max(float(min_thickness), T.MIN_MEMBER * h)
    # (the grid's own limit, 6 mm at draft, is no reason to cut a 2 mm skin: nothing asked for, nothing cut)
    asked = max(float(min_thickness), float(skin))
    crop = body if avoid_body is None else F.subtract(body, avoid_body)

    def shape(density, cut):
        check_cancel()   # (the smoothing and its analysis take a while: Stop between them)
        _room_for_smoothing(np.shape(density), smoothing)
        source = density
        if smooth is not None:   # (dead ends cut are empty in it too)
            source = smooth if cut is None else np.where(cut > 0, 0.0, smooth).astype(np.float32)
        try:
            # the smoothed samples as a smooth field of their own (a cubic B-spline), not the distance to a mesh
            # of them: that mesh's facets showed as stripes along every member (FINDINGS, "Polishing")
            made = T.density_field(source, state["phi"], origin, h, blur=smoothing,
                                   space=state["space"] if "space" in state else None,  # (older states have none)
                                   min_thickness=thinnest, crop=crop, cancel=check_cancel,
                                   conditions=conds, tips_under=min(thinnest, asked), fixed=fixed,
                                   even=even if smooth is not None else 0.0, axis=axis, avoid=avoid_body)
        except ConditionError as err:
            raise GraphError(str(err)) from err
        out = F.sampled_body(made.values, made.start, made.voxel, made.bounds)
        made.metadata.pop("bars", None)
        made.metadata.pop("no_bars", None)
        if members_as == "bars" and smooth is not None:
            # the members drawn again as bars along their centre lines, joined to the same bosses and Keep
            bars, about = _bars(made, body, avoid_body, fixed, bar_size, 0.0 if bar_size > 0 else thinnest, walls,
                                float(fillet), float(edge), edge_shape, mirror, follow=bar_form == "follow")
            if bars is not None:
                out, made.metadata["bars"] = bars, about
            else:
                made.metadata["no_bars"] = about
        # (bars join the bosses and Keep with their own middle fillet)
        join = FILLET_SIZE * made.metadata["bars"].fillet if "bars" in made.metadata else blend
        if smooth is None:   # (the densities, with the skin and Keep in them as elements)
            out = F.intersect_edged(body, out, edge, edge=edge_shape)
            if avoid_body is not None:  # exactly its faces (smoothing may round a little into it)
                out = F.intersect_edged(out, -avoid_body, edge, bounds="first", edge=edge_shape)
            return made, (out if keep_body is None else F.union(out, keep_body))
        # the members, then the bosses at supports and loads (true cylinders round holes and pins) and Keep,
        # each joined with the fillet; the design space crops it last. Cropped first, the members had the design
        # space's faces as their own, and where Keep is part of the design space (a plate over a web) those
        # are Keep's faces too: no corner between two bodies was left for a fillet to fill, only half of one.
        within = body
        if around is not None:
            out = F.smooth_union(out, around, join, 2)
        if keep_body is not None:
            out = F.smooth_union(out, keep_body, join, 2)
            within = F.union(body, keep_body)
        # (the edges this crop makes, where a member's side meets a face of the design space, can be rounded:
        # no CAD model has them. The design space's and Keep's own edges are in both bodies and stay)
        out = F.intersect_edged(within, out, edge, keep=keep_body, edge=edge_shape)
        if avoid_body is not None:  # exactly its faces (smoothing may round a little into it)
            # (bars in walls reach just past the hollow's face, their rims already broken: breaking the edge
            # between two faces a hair apart left streaks along every bar)
            inner = 0.0 if walls > 0 and "bars" in made.metadata else edge
            out = F.intersect_edged(out, -avoid_body, inner, bounds="first", edge=edge_shape)
        return made, out

    # dead ends found before are on the SSD (each round of finding them is a smoothing and an analysis)
    trim = trim and check is not None
    store = ARRAY_CACHE.disk if trim else None
    cut_key = ("topology-dead-ends", base_key, T.DEAD_SHARE, T.DEAD_IDLE, T.DEAD_PASSES, "sticking out past min",
               "members" if smooth is not None else "sampled body", *((("even", float(even)),) if even else ()))
    design = float((density >= 0.5).sum())
    cut = store.load(cut_key) if store is not None else None
    if cut is not None and cut.shape == density.shape:
        density = np.where(cut > 0, 0.0, density).astype(np.float32)
    else:
        cut = None
    # (minutes on a fine grid, after the last iteration: the card said iteration 94 / 125 all that time, and
    # the part looked stuck)
    stages = 3 if trim and cut is None else 2

    def analyse(out, step):
        # (sized from the shape's own volume, not on the optimization's grid: a web of struts there came out 38%
        # too stiff, deflection 0.234 mm instead of 0.375 at the check's own 0.49 mm; FINDINGS, "The result's check")
        return _check(out, *check, progress=_solver_progress(step))
    report_progress(1, stages, "smoothing the shape")
    made, out = shape(density, cut)
    report_progress(2, stages, "analysing the shape")
    analysis, text, warnings = analyse(out, "analysing the shape") if check is not None else (None, "", [])
    if trim and cut is None and analysis is not None:
        report_progress(3, stages, "looking for dead ends")
        centres = Grid.block(origin + h / 2, h, (0, 0, 0), density.shape)
        cut = np.zeros(density.shape, dtype=np.uint16)
        for _ in range(T.DEAD_PASSES):  # (a cut branch's root is the next dead end)
            check_cancel()
            stress = backend.to_host(F.Context(centres).eval_full(analysis.fields["stress"]))
            found, count = T.dead_ends(density, stress, origin, h, check[0], skin, reach=thinnest)
            del stress
            if not count:
                break
            removing = f"removing {count} dead end{'s' if count > 1 else ''}"
            report_progress(3, stages, removing)
            cut = np.where(found > 0, found + int(cut.max()), cut).astype(np.uint16)
            density = np.where(found > 0, 0.0, density).astype(np.float32)
            made, out = shape(density, cut)
            analysis, text, warnings = analyse(out, f"{removing}: analysing the shape again")
            if analysis is None:
                break
        if analysis is not None and store is not None:
            store.save(cut_key, cut)
    trimmed, rho_key = "", f"{key}-{base_key[1]}-rho"
    if cut is not None and cut.any():
        total, removed = int(cut.max()), float((cut > 0).sum()) / max(design, 1.0)
        trimmed = f"{total} dead end{'s' if total > 1 else ''} removed (-{100 * removed:.1f}% material)"
        rho_key += "-trimmed"
    added = float(made.metadata.get("thickened", 0.0))
    grown = f"members thickened to {thinnest:.2g} mm (+{100 * added:.0f}% material)" if added >= 0.005 else ""
    tips = int(made.metadata.get("tips", 0))
    tips = f"{tips} thin tip{'s' if tips > 1 else ''} cut off" if tips else ""
    about = made.metadata.get("bars")
    if about is not None:
        low, high = about.width
        wide = f"{low:.2g} mm" if high - low < 0.05 else f"{low:.2g} to {high:.2g} mm"
        tips = " · ".join(t for t in (tips, f"drawn as {about.bars} bar{'s' if about.bars != 1 else ''} {wide} wide") if t)
    elif members_as == "bars" and made.metadata.get("no_bars") == "loose":
        warnings = [*warnings, "Members as bars: drawn as bars this part falls apart (its members are plates, or "
                               "shorter than a bar is wide), so the optimized shape is shown. Bars suit a shape "
                               "made of struts: try Walls, a Max thickness with Members: struts only, or a "
                               "smaller Bar size"]
    elif members_as == "bars":
        warnings = [*warnings, "Members as bars: no members to draw as bars (a result optimized by an earlier "
                               "version has to be optimized again), so the optimized shape is shown"]
    centres = SimpleNamespace(origin=origin + h / 2, h=h)
    rho = FA.nodal_field(density, centres, rho_key, "density", "", (0.0, 1.0), body.bounds)
    missed = [f"{label}: the shape does not reach this load, so nothing carries it" if share > 0.99 else
              f"{label}: the shape does not reach {100 * share:.0f}% of this load, so nothing carries that part of it"
              for label, share in made.metadata.get("unreached", ())]
    if missed:   # (a member thinner than the filter's radius carries nothing, so too little material breaks the path)
        missed[0] += (". Too little material for members this thick? A higher Keep %, a finer Detail or a smaller Min "
                      "thickness gives it more to reach them with")
    checked = (" · ".join(t for t in (trimmed, grown, tips, text) if t), [*missed, *warnings])
    fields = analysis.fields if analysis is not None else None
    _TOPO_RESULTS[memo_key] = (out, rho, checked, fields)
    while len(_TOPO_RESULTS) > 4:
        _TOPO_RESULTS.popitem(last=False)
    return out, rho, checked, fields


@node("sim.topology", "Topology Optimization", SIM,
      [Port("space", "part", many=True, label="Design space",
            doc="The room the part may fill: a CAD body (model it in SolidWorks as the largest envelope, with "
                "its bolt holes and mounting faces) or an implicit body."),
       Port("material", "material", doc="From Material or Custom Material."),
       Port("conditions", "conditions", label="Supports & loads",
            doc="From Fixed Faces and Load (chained with Add to), picked on the design space's faces."),
       Port("keep", "number", 30.0, label="Keep", unit="%", slider=(0, 100),
            doc="How much of the design space's mass to keep, %. Less = lighter but bendier."),
       Port("min_thickness", "number", 0.0, label="Min thickness", unit="mm", zero="auto",
            doc="Thinnest member, mm (thicker members print better). 0 = automatic: 3 elements, and never less than "
                "that. It holds on the final shape, after Smoothing: members that come out thinner are thickened (the "
                "note says how much material that adds). Thin tips the design space's faces leave, thinner than this "
                "or Skin as set here, are cut off, unless a support or load stands on them."),
       Port("max_thickness", "number", 0.0, label="Max thickness", unit="mm", zero="no limit",
            doc="Thickest member, mm; 0 = no limit. Thick parts then split into more, thinner members: a branchier, "
                "more organic look (a little less stiff for its mass). Use a finer detail for thin values: it has to "
                "be at least ~5 elements."),
       Port("members", "text", "plates and struts", choices=["plates and struts", "struts only"], advanced=True,
            doc="With a Max thickness: plates and struts = wide thin plates are allowed (often the stiffest). "
                "struts only = no plates: everything becomes round-ish struts no thicker than Max thickness (and "
                "at least 0.65 of it), so a big flat flange turns into a web of branches. Plates just thin enough "
                "to pass can stay: Max section takes those out too."),
       Port("max_section", "number", 0.0, label="Max section", unit="mm²", zero="no limit",
            doc="Largest cross-section of a member, mm²; 0 = no limit. Unlike a thickness this also limits how wide a "
                "member is, and a plate is one very wide member, so plates have to break up into struts. A round strut "
                "of 50 mm² is 8 mm across; members come out at least about half that across, and where several meet "
                "they hold more. Smaller = more, thinner struts. Past the design space's largest section it has no "
                "effect (the note says so). Expect a real loss of stiffness, and no more than about "
                "15-20% of the design space fits (the note says when Keep % was not reached): set Keep % near that for "
                "clean struts, since a higher Keep % leaves material free to fill the space with extra ones. Use a finer "
                "detail for small values: a round member has to be at least ~5 elements across."),
       Port("skin", "number", 2.0, unit="mm", zero="none",
            doc="Material kept solid at the fixed and loaded faces, mm, so bolt holes and contact faces stay (0 = "
                "none; never thinner than a Min thickness you set). On the finished part it is exact: a boss round a "
                "hole is a true cylinder, this much thicker than the hole. With a bearing load the optimizer only "
                "counts on the side the pin presses on, but the part still gets the whole ring round the hole, so "
                "the eye is closed (a little more material than Keep %)."),
       Port("gap", "number", 0.0, label="Gap to faces", advanced=True, unit="mm", zero="none",
            doc="mm: keeps the shape this far inside the design space's faces, e.g. for clearance to what is around "
                "it. Around the faces of supports and loads (and Keep) material still reaches the faces. Note: a plate "
                "that lay flat against a face moves in by the gap but stays a flat plate (the outermost place is the "
                "stiffest for it); to break plates up use Max thickness with Members: struts only, or model a rounded "
                "design space. 0 = off. Keep % stays a share of the whole design space."),
       Port("smoothness", "number", 0.0, advanced=True, zero="off", slider=(0, 2),
            doc="While optimizing: weighs the part's surface area against its stiffness, so it makes fewer, rounder "
                "members and cleaner webs (it minimizes compliance x surface^Smoothness). 0 = off, 0.5 = a good start, "
                "1 = strong, at most 2. Key box: 0.5 = 9% less surface for 2% less stiffness, 1 = 14% for 7%; with "
                "Members: struts only it even came out stiffer. It works against a tight Max thickness (it wants "
                "fewer, thicker members): then less of Keep % fits. Too much on a thin part can break a member: "
                "compare the stiffness in the note with Smoothness 0. Smoothing (below) smooths the finished surface "
                "instead."),
       Port("symmetry", "text", "none", choices=list(T.SYMMETRY), advanced=True,
            doc="Mirror planes through the design space's middle: x = the two halves on either side of the "
                "plane square to X are mirror images (y, z alike; several = symmetric across each). Use it "
                "for a symmetric part whose supports and loads are symmetric too; the log says where the "
                "planes are."),
       Port("extrude", "text", "none", choices=list(T.EXTRUDE), label="Extrude along", advanced=True,
            doc="x, y or z = every member runs straight through the design space along that axis, with flat "
                "sides, as if cut from a plate: a web between flanges, a part to mill or laser-cut, the look of "
                "a truss. Min and Max thickness are then the members' width seen along the axis; Members and "
                "Max section have no effect. The design space may still taper: its own faces cut the web. "
                "none = members are free in all three directions (bone-like)."),
       Port("walls", "number", 0.0, label="Walls", unit="mm", zero="solid", slider=(0, 10), advanced=True,
            doc="mm: the part becomes hollow, with walls this thick along the design space's faces, and the "
                "optimizer cuts its openings in those walls. Every member is then a flat strip of a wall: flat "
                "outside and inside, a square or rectangular section, the look of a cast or machined frame "
                "(with Edges and Edge shape: chamfer, bevelled corners). Each wall gets its own pattern, "
                "so members lie at different depths. Use a Min thickness no larger than the walls. Keep and the "
                "skin round supports and loads stay solid; where the part is thinner than two walls nothing is "
                "hollow. Keep % is then a share of the walls. 0 = the whole design space may be filled. For a "
                "hollow of another shape connect your own body to Avoid instead."),
       Port("keep_body", "implicit", optional=True, label="Keep",
            doc="Optional: a body that is always solid (a boss, a mounting tab)."),
       Port("avoid_body", "implicit", optional=True, label="Avoid",
            doc="Optional: a body that must stay empty (room for a screw head, a cable, another part)."),
       Port("prefer_body", "implicit", optional=True, label="Prefer", advanced=True,
            doc="Optional: a guide body (e.g. a Ball, a Bar along the path a member should take, or a body from "
                "your STEP file). The optimizer prefers to put material there, softly: members grow to it and "
                "merge as it shapes them, so it stays organic, not a copy of the guide."),
       Port("prefer_strength", "number", 0.5, label="Prefer strength", advanced=True,
            doc="How strongly to follow Prefer: 0.5 = a nudge (wall bracket: a ball filled at no stiffness cost), "
                "1 = firm, 2 = very firm (it bulges round the guide, 7% less stiff). Negative = keep material "
                "away from it, softly (unlike Avoid, which keeps it exactly empty)."),
       Port("detail", "text", "draft", choices=list(DETAIL),
            doc="How fine the optimization grid is: draft ~40,000 elements (about half a minute on the GPU), "
                "medium ~250,000 (a few minutes), fine ~1 million. Start with draft. Finer = thinner members "
                "possible and a smoother surface."),
       Port("coarse_start", "text", "yes", choices=["yes", "no"], label="Start coarse", when={"detail": ["fine"]},
            advanced=True,
            doc="Fine detail: yes = the optimization starts on draft's and medium's bigger elements (a first shape "
                "in the 3D view within seconds) and moves on to finer ones as the shape settles. Usually as good "
                "and 25-40% faster (wall bracket: 66 s instead of 110). A level too coarse for the Max thickness "
                "is skipped. no = every iteration on fine elements (the shape can come out a little different: "
                "bolted bracket 5% stiffer). Draft and medium always run on their own elements (quick anyway)."),
       Port("smoothing", "number", T.SMOOTHING,
            doc="How much the optimized shape's surface is smoothed, in elements (the grid's cubes; their size "
                "is in the log). 1 = closest to the optimized shape, already without steps; 2 = rounder; 3 and "
                "up = rounder still. 0 = as optimized (waves at the elements' scale show). Members keep their "
                "thickness and the part its mass (members never get thinner than Min thickness or vanish). "
                "Changing it does not optimize again. The surface is a smooth one at any detail of the view, "
                "not a mesh of the elements; a result optimized before 2026-11-12 gets its truest surface when "
                "optimized again."),
       Port("even", "number", T.EVEN, label="Even out", unit="×", zero="off", slider=(0, 3),
            doc="Evens the optimized members out: bulges and necks along a member level out, the corners of "
                "openings and the joints between members get round, and loose ends pull back. Straight members "
                "and plates keep their thickness, and the part its mass. Small holes shrink too: at 1 those "
                "under 2 elements' radius close and larger ones hardly change; 2 and 3 work twice and three "
                "times as long (holes under 2.8 and 3.5 elements close, corners get rounder). 0 = the smoothed "
                "shape as it is. Changing it does not optimize again."),
       Port("fillet", "number", 0.0, label="Fillet", unit="mm", zero="sharp", slider=(0, 5),
            doc="Rounds the inside corners where the optimized members meet a Keep body (a web under a plate) "
                "and the bosses at supports and loads, mm: a smooth blend as deep as a round fillet of this "
                "radius. 0 = sharp corners. The design space's own faces stay as they are (model a fillet "
                "between two of its faces in CAD). Changing it does not optimize again; it adds a little "
                "material, and the note's safety factor and deflection are of the part with it. With Members "
                "as bars it is also the fillet where bars meet, and then the least one: a bar's fillets are "
                "0.4 x its width where that is more (the narrower of two bars decides), so wide bars flow "
                "into each other; 0 = no fillets at all."),
       Port("edge", "number", 0.0, label="Edges", unit="mm", zero="sharp", slider=(0, 3),
            doc="Breaks the edges where the optimized members meet the faces of the design space, by this "
                "many mm: the rims of the openings in a flat-sided web (Extrude along) or in Walls, and "
                "wherever a member is cut off by a face. Edge shape says how: round or chamfer. 0 = sharp. "
                "The design space's and a Keep body's own edges stay as they are (break those in CAD). More "
                "than half a member's thickness takes its whole side. It takes a little material off, and the "
                "note's safety factor and deflection are of the part with it. Changing it does not optimize "
                "again."),
       Port("edge_shape", "text", "round", choices=["round", "chamfer"], label="Edge shape",
            doc="How Edges breaks those edges: round = an arc of that radius; chamfer = a flat bevel that "
                "starts that far from the edge on both faces, the look of a machined or cast part. With Walls "
                "the members are flat strips, and a chamfer makes their sections squares with bevelled "
                "corners."),
       Port("members_as", "text", "optimized", choices=["optimized", "bars"], label="Members as",
            doc="optimized = the optimizer's own shape, smoothed. bars = the optimization only says where "
                "material goes: its members are traced along their centre lines and drawn again as straight "
                "bars with a square section and flat faces, joined by fillets (Fillet) and with their rims "
                "broken (Edges, Edge shape): the look of a designed frame. Short loose ends are left out. A bar's "
                "faces lie as the nearest face of the design space does. With Walls, every bar is a strip of a "
                "wall. The note's safety factor and deflection are of the bars. Changing it does not optimize "
                "again."),
       Port("bar_form", "text", "follow", choices=["follow", "straight"], label="Bar form",
            when={"members_as": ["bars"]},
            doc="follow = every bar keeps its member's form: it runs along the member's own centre line, "
                "curved where the member is, and (with Bar size auto) swells and tapers as the member does, "
                "wider towards a joint that carries more. straight = straight bars between the joints, each "
                "of one width: the plainest frame."),
       Port("bar_size", "number", 0.0, label="Bar size", unit="mm", zero="auto", slider=(0, 12),
            when={"members_as": ["bars"]},
            doc="How wide every bar is, mm. 0 = as wide as the optimized member: its own section, along the "
                "bar with Bar form follow, on average with straight (so thick members stay thick, and the "
                "bars weigh about what the optimized members do). One size for all is plainer, and lighter "
                "or heavier than the optimized shape: read the note."),
       Port("dead_ends", "text", "remove", choices=["remove", "keep"], label="Dead ends", advanced=True,
            doc="Branches and fins that end in nothing and carry next to no load (stress under a tenth of the "
                "typical), which the optimizer grows when material is cheap, e.g. when Max thickness leaves no "
                "room for Keep %. remove = while optimizing, every 5 iterations it analyses its design and keeps "
                "the dead ends empty, so the material goes to members that carry (bolted bracket: 23% stiffer); "
                "what the last iterations leave is cut off the finished part (the note says how many; supports, "
                "loads and their skin stay). keep = the optimizer's shape as it is. A result optimized with the "
                "other setting is not reused: press Optimize."),
       Port("max_iterations", "int", T.MAX_ITERATIONS, label="Max iterations", advanced=True,
            doc="The most iterations an optimization runs (in all, with Start coarse). It stops earlier when the "
                "design no longer changes; the note says when it stopped here instead, still changing. Raise it "
                "and press Optimize: the run goes on from where it stopped (no start over). Each fine iteration "
                "takes 1-3 s on the GPU. More is not always better: when Keep % does not fit (Max thickness), the "
                "design keeps shifting instead of settling (bolted bracket: no better at 200 than at 125, with "
                "Dead ends: remove worse)."),
       Port("show", "text", "shape", choices=["shape", *SHOW],
            doc="What colours the optimized part in the 3D view: shape = plain; stress (von Mises, MPa), safety "
                "factor or displacement (mm) of the part as shown, from the same Static Analysis as the safety "
                "factor in the note (at this node's detail). Dark blue stress = material that hardly works."),
       Port("compute", "int", 0, hidden=True), Port("stamp", "int", 0, hidden=True)],
      [Port("body", "implicit",
            doc="The optimized part, cropped to the design space (until you press Optimize: the design space). "
                "Smoothen it, texture it, mesh it."),
       Port("density", S, doc="0..1: how much material the optimizer put at each point (e.g. Remap -> a lattice's "
                              "thickness)."),
       Port("result", S, label="Shown result",
            doc="The field chosen by Show (nothing while Show is shape or the part is not optimized yet).")],
      paint="result")
def topology(space, material, conditions, keep, min_thickness, skin, keep_body, avoid_body, detail, compute=0,
             stamp=0, max_thickness=0.0, members="plates and struts", smoothing=T.SMOOTHING, gap=0.0,
             smoothness=0.0, symmetry="none", coarse_start="yes", prefer_body=None, prefer_strength=0.5,
             dead_ends="remove", max_iterations=T.MAX_ITERATIONS, show="shape", max_section=0.0, extrude="none",
             fillet=0.0, even=T.EVEN, edge=0.0, edge_shape="round", walls=0.0, members_as="optimized",
             bar_size=0.0, bar_form="follow"):
    """The stiffest shape for a given mass: keeps Keep % of the design space where the loads need it.
    Press Optimize on the node (it takes a while; the shape grows in the 3D view, Stop keeps the
    shape so far). Previews show the last result for these settings."""
    _validate(material, conditions)
    body, volume, _ = _part(space)
    if detail not in DETAIL:
        raise GraphError(f"detail must be one of: {', '.join(DETAIL)}")
    if not 0 < keep < 100:    # (also before Optimize: a preview took Keep 0 and 150 without a word)
        raise GraphError("Keep % must be between 0 and 100")
    if max_thickness < 0 or min_thickness < 0:
        raise GraphError("thicknesses must be 0 or more")
    if max_section < 0:
        raise GraphError("max section must be 0 or more")
    if not 0 <= smoothing <= 5:
        raise GraphError("smoothing must be between 0 and 5 elements")
    if not 0 <= even <= 5:
        raise GraphError("even out must be between 0 and 5")
    if not 0 <= fillet <= 20:
        raise GraphError("the fillet must be between 0 and 20 mm")
    if not 0 <= edge <= 20:
        raise GraphError("edges must be between 0 and 20 mm")
    if edge_shape not in ("round", "chamfer"):
        raise GraphError("edge shape must be round or chamfer")
    if gap < 0:
        raise GraphError("the gap must be 0 or more")
    if members not in T.MEMBERS:
        raise GraphError(f"members must be one of: {', '.join(T.MEMBERS)}")
    if not 0 <= smoothness <= 2:
        raise GraphError("smoothness must be between 0 and 2")
    if symmetry not in T.SYMMETRY:
        raise GraphError(f"symmetry must be one of: {', '.join(T.SYMMETRY)}")
    if extrude not in T.EXTRUDE:
        raise GraphError(f"extrude along must be one of: {', '.join(T.EXTRUDE)}")
    if coarse_start not in ("yes", "no"):
        raise GraphError("start coarse must be yes or no")
    coarse = coarse_start == "yes"
    if not -3 <= prefer_strength <= 3:
        raise GraphError("prefer strength must be between -3 and 3")
    if show != "shape" and show not in SHOW:
        raise GraphError(f"show must be one of: shape, {', '.join(SHOW)}")
    if dead_ends not in ("remove", "keep"):
        raise GraphError("dead ends must be remove or keep")
    if not 10 <= max_iterations <= 2000:
        raise GraphError("max iterations must be between 10 and 2000")
    if not 0 <= walls <= 1000:
        raise GraphError("walls must be 0 or more (mm)")
    if members_as not in ("optimized", "bars"):
        raise GraphError("members as must be optimized or bars")
    if not 0 <= bar_size <= 1000:
        raise GraphError("bar size must be 0 (automatic) or more (mm)")
    if bar_form not in ("follow", "straight"):
        raise GraphError("bar form must be follow or straight")
    if walls > 0:   # (an Avoid body like any other: in the topology key through its own)
        hollow = _hollow(space, float(walls), keep_body, conditions, float(skin), float(min_thickness))
        avoid_body = hollow if avoid_body is None else F.union(avoid_body, hollow)
    prune = dead_ends == "remove"
    setup = dict(max_thickness=max_thickness, members=members, gap=gap, smoothness=smoothness, symmetry=symmetry,
                 coarse=coarse, prefer_body=prefer_body, prefer_strength=prefer_strength, max_section=max_section,
                 extrude=extrude)
    args = (body, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, detail)
    key = T.topology_key(*args, **setup, prune=prune)
    state = load_topology(key)
    finished = None
    if state is not None and T.finished(state, max_iterations):
        # (said before the result is built: minutes on a fine grid, with Optimize doing nothing a designer can see)
        finished = _nothing_left(state)
        report_finished(finished)
        if compute:
            report_message(f"Topology Optimization: {finished}")
    elif compute:
        try:
            state = _optimize(key, body, volume, conditions, material, keep, min_thickness, skin, keep_body,
                              avoid_body, detail, state, max_thickness, members, gap, smoothness, symmetry, coarse,
                              prefer_body, prefer_strength, prune, int(max_iterations), max_section, extrude)
        except NotConverged as err:
            raise GraphError(f"{err}: is the part held well enough? A part that can turn about its supports has no "
                             f"answer, and a very slender one may need a finer Detail") from err
        finished = _nothing_left(state) if T.finished(state, max_iterations) else None
    unpruned = ""
    if state is None and prune:   # optimized before dead ends were pruned while optimizing: shown, trimmed
        state = load_topology(T.topology_key(*args, **setup))
        if state is not None:
            key = T.topology_key(*args, **setup)
            unpruned = " · optimized without pruning dead ends: press Optimize for a cleaner, stiffer result"
    if state is None:
        same = F.Field(lambda ctx: ctx.eval(body), F._digest("not-optimized", body.key), body.name, body.bounds,
                       children=(body,))
        same.note = "not optimized for these settings yet: press Optimize"
        return {"body": same, "density": F._lift(0.0), "result": None}
    def members_of(saved, key=key):
        # a result saved before the members' own field was: from its design variables, on the problem as it was
        # optimized (seconds; half a minute at fine), and saved with it
        level = str(saved["level"]) if "level" in saved else detail
        report_progress(0, 3, "reading a result saved by an earlier version")
        try:
            problem = T.prepare(body, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, level,
                                xp=lambda n: _array_module(n, TOPOLOGY_GPU_MIN, TOPOLOGY_GPU_BYTES_PER_ELEMENT),
                                cache=ARRAY_CACHE, volume=volume, max_thickness=max_thickness, members=members,
                                gap=gap, smoothness=smoothness, symmetry=symmetry, prefer_body=prefer_body,
                                prefer_strength=prefer_strength, prune=prune, max_section=max_section,
                                extrude=extrude)
            if tuple(problem.grid.shape) != tuple(saved["x"].shape):
                return None
            field = T.members_field(problem, saved["x"])
        except (ConditionError, GraphError):
            return None
        finally:
            backend.release_gpu_memory()
        save_topology(key, **{**saved, "members": field, "members_version": np.array(T.MEMBERS_FIELD)})
        return field
    out, rho, (checked, warnings), fields = _result(key, state, body, keep_body, float(smoothing), avoid_body,
                                                    min_thickness, check=(conditions, material, detail),
                                                    skin=float(skin), trim=dead_ends == "remove",
                                                    fillet=float(fillet), members=members_of,
                                                    even=float(even), axis=T.extrude_axis(extrude),
                                                    edge=float(edge), edge_shape=edge_shape,
                                                    members_as=members_as, bar_size=float(bar_size),
                                                    walls=float(walls), bar_form=bar_form,
                                                    mirror=[(a, 0.5 * float(body.bounds[0][a] + body.bounds[1][a]))
                                                            for a in T.mirror_axes(symmetry)])
    history = np.asarray(state["history"]).reshape(-1, 4)
    line = T.summary([tuple(r) for r in history], float(state["full_compliance"]), int(state["iteration"]),
                     float(state["compliance"]) if "compliance" in state else None)
    if len(history) and (max_thickness > 0 or max_section > 0) and history[-1][1] < keep / 100 - 0.01:
        limit = "max section" if max_section else "max thickness"
        line += f" (Keep {keep:g}%: the {limit}{' with Smoothness' if smoothness else ''} leaves room for no more)"
    if "dead_ends" in state and int(state["dead_ends"]):
        line += f" · {int(state['dead_ends'])} dead ends kept empty while optimizing"
    if checked:
        line += f" · {checked}"
    line += unpruned
    for text in warnings:  # (again on every run: the analysis is kept with the shape)
        report_message(text, "warning")
    if bool(state["done"]):
        if not T.settled(state):   # (the design was still changing)
            line += " · stopped at Max iterations, still changing" + (
                f": Optimize goes on to {int(max_iterations)}" if int(state["iteration"]) < max_iterations else "")
        out.note = line
    else:
        level = str(state["level"]) if "level" in state else detail
        on = "" if level == detail else f" on {level} elements"
        out.note = f"stopped at iteration {int(state['iteration'])}{on} (Optimize continues): {line}"
    out.finished = finished
    shown = fields[SHOW[show]] if fields is not None and show != "shape" else None
    return {"body": out, "density": rho, "result": shown}
