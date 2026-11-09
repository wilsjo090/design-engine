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
    for k, body in enumerate(list(cad) if isinstance(cad, list) else [cad]):
        ids = facemask.cad_faces(body)
        if ids is None:
            raise GraphError("this body has no CAD faces: connect a body from Import STEP")
        chosen = facemask.faces_for_body(faces, facemask.cad_body(body))
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
            volumes.append(float(abs(item.volume)) if item.is_watertight else None)
        else:
            raise GraphError("Part takes a CAD body or an implicit body")
    body = bodies[0]
    for other in bodies[1:]:
        body = F.union(body, other)
    if body.bounds is None:
        raise GraphError("this part has no size of its own (e.g. a lattice): fill a body with it first")
    volume = sum(volumes) if all(v is not None for v in volumes) else None
    return body, volume, len(items)


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
    if count > 1:
        result.report["notes"].append(f"{count} bodies analysed as one piece, joined where they touch")
    for line in result.summary().splitlines():
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
              max_section=0.0):
    """Run (or continue) the optimization, saving its state as it goes; with `coarse`, on the coarser
    details first (`T.levels`), each handing its design over to the next; with `prune`, dead ends kept
    empty as they show; at most `max_iterations` in all. Returns the final state."""
    levels = T.levels(body, detail, coarse, volume, min_thickness, max_thickness, max_section)
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
                                prune=prune, max_section=max_section)
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
            limit = (f"Max section ({problem.section_area:g} mm²)" if problem.section_sigma
                     else f"Max thickness ({problem.max_member:g} mm)")
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


def _result(key, state, body, keep_body, smoothing=T.SMOOTHING, avoid_body=None, min_thickness=0.0, check=None,
            skin=2.0, trim=True):
    """(body, density field, (note text, warnings) of its Static Analysis, that analysis' fields or
    None) of a saved state, kept for a few states. Members are at least the min thickness (automatic:
    MIN_MEMBER elements) after the smoothing, whatever it is. `check`: (conditions, material, detail)
    to analyse the body with. `trim`: dead ends that analysis finds (`T.dead_ends`) are emptied in the
    densities and the body made and analysed again."""
    # (the material's strength is not in the topology key, but the safety factor needs it)
    memo_key = (key, int(state["iteration"]), bool(state["done"]), str(state.get("level", "")), smoothing,
                check[1].key if check else None, bool(trim) and check is not None, float(skin))
    if memo_key in _TOPO_RESULTS:
        _TOPO_RESULTS.move_to_end(memo_key)
        return _TOPO_RESULTS[memo_key]
    g = state["grid"]
    origin, h = g[:3], float(g[3])
    density = state["density"]
    thinnest = max(float(min_thickness), T.MIN_MEMBER * h)
    # (the grid's own limit, 6 mm at draft, is no reason to cut a 2 mm skin: nothing asked for, nothing cut)
    asked = max(float(min_thickness), float(skin))
    crop = body if avoid_body is None else F.subtract(body, avoid_body)

    def shape(density):
        check_cancel()   # (the smoothing and its analysis take a while: Stop between them)
        _room_for_smoothing(np.shape(density), smoothing)
        try:
            mesh = T.density_mesh(density, state["phi"], origin, h, blur=smoothing,
                                  space=state["space"] if "space" in state else None,  # (older states have none)
                                  min_thickness=thinnest, crop=crop, cancel=check_cancel,
                                  conditions=check[0] if check is not None else (),
                                  tips_under=min(thinnest, asked))
        except ConditionError as err:
            raise GraphError(str(err)) from err
        out = F.intersect(body, F.body_from_mesh(mesh, 2.0))
        if avoid_body is not None:  # exactly its faces (smoothing may round a little into it)
            out = F.subtract(out, avoid_body)
        if keep_body is not None:
            out = F.union(out, keep_body)
        return mesh, out

    # dead ends found before are on the SSD (each round of finding them is a smoothing and an analysis)
    trim = trim and check is not None
    store = ARRAY_CACHE.disk if trim else None
    cut_key = ("topology-dead-ends", memo_key, T.DEAD_SHARE, T.DEAD_IDLE, T.DEAD_PASSES, "sticking out past min")
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
    mesh, out = shape(density)
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
            mesh, out = shape(density)
            analysis, text, warnings = analyse(out, f"{removing}: analysing the shape again")
            if analysis is None:
                break
        if analysis is not None and store is not None:
            store.save(cut_key, cut)
    trimmed, rho_key = "", f"{key}-{memo_key[1]}-rho"
    if cut is not None and cut.any():
        total, removed = int(cut.max()), float((cut > 0).sum()) / max(design, 1.0)
        trimmed = f"{total} dead end{'s' if total > 1 else ''} removed (-{100 * removed:.1f}% material)"
        rho_key += "-trimmed"
    added = float(mesh.metadata.get("thickened", 0.0))
    grown = f"members thickened to {thinnest:.2g} mm (+{100 * added:.0f}% material)" if added >= 0.005 else ""
    tips = int(mesh.metadata.get("tips", 0))
    tips = f"{tips} thin tip{'s' if tips > 1 else ''} cut off" if tips else ""
    centres = SimpleNamespace(origin=origin + h / 2, h=h)
    rho = FA.nodal_field(density, centres, rho_key, "density", "", (0.0, 1.0), body.bounds)
    missed = [f"{label}: the shape does not reach this load, so nothing carries it" if share > 0.99 else
              f"{label}: the shape does not reach {100 * share:.0f}% of this load, so nothing carries that part of it"
              for label, share in mesh.metadata.get("unreached", ())]
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
                "none; never thinner than a Min thickness you set). A bearing load only keeps the side the pin presses "
                "on; for a closed ring around the hole, add it as a Keep body."),
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
                "is in the log). 1 = closest to the optimized shape (small steps and waves at the elements' "
                "scale can show); 2 = smooth; 3 and up = smoother still. Members keep their thickness and the "
                "part its mass (members never get thinner than Min thickness or vanish). Changing it does not "
                "optimize again."),
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
             dead_ends="remove", max_iterations=T.MAX_ITERATIONS, show="shape", max_section=0.0):
    """The stiffest shape for a given mass: keeps Keep % of the design space where the loads need it.
    Press Optimize on the node (it takes a while; the shape grows in the 3D view, Stop keeps the
    shape so far). Previews show the last result for these settings."""
    _validate(material, conditions)
    body, volume, _ = _part(space)
    if detail not in DETAIL:
        raise GraphError(f"detail must be one of: {', '.join(DETAIL)}")
    if max_thickness < 0 or min_thickness < 0:
        raise GraphError("thicknesses must be 0 or more")
    if max_section < 0:
        raise GraphError("max section must be 0 or more")
    if not 0 <= smoothing <= 5:
        raise GraphError("smoothing must be between 0 and 5 elements")
    if gap < 0:
        raise GraphError("the gap must be 0 or more")
    if members not in T.MEMBERS:
        raise GraphError(f"members must be one of: {', '.join(T.MEMBERS)}")
    if not 0 <= smoothness <= 2:
        raise GraphError("smoothness must be between 0 and 2")
    if symmetry not in T.SYMMETRY:
        raise GraphError(f"symmetry must be one of: {', '.join(T.SYMMETRY)}")
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
    prune = dead_ends == "remove"
    setup = dict(max_thickness=max_thickness, members=members, gap=gap, smoothness=smoothness, symmetry=symmetry,
                 coarse=coarse, prefer_body=prefer_body, prefer_strength=prefer_strength, max_section=max_section)
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
                              prefer_body, prefer_strength, prune, int(max_iterations), max_section)
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
    out, rho, (checked, warnings), fields = _result(key, state, body, keep_body, float(smoothing), avoid_body,
                                                    min_thickness, check=(conditions, material, detail),
                                                    skin=float(skin), trim=dead_ends == "remove")
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
