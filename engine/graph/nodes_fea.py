"""Simulation nodes: materials, supports and loads picked as CAD faces, static stress analysis.

Units: mm, N, MPa (N/mm^2), g/cm^3. See engine/fea and docs/TOPOLOGY_PLAN.md.
"""
import math
import re

import numpy as np

from .. import backend, facemask
from .. import fields as F
from ..fea import analysis as FA
from ..fea.materials import PRESETS, custom
from ..fea.model import (DETAIL, DIRECTIONS, LOAD_KINDS, Condition, ConditionError, Conditions, markers,
                         parse_vector)
from ..cache import available_memory, memory_limit
from ..fea.solver import NotConverged
from .core import FREE_TIGHT, GraphError, Port, node, report_message, report_progress
from .nodes import ARRAY_CACHE, S

SIM = "Simulation"
SHOW = {"stress": "stress", "safety factor": "safety", "displacement": "displacement"}
GPU_MIN_ELEMENTS = 60_000   # smaller analyses are as fast on the CPU (and need no CUDA context)
# Memory of an analysis per element (grid box, not just the part), in units of the float size
# (4 bytes on the GPU, 8 on the CPU): ~16 nodal vectors of 3 values, the multigrid levels, the
# setup arrays. An estimate until measured (FINDINGS.md).
BYTES_PER_ELEMENT = 75


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
      [Port("stiffness", "number", 3500.0, label="Stiffness E",
            doc="Young's modulus, MPa (N/mm²): how hard it is to stretch. PLA ~3500, PETG ~2000, aluminium 69000, "
                "steel 210000. The datasheet's tensile modulus."),
       Port("poisson", "number", 0.35, label="Poisson's ratio",
            doc="How much it gets thinner when stretched: plastics 0.35-0.4, metals ~0.3."),
       Port("density", "number", 1.24, doc="g/cm³, for the mass in the report."),
       Port("strength", "number", 50.0,
            doc="MPa: the stress at which it yields (metals) or breaks (plastics, tensile strength). The safety "
                "factor is strength / stress.")],
      [Port("material", "material", doc="For Static Analysis.")])
def custom_material(stiffness, poisson, density, strength):
    """A material with your own values (from a datasheet)."""
    try:
        return custom(stiffness, poisson, density, strength)
    except ValueError as err:
        raise GraphError(str(err)) from err


# --- supports and loads ------------------------------------------------------------------------
def _chosen_triangles(cad, faces):
    """Corner positions (k, 3, 3) of the triangles of the chosen CAD faces on one body or a list."""
    unknown = re.findall(r"[A-Za-z_][\w-]*", str(faces or ""))
    if unknown:
        raise GraphError(f"no face selection named '{unknown[0]}' in this recipe")
    out = []
    for body in (list(cad) if isinstance(cad, list) else [cad]):
        ids = facemask.cad_faces(body)
        if ids is None:
            raise GraphError("this body has no CAD faces: connect a body from Import STEP")
        chosen = facemask.faces_for_body(faces, facemask.cad_body(body))
        wrong = [f for f in chosen if f > ids.max()]
        if wrong:
            raise GraphError(f"face {wrong[0]} does not exist (the body has faces 0-{ids.max()})")
        out.append(np.asarray(body.triangles)[np.isin(ids, chosen)])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def _with(add_to, condition):
    """`add_to` plus the new condition, carrying the new one's markers for the 3D view."""
    if add_to is not None and not isinstance(add_to, Conditions):
        raise GraphError("Add to takes supports and loads (from Fixed Faces or Load)")
    out = (add_to or Conditions()).add(condition)
    if len(condition.triangles):
        pts, dirs = markers(condition)
        push = (np.einsum("ij,ij->i", dirs, _normals_at(condition, pts)) < 0).tolist()
        out.markers = {"kind": condition.kind, "label": condition.label,
                       "points": np.round(pts, 4).tolist(), "dirs": np.round(dirs, 4).tolist(), "push": push}
    else:
        out.markers = {"kind": condition.kind, "label": condition.label, "points": [], "dirs": [], "push": []}
    return out


def _normals_at(condition, points):
    """Outward normal of the chosen triangle nearest to each point."""
    t = condition.triangles
    centres = t.mean(axis=1)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    from scipy.spatial import cKDTree
    return n[cKDTree(centres).query(points)[1]]


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
    return _with(add_to, Condition("fixed", _chosen_triangles(cad, faces), label=f"Fixed faces {_faces_label(faces)}"))


@node("sim.load", "Load", SIM,
      [Port("cad", "cad", many=True, label="Body",
            doc="The CAD body the faces are picked on: the part you analyse."),
       Port("faces", "faces", "", doc="The faces the load acts on. Click Pick, then click them in the 3D view."),
       Port("kind", "text", "force", choices=list(LOAD_KINDS),
            doc="force = a total force in N, spread evenly over the faces. pressure = MPa (N/mm²) pushing on every "
                "bit of the faces, always square to them (Direction is not used)."),
       Port("amount", "number", 100.0,
            doc="Force: N in total (100 N is about the weight of 10 kg). Pressure: MPa (0.1 MPa = 1 bar). "
                "Negative turns it round."),
       Port("direction", "text", "into the face", choices=list(DIRECTIONS),
            doc="Force only. into the face = pushing on it, square to it everywhere (on a curved face the pushes "
                "partly cancel). out of the face = pulling. +X ... -Z = along an axis (-Z: downwards). custom = "
                "along Vector."),
       Port("vector", "text", "0, 0, -1", doc="Custom direction only: x, y, z (any length), e.g. 1, 0, -1 for 45° "
                                              "down in the X-Z plane."),
       Port("add_to", "conditions", optional=True, label="Add to",
            doc="Optional: supports and loads from a Fixed Faces or another Load, so one link carries them all.")],
      [Port("conditions", "conditions", label="Supports & loads",
            doc="This load (plus what came in on Add to). Connect to Static Analysis, or to the next node's Add to. "
                "Shown as red arrows in the 3D view.")],
      preview=True)
def load(cad, faces, kind, amount, direction, vector, add_to):
    """A force or a pressure on faces of the part."""
    if kind not in LOAD_KINDS:
        raise GraphError(f"kind must be one of: {', '.join(LOAD_KINDS)}")
    if direction not in DIRECTIONS:
        raise GraphError(f"direction must be one of: {', '.join(DIRECTIONS)}")
    try:
        vec = parse_vector(vector) if (kind == "force" and direction == "custom") else (0.0, 0.0, -1.0)
        if kind == "force" and direction == "custom" and np.linalg.norm(vec) == 0:
            raise ConditionError("the vector must not be 0, 0, 0")
    except ConditionError as err:
        raise GraphError(str(err)) from err
    unit = "N" if kind == "force" else "MPa"
    label = f"{float(amount):g} {unit} on {_faces_label(faces) or 'no faces'}"
    return _with(add_to, Condition("load", _chosen_triangles(cad, faces), label=label, load=kind,
                                   amount=float(amount), direction=direction, vector=vec))


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


def _array_module(elements):
    """CuPy for big analyses when the GPU is on and has the memory, else numpy (if the PC has the
    memory: else an error rather than paging to a hard disk)."""
    if elements >= GPU_MIN_ELEMENTS and backend.gpu_enabled():
        free, _ = backend.gpu_memory()
        if elements * BYTES_PER_ELEMENT * 4 < free - backend.GPU_HEADROOM:
            return backend.cupy()
    need = elements * BYTES_PER_ELEMENT * 8
    room = min(memory_limit(), available_memory() - FREE_TIGHT)
    if need > room:
        raise GraphError(f"this analysis needs about {need / 2**30:.1f} GB of memory ({elements:,} elements) but "
                         f"only {max(room, 0) / 2**30:.1f} GB can be used: choose a lower detail")
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
        done = math.log(max(residual, 1e-30)) / math.log(1e-6)
        report_progress(min(99, max(1, int(100 * done))), 100)

    try:
        result = FA.static_analysis(body, conditions, material, detail, xp=_array_module, cache=ARRAY_CACHE,
                                    store=ARRAY_CACHE.disk, callback=progress, volume=volume)
    except ConditionError as err:
        raise GraphError(str(err)) from err
    except NotConverged as err:
        raise GraphError(f"{err}: is the part held well enough? A part that can turn about its supports has no "
                         f"answer") from err
    if count > 1:
        result.report["notes"].append(f"{count} bodies analysed as one piece, joined where they touch")
    for line in result.summary().splitlines():
        report_message(line)
    f = result.fields
    return {"result": f[SHOW[show]], "body": body, "stress": f["stress"], "safety": f["safety"],
            "displacement": f["displacement"]}
