"""Simulation nodes: materials, supports and loads picked as CAD faces, static stress analysis.

Units: mm, N, MPa (N/mm^2), g/cm^3. See engine/fea and docs/TOPOLOGY_PLAN.md.
"""
import math
import re
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from .. import backend, facemask
from .. import fields as F
from ..cache import available_memory, load_topology, memory_limit, save_topology
from ..fea import analysis as FA
from ..fea import topology as T
from ..fea.materials import PRESETS, custom
from ..fea.model import (DETAIL, DIRECTIONS, LOAD_KINDS, OFFSET_KINDS, Condition, ConditionError, Conditions,
                         markers, parse_vector)
from ..fea.solver import SLAB_BYTES_GPU, NotConverged
from .core import (FREE_TIGHT, Cancelled, GraphError, Port, check_cancel, node, report_message, report_progress,
                   report_view)
from .nodes import ARRAY_CACHE, S

SIM = "Simulation"
SHOW = {"stress": "stress", "safety factor": "safety", "displacement": "displacement"}
GPU_MIN_ELEMENTS = 60_000   # smaller analyses are as fast on the CPU (and need no CUDA context)
# Memory of an analysis per element of the grid box (not just the part), with a margin. Measured on
# the L bracket (17% of its box filled): CPU 480-620 bytes, GPU 540 (float64 CG vectors over the
# whole box are most of it).
CPU_BYTES_PER_ELEMENT = 800
GPU_BYTES_PER_ELEMENT = 600


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
        if condition.kind == "load":  # for the big arrow: the faces' centre and the force (torque axis)
            arrow = condition.arrow()
            out.markers.update(load=condition.load, centre=np.round(condition.geometry[1], 4).tolist(),
                               arrow=None if arrow is None else np.round(arrow, 4).tolist())
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
                "bit of the faces, always square to them (Direction is not used). bearing = a pin, bolt or shaft "
                "in a hole pressing on it: only the side of the hole the force points to carries it, most at the "
                "middle of that side, none at the far side (as SolidWorks' bearing load). Needs a direction "
                "such as -Z. torque = twisting, N·m, about the Direction axis (right-hand rule: thumb along the "
                "axis, the fingers show the turn), as through a rigid connection: the further from the centre "
                "of the faces, the more each bit carries (a shaft's torque on its hole, bolts on a circle)."),
       Port("amount", "number", 100.0,
            doc="Force and bearing: N in total (100 N is about the weight of 10 kg). Pressure: MPa (0.1 MPa = "
                "1 bar). Torque: N·m (10 N on a 10 cm lever = 1 N·m). Negative turns it round."),
       Port("direction", "text", "into the face", choices=list(DIRECTIONS),
            doc="Force and bearing. into the face = pushing on it, square to it everywhere (on a curved face the "
                "pushes partly cancel). out of the face = pulling. +X ... -Z = along an axis (-Z: downwards; a "
                "bearing load -Z presses on the bottom of the hole). custom = along Vector. Torque: the axis it "
                "turns about (into the face on a flat face = turning like driving a screw into it). Easiest: "
                "select this node and drag the head of the yellow arrow in the 3D view (it snaps to the axes "
                "and 45°)."),
       Port("vector", "text", "0, 0, -1", doc="Custom direction only: x, y, z (any length), e.g. 1, 0, -1 for 45° "
                                              "down in the X-Z plane."),
       Port("offset", "text", "0, 0, 0",
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
        condition = Condition("load", _chosen_triangles(cad, faces), load=kind, amount=float(amount),
                              direction=direction, vector=vec, offset=shift,
                              label=f"{float(amount):g} {unit} {kind + ' ' if kind in ('bearing', 'torque') else ''}"
                                    f"on {_faces_label(faces) or 'no faces'}")
        condition.check()  # the direction (a bearing load: that something faces it), now
    except ConditionError as err:
        raise GraphError(str(err)) from err
    return _with(add_to, condition)


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


def _array_module(elements, gpu_min=GPU_MIN_ELEMENTS):
    """CuPy for big analyses when the GPU is on and has the memory, else numpy (if the PC has the
    memory: else an error rather than paging to a hard disk)."""
    if elements >= gpu_min and backend.gpu_enabled():
        free, _ = backend.gpu_memory()
        batches = 3 * SLAB_BYTES_GPU  # a product's temporaries
        if elements * GPU_BYTES_PER_ELEMENT + batches < free - backend.GPU_HEADROOM:
            return backend.cupy()
    need = elements * CPU_BYTES_PER_ELEMENT
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
    finally:
        backend.release_gpu_memory()  # CuPy keeps its pool otherwise (3.6 GB after a fine analysis)
    if count > 1:
        result.report["notes"].append(f"{count} bodies analysed as one piece, joined where they touch")
    for line in result.summary().splitlines():
        report_message(line)
    f = result.fields
    return {"result": f[SHOW[show]], "body": body, "stress": f["stress"], "safety": f["safety"],
            "displacement": f["displacement"]}


# --- topology optimization ---------------------------------------------------------------------
LIVE_EVERY = 3          # iterations between shapes shown in the 3D view while optimizing
# An optimization is ~70 solves: the GPU pays off sooner than for one analysis (3D cantilever,
# 47k elements: 0.3 s per iteration on the GPU, 1.0 s on the CPU)
TOPOLOGY_GPU_MIN = 10_000
CHECKPOINT_EVERY = 5    # iterations between saved states (Stop, a crash or a restart lose at most this)
_TOPO_RESULTS = OrderedDict()  # (key, iteration, done) -> (body, density): building the body takes a while


def _validate(material, conditions):
    if material is None:
        raise GraphError("connect a Material")
    if not isinstance(conditions, Conditions) or not len(conditions):
        raise GraphError("connect Fixed Faces and a Load (Supports & loads)")
    for cond in conditions:
        if not len(cond.triangles):
            raise GraphError(f"{'Fixed Faces' if cond.kind == 'fixed' else 'Load'}: no faces chosen yet (Pick)")


def _optimize(key, body, volume, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, detail,
              state, max_thickness=0.0, members="plates and struts", gap=0.0):
    """Run (or continue) the optimization, saving its state as it goes. Returns the final state."""
    try:
        problem = T.prepare(body, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, detail,
                            xp=lambda n: _array_module(n, TOPOLOGY_GPU_MIN), cache=ARRAY_CACHE, volume=volume,
                            max_thickness=max_thickness, members=members, gap=gap)
    except ConditionError as err:
        raise GraphError(str(err)) from err
    resume = state is not None and tuple(state["x"].shape) == tuple(problem.grid.shape)
    opt = T.Optimizer(problem, state if resume else None)
    device = "GPU" if problem.xp is not np else "CPU"
    free = int(problem.xp.sum(problem.free))
    report_message(f"Topology Optimization ({detail}, {problem.grid.describe()}, {free:,} design elements, "
                   f"{device}): " + (f"continuing at iteration {opt.iteration}" if resume else "starting"))
    for n in problem.notes:
        report_message(f"note: {n}")
    live = T.LiveSurface(backend.to_host(problem.phi), problem.grid.origin, problem.grid.h, problem.space)
    t0 = time.perf_counter()

    def on_iteration(o, compliance, kept, change, seconds):
        report_progress(o.iteration, T.MAX_ITERATIONS)  # also where Stop takes effect
        if o.iteration % LIVE_EVERY == 0:
            try:
                report_view(live(o.density()))
            except (ConditionError, ValueError):
                pass  # no surface yet (all densities on one side of 0.5)
        if o.iteration % CHECKPOINT_EVERY == 0:
            save_topology(key, **o.state(stopped=True))
        if o.iteration % 10 == 0:
            local = (f", fullest neighbourhood {100 * o.local_max:.0f}% (max {100 * T.LOCAL_SHARE:.0f}%)"
                     if o.local_max is not None else "")
            report_message(f"iteration {o.iteration}: {100 * kept:.1f}% kept, stiffness "
                           f"{100 * o.full_compliance / compliance:.0f}% of the full design space{local} "
                           f"({seconds:.1f} s per iteration)")

    try:
        opt.run(on_iteration, solver_callback=lambda it, r: check_cancel())
    except (Cancelled, GraphError):  # Stop, or the memory watchdog: keep what is done
        save_topology(key, **opt.state(stopped=True))
        report_message(f"Topology Optimization stopped at iteration {opt.iteration}: the shape so far is "
                       f"shown; Optimize continues from there")
        raise
    finally:
        backend.release_gpu_memory()
    final = opt.state()
    save_topology(key, **final)
    report_message(f"Topology Optimization done in {time.perf_counter() - t0:.0f} s: {opt.summary()}")
    if opt.history and opt.history[-1][1] < keep / 100 - 0.01:
        report_message(f"note: {100 * opt.history[-1][1]:.1f}% kept instead of {keep:g}%: no more material fits "
                       f"without a member thicker than Max thickness ({max_thickness:g} mm). A larger Max "
                       f"thickness or a lower Keep % avoids this.")
    return final


def _result(key, state, body, keep_body, smoothing=T.SMOOTHING, avoid_body=None):
    """(body, density field) of a saved state, kept for a few states."""
    memo_key = (key, int(state["iteration"]), bool(state["done"]), smoothing)
    if memo_key in _TOPO_RESULTS:
        _TOPO_RESULTS.move_to_end(memo_key)
        return _TOPO_RESULTS[memo_key]
    g = state["grid"]
    origin, h = g[:3], float(g[3])
    density = state["density"]
    try:
        mesh = T.density_mesh(density, state["phi"], origin, h, blur=smoothing,
                              space=state["space"] if "space" in state else None)  # (older states have none)
    except ConditionError as err:
        raise GraphError(str(err)) from err
    out = F.intersect(body, F.body_from_mesh(mesh, 2.0))
    if avoid_body is not None:  # exactly its faces (smoothing may round a little into it)
        out = F.subtract(out, avoid_body)
    if keep_body is not None:
        out = F.union(out, keep_body)
    centres = SimpleNamespace(origin=origin + h / 2, h=h)
    rho = FA.nodal_field(density, centres, f"{key}-{memo_key[1]}-rho", "density", "", (0.0, 1.0), body.bounds)
    _TOPO_RESULTS[memo_key] = (out, rho)
    while len(_TOPO_RESULTS) > 4:
        _TOPO_RESULTS.popitem(last=False)
    return out, rho


@node("sim.topology", "Topology Optimization", SIM,
      [Port("space", "part", many=True, label="Design space",
            doc="The room the part may fill: a CAD body (model it in SolidWorks as the largest envelope, with "
                "its bolt holes and mounting faces) or an implicit body."),
       Port("material", "material", doc="From Material or Custom Material."),
       Port("conditions", "conditions", label="Supports & loads",
            doc="From Fixed Faces and Load (chained with Add to), picked on the design space's faces."),
       Port("keep", "number", 30.0, label="Keep %",
            doc="How much of the design space's mass to keep, %. Less = lighter but bendier."),
       Port("min_thickness", "number", 0.0, label="Min thickness",
            doc="Thinnest member, mm (thicker members print better). 0 = automatic: 3 elements."),
       Port("max_thickness", "number", 0.0, label="Max thickness",
            doc="Thickest member, mm; 0 = no limit. Thick parts then split into more, thinner members: a "
                "branchier, more organic look (a little less stiff for its mass). Use a finer detail for "
                "thin values: it has to be at least ~5 elements."),
       Port("members", "text", "plates and struts", choices=["plates and struts", "struts only"],
            doc="With a Max thickness: plates and struts = wide thin plates are allowed (often the stiffest). "
                "struts only = no plates: everything becomes round-ish struts no thicker than Max thickness (and "
                "at least 0.65 of it), so a big flat flange turns into a web of branches."),
       Port("skin", "number", 2.0,
            doc="Material kept solid at the fixed and loaded faces, mm, so bolt holes and contact faces stay "
                "(0 = none). A bearing load only keeps the side the pin presses on; for a closed ring around "
                "the hole, add it as a Keep body."),
       Port("gap", "number", 0.0, label="Gap to faces",
            doc="mm: keeps the shape this far inside the design space's faces, e.g. for clearance to what is "
                "around it. Around the faces of supports and loads (and Keep) material still reaches the faces. "
                "Note: a plate that lay flat against a face moves in by the gap but stays a flat plate (the "
                "outermost place is the stiffest for it); to break plates up use Max thickness with Members: "
                "struts only, or model a rounded design space. 0 = off. Keep % stays a share of the whole "
                "design space."),
       Port("keep_body", "implicit", optional=True, label="Keep",
            doc="Optional: a body that is always solid (a boss, a mounting tab)."),
       Port("avoid_body", "implicit", optional=True, label="Avoid",
            doc="Optional: a body that must stay empty (room for a screw head, a cable, another part)."),
       Port("detail", "text", "draft", choices=list(DETAIL),
            doc="How fine the optimization grid is: draft ~40,000 elements (about half a minute on the GPU), "
                "medium ~250,000 (a few minutes), fine ~1 million. Start with draft. Finer = thinner members "
                "possible and a smoother surface."),
       Port("smoothing", "number", T.SMOOTHING,
            doc="How much the optimized shape's surface is smoothed, in elements (the grid's cubes; their size "
                "is in the log). 1 = true to the optimized shape (small steps can show on gently sloped faces). "
                "Above 1 it gets rounder, but thin members get thinner and can vanish, leaving stubs: a slender "
                "bracket came out 25% less stiff at 2. The mass stays the same. For a smoother surface that "
                "keeps its strength use a finer Detail. Changing it does not optimize again."),
       Port("compute", "int", 0, hidden=True), Port("stamp", "int", 0, hidden=True)],
      [Port("body", "implicit",
            doc="The optimized part, cropped to the design space (until you press Optimize: the design space). "
                "Smoothen it, texture it, mesh it."),
       Port("density", S, doc="0..1: how much material the optimizer put at each point (e.g. Remap -> a lattice's "
                              "thickness).")])
def topology(space, material, conditions, keep, min_thickness, skin, keep_body, avoid_body, detail, compute=0,
             stamp=0, max_thickness=0.0, members="plates and struts", smoothing=T.SMOOTHING, gap=0.0):
    """The stiffest shape for a given mass: keeps Keep % of the design space where the loads need it.
    Press Optimize on the node (it takes a while; the shape grows in the 3D view, Stop keeps the
    shape so far). Previews show the last result for these settings."""
    _validate(material, conditions)
    body, volume, _ = _part(space)
    if detail not in DETAIL:
        raise GraphError(f"detail must be one of: {', '.join(DETAIL)}")
    if max_thickness < 0 or min_thickness < 0:
        raise GraphError("thicknesses must be 0 or more")
    if not 0 <= smoothing <= 5:
        raise GraphError("smoothing must be between 0 and 5 elements")
    if gap < 0:
        raise GraphError("the gap must be 0 or more")
    if members not in T.MEMBERS:
        raise GraphError(f"members must be one of: {', '.join(T.MEMBERS)}")
    key = T.topology_key(body, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, detail,
                         max_thickness=max_thickness, members=members, gap=gap)
    state = load_topology(key)
    if compute and not (state is not None and bool(state["done"])):
        state = _optimize(key, body, volume, conditions, material, keep, min_thickness, skin, keep_body,
                          avoid_body, detail, state, max_thickness, members, gap)
    if state is None:
        same = F.Field(lambda ctx: ctx.eval(body), F._digest("not-optimized", body.key), body.name, body.bounds,
                       children=(body,))
        same.note = "not optimized for these settings yet: press Optimize"
        return {"body": same, "density": F._lift(0.0)}
    out, rho = _result(key, state, body, keep_body, float(smoothing), avoid_body)
    history = np.asarray(state["history"]).reshape(-1, 4)
    line = T.summary([tuple(r) for r in history], float(state["full_compliance"]), int(state["iteration"]))
    if len(history) and max_thickness > 0 and history[-1][1] < keep / 100 - 0.01:
        line += f" (Keep {keep:g}%: the max thickness leaves room for no more)"
    if bool(state["done"]):
        out.note = line
    else:
        out.note = f"stopped at iteration {int(state['iteration'])} (Optimize continues): {line}"
    return {"body": out, "density": rho}
