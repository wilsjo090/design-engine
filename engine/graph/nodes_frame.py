"""Frame optimization nodes: parts built from bars (docs/FRAME_PLAN.md).

First editor version: round bars with plain joints. The layout and sizing run on Optimize (a
queued job like Topology Optimization's, ~10-20 s) and are kept on the SSD; previews read them.
"""
import json

import numpy as np

from .. import fields as F
from ..cache import load_frame, save_frame
from ..fea.model import ConditionError, Conditions
from ..fields import _digest
from ..frame import optimize as FO
from ..frame.geometry import JOINTS, frame_body, join_body, seamless
from ..frame.profiles import KINDS, Profile
from .core import GraphError, Port, node, report_message
from .nodes import ARRAY_CACHE
from .nodes_fea import SIM, _part, _validate

FRAME_VERSION = 2     # bump when the method changes: saved frames are then ignored
SUPPORTS = ("connect all", "only needed")
MID_BAR = ("on", "off")


def _sizes(text):
    """None for "any", else the listed sizes (mm)."""
    t = str(text or "").strip().lower()
    if t in ("", "any"):
        return None
    try:
        sizes = sorted({float(p) for p in t.replace(";", " ").replace(",", " ").split()})
    except ValueError as err:
        raise GraphError(f"sizes must be 'any' or numbers like 3, 4, 6, 8 (got '{text}')") from err
    if not sizes or sizes[0] <= 0:
        raise GraphError("sizes must be greater than 0")
    return sizes


@node("sim.frame", "Frame Optimization", SIM,
      [Port("space", "part", many=True, label="Design space",
            doc="The room the bars must stay inside: a CAD body (the largest envelope, with its bolt holes and "
                "mounting faces) or an implicit body. Several bodies: one frame over all of them (bars cross "
                "where they touch or overlap), optimized together."),
       Port("material", "material", doc="From Material or Custom Material (Nylon PA12 for SLS)."),
       Port("conditions", "conditions", label="Supports & loads",
            doc="From Fixed Faces and Load (chained with Add to), picked on the design space's faces. A bolt or "
                "pin hole becomes one node at its middle."),
       Port("profile", "text", "round", choices=list(KINDS),
            doc="The bars' cross-section, the same along each bar: round, round tube, square, square tube, "
                "rectangle, rect tube, I or angle. Its size is its width. Non-round bars are turned so their "
                "stronger direction points up (world Z), and keep that along curves (no twist). Hollow ones trap "
                "powder in SLS where their ends are closed (drain holes: not yet)."),
       Port("ratio", "number", 1.5, label="Height / width",
            doc="Rectangle, rect tube and I: the section's height over its width."),
       Port("wall", "number", 0.15,
            doc="Tubes and angle: wall thickness as a share of the width (at least 0.8 mm, the thinnest SLS wall)."),
       Port("min_size", "number", 2.0, label="Min size",
            doc="Thinnest bar, mm (its width; round bars: the diameter). Every bar costs at least this much, so "
                "the frame comes out with few, well used bars."),
       Port("sizes", "text", "any",
            doc="any = each bar its own size, in steps of Size step; or a list such as 3, 4, 6, 8 (mm)."),
       Port("size_step", "number", 0.5, label="Size step", doc="mm, for sizes 'any'."),
       Port("max_sizes", "int", 0, label="Max sizes",
            doc="0 = no limit; else the bars are grouped into at most this many sizes (a calmer look, a little "
                "heavier)."),
       Port("safety", "number", 2.0, label="Safety factor",
            doc="On the material's strength, for stress and for buckling of the bars in compression."),
       Port("joint_cost", "number", FO.JOINT_COST, label="Joint cost",
            doc="Each bar costs as if it were this share of the design space's diagonal longer: higher gives "
                "fewer, longer bars (more hand made), 0 the lightest frame."),
       Port("curving", "number", 0.5,
            doc="0 = straight bars only. Above 0, bars that go on nearly straight through a joint (bending by at "
                "most 70 degrees) become one smoothly curved bar along their whole length, still passing every "
                "joint; 1 = a full, even arc, smaller = flatter (closer to the straight bars, never a kink). The frame "
                "check sizes the curved bar as it is: a curve under load also bends, so more curving can cost mass "
                "(wall bracket: 3.9 g at 0.5, 4.8 g at 1)."),
       Port("joints", "text", JOINTS[0], choices=list(JOINTS),
            doc="How bars meet: fillet = they blend into each other and into the bolt bosses with a rounded fillet "
                "(no sharp inner corners, where stress gathers); plain = they just overlap; ball = a ball at every "
                "joint, filleted on (a space frame look). Changing it does not optimize again."),
       Port("fillet", "number", 0.0,
            doc="Fillet size at joints, mm; 0 = automatic (0.6 x the thinner bar). Changing it does not optimize "
                "again."),
       Port("spacing", "number", 0.0, label="Node spacing",
            doc="mm between the points bars may meet at; 0 = automatic (the design space's diagonal / 15). "
                "Smaller: more possible bars, slower."),
       Port("skin", "number", 2.0,
            doc="Material kept around the faces of supports and loads, mm (bolt bosses, a pin's eye); 0 = none."),
       Port("supports", "text", SUPPORTS[0], choices=list(SUPPORTS), label="Supports",
            doc="connect all = every support (each bolt hole, each mounting face) gets bars: one the layout does "
                "not need is joined to the frame by the shortest bars, so it shares the load and no boss is left "
                "loose. only needed = only the supports the layout uses (the others get no boss either)."),
       Port("mid_bar", "text", MID_BAR[0], choices=list(MID_BAR), label="Mid-bar joints",
            doc="on = a bar may land part way along another bar, which runs on through the joint as one bar "
                "(straight or curved); off = bars only meet where they end."),
       Port("connect_to", "frame", optional=True, many=True, label="Connect to",
            doc="Optional: another frame (its Frame output) or a CAD body this frame joins: its bars or surface "
                "near this design space hold this frame like a support; the log says what this frame puts on "
                "it. Join the two with Join Frames."),
       Port("compute", "int", 0, hidden=True), Port("stamp", "int", 0, hidden=True)],
      [Port("body", "implicit",
            doc="The frame as a part: its bars with their profile, the joints, the skin at supports and loads, "
                "inside the design space (until you press Optimize: the design space). Mesh it, export it, check "
                "it with Static Analysis."),
       Port("frame", "frame", doc="The bars themselves, for Connect to of another frame and for Join Frames.")])
def frame(space, material, conditions, min_size, sizes, size_step, max_sizes, safety, joint_cost, spacing, skin,
          compute=0, stamp=0, supports=None, profile="round", ratio=1.5, wall=0.15, curving=0.5, joints=JOINTS[0],
          fillet=0.0, mid_bar=MID_BAR[0], connect_to=None):
    """The lightest frame of bars that carries the loads to the supports: a truss chosen from
    thousands of possible bars, sized for stress and buckling. Press Optimize (seconds)."""
    _validate(material, conditions)
    body, volume, count = _part(space)
    listed = _sizes(sizes)
    if min_size <= 0 or size_step <= 0:
        raise GraphError("min size and size step must be greater than 0")
    if safety < 1:
        raise GraphError("the safety factor must be at least 1")
    if joint_cost < 0 or spacing < 0 or skin < 0 or max_sizes < 0:
        raise GraphError("joint cost, node spacing, skin and max sizes must be 0 or more")
    if not 0 <= curving <= 1:
        raise GraphError("curving must be between 0 and 1")
    supports = supports or SUPPORTS[0]
    if supports not in SUPPORTS:
        raise GraphError(f"supports must be one of: {', '.join(SUPPORTS)}")
    if joints not in JOINTS:
        raise GraphError(f"joints must be one of: {', '.join(JOINTS)}")
    if mid_bar not in MID_BAR:
        raise GraphError(f"mid-bar joints must be one of: {', '.join(MID_BAR)}")
    if fillet < 0:
        raise GraphError("the fillet must be 0 or more")
    try:
        prof = Profile(profile, ratio=float(ratio) if profile in ("rectangle", "rect tube", "I") else 1.0,
                       wall=float(wall) if profile in ("round tube", "square tube", "rect tube", "angle") else 0.15)
    except ValueError as err:
        raise GraphError(str(err)) from err
    attach = _items(connect_to, "Connect to")
    key = _digest("frame", FRAME_VERSION, body.key, conditions.key, material.key, float(min_size), listed,
                  float(size_step), int(max_sizes), float(safety), float(joint_cost), float(spacing), supports,
                  prof, float(curving), mid_bar, tuple(_item_key(a) for a in attach))
    state = load_frame(key)
    if compute and state is None:
        try:
            result = FO.optimize_frame(body, conditions, material, prof, listed, size_step, int(max_sizes),
                                       min_size, safety, spacing, joint_cost=joint_cost,
                                       connect_supports=supports == SUPPORTS[0], curving=float(curving),
                                       mid_bar=mid_bar == MID_BAR[0], attach=attach, cache=ARRAY_CACHE)
        except ConditionError as err:
            raise GraphError(str(err)) from err
        line = result.summary()
        for n in result.notes:
            report_message(f"note: {n}")
        r = result.report
        report_message(f"Frame Optimization done in {r['seconds']:.0f} s: {line}")
        state = {"nodes": result.nodes, "bars": result.bars, "chains": result.chains, "sizes": result.sizes,
                 "utilisation": result.utilisation, "axial": result.check.axial,
                 "summary": np.array(line), "report": np.array(json.dumps(r))}
        save_frame(key, **state)
    if state is None:
        same = F.Field(lambda ctx: ctx.eval(body), F._digest("not-optimized", body.key), body.name, body.bounds,
                       children=(body,))
        same.note = "not optimized for these settings yet: press Optimize"
        return {"body": same, "frame": NotOptimized()}
    chains, sizes_ = np.asarray(state["chains"]), np.asarray(state["sizes"])
    crop = seamless(body) if count > 1 else body    # several design spaces: their shared faces are inside
    out = frame_body(crop, state["nodes"], state["bars"], chains, sizes_, key, conditions, float(skin), prof, joints,
                     float(fillet))
    out.note = str(state["summary"])
    ref = FO.FrameRef(np.asarray(state["nodes"]), np.asarray(state["bars"]), chains, sizes_, prof, body,
                      tuple(conditions), float(skin), key, np.asarray(state["utilisation"]), np.asarray(state["axial"]))
    out.frame_view = frame_view(ref)
    return {"body": out, "frame": ref}


def frame_view(ref):
    """The bars for the editor's frame view (JSON): nodes, pieces, each piece's bar, size per bar,
    utilisation per bar, axial force per piece, length per bar."""
    nodes, bars, chains = np.asarray(ref.nodes), np.asarray(ref.bars), np.asarray(ref.chains)
    lengths = np.bincount(chains, weights=np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1),
                          minlength=len(ref.sizes))
    return {"profile": ref.profile.kind, "nodes": np.round(nodes, 4).tolist(), "bars": bars.tolist(),
            "chains": chains.tolist(), "sizes": np.asarray(ref.sizes, dtype=float).round(4).tolist(),
            "utilisation": np.asarray(ref.utilisation, dtype=float).round(4).tolist(),
            "axial": np.asarray(ref.axial, dtype=float).round(3).tolist(), "lengths": lengths.round(2).tolist()}


def _items(value, what):
    """Frames and CAD bodies from an input (one, or a list); an unoptimized frame is an error."""
    if value is None:
        return []
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    out = []
    for v in items:
        if v is None:
            continue
        if isinstance(v, NotOptimized):
            raise GraphError(f"{what}: that frame is not optimized yet: press Optimize on it first")
        if not (isinstance(v, FO.FrameRef) or hasattr(v, "triangles")):
            raise GraphError(f"{what} takes frames (a Frame Optimization's Frame output) or CAD bodies")
        out.append(v)
    return out


class NotOptimized:
    """A Frame Optimization's Frame output before it is optimized (None would read as "not
    connected": Connect to would quietly optimize without the other frame)."""


def _item_key(v):
    return v.key if isinstance(v, FO.FrameRef) else F.mesh_fingerprint(v)


@node("sim.join_frames", "Join Frames", SIM,
      [Port("a", "frame", label="Frame A", doc="A frame (a Frame Optimization's Frame output) or a CAD body."),
       Port("b", "frame", optional=True, label="Frame B", doc="Another frame or CAD body to join."),
       Port("c", "frame", optional=True, label="Frame C", doc="Optional: one more."),
       Port("d", "frame", optional=True, label="Frame D", doc="Optional: one more."),
       Port("joints", "text", JOINTS[0], choices=list(JOINTS),
            doc="How bars meet, also bars of different frames: fillet, plain or ball (as in Frame Optimization)."),
       Port("fillet", "number", 0.0, doc="Fillet size, mm; 0 = automatic (0.6 x the thinner bar).")],
      [Port("body", "implicit", doc="The frames and bodies as one part, filleted where they meet.")])
def join_frames(a, b=None, c=None, d=None, joints=JOINTS[0], fillet=0.0):
    """Frames made separately (one connected to another with Connect to) and CAD bodies as one part:
    where a bar of one meets a bar of another, the joint is filleted as inside one frame."""
    items = []
    for name, v in (("A", a), ("B", b), ("C", c), ("D", d)):
        items += _items(v, f"Frame {name}")
    frames = [v for v in items if isinstance(v, FO.FrameRef)]
    bodies = [v for v in items if not isinstance(v, FO.FrameRef)]
    if not frames:
        raise GraphError("connect at least one frame (a Frame Optimization's Frame output)")
    if joints not in JOINTS:
        raise GraphError(f"joints must be one of: {', '.join(JOINTS)}")
    if fillet < 0:
        raise GraphError("the fillet must be 0 or more")
    return join_body(frames, bodies, joints, float(fillet))
