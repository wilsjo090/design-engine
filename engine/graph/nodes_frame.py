"""Frame optimization nodes: parts built from bars (docs/FRAME_PLAN.md).

First editor version: round bars with plain joints. The layout and sizing run on Optimize (a
queued job like Topology Optimization's, ~10-20 s) and are kept on the SSD; previews read them.
"""
import json
import time

import numpy as np

from .. import fields as F
from ..cache import load_frame, save_frame
from ..fea.model import ConditionError, Conditions, sample_triangles
from ..fields import _digest
from ..frame import ground as G
from ..frame import optimize as FO
from ..frame import sizing as SZ
from ..frame.geometry import CONTINUITY, JOINTS, _cylinder, frame_body, join_body, seamless
from ..frame.profiles import KINDS, Profile
from .core import (Cancelled, GraphError, Port, check_cancel, node, report_finished, report_frame, report_message,
                   report_progress)
from .nodes import ARRAY_CACHE, S
from .nodes_fea import SHOW, SIM, _check, _part, _validate, check_result, safety_text

FRAME_VERSION = 5     # bump when the method changes: saved frames are then ignored
# The beam model sees neither the joints' fillets nor the bosses at holes, where the body is most
# stressed: the bracket's body reached safety factor 1.42-1.57 of 2. So the body is analysed (at
# CHECK_DETAIL) and the bars at its most stressed spot made thicker, for up to CHECK_ROUNDS rounds.
# Medium, as Static Analysis by default (3-4 s a round on the GPU). The peak at a fillet still grows
# with the detail: sized at medium to 2.38, the bracket's body had 1.90 at fine, where its sparse grid
# box no longer fitted the GPU and one analysis took 87 s on the CPU.
CHECK_DETAIL = "medium"
CHECK_ROUNDS = 6
# Still under the safety factor within SPREAD_NEAR mm (plus the skin) of a support's or load's faces: those
# faces get several nodes and the frame is laid out again (`_spread_where_weak`), up to SPREAD_ROUNDS times
SPREAD_NEAR = 3.0
SPREAD_ROUNDS = 2
LIVE_EVERY = 0.02     # s: the live view while optimizing, at most this often (but every new stage); the
                      # editor keeps the steps for Replay (joints move ~25 ms a step, 0.7 s in all on the wall bracket)
TICK_EVERY = 0.5      # s: an LP's iterations on the card, at most this often
SUPPORTS = ("connect all", "only needed")
MID_BAR = ("on", "off")
LIGHT_BARS = ("keep", "remove")
MOVE_JOINTS = ("on", "off")
NODES_SHOW = "candidate nodes"   # Show: the ground structure's nodes (`_ground_view`), before Optimize too


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
       Port("ratio", "number", 1.5, label="Height / width", when={"profile": ["rectangle", "rect tube", "I"]},
            unit="×", doc="Rectangle, rect tube and I: the section's height over its width."),
       Port("wall", "number", 0.15, when={"profile": ["round tube", "square tube", "rect tube", "angle"]},
            unit="× width",
            doc="Tubes and angle: wall thickness as a share of the width (at least 0.8 mm, the thinnest SLS wall)."),
       Port("min_size", "number", 2.0, label="Min size", unit="mm",
            doc="Thinnest bar, mm (its width; round bars: the diameter). Every bar costs at least this much, so the "
                "frame comes out with few, well used bars."),
       Port("max_size", "number", 0.0, label="Max size", unit="mm", zero="no limit",
            doc="Thickest bar, mm (its width; round bars: the diameter); 0 = no limit. Where one bar would need "
                "more, the load is shared by more bars: a branchier frame, a little heavier. If even that cannot "
                "carry the loads (a load on one hole, a long bar in compression), the node says so: a larger Max "
                "size or a lower Safety factor."),
       Port("sizes", "text", "any",
            doc="any = each bar its own size, in steps of Size step; or a list such as 3, 4, 6, 8 (mm)."),
       Port("size_step", "number", 0.5, label="Size step", when={"sizes": ["any", ""]}, advanced=True,
            unit="mm", doc="mm, for sizes 'any'."),
       Port("max_sizes", "int", 0, label="Max sizes", advanced=True, zero="no limit",
            doc="0 = no limit; else the bars are grouped into at most this many sizes (a calmer look, a little "
                "heavier)."),
       Port("safety", "number", 2.0, label="Safety factor",
            doc="On the material's strength, for stress and for buckling of the bars in compression. The "
                "body itself (joints, fillets, bosses) is then checked with Static Analysis at medium detail "
                "and the bars at its most stressed spot made thicker until it reaches this too."),
       Port("joint_cost", "number", FO.JOINT_COST, label="Joint cost", advanced=True,
            doc="Each bar costs as if it were this share of the design space's diagonal longer: higher gives "
                "fewer, longer bars (more hand made), 0 the lightest frame."),
       Port("curving", "number", 0.5, zero="straight", slider=(0, 1),
            doc="0 = straight bars only. Above 0, bars that go on nearly straight through a joint (bending by at most "
                "70 degrees) become one smoothly curved bar along their whole length, still passing every joint; 1 = a "
                "full, even arc, smaller = flatter (closer to the straight bars, never a kink). The frame check sizes "
                "the curved bar as it is: a curve under load also bends, so more curving can cost mass (wall bracket: "
                "3.9 g at 0.5, 4.8 g at 1)."),
       Port("joints", "text", JOINTS[0], choices=list(JOINTS),
            doc="How bars meet: fillet = they blend into each other and into the bolt bosses with a rounded fillet "
                "(no sharp inner corners, where stress gathers); plain = they just overlap; ball = a ball at every "
                "joint, filleted on (a space frame look). Changing it does not optimize again."),
       Port("fillet", "number", 0.0, when={"joints": ["fillet", "ball"]}, unit="mm", zero="auto",
            doc="Fillet size at joints, mm; 0 = automatic (0.6 x the thinner bar). Changing it does not optimize "
                "again."),
       Port("continuity", "text", "G2", choices=list(CONTINUITY), label="Fillet continuity",
            when={"joints": ["fillet", "ball"]}, advanced=True,
            doc='How smoothly fillets run into the bars and bosses: G1 = tangent (the curvature jumps where a fillet '
                'meets a bar: a faint ring in reflections), G2 = curvature continuous, G3 = one step smoother still. '
                'G2 and G3 fillets reach farther along the bars, with about the same material at the joint as G1 and a '
                'gentler middle. Changing it does not optimize again.'),
       Port("spacing", "number", 0.0, label="Node spacing", advanced=True, unit="mm", zero="auto",
            doc="mm between the points bars may meet at; 0 = automatic (the design space's diagonal / 15, finer when "
                "that leaves a support or load without bars, as in thin walls; the log says so). Smaller: more "
                "possible bars, slower."),
       Port("skin", "number", 2.0, unit="mm", zero="none",
            doc="Material kept around the faces of supports and loads, mm (bolt bosses, a pin's eye); 0 = none."),
       Port("supports", "text", SUPPORTS[0], choices=list(SUPPORTS), label="Supports", advanced=True,
            doc="connect all = every support (each bolt hole, each mounting face) gets bars: one the layout does "
                "not need is joined to the frame by the shortest bars, so it shares the load and no boss is left "
                "loose. only needed = only the supports the layout uses (the others get no boss either)."),
       Port("mid_bar", "text", MID_BAR[0], choices=list(MID_BAR), label="Mid-bar joints", advanced=True,
            doc="on = a bar may land part way along another bar, which runs on through the joint as one bar "
                "(straight or curved); off = bars only meet where they end."),
       Port("move_joints", "text", MOVE_JOINTS[0], choices=list(MOVE_JOINTS), label="Move joints", advanced=True,
            doc="on = after the layout, the joints that hold no support or load move off the grid of node points "
                "to where the frame is lightest: bars run straighter, bars left carrying nothing go and joints "
                "that meet become one; every bar stays inside the design space. off = joints stay on the grid."),
       Port("light_bars", "text", LIGHT_BARS[0], choices=list(LIGHT_BARS), label="Light bars", advanced=True,
            doc="remove = after sizing, bars that use under 35% of their strength (often bracing at Min size) are "
                "taken out one at a time, each only if the frame still passes its check without it and weighs "
                "less; every support and load keeps a bar. keep = the frame as laid out."),
       Port("connect_to", "frame", optional=True, many=True, label="Connect to", advanced=True,
            doc="Optional: another frame (its Frame output) or a CAD body this frame joins: its bars or surface "
                "near this design space hold this frame like a support; the log says what this frame puts on "
                "it. Join the two with Join Frames."),
       Port("show", "text", "shape", choices=["shape", *SHOW, NODES_SHOW],
            doc="What colours the optimized frame in the 3D view: shape = plain; stress (von Mises, MPa), safety "
                "factor or displacement (mm) of the frame as shown, from the same Static Analysis (medium) as the "
                "body check in the note. Dark blue stress = a bar or joint that hardly works. candidate nodes = the "
                "points bars may meet at, also before Optimize: grey inside the design space (a wall without dots "
                "is thinner than the Node spacing), green at supports, red at loads, a large red ball where no bar "
                "can reach; with the section on, the layer of dots at the cut."),
       Port("compute", "int", 0, hidden=True), Port("stamp", "int", 0, hidden=True)],
      [Port("body", "implicit",
            doc="The frame as a part: its bars with their profile, the joints, the skin at supports and loads, "
                "inside the design space (until you press Optimize: the design space). Mesh it, export it, check "
                "it with Static Analysis."),
       Port("frame", "frame", doc="The bars themselves, for Connect to of another frame and for Join Frames."),
       Port("result", S, label="Shown result",
            doc="The field chosen by Show (nothing while Show is shape or the frame is not optimized yet).")],
      paint="result")
def frame(space, material, conditions, min_size, sizes, size_step, max_sizes, safety, joint_cost, spacing, skin,
          compute=0, stamp=0, supports=None, profile="round", ratio=1.5, wall=0.15, curving=0.5, joints=JOINTS[0],
          fillet=0.0, mid_bar=MID_BAR[0], connect_to=None, continuity="G2", show="shape", light_bars=LIGHT_BARS[0],
          move_joints=MOVE_JOINTS[0], max_size=0.0):
    """The lightest frame of bars that carries the loads to the supports: a truss chosen from
    thousands of possible bars, sized for stress and buckling. Press Optimize (seconds)."""
    _validate(material, conditions)
    if len(conditions.cases()) > 1:
        raise GraphError("Frame Optimization takes one load case (all its loads act together): connect the supports "
                         "and loads without Load Case nodes, or one case's chain")
    body, volume, count = _part(space)
    listed = _sizes(sizes)
    if min_size <= 0 or size_step <= 0:
        raise GraphError("min size and size step must be greater than 0")
    if max_size < 0 or (max_size and max_size < min_size):
        raise GraphError(f"Max size must be 0 (no limit) or at least Min size ({min_size:g} mm)")
    if max_size and listed:   # (a list's sizes above it are left out)
        listed = [s for s in listed if s <= max_size * (1 + 1e-9)]
        if not listed:
            raise GraphError(f"no size in the list is at most Max size ({max_size:g} mm)")
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
    if continuity not in CONTINUITY:
        raise GraphError(f"fillet continuity must be one of: {', '.join(CONTINUITY)}")
    if move_joints not in MOVE_JOINTS:
        raise GraphError(f"move joints must be one of: {', '.join(MOVE_JOINTS)}")
    if light_bars not in LIGHT_BARS:
        raise GraphError(f"light bars must be one of: {', '.join(LIGHT_BARS)}")
    if show not in ("shape", NODES_SHOW) and show not in SHOW:
        raise GraphError(f"show must be one of: shape, {', '.join(SHOW)}, {NODES_SHOW}")
    try:
        prof = Profile(profile, ratio=float(ratio) if profile in ("rectangle", "rect tube", "I") else 1.0,
                       wall=float(wall) if profile in ("round tube", "square tube", "rect tube", "angle") else 0.15)
    except ValueError as err:
        raise GraphError(str(err)) from err
    attach = _items(connect_to, "Connect to")
    key = _digest("frame", FRAME_VERSION, body.key, conditions.key, material.key, float(min_size), listed,
                  float(size_step), int(max_sizes), float(safety), float(joint_cost), float(spacing), supports,
                  prof, float(curving), mid_bar, tuple(_item_key(a) for a in attach),
                  *(("light bars removed", FO.LIGHT_SHARE) if light_bars == "remove" else ()),
                  *(("joints moved",) if move_joints == "on" else ()),
                  *(("max size", float(max_size), FO.MAX_SIZE_SHARES) if max_size else ()))
    state = load_frame(key)
    crop = seamless(body) if count > 1 else body    # several design spaces: their shared faces are inside
    shape = {"skin": float(skin), "joints": joints, "fillet": float(fillet)}   # the body's own settings
    if continuity != "G1":   # (states sized before it was a setting were G1)
        shape["continuity"] = continuity
    live = _Live(prof, material.strength / safety, min_size, material.density)

    def lay_out(spread=()):
        try:
            result = FO.optimize_frame(body, conditions, material, prof, listed, size_step, int(max_sizes),
                                       min_size, safety, spacing, joint_cost=joint_cost,
                                       connect_supports=supports == SUPPORTS[0], curving=float(curving),
                                       mid_bar=mid_bar == MID_BAR[0], attach=attach, cache=ARRAY_CACHE, spread=spread,
                                       cancel=check_cancel, light_bars=light_bars == "remove",
                                       move_joints=move_joints == "on", live=live, max_size=float(max_size))
        except G.Unreached as err:
            balls = " (a large red ball: what no bar reaches)" if len(err.lonely) else ""
            raise GraphError(f"{err} · Show: {NODES_SHOW} draws the points{balls}") from err
        except ConditionError as err:
            raise GraphError(str(err)) from err
        for n in result.notes:
            report_message(f"note: {n}")
        r = result.report
        report_message(f"Frame Optimization done in {r['seconds']:.0f} s: {result.summary()}")
        return {"nodes": result.nodes, "bars": result.bars, "chains": result.chains, "sizes": result.sizes,
                "beam_sizes": result.sizes, "utilisation": result.utilisation, "axial": result.check.axial,
                "report": np.array(json.dumps(r))}

    def sized(state):
        state = dict(state)
        # (never thicker than fits the design space: without a skin the body check made 25.5 mm bars in a 20 mm block)
        rep = json.loads(str(state["report"]))
        room = float(rep.get("room") or 0.0)
        largest = min(float(max_size), room) if max_size and room else float(max_size or room)
        state.update(_size_for_body(crop, state, conditions, material, prof, shape, key, safety, min_size, listed,
                                    size_step, int(max_sizes), live, largest=largest,
                                    room=bool(room) and largest == room and not (max_size and max_size <= room)))
        return state

    ground = _ground_view(body, conditions, spacing, min_size, joint_cost, attach) if show == NODES_SHOW else None
    fresh = compute and state is None
    if fresh:
        state = lay_out()
    if compute and state is not None and json.loads(str(state.get("shape", "null"))) != shape:
        # (also when only skin, joints or fillet changed: the bars are sized again for this body)
        state = sized(state)
        if fresh:
            state = _spread_where_weak(state, lay_out, sized, conditions, safety, float(skin))
        save_frame(key, **state)
    if state is None:
        same = F.Field(lambda ctx: ctx.eval(body), F._digest("not-optimized", body.key), body.name, body.bounds,
                       children=(body,))
        same.note = "not optimized for these settings yet: press Optimize"
        same.ground_view = ground
        return {"body": same, "frame": NotOptimized(), "result": None}
    chains, sizes_ = np.asarray(state["chains"]), np.asarray(state["sizes"])
    finished = None
    if json.loads(str(state.get("shape", "null"))) == shape:
        # (Optimize would do nothing: it finished in a second and looked dead, as Topology Optimization's did)
        finished = ("Laid out and sized for these settings: nothing left to optimize. Change a setting to run it "
                    "again (Show needs no new run).")
        report_finished(finished)
    out = frame_body(crop, state["nodes"], state["bars"], chains, sizes_, _digest(key, tuple(sizes_.tolist())),
                     conditions, float(skin), prof, joints, float(fillet), continuity)
    out.finished = finished
    note = str(state["summary"])
    if json.loads(str(state.get("shape", "null"))) == shape:   # checked as it is
        check = json.loads(str(state["check"]))
    else:   # skin, joints or fillet changed since: check this body (kept by Static Analysis)
        r, text, warnings = _body_check(out, conditions, material)
        check = {"text": text, "warnings": warnings, "safety": r["min_safety"] if r else None}
        if r is not None and r["min_safety"] < safety:
            check["warnings"] = check["warnings"] + [
                f"the bars were sized for another skin, joints or fillet (size or continuity): press Optimize to "
                f"size them for these "
                f"(the body now reaches safety factor {safety_text(r['min_safety'])} of {safety:g})"]
    if check["text"]:
        note += f" · body ({CHECK_DETAIL} Static Analysis): {check['text']}"
    for text in check["warnings"]:
        report_message(text, "warning")
    out.note = note
    ref = FO.FrameRef(np.asarray(state["nodes"]), np.asarray(state["bars"]), chains, sizes_, prof, body,
                      tuple(conditions), float(skin), key, np.asarray(state["utilisation"]), np.asarray(state["axial"]))
    out.frame_view = frame_view(ref)
    out.ground_view = ground
    shown = None
    if show in SHOW:   # (the body check's analysis: its displacements are on the SSD, kept by key)
        analysis, _, _ = _check(out, conditions, material, CHECK_DETAIL)
        shown = analysis.fields[SHOW[show]] if analysis is not None else None
    return {"body": out, "frame": ref, "result": shown}


def _body_check(body, conditions, material):
    """check_result of the frame's body at CHECK_DETAIL; a frame held only by what it connects to (Connect to, no
    Fixed Faces) is not checked: Static Analysis sees none of that, and said "add Fixed Faces"."""
    if len(conditions) and not any(c.kind == "fixed" for c in conditions):
        return None, "", ["the body is not checked with Static Analysis: it is held only by what it connects to "
                          "(Connect to), which Static Analysis does not see. Check it joined to that part (Join Frames), "
                          "with its supports"]
    return check_result(body, conditions, material, CHECK_DETAIL)


def _spread_where_weak(state, lay_out, sized, conditions, safety, skin):
    """When the body stays under the safety factor by a support's or load's faces (a bolt or pin hole is
    one node, so its whole load goes into the boss through the bars meeting there), those faces are
    spread over several nodes (`ground` "around") and the frame laid out and sized again; the safer body
    is kept. Up to SPREAD_ROUNDS times (the weak spot can move to another hole)."""
    spread = ()
    for _ in range(SPREAD_ROUNDS):
        check_cancel()
        check = json.loads(str(state["check"]))
        if check.get("safety") is None or check["safety"] >= safety or not check.get("at"):
            break
        at = np.asarray(check["at"], dtype=float)
        near = [i for i, c in enumerate(conditions) if i not in spread and
                float(np.linalg.norm(sample_triangles(c.triangles, 1.0, stretch=False)[0] - at, axis=1).min())
                <= skin + SPREAD_NEAR]
        if not near:
            break
        names = ", ".join(conditions[i].label or conditions[i].kind for i in near)
        report_message(f"body check: safety factor {safety_text(check['safety'])} by {names}: laid out again with it "
                       f"spread over several bars")
        try:
            other = sized(lay_out(spread + tuple(near)))
        except Cancelled:
            raise
        except GraphError as err:
            report_message(f"spread, {names} gave no frame ({err}): the first layout is kept", "warning")
            break
        theirs = json.loads(str(other["check"])).get("safety")
        keep = theirs is not None and theirs > check["safety"]
        report_message(f"body check with {names} spread: safety factor {safety_text(theirs or 0.0)} "
                       f"({'kept' if keep else 'the layout before is kept'})")
        if not keep:
            break
        state, spread = other, spread + tuple(near)
    return state


def _segment_distance(p, a, b):
    """Distance from point p to each segment a[i]-b[i]."""
    d = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * d - p, axis=1)


def _size_for_body(crop, state, conditions, material, prof, shape, key, safety, min_size, listed, size_step,
                   max_sizes, live=None, largest=0.0, room=False):
    """Bar sizes with which the frame's body (with `shape`: skin, joints, fillet) reaches the safety
    factor in Static Analysis: the beam model's sizes, then round by round the bars at the body's most
    stressed spot made thicker (bending stress goes as 1 / size^3), never above `largest` (Max size, 0:
    none; `room`: it is what fits the design space, not Max size). Returns the state's new entries: sizes,
    summary, the body settings checked (shape) and the check. `live`: a `_Live`, shown every round."""
    nodes, bars, chains = (np.asarray(state[k]) for k in ("nodes", "bars", "chains"))
    beam = np.asarray(state["beam_sizes"], dtype=float)
    sizes = beam.copy()
    a, b = nodes[bars[:, 0]], nodes[bars[:, 1]]
    lengths = np.bincount(chains, weights=np.linalg.norm(b - a, axis=1), minlength=len(sizes))
    rounds, misses, stuck = 0, 0, None
    best = None   # (safety factor, sizes, check) of the safest body so far
    while True:
        check_cancel()
        body = frame_body(crop, nodes, bars, chains, sizes, _digest(key, tuple(sizes.tolist())), conditions,
                          shape["skin"], prof, shape["joints"], shape["fillet"], shape.get("continuity", "G1"))
        r, text, warnings = _body_check(body, conditions, material)
        if live is not None and r is not None:
            spot = [float(v) for v in r["max_away_at"]]
            live("body check", f"round {rounds + 1}: safety factor {safety_text(r['min_safety'])} of "
                 f"{safety:g} at ({spot[0]:.0f}, {spot[1]:.0f}, {spot[2]:.0f})", nodes, bars,
                 forces=np.asarray(state["axial"], dtype=float), chains=chains, sizes=sizes,
                 utilisation=np.asarray(state["utilisation"], dtype=float), spot=spot)
        better = r is not None and (best is None or r["min_safety"] >= 1.01 * best[0])
        if r is not None and (best is None or r["min_safety"] > best[0]):
            best = (r["min_safety"], sizes.copy(), (r, text, warnings))
        if r is None or r["min_safety"] >= safety or rounds == CHECK_ROUNDS:
            break
        misses = 0 if better else misses + 1
        if misses == 2:   # (one round may dip: the thicker joint draws more load before the bars round it
            stuck = "thicker bars there did not raise it"   # carry it; wall bracket 1.24, 1.19, 1.40, 1.60)
            break
        hot = np.asarray(r["max_away_at"])
        near_chain = np.full(len(sizes), np.inf)
        np.minimum.at(near_chain, chains, _segment_distance(hot, a, b))
        near = near_chain <= sizes + shape["skin"] + 2.0       # the bars meeting at that spot
        if not near.any():
            near = near_chain == near_chain.min()
        grow = 1.02 * (safety / r["min_safety"]) ** (1 / 3)
        before = sizes.copy()
        try:
            for c in np.flatnonzero(near):
                if largest and sizes[c] >= largest * (1 - 1e-9):
                    continue
                sizes[c] = max(SZ.round_up(sizes[c] * grow, min_size, size_step, listed),
                               SZ.next_size(sizes[c], size_step, listed))
                if largest:
                    sizes[c] = min(sizes[c], largest)
        except ConditionError as err:
            sizes, stuck = before, str(err)
            break
        if (sizes == before).all():   # (all of them at Max size)
            stuck = (f"the bars there are as thick as fits the design space ({largest:g} mm)" if room else
                     f"the bars there are at Max size ({largest:g} mm)")
            break
        if max_sizes > 0:
            sizes = SZ.group(sizes, lengths, prof, max_sizes)
        changed = ", ".join(f"{before[c]:g} -> {sizes[c]:g}" for c in np.flatnonzero(sizes != before))
        report_message(f"body check: safety factor {safety_text(r['min_safety'])} at ({hot[0]:.0f}, {hot[1]:.0f}, "
                       f"{hot[2]:.0f}); bars there made thicker ({changed} mm)")
        rounds += 1
    if best is not None and r is not None and best[0] > r["min_safety"]:
        # the last bars made it worse (another spot took the stress): back to the safest (bolted bracket
        # held by two bolts: 1.56 the round before, 1.13 kept)
        _, sizes, (r, text, warnings) = best
    if r is not None and r["min_safety"] < safety:
        at = np.asarray(r["max_away_at"], dtype=float)
        on = [c for c in conditions if c.kind == "load" and float(np.linalg.norm(
            sample_triangles(c.triangles, 1.0, stretch=False)[0] - at, axis=1).min()) <= 1.5 * float(r.get("h", 1.0))]
        # on a pin's or bolt's hole the stress is the load's own pressure on it: no bar, skin or layout changes that
        # (bolted bracket: 1.55 with a 2, 3.5 or 5 mm skin and with the hole spread over several bars). On flat faces
        # it is the skin between the bars that carry them: held at both ends and loaded on its top, the block reached
        # 1.2 with a 2 mm skin, 1.32 with 4; without a skin, 1.57 where 2 mm gave 2.1)
        pressed = [c for c in on if _cylinder(np.asarray(c.triangles, dtype=np.float64).reshape(-1, 3, 3)) is not None]
        spread = [c for c in on if c not in pressed]
        hint = (f"the most stressed spot is on the faces of {pressed[0].label or 'a load'}, where the load itself "
                f"presses: bars, Skin and layout do not change that; a stronger material, a lower Safety factor or "
                f"more face for it to press on (a longer hole, a bigger pin) helps" if pressed else
                f"the most stressed spot is by the faces of {spread[0].label or 'a load'}, which the Skin "
                f"({shape['skin']:g} mm) carries to the bars there: a thicker Skin, or more bars under the faces (a "
                f"smaller Node spacing), helps" if spread else
                "a larger Skin or Fillet, a stronger material or a larger design space may help" if room else
                "a larger Max size, Skin or Fillet, or another layout, may help" if largest else
                "a larger Skin or Fillet, or another layout, may help")
        warnings = warnings + [f"the body reaches safety factor {safety_text(r['min_safety'])} of {safety:g} "
                               f"({stuck or f'after {rounds} rounds of thicker bars'}); {hint}"]
    rep = json.loads(str(state["report"]))
    mass = sum(prof.section(s).A * L for s, L in zip(sizes, lengths)) * material.density / 1000.0
    curved = f", {rep['curved']} curved" if rep.get("curved") else ""
    summary = (f"{rep['bars']} bars ({rep['pieces']} pieces, {rep['joints']} joints{curved}), "
               f"{len(np.unique(sizes))} sizes ({', '.join(f'{s:g}' for s in np.unique(sizes))} mm {prof.kind}), "
               f"{mass:.3g} g of bars · beam model: max utilisation {rep['utilisation']:.2f}, max deflection "
               f"{rep['deflection']:.3g} mm, buckling factor {rep['buckling']:.3g}")
    if rounds:
        report_message(f"body check: bars thickened in {rounds} {'round' if rounds == 1 else 'rounds'} for safety factor "
                       f"{safety:g}: {summary}")
    check = {"text": text, "warnings": warnings, "safety": r["min_safety"] if r is not None else None,
             "at": [float(v) for v in r["max_away_at"]] if r is not None else None}   # (the weakest spot)
    return {"sizes": sizes, "summary": np.array(summary), "shape": np.array(json.dumps(shape)),
            "check": np.array(json.dumps(check))}


class _Live:
    """Frame Optimization's live view: the frame after every step of a run (`optimize_frame`'s `live`, the
    body check's rounds) as a `live_frame` event, at most every LIVE_EVERY s but at once for a new stage, which
    also moves the card's progress. Bars not sized yet are drawn as thick as their force needs."""

    def __init__(self, profile, allow, min_size, density=0.0):
        self.profile, self.allow, self.min_size, self.density = profile, allow, min_size, density
        self.stage, self.last, self.ticked = None, 0.0, 0.0

    def __call__(self, stage, what, nodes, bars, forces=None, chains=None, sizes=None, utilisation=None, spot=None,
                 numbers=None):
        now = time.perf_counter()
        if stage == self.stage and now - self.last < LIVE_EVERY:
            return
        if stage != self.stage:
            report_progress(FO.STAGES.index(stage), len(FO.STAGES), stage)
        self.stage, self.last = stage, now
        report_frame(live_view(nodes, bars, forces, chains, sizes, utilisation, spot, self.profile, self.allow,
                               self.min_size, stage, what, numbers, self.density))

    def tick(self, stage, text):
        """What a long step is doing now (an LP's iterations): the card's progress text, at most every TICK_EVERY s.
        Raises when the run is stopped."""
        now = time.perf_counter()
        if now - self.ticked < TICK_EVERY:
            check_cancel()
            return
        self.ticked = now
        report_progress(FO.STAGES.index(stage), len(FO.STAGES), f"{stage} ({text})")


def live_view(nodes, bars, forces, chains, sizes, utilisation, spot, profile, allow, min_size, stage, what,
              numbers=None, density=0.0):
    """A frame view (as `frame_view`) of a frame being made: only the nodes its bars use, each piece its own
    bar until bars are joined (`chains`), sizes (None: the round bar's diameter its largest force needs at the
    allowed stress, at least Min size), and `live`, `stage`, `what`, the body check's weakest `spot`, what the step
    did (`numbers`: an LP's counts and volumes, `layout.optimize`) and the bars' `mass` as drawn, g (`density`,
    g/cm3; the sections of `profile` once sized)."""
    nodes, bars = np.asarray(nodes, dtype=float), np.asarray(bars, dtype=np.int64).reshape(-1, 2)
    used = np.unique(bars)
    remap = np.full(len(nodes), -1)
    remap[used] = np.arange(len(used))
    bars = remap[bars]
    nodes = nodes[used]
    forces = np.zeros(len(bars)) if forces is None else np.asarray(forces, dtype=float)
    chains = np.arange(len(bars)) if chains is None else np.unique(np.asarray(chains), return_inverse=True)[1]
    n = int(chains.max()) + 1 if len(chains) else 0
    sized = sizes is not None
    if sized:
        sizes = np.asarray(sizes, dtype=float)
    else:
        worst = np.zeros(n)
        np.maximum.at(worst, chains, np.abs(forces))
        sizes = np.maximum(min_size, np.sqrt(4.0 * worst / (allow * np.pi)))
    lengths = np.bincount(chains, weights=np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1), minlength=n)
    utilisation = np.zeros(n) if utilisation is None else np.asarray(utilisation, dtype=float)
    areas = (np.array([profile.section(s).A for s in sizes]) if sized and len(sizes) else np.pi / 4 * sizes ** 2)
    mass = float((lengths * areas).sum()) * density / 1000.0
    numbers = {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in (numbers or {}).items()}
    return {"profile": profile.kind, "nodes": np.round(nodes, 4).tolist(), "bars": bars.tolist(),
            "chains": chains.tolist(), "sizes": np.round(sizes, 3).tolist(), "utilisation": utilisation.round(4).tolist(),
            "axial": forces.round(3).tolist(), "lengths": lengths.round(2).tolist(), "live": True, "sized": sized,
            "stage": stage, "what": what, "spot": spot, "numbers": numbers, "mass": round(mass, 4)}


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


def _ground_view(body, conditions, spacing, min_size, joint_cost, attach):
    """Show: candidate nodes, for the editor (JSON): the ground structure Optimize starts from (as
    `optimize_frame` builds it): its lattice nodes, the nodes at supports, loads and connections (kind
    and label of each) and which of them no bar reaches (`lonely`; Optimize then stops). None, with a
    warning, when there is none to show (nothing held, no load)."""
    lonely, error = (), None
    try:
        gs = G.build(body, conditions, spacing, min_size=min_size, overlap=joint_cost > 0, cache=ARRAY_CACHE,
                     attach=attach)
    except G.Unreached as err:
        if err.ground is None:
            raise
        gs, lonely, error = err.ground, err.lonely, str(err)
        report_message(f"{error} · it is the large red ball", "warning")
    except ConditionError as err:
        report_message(f"candidate nodes: {err}", "warning")
        return None
    n = len(conditions)
    lattice = gs.owner < 0
    at = np.flatnonzero(~lattice)
    owner = [int(o) for o in gs.owner[at]]
    return {"spacing": float(gs.spacing), "auto": spacing <= 0, "coarser": gs.coarser,
            "lattice": np.round(gs.nodes[lattice], 2).tolist(), "points": np.round(gs.nodes[at], 2).tolist(),
            "kinds": ["attach" if o >= n else conditions[o].kind for o in owner],
            "labels": [f"Connect to #{o - n + 1}" if o >= n else conditions[o].label or conditions[o].kind
                       for o in owner],
            "lonely": np.searchsorted(at, lonely).tolist(), "bars": int(len(gs.bars)), "error": error}


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
       Port("fillet", "number", 0.0, when={"joints": ["fillet", "ball"]}, unit="mm", zero="auto",
            doc="Fillet size, mm; 0 = automatic (0.6 x the thinner bar)."),
       Port("continuity", "text", "G2", choices=list(CONTINUITY), label="Fillet continuity",
            when={"joints": ["fillet", "ball"]},
            doc="How smoothly fillets run into the bars: G1, G2 or G3 (as in Frame Optimization).")],
      [Port("body", "implicit", doc="The frames and bodies as one part, filleted where they meet.")])
def join_frames(a, b=None, c=None, d=None, joints=JOINTS[0], fillet=0.0, continuity="G2"):
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
    if continuity not in CONTINUITY:
        raise GraphError(f"fillet continuity must be one of: {', '.join(CONTINUITY)}")
    return join_body(frames, bodies, joints, float(fillet), continuity)
