"""Frame optimization from design space, supports and loads to sized, checked bars.

ground structure -> minimum-volume layout (LP) -> continuous bars -> sizes from the layout's
forces (stress and buckling over the unbraced length) -> max sizes grouping -> frame check with
the real sections; bars over their limit, or a frame that buckles below the safety factor, get
the next size up until it passes (fully stressed design, a few rounds).
"""
import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..fea.model import ConditionError
from . import beam, curves as CV, ground as G, joints as J, layout as LY, sizing as S
from .profiles import Profile

# rounds of the beam check, each a size step thicker where it fails: a bar bent at a clamped support
# draws more moment the stiffer it gets (bolted bracket held by two bolts, round bars: 0.3-0.6 MPa less a
# round, through at 17; 12 rounds failed)
MAX_ROUNDS = 30
MIN_GROWTH = 0.02     # ... each at least this share thicker (so a 0.5 mm size step is one step up to 25 mm bars)
LP_LOADS = (1e-2, 1e5)   # N: loads on the ground's nodes outside this are scaled for the LPs (`_frame`)
# joint cost (share of the design space's diagonal added to every bar): on the wall brackets 0
# gave 24 and 46 bars, 0.02 gave 9 and 15, 0.05 gave 9 and 9 (bolted, glued)
JOINT_COST = 0.05
# the steps a run goes through (the live view's progress); the node's body check is the last
STAGES = ("layout", "mid-bar joints", "moving joints", "sizing", "body check")
# Light bars "remove": a bar using under this share of its strength in the beam check is tried without (the
# wall bracket's bracing at its struts' bend, 2 mm at Min size, used 5-30%)
LIGHT_SHARE = 0.35
# Max size: in the layout a bar may carry at most this share of what a bar of Max size carries (in
# compression, over its own length); when bars at Max size then fail the beam check, the next share is tried.
# Bars that share a load in the layout do not share it so in the elastic frame (the stiffer path draws more),
# and the check adds bending at the clamped joints. Block 60 x 20 x 30 mm, 200 N, PLA, safety 2: Max 4 mm
# passes at 0.8 (9.6 g, unlimited 10.5 g); Max 3 mm failed at 0.8 (a bar at 1.38 of its strength) and passed
# at 0.6 (163 bars, 18.6 g)
MAX_SIZE_SHARES = (0.8, 0.6, 0.45)
# Mid-bar joints are left out when they make more candidates than this: their nodes' bars reach twice as far, and the
# LPs grow dear fast (the block at 3 mm spacing: 110k candidates, 77 s in all, 13.2 g; at 2 mm 374k, 11 minutes)
MID_BAR_LP = 150_000


class MaxSizeShort(ConditionError):
    """Bars at Max size fail the frame check (laid out again with a lower share, MAX_SIZE_SHARES). `room`: the
    largest size that fits the design space when that, not Max size, held them (`_room_size`), with `need`,
    the thickest bar the layout asked for, and `thickest`, the design space's thickest spot (mm)."""

    def __init__(self, message, room=0.0, need=0.0, thickest=0.0):
        super().__init__(message)
        self.room, self.need, self.thickest = room, need, thickest


@dataclass
class FrameRef:
    """A finished frame as a node passes it on (Frame Optimization's `frame` output): its bars, where
    it lives and what holds it, for Connect to and Join Frames."""
    nodes: np.ndarray
    bars: np.ndarray
    chains: np.ndarray
    sizes: np.ndarray
    profile: object
    space: object            # implicit design space (the part is cropped by it)
    conditions: tuple        # its supports and loads (the skin)
    skin: float
    key: str
    utilisation: np.ndarray = None
    axial: np.ndarray = None


@dataclass
class Frame:
    nodes: np.ndarray        # (n, 3)
    bars: np.ndarray         # (k, 2) pieces between nodes
    chains: np.ndarray       # (k,) continuous bar of each piece
    sizes: np.ndarray        # (chains,) mm
    profile: Profile
    supports: np.ndarray     # node indices
    loads: np.ndarray        # (n, 3) N
    check: object            # beam.Check of the final sizes
    utilisation: np.ndarray  # (chains,) stress / allowable (1 = at the limit)
    layout_volume: float     # mm^3 of the plastic layout
    mass: float              # g
    report: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def lengths(self):
        return np.linalg.norm(self.nodes[self.bars[:, 1]] - self.nodes[self.bars[:, 0]], axis=1)

    def summary(self):
        r = self.report
        sizes = ", ".join(f"{s:g}" for s in np.unique(self.sizes))
        curved = f", {r['curved']} curved" if r.get("curved") else ""
        return (f"{r['bars']} bars ({r['pieces']} pieces, {r['joints']} joints{curved}), {len(np.unique(self.sizes))} sizes "
                f"({sizes} mm {self.profile.kind}), {self.mass:.3g} g · max utilisation {r['utilisation']:.2f} · "
                f"max deflection {r['deflection']:.3g} mm · buckling factor {r['buckling']:.3g}")


def unbraced_lengths(nodes, bars, chains, braced):
    """Per continuous bar: its longest stretch between nodes where something holds it sideways
    (another bar, a support, a load)."""
    degree = np.bincount(bars.ravel(), minlength=len(nodes))
    hold = braced | (degree > 2)
    L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    out = np.zeros(chains.max() + 1)
    for c in range(len(out)):
        pieces = np.flatnonzero(chains == c)
        if len(pieces) == 1:
            out[c] = L[pieces[0]]
            continue
        # walk the chain: pieces meet at nodes with two pieces of this chain
        ends = bars[pieces]
        count = np.bincount(ends.ravel(), minlength=len(nodes))
        stretch, best = 0.0, 0.0
        start = next((int(v) for v in ends.ravel() if count[v] == 1), int(ends[0, 0]))
        node, left = start, set(pieces.tolist())
        while left:
            nxt = next((p for p in left if node in bars[p]), None)
            if nxt is None:
                break
            left.discard(nxt)
            stretch += L[nxt]
            node = int(bars[nxt, 1] if bars[nxt, 0] == node else bars[nxt, 0])
            if hold[node]:
                best, stretch = max(best, stretch), 0.0
        out[c] = max(best, stretch)
    return out


def optimize_frame(*args, max_size=0.0, **kw):
    """`_frame` (the arguments are its own); with a Max size, laid out again at the next of MAX_SIZE_SHARES
    while bars at Max size fail the frame check. Bars thicker than the design space allows (`_room_size`: they
    would be cut off) fail too: then it is laid out again with that as Max size (safety factor 100 had made
    113 mm bars in a 20 mm block, and its "4.85 kg of bars" were a 45 g block)."""
    try:
        return _shared(args, kw, max_size)
    except MaxSizeShort as err:
        if not err.room:
            raise
        first = err
    log = kw.get("log")
    if log:
        log(f"{first}: laid out again with bars of at most {first.room:g} mm")
    wanted = f"{first.need:g} mm" if first.need > first.room else f"more than {first.room:g} mm"
    try:
        frame = _shared(args, kw, first.room)
    except ConditionError as err:
        raise ConditionError(f"the bars would have to be thicker than the design space: {wanted} where it is at most "
                             f"{G.tidy(first.thickest):g} mm thick (they would be cut off), and no frame of bars up to "
                             f"{first.room:g} mm passes its check. A stronger material, a lower Safety factor, smaller "
                             f"loads or a larger design space helps") from err
    frame.notes.append(f"bars of {wanted} were needed, more than fits the design space (at most "
                       f"{G.tidy(first.thickest):g} mm thick): laid out again with bars of at most {first.room:g} mm "
                       f"sharing the loads, as Max size does")
    return frame


def _shared(args, kw, max_size):
    """`_frame` with a Max size: at each of MAX_SIZE_SHARES until bars at Max size pass the frame check."""
    if not max_size:
        return _frame(*args, **kw)
    for i, share in enumerate(MAX_SIZE_SHARES):
        try:
            frame = _frame(*args, max_size=max_size, share=share, **kw)
        except MaxSizeShort as err:
            if err.room or i == len(MAX_SIZE_SHARES) - 1:
                raise
            if kw.get("log"):
                kw["log"](f"{err}: laid out again with bars carrying at most {MAX_SIZE_SHARES[i + 1]:.0%} of that")
            continue
        if i:
            frame.notes.append(f"Max size: laid out with each bar carrying at most {share:.0%} of what a "
                               f"{max_size:g} mm bar carries (at {MAX_SIZE_SHARES[i - 1]:.0%}, bars at Max size "
                               f"failed the frame check: the elastic frame does not share loads as the layout did)")
        return frame


def _frame(space, conditions, material, profile=None, sizes=None, size_step=0.5, max_sizes=0,
           min_size=2.0, safety=2.0, spacing=0.0, reach=3.0, joint_cost=JOINT_COST, buckling_k=1.0,
           connect_supports=True, curving=0.0, mid_bar=True, attach=(), cache=None, log=None, spread=(),
           cancel=None, light_bars=False, move_joints=False, live=None, max_size=0.0, share=MAX_SIZE_SHARES[0]):
    """A sized, checked frame for a design space (implicit body) with supports and loads. Joint cost:
    each bar costs as if it were this share of the design space's diagonal longer (fewer, longer
    bars). `max_size` (mm, 0: none): no bar thicker; the layout shares a load among more bars where one
    would need more, each carrying at most `share` of what a bar of Max size carries. `spread`: indices of
    conditions whose holes take several bars (`ground.build`). `cancel`:
    called between the long steps (it raises to stop the run; big LPs can then be ended at once).
    `move_joints`: the layout's free joints move off the lattice where that makes it lighter
    (`joints.move_joints`); `light_bars`: bars using little of their strength are tried without.
    `live(stage, what, nodes, bars, forces=, chains=, sizes=, utilisation=)`: called with the frame as it is
    after every step (stage: one of STAGES; the editor's live view)."""
    t0 = time.perf_counter()
    profile = profile or Profile()
    if safety < 1:
        raise ConditionError("the safety factor must be at least 1")
    if min_size <= 0:
        raise ConditionError("min size must be greater than 0")
    if max_size and max_size < min_size:
        raise ConditionError(f"Max size ({max_size:g} mm) must be at least Min size ({min_size:g} mm)")
    sizes = sorted(float(s) for s in sizes) if sizes else None
    limit = None
    if max_size:
        top = profile.section(max_size)
        limit = lambda lengths: S.force_limits(top, lengths, material, safety, buckling_k, share)
    gs = G.build(space, conditions, spacing, reach, min_size, overlap=joint_cost > 0, cache=cache, attach=attach,
                 spread=spread)
    thickest = gs.inside.thickest() if gs.inside is not None else 0.0
    room = _room_size(profile, thickest, min_size, size_step, sizes)
    largest = min(max_size, room) if max_size and room else (max_size or room)   # (no bar thicker)
    by_room = bool(room) and (not max_size or room < max_size * (1 - 1e-9))
    allow = material.strength / safety
    # The LPs' tolerances are absolute: 1e-9 N came back as noise, every bar dropped. When the loads are far from
    # newtons, the LPs get them and the smallest bar's area scaled alike (the same bars; forces, costs and volume of
    # ordinary size: scaling the allowed stress instead left costs of 1e-10 and member adding asked for every bar)
    biggest = float(np.abs(gs.loads).max())
    scale = 1.0 if LP_LOADS[0] <= biggest <= LP_LOADS[1] else 10.0 ** round(math.log10(biggest))
    lp_allow, lp_limit, lp_live, lp_area = allow, limit, live, profile.section(min_size).A
    if scale != 1.0:
        gs = dataclasses.replace(gs, loads=gs.loads / scale)
        lp_area = lp_area / scale
        lp_limit = None if limit is None else (lambda lengths: tuple(v / scale for v in limit(lengths)))
        lp_live = None if live is None else (
            lambda stage, what, n, b, forces=None, numbers=None, **k: live(
                stage, what, n, b, **k, forces=None if forces is None else np.asarray(forces) * scale,
                numbers=None if numbers is None else {key: v * scale if key in ("volume", "real") else v
                                                      for key, v in numbers.items()}))
    lo, hi = (np.asarray(b) for b in space.bounds)
    joint_length = joint_cost * float(np.linalg.norm(hi - lo))
    on_ground = lambda stage: None if live is None else (
        lambda what, q, numbers=None: _ground_view(lp_live, stage, what, gs, q, numbers))
    # (an LP of a fine spacing runs for seconds: its iterations go to the card, "layout (LP 8, interior point iteration
    # 28)")
    ticker = lambda stage: (lambda text: live.tick(stage, text)) if hasattr(live, "tick") else None
    try:
        lay = LY.optimize(gs, lp_allow, joint_length=joint_length, min_area=lp_area, log=log, cancel=cancel,
                          show=on_ground("layout"), limit=lp_limit, tick=ticker("layout"))
    except LY.NoFrame as err:
        raise ConditionError(f"{err}{_why_none(min_size, thickest, gs)}") from err
    # (each sample by the share of its voxel inside: counting those inside, samples on a box's faces were left out)
    held = (float(np.clip(0.5 - gs.inside.values / gs.inside.step, 0.0, 1.0).sum()) * gs.inside.step ** 3
            if gs.inside is not None else 0.0)
    if held and lay.least * scale > 2 * held:
        # (no truss of these bars carries the loads in less: 1e5 N on the block was laid out again at the largest size
        # that fits for 15 minutes before it said so)
        raise ConditionError(f"the loads need at least {lay.least * scale / 1000:.3g} cm3 of material at the allowed "
                             f"stress ({allow:.3g} MPa: strength / Safety factor), more than twice what the design "
                             f"space holds ({held / 1000:.3g} cm3): no frame fits in it. A stronger material, a lower "
                             f"Safety factor, smaller loads or a larger design space helps")
    skipped = None
    if mid_bar:  # nodes along the bars used, and a second layout: branches may land part way along a bar
        finer = G.refine(gs, lay, gs.spacing, reach, min_size)
        if finer is not gs and len(finer.bars) > MID_BAR_LP:
            # (minutes of LPs for each round: Max size = Min size spread the load over so many bars that the nodes
            # along them made 165k candidates, over an hour in all; 2 mm spacing 13 of its 18 minutes)
            skipped, finer = len(finer.bars), gs
        if finer is not gs:
            index = {tuple(sorted(p)): i for i, p in enumerate(finer.bars.tolist())}
            start = np.zeros(len(finer.bars))   # the first layout's forces, to start from
            for (a, b), q in zip(lay.ground_index[lay.bars].tolist(), lay.forces):
                start[index[(min(a, b), max(a, b))]] = q
            gs = finer
            lay = LY.optimize(gs, lp_allow, joint_length=joint_length, min_area=lp_area, log=log, start=start,
                              cancel=cancel, show=on_ground("mid-bar joints"), limit=lp_limit,
                              tick=ticker("mid-bar joints"))
    notes = list(gs.notes)
    if skipped:
        notes.append(f"mid-bar joints left out: with them the layout would have had {skipped:,} candidate bars, "
                     f"minutes of LPs a round (a larger Node spacing or Max size keeps them)")
    moves = None
    if move_joints:  # joints off the lattice: bars straighter, idle ones gone
        lay, moves = J.move_joints(lay, gs, lp_allow, lp_allow, joint_length=joint_length, min_area=lp_area,
                                   min_size=min_size, cancel=cancel,
                                   show=None if live is None else
                                   lambda what, n, b, q: lp_live("moving joints", what, n, b, forces=q), limit=lp_limit)
        if moves.get("joints"):
            less = 1.0 - moves["volume_after"] / max(moves["volume_before"], 1e-12)
            gone = moves["dropped"] + moves["merged"] + moves["straightened"]
            notes.append(f"joints moved: {moves['joints']} joints up to {moves['farthest']:.1f} mm off the lattice"
                         + (f", {gone} bars fewer" if gone else "") + f", layout {less:.0%} lighter")
        if live is not None:
            lp_live("moving joints", notes[-1] if moves.get("joints") else "no joint moved", lay.nodes, lay.bars,
                    forces=lay.forces)
    if connect_supports:  # a support the layout left out would be a loose boss in the part
        # (per piece: a hole held at several nodes is used when any of them is; its other nodes, joined, were
        # bars between support nodes, which carry nothing)
        used = set(lay.ground_index.tolist())
        used_pieces = {int(gs.piece[i]) for i in used if gs.piece is not None and gs.piece[i] >= 0}
        left_out = [int(i) for i in gs.supports if i not in used and not gs.attached[i]
                    and (gs.piece is None or gs.piece[i] < 0 or int(gs.piece[i]) not in used_pieces)]
        lay, joined = LY.connect(lay, gs, left_out)
        for p in joined:
            notes.append(f"the support at ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) was not needed by the layout: joined to "
                         f"the frame by the shortest bars")
    if scale != 1.0:   # (back to newtons and mm^3)
        lay = dataclasses.replace(lay, forces=lay.forces * scale, loads=lay.loads * scale, volume=lay.volume * scale)
        if moves is not None and "volume_before" in moves:
            moves["volume_before"] *= scale
            moves["volume_after"] *= scale
    t_layout = time.perf_counter() - t0
    owner = gs.owner[lay.ground_index]       # which support, load or connected item a node belongs to
    nodes, bars, forces, chains, fixed, loads = lay.nodes, lay.bars, lay.forces, lay.chains, lay.fixed, lay.loads
    if not mid_bar:  # bars end at every joint (none runs on through one)
        chains = LY.continuous_bars(nodes, bars, through_joints=False)
    if curving > 0:  # round the free two-bar kinks (curves stay half the smallest size inside the space)
        nodes, bars, forces, chains, fixed, loads = CV.smooth_bars(nodes, bars, forces, chains, fixed, loads, curving,
                                                                   gs.inside, 0.5 * min_size, gs.free[lay.ground_index],
                                                                   through_joints=mid_bar)
    supports = np.flatnonzero(fixed.any(axis=1))
    loaded = np.abs(loads).sum(axis=1) > 0
    n_chains = int(chains.max()) + 1
    zaxes, _ = CV.orientation(nodes, bars, chains)
    if live is not None:
        live("sizing", f"{n_chains} continuous bars" + (", curved" if curving > 0 else ""), nodes, bars,
             forces=forces, chains=chains)

    # sizes from the layout's forces
    L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    span = unbraced_lengths(nodes, bars, chains, loaded | fixed.any(axis=1))
    need = np.zeros(n_chains)
    for c in range(n_chains):
        mine = chains == c
        q = forces[mine]
        worst_t, worst_c = max(q.max(), 0.0), max(-q.min(), 0.0)
        s_t = S.required_size(profile, worst_t, span[c], material, safety, buckling_k) if worst_t else 0.0
        s_c = S.required_size(profile, -worst_c, span[c], material, safety, buckling_k) if worst_c else 0.0
        need[c] = S.round_up(max(s_t, s_c), min_size, size_step, sizes)
    wanted = float(need.max())
    if largest:   # (a bar of pieces in line buckles over all of them: the check below says if it fails)
        need = np.minimum(need, float(largest))
    weight = np.bincount(chains, weights=L, minlength=n_chains)
    current = S.group(need, weight, profile, max_sizes)

    # check the frame with the real sections; bump what fails
    label = "the design space's thickness" if by_room else "Max size"
    sizer = lambda bars_, chains_, current_, show=None: _checked(
        nodes, bars_, chains_, current_, profile, material, safety, supports, loads, loaded | fixed.any(axis=1),
        size_step, sizes, max_sizes, buckling_k, cancel, show, largest=largest, label=label)
    sized = None if live is None else lambda what, b, c, s, f, u: live("sizing", what, nodes, b, forces=f, chains=c,
                                                                         sizes=s, utilisation=u)
    try:
        current, res, util, rnd = sizer(bars, chains, current, sized)
    except MaxSizeShort as err:
        if by_room:   # (`optimize_frame` lays it out again with the room as Max size)
            raise MaxSizeShort(str(err), room, wanted, thickest) from err
        raise
    if light_bars:
        bars, chains, current, res, util, taken = _drop_light(nodes, bars, chains, current, res, util, profile,
                                                              loaded, supports if connect_supports else (), sizer,
                                                              cancel, sized)
        if taken:
            notes.append(f"light bars: {taken} taken out (each used under {LIGHT_SHARE:.0%} of its strength; the "
                         f"frame passes its check without them and weighs less)")
        n_chains = int(chains.max()) + 1
        L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)

    for k in range(len(attach)):  # what this frame puts on each frame or body it connects to
        mine = np.flatnonzero(owner == len(conditions) + k)
        if len(mine):
            push = np.round(-res.reactions[mine, :3].sum(axis=0), 1) + 0.0   # + 0.0: no "-0"
            notes.append(f"connect to #{k + 1}: joined at {len(mine)} {'point' if len(mine) == 1 else 'points'}, "
                         f"this frame puts ({push[0]:g}, {push[1]:g}, {push[2]:g}) N on it")
    area = np.array([profile.section(current[c]).A for c in chains])
    mass = float((area * L).sum() * material.density / 1000.0)
    load_nodes = np.flatnonzero(loaded)
    deflection = float(np.linalg.norm(res.displacements[load_nodes, :3], axis=1).max()) if len(load_nodes) else 0.0
    degree = np.bincount(bars.ravel(), minlength=len(nodes))
    curved = sum(1 for c in range(n_chains) if _bends(nodes, bars[chains == c]))
    report = {"bars": n_chains, "pieces": len(bars), "joints": int((degree > 2).sum()), "utilisation": float(util.max()),
              "deflection": deflection, "buckling": float(res.buckling), "rounds": rnd + 1, "curved": curved,
              "layout_seconds": t_layout, "seconds": time.perf_counter() - t0, "candidates": len(gs.bars),
              "nodes": len(gs.nodes), "active": lay.candidates, "adding_rounds": lay.rounds,
              "layout_mass": lay.volume * material.density / 1000.0, "thickest": thickest, "largest": largest,
              "room": room}
    if moves is not None:
        report["moves"] = moves
    return Frame(nodes, bars, chains, current, profile, supports, loads, res, util, lay.volume, mass, report, notes)


def _ground_view(live, stage, what, ground, q, numbers=None):
    """`live` with the candidate bars of a ground structure that carry force (`q`: on all of them), and what the
    LP did (`layout.optimize`'s `show`)."""
    used = np.flatnonzero(np.abs(q) > LY.DROP * np.abs(q).max())
    live(stage, what, ground.nodes, ground.bars[used], forces=q[used], numbers=numbers)


def _checked(nodes, bars, chains, current, profile, material, safety, supports, loads, braced, size_step, sizes,
             max_sizes, buckling_k, cancel=None, show=None, largest=0.0, label="Max size"):
    """The frame checked with its real sections (beam model), bars that fail made a size thicker until it
    passes: (sizes, check, utilisation per bar, rounds). Nodes without bars (Light bars took them out) are
    held in the check, where they meet nothing. `show(what, bars, chains, sizes, forces, utilisation)`:
    called every round (the live view). `largest`: Max size (0: none); bars at it stay, and when only such
    bars fail the frame is not made."""
    n_chains = int(chains.max()) + 1
    current = np.asarray(current, dtype=np.float64).copy()
    allow = material.strength / safety
    L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    weight = np.bincount(chains, weights=L, minlength=n_chains)
    span = unbraced_lengths(nodes, bars, chains, braced)
    zaxes, _ = CV.orientation(nodes, bars, chains)
    held = np.union1d(np.asarray(supports, dtype=np.int64),
                      np.flatnonzero(np.bincount(bars.ravel(), minlength=len(nodes)) == 0))
    for rnd in range(MAX_ROUNDS):
        if cancel is not None:
            cancel()
        secs = {s: profile.section(s) for s in np.unique(current)}
        sections = [secs[current[c]] for c in chains]
        res = beam.check(nodes, bars, sections, material, held, loads, [profile.kind] * len(bars), zaxes=zaxes)
        util = np.zeros(n_chains)
        np.maximum.at(util, chains, res.stress / allow)
        # each bar in compression against buckling over its unbraced length, with the frame's forces
        for c in range(n_chains):
            comp = -res.axial[chains == c].min()
            if comp > 0:
                _, cap = S.capacity(secs[current[c]], span[c], material, safety, buckling_k)
                util[c] = max(util[c], comp / cap)
        bump = util > 1 + 1e-6
        if res.buckling < safety:
            bump |= np.bincount(chains, weights=np.minimum(res.axial, 0.0), minlength=n_chains) < 0
        if show is not None:
            show(f"sized by the beam model, round {rnd + 1}: utilisation up to {util.max():.2f}", bars, chains,
                 current, res.axial, util)
        if not bump.any():
            return current, res, util, rnd
        if largest:   # bars at Max size cannot grow; when nothing else can, no frame passes
            at_max = bump & (current >= largest * (1 - 1e-9))
            if at_max.any() and not (bump & ~at_max).any():
                worst = int(np.argmax(np.where(at_max, util, -1.0)))
                at = nodes[bars[chains == worst]].mean(axis=(0, 1))
                raise MaxSizeShort(f"bars at {label} ({largest:g} mm) do not pass the frame check (at ({at[0]:.0f}, "
                                     f"{at[1]:.0f}, {at[2]:.0f}): {util[worst]:.2f} of their strength, buckling "
                                     f"factor {res.buckling:.2g}): a larger Max size, a lower Safety factor or a "
                                     f"smaller node spacing helps")
            bump &= ~at_max
        for c in np.flatnonzero(bump):
            current[c] = _grown(current[c], size_step, sizes, largest)
        if max_sizes > 0:
            current = S.group(current, weight, profile, max_sizes)
    worst = int(np.argmax(res.stress))
    at = nodes[bars[worst]].mean(axis=0)
    if largest and ((util > 1 + 1e-6) & (current >= largest * (1 - 1e-9))).any():
        # (some at the largest size, the others still growing: 1e5 N on the block; laid out again as at Max size)
        raise MaxSizeShort(f"bars at {label} ({largest:g} mm) and others still do not pass the frame check after "
                           f"{MAX_ROUNDS} rounds of thicker bars (at ({at[0]:.0f}, {at[1]:.0f}, {at[2]:.0f}): "
                           f"{res.stress[worst]:.3g} MPa where {allow:.3g} is allowed)")
    raise ConditionError(f"the frame did not pass its check after {MAX_ROUNDS} rounds of thicker bars (at "
                         f"({at[0]:.0f}, {at[1]:.0f}, {at[2]:.0f}): {res.stress[worst]:.3g} MPa where {allow:.3g} "
                         f"is allowed, buckling factor {res.buckling:.2g}): a stronger material, a lower safety "
                         f"factor, Curving 0 or another node spacing may help")


def _grown(size, step, sizes, largest):
    """The next size up for a bar that fails the frame check, at least MIN_GROWTH larger (a listed size: the
    next one). One size step at a time, a Size step of 0.01 mm grew a bar 0.3 mm in MAX_ROUNDS rounds, and a
    60 m part's bars 15 mm: neither frame passed, though thicker bars would have."""
    up = S.next_size(size, step, sizes, largest)
    if sizes:
        return up
    target = size * (1 + MIN_GROWTH)
    return max(up, S.round_up(min(target, largest) if largest else target, 0.0, step, None, largest))


def _why_none(min_size, thickest, ground):
    """What may have left the layout without a frame, for its error: Min size about as thick as the design space
    (15 mm bars in a 20 mm block, 1 mm in a 1 mm plate: their centres fit on a sheet in its middle, and few bars do)."""
    if thickest and min_size > 0.7 * thickest:
        return (f": Min size {min_size:g} mm is nearly as thick as the design space (at most {G.tidy(thickest):g} mm), "
                f"so bars fit only along its middle: a smaller Min size helps")
    return (f" with nodes {ground.spacing:g} mm apart ({len(ground.bars):,} candidate bars): a smaller Node spacing gives "
            f"it more to choose from")


def _room_size(profile, thickest, min_size, step, sizes):
    """The largest bar size whose section (its thinner side) fits the design space's thickest spot, on the
    Size step or in the list (mm; 0 when the thickness is not known). Raises when even the smallest size the
    settings allow does not fit (a Size step of 50 mm made every bar 50 mm in a 20 mm block)."""
    if not thickest > 0:
        return 0.0
    unit = profile.section(1.0)
    # its thinner side (per mm of size): bars are turned, a 2 x 20 mm rectangle stands in a 20 x 30 mm block's 30 mm
    per = min(unit.width, unit.height)
    fits = thickest * (1 + 1e-3) / per
    smallest = S.round_up(min_size, min_size, step, sizes)
    if smallest > fits:
        tall = f", {smallest * per:g} mm across its thinner side" if per < 1 - 1e-9 else ""
        what = ("the smallest size in the list" if sizes else "Min size rounded up to the Size step"
                if smallest > min_size * (1 + 1e-9) else "Min size")
        raise ConditionError(f"{what} ({smallest:g} mm{tall}) does not fit in the design space, which is at most "
                             f"{G.tidy(thickest):g} mm thick")
    if sizes:
        return max(s for s in sizes if s <= fits)
    return max(smallest, math.floor(fits / step + 1e-9) * step)


def _drop_light(nodes, bars, chains, current, res, util, profile, loaded, keep_held, sizer, cancel=None, show=None):
    """Light bars taken out (Light bars: remove): one at a time, the one carrying least first, of those using
    under LIGHT_SHARE of their strength; a removal stays when the frame, sized again, passes its check and
    weighs less. Bars it leaves ending in the air go with it; every load and (`keep_held`) support keeps a
    bar. Returns (bars, chains, sizes, check, utilisation, how many were taken out). `show` as `_checked`'s,
    after every removal that stays."""
    ident = np.arange(int(chains.max()) + 1)        # the bars' first numbers (tried ones are not tried again)
    tried, taken = set(), 0
    anchored = np.zeros(len(nodes), dtype=bool)
    anchored[np.asarray(keep_held, dtype=np.int64)] = True
    anchored |= loaded

    def volume(bars_, chains_, sizes_):
        area = np.array([profile.section(sizes_[c]).A for c in chains_])
        return float((area * np.linalg.norm(nodes[bars_[:, 1]] - nodes[bars_[:, 0]], axis=1)).sum())
    while True:
        light = [c for c in range(len(ident)) if util[c] < LIGHT_SHARE and int(ident[c]) not in tried]
        if not light:
            break
        c = min(light, key=lambda c: (float(np.abs(res.axial[chains == c]).max()), c))
        tried.add(int(ident[c]))
        keep = chains != c
        while True:   # bars now ending in the air carry nothing: they go too
            degree = np.bincount(bars[keep].ravel(), minlength=len(nodes))
            loose = keep & (((degree[bars[:, 0]] == 1) & ~anchored[bars[:, 0]])
                            | ((degree[bars[:, 1]] == 1) & ~anchored[bars[:, 1]]))
            if not loose.any():
                break
            keep &= ~loose
        degree = np.bincount(bars[keep].ravel(), minlength=len(nodes))
        if not keep.any() or (degree[anchored] == 0).any():
            continue
        used, new_chains = np.unique(chains[keep], return_inverse=True)
        try:
            trial = sizer(bars[keep], new_chains, current[used])
        except (ConditionError, RuntimeError, ValueError, ArithmeticError):
            continue   # (it fails its check, or the frame can turn: a singular stiffness)
        if not np.isfinite(trial[1].displacements).all() or \
                volume(bars[keep], new_chains, trial[0]) >= volume(bars, chains, current) - 1e-9:
            continue
        bars, chains, ident = bars[keep], new_chains, ident[used]
        current, res, util, _ = trial
        taken += 1
        if show is not None:
            show(f"light bars: {taken} taken out", bars, chains, current, res.axial, util)
        if cancel is not None:
            cancel()
    return bars, chains, current, res, util, taken


def _bends(nodes, pieces):
    """Whether a continuous bar's pieces are not all on one line."""
    if len(pieces) < 2:
        return False
    d = nodes[pieces[:, 1]] - nodes[pieces[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return bool(np.abs(d @ d[0]).min() < 0.9995)
