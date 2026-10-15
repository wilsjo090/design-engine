"""Frame optimization from design space, supports and loads to sized, checked bars.

ground structure -> minimum-volume layout (LP) -> continuous bars -> sizes from the layout's
forces (stress and buckling over the unbraced length) -> max sizes grouping -> frame check with
the real sections; bars over their limit, or a frame that buckles below the safety factor, get
the next size up until it passes (fully stressed design, a few rounds).
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..fea.model import ConditionError
from . import beam, curves as CV, ground as G, layout as LY, sizing as S
from .profiles import Profile

MAX_ROUNDS = 12
# joint cost (share of the design space's diagonal added to every bar): on the wall brackets 0
# gave 24 and 46 bars, 0.02 gave 9 and 15, 0.05 gave 9 and 9 (bolted, glued)
JOINT_COST = 0.05


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


def optimize_frame(space, conditions, material, profile=None, sizes=None, size_step=0.5, max_sizes=0,
                   min_size=2.0, safety=2.0, spacing=0.0, reach=3.0, joint_cost=JOINT_COST, buckling_k=1.0,
                   connect_supports=True, curving=0.0, cache=None, log=None):
    """A sized, checked frame for a design space (implicit body) with supports and loads. Joint cost:
    each bar costs as if it were this share of the design space's diagonal longer (fewer, longer
    bars)."""
    t0 = time.perf_counter()
    profile = profile or Profile()
    if safety < 1:
        raise ConditionError("the safety factor must be at least 1")
    if min_size <= 0:
        raise ConditionError("min size must be greater than 0")
    sizes = sorted(float(s) for s in sizes) if sizes else None
    gs = G.build(space, conditions, spacing, reach, min_size, overlap=joint_cost > 0, cache=cache)
    allow = material.strength / safety
    lo, hi = (np.asarray(b) for b in space.bounds)
    joint_length = joint_cost * float(np.linalg.norm(hi - lo))
    lay = LY.optimize(gs, allow, joint_length=joint_length, min_area=profile.section(min_size).A, log=log)
    notes = list(gs.notes)
    if connect_supports:  # a support the layout left out would be a loose boss in the part
        left_out = [int(i) for i in gs.supports if i not in set(lay.ground_index.tolist())]
        lay, joined = LY.connect(lay, gs, left_out)
        for p in joined:
            notes.append(f"the support at ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) was not needed by the layout: joined to "
                         f"the frame by the shortest bars")
    t_layout = time.perf_counter() - t0
    nodes, bars, forces, chains, fixed, loads = lay.nodes, lay.bars, lay.forces, lay.chains, lay.fixed, lay.loads
    if curving > 0:  # round the free two-bar kinks (curves stay half the smallest size inside the space)
        nodes, bars, forces, chains, fixed, loads = CV.smooth_bars(nodes, bars, forces, chains, fixed, loads, curving,
                                                                   gs.inside, 0.5 * min_size, gs.free[lay.ground_index])
    supports = np.flatnonzero(fixed.any(axis=1))
    loaded = np.abs(loads).sum(axis=1) > 0
    n_chains = int(chains.max()) + 1
    zaxes, _ = CV.orientation(nodes, bars, chains)

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
    weight = np.bincount(chains, weights=L, minlength=n_chains)
    current = S.group(need, weight, profile, max_sizes)

    # check the frame with the real sections; bump what fails
    for rnd in range(MAX_ROUNDS):
        secs = {s: profile.section(s) for s in np.unique(current)}
        sections = [secs[current[c]] for c in chains]
        res = beam.check(nodes, bars, sections, material, supports, loads, [profile.kind] * len(bars), zaxes=zaxes)
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
        if not bump.any():
            break
        for c in np.flatnonzero(bump):
            current[c] = S.next_size(current[c], size_step, sizes)
        if max_sizes > 0:
            current = S.group(current, weight, profile, max_sizes)
    else:
        raise ConditionError("the frame did not pass its check after resizing: try a smaller node spacing")

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
              "layout_mass": lay.volume * material.density / 1000.0}
    return Frame(nodes, bars, chains, current, profile, supports, loads, res, util, lay.volume, mass, report, notes)


def _bends(nodes, pieces):
    """Whether a continuous bar's pieces are not all on one line."""
    if len(pieces) < 2:
        return False
    d = nodes[pieces[:, 1]] - nodes[pieces[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return bool(np.abs(d @ d[0]).min() < 0.9995)
