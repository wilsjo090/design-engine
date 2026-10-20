"""Layout optimization: the lightest set of bars that carries the loads to the supports.

Plastic minimum-volume truss (Dorn, Gomory, Greenberg 1964): every candidate bar i carries a
tension q+ >= 0 and a compression q- >= 0, nodal equilibrium B (q+ - q-) = f holds at every free
degree of freedom, and the volume sum L (q+ / sigma_t + q- / sigma_c) is minimised: a linear
program (HiGHS). *Member adding* (Gilbert & Tyas 2003) solves it on a few short bars first and adds
only bars whose virtual strain (from the equilibrium duals) breaks the bound 1 / sigma; when none
do, the optimum over all candidates is reached.

A *joint length* s adds s to every bar's length in the cost (Parkes, Fairclough & Gilbert 2015):
each bar then costs a little extra, so fewer, longer bars win.

A smallest bar (*min area*) makes the plain LP's answer heavy: it spreads the load over hundreds
of bars carrying a few N each, and every one of them is then printed at the smallest size (the
wall bracket: 1.7 g of plastic layout, 17 g once sized, 323 bars). The real volume of a bar is
L max(|q| / sigma, A_min), so the LP is solved again with each used bar's cost per N raised to
L A_min / |q| where its force is below what the smallest size carries (iteratively reweighted,
as for sparse solutions): a bar's load moves onto fewer, fuller bars. The lightest real layout
seen is kept.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ..fea.model import ConditionError

FULL_LP = 80_000         # up to this many candidates the LP takes them all at once (seconds)
START_REACH = 1.8        # member adding: first bars up to this many spacings (neighbours, diagonals)
START_PER_FACE_NODE = 12 # ... and this many of the shortest bars from every node on a face
ADD_TOL = 1e-4           # a bar is added when its virtual strain breaks the bound by more than this
MAX_ROUNDS = 60
REWEIGHT_ROUNDS = 12     # smallest size: LPs with reweighted costs at most (stops when nothing changes)
DROP = 1e-3              # bars with less than this share of the largest force are dropped
STRAIGHT = np.cos(np.radians(178.0))   # two bars at a node this close to opposite: one bar through it


@dataclass
class Layout:
    nodes: np.ndarray       # (n, 3) used nodes
    bars: np.ndarray        # (k, 2) node indices of the used bars (pieces between nodes)
    forces: np.ndarray      # (k,) N, + = tension
    chains: np.ndarray      # (k,) continuous bar each piece belongs to
    fixed: np.ndarray       # (n, 3)
    loads: np.ndarray       # (n, 3)
    volume: float           # mm^3 of the plastic layout (bars sized exactly to their force)
    candidates: int         # bars in the last LP
    rounds: int             # member adding rounds
    ground_index: np.ndarray = None   # (n,) each node's index in the ground structure

    @property
    def lengths(self):
        return np.linalg.norm(self.nodes[self.bars[:, 1]] - self.nodes[self.bars[:, 0]], axis=1)


def equilibrium_matrix(nodes, bars):
    """B (3n x m): column i holds bar i's unit vector at its second node and minus it at its first,
    so B q = f for bar forces q (tension +) and external nodal forces f."""
    e = nodes[bars[:, 1]] - nodes[bars[:, 0]]
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    m = len(bars)
    rows = np.concatenate([3 * bars[:, 0, None] + np.arange(3), 3 * bars[:, 1, None] + np.arange(3)], axis=1)
    vals = np.concatenate([-e, e], axis=1)
    cols = np.repeat(np.arange(m), 6)
    return sparse.csc_matrix((vals.ravel(), (rows.ravel(), cols)), shape=(3 * len(nodes), m))


def _solve(B, cost_t, cost_c, f):
    m = B.shape[1]
    # interior point (with crossover to a vertex): 0.9 s on the wall bracket's 17k candidates, the
    # simplex methods 2.5 s, same optimum
    res = linprog(np.concatenate([cost_t, cost_c]), A_eq=sparse.hstack([B, -B]).tocsc(), b_eq=f,
                  bounds=(0, None), method="highs-ipm")
    if res.status == 2:
        return None
    if res.status != 0:
        raise ConditionError(f"the layout optimization failed: {res.message}")
    x = res.x
    return x[:m] - x[m:], np.asarray(res.eqlin.marginals), float(res.fun)


def optimize(ground, sigma_t, sigma_c=None, joint_length=0.0, member_adding=None, min_area=0.0, log=None,
             start=None):
    """The minimum-volume layout of a Ground structure for allowable stresses (MPa); with a smallest
    bar area (mm^2), the lightest layout once every bar is at least that thick. `start`: forces of
    a previous layout on these candidates, to begin the reweighting from (a refined ground)."""
    sigma_c = sigma_t if sigma_c is None else sigma_c
    nodes, bars, L = ground.nodes, ground.bars, ground.lengths
    free = ~ground.fixed.ravel()
    B_all = equilibrium_matrix(nodes, bars).tocsr()[free].tocsc()
    f = ground.loads.ravel()[free]
    if not np.abs(f).max() > 0:
        raise ConditionError("the loads add up to nothing (or sit on the supports)")
    if member_adding is None:
        member_adding = len(bars) > FULL_LP
    cost = L + joint_length
    if start is not None and min_area > 0:
        q, idx, rounds = np.asarray(start, dtype=np.float64), np.arange(len(bars)), 0
    else:
        q, idx, rounds = _layout(ground, B_all, f, cost / sigma_t, cost / sigma_c, member_adding, log)
    real = lambda q: float((L * np.maximum(np.where(q > 0, q / sigma_t, -q / sigma_c), min_area)
                            * (np.abs(q) > DROP * np.abs(q).max())).sum())
    if min_area > 0:
        best, best_q, used = real(q), q, np.abs(q) > DROP * np.abs(q).max()
        carry = min_area * min(sigma_t, sigma_c)    # the force the smallest bar carries
        seen = {used.tobytes()}   # it can cycle (the wall bracket: 12, 9, 9, 12, ... bars): stop on a repeat
        for r in range(REWEIGHT_ROUNDS):
            ref = np.where(used, np.maximum(np.abs(q), 1e-6 * carry), carry)
            scale = np.maximum(1.0, carry / ref)
            q, idx, _ = _layout(ground, B_all, f, cost / sigma_t * scale, cost / sigma_c * scale, member_adding, log)
            now = np.abs(q) > DROP * np.abs(q).max()
            v = real(q)
            if log:
                log(f"min size round {r + 1}: {int(now.sum())} bars, volume with the smallest size {v:.6g}")
            if v < best - 1e-9 * best:
                best, best_q = v, q
            if now.tobytes() in seen:
                break
            seen.add(now.tobytes())
            used = now
        q = best_q
    return _clean(ground, q, float((L * np.where(q > 0, q / sigma_t, -q / sigma_c)).sum()), len(idx), rounds)


def _layout(ground, B_all, f, ct, cc, member_adding, log):
    """(forces of all candidates, indices in the last LP, rounds) of one minimum-cost layout."""
    bars, L = ground.bars, ground.lengths
    if member_adding:
        active = L <= START_REACH * ground.spacing * (1 + 1e-9)
        face = np.flatnonzero(ground.owner >= 0)
        for node in face:
            mine = np.flatnonzero((bars[:, 0] == node) | (bars[:, 1] == node))
            active[mine[np.argsort(L[mine])[:START_PER_FACE_NODE]]] = True
    else:
        active = np.ones(len(bars), dtype=bool)
    rounds = 0
    while True:
        rounds += 1
        idx = np.flatnonzero(active)
        out = _solve(B_all[:, idx], ct[idx], cc[idx], f)
        if out is None:  # infeasible with these bars: allow longer ones
            if active.all():
                raise ConditionError("no frame inside the design space can carry the loads to the supports (a "
                                     "load not connected to a support? try a smaller node spacing or larger reach)")
            longer = L <= max(L[active].max() * 1.5, ground.spacing * 2.5)
            active |= longer if longer.sum() > active.sum() else np.ones_like(active)
            continue
        q_active, y, volume = out
        if not member_adding or rounds >= MAX_ROUNDS:
            break
        # virtual strain of every bar from the duals: bars outside that break the bounds come in
        v = B_all.T @ y
        excess = np.maximum(v / ct, -v / cc)
        excess[active] = 0.0
        add = np.flatnonzero(excess > 1 + ADD_TOL)
        if log:
            log(f"member adding round {rounds}: {len(idx):,} bars, volume {volume:.6g}, {len(add)} to add")
        if not len(add):
            break
        limit = max(1000, len(idx))
        if len(add) > limit:
            add = add[np.argsort(-excess[add])[:limit]]
        active[add] = True
    q = np.zeros(len(bars))
    q[idx] = q_active
    return q, idx, rounds


def _held_bars(ground, used):
    """`used` without leftovers of the solver's tolerance: bars that end at a node no other bar,
    support or load meets, and pieces of bars that reach no support (with tiny forces dropped, a
    0.1 N bar could stay on its own and make the frame check singular)."""
    from scipy.sparse.csgraph import connected_components
    n = len(ground.nodes)
    anchored = ground.fixed.any(axis=1) | (np.abs(ground.loads).sum(axis=1) > 0)
    while len(used):
        degree = np.bincount(ground.bars[used].ravel(), minlength=n)
        loose = (degree == 1) & ~anchored
        drop = loose[ground.bars[used]].any(axis=1)
        if not drop.any():
            break
        used = used[~drop]
    if len(used):
        b = ground.bars[used]
        graph = sparse.coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        held = np.zeros(n, dtype=bool)
        held[np.unique(label[np.flatnonzero(ground.fixed.any(axis=1))])] = True
        used = used[held[label[b[:, 0]]]]
    return used


def _clean(ground, q, volume, candidates, rounds):
    """Used bars only (tiny forces dropped), their nodes, and continuous bars: pieces that run
    straight on through a node are one bar (also where other bars meet it there)."""
    big = np.abs(q).max()
    used = _held_bars(ground, np.flatnonzero(np.abs(q) > DROP * big))
    bars = ground.bars[used]
    keep = np.zeros(len(ground.nodes), dtype=bool)
    keep[bars.ravel()] = True
    remap = np.cumsum(keep) - 1
    nodes = ground.nodes[keep]
    bars = remap[bars]
    return Layout(nodes, bars, q[used], continuous_bars(nodes, bars), ground.fixed[keep], ground.loads[keep],
                  volume, candidates, rounds, np.flatnonzero(keep))


def connect(layout, ground, join):
    """The layout with every ground node in `join` (e.g. a support the layout left out) joined to it
    by the shortest chain of candidate bars, ending at a node that is not held (so it shares the
    load) if there is one. The new bars carry no force in the layout; the frame check sizes them.
    Returns (layout, the joined nodes' positions)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    join = [int(g) for g in join if g not in set(layout.ground_index.tolist())]
    if not join:
        return layout, []
    n = len(ground.nodes)
    b, L = ground.bars, ground.lengths
    graph = coo_matrix((np.concatenate([L, L]), (np.concatenate([b[:, 0], b[:, 1]]), np.concatenate([b[:, 1], b[:, 0]]))),
                       shape=(n, n)).tocsr()
    in_frame = np.zeros(n, dtype=bool)
    in_frame[layout.ground_index] = True
    targets = in_frame & ~ground.fixed.any(axis=1)
    if not targets.any():
        targets = in_frame
    pairs = {tuple(sorted(p)) for p in layout.ground_index[layout.bars].tolist()}
    added, joined = [], []
    for g in join:
        dist, pred = dijkstra(graph, indices=g, return_predecessors=True)
        reach = np.where(targets, dist, np.inf)
        end = int(np.argmin(reach))
        if not np.isfinite(reach[end]):
            continue  # no bar path inside the design space: left as it is
        node = end
        while node != g:
            prev = int(pred[node])
            key = (min(prev, node), max(prev, node))
            if key not in pairs:
                pairs.add(key)
                added.append(key)
            if not ground.fixed[prev].any():
                targets[prev] = True   # the chain is part of the frame now: later ones may end on it
            node = prev
        joined.append(ground.nodes[g])
    if not added:
        return layout, []
    keep = np.zeros(n, dtype=bool)
    keep[layout.ground_index] = True
    keep[np.asarray(added).ravel()] = True
    index = np.flatnonzero(keep)
    remap = np.full(n, -1)
    remap[index] = np.arange(len(index))
    bars = np.concatenate([remap[layout.ground_index[layout.bars]], remap[np.asarray(added)]])
    forces = np.concatenate([layout.forces, np.zeros(len(added))])
    nodes = ground.nodes[index]
    return Layout(nodes, bars, forces, continuous_bars(nodes, bars), ground.fixed[index], ground.loads[index],
                  layout.volume, layout.candidates, layout.rounds, index), joined


def continuous_bars(nodes, bars, through_joints=True):
    """Chain id per piece: at each node, pairs of pieces pointing (nearly) opposite ways are joined,
    the straightest pairs first; without `through_joints` only at nodes no other bar meets (bars
    then end at every joint)."""
    parent = np.arange(len(bars))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = nodes[bars[:, 1]] - nodes[bars[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for node in range(len(nodes)):
        at = np.flatnonzero((bars[:, 0] == node) | (bars[:, 1] == node))
        if len(at) < 2 or (not through_joints and len(at) > 2):
            continue
        out = np.where((bars[at, 0] == node)[:, None], d[at], -d[at])   # pointing away from the node
        c = out @ out.T
        np.fill_diagonal(c, 1.0)
        taken = np.zeros(len(at), dtype=bool)
        for flat in np.argsort(c, axis=None):
            i, j = divmod(int(flat), len(at))
            if c[i, j] > STRAIGHT:
                break
            if i < j and not taken[i] and not taken[j]:
                taken[i] = taken[j] = True
                parent[root(at[i])] = root(at[j])
    roots = np.array([root(i) for i in range(len(bars))])
    return np.unique(roots, return_inverse=True)[1]
