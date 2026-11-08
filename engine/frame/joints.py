"""Geometry optimization: the layout's joints moved to where the frame is lighter.

The layout picks its joints from a lattice of candidate nodes (`ground.build`), so a bar often bends at a
lattice node where a joint a little elsewhere would have let it run straight, and bars then hold that bend
sideways while carrying little (the wall bracket's struts to its pin). Here the free joints move (not those
at supports, loads or frames this one connects to), by sequential linear programming (He & Gilbert 2015,
"Rationalization of trusses generated via layout optimization"): each step is one LP in the bars' forces and
the joints' moves together, the balance of every joint linearized in the moves (its bars' directions turn
with them) and the bars' lengths too, each move within a box of the step's size; the moves stay when the
layout's LP over the moved joints costs less and every bar is still inside the design space, else the box
shrinks. Then joints that met merge, bars left carrying nothing go, joints a bar now runs straight through
are taken out, and it starts again.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import layout as LY
from .ground import _stays_inside

STEPS = 80           # moves at most per round
ROUNDS = 4           # rounds of moves, then merging and dropping
FIRST_STEP = 0.25    # x node spacing: the largest move of a joint in one step
LAST_STEP = 0.01     # x node spacing: moves stop below this
MERGE = 0.2          # x node spacing (at least Min size): joints closer than this become one
GAIN = 1e-5          # a step stays when the cost falls by more than this share
SLACK = 1000.0       # x the dearest bar's cost per newton: a joint's imbalance (see `_solve`)
REWEIGHT = 3         # LPs, each weighted by the forces of the one before, when a state is judged


def move_joints(layout, ground, sigma_t, sigma_c, joint_length=0.0, min_area=0.0, min_size=2.0, cancel=None,
                show=None, limit=None):
    """The layout with its free joints moved to lower its volume (`layout.Layout`, same supports and loads,
    every bar carrying force), and a report: volume before and after (bars at least the smallest size, no
    joint cost), steps, joints moved and how far at most, merged joints, dropped bars, joints taken out of
    straight bars, and (steps, volume) at the start and per round. The steps lower the LP's own cost with the smallest
    size's weighting held for the round; states are judged by their volume, each after a few reweighted
    LPs (`_settle`), and the lightest is kept. `show(what, nodes, bars, forces)`: called after every step
    and tidy (the live view). `limit(lengths)`: Max size's force bounds per bar (`layout.optimize`), or None."""
    nodes, bars = layout.nodes.copy(), layout.bars.copy()
    fixed, loads, index = layout.fixed, layout.loads, layout.ground_index
    anchored = fixed.any(axis=1) | (np.abs(loads).sum(axis=1) > 0) | (ground.owner[index] >= 0)
    if ground.attach is not None:
        anchored |= ground.attach[index]
    free_r = ground.free[index] if ground.free is not None else np.zeros(len(nodes))
    spacing = ground.spacing
    weights = dict(sigma_t=sigma_t, sigma_c=sigma_c, joint_length=joint_length, min_area=min_area, limit=limit)
    sizes = lambda nodes_, bars_, q_: _volume(nodes_, bars_, q_, sigma_t, sigma_c, 0.0, min_area)

    out = _settle(nodes, bars, fixed, loads, layout.forces, weights)
    if out is None:
        return layout, {"moved": False}
    # the restricted LP may need slack (the layout dropped bars under DROP of the largest force): no state
    # may need more than the layout did, with as much again to spare
    allowed = 2.0 * out.slack + LY.DROP * float(np.abs(out.q).max())
    report = {"volume_before": sizes(nodes, bars, layout.forces), "steps": 0,
              "rounds": [(0, round(sizes(nodes, bars, out.q), 3))]}   # (steps, volume) at the start, per round
    tally = {"merged": 0, "dropped": 0, "straightened": 0}
    best = (out.volume, nodes, bars, out.q, fixed, loads, index, anchored, dict(tally))
    for rnd in range(ROUNDS + 1):   # (round 0 only tidies: idle bars and joints with two bars in line first)
        box = FIRST_STEP * spacing
        steps = 0
        q_ref = out.q   # the smallest size's weighting for this round
        cur = _solve(nodes, bars, fixed, loads, q_prev=q_ref, **weights)
        for _ in range(STEPS if rnd and cur is not None and not anchored.all() else 0):
            if cancel is not None:
                cancel()
            held = anchored.copy()
            while True:   # joints whose bars would leave the design space are held for this step
                move = _step(nodes, bars, fixed, loads, cur, ~held, box, q_ref, **weights)
                if move is None:
                    break
                trial = nodes + move
                outside = ~_stays_inside(trial, bars, free_r, ground.inside, min_size)
                if not outside.any():
                    break
                hold = np.zeros(len(nodes), dtype=bool)
                hold[bars[outside].ravel()] = True
                hold &= ~held & (np.abs(move).sum(axis=1) > 0)
                if not hold.any():
                    move = None
                    break
                held |= hold
            moved = None if move is None or not np.abs(move).any() else \
                _solve(trial, bars, fixed, loads, q_prev=q_ref, **weights)
            if moved is not None and moved.cost < cur.cost * (1 - GAIN):
                nodes, cur = trial, moved
                steps += 1
                box = min(box * 1.5, FIRST_STEP * spacing)
                if show is not None:
                    show(f"step {report['steps'] + steps}", nodes, bars, cur.q)
            else:
                box *= 0.5
                if box < LAST_STEP * spacing:
                    break
        report["steps"] += steps
        if steps:
            out = _settle(nodes, bars, fixed, loads, cur.q, weights) or out
        changed = False
        tidy = _tidy(nodes, bars, out.q, anchored, max(MERGE * spacing, min_size), ground, free_r, min_size)
        if tidy is not None:
            t_nodes, t_bars, keep_nodes, counts = tidy
            t_fixed, t_loads = fixed[keep_nodes], loads[keep_nodes]
            t_out = _settle(t_nodes, t_bars, t_fixed, t_loads, None, weights)
            if t_out is not None and t_out.slack <= allowed and t_out.volume <= out.volume * (1 + GAIN):
                nodes, bars, out, fixed, loads = t_nodes, t_bars, t_out, t_fixed, t_loads
                index, anchored, free_r = index[keep_nodes], anchored[keep_nodes], free_r[keep_nodes]
                for k, v in counts.items():
                    tally[k] += v
                changed = True
                if show is not None:
                    show(f"joints that met merged, idle bars out (round {rnd + 1})", nodes, bars, out.q)
        report["rounds"].append((steps, round(sizes(nodes, bars, out.q), 3)))
        if out.slack <= allowed and out.volume < best[0]:
            best = (out.volume, nodes, bars, out.q, fixed, loads, index, anchored, dict(tally))
        if rnd and not (steps or changed):
            break
    _, nodes, bars, q, fixed, loads, index, anchored, tally = best
    nodes, bars, q, keep_nodes = _carrying(nodes, bars, q, anchored)   # (as the layout itself is)
    fixed, loads, index = fixed[keep_nodes], loads[keep_nodes], index[keep_nodes]
    report.update(tally, volume_after=sizes(nodes, bars, q))
    shift = np.linalg.norm(nodes - ground.nodes[index], axis=1)
    report["joints"], report["farthest"] = int((shift > 1e-6).sum()), float(shift.max(initial=0.0))
    plastic = float((np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
                     * np.where(q > 0, q / sigma_t, -q / sigma_c)).sum())
    moved = LY.Layout(nodes, bars, q, LY.continuous_bars(nodes, bars), fixed, loads, plastic, layout.candidates,
                      layout.rounds, index)
    report["moved"] = True
    report["bars"] = len(bars)
    return moved, report


def _settle(nodes, bars, fixed, loads, q_start, weights):
    """The lightest (by volume) of REWEIGHT LPs over these bars, the first weighted by `q_start` (None:
    plain), each next by the forces of the one before: the measure states are compared by."""
    best, q = None, q_start
    for _ in range(REWEIGHT):
        s = _solve(nodes, bars, fixed, loads, q_prev=q, **weights)
        if s is None:
            break
        if best is None or s.volume < best.volume:
            best = s
        q = s.q
    return best


def _carrying(nodes, bars, q, anchored):
    """Bars under DROP of the largest force taken out, and those then hanging loose; (nodes, bars, forces,
    the kept nodes' indices)."""
    keep = np.abs(q) > LY.DROP * np.abs(q).max(initial=0.0)
    while True:
        degree = np.bincount(bars[keep].ravel(), minlength=len(nodes))
        loose = keep & (((degree[bars[:, 0]] == 1) & ~anchored[bars[:, 0]])
                        | ((degree[bars[:, 1]] == 1) & ~anchored[bars[:, 1]]))
        if not loose.any():
            break
        keep &= ~loose
    used = np.unique(bars[keep])
    remap = np.full(len(nodes), -1)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[bars[keep]], q[keep], used


class _Solved:
    """One LP solve: forces `q`, its `cost` (what the LP minimized), `volume` (with the smallest bar size:
    the frame's real measure), `slack` (N of imbalance, summed over the joints) and `per_length` (each
    bar's cost per mm at these forces)."""
    __slots__ = ("q", "cost", "volume", "slack", "per_length")

    def __init__(self, q, cost, volume, slack, per_length):
        self.q, self.cost, self.volume, self.slack, self.per_length = q, cost, volume, slack, per_length


def _weighting(q_prev, sigma_t, sigma_c, joint_length, min_area, lengths):
    """(cost per newton in tension, in compression) of every bar: its length plus the joint's, over the
    allowed stress, times the smallest size's weight (a bar carrying less than the smallest one can costs as
    if it carried that much, as in the layout's reweighting; `q_prev`: the forces the weight is from)."""
    scale = np.ones(len(lengths))
    carry = min_area * min(sigma_t, sigma_c)
    if min_area > 0 and q_prev is not None:
        scale = np.maximum(1.0, carry / np.maximum(np.abs(q_prev), 1e-6 * carry))
    cost = lengths + joint_length
    return cost * scale / sigma_t, cost * scale / sigma_c


def _force_bounds(limit, lengths, extra):
    """linprog bounds: each bar's tension and compression up to `limit(lengths)` (Max size; None: no bound),
    then `extra` more (lower, upper) pairs for the other variables."""
    if limit is None:
        return [(0, None)] * (2 * len(lengths)) + list(extra)
    t, c = limit(lengths)
    return [(0.0, float(u)) for u in np.concatenate([t, c])] + list(extra)


def _solve(nodes, bars, fixed, loads, sigma_t, sigma_c, joint_length, min_area, q_prev, limit=None):
    """The layout's LP over these bars (`_Solved`), or None. Each joint may also be out of balance at SLACK
    times the dearest bar's cost per newton: the layout drops bars under DROP of the largest force, and
    without them its bars were ~1 N out of balance, no answer (the slack is not in the volume: states
    needing more than the layout did are not kept). `limit`: Max size's force bounds, or None."""
    from scipy import sparse
    from scipy.optimize import linprog
    L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    if (L < 1e-9).any():
        return None
    ct, cc = _weighting(q_prev, sigma_t, sigma_c, joint_length, min_area, L)
    free = ~fixed.ravel()
    B = LY.equilibrium_matrix(nodes, bars).tocsr()[free].tocsc()
    k, m = B.shape[1], B.shape[0]
    penalty = SLACK * float(max(ct.max(), cc.max()))
    eye = sparse.identity(m, format="csc")
    res = linprog(np.concatenate([ct, cc, np.full(2 * m, penalty)]), A_eq=sparse.hstack([B, -B, eye, -eye]).tocsc(),
                  b_eq=loads.ravel()[free], bounds=_force_bounds(limit, L, [(0, None)] * (2 * m)), method="highs")
    if res.status != 0:
        return None
    tension, compression = res.x[:k], res.x[k:2 * k]
    q = tension - compression
    per_length = (ct * tension + cc * compression) / (L + joint_length)
    return _Solved(q, float(res.fun), _volume(nodes, bars, q, sigma_t, sigma_c, joint_length, min_area),
                   float(res.x[2 * k:].sum()), per_length)


def _volume(nodes, bars, q, sigma_t, sigma_c, joint_length, min_area):
    """The bars' volume, each at least the smallest size (bars under DROP of the largest force left out),
    each as long as it is plus `joint_length`."""
    L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1) + joint_length
    used = np.abs(q) > LY.DROP * np.abs(q).max(initial=0.0)
    return float((L * np.maximum(np.where(q > 0, q / sigma_t, -q / sigma_c), min_area) * used).sum())


def _step(nodes, bars, fixed, loads, solved, movable, box, q_ref, sigma_t, sigma_c, joint_length, min_area,
          limit=None):
    """Joint moves (n, 3; 0 where not `movable`) from the LP in forces and moves together, linearized about
    the current joints and forces (`solved`): a bar's direction e turns by (I - e e^T) / L times the move of
    one end against the other, so its force at the old size pushes its joints by that much more; its cost
    grows by its cost per mm times e . (move). Every coordinate moves at most `box`. None when it fails."""
    from scipy import sparse
    from scipy.optimize import linprog
    n = len(nodes)
    mov = np.flatnonzero(movable)
    if not len(mov):
        return None
    a, b = bars[:, 0], bars[:, 1]
    d = nodes[b] - nodes[a]
    L = np.linalg.norm(d, axis=1)
    e = d / L[:, None]
    ct, cc = _weighting(q_ref, sigma_t, sigma_c, joint_length, min_area, L)
    turn = solved.q[:, None, None] * (np.eye(3)[None] - e[:, :, None] * e[:, None, :]) / L[:, None, None]
    column = np.full(n, -1)
    column[mov] = np.arange(len(mov))
    rows, cols, vals = [], [], []
    r3, c3 = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    for at, sign_at in ((b, 1.0), (a, -1.0)):          # the balance at this end of each bar
        for by, sign_by in ((b, 1.0), (a, -1.0)):      # changes with the move of this end
            ok = column[by] >= 0
            rows.append((3 * at[ok])[:, None, None] + r3[None])
            cols.append((3 * column[by[ok]])[:, None, None] + c3[None])
            vals.append(sign_at * sign_by * turn[ok])
    J = sparse.coo_matrix((np.concatenate([v.ravel() for v in vals]),
                           (np.concatenate([r.ravel() for r in rows]), np.concatenate([c.ravel() for c in cols]))),
                          shape=(3 * n, 3 * len(mov))).tocsr()
    free = ~fixed.ravel()
    B = LY.equilibrium_matrix(nodes, bars).tocsr()[free].tocsc()
    J = J[free].tocsc()
    k, m = B.shape[1], B.shape[0]
    grow = np.zeros_like(nodes)
    w = solved.per_length[:, None] * e
    np.add.at(grow, b, w)
    np.add.at(grow, a, -w)
    penalty = SLACK * float(max(ct.max(), cc.max()))
    eye = sparse.identity(m, format="csc")
    c = np.concatenate([ct, cc, grow[mov].ravel(), np.full(2 * m, penalty)])
    bounds = _force_bounds(limit, L, [(-box, box)] * (3 * len(mov)) + [(0, None)] * (2 * m))
    res = linprog(c, A_eq=sparse.hstack([B, -B, J, eye, -eye]).tocsc(), b_eq=loads.ravel()[free], bounds=bounds,
                  method="highs")
    if res.status != 0:
        return None
    move = np.zeros_like(nodes)
    move[mov] = res.x[2 * k:2 * k + 3 * len(mov)].reshape(-1, 3)
    return move


def _tidy(nodes, bars, q, anchored, near, ground, free_r, min_size):
    """Joints closer than `near` merged (a free one into an anchored one, two free ones at their middle),
    bars carrying nothing dropped (and what then hangs loose), free joints with two bars in line taken out
    (one bar through), all only while every bar stays inside. (nodes, bars, kept node indices, counts) or
    None when nothing changes."""
    n = len(nodes)
    nodes = nodes.copy()
    target = np.arange(n)
    merged = 0
    for i, j in sorted(cKDTree(nodes).query_pairs(near)):
        ri, rj = _root(target, i), _root(target, j)
        if ri == rj or (anchored[ri] and anchored[rj]):
            continue
        if anchored[ri]:
            ri, rj = rj, ri     # rj stays (anchored), ri goes into it
        elif not anchored[rj]:
            nodes[rj] = 0.5 * (nodes[ri] + nodes[rj])
        target[ri] = rj
        merged += 1
    roots = np.array([_root(target, i) for i in range(n)])
    b = roots[bars]
    keep = (b[:, 0] != b[:, 1]) & (np.abs(q) > LY.DROP * np.abs(q).max(initial=0.0))
    dropped = int((~keep & (b[:, 0] != b[:, 1])).sum())
    b = np.unique(np.sort(b[keep], axis=1), axis=0)
    while len(b):   # bars ending at a joint nothing else holds carry nothing
        degree = np.bincount(b.ravel(), minlength=n)
        loose = ((degree[b[:, 0]] == 1) & ~anchored[b[:, 0]]) | ((degree[b[:, 1]] == 1) & ~anchored[b[:, 1]])
        if not loose.any():
            break
        dropped += int(loose.sum())
        b = b[~loose]
    straightened = 0
    for node in range(n):   # a free joint with two bars in line: one bar through it
        at = np.flatnonzero((b[:, 0] == node) | (b[:, 1] == node))
        if len(at) != 2 or anchored[node]:
            continue
        ends = [int(b[k, 1] if b[k, 0] == node else b[k, 0]) for k in at]
        u, v = nodes[ends[0]] - nodes[node], nodes[ends[1]] - nodes[node]
        if u @ v / (np.linalg.norm(u) * np.linalg.norm(v)) > LY.STRAIGHT:
            continue
        joined = np.array([sorted(ends)])
        if not _stays_inside(nodes, joined, free_r, ground.inside, min_size).all():
            continue
        b = np.concatenate([np.delete(b, at, axis=0), joined])
        straightened += 1
    if not len(b) or not (merged or dropped or straightened):
        return None
    if not _stays_inside(nodes, b, free_r, ground.inside, min_size).all():
        return None
    used = np.unique(b)
    remap = np.full(n, -1)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[b], used, {"merged": merged, "dropped": dropped, "straightened": straightened}


def _root(target, i):
    while target[i] != i:
        i = target[i]
    return i
