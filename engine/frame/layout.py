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

A largest bar (*Max size*) bounds each candidate's tension and compression by what that size carries
over the candidate's length (`limit`): where one bar would need more, the load is shared by more bars.

The LPs of one layout are one model kept in HiGHS (`_Model`, scipy's own copy of it): member adding adds
columns, the smallest size's rounds change costs. Each is solved by interior point without crossover: the
vertex crossover finds costs as much again (8 of 17 s at 2 mm spacing), and the interior duals name fewer bars
to add (1,150 in the second round where the vertex's named 7,380; 4 rounds to the optimum instead of 5, each
9 s instead of 22). A simplex from the last basis took longer than 5 minutes, PDLP and the dual simplex over 2.
The solve is stopped from HiGHS's callback (Stop takes effect at once, in this process) and reports its
iterations (`tick`).
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ..fea.model import ConditionError

try:   # (a scipy without its HiGHS bindings: linprog, each LP built anew and solved apart)
    from scipy.optimize._highspy import _core as _hc
    _Highs = _hc._Highs
except Exception:   # pragma: no cover
    _hc = _Highs = None

FULL_LP = 80_000         # up to this many candidates the LP takes them all at once (seconds)
START_REACH = 1.8        # member adding: first bars up to this many spacings (neighbours, diagonals)
START_PER_FACE_NODE = 12 # ... and this many of the shortest bars from every node on a face
ADD_TOL = 1e-4           # a bar is added when its virtual strain breaks the bound by more than this
MAX_ROUNDS = 60
ADD_STALL = 1e-4         # member adding stops when the volume falls by less than this share ...
STALL_ROUNDS = 1         # ... this many rounds in a row (2 with the vertex duals of a crossover, which named
                         # ~1000 bars a round that lowered nothing; the interior duals name few: 9 s an LP saved)
REWEIGHT_ROUNDS = 12     # smallest size: LPs with reweighted costs at most (stops when nothing changes)
PATIENCE = 3             # ... with member adding, after this many rounds that found none lighter (Max size = Min size
                         # on the block: 165k candidates, rounds of 2-5 minutes that cycled without repeating)
DROP = 1e-3              # bars with less than this share of the largest force are dropped
STRAIGHT = np.cos(np.radians(178.0))   # two bars at a node this close to opposite: one bar through it
# With `cancel`, LPs over this many candidates run in a process of their own, ended at once on Stop: a
# solve can take minutes at a fine node spacing (139k candidates), and none can be interrupted in this
# process. (Starting the process takes ~2 s, as long as a whole LP of the wall bracket.) Only without `_Highs`.
APART = 30_000
TICK = 0.25              # s: a solve's callback looks at Stop and reports its iterations at most this often


class NoFrame(ConditionError):
    """No frame of the candidate bars carries the loads to the supports (the LP has no answer)."""


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
    least: float = 0.0      # mm^3 no truss of these candidates carrying the loads can do with less (`optimize`)

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


def _solve(B, cost_t, cost_c, f, cancel=None, upper=None):
    """(forces, equilibrium duals, volume) of the LP over the candidates in B, or None if it has no
    answer. `cancel`: called before (and, apart, during) the solve; raises when the run is stopped.
    `upper`: (tension, compression) the most each candidate may carry, N (Max size), or None."""
    m = B.shape[1]
    # interior point (with crossover to a vertex): 0.9 s on the wall bracket's 17k candidates, the
    # simplex methods 2.5 s, same optimum
    c = np.concatenate([cost_t, cost_c])
    bounds = (0, None) if upper is None else np.column_stack([np.zeros(2 * m), np.concatenate(upper)])
    options = dict(A_eq=sparse.hstack([B, -B]).tocsc(), b_eq=f, bounds=bounds, method="highs-ipm")
    if cancel is not None:
        cancel()
    res = _linprog_apart(c, options, cancel) if cancel is not None and m > APART else linprog(c, **options)
    if res.status == 2:
        return None
    if res.status != 0:
        raise ConditionError(f"the layout optimization failed: {res.message}")
    x = res.x
    return x[:m] - x[m:], np.asarray(res.eqlin.marginals), float(res.fun)


def _linprog_apart(c, options, cancel):
    """linprog in a process of its own (a pool of one), ended at once when `cancel` raises."""
    import multiprocessing
    pool = multiprocessing.get_context("spawn").Pool(1)
    try:
        job = pool.apply_async(linprog, (c,), options)
        while not job.ready():
            cancel()
            job.wait(0.2)
        return job.get()
    finally:
        pool.terminate()
        pool.join()


class _Model:
    """A layout's LP kept between solves: a row per free degree of freedom (its balance), a tension and a
    compression column per candidate bar taken in (`add`, never taken out: member adding only adds), costs
    changed in place (`costs`). `solve` gives (forces of all candidates, 0 for those not in, equilibrium duals,
    cost), or None when no frame of these bars carries the loads. `cancel()` and `tick(iteration)` are called
    while it solves (at most every TICK s); either may raise (Stop), which ends the solve at once.
    Without scipy's HiGHS bindings each solve is a `linprog` over the columns in."""

    def __init__(self, B_all, f, upper=None, cancel=None, tick=None):
        self.B, self.f, self.upper, self.cancel, self.tick = B_all, f, upper, cancel, tick
        self.col = np.full(B_all.shape[1], -1, dtype=np.int64)   # each bar's tension column (+1: compression)
        self.bars = np.zeros(0, dtype=np.int64)                  # the bars in, in column order
        self.ct = self.cc = None
        self.h = None
        self.iterations = 0      # of the last solve
        self.solves = 0
        self.label = "LP"        # what `tick` says is being solved
        if _Highs is not None:
            h = _Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("solver", "ipm")
            h.setOptionValue("run_crossover", "off")
            n = len(f)
            h.addRows(n, f, f, 0, np.zeros(n, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0))
            h.setCallback(self._callback, None)
            h.startCallback(_hc.cb.HighsCallbackType.kCallbackIpmInterrupt)
            self.h = h

    @property
    def taken(self):
        """Per candidate: in the LP."""
        return self.col >= 0

    def add(self, idx, ct, cc):
        """The bars `idx` not in yet, at costs per newton `ct`, `cc` (of all candidates)."""
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[self.col[idx] < 0]
        if not len(idx):
            return
        self.col[idx] = 2 * (len(self.bars) + np.arange(len(idx)))
        self.bars = np.concatenate([self.bars, idx])
        if self.h is None:
            return
        k = len(idx)
        sub = self.B[:, idx]
        both = sparse.hstack([sub, -sub]).tocsc()[:, np.column_stack([np.arange(k), k + np.arange(k)]).ravel()]
        if self.upper is None:
            upper = np.full(2 * k, _hc.kHighsInf)
        else:
            upper = np.column_stack([self.upper[0][idx], self.upper[1][idx]]).ravel().astype(np.float64)
        self.h.addCols(2 * k, np.column_stack([ct[idx], cc[idx]]).ravel().astype(np.float64), np.zeros(2 * k), upper,
                       both.nnz, both.indptr[:-1].astype(np.int32), both.indices.astype(np.int32),
                       both.data.astype(np.float64))

    def costs(self, ct, cc):
        """Costs per newton of all candidates (those in the LP take theirs now, later ones when added)."""
        self.ct, self.cc = ct, cc
        if self.h is None or not len(self.bars):
            return
        c = np.column_stack([ct[self.bars], cc[self.bars]]).ravel().astype(np.float64)
        self.h.changeColsCost(len(c), np.arange(len(c), dtype=np.int32), c)

    def solve(self):
        self.solves += 1
        if self.h is None:
            ct, cc = self.ct[self.bars], self.cc[self.bars]
            upper = None if self.upper is None else (self.upper[0][self.bars], self.upper[1][self.bars])
            out = _solve(self.B[:, self.bars], ct, cc, self.f, self.cancel, upper)
            if out is None:
                return None
            q = np.zeros(self.B.shape[1])
            q[self.bars] = out[0]
            return q, out[1], out[2]
        if self.cancel is not None:
            self.cancel()
        self.stopped, self.last, self.iterations = None, time.perf_counter(), 0
        status = self._run()
        if status == _hc.HighsModelStatus.kInfeasible:
            return None
        if status != _hc.HighsModelStatus.kOptimal:
            # (numerical trouble; not seen on frames yet) once more with crossover, then linprog
            self.h.setOptionValue("run_crossover", "on")
            try:
                status = self._run()
            finally:
                self.h.setOptionValue("run_crossover", "off")
            if status == _hc.HighsModelStatus.kInfeasible:
                return None
            if status != _hc.HighsModelStatus.kOptimal:
                h, self.h = self.h, None
                try:
                    return self.solve()
                finally:
                    self.h = h
        solution = self.h.getSolution()
        x = np.asarray(solution.col_value)
        q = np.zeros(self.B.shape[1])
        q[self.bars] = x[0::2] - x[1::2]
        return q, np.asarray(solution.row_dual), float(self.h.getInfo().objective_function_value)

    def _run(self):
        self.h.run()
        if self.stopped is not None:
            stopped, self.stopped = self.stopped, None
            raise stopped
        return self.h.getModelStatus()

    def _callback(self, kind, message, out, given, user):
        """HiGHS's interior point callback (called very often, also inside its iterations, with -1)."""
        self.iterations = max(self.iterations, int(out.ipm_iteration_count))
        now = time.perf_counter()
        if now - self.last < TICK or self.stopped is not None:
            return
        self.last = now
        try:
            if self.cancel is not None:
                self.cancel()
            if self.tick is not None:
                self.tick(self.iterations)
        except BaseException as err:   # (nothing may be raised through HiGHS: kept, raised once it has stopped)
            self.stopped = err
            given.user_interrupt = True


def optimize(ground, sigma_t, sigma_c=None, joint_length=0.0, member_adding=None, min_area=0.0, log=None,
             start=None, cancel=None, show=None, limit=None, tick=None):
    """The minimum-volume layout of a Ground structure for allowable stresses (MPa); with a smallest
    bar area (mm^2), the lightest layout once every bar is at least that thick. `start`: forces of
    a previous layout on these candidates, to begin the reweighting from (a refined ground).
    `cancel`: called between LP solves and while one runs; raises to stop. `show(what, forces, numbers)`:
    called with every LP's forces on all candidates and a dict of what it did (the live view: `lp`, `in_lp`,
    `candidates`, `adding`, `volume`, `real`: the volume with every bar at least the smallest size).
    `tick(text)`: called while an LP solves (its iterations). `limit(lengths)`: (tension, compression) the most
    a candidate of each length may carry, N (Max size), or None."""
    sigma_c = sigma_t if sigma_c is None else sigma_c
    nodes, bars, L = ground.nodes, ground.bars, ground.lengths
    upper = None if limit is None else limit(L)
    free = ~ground.fixed.ravel()
    B_all = equilibrium_matrix(nodes, bars).tocsr()[free].tocsc()
    f = ground.loads.ravel()[free]
    if not np.abs(f).max() > 0:
        raise ConditionError("the loads add up to nothing (or sit on the supports)")
    if member_adding is None:
        member_adding = len(bars) > FULL_LP
    cost = L + joint_length
    model = _Model(B_all, f, upper, cancel,
                   None if tick is None else (lambda it: tick(f"{model.label}, interior point iteration {it}")))
    real = lambda q: float((L * np.maximum(np.where(q > 0, q / sigma_t, -q / sigma_c), min_area)
                            * (np.abs(q) > DROP * np.abs(q).max())).sum())
    plastic = lambda q: float((L * np.where(q > 0, q / sigma_t, -q / sigma_c)).sum())
    report = None if show is None else (lambda what, q, **numbers: show(what, q, dict(
        numbers, lp=model.solves, in_lp=len(model.bars), candidates=len(bars), volume=plastic(q), real=real(q))))
    least = 0.0   # (not known: started from another layout's forces)
    if start is not None and min_area > 0:
        q, rounds = np.asarray(start, dtype=np.float64), 0
        model.costs(cost / sigma_t, cost / sigma_c)
        model.add(np.flatnonzero(q), cost / sigma_t, cost / sigma_c)
    else:
        q, rounds = _layout(ground, model, B_all, cost / sigma_t, cost / sigma_c, member_adding, log, report,
                            "the lightest layout")
        # its cost is the least of all; each bar's length is at least L / (L + joint length) of its cost, for the
        # shortest L (an LP without the joint length would find no less)
        least = float((cost * np.where(q > 0, q / sigma_t, -q / sigma_c)).sum()) * L.min() / (L.min() + joint_length)
        if report is not None:
            report("the lightest layout", q)
    if min_area > 0:
        best, best_q, used = real(q), q, np.abs(q) > DROP * np.abs(q).max()
        carry = min_area * min(sigma_t, sigma_c)    # the force the smallest bar carries
        seen = {used.tobytes()}   # it can cycle (the wall bracket: 12, 9, 9, 12, ... bars): stop on a repeat
        since = 0                 # rounds since the lightest
        for r in range(REWEIGHT_ROUNDS):
            ref = np.where(used, np.maximum(np.abs(q), 1e-6 * carry), carry)
            scale = np.maximum(1.0, carry / ref)
            # (member adding goes on from the bars the rounds before had, which the model keeps: from the shortest
            # bars again, every round asked for most of the same ones)
            what = f"bars at least Min size, round {r + 1}"
            q, _ = _layout(ground, model, B_all, cost / sigma_t * scale, cost / sigma_c * scale, member_adding, log,
                           report, what)
            now = np.abs(q) > DROP * np.abs(q).max()
            v = real(q)
            if log:
                log(f"min size round {r + 1}: {int(now.sum())} bars, volume with the smallest size {v:.6g}")
            if report is not None:
                report(what, q)
            if v < best - 1e-9 * best:
                best, best_q, since = v, q, 0
            else:
                since += 1
            if now.tobytes() in seen:
                break
            if member_adding and since >= PATIENCE:   # (big grounds only: each round there is minutes of LPs)
                break
            seen.add(now.tobytes())
            used = now
        q = best_q
    out = _clean(ground, q, plastic(q), len(model.bars), rounds)
    out.least = least
    return out


def _layout(ground, model, B_all, ct, cc, member_adding, log, report=None, what=""):
    """(forces of all candidates, rounds) of one minimum-cost layout, solved in `model` (`_Model`; it keeps the
    bars earlier layouts took in). `report(what, forces, **numbers)`: called after each member adding round.
    (A smallest size's round solved once, its bars to add left for the next round, was 8 % heavier at 2 mm spacing,
    19.2 g instead of 17.7: the 124 bars its second LP took in moved the layout's next round elsewhere.)"""
    bars, L = ground.bars, ground.lengths
    model.costs(ct, cc)
    if member_adding:
        active = L <= START_REACH * ground.spacing * (1 + 1e-9)
        face = np.flatnonzero(ground.owner >= 0)
        for node in face:
            mine = np.flatnonzero((bars[:, 0] == node) | (bars[:, 1] == node))
            active[mine[np.argsort(L[mine])[:START_PER_FACE_NODE]]] = True
    else:
        active = np.ones(len(bars), dtype=bool)
    model.add(np.flatnonzero(active & ~model.taken), ct, cc)
    rounds, stalled, before = 0, 0, np.inf
    while True:
        rounds += 1
        model.label = f"LP {model.solves + 1}" + (f", member adding round {rounds}" if member_adding else "")
        out = model.solve()
        if out is None:  # infeasible with these bars: allow longer ones
            taken = model.taken
            if taken.all():
                if model.upper is not None:
                    raise ConditionError("no frame of bars up to Max size inside the design space can carry the "
                                         "loads to the supports: a larger Max size, a lower Safety factor or a "
                                         "smaller node spacing (more bars to share the load) helps")
                raise NoFrame("no frame inside the design space can carry the loads to the supports")
            longer = L <= max(L[taken].max() * 1.5, ground.spacing * 2.5)
            more = longer if longer.sum() > taken.sum() else np.ones_like(taken)
            model.add(np.flatnonzero(more & ~taken), ct, cc)
            continue
        q, y, volume = out
        if not member_adding or rounds >= MAX_ROUNDS:
            break
        # (the duals of a degenerate optimum keep naming bars that lower nothing: at 2 mm spacing the block's
        # volume was 2616.53 from round 3 on while ~1000 bars a round came in, 30 s an LP, for up to 60 rounds)
        stalled = stalled + 1 if volume >= before * (1 - ADD_STALL) else 0
        before = min(before, volume)
        in_lp = len(model.bars)
        if stalled >= STALL_ROUNDS:
            if log:
                log(f"member adding round {rounds}: {in_lp:,} bars, volume {volume:.6g}: no longer falling, stopped")
            break
        # virtual strain of every bar from the duals: bars outside that break the bounds come in
        v = B_all.T @ y
        excess = np.maximum(v / ct, -v / cc)
        excess[model.taken] = 0.0
        add = np.flatnonzero(excess > 1 + ADD_TOL)
        if log:
            log(f"member adding round {rounds}: {in_lp:,} bars, volume {volume:.6g}, {len(add)} to add")
        if not len(add):
            break
        limit = max(1000, in_lp)
        if len(add) > limit:
            add = add[np.argsort(-excess[add])[:limit]]
        if report is not None:
            report(f"{what}: member adding round {rounds}", q, adding=len(add))
        model.add(add, ct, cc)
    return q, rounds


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
    moved = np.zeros(n, dtype=bool)   # joints moved off their ground node (`joints.move_joints`): ground
    moved[layout.ground_index] = np.abs(layout.nodes - ground.nodes[layout.ground_index]).max(axis=1) > 1e-9
    b, L = ground.bars, ground.lengths   # bars to them may not lie inside, no path goes through them
    b, L = b[~moved[b].any(axis=1)], L[~moved[b].any(axis=1)]
    graph = coo_matrix((np.concatenate([L, L]), (np.concatenate([b[:, 0], b[:, 1]]), np.concatenate([b[:, 1], b[:, 0]]))),
                       shape=(n, n)).tocsr()
    in_frame = np.zeros(n, dtype=bool)
    in_frame[layout.ground_index] = True
    targets = in_frame & ~ground.fixed.any(axis=1) & ~moved
    if not targets.any():
        targets = in_frame & ~moved
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
    nodes[remap[layout.ground_index]] = layout.nodes   # (moved joints stay where they are)
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
