"""Curved bars and the orientation of bars along their length.

Curving: the layout's bars are straight and meet at kinks. At a node that nothing holds or loads,
the two bars that go on most nearly straight (turning by at most MAX_TURN) are linked; linked bars
form a path from one end to the other (a bolt, a joint where the path stops, the pin), and the
whole path becomes one smooth curved bar: a cubic Hermite spline through every node of the path
(so bars that land on it at a joint still meet it there), with one tangent direction per node
(the bisector of the two chords at an inner node; at the ends the direction that makes the end
span a parabola). *curving* (0..1) scales the tangents: 1 an even curve all along the bar, smaller
values keep it closer to the straight bars; the tangent's direction is the same on both sides of
a node, so there is never a kink. Where a curve would leave the design space it is made flatter,
else the path stays straight. A curve is a polyline of short pieces (the frame check and the body
use them) and one continuous bar of one size.

(First version: only a short quadratic stretch around each bend, straight elsewhere. It looked
unnatural: a straight bar, a tight local bend, straight again.)

Orientation: a bar's local z (its profile's strong axis) is world Z made square to the bar, and
along a continuous bar it is carried from piece to piece (a rotation-minimising frame), so a
square tube does not twist around a curve.
"""
import math

import numpy as np

MAX_TURN = math.radians(70.0)    # sharper bends stay corners (a curve round them looks bent double)
STEP = math.radians(4.0)          # one piece per this much turning


def _hermite(P0, P1, m0, m1, s):
    s = s[:, None]
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * P0 + (s ** 3 - 2 * s ** 2 + s) * m0
            + (-2 * s ** 3 + 3 * s ** 2) * P1 + (s ** 3 - s ** 2) * m1)


def _angle(u, v):
    return math.acos(float(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)))


def smooth_bars(nodes, bars, forces, chains, fixed, loads, curving, inside=None, depth=0.0, free=None):
    """(nodes, bars, forces, chains, fixed, loads) with each path of linked bars made one smooth
    curved bar. `inside(points)`: the design space's signed distance, curves stay `depth` inside,
    except within `free[node]` mm of a node (a bolt's or pin's node sits in its hole: the bars there
    cross the hole, as in the ground structure; checking it there dropped every curve to the pin)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    bars = np.asarray(bars)
    if curving <= 0 or not len(bars):
        return nodes, bars, forces, chains, fixed, loads
    n = len(nodes)
    held = fixed.any(axis=1) | (np.abs(loads).sum(axis=1) > 0)
    at = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bars):
        at[a].append(i)
        at[b].append(i)
    other = lambda i, v: int(bars[i, 0] if bars[i, 1] == v else bars[i, 1])

    # links: at each free node, the pair of bars that goes on most nearly straight
    link = {}
    for v in range(n):
        if held[v] or len(at[v]) < 2:
            continue
        pairs = [(math.pi - _angle(nodes[other(i, v)] - nodes[v], nodes[other(j, v)] - nodes[v]), i, j)
                 for x, i in enumerate(at[v]) for j in at[v][x + 1:]]
        turn, i, j = min(pairs)
        if turn <= MAX_TURN:
            link[(i, v)], link[(j, v)] = j, i

    # paths of linked bars, walked from an end
    seen = np.zeros(len(bars), dtype=bool)
    paths = []
    for first in range(len(bars)):
        if seen[first]:
            continue
        # back to the start of this path (a free end, or round a loop to the first bar)
        bar, node = first, int(bars[first, 0])
        for _ in range(len(bars)):
            nxt = link.get((bar, node))
            if nxt is None or nxt == first:
                break
            bar, node = nxt, other(nxt, node)
        start_bar, start_node = bar, node
        seq_nodes, seq_bars = [start_node], []
        bar, node = start_bar, start_node
        while bar is not None and not seen[bar]:
            seen[bar] = True
            seq_bars.append(bar)
            node = other(bar, node)
            seq_nodes.append(node)
            bar = link.get((bar, node))
        paths.append((seq_nodes, seq_bars))

    new_nodes = [nodes]
    count = n
    pieces, piece_force, piece_chain = [], [], []
    chain_of = np.asarray(chains).copy()
    next_chain = int(chain_of.max()) + 1 if len(chain_of) else 0
    for seq_nodes, seq_bars in paths:
        P = nodes[seq_nodes]
        if len(seq_bars) < 2:
            pieces.append(tuple(bars[seq_bars[0]]))
            piece_force.append(forces[seq_bars[0]])
            piece_chain.append(int(chain_of[seq_bars[0]]))
            continue
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        u = np.diff(P, axis=0) / d[:, None]
        T = np.zeros_like(P)
        T[1:-1] = u[:-1] + u[1:]
        T[1:-1] /= np.maximum(np.linalg.norm(T[1:-1], axis=1, keepdims=True), 1e-12)
        T[0], T[-1] = 2 * u[0] - T[1], 2 * u[-1] - T[-2]
        T[[0, -1]] /= np.linalg.norm(T[[0, -1]], axis=1, keepdims=True)
        bends = max(_angle(T[k], T[k + 1]) for k in range(len(d)))
        spans = None
        c = float(curving)
        if bends > math.radians(1.0):
            for _ in range(4):  # flatter where the curve would leave the design space
                trial = []
                for k in range(len(d)):
                    m = max(1, min(24, math.ceil(_angle(T[k], T[k + 1]) / STEP) + 1))
                    s = np.linspace(0.0, 1.0, m + 1)[1:-1]
                    trial.append(_hermite(P[k], P[k + 1], c * d[k] * T[k], c * d[k] * T[k + 1], s))
                pts = np.concatenate([t for t in trial if len(t)]) if any(len(t) for t in trial) else np.zeros((0, 3))
                ok = inside is None or not len(pts)
                if not ok:
                    out = inside(pts) > -depth
                    if free is not None and out.any():
                        for v in (seq_nodes[0], seq_nodes[-1]):
                            out &= np.linalg.norm(pts - nodes[v], axis=1) > free[v]
                    ok = not out.any()
                if ok:
                    spans = trial
                    break
                c /= 2
        if spans is None:  # straight (no bend, or no curve fits): the bars as they are
            for i in seq_bars:
                pieces.append(tuple(bars[i]))
                piece_force.append(forces[i])
                piece_chain.append(int(chain_of[i]))
            continue
        cid = next_chain
        next_chain += 1
        for k, (i, inner) in enumerate(zip(seq_bars, spans)):
            idx = list(range(count, count + len(inner)))
            new_nodes.append(inner)
            count += len(inner)
            chain = [seq_nodes[k]] + idx + [seq_nodes[k + 1]]
            for a, b in zip(chain[:-1], chain[1:]):
                pieces.append((a, b))
                piece_force.append(forces[i])
                piece_chain.append(cid)
    nodes = np.concatenate(new_nodes)
    extra = len(nodes) - n
    fixed = np.concatenate([fixed, np.zeros((extra, 3), dtype=bool)])
    loads = np.concatenate([loads, np.zeros((extra, 3))])
    return (nodes, np.asarray(pieces, dtype=np.int64), np.asarray(piece_force, dtype=np.float64),
            np.unique(np.asarray(piece_chain), return_inverse=True)[1], fixed, loads)


def orientation(nodes, bars, chains, up=(0.0, 0.0, 1.0)):
    """(z axes (k, 3), forward (k,) bool) per piece: its local z, carried along each continuous bar
    from its first piece (world `up` made square to it; world X for a bar along `up`), and whether
    the piece's (a, b) runs the way the bar is walked (a bar's profile is drawn in walk order)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = np.zeros((len(bars), 3))
    forward = np.ones(len(bars), dtype=bool)
    for c in np.unique(chains):
        mine = [int(p) for p in np.flatnonzero(chains == c)]
        ends = np.bincount(bars[mine].ravel(), minlength=len(nodes))
        start = next((int(v) for v in bars[mine].ravel() if ends[v] == 1), int(bars[mine[0], 0]))
        node, left, ref = start, set(mine), None
        while left:
            p = next((q for q in left if node in bars[q]), None)
            if p is None:  # (a branch: start again from any piece left)
                p = next(iter(left))
                node = int(bars[p, 0])
            left.discard(p)
            a, b = int(bars[p, 0]), int(bars[p, 1])
            forward[p] = a == node
            nxt = b if forward[p] else a
            ex = nodes[nxt] - nodes[node]
            ex /= np.linalg.norm(ex)
            if ref is None:
                ref = up if abs(ex @ up) < 0.999 else np.array([1.0, 0.0, 0.0])
            ez = ref - (ref @ ex) * ex
            if np.linalg.norm(ez) < 1e-9:
                ez = np.array([1.0, 0.0, 0.0]) - ex[0] * ex
            ez /= np.linalg.norm(ez)
            z[p], ref = ez, ez
            node = nxt
    return z, forward
