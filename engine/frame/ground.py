"""Ground structure: the candidate nodes and bars a frame's layout is chosen from.

Nodes: a regular lattice inside the design space (deep enough for a bar of the smallest size),
plus nodes on the faces of every support and load: the face is sampled and clustered (k-means)
into 1-6 nodes, each at the sample nearest its cluster's centre, so it lies on the face itself (a
bolt hole's wall, a pin's contact side). A load's nodes carry the sum of its traction over their
cluster (the total is exact, as in the FE model); a support's nodes are held in x, y and z.

Bars: every pair of nodes within *reach* node spacings whose segment stays inside the design
space (its distance field sampled along the bar, at least half the smallest size deep, except
within half a spacing of a node on a face, where bars start at the surface). A bar between lattice
nodes that passes through another lattice node (its step has a common divisor) is left out: the
layout chains the short ones instead, unless *overlap* asks for them (joint costs).
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from ..fea.model import ConditionError, sample_triangles
from ..fields import Context
from ..grid import Grid

AUTO_NODES = 15          # automatic spacing: the design space's diagonal / this
MAX_LATTICE = 30_000     # lattice nodes at most (a finer spacing: an error that says so)
FACE_NODES = (1, 6)      # nodes per support or load face, by its area
TURNING_NODES = 4        # at least this many for a torque or an offset force (they carry a moment)
PARALLEL = 0.9           # a load whose resultant is less than this share of its pushes: one node
ROUND = 0.5              # faces whose mean normal is shorter than this (a hole): one node at the axis
DEPTH = 0.5              # bar centres at least this x the smallest size inside the design space


@dataclass
class Ground:
    nodes: np.ndarray         # (n, 3) mm
    bars: np.ndarray          # (m, 2) node indices, first < second
    lengths: np.ndarray       # (m,) mm
    fixed: np.ndarray         # (n, 3) bool: held in that direction
    loads: np.ndarray         # (n, 3) N
    spacing: float            # mm
    owner: np.ndarray         # (n,) the condition a node belongs to (-1: lattice)
    notes: list = field(default_factory=list)
    free: np.ndarray = None   # (n,) mm around a node where its bars may leave the design space
    inside: object = None     # points -> the design space's signed distance (curves are checked with it)

    @property
    def supports(self):
        return np.flatnonzero(self.fixed.any(axis=1))


def tidy(x):
    """Two significant digits (6.27 -> 6.3)."""
    step = 10 ** (math.floor(math.log10(x)) - 1)
    return round(round(x / step) * step, 10)


class _Inside:
    """Signed distance of the design space anywhere in its box (trilinear on a sampled grid)."""

    def __init__(self, space, step, cache=None):
        lo, hi = (np.asarray(b, dtype=np.float64) for b in space.bounds)
        self.grid = Grid(lo - 2 * step, hi + 2 * step, step)
        self.values = np.asarray(Context(self.grid, cache, np).eval_full(space), dtype=np.float32)
        self.step = step

    def __call__(self, points):
        coords = (np.asarray(points, dtype=np.float64) - self.grid.origin) / self.step
        return ndimage.map_coordinates(self.values, coords.T, order=1, mode="nearest")


def _pieces(triangles):
    """Connected pieces of a set of triangles (sharing a corner): one per bolt hole of a support."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    t = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    _, corner = np.unique(np.round(t.reshape(-1, 3), 6), axis=0, return_inverse=True)
    k, v = len(t), int(corner.max()) + 1
    rows = np.repeat(np.arange(k), 3)
    graph = coo_matrix((np.ones(3 * k), (rows, corner.ravel() + k)), shape=(k + v, k + v))
    _, label = connected_components(graph, directed=False)
    label = label[:k]
    return [t[label == c] for c in np.unique(label)]


def _face_nodes(cond, spacing, rng):
    """(positions, forces, free radii) of a support or load: per connected piece of its faces."""
    out = [_piece_nodes(cond, tri, spacing, rng) for tri in _pieces(cond.triangles)]
    if not out:
        raise ConditionError(f"{cond.label or cond.kind}: no faces chosen")
    return tuple(np.concatenate(a) for a in zip(*out))


def _piece_nodes(cond, triangles, spacing, rng):
    """(positions (k, 3), forces (k, 3), free radius (k,)) of one piece of a support or load. Bars
    may leave the design space within a node's free radius (they start at its surface)."""
    pts, areas, normals = sample_triangles(triangles, spacing / 4)
    if not len(pts):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    force = cond.traction(normals, pts) * areas[:, None] if cond.kind == "load" else np.zeros_like(pts)
    turning = cond.kind == "load" and (cond.load == "torque" or cond.offset_moment() is not None)
    # a hole or shaft (faces facing all ways), and a load whose pushes partly cancel (a pin in a
    # hole), act on the frame at one node in the middle, as a bolt or pin does: split over nodes on
    # a hole's wall, the sideways pushes had to be carried across by a nest of extra bars
    round_face = np.linalg.norm((normals * areas[:, None]).sum(axis=0)) < ROUND * areas.sum()
    cancelling = cond.kind == "load" and np.linalg.norm(force.sum(axis=0)) < PARALLEL * np.linalg.norm(force, axis=1).sum()
    if not turning and (round_face or cancelling):
        centre = (pts * areas[:, None]).sum(axis=0) / areas.sum()
        reach = float(np.linalg.norm(pts - centre, axis=1).max())
        return centre[None], force.sum(axis=0)[None], np.array([reach + spacing / 2])
    area = float(areas.sum())
    k = int(np.clip(round(area / (2 * spacing ** 2)), *FACE_NODES))
    if turning:
        k = max(k, TURNING_NODES)
    k = min(k, len(pts))
    if k > 1:
        _, label = kmeans2(pts, k, minit="++", rng=rng)
    else:
        label = np.zeros(len(pts), dtype=int)
    nodes, forces = [], []
    for j in np.unique(label):
        mine = label == j
        centre = (pts[mine] * areas[mine, None]).sum(axis=0) / areas[mine].sum()
        nodes.append(pts[mine][np.argmin(np.linalg.norm(pts[mine] - centre, axis=1))])
        forces.append(force[mine].sum(axis=0))
    return np.asarray(nodes), np.asarray(forces), np.full(len(nodes), spacing / 2)


def build(space, conditions, spacing=0.0, reach=3.0, min_size=2.0, overlap=False, cache=None, seed=0):
    """The ground structure of a design space (implicit body) with supports and loads."""
    if space.bounds is None:
        raise ConditionError("the design space has no size of its own: fill a body first")
    lo, hi = (np.asarray(b, dtype=np.float64) for b in space.bounds)
    diagonal = float(np.linalg.norm(hi - lo))
    spacing = float(spacing) if spacing > 0 else tidy(diagonal / AUTO_NODES)
    if reach < 1:
        raise ConditionError("reach must be at least 1 node spacing")
    notes = []
    inside = _Inside(space, min(spacing / 3, max(min_size, spacing / 8)), cache)

    # the lattice, centred in the design space's box
    count = np.floor((hi - lo) / spacing).astype(int) + 1
    if int(np.prod(count)) > 8 * MAX_LATTICE:
        raise ConditionError(f"node spacing {spacing:g} mm gives too many nodes here: use a larger spacing")
    start = lo + ((hi - lo) - (count - 1) * spacing) / 2
    ijk = np.stack(np.meshgrid(*[np.arange(c) for c in count], indexing="ij"), axis=-1).reshape(-1, 3)
    lattice = start + ijk * spacing
    keep = inside(lattice) <= -DEPTH * min_size
    ijk, lattice = ijk[keep], lattice[keep]
    if len(lattice) > MAX_LATTICE:
        raise ConditionError(f"node spacing {spacing:g} mm gives {len(lattice):,} nodes (at most {MAX_LATTICE:,}): "
                             f"use a larger spacing")

    # nodes on the faces of supports and loads
    rng = np.random.default_rng(seed)
    nodes, owner, fixed, loads = [lattice], [np.full(len(lattice), -1)], [np.zeros((len(lattice), 3), bool)], \
        [np.zeros((len(lattice), 3))]
    free = [np.zeros(len(lattice))]
    if not any(c.kind == "fixed" for c in conditions):
        raise ConditionError("the part is not held anywhere: add Fixed Faces")
    if not any(c.kind == "load" for c in conditions):
        raise ConditionError("there is no load: add a Load")
    for ci, cond in enumerate(conditions):
        pos, force, radius = _face_nodes(cond, spacing, rng)
        nodes.append(pos)
        owner.append(np.full(len(pos), ci))
        fixed.append(np.full((len(pos), 3), cond.kind == "fixed"))
        loads.append(force)
        free.append(radius)
    nodes, owner = np.concatenate(nodes), np.concatenate(owner)
    fixed, loads, free = np.concatenate(fixed), np.concatenate(loads), np.concatenate(free)
    # face nodes at one point (two loads on one pin, a load on a held face): one node
    n_lat = len(lattice)
    _, first, which = np.unique(np.round(nodes[n_lat:], 6), axis=0, return_index=True, return_inverse=True)
    if len(first) < len(nodes) - n_lat:
        which = which.ravel()
        merged = np.zeros((len(first), 3))
        np.add.at(merged, which, loads[n_lat:])
        held = np.zeros((len(first), 3), dtype=bool)
        np.logical_or.at(held, which, fixed[n_lat:])
        wide = np.zeros(len(first))
        np.maximum.at(wide, which, free[n_lat:])
        nodes = np.concatenate([lattice, nodes[n_lat:][first]])
        owner = np.concatenate([owner[:n_lat], owner[n_lat:][first]])
        fixed, loads = np.concatenate([fixed[:n_lat], held]), np.concatenate([loads[:n_lat], merged])
        free = np.concatenate([free[:n_lat], wide])
    loads[fixed] = 0.0  # a load on a held node goes straight into the support

    # candidate bars: within reach, and from every node at a support or load to every other node
    # (a bolt's or pin's node is off the lattice: limited to the reach, a straight bar from a bolt
    # to a pin could only be imitated by zig-zag chains)
    pairs = np.array(sorted(cKDTree(nodes).query_pairs(reach * spacing * (1 + 1e-9))), dtype=np.int64).reshape(-1, 2)
    face = np.flatnonzero(owner >= 0)
    i, j = np.repeat(face, len(nodes)), np.tile(np.arange(len(nodes)), len(face))
    longer = np.sort(np.column_stack([i, j])[i != j], axis=1)
    pairs = np.unique(np.concatenate([pairs, longer]), axis=0)
    a, b = pairs[:, 0], pairs[:, 1]
    keep = ~((owner[a] >= 0) & (owner[a] == owner[b]))          # not across one face (a hole)
    n_lat = len(lattice)
    both = (a < n_lat) & (b < n_lat)
    if not overlap:
        step = np.abs(ijk[np.minimum(a, n_lat - 1)] - ijk[np.minimum(b, n_lat - 1)])
        keep &= ~both | (np.gcd.reduce(step, axis=1) == 1)
    pairs = pairs[keep]
    pairs = pairs[_stays_inside(nodes, pairs, free, inside, min_size)]
    if not len(pairs):
        raise ConditionError("no bar fits inside the design space: use a smaller node spacing or min size")
    lengths = np.linalg.norm(nodes[pairs[:, 1]] - nodes[pairs[:, 0]], axis=1)

    # nodes without bars go; a load or support without bars is an error
    used = np.zeros(len(nodes), dtype=bool)
    used[pairs.ravel()] = True
    lonely = ~used & (owner >= 0)
    if lonely.any():
        c = conditions[int(owner[np.flatnonzero(lonely)[0]])]
        raise ConditionError(f"{c.label or c.kind}: no bar reaches it inside the design space (try a smaller node "
                             f"spacing or min size)")
    remap = np.cumsum(used) - 1
    ground = Ground(nodes[used], remap[pairs], lengths, fixed[used], loads[used], spacing, owner[used], notes,
                    free[used], inside)
    notes.append(f"{len(ground.nodes):,} candidate nodes {spacing:g} mm apart, {len(ground.bars):,} candidate bars")
    return ground


def _stays_inside(nodes, pairs, free, inside, min_size, chunk=200_000):
    """Which candidate bars stay at least DEPTH x min size inside the design space (not counting the
    stretch within a face node's free radius, where a bar leaves the surface)."""
    ok = np.ones(len(pairs), dtype=bool)
    ds = inside.step / 2
    for s in range(0, len(pairs), chunk):
        p = pairs[s:s + chunk]
        a, b = nodes[p[:, 0]], nodes[p[:, 1]]
        L = np.linalg.norm(b - a, axis=1)
        n = np.maximum(2, np.ceil(L / ds).astype(np.int64) + 1)
        bar = np.repeat(np.arange(len(p)), n)
        first = np.repeat(np.cumsum(n) - n, n)
        t = (np.arange(len(bar)) - first) / (n[bar] - 1)
        pts = a[bar] + t[:, None] * (b - a)[bar]
        d = inside(pts)
        along = t * L[bar]
        loose = (along <= free[p[:, 0]][bar]) | (L[bar] - along <= free[p[:, 1]][bar])
        bad = (d > -DEPTH * min_size) & ~loose
        ok[s:s + chunk] = ~np.bincount(bar[bad], minlength=len(p)).astype(bool)
    return ok
