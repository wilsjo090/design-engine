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
    attach: np.ndarray = None  # (n,) bool: held because it is on a frame or body this one connects to

    @property
    def attached(self):
        return np.zeros(len(self.nodes), dtype=bool) if self.attach is None else self.attach

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
        values = np.asarray(Context(self.grid, cache, np).eval_full(space), dtype=np.float32)
        # Several design spaces are one union: where they only touch, the union's field (the smaller
        # of theirs) is 0 along the shared face, as if it were a surface, and every bar across it
        # was refused. Where the field and the distance from the voxels inside disagree by more
        # than a voxel or two (only at such seams), take the voxels' distance.
        inside = values < 0
        voxels = (np.where(inside, -(ndimage.distance_transform_edt(inside) - 0.5),
                           ndimage.distance_transform_edt(~inside) - 0.5) * step).astype(np.float32)
        self.values = np.where(np.abs(voxels - values) > 2 * step, voxels, values)
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


def _attach_nodes(item, spacing, inside, rng):
    """(positions, free radii) where this frame may connect to another frame or body: the other
    frame's nodes and points along its bars, or points on the body's surface (clustered), all within
    a node spacing of this design space."""
    if hasattr(item, "bars") and hasattr(item, "chains"):          # a frame
        nodes, bars = np.asarray(item.nodes, dtype=np.float64), np.asarray(item.bars)
        width = np.asarray(item.sizes, dtype=np.float64)[np.asarray(item.chains)]
        pts, rad = [nodes], [np.zeros(len(nodes))]
        for (a, b), w in zip(bars, width):
            m = int(np.floor(np.linalg.norm(nodes[b] - nodes[a]) / (spacing / 2)))
            if m >= 2:
                pts.append(nodes[a] + np.arange(1, m)[:, None] / m * (nodes[b] - nodes[a]))
                rad.append(np.full(m - 1, w / 2))
        r_node = np.zeros(len(nodes))
        np.maximum.at(r_node, bars.ravel(), np.repeat(width / 2, 2))
        rad[0] = r_node
        pts, rad = np.concatenate(pts), np.concatenate(rad)
        keep = inside(pts) <= spacing
        return pts[keep], rad[keep] + spacing / 2
    tri = np.asarray(getattr(item, "triangles", item), dtype=np.float64).reshape(-1, 3, 3)   # a body
    pts, areas, _ = sample_triangles(tri, spacing / 4)
    keep = inside(pts) <= spacing
    pts, areas = pts[keep], areas[keep]
    if not len(pts):
        return np.zeros((0, 3)), np.zeros(0)
    k = int(np.clip(round(areas.sum() / (2 * spacing ** 2)), 1, 12))
    k = min(k, len(pts))
    label = kmeans2(pts, k, minit="++", rng=rng)[1] if k > 1 else np.zeros(len(pts), dtype=int)
    out = []
    for j in np.unique(label):
        mine = label == j
        centre = (pts[mine] * areas[mine, None]).sum(axis=0) / areas[mine].sum()
        out.append(pts[mine][np.argmin(np.linalg.norm(pts[mine] - centre, axis=1))])
    return np.asarray(out), np.full(len(out), spacing / 2)


def build(space, conditions, spacing=0.0, reach=3.0, min_size=2.0, overlap=False, cache=None, seed=0, attach=()):
    """The ground structure of a design space (implicit body) with supports and loads. `attach`:
    frames or CAD bodies this frame connects to (their bars or surface hold it, like a support)."""
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
    if not any(c.kind == "fixed" for c in conditions) and not attach:
        raise ConditionError("the part is not held anywhere: add Fixed Faces (or Connect to a frame or body)")
    if not any(c.kind == "load" for c in conditions):
        raise ConditionError("there is no load: add a Load")
    for ci, cond in enumerate(conditions):
        pos, force, radius = _face_nodes(cond, spacing, rng)
        nodes.append(pos)
        owner.append(np.full(len(pos), ci))
        fixed.append(np.full((len(pos), 3), cond.kind == "fixed"))
        loads.append(force)
        free.append(radius)
    for ai, item in enumerate(attach):
        pos, radius = _attach_nodes(item, spacing, inside, rng)
        if not len(pos):
            notes.append(f"connect to #{ai + 1}: nothing of it lies within {spacing:g} mm of the design space")
        nodes.append(pos)
        owner.append(np.full(len(pos), len(conditions) + ai))
        fixed.append(np.ones((len(pos), 3), dtype=bool))
        loads.append(np.zeros((len(pos), 3)))
        free.append(radius)
    nodes, owner = np.concatenate(nodes), np.concatenate(owner)
    fixed, loads, free = np.concatenate(fixed), np.concatenate(loads), np.concatenate(free)
    if not fixed.any():
        raise ConditionError("nothing holds the part: no Fixed Faces, and what it connects to is not near its design "
                             "space")
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
    face = np.flatnonzero((owner >= 0) & (owner < len(conditions)))   # (connection points: within reach only)
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
    lonely = ~used & (owner >= 0) & (owner < len(conditions))
    if lonely.any():
        c = conditions[int(owner[np.flatnonzero(lonely)[0]])]
        raise ConditionError(f"{c.label or c.kind}: no bar reaches it inside the design space (try a smaller node "
                             f"spacing or min size)")
    remap = np.cumsum(used) - 1
    ground = Ground(nodes[used], remap[pairs], lengths, fixed[used], loads[used], spacing, owner[used], notes,
                    free[used], inside, (owner >= len(conditions))[used])
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


def refine(ground, layout, step, reach, min_size):
    """The ground structure with nodes added along every bar the layout uses (about `step` mm
    apart) and candidate bars from them, so a bar can land part way along another one (a mid-bar
    joint): the next layout chooses where. Each used bar also comes as the pieces between its new
    nodes, so it can run straight on through a joint."""
    used = layout.ground_index[layout.bars]               # (k, 2) ground node indices
    n0 = len(ground.nodes)
    new_nodes, pieces, host = [], [], []   # host: the nodes along each refined bar
    count = n0
    existing = cKDTree(ground.nodes)
    for a, b in used:
        A, B = ground.nodes[a], ground.nodes[b]
        m = int(np.floor(np.linalg.norm(B - A) / step))
        if m < 2:
            continue
        pts = A + np.arange(1, m)[:, None] / m * (B - A)
        # not on top of a node already there (a long bar over lattice nodes: zero-length bars)
        pts = pts[existing.query(pts)[0] > 0.3 * step]
        if not len(pts):
            continue
        idx = list(range(count, count + len(pts)))
        new_nodes.append(pts)
        count += len(pts)
        chain = [int(a)] + idx + [int(b)]
        pieces += list(zip(chain[:-1], chain[1:]))
        host.append(set(chain))
    if not new_nodes:
        return ground
    added = np.concatenate(new_nodes)
    nodes = np.concatenate([ground.nodes, added])
    m_new = len(added)
    # bars from the new nodes to the frame's own nodes, the new nodes of other bars and the support
    # and load nodes within twice the reach, not along their own bar (to every node within reach:
    # 18,000 more candidates and three times the time, for the same frames on the wall brackets)
    own = {}
    for h in host:
        for v in h:
            own.setdefault(v, set()).update(h)
    targets = np.unique(np.concatenate([np.arange(n0, len(nodes)), used.ravel(), np.flatnonzero(ground.owner >= 0)]))
    near = cKDTree(nodes[targets])
    pairs = set()
    for i in range(n0, len(nodes)):
        for x in near.query_ball_point(nodes[i], 2 * reach * ground.spacing):
            j = int(targets[x])
            if j != i and j not in own.get(i, ()):
                pairs.add((min(i, j), max(i, j)))
    pairs = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    free = np.concatenate([ground.free if ground.free is not None else np.zeros(n0), np.zeros(m_new)])
    if len(pairs):
        pairs = pairs[_stays_inside(nodes, pairs, free, ground.inside, min_size)]
    bars = np.unique(np.concatenate([ground.bars, np.asarray(pieces, dtype=np.int64).reshape(-1, 2), pairs]), axis=0)
    lengths = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    bars, lengths = bars[lengths > 1e-6 * ground.spacing], lengths[lengths > 1e-6 * ground.spacing]
    notes = list(ground.notes) + [f"{m_new} nodes added along the bars used, {len(bars) - len(ground.bars):,} more "
                                  f"candidate bars (mid-bar joints)"]
    return Ground(nodes, bars, lengths, np.concatenate([ground.fixed, np.zeros((m_new, 3), dtype=bool)]),
                  np.concatenate([ground.loads, np.zeros((m_new, 3))]), ground.spacing,
                  np.concatenate([ground.owner, np.full(m_new, -1)]), notes, free, ground.inside,
                  np.concatenate([ground.attached, np.zeros(m_new, dtype=bool)]))
