"""Ground structure: the candidate nodes and bars a frame's layout is chosen from.

Nodes: a regular lattice inside the design space (deep enough for a bar of the smallest size; an
automatic spacing gets finer while a support or load is left without bars), plus nodes on the faces of every support and load: the face is sampled and clustered (k-means)
into 1-6 nodes, each at the sample nearest its cluster's centre, so it lies on the face itself (a
bolt hole's wall, a pin's contact side). A load's nodes carry the sum of its traction over their
cluster (the total is exact, as in the FE model); a support's nodes are held in x, y and z.

Bars: every pair of nodes within *reach* node spacings whose segment stays inside the design
space (its distance field sampled along the bar, at least half the smallest size deep, except
within half a spacing of a node on a face, where bars start at the surface). A bar between lattice
nodes that passes through another lattice node (its step has a common divisor) is left out: the
layout chains the short ones instead, unless *overlap* asks for them (joint costs).
"""
import copy
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
FINER = 0.8              # ... this x finer while a support or load gets no bar
AUTO_FLOOR = 1 / 3       # ... down to this share of it
MAX_LATTICE = 30_000     # lattice nodes at most (a finer spacing: an error that says so)
FACE_NODES = (1, 6)      # nodes per support or load face, by its area
TURNING_NODES = 4        # at least this many for a torque or an offset force (they carry a moment)
PARALLEL = 0.9           # a load whose resultant is less than this share of its pushes: one node
ROUND = 0.5              # faces whose mean normal is shorter than this (a hole): one node at the axis
AROUND = (3, 6)          # nodes around a hole spread out (`_piece_nodes` "around"), by the area it acts on
SET_IN = 0.25            # ... set this x the spacing into the material (on the wall, bars had to skim it)
DEPTH = 0.5              # bar centres at least this x the smallest size inside the design space
# bars from a support's or load's node may leave the design space within half a node spacing of it, but at
# least this x the smallest size: a bar starting on a face reaches DEPTH x min size inside only that far in
# (a 5 mm cube's automatic 0.58 mm spacing reached none of its faces, and finer spacings were worse)
FREE_SIZE = 0.75
MAX_INSIDE = 16_000_000  # samples of the design space's distance at most (`_Inside`)
SLENDER = 40             # all nodes on a line in a design space this many times as long as thick: no frame (`_across`)


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
    piece: np.ndarray = None  # (n,) the piece of faces (a hole, a face) a node is on (-1: none)
    coarser: float = None     # the automatic spacing that left a support or load without bars (this one is finer)

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
    """Signed distance of the design space anywhere in its box (trilinear on a sampled grid; outside the
    grid, its edge value plus the distance to it)."""

    def __init__(self, space, step, cache=None):
        lo, hi = (np.asarray(b, dtype=np.float64) for b in space.bounds)
        # (a typed node spacing sets the step: at most MAX_INSIDE samples, a 0.01 mm spacing asked for 341 GB)
        step = max(step, float(np.prod(hi - lo + 4 * step) / MAX_INSIDE) ** (1 / 3))
        self.space, self.cache, self._thickest = space, cache, None
        self.grid = Grid(lo - 2 * step, hi + 2 * step, step)
        values = np.asarray(Context(self.grid, cache, np).eval_full(space), dtype=np.float32)
        # Several design spaces are one union: where they only touch, the union's field (the smaller
        # of theirs) is 0 along the shared face, as if it were a surface, and every bar across it
        # was refused. Where the field and the distance from the voxels inside disagree by more
        # than a voxel or two (only at such seams), take the voxels' distance.
        inside = values < 0
        if inside.any():   # (with no sample inside, a wall thinner than the step, there is nothing to measure from:
            voxels = (np.where(inside, -(ndimage.distance_transform_edt(inside) - 0.5),   # 0.75 mm came out everywhere)
                               ndimage.distance_transform_edt(~inside) - 0.5) * step).astype(np.float32)
            values = np.where(np.abs(voxels - values) > 2 * step, voxels, values)
        self.values = values
        self.step = step

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float64)
        coords = (points - self.grid.origin) / self.step
        d = ndimage.map_coordinates(self.values, coords.T, order=1, mode="nearest")
        # (beyond the grid: its edge value had made a body 500 mm away "within a node spacing" for Connect to)
        top = self.grid.origin + (np.asarray(self.grid.shape) - 1) * self.step
        away = np.linalg.norm(np.maximum(self.grid.origin - points, 0.0) + np.maximum(points - top, 0.0), axis=-1)
        return d + away

    def thickest(self):
        """The thickest bar that fits anywhere in the design space (mm): twice its greatest depth, measured
        again on a fine grid round the deepest sample (a plate thinner than the step is found too)."""
        if self._thickest is None:
            i = np.unravel_index(int(np.argmin(self.values)), self.values.shape)
            depth = -float(self.values[i])
            centre, step = self.grid.origin + np.asarray(i) * self.step, self.step
            for _ in range(2):   # (a tenth of the step, then a hundredth)
                fine = Grid(centre - 1.5 * step, centre + 1.5 * step, step / 10)
                values = np.asarray(Context(fine, self.cache, np).eval_full(self.space))
                j = np.unravel_index(int(np.argmin(values)), values.shape)
                depth = max(depth, -float(values[j]))
                centre, step = fine.origin + np.asarray(j) * fine.voxel, step / 10
            self._thickest = 2.0 * max(depth, 0.0)
        return self._thickest


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


def _face_nodes(cond, spacing, rng, mode=None, memo=None):
    """(positions, forces, free radii, piece numbers) of a support or load: per connected piece of its
    faces. `mode`: how a hole is held or loaded (`_piece_nodes`). `memo`: the random state each piece was
    first clustered with, so a piece chosen again (a load on a held face, two loads on one face) gets the
    same nodes, which `_build` then merges (else a load on its own support became a 1.3 g frame of short
    bars between the two clusterings, and loads that cancel a frame carrying nothing)."""
    out = []
    for tri in _pieces(cond.triangles):
        use = rng
        if memo is not None:
            key = (np.round(tri, 6).tobytes(), mode)
            if key in memo:
                use = copy.deepcopy(memo[key])
            else:
                memo[key] = copy.deepcopy(rng)
        out.append(_piece_nodes(cond, tri, spacing, use, mode))
    if not out:
        raise ConditionError(f"{cond.label or cond.kind}: no faces chosen")
    piece = np.concatenate([np.full(len(o[0]), k) for k, o in enumerate(out)])
    return (*(np.concatenate(a) for a in zip(*out)), piece)


def _piece_nodes(cond, triangles, spacing, rng, mode=None):
    """(positions (k, 3), forces (k, 3), free radius (k,)) of one piece of a support or load. Bars
    may leave the design space within a node's free radius (they start at its surface). `mode`: None,
    a hole (or pin) at one node on its axis; "axis", a held hole at two nodes on its axis (supports on
    one line); "around", a hole at AROUND nodes around its wall (a load: where it presses), set
    SET_IN into the material, so several bars take it (the frame's body was overstressed there)."""
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
    centre = (pts * areas[:, None]).sum(axis=0) / areas.sum()
    if not turning and (round_face or cancelling) and _compact(triangles, pts, centre, spacing):
        if mode == "around":
            on = np.linalg.norm(force, axis=1) > 0 if cond.kind == "load" else np.ones(len(pts), dtype=bool)
            p, a, n, f = pts[on], areas[on], normals[on], force[on]
            k = min(int(np.clip(round(a.sum() / (2 * spacing ** 2)), *AROUND)), len(p))
            _, label = kmeans2(p, k, minit="++", rng=rng)
            nodes, forces = [], []
            for j in np.unique(label):
                mine = label == j
                middle = (p[mine] * a[mine, None]).sum(axis=0) / a[mine].sum()
                i = np.argmin(np.linalg.norm(p[mine] - middle, axis=1))
                nodes.append(p[mine][i] - SET_IN * spacing * n[mine][i])   # (outward normals point into the hole)
                forces.append(f[mine].sum(axis=0))
            return np.asarray(nodes), np.asarray(forces), np.full(len(nodes), spacing / 2)
        if mode == "axis" and round_face:
            # a bolt holds along its hole: two nodes on the axis (the direction no normal takes), a
            # quarter of the hole's length in from each end (on its ends, bars had to leave the
            # wall's face; on the wall, most cut through the hole)
            axis = np.linalg.eigh((normals * areas[:, None]).T @ normals)[1][:, 0]
            t = (pts - centre) @ axis
            ends = centre + np.outer([0.75 * t.min() + 0.25 * t.max(), 0.25 * t.min() + 0.75 * t.max()], axis)
            reach = max(float(np.linalg.norm(pts - e, axis=1).max()) for e in ends)
            return ends, np.zeros((2, 3)), np.full(2, reach + spacing / 2)
        reach = float(np.linalg.norm(pts - centre, axis=1).max())
        return centre[None], force.sum(axis=0)[None], np.array([reach + spacing / 2])
    area = float(areas.sum())
    k = int(np.clip(round(area / (2 * spacing ** 2)), *FACE_NODES))
    if turning or mode:   # (a face spread out: never on one line, which bars on pins could turn about)
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


def _compact(triangles, pts, centre, spacing):
    """Whether a piece of faces is a hole or a pin (a cylinder, `geometry._cylinder`) or small (within
    1.5 node spacings of its middle): only then does it act at one node there, whose bars may leave the
    design space within its reach. (An end face and a large face, picked together, whose pushes cancelled,
    became one node in the middle of the part with bars free to leave it within 96 mm: they ran through
    empty space and the design space cut them off the part.)"""
    from .geometry import _cylinder
    return float(np.linalg.norm(pts - centre, axis=1).max()) <= 1.5 * spacing or _cylinder(triangles) is not None


def _on_a_line(points, spacing):
    """Whether points (n, 3) all lie on one line (or are one point), within a hundredth of a spacing."""
    if len(points) < 3:
        return True
    s = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return bool(s[1] < 0.01 * spacing * math.sqrt(len(points)))


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


class Unreached(ConditionError):
    """A support or load that no candidate bar reaches (`what`: which). `ground`: the candidate nodes
    and bars there were (nodes without bars too, at supports and loads), `lonely`: its nodes no bar
    reaches (the editor draws them, Show: candidate nodes)."""

    def __init__(self, message, what, ground=None, lonely=()):
        super().__init__(message)
        self.what, self.ground, self.lonely = what, ground, np.asarray(lonely, dtype=np.int64)


class Flat(Unreached):
    """Every candidate node on one line (or in one plane) with a load across it (`_across`). `suggest`: a
    node spacing that puts nodes across the design space (mm), or None."""

    def __init__(self, message, ground=None, suggest=None):
        super().__init__(message, "the loads", ground)
        self.suggest = suggest


def build(space, conditions, spacing=0.0, reach=3.0, min_size=2.0, overlap=False, cache=None, seed=0, attach=(),
          spread=()):
    """The ground structure of a design space (implicit body) with supports and loads. `attach`:
    frames or CAD bodies this frame connects to (their bars or surface hold it, like a support).
    `spread`: indices of conditions whose holes are held or loaded at nodes around their walls.
    `spacing` 0: automatic, the diagonal / AUTO_NODES, made FINER while that leaves a support or load
    without bars, down to AUTO_FLOOR of it (a lattice coarser than a wall can miss it entirely: the
    table bracket's 8 mm plate lay between two layers 13 mm apart, a load on it reached nothing)."""
    if space.bounds is None:
        raise ConditionError("the design space has no size of its own: fill a body first")
    args = (reach, min_size, overlap, cache, seed, attach, spread)
    if spacing > 0:
        return _build(space, conditions, float(spacing), *args)
    lo, hi = (np.asarray(b, dtype=np.float64) for b in space.bounds)
    auto = step = tidy(float(np.linalg.norm(hi - lo)) / AUTO_NODES)
    first = last = None
    while True:
        try:
            ground = _build(space, conditions, step, *args)
        except Unreached as err:   # (`Flat` too: finer down to AUTO_FLOOR, not to the spacing it suggests; a 300 x 4
            first, last = first or err, err   # x 4 mm bar at 1.5 mm made an LP of 6 GB)
            step = tidy(step * FINER)
            if step >= AUTO_FLOOR * auto:
                continue
            raise
        except ConditionError:
            if last is None:
                raise
            raise last from None   # (finer: too many nodes)
        if isinstance(first, Flat):
            ground.notes.insert(0, f"with nodes {auto:g} mm apart they all lay on one line or plane, which bars cannot "
                                   f"carry the load across: {step:g} mm used")
            ground.coarser = auto
        elif first is not None:
            ground.notes.insert(0, f"with nodes {auto:g} mm apart no bar reached {first.what} (the lattice can miss "
                                   f"walls thinner than that): {step:g} mm used")
            ground.coarser = auto
        return ground


def _build(space, conditions, spacing, reach, min_size, overlap, cache, seed, attach, spread):
    """`build` at a given node spacing (mm)."""
    lo, hi = (np.asarray(b, dtype=np.float64) for b in space.bounds)
    if reach < 1:
        raise ConditionError("reach must be at least 1 node spacing")
    notes = []
    # the lattice, centred in the design space's box (counted before the design space is sampled at a third of
    # the spacing: 0.01 mm asked for 341 GB there)
    count = np.floor((hi - lo) / spacing).astype(int) + 1
    boxed = float(np.prod(count.astype(np.float64)))
    if boxed > 8 * MAX_LATTICE:
        raise ConditionError(f"node spacing {spacing:g} mm gives too many nodes here: use "
                             f"{_up(spacing * (boxed / (8 * MAX_LATTICE)) ** (1 / 3)):g} mm or more")
    step = min(spacing / 3, max(min_size, spacing / 8))
    inside = _Inside(space, step, cache)
    thick = inside.thickest()
    if 0 < thick < 4 * step:
        # (interpolated over so few samples, its middle looked shallower than it is: the centre of a 4 mm bar sampled
        # every 2.5 mm was 0.96 mm deep, not 2, and no 2 mm bar fitted)
        inside = _Inside(space, thick / 4, cache)
        thick = inside.thickest()   # (measured again, finer)
    if min_size > thick * (1 + 1e-3):
        raise ConditionError(f"Min size {min_size:g} mm: no bar that thick fits in the design space, which is at most "
                             f"{tidy(thick) if thick > 0 else 0:g} mm thick")
    start = lo + ((hi - lo) - (count - 1) * spacing) / 2
    ijk = np.stack(np.meshgrid(*[np.arange(c) for c in count], indexing="ij"), axis=-1).reshape(-1, 3)
    lattice = start + ijk * spacing
    keep = inside(lattice) <= -DEPTH * min_size
    ijk, lattice = ijk[keep], lattice[keep]
    if len(lattice) > MAX_LATTICE:
        raise ConditionError(f"node spacing {spacing:g} mm gives {len(lattice):,} nodes (at most {MAX_LATTICE:,}): "
                             f"use {_up(spacing * (len(lattice) / MAX_LATTICE) ** (1 / 3)):g} mm or more")

    # nodes on the faces of supports and loads
    rng = np.random.default_rng(seed)
    nodes, owner, fixed, loads = [lattice], [np.full(len(lattice), -1)], [np.zeros((len(lattice), 3), bool)], \
        [np.zeros((len(lattice), 3))]
    free = [np.zeros(len(lattice))]
    piece = [np.full(len(lattice), -1)]
    if not any(c.kind == "fixed" for c in conditions) and not attach:
        raise ConditionError("the part is not held anywhere: add Fixed Faces (or Connect to a frame or body)")
    if not any(c.kind == "load" for c in conditions):
        raise ConditionError("there is no load: add a Load")
    memo = {}
    faces = [_face_nodes(cond, spacing, rng, "around" if ci in spread else None, memo)
             for ci, cond in enumerate(conditions)]
    held = [f[0] for f, c in zip(faces, conditions) if c.kind == "fixed"]
    if held and not attach and _on_a_line(np.concatenate(held), spacing):
        # The supports are points on one line (two bolt holes, one hole: each held at one node on its
        # axis): bars on pin joints could turn about it, so no frame carried a load with a lever about
        # it (bolted bracket held by its two lower bolts: "no frame ... can carry the loads"). A bolt does
        # not let it turn: a hole is held along its axis (two nodes), a face at four points or more.
        faces = [_face_nodes(c, spacing, rng, "axis", memo) if c.kind == "fixed" and ci not in spread else f
                 for ci, (f, c) in enumerate(zip(faces, conditions))]
        notes.append("the supports lie on one line (as two bolt holes do): a frame of bars could turn about it, so "
                     "each bolt hole is held along its axis")
        held = [f[0] for f, c in zip(faces, conditions) if c.kind == "fixed"]
        if _on_a_line(np.concatenate(held), spacing):
            # holes on one axis (an axle's lugs): their axes are that line too, so they are held at nodes
            # round their walls (Fixed Faces clamp the hole all round, as Static Analysis does)
            faces = [_face_nodes(c, spacing, rng, "around", memo) if c.kind == "fixed" and ci not in spread else f
                     for ci, (f, c) in enumerate(zip(faces, conditions))]
            notes[-1] = ("the supports lie on one axis (one hole, or holes in line as an axle's lugs): a frame of bars "
                         "could turn about it, so each hole is held at points round its wall")
    pushed = sum(float(np.abs(f[1]).sum()) for f in faces)   # (before loads on one node add up)
    pieces_before = 0
    for ci, (cond, (pos, force, radius, part)) in enumerate(zip(conditions, faces)):
        nodes.append(pos)
        owner.append(np.full(len(pos), ci))
        fixed.append(np.full((len(pos), 3), cond.kind == "fixed"))
        loads.append(force)
        free.append(np.maximum(radius, FREE_SIZE * min_size))
        piece.append(part + pieces_before)
        pieces_before += int(part.max()) + 1 if len(part) else 0
    for ai, item in enumerate(attach):
        pos, radius = _attach_nodes(item, spacing, inside, rng)
        if not len(pos):
            notes.append(f"connect to #{ai + 1}: nothing of it lies within {spacing:g} mm of the design space")
        nodes.append(pos)
        owner.append(np.full(len(pos), len(conditions) + ai))
        fixed.append(np.ones((len(pos), 3), dtype=bool))
        loads.append(np.zeros((len(pos), 3)))
        free.append(np.maximum(radius, FREE_SIZE * min_size))
        piece.append(np.full(len(pos), -1))
    nodes, owner, piece = np.concatenate(nodes), np.concatenate(owner), np.concatenate(piece)
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
        piece = np.concatenate([piece[:n_lat], piece[n_lat:][first]])
        fixed, loads = np.concatenate([fixed[:n_lat], held]), np.concatenate([loads[:n_lat], merged])
        free = np.concatenate([free[:n_lat], wide])
    on_held = float(np.abs(loads[fixed]).sum())
    loads[fixed] = 0.0  # a load on a held node goes straight into the support
    if not np.abs(loads).max() > 0:
        if not pushed > 0:
            raise ConditionError("the loads are 0: give the Load an amount other than 0")
        if on_held > 0:
            raise ConditionError("all of the load sits on Fixed Faces: it goes straight into the supports and the "
                                 "frame has nothing to carry. Put the load on other faces")
        raise ConditionError("the loads cancel each other out (equal and opposite on the same faces): the frame has "
                             "nothing to carry")
    if on_held > 0.01 * pushed:
        notes.append(f"{on_held / pushed:.0%} of the load sits on Fixed Faces: it goes straight into the supports")

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
    if not overlap and n_lat:   # (no lattice node at all: Joint cost 0 with a spacing larger than the part crashed)
        step = np.abs(ijk[np.minimum(a, n_lat - 1)] - ijk[np.minimum(b, n_lat - 1)])
        keep &= ~both | (np.gcd.reduce(step, axis=1) == 1)
    pairs = pairs[keep]
    pairs = pairs[_stays_inside(nodes, pairs, free, inside, min_size)]
    lengths = np.linalg.norm(nodes[pairs[:, 1]] - nodes[pairs[:, 0]], axis=1)

    # nodes without bars go; a load or support without bars is an error
    used = np.zeros(len(nodes), dtype=bool)
    used[pairs.ravel()] = True
    lonely = ~used & (owner >= 0) & (owner < len(conditions))
    if lonely.any():
        shown = used | (owner >= 0)   # (what there was, for the editor)
        remap = np.cumsum(shown) - 1
        partial = Ground(nodes[shown], remap[pairs], lengths, fixed[shown], loads[shown], spacing, owner[shown], notes,
                         free[shown], inside, (owner >= len(conditions))[shown], piece[shown])
        lonely_shown = np.flatnonzero(lonely[shown])
        if not len(pairs):
            raise Unreached(f"no bar fits inside the design space with nodes {spacing:g} mm apart (it is at most "
                            f"{tidy(thick):g} mm thick, and bars of Min size {min_size:g} mm keep their centres "
                            f"{DEPTH * min_size:g} mm inside): use a smaller node spacing or min size",
                            "the supports and loads", partial, lonely_shown)
        i = np.flatnonzero(lonely)[0]
        what = conditions[int(owner[i])].label or conditions[int(owner[i])].kind
        x, y, z = nodes[i]
        raise Unreached(f"{what}: no bar reaches its point at ({x:.0f}, {y:.0f}, {z:.0f}) mm inside the design space "
                        f"with nodes {spacing:g} mm apart (try a smaller node spacing or min size)", what, partial,
                        lonely_shown)
    remap = np.cumsum(used) - 1
    ground = Ground(nodes[used], remap[pairs], lengths, fixed[used], loads[used], spacing, owner[used], notes,
                    free[used], inside, (owner >= len(conditions))[used], piece[used])
    _joined(ground, conditions, min_size)
    _across(ground, min_size, thick)
    notes.append(f"{len(ground.nodes):,} candidate nodes {spacing:g} mm apart, {len(ground.bars):,} candidate bars")
    return ground


def _joined(ground, conditions, min_size):
    """Raises Unreached when a load's nodes have bars, but no chain of them reaches anything held: the design
    space is too thin for bars of Min size somewhere between them (a neck, a gap) or they are separate bodies.
    (The layout's LP then only said it found no frame, "a load not connected to a support?")"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(ground.nodes)
    graph = coo_matrix((np.ones(len(ground.bars)), (ground.bars[:, 0], ground.bars[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    cut = (np.abs(ground.loads).sum(axis=1) > 0) & ~np.isin(label, label[ground.fixed.any(axis=1)])
    if cut.any():
        i = int(np.flatnonzero(cut)[0])
        what = conditions[int(ground.owner[i])].label or "a load"
        x, y, z = ground.nodes[i]
        raise Unreached(f"{what}: no chain of bars inside the design space joins its point at ({x:.0f}, {y:.0f}, "
                        f"{z:.0f}) mm to a support with nodes {ground.spacing:g} mm apart: somewhere between them the "
                        f"design space is too thin for bars of Min size {min_size:g} mm (a neck or a gap), or they are "
                        f"on separate bodies", what, ground, np.flatnonzero(cut))


def _across(ground, min_size, thick):
    """Raises Flat when every candidate node lies on one line (or in one plane) and a load acts across it: no
    frame of bars there carries it (a 300 x 4 x 4 mm bar's automatic spacing put one row of nodes down its
    middle). It suggests a spacing that puts nodes across the design space."""
    pts = ground.nodes - ground.nodes.mean(axis=0)
    _, s, vt = np.linalg.svd(pts, full_matrices=False)
    # (within a tenth of a spacing: the support's and load's nodes sat ~1 mm off the row, a truss of 1 mm levers)
    flat = np.concatenate([s, np.zeros(3 - len(s))]) < 0.1 * ground.spacing * math.sqrt(len(pts))
    if not flat[2]:
        return
    along = vt[0] if flat[1] else None          # a line's direction, else the plane's normal is vt[2]
    f = ground.loads[np.abs(ground.loads).sum(axis=1) > 0]
    across = f - np.outer(f @ along, along) if along is not None else np.outer(f @ vt[2], vt[2])
    if not (np.linalg.norm(across, axis=1) > 0.01 * np.linalg.norm(f, axis=1)).any():
        return   # (along the line or in the plane, as near as the nodes are to it: bars there carry it)
    # a spacing that puts nodes a bar apart across it, if there is room for a frame there at all: in a 4 mm bar,
    # 2 mm bars 1 mm apart made an LP that took 4 minutes and 3.9 GB, for 963 bars filling it
    # (nor in a rod: a 300 x 4 x 4 mm bar with 1 mm bars 1.5 mm apart made an LP of 6 GB, the bars from its ends to every
    # node of the row nearly dense for HiGHS)
    room = (thick - min_size) / 2
    long = float(np.ptp(ground.nodes, axis=0).max())
    slender = long > SLENDER * thick
    suggest = _down(room) if room >= min_size and not slender else None
    shape = "on one line" if along is not None else "in one plane"
    hint = (f"use a smaller Node spacing (about {suggest:g} mm)" if suggest else
            f"it is too thin for a frame of {min_size:g} mm bars (that needs about {3 * min_size:g} mm across): a "
            f"smaller Min size, a thicker design space, or Topology Optimization" if room < min_size else
            f"it is too slender for a frame ({long / thick:.0f} times as long as it is thick): Topology Optimization, or "
            f"a thicker design space")
    raise Flat(f"all candidate nodes lie {shape} with nodes {ground.spacing:g} mm apart (the design space is at most "
               f"{tidy(thick):g} mm thick, and bar centres stay {DEPTH * min_size:g} mm inside it), and bars {shape} "
               f"cannot carry a load across it: {hint}", ground, suggest)


def _up(x):
    """Rounded up to two significant digits."""
    step = 10 ** (math.floor(math.log10(x)) - 1)
    return round(math.ceil(x / step - 1e-9) * step, 10)


def _down(x):
    """Rounded down to two significant digits."""
    step = 10 ** (math.floor(math.log10(x)) - 1)
    return round(math.floor(x / step + 1e-9) * step, 10)


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
                  np.concatenate([ground.attached, np.zeros(m_new, dtype=bool)]),
                  None if ground.piece is None else np.concatenate([ground.piece, np.full(m_new, -1)]))
