"""Offset Faces: picked CAD faces moved out (or in) by a distance, as one piece, as a field.

Near the chosen faces (the patch) every grid sample p gets two coordinates from the CAD mesh
(`geometry`): its height over the patch, hgt, and a sideways one, lat: 0 on the walls that stand on
the patch's border along the faces' normals, negative over the patch, positive beyond it. With c
the closest point of p on the patch (igl's AABB, or the Warp kernel on the GPU) and n the faces' own
normal at c (angle-weighted vertex normals per CAD face, interpolated across the triangle):
- c inside the patch: hgt = +-|p - c| (the patch's own offset: where two chosen faces fold outwards
  the moved faces join round, as Offset Body does; where they fold inwards they meet sharp),
  lat = -(distance from c to the nearest border edge);
- c on the border: hgt = (p - c).n, lat = what is left of |p - c| sideways (outside a corner of the
  border: the distance to the line through the corner along n).
So {lat <= 0, 0 <= hgt <= d} is the space over the faces, d high, with straight walls, and a field
d (a varying offset) is read sample by sample.

Sides straight (`offset_faces`): the part with a raised slab U = {lat <= 0, -e <= hgt <= d} added and
a recess K = {lat <= 0, d <= hgt <= e} cut away. For d > 0 the recess is empty, for d < 0 the slab
lies inside the part, so one formula serves a Distance field that is positive in some places and
negative in others. The slab reaches e into the part and the recess e out of it, so that no surface
of theirs lies on the part's: e = |d| (the inside of the result stays a good distance), but at most
half the wall under the faces (and half the free space over them), so a slab never comes out behind
a thin wall (`walls`). Sides round: U and K are the patch's own offset, |p - patch| <= |d|, but not
deeper than e straight under (over) the faces.

Edge (the step's outside edge) and Foot (its inside edge): raised, Edge shapes the slab's own top
rim and Foot the union of part and slab; sunk, Edge shapes the cut's rim (part minus recess) and
Foot the recess's own bottom corner. chamfer and round are the Booleans' (Mercury's rounded
intersection / union, exact where the surfaces meet at 90 degrees, as the step's always do); smooth
is the G2 blend (`fields.bulge` order 2) at SMOOTH_SIZE x the size, which makes its middle as deep
as round's. Sizes are cut to the step's height: two parallel surfaces (the moved face and the
original one) closer than a rounded union's size bulge into each other, so this also fades a
fillet out where a field Distance goes to 0.

Whether a step has a foot or an edge depends on the neighbouring face. A box's top raised: its walls
continue the box's sides and there is no foot, and a rounded union of two surfaces that continue
each other bulges (Mercury's union is -0.41 r where both are 0); sunk, the recess's rounded bottom
corner left a knife-thin lip on the side. A pocket's floor raised or deepened: its walls continue
the pocket's and there is no edge (the rim rounding made a groove). So every border edge gets
s = sin(theta), theta the fold from the chosen face to its neighbour across that edge, measured with
both faces' own normals at the edge (0: the neighbour lies in the same plane or runs on
tangentially, a fillet; +1: it folds away 90 degrees or more, a convex edge; -1: it folds up 90
degrees or more, a concave one), and Foot works at its size x (1 - s), Edge at its size x (1 + s),
clipped to 0..1. In between (a 45-degree chamfer next to the face) the smaller size makes Mercury's
round about as deep along the bisector as a true rolling-ball fillet of the full size at that angle
(0.17 r against 0.155 r at 30 degrees). s is constant along a border edge and, within RUNOUT mm of a
border vertex, moves to the vertex's value (the incident edge's with the largest |s|): a foot
fillet along a pad that reaches a box's side runs out there instead of jumping.

Every array the field reads is a pure function of position (blocks stay exact): the per-sample
values come from exact closest points, farther than `band` from the faces the slab and the recess
are FAR, and within TAPER of the band's end they grow to FAR smoothly, so the result has no jump.
"""
import math
import threading
from collections import OrderedDict

import numpy as np

from .facemask import nearest_on_grid
from .fields import Field, _core, _digest, _grown, _lift, _round_max, _round_min, mesh_fingerprint
from .grid import upsample_linear
from .sdf import mesh_query
from .shared import Shared

SIDES = ("straight", "round", "blend")
EDGES = ("sharp", "chamfer", "round", "smooth")
# A smooth (G2) blend of k = SMOOTH_SIZE x size is as deep in its middle as a round one of that size at a
# right angle: along the bisector the cubic's surface lies sqrt(2) k / 6 from the corner, Mercury's round
# (sqrt(2) - 1) r. (The frames' CONTINUITY_SIZE[2] = 1.5 matches their quadratic G1 blend, k / 4 deep,
# which is shallower than a circular fillet; with it smooth came out 15% shallower than round.)
SMOOTH_SIZE = 3.0 * (2.0 - math.sqrt(2.0))
GAP = 0.05            # mm: the slab reaches at least this far into the part (and a recess out of it)
WALL_SHARE = 0.5      # ... and at most this share of the wall under the faces (the free space over them)
WALL_QUANTILE = 0.02  # the walls' thickness: their thinnest 2% by area (a sliver's ray does not count)
RUNOUT = 1.5          # mm: along the border, Edge and Foot move to a corner's value within this much of it
TAPER = 1.0           # mm: before the end of the computed band the slab's and recess's values grow to FAR
ON_BORDER = 1e-4      # mm: a closest point this near a border edge lies on it (the GPU's are float32)
CHUNK = 500_000       # samples per pass of the exact queries (~1 KB of temporaries each)
FAR = np.float32(1e4)
_SQRT_HALF = math.sqrt(0.5)
_BORDERS = OrderedDict()   # patch key -> igl AABB over its border edges, per process
_LOCK = threading.Lock()


def _corner_normals(V, F, ids):
    """Normal at every triangle corner, (n, 3, 3): the angle-weighted mean over the triangles of the
    same CAD face around that vertex (so the normal is the face's own on both sides of a fold)."""
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    unit = cross / np.maximum(np.linalg.norm(cross, axis=1), 1e-300)[:, None]
    angles = np.empty((len(F), 3))
    for k in range(3):
        u, w = tri[:, (k + 1) % 3] - tri[:, k], tri[:, (k + 2) % 3] - tri[:, k]
        cos = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300)
        angles[:, k] = np.arccos(np.clip(cos, -1.0, 1.0))
    ids = np.asarray(ids, dtype=np.int64)
    key = np.asarray(F, dtype=np.int64) * (int(ids.max()) + 1 if len(ids) else 1) + ids[:, None]
    _, inverse = np.unique(key.ravel(), return_inverse=True)
    weight = (angles[..., None] * unit[:, None, :]).reshape(-1, 3)
    sums = np.stack([np.bincount(inverse, weights=weight[:, c]) for c in range(3)], axis=1)
    sums /= np.maximum(np.linalg.norm(sums, axis=1), 1e-300)[:, None]
    return sums[inverse].reshape(len(F), 3, 3)


def _quantile(values, weights, q):
    """The weighted q-quantile of values (inf counts as larger than everything; inf if none)."""
    values = np.where(np.isnan(values), np.inf, values)
    if not len(values) or weights.sum() <= 0:
        return math.inf
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][min(np.searchsorted(cum, q * cum[-1]), len(values) - 1)])


class Patch:
    """The chosen faces of one CAD body as a mesh of their own (vertices `V`, triangles `F`, CAD face
    `ids`, corner `normals`), and their border: `edges` (vertex pairs), `s` per edge (see the module
    doc) and `ends`, s at each edge's two vertices."""

    def __init__(self, mesh, ids, faces, invert=False):
        self.out = np.zeros((0, 3))   # per border edge: the direction away from the chosen face, across it
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        self.faces = tuple(sorted(int(f) for f in faces))
        self.invert = bool(invert)
        selected = np.isin(ids, self.faces)
        if self.invert:
            selected = ~selected
        self.key = _digest("offset-faces-patch", mesh_fingerprint(mesh), self.faces, self.invert)
        rows = np.flatnonzero(selected)
        used, inverse = np.unique(F[rows], return_inverse=True)
        self.V = V[used]
        self.F = inverse.reshape(-1, 3).astype(np.int64)
        self.ids = ids[rows]
        self.normals = _corner_normals(self.V, self.F, self.ids) if len(rows) else np.zeros((0, 3, 3))
        self.edges = np.zeros((0, 2), dtype=np.int64)
        self.s = np.zeros(0)
        self.ends = np.zeros((0, 2))
        self.unmatched = 0
        if len(rows) and not selected.all():
            self._border(V, F, ids, selected, used)

    def __len__(self):
        return len(self.F)

    def _border(self, V, F, ids, selected, used):
        Fs = self.F
        e = np.stack([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]], axis=1).reshape(-1, 2)
        n = len(self.V)
        key = np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        rows = np.flatnonzero(counts[inverse] == 1)        # edges of one chosen triangle only: the border
        if not len(rows):
            return
        owner, corner = rows // 3, rows % 3
        edges = e[rows]
        third = Fs[owner, (corner + 2) % 3]
        a, b = self.V[edges[:, 0]], self.V[edges[:, 1]]
        # the chosen face's own normal along the edge, and its outward direction across it
        nc = self.normals[owner, corner] + self.normals[owner, (corner + 1) % 3]
        nc /= np.maximum(np.linalg.norm(nc, axis=1), 1e-300)[:, None]
        m = np.cross(b - a, nc)
        m /= np.maximum(np.linalg.norm(m, axis=1), 1e-300)[:, None]
        m *= np.where(np.einsum("ij,ij->i", m, self.V[third] - a) > 0, -1.0, 1.0)[:, None]
        no = self._neighbour_normals(V, F, ids, selected, used[edges])
        found = np.linalg.norm(no, axis=1) > 0.5
        self.unmatched = int((~found).sum())
        sin, cos = np.einsum("ij,ij->i", no, m), np.einsum("ij,ij->i", no, nc)
        s = np.where(cos >= 0, sin, np.sign(sin))       # past 90 degrees the fold counts as 90
        s = np.where(found, np.clip(s, -1.0, 1.0), 0.0)
        # each border vertex takes the s of its incident edge with the largest |s|
        flat, values = edges.ravel(), np.repeat(s, 2)
        biggest = np.zeros(n)
        np.maximum.at(biggest, flat, np.abs(values))
        at_vertex = np.zeros(n)
        hit = np.abs(values) >= biggest[flat] - 1e-12
        at_vertex[flat[hit]] = values[hit]
        self.edges, self.s, self.ends, self.out = edges, s, at_vertex[edges], m

    def _neighbour_normals(self, V, F, ids, selected, pairs):
        """The neighbouring (not chosen) face's own normal along each border edge (full-mesh vertex
        pairs), zero where none is found: the triangle across the edge, else (a crack: the faces' meshes
        do not share the edge) the mean of the unchosen triangles at its ends."""
        touches = np.zeros(len(V), dtype=bool)
        touches[pairs.ravel()] = True
        cand = np.flatnonzero(~selected & touches[F].any(axis=1))
        out = np.zeros((len(pairs), 3))
        if not len(cand):
            return out
        Fc = F[cand]
        normals = _corner_normals(V, Fc, ids[cand])
        nv = len(V)
        e = np.stack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]], axis=1).reshape(-1, 2)
        ekey = np.minimum(e[:, 0], e[:, 1]) * nv + np.maximum(e[:, 0], e[:, 1])
        order = np.argsort(ekey, kind="stable")
        ekey = ekey[order]
        want = np.minimum(pairs[:, 0], pairs[:, 1]) * nv + np.maximum(pairs[:, 0], pairs[:, 1])
        at = np.minimum(np.searchsorted(ekey, want), len(ekey) - 1)
        match = ekey[at] == want
        row = order[at[match]]
        t, k = row // 3, row % 3
        out[match] = normals[t, k] + normals[t, (k + 1) % 3]
        if not match.all():  # a crack: every unchosen triangle at the edge's two vertices
            per_vertex = np.zeros((nv, 3))
            for c in range(3):
                np.add.at(per_vertex, Fc[:, c], normals[:, c])
            miss = np.flatnonzero(~match)
            out[miss] = per_vertex[pairs[miss, 0]] + per_vertex[pairs[miss, 1]]
        length = np.linalg.norm(out, axis=1)
        return np.where(length[:, None] > 1e-12, out / np.maximum(length, 1e-300)[:, None], 0.0)

    def border_share(self, test):
        """Share of the border's length whose s passes test(s)."""
        if not len(self.edges):
            return 0.0
        length = np.linalg.norm(self.V[self.edges[:, 1]] - self.V[self.edges[:, 0]], axis=1)
        return float(length[test(self.s)].sum() / max(length.sum(), 1e-300))


def walls(mesh, patch, out_reach, in_reach):
    """(e_down, e_up) caps, mm: WALL_SHARE of the wall under the chosen faces (for raising them) and of
    the free space over them (for sinking them), their thinnest WALL_QUANTILE by area, measured by ray
    casts from the triangles' middles (inf where farther than the step needs)."""
    from .checks import ray_first_hit, ray_thickness  # Warp: only needed here
    if not len(patch):
        return math.inf, math.inf
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    tri = patch.V[patch.F]
    centres = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(2 * area, 1e-300)[:, None]
    down = up = math.inf
    if out_reach > 0:
        reach = 2 * float(out_reach) + 1.0
        down = WALL_SHARE * _quantile(ray_thickness(V, F, centres, normals, reach).astype(np.float64), area,
                                      WALL_QUANTILE)
    if in_reach > 0:
        reach = 2 * float(in_reach) + 1.0
        hit, _ = ray_first_hit(V, F, centres + 1e-3 * normals, normals, reach)
        gap = np.where(hit >= 0, hit.astype(np.float64) + 1e-3, np.inf)
        up = WALL_SHARE * _quantile(gap, area, WALL_QUANTILE)
    return down, up


def _border_tree(key, V, E):
    import igl
    with _LOCK:
        tree = _BORDERS.get(key)
        if tree is not None:
            _BORDERS.move_to_end(key)
            return tree
    tree = igl.AABB()
    tree.init(V, E)
    with _LOCK:
        _BORDERS[key] = tree
        while len(_BORDERS) > 16:
            _BORDERS.popitem(last=False)
    return tree


def _fall(x):
    """1 at 0, falling smoothly to 0 at 1 and beyond."""
    t = np.clip(x, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _local(points, tri, closest, V, F, N, edges, s_edge, ends, away, key):
    """(lat, hgt, s) for points whose closest point on the patch is `closest`, on triangle `tri`."""
    corners = V[F[tri]]
    e1, e2, w = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], closest - corners[:, 0]
    d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    w1, w2 = (w * e1).sum(1), (w * e2).sum(1)
    det = np.maximum(d11 * d22 - d12 * d12, 1e-300)
    u, v = (d22 * w1 - d12 * w2) / det, (d11 * w2 - d12 * w1) / det
    bary = np.clip(np.column_stack([1 - u - v, u, v]), 0.0, None)
    bary /= np.maximum(bary.sum(axis=1), 1e-300)[:, None]
    n = np.einsum("ki,kij->kj", bary, N[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    diff = points - closest
    dist = np.linalg.norm(diff, axis=1)
    height = np.where(np.einsum("ij,ij->i", diff, n) < 0, -dist, dist)
    if not len(edges):  # the faces close round: no border, no walls
        return np.full(len(points), -FAR, dtype=np.float64), height, np.zeros(len(points))
    sq, k, at = _border_tree(key, V, edges).squared_distance(V, edges, np.ascontiguousarray(closest))
    k = np.asarray(k, dtype=np.int64)
    inside = np.sqrt(np.maximum(sq, 0.0))
    on = inside < ON_BORDER
    db = points - at
    hb = np.einsum("ij,ij->i", db, n)
    side = np.sqrt(np.maximum((db * db).sum(1) - hb * hb, 0.0))
    # sideways from the border point: + away from the faces (a closest point a hair inside the border,
    # within ON_BORDER, gives a hair less than 0, as it would have from `inside`)
    side = np.where(np.einsum("ij,ij->i", db - hb[:, None] * n, away[k]) < 0, -side, side)
    lat = np.where(on, side, -inside)
    hgt = np.where(on, hb, height)
    a, b = V[edges[k, 0]], V[edges[k, 1]]
    length = np.linalg.norm(b - a, axis=1)
    along = np.minimum(np.linalg.norm(at - a, axis=1), length)
    ramp = np.maximum(np.minimum(length / 2, RUNOUT), 1e-9)
    s = s_edge[k] + (ends[k, 0] - s_edge[k]) * _fall(along / ramp) + (ends[k, 1] - s_edge[k]) * _fall((length - along) / ramp)
    return lat, hgt, s


def _near(V, F, grid, band, key, gpu):
    """(flat sample indices, triangle, closest point) of the samples within `band` mm of the mesh, in
    chunks of at most CHUNK: facemask.nearest_on_grid (exact queries only near the mesh, found from a
    coarse pass), without holding every candidate's points and answers at once."""
    if gpu:
        near, tri, closest = nearest_on_grid(V, F, grid, band, key=key, gpu=True)
        for s in range(0, len(near), CHUNK):
            yield near[s:s + CHUNK], tri[s:s + CHUNK], closest[s:s + CHUNK]
        return
    query = mesh_query(V, F, key)
    if grid.size <= 20_000:
        candidates = np.arange(grid.size)
    else:
        factor = 2 if grid.size < 2_000_000 else 4
        coarse = grid.coarser(factor)
        d = np.sqrt(query.tree.squared_distance(query.V, query.F, coarse.points())[0]).astype(np.float32)
        candidates = np.flatnonzero(upsample_linear(d.reshape(coarse.shape), factor, grid.shape)
                                    < band + coarse.voxel * np.sqrt(3))
    start = np.asarray(grid.start)
    for s in range(0, len(candidates), CHUNK):
        part = candidates[s:s + CHUNK]
        points = grid.base + (start + np.column_stack(np.unravel_index(part, grid.shape))) * grid.voxel
        sq, tri, closest = query.tree.squared_distance(query.V, query.F, points)
        keep = sq <= band * band
        yield part[keep], np.asarray(tri)[keep], np.asarray(closest)[keep]


def geometry(patch, band):
    """Field of (lat, hgt, s) at every sample, shape (3, *grid.shape): exact within `band` mm of the
    chosen faces, (FAR, FAR, 0) farther away (see the module doc)."""
    V, F, N = Shared(patch.V), Shared(patch.F), Shared(patch.normals)
    edges, s_edge, ends, away = (Shared(a) for a in (patch.edges, patch.s, patch.ends, patch.out))
    key, band = patch.key, float(band)

    def fn(ctx):
        grid = ctx.grid
        out = np.empty((3,) + tuple(grid.shape), dtype=np.float32)
        out[0], out[1], out[2] = FAR, FAR, 0.0
        flat = [out[c].reshape(-1) for c in range(3)]
        start = np.asarray(grid.start)
        arrays = (V.get(), F.get(), N.get(), edges.get(), s_edge.get(), ends.get(), away.get())
        for near, tri, closest in _near(V, F, grid, band, key, ctx.xp is not np):
            if not len(near):
                continue
            points = grid.base + (start + np.column_stack(np.unravel_index(near, grid.shape))) * grid.voxel
            values = _local(points, np.asarray(tri, dtype=np.int64), np.asarray(closest, dtype=np.float64), *arrays,
                            key)
            for c in range(3):
                flat[c][near] = values[c]
        return ctx.xp.asarray(out)
    field = Field(fn, _digest("offset-faces-geometry", key, band), f"offset_faces_geometry({len(patch.F)})",
                  cache=True)
    field.mesh = (key, len(patch.F))  # every process evaluating it builds a search tree for the faces
    return field


def _radius(kind, size, step):
    """A size cut to the step's height (smooth: its blend width k = SMOOTH_SIZE x size)."""
    if kind == "sharp":
        return 0.0
    return np.minimum(float(size), step / SMOOTH_SIZE if kind == "smooth" else step)


def _smooth(xp, a, b, r):
    k = r * SMOOTH_SIZE
    h = xp.maximum(1.0 - xp.abs(a - b) / xp.maximum(k, 1e-12), 0.0)
    return k * h * h * h / 6.0


def _corner(kind, xp, a, b, r):
    """max(a, b), the corner of a prism (surfaces at 90 degrees), shaped: sharp = the exact distance
    round it, chamfer, round, smooth (G2)."""
    if kind == "sharp":
        return _round_max(xp, a, b, 0.0)
    if kind == "chamfer":
        return xp.maximum(xp.maximum(a, b), (a + b + r) * _SQRT_HALF)
    if kind == "round":
        return _round_max(xp, a, b, r)
    return xp.maximum(a, b) + _smooth(xp, a, b, r)


def _cut(kind, xp, a, b, r):
    """max(a, b) of a part and a cut, its edge shaped (sharp: as Boolean Subtract)."""
    return xp.maximum(a, b) if kind == "sharp" else _corner(kind, xp, a, b, r)


def _join(kind, xp, a, b, r):
    """min(a, b), a union, the inner edge filled: sharp, chamfer, round (as Boolean Union), smooth."""
    if kind == "sharp":
        return xp.minimum(a, b)
    if kind == "chamfer":
        return xp.minimum(xp.minimum(a, b), (a + b - r) * _SQRT_HALF)
    if kind == "round":
        return _round_min(xp, a, b, r)
    return xp.minimum(a, b) - _smooth(xp, a, b, r)


def band_for(reach, edge, edge_size, foot, foot_size):
    """How far from the faces the geometry is computed, mm: the step, the largest blend, the taper,
    rounded up to half a mm (so trying out sizes mostly keeps the cached geometry)."""
    blends = [min(size * (SMOOTH_SIZE if kind == "smooth" else 1.0), reach)
              for kind, size in ((edge, edge_size), (foot, foot_size)) if kind != "sharp" and size > 0]
    return math.ceil((reach + max(blends, default=0.0) + TAPER + 0.1) * 2) / 2


def offset_faces(body, patch, distance, low, high, sides="straight", edge="sharp", edge_size=0.0, foot="sharp",
                 foot_size=0.0, caps=(math.inf, math.inf)):
    """`body` with the faces of `patch` moved by `distance` mm (a number or a field, cut to low..high),
    sides straight or round, Edge and Foot shaped (see the module doc); caps = walls(...)."""
    if sides not in ("straight", "round"):
        raise ValueError("sides must be straight or round here (blend is a Texture with a Face Mask)")
    if edge not in EDGES or foot not in EDGES:
        raise ValueError(f"Edge and Foot must be one of {', '.join(EDGES)}")
    body = _lift(body)
    low, high = float(low), float(high)
    reach = max(abs(low), abs(high))
    band = band_for(reach, edge, edge_size, foot, foot_size)
    geom = geometry(patch, band)
    field = distance if isinstance(distance, Field) else None
    plain = None if field is not None else np.float32(min(max(float(distance), low), high))
    down, up = (float(c) for c in caps)
    edge_size, foot_size = float(edge_size), float(foot_size)

    # a plain distance at least GAP out has no recess, one at least GAP in no slab (their values could
    # only differ off the surface): leaving them out saves time and memory
    raising = field is not None or plain > -GAP
    sinking = field is not None or plain < GAP

    def weight(xp, kind, size, step, s, sign):
        """Edge (sign +1) or Foot (-1) size at every sample: cut to the step, times 1 +- s."""
        if kind == "sharp" or size <= 0:
            return 0.0
        return _radius(kind, size, step) * xp.clip(1.0 + sign * s, 0.0, 1.0)

    def fn(ctx):
        xp = ctx.xp
        g = ctx.eval(geom)
        lat, hgt, s = g[0], g[1], g[2]
        d = plain if field is None else xp.clip(xp.asarray(ctx.eval(field), dtype=xp.float32), low, high)
        out, sunk = xp.maximum(d, 0.0), xp.maximum(-d, 0.0)
        dist = xp.maximum(lat, 0.0)                              # distance to the faces
        dist *= dist
        dist += hgt * hgt
        xp.sqrt(dist, out=dist)
        u = (dist - (band - TAPER)) / TAPER
        xp.clip(u, 0.0, 1.0, out=u)
        grow = xp.where(u < 1.0, TAPER * u / xp.maximum(1.0 - u, 1e-6), FAR)
        del u
        result = ctx.eval(body)
        if raising:
            e_down = xp.minimum(xp.maximum(out, GAP), down)
            if sides == "straight":
                # where the walls run along a neighbouring face with the part on both sides (a raised pocket
                # floor), or below with the air on both (a sunk box top, a hole's wall), they stand GAP
                # beyond the border: on one plane both values were 0, a sheet of zeros the mesher took for
                # a surface
                side = lat - GAP * xp.minimum(s, 0.0) ** 2
                slab = _corner(edge, xp, side, hgt - d, weight(xp, edge, edge_size, out, s, 1.0))
                del side
                xp.maximum(slab, -hgt - e_down, out=slab)
            else:  # (hgt - d changes nothing inside the tube; on the other side of 0 it keeps it empty)
                slab = xp.maximum(dist - out, hgt - d)
                xp.maximum(slab, xp.minimum(-hgt - e_down, -lat), out=slab)
            slab += grow
            xp.minimum(slab, FAR, out=slab)
            result = _join(foot, xp, result, slab, weight(xp, foot, foot_size, out, s, -1.0))
            del slab
        if sinking:
            e_up = xp.minimum(xp.maximum(sunk, GAP), up)
            if sides == "straight":
                side = lat - GAP * xp.maximum(s, 0.0) ** 2
                recess = _corner(foot, xp, side, d - hgt, weight(xp, foot, foot_size, sunk, s, -1.0))
                del side
                xp.maximum(recess, hgt - e_up, out=recess)
            else:
                recess = xp.maximum(dist - sunk, d - hgt)
                xp.maximum(recess, xp.minimum(hgt - e_up, -lat), out=recess)
            recess += grow
            xp.minimum(recess, FAR, out=recess)
            xp.negative(recess, out=recess)
            result = _cut(edge, xp, result, recess, weight(xp, edge, edge_size, sunk, s, 1.0))
        return xp.asarray(result, dtype=xp.float32)

    key = _digest("offset-faces", geom.key, body.key, None if field is None else field.key,
                  None if plain is None else float(plain), low, high, sides, edge, edge_size, foot, foot_size,
                  down, up)
    children = (geom, body) + ((field,) if field is not None else ())
    out_reach = max(high, 0.0)
    bounds, grown = body.bounds, body.grown
    if out_reach > 0 and body.bounds is not None:  # raised faces reach out of the part's box
        core, before = _core(body)
        bounds, grown = _grown(body.bounds, out_reach), (core, before + out_reach)
    result = Field(fn, key, f"offset_faces({body.name})", bounds, children=children)
    result.grown = grown
    return result
