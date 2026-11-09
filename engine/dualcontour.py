"""Dual contouring: a mesh from a field sampled on a grid that keeps sharp edges and corners
(docs/DUAL_CONTOURING_PLAN.md).

Marching cubes puts every vertex on a grid edge, so a CAD edge or a Boolean's crease that runs between
grid edges is cut off, about a voxel wide. Here each cell the surface passes through gets a vertex where
the planes through the surface's crossings of the cell's edges (point and normal) meet best, which is on
the crease or corner when the cell holds one; each crossed grid edge becomes a quad of the four cells
around it.

- Normals: the field's gradient by central differences at the edge's two samples, interpolated to the
  crossing. They are exact where the field is straight around both samples; a voxel from a crease they
  blend both sides. Each crossing is trusted by how straight the field is there (forward and backward
  differences agree: `_slopes`), and the planes are weighted by that trust.
- A cell next to a crease has no straight crossings of its own (all of them are within a voxel of the
  crease), so a piece with an untrusted crossing also fits the planes of the crossings up to RINGS steps
  away along the surface (sharing a quad with it, then with those): the faces' own planes, which meet on
  the crease. Following the surface, not a box of cells, keeps a thin wall's other side out. That fit is
  taken only where the planes meet cleanly (CREASE_FIT) and the field is a distance field there
  (slope 1: a CAD body, its Booleans, holes): a texture's ridge may be rounded finer than a voxel,
  which the samples cannot tell from sharp, and its sides' planes meet above it. (So can a CAD fillet
  under a voxel: it comes out sharp.) A cheap first look at the slope around each piece skips the rest
  where it is not 1 (on textures nearly every piece is beside a crease). Elsewhere the piece's own
  planes give a plain dual contouring vertex (as good as marching cubes on smooth surfaces, and closer
  on ridges).
- Manifold: a cell gets a vertex per piece of surface in it (Manifold Dual Contouring, Schaefer, Ju and
  Warren 2007): its crossed edges are grouped by how the surface links them on the cell's faces; on a
  face with four crossings, whether its inside corners join across the middle is decided as marching
  cubes does (the bilinear surface's saddle), the same in both cells that share it. A thin wall or two
  sheets in one cell keep their own vertices. Where a piece crosses a face twice beside one that does
  too, the two vertices would share four quads: each arc on that face gets a vertex of its own.
- The fit (QEF) is solved with a pull towards the piece's own crossings' mean (HOLD): a flat piece's
  vertex is the mean moved onto its plane, an edge's lands on its line, and a weak plane (a neighbour's
  that barely fits) moves it only a little; a sharp one is kept within MARGIN of the cell, the others
  inside it.
- Each quad is split along the diagonal whose triangles face best along the field's normal at its
  crossing; a sliver that still faces against it (vertices that left their cells crossed over) has its
  vertices put back into their cells, then onto their crossings' mean.
- Big grids are done in blocks (contour_block, join_blocks; tiles.mesh_tiled): each keeps its own
  cells' vertices and its own edges' quads and names vertices by their global cell, so the blocks join
  by name.

Measured on a turned box at 0.3 mm (tests/test_dualcontour.py): edges 0.03 voxel from the true ones on
average (marching cubes 0.34), corners 0.09 (0.54); real parts in docs/DUAL_CONTOURING_PLAN.md.
"""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sparse
import trimesh

from . import trace
from .backend import to_host
from .meshing import finite_values, touches_box

TRUST_SCALE = 0.2    # a crossing's trust: exp(-(bend at its samples) / this); a right-angle crease bends ~1,
                     # a sphere 8 voxels across ~0.03
CREASE_TRUST = 0.5   # a piece with a crossing trusted less than this takes its neighbours' planes
RINGS = 3            # ...from this many steps away along the surface,
REACH = 2.5          # weighted by exp(-(distance / (REACH voxels))^2) from the piece's own crossings' mean
ON_PLANE = 0.2       # ...and by exp(-(gap / ON_PLANE)^2), gap: voxels from the plane to its nearest own crossing
CREASE_FIT = 0.1     # ...and takes their fit only if it lies within this RMS distance of them (voxels)
DISTANCE_FIELD = 0.03  # ...and the field's slope there is within this of 1 (a distance field),
STRAIGHT = 0.9       # judged on the planes trusted this much
GUESS = 3.0          # pieces whose cheap first look at the slope is off by more than this x DISTANCE_FIELD
                     # skip the fit altogether
HOLD = 0.01          # a vertex's fit is pulled towards its crossings' mean this much (share of its strongest plane sum)
MARGIN = 0.5         # voxels a vertex may lie outside its cell: a corner that clips a cell without enclosing
                     # any of its samples gives that cell no vertex, so its neighbour's must reach it
PLAIN_MARGIN = -0.01  # ...a vertex that is not on a sharp edge, likewise (negative: that far inside)
INSET = 0.01         # voxels a vertex put back into its cell stays inside it: on its border, it met its
                     # neighbour's on their shared corner, one vertex in a float32 file (edges of 4 triangles)
DENSE_CELLS = 60_000_000  # up to this many cells, a cell's number among the surface cells comes from a table
CREASE_CHUNK = 20_000  # crease pieces whose neighbourhoods are built at once (~10 kB each: all at once, the
                       # holed vent case took 150 bytes per voxel, marching cubes 34)


def _cell_edges():
    """The corners (bit 0 = +x, 1 = +y, 2 = +z) of a cell's 12 edges: 0-3 along x (dy + 2 dz), 4-7
    along y (dx + 2 dz), 8-11 along z (dx + 2 dy)."""
    out = []
    for a in range(3):
        p, q = [b for b in range(3) if b != a]
        for s in range(4):
            c0 = ((s & 1) << p) | ((s >> 1) << q)
            out.append((c0, c0 | (1 << a)))
    return out


EDGES = _cell_edges()


def _cell_faces():
    """A cell's 6 faces: their corners in turn round the face (0 and 2 diagonal) and their edges."""
    out = []
    for a in range(3):
        p, q = [b for b in range(3) if b != a]
        for side in (0, 1):
            c0 = side << a
            corners = [c0, c0 | (1 << p), c0 | (1 << p) | (1 << q), c0 | (1 << q)]
            on = [e for e, (e0, e1) in enumerate(EDGES) if (e0 >> a) & 1 == side and (e1 >> a) & 1 == side]
            out.append((corners, on))
    return out


FACES = _cell_faces()


def _component_table():
    """For each of the 256 sign cases of a cell and each choice on its faces with four crossings (bit f
    of 0..63 set: on face f the inside corners join across the middle, else the outside ones do), the
    piece of surface (0..3) each crossed edge belongs to (-1: not crossed), and how many pieces there
    are."""
    table = np.full((256, 64, 12), -1, dtype=np.int8)
    count = np.zeros((256, 64), dtype=np.int64)
    for case in range(256):
        inside = [(case >> c) & 1 for c in range(8)]
        for choice in range(64):
            parent = {e: e for e, (c0, c1) in enumerate(EDGES) if inside[c0] != inside[c1]}

            def find(e):
                while parent[e] != e:
                    e = parent[e]
                return e

            for f, (corners, on) in enumerate(FACES):
                cut = [e for e in on if e in parent]
                if len(cut) == 2:
                    parent[find(cut[0])] = find(cut[1])
                elif len(cut) == 4:  # the corners kept apart get an arc each: link their two crossings
                    apart = 0 if (choice >> f) & 1 else 1
                    for c in corners:
                        if inside[c] == apart:
                            e, g = [e for e in cut if c in EDGES[e]]
                            parent[find(e)] = find(g)
            roots = sorted({find(e) for e in parent})
            for e in parent:
                table[case, choice, e] = roots.index(find(e))
            count[case, choice] = len(roots)
    return table, count


TABLE, COUNT = _component_table()


def _slopes(v, nodes):
    """At the nodes (three index arrays): the gradient of `v` by central differences (per sample step;
    one-sided at the array's edge), and how far the field bends there: the squared difference of the
    forward and backward differences over the gradient's squared length (0 where it is straight)."""
    g = np.empty((len(nodes[0]), 3), dtype=np.float64)
    bend = np.zeros(len(nodes[0]))
    here = v[nodes].astype(np.float64)
    for b in range(3):
        n = v.shape[b]
        up, down = list(nodes), list(nodes)
        up[b] = np.minimum(nodes[b] + 1, n - 1)
        down[b] = np.maximum(nodes[b] - 1, 0)
        ahead, behind = v[tuple(up)].astype(np.float64) - here, here - v[tuple(down)]
        g[:, b] = (ahead + behind) / (up[b] - down[b])
        both = (up[b] > nodes[b]) & (down[b] < nodes[b])
        bend += np.where(both, (ahead - behind) ** 2, 0.0)
    return g, bend / np.maximum(np.einsum("ij,ij->i", g, g), 1e-30)


def _on_plane(own, pieces, planes, points, normals, chunk=500_000):
    """For each (piece, plane crossing) pair: how well the piece's own crossings lie on that crossing's
    plane, exp(-(nearest one's distance / ON_PLANE)^2) (distances in voxels)."""
    own = own.tocsr()
    out = np.empty(len(pieces))
    for s in range(0, len(pieces), chunk):
        p, j = pieces[s:s + chunk], planes[s:s + chunk]
        rows = own[p]  # the pairs' own crossings, row by row
        pair = np.repeat(np.arange(len(p)), np.diff(rows.indptr))
        gap = np.abs(np.einsum("ij,ij->i", normals[j][pair], points[rows.indices] - points[j][pair]))
        nearest = np.full(len(p), np.inf)
        np.minimum.at(nearest, pair, gap)
        out[s:s + chunk] = np.exp(-(nearest / ON_PLANE) ** 2)
    return out


def _fit(pieces, cols, weight, points, normals, mean):
    """Each vertex's least-squares point for the planes of the crossings `cols` given to it (`pieces`:
    vertex numbers, one per plane) with these weights, pulled towards its `mean` with HOLD x its
    strongest direction (a direction the planes fix firmly, a face or both faces at an edge, is
    followed; a weak one only partly); and its RMS distance to those planes (voxels)."""
    count = len(mean)
    n, d = normals[cols], np.einsum("ij,ij->i", normals[cols], points[cols])
    sums = lambda x: np.bincount(pieces, weights=x, minlength=count)
    M = np.empty((count, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            M[:, i, j] = M[:, j, i] = sums(weight * n[:, i] * n[:, j])
    rhs = np.stack([sums(weight * n[:, i] * d) for i in range(3)], axis=1)
    total, square = sums(weight), sums(weight * d * d)
    lam, Q = np.linalg.eigh(M)
    hold = HOLD * np.maximum(lam[:, 2:3], 1e-12)
    r = rhs - np.einsum("kij,kj->ki", M, mean)
    x = mean + np.einsum("kij,kj->ki", Q, np.einsum("kji,kj->ki", Q, r) / (np.maximum(lam, 0.0) + hold))
    miss = np.einsum("ki,kij,kj->k", x, M, x) - 2 * np.einsum("ki,ki->k", x, rhs) + square
    return x, np.sqrt(np.maximum(miss, 0.0) / np.maximum(total, 1e-30))


def _split(x, quads, out):
    """Each quad (four vertex numbers, in turn round its edge) as two triangles, split along the
    diagonal whose triangles face best along `out` (the field's normal at the quad's crossing; the
    other split can fold over at a corner); and which triangles still face against it."""
    a, b, c, d = (quads[:, k] for k in range(4))

    def facing(p, q, s):  # how well triangle p q s faces out (its unit normal along the edge)
        n = np.cross(x[q] - x[p], x[s] - x[p])
        return np.einsum("ij,ij->i", n, out) / np.maximum(np.linalg.norm(n, axis=1), 1e-30)

    f_abc, f_acd, f_abd, f_bcd = facing(a, b, c), facing(a, c, d), facing(a, b, d), facing(b, c, d)
    on_ac = np.minimum(f_abc, f_acd) >= np.minimum(f_abd, f_bcd)
    faces = np.concatenate([
        np.where(on_ac[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1)),
        np.where(on_ac[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1)),
    ])
    folded = np.concatenate([np.where(on_ac, f_abc, f_abd), np.where(on_ac, f_acd, f_bcd)]) < 0
    return faces, folded


def _tunnels(quads):
    """(quads, 4) bool: whether the quad's edge from its corner k to k + 1 is shared by more than two
    quads. A piece that crosses a cell face twice (two arcs) next to one that does too: both arcs join
    the same two vertices, so four quads share that edge."""
    nxt = np.roll(quads, -1, axis=1)
    key = np.minimum(quads, nxt).astype(np.int64) * (int(quads.max()) + 1) + np.maximum(quads, nxt)
    _, inv, count = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    return (count[inv] > 2).reshape(quads.shape)


def _arc_vertices(quads, tunnel, inner, points, count, emit):
    """For the quads with an edge shared too often (see _tunnels): a vertex per arc on the cell face
    under that edge, at the mean of the arc's crossings (the quads' crossings there that share an inside
    sample: a face's inside corners are kept apart, so each arc goes round one), put between the edge's
    two vertices; each of those quads that `emit` marks becomes a fan of triangles round it. Returns
    the arcs ((lower vertex, upper vertex, inside sample) per row), their vertices (numbered from
    `count` on in the triangles) and the triangles."""
    q, k = np.nonzero(tunnel)
    a, b = quads[q, k], quads[q, (k + 1) % 4]
    arcs = defaultdict(list)  # (edge's lower vertex, its upper one, inside sample) -> the quads
    for qq, lo_v, hi_v in zip(q, np.minimum(a, b), np.maximum(a, b)):
        arcs[(int(lo_v), int(hi_v), int(inner[qq]))].append(qq)
    number = {arc: count + i for i, arc in enumerate(arcs)}
    at = {(int(qq), int(kk)): number[(int(min(lo_v, hi_v)), int(max(lo_v, hi_v)), int(inner[qq]))]
          for qq, kk, lo_v, hi_v in zip(q, k, a, b)}
    triangles = []
    for qq in np.flatnonzero(tunnel.any(axis=1) & emit):
        ring = []
        for kk in range(4):
            ring.append(int(quads[qq, kk]))
            if (int(qq), kk) in at:
                ring.append(at[(int(qq), kk)])
        start = next(i for i, v in enumerate(ring) if v >= count)  # a fan round the first arc vertex
        ring = ring[start:] + ring[:start]
        triangles += [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]
    new = np.array([points[members].mean(axis=0) for members in arcs.values()]).reshape(-1, 3)
    return (np.array(list(arcs), dtype=np.int64).reshape(-1, 3), new,
            np.array(triangles, dtype=np.int64).reshape(-1, 3))


def dual_contour(values, grid, debug=None):
    """The zero level of `values` (numpy or CuPy, inside negative) as a closed mesh in world
    coordinates, like meshing.field_to_mesh (the same checks and `metadata`), with sharp edges kept.
    `debug`: a dict that gets per-vertex internals (for measuring)."""
    values, nonfinite = finite_values(to_host(values), grid.voxel)
    clipped = touches_box(values)
    if float(values.min()) >= 0:
        raise ValueError("Field has no surface inside the grid (no sign change): nothing is inside, or the body "
                         "is thinner than the voxel size")
    if float(values.max()) <= 0:
        raise ValueError("Field has no surface inside the grid (no sign change): the body fills the whole box it "
                         "is meshed in (more Padding, or a smaller offset?)")
    values = off_zero(values, grid.voxel)
    # outside all round, so surfaces touching the grid's box still close
    v = np.pad(values, 1, mode="constant", constant_values=max(float(values.max()), grid.voxel))
    del values
    with trace.span("dual contour", device="cpu"):
        dc = _vertices(v, grid.voxel, debug)
        del v
        faces, _, _, new, info = _faces(dc.x, dc.lo, dc.mean, dc.quads, dc.normals, dc.points, dc.inner)
    if debug is not None:
        debug.update(info)
    verts = grid.origin + (np.concatenate([dc.x, new]) - 1.0) * grid.voxel  # (the padding moved every index by one)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.metadata.update(nonfinite=nonfinite, clipped=clipped, voxel=grid.voxel, folded=info["folded"])
    return mesh


def off_zero(values, voxel):
    """Samples (almost) exactly zero moved off it (as meshing.field_to_mesh does), float32."""
    eps = np.float32(0.01 * voxel)  # float32: a Python float made the inner where float64
    return np.where(np.abs(values) < eps, np.where(values < 0, -eps, eps), values).astype(np.float32)


def _vertices(v, voxel, debug=None):
    """The vertices of the zero level of the samples `v` (inside negative, outside on every outer
    sample), in v's index units: a vertex per piece of surface in each cell, and a quad per crossed
    grid edge. Returns a SimpleNamespace: x, mean (of its crossings), lo (its cell's lower corner),
    cell (its cell's flat index in v's cells) per vertex; quads (its four vertices, facing out),
    normals, points and inner (the inside sample's flat index in v) per crossing."""
    inside = v < 0
    shape = np.array(v.shape)
    cells = shape - 1

    points, normals, weights, slopes, cell_ids, local, lower_inside, inner = [], [], [], [], [], [], [], []
    for a in range(3):
        p, q = [b for b in range(3) if b != a]
        u, w = (p, q) if a != 1 else (q, p)  # (u, w, a) right-handed: the quad goes round +a
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        nodes = np.nonzero(inside[tuple(lo)] != inside[tuple(hi)])
        upper = list(nodes)
        upper[a] = nodes[a] + 1
        v0, v1 = v[nodes].astype(np.float64), v[tuple(upper)].astype(np.float64)
        t = v0 / (v0 - v1)
        point = np.stack(nodes, axis=1).astype(np.float64)
        point[:, a] += t
        (g0, bend0), (g1, bend1) = _slopes(v, nodes), _slopes(v, tuple(upper))
        g = (1 - t)[:, None] * g0 + t[:, None] * g1
        length = np.linalg.norm(g, axis=1)
        slopes.append(length / voxel)
        normals.append(g / np.maximum(length, 1e-30)[:, None])
        weights.append(np.where(length > 0, np.exp(-np.maximum(bend0, bend1) / TRUST_SCALE), 0.0))
        points.append(point)
        lower_inside.append(inside[nodes])
        inner.append(np.where(inside[nodes], np.ravel_multi_index(nodes, v.shape),
                              np.ravel_multi_index(tuple(upper), v.shape)))  # the edge's inside sample
        # the four cells around the edge, in turn round +a; the edge's number in each
        ids, edge = [], []
        for du, dw in ((0, 0), (1, 0), (1, 1), (0, 1)):
            c = list(nodes)
            c[u] = nodes[u] - du
            c[w] = nodes[w] - dw
            ids.append((c[0] * cells[1] + c[1]) * cells[2] + c[2])
            d = {u: du, w: dw}
            edge.append(np.full(len(t), 4 * a + d[p] + 2 * d[q], dtype=np.int64))
        cell_ids.append(np.stack(ids, axis=1))
        local.append(np.stack(edge, axis=1))
    points, normals, weights = np.concatenate(points), np.concatenate(normals), np.concatenate(weights)
    slopes = np.concatenate(slopes)
    cell_ids, local, lower_inside = np.concatenate(cell_ids), np.concatenate(local), np.concatenate(lower_inside)
    inner = np.concatenate(inner)

    # the surface cells, their sign cases and their vertices (one per piece of surface)
    total = int(np.prod(cells))
    if total <= DENSE_CELLS:
        used = np.zeros(total, dtype=bool)
        used[cell_ids.ravel()] = True
        surface = np.flatnonzero(used)
        number = np.empty(total, dtype=np.int64)
        number[surface] = np.arange(len(surface))
        which = number[cell_ids]
        del used, number
    else:
        surface, which = np.unique(cell_ids, return_inverse=True)
        which = which.reshape(cell_ids.shape)
    corner = np.stack(np.unravel_index(surface, tuple(cells)), axis=1)
    case = np.zeros(len(surface), dtype=np.int64)
    value = np.empty((len(surface), 8))
    for c in range(8):
        at = corner + np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1])
        value[:, c] = v[at[:, 0], at[:, 1], at[:, 2]]
        case |= inside[at[:, 0], at[:, 1], at[:, 2]].astype(np.int64) << c
    # faces with four crossings: do the inside corners join across the middle? As marching cubes decides
    # it (the asymptotic decider): whether the bilinear surface's saddle is inside. The same for both
    # cells that share the face.
    choice = np.zeros(len(surface), dtype=np.int64)
    for f, (fc, _) in enumerate(FACES):
        v0, v1, v2, v3 = (value[:, c] for c in fc)
        crossed = (v0 < 0) == (v2 < 0)
        crossed &= ((v1 < 0) == (v3 < 0)) & ((v0 < 0) != (v1 < 0))
        den = v0 + v2 - v1 - v3
        saddle = (v0 * v2 - v1 * v3) / np.where(den == 0, 1.0, den)
        choice |= (crossed & (den != 0) & (saddle < 0)).astype(np.int64) << f
    first = np.concatenate([[0], np.cumsum(COUNT[case, choice])])
    piece = TABLE[case[which], choice[which], local]
    if (piece < 0).any():
        raise AssertionError("a crossing in a cell whose sign case does not cross that edge")
    vid = first[which] + piece  # (crossings, 4)
    count = int(first[-1])

    # each vertex's planes: sums for its least-squares fit; the mean of its own crossings
    flat = vid.ravel()
    crossing = np.repeat(np.arange(len(points)), 4)
    hits = np.bincount(flat, minlength=count).astype(np.float64)
    mean = np.stack([np.bincount(flat, weights=points[crossing, i], minlength=count) for i in range(3)], 1) / hits[:, None]
    # every piece: the fit of its own crossings' planes (on a smooth surface that is all it needs)
    x, _ = _fit(flat, crossing, np.ones(len(flat)), points, normals, mean)
    # pieces beside a crease (an untrusted crossing): the fit of the planes up to RINGS steps away along
    # the surface (a crossing's quad joins four pieces), weighted by trust, distance and how well the
    # piece's own crossings lie on each; taken where those planes meet cleanly, on a true edge or corner
    # (on a surface that only curves tightly, a fillet or a smoothed texture, they do not)
    is_crease = np.bincount(flat, weights=weights[crossing] < CREASE_TRUST, minlength=count) > 0
    crease = np.flatnonzero(is_crease)
    sharp = np.zeros(count, dtype=bool)
    if len(crease):
        own = sparse.csr_matrix((np.ones(len(flat)), (flat, crossing)), shape=(count, len(points)))
        # a first look at the field's slope around them, cheap (products with vectors, not the rings
        # below): the straight crossings the same RINGS steps away, counted as often as the steps reach
        # them. Where it is far from 1 (a texture), the fit below would be refused anyway; where there is
        # no straight crossing to tell, it is refused too (a texture's dense ridges: 99% of the wave's
        # sharp vertices, and all of them that came out over a quarter voxel off, had none; every one on
        # the vent case's edges and holes had some).
        level = weights >= STRAIGHT
        off, seen = own @ np.where(level, np.abs(slopes - 1.0), 0.0), own @ level.astype(np.float64)
        for _ in range(RINGS):
            off, seen = own @ (own.T @ off), own @ (own.T @ seen)
        guess = off[crease] / np.maximum(seen[crease], 1e-30)
        if debug is not None:
            debug.update(guess=np.where(seen[crease] > 0, guess, np.nan), guessed=crease)
        crease = crease[(seen[crease] > 0) & (guess < GUESS * DISTANCE_FIELD)]
    for start in range(0, len(crease), CREASE_CHUNK):  # (a few at a time: their neighbourhoods are big)
        part = crease[start:start + CREASE_CHUNK]
        near = own[part]
        for _ in range(RINGS):  # the pieces sharing a crossing with those reached, and their crossings
            near = near @ own.T @ own
        near = near.tocoo()
        rows, cols = near.row, near.col  # (rows: the pieces' places in `part`)
        del near
        # a neighbour's plane counts as far as the piece's own crossings lie on it: beside a crease, a
        # piece on one face must not take the other face's plane
        on = _on_plane(own, part[rows], cols, points, normals)
        gap = points[cols] - mean[part][rows]
        weight = np.maximum(weights[cols], 1e-6) * on * np.exp(-np.einsum("ij,ij->i", gap, gap) / REACH ** 2)
        del on, gap
        edge, miss = _fit(rows, cols, weight, points, normals, mean[part])
        # ...and only on a distance field (slope 1: a CAD body, its Booleans and holes), whose creases are
        # sharp; a texture's ridges may be rounded finer than a voxel, which the samples cannot tell from
        # sharp, and their sides' planes meet above them (a smoothed wave came out 0.14 mm off with them,
        # 0.06 without). Judged on the clearly straight planes that matter (a tenth of the strongest or
        # more): near a crease a distance field's differences come out shorter too.
        strongest = np.zeros(len(part))
        np.maximum.at(strongest, rows, weight)
        straight = (weight >= 0.1 * strongest[rows]) & (weights[cols] >= STRAIGHT)
        rows, cols, weight = rows[straight], cols[straight], weight[straight]
        bent = np.bincount(rows, weights=weight * np.abs(slopes[cols] - 1.0), minlength=len(part))
        judged = np.bincount(rows, weights=weight, minlength=len(part))
        bent /= np.maximum(judged, 1e-30)
        ok = (miss < CREASE_FIT) & (judged > 0) & (bent < DISTANCE_FIELD)
        sharp[part[ok]] = True
        x[part[ok]] = edge[ok]
        del rows, cols, weight
    # in its cell
    owner = np.repeat(np.arange(len(surface)), COUNT[case, choice])
    lo = corner[owner].astype(np.float64)
    if debug is not None:
        debug.update(crease=is_crease, sharp=sharp, fit=x.copy(), cell=lo, mean=mean)
    reach = np.where(sharp, MARGIN, PLAIN_MARGIN)[:, None]
    x = np.clip(x, lo - reach, lo + 1.0 + reach)
    # a quad per crossing, facing from the inside sample to the outside one
    quads = np.where(lower_inside[:, None], vid, vid[:, ::-1])
    return SimpleNamespace(x=x, mean=mean, lo=lo, cell=surface[owner], quads=quads, normals=normals,
                           points=points, inner=inner)


def _faces(x, lo, mean, quads, normals, points, inner, emit=None):
    """The quads as triangles (all, or those `emit` marks: the others only help find edges shared too
    often). Where a triangle still faces against the field's normal at its crossing (a sliver folded over
    at a crease: vertices that left their cells crossed over), its vertices go back into their cells,
    and then to their crossings' mean (x is changed in place). Returns (triangles, how far each vertex
    was moved: 0, 1 = into its cell, 2 = to the mean, the arcs and arc vertices of _arc_vertices
    (numbered from len(x) on in the triangles), counts for `debug`)."""
    tunnel = _tunnels(quads)
    crossed = tunnel.any(axis=1)
    if emit is None:
        emit = np.ones(len(quads), dtype=bool)
    plain, out = quads[emit & ~crossed], normals[emit & ~crossed]
    stage = np.zeros(len(x), dtype=np.int8)
    for level, fallback in ((1, lambda v: np.clip(x[v], lo[v] + INSET, lo[v] + 1.0 - INSET)), (2, lambda v: mean[v])):
        faces, folded = _split(x, plain, out)
        if not folded.any():
            break
        moved = np.unique(faces[folded])
        x[moved] = fallback(moved)
        stage[moved] = level
    else:
        faces, folded = _split(x, plain, out)
    arcs, new = np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))
    if (crossed & emit).any():
        arcs, new, more = _arc_vertices(quads, tunnel, inner, points, len(x), emit)
        faces = np.concatenate([faces, more])
    return faces, stage, arcs, new, dict(folded=int(folded.sum()), tunnels=int((crossed & emit).sum()))


# --- in blocks (tiles.mesh_tiled) -----------------------------------------------------------------
BLOCK_HALO = 12  # samples around each block: a vertex depends on samples up to ~RINGS + 3 cells away, and a
                 # block's quads use the vertices of the cells just outside it (a multiple of tiles.ALIGN)
PIECES = 4       # pieces of surface a cell can hold at most: a vertex is named cell number x PIECES + piece


def contour_block(values, start, block, b, blocks, voxel):
    """Dual contouring of block `b` of a box of `blocks` blocks of `block` cells (tiles.mesh_tiled).
    `values`: the samples from global sample index `start`, BLOCK_HALO more than the block on every side
    (outside the box already outside). The block keeps the vertices of the cells it owns and the quads
    of the grid edges it owns (a cell, or an edge by its lower sample, belongs to block
    clip(index // block, 0, blocks - 1): the box's outer layer of cells to its outermost blocks), named
    by their global cell, so the blocks' pieces join by name (join_blocks), not by position. Everything
    a kept vertex or quad depends on lies inside the halo, so they come out as on one grid. Returns a
    dict (positions in global sample units): the vertices it owns (names, x, mean, stage: how far its
    folds moved them, see _faces), the moves it asks of others' vertices, its triangles (vertex names;
    arc vertices as -1 - their row in `arcs`), the arcs (two vertex names and the inside sample's
    number) with their vertices, and `needs`: the other blocks whose vertices or quads these use."""
    with trace.span("dual contour", device="cpu"):
        return _contour_block(values, start, block, b, blocks, voxel)


def _contour_block(values, start, block, b, blocks, voxel):
    start, blocks, b = (np.asarray(a, dtype=np.int64) for a in (start, blocks, b))
    dims = blocks * block + 2                  # cells -1 .. blocks x block per side
    v = np.pad(values, 1, mode="constant", constant_values=voxel)  # (the same in every block)
    dc = _vertices(v, voxel)
    shift = start - 1                          # index in v -> global index
    cell = np.stack(np.unravel_index(dc.cell, tuple(np.array(v.shape) - 1)), axis=1) + shift
    piece = np.arange(len(dc.x)) - np.searchsorted(dc.cell, dc.cell)
    name = (((cell[:, 0] + 1) * dims[1] + cell[:, 1] + 1) * dims[2] + cell[:, 2] + 1) * PIECES + piece
    owner = lambda index: np.clip(index // block, 0, blocks - 1)
    mine_v = (owner(cell) == b).all(axis=1)
    lower = np.floor(dc.points).astype(np.int64) + shift  # each crossing's lower sample
    mine_q = (owner(lower) == b).all(axis=1)
    # the blocks this one's quads and cells reach into: a quad joins the vertices of four cells, and a
    # cell's vertex is joined by the quads of all its crossed edges
    reach = mine_q | mine_v[dc.quads].any(axis=1)
    used = np.concatenate([owner(lower[reach]), owner(cell[dc.quads[reach]].reshape(-1, 3))])
    used = np.unique((used[:, 0] * blocks[1] + used[:, 1]) * blocks[2] + used[:, 2])
    needs = [tuple(int(i) for i in n) for n in np.stack(np.unravel_index(used, tuple(blocks)), axis=1)
             if (n != b).any()]

    x0 = dc.x.copy()
    faces, stage, arcs, new, info = _faces(dc.x, dc.lo, dc.mean, dc.quads, dc.normals, dc.points, dc.inner,
                                           emit=mine_q)
    count = len(dc.x)
    plain = faces < count
    asked = np.zeros(count, dtype=bool)
    asked[faces[plain]] = True
    asked &= ~mine_v & (stage > 0)
    kept = np.unique(faces[~plain]) - count    # the arcs its triangles use
    row = np.full(len(arcs), -1, dtype=np.int64)
    row[kept] = np.arange(len(kept))
    triangles = np.empty_like(faces)
    triangles[plain] = name[faces[plain]]
    triangles[~plain] = -1 - row[faces[~plain] - count]
    arcs = arcs[kept]
    ends = np.sort(name[arcs[:, :2]], axis=1)
    side = dims + 1                            # samples -1 .. blocks x block + 1 per side
    s = np.stack(np.unravel_index(arcs[:, 2], v.shape), axis=1) + shift + 1
    return dict(names=name[mine_v], x=x0[mine_v] + shift, mean=dc.mean[mine_v] + shift, stage=stage[mine_v],
                asked=name[asked], asked_stage=stage[asked], triangles=triangles,
                arcs=np.column_stack([ends, (s[:, 0] * side[1] + s[:, 1]) * side[2] + s[:, 2]]),
                arc_x=new[kept] + shift, needs=needs, folded=info["folded"])


def join_blocks(pieces, block, blocks):
    """The vertices (global sample units) and triangles of contour_block's pieces (a list, emptied on
    the way): each vertex from the block that owns it, moved as far as any block's folds asked; the arc
    vertices, which neighbouring blocks may both make, once."""
    with trace.span("join blocks", device="cpu", blocks=len(pieces)):
        return _join_blocks(pieces, block, blocks)


def _join_blocks(pieces, block, blocks):
    dims = np.asarray(blocks, dtype=np.int64) * block + 2
    take = lambda key: np.concatenate([p[key] for p in pieces])
    names = take("names")
    order = np.argsort(names)
    names = names[order]
    if (names[1:] == names[:-1]).any():
        raise AssertionError("a vertex made by two blocks")
    x, mean, stage = take("x")[order], take("mean")[order], take("stage")[order]
    del order

    def find(wanted):
        at = np.minimum(np.searchsorted(names, wanted), max(len(names) - 1, 0))
        if len(wanted) and (names[at] != wanted).any():
            raise AssertionError("a block uses a vertex of a block that was not computed")
        return at

    np.maximum.at(stage, find(take("asked")), take("asked_stage"))
    cell = np.stack(np.unravel_index(names // PIECES, tuple(dims)), axis=1) - 1.0
    x = np.where((stage == 1)[:, None], np.clip(x, cell + INSET, cell + 1.0 - INSET), x)
    x = np.where((stage == 2)[:, None], mean, x)
    del mean, cell
    keys, arc_x = take("arcs"), take("arc_x")
    number = np.zeros(0, dtype=np.int64)
    if len(keys):
        keys, first, number = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        arc_x, number = arc_x[first], number.ravel()
    triangles, done = [], 0
    while pieces:
        p = pieces.pop(0)
        t = p["triangles"]
        out = np.empty_like(t)
        plain = t >= 0
        out[plain] = find(t[plain])
        out[~plain] = len(names) + number[done - 1 - t[~plain]]
        done += len(p["arcs"])
        triangles.append(out)
    return np.concatenate([x, arc_x]), np.concatenate(triangles) if triangles else np.zeros((0, 3), np.int64)
