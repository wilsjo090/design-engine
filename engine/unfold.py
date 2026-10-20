"""Unfolding CAD faces that join smoothly into one flat sheet, so a face pattern runs across their
borders without a seam (engine/holes.py, Join Faces).

1. `sheets`: the chosen triangles split into sheets, joined across every edge except folds (edges
   between two CAD faces that bend more than a set angle). Vertices on a fold get one copy per
   side, so the sheet is cut open there (a fold's holes are dropped by the edge margin anyway, and
   a box corner could not be flattened without tearing).
2. `closes_around`: a sheet that is not a disc, or a disc with holes, cannot be flattened (the two
   halves of a cylinder, a tube with its fillets): it keeps a pattern per face, which closes around.
   Told apart by the turning of each border loop measured on the surface: 2 pi around a flat
   outline, -2 pi around a hole in it, about 0 around a tube's end.
3. `flatten`: a least-squares conformal map to start (libigl LSCM: angles kept, sizes not), then
   as-rigid-as-possible steps (Liu et al. 2008, "A local/global approach to mesh
   parameterization"): each triangle's best rotation onto its own flat copy, then one sparse solve
   (cotangent weights, factorised once) that puts the vertices where those turned triangles fit
   best. A developable sheet (planes, cylinders, cones: a floor with straight fillets) comes out
   isometric; where the surface curves both ways (a fillet around a corner) the stretch is spread
   thin. On the key box's inside (13 faces, 6400 triangles, rising 69 degrees) 95% of the area
   stretches less than 2.3%, at most 7.6%.
"""
import hashlib
import math

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized

ITERATIONS = 300       # as-rigid-as-possible steps at most
SETTLED = 1e-7         # stop when no vertex moves more than this share of the sheet's size in a step
_FLAT = {}             # sheet fingerprint -> (X, stretch): unfolding again on every edit is wasted work


def sheets(V, F, ids, tri, fold):
    """The triangles `tri` of a body (indices into F; ids: CAD face per triangle of F) as sheets
    joined across every edge except folds between two faces (bend over `fold` radians). Returns a
    list of (triangles of F, (k, 3) corners numbered in the sheet's own vertices, the body vertex
    of each sheet vertex)."""
    tri = np.asarray(tri, dtype=np.int64)
    if not len(tri):
        return []
    Ft = np.asarray(F)[tri]
    sub = trimesh.Trimesh(V, Ft, process=False)       # vertex numbers stay the body's
    pairs, edges, bends = sub.face_adjacency, sub.face_adjacency_edges, sub.face_adjacency_angles
    face = np.asarray(ids)[tri]
    joined = (face[pairs[:, 0]] == face[pairs[:, 1]]) | (bends <= fold)
    a, b, edges = pairs[joined, 0], pairs[joined, 1], edges[joined]
    k = len(tri)
    _, piece = connected_components(sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(k, k)), directed=False)

    # corner (t, j) is node 3 t + j; across a joined edge both triangles' corners at its two ends are
    # one vertex, so a vertex splits where the fan of triangles around it is cut by a fold
    def corner(t, v):
        return 3 * t + np.argmax(Ft[t] == v[:, None], axis=1)
    one = np.concatenate([corner(a, edges[:, 0]), corner(a, edges[:, 1])])
    two = np.concatenate([corner(b, edges[:, 0]), corner(b, edges[:, 1])])
    _, vertex = connected_components(sp.coo_matrix((np.ones(len(one)), (one, two)), shape=(3 * k, 3 * k)),
                                     directed=False)
    out = []
    for p in range(piece.max() + 1):
        mine = np.flatnonzero(piece == p)
        used, local = np.unique(vertex.reshape(-1, 3)[mine].ravel(), return_inverse=True)
        source = np.empty(len(used), dtype=np.int64)
        source[local] = Ft[mine].ravel()
        out.append((tri[mine], local.reshape(-1, 3), source))
    return out


def corner_angles(P):
    """Angle at each corner of triangles P (k, 3, 3): (k, 3)."""
    out = np.zeros(P.shape[:2])
    for j in range(3):
        a, b = P[:, (j + 1) % 3] - P[:, j], P[:, (j + 2) % 3] - P[:, j]
        out[:, j] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
    return out


def closes_around(V, F):
    """Why a connected sheet cannot be unfolded flat, or None if it can: it has no border (closed),
    a handle, or a border loop that is not the outline or a hole of a flat sheet (the end of a tube)."""
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    border = unique[count == 1]
    if not len(border):
        return "they close around completely"
    n = len(V)
    loops, label = connected_components(sp.coo_matrix((np.ones(len(border)), (border[:, 0], border[:, 1])),
                                                      shape=(n, n)), directed=False)
    on = np.unique(border)
    euler = len(np.unique(F)) - len(unique) + len(F)
    found = len(np.unique(label[on]))
    if euler + found != 2:
        return "they form a ring or a handle"
    if found == 1:
        return None
    total = np.bincount(F.ravel(), weights=corner_angles(V[F]).ravel(), minlength=n)
    turning = np.bincount(label[on], weights=math.pi - total[on], minlength=loops)[np.unique(label[on])]
    # the outline turns the most; every other loop must be a hole (about -2 pi), not a tube's end (about 0)
    if np.all(np.delete(turning, np.argmax(turning)) < -math.pi):
        return None
    return "they close around (like the halves of a cylinder)"


def flatten(V, F, check=None):
    """2D positions (mm) of the vertices of a connected sheet with a border (V (n, 3), F (k, 3)),
    stretched as little as possible, and per triangle the largest and smallest factor a length in it
    changes by, (k, 2) (0 for degenerate triangles). Kept per process by the sheet's fingerprint."""
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    key = hashlib.sha1(V.tobytes() + F.tobytes()).hexdigest()
    if key in _FLAT:
        return _FLAT[key]
    n, k = len(V), len(F)
    P = V[F]
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    cross = np.cross(e1, e2)
    twice = np.linalg.norm(cross, axis=1)                     # twice each triangle's area
    good = twice > 1e-12 * max(float(twice.max()), 1e-300)
    # each triangle flat in its own plane, the same way round as seen from outside
    l1 = np.linalg.norm(e1, axis=1)
    x = e1 / np.where(l1 > 0, l1, 1.0)[:, None]
    y = np.cross(np.where(good[:, None], cross / np.where(good, twice, 1.0)[:, None], 0.0), x)
    Q = np.zeros((k, 3, 2))
    Q[:, 1, 0] = l1
    Q[:, 2] = np.column_stack([np.einsum("ij,ij->i", e2, x), np.einsum("ij,ij->i", e2, y)])
    inverse = np.zeros((k, 2, 2))
    inverse[good] = np.linalg.inv(np.stack([Q[good, 1], Q[good, 2]], axis=2))

    # the energy sum w |x_i - x_j - R (q_i - q_j)|^2 over every triangle's edges, w = cot(opposite) / 2
    rows, cols, vals, br, bc, bx, by = [], [], [], [], [], [], []
    for j in range(3):
        i0, i1 = F[:, (j + 1) % 3], F[:, (j + 2) % 3]
        a, b = P[:, (j + 1) % 3] - P[:, j], P[:, (j + 2) % 3] - P[:, j]
        w = np.where(good, np.einsum("ij,ij->i", a, b) / np.where(good, twice, 1.0), 0.0) / 2
        rows += [i0, i1, i0, i1]
        cols += [i1, i0, i0, i1]
        vals += [-w, -w, w, w]
        e = (Q[:, (j + 1) % 3] - Q[:, (j + 2) % 3]) * w[:, None]
        br += [i0, i1]
        bc += [np.arange(k)] * 2
        bx += [e[:, 0], -e[:, 0]]
        by += [e[:, 1], -e[:, 1]]
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    used = np.bincount(F[good].ravel(), minlength=n) > 0     # vertices of degenerate triangles only: filled in
    pin = int(np.flatnonzero(used)[0])                         # held at (0, 0)
    free = np.flatnonzero(used & (np.arange(n) != pin))
    A = L[free][:, free].tocsc()
    solve = factorized(A + sp.identity(len(free), format="csc") * (1e-12 * float(A.diagonal().mean())))
    r, c = np.concatenate(br), np.concatenate(bc)
    ex = sp.csr_matrix((np.concatenate(bx), (r, c)), shape=(n, k))[free]
    ey = sp.csr_matrix((np.concatenate(by), (r, c)), shape=(n, k))[free]

    X = np.zeros((n, 2))
    X[used] = _start(V[used], np.searchsorted(np.flatnonzero(used), F[good]))
    X -= X[pin]
    if np.sum(_signed_twice(X, F)) < 0:      # the plane seen from outside
        X[:, 1] *= -1
    size = float(np.linalg.norm(np.ptp(V, axis=0)))
    for _ in range(ITERATIONS):
        if check is not None:
            check()
        J = _jacobians(X, F, inverse)
        turn = np.arctan2(J[:, 1, 0] - J[:, 0, 1], J[:, 0, 0] + J[:, 1, 1])   # closest rotation
        cos, sin = np.cos(turn), np.sin(turn)
        # R e = (cos e_x - sin e_y, sin e_x + cos e_y), summed into each edge's two vertices
        new = X.copy()
        new[free] = np.column_stack([solve(ex @ cos - ey @ sin), solve(ex @ sin + ey @ cos)])
        moved = float(np.abs(new - X).max())
        X = new
        if moved < SETTLED * size:
            break
    if not used.all():
        from scipy.spatial import cKDTree
        X[~used] = X[used][cKDTree(V[used]).query(V[~used])[1]]
    stretch = np.zeros((k, 2))
    stretch[good] = np.linalg.svd(_jacobians(X, F, inverse)[good], compute_uv=False)
    stretch[good & (_signed_twice(X, F) < 0)] *= -1           # folded over: negative
    _FLAT[key] = (X, stretch)
    while len(_FLAT) > 16:
        _FLAT.pop(next(iter(_FLAT)))
    return X, stretch


def _jacobians(X, F, inverse):
    U = X[F]
    return np.stack([U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]], axis=2) @ inverse


def _signed_twice(X, F):
    a, b = X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _start(V, F):
    """Where the steps start: libigl's least-squares conformal map, two far-apart border vertices
    held at their distance; the sheet projected onto its mean plane if that fails."""
    import igl
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    border = np.unique(unique[count == 1])
    if len(border) >= 2:
        first = border[0]
        far = border[int(np.argmax(np.linalg.norm(V[border] - V[first], axis=1)))]
        gap = float(np.linalg.norm(V[far] - V[first]))
        try:
            X, _ = igl.lscm(V, F, np.array([first, far], dtype=np.int64), np.array([[0.0, 0.0], [gap, 0.0]]))
            X = np.asarray(X, dtype=np.float64)
            if np.isfinite(X).all() and np.ptp(X, axis=0).min() > 1e-6 * gap:
                return X
        except Exception:  # noqa: BLE001 (libigl raises plain RuntimeErrors on bad meshes)
            pass
    _, _, axes = np.linalg.svd(V - V.mean(axis=0), full_matrices=False)
    return (V - V.mean(axis=0)) @ axes[:2].T
