"""Signed distance fields from triangle meshes (negative inside, positive outside)."""
from collections import OrderedDict

import igl
import numpy as np
import trimesh

from . import shared
from .cache import is_mapped
from .grid import upsample_linear

_WINDING = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
_CHUNK = 2_000_000


class _Cracks:
    """Open edges of a mesh (e.g. T-junctions at fillet corners). The pseudonormal sign is only
    unreliable for query points whose closest surface point lies on one of these edges."""

    def __init__(self, a, b, faces, tol=1e-3):
        self.a, self.b = a, b  # each crack edge's ends
        self.faces = faces     # per triangle: touches a crack
        self.tol = tol
        self.count = len(a)

    @staticmethod
    def arrays(V, F):
        """(a, b, faces) for a mesh (V, F)."""
        sorted_edges = trimesh.Trimesh(V, F, process=False).edges_sorted
        edges = sorted_edges[trimesh.grouping.group_rows(sorted_edges, require_count=1)]
        touched = np.zeros(len(V), dtype=bool)
        touched[edges.ravel()] = True
        return V[edges[:, 0]], V[edges[:, 1]], touched[F].any(axis=1)

    def near(self, face, closest):
        """Mask of points whose closest point (on `face`) lies on a crack edge."""
        out = np.zeros(len(face), dtype=bool)
        cand = np.flatnonzero(self.faces[face])
        if len(cand) == 0:
            return out
        c = closest[cand]
        best = np.full(len(cand), np.inf)
        for a, b in zip(self.a, self.b):
            ab = b - a
            t = np.clip((c - a) @ ab / max(ab @ ab, 1e-30), 0, 1)
            best = np.minimum(best, np.linalg.norm(c - (a + t[:, None] * ab), axis=1))
        out[cand[best < self.tol]] = True
        return out


class MeshQuery:
    """Signed-distance queries against one mesh. The search tree and normals are built once and
    reused: rebuilding them costs ~0.1 s per call, which dominated small (preview) grids."""

    def __init__(self, V, F, prepared=None):
        """`prepared`: query_arrays(V, F), worked out by another process (shared.prepared), else here."""
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        self.tree = igl.AABB()
        self.tree.init(self.V, self.F)
        p = prepared if prepared is not None else query_arrays(self.V, self.F)
        self.FN, self.VN, self.EN, self.EMAP = p["FN"], p["VN"], p["EN"], p["EMAP"]
        self.cracks = _Cracks(p["crack_a"], p["crack_b"], p["crack_faces"])
        self._fwn = None  # winding-number tree, built on first need

    @property
    def nbytes(self):
        """RAM of its own: the arrays not mapped from the shared folder, plus TREE_BYTES per triangle for
        each search tree (libigl's are not numpy arrays)."""
        arrays = sum(a.nbytes for a in (self.V, self.F, self.FN, self.VN, self.EN, self.EMAP) if not is_mapped(a))
        return arrays + TREE_BYTES * len(self.F) * (1 if self._fwn is None else 2)

    def _inside(self, points):
        """Winding-number inside test (robust to cracks). Many grid points can sit in a crack
        vertex's corner region, so this uses a prebuilt tree rather than the exact O(faces)
        per-point formula. Approximate values are fine: only > 0.5 matters."""
        if self._fwn is None:
            self._fwn = igl.FastWindingNumberBVH()
            self._fwn.init(self.V, self.F)
        return self._fwn.winding_number(np.ascontiguousarray(points, dtype=np.float64)) > 0.5

    def _pseudonormal_sign(self, P, face, C):
        """+1 outside / -1 inside, from the normal of the closest feature (face, edge or vertex)
        at the closest point C: the angle-weighted pseudonormal test (Baerentzen & Aanaes)."""
        tri = self.F[face]
        a, b, c = self.V[tri[:, 0]], self.V[tri[:, 1]], self.V[tri[:, 2]]
        v0, v1, v2 = b - a, c - a, C - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denom = d00 * d11 - d01 * d01
        denom[np.abs(denom) < 1e-300] = 1.0
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        bary = np.stack([1.0 - v - w, v, w], axis=1)
        zero = bary < 1e-7
        n_zero = zero.sum(axis=1)

        normal = self.FN[face].copy()
        on_edge = n_zero == 1  # edge k is the one opposite corner k (libigl's EMAP layout)
        if on_edge.any():
            k = np.argmax(zero[on_edge], axis=1)
            normal[on_edge] = self.EN[self.EMAP[face[on_edge] + k * len(self.F)]]
        at_vertex = n_zero >= 2
        if at_vertex.any():
            j = np.argmax(bary[at_vertex], axis=1)
            normal[at_vertex] = self.VN[tri[at_vertex, j]]
        return np.where(np.einsum("ij,ij->i", P - C, normal) < 0, -1.0, 1.0)

    def signed(self, points):
        """Exact signed distance for (N, 3) points, float32."""
        out = np.empty(len(points), dtype=np.float32)
        for s in range(0, len(points), _CHUNK):
            p = np.ascontiguousarray(points[s:s + _CHUNK], dtype=np.float64)
            if self.cracks.count > 500:  # badly broken mesh: winding numbers everywhere
                out[s:s + _CHUNK] = igl.signed_distance(p, self.V, self.F, sign_type=_WINDING)[0]
                continue
            sq, face, closest = self.tree.squared_distance(self.V, self.F, p)
            sign = self._pseudonormal_sign(p, face, closest)
            redo = self.cracks.near(face, closest)
            if redo.any():  # winding number for the points whose nearest feature is a crack
                sign[redo] = np.where(self._inside(p[redo]), -1.0, 1.0)
            out[s:s + _CHUNK] = np.sqrt(sq) * sign
        return out


_QUERIES = OrderedDict()  # per-process cache: mesh key -> MeshQuery
_MAX_QUERIES = 24  # each holds a search tree (~5-10 MB for a 25k-triangle body)
# ... and at most this much in all (the newest is always kept): every new shape (a topology result,
# another texture) is a new key, and 24 medium topology results held ~1 GB per process. Measured
# 345-580 B per triangle in all (test part, key box, a 330k-triangle sphere), the winding-number
# tree of a cracked mesh 70-1100 more.
MAX_QUERY_BYTES = 768 * 2**20
TREE_BYTES = 450


# Meshes from this many triangles get their normals and cracks worked out once for all processes and mapped
# from the shared folder (shared.prepared): per worker ~96 B per triangle kept and ~240 B while finding the
# cracks, ~2.5 GB over 8 workers for a 2.4M-triangle part. Smaller ones (a lamp petal) are quicker built
# than written and read.
PREPARE_TRIANGLES = 100_000


def query_arrays(V, F):
    """A mesh's normals (face, angle-weighted vertex, edge; libigl's edge map) and cracks, for MeshQuery."""
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    EN, _, emap = igl.per_edge_normals(V, F, igl.PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM)[:3]
    a, b, faces = _Cracks.arrays(V, F)
    return {"FN": igl.per_face_normals(V, F),
            "VN": igl.per_vertex_normals(V, F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE),
            "EN": EN, "EMAP": np.asarray(emap).ravel(), "crack_a": a, "crack_b": b, "crack_faces": faces}


def arrays(*items):
    """Plain arrays from arrays or shared.Shared arrays (those are read from the SSD here)."""
    return [a.get() if hasattr(a, "get") else a for a in items]


def _new_query(V, F, key):
    V, F = arrays(V, F)
    prepared = None
    if key is not None and len(F) >= PREPARE_TRIANGLES:
        prepared = shared.prepared("mesh query", key, lambda: query_arrays(V, F))
    return MeshQuery(V, F, prepared)


def mesh_query(V, F, key=None):
    """Search structures for the mesh (V, F), built once per process for each `key`. V and F
    may be shared.Shared arrays: they are only read when the structures have to be built. For a
    big mesh with a key, everything but the search tree is worked out once for all processes."""
    if key is None:
        return MeshQuery(*arrays(V, F))
    query = _QUERIES.get(key)
    if query is None:
        query = _QUERIES[key] = _new_query(V, F, key)
        while len(_QUERIES) > _MAX_QUERIES or (len(_QUERIES) > 1 and
                                               sum(q.nbytes for q in _QUERIES.values()) > MAX_QUERY_BYTES):
            _QUERIES.popitem(last=False)
    else:
        _QUERIES.move_to_end(key)
    return query


def mesh_to_sdf(mesh, grid, band, coarse_factor=None, key=None):
    """Sample a mesh's signed distance on `grid`. `mesh`: a trimesh, or a (vertices, faces) pair.

    Only voxels within `band` mm of the surface get an exact distance. Everything else is
    interpolated from a coarse grid, which keeps the sign correct and the value approximate.
    Make `band` at least as large as the biggest offset you plan to apply.
    `key` (e.g. the mesh fingerprint) lets repeated calls reuse the mesh's search structures.
    """
    query = mesh_query(*(mesh if isinstance(mesh, tuple) else (mesh.vertices, mesh.faces)), key)
    if grid.size <= 20_000:
        return query.signed(grid.points()).reshape(grid.shape)
    if coarse_factor is None:
        # A finer coarse pass narrows the safety margin around the band (fewer exact queries);
        # on big grids the coarse pass itself would get expensive, so go coarser there.
        coarse_factor = 2 if grid.size < 2_000_000 else 4

    coarse = grid.coarser(coarse_factor)
    coarse_voxel = coarse.voxel
    coarse_sdf = query.signed(coarse.points()).reshape(coarse.shape)

    sdf = upsample_linear(coarse_sdf, coarse_factor, grid.shape)

    near = np.abs(sdf) < band + coarse_voxel * np.sqrt(3)
    sdf[near] = query.signed(grid.points(near))
    return sdf
