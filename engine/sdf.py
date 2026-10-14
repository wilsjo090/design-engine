"""Signed distance fields from triangle meshes (negative inside, positive outside)."""
from collections import OrderedDict

import igl
import numpy as np
import trimesh

from .grid import upsample_linear

_WINDING = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
_CHUNK = 2_000_000


class _Cracks:
    """Open edges of a mesh (e.g. T-junctions at fillet corners). The pseudonormal sign is only
    unreliable for query points whose closest surface point lies on one of these edges."""

    def __init__(self, mesh, tol=1e-3):
        edges = mesh.edges_sorted[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        self.a = mesh.vertices[edges[:, 0]]
        self.b = mesh.vertices[edges[:, 1]]
        touched = np.zeros(len(mesh.vertices), dtype=bool)
        touched[edges.ravel()] = True
        self.faces = touched[mesh.faces].any(axis=1)
        self.tol = tol
        self.count = len(edges)

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

    def __init__(self, V, F):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        self.tree = igl.AABB()
        self.tree.init(self.V, self.F)
        self.FN = igl.per_face_normals(self.V, self.F)
        self.VN = igl.per_vertex_normals(self.V, self.F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE)
        self.EN, _, emap = igl.per_edge_normals(self.V, self.F, igl.PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM)[:3]
        self.EMAP = np.asarray(emap).ravel()
        self.cracks = _Cracks(trimesh.Trimesh(self.V, self.F, process=False))
        self._fwn = None  # winding-number tree, built on first need

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


def arrays(*items):
    """Plain arrays from arrays or shared.Shared arrays (those are read from the SSD here)."""
    return [a.get() if hasattr(a, "get") else a for a in items]


def mesh_query(V, F, key=None):
    """Search structures for the mesh (V, F), built once per process for each `key`. V and F
    may be shared.Shared arrays: they are only read when the structures have to be built."""
    if key is None:
        return MeshQuery(*arrays(V, F))
    query = _QUERIES.get(key)
    if query is None:
        query = _QUERIES[key] = MeshQuery(*arrays(V, F))
        while len(_QUERIES) > _MAX_QUERIES:
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
