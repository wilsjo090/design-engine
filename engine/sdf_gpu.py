"""Signed distance fields on the GPU (NVIDIA Warp), ~50x faster than the CPU path.

Same narrow-band scheme as engine/sdf.py: an exact pass on a coarse grid, trilinear upsampling,
then exact distances only for points within `band` of the surface (Warp's BVH query takes a
maximum distance, so far points are rejected almost for free). Signs come from Warp's
winding-number test, which is robust to the small cracks OpenCascade leaves in tessellations.
Grid points are generated on the GPU, results stay there as CuPy arrays.
"""
from collections import OrderedDict

import igl
import numpy as np
import trimesh
import warp as wp

from . import backend
from .grid import upsample_linear
from .sdf import arrays

wp.config.quiet = True
wp.init()


@wp.kernel
def _sdf_grid(mesh: wp.uint64, indices: wp.array(dtype=int), face_n: wp.array(dtype=wp.vec3),
              vert_n: wp.array(dtype=wp.vec3), edge_n: wp.array(dtype=wp.vec3), emap: wp.array(dtype=int),
              nfaces: int, crack: wp.array(dtype=wp.int8), has_cracks: int,
              origin: wp.vec3, sx: int, sy: int, sz: int, voxel: float, ny: int, nz: int, max_dist: float,
              out: wp.array(dtype=float)):
    """Signed distance for every point of a grid (flattened C order). Points with no surface
    within max_dist keep the value already in `out`.

    Sign: angle-weighted pseudonormal of the closest feature (face, edge or corner), the same
    test as the CPU path; exact on closed surfaces, including ties such as a point on the axis
    of a hole. (Warp's own normal sign uses one face normal and gets such ties wrong.) Points
    whose closest triangle touches a crack in the mesh are redone with the winding number."""
    tid = wp.tid()
    i = tid // (ny * nz)
    j = (tid // nz) % ny
    k = tid % nz
    p = origin + wp.vec3(float(i + sx), float(j + sy), float(k + sz)) * voxel  # origin = grid.base
    q = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    if q.result:
        f = q.face
        c = wp.mesh_eval_position(mesh, f, q.u, q.v)
        b0 = q.u
        b1 = q.v
        b2 = 1.0 - q.u - q.v
        eps = float(1.0e-5)
        zeros = 0
        if b0 < eps:
            zeros += 1
        if b1 < eps:
            zeros += 1
        if b2 < eps:
            zeros += 1
        n = face_n[f]
        if zeros == 1:  # on an edge: edge k is opposite corner k
            corner = 2
            if b0 < eps:
                corner = 0
            elif b1 < eps:
                corner = 1
            n = edge_n[emap[f + corner * nfaces]]
        elif zeros >= 2:  # at a corner: the one whose coordinate is not zero
            corner = 2
            if b0 >= eps:
                corner = 0
            elif b1 >= eps:
                corner = 1
            n = vert_n[indices[f * 3 + corner]]
        s = float(1.0)
        if wp.dot(p - c, n) < 0.0:
            s = -1.0
        if has_cracks != 0:
            if crack[f] != wp.int8(0):
                w = wp.mesh_query_point_sign_winding_number(mesh, p, max_dist, 2.0, 0.5)
                if w.result:
                    s = w.sign
        out[tid] = s * wp.length(p - c)


@wp.kernel
def _nearest_grid(mesh: wp.uint64, origin: wp.vec3, sx: int, sy: int, sz: int, voxel: float, ny: int, nz: int,
                  max_dist: float, face_out: wp.array(dtype=int), closest_out: wp.array(dtype=wp.vec3)):
    """Closest triangle and point on it for every grid point within max_dist (others: face -1)."""
    tid = wp.tid()
    i = tid // (ny * nz)
    j = (tid // nz) % ny
    k = tid % nz
    p = origin + wp.vec3(float(i + sx), float(j + sy), float(k + sz)) * voxel
    q = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    if q.result:
        face_out[tid] = q.face
        closest_out[tid] = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    else:
        face_out[tid] = -1


_MESHES = OrderedDict()  # key -> (wp.Mesh, its point and index arrays): BVH built once per mesh
_MAX_MESHES = 48
MAX_EDGE = 2.0  # mm. Warp works in float32: on long thin triangles (CAD tessellations of flat
                # faces have 50 mm edges) its closest-point maths was off by up to 0.03 mm.
                # Splitting them (geometry unchanged) brings the error to ~0.0005 mm.


class _GpuMesh:
    """A mesh's Warp BVH plus a per-triangle flag for triangles that touch a crack."""

    def __init__(self, V, F):
        V, F, parent = trimesh.remesh.subdivide_to_size(np.asarray(V, dtype=np.float64), np.asarray(F),
                                                        max_edge=MAX_EDGE, max_iter=12, return_index=True)
        F = np.asarray(F)
        self.parent = np.asarray(parent, dtype=np.int64)  # original triangle of each split one
        edges = trimesh.Trimesh(V, F, process=False).edges_sorted
        open_edges = edges[trimesh.grouping.group_rows(edges, require_count=1)]
        on_crack = np.zeros(len(V), dtype=bool)
        on_crack[open_edges.ravel()] = True
        flags = on_crack[F].any(axis=1).astype(np.int8)
        self.has_cracks = int(flags.any())
        # pseudonormals, computed once on the CPU (same as engine/sdf.py)
        V64, F64 = np.ascontiguousarray(V, dtype=np.float64), np.ascontiguousarray(F, dtype=np.int64)
        face_n = igl.per_face_normals(V64, F64)
        vert_n = igl.per_vertex_normals(V64, F64, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE)
        edge_n, _, emap = igl.per_edge_normals(V64, F64, igl.PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM)[:3]
        vec = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float32), dtype=wp.vec3)
        self.face_n, self.vert_n, self.edge_n = vec(face_n), vec(vert_n), vec(edge_n)
        self.emap = wp.array(np.asarray(emap, dtype=np.int32).ravel(), dtype=int)
        self.nfaces = len(F)
        self.points = wp.array(np.ascontiguousarray(V, dtype=np.float32), dtype=wp.vec3)
        self.indices = wp.array(np.ascontiguousarray(F, dtype=np.int32).ravel(), dtype=int)
        self.crack = wp.array(flags if self.has_cracks else np.zeros(1, np.int8), dtype=wp.int8)
        # winding numbers need extra BVH data: only built for meshes that have cracks
        self.mesh = wp.Mesh(points=self.points, indices=self.indices, support_winding_number=bool(self.has_cracks))


def _mesh(V, F, key):
    entry = _MESHES.get(key) if key is not None else None
    if entry is None:
        entry = _GpuMesh(*arrays(V, F))  # Shared arrays are read from the SSD only here
        if key is not None:
            _MESHES[key] = entry
            while len(_MESHES) > _MAX_MESHES:
                _MESHES.popitem(last=False)
    elif key is not None:
        _MESHES.move_to_end(key)
    return entry


def _launch(m, grid, max_dist, out_cupy):
    cp = backend.cupy()
    cp.cuda.Device().synchronize()  # Warp runs on its own stream: let CuPy finish writing `out`
    wp.launch(_sdf_grid, dim=grid.size,
              inputs=[m.mesh.id, m.indices, m.face_n, m.vert_n, m.edge_n, m.emap, m.nfaces, m.crack, m.has_cracks,
                      wp.vec3(*[float(c) for c in grid.base]), *grid.start, float(grid.voxel),
                      grid.shape[1], grid.shape[2],
                      float(max_dist), wp.from_dlpack(out_cupy)])
    wp.synchronize_device()


def nearest_on_grid_gpu(V, F, grid, max_dist, key=None):
    """For grid points within max_dist of the mesh: (flat grid indices, original triangle index,
    closest point), as numpy arrays."""
    mesh = _mesh(V, F, key)
    face = wp.empty(grid.size, dtype=int)
    closest = wp.empty(grid.size, dtype=wp.vec3)
    wp.launch(_nearest_grid, dim=grid.size,
              inputs=[mesh.mesh.id, wp.vec3(*[float(c) for c in grid.base]), *grid.start, float(grid.voxel),
                      grid.shape[1], grid.shape[2], float(max_dist), face, closest])
    wp.synchronize_device()
    face = face.numpy()
    near = np.nonzero(face >= 0)[0]
    return near, mesh.parent[face[near]], closest.numpy()[near].astype(np.float64)


def mesh_to_sdf_gpu(V, F, grid, band, key=None, coarse_factor=None):
    """Signed distance of the mesh (V, F) on `grid` as a CuPy array (negative inside). Outside
    the band the values are interpolated, as on the CPU (same coarse factors, so both agree)."""
    cp = backend.cupy()
    mesh = _mesh(V, F, key)
    if grid.size <= 20_000:  # tiny grid: every point exactly, in one pass
        out = cp.zeros(grid.size, dtype=cp.float32)
        _launch(mesh, grid, 1e9, out)
        return out.reshape(grid.shape)
    if coarse_factor is None:
        coarse_factor = 2 if grid.size < 2_000_000 else 4

    coarse = grid.coarser(coarse_factor)
    coarse_voxel = coarse.voxel
    coarse_sdf = cp.zeros(coarse.size, dtype=cp.float32)
    _launch(mesh, coarse, 1e9, coarse_sdf)
    sdf = upsample_linear(coarse_sdf.reshape(coarse.shape), coarse_factor, grid.shape)
    flat = sdf.reshape(-1)
    _launch(mesh, grid, band + coarse_voxel * np.sqrt(3), flat)  # exact inside the band only
    return flat.reshape(grid.shape)

