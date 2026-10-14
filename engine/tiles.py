"""Meshing big bodies block by block.

A dense grid over a big part's box does not fit in memory (the lamp center at 0.1 mm is 2.7
billion voxels, 10 GB per field array) and is mostly empty space. Here the box is split into
blocks of BLOCK voxels. A coarse probe finds the blocks near the surface; those are computed
on their own small grids (in the worker processes, on the GPU when it is on), meshed, simplified
with their borders locked, and stitched. A block whose surface runs into a neighbour that was
not computed pulls that neighbour into the next round, so a probe that misses a block cannot
leave a hole.

Seams close exactly because:
- every block computes a shared sample from the same global index (Grid.block), and block
  starts are multiples of ALIGN, so the coarse sampling grids inside noise and distance fields
  coincide between blocks;
- smoothing gets a margin of extra samples around each block (fields.halo);
- border vertices are matched by the grid edge they lie on (integers), not by position.
"""
import numpy as np
import trimesh
from skimage.measure import marching_cubes

from . import backend
from .fields import Context, halo as field_halo, mesh_triangles, release_mesh_caches
from .grid import Grid
from .meshing import reduce, signed_volume, simplify, snap_float32

BLOCK = 96             # voxels per block side; a multiple of ALIGN
ALIGN = 12             # coarse sampling factors (noise 2-4, distance fields 2 and 4) all divide this
TILE_MIN_VOXELS = 32_000_000       # grids bigger than this are meshed block by block
MAX_TILED_VOXELS = 30_000_000_000  # box size limit (the lamp center at 0.05 mm is ~22 billion)
PROBE = 8              # probe samples per block side when looking for the surface
LIPSCHITZ = 2.5        # assumed steepest slope of a body field (distance: 1, textures add some)
BATCH = 6              # blocks per worker task


class Plan:
    """The block layout for meshing `field` inside `bounds` (plain data: sent to the workers)."""

    def __init__(self, field, bounds, voxel, padding):
        grid = Grid.from_bounds(bounds, voxel, padding)
        self.base, self.voxel, self.block = grid.base, grid.voxel, BLOCK
        self.blocks = tuple(max(1, -(-(n - 1) // BLOCK)) for n in grid.shape)  # ceil((n-1)/BLOCK)
        self.halo = -(-field_halo(field, voxel) // ALIGN) * ALIGN  # rounded up to keep alignment
        self.dense = grid.size

    @property
    def count(self):
        return int(np.prod(self.blocks))

    def eval_grid(self, b):
        """Samples of block b (BLOCK + 1 per side, sharing the outer ones with its neighbours)
        plus the halo."""
        n = self.block + 1 + 2 * self.halo
        return Grid.block(self.base, self.voxel, [i * self.block - self.halo for i in b], (n, n, n))


def worker_memory(plan, field):
    """Bytes one worker needs for this body's blocks: the block's field arrays plus search
    structures for the CAD meshes the field reads (the same in every worker: a 2.4M-triangle part
    took ~1.2 GB per worker on the CPU, 0.8 GB on the GPU, so big parts get fewer workers)."""
    return int(WORKER_BASE + (plan.block + 1 + 2 * plan.halo) ** 3 * 4 * 12 + MESH_BYTES * mesh_triangles(field))


WORKER_BASE = int(0.6 * 2**30)  # a worker process itself: libraries plus a CUDA context
MESH_BYTES = 500      # RAM per triangle per worker for distance-field search structures
MESH_GPU_BYTES = 150  # GPU memory per triangle per worker (Warp BVH, normals)


def find_surface(field, plan):
    """Blocks the surface probably passes through, from the field sampled every BLOCK/PROBE
    voxels: a sign change, or a value small enough that the surface could be between samples."""
    step = plan.block // PROBE
    probe = Grid.block(plan.base, plan.voxel * step, (0, 0, 0), [n * PROBE + 1 for n in plan.blocks])
    xp = backend.cupy() if backend.use_gpu_for(probe.size) else np
    try:
        values = backend.to_host(Context(probe, None, xp).eval_full(field))
    finally:
        release_mesh_caches()  # the workers do the real work; don't keep a big part's copy here
        if xp is not np:
            backend.release_gpu_memory()
    near = LIPSCHITZ * step * plan.voxel * np.sqrt(3) / 2  # half a probe cell's diagonal
    found = []
    for b in np.ndindex(*plan.blocks):
        v = values[tuple(slice(i * PROBE, (i + 1) * PROBE + 1) for i in b)]
        if (v.min() < 0 < v.max()) or np.abs(v).min() < near:
            found.append(b)
    return found


def mesh_blocks(field, plan, blocks, keep, gpu, tolerance=0.0):
    """Mesh a few blocks (runs in a worker). One result per block: (block, vertices in global
    sample units, faces, border-vertex mask, crossings), where crossings[axis][side] tells whether
    the surface runs into the neighbour on that side (side 0 = lower)."""
    xp = backend.cupy() if gpu and backend.gpu_enabled() else np
    try:
        return [_mesh_block(field, plan, b, keep, xp, tolerance) for b in blocks]
    finally:
        if xp is not np:
            backend.release_gpu_memory()


def _mesh_block(field, plan, b, keep, xp, tolerance=0.0):
    values = Context(plan.eval_grid(b), None, xp).eval_full(field)
    h = plan.halo
    if h:
        values = values[h:-h, h:-h, h:-h]
    values = backend.to_host(values)
    # same nudge off zero as meshing.field_to_mesh; identical in every block for a shared sample
    eps = 0.01 * plan.voxel
    values = np.where(np.abs(values) < eps, np.where(values < 0, -eps, eps), values).astype(np.float32)
    crossings = [[bool(f.min() < 0 < f.max()) for f in (values.take(0, axis=a), values.take(-1, axis=a))]
                 for a in range(3)]
    if values.min() > 0 or values.max() < 0:
        return b, None, None, None, crossings
    # close the surface on the outer faces of the whole box (inner faces are shared with neighbours)
    pad = [(int(b[a] == 0), int(b[a] == plan.blocks[a] - 1)) for a in range(3)]
    if any(lo or hi for lo, hi in pad):
        values = np.pad(values, pad, mode="constant", constant_values=max(float(values.max()), plan.voxel))
    verts, faces, _, _ = marching_cubes(values, level=0.0)
    verts = verts.astype(np.float64) + np.array([b[a] * plan.block - pad[a][0] for a in range(3)], dtype=np.float64)
    faces = faces.astype(np.int64)
    if tolerance > 0:  # reduce to a tolerance here, in the worker: borders stay for the neighbours
        verts, faces = reduce(verts, faces, tolerance / plan.voxel, snap=False)
    elif keep < 1.0:
        verts, faces = _simplify(verts, faces, keep, plan.voxel)
    return b, verts, faces.astype(np.int32), _border(faces, len(verts)), crossings


def _simplify(verts, faces, keep, voxel):
    """Keep about `keep` of the triangles without moving the block's open border, so it still
    meets the neighbours (same algorithm and settings as meshing.decimate)."""
    # The algorithm does not check that a collapse keeps the surface manifold: in ~1 block of 200
    # it leaves an edge shared by 4 triangles. Such a block is kept as it was (~1% more triangles).
    return simplify(verts, faces, keep, voxel, preserve_border=True) or (verts, faces)


def _border(faces, count):
    """Mask of vertices on open edges: in a block's mesh, exactly the ones on shared faces."""
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    mask = np.zeros(count, dtype=bool)
    mask[edges[trimesh.grouping.group_rows(edges, require_count=1)].ravel()] = True
    return mask


def stitch(results, plan, tolerance=0.0):
    """One mesh from the blocks' pieces (a dict, emptied on the way to save memory): border
    vertices that lie on the same grid edge are the same vertex. With a tolerance the blocks were
    reduced with their borders kept: the seams are reduced here (half the tolerance, as they now
    sit next to reduced surface)."""
    pieces = [(V, F, B) for _, V, F, B, _ in results.values() if V is not None and len(F)]
    results.clear()
    if not pieces:
        raise ValueError("Field has no surface inside the grid (no sign change)")
    offsets = np.cumsum([0] + [len(V) for V, _, _ in pieces])
    V = np.concatenate([p[0] for p in pieces])
    B = np.concatenate([p[2] for p in pieces])
    F = np.concatenate([p[1].astype(np.int64) + o for p, o in zip(pieces, offsets)])
    del pieces

    idx = np.nonzero(B)[0]
    c = V[idx]
    r = np.round(c)
    whole = np.abs(c - r) < 1e-6  # two coordinates are whole sample indices, one lies in between
    axis = np.argmin(whole, axis=1)
    lo = np.where(whole, r, np.floor(c)).astype(np.int64) + 1
    dims = np.array(plan.blocks, dtype=np.int64) * plan.block + 4
    key = ((lo[:, 0] * dims[1] + lo[:, 1]) * dims[2] + lo[:, 2]) * 3 + axis
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    remap = np.arange(len(V))
    remap[idx] = idx[first][inverse.ravel()]

    F = remap[F]
    if tolerance > 0:
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        on_seam = np.zeros(len(V), dtype=bool)  # on an inner block boundary plane
        for a in range(3):
            k = np.round(V[:, a] / plan.block)
            on_seam |= (V[:, a] == k * plan.block) & (k > 0) & (k < plan.blocks[a])
        seam = on_seam[F].any(axis=1)
        if seam.any():
            V, F = reduce(V, F, tolerance / 2 / plan.voxel, frozen=~seam, snap=False)
    if signed_volume(V, F) < 0:
        F = F[:, ::-1]  # same as meshing.field_to_mesh: outward-facing triangles
    mesh = trimesh.Trimesh(plan.base + V * plan.voxel, F, process=False)
    del V, F
    mesh.remove_unreferenced_vertices()
    return snap_float32(mesh)


def mesh_tiled(field, bounds, voxel, padding, keep, map_fn, tolerance=0.0):
    """Mesh `field` block by block. `map_fn(fn, arg_list, need=, gpu=, gpu_need=)` runs the block tasks
    (in parallel when it can). Returns (mesh, blocks computed, blocks in the box)."""
    plan = Plan(field, bounds, voxel, padding)
    gpu = backend.gpu_enabled()
    done, todo = {}, find_surface(field, plan)
    while todo:
        tasks = [dict(field=field, plan=plan, blocks=todo[i:i + BATCH], keep=keep, gpu=gpu, tolerance=tolerance)
                 for i in range(0, len(todo), BATCH)]
        for results in map_fn(mesh_blocks, tasks, need=worker_memory(plan, field), gpu=gpu,
                              gpu_need=MESH_GPU_BYTES * mesh_triangles(field)):
            for r in results:
                done[r[0]] = r
        missing = set()
        for b in todo:
            for a in range(3):
                for side, step in ((0, -1), (1, 1)):
                    n = list(b)
                    n[a] += step
                    if done[b][4][a][side] and 0 <= n[a] < plan.blocks[a] and tuple(n) not in done:
                        missing.add(tuple(n))
        todo = sorted(missing)
    computed = len(done)
    return stitch(done, plan, tolerance), computed, plan.count
