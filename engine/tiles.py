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
from .dualcontour import BLOCK_HALO, contour_block, join_blocks, off_zero
from .fields import Context, halo as field_halo, mesh_triangles, release_mesh_caches
from .grid import Grid
from .meshing import decimate, finite_values, reduce, signed_volume, simplify, snap_float32

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

    def eval_grid(self, b, extra=0):
        """Samples of block b (BLOCK + 1 per side, sharing the outer ones with its neighbours)
        plus the halo, and `extra` more samples on every side (a multiple of ALIGN)."""
        n = self.block + 1 + 2 * (self.halo + extra)
        return Grid.block(self.base, self.voxel, [i * self.block - self.halo - extra for i in b], (n, n, n))


def worker_memory(plan, field, dual=False):
    """Bytes one worker needs for this body's blocks: the block's field arrays plus search
    structures for the CAD meshes the field reads (the same in every worker: a 2.4M-triangle part
    took ~1.2 GB per worker on the CPU, 0.8 GB on the GPU, so big parts get fewer workers); dual
    contouring's block is bigger by its halo and needs more per sample."""
    n = plan.block + 1 + 2 * (plan.halo + (BLOCK_HALO if dual else 0))
    return int(WORKER_BASE + n ** 3 * (DUAL_BYTES if dual else 4 * 12) + MESH_BYTES * mesh_triangles(field))


WORKER_BASE = int(0.6 * 2**30)  # a worker process itself: libraries plus a CUDA context
MESH_BYTES = 500      # RAM per triangle per worker for distance-field search structures
MESH_GPU_BYTES = 150  # GPU memory per triangle per worker (Warp BVH, normals)
DUAL_BYTES = 200      # RAM per sample of a dual contouring block (its field arrays and the crossings' data)


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


def mesh_blocks(field, plan, blocks, keep, gpu, tolerance=0.0, close=True):
    """Mesh a few blocks (runs in a worker). One result per block: (block, vertices in global
    sample units, faces, border-vertex mask, crossings, (inside, nonfinite)), where
    crossings[axis][side] tells whether the surface runs into the neighbour on that side (side 0 =
    lower), inside[axis][side] whether the body reaches that face of the block, and nonfinite
    counts the NaN and infinite samples (see meshing.finite_values)."""
    xp = backend.cupy() if gpu and backend.gpu_enabled() else np
    try:
        return [_mesh_block(field, plan, b, keep, xp, tolerance, close) for b in blocks]
    finally:
        if xp is not np:
            backend.release_gpu_memory()


def _mesh_block(field, plan, b, keep, xp, tolerance=0.0, close=True):
    values = Context(plan.eval_grid(b), None, xp).eval_full(field)
    h = plan.halo
    if h:
        values = values[h:-h, h:-h, h:-h]
    values, nonfinite = finite_values(backend.to_host(values), plan.voxel)
    # same nudge off zero as meshing.field_to_mesh; identical in every block for a shared sample
    eps = np.float32(0.01 * plan.voxel)  # float32: a Python float made the inner where float64
    values = np.where(np.abs(values) < eps, np.where(values < 0, -eps, eps), values).astype(np.float32)
    faces_ = [(values.take(0, axis=a), values.take(-1, axis=a)) for a in range(3)]
    crossings = [[bool(f.min() < 0 < f.max()) for f in pair] for pair in faces_]
    found = ([[bool(f.min() < 0) for f in pair] for pair in faces_], nonfinite)
    del faces_
    if values.min() > 0 or values.max() < 0:
        return b, None, None, None, crossings, found
    # close the surface on the outer faces of the whole box (inner faces are shared with neighbours)
    pad = [(int(close and b[a] == 0), int(close and b[a] == plan.blocks[a] - 1)) for a in range(3)]
    if any(lo or hi for lo, hi in pad):
        values = np.pad(values, pad, mode="constant", constant_values=max(float(values.max()), plan.voxel))
    verts, faces, _, _ = marching_cubes(values, level=0.0)
    verts = verts.astype(np.float64) + np.array([b[a] * plan.block - pad[a][0] for a in range(3)], dtype=np.float64)
    faces = faces.astype(np.int64)
    if tolerance > 0:  # reduce to a tolerance here, in the worker: borders stay for the neighbours
        verts, faces = reduce(verts, faces, tolerance / plan.voxel, snap=False)
    elif keep < 1.0:
        verts, faces = _simplify(verts, faces, keep, plan.voxel)
    return b, verts, faces.astype(np.int32), _border(faces, len(verts)), crossings, found


def _simplify(verts, faces, keep, voxel):
    """Keep about `keep` of the triangles without moving the block's open border, so it still
    meets the neighbours (same algorithm and settings as meshing.decimate)."""
    # The algorithm does not check that a collapse keeps the surface manifold: in ~1 block of 200
    # it leaves an edge shared by 4 triangles. Such a block is kept as it was (~1% more triangles).
    return simplify(verts, faces, keep, voxel, preserve_border=True) or (verts, faces)


def dual_blocks(field, plan, blocks, gpu):
    """Dual contouring of a few blocks (runs in a worker). One (block, piece) per block: the piece as
    from dualcontour.contour_block (only `needs` where the block has no surface), plus `nonfinite`
    (its own samples that were not finite) and `clipped` (the body reaches the box's outer samples)."""
    xp = backend.cupy() if gpu and backend.gpu_enabled() else np
    try:
        return [_dual_block(field, plan, b, xp) for b in blocks]
    finally:
        if xp is not np:
            backend.release_gpu_memory()


def _dual_block(field, plan, b, xp):
    values = Context(plan.eval_grid(b, BLOCK_HALO), None, xp).eval_full(field)
    h = plan.halo
    if h:
        values = values[h:-h, h:-h, h:-h]
    values = backend.to_host(values)
    start = [i * plan.block - BLOCK_HALO for i in b]
    top = [n * plan.block for n in plan.blocks]  # the box's last sample
    own = tuple(slice(BLOCK_HALO, BLOCK_HALO + plan.block + int(b[a] == plan.blocks[a] - 1)) for a in range(3))
    nonfinite = int(np.count_nonzero(~np.isfinite(values[own])))
    values = off_zero(finite_values(values, plan.voxel)[0], plan.voxel)
    clipped = False
    for a in range(3):
        index = np.arange(values.shape[a]) + start[a]
        for edge in (0, top[a]):  # the box's outer samples: is the body cut off there?
            if index[0] <= edge <= index[-1]:
                clipped |= bool(values.take(edge - start[a], axis=a).min() < 0)
        outside = (index < 0) | (index > top[a])  # beyond the box: outside, so the surface closes on it
        if outside.any():
            values[(slice(None),) * a + (outside,)] = plan.voxel
    found = dict(needs=[], nonfinite=nonfinite, clipped=clipped)
    if values.min() > 0 or values.max() < 0:
        return b, found
    found.update(contour_block(values, start, plan.block, b, plan.blocks, plan.voxel))
    return b, found


def _neighbours(result, plan):
    """The neighbours a marching cubes block's surface runs into."""
    b, crossings = result[0], result[4]
    for a in range(3):
        for side, step in ((0, -1), (1, 1)):
            n = list(b)
            n[a] += step
            if crossings[a][side] and 0 <= n[a] < plan.blocks[a]:
                yield tuple(n)


def _border(faces, count):
    """Mask of vertices on open edges: in a block's mesh, exactly the ones on shared faces."""
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    mask = np.zeros(count, dtype=bool)
    mask[edges[trimesh.grouping.group_rows(edges, require_count=1)].ravel()] = True
    return mask


def stitch(results, plan, tolerance=0.0, close=True):
    """One mesh from the blocks' pieces (a dict, emptied on the way to save memory): border
    vertices that lie on the same grid edge are the same vertex. With a tolerance the blocks were
    reduced with their borders kept: the seams are reduced here (half the tolerance, as they now
    sit next to reduced surface). An open piece (close=False) has no volume to orient it by; marching
    cubes already faces it outward."""
    pieces = [(V, F, B) for _, V, F, B, *_ in results.values() if V is not None and len(F)]
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
    if close and signed_volume(V, F) < 0:
        F = F[:, ::-1]  # same as meshing.field_to_mesh: outward-facing triangles
    mesh = trimesh.Trimesh(plan.base + V * plan.voxel, F, process=False)
    del V, F
    mesh.remove_unreferenced_vertices()
    return snap_float32(mesh)


def mesh_tiled(field, bounds, voxel, padding, keep, map_fn, tolerance=0.0, close=True, surface=None,
               method="marching cubes"):
    """Mesh `field` block by block. `map_fn(fn, arg_list, need=, gpu=, gpu_need=)` runs the block tasks
    (in parallel when it can). Returns (mesh, blocks computed, blocks in the box); `mesh.metadata`
    has `nonfinite`, `clipped` and `voxel` as from meshing.field_to_mesh. close=False leaves the surface
    open where it leaves the box (a piece of a body: the editor's detail view; marching cubes only);
    `surface`: the blocks find_surface gave for the same layout, if already known. Dual contouring
    (`method`) joins its blocks by vertex names (dualcontour.contour_block) and reduces the whole mesh
    here, in this process, for `tolerance` or `keep`."""
    dual = method == "dual contouring"
    if dual and not close:
        raise ValueError("dual contouring in blocks makes closed meshes only")
    plan = Plan(field, bounds, voxel, padding)
    gpu = backend.gpu_enabled()
    done, todo = {}, (find_surface(field, plan) if surface is None else list(surface))
    while todo:
        tasks = [dict(field=field, plan=plan, blocks=todo[i:i + BATCH], gpu=gpu) if dual else
                 dict(field=field, plan=plan, blocks=todo[i:i + BATCH], keep=keep, gpu=gpu, tolerance=tolerance,
                      close=close)
                 for i in range(0, len(todo), BATCH)]
        for results in map_fn(dual_blocks if dual else mesh_blocks, tasks, need=worker_memory(plan, field, dual),
                              gpu=gpu, gpu_need=MESH_GPU_BYTES * mesh_triangles(field)):
            for r in results:
                done[r[0]] = r
        missing = set()
        for b in todo:  # blocks the surface runs into that were not computed: the next round
            missing.update(n for n in (done[b][1]["needs"] if dual else _neighbours(done[b], plan)) if n not in done)
        todo = sorted(missing)
    computed = len(done)
    if dual:
        clipped = any(r[1]["clipped"] for r in done.values())
        nonfinite = sum(r[1]["nonfinite"] for r in done.values())
        folded = sum(r[1].get("folded", 0) for r in done.values())
        pieces = [r[1] for r in done.values() if "names" in r[1]]
        done.clear()
        if not pieces:
            raise ValueError("Field has no surface inside the grid (no sign change)")
        V, F = join_blocks(pieces, plan.block, plan.blocks)
        mesh = trimesh.Trimesh(plan.base + V * plan.voxel, F, process=False)
        del V, F
        mesh.remove_unreferenced_vertices()
        if tolerance > 0:
            mesh = trimesh.Trimesh(*reduce(mesh.vertices, mesh.faces, tolerance), process=False)
        elif keep < 1.0:
            mesh = decimate(mesh, keep, plan.voxel)
        mesh = snap_float32(mesh)
        mesh.metadata["folded"] = folded
    else:
        clipped = any(r[5][0][a][side] for b, r in done.items() for a in range(3) for side in (0, 1)
                      if b[a] == (0 if side == 0 else plan.blocks[a] - 1))
        nonfinite = sum(r[5][1] for r in done.values())
        mesh = stitch(done, plan, tolerance, close)
    mesh.metadata.update(nonfinite=nonfinite, clipped=clipped, voxel=plan.voxel)
    return mesh, computed, plan.count
