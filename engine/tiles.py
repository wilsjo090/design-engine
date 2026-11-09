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

from . import backend, blockmesh, stages, trace
from .blockmesh import dual_block_values as _dual_block_values, mesh_block_values as _mesh_block_values
from .dualcontour import BLOCK_HALO, join_blocks
from .fields import Context, halo as field_halo, mesh_triangles, release_mesh_caches
from .grid import Grid
from .meshing import FIELD_NORMAL, decimate, reduce, signed_volume, snap_float32

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


def worker_memory(plan, field, dual=False, normals=False):
    """Bytes one worker needs for this body's blocks: the block's field arrays plus search
    structures for the CAD meshes the field reads (the same in every worker: a 2.4M-triangle part
    took ~1.2 GB per worker on the CPU, 0.8 GB on the GPU, so big parts get fewer workers); dual
    contouring's block is bigger by its halo and needs more per sample, one with normals by NORMAL_MORE."""
    n = plan.block + 1 + 2 * (plan.halo + (BLOCK_HALO if dual else 0)) + (blockmesh.NORMAL_MORE if normals else 0)
    return int(WORKER_BASE + n ** 3 * (DUAL_BYTES if dual else 4 * 12) + MESH_BYTES * mesh_triangles(field))


def cpu_memory(plan, dual=False):
    """Bytes a CPU worker needs for one task's CPU stage (engine/stages.py): the task's blocks' samples,
    handed over, and one block meshed and reduced (dual contoured) at a time."""
    n = plan.block + 1 + (2 * BLOCK_HALO if dual else 0)
    return int(BATCH * n ** 3 * 4 + n ** 3 * (DUAL_BYTES if dual else 4 * 12))


WORKER_BASE = int(0.6 * 2**30)  # a worker process itself: libraries plus a CUDA context
# RAM per triangle per worker for distance-field search structures: mostly libigl's tree, since a big mesh's
# vertices, triangles and normals are mapped from the shared folder, one copy for all (500 before: 1.31M
# triangles held 0.48 GB per worker, now 0.32, 2026-11-09)
MESH_BYTES = 370
MESH_GPU_BYTES = 150  # GPU memory per triangle per worker (Warp BVH, normals)
DUAL_BYTES = 200      # RAM per sample of a dual contouring block (its field arrays and the crossings' data)


def find_surface(field, plan):
    """Blocks the surface probably passes through, from the field sampled every BLOCK/PROBE
    voxels: a sign change, or a value small enough that the surface could be between samples."""
    step = plan.block // PROBE
    probe = Grid.block(plan.base, plan.voxel * step, (0, 0, 0), [n * PROBE + 1 for n in plan.blocks])
    xp = backend.cupy() if backend.use_gpu_for(probe.size) else np
    try:
        with trace.span("probe", xp=xp):
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


def mesh_blocks(field, plan, blocks, keep, gpu, tolerance=0.0, close=True, normals=False):
    """Mesh a few blocks (runs in a worker). One result per block: (block, vertices in global
    sample units, faces, border-vertex mask, crossings, (inside, nonfinite)), where
    crossings[axis][side] tells whether the surface runs into the neighbour on that side (side 0 =
    lower), inside[axis][side] whether the body reaches that face of the block, and nonfinite
    counts the NaN and infinite samples (see meshing.finite_values). In a two-stage map (engine/stages.py)
    blocks that are reduced (Keep, Tolerance) are meshed and reduced by a CPU worker: this worker only
    computes their samples. normals (no Keep or Tolerance): the field's normals as a 7th item, each block's
    cells one sample up (blockmesh.normals_block)."""
    xp = backend.cupy() if gpu and backend.gpu_enabled() else np
    try:
        if (tolerance > 0 or keep < 1.0) and stages.splitting():
            values = []
            for b in blocks:
                with trace.span("block", item=b):
                    values.append(_block_values(field, plan, b, xp))
            return stages.cpu_stage(blockmesh.mesh_values, plan=blockmesh.layout(plan), blocks=blocks,
                                    values=values, keep=keep, tolerance=tolerance, close=close)
        out = []
        for b in blocks:
            with trace.span("block", item=b):
                out.append(_mesh_block(field, plan, b, keep, xp, tolerance, close, normals))
        return out
    finally:
        if xp is not np:
            backend.release_gpu_memory()


def _block_values(field, plan, b, xp, more=0):
    """Block b's own samples (without the halo), on the host; `more`: that many more past its far faces (the
    grid's start, a multiple of ALIGN, stays; the halo goes round them too)."""
    grid = plan.eval_grid(b)
    if more:
        grid = Grid.block(grid.base, grid.voxel, grid.start, [s + more for s in grid.shape])
    with trace.span("field", xp=xp):
        values = Context(grid, None, xp).eval_full(field)
    h = plan.halo
    if h:
        n = plan.block + 1 + more
        values = values[h:h + n, h:h + n, h:h + n]
    return backend.to_host(values)


def _mesh_block(field, plan, b, keep, xp, tolerance=0.0, close=True, normals=False):
    more = blockmesh.NORMAL_MORE if normals else 0
    return _mesh_block_values(_block_values(field, plan, b, xp, more), plan, b, keep, tolerance, close, normals)


def dual_blocks(field, plan, blocks, gpu):
    """Dual contouring of a few blocks (runs in a worker). One (block, piece) per block: the piece as
    from dualcontour.contour_block (only `needs` where the block has no surface), plus `nonfinite`
    (its own samples that were not finite) and `clipped` (the body reaches the box's outer samples).
    In a two-stage map (engine/stages.py) a CPU worker does the contouring: this worker only computes the
    samples."""
    xp = backend.cupy() if gpu and backend.gpu_enabled() else np
    try:
        if stages.splitting():
            values = []
            for b in blocks:
                with trace.span("block", item=b):
                    values.append(_dual_values(field, plan, b, xp))
            return stages.cpu_stage(blockmesh.dual_contour_values, plan=blockmesh.layout(plan), blocks=blocks,
                                    values=values)
        out = []
        for b in blocks:
            with trace.span("block", item=b):
                out.append(_dual_block(field, plan, b, xp))
        return out
    finally:
        if xp is not np:
            backend.release_gpu_memory()


def _dual_values(field, plan, b, xp):
    """Block b's samples with dual contouring's halo (without the field's), on the host."""
    with trace.span("field", xp=xp):
        values = Context(plan.eval_grid(b, BLOCK_HALO), None, xp).eval_full(field)
    h = plan.halo
    if h:
        values = values[h:-h, h:-h, h:-h]
    return backend.to_host(values)


def _dual_block(field, plan, b, xp):
    return _dual_block_values(_dual_values(field, plan, b, xp), plan, b)


def _neighbours(result, plan):
    """The neighbours a marching cubes block's surface runs into."""
    b, crossings = result[0], result[4]
    for a in range(3):
        for side, step in ((0, -1), (1, 1)):
            n = list(b)
            n[a] += step
            if crossings[a][side] and 0 <= n[a] < plan.blocks[a]:
                yield tuple(n)


def stitch(results, plan, tolerance=0.0, close=True):
    """One mesh from the blocks' pieces (a dict, emptied on the way to save memory): border
    vertices that lie on the same grid edge are the same vertex. With a tolerance the blocks were
    reduced with their borders kept: the seams are reduced here (half the tolerance, as they now
    sit next to reduced surface). An open piece (close=False) has no volume to orient it by; marching
    cubes already faces it outward. Pieces with the field's normals (mesh_blocks(normals=True)) give them to
    the mesh (meshing.FIELD_NORMAL; a seam's vertex has the same normal in both blocks)."""
    with trace.span("stitch", blocks=len(results)):
        return _stitch(results, plan, tolerance, close)


def _stitch(results, plan, tolerance, close):
    pieces = [(r[1], r[2], r[3], r[6] if len(r) > 6 else None) for r in results.values()
              if r[1] is not None and len(r[2])]
    results.clear()
    if not pieces:
        raise ValueError("Field has no surface inside the grid (no sign change)")
    offsets = np.cumsum([0] + [len(p[0]) for p in pieces])
    V = np.concatenate([p[0] for p in pieces])
    B = np.concatenate([p[2] for p in pieces])
    F = np.concatenate([p[1].astype(np.int64) + o for p, o in zip(pieces, offsets)])
    N = np.concatenate([p[3] for p in pieces]) if tolerance <= 0 and all(p[3] is not None for p in pieces) else None
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
        F = F[:, ::-1]  # same as meshing.field_to_mesh: outward-facing triangles (the normals point out already)
    mesh = trimesh.Trimesh(plan.base + V * plan.voxel, F, process=False,
                           vertex_attributes=None if N is None else {FIELD_NORMAL: N})
    del V, F, N
    mesh.remove_unreferenced_vertices()  # (trimesh keeps vertex_attributes in step, also in snap_float32)
    return snap_float32(mesh)


def mesh_tiled(field, bounds, voxel, padding, keep, map_fn, tolerance=0.0, close=True, surface=None,
               method="marching cubes", normals=False):
    """Mesh `field` block by block. `map_fn(fn, arg_list, need=, gpu=, gpu_need=)` runs the block tasks
    (in parallel when it can). Returns (mesh, blocks computed, blocks in the box); `mesh.metadata`
    has `nonfinite`, `clipped` and `voxel` as from meshing.field_to_mesh. close=False leaves the surface
    open where it leaves the box (a piece of a body: the editor's detail view; marching cubes only);
    `surface`: the blocks find_surface gave for the same layout, if already known. Dual contouring
    (`method`) joins its blocks by vertex names (dualcontour.contour_block) and reduces the whole mesh
    here, in this process, for `tolerance` or `keep`. normals: the field's normals at the vertices
    (meshing.FIELD_NORMAL; the editor's views and details: marching cubes, no Keep or Tolerance)."""
    dual = method == "dual contouring"
    if dual and not close:
        raise ValueError("dual contouring in blocks makes closed meshes only")
    normals = bool(normals) and not dual and keep >= 1.0 and tolerance <= 0
    plan = Plan(field, bounds, voxel, padding)
    gpu = backend.gpu_enabled()
    done, todo = {}, (find_surface(field, plan) if surface is None else list(surface))
    while todo:
        tasks = [dict(field=field, plan=plan, blocks=todo[i:i + BATCH], gpu=gpu) if dual else
                 dict(field=field, plan=plan, blocks=todo[i:i + BATCH], keep=keep, gpu=gpu, tolerance=tolerance,
                      close=close, **({"normals": True} if normals else {}))
                 for i in range(0, len(todo), BATCH)]
        for results in map_fn(dual_blocks if dual else mesh_blocks, tasks,
                              need=worker_memory(plan, field, dual, normals),
                              gpu=gpu, gpu_need=MESH_GPU_BYTES * mesh_triangles(field),
                              cpu_need=cpu_memory(plan, dual)):
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
