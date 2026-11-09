"""The CPU part of meshing a block (engine/tiles.py): its samples to a mesh piece (marching cubes, then reduced
with the border kept) or a dual contouring piece. Here and not in tiles so that a CPU worker (engine/stages.py),
which runs it in a two-stage map, loads only these modules (numpy, trimesh, scikit-image, pyfqmr), not the
field library. A plan here is anything with the block layout's `voxel`, `block` and `blocks` (tiles.Plan, or
`layout(plan)` to hand over)."""
from types import SimpleNamespace

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from . import trace
from .dualcontour import BLOCK_HALO, contour_block, off_zero
from .meshing import field_normals, finite_values, reduce, simplify


def layout(plan):
    """What a block's CPU part needs of a tiles.Plan: the voxel and the blocks' size and count."""
    return SimpleNamespace(voxel=plan.voxel, block=plan.block, blocks=tuple(plan.blocks))


def mesh_values(plan, blocks, values, keep, tolerance, close):
    """tiles.mesh_blocks' CPU stage: the blocks meshed from their samples (a list, emptied on the way)."""
    out = []
    for b in blocks:
        with trace.span("block", item=b):
            out.append(mesh_block_values(values.pop(0), plan, b, keep, tolerance, close))
    return out


def mesh_block_values(values, plan, b, keep, tolerance=0.0, close=True, normals=False):
    """Block b's mesh piece from its own samples (see tiles.mesh_blocks). normals: `values` reach NORMAL_MORE samples
    past the block's far faces, and the piece carries the field's normals (normals_block)."""
    if normals:
        n = plan.block + 1
        nonfinite = int(np.count_nonzero(~np.isfinite(values[:n, :n, :n])))  # (its own samples, as without normals)
        values = finite_values(values, plan.voxel)[0]
    else:
        values, nonfinite = finite_values(values, plan.voxel)
    # same nudge off zero as meshing.field_to_mesh; identical in every block for a shared sample
    eps = np.float32(0.01 * plan.voxel)  # float32: a Python float made the inner where float64
    values = np.where(np.abs(values) < eps, np.where(values < 0, -eps, eps), values).astype(np.float32)
    if normals:
        return normals_block(values, plan, b, close, nonfinite)
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
    with trace.span("marching cubes", device="cpu"):
        verts, faces, _, _ = marching_cubes(values, level=0.0)
    verts = verts.astype(np.float64) + np.array([b[a] * plan.block - pad[a][0] for a in range(3)], dtype=np.float64)
    faces = faces.astype(np.int64)
    if tolerance > 0:  # reduce to a tolerance here, in the worker: borders stay for the neighbours
        verts, faces = reduce(verts, faces, tolerance / plan.voxel, snap=False)
    elif keep < 1.0:
        verts, faces = simplify_block(verts, faces, keep, plan.voxel)
    return b, verts, faces.astype(np.int32), border(faces, len(verts)), crossings, found


NORMAL_MORE = 2  # samples a block with normals computes past its far faces (normals_block)


def normals_block(values, plan, b, close, nonfinite):
    """Block b's mesh piece with the field's normals (meshing.field_normals, per vertex, the 7th item). Central
    differences at a sample need its neighbours on both sides. A block's grid cannot start a sample earlier (its
    start must stay a multiple of tiles.ALIGN so that coarse sampling lattices coincide: a layer before would cost
    ALIGN layers, ~1.5x the samples), so its cells move one up instead: block b meshes the cells from b * block + 1 to
    (b + 1) * block + 1, its samples reach NORMAL_MORE past (b + 1) * block (6% more), and every sample it needs a
    gradient at has both neighbours in it, the same bit for bit in both blocks at a seam. The first block along an
    axis also has the cells from the box's first sample (and its pad), the last one stops at the box's last sample
    (and its pad). Same triangles as mesh_block_values, from other blocks. `values`: finite and off zero, from
    b * block on."""
    n = plan.block
    first = [b[a] == 0 for a in range(3)]
    last = [b[a] == plan.blocks[a] - 1 for a in range(3)]
    lo = [0 if first[a] else 1 for a in range(3)]  # the first and last sample of the block's cells, in values
    hi = [n if last[a] else n + 1 for a in range(3)]
    own = values[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    faces_ = [(own.take(0, axis=a), own.take(-1, axis=a)) for a in range(3)]
    crossings = [[bool(f.min() < 0 < f.max()) for f in pair] for pair in faces_]
    found = ([[bool(f.min() < 0) for f in pair] for pair in faces_], nonfinite)
    del faces_
    if own.min() > 0 or own.max() < 0:
        return b, None, None, None, crossings, found, None
    fill = max(float(own.max()), plan.voxel)  # "outside": closes the surface on the box's outer faces
    pad = [(int(close and first[a]), int(close and last[a])) for a in range(3)]
    samples, before = values, [0, 0, 0]
    for a in range(3):
        if first[a]:  # a layer before the box's first sample: the pad, or (an open piece) the samples continued straight
            edge = samples.take(0, axis=a)
            layer = np.full_like(edge, fill) if close else 2 * edge - samples.take(1, axis=a)
            samples = np.concatenate([np.expand_dims(layer, a), samples], axis=a)
            before[a] = 1
        if last[a] and close:  # past the box's last sample: the pad
            beyond = [slice(None)] * 3
            beyond[a] = slice(before[a] + n + 1, None)
            samples[tuple(beyond)] = fill
    start = [before[a] + lo[a] - pad[a][0] for a in range(3)]
    stop = [before[a] + hi[a] + pad[a][1] + 1 for a in range(3)]
    with trace.span("marching cubes", device="cpu"):
        verts, faces, _, _ = marching_cubes(samples[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]], level=0.0)
    normals = field_normals(samples, verts, start)
    verts = verts.astype(np.float64) + np.array([b[a] * n + lo[a] - pad[a][0] for a in range(3)], dtype=np.float64)
    faces = faces.astype(np.int64)
    return b, verts, faces.astype(np.int32), border(faces, len(verts)), crossings, found, normals


def simplify_block(verts, faces, keep, voxel):
    """Keep about `keep` of the triangles without moving the block's open border, so it still
    meets the neighbours (same algorithm and settings as meshing.decimate)."""
    # The algorithm does not check that a collapse keeps the surface manifold: in ~1 block of 200
    # it leaves an edge shared by 4 triangles. Such a block is kept as it was (~1% more triangles).
    return simplify(verts, faces, keep, voxel, preserve_border=True) or (verts, faces)


def border(faces, count):
    """Mask of vertices on open edges: in a block's mesh, exactly the ones on shared faces."""
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    mask = np.zeros(count, dtype=bool)
    mask[edges[trimesh.grouping.group_rows(edges, require_count=1)].ravel()] = True
    return mask


def dual_contour_values(plan, blocks, values):
    """tiles.dual_blocks' CPU stage: the blocks contoured from their samples (a list, emptied on the way)."""
    out = []
    for b in blocks:
        with trace.span("block", item=b):
            out.append(dual_block_values(values.pop(0), plan, b))
    return out


def dual_block_values(values, plan, b):
    """Block b's dual contouring piece from its samples with BLOCK_HALO more (see tiles.dual_blocks)."""
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
