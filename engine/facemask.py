"""Face masks: a field that is 1 on chosen CAD faces and 0 elsewhere, fading over a set width at
the edge of the selection. Multiply a texture by it to apply the texture only to those faces.

How a point gets its value: its closest point on the body decides whether it is on a chosen face
(+) or not (-), and the distance from that closest point to the selection's border (the edges
where chosen and other faces meet) sets how far into the fade it is. Measuring from the closest
point keeps the value the same through the thickness of an offset, and measuring to the border
(not to the other faces) keeps thin walls right: the inside of a 1 mm petal wall does not count
as "1 mm from the selection".
"""
import os
import re

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .fields import Field, _digest, mesh_fingerprint
from .grid import upsample_linear
from .sdf import arrays, mesh_query
from .shared import Shared

APPLY = ("chosen faces", "all other faces")  # where the effect goes: the picked faces, or all but them
POSITIONS = ("inside", "center", "outside")
REACHES = ("surface", "volume")
_TREES = {}  # (mesh fingerprint, faces, which) -> igl AABB of that subset of triangles


def cad_faces(mesh):
    """Per-triangle CAD face numbers of a tessellated body, or None (e.g. a plain mesh)."""
    ids = getattr(mesh, "face_attributes", {}).get("cad_face")
    return np.asarray(ids) if ids is not None and len(ids) == len(mesh.faces) else None


def cad_body(mesh, default=0):
    """The body's number in its STEP file (set by Import STEP)."""
    return int(getattr(mesh, "metadata", {}).get("cad_body", default))


def faces_for_body(text, body):
    """Face numbers chosen on body number `body`: "15" means face 15 on every body (a list of
    identical bodies, like the lamp's petals), "2:15" face 15 of body 2 only."""
    if isinstance(text, (list, tuple)):
        return sorted({int(f) for f in text})
    out = set()
    for b, f, plain in re.findall(r"(\d+)\s*:\s*(\d+)|(\d+)", str(text or "")):
        if plain:
            out.add(int(plain))
        elif int(b) == body:
            out.add(int(f))
    return sorted(out)


def parse_faces(text):
    """'3, 7 12' -> [3, 7, 12] (body-qualified entries "2:15" count as face 15)."""
    if isinstance(text, (list, tuple)):
        return sorted({int(f) for f in text})
    return sorted({int(f or p) for _, f, p in re.findall(r"(\d+)\s*:\s*(\d+)|(\d+)", str(text or ""))})


def fade_range(fade, position):
    """(signed distance where the mask starts rising, where it reaches 1)."""
    fade = max(float(fade), 0.0)
    return {"inside": (0.0, fade), "center": (-fade / 2, fade / 2), "outside": (-fade, 0.0)}[position]


def border_segments(V, F, selected):
    """Mesh edges where a chosen triangle meets another one: (n, 2, 3) endpoints."""
    m = trimesh.Trimesh(V, F, process=False)
    pairs, edges = m.face_adjacency, m.face_adjacency_edges
    cross = selected[pairs[:, 0]] != selected[pairs[:, 1]]
    return np.asarray(V)[edges[cross]]


def sample_segments(segments, spacing):
    """Points along the segments, at most `spacing` apart (endpoints included)."""
    if len(segments) == 0:
        return np.zeros((0, 3))
    a, b = segments[:, 0], segments[:, 1]
    n = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / spacing).astype(np.int64), 1)
    seg = np.repeat(np.arange(len(segments)), n + 1)
    t = (np.arange(len(seg)) - np.repeat(np.cumsum(n + 1) - (n + 1), n + 1)) / np.repeat(n, n + 1)
    return a[seg] + (b - a)[seg] * t[:, None]


def nearest_on_grid(V, F, grid, max_dist, key=None, gpu=False, coarse_factor=None):
    """Grid points within max_dist of the mesh: (flat indices, triangle index, closest point).
    Exact queries only near the surface, found from a coarse pass (as in sdf.mesh_to_sdf);
    `coarse_factor` fixes that pass's factor (a grid that is itself coarse needs one that keeps the
    total dividing tiles.ALIGN)."""
    if gpu:
        from .sdf_gpu import nearest_on_grid_gpu
        return nearest_on_grid_gpu(V, F, grid, max_dist, key=key)
    query = mesh_query(V, F, key)
    if grid.size <= 20_000:
        candidates = np.arange(grid.size)
    else:
        factor = coarse_factor or (2 if grid.size < 2_000_000 else 4)
        coarse = grid.coarser(factor)
        d = np.sqrt(query.tree.squared_distance(query.V, query.F, coarse.points())[0]).astype(np.float32)
        d = upsample_linear(d.reshape(coarse.shape), factor, grid.shape)
        candidates = np.flatnonzero(d < max_dist + coarse.voxel * np.sqrt(3))
    points = grid.base + (np.asarray(grid.start) + np.column_stack(np.unravel_index(candidates, grid.shape))) * grid.voxel
    sq, tri, closest = query.tree.squared_distance(query.V, query.F, points)
    keep = sq <= max_dist * max_dist
    return candidates[keep], np.asarray(tri)[keep], np.asarray(closest)[keep]


def _subset_distance(V, F, key, points):
    """Distance from points to a subset of a mesh's triangles (igl AABB, cached per process)."""
    import igl
    V, F = arrays(V, F)
    tree = _TREES.get(key)
    if tree is None:
        tree = igl.AABB()
        tree.init(V, F)
        _TREES[key] = tree
        while len(_TREES) > 16:
            _TREES.pop(next(iter(_TREES)))
    return np.sqrt(tree.squared_distance(V, F, points)[0])


def _volume_mask(V, F, selected, fingerprint, faces, fade):
    """fn for reach="volume": 1 at the chosen faces, fading smoothly to 0 at `fade` mm from them,
    through the whole part (the other faces play no part). Sampled on a grid 4x coarser and
    interpolated (the mask is smooth), so a fade below ~4 voxels is softened.

    (A first version put the border halfway between the chosen and the other faces: on a flat board
    the top and bottom then limited a chosen side's reach to half the board's thickness, and with a
    longer fade the mask never got near 1.)"""
    V, Fs = Shared(V, key=fingerprint), Shared(np.ascontiguousarray(F[selected]))

    def fn(ctx):
        grid = ctx.grid
        factor = 4 if grid.size > 50_000 else 1
        target = grid.coarser(factor) if factor > 1 else grid
        d = _subset_distance(V, Fs, (fingerprint, faces, "chosen"), target.points())
        if fade < 1e-9:
            values = (d < 1e-6).astype(np.float32)
        else:
            t = np.clip(d / fade, 0.0, 1.0)
            values = (1 - t * t * (3 - 2 * t)).astype(np.float32)
        values = values.reshape(target.shape)
        if factor > 1:
            values = upsample_linear(values, factor, grid.shape)
        return ctx.xp.asarray(values)
    return fn


# A surface mask with a wide fade is smooth: sampled about 6 times across the fade on a grid this
# much coarser and interpolated, it is the same within ~1% (a hard edge is sampled at full
# resolution). Factors divide tiles.ALIGN also with nearest_on_grid's own coarse pass of 2, so
# block meshing stays exact. (Key cabinet, Medium: 14.6 s -> see docs/FINDINGS.md.)
SURFACE_FACTORS = (6, 3, 2)
SAMPLES_ACROSS_FADE = 6


def sample_factor(span, voxel, size):
    """How much coarser than a grid with `voxel` mm samples a mask fading over `span` mm is sampled."""
    if size <= 50_000:
        return 1
    return next((f for f in SURFACE_FACTORS if f * voxel <= span / SAMPLES_ACROSS_FADE), 1)


def face_mask(mesh, ids, faces, fade, position="inside", band=2.0, reach="surface", invert=False):
    """Field: 1 on CAD faces `faces` of `mesh` (`ids` = CAD face per triangle), 0 elsewhere, with a
    smooth fade `fade` mm wide placed inside, centred on, or outside the selection's border.
    reach="surface": computed within `band` mm of the surface (0 farther away), the fade measured
    along the surface (right on thin walls; for textures). reach="volume": 1 at the chosen faces,
    fading to 0 at `fade` mm from them through the whole part (for lattices); `position` and `band`
    do not apply. invert: the effect goes on every face except `faces` (easier when only a few
    faces should stay plain); the fade and position then refer to those other faces."""
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {', '.join(POSITIONS)}")
    if reach not in REACHES:
        raise ValueError(f"reach must be one of {', '.join(REACHES)}")
    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    fingerprint = mesh_fingerprint(mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    faces = tuple(parse_faces(faces))
    selected = np.isin(np.asarray(ids), faces)
    if invert:
        selected = ~selected
    segments = border_segments(V, F, selected) if selected.any() and not selected.all() else np.zeros((0, 2, 3))
    lo, hi = fade_range(fade, position)
    key = _digest("facemask", "sampled", fingerprint, faces, bool(invert), float(fade), position, float(band), reach)
    if reach == "volume":
        key = _digest("facemask-distance", fingerprint, faces, bool(invert), float(fade))
        if not selected.any():
            return Field(lambda ctx: np.float32(0.0), key, "face_mask(none)", range=(0.0, 0.0))
        field = Field(_volume_mask(V, F, selected, fingerprint, (faces, bool(invert)), float(fade)), key,
                      f"face_mask({len(faces)} faces, volume)", cache=True, range=(0.0, 1.0))
        field.extent = (mesh.bounds[0] - 1.0, mesh.bounds[1] + 1.0)
        return field
    span = max(abs(lo), abs(hi))  # how far from the border the fade reaches
    any_selected = bool(selected.any())
    # pickled into every task sent to a worker: big arrays as Shared (read once per worker)
    V, F, selected = Shared(V, key=fingerprint), Shared(F, key=fingerprint), Shared(selected)

    def values_on(grid, reach, spacing, gpu, coarse_factor=None):
        near, tri, closest = nearest_on_grid(V, F, grid, reach, key=fingerprint, gpu=gpu, coarse_factor=coarse_factor)
        # distance along the surface to the selection's border, + on chosen faces, - elsewhere
        d = np.full(len(near), span + 1.0)
        if len(segments):
            border = sample_segments(segments, spacing)
            threads = int(os.environ.get("IGL_NUM_THREADS") or os.cpu_count() or 1)  # a worker's share
            d = np.minimum(cKDTree(border).query(closest, distance_upper_bound=span + 1e-6, workers=threads)[0], d)
        s = np.where(selected.get()[tri], d, -d)
        if hi - lo < 1e-9:  # no fade: a hard edge
            values = ((s >= lo) if position == "outside" else (s > lo)).astype(np.float32)
        else:
            t = np.clip((s - lo) / (hi - lo), 0.0, 1.0)
            values = (t * t * (3 - 2 * t)).astype(np.float32)
        out = np.zeros(grid.size, dtype=np.float32)
        out[near] = values
        return out.reshape(grid.shape)

    def fn(ctx):
        if not any_selected:
            return np.float32(0.0)
        grid = ctx.grid
        gpu = ctx.xp is not np
        spacing = max(min(span / 8, grid.voxel / 2), 1e-3)  # border samples: as fine as the grid
        factor = sample_factor(span, grid.voxel, grid.size) if hi - lo > 1e-9 else 1
        if factor == 1:
            return ctx.xp.asarray(values_on(grid, band, spacing, gpu))
        coarse = grid.coarser(factor)
        values = values_on(coarse, band + coarse.voxel * np.sqrt(3), spacing, gpu, coarse_factor=2)
        return ctx.xp.asarray(upsample_linear(values, factor, grid.shape))
    field = Field(fn, key, f"face_mask({len(faces)} faces)", cache=True, range=(0.0, 1.0))
    field.extent = (bounds[0] - band - 1.0, bounds[1] + band + 1.0)
    return field


def tangent_groups(mesh, ids, angle=np.radians(30)):
    """Group number per CAD face: faces joined by smooth (tangent) edges share a group, so a
    fillet made of several faces can be picked at once. Two faces count as smoothly joined when
    the median bend between the triangles along their shared edge is under `angle` (the
    triangles near a tangent edge can lean by up to half the tessellation angle on each side)."""
    ids = np.asarray(ids)
    count = int(ids.max()) + 1 if len(ids) else 0
    parent = np.arange(count)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    m = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    pairs, bends = m.face_adjacency, m.face_adjacency_angles
    a, b = ids[pairs[:, 0]], ids[pairs[:, 1]]
    across = a != b
    if across.any():
        lo, hi = np.minimum(a, b)[across], np.maximum(a, b)[across]
        key = lo.astype(np.int64) * count + hi
        order = np.argsort(key, kind="stable")
        key, bend = key[order], bends[across][order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        medians = [np.median(chunk) for chunk in np.split(bend, starts[1:])]
        for k, med in zip(key[starts], medians):
            if med < angle:
                parent[root(int(k // count))] = root(int(k % count))
    return np.array([root(i) for i in range(count)], dtype=np.int64)
