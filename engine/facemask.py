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
import threading
from collections import OrderedDict

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
_FACE_TREES = OrderedDict()  # mesh fingerprint -> (igl AABB of all its triangles, triangle count), newest last
_FACE_TREES_LOCK = threading.Lock()
MAX_TREE_TRIANGLES = 2_000_000  # nearest_face keeps search trees for about this many triangles in all
_FACE_TABLES = OrderedDict()  # mesh fingerprint -> face_info of every CAD face of that body, newest last
SIMILAR = ("size", "plane", "type")  # Select similar: the same size, in the same plane, the same kind of surface


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


# OpenCascade's GeomAbs_SurfaceType numbers, as kept in mesh.metadata["cad_surfaces"][:, 0]
SURFACE_TYPES = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "bezier", 6: "bspline",
                 7: "revolution", 8: "extrusion", 9: "offset"}


def face_info(meshes, body, face):
    """What one CAD face is, for the face menu of the 3D view: a dict with `type` (plane, cylinder,
    cone, sphere, torus, bspline, ... or other), `area` (mm²), `centre` (area-weighted centroid of its
    triangles), `triangles`, and for a plane its unit `normal`, for a cylinder its `axis` (unit
    vector, sign fixed), `radius` and `hole` (the faces' normals point towards the axis: a bore, not a
    shaft); entries that do not apply are None (hole: False). `meshes` is a tessellated body or a
    list of them; `body` is the body number as /api/faces reports it (the first mesh with that
    number is used: copies share faces). None if the body or the face is not there."""
    items = list(meshes) if isinstance(meshes, (list, tuple)) else [meshes]
    for k, mesh in enumerate(items):
        ids = cad_faces(mesh)
        if ids is None or cad_body(mesh, k) != body:
            continue
        on = ids == face
        if on.any():  # with the body's name in the STEP file, when known (Import STEP's cad_name)
            return {**_describe_face(mesh, np.flatnonzero(on), face), "body_name": mesh.metadata.get("cad_name")}
    return None


def face_table(mesh):
    """face_info of every CAD face of one body, {face: info}; kept per process for the last few
    bodies (Select similar asks about one body again and again)."""
    key = mesh_fingerprint(mesh)
    with _FACE_TREES_LOCK:
        if key in _FACE_TABLES:
            _FACE_TABLES.move_to_end(key)
            return _FACE_TABLES[key]
    ids = cad_faces(mesh)
    order = np.argsort(ids, kind="stable")
    faces, starts = np.unique(ids[order], return_index=True)
    table = {int(f): _describe_face(mesh, rows, int(f)) for f, rows in zip(faces, np.split(order, starts[1:]))}
    with _FACE_TREES_LOCK:
        _FACE_TABLES[key] = table
        while len(_FACE_TABLES) > 8:
            _FACE_TABLES.popitem(last=False)
    return table


def similar_faces(meshes, body, face, mode):
    """The faces of body `body` that are like its face `face` (that one included, sorted), for Select
    similar in the 3D view. `mode`: size = the same kind of surface and size (holes or shafts of the
    same diameter, other faces of the same area), plane = planes lying in the same plane and facing
    the same way, type = the same kind of surface (every bore, every shaft, every plane...). `meshes`
    and `body` as in face_info; None if the body or the face is not there."""
    if mode not in SIMILAR:
        raise ValueError(f"mode must be one of: {', '.join(SIMILAR)}")
    items = list(meshes) if isinstance(meshes, (list, tuple)) else [meshes]
    for k, mesh in enumerate(items):
        ids = cad_faces(mesh)
        if ids is None or cad_body(mesh, k) != body or not np.any(ids == face):
            continue
        table = face_table(mesh)
        return sorted(f for f, info in table.items() if _alike(table[face], info, mode))
    return None


def _alike(a, b, mode):
    """Whether face info b is like a for similar_faces' `mode`."""
    if a["type"] != b["type"] or a["hole"] != b["hole"]:
        return False
    if mode == "type":
        return True
    if mode == "plane":
        if a["normal"] is None or b["normal"] is None:
            return False
        n = np.asarray(a["normal"])
        return float(n @ b["normal"]) > 0.9999 and abs(float(n @ (np.subtract(b["centre"], a["centre"])))) < 0.01
    if a["radius"] is not None or b["radius"] is not None:
        return a["radius"] is not None and b["radius"] is not None and \
            abs(a["radius"] - b["radius"]) <= max(1e-3, 2e-3 * a["radius"])
    return abs(a["area"] - b["area"]) <= 0.01 * max(a["area"], b["area"])


def _describe_face(mesh, rows, face):
    corners = np.asarray(mesh.faces)[rows]
    tri = np.asarray(mesh.vertices, dtype=np.float64)[corners]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    size = np.linalg.norm(cross, axis=1)
    area = float(0.5 * size.sum())
    centre = (0.5 * size[:, None] * tri.mean(axis=1)).sum(axis=0) / area if area > 0 else tri.mean(axis=(0, 1))
    normals = cross / np.maximum(size, 1e-300)[:, None]
    kinds = np.asarray(mesh.metadata.get("cad_surfaces", np.zeros((0, 3))))
    kind = SURFACE_TYPES.get(int(kinds[face][0]), "other") if face < len(kinds) else "other"
    info = {"type": kind, "area": area, "centre": centre.tolist(), "normal": None, "axis": None, "radius": None,
            "hole": False, "triangles": len(rows)}
    if kind == "plane" and area > 0:
        total = cross.sum(axis=0)   # sum of (2 x area x normal)
        info["normal"] = (total / max(np.linalg.norm(total), 1e-300)).tolist()
    elif kind == "cylinder" and area > 0:
        fit = _fit_cylinder(tri, corners, normals, 0.5 * size)
        if fit is not None:
            info.update(fit)
    return info


def _fit_cylinder(tri, corners, normals, areas):
    """{axis, radius, hole} of a cylinder from its triangles, or None if they do not show one. The
    axis is the direction the normals avoid (smallest eigenvector of the sum of n nᵀ, which holds
    for a part of a cylinder as well as a whole one); the circle the corner points make across the
    axis is fitted (Kåsa), as the centroid of an arc is not on its axis."""
    _, vectors = np.linalg.eigh(np.einsum("k,ki,kj->ij", areas, normals, normals))
    axis = vectors[:, 0]
    axis = axis if axis[np.argmax(np.abs(axis))] > 0 else -axis
    side = vectors[:, 1:]                               # two directions across the axis
    points = tri.reshape(-1, 3)[np.unique(corners.ravel(), return_index=True)[1]]
    flat = (points - points.mean(axis=0)) @ side        # corner points seen along the axis
    A = np.column_stack([2 * flat, np.ones(len(flat))])
    solution, _, rank, _ = np.linalg.lstsq(A, (flat ** 2).sum(axis=1), rcond=None)
    if rank < 3 or not np.all(np.isfinite(solution)):
        return None
    middle = solution[:2]
    radius = float(np.linalg.norm(flat - middle, axis=1).mean())
    if not np.isfinite(radius) or radius <= 0:
        return None
    # the normals of a bore look at the axis: on average against the way out from it
    out = tri.mean(axis=1) - (points.mean(axis=0) + side @ middle)
    out -= np.outer(out @ axis, axis)
    hole = float((areas * np.einsum("ij,ij->i", out, normals)).sum()) < 0
    return {"axis": axis.tolist(), "radius": radius, "hole": hole}


def same_part(meshes, ids):
    """The bodies are copies of one part (the petals, also when scaled): the same faces, the same
    triangles per face and the same share of the surface per face, so they share face numbers.
    `ids` is the CAD face per triangle of each body (`cad_faces`)."""
    counts = [np.bincount(i) for i in ids]
    if not all(len(c) == len(counts[0]) and np.array_equal(c, counts[0]) for c in counts):
        return False
    shares = []
    for m, i in zip(meshes, ids):
        areas = _triangle_areas(m)
        shares.append(np.bincount(i, weights=areas) / max(float(areas.sum()), 1e-12))
    return all(np.allclose(a, shares[0], atol=1e-4) for a in shares)


def _triangle_areas(mesh, chunk=1_000_000):
    """Area of every triangle, in chunks, without filling trimesh's cache on a kept mesh (`area_faces`
    keeps 104 B per triangle on it; /api/faces runs `same_part` for every CAD view the editor shows)."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    out = np.empty(len(F))
    for s in range(0, len(F), chunk):
        a, b, c = (V[F[s:s + chunk, k]] for k in range(3))
        out[s:s + chunk] = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return out


def _face_tree(mesh):
    """igl AABB over all of a body's triangles, kept per process for the bodies asked about last (a
    click on the 3D view asks about a few bodies again and again; the lamp's 145 petals are small, its
    centre part is not)."""
    import igl
    key = mesh_fingerprint(mesh)
    with _FACE_TREES_LOCK:
        if key in _FACE_TREES:
            _FACE_TREES.move_to_end(key)
            return _FACE_TREES[key][0]
    tree = igl.AABB()
    tree.init(np.ascontiguousarray(mesh.vertices, dtype=np.float64), np.ascontiguousarray(mesh.faces, dtype=np.int64))
    with _FACE_TREES_LOCK:
        _FACE_TREES[key] = (tree, len(mesh.faces))
        while len(_FACE_TREES) > 1 and sum(n for _, n in _FACE_TREES.values()) > MAX_TREE_TRIANGLES:
            _FACE_TREES.popitem(last=False)
    return tree


def nearest_face(meshes, point, within=np.inf, index=False):
    """The CAD face closest to `point`: (face number, body number, distance in mm), plus the body's
    position in the list with `index`, or None when no body with CAD faces lies closer than `within`
    mm. `meshes` is a tessellated body or a list of them; the body number is the one /api/faces
    reports (`cad_body`, the list index for a body without one). Bodies are tried nearest bounding
    box first and the search stops at the first box that is not closer than the best distance so
    far, so a list of 145 petals costs a handful of tree queries (a box is never nearer than its
    triangles)."""
    items = list(meshes) if isinstance(meshes, (list, tuple)) else [meshes]
    point = np.asarray(point, dtype=np.float64).reshape(3)
    ranked = []
    for k, mesh in enumerate(items):
        ids = cad_faces(mesh)
        if ids is None or len(ids) == 0:
            continue
        vertices = np.asarray(mesh.vertices)
        reach = np.linalg.norm(np.maximum(vertices.min(axis=0) - point, 0) + np.maximum(point - vertices.max(axis=0), 0))
        ranked.append((float(reach), k, ids))
    best = None
    for reach, k, ids in sorted(ranked, key=lambda r: r[0]):
        if reach >= within:
            break
        mesh = items[k]
        sq, tri, _ = _face_tree(mesh).squared_distance(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64), np.ascontiguousarray(mesh.faces, dtype=np.int64),
            point[None])
        distance = float(np.sqrt(sq[0]))
        if distance < within:
            within, best = distance, (int(ids[int(tri[0])]), cad_body(mesh, k), distance) + ((k,) if index else ())
    return best


def face_region(meshes, index, face, V, F, reach):
    """The triangles of a shown mesh (V, F) that lie on CAD face `face` of body `meshes[index]`, for
    highlighting the face the face menu acts on when the shown triangles carry no faces (a textured
    or offset body): those whose centre is within `reach` mm of that body, lies straightest off that
    face (`_straightest`), and is not closer to another body of the list. Sorted triangle numbers
    (int64)."""
    items = list(meshes) if isinstance(meshes, (list, tuple)) else [meshes]
    mesh = items[index]
    ids = cad_faces(mesh)
    V, F = np.asarray(V), np.asarray(F)
    if ids is None or not (ids == face).any() or len(F) == 0:
        return np.zeros(0, dtype=np.int64)
    Vb = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    Fb = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    corners = Vb[np.unique(Fb[ids == face])]
    lo, hi = corners.min(axis=0) - reach, corners.max(axis=0) + reach
    near = np.all((V >= lo) & (V <= hi), axis=1)
    rows = np.flatnonzero(near[F].any(axis=1))
    if not len(rows):
        return rows
    centres = V[F[rows]].astype(np.float64).mean(axis=1)
    sq, tri, closest = _face_tree(mesh).squared_distance(Vb, Fb, centres)
    keep = (ids[_straightest(mesh, centres, tri, closest)] == face) & (sq <= reach * reach)
    rows, centres, sq = rows[keep], centres[keep], sq[keep]
    # where bodies of a list meet, a surface nearer to the other body is that body's
    lo, hi = (centres.min(axis=0) - reach, centres.max(axis=0) + reach) if len(rows) else (hi, lo)
    for k, other in enumerate(items):
        box = other.bounds if k != index and cad_faces(other) is not None and len(other.faces) else None
        if not len(rows) or box is None or np.any(box[0] > hi) or np.any(box[1] < lo):
            continue
        mine = sq <= _face_tree(other).squared_distance(
            np.ascontiguousarray(other.vertices, dtype=np.float64), np.ascontiguousarray(other.faces, dtype=np.int64),
            centres)[0]
        rows, centres, sq = rows[mine], centres[mine], sq[mine]
    return rows


def _straightest(mesh, points, tri, closest):
    """The triangle of `mesh` each point lies straightest off: of the triangles through its closest
    point (`closest`, found on triangle `tri`), the one whose plane is farthest from the point. That is
    `tri` unless the closest point is on an edge or a corner: around a convex edge an offset surface is
    as close to both sides, and the nearest triangle alone picked either face at random (a speckled
    band); this splits the rounding down its middle, and a rounded corner into thirds."""
    Vb = np.asarray(mesh.vertices, dtype=np.float64)
    Fb = np.asarray(mesh.faces, dtype=np.int64)
    tri = np.asarray(tri, dtype=np.int64).copy()
    # barycentric weights of the closest points: one is 0 on an edge, two at a corner
    a, b, c = (Vb[Fb[tri, k]] for k in range(3))
    e1, e2, w = b - a, c - a, closest - a
    d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    w1, w2 = (w * e1).sum(1), (w * e2).sum(1)
    det = np.maximum(d11 * d22 - d12 * d12, 1e-300)
    u, v = (d22 * w1 - d12 * w2) / det, (d11 * w2 - d12 * w1) / det
    weights = np.column_stack([1 - u - v, u, v])
    border = np.flatnonzero(weights.min(axis=1) < 1e-6)
    if not len(border):
        return tri
    m = np.arange(len(border))
    rank = np.argsort(-weights[border], axis=1)
    corners = Fb[tri[border]]
    first = corners[m, rank[:, 0]]                                     # the corner nearest the point
    second = np.where(weights[border, rank[:, 1]] >= 1e-6, corners[m, rank[:, 1]], -1)   # on an edge: its other end
    # the triangles around `first` (vertex -> triangle lists, CSR), holding `second` too on an edge
    order = np.argsort(Fb.ravel(), kind="stable")
    start = np.concatenate([[0], np.cumsum(np.bincount(Fb.ravel(), minlength=len(Vb)))])
    count = start[first + 1] - start[first]
    owner = np.repeat(m, count)
    groups = np.concatenate([[0], np.cumsum(count)[:-1]])
    cand = order[np.repeat(start[first] - groups, count) + np.arange(count.sum())] // 3
    holds = (second[owner] < 0) | (Fb[cand] == second[owner][:, None]).any(axis=1)
    away = points[border][owner] - closest[border][owner]
    score = np.where(holds, np.abs(np.einsum("ij,ij->i", away, np.asarray(mesh.face_normals)[cand])), -1.0)
    tri[border] = cand[np.lexsort((-score, owner))[groups]]
    return tri


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
