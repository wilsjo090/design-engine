"""Turn an evaluated field back into a triangle mesh and export it."""
import numpy as np
import trimesh
from skimage.measure import marching_cubes

from .backend import to_host, xp_of


def finite_values(values, voxel):
    """`values` with NaN and infinite samples replaced, and how many there were. Marching cubes
    would put vertices at NaN positions there (an STL full of broken triangles). NaN and +inf
    count as one voxel outside, -inf as one voxel inside (the same everywhere, so blocks meshed
    separately still agree)."""
    xp = xp_of(values)
    bad = ~xp.isfinite(values)
    count = int(bad.sum())
    if count:
        values = xp.where(bad, xp.where(values < 0, -voxel, voxel), values).astype(xp.float32)
    return values, count


def touches_box(values):
    """Whether the body reaches the outermost samples of the grid: there it is cut flat."""
    return any(float(values.take(i, axis=a).min()) < 0 for a in range(3) for i in (0, -1))


def field_to_mesh(values, grid):
    """Extract the zero level of `values` (numpy or CuPy) as a watertight mesh in world coordinates.
    `mesh.metadata` says how many samples were not finite (`nonfinite`) and whether the body was
    cut off by the grid's box (`clipped`)."""
    xp = xp_of(values)
    values, nonfinite = finite_values(values, grid.voxel)
    clipped = touches_box(values)
    if float(values.min()) >= 0:
        raise ValueError("Field has no surface inside the grid (no sign change): nothing is inside, or the body "
                         "is thinner than the voxel size")
    if float(values.max()) <= 0:
        raise ValueError("Field has no surface inside the grid (no sign change): the body fills the whole box it "
                         "is meshed in (more Padding, or a smaller offset?)")
    # Values (almost) exactly zero put several marching-cubes vertices on the same grid corner.
    # They are distinct in memory, but STL stores no connectivity, so slicers re-merge them by
    # position and the surface gets pinched. Pushing them 1% of a voxel off zero keeps every
    # vertex distinct (surface moves ~0.001 mm).
    eps = 0.01 * grid.voxel
    values = xp.where(xp.abs(values) < eps, xp.where(values < 0, -eps, eps), values).astype(xp.float32)
    # Pad with "outside" so surfaces touching the grid boundary still close.
    padded = xp.pad(values, 1, mode="constant", constant_values=max(float(values.max()), grid.voxel))
    del values
    if xp is np:
        verts, faces, _, _ = marching_cubes(padded, level=0.0, spacing=(grid.voxel,) * 3)
    else:
        verts, faces = _marching_cubes_gpu(padded, grid.voxel)
    del padded
    verts += grid.origin - grid.voxel
    # Marching cubes output is already closed and shares vertices correctly. trimesh's
    # processing (merging near-coincident vertices, dropping zero-area triangles) can pinch or
    # open it where the surface passes through grid corners, so keep its topology untouched.
    mesh = trimesh.Trimesh(verts, faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    mesh.metadata.update(nonfinite=nonfinite, clipped=clipped)
    return mesh


def _marching_cubes_gpu(values, voxel):
    """Marching cubes of a CuPy array with NVIDIA Warp: ~8x faster than skimage on the CPU and only
    the mesh comes back to the CPU. Same vertices and triangles as skimage (one vertex per crossed
    grid edge, closed surface); vertex positions are computed in float32 (~1e-6 mm here)."""
    import warp as wp
    import warp.geometry
    from . import backend
    backend.cupy().cuda.Device().synchronize()  # Warp runs on its own CUDA stream
    verts, indices = warp.geometry.IsoSurfaceMarchingCubes.extract(wp.from_dlpack(values), 0.0)
    wp.synchronize_device()
    return verts.numpy().astype(np.float64) * voxel, indices.numpy().reshape(-1, 3).astype(np.int64)


AGGRESSION = 3  # quadric collapse: low values avoid sliver triangles (specks on curved areas)


def simplify(verts, faces, keep, voxel, preserve_border=False):
    """Keep about `keep` of the triangles (quadric edge collapse, pyfqmr). The collapse error
    threshold is in squared length, so it is scaled to the voxel size: the same result at any
    voxel size. (Unscaled, at 0.1 mm voxels it stopped at 48% of a petal's triangles for keep =
    0.3, after 0.45 s; scaled it reaches 30% in 0.21 s, within 0.005 mm of the full mesh.)
    Returns None if the result is not a clean surface (an edge shared by more than 2 triangles,
    which the algorithm doesn't rule out)."""
    import pyfqmr
    s = pyfqmr.Simplify()
    s.setMesh(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    s.simplify_mesh(target_count=max(4, int(len(faces) * keep)), aggressiveness=AGGRESSION,
                    preserve_border=preserve_border, alpha=1e-9 / voxel ** 2, verbose=False)
    V, F, _ = s.getMesh()
    F = np.asarray(F, dtype=np.int64)
    edges = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    if (np.unique(edges[:, 0] * np.int64(len(V)) + edges[:, 1], return_counts=True)[1] > 2).any():
        return None
    return np.asarray(V, dtype=np.float64), F


def decimate(mesh, keep, voxel):
    """Keep about `keep` (0..1] of the triangles of a mesh made with `voxel` mm voxels."""
    if keep >= 1.0:
        return mesh
    simple = simplify(mesh.vertices, mesh.faces, keep, voxel)
    if simple is None:  # rare: the old way (unscaled threshold, stops earlier) on this body
        return snap_float32(mesh.simplify_quadric_decimation(percent=1.0 - keep, aggression=2))
    return snap_float32(trimesh.Trimesh(*simple, process=False))


def snap_float32(mesh):
    """Decimation and Boolean unions can leave vertices that are distinct in memory but identical
    once written as float32 (STL/3MF), which pinches the surface on reload. Snap to float32,
    merge, and drop the triangles that collapse."""
    mesh.vertices = mesh.vertices.astype(np.float32).astype(np.float64)
    mesh.merge_vertices()
    f = mesh.faces
    mesh.update_faces((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2]))
    mesh.remove_unreferenced_vertices()
    return mesh


SPECK_VOLUME = 1.0  # mm³. Textures can pinch off tiny solid specks and bubbles inside the solid
                    # (the textured lamp center: 5 specks and 47 bubbles, all under 0.1 mm³).


def fuse(meshes):
    """Boolean union of closed meshes into one watertight mesh (manifold3d). Meshes only join
    where they overlap; touching is not enough. Separate pieces under SPECK_VOLUME are dropped.
    Returns (mesh, number of separate solids, number of specks dropped)."""
    import manifold3d as m3d  # imported here: only needed when fusing

    solids = []
    for i, mesh in enumerate(meshes):
        m = m3d.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                     tri_verts=np.asarray(mesh.faces, np.uint32))
        solid = m3d.Manifold(m)
        if solid.status() != m3d.Error.NoError and m.merge():  # weld vertices duplicated by position
            solid = m3d.Manifold(m)
        if solid.status() != m3d.Error.NoError:
            raise ValueError(f"mesh {i + 1} of {len(meshes)} is not closed ({solid.status().name}), "
                             f"so it cannot be fused")
        solids.append(solid)
    result = m3d.Manifold.batch_boolean(solids, m3d.OpType.Add) if len(solids) > 1 else solids[0]
    del solids
    parts = result.decompose()
    keep = [p for p in parts if abs(p.volume()) >= SPECK_VOLUME] or parts
    if len(keep) < len(parts):
        result = m3d.Manifold.compose(keep) if len(keep) > 1 else keep[0]
    out = result.to_mesh()
    mesh = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=False)
    # slivers where surfaces meet almost tangentially collapse to zero area in float32
    return snap_float32(mesh), len(keep), len(parts) - len(keep)


def signed_volume(V, F, chunk=1_000_000):
    """Volume enclosed by the triangles, in chunks (trimesh's .volume builds several arrays of
    the mesh's full size: +3.9 GB for a 12M-triangle part)."""
    total = 0.0
    for s in range(0, len(F), chunk):
        a, b, c = (V[F[s:s + chunk, k]] for k in range(3))
        total += float(np.einsum("ij,ij->", a, np.cross(b, c)))
    return total / 6.0


def export(mesh, path):
    """Write STL, 3MF, OBJ or PLY depending on the file extension."""
    if str(path).lower().endswith(".stl"):
        _export_stl(mesh, path)
    else:  # without a CAD body's per-triangle data (PLY would try to write the corner UVs)
        trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).export(str(path))


def _export_stl(mesh, path, chunk=1_000_000):
    """Binary STL written in chunks: trimesh's exporter keeps the full triangle array and normals
    of the mesh in memory (~1.3 GB for 14M triangles)."""
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces)
    record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")])
    with open(path, "wb") as f:
        f.write(b"design engine".ljust(80, b" "))
        f.write(np.uint32(len(F)).tobytes())
        for s in range(0, len(F), chunk):
            tri = V[F[s:s + chunk]]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            out = np.zeros(len(tri), dtype=record)
            out["normal"], out["corners"] = n, tri
            f.write(out.tobytes())


# --- reduction by tolerance -------------------------------------------------------------------
REDUCE_ROUNDS = 4  # rounds of "keep the original triangles around what went wrong, try again"
REDUCE_ITERATIONS = 40
REDUCE_AGGRESSION = 3


def reduce(vertices, faces, tolerance, check=None, frozen=None, snap=True):
    """Fewer triangles only where they are not needed: edges collapse while the surface moves less
    than about `tolerance` (mm, or the vertices' units) by quadric error (pyfqmr), so a flat face
    ends up as a few big triangles and a texture keeps what it needs. (Key cabinet, 5.7M
    triangles: at 0.01 mm 1.10M left, surface within 0.006 mm, 28 s; at 0.05 mm 0.46M, within
    0.09 mm: on curved detail the quadric error can let it move up to ~2x the tolerance.)

    The collapses can fold two triangles onto each other or pinch an edge (~1 in 100k-1M
    triangles): folds are removed, and around anything still wrong the original triangles are
    kept and the rest is reduced again with that patch's edges locked. Open edges of the input (a
    block's border) stay exactly where they are. `frozen`: triangles to keep as they are from the
    start. `snap`: snap to float32 as files store it (blocks are snapped after stitching). Every
    edge ends up used by exactly two triangles (open input edges: once); if that cannot be reached
    the input is returned. Returns (vertices, faces)."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if not len(F) or tolerance <= 0:
        return V, F
    frozen = np.zeros(len(F), dtype=bool) if frozen is None else np.array(frozen, dtype=bool)
    radius = max(30.0 * tolerance, 0.5)  # kept around each problem (doubles each round)
    centres = open_keys = None
    for _ in range(REDUCE_ROUNDS):
        if check is not None:
            check()
        v, f = _reduce_rest(V, F, frozen, tolerance, snap)
        overused, single = _edge_problems(v, f)
        if len(single):  # edges used once: fine if they were open in the input (a block border)
            if open_keys is None:
                open_keys = _keys(V[_open_vertices(F)])
            single = single[~np.isin(_keys(v[single]), open_keys)]
        problems = np.union1d(overused, single)
        if not len(problems):
            return v, f
        if centres is None:
            centres = V[F].mean(axis=1)
        from scipy.spatial import cKDTree
        frozen |= cKDTree(v[problems]).query(centres, distance_upper_bound=radius)[0] <= radius
        radius *= 2
    return V, F


def _reduce_rest(V, F, frozen, tolerance, snap=True):
    """Reduce the triangles that are not frozen (their border with the frozen ones stays put) and
    join both parts, folds removed (float32-snapped if `snap`)."""
    import pyfqmr
    s = pyfqmr.Simplify()
    s.setMesh(V, F[~frozen])
    # The collapse threshold grows each iteration (alpha * (iteration + K)^aggressiveness) and is
    # made to end at tolerance² (squared distance): collapses go on until the surface would move
    # more than that, whatever the triangle count. (pyfqmr's own lossless mode cannot keep borders
    # in place, and simplify_mesh(lossless=True) hardly reduces at all.)
    s.simplify_mesh(target_count=4, update_rate=5, aggressiveness=REDUCE_AGGRESSION, max_iterations=REDUCE_ITERATIONS,
                    alpha=tolerance * tolerance / (REDUCE_ITERATIONS - 1 + 3) ** REDUCE_AGGRESSION, K=3,
                    preserve_border=True, verbose=False)
    v, f, _ = s.getMesh()
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)
    if frozen.any():  # the border vertices did not move: they meet the frozen part by position
        v, f = np.concatenate([V, v]), np.concatenate([F[frozen], f + len(V)])
    return _clean(v.astype(np.float32).astype(np.float64) if snap else v, f)


def _clean(v, f):
    """Merge vertices at the same position, drop collapsed triangles and folds (two triangles on
    the same three vertices), drop unused vertices."""
    v, inverse = np.unique(v, axis=0, return_inverse=True)
    f = inverse.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    # copies of a triangle on the same three vertices cancel in pairs (a fold is two of them back
    # to back); an odd one out is part of the surface and stays
    _, first, count = np.unique(np.sort(f, axis=1), axis=0, return_index=True, return_counts=True)
    f = f[np.sort(first[count % 2 == 1])]
    used, f = np.unique(f, return_inverse=True)
    return v[used], f.reshape(-1, 3)


def _edge_counts(n, f):
    """(edge keys, how many triangles use each edge) for faces `f` over `n` vertices."""
    e = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    return np.unique(e[:, 0] * np.int64(n) + e[:, 1], return_counts=True)


def _edge_problems(v, f):
    """(vertices of edges used by more than two triangles, vertices of edges used by one)."""
    key, count = _edge_counts(len(v), f)
    ends = lambda k: np.unique(np.concatenate([k // len(v), k % len(v)])).astype(np.int64)
    return ends(key[count > 2]), ends(key[count == 1])


def _open_vertices(F):
    """Vertices on edges used by only one triangle."""
    n = int(F.max()) + 1 if len(F) else 0
    key, count = _edge_counts(n, F)
    k = key[count == 1]
    return np.unique(np.concatenate([k // n, k % n])).astype(np.int64)


def _keys(points):
    """Exact positions as single comparable values."""
    return np.ascontiguousarray(points, dtype=np.float64).view(np.dtype((np.void, 24))).ravel()
