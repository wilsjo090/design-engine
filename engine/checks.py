"""Print checks on a displayed mesh: size, volume and weight, separate bodies and enclosed
cavities, open edges, and the wall thickness at every vertex. (Imports NVIDIA Warp: the server
imports this module only when a check is asked for.)"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import warp as wp

from . import backend

wp.config.quiet = True
wp.init()

CAVITY_MIN = 1e-3  # mm³: smaller closed shells are numerical dust, not cavities


def mesh_report(V, F, density=1.24):
    """Numbers for a mesh (V float (n, 3), F int (m, 3)). `density` in g/cm³ (PLA ~1.24)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    report = {"triangles": int(len(F)), "vertices": int(len(V))}
    if not len(F):
        return report
    lo, hi = V.min(axis=0), V.max(axis=0)
    report["size"] = [float(x) for x in hi - lo]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    report["area"] = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    # each closed shell on its own: shells with negative volume are cavities (surfaces facing inwards)
    n = len(V)
    edges = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    graph = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    shell = label[F[:, 0]]
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    shell_volume = np.bincount(shell, weights=volume)
    shell_volume = shell_volume[np.unique(shell)]
    bodies = shell_volume[shell_volume > CAVITY_MIN]
    cavities = shell_volume[shell_volume < -CAVITY_MIN]
    report["volume"] = float(shell_volume.sum())
    report["weight"] = report["volume"] / 1000.0 * float(density)
    report["bodies"] = int(len(bodies))
    report["cavities"] = int(len(cavities))
    report["cavity_volume"] = float(-cavities.sum())
    key = np.sort(edges, axis=1)
    _, uses = np.unique(key[:, 0] * np.int64(n) + key[:, 1], return_counts=True)
    report["open_edges"] = int((uses != 2).sum())
    return report


def vertex_normals(V, F):
    """Area-weighted vertex normals (unit length)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # length = 2 x area
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, F[:, k], fn)
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(length > 0, length, 1.0)


@wp.kernel
def _thickness(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
               max_t: float, first: int, out: wp.array(dtype=float)):
    """Distance from each vertex straight into the part to the opposite wall (-1: none within
    max_t). Hits from outside (the ray left through a neighbouring triangle at a crease) and hits
    right at the start are stepped over."""
    tid = wp.tid() + first
    d = -normals[tid]
    p = points[tid] + d * 1.0e-4
    travelled = float(1.0e-4)
    result = float(-1.0)
    tries = int(0)
    while tries < 6:
        q = wp.mesh_query_ray(mesh, p, d, max_t)
        if not q.result:
            tries = 6
        elif q.sign < 0.0 and travelled + q.t > 1.0e-3:
            result = travelled + q.t
            tries = 6
        else:
            step = q.t + 1.0e-4
            p = p + d * step
            travelled = travelled + step
            tries += 1
    out[tid] = result


@wp.kernel
def _first_hit(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3), max_t: float,
               first: int, hit: wp.array(dtype=float), sign: wp.array(dtype=float)):
    """Distance along each ray to the first triangle it meets (-1: none within max_t) and the
    hit's side (+1: the triangle faces the ray, i.e. the ray enters the part; -1: it leaves)."""
    tid = wp.tid() + first
    q = wp.mesh_query_ray(mesh, points[tid], dirs[tid], max_t)
    if q.result:
        hit[tid] = q.t
        sign[tid] = q.sign
    else:
        hit[tid] = -1.0
        sign[tid] = 0.0


def _cast(kernel, V, F, points, dirs, max_t, outputs=1):
    """Run a ray kernel for every (point, direction) against the mesh (V, F): on the GPU when it
    is enabled, else on Warp's CPU device in threads. Returns `outputs` float32 arrays."""
    device = "cuda" if backend.gpu_enabled() else "cpu"
    mesh = wp.Mesh(points=wp.array(np.ascontiguousarray(V, dtype=np.float32), dtype=wp.vec3, device=device),
                   indices=wp.array(np.ascontiguousarray(F, dtype=np.int32).ravel(), dtype=int, device=device))
    p = wp.array(np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device=device)
    d = wp.array(np.ascontiguousarray(dirs, dtype=np.float32), dtype=wp.vec3, device=device)
    n = len(points)
    if not n:
        return [np.zeros(0, dtype=np.float32) for _ in range(outputs)]
    if device != "cpu":
        outs = [wp.empty(n, dtype=float, device=device) for _ in range(outputs)]
        wp.launch(kernel, dim=n, inputs=[mesh.id, p, d, float(max_t), 0, *outs], device=device)
        wp.synchronize_device(device)
        return [np.asarray(o.numpy(), dtype=np.float32) for o in outs]
    results = [np.empty(n, dtype=np.float32) for _ in range(outputs)]
    targets = [wp.array(r, dtype=float, device="cpu", copy=False) for r in results]
    threads = os.cpu_count() or 4
    step = max(1, -(-n // (4 * threads)))
    launch = lambda s: wp.launch(kernel, dim=min(step, n - s), inputs=[mesh.id, p, d, float(max_t), s, *targets],
                                 device="cpu")
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(launch, range(0, n, step)))
    return results


def wall_thickness(V, F, max_t=100.0):
    """Wall thickness at every vertex (float32 mm; NaN where no opposite wall was found within
    max_t): the length of a ray from the vertex straight into the part. Runs on the GPU when it
    is enabled, else on Warp's CPU device in threads."""
    return ray_thickness(V, F, V, vertex_normals(V, F), max_t)


def ray_thickness(V, F, points, normals, max_t=100.0):
    """Distance from each point along -normal to where the ray leaves the part through the opposite
    wall (float32 mm; NaN if not within max_t). Walls the ray first enters from outside are
    stepped over, so points a little above the surface measure from where they are."""
    result = _cast(_thickness, V, F, points, normals, max_t)[0]
    result[result < 0] = np.nan
    return result


def ray_first_hit(V, F, points, dirs, max_t=100.0):
    """(distance, side) of the first triangle along each ray: distance -1 when none is within
    max_t; side +1 when the ray enters the part there, -1 when it leaves it."""
    return tuple(_cast(_first_hit, V, F, points, dirs, max_t, outputs=2))


def thickness_report(thickness, min_wall):
    """Summary of per-vertex wall thickness: thinnest, a robust low value and the share of the
    surface below `min_wall` mm."""
    t = thickness[np.isfinite(thickness)]
    if not len(t):
        return {}
    return {"thinnest": float(t.min()), "p1": float(np.quantile(t, 0.01)), "median": float(np.median(t)),
            "below": float((t < min_wall).mean()), "min_wall": float(min_wall)}
