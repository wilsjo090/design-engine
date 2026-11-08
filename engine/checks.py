"""Print checks on a displayed mesh: size, volume and weight, separate bodies and enclosed
cavities, open edges, and the wall thickness at every vertex. (Imports NVIDIA Warp: the server
imports this module only when a check is asked for.)"""
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import warp as wp

from . import backend

wp.config.quiet = True
wp.init()

CAVITY_MIN = 1e-3  # mm³: smaller closed shells are numerical dust, not cavities
# Ray casts one at a time: the server answers requests in threads, and two loading this module's kernels on the GPU
# at once (Measure asks for the wall and the section together) failed both with CUDA error 700, an illegal memory
# access that spoils the process's CUDA context for good (every later cast: "Failed to allocate").
_CAST_LOCK = threading.Lock()


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
    is enabled, else on Warp's CPU device in threads. Returns `outputs` float32 arrays. One at a
    time (`_CAST_LOCK`)."""
    with _CAST_LOCK:
        return _cast_now(kernel, V, F, points, dirs, max_t, outputs)


def _cast_now(kernel, V, F, points, dirs, max_t, outputs):
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


# Member sections (Measure): the smallest closed cut through the middle of a member, its plane found by
# turning it round the clicked point's normal (SECTION_TURNS directions, 180° / SECTION_TURNS apart), then a pattern
# search that may also lean it up to SECTION_TILT towards the normal. Only triangles near the middle are cut, out to
# SECTION_START wall thicknesses, doubled while no cut closes, up to SECTION_REACH: farther, it is a wall or a plate.
SECTION_TURNS = 18
SECTION_TILT = math.radians(60.0)
SECTION_START = 1.5
SECTION_REACH = 24.0
SECTION_STEPS = (math.radians(5.0), math.radians(0.25))   # the pattern search's first and last step
SECTION_POINTS = 4000      # most outline points sent back per loop


def _loops(P, T, c, d, e1, e2):
    """The cut of triangles T (local indices into P, (k, 3)) by the plane through c with normal d: closed loops
    as (2D points in (e1, e2) from c, signed area, 3D points), the outer ones counter-clockwise (positive) as seen
    from +d, holes clockwise; and whether any chain ran open (out of the triangles given). Loops are linked by
    the mesh edges their points lie on, so no tolerance is needed."""
    s = (P - c) @ d
    up = s > 0
    side = up[T]
    cut = side.any(axis=1) & ~side.all(axis=1)
    T, side = T[cut], side[cut]
    if not len(T):
        return [], False
    k = len(T)
    starts, ends = np.zeros(k, dtype=np.int64), np.zeros(k, dtype=np.int64)
    a_s, b_s = np.zeros(k, dtype=np.int64), np.zeros(k, dtype=np.int64)
    n = np.int64(len(P))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = T[:, i].astype(np.int64), T[:, j].astype(np.int64)
        key = np.minimum(a, b) * n + np.maximum(a, b)
        out = side[:, i] & ~side[:, j]     # the segment starts where the edge leaves the + side ...
        into = ~side[:, i] & side[:, j]    # ... and ends where one comes back into it (solid on its left)
        starts[out], ends[into] = key[out], key[into]
        a_s[out], b_s[out] = np.minimum(a, b)[out], np.maximum(a, b)[out]
    # each segment's first point, on its start edge, the same whichever triangle computes it
    sa, sb = s[a_s], s[b_s]
    pts = P[a_s] + (P[b_s] - P[a_s]) * (sa / (sa - sb))[:, None]
    order = np.argsort(starts, kind="stable")
    at = np.searchsorted(starts[order], ends)
    found = (at < k) & (starts[order[np.minimum(at, k - 1)]] == ends)
    nxt = np.where(found, order[np.minimum(at, k - 1)], -1)
    flat = np.column_stack([(pts - c) @ e1, (pts - c) @ e2])
    seen = np.zeros(k, dtype=bool)
    loops, open_ = [], False
    for first in range(k):
        if seen[first]:
            continue
        chain, i = [], first
        while i >= 0 and not seen[i]:
            seen[i] = True
            chain.append(i)
            i = int(nxt[i])
        if i != first:     # ran out of triangles, or into a chain already walked from its middle
            open_ = True
            continue
        q = flat[chain]
        x, y = q[:, 0], q[:, 1]
        area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
        loops.append((q, area, pts[chain]))
    return loops, open_


def _inside(q, p):
    """Whether the 2D point p is inside the closed polygon q (n, 2) (even-odd crossings)."""
    x, y = q[:, 0], q[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = (y > p[1]) != (y2 > p[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        at = x + (p[1] - y) * (x2 - x) / (y2 - y)
    return bool((cross & (at > p[0])).sum() % 2)


def _member_cut(P, T, c, d, e1, e2, reach):
    """The region of the cut (see _loops) that contains c, the plane's centre: (area, outer loop, holes) or None
    when no closed loop goes round it (the cut runs out of the triangles given) or its outline reaches farther
    than `reach` mm from c (a plate of a few big triangles closes within any crop)."""
    loops, _ = _loops(P, T, c, d, e1, e2)
    origin = np.zeros(2)
    around = [lp for lp in loops if lp[1] > 0 and _inside(lp[0], origin)]
    if not around:
        return None
    outer = min(around, key=lambda lp: lp[1])
    if float(np.einsum("ij,ij->i", outer[0], outer[0]).max()) > reach * reach:
        return None
    within = [lp for lp in loops if lp is not outer and _inside(outer[0], lp[0][0])]
    islands = [lp for lp in within if lp[1] > 0]    # (pieces standing in its holes: not its own)
    holes = [lp for lp in within if lp[1] < 0 and not any(_inside(i[0], lp[0][0]) for i in islands)]
    return outer[1] + sum(h[1] for h in holes), outer, holes


def _widths(q):
    """(narrowest, widest) across a 2D outline: its extent along every direction, half a degree apart."""
    from scipy.spatial import ConvexHull
    try:
        q = q[ConvexHull(q).vertices]
    except Exception:   # (degenerate: all on a line)
        pass
    a = np.linspace(0.0, math.pi, 360, endpoint=False)
    span = np.ptp(q @ np.stack([np.cos(a), np.sin(a)]), axis=0)
    return float(span.min()), float(span.max())


def _thin_out(points):
    step = max(1, -(-len(points) // SECTION_POINTS))
    return np.asarray(points[::step], dtype=np.float64).round(4).tolist()


def member_section(V, F, point, normal, thickness=None):
    """The cross-section of the member under a point of a closed mesh's surface (`normal`: the surface's outward
    normal there): the smallest closed cut through the member's middle, the point halfway to the opposite wall
    straight in (`thickness`: that wall thickness when known). Returns None when there is no opposite wall; a
    dict {"open": True, "thickness", "centre"} when every cut runs out farther than SECTION_REACH wall thicknesses
    (a wall or a plate, not a member); else {"area" (mm2, holes taken off), "diameter" (of a round member of that
    area), "narrowest", "widest" (mm across the outline), "thickness", "centre", "normal" (of the cut), "outline"
    (3D points of the outer loop), "holes" (lists of 3D points)}."""
    V = np.asarray(V)
    F = np.asarray(F)
    n = np.asarray(normal, dtype=np.float64)
    if not len(F) or not np.linalg.norm(n) > 0:
        return None
    n /= np.linalg.norm(n)
    point = np.asarray(point, dtype=np.float64)
    t = float(thickness) if thickness is not None else float(
        ray_thickness(V, F, point[None].astype(np.float32), n[None].astype(np.float32), 1000.0)[0])
    if not np.isfinite(t) or t <= 0:
        return None
    c = point - n * (t / 2)
    # how far each triangle's nearest point can be from c, at least (its bounding sphere), in chunks
    near = np.empty(len(F), dtype=np.float32)
    for s in range(0, len(F), 1 << 20):
        tri = V[F[s:s + (1 << 20)]].astype(np.float32)
        mid = tri.mean(axis=1)
        rad = np.sqrt(((tri - mid[:, None]) ** 2).sum(axis=2).max(axis=1))
        near[s:s + len(tri)] = np.linalg.norm(mid - c.astype(np.float32), axis=1) - rad
        del tri, mid, rad
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    def plane(theta, phi):
        d = math.cos(phi) * (math.cos(theta) * u + math.sin(theta) * v) + math.sin(phi) * n
        e1 = np.cross(n, d)     # (in the plane, as long as it leans less than 90°; then e2 so that e1 x e2 = d)
        e1 /= np.linalg.norm(e1)
        return d, e1, np.cross(d, e1)

    def crop(r):
        tris = np.flatnonzero(near <= r)
        used, local = np.unique(F[tris], return_inverse=True)
        # (corners at the same place are one: an unwelded mesh's triangles would otherwise share no edges)
        P, weld = np.unique(V[used].astype(np.float64), axis=0, return_inverse=True)
        return P, weld.ravel()[local.ravel()].reshape(-1, 3)

    def search(P, T, start=None):
        """((theta, phi), cut) of the smallest closed cut, or None."""
        cache = {}

        def area(theta, phi):
            k = (round(theta, 9), round(phi, 9))
            if k not in cache:
                cache[k] = _member_cut(P, T, c, *plane(theta, phi), SECTION_REACH * t)
            return cache[k]
        if start is None:
            tries = [(math.pi * i / SECTION_TURNS, 0.0) for i in range(SECTION_TURNS)]
        else:
            tries = [start]
        best = None
        for at in tries:
            got = area(*at)
            if got is not None and (best is None or got[0] < best[1][0]):
                best = (at, got)
        if best is None:
            return None
        step = SECTION_STEPS[0]
        while step >= SECTION_STEPS[1]:
            (theta, phi), cut = best
            moved = False
            for at in ((theta + step, phi), (theta - step, phi), (theta, phi + step), (theta, phi - step)):
                if abs(at[1]) > SECTION_TILT:
                    continue
                got = area(*at)
                if got is not None and got[0] < cut[0]:
                    best, moved = (at, got), True
                    break
            if not moved:
                step /= 2
        return best

    r, found = SECTION_START * t, None
    while found is None and r <= SECTION_REACH * t:
        found = search(*crop(r))
        if found is None:
            r *= 2
    if found is None:
        return {"open": True, "thickness": t, "centre": c.round(4).tolist()}
    # cut again with room round the whole outline, so leaning cuts near the best one close too
    reach = float(np.linalg.norm(found[1][1][0], axis=1).max())
    if 2.0 * reach > r:
        found = search(*crop(2.0 * reach), start=found[0]) or found
    (theta, phi), (area, outer, holes) = found
    d = plane(theta, phi)[0]
    narrowest, widest = _widths(outer[0])
    return {"area": float(area), "diameter": float(2.0 * math.sqrt(max(area, 0.0) / math.pi)),
            "narrowest": narrowest, "widest": widest, "thickness": t, "centre": c.round(4).tolist(),
            "normal": d.round(6).tolist(), "outline": _thin_out(outer[2]), "holes": [_thin_out(h[2]) for h in holes]}


def thickness_report(thickness, min_wall):
    """Summary of per-vertex wall thickness: thinnest, a robust low value and the share of the
    surface below `min_wall` mm."""
    t = thickness[np.isfinite(thickness)]
    if not len(t):
        return {}
    return {"thinnest": float(t.min()), "p1": float(np.quantile(t, 0.01)), "median": float(np.median(t)),
            "below": float((t < min_wall).mean()), "min_wall": float(min_wall)}
