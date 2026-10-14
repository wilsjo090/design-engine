"""Distance from grid points to the nearest of many line segments (the Voronoi foam's struts).

One NVIDIA Warp kernel for both devices: a BVH over the segments' boxes, and for each point a
cube search that doubles until the nearest segment found is no farther than the cube's half size
(every segment outside the cube is farther than that, so the result is exact). On the GPU it is
~750x faster than the KD-tree + numpy version it replaced; on the CPU Warp runs the same kernel
in threads (~80x faster with 16 threads).
"""
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import warp as wp

from . import backend

wp.config.quiet = True
wp.init()


@wp.kernel
def _segment_grid(bvh: wp.uint64, a: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3),
                  inv: wp.array(dtype=float), radius: float, origin: wp.vec3, sx: int, sy: int, sz: int,
                  voxel: float, ny: int, nz: int, first: int, out: wp.array(dtype=float)):
    """Distance to the nearest segment a + t*d (t in 0..1) for grid points first, first+1, ...
    (flattened C order)."""
    tid = wp.tid() + first
    i = tid // (ny * nz)
    j = (tid // nz) % ny
    k = tid % nz
    p = origin + wp.vec3(float(i + sx), float(j + sy), float(k + sz)) * voxel  # origin = grid.base
    r = radius
    best = float(1.0e30)
    tries = int(0)
    while tries < 40:
        query = wp.bvh_query_aabb(bvh, p - wp.vec3(r, r, r), p + wp.vec3(r, r, r))
        n = int(0)
        while wp.bvh_query_next(query, n):
            pa = p - a[n]
            t = wp.clamp(wp.dot(pa, d[n]) * inv[n], 0.0, 1.0)
            best = wp.min(best, wp.length(pa - t * d[n]))
        if best <= r:
            tries = 40
        r = r * 2.0
        tries += 1
    out[tid] = best


_TREES = OrderedDict()  # (key, device) -> (wp.Bvh, its arrays): built once per process
PIECE = 4_000_000       # grid points per CPU launch (each thread takes pieces of this size)


def _tree(segments, key, device):
    entry = _TREES.get((key, device)) if key is not None else None
    if entry is None:
        seg = np.asarray(segments.get() if hasattr(segments, "get") else segments, dtype=np.float64)
        a, b = seg[:, 0], seg[:, 1]
        d = b - a
        pad = 1e-3  # mm: float32 rounding never drops a segment from a box that touches it
        vec = lambda x: wp.array(np.ascontiguousarray(x, dtype=np.float32), dtype=wp.vec3, device=device)
        arrs = (vec(a), vec(d), wp.array((1.0 / np.maximum((d * d).sum(axis=1), 1e-12)).astype(np.float32),
                                         dtype=float, device=device),
                vec(np.minimum(a, b) - pad), vec(np.maximum(a, b) + pad))
        entry = (wp.Bvh(arrs[3], arrs[4]), arrs)
        if key is not None:
            _TREES[(key, device)] = entry
            while len(_TREES) > 4:
                _TREES.popitem(last=False)
    elif key is not None:
        _TREES.move_to_end((key, device))
    return entry


def segment_distance(segments, grid, radius, key=None, gpu=False):
    """Distance from every point of `grid` to the nearest of `segments` ((n, 2, 3) array or a
    shared.Shared one): a CuPy array on the GPU, else numpy. `radius`: first search size in mm
    (about the typical distance; each doubling repeats the search). `key` names the segments,
    so the BVH is built once per process."""
    device = "cuda" if gpu else "cpu"
    bvh, (a, d, inv, _, _) = _tree(segments, key, device)
    common = [bvh.id, a, d, inv, float(radius), wp.vec3(*[float(c) for c in grid.base]), *grid.start,
              float(grid.voxel), grid.shape[1], grid.shape[2]]
    if gpu:
        cp = backend.cupy()
        out = cp.empty(grid.size, dtype=cp.float32)
        cp.cuda.Device().synchronize()  # Warp runs on its own stream
        wp.launch(_segment_grid, dim=grid.size, inputs=[*common, 0, wp.from_dlpack(out)], device=device)
        wp.synchronize_device(device)
        return out.reshape(grid.shape)
    out = np.empty(grid.size, dtype=np.float32)
    target = wp.array(out, dtype=float, device="cpu", copy=False)
    # Warp's CPU launches release the GIL: threads (as many as this process may use; worker
    # processes get their share in IGL_NUM_THREADS) run pieces of the grid side by side
    threads = int(os.environ.get("IGL_NUM_THREADS") or os.cpu_count() or 4)
    step = max(1, min(PIECE, -(-grid.size // (4 * threads))))
    starts = range(0, grid.size, step)
    launch = lambda s: wp.launch(_segment_grid, dim=min(step, grid.size - s), inputs=[*common, s, target],
                                 device="cpu")
    if threads <= 1 or len(starts) == 1:
        for s in starts:
            launch(s)
    else:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(launch, starts))
    return out.reshape(grid.shape)
