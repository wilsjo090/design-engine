"""The frame body's Warp kernel (see `geometry.py`): imported only when a frame is drawn, so the
editor and its workers do not start Warp for nothing."""
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import warp as wp

from .. import backend

wp.config.quiet = True
wp.init()


@wp.func
def _box2(qy: float, qz: float, r: wp.vec4) -> float:
    dy = wp.abs(qy - (r[0] + r[1]) * 0.5) - (r[1] - r[0]) * 0.5
    dz = wp.abs(qz - (r[2] + r[3]) * 0.5) - (r[3] - r[2]) * 0.5
    oy = wp.max(dy, 0.0)
    oz = wp.max(dz, 0.0)
    return wp.sqrt(oy * oy + oz * oz) + wp.min(wp.max(dy, dz), 0.0)


@wp.kernel
def _frame_grid(bvh: wp.uint64, kind: wp.array(dtype=int), chain: wp.array(dtype=int), width: wp.array(dtype=float),
                a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), ey: wp.array(dtype=wp.vec3),
                ez: wp.array(dtype=wp.vec3), p0: wp.array(dtype=wp.vec3), n0: wp.array(dtype=wp.vec3),
                p1: wp.array(dtype=wp.vec3), n1: wp.array(dtype=wp.vec3), rad: wp.array(dtype=float),
                wall: wp.array(dtype=float), rects: wp.array(dtype=wp.vec4), rsign: wp.array(dtype=int),
                cen: wp.array(dtype=wp.vec2), rot: wp.array(dtype=wp.vec2), fillet: float, factor: float,
                cap: float, origin: wp.vec3, sx: int, sy: int, sz: int, voxel: float, ny: int, nz: int, first: int,
                out: wp.array(dtype=float)):
    tid = wp.tid() + first
    i = tid // (ny * nz)
    j = (tid // nz) % ny
    k = tid % nz
    p = origin + wp.vec3(float(i + sx), float(j + sy), float(k + sz)) * voxel
    d1 = cap
    c1 = int(-2)
    w1 = float(0.0)
    d2 = cap
    c2 = int(-2)
    w2 = float(0.0)
    query = wp.bvh_query_aabb(bvh, p, p)
    n = int(0)
    while wp.bvh_query_next(query, n):
        kd = kind[n]
        d = float(0.0)
        if kd == 0:  # round bar: capsule
            ab = b[n] - a[n]
            t = wp.clamp(wp.dot(p - a[n], ab) / wp.max(wp.dot(ab, ab), 1.0e-12), 0.0, 1.0)
            d = wp.length(p - (a[n] + t * ab)) - rad[n]
        elif kd == 3:  # ball
            d = wp.length(p - a[n]) - rad[n]
        else:
            rel = p - a[n]
            y = wp.dot(rel, ey[n])
            z = wp.dot(rel, ez[n])
            flat = float(0.0)
            if kd == 1:  # round tube
                rho = wp.sqrt(y * y + z * z)
                flat = wp.abs(rho - (rad[n] - wall[n] * 0.5)) - wall[n] * 0.5
            else:  # rectangles: union of the solid ones, minus the holes
                cs = rot[n]
                cc = cen[n]
                qy = cc[0] + y * cs[0] - z * cs[1]
                qz = cc[1] + y * cs[1] + z * cs[0]
                solid = float(1.0e30)
                hole = float(-1.0e30)
                for r in range(4):
                    sg = rsign[n * 4 + r]
                    if sg > 0:
                        solid = wp.min(solid, _box2(qy, qz, rects[n * 4 + r]))
                    if sg < 0:
                        hole = wp.max(hole, -_box2(qy, qz, rects[n * 4 + r]))
                flat = wp.max(solid, hole)
            along = wp.max(wp.dot(p - p0[n], n0[n]), wp.dot(p - p1[n], n1[n]))
            if flat > 0.0 and along > 0.0:
                d = wp.sqrt(flat * flat + along * along)
            else:
                d = wp.max(flat, along)
        cn = chain[n]
        wn = width[n]
        # the two nearest different bars (each at its nearest piece)
        if cn == c1:
            d1 = wp.min(d1, d)
        elif cn == c2:
            d2 = wp.min(d2, d)
            if d2 < d1:
                td = d1
                tc = c1
                tw = w1
                d1 = d2
                c1 = c2
                w1 = w2
                d2 = td
                c2 = tc
                w2 = tw
        elif d < d1:
            d2 = d1
            c2 = c1
            w2 = w1
            d1 = d
            c1 = cn
            w1 = wn
        elif d < d2:
            d2 = d
            c2 = cn
            w2 = wn
    res = d1
    if c2 != -2:
        kk = fillet
        if kk <= 0.0:
            kk = factor * wp.min(w1, w2)
        if kk > 0.0:
            h = wp.max(kk - wp.abs(d1 - d2), 0.0) / kk
            res = wp.min(d1, d2) - h * h * kk * 0.25
    out[tid] = res


_KERNEL_DATA = OrderedDict()  # (key, device) -> (wp.Bvh, its arrays): built once per process
PIECE = 4_000_000             # grid points per CPU launch


def _device_data(data, key, device):
    entry = _KERNEL_DATA.get((key, device))
    if entry is None:
        vec = lambda x, t=wp.vec3: wp.array(np.ascontiguousarray(x, dtype=np.float32), dtype=t, device=device)
        ints = lambda x: wp.array(np.ascontiguousarray(x, dtype=np.int32).reshape(-1), dtype=int, device=device)
        flts = lambda x: wp.array(np.ascontiguousarray(x, dtype=np.float32), dtype=float, device=device)
        arrays = [ints(data["kind"]), ints(data["chain"]), flts(data["width"]),
                  vec(data["a"]), vec(data["b"]), vec(data["ey"]), vec(data["ez"]), vec(data["p0"]), vec(data["n0"]),
                  vec(data["p1"]), vec(data["n1"]), flts(data["rad"]), flts(data["wall"]),
                  vec(data["rects"].reshape(-1, 4), wp.vec4), ints(data["rsign"]),
                  vec(data["cen"], wp.vec2), vec(data["rot"], wp.vec2)]
        lo, hi = vec(data["lo"]), vec(data["hi"])
        entry = (wp.Bvh(lo, hi), arrays, (lo, hi))
        _KERNEL_DATA[(key, device)] = entry
        while len(_KERNEL_DATA) > 8:
            _KERNEL_DATA.popitem(last=False)
    else:
        _KERNEL_DATA.move_to_end((key, device))
    return entry


def evaluate(data, grid, key, gpu, fillet, factor):
    device = "cuda" if gpu else "cpu"
    bvh, arrays, _ = _device_data(data, key, device)
    common = [bvh.id, *arrays, float(fillet), float(factor), float(data["cap"]),
              wp.vec3(*[float(c) for c in grid.base]), *grid.start, float(grid.voxel), grid.shape[1], grid.shape[2]]
    if gpu:
        cp = backend.cupy()
        out = cp.empty(grid.size, dtype=cp.float32)
        cp.cuda.Device().synchronize()  # Warp runs on its own stream
        wp.launch(_frame_grid, dim=grid.size, inputs=[*common, 0, wp.from_dlpack(out)], device=device)
        wp.synchronize_device(device)
        return out.reshape(grid.shape)
    out = np.empty(grid.size, dtype=np.float32)
    target = wp.array(out, dtype=float, device="cpu", copy=False)
    threads = int(os.environ.get("IGL_NUM_THREADS") or os.cpu_count() or 4)
    step = max(1, min(PIECE, -(-grid.size // (4 * threads))))
    starts = range(0, grid.size, step)
    launch = lambda s: wp.launch(_frame_grid, dim=min(step, grid.size - s), inputs=[*common, s, target], device="cpu")
    if threads <= 1 or len(starts) == 1:
        for s in starts:
            launch(s)
    else:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(launch, starts))
    return out.reshape(grid.shape)


