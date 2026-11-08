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

SLOTS = 12  # the nearest different bars kept per point (more within two fillets' reach: a seam again)
_vecf = wp.types.vector(length=SLOTS, dtype=float)
_veci = wp.types.vector(length=SLOTS, dtype=int)


@wp.func
def _bulge(h: float, k: float, order: int) -> float:
    """What a smooth minimum of size k adds to min(a, b), h = 1 - |a - b| / k (0..1): G1 the quadratic
    one, G2 a cubic (curvature continuous), G3 a quintic, also smooth through the fillet's middle
    (`fields.bulge`)."""
    if order >= 3:
        return k * h * h * h * h * (5.0 - 2.0 * h) * 0.05
    if order == 2:
        return k * h * h * h / 6.0
    return h * h * k * 0.25


@wp.func
def _reach(w: float, fillet: float, factor: float) -> float:
    """A bar's fillet size (0: plain joints)."""
    if fillet > 0.0:
        return fillet
    return factor * w


@wp.func
def _far(d: float, w: float, fillet: float, factor: float) -> float:
    """How far a bar is in its fillet sizes (the one dropped when more bars are near than SLOTS)."""
    kk = _reach(w, fillet, factor)
    if kk > 0.0:
        return d / kk
    return d


@wp.func
def _fade(x: float) -> float:
    """1 up to x = 1, 0 from 2, C3 between: a bar more than one fillet size away blends ever less
    (`geometry.fade`)."""
    if x <= 1.0:
        return 1.0
    if x >= 2.0:
        return 0.0
    y = 2.0 - x
    return y * y * y * y * (35.0 + y * (-84.0 + y * (70.0 - 20.0 * y)))


@wp.func
def _ramp(s: float, size: float, order: int) -> float:
    """max(s, 0) made as smooth as the fillets over -size..size: a round bar's end is where
    sqrt(rho^2 + ramp(s)^2) = its radius, a half ball for G1 (meeting the bar with a curvature jump),
    a C2 / C3 rounding for G2 / G3 (the bar narrows by ~1% at its end node; `geometry.ramp`)."""
    if order <= 1:
        return wp.max(s, 0.0)
    if s >= size:
        return s
    if s <= -size:
        return 0.0
    x = (s + size) / (2.0 * size)
    if order == 2:
        return 2.0 * size * x * x * x * (1.0 - 0.5 * x)
    return 2.0 * size * x * x * x * x * (2.5 + x * (x - 3.0))


@wp.func
def _on_curve(c0: wp.vec3, c1: wp.vec3, c2: wp.vec3, c3: wp.vec3, c4: wp.vec3, c5: wp.vec3, t: float) -> wp.vec3:
    """A bar's curve c0 + c1 t + ... + c5 t^5, going on straight past t = 0 and 1 (`ends`)."""
    if t < 0.0:
        return c0 + t * c1
    if t > 1.0:
        return c0 + c1 + c2 + c3 + c4 + c5 + (t - 1.0) * (c1 + 2.0 * c2 + 3.0 * c3 + 4.0 * c4 + 5.0 * c5)
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))


@wp.func
def _closest(p: wp.vec3, c0: wp.vec3, c1: wp.vec3, c2: wp.vec3, c3: wp.vec3, c4: wp.vec3, c5: wp.vec3, lo: float,
             hi: float) -> float:
    """The parameter (lo..hi) of the curve's point nearest p: from the chord, four Newton steps (a fixed
    number, so a smooth function of p; the pieces are short and nearly straight)."""
    ch = c1 + c2 + c3 + c4 + c5
    t = wp.clamp(wp.dot(p - c0, ch) / wp.max(wp.dot(ch, ch), 1.0e-12), lo, hi)
    for it in range(4):
        d1 = c1 + t * (2.0 * c2 + t * (3.0 * c3 + t * (4.0 * c4 + t * 5.0 * c5)))
        d2 = 2.0 * c2 + t * (6.0 * c3 + t * (12.0 * c4 + t * 20.0 * c5))
        if t < 0.0:
            d1 = c1
            d2 = wp.vec3(0.0, 0.0, 0.0)
        if t > 1.0:
            d1 = c1 + 2.0 * c2 + 3.0 * c3 + 4.0 * c4 + 5.0 * c5
            d2 = wp.vec3(0.0, 0.0, 0.0)
        r = _on_curve(c0, c1, c2, c3, c4, c5, t) - p
        slope = wp.dot(d1, d1) + wp.dot(r, d2)
        if slope > 1.0e-12:
            t = wp.clamp(t - wp.dot(r, d1) / slope, lo, hi)
    return t


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
                cen: wp.array(dtype=wp.vec2), rot: wp.array(dtype=wp.vec2), curve: wp.array(dtype=wp.vec3),
                ends: wp.array(dtype=int), tips: wp.array(dtype=wp.vec3), near: wp.array(dtype=int),
                fillet: float, factor: float, order: int,
                cap: float, origin: wp.vec3, sx: int, sy: int, sz: int, voxel: float, ny: int, nz: int, first: int,
                out: wp.array(dtype=float)):
    tid = wp.tid() + first
    i = tid // (ny * nz)
    j = (tid // nz) % ny
    k = tid % nz
    p = origin + wp.vec3(float(i + sx), float(j + sy), float(k + sz)) * voxel
    nd = _vecf()   # the nearest different bars: each at its nearest piece (round ones: rho - radius)
    nc = _veci()
    nw = _vecf()
    nr = _vecf()   # round bars and balls: the distance to the curve (centre) there, the end rounding's
    nq = _vecf()   # square, the radius (a bar's ends are rounded from its curve's nearest point, not per
    na = _vecf()   # piece: a piece farther from the end, less rounded, then won and creased the surface)
    m = int(0)
    query = wp.bvh_query_aabb(bvh, p, p)
    n = int(0)
    while wp.bvh_query_next(query, n):
        kd = kind[n]
        d = float(0.0)
        rho_n = float(0.0)
        qq = float(0.0)
        rad_n = float(0.0)
        if kd == 0 or kd == 1:  # round bar or tube: around its bar's curve (`geometry._curves`)
            c0 = curve[n * 6]
            c1 = curve[n * 6 + 1]
            c2 = curve[n * 6 + 2]
            c3 = curve[n * 6 + 3]
            c4 = curve[n * 6 + 4]
            c5 = curve[n * 6 + 5]
            # past its ends the curve goes on straight as far as the end's rounding and any fillet on it
            # reach (farther, a curved bar's straight ray could pass by the bar itself, nearer than it)
            lo = float(0.0)
            hi = float(1.0)
            run = fillet
            if run <= 0.0:
                run = factor * width[n]
            run = rad[n] + 2.0 * run + 1.0
            if ends[n * 2] == 1:
                lo = -run / wp.max(wp.length(c1), 1.0e-9)
            if ends[n * 2 + 1] == 1:
                hi = 1.0 + run / wp.max(wp.length(c1 + 2.0 * c2 + 3.0 * c3 + 4.0 * c4 + 5.0 * c5), 1.0e-9)
            t = _closest(p, c0, c1, c2, c3, c4, c5, lo, hi)
            c = _on_curve(c0, c1, c2, c3, c4, c5, t)
            rho = wp.length(p - c)   # (which piece of the bar is nearest: by this)
            if kd == 0:   # (its ends rounded below, from the bar's nearest point)
                # on the straight run past an end, and beyond it: from the line through the end (the run's
                # end point gave too small values farther out)
                if t > 1.0:
                    e = tips[n * 4 + 2]
                    c = e + wp.dot(p - e, tips[n * 4 + 3]) * tips[n * 4 + 3]
                    rho = wp.length(p - c)
                if t < 0.0:
                    e = tips[n * 4]
                    c = e + wp.dot(p - e, tips[n * 4 + 1]) * tips[n * 4 + 1]
                    rho = wp.length(p - c)
                if near[n * 2] == 1:
                    q = _ramp(wp.dot(c - tips[n * 4], tips[n * 4 + 1]), rad[n], order)
                    qq += q * q
                if near[n * 2 + 1] == 1:
                    q = _ramp(wp.dot(c - tips[n * 4 + 2], tips[n * 4 + 3]), rad[n], order)
                    qq += q * q
                rho_n = rho
                rad_n = rad[n]
                d = rho - rad[n]
            else:
                flat = wp.abs(rho - (rad[n] - wall[n] * 0.5)) - wall[n] * 0.5
                along = wp.max(wp.dot(p - p0[n], n0[n]), wp.dot(p - p1[n], n1[n]))
                if flat > 0.0 and along > 0.0:
                    d = wp.sqrt(flat * flat + along * along)
                else:
                    d = wp.max(flat, along)
        elif kd == 3:  # ball
            rho_n = wp.length(p - a[n])
            rad_n = rad[n]
            d = rho_n - rad[n]
        else:
            rel = p - a[n]
            y = wp.dot(rel, ey[n])
            z = wp.dot(rel, ez[n])
            flat = float(0.0)
            if kd == 2:  # rectangles: union of the solid ones, minus the holes
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
        found = int(-1)
        far = int(0)
        for u in range(m):   # (a loop to m: Warp unrolls constant ones, and compiles longer)
            if nc[u] == cn:
                found = u
            if _far(nd[u], nw[u], fillet, factor) > _far(nd[far], nw[far], fillet, factor):
                far = u
        slot = int(-1)
        if found >= 0:
            if d < nd[found]:
                slot = found
        elif m < SLOTS:
            slot = m
            m += 1
        elif _far(d, width[n], fillet, factor) < _far(nd[far], nw[far], fillet, factor):
            # (a bar dropped here is farther than all kept: its later pieces too)
            slot = far
        if slot >= 0:
            nd[slot] = d
            nc[slot] = cn
            nw[slot] = width[n]
            nr[slot] = rho_n
            nq[slot] = qq
            na[slot] = rad_n
    for u in range(m):   # round bars: rounded at their ends
        if na[u] > 0.0:
            nd[u] = wp.sqrt(nr[u] * nr[u] + nq[u]) - na[u]
    # thickest first (then by bar number), each bar blended into those before it with a fillet sized by
    # itself, the thinner of each pair: one blend as smooth as its order however many bars meet. (The
    # union of every pair's fillet creased where two fillets crossed; blending only the two nearest tore a
    # seam where the second and third swapped.)
    for u in range(m):
        for v in range(m - 1 - u):
            if nw[v + 1] > nw[v] or (nw[v + 1] == nw[v] and nc[v + 1] < nc[v]):
                td = nd[v]
                nd[v] = nd[v + 1]
                nd[v + 1] = td
                tc = nc[v]
                nc[v] = nc[v + 1]
                nc[v + 1] = tc
                tw = nw[v]
                nw[v] = nw[v + 1]
                nw[v + 1] = tw
    res = cap   # (also with the first bar: a point entering its box, where it is farther than cap, so no jump)
    if m > 0:
        res = wp.min(cap, nd[0])
    # (a bar more than a fillet size away fades out: blended with what is before it, far bars changed the
    # values later bars blend with, so every bar in the boxes mattered and one dropping out made a crease)
    for u in range(1, m):
        kk = _reach(nw[u], fillet, factor)
        if kk > 0.0:
            kk = kk * _fade(nd[u] / kk)
        if kk > 0.0:
            h = wp.max(kk - wp.abs(res - nd[u]), 0.0) / kk
            res = wp.min(res, nd[u]) - _bulge(h, kk, order)
        else:
            res = wp.min(res, nd[u])
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
                  vec(data["cen"], wp.vec2), vec(data["rot"], wp.vec2), vec(data["curve"].reshape(-1, 3)),
                  ints(data["ends"]), vec(data["tips"].reshape(-1, 3)), ints(data["near"])]
        lo, hi = vec(data["lo"]), vec(data["hi"])
        entry = (wp.Bvh(lo, hi), arrays, (lo, hi))
        _KERNEL_DATA[(key, device)] = entry
        while len(_KERNEL_DATA) > 8:
            _KERNEL_DATA.popitem(last=False)
    else:
        _KERNEL_DATA.move_to_end((key, device))
    return entry


def evaluate(data, grid, key, gpu, fillet, factor, order=1):
    device = "cuda" if gpu else "cpu"
    bvh, arrays, _ = _device_data(data, key, device)
    common = [bvh.id, *arrays, float(fillet), float(factor), int(order), float(data["cap"]),
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


