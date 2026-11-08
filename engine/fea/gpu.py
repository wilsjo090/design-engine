"""The FEA solver's and the topology update's work on the GPU as fused Warp kernels (used by
engine/fea/solver.py and topology.Optimizer for CuPy arrays; solver.GPU_KERNELS switches them off).

Written with CuPy, a stiffness product was ~50 small kernels (8 gathers, a matmul, 8 scatters, the
mask), and launching one from Python costs ~20 us whatever its size: on draft grids that was nearly
all of it (bolted wall bracket, draft: 1.2 ms a product, ~190 products an iteration). Here one kernel
does a product with the smoother's or the residual's vector updates fused in, one a restriction and
one a prolongation (with the masks and the correction added in), and the optimality criteria's root
finding takes two kernels per value tried (`Volume`).

A thread per node sums its rows of the 8 elements around it, so every node is written by one thread
in a fixed order: no atomics, the same result on every run (sums go per block, in a fixed order, too).
Products are in the arrays' precision (float32 multigrid, float64 for the exact fine level: see
solver.MIXED). Launches are recorded once (`wp.Launch`; only parameters that changed are set again,
~1.5 us each, a launch ~13 us instead of ~50 with array wrappers) and go to CuPy's current stream, so
CuPy and Warp work stay in order without synchronizing. Loops over constants are unrolled by Warp:
the loops over an element's neighbours and children run to a count passed in (unrolled, the product
kernels took ~60 s each to compile, now ~1 s).
"""
import math
import weakref

import numpy as np
import warp as wp

from .. import backend

wp.config.quiet = True
wp.init()

KINDS = {"fine": 0, "children": 1, "stored": 2}
BLOCK = 256   # threads per block of the volume kernel (one partial sum per block)
_KERNELS = {}
_STREAMS = {}


def _make(T):
    """The kernels for arrays of scalar type T (wp.float32 or wp.float64). Arrays come as pointers
    (uint64) with their sizes and are viewed as flat arrays inside the kernels."""
    UE = wp.types.vector(length=24, dtype=T)
    V3 = wp.types.vector(length=3, dtype=T)

    @wp.func
    def rows(u: wp.array(dtype=T), per: wp.array(dtype=T), const: wp.array(dtype=T),
             rowmap: wp.array(dtype=wp.int32), sparse: int, kind: int, eight: int, kids: int, i: int, j: int, k: int,
             nx: int, ny: int, nz: int):
        """(K u) at node (i, j, k): its three rows of the matrices of the (up to 8) elements around it,
        summed. K_e = per[e] KE (fine), sum_c per[e, c] M_c (children) or per[e] (stored); `rowmap`
        gives an element's row of `per` on a level stored element by element (-1: no material).
        `eight` is 8 and `kids` the children per element (8, or 4 or 2 where the level above halved only
        two axes or one: a sheet, a bar), passed in: with a constant Warp unrolls the loops over the
        elements and the children, and the kernels took ~60 s each to compile."""
        Y = ny + 1
        Z = nz + 1
        s0 = T(0.0)
        s1 = T(0.0)
        s2 = T(0.0)
        for n in range(eight):
            a = n // 4   # this node is corner (a, b, c) of element (i - a, j - b, k - c)
            b = (n // 2) % 2
            c = n % 2
            ei = i - a
            ej = j - b
            ek = k - c
            row = int(-1)
            if ei >= 0 and ei < nx and ej >= 0 and ej < ny and ek >= 0 and ek < nz:
                row = (ei * ny + ej) * nz + ek
                if sparse != 0:
                    row = int(rowmap[row])
            if row >= 0:
                ue = UE()
                for m in range(8):   # the element's 24 values, corners in element.CORNERS order
                    nm = 3 * (((ei + m // 4) * Y + ej + (m // 2) % 2) * Z + ek + m % 2)
                    ue[3 * m] = u[nm]
                    ue[3 * m + 1] = u[nm + 1]
                    ue[3 * m + 2] = u[nm + 2]
                r0 = 3 * n
                t0 = T(0.0)
                t1 = T(0.0)
                t2 = T(0.0)
                if kind == 2:
                    base = row * 576 + r0 * 24
                    for l in range(24):
                        t0 += per[base + l] * ue[l]
                        t1 += per[base + 24 + l] * ue[l]
                        t2 += per[base + 48 + l] * ue[l]
                elif kind == 1:
                    for ch in range(kids):
                        q0 = T(0.0)
                        q1 = T(0.0)
                        q2 = T(0.0)
                        for l in range(24):
                            kk = (ch * 24 + l) * 24 + r0
                            q0 += ue[l] * const[kk]
                            q1 += ue[l] * const[kk + 1]
                            q2 += ue[l] * const[kk + 2]
                        w = per[row * kids + ch]
                        t0 += q0 * w
                        t1 += q1 * w
                        t2 += q2 * w
                else:
                    for l in range(24):
                        kk = l * 24 + r0
                        t0 += ue[l] * const[kk]
                        t1 += ue[l] * const[kk + 1]
                        t2 += ue[l] * const[kk + 2]
                    w = per[row]
                    t0 = t0 * w
                    t1 = t1 * w
                    t2 = t2 * w
                s0 += t0
                s1 += t1
                s2 += t2
        return V3(s0, s1, s2)

    @wp.kernel(module="unique")
    def product(u_p: wp.uint64, r_p: wp.uint64, res_p: wp.uint64, d_p: wp.uint64, x_p: wp.uint64,
                free_p: wp.uint64, inv_p: wp.uint64, c1: T, c2: T, mode: int, flag: int,
                per_p: wp.uint64, const_p: wp.uint64, rowmap_p: wp.uint64, n_per: int, n_const: int, n_rows: int,
                sparse: int, kind: int, eight: int, kids: int, nx: int, ny: int, nz: int):
        """A product K u with what is done with it (one kernel, so it compiles once):
        mode 0, apply: res = K u (flag: times free);
        mode 1, residual: res = r - K u on the free DOFs (flag: also d = D^-1 res / c1, a smoothing
        step's start);
        mode 2, Chebyshev step with u = d_in: res -= K u on the free DOFs, d = c1 u + c2 D^-1 res,
        x += d (flag: x += u first, the update the step before left to this one).
        Every thread reads u around its node and writes only its own node: d must not be u. `free` is
        float32 (0 or 1) in both precisions: the exact float64 fine level uses the float32 level's
        mask as it is (a float64 copy was ~130 MB on a 5.3M-element fine analysis)."""
        t = wp.tid()
        n3 = 3 * (nx + 1) * (ny + 1) * (nz + 1)
        u = wp.array(ptr=u_p, shape=(n3,), dtype=T)
        r = wp.array(ptr=r_p, shape=(n3,), dtype=T)
        res = wp.array(ptr=res_p, shape=(n3,), dtype=T)
        dd = wp.array(ptr=d_p, shape=(n3,), dtype=T)
        x = wp.array(ptr=x_p, shape=(n3,), dtype=T)
        free = wp.array(ptr=free_p, shape=(n3,), dtype=wp.float32)
        inv = wp.array(ptr=inv_p, shape=(n3,), dtype=T)
        per = wp.array(ptr=per_p, shape=(n_per,), dtype=T)
        const = wp.array(ptr=const_p, shape=(n_const,), dtype=T)
        rowmap = wp.array(ptr=rowmap_p, shape=(n_rows,), dtype=wp.int32)
        Z = nz + 1
        YZ = (ny + 1) * Z
        v = rows(u, per, const, rowmap, sparse, kind, eight, kids, t // YZ, (t // Z) % (ny + 1), t % Z, nx, ny, nz)
        for d in range(3):
            q = 3 * t + d
            if mode == 0:
                y = v[d]
                if flag != 0:
                    y = y * T(free[q])
                res[q] = y
            elif mode == 1:
                rr = r[q] - v[d] * T(free[q])
                res[q] = rr
                if flag != 0:
                    dd[q] = inv[q] * rr / c1
            else:
                rr = res[q] - v[d] * T(free[q])
                res[q] = rr
                dn = u[q] * c1 + inv[q] * rr * c2
                dd[q] = dn
                xv = x[q]
                if flag != 0:
                    xv = xv + u[q]
                x[q] = xv + dn

    @wp.kernel(module="unique")
    def start(r_p: wp.uint64, res_p: wp.uint64, d_p: wp.uint64, x_p: wp.uint64, inv_p: wp.uint64, theta: T,
              n3: int):
        """A smoothing step from x = 0: res = r, d = x = D^-1 r / theta."""
        q = wp.tid()
        r = wp.array(ptr=r_p, shape=(n3,), dtype=T)
        res = wp.array(ptr=res_p, shape=(n3,), dtype=T)
        dd = wp.array(ptr=d_p, shape=(n3,), dtype=T)
        x = wp.array(ptr=x_p, shape=(n3,), dtype=T)
        inv = wp.array(ptr=inv_p, shape=(n3,), dtype=T)
        rr = r[q]
        res[q] = rr
        dv = inv[q] * rr / theta
        dd[q] = dv
        x[q] = dv

    @wp.kernel(module="unique")
    def restrict(fine_p: wp.uint64, coarse_p: wp.uint64, free_p: wp.uint64, X: int, Y: int, Z: int,
                 Xc: int, Yc: int, Zc: int, fx: int, fy: int, fz: int):
        """coarse = (P^T fine) times the coarse level's free: each coarse node sums the fine nodes it
        interpolates to (weight 1 at its own, 1/2 at the neighbours halfway along a halved axis)."""
        t = wp.tid()
        fine = wp.array(ptr=fine_p, shape=(3 * X * Y * Z,), dtype=T)
        coarse = wp.array(ptr=coarse_p, shape=(3 * Xc * Yc * Zc,), dtype=T)
        free = wp.array(ptr=free_p, shape=(3 * Xc * Yc * Zc,), dtype=T)
        I = t // (Yc * Zc)
        J = (t // Zc) % Yc
        K = t % Zc
        s0 = T(0.0)
        s1 = T(0.0)
        s2 = T(0.0)
        for oi in range(-1, 2):
            for oj in range(-1, 2):
                for ok in range(-1, 2):
                    i = fx * I + oi
                    j = fy * J + oj
                    k = fz * K + ok
                    if (fx == 2 or oi == 0) and (fy == 2 or oj == 0) and (fz == 2 or ok == 0):
                        if i >= 0 and i < X and j >= 0 and j < Y and k >= 0 and k < Z:
                            w = T(1.0)
                            if oi != 0:
                                w = w * T(0.5)
                            if oj != 0:
                                w = w * T(0.5)
                            if ok != 0:
                                w = w * T(0.5)
                            n = 3 * ((i * Y + j) * Z + k)
                            s0 += w * fine[n]
                            s1 += w * fine[n + 1]
                            s2 += w * fine[n + 2]
        coarse[3 * t] = s0 * free[3 * t]
        coarse[3 * t + 1] = s1 * free[3 * t + 1]
        coarse[3 * t + 2] = s2 * free[3 * t + 2]

    @wp.kernel(module="unique")
    def prolong_add(coarse_p: wp.uint64, x_p: wp.uint64, free_p: wp.uint64, X: int, Y: int, Z: int,
                    Xc: int, Yc: int, Zc: int, fx: int, fy: int, fz: int):
        """x += free * (P coarse): trilinear interpolation of the coarse correction, added in."""
        t = wp.tid()
        coarse = wp.array(ptr=coarse_p, shape=(3 * Xc * Yc * Zc,), dtype=T)
        x = wp.array(ptr=x_p, shape=(3 * X * Y * Z,), dtype=T)
        free = wp.array(ptr=free_p, shape=(3 * X * Y * Z,), dtype=T)
        i = t // (Y * Z)
        j = (t // Z) % Y
        k = t % Z
        s0 = T(0.0)
        s1 = T(0.0)
        s2 = T(0.0)
        for ci in range(2):
            for cj in range(2):
                for ck in range(2):
                    # along a halved axis an even node is a coarse node, an odd one halfway between two
                    I = i
                    wi = T(1.0)
                    use = 1
                    if fx == 2:
                        I = (i - i % 2) // 2 + ci
                        if i % 2 == 1:
                            wi = T(0.5)
                        elif ci == 1:
                            use = 0
                    elif ci == 1:
                        use = 0
                    J = j
                    wj = T(1.0)
                    if fy == 2:
                        J = (j - j % 2) // 2 + cj
                        if j % 2 == 1:
                            wj = T(0.5)
                        elif cj == 1:
                            use = 0
                    elif cj == 1:
                        use = 0
                    Kc = k
                    wk = T(1.0)
                    if fz == 2:
                        Kc = (k - k % 2) // 2 + ck
                        if k % 2 == 1:
                            wk = T(0.5)
                        elif ck == 1:
                            use = 0
                    elif ck == 1:
                        use = 0
                    if use != 0:
                        n = 3 * ((I * Yc + J) * Zc + Kc)
                        w = wi * wj * wk
                        s0 += w * coarse[n]
                        s1 += w * coarse[n + 1]
                        s2 += w * coarse[n + 2]
        q = 3 * t
        x[q] = x[q] + free[q] * s0
        x[q + 1] = x[q + 1] + free[q + 1] * s1
        x[q + 2] = x[q + 2] + free[q + 2] * s2

    @wp.kernel(module="unique")
    def oc_design(x_p: wp.uint64, ndc_p: wp.uint64, ndv_p: wp.uint64, ndg_p: wp.uint64, lo_p: wp.uint64,
                  hi_p: wp.uint64, free_p: wp.uint64, domain_p: wp.uint64, xn_p: wp.uint64, xd_p: wp.uint64,
                  l1: T, l2: T, n: int):
        """The optimality criteria's design for multipliers l1, l2: xn = x sqrt(ndc / (l1 ndv + l2 ndg))
        within [lo, hi] where free (else x), and xd = xn times the domain (what the filter takes)."""
        e = wp.tid()
        x = wp.array(ptr=x_p, shape=(n,), dtype=T)
        ndc = wp.array(ptr=ndc_p, shape=(n,), dtype=T)
        ndv = wp.array(ptr=ndv_p, shape=(n,), dtype=T)
        ndg = wp.array(ptr=ndg_p, shape=(n,), dtype=T)
        lo = wp.array(ptr=lo_p, shape=(n,), dtype=T)
        hi = wp.array(ptr=hi_p, shape=(n,), dtype=T)
        free = wp.array(ptr=free_p, shape=(n,), dtype=T)
        domain = wp.array(ptr=domain_p, shape=(n,), dtype=T)
        xn = wp.array(ptr=xn_p, shape=(n,), dtype=T)
        xd = wp.array(ptr=xd_p, shape=(n,), dtype=T)
        y = x[e]
        if free[e] != T(0.0):
            y = wp.min(wp.max(y * wp.sqrt(ndc[e] / wp.max(l1 * ndv[e] + l2 * ndg[e], T(1e-30))), lo[e]), hi[e])
        xn[e] = y
        xd[e] = y * domain[e]

    @wp.kernel(module="unique")
    def oc_volume(xd_p: wp.uint64, conv_p: wp.uint64, offsets_p: wp.uint64, weights_p: wp.uint64,
                  norm_p: wp.uint64, phi_p: wp.uint64, solid_p: wp.uint64, domain_p: wp.uint64, partial_p: wp.uint64,
                  taps: int, direct: int, beta: T, eta: T, a: T, ab: T, nx: int, ny: int, nz: int, blocks: int):
        """Sums per block of BLOCK elements (in float64, added up on the host: the same total on every
        run) of phi times the physical densities (Optimizer.physical: the filter, the projection, solid
        regions full) of the design xd (times the domain). The filter's convolution is done here when
        `direct` (its nonzero taps: offsets (i, j, k) and weights, zero outside the grid), else taken from
        `conv` (by FFT); elements outside the design space and solid ones do not need it."""
        b, lane = wp.tid()
        n = nx * ny * nz
        xd = wp.array(ptr=xd_p, shape=(n,), dtype=T)
        conv = wp.array(ptr=conv_p, shape=(n,), dtype=T)
        offsets = wp.array(ptr=offsets_p, shape=(3 * taps,), dtype=wp.int32)
        weights = wp.array(ptr=weights_p, shape=(taps,), dtype=T)
        norm = wp.array(ptr=norm_p, shape=(n,), dtype=T)
        phi = wp.array(ptr=phi_p, shape=(n,), dtype=T)
        solid = wp.array(ptr=solid_p, shape=(n,), dtype=T)
        domain = wp.array(ptr=domain_p, shape=(n,), dtype=T)
        partial = wp.array(ptr=partial_p, shape=(blocks,), dtype=wp.float64)
        e = b * BLOCK + lane
        v = wp.float64(0.0)
        if e < n:
            xb = T(1.0)
            if phi[e] * domain[e] == T(0.0):
                xb = T(0.0)
            elif solid[e] == T(0.0):
                c = T(0.0)
                if direct != 0:
                    i = e // (ny * nz)
                    j = (e // nz) % ny
                    k = e % nz
                    for m in range(taps):
                        ii = i + int(offsets[3 * m])
                        jj = j + int(offsets[3 * m + 1])
                        kk = k + int(offsets[3 * m + 2])
                        if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
                            c += weights[m] * xd[(ii * ny + jj) * nz + kk]
                else:
                    c = conv[e]
                xb = (a + wp.tanh(beta * (c * norm[e] - eta))) / ab
            v = wp.float64(phi[e] * (xb * domain[e]))
        wp.tile_store(partial, wp.tile_sum(wp.tile(v)), offset=b)

    return {"product": product, "start": start, "restrict": restrict, "prolong_add": prolong_add,
            "oc_design": oc_design, "oc_volume": oc_volume}


def _kernels(dtype):
    dtype = np.dtype(dtype)
    if dtype not in _KERNELS:
        _KERNELS[dtype] = _make(wp.float32 if dtype == np.float32 else wp.float64)
    return _KERNELS[dtype]


def _device_and_stream():
    """Warp's device and a Warp stream on CuPy's current stream (which may be the default one)."""
    cp = backend.cupy()
    key = (cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream().ptr)
    if key not in _STREAMS:
        device = wp.get_device(f"cuda:{key[0]}")
        _STREAMS[key] = device, wp.Stream(device, cuda_stream=key[1])
    return _STREAMS[key]


def _ptr(a):
    return int(a.data.ptr)


def _launch(cmds, kernel, dim, head, tail=(), block_dim=256):
    """Run `kernel` on CuPy's current stream with parameters head + tail: recorded the first time
    (kept in `cmds`), afterwards only the head's parameters that changed are set again (~1.5 us each,
    a launch ~13 us)."""
    device, stream = _device_and_stream()
    key = (kernel, dim, stream)
    kept = cmds.get(key)
    if kept is None:
        cmd = wp.launch(kernel, dim=dim, inputs=list(head) + list(tail), device=device, stream=stream,
                        block_dim=block_dim, record_cmd=True)
        kept = cmds[key] = (cmd, list(head))
    else:
        cmd, last = kept
        for i, value in enumerate(head):
            if last[i] != value:
                cmd.set_param_at_index(i, value)
                last[i] = value
    kept[0].launch(stream=stream)


class Volume:
    """The optimality criteria update's root finding (topology.Optimizer._update and _update_two):
    for multipliers (l1, l2) the design, and the share of the design space its physical densities
    keep. Two kernels and a sum per value tried instead of ~20 CuPy kernels and a filter (wall
    bracket, draft: ~60 values tried an iteration). fft: the filter's convolution for kernels too big
    to convolve directly (topology.FFT_TAPS), else the volume kernel convolves."""

    def __init__(self, problem, x, ndc, ndv, ndg, lo, hi, beta, eta, fft=None):
        xp = problem.xp
        self.p, self.xp, self.fft = problem, xp, fft
        self.dtype = np.dtype(problem.dtype)
        self.kernels = _kernels(self.dtype)
        self.shape = tuple(int(n) for n in problem.phi.shape)
        flat = lambda a: xp.ascontiguousarray(xp.asarray(a, dtype=self.dtype)).ravel()
        self.design_in = [flat(a) for a in (x, ndc, ndv, ndg, lo, hi, problem.free, problem.domain)]
        self.n = int(np.prod(self.shape))
        self.xn, self.xd = (xp.empty(self.n, dtype=self.dtype) for _ in range(2))
        kernel = np.asarray(backend.to_host(problem.kernel))
        taps = np.argwhere(kernel != 0)   # (the cone's corners are 0: about half the taps)
        self.offsets = xp.asarray((taps - (kernel.shape[0] - 1) // 2).astype(np.int32).ravel())
        self.weights = xp.asarray(kernel[kernel != 0], dtype=self.dtype)
        self.volume_in = [flat(a) for a in (problem.norm, problem.phi, problem.solid, problem.domain)]
        self.blocks = (self.n + BLOCK - 1) // BLOCK
        self.partial = xp.empty(self.blocks, dtype=np.float64)
        a, b = math.tanh(beta * eta), math.tanh(beta * (1 - eta))
        self.scalars = [len(taps), int(fft is None), float(beta), float(eta), a, a + b, *self.shape, self.blocks]
        self._cmds = {}

    def _design(self, l1, l2, xn):
        _launch(self._cmds, self.kernels["oc_design"], self.n,
                [*map(_ptr, self.design_in), _ptr(xn), _ptr(self.xd), float(l1), float(l2), self.n])

    def design(self, l1, l2):
        """The design for (l1, l2), a new array in the grid's shape."""
        xn = self.xp.empty(self.n, dtype=self.dtype)
        self._design(l1, l2, xn)
        return xn.reshape(self.shape)

    def volume(self, l1, l2):
        """The share of the design space that design keeps (Problem.volume of its physical densities)."""
        self._design(l1, l2, self.xn)
        conv = self.xd
        if self.fft is not None:
            conv = self.xp.ascontiguousarray(self.fft(self.xd.reshape(self.shape), self.p.kernel),
                                             dtype=self.dtype).ravel()
        _launch(self._cmds, self.kernels["oc_volume"], (self.blocks, BLOCK),
                [_ptr(self.xd), _ptr(conv), _ptr(self.offsets), _ptr(self.weights), *map(_ptr, self.volume_in),
                 _ptr(self.partial), *self.scalars], block_dim=BLOCK)
        return float(np.sum(self.partial.get())) / self.p.total


class LevelKernels:
    """A level's element data in the kernels' layout, and its recorded launches. It refers to its level
    weakly: the level keeps it (Level._gpu), and with a reference back every iteration's hierarchy
    waited for Python's cycle collector, holding GPU memory (wall bracket, medium: up to 47 levels,
    383 MB instead of 21; a fine result's Static Analysis check no longer fit the GPU)."""

    def __init__(self, level):
        xp = level.xp
        self._level = weakref.ref(level)
        self.dtype = np.dtype(level.per.dtype)
        self.kernels = _kernels(self.dtype)
        self.per = xp.ascontiguousarray(level.per)
        const = level.const if level.const is not None else xp.zeros(1, dtype=self.dtype)
        self.const = xp.ascontiguousarray(xp.asarray(const, dtype=self.dtype))
        nx, ny, nz = level.shape
        if level.cells is None:
            self.rowmap, sparse = xp.zeros(1, dtype=xp.int32), 0
        else:
            self.rowmap = xp.full(nx * ny * nz, -1, dtype=xp.int32)
            self.rowmap[level.cells] = xp.arange(level.count, dtype=xp.int32)
            sparse = 1
        kids = int(level.const.shape[0]) if level.kind == "children" else 8
        self.tail = [_ptr(self.per), _ptr(self.const), _ptr(self.rowmap), int(self.per.size), int(self.const.size),
                     int(self.rowmap.size), sparse, KINDS[level.kind], 8, kids, nx, ny, nz]
        self.nodes = (nx + 1) * (ny + 1) * (nz + 1)
        self._vectors = {}
        self._cmds = {}

    @property
    def level(self):
        return self._level()

    def vector(self, name, dtype=None):
        """The level's `free` or `inv_diag` in this precision, or `dtype` (no copy when it has it)."""
        a = getattr(self.level, name)
        dtype = np.dtype(dtype or self.dtype)
        kept = self._vectors.get((name, dtype))
        if kept is None or kept[0] is not a:
            xp = self.level.xp
            kept = self._vectors[(name, dtype)] = (a, xp.ascontiguousarray(xp.asarray(a, dtype=dtype)))
        return kept[1]

    def _launch(self, name, dim, head, tail=True):
        _launch(self._cmds, self.kernels[name], dim, head, self.tail if tail else [])

    def _check(self, *arrays):
        for a in arrays:
            if a.dtype != self.dtype or not a.flags.c_contiguous:
                raise ValueError("solver kernels need C-contiguous arrays of the level's type")

    def _product(self, mode, flag, u, res, r=None, d=None, x=None, c1=1.0, c2=0.0):
        """The `product` kernel (see there); arrays it does not use in `mode` are passed as u (apply
        never reads inv_diag: the exact float64 level would otherwise get a float64 copy of it)."""
        self._check(*(a for a in (u, res, r, d, x) if a is not None))
        inv = u if mode == 0 else self.vector("inv_diag")
        self._launch("product", self.nodes, [_ptr(u), _ptr(u if r is None else r), _ptr(res),
                                             _ptr(u if d is None else d), _ptr(u if x is None else x),
                                             _ptr(self.vector("free", np.float32)), _ptr(inv),
                                             float(c1), float(c2), mode, int(bool(flag))])

    def apply(self, u, out=None, masked=True):
        xp = self.level.xp
        u = xp.ascontiguousarray(u)
        y = xp.empty(u.shape, dtype=self.dtype) if out is None else out
        self._product(0, masked, u, y)
        return y

    def residual(self, x, r, d=None, theta=1.0):
        """r - K x on the free DOFs (and, given d, d = D^-1 (r - K x) / theta)."""
        res = self.level.xp.empty_like(r)
        self._product(1, d is not None, x, res, r=r, d=d, c1=theta)
        return res

    def smooth(self, r, x, coefficients):
        """Level.smooth with the same coefficients: (theta, [(ca, cb) per further step])."""
        xp = self.level.xp
        theta, steps = coefficients
        r = xp.ascontiguousarray(r)
        d, d2 = xp.empty_like(r), xp.empty_like(r)
        if x is None:
            x, res = xp.empty_like(r), xp.empty_like(r)
            self._check(r, x)
            self._launch("start", r.size, [_ptr(r), _ptr(res), _ptr(d), _ptr(x), _ptr(self.vector("inv_diag")),
                                           float(theta), int(r.size)], tail=False)
            pending = 0
        else:
            res = self.residual(x, r, d, theta)
            pending = 1   # x += d is left to the first step
        for ca, cb in steps:
            self._product(2, pending, d, res, d=d2, x=x, c1=ca, c2=cb)
            d, d2 = d2, d
            pending = 0
        if pending:
            x += d
        return x

    def restrict(self, fine, coarse_level):
        """P^T fine on the next coarser level's nodes, times its free."""
        xp = self.level.xp
        Xc, Yc, Zc = coarse_level.nodes
        out = xp.empty((Xc, Yc, Zc, 3), dtype=self.dtype)
        free = xp.ascontiguousarray(coarse_level.free, dtype=self.dtype)
        self._check(fine, out, free)
        self._launch("restrict", Xc * Yc * Zc, [_ptr(fine), _ptr(out), _ptr(free), *self.level.nodes, Xc, Yc, Zc,
                                                *self.level.factors], tail=False)
        return out

    def prolong_add(self, coarse, x):
        """x += free * P coarse."""
        xp = self.level.xp
        coarse = xp.ascontiguousarray(coarse, dtype=self.dtype)
        self._check(coarse, x)
        X, Y, Z = self.level.nodes
        Xc, Yc, Zc = coarse.shape[:3]
        self._launch("prolong_add", X * Y * Z, [_ptr(coarse), _ptr(x), _ptr(self.vector("free")), X, Y, Z,
                                                Xc, Yc, Zc, *self.level.factors], tail=False)
        return x
