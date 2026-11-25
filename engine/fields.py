"""Scalar fields and implicit-body operations.

A Field is a lazy expression that evaluates to a float32 array on a Grid. Implicit bodies are
just fields whose zero level is the surface (negative inside). Fields support normal Python
math, so a node graph reads as plain code:

    wave = 0.35 * acos(cos(2 * 2 * X / period))
    textured = offset(part, wave)

Every field carries a `key`: a fingerprint of the whole expression below it. Two fields built
the same way have the same key, which lets evaluated arrays be cached across graph runs.
Fields may also carry `bounds` (the box their surface lives in), used to size the mesh grid.
"""
import hashlib
import math

import numpy as np

from .backend import gaussian_filter as backend_gaussian, to_device, to_host
from .noise import PATTERNS, fractal_on_grid, wave3d_on_grid
from .sdf import mesh_to_sdf
from .shared import Shared


def _digest(*parts):
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:20]


def mesh_fingerprint(mesh):
    h = hashlib.sha1(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:20]


def _merge_bounds(*bounds, mode="union"):
    bs = [np.asarray(b) for b in bounds if b is not None]
    if not bs:
        return None
    if mode == "intersect":
        return np.array([np.max([b[0] for b in bs], axis=0), np.min([b[1] for b in bs], axis=0)])
    return np.array([np.min([b[0] for b in bs], axis=0), np.max([b[1] for b in bs], axis=0)])


def _grown(bounds, by):
    """`bounds` grown by `by` mm on every side (unchanged if `by` is unknown, not positive or not
    finite)."""
    if bounds is None or by is None or not math.isfinite(by) or by <= 0:
        return bounds
    return np.array([bounds[0] - by, bounds[1] + by])


def _core(field):
    """(the box before offsets grew it, how far they reach beyond it) of a field with bounds."""
    return field.grown if field.grown is not None else (field.bounds, 0.0)


def _merged_core(fields, mode="union"):
    """`grown` of a field whose bounds merge those of `fields` (union or intersect): the merged
    original boxes with the largest reach, which contains every grown box."""
    boxed = [f for f in fields if f.bounds is not None]
    if not any(f.grown is not None for f in boxed):
        return None
    cores = [_core(f) for f in boxed]
    return _merge_bounds(*[c for c, _ in cores], mode=mode), max(r for _, r in cores)


def mesh_box(field, padding, voxel=0.0):
    """The box and padding to mesh `field` in with `voxel` mm voxels. Offsets by a known amount grow
    a field's bounds; here they get enough room around the box they grew from instead (the grid adds
    two voxels to the padding; one is enough to close the surface), so the grid, and the distance
    fields cached on it, stay the same while Padding covers the offset, e.g. while you try out a
    texture's amplitude."""
    if field.grown is None:
        return field.bounds, padding
    core, reach = field.grown
    return core, max(padding, reach - voxel)


def _offset_reach(sym, body, value):
    """How far `body sym value` can reach out of the body's box, when `value` is a plain amount
    (no box of its own): body - d grows by up to max(d), body + d by up to -min(d). An offset by
    a texture of known range then keeps all of its surface inside the box."""
    if body.bounds is None or value.bounds is not None or value.range is None:
        return None
    return value.range[1] if sym == "-" else -value.range[0]


class Context:
    """Evaluates fields on one grid, keeping memory low.

    Before evaluating, it counts how often each sub-expression will be asked for; an
    intermediate array is kept only until its last use, then released. (Holding every step of
    a 4-million-voxel petal costs ~20 x 16 MB; most steps are cheap to recompute anyway.)
    Only fields marked `cache=True` (expensive ones: CAD distance fields, noise) go into the
    shared ArrayCache, which survives between runs; pass `cache=None` to skip it.

    `xp` is the array module: numpy (CPU) or cupy (GPU). The cache always holds numpy arrays.
    """

    def __init__(self, grid, cache=None, xp=np):
        self.grid = grid
        self.cache = cache
        self.xp = xp
        self._memo = {}   # key -> array still needed later in this evaluation
        self._uses = {}   # key -> remaining requests
        self._depth = 0

    def axis(self, i):
        """Coordinate along axis i as a broadcastable array (full size only once combined)."""
        shape = [1, 1, 1]
        shape[i] = self.grid.shape[i]
        return self.xp.asarray(self.grid.axes()[i].astype(np.float32).reshape(shape))

    def _plan(self, root):
        """Count requests per unique sub-expression (each edge from a parent is one request)."""
        self._uses[root.key] = self._uses.get(root.key, 0) + 1
        seen, stack = set(), [root]
        while stack:
            f = stack.pop()
            if f.key in seen:
                continue
            seen.add(f.key)
            for child in f.children:
                self._uses[child.key] = self._uses.get(child.key, 0) + 1
                stack.append(child)

    def eval(self, field):
        if self._depth == 0:
            self._plan(field)
        key = field.key
        value = self._memo.get(key)
        if value is None:
            store = self.cache if field.cache else None
            if store is not None:
                value = store.get((key, self.grid.key))
                if value is not None:
                    value = to_device(value, self.xp)
            if value is None:
                self._depth += 1
                try:
                    value = field.fn(self)
                finally:
                    self._depth -= 1
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False  # may be shared: never modify in place
                if store is not None:
                    store.put((key, self.grid.key), to_host(value))
        left = self._uses.get(key, 1) - 1
        if left > 0:
            self._memo[key] = value
            self._uses[key] = left
        else:
            self._memo.pop(key, None)
            self._uses.pop(key, None)
        return value

    def eval_full(self, field):
        """Like eval, but always returns a full grid-shaped float32 array (on xp's device)."""
        xp = self.xp
        v = xp.asarray(self.eval(field))
        return xp.ascontiguousarray(xp.broadcast_to(v, self.grid.shape), dtype=xp.float32)


def _lift(value):
    if isinstance(value, Field):
        return value
    v = np.float32(value)
    return Field(lambda ctx: v, _digest("const", float(v)), name=f"{float(v):g}", range=(float(v), float(v)))


# --- value ranges ----------------------------------------------------------------------------
# Every field knows the range its values can take, when that follows from the formula alone
# (noise -1..1, a mask 0..1, sums and products of those...): shown on node cards, and used by
# Remap's Auto. None = depends on the part (distances, coordinates).
def _interval(sym, a, b):
    if a is None or b is None:
        return None
    (a0, a1), (b0, b1) = a, b
    if sym == "+":
        return (a0 + b0, a1 + b1)
    if sym == "-":
        return (a0 - b1, a1 - b0)
    if sym in ("*", "/"):
        if sym == "/":
            if b0 <= 0 <= b1:
                return None
            b0, b1 = 1 / b1, 1 / b0
        p = (a0 * b0, a0 * b1, a1 * b0, a1 * b1)
        return (min(p), max(p))
    if sym == "min":
        return (min(a0, b0), min(a1, b1))
    if sym == "max":
        return (max(a0, b0), max(a1, b1))
    return None


def _unary_range(label, r):
    if label in ("sin", "cos"):
        return (-1.0, 1.0)
    if label in ("acos", "asin"):
        # the input is clipped to -1..1, also when its whole range lies outside
        lo, hi = (-1.0, 1.0) if r is None else (min(max(r[0], -1.0), 1.0), max(min(r[1], 1.0), -1.0))
        return (math.acos(hi), math.acos(lo)) if label == "acos" else (math.asin(lo), math.asin(hi))
    if r is None:
        return None
    if label == "abs":
        return r if r[0] >= 0 else (-r[1], -r[0]) if r[1] <= 0 else (0.0, max(-r[0], r[1]))
    if label == "sqrt":
        return (math.sqrt(max(r[0], 0.0)), math.sqrt(max(r[1], 0.0)))
    return None


class Field:
    """`children`: the fields `fn` evaluates, once each per listed entry (lets a Context free
    intermediate arrays early). `cache`: keep the result in the shared ArrayCache; for
    expensive fields only."""

    def __init__(self, fn, key, name="field", bounds=None, children=(), cache=False, range=None):
        self.fn = fn
        self.key = key
        self.name = name
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        self.children = tuple(children)
        self.cache = cache
        self.range = None if range is None else (float(range[0]), float(range[1]))  # see _interval
        # (box, reach): set when offsets grew `bounds` beyond the box they started from (see mesh_box)
        self.grown = None
        # where the field belongs, for showing a list of per-body fields (slices, colours) when it has
        # no bounds of its own (a face mask is defined everywhere but belongs to one body)
        self.extent = None

    def _binary(self, other, op, sym, swap=False, bounds="union"):
        a, b = (_lift(other), self) if swap else (self, _lift(other))
        if bounds == "first":
            bb, grown = a.bounds, a.grown
        else:
            bb, grown = _merge_bounds(a.bounds, b.bounds, mode=bounds), _merged_core((a, b), bounds)
        reach = None
        if sym == "-":
            reach = _offset_reach(sym, a, b)
        elif sym == "+":
            reach = _offset_reach(sym, a, b) if a.bounds is not None else _offset_reach(sym, b, a)
        if _grown(bb, reach) is not bb:
            core, before = grown if grown is not None else (bb, 0.0)
            bb, grown = _grown(bb, reach), (core, before + reach)
        f = Field(lambda ctx: op(ctx.eval(a), ctx.eval(b)), _digest(sym, a.key, b.key),
                  f"({a.name} {sym} {b.name})", bb, children=(a, b), range=_interval(sym, a.range, b.range))
        f.grown = grown
        return f

    def __add__(self, o): return self._binary(o, np.add, "+")
    def __radd__(self, o): return self._binary(o, np.add, "+", swap=True)
    # body - distance keeps the body's bounds (that is what offset does), grown by the distance
    def __sub__(self, o): return self._binary(o, np.subtract, "-", bounds="first")
    def __rsub__(self, o): return self._binary(o, np.subtract, "-", swap=True, bounds="first")
    def __mul__(self, o): return self._binary(o, np.multiply, "*")
    def __rmul__(self, o): return self._binary(o, np.multiply, "*", swap=True)
    def __truediv__(self, o): return self._binary(o, np.divide, "/")
    def __rtruediv__(self, o): return self._binary(o, np.divide, "/", swap=True)
    def __neg__(self):
        r = None if self.range is None else (-self.range[1], -self.range[0])
        f = Field(lambda ctx: -ctx.eval(self), _digest("neg", self.key), f"-{self.name}", self.bounds,
                  children=(self,), range=r)
        f.grown = self.grown
        return f

    def __repr__(self):
        return f"Field<{self.name}>"


def _unary(np_fn, label):
    def apply(f):
        f = _lift(f)
        out = Field(lambda ctx: np_fn(ctx.eval(f)), _digest(label, f.key), f"{label}({f.name})", f.bounds,
                    children=(f,), range=_unary_range(label, f.range))
        out.grown = f.grown
        return out
    apply.__name__ = label
    return apply


sin, cos, tan = _unary(np.sin, "sin"), _unary(np.cos, "cos"), _unary(np.tan, "tan")
abs_ = _unary(np.abs, "abs")
acos = _unary(lambda v: np.arccos(np.clip(v, -1, 1)), "acos")
asin = _unary(lambda v: np.arcsin(np.clip(v, -1, 1)), "asin")
sqrt = _unary(lambda v: np.sqrt(np.maximum(v, 0)), "sqrt")


def minimum(a, b): return _lift(a)._binary(b, np.minimum, "min")
def maximum(a, b): return _lift(a)._binary(b, np.maximum, "max")
def clamp(f, lo, hi): return minimum(maximum(f, lo), hi)


def smoothstep(f, edge0, edge1):
    """0 below edge0, 1 above edge1, smooth S-curve in between. Handy for fading effects in/out.
    Where the edges are equal it is a sharp step: 1 at and above them."""
    if isinstance(edge0, Field) or isinstance(edge1, Field):
        x, e0, e1 = _lift(f), _lift(edge0), _lift(edge1)

        def fn(ctx):
            xp, v, a, b = ctx.xp, ctx.eval(x), ctx.eval(e0), ctx.eval(e1)
            span = b - a
            same = span == 0
            t = xp.where(same, (v >= a).astype(xp.float32), (v - a) / xp.where(same, 1.0, span))
            t = xp.clip(t, 0, 1)
            return (t * t * (3 - 2 * t)).astype(xp.float32)
        return Field(fn, _digest("smoothstep", x.key, e0.key, e1.key), f"smoothstep({x.name})", x.bounds,
                     children=(x, e0, e1), range=(0.0, 1.0))
    if float(edge1) == float(edge0):
        x, edge = _lift(f), np.float32(edge0)
        return Field(lambda ctx: (ctx.eval(x) >= edge).astype(ctx.xp.float32), _digest("step", x.key, float(edge)),
                     f"step({x.name})", x.bounds, children=(x,), range=(0.0, 1.0))
    t = clamp((_lift(f) - edge0) / (edge1 - edge0), 0, 1)
    out = t * t * (3 - 2 * t)
    out.range = (0.0, 1.0)  # interval arithmetic would say 0..3 (t appears three times)
    return out


# --- coordinate fields -------------------------------------------------------------------
X = Field(lambda ctx: ctx.axis(0), "X", "X")
Y = Field(lambda ctx: ctx.axis(1), "Y", "Y")
Z = Field(lambda ctx: ctx.axis(2), "Z", "Z")
AXES = {"x": X, "y": Y, "z": Z}


def radial(axis="z"):
    """Distance from the given coordinate axis."""
    u, v = [AXES[c] for c in "xyz" if c != axis]
    return sqrt(u * u + v * v)


def angle(axis="z"):
    """Angle around the given coordinate axis, radians (-pi..pi)."""
    u, v = [AXES[c] for c in "xyz" if c != axis]
    return Field(lambda ctx: np.arctan2(ctx.eval(v), ctx.eval(u)), _digest("angle", axis), f"angle_{axis}",
                 children=(v, u), range=(-math.pi, math.pi))


Radius = radial("z")
Angle = angle("z")


def sample(field, point):
    """Value of a field at one point (evaluated on a 1-voxel grid there)."""
    if not isinstance(field, Field):
        return float(field)
    from .grid import Grid
    p = np.asarray(point, dtype=np.float64)
    return float(Context(Grid(p, p, 1.0)).eval_full(field).ravel()[0])


def remap(f, in_lo, in_hi, out_lo, out_hi):
    """Linearly map in_lo..in_hi to out_lo..out_hi (not clamped)."""
    return (_lift(f) - in_lo) * ((out_hi - out_lo) / (in_hi - in_lo)) + out_lo


# --- noise and 3D patterns ---------------------------------------------------------------
def noise(scale=10.0, octaves=3, roughness=0.5, seed=0):
    """Smooth fractal noise spanning -1..1 (`noise.gain`), with features about `scale` mm across."""
    from .noise import gain
    args = (float(scale), int(octaves), float(roughness), int(seed))
    by = np.float32(gain(args[1], args[2]))

    def fn(ctx):
        values = fractal_on_grid(ctx.grid, *args, xp=ctx.xp)
        values *= by
        return ctx.xp.clip(values, -1.0, 1.0, out=values)
    return Field(fn, _digest("noise", "spans -1..1", *args), f"noise{args}", cache=True, range=(-1.0, 1.0))


def wave3d(pattern="ridges", period=5.0, warp=0.0, warp_scale=8.0, seed=0, axis="y"):
    """Wave pattern -1..1 (ridges along `axis`, or gyroid / schwarz / diamond), optionally bent
    in 3D by noise."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {', '.join(PATTERNS)}")
    if axis not in AXES:
        raise ValueError("axis must be x, y or z")
    args = (pattern, float(period), float(warp), float(warp_scale), int(seed), axis)
    return Field(lambda ctx: wave3d_on_grid(ctx.grid, *args, xp=ctx.xp), _digest("wave3d", *args), f"wave3d{args}",
                 cache=True, range=(-1.0, 1.0))


# --- simple shapes (guides for Topology Optimization, Keep / Avoid bodies) ------------------
def ball(centre, radius):
    """A ball, exact distance."""
    c, r = tuple(float(v) for v in centre), float(radius)

    def fn(ctx):
        return ctx.xp.sqrt(sum((ctx.axis(k) - c[k]) ** 2 for k in range(3))) - r
    return Field(fn, _digest("ball", c, r), "ball", (np.subtract(c, r), np.add(c, r)))


def bar(start, end, radius):
    """A rod with round ends from `start` to `end` (a capsule), exact distance."""
    a, b, r = np.asarray(start, float), np.asarray(end, float), float(radius)
    d = b - a
    length2 = float(d @ d)

    def fn(ctx):
        xp = ctx.xp
        p = [ctx.axis(k) - a[k] for k in range(3)]
        t = xp.clip(sum(p[k] * d[k] for k in range(3)) / length2, 0.0, 1.0) if length2 > 0 else 0.0
        return xp.sqrt(sum((p[k] - t * d[k]) ** 2 for k in range(3))) - r
    return Field(fn, _digest("bar", tuple(a), tuple(b), r), "bar", (np.minimum(a, b) - r, np.maximum(a, b) + r))


def box(centre, size, rounding=0.0):
    """A box `size` across, its edges rounded by `rounding` mm; exact distance."""
    c, half = np.asarray(centre, float), 0.5 * np.abs(np.asarray(size, float))
    rr = min(max(float(rounding), 0.0), float(half.min()))

    def fn(ctx):
        xp = ctx.xp
        q = [xp.abs(ctx.axis(k) - c[k]) - (half[k] - rr) for k in range(3)]
        outside = xp.sqrt(sum(xp.maximum(v, 0.0) ** 2 for v in q))
        inside = xp.minimum(xp.maximum(xp.maximum(q[0], q[1]), q[2]), 0.0)
        return outside + inside - rr
    return Field(fn, _digest("box", tuple(c), tuple(half), rr), "box", (c - half, c + half))


# --- implicit bodies ---------------------------------------------------------------------
def body_from_mesh(mesh, band):
    """Implicit body from a triangle mesh (e.g. a tessellated CAD body)."""
    # Keep only the raw arrays (a trimesh object drags its caches along, ~10x the size), as Shared
    # arrays: this field is pickled into every task sent to a worker, and each worker only needs
    # the mesh once, to build its search structures.
    fingerprint = mesh_fingerprint(mesh)
    V = Shared(np.ascontiguousarray(mesh.vertices, dtype=np.float64), key=fingerprint)
    Fc = Shared(np.ascontiguousarray(mesh.faces, dtype=np.int64), key=fingerprint)

    def fn(ctx):
        if ctx.xp is not np:  # GPU: NVIDIA Warp, ~50x faster
            from .sdf_gpu import mesh_to_sdf_gpu
            return mesh_to_sdf_gpu(V, Fc, ctx.grid, band, key=fingerprint)
        return mesh_to_sdf((V, Fc), ctx.grid, band, key=fingerprint)
    f = Field(fn, _digest("mesh", fingerprint, float(band)), "CADBody", mesh.bounds, cache=True)
    f.mesh = (fingerprint, len(mesh.faces))  # every process evaluating it builds search structures for the mesh
    return f


def _bspline_weights(u):
    """The four weights of a uniform cubic B-spline at u (0..1) between its second and third control value."""
    v = 1.0 - u
    return (v * v * v / 6.0, (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
            (3.0 * v * v * v - 6.0 * v * v + 4.0) / 6.0, u * u * u / 6.0)


def sampled_body(values, start, voxel, bounds=None):
    """An implicit body from values at samples `voxel` mm apart, the first at `start` (mm, negative inside):
    a cubic B-spline with the samples as its control values. Twice continuously differentiable, so a surface
    without the facets and stripes of a mesh of those samples (Topology Optimization's body was the distance to
    one), and a point's value never leaves the range of the 64 samples round it: no ripples between them, and no
    loose bits that the samples do not have (a spline through the samples rings: islands of 0.01-0.1 mm3 by
    every hard edge in them). At a sample it is the samples' [1, 4, 1] / 6 average along each axis, a blur of
    0.58 samples. Beyond the samples: the value at their edge plus the distance out. A pure function of
    position (blocks get the same values bit for bit: positions and weights come from the grid's own float64
    axes), worked out one axis at a time on the part of the samples a grid covers, on the CPU or the GPU. The
    samples are a Shared array: workers map them."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    shape = array.shape
    if len(shape) != 3 or min(shape) < 4:
        raise ValueError("a sampled body needs at least 4 samples along each axis")
    samples = Shared(array)
    start, voxel = tuple(float(v) for v in start), float(voxel)
    if bounds is None:
        bounds = (np.asarray(start), np.asarray(start) + voxel * (np.asarray(shape) - 1))

    def fn(ctx):
        return spline_values(samples.get(), start, voxel, ctx.grid.axes(), ctx.xp)
    return Field(fn, _digest("sampled", samples.key, start, voxel), "body", np.asarray(bounds, dtype=np.float64))


def spline_values(samples, start, voxel, axes, xp):
    """`sampled_body`'s values: the cubic B-spline with `samples` (3D, `voxel` mm apart, the first at `start`) as
    its control values, at the positions `axes` (three float64 arrays: a grid's axes, or a part of them)."""
    shape = samples.shape
    first, weights, beyond = [], [], []
    for k, positions in enumerate(axes):
        t = (positions - start[k]) / voxel
        inside = np.clip(t, 1.0, shape[k] - 2.0)
        lower = np.minimum(np.floor(inside), shape[k] - 3.0)
        first.append(lower.astype(np.int64) - 1)
        weights.append([w.astype(np.float32) for w in _bspline_weights(inside - lower)])
        beyond.append(np.abs(t - inside) * voxel)
    lo = [int(f.min()) for f in first]
    out = xp.asarray(np.ascontiguousarray(samples[tuple(slice(lo[k], int(first[k].max()) + 4) for k in range(3))]))
    for k in range(3):
        across = [-1 if a == k else 1 for a in range(3)]
        at = xp.asarray(first[k] - lo[k])
        total = None
        for tap in range(4):
            term = xp.take(out, at + tap, axis=k) * xp.asarray(weights[k][tap].reshape(across))
            total = term if total is None else total + term
        out = total
    if any(float(b.max()) > 0 for b in beyond):
        x, y, z = (xp.asarray(b.reshape([-1 if a == k else 1 for a in range(3)])) for k, b in enumerate(beyond))
        out = out + xp.sqrt(x * x + y * y + z * z).astype(np.float32)
    return out


def display_extent(field):
    """The box a field belongs to when shown with other bodies' fields: its bounds, else its extent."""
    if field.bounds is not None:
        return field.bounds
    extent = getattr(field, "extent", None)
    return None if extent is None else np.asarray(extent, dtype=np.float64)


def thinnest(field):
    """The thinnest wall or strut the bodies `field` is made of say they have, mm (None: none says): a
    lattice's or a Shell's `thinnest`. A mesher warns when its voxel cannot show it."""
    found, seen, stack = None, set(), [field]
    while stack:
        f = stack.pop()
        if f.key in seen:
            continue
        seen.add(f.key)
        thin = getattr(f, "thinnest", None)
        if thin is not None and thin > 0:
            found = thin if found is None else min(found, thin)
        stack.extend(f.children)
    return found


def surface_band(field):
    """The smallest `surface_band` among the fields `field` is made of, mm (None: none has one): a Face
    Mask, a Surface Pattern, a Picture or Text is only computed that far from the surface, 0 beyond."""
    found, seen, stack = None, set(), [field]
    while stack:
        f = stack.pop()
        if f.key in seen:
            continue
        seen.add(f.key)
        band = getattr(f, "surface_band", None)
        if band is not None:
            found = band if found is None else min(found, band)
        stack.extend(f.children)
    return found


def mesh_triangles(field):
    """Triangles of all distinct CAD meshes a field reads (each evaluating process builds search
    structures for them: ~0.5 KB per triangle)."""
    meshes, seen, stack = {}, set(), [field]
    while stack:
        f = stack.pop()
        if f.key in seen:
            continue
        seen.add(f.key)
        if getattr(f, "mesh", None):
            meshes[f.mesh[0]] = f.mesh[1]
        stack.extend(f.children)
    return sum(meshes.values())


def release_mesh_caches():
    """Drop this process's search structures for meshes (rebuilt when needed)."""
    from . import sdf, shared
    sdf._QUERIES.clear()
    shared.forget()
    import sys
    if "engine.segments" in sys.modules:
        sys.modules["engine.segments"]._TREES.clear()
    if "engine.sdf_gpu" in sys.modules:
        sys.modules["engine.sdf_gpu"]._MESHES.clear()


def offset(body, distance):
    """Grow the body outward by `distance` mm (a number or a field). Negative shrinks it."""
    return _lift(body) - distance


def shell(body, thickness):
    """Hollow wall of `thickness` mm centred on the original surface."""
    body = _lift(body)
    return abs_(body)._binary(_lift(thickness) * 0.5, np.subtract, "-", bounds="first")


def union(a, b): return minimum(a, b)
def intersect(a, b): return _lift(a)._binary(b, np.maximum, "max", bounds="intersect")
def subtract(a, b): return _lift(a)._binary(-_lift(b), np.maximum, "max", bounds="first")


EDGES = ("sharp", "round", "chamfer")


def _edged(a, b, radius, label, op, bounds, grown=None):
    a, b = _lift(a), _lift(b)
    r = float(radius)
    f = Field(lambda ctx: op(ctx.xp, ctx.eval(a), ctx.eval(b), r), _digest(label, a.key, b.key, r),
              f"{label}({a.name}, {b.name})", bounds, children=(a, b))
    f.grown = grown
    return f


def _round_max(xp, a, b, r):
    """max(a, b) with the edge rounded to radius r (exact where the surfaces meet at 90°; Mercury's
    hg_sdf fOpIntersectionRound)."""
    ua, ub = xp.maximum(r + a, 0.0), xp.maximum(r + b, 0.0)
    return xp.minimum(-r, xp.maximum(a, b)) + xp.sqrt(ua * ua + ub * ub)


def _round_min(xp, a, b, r):
    ua, ub = xp.maximum(r - a, 0.0), xp.maximum(r - b, 0.0)
    return xp.maximum(r, xp.minimum(a, b)) - xp.sqrt(ua * ua + ub * ub)


def subtract_edged(a, b, edge, radius):
    """A with B cut away, the edges where B's surface meets A's rounded (radius) or chamfered (legs
    of `radius` along both surfaces)."""
    if edge == "sharp" or radius <= 0:
        return subtract(a, b)
    a = _lift(a)
    if edge == "round":
        return _edged(a, b, radius, "subtract_round", lambda xp, x, y, r: _round_max(xp, x, -y, r), a.bounds, a.grown)
    return _edged(a, b, radius, "subtract_chamfer",
                  lambda xp, x, y, r: xp.maximum(xp.maximum(x, -y), (x - y + r) * math.sqrt(0.5)), a.bounds, a.grown)


FAN = (0.7, 0.95)   # a body's values count as fanning round an edge between these (`_meeting`'s curl)
CURL_GAIN = 1.25    # second differences read a fan's curl ~0.83 one sample from its edge (1 farther out)


def _meeting(xp, fa, fb, shape, rows):
    """For the samples `rows` (flat indices) of two bodies' values: (along, curl_a, curl_b).
    along, 0..1: the product of their unit gradients, cut to 0..1: 0 where the surfaces cross at 90
    degrees or more, 1 where they run along each other (faces the bodies share).
    curl, 0..1: value x Laplacian of each body (for a distance field: 0 beside a flat face, 1 where the
    values fan round an edge, R-to-distance in between beside a cylinder).
    Central differences; all 0 for samples on the array's outside (and where a gradient vanishes)."""
    n0, n1, n2 = shape
    i, j, k = xp.unravel_index(rows, shape)
    inner = (i > 0) & (i < n0 - 1) & (j > 0) & (j < n1 - 1) & (k > 0) & (k < n2 - 1)
    sub = rows[inner]
    a0, b0 = fa[sub], fb[sub]
    dot = xp.zeros(len(sub), dtype=np.float32)
    na, nb, la, lb = dot.copy(), dot.copy(), dot.copy(), dot.copy()
    for stride in (n1 * n2, n2, 1):
        ap, am, bp, bm = fa[sub + stride], fa[sub - stride], fb[sub + stride], fb[sub - stride]
        da, db = ap - am, bp - bm
        dot += da * db
        na += da * da
        nb += db * db
        la += ap + am - 2.0 * a0
        lb += bp + bm - 2.0 * b0
    norm = xp.sqrt(na * nb)
    known = norm > 1e-12
    out = [xp.zeros(len(rows), dtype=np.float32) for _ in range(3)]
    out[0][inner] = xp.where(known, xp.clip(dot / xp.where(known, norm, 1.0), 0.0, 1.0), 0.0)
    # (the squared gradient is over two samples: a x Laplacian = 4 a x second differences / it)
    gain = 4.0 * CURL_GAIN
    out[1][inner] = xp.where(na > 1e-12, xp.clip(gain * a0 * la / xp.where(na > 1e-12, na, 1.0), 0.0, 1.0), 0.0)
    out[2][inner] = xp.where(nb > 1e-12, xp.clip(gain * b0 * lb / xp.where(nb > 1e-12, nb, 1.0), 0.0, 1.0), 0.0)
    return out


def _round_union(xp, a, b, r, g, ca, cb):
    """A rounded union's value where a and b are both under r: r minus the distance to where both are
    at least r (what a ball of radius r rolling over both bodies cannot reach is filled). With the
    surfaces' angle (g) and each body's curl: the nearest such point lies straight out from one body
    where the other is at least r there too (worked out as a flat face's values, or an edge's fan, by the
    curl), else on the line where the two r-surfaces meet (as planes at that angle). g = 0 and no curl:
    Mercury's formula."""
    u, v = r - a, r - b
    flat = b + u * g
    b_there = flat + cb * (xp.sqrt(xp.maximum(b * b + u * u + 2.0 * b * u * g, 0.0)) - flat)
    flat = a + v * g
    a_there = flat + ca * (xp.sqrt(xp.maximum(a * a + v * v + 2.0 * a * v * g, 0.0)) - flat)
    line = xp.sqrt(xp.maximum(u * u + v * v - 2.0 * g * u * v, 0.0) / xp.maximum(1.0 - g * g, 1e-6))
    far = xp.where(b_there >= r, u, xp.where(a_there >= r, v, line))
    return xp.minimum(r - far, xp.minimum(a, b))


def _chamfer_union(xp, a, b, r, g, ca, cb):
    """A chamfered union's value: min(a, b) cut by the plane through the two points where a fillet of
    radius r would touch the faces (at 90 degrees, g = 0: r from the corner along both, (a + b - r) /
    sqrt 2; at a flatter angle less, down to nothing where the surfaces run along each other). No cut
    where a body's values fan round an edge (`FAN`)."""
    cut = (a + b - r * (1.0 - g)) / xp.sqrt(2.0 + 2.0 * g)
    fan = xp.clip((xp.maximum(ca, cb) - FAN[0]) / (FAN[1] - FAN[0]), 0.0, 1.0)
    return xp.minimum(xp.minimum(a, b), cut + 4.0 * r * fan)


def _union_edged(a, b, radius, label, bounds, grown):
    """A rounded or chamfered union that works out, at every sample near both bodies, the angle at which
    their surfaces meet and whether a body's values fan round an edge there (`_meeting`). Mercury's
    formulas assume two flat faces at 90 degrees: where the bodies share a face (two boxes side by side,
    a = b) they moved it out by 0.29 x radius (round) or half of it (chamfer), 2026-11-10: two 20 mm
    boxes, round 2 mm, +0.59 mm on the faces they share; and a body's edge lying hidden on the other's
    face read as a corner to fill."""
    r = np.float32(radius)
    kind = _round_union if "round" in label else _chamfer_union
    reach = r if kind is _round_union else np.float32(2.0) * r   # beyond it the result is min(a, b)

    def values(xp, va, vb):
        out = xp.minimum(va, vb)
        rows = xp.flatnonzero((va < reach) & (vb < reach))
        if len(rows):
            fa, fb = va.reshape(-1), vb.reshape(-1)
            out.reshape(-1)[rows] = kind(xp, fa[rows], fb[rows], r, *_meeting(xp, fa, fb, va.shape, rows))
        return out

    return _with_neighbours((a, b), values, _digest(label, a.key, b.key, float(r)), label, bounds, grown)


def _with_neighbours(bodies, values, key, label, bounds, grown):
    """The field values(xp, *the bodies' values), for `values` that read each sample's neighbours
    (`_meeting`)."""
    def fn(ctx):
        xp, grid = ctx.xp, ctx.grid
        if min(grid.shape) >= 3:
            return values(xp, *(ctx.eval_full(f) for f in bodies))
        # a few samples (a value read at a point): with their neighbours, from a grid one sample bigger
        from .grid import Grid
        more = Grid.block(grid.base, grid.voxel, [s - 1 for s in grid.start], [n + 2 for n in grid.shape])
        try:
            inner = Context(more, ctx.cache, xp)
            return values(xp, *(inner.eval_full(f) for f in bodies))[1:-1, 1:-1, 1:-1]
        except ValueError:   # a body that samples coarser lattices cannot start one sample earlier
            return values(xp, *(ctx.eval_full(f) for f in bodies))
    f = Field(fn, key, f"{label}({', '.join(f.name for f in bodies[:2])})", bounds, children=tuple(bodies))
    f.grown = grown
    f.neighbours = 1   # reads one sample on every side: blocks need it as a margin (`halo`)
    return f


def union_edged(a, b, edge, radius):
    """Both bodies, the inner edges where they meet filled round (radius) or with a chamfer, at the
    angle they meet at; nothing is added where the bodies' surfaces run along each other."""
    if edge == "sharp" or radius <= 0:
        return union(a, b)
    a, b = _lift(a), _lift(b)
    bounds, grown = _merge_bounds(a.bounds, b.bounds), _merged_core((a, b))
    return _union_edged(a, b, radius, "union_round_at" if edge == "round" else "union_chamfer_at", bounds, grown)


KEPT_EDGE = 0.25    # x the radius: `intersect_edged` rounds b in full where it is this much deeper than `keep`


def intersect_edged(a, b, radius, bounds="intersect", keep=None, edge="round"):
    """What the two bodies share, the edges where their surfaces cross rounded to `radius` at the angle they
    cross at (`_meeting`); where the surfaces run along each other (faces the bodies share: a part and the
    box that crops it) nothing changes, where Mercury's formula sinks them by 0.29 x the radius. Exact
    inside and up to half the radius outside; farther out the rounding fades to the plain max(a, b) at one
    radius (a lower bound of the distance there). `bounds`: "intersect", or "first" (b has none: a body
    minus another, given as its negative). `edge` "chamfer": a flat cut instead of the arc, through the two
    lines where the arc would touch the surfaces (legs of `radius` at 90 degrees, as the Booleans'
    `_chamfer_union`).
    `keep`: a part of b whose own surface is left as it is (a CAD body joined into b: Topology
    Optimization's Keep). An edge of b's that lies on a face of a's (a plate's lower edge on the face of the
    box that crops it) reads as two surfaces crossing only on its one side, and came out half rounded, with
    a step along the edge: no rounding where b is `keep` (fading in to full where b is `KEPT_EDGE` x the
    radius deeper: the other bodies in b, and the fillets between them and `keep`), and nothing is taken
    off `keep` itself."""
    a, b = _lift(a), _lift(b)
    box = a.bounds if bounds == "first" else _merge_bounds(a.bounds, b.bounds, mode="intersect")
    if radius <= 0 or edge == "sharp":
        return a._binary(b, np.maximum, "max", bounds=bounds)
    r = np.float32(radius)
    bodies = (a, b) if keep is None else (a, b, _lift(keep))
    kind = _chamfer_union if edge == "chamfer" else _round_union

    def values(xp, va, vb, vk=None):
        out = xp.maximum(va, vb)
        rows = xp.flatnonzero((va > -r) & (vb > -r) & (out < r))
        if len(rows):
            fa, fb = va.reshape(-1), vb.reshape(-1)
            along = _meeting(xp, fa, fb, va.shape, rows)[0]
            plain = out.reshape(-1)[rows]
            # (the rounded union of what is outside the two, seen from inside; no curl: only crossing surfaces)
            none = np.float32(0.0)
            more = xp.maximum(-kind(xp, -fa[rows], -fb[rows], r, along, none, none) - plain, 0.0)
            more *= xp.clip(np.float32(2.0) - plain * (np.float32(2.0) / r), 0.0, 1.0)
            if vk is not None:
                kept = vk.reshape(-1)[rows]
                more *= xp.clip((kept - fb[rows]) / (np.float32(KEPT_EDGE) * r), 0.0, 1.0)
                out.reshape(-1)[rows] = xp.minimum(plain + more, xp.maximum(fa[rows], kept))
            else:
                out.reshape(-1)[rows] = plain + more
        return out
    grown = a.grown if bounds == "first" else _merged_core((a, b), "intersect")
    label = "intersect_chamfer_at" if edge == "chamfer" else "intersect_round_at"
    return _with_neighbours(bodies, values, _digest(label, *(f.key for f in bodies), float(r)), label, box, grown)


def bulge(h, k, order):
    """What a smooth minimum of size k adds below min(a, b), h = 1 - |a - b| / k (0..1): order 1 the
    quadratic one (G1: tangent, the curvature jumps where the blend meets the bodies), 2 a cubic (G2:
    curvature continuous), 3 a quintic (G3, also smooth through the blend's middle, where the cubic's
    third derivative jumps). Each keeps the field's slope within that of a and b. The frame kernel has
    the same (`frame.kernel._bulge`)."""
    if order >= 3:
        return k * h ** 4 * (5.0 - 2.0 * h) * 0.05
    if order == 2:
        return k * h ** 3 / 6.0
    return h * h * k * 0.25


def smooth_union(a, b, radius, order=1):
    """Union with a fillet-like blend of roughly `radius` mm where the bodies meet (0 = a plain
    union); `order` its continuity (`bulge`)."""
    if radius < 0:
        raise ValueError("radius must be 0 or more")
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2 or 3")
    if radius == 0:
        return union(a, b)
    a, b = _lift(a), _lift(b)

    def fn(ctx):
        da, db = ctx.eval(a), ctx.eval(b)
        if order > 1:
            h = np.maximum(1.0 - np.abs(da - db) / radius, 0.0)
            return np.minimum(da, db) - bulge(h, radius, order)
        h = np.clip(0.5 + 0.5 * (db - da) / radius, 0, 1)
        return db * (1 - h) + da * h - radius * h * (1 - h)
    key = _digest("smooth_union", a.key, b.key, float(radius), *((order,) if order > 1 else ()))
    f = Field(fn, key, f"smooth_union({a.name}, {b.name})", _merge_bounds(a.bounds, b.bounds), children=(a, b))
    f.grown = _merged_core((a, b))
    return f


def smoothen(body, iterations=3, grid_size=0.1):
    """Round off sharp creases, like `iterations` passes of a 3x3x3 box filter on a `grid_size` mm
    grid (the Smoothen Body node's inputs). Implemented as the equivalent Gaussian blur in mm, so the
    result looks the same whether it is evaluated on a coarse preview grid or the final one."""
    body = _lift(body)
    sigma_mm = grid_size * math.sqrt(iterations * 2.0 / 3.0)  # variance of n box passes

    def fn(ctx):
        v = ctx.eval_full(body)
        sigma = sigma_mm / ctx.grid.voxel
        if sigma < 0.05:
            return v
        return backend_gaussian(ctx.xp)(v, sigma=sigma, mode="nearest", truncate=SMOOTH_TRUNCATE)
    f = Field(fn, _digest("smoothen", body.key, int(iterations), float(grid_size)),
              f"smoothen({body.name})", body.bounds, children=(body,))
    f.grown = body.grown
    f.blur = sigma_mm
    return f


SMOOTH_TRUNCATE = 3.0  # Gaussian cut off at 3 sigma


def halo(field, voxel):
    """Samples of margin a block needs on every side so that `field` gives the same values there
    as on the whole grid: smoothing reads its neighbours (nested smoothing adds up)."""
    memo = {}

    def reach(f):
        if f.key not in memo:
            blur = getattr(f, "blur", 0.0) / voxel
            own = int(SMOOTH_TRUNCATE * blur + 0.5) + 1 if blur >= 0.05 else 0  # scipy's radius + 1
            own = max(own, int(getattr(f, "neighbours", 0)))                    # e.g. a gradient's one sample
            memo[f.key] = own + max((reach(c) for c in f.children), default=0)
        return memo[f.key]
    return reach(field)
