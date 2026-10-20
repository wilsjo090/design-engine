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
                 children=(v, u))


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
    """Smooth fractal noise, roughly -1..1, with features about `scale` mm across."""
    args = (float(scale), int(octaves), float(roughness), int(seed))
    return Field(lambda ctx: fractal_on_grid(ctx.grid, *args, xp=ctx.xp), _digest("noise", *args), f"noise{args}",
                 cache=True, range=(-1.0, 1.0))


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


def display_extent(field):
    """The box a field belongs to when shown with other bodies' fields: its bounds, else its extent."""
    if field.bounds is not None:
        return field.bounds
    extent = getattr(field, "extent", None)
    return None if extent is None else np.asarray(extent, dtype=np.float64)


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


def union_edged(a, b, edge, radius):
    """Both bodies, the inner edges where they meet filled round (radius) or with a chamfer."""
    if edge == "sharp" or radius <= 0:
        return union(a, b)
    a, b = _lift(a), _lift(b)
    bounds, grown = _merge_bounds(a.bounds, b.bounds), _merged_core((a, b))
    if edge == "round":
        return _edged(a, b, radius, "union_round", _round_min, bounds, grown)
    return _edged(a, b, radius, "union_chamfer",
                  lambda xp, x, y, r: xp.minimum(xp.minimum(x, y), (x + y - r) * math.sqrt(0.5)), bounds, grown)


def smooth_union(a, b, radius):
    """Union with a fillet-like blend of roughly `radius` mm where the bodies meet (0 = a plain
    union)."""
    if radius < 0:
        raise ValueError("radius must be 0 or more")
    if radius == 0:
        return union(a, b)
    a, b = _lift(a), _lift(b)

    def fn(ctx):
        da, db = ctx.eval(a), ctx.eval(b)
        h = np.clip(0.5 + 0.5 * (db - da) / radius, 0, 1)
        return db * (1 - h) + da * h - radius * h * (1 - h)
    f = Field(fn, _digest("smooth_union", a.key, b.key, float(radius)),
              f"smooth_union({a.name}, {b.name})", _merge_bounds(a.bounds, b.bounds), children=(a, b))
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
            memo[f.key] = own + max((reach(c) for c in f.children), default=0)
        return memo[f.key]
    return reach(field)
