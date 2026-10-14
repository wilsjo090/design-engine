"""Lattices as implicit bodies: TPMS surfaces, periodic strut lattices and a Voronoi foam.

All of them are pure functions of position (no grid-dependent sampling), so they work at any
preview quality and in block meshing. Values are distances in mm (exact for struts, a first-order
estimate f/|grad f| for TPMS), negative inside the material, so thickness inputs are in mm and
can be fields (graded lattices).
"""
import math

import numpy as np
from scipy.spatial import Voronoi

from .fields import Context, Field, _digest, _lift
from .shared import Shared

# --- density: the share of solid material instead of a thickness ----------------------------
# How much of a lattice is solid grows very unevenly with strut thickness: a Voronoi foam with
# 5 mm cells is 42% solid at 2 mm, 98% at 5 mm and completely solid from ~7 mm on. A fade in mm
# therefore looks like a sharp step. In density units the value is the solid share itself (the
# thickness is looked up from the lattice's own curve), so a fade changes evenly.
UNITS = ("mm", "density")
DENSITY_LEVELS = np.linspace(0.0, 1.0, 201)


def _density_curve(distances, scale):
    """Thickness (scale 2: material within t/2 of the struts or sheet) or offset (scale 1:
    material where the distance is below it) for each density in DENSITY_LEVELS: the share of
    sample points that end up solid. The ends get a 5% margin, so 1 is really solid and 0 empty."""
    q = np.quantile(np.asarray(distances, dtype=np.float64).ravel(), DENSITY_LEVELS) * scale
    margin = 0.05 * max(q[-1] - q[0], 1e-6)
    q[-1] += margin
    if scale == 1:
        q[0] -= margin
    return q


def _lookup(field, ys, label):
    """`field` (a density, 0..1) turned into the thickness or offset `ys` gives for it (linear in
    between, clamped at the ends)."""
    field = _lift(field)
    xs32, ys32 = DENSITY_LEVELS.astype(np.float32), np.asarray(ys, dtype=np.float32)

    def fn(ctx):
        xp = ctx.xp
        v = xp.asarray(ctx.eval(field), dtype=xp.float32)
        return xp.interp(v, xp.asarray(xs32), xp.asarray(ys32)).astype(xp.float32)
    rng = None
    if field.range is not None:
        rng = tuple(float(np.interp(r, DENSITY_LEVELS, ys)) for r in field.range)
    return Field(fn, _digest("density", field.key, ys32.tobytes()), label, field.bounds, children=(field,), range=rng)


def _cell_samples(field, cell, n=48):
    """Values of a periodic lattice field over one cell (n³ points)."""
    from .grid import Grid
    step = float(cell) / n
    return Context(Grid((0.0, 0.0, 0.0), (cell - step,) * 3, step)).eval_full(field)


def _check_units(units):
    if units not in UNITS:
        raise ValueError(f"units must be one of {', '.join(UNITS)}")


# --- TPMS ----------------------------------------------------------------------------------
TPMS = {
    "gyroid": lambda x, y, z: np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x),
    "schwarz_p": lambda x, y, z: np.cos(x) + np.cos(y) + np.cos(z),
    "diamond": lambda x, y, z: (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
                                + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z)),
    "neovius": lambda x, y, z: 3 * (np.cos(x) + np.cos(y) + np.cos(z)) + 4 * np.cos(x) * np.cos(y) * np.cos(z),
    "lidinoid": lambda x, y, z: (0.5 * (np.sin(2 * x) * np.cos(y) * np.sin(z) + np.sin(2 * y) * np.cos(z) * np.sin(x)
                                        + np.sin(2 * z) * np.cos(x) * np.sin(y))
                                 - 0.5 * (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
                                          + np.cos(2 * z) * np.cos(2 * x)) + 0.15),
}
TPMS_MODES = ("sheet", "solid", "solid (other side)")


def _trig(x, y, z):
    return np.sin(x), np.cos(x), np.sin(y), np.cos(y), np.sin(z), np.cos(z)


def _gyroid_gradient(x, y, z):
    sx, cx, sy, cy, sz, cz = _trig(x, y, z)
    return cx * cy - sz * sx, cy * cz - sx * sy, cz * cx - sy * sz


def _schwarz_p_gradient(x, y, z):
    return -np.sin(x), -np.sin(y), -np.sin(z)


def _diamond_gradient(x, y, z):
    sx, cx, sy, cy, sz, cz = _trig(x, y, z)
    return (cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz,
            sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz,
            sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz)


def _neovius_gradient(x, y, z):
    sx, cx, sy, cy, sz, cz = _trig(x, y, z)
    return -3 * sx - 4 * sx * cy * cz, -3 * sy - 4 * cx * sy * cz, -3 * sz - 4 * cx * cy * sz


def _lidinoid_gradient(x, y, z):
    sx, cx, sy, cy, sz, cz = _trig(x, y, z)
    s2x, c2x, s2y, c2y, s2z, c2z = _trig(2 * x, 2 * y, 2 * z)
    return (0.5 * (2 * c2x * cy * sz + s2y * cz * cx - s2z * sx * sy) + s2x * (c2y + c2z),
            0.5 * (-s2x * sy * sz + 2 * c2y * cz * sx + s2z * cx * cy) + s2y * (c2x + c2z),
            0.5 * (s2x * cy * cz - s2y * sz * sx + 2 * c2z * cx * sy) + s2z * (c2x + c2y))


# the exact gradients of the TPMS functions (6 extra evaluations by central differences made the
# lattice ~3x slower)
TPMS_GRADIENT = {"gyroid": _gyroid_gradient, "schwarz_p": _schwarz_p_gradient, "diamond": _diamond_gradient,
                 "neovius": _neovius_gradient, "lidinoid": _lidinoid_gradient}


def tpms(kind, cell, thickness, mode="sheet", offset=0.0, bounds=None, units="mm"):
    """A triply periodic minimal surface lattice with `cell` mm repeats. sheet: a wall `thickness`
    mm thick centred on the surface; solid: the material on one side of the surface, moved by
    `offset` mm (0 = about half the volume). `thickness`/`offset` may be fields. units="density":
    `thickness` (sheet) or `offset` (solid modes) is the share of solid material, 0..1."""
    if kind not in TPMS:
        raise ValueError(f"type must be one of {', '.join(TPMS)}")
    if mode not in TPMS_MODES:
        raise ValueError(f"mode must be one of {', '.join(TPMS_MODES)}")
    _check_units(units)
    if units == "density":
        base = _cell_samples(tpms(kind, cell, 0.0, mode, 0.0), cell)  # |distance| (sheet) or distance
        if mode == "sheet":
            thickness = _lookup(thickness, _density_curve(base, 2.0), "thickness from density")
        else:
            offset = _lookup(offset, _density_curve(base, 1.0), "offset from density")
    f, gradient = TPMS[kind], TPMS_GRADIENT[kind]
    k = 2 * math.pi / float(cell)
    amount = _lift(thickness if mode == "sheet" else offset)

    def fn(ctx):
        xp = ctx.xp
        x, y, z = k * ctx.axis(0), k * ctx.axis(1), k * ctx.axis(2)
        value = f(x, y, z)
        # distance estimate: value / |gradient| (the gradient in mm units)
        gx, gy, gz = gradient(x, y, z)
        slope = k * xp.sqrt(gx * gx + gy * gy + gz * gz)
        del gx, gy, gz
        dist = value / xp.maximum(slope, 0.1 * k)  # no blow-up at the surface's saddle points
        a = ctx.eval(amount)
        if mode == "sheet":
            return (xp.abs(dist) - a * 0.5).astype(xp.float32)
        return ((dist if mode == "solid" else -dist) - a).astype(xp.float32)
    return Field(fn, _digest("tpms", "exact gradient", kind, float(cell), mode, amount.key),
                 f"tpms({kind}, {cell:g} mm)", bounds, children=(amount,))


# --- strut lattices -------------------------------------------------------------------------
def _cube_edges():
    corners = [np.array(c, float) for c in np.ndindex(2, 2, 2)]
    return [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:] if np.abs(a - b).sum() == 1]


def _face_diagonals():
    out = []
    for axis in range(3):
        for side in (0.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            for (su, sv), (eu, ev) in (((0, 0), (1, 1)), ((1, 0), (0, 1))):
                p, q = np.zeros(3), np.zeros(3)
                p[axis] = q[axis] = side
                p[u], p[v], q[u], q[v] = su, sv, eu, ev
                out.append((p, q))
    return out


def _octahedron():
    centres = []
    for axis in range(3):
        for side in (0.0, 1.0):
            c = np.full(3, 0.5)
            c[axis] = side
            centres.append(c)
    # neighbouring face centres only (0.71 apart), not opposite ones (1.0 apart, through the centre)
    return [(a, b) for i, a in enumerate(centres) for b in centres[i + 1:] if np.linalg.norm(a - b) < 0.9]


_BCC = [(np.array(a, float), 1 - np.array(a, float)) for a in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))]
STRUTS = {
    "cubic": _cube_edges(),
    "bcc": _BCC,
    "fcc": _face_diagonals(),
    "octet": _face_diagonals() + _octahedron(),
    "cubic + bcc": _cube_edges() + _BCC,
}


def _clip(a, b, lo=0.0, hi=0.5):
    """The part of segment a-b inside the box [lo, hi]^3, or None (Liang-Barsky)."""
    t0, t1, d = 0.0, 1.0, b - a
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if a[i] < lo - 1e-12 or a[i] > hi + 1e-12:
                return None
            continue
        ta, tb = (lo - a[i]) / d[i], (hi - a[i]) / d[i]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1 + 1e-12:
            return None
    p, q = a + t0 * d, a + t1 * d
    return None if np.linalg.norm(q - p) < 1e-9 else (p, q)


def folded_struts(kind):
    """The lattice's struts inside one octant of a cell, [0, 0.5]^3 in cell units. These lattices
    are mirror-symmetric about every cell face and mid-plane, so folding a point into the octant
    (distance to the nearest whole cell coordinate) and measuring to these struts gives the
    distance to the whole infinite lattice: the nearest strut can always be mirrored into it."""
    segs, seen = [], set()
    for a, b in STRUTS[kind]:
        c = _clip(a, b)
        if c is None:
            continue
        key = tuple(sorted((tuple(np.round(c[0], 9)), tuple(np.round(c[1], 9)))))
        if key not in seen:
            seen.add(key)
            segs.append(c)
    return segs


def _segment_distance(px, py, pz, a, b, xp):
    """Distance from points (broadcastable float32 arrays) to segment a-b. Endpoints as plain
    floats: numpy float64 scalars would turn every array float64 (twice the memory)."""
    a0, a1, a2 = (float(v) for v in a)
    d0, d1, d2 = (float(v) for v in np.asarray(b) - np.asarray(a))
    t = xp.clip(((px - a0) * d0 + (py - a1) * d1 + (pz - a2) * d2) / (d0 * d0 + d1 * d1 + d2 * d2), 0.0, 1.0)
    return xp.sqrt((px - a0 - t * d0) ** 2 + (py - a1 - t * d1) ** 2 + (pz - a2 - t * d2) ** 2)


def _smooth_min(a, b, k, xp):
    """Polynomial smooth minimum: rounds where the two meet by about k (k = 0: plain min)."""
    if k <= 0:
        return xp.minimum(a, b)
    h = xp.maximum(k - xp.abs(a - b), 0.0) / k
    return xp.minimum(a, b) - h * h * k * 0.25


def struts(kind, cell, thickness, blend=0.0, bounds=None, units="mm"):
    """A periodic strut lattice with `cell` mm cells and struts `thickness` mm thick (a field gives
    graded struts); `blend` mm rounds the nodes where struts meet. units="density": `thickness`
    is the share of solid material, 0..1."""
    if kind not in STRUTS:
        raise ValueError(f"type must be one of {', '.join(STRUTS)}")
    _check_units(units)
    if units == "density":
        base = _cell_samples(struts(kind, cell, 0.0, blend), cell)  # distance to the struts
        thickness = _lookup(thickness, _density_curve(base, 2.0), "thickness from density")
    segs = folded_struts(kind)
    cell, blend = float(cell), float(blend)
    thickness = _lift(thickness)

    def fn(ctx):
        xp = ctx.xp
        u = [ctx.axis(i) / cell for i in range(3)]
        u = [xp.abs(c - xp.rint(c)) for c in u]  # fold into [0, 0.5] per axis
        d = None
        for a, b in segs:
            s = _segment_distance(*u, a, b, xp) * cell
            d = s if d is None else _smooth_min(d, s, blend, xp)
        return (d - ctx.eval(thickness) * 0.5).astype(xp.float32)
    return Field(fn, _digest("struts", kind, cell, blend, thickness.key), f"struts({kind}, {cell:g} mm)", bounds,
                 children=(thickness,))


# --- Voronoi foam ---------------------------------------------------------------------------
def voronoi_struts(bounds, cell, randomness=1.0, seed=0):
    """Edges of the Voronoi cells of jittered grid points filling `bounds` (+ a margin), as
    (n, 2, 3) segments. randomness 0 = a regular grid of cells, 1 = fully random positions."""
    lo, hi = np.asarray(bounds, float)
    lo, hi = lo - 1.5 * cell, hi + 1.5 * cell
    counts = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    if np.prod(counts) > 2_000_000:
        raise ValueError(f"{np.prod(counts):,} Voronoi cells: use a larger cell size")
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in counts], indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-0.5, 0.5, idx.shape) * max(float(randomness), 0.02)  # a little keeps Qhull happy
    points = lo + (idx + 0.5 + jitter) * cell
    vor = Voronoi(points)
    edges = set()
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        for i in range(len(ridge)):
            a, b = ridge[i], ridge[(i + 1) % len(ridge)]
            edges.add((a, b) if a < b else (b, a))
    seg = vor.vertices[np.array(sorted(edges))]
    inside = np.all((seg >= lo).all(axis=2) & (seg <= hi).all(axis=2), axis=1)
    return seg[inside]


def _split(segs, longest):
    """Cut segments into pieces no longer than `longest`."""
    n = np.maximum(np.ceil(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) / longest).astype(int), 1)
    seg = np.repeat(np.arange(len(segs)), n)
    i = np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n)
    t0, t1 = (i / n[seg])[:, None], ((i + 1) / n[seg])[:, None]
    a, d = segs[seg, 0], segs[seg, 1] - segs[seg, 0]
    return np.stack([a + t0 * d, a + t1 * d], axis=1)


def voronoi(bounds, cell, thickness, randomness=1.0, seed=0, units="mm"):
    """A Voronoi foam of struts `thickness` mm thick (a field gives graded struts) filling the box
    `bounds`, cells about `cell` mm across. units="density": `thickness` is the share of solid
    material, 0..1 (measured on this foam)."""
    _check_units(units)
    if units == "density":
        base = voronoi(bounds, cell, 0.0, randomness, seed)
        thickness = _lookup(thickness, _density_curve(_foam_samples(base, bounds, cell), 2.0),
                            "thickness from density")
    segs = voronoi_struts(bounds, float(cell), randomness, seed)
    if len(segs) == 0:
        raise ValueError("no Voronoi cells inside the box")
    # Exact distance to the nearest strut (engine/segments.py). Struts are cut into pieces of at
    # most a third of a cell, so the search boxes around them stay tight. The pieces travel to
    # worker processes as a Shared array: once per worker, not in every task.
    pieces = Shared(_split(segs, float(cell) / 3))
    thickness = _lift(thickness)
    key = _digest("voronoi", tuple(np.round(np.asarray(bounds, float).ravel(), 6)), float(cell), float(randomness),
                  int(seed), thickness.key)

    def fn(ctx):
        from .segments import segment_distance
        t = ctx.eval(thickness)
        dist = segment_distance(pieces, ctx.grid, float(cell) / 4, key=key, gpu=ctx.xp is not np)
        return dist - ctx.xp.asarray(t, dtype=ctx.xp.float32) * 0.5
    return Field(fn, key, f"voronoi({cell:g} mm)", np.asarray(bounds, float), children=(thickness,), cache=True)


def _foam_samples(field, bounds, cell, most=400_000):
    """Distances to the struts over the foam's box (its middle part if the box is large): cell/8
    apart, at most `most` points."""
    from .grid import Grid
    lo, hi = np.asarray(bounds, dtype=np.float64)
    centre, half = (lo + hi) / 2, np.minimum((hi - lo) / 2, 4 * cell)
    lo, hi = centre - half, centre + half
    step = max(cell / 8, (np.prod(np.maximum(hi - lo, cell / 8)) / most) ** (1 / 3))
    return Context(Grid(lo, np.maximum(hi, lo + step), step)).eval_full(field)


# --- putting a lattice into a part ---------------------------------------------------------
def fill(body, lattice, skin=0.0):
    """The lattice inside the body, plus a solid skin `skin` mm thick along the body's surface
    (0 = no skin: the lattice is just cut to the body's shape)."""
    from . import fields as F
    inside = F.intersect(body, lattice)
    if not isinstance(skin, Field) and float(skin) <= 0:
        return inside
    shell = F.subtract(body, F.offset(body, -_lift(skin)))
    return F.union(shell, inside)
