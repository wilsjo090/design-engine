"""2D shapes and layouts for face patterns (vent holes now, raised or sunk textures later).

A shape is an exact signed distance in mm (negative inside), in the face's own 2D frame: the
standard closed forms for a circle, a rounded regular polygon, a stadium (slot) and a rhombus; the
cross is the union of two rounded boxes (exact outside, a little short of the true distance
inside its centre). So diameters, flats and webs come out exact up to the voxel size. Shapes and
layouts are separate: every shape works in every layout. Written against an array module `xp`, so
it runs unchanged on numpy or CuPy.

Size is the shape's smallest width: the diameter of a round hole, across the flats of a polygon
(for an odd number of sides, the diameter of the circle that fits inside), the width of a slot,
of a diamond and of a cross's arms: always the largest round pin that fits through.

Voronoi cells are the exception: each centre's own cell of the layout (`voronoi_cells`, convex
polygons moved in by half the web), so every one has its own outline (`polygon_sdf`).
"""
import math

import numpy as np

SHAPES = ("round", "square", "hexagon", "triangle", "octagon", "polygon", "slot", "diamond", "cross", "rectangle",
          "star", "chevron", "voronoi", "delaunay")
SIDES = {"square": 4, "hexagon": 6, "triangle": 3, "octagon": 8}
LONG = ("slot", "diamond", "cross", "rectangle", "star", "chevron")    # shapes that also have a Length
CELLS = ("voronoi", "delaunay")          # shapes made from the layout's own cells (each its own outline)
LAYOUTS = ("grid", "staggered", "rings", "spiral")
HONEYCOMB = ("round", "hexagon", "voronoi", "delaunay")  # staggered rows packed as a honeycomb (sqrt(3)/2)
GOLDEN = math.pi * (3 - math.sqrt(5))    # the sunflower's turn between neighbouring seeds (137.5°)
CHEVRON = math.radians(30)               # a chevron's arms slope this much below the horizontal
ALTERNATE = ("none", "checker 90", "checker 180", "rows 180")   # every other cell (or row) turned


class Shape:
    """One hole's outline. `rotation` in degrees turns it counter-clockwise; `corner` rounds the
    convex corners of polygons, rectangles, diamonds, crosses, stars and chevrons (the size stays)."""

    def __init__(self, kind, size, length=0.0, sides=6, corner=0.0, rotation=0.0):
        if kind not in SHAPES:
            raise ValueError(f"shape must be one of {', '.join(SHAPES)}")
        if size <= 0:
            raise ValueError("size must be greater than 0")
        self.kind, self.size, self.rotation = kind, float(size), float(rotation)
        self.sides = SIDES.get(kind, int(sides))
        if kind in ("polygon", "star") and self.sides < 3:
            raise ValueError(f"a {kind} needs at least 3 {'sides' if kind == 'polygon' else 'points'}")
        self.length = max(float(length), self.size) if kind in LONG else self.size
        # a corner radius beyond the shape's inner radius would turn it into a circle of another size
        inner = _apothem(self.size / 2, self.length / 2) if kind == "diamond" else self.size / 2
        if kind in CELLS:   # size = the web between cells; corners are limited by each cell
            inner = math.inf
        self.corner = min(max(float(corner), 0.0), 0.999 * inner) if kind not in ("round", "slot") else 0.0
        a = math.radians(self.rotation)
        self._cos, self._sin = math.cos(a), math.sin(a)
        # odd polygons are centred half way between their flat side and their tip, so that turned
        # 180° they fill the same band (alternating triangles tile)
        self._lift = 0.0
        if self.polygonal and self.sides % 2:
            w, r = self.size / 2, self.corner
            self._lift = ((w - r) / math.cos(math.pi / self.sides) + r - w) / 2
        self._poly = None   # stars and chevrons: their outline as a polygon (before rounding)
        if kind == "star":
            self._poly = _star(self.sides, self.size / 2 - self.corner, self.length / 2 - self.corner)
        elif kind == "chevron":
            self._poly = _chevron(self.size - 2 * self.corner, self.length - 2 * self.corner)

    @property
    def polygonal(self):
        return self.kind in SIDES or self.kind == "polygon"

    def key(self):
        return (self.kind, self.size, self.length, self.sides if self.polygonal or self.kind == "star" else 0,
                self.corner, self.rotation)

    def sdf(self, x, y, xp=np):
        """Signed distance (mm) at 2D points (x, y) relative to the shape's centre."""
        u = x * self._cos + y * self._sin   # into the shape's own frame (turned back by its rotation)
        v = y * self._cos - x * self._sin
        w, r = self.size / 2, self.corner
        if self.kind == "round":
            return xp.sqrt(u * u + v * v) - w
        if self.polygonal:
            return _polygon(u, v + self._lift, self.sides, w - r, xp) - r
        if self.kind == "slot":
            return xp.sqrt(xp.maximum(xp.abs(u) - (self.length - self.size) / 2, 0.0) ** 2 + v * v) - w
        if self.kind == "diamond":
            bx, by = w, self.length / 2
            s = 1.0 - r / _apothem(bx, by)  # moved in by r on every side, then grown back round
            return _rhombus(u, v, bx * s, by * s, xp) - r
        if self.kind == "rectangle":
            return _box(u, v, self.length / 2 - r, w - r, xp) - r
        if self._poly is not None:          # star, chevron
            return simple_polygon_sdf(u, v, self._poly, xp) - r
        hl = self.length / 2 - r                    # cross: two boxes
        return xp.minimum(_box(u, v, hl, w - r, xp), _box(u, v, w - r, hl, xp)) - r

    def _corners(self):
        """The outline's corners before rounding, in the shape's own frame (polygonal shapes)."""
        w, r = self.size / 2, self.corner
        if self.kind == "diamond":
            k = 1.0 - r / _apothem(w, self.length / 2)
            return np.array([[w * k, 0], [-w * k, 0], [0, self.length / 2 * k], [0, -self.length / 2 * k]])
        if self.kind in ("rectangle", "cross"):
            hl, hw = self.length / 2 - r, w - r
            box = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
            return box if self.kind == "rectangle" else np.concatenate([box, box[:, ::-1]])
        if self._poly is not None:
            return self._poly
        n = self.sides
        offset = 0.0 if n % 2 == 0 else -math.pi / 2
        a = offset + math.pi / n + 2 * math.pi * np.arange(n) / n
        return (w - r) / math.cos(math.pi / n) * np.column_stack([np.cos(a), np.sin(a)]) - [0, self._lift]

    def circumradius(self):
        """Distance from the centre to the farthest point of the outline."""
        w, r = self.size / 2, self.corner
        if self.kind == "round":
            return w
        if self.kind == "slot":
            return self.length / 2
        return float(np.linalg.norm(self._corners(), axis=1).max()) + r

    def extent(self):
        """Half width and half height of the turned shape's bounding box."""
        c, s, w, r = abs(self._cos), abs(self._sin), self.size / 2, self.corner
        if self.kind == "round":
            return w, w
        if self.kind == "slot":
            h = (self.length - self.size) / 2
            return h * c + w, h * s + w
        corners = self._corners()
        x = corners[:, 0] * self._cos - corners[:, 1] * self._sin
        y = corners[:, 0] * self._sin + corners[:, 1] * self._cos
        return float(np.abs(x).max()) + r, float(np.abs(y).max()) + r

    def area(self):
        """Closed form, mm² (stars and chevrons with rounded corners: counted on a fine raster)."""
        w, r = self.size / 2, self.corner
        if self.kind == "round":
            return math.pi * w * w
        if self.polygonal:
            n, a = self.sides, w - r
            return n * a * a * math.tan(math.pi / n) + 2 * n * a * math.tan(math.pi / n) * r + math.pi * r * r
        if self.kind == "slot":
            return math.pi * w * w + (self.length - self.size) * self.size
        if self.kind == "diamond":
            s = 1.0 - r / _apothem(w, self.length / 2)
            bx, by = w * s, self.length / 2 * s
            return 2 * bx * by + 4 * math.hypot(bx, by) * r + math.pi * r * r
        if self.kind == "rectangle":
            return self.length * self.size - (4 - math.pi) * r * r
        if self._poly is not None:
            if r == 0:
                x, y = self._poly[:, 0], self._poly[:, 1]
                return abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) / 2
            R = self.circumradius()
            axis = np.linspace(-R, R, 2001)
            gx, gy = np.meshgrid(axis, axis, indexing="ij")
            return float((self.sdf(gx, gy) < 0).sum() * (axis[1] - axis[0]) ** 2)
        return 2 * (self.length * self.size - (4 - math.pi) * r * r) - self.size ** 2

    def outline(self, n=96):
        """`n` points on the outline (n, 2), found along rays from the centre (every shape here is
        star-shaped about its centre)."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        d = np.column_stack([np.cos(t), np.sin(t)])
        lo, hi = np.zeros(n), np.full(n, 1.01 * self.circumradius())
        for _ in range(40):
            mid = (lo + hi) / 2
            inside = self.sdf(d[:, 0] * mid, d[:, 1] * mid) < 0
            lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
        return d * ((lo + hi) / 2)[:, None]

    def describe(self):
        if self.kind == "voronoi":
            return f"Voronoi cells, web {self.size:g} mm"
        if self.kind == "delaunay":
            return f"triangle cells, web {self.size:g} mm"
        if self.kind == "round":
            return f"round {self.size:g} mm"
        if self.kind == "star":
            return f"{self.sides}-point star {self.size:g} / {self.length:g} mm"
        if self.kind in LONG:
            return f"{self.kind} {self.size:g} x {self.length:g} mm"
        name = f"{self.sides}-sided polygon" if self.kind == "polygon" else self.kind
        return f"{name} {self.size:g} mm across"


def _star(points, inner, outer):
    """A star's outline: `points` tips at radius `outer` (one pointing up), valleys at `inner`."""
    a = math.pi / 2 + math.pi * np.arange(2 * points) / points
    radius = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def _chevron(width, span):
    """A chevron pointing up: two arms `width` thick (square ends) sloping CHEVRON below the
    horizontal, `span` from tip to tip, centred in the middle of its apex: every point of both arms
    is in sight from there (the centre of its box lies in the notch, outside it)."""
    c, s = math.cos(CHEVRON), math.sin(CHEVRON)
    apex = np.array([0.0, width / (2 * c)])
    left = apex + (span / 2) / c * np.array([-c, -s])
    right = apex + (span / 2) / c * np.array([c, -s])
    inner = apex - [0.0, width / c]
    return np.array([apex, left, left + width * np.array([s, -c]), inner, right + width * np.array([-s, -c]), right])


def simple_polygon_sdf(u, v, P, xp=np):
    """Exact signed distance to any simple polygon P (m, 2) (convex or not; Inigo Quilez's
    crossing-number test for the sign)."""
    d = inside = None
    m = len(P)
    for i in range(m):
        ax, ay, bx, by = float(P[i, 0]), float(P[i, 1]), float(P[i - 1, 0]), float(P[i - 1, 1])
        ex, ey = bx - ax, by - ay
        wx, wy = u - ax, v - ay
        t = xp.clip((wx * ex + wy * ey) / max(ex * ex + ey * ey, 1e-30), 0.0, 1.0)
        dx, dy = wx - ex * t, wy - ey * t
        dd = dx * dx + dy * dy
        c1, c2, c3 = v >= ay, v < by, ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        d = dd if d is None else xp.minimum(d, dd)
        inside = flip if inside is None else inside ^ flip
    return xp.where(inside, -1.0, 1.0) * xp.sqrt(d)


def _apothem(bx, by):
    return bx * by / math.hypot(bx, by)


def _polygon(u, v, n, a, xp):
    """Regular n-gon with inner radius a. Even n: a flat faces +x (a hexagon's neighbours in a
    honeycomb row face its flats); odd n: a flat faces -y (the triangle points up)."""
    offset = 0.0 if n % 2 == 0 else -math.pi / 2
    sector = 2 * math.pi / n
    t = xp.arctan2(v, u) - offset
    t = t - sector * xp.round(t / sector)        # folded onto the nearest flat's normal
    rho = xp.sqrt(u * u + v * v)
    px, py = rho * xp.cos(t), xp.abs(rho * xp.sin(t))
    h = a * math.tan(math.pi / n)                # half the side
    return xp.where(py > h, xp.sqrt((px - a) ** 2 + (py - h) ** 2), px - a)


def _box(u, v, bx, by, xp):
    qx, qy = xp.abs(u) - bx, xp.abs(v) - by
    return xp.sqrt(xp.maximum(qx, 0.0) ** 2 + xp.maximum(qy, 0.0) ** 2) + xp.minimum(xp.maximum(qx, qy), 0.0)


def _rhombus(u, v, bx, by, xp):
    """Exact distance to a rhombus with half diagonals bx (along x) and by (Inigo Quilez)."""
    px, py = xp.abs(u), xp.abs(v)
    h = xp.clip(((bx - 2 * px) * bx - (by - 2 * py) * by) / (bx * bx + by * by), -1.0, 1.0)
    dx, dy = px - 0.5 * bx * (1 - h), py - 0.5 * by * (1 + h)
    return xp.sqrt(dx * dx + dy * dy) * xp.sign(px * by + py * bx - bx * by)


# --- layouts -------------------------------------------------------------------------------------
SPIRAL_GAP = 1.68  # nearest neighbours of the sunflower spiral r = c sqrt(k), angle k * GOLDEN: 1.68 c


def steps(shape, spacing, layout, alternate="none"):
    """(step along x, step between rows) in mm. Spacing is centre to centre; shapes with a Length
    are spaced farther apart along it, so the web between their ends is the same as between their
    sides (spacing - size). Staggered round and hexagonal holes (and Voronoi cells) pack as a
    honeycomb: every neighbour `spacing` away. Every other cell turned 90°: room both ways.
    Triangles on a grid, every other one turned 180°: rows as far apart as makes the web between
    rows the same as between neighbours in a row (a triangle tiling)."""
    sx = sy = float(spacing)
    if shape.kind in LONG and shape.kind != "star":   # a star is as wide every way: Spacing as it is
        ex, ey = shape.extent()
        sx += max(0.0, 2 * ex - shape.size)
        if shape.kind != "chevron":                   # chevrons nest one row inside the next
            sy += max(0.0, 2 * ey - shape.size)
    if layout == "staggered" and shape.kind in HONEYCOMB:
        sy *= math.sqrt(3) / 2
    if alternate == "checker 90":
        sx = sy = max(sx, sy)
    if alternate == "checker 180" and layout == "grid" and shape.polygonal and shape.sides == 3:
        r = shape.size / 2   # web in a row: sx cos 30° - 1.5 r; between rows: sy - 3 r
        sy = sx * math.cos(math.pi / 6) + 1.5 * r
    return sx, sy


def lattice(layout, sx, sy, box, origin=(0.0, 0.0), center=(0.0, 0.0), pad=0.0, randomness=0.0, seed=0,
            wrap=(0, 0)):
    """Centres of a layout covering box = (xmin, ymin, xmax, ymax) grown by `pad` mm (the
    neighbours a Voronoi cell needs), mm: arrays x, y, turn (extra rotation of each hole in
    radians: rings and the spiral turn every hole with its angle around the centre, so a slot stays
    radial) and the integer cell (i, j) of each.

    grid / staggered: x = origin + i * sx (staggered: odd rows moved half a step), y = origin +
    j * sy. rings: a hole at `center`, then rings sx apart with holes about sy apart along them.
    spiral: a sunflower around `center`, neighbours about sx apart.

    randomness 0..1 moves every centre by up to that share of half a step in x and in y, the same
    for the same cell every time (`seed` picks another set); `wrap` = (cells around in x, in y) on
    a face that closes around, so cells i and i + wrap move alike and the pattern still closes."""
    xmin, ymin, xmax, ymax = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    if layout in ("rings", "spiral"):
        cx, cy = center
        corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]) - (cx, cy)
        reach = float(np.hypot(corners[:, 0], corners[:, 1]).max())
        if layout == "spiral":
            c = sx / SPIRAL_GAP
            k = np.arange(int((reach / c) ** 2) + 2)
            k = k[(k == 0) | (k >= 3)]              # the 2nd and 3rd seeds sit too close to the 1st
            if len(k) > 4_000_000:
                raise ValueError(f"about {len(k):,} cells: the spacing is far too small for this face")
            t = k * GOLDEN
            x, y = cx + c * np.sqrt(k) * np.cos(t), cy + c * np.sqrt(k) * np.sin(t)
            i, j = k, np.zeros(len(k), dtype=np.int64)
        else:
            rings = int(np.ceil(reach / sx)) + 1
            xs, ys, turn, ii, jj = [np.array([cx])], [np.array([cy])], [np.zeros(1)], [np.zeros(1, int)], [np.zeros(1, int)]
            for k in range(1, rings + 1):
                count = max(1, int(round(2 * np.pi * k * sx / sy)))
                a = 2 * np.pi * np.arange(count) / count
                xs.append(cx + k * sx * np.cos(a))
                ys.append(cy + k * sx * np.sin(a))
                turn.append(a)
                ii.append(np.full(count, k))
                jj.append(np.arange(count))
            x, y, t = np.concatenate(xs), np.concatenate(ys), np.concatenate(turn)
            i, j = np.concatenate(ii), np.concatenate(jj)
        if randomness > 0:
            x, y = jitter(layout, x, y, i, j, randomness, sx, sy, seed)
        keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        return x[keep], y[keep], t[keep], i[keep], j[keep]
    ox, oy = origin
    j = np.arange(int(np.floor((ymin - oy) / sy)) - 1, int(np.ceil((ymax - oy) / sy)) + 2)
    i = np.arange(int(np.floor((xmin - ox) / sx)) - 2, int(np.ceil((xmax - ox) / sx)) + 3)
    if len(i) * len(j) > 4_000_000:
        raise ValueError(f"about {len(i) * len(j):,} cells: the spacing is far too small for this face")
    i, j = np.meshgrid(i, j, indexing="ij")
    i, j = i.ravel(), j.ravel()
    shift = 0.5 * (j % 2) if layout == "staggered" else 0.0
    x, y = ox + (i + shift) * sx, oy + j * sy
    if randomness > 0:
        x, y = jitter(layout, x, y, i, j, randomness, sx, sy, seed, wrap)
    keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return x[keep], y[keep], np.zeros(int(keep.sum())), i[keep], j[keep]


def jitter(layout, x, y, i, j, amount, sx, sy, seed=0, wrap=(0, 0)):
    """Centres of cells (i, j) moved at random by up to `amount` (0..1: a number, or one per
    centre) of half a step in x and in y: the same for the same cell every time; with `wrap`,
    cells i and i + wrap (j and j + wrap) move alike. Rings and the spiral use sx both ways."""
    if layout in ("rings", "spiral"):
        wi, wj, ax, ay = i, j, sx, sx
    else:
        wi = i % wrap[0] if wrap[0] else i
        wj = j % wrap[1] if wrap[1] else j
        ax, ay = sx, sy
    return (x + (_unit_random(wi, wj, seed, 0) - 0.5) * amount * ax,
            y + (_unit_random(wi, wj, seed, 1) - 0.5) * amount * ay)


def _unit_random(i, j, seed, salt):
    """A number 0..1 for every integer cell (i, j): the same each time (splitmix64 of i, j, seed)."""
    with np.errstate(over="ignore"):
        x = (np.asarray(i, dtype=np.int64).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ^ np.asarray(j, dtype=np.int64).astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
             ^ np.uint64((int(seed) * 4 + salt) % 2**63) * np.uint64(0x165667B19E3779F9)
             ^ np.uint64(0x632BE59BD9B4E019))
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / 2.0 ** 53


# --- Voronoi cells -------------------------------------------------------------------------------
def voronoi_cells(x, y, inset, bound):
    """Every point's Voronoi cell among the points (x, y), moved in by `inset` mm on every side, as
    vertices around the point (counter-clockwise, relative to it): (n, m, 2), shorter cells padded
    by repeating their last vertex; cells that vanish are NaN. Cells at the edge of the point set
    are capped at `bound` mm from their point."""
    pts = np.column_stack([x, y]).astype(np.float64)
    n = len(pts)
    neighbours = [[] for _ in range(n)]
    try:
        from scipy.spatial import Voronoi
        for a, b in Voronoi(pts).ridge_points:
            neighbours[a].append(b)
            neighbours[b].append(a)
    except Exception:  # too few points, or all in a row: every other point within reach is a neighbour
        from scipy.spatial import cKDTree
        neighbours = cKDTree(pts).query_ball_point(pts, 2.5 * bound)
    square = bound * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    cells = []
    for a in range(n):
        poly = square
        for b in neighbours[a]:
            d = pts[b] - pts[a]
            length = math.hypot(d[0], d[1])
            if b == a or length < 1e-12:
                continue
            poly = _clip(poly, d / length, length / 2 - inset)
            if not len(poly):
                break
        cells.append(poly)
    m = max([len(c) for c in cells] + [3])
    out = np.full((n, m, 2), np.nan)
    for a, c in enumerate(cells):
        if len(c) >= 3:
            out[a, :len(c)] = c
            out[a, len(c):] = c[-1]
    return out


def delaunay_cells(x, y, inset):
    """The Delaunay triangles of the points (x, y), each moved in by `inset` mm on every side (a
    triangle shrinks about the centre of its inscribed circle): (centre x, centre y, cells), every
    cell counter-clockwise around its own centroid, (n, 3, 2); cells that vanish are NaN."""
    from scipy.spatial import Delaunay
    pts = np.column_stack([x, y]).astype(np.float64)
    try:
        tri = pts[Delaunay(pts).simplices]                 # (n, 3, 2)
    except Exception:  # too few points, or all in a row
        return np.zeros(0), np.zeros(0), np.zeros((0, 3, 2))
    e = np.roll(tri, -1, axis=1) - tri
    turn = e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]
    tri[turn < 0] = tri[turn < 0, ::-1]                    # counter-clockwise
    side = np.linalg.norm(np.roll(tri, -1, axis=1) - tri, axis=2)          # side k: corner k to k + 1
    facing = np.roll(side, -1, axis=1)                     # the side facing each corner
    perimeter = side.sum(axis=1)
    incentre = (facing[..., None] * tri).sum(axis=1) / perimeter[:, None]
    inradius = np.abs(turn) / perimeter                    # area / half the perimeter
    k = (inradius - inset) / np.maximum(inradius, 1e-12)
    cells = incentre[:, None] + (tri - incentre[:, None]) * np.where(k > 0, k, np.nan)[:, None, None]
    middle = cells.mean(axis=1)
    return middle[:, 0], middle[:, 1], cells - middle[:, None]


def centroids(cells):
    """Area centroid of every polygon (n, m, 2) (padding vertices add nothing); NaN cells stay NaN."""
    x, y = cells[..., 0], cells[..., 1]
    x1, y1 = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * y1 - x1 * y
    area = cross.sum(axis=1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = ((x + x1) * cross).sum(axis=1) / (6 * area)
        cy = ((y + y1) * cross).sum(axis=1) / (6 * area)
    tiny = ~(np.abs(area) > 1e-12)
    cx[tiny], cy[tiny] = x[tiny].mean(axis=1), y[tiny].mean(axis=1)
    return np.column_stack([cx, cy])


def _clip(poly, normal, offset):
    """The part of convex polygon `poly` (k, 2) where normal . q <= offset."""
    s = poly @ normal - offset
    if np.all(s <= 0):
        return poly
    if np.all(s > 0):
        return poly[:0]
    out = []
    for k in range(len(poly)):
        a, b, sa, sb = poly[k], poly[(k + 1) % len(poly)], s[k], s[(k + 1) % len(poly)]
        if sa <= 0:
            out.append(a)
        if (sa <= 0) != (sb <= 0):
            out.append(a + (b - a) * (sa / (sa - sb)))
    return np.array(out)


def polygon_sdf(u, v, P, xp=np):
    """Exact signed distance to a convex polygon: P (m, 2) counter-clockwise, or one per point
    (N, m, 2) for points u, v (N,). Repeated vertices (padding) are harmless."""
    m = P.shape[-2]
    best = inside = None
    for k in range(m):
        ax, ay, bx, by = P[..., k, 0], P[..., k, 1], P[..., (k + 1) % m, 0], P[..., (k + 1) % m, 1]
        ex, ey = bx - ax, by - ay
        wx, wy = u - ax, v - ay
        t = xp.clip((wx * ex + wy * ey) / xp.maximum(ex * ex + ey * ey, 1e-30), 0.0, 1.0)
        dx, dy = wx - t * ex, wy - t * ey
        d = dx * dx + dy * dy
        side = ex * wy - ey * wx >= 0
        best = d if best is None else xp.minimum(best, d)
        inside = side if inside is None else inside & side
    dist = xp.sqrt(best)
    return xp.where(inside, -dist, dist)
