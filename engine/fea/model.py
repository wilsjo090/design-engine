"""The finite element model of a part: its grid of cube elements, how full each element is, and
the supports and loads taken from CAD faces."""
import hashlib
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..backend import to_host
from ..fields import Context
from ..grid import Grid
from .element import CORNERS

# detail -> (elements inside the part, most elements in the grid's box)
DETAIL = {"draft": (40_000, 400_000), "medium": (250_000, 2_500_000), "fine": (1_000_000, 8_000_000)}
PHI_MIN = 0.02            # elements filled less than this are left out
# elements on the coarsest multigrid level (factorized directly, and again every topology
# iteration): 3000 elements (11k unknowns) took 1.0 s to factorize, 375 took 0.012 s. A static
# analysis factorizes once, and a bigger coarsest level saves iterations (bracket, medium: 33 -> 24).
MAX_COARSE = 500
MAX_COARSE_STATIC = 3000
MAX_PADDING = 1.15        # an axis is padded by at most this much to allow another multigrid level
MAX_SAMPLES = 4_000_000   # points sampled on the faces of one support or load
# Points per spacing across a long thin triangle (sample_triangles): with 1, the stress next to a
# loaded hole came out up to 0.9% off a 8x finer sampling (the split into m x m, dense across such
# triangles, 0.12%); with 4, 0.12%
ACROSS = 4
FIXED_NEAR = 0.2          # a support fixes the corners whose trilinear weight is at least this per axis

DIRECTIONS = ("into the face", "out of the face", "+X", "-X", "+Y", "-Y", "+Z", "-Z", "custom")
LOAD_KINDS = ("force", "pressure", "bearing", "torque")
OFFSET_KINDS = ("force", "bearing")   # loads acting at a point, which Offset moves off the faces


class ConditionError(ValueError):
    pass


def _digest(*parts):
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:20]


# --- the grid ------------------------------------------------------------------------------------
@dataclass
class FEGrid:
    origin: np.ndarray       # position of node (0, 0, 0)
    h: float                 # element size, mm
    shape: tuple             # elements per axis
    coarsenings: tuple       # multigrid: how often each axis is halved

    @property
    def nodes(self):
        return tuple(n + 1 for n in self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def centres(self):
        """The grid of element centres (for sampling fields per element)."""
        return Grid.block(self.origin + self.h / 2, self.h, (0, 0, 0), self.shape)

    def describe(self):
        return f"{self.shape[0]}x{self.shape[1]}x{self.shape[2]} elements of {self.h:g} mm"


def tidy(h):
    """Round an element size up to two significant digits (1.37 -> 1.4)."""
    step = 10 ** (math.floor(math.log10(h)) - 1)
    return round(math.ceil(h / step - 1e-9) * step, 10)


def element_size(bounds, volume, detail):
    if detail not in DETAIL:
        raise ValueError(f"detail must be one of {', '.join(DETAIL)}")
    target, box_cap = DETAIL[detail]
    extent = np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float)
    h = (max(volume, 1e-9) / target) ** (1 / 3)
    for _ in range(20):  # the box cap: thin or hollow parts have far more grid than part
        if np.prod(np.ceil(extent / h) + 1) <= box_cap:
            break
        h *= 1.05
    return tidy(h)


def plan_coarsenings(shape, max_coarse=MAX_COARSE):
    """(coarsenings per axis, padded element counts): halve the axes until the coarsest grid has
    at most `max_coarse` elements, padding an axis by at most MAX_PADDING for that."""
    n = [int(v) for v in shape]
    best = None
    for depth in range(0, 12):
        ks, padded = [], []
        for ni in n:
            k = 0
            for cand in range(min(depth, int(math.log2(max(ni, 1)))), 0, -1):
                p = -(-ni // 2 ** cand) * 2 ** cand
                if p <= MAX_PADDING * ni:
                    k = cand
                    break
            ks.append(k)
            padded.append(-(-ni // 2 ** k) * 2 ** k)
        coarse = int(np.prod([p // 2 ** k for p, k in zip(padded, ks)]))
        if best is not None and ks == best[0]:
            break  # no axis can be halved any more
        best = (ks, padded)
        if coarse <= max_coarse:
            break
    return tuple(best[0]), tuple(best[1])


def make_grid(bounds, h, max_coarse=MAX_COARSE):
    """Elements from the part's lowest corner on: faces at the low end of its box (a base, a flat
    end) then coincide with element faces, and are exact rather than half-filled elements."""
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    n = [max(1, int(math.ceil((hi[a] - lo[a]) / h - 1e-9))) for a in range(3)]
    ks, padded = plan_coarsenings(n, max_coarse)
    return FEGrid(lo.copy(), float(h), padded, ks)


def estimate_volume(part, bounds, cells=64):
    """A part's volume from its distance field sampled on a coarse grid (mm^3)."""
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    v = max(float((hi - lo).max()) / cells, 1e-3)
    grid = Grid(lo, hi, v)
    d = to_host(Context(grid, None, np).eval_full(part)).astype(np.float64)
    return float(np.clip(0.5 - d / v, 0, 1).sum() * v ** 3)


def element_fill(part, grid, xp=np, cache=None):
    """How full each element is (0..1), from the part's signed distance at its centre: exact for a
    flat surface parallel to the element's faces, close otherwise."""
    d = to_host(Context(grid.centres(), cache, xp).eval_full(part)).astype(np.float64)
    phi = np.clip(0.5 - d / grid.h, 0.0, 1.0)
    phi[phi < PHI_MIN] = 0.0
    return phi


def active_nodes(phi):
    """Nodes touching an element with material."""
    nx, ny, nz = phi.shape
    act = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    filled = phi > 0
    for cx, cy, cz in CORNERS:
        act[cx:cx + nx, cy:cy + ny, cz:cz + nz] |= filled
    return act


# --- supports and loads --------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """A support or a load on a set of CAD triangles ((k, 3, 3) corner positions, outward order)."""
    kind: str                       # "fixed" or "load"
    triangles: np.ndarray = field(repr=False)
    label: str = ""
    load: str = "force"             # force (N, total), pressure (MPa), bearing (N) or torque (N m)
    amount: float = 0.0
    direction: str = "into the face"
    vector: tuple = (0.0, 0.0, -1.0)
    offset: tuple = (0.0, 0.0, 0.0)  # force and bearing: where the force acts, mm from the faces' centre

    @property
    def key(self):
        tri = np.ascontiguousarray(self.triangles, dtype=np.float64)
        parts = (self.kind, hashlib.sha1(tri.tobytes()).hexdigest(), self.load, float(self.amount),
                 self.direction, tuple(float(v) for v in self.vector))
        if self.offset_moment() is not None:  # (loads without an offset keep their keys)
            parts += (tuple(float(v) for v in self.offset),)
        return _digest(*parts)

    @property
    def area(self):
        t = self.triangles
        return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

    @cached_property
    def geometry(self):
        """(area, centre, inertia tensor of the area about the centre, area-weighted mean normal) of
        the faces, exact for flat triangles (int r r^T dA = A/12 (sum v v^T + s s^T) per triangle)."""
        t = np.asarray(self.triangles, dtype=np.float64).reshape(-1, 3, 3)
        cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        a = 0.5 * np.linalg.norm(cross, axis=1)
        area = float(a.sum())
        if area <= 0:
            return 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros(3)
        centre = (a[:, None] * t.mean(axis=1)).sum(axis=0) / area
        v = t - centre
        s = v.sum(axis=1)
        second = (np.einsum("k,kci,kcj->ij", a, v, v) + np.einsum("k,ki,kj->ij", a, s, s)) / 12
        inertia = np.trace(second) * np.eye(3) - second
        return area, centre, inertia, 0.5 * cross.sum(axis=0) / area

    def traction(self, normals, points=None):
        """Force per mm^2 at points on the faces with these outward normals (n, 3). A torque and a
        force with an offset also need the points."""
        if self.kind != "load":
            return np.zeros_like(normals)
        if self.load == "pressure":
            return -float(self.amount) * normals  # a positive pressure pushes into the face
        if self.load == "torque":
            return self._turning(points, 1000.0 * float(self.amount) * self.torque_axis())  # N m -> N mm
        if self.load == "bearing":
            # a pin pressing on the side of a hole the force points to: pressure square to the face,
            # largest where the face looks straight against the force, falling as the cosine to 0 at
            # the sides, none on the far side (as SolidWorks' bearing load); total = the force
            d = self.bearing_direction()
            w = np.maximum(-(normals @ d), 0.0)
            t = (float(self.amount) / self._bearing_norm(d)) * w[:, None] * -normals
        else:
            per_area = float(self.amount) / max(self.area, 1e-12)
            if self.direction == "into the face":
                t = -per_area * normals
            elif self.direction == "out of the face":
                t = per_area * normals
            else:
                t = np.broadcast_to(per_area * direction_vector(self.direction, self.vector), normals.shape).copy()
        moment = self.offset_moment()
        return t if moment is None else t + self._turning(points, moment)

    def resultant(self):
        """The load's total force (N), as a vector."""
        if self.kind != "load" or self.load == "torque":
            return np.zeros(3)
        if self.load == "pressure":
            area, _, _, mean_normal = self.geometry
            return -float(self.amount) * area * mean_normal
        if self.load == "bearing" or self.direction not in ("into the face", "out of the face"):
            return float(self.amount) * direction_vector(self.direction, self.vector)
        sign = -1.0 if self.direction == "into the face" else 1.0
        return sign * float(self.amount) * self.geometry[3]

    def offset_moment(self):
        """The moment (N mm, about the faces' centre) of a force acting at Offset, or None."""
        if self.kind != "load" or self.load not in OFFSET_KINDS or not any(self.offset):
            return None
        return np.cross(np.asarray(self.offset, dtype=float), self.resultant())

    def torque_axis(self):
        """The axis a torque turns about (right-hand rule). into the face on a flat face: the way a
        screw is driven into it (clockwise seen from outside); out of the face: loosening it."""
        if self.direction not in ("into the face", "out of the face"):
            return direction_vector(self.direction, self.vector)
        n = self.geometry[3]
        if np.linalg.norm(n) < 0.5:
            raise ConditionError("a torque on a round face (a hole, a shaft) or on faces facing several ways needs "
                                 "the axis it turns about, e.g. +Y (or drag the yellow arrow): into the face only "
                                 "works on a flat face")
        n = n / np.linalg.norm(n)
        return -n if self.direction == "into the face" else n

    def arrow(self):
        """The direction to draw the load's big arrow: its force, or a torque's axis (None for none)."""
        if self.kind != "load" or self.load == "pressure":
            return None
        v = self.torque_axis() if self.load == "torque" else self.resultant()
        size = np.linalg.norm(v)
        return v / size if size > 1e-12 else None

    def _turning(self, points, moment):
        """Traction adding up to no force and to `moment` (N mm) about the faces' centre: a turn
        about the centre that grows with the distance from it, as through a rigid connection (a
        shaft's torque on its hole, bolts on a circle; SolidWorks' rigid remote load). For the turn
        w: moment = sum of r x (w x r) dA = inertia w."""
        if points is None:
            raise ValueError("this load needs the points as well as the normals")
        area, centre, inertia, _ = self.geometry
        if area <= 0:
            return np.zeros((len(points), 3))
        spin = np.linalg.lstsq(inertia, np.asarray(moment, dtype=float), rcond=1e-9)[0]
        return np.cross(spin, np.asarray(points, dtype=float) - centre)

    def check(self):
        """Raise ConditionError now for settings this load cannot use (rather than in the analysis)."""
        if self.kind == "load" and len(self.triangles):
            self.traction(np.zeros((0, 3)), np.zeros((0, 3)))
            self.arrow()

    def presses(self, normals):
        """Where the load acts on its own (not counting an offset's moment): a bearing load only on
        the contact side, the others everywhere."""
        if self.kind == "load" and self.load == "bearing":
            return -(normals @ self.bearing_direction()) > 0
        return np.ones(len(normals), dtype=bool)

    def bearing_direction(self):
        if self.direction in ("into the face", "out of the face"):
            raise ConditionError("a bearing load needs a direction (e.g. -Z): the side of the hole the pin "
                                 "presses on")
        return direction_vector(self.direction, self.vector)

    def _bearing_norm(self, d):
        """sum over the faces of area x w^2 (w = cos of the angle against the force): the pressure
        p0 w along -n then adds up to p0 x this along the force."""
        t = self.triangles
        cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        double = np.linalg.norm(cross, axis=1)
        n = cross / np.maximum(double, 1e-300)[:, None]
        w = np.maximum(-(n @ d), 0.0)
        total = float((0.5 * double * w * w).sum())
        if total <= 0:
            raise ConditionError("bearing load: no part of the faces faces against the force (check the "
                                 "direction: a pin pulled down presses on the bottom of the hole: -Z)")
        return total


def direction_vector(direction, vector=(0, 0, -1)):
    axes = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}
    if direction in axes:
        return np.array(axes[direction], dtype=float)
    if direction != "custom":
        raise ConditionError(f"direction must be one of: {', '.join(DIRECTIONS)}")
    v = np.asarray(vector, dtype=float)
    n = np.linalg.norm(v)
    if v.shape != (3,) or n < 1e-12:
        raise ConditionError("the custom direction needs three numbers that are not all 0, e.g. 0, 0, -1")
    return v / n


def parse_vector(text):
    parts = [p for p in str(text or "").replace(";", " ").replace(",", " ").split() if p]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        values = []
    if len(values) != 3:
        raise ConditionError(f"the vector must be three numbers like 0, 0, -1 (got '{text}')")
    return tuple(values)


class Conditions(tuple):
    """All supports and loads of an analysis (a node input takes one link, so they are chained)."""

    @property
    def key(self):
        return _digest("conditions", tuple(c.key for c in self))

    def add(self, condition):
        return Conditions(tuple(self) + (condition,))


def _subdivision(m):
    """Barycentric centres of the m^2 equal triangles a triangle splits into, m per edge."""
    i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    up = (i + j) <= m - 1
    down = (i + j) <= m - 2
    a = np.concatenate([(i[up] + 1 / 3) / m, (i[down] + 2 / 3) / m])
    b = np.concatenate([(j[up] + 1 / 3) / m, (j[down] + 2 / 3) / m])
    return np.stack([a, b], axis=1)


def _stretched(m, n):
    """A grid stretched over a triangle: m rows from its apex to the opposite edge, n points along
    each, as (s1, s2, share of the area) with point = apex + s1 (p1 - apex + s2 (p2 - p1)). Rows sit
    at their area centroid, so linear functions (a torque's turn) add up exactly, as on _subdivision."""
    i = np.arange(m)
    s1 = 2 * (3 * i * i + 3 * i + 1) / (3 * m * (2 * i + 1))
    share = (2 * i + 1) / (m * m * n)
    return np.repeat(s1, n), np.tile((np.arange(n) + 0.5) / n, m), np.repeat(share, n)


def sample_triangles(triangles, spacing, stretch=True):
    """Points on triangles about `spacing` apart: (points, area each, outward normal each).

    A triangle is split into m x m equal ones, m from its longest edge. Long thin CAD triangles (the
    strips along a hole) would get m^2 points across a width that needs a few: with `stretch`, those
    where that is more than twice what a grid stretched to the triangle needs (m rows from the
    corner opposite the shortest edge, ACROSS points per spacing along each) get that grid. The wall
    bracket's pin hole at medium: 2.0M points -> 0.12M. Halving the long edges instead (a remesh that
    keeps the faces) leaves thin pieces: 0.36M with one point each. Fine for sums over the faces
    (loads); tests of what lies near a face (supports, topology's skin) keep stretch=False: their
    answer depends on how densely the faces are covered (bracket, fine: 370 fewer fixed nodes and 7%
    less stress away from the supports with stretched grids)."""
    t = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    cross = np.cross(b - a, c - a)
    double = np.linalg.norm(cross, axis=1)
    keep = double > 1e-12
    a, b, c, cross, double = a[keep], b[keep], c[keep], cross[keep], double[keep]
    if not len(a):
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    normal = cross / double[:, None]
    edges = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=1)
    m = np.maximum(1, np.ceil(edges.max(axis=1) / spacing)).astype(np.int64)
    n = np.maximum(1, np.ceil(ACROSS * edges.min(axis=1) / spacing)).astype(np.int64)
    thin = (2 * n <= m) & stretch
    total = int(np.where(thin, m * n, m * m).sum())
    if total > MAX_SAMPLES:
        f = math.sqrt(MAX_SAMPLES / total)
        m = np.maximum(1, np.ceil(m * f)).astype(np.int64)
        n = np.maximum(1, np.ceil(n * f)).astype(np.int64)
    points, areas, normals = [], [], []
    for mm in np.unique(m[~thin]):
        sel = (m == mm) & ~thin
        bary = _subdivision(int(mm))
        p = a[sel, None] + bary[None, :, :1] * (b - a)[sel, None] + bary[None, :, 1:] * (c - a)[sel, None]
        points.append(p.reshape(-1, 3))
        areas.append(np.repeat(0.5 * double[sel] / (mm * mm), mm * mm))
        normals.append(np.repeat(normal[sel], mm * mm, axis=0))
    rows = np.flatnonzero(thin)
    if len(rows):
        corners = np.stack([a[rows], b[rows], c[rows]], axis=1)
        apex = (edges[rows].argmin(axis=1) + 2) % 3        # edge k joins corners k and k + 1
        k = np.arange(len(rows))
        apex, p1, p2 = corners[k, apex], corners[k, (apex + 1) % 3], corners[k, (apex + 2) % 3]
        mt, nt = m[rows], n[rows]
        for mm, nn in np.unique(np.stack([mt, nt], axis=1), axis=0):
            sel = (mt == mm) & (nt == nn)
            s1, s2, share = _stretched(int(mm), int(nn))
            p = apex[sel, None] + s1[None, :, None] * ((p1 - apex)[sel, None] + s2[None, :, None] * (p2 - p1)[sel, None])
            points.append(p.reshape(-1, 3))
            areas.append((0.5 * double[rows][sel, None] * share).ravel())
            normals.append(np.repeat(normal[rows][sel], mm * nn, axis=0))
    return np.concatenate(points), np.concatenate(areas), np.concatenate(normals)


def markers(condition, count=24):
    """A few well-spread points on the condition's faces and the direction to draw there (the load's
    direction; into the part for supports), for the 3D view."""
    area = condition.area
    if area <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    cell = math.sqrt(area / count)
    pts, _, normals = sample_triangles(condition.triangles, cell / 3)
    if condition.kind == "load":  # only where the load acts (a bearing load: the contact side)
        size = np.linalg.norm(condition.traction(normals, pts), axis=1)
        on = size > 0.05 * size.max() if size.max() > 0 else np.ones(len(size), dtype=bool)
        pts, normals = pts[on], normals[on]
        cell = math.sqrt(area * on.mean() / count) if on.any() else cell
    keys = np.floor(pts / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    if len(first) > count:
        first = np.sort(np.random.default_rng(0).choice(first, count, replace=False))
    pts, normals = pts[first], normals[first]
    if condition.kind == "fixed":
        dirs = -normals
    else:
        dirs = condition.traction(normals, pts)
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = np.where(norm > 0, dirs / np.maximum(norm, 1e-300), -normals)
    return pts, dirs


def apply_conditions(grid, phi, conditions):
    """Fixed DOFs (X, Y, Z, 3) bool, nodal forces (X, Y, Z, 3) in N, and notes for the report.

    Points sampled on the chosen faces (h/3 apart) are located in the grid. Supports: the nodes of
    the element around a point that are near it (within 0.8 h along each axis; on a face lying on
    element faces, only that layer of nodes) and have material are fixed in x, y and z. Loads: each point's force
    (area x traction) goes to the corners of its element with trilinear weights, shared among the
    corners that have material; the total is scaled back to the full amount when part of the faces
    lies off the part."""
    nodes = grid.nodes
    act = active_nodes(phi).ravel()
    fixed = np.zeros(nodes + (3,), dtype=bool)
    force = np.zeros((int(np.prod(nodes)), 3))
    notes = []
    # node numbers and trilinear weights of a point's element corners, per axis: (s, 8, 3) arrays
    # took 0.7 of 1.2 s for the 1.2M points of a fine analysis
    cx, cy, cz = CORNERS[:, 0], CORNERS[:, 1], CORNERS[:, 2]
    offsets = (cx * nodes[1] + cy) * nodes[2] + cz
    for cond in conditions:
        pts, areas, normals = sample_triangles(cond.triangles, grid.h / 3, stretch=cond.kind != "fixed")
        if not len(pts):
            raise ConditionError(f"{cond.label or cond.kind}: no faces chosen")
        rel = (pts - grid.origin) / grid.h
        e = np.clip(np.floor(rel).astype(np.int64), 0, np.asarray(grid.shape) - 1)
        t = np.clip(rel - e, 0.0, 1.0)
        ends = np.stack([1 - t, t], axis=1)                                   # (s, 2, 3): weights at each end
        flat = ((e[:, 0] * nodes[1] + e[:, 1]) * nodes[2] + e[:, 2])[:, None] + offsets   # (s, 8) corners
        on = act[flat]
        on &= np.all((rel >= -0.5) & (rel <= np.asarray(grid.shape) + 0.5), axis=1)[:, None]  # off the grid
        hit = on.any(axis=1)
        share = float(areas[hit].sum() / areas.sum())
        if not hit.any():
            raise ConditionError(f"{cond.label or cond.kind}: its faces are not on the part (no material there)")
        if share < 0.9:
            notes.append(f"{cond.label or cond.kind}: {100 * (1 - share):.0f}% of its faces lie off the part")
        if cond.kind == "fixed":
            close = ends >= FIXED_NEAR
            near = close[:, cx, 0] & close[:, cy, 1] & close[:, cz, 2]
            fixed.reshape(-1, 3)[flat[on & near]] = True
            continue
        w = ends[:, cx, 0] * ends[:, cy, 1] * ends[:, cz, 2]
        ws = w * on
        total = ws.sum(axis=1)
        ok = total > 0
        ws, total, flat = ws[ok], total[ok], flat[ok]
        ws /= total[:, None]
        keep_full = areas.sum() / areas[ok].sum()   # the share off the part is put on the rest
        f = cond.traction(normals[ok], pts[ok]) * areas[ok, None] * keep_full
        moment = cond.offset_moment()
        if moment is not None:
            notes.append(f"{cond.label or 'load'}: acts {np.linalg.norm(cond.offset):.3g} mm from the centre of its "
                         f"faces, so they also carry a moment of {np.linalg.norm(moment) / 1000:.3g} N·m")
        flat = flat.ravel()
        lo, hi = int(flat.min()), int(flat.max())
        for d in range(3):
            force[lo:hi + 1, d] += np.bincount(flat - lo, weights=(ws * f[:, None, d]).ravel(), minlength=hi - lo + 1)
    force = force.reshape(nodes + (3,))
    if not fixed.any():
        raise ConditionError("the part is not held anywhere: add Fixed Faces")
    if not np.abs(force).max() > 0:
        raise ConditionError("there is no load: add a Load (with an amount other than 0)")
    lost = force[fixed].reshape(-1)
    if np.abs(lost).sum() > 0.5 * np.abs(force).sum():
        notes.append("most of the load sits on fixed faces and goes straight into the support")
    force[fixed] = 0.0
    return fixed, force, notes
