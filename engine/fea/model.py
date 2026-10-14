"""The finite element model of a part: its grid of cube elements, how full each element is, and
the supports and loads taken from CAD faces."""
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from ..backend import to_host
from ..fields import Context
from ..grid import Grid
from .element import CORNERS

# detail -> (elements inside the part, most elements in the grid's box)
DETAIL = {"draft": (40_000, 400_000), "medium": (250_000, 2_500_000), "fine": (1_000_000, 8_000_000)}
PHI_MIN = 0.02            # elements filled less than this are left out
MAX_COARSE = 4000         # elements on the coarsest multigrid level (solved directly)
MAX_PADDING = 1.15        # an axis is padded by at most this much to allow another multigrid level
MAX_SAMPLES = 4_000_000   # points sampled on the faces of one support or load
FIXED_NEAR = 0.2          # a support fixes the corners whose trilinear weight is at least this per axis

DIRECTIONS = ("into the face", "out of the face", "+X", "-X", "+Y", "-Y", "+Z", "-Z", "custom")
LOAD_KINDS = ("force", "pressure")


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


def plan_coarsenings(shape):
    """(coarsenings per axis, padded element counts): halve the axes until the coarsest grid is
    small, padding an axis by at most MAX_PADDING for that."""
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
        if coarse <= MAX_COARSE:
            break
    return tuple(best[0]), tuple(best[1])


def make_grid(bounds, h):
    """Elements from the part's lowest corner on: faces at the low end of its box (a base, a flat
    end) then coincide with element faces, and are exact rather than half-filled elements."""
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    n = [max(1, int(math.ceil((hi[a] - lo[a]) / h - 1e-9))) for a in range(3)]
    ks, padded = plan_coarsenings(n)
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
    load: str = "force"             # force (N, total) or pressure (MPa)
    amount: float = 0.0
    direction: str = "into the face"
    vector: tuple = (0.0, 0.0, -1.0)

    @property
    def key(self):
        tri = np.ascontiguousarray(self.triangles, dtype=np.float64)
        return _digest(self.kind, hashlib.sha1(tri.tobytes()).hexdigest(), self.load, float(self.amount),
                       self.direction, tuple(float(v) for v in self.vector))

    @property
    def area(self):
        t = self.triangles
        return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

    def traction(self, normals):
        """Force per mm^2 at points with these outward normals (n, 3)."""
        if self.kind != "load":
            return np.zeros_like(normals)
        if self.load == "pressure":
            return -float(self.amount) * normals  # a positive pressure pushes into the face
        per_area = float(self.amount) / max(self.area, 1e-12)
        if self.direction == "into the face":
            return -per_area * normals
        if self.direction == "out of the face":
            return per_area * normals
        return np.broadcast_to(per_area * direction_vector(self.direction, self.vector), normals.shape).copy()


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


def sample_triangles(triangles, spacing):
    """Points on triangles about `spacing` apart: (points, area each, outward normal each)."""
    t = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    cross = np.cross(b - a, c - a)
    double = np.linalg.norm(cross, axis=1)
    keep = double > 1e-12
    a, b, c, cross, double = a[keep], b[keep], c[keep], cross[keep], double[keep]
    if not len(a):
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    normal = cross / double[:, None]
    longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=0)
    m = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
    total = int((m * m).sum())
    if total > MAX_SAMPLES:
        m = np.maximum(1, np.ceil(m * math.sqrt(MAX_SAMPLES / total))).astype(np.int64)
    points, areas, normals = [], [], []
    for mm in np.unique(m):
        sel = m == mm
        bary = _subdivision(int(mm))
        p = a[sel, None] + bary[None, :, :1] * (b - a)[sel, None] + bary[None, :, 1:] * (c - a)[sel, None]
        points.append(p.reshape(-1, 3))
        areas.append(np.repeat(0.5 * double[sel] / (mm * mm), mm * mm))
        normals.append(np.repeat(normal[sel], mm * mm, axis=0))
    return np.concatenate(points), np.concatenate(areas), np.concatenate(normals)


def markers(condition, count=24):
    """A few well-spread points on the condition's faces and the direction to draw there (the load's
    direction; into the part for supports), for the 3D view."""
    area = condition.area
    if area <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    cell = math.sqrt(area / count)
    pts, _, normals = sample_triangles(condition.triangles, cell / 3)
    keys = np.floor(pts / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    if len(first) > count:
        first = np.sort(np.random.default_rng(0).choice(first, count, replace=False))
    pts, normals = pts[first], normals[first]
    if condition.kind == "fixed":
        dirs = -normals
    else:
        dirs = condition.traction(normals)
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
    act = active_nodes(phi)
    fixed = np.zeros(nodes + (3,), dtype=bool)
    force = np.zeros((int(np.prod(nodes)), 3))
    notes = []
    for cond in conditions:
        pts, areas, normals = sample_triangles(cond.triangles, grid.h / 3)
        if not len(pts):
            raise ConditionError(f"{cond.label or cond.kind}: no faces chosen")
        rel = (pts - grid.origin) / grid.h
        e = np.clip(np.floor(rel).astype(np.int64), 0, np.asarray(grid.shape) - 1)
        t = np.clip(rel - e, 0.0, 1.0)
        corners = e[:, None, :] + CORNERS[None]                                # (s, 8, 3)
        w = np.prod(np.where(CORNERS[None] == 1, t[:, None, :], 1 - t[:, None, :]), axis=2)
        on = act[corners[..., 0], corners[..., 1], corners[..., 2]]           # (s, 8)
        on &= np.all((rel >= -0.5) & (rel <= np.asarray(grid.shape) + 0.5), axis=1)[:, None]  # off the grid
        hit = on.any(axis=1)
        share = float(areas[hit].sum() / areas.sum())
        if not hit.any():
            raise ConditionError(f"{cond.label or cond.kind}: its faces are not on the part (no material there)")
        if share < 0.9:
            notes.append(f"{cond.label or cond.kind}: {100 * (1 - share):.0f}% of its faces lie off the part")
        if cond.kind == "fixed":
            near = np.all(np.where(CORNERS[None] == 1, t[:, None, :], 1 - t[:, None, :]) >= FIXED_NEAR, axis=2)
            c = corners[on & near]
            fixed[c[:, 0], c[:, 1], c[:, 2]] = True
            continue
        ws = w * on
        total = ws.sum(axis=1)
        ok = total > 0
        ws, total, corners = ws[ok], total[ok], corners[ok]
        ws /= total[:, None]
        keep_full = areas.sum() / areas[ok].sum()   # the share off the part is put on the rest
        f = cond.traction(normals[ok]) * areas[ok, None] * keep_full
        flat = np.ravel_multi_index((corners[..., 0], corners[..., 1], corners[..., 2]), nodes).ravel()
        for d in range(3):
            force[:, d] += np.bincount(flat, weights=(ws * f[:, None, d]).ravel(), minlength=len(force))
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
