"""Pipes: a tube along a path through points (node Pipe, `implicit.pipe`): a channel to cut through a
part (a cable, cooling, a hidden duct) or a stem or handle to add to one. After LEAP 71's ShapeKernel
(BasePipe on a spine, LatticeManifold's printable tips), as a field.

The path: the points joined straight, or a smooth curve through them: a cubic Hermite spline with one
tangent direction per point (the bisector of the chords there; at the ends the direction that makes the
end span a parabola), each span's tangents as long as its chord. It is cut into straight pieces within
`TOLERANCE` of the curve (Douglas-Peucker on a sampling `DENSE` apart) and at most `_max_piece` long. The
radius changes linearly with the length along the path.

The section: a ball (round), or a teardrop: the ball with a cone on top whose sides rise at Roof from the
horizontal, its apex radius / cos(Roof) above the centre along Up. The pipe is the sweep of the section
along the path; the cone always points up, so a horizontal channel's roof is pointed and prints without
supports, and on a vertical stretch it lies inside the pipe. A piece is the convex hull of the sections at
its ends (the union of the sections in between), and its distance is exact: the smallest, over the piece,
of the section's distance there. The section's distance is the ball's, the apex's or the cone side's
(iq's uneven capsule, by which one the point is nearest), and each of those three can be minimized over the
piece in closed form (`_argmin_parameter`); the section's distance at the three minimizers, the smallest of
them, is the exact minimum (the ball's and the apex's distances are never below the section's, the side's
never above it, and equal to it where it applies). Neighbouring pieces share their end sections, so a curve
has no seams, and the apex runs along a line over every piece, so a curved roof has no notches.

Values are exact up to `reach` mm outside (at least PAD samples), and that value farther out: each piece is
computed only on the grid samples within reach of its box (a pure function of position, the same in blocks
bit for bit). `field.regrow(distance)` gives a pipe exact farther out (rounded or chamfered Boolean edges,
Smooth Union, Offset, Shell). Flat ends cut the end sections at the plane square to the path through the end
point (within `_cut_radius` of it).
"""
import hashlib
import math

import numpy as np

from .fields import Field

SECTIONS = ("round", "teardrop")
PATHS = ("smooth", "straight")
ENDS = ("round", "flat")
TOLERANCE = 0.005   # mm: the straight pieces stay this close to the smooth curve
DENSE = 0.25        # mm: the curve's sampling before it is cut into pieces
SAME = 1e-6         # mm: points closer than this are one point
PAD = 3             # samples around every piece computed exactly, at least
REACH = 1.0         # mm: ... and at least this far
CHUNK = 1 << 21     # samples per evaluation step (bounds the temporaries)
ROOF_RANGE = (10.0, 80.0)  # degrees
VERSION = 1


def parse_points(text):
    """Points from text: x, y, z in mm, the points separated by ; or new lines."""
    points = []
    for part in str(text or "").replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        values = [v for v in part.replace(",", " ").split() if v]
        try:
            xyz = [float(v) for v in values]
        except ValueError:
            xyz = []
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError(f"each point must be three numbers x, y, z in mm, like 10, 0, 25 (got '{part}')")
        points.append(xyz)
    if not points:
        raise ValueError("the path needs at least one point: x, y, z in mm, points separated by ;")
    return np.asarray(points, dtype=np.float64)


def _distinct(points):
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > SAME:
            keep.append(i)
    return points[keep]


def _tangents(points):
    """One unit tangent per point: the bisector of the chords; the ends' make their spans parabolas."""
    chord = np.diff(points, axis=0)
    u = chord / np.linalg.norm(chord, axis=1)[:, None]
    t = np.zeros_like(points)
    for i in range(1, len(points) - 1):
        b = u[i - 1] + u[i]
        n = np.linalg.norm(b)
        t[i] = b / n if n > 1e-9 else u[i]   # (a turn of 180°: no bisector)
    if len(points) == 2:
        t[0] = t[1] = u[0]
        return t
    t[0] = 2.0 * (t[1] @ u[0]) * u[0] - t[1]
    t[-1] = 2.0 * (t[-2] @ u[-1]) * u[-1] - t[-2]
    for i in (0, -1):
        n = np.linalg.norm(t[i])
        t[i] = t[i] / n if n > 1e-9 else u[i]
    return t


def _hermite(p0, p1, m0, m1, s):
    s = s[:, None]
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1


def _simplify(points, keep, tol):
    """Indices of `points` (a dense polyline) to keep so that it stays within `tol` of them, `keep`
    always among them (Douglas-Peucker)."""
    kept = np.zeros(len(points), dtype=bool)
    kept[list(keep)] = True
    stack = [(a, b) for a, b in zip(sorted(keep)[:-1], sorted(keep)[1:])]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, ab = points[i], points[j] - points[i]
        inner = points[i + 1:j] - a
        t = np.clip(inner @ ab / max(float(ab @ ab), 1e-300), 0.0, 1.0)
        off = np.linalg.norm(inner - t[:, None] * ab, axis=1)
        k = int(np.argmax(off))
        if off[k] > tol:
            m = i + 1 + k
            kept[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(kept)


def path(points, smooth=True):
    """(joints (n, 3), share of the path's length at each joint (n,)): the straight pieces of the path."""
    points = _distinct(np.asarray(points, dtype=np.float64))
    if len(points) == 1:
        return points, np.zeros(1)
    if smooth and len(points) > 2:
        t = _tangents(points)
        dense, at = [], []
        last = len(points) - 2
        for i in range(len(points) - 1):
            chord = float(np.linalg.norm(points[i + 1] - points[i]))
            n = max(16, int(math.ceil(3.0 * chord / DENSE)))   # a curved span is up to ~1.6 x its chord
            s = np.linspace(0.0, 1.0, n + 1)[: None if i == last else -1]
            at.append(sum(len(d) for d in dense))
            dense.append(_hermite(points[i], points[i + 1], t[i] * chord, t[i + 1] * chord, s))
        dense = np.concatenate(dense)
        at.append(len(dense) - 1)
        length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
        idx = _simplify(dense, at, TOLERANCE)
        joints, along = dense[idx], length[idx]
    else:
        joints = points
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = along[-1]
    return joints, (along / total if total > 0 else np.zeros(len(joints)))


def _split_long(joints, share, longest):
    """Pieces longer than `longest` cut evenly (their boxes stay small, the shape is the same)."""
    out_j, out_s = [joints[:1]], [share[:1]]
    for i in range(len(joints) - 1):
        n = max(1, int(math.ceil(np.linalg.norm(joints[i + 1] - joints[i]) / longest)))
        f = np.arange(1, n + 1) / n
        out_j.append(joints[i] + f[:, None] * (joints[i + 1] - joints[i]))
        out_s.append(share[i] + f * (share[i + 1] - share[i]))
    return np.concatenate(out_j), np.concatenate(out_s)


def _argmin_parameter(m_scale, slope, length):
    """For g(s) = A sqrt((m - s D)^2 + n^2) + beta s on 0 <= s <= 1, with c = beta / (A D) (the same for
    every sample): (factor, constant). Where |c| < 1, the minimizer is s = (m - factor n) / D clipped to
    0..1 (factor = c / sqrt(1 - c^2)), else the constant end (g only rises or only falls)."""
    if length <= 0:
        return None, (0.0 if slope >= 0 else 1.0)
    c = slope / (m_scale * length)
    if c >= 1.0:
        return None, 0.0
    if c <= -1.0:
        return None, 1.0
    return c / math.sqrt(1.0 - c * c), None


class _Shape:
    """The pipe's pieces as plain numbers (they travel inside worker tasks)."""

    def __init__(self, joints, radii, section, up, roof, ends):
        self.joints = np.ascontiguousarray(joints, dtype=np.float64)
        self.radii = np.ascontiguousarray(radii, dtype=np.float64)
        self.teardrop = section == "teardrop"
        self.up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        self.cos, self.sin = math.cos(math.radians(roof)), math.sin(math.radians(roof))
        self.roof = float(roof)
        self.flat = ends == "flat" and len(joints) > 1
        apex = self.radii / self.cos if self.teardrop else self.radii
        self.extent = np.maximum(self.radii, apex) if self.teardrop else self.radii
        r = self.radii[:, None]
        lo, hi = self.joints - r, self.joints + r
        if self.teardrop:
            tip = self.joints + apex[:, None] * self.up
            lo, hi = np.minimum(lo, tip), np.maximum(hi, tip)
        self.lo, self.hi = lo, hi   # each joint's section's box

    def bounds(self):
        return np.array([self.lo.min(axis=0), self.hi.max(axis=0)])

    def key(self):
        h = hashlib.sha1(repr(("pipe", VERSION, self.teardrop, tuple(self.up), self.roof, self.flat)).encode())
        h.update(self.joints.tobytes())
        h.update(self.radii.tobytes())
        return h.hexdigest()[:20]

    def pieces(self):
        """Per piece: (box lo, box hi, start point, its numbers for `_piece`); a single point is one piece."""
        P, r = self.joints, self.radii
        if len(P) == 1:
            yield self.lo[0], self.hi[0], P[0], None
            return
        u = self.up
        for k in range(len(P) - 1):
            d = P[k + 1] - P[k]
            length = float(np.linalg.norm(d))
            dr = float(r[k + 1] - r[k])
            e = d / length
            q = {"d": tuple(float(v) for v in d), "e": tuple(float(v) for v in e), "length": length,
                 "r0": float(r[k]), "dr": dr, "ball": _argmin_parameter(1.0, -dr, length)}
            if self.teardrop:
                du = float(d @ u)
                lift = d + (dr / self.cos) * u      # the apex's line
                lift2 = float(lift @ lift)
                q["tip"] = (tuple(float(v) for v in lift / lift2), float(r[k] / self.cos * (lift @ u) / lift2))
                across = d - du * u                  # the piece seen along Up
                width = float(np.linalg.norm(across))
                if width > 1e-9 * length:
                    ep = across / width
                    q["side"] = (tuple(float(v) for v in ep), tuple(float(v) for v in np.cross(u, ep)), width,
                                 _argmin_parameter(self.sin, -self.cos * du - dr, width))
                else:   # a piece along Up: the side's distance changes linearly along it
                    q["side"] = (None, None, 0.0, _argmin_parameter(self.sin, -self.cos * du - dr, 0.0))
            lo, hi = np.minimum(self.lo[k], self.lo[k + 1]), np.maximum(self.hi[k], self.hi[k + 1])
            yield lo, hi, P[k], q

    def cuts(self, far, voxel):
        """Flat ends: (end point, outward direction, radius of the ball in which the cut applies). The
        ball reaches `far` past every section, so the cut changes nothing at its border (the values there
        are `far` both ways)."""
        if not self.flat:
            return []
        P = self.joints
        reach = float(self.extent.max()) + far + voxel
        first, last = P[0] - P[1], P[-1] - P[-2]
        return [(P[0], first / np.linalg.norm(first), reach), (P[-1], last / np.linalg.norm(last), reach)]

    # --- distances (xp: numpy or cupy; float32 arrays; everything else Python floats) ---------------------
    def section(self, xp, wx, wy, wz, r):
        """The section's signed distance at offsets w from its centre, radius r (array or number)."""
        if not self.teardrop:
            return xp.sqrt(wx * wx + wy * wy + wz * wz) - r
        ux, uy, uz = (float(v) for v in self.up)
        c, s = self.cos, self.sin
        y = wx * ux + wy * uy + wz * uz
        cx, cy, cz = wy * uz - wz * uy, wz * ux - wx * uz, wx * uy - wy * ux
        rho2 = cx * cx + cy * cy + cz * cz
        rho = xp.sqrt(rho2)
        h = r * (1.0 / c)
        k = s * y - c * rho
        ball = xp.sqrt(rho2 + y * y) - r
        tip = xp.sqrt(rho2 + (y - h) * (y - h))
        side = s * rho + c * y - r
        return xp.where(k < 0, ball, xp.where(k > s * h, tip, side))

    def piece(self, xp, x, y, z, q):
        """Exact distance to piece `q` at offsets (x, y, z) from its start (broadcastable float32)."""
        if q is None:   # a single point: its section
            return self.section(xp, x, y, z, float(self.radii[0]))
        (dx, dy, dz), (ex, ey, ez), length = q["d"], q["e"], q["length"]

        def clip(v):
            return xp.clip(v, 0.0, 1.0)

        def minimizer(rule, m, n, scale):
            factor, constant = rule
            if constant is not None:
                return constant
            return clip((m - factor * n) * (1.0 / scale))

        m = x * ex + y * ey + z * ez
        cx, cy, cz = y * ez - z * ey, z * ex - x * ez, x * ey - y * ex
        candidates = [minimizer(q["ball"], m, xp.sqrt(cx * cx + cy * cy + cz * cz), length)]
        if self.teardrop:
            (tx, ty, tz), offset = q["tip"]
            candidates.append(clip(x * tx + y * ty + z * tz - offset))
            across, g, width, rule = q["side"]
            if across is None:
                candidates.append(rule[1])
            else:
                (px, py, pz), (gx, gy, gz) = across, g
                candidates.append(minimizer(rule, x * px + y * py + z * pz, xp.abs(x * gx + y * gy + z * gz), width))
        out = None
        for s in candidates:
            v = self.section(xp, x - s * dx, y - s * dy, z - s * dz, q["r0"] + s * q["dr"])
            out = v if out is None else xp.minimum(out, v)
        return out


def _window(grid, lo, hi, pad):
    """First and last + 1 sample index (per axis, of this grid) within `pad` mm of the box lo..hi; from
    the global sample index, so the same samples in every block."""
    start, shape = np.asarray(grid.start), np.asarray(grid.shape)
    first = np.ceil((np.asarray(lo) - pad - grid.base) / grid.voxel).astype(np.int64) - start
    last = np.floor((np.asarray(hi) + pad - grid.base) / grid.voxel).astype(np.int64) - start + 1
    return np.maximum(first, 0), np.minimum(last, shape)


def _chunks(xp, axes, first, last, origin):
    """Coordinates (float32, relative to `origin`) of the samples first..last, in slabs of at most CHUNK."""
    n1, n2 = int(last[1] - first[1]), int(last[2] - first[2])
    y = xp.asarray((axes[1][first[1]:last[1]] - origin[1]).astype(np.float32))[None, :, None]
    z = xp.asarray((axes[2][first[2]:last[2]] - origin[2]).astype(np.float32))[None, None, :]
    step = max(1, CHUNK // max(n1 * n2, 1))
    for i in range(int(first[0]), int(last[0]), step):
        j = min(i + step, int(last[0]))
        x = xp.asarray((axes[0][i:j] - origin[0]).astype(np.float32))[:, None, None]
        yield x, y, z, (slice(i, j), slice(int(first[1]), int(last[1])), slice(int(first[2]), int(last[2])))


def _max_piece(extent):
    return 4.0 * float(extent) + 4.0


def pipe(points, radius, end_radius=0.0, section="round", up=(0.0, 0.0, 1.0), roof=45.0, smooth=True,
         ends="round", reach=REACH):
    """The pipe along `points` ((n, 3) mm) as an implicit body (see the module doc). `end_radius` 0 = the
    same radius all along."""
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {', '.join(SECTIONS)}")
    if ends not in ENDS:
        raise ValueError(f"ends must be one of {', '.join(ENDS)}")
    radius, end_radius = float(radius), float(end_radius or 0.0)
    if not radius > 0:
        raise ValueError("radius must be more than 0 mm")
    if end_radius < 0:
        raise ValueError("end radius must be 0 (the same) or more")
    up = np.asarray(up, dtype=np.float64)
    if section == "teardrop" and not np.linalg.norm(up) > 0:
        raise ValueError("up must be a direction, like 0, 0, 1")
    if section == "round":
        up, roof = np.array([0.0, 0.0, 1.0]), 45.0
    roof = min(max(float(roof), ROOF_RANGE[0]), ROOF_RANGE[1])
    joints, share = path(points, smooth)
    last = end_radius if end_radius > 0 else radius
    longest = _max_piece(max(radius, last) / (math.cos(math.radians(roof)) if section == "teardrop" else 1.0))
    joints, share = _split_long(joints, share, longest)
    radii = radius + share * (last - radius)
    shape = _Shape(joints, radii, section, up, roof, ends)
    pieces = list(shape.pieces())
    reach = float(reach)

    def fn(ctx):
        xp, grid = ctx.xp, ctx.grid
        far = max(PAD * grid.voxel, reach)
        out = xp.full(grid.shape, far, dtype=xp.float32)
        axes = grid.axes()
        for lo, hi, origin, q in pieces:
            first, last = _window(grid, lo, hi, far)
            if np.any(last <= first):
                continue
            for x, y, z, box in _chunks(xp, axes, first, last, origin):
                block = out[box]
                xp.minimum(block, shape.piece(xp, x, y, z, q).astype(xp.float32), out=block)
        for centre, outward, ball in shape.cuts(far, grid.voxel):
            first, last = _window(grid, centre, centre, ball)
            if np.any(last <= first):
                continue
            for x, y, z, box in _chunks(xp, axes, first, last, centre):
                plane = x * float(outward[0]) + y * float(outward[1]) + z * float(outward[2])
                block = out[box]
                cut = xp.minimum(xp.maximum(block, plane), far)
                block[...] = xp.where(x * x + y * y + z * z < ball * ball, cut, block)
        return out

    bounds = shape.bounds()
    key = hashlib.sha1(repr((shape.key(), reach)).encode()).hexdigest()[:20]
    field = Field(fn, key, f"pipe({len(joints)} joints)", bounds, cache=True)
    field.extent = bounds
    field.regrow = lambda distance: pipe(points, radius, end_radius, section, up, roof, smooth, ends,
                                         max(reach, 1.25 * float(distance)))
    field.joints = joints   # (for tests and notes)
    return field
