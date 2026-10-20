"""Patterns of whole shapes laid out on picked CAD faces (shapes and layouts: engine/pattern2d.py):
a body of holes, pockets or raised shapes for the Booleans (`cutter`, node Hole Pattern), or a
0..1 relief for Offset Body (`relief`, node Surface Pattern).

1. Each chosen face gets a 2D frame in mm from the surface parameters the import stores per
   triangle corner (`cad_uv`): one scale for u and one for v, the area-weighted mean of the
   triangles' Jacobians (1 on planes, the radius for a cylinder's angle), turned so that x follows
   Align (a world axis projected onto the face) plus Angle. Faces that close around (cylinders,
   cones) get a whole number of cells around, so the pattern has no seam there.
   Join Faces "tangent": chosen faces joined smoothly are unfolded together into one flat sheet
   (engine/unfold.py), laid onto the CAD frame of its largest face, and that sheet is the frame
   instead, so one pattern runs across their borders. Sheets that close around (the halves of a
   cylinder) or stretch too much keep a frame per face.
2. The layout's cell centres are looked up in the face's own triangulation in (u, v): centres
   outside the trimmed face (in a cut-out) find no triangle and are dropped; the others are placed
   in 3D by barycentric coordinates, with the triangle's normal as the hole's axis.
3. A hole is kept only if its whole outline stays Edge margin away from the selection's border
   (the edges to faces that are not chosen, and sharp folds between chosen ones), and not too
   close to a hole from another face (two faces' frames don't line up).
4. Through holes are cut as deep as the wall behind them (ray casts at the centre and around the
   outline, plus EXTRA), and start OUTSIDE mm above the face (more where the face curves up).
   Pockets (Depth) are cut that deep; one as deep as the wall goes through, no further.
   Raised shapes (Height, for Boolean Union) stand Height mm above the face at their centre and
   reach just far enough into the wall to join it: as deep as the face falls away under their
   outline, plus JOIN, never more than most of the wall (so they don't poke out behind it).
5. The cutter: every hole is a straight prism along its axis (as if drilled), max(shape distance
   across, depth window along). It is computed hole by hole on the part of the grid around each
   one: the field depends only on position (block meshing stays exact), and nothing is computed
   away from the holes. (Measuring every voxel from its closest point on the faces would follow
   the curvature exactly, but cost a mesh query per voxel and cut holes in half at the border
   between two chosen faces.)
6. The relief: the same per-shape evaluation, a smooth step from the outline inwards instead of
   a prism, constant along each shape's axis within Band of the face.
"""
import hashlib
import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import pattern2d as P2
from . import unfold
from .facemask import border_segments, sample_segments
from .fields import Field, _digest, sample

ALIGN = ("as in CAD", "x", "y", "z")
JOIN_FACES = ("tangent", "no")  # chosen faces joined smoothly: one unfolded pattern, or one pattern per face
MAX_STRETCH = 0.25      # a sheet that only unfolds stretching more (on 5% of its area) keeps a pattern per face
MIN_HOLE = 1.0          # mm: smaller holes print poorly (SLS: the powder hardly clears)
MIN_WEB = 0.8           # mm: thinner webs between holes are fragile
FOLD = math.radians(30)  # chosen faces meeting at a sharper bend have a border between them
OUTSIDE = 1.0           # mm: the cutter starts this far above the face
EXTRA = 0.5             # mm: through holes are cut this much deeper than the wall
JOIN = 0.3              # mm: raised shapes reach this much deeper than the face under them
FAR = np.float32(1e4)   # the cutter's value away from every hole (outside it)
PAD = 3                 # voxels around each hole's box computed exactly
OUTLINE = 64            # points on each outline for webs and margins
RAYS = 8                # extra wall-thickness rays around each hole's outline
# what cells of a regular layout (Randomness 0) all are: Voronoi cells of the staggered (honeycomb)
# layout look exactly like Shape hexagon, so picking voronoi alone seems to change nothing
REGULAR_CELLS = {("voronoi", "staggered"): "hexagons", ("voronoi", "grid"): "squares",
                 ("delaunay", "staggered"): "triangles", ("delaunay", "grid"): "right triangles"}


class HoleError(ValueError):
    pass


class Holes:
    """The holes laid out on one body (arrays over the holes) and the report about them."""

    def __init__(self, shape, bounds, title="Hole Pattern"):
        self.shape = shape
        self.title = title            # the node, at the start of every report line
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.centre = np.zeros((0, 3))
        self.normal = np.zeros((0, 3))
        self.t1 = np.zeros((0, 3))
        self.t2 = np.zeros((0, 3))
        self.scale = np.zeros(0)
        self.turn = np.zeros(0)       # extra rotation of each hole (rings), radians
        self.outer = np.zeros(0)      # mm the cutter reaches above the face
        self.depth = np.zeros(0)      # mm it reaches below
        self.face = np.zeros(0, dtype=np.int64)
        self.cells = None             # Voronoi: each shape's own outline, (n, m, 2) unit polygons
        self.wall = None              # Surface Pattern: wall thickness under each shape, mm (NaN: none)
        self.face_normal = None       # the face's normal at each centre (the axis, unless tilted)
        self.shape_b = None           # Morph to: the second outline, and each hole's share of it (0..1)
        self.morph = None
        self.lines = []               # (text, level) for the editor's log
        self.smallest_web = None
        self.regular = None           # cells of a regular layout: what they all are ("hexagons"), else None

    def __len__(self):
        return len(self.centre)

    def take(self, keep):
        self._outlines = None
        for name in ("centre", "normal", "t1", "t2", "scale", "turn", "outer", "depth", "face", "cells", "wall",
                     "morph", "face_normal"):
            value = getattr(self, name)
            if value is not None and len(value) == len(keep):  # depths are only set at the end
                setattr(self, name, value[keep])

    def _sdf_of(self, shape, x, y, index):
        if shape.kind in P2.CELLS:
            return P2.polygon_sdf(x, y, self.cells[index]) - shape.corner
        return shape.sdf(x, y)

    def unit_sdf(self, x, y, index):
        """The 2D distance (at scale 1) of points x, y in the frames of shapes `index`. A morph
        blends the two shapes' distances (a convex blend of distances: no steeper than either)."""
        first = self._sdf_of(self.shape, x, y, index)
        if self.shape_b is None:
            return first
        return first + self.morph[index] * (self._sdf_of(self.shape_b, x, y, index) - first)

    def _radius_of(self, shape):
        if shape.kind in P2.CELLS:
            return np.linalg.norm(self.cells, axis=2).max(axis=1) + shape.corner
        return np.full(len(self), shape.circumradius())

    def radius(self):
        """Each shape's distance from its centre to its farthest point, at scale 1 (a morph: the
        larger of its two shapes)."""
        r = self._radius_of(self.shape)
        return r if self.shape_b is None else np.maximum(r, self._radius_of(self.shape_b))

    def widths(self):
        """Each shape's inner width, mm: Size, or for a cell or a morph twice its distance from its
        centre to its outline."""
        if self.cells is None and self.shape_b is None:
            return self.scale * self.shape.size
        index = np.arange(len(self))
        return -2 * self.unit_sdf(np.zeros(len(self)), np.zeros(len(self)), index) * self.scale

    def unit_outlines(self, index, n):
        """n points on the outlines of shapes `index` at scale 1, in their turned frames:
        (len(index), n, 2), found along rays from each centre (every shape is star-shaped)."""
        if self.cells is None and self.shape_b is None:
            return np.broadcast_to(self.shape.outline(n), (len(index), n, 2))
        cached = getattr(self, "_outlines", None)
        if cached is None or cached[0] != n or len(cached[1]) != len(self):  # once for all shapes
            every = np.arange(len(self))
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            d = np.column_stack([np.cos(t), np.sin(t)])
            k = np.repeat(every, n)
            hi = np.repeat(1.01 * self.radius(), n)
            lo = np.zeros(len(k))
            dx, dy = np.tile(d[:, 0], len(self)), np.tile(d[:, 1], len(self))
            for _ in range(40):
                mid = (lo + hi) / 2
                inside = self.unit_sdf(dx * mid, dy * mid, k) < 0
                lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
            r = (lo + hi) / 2
            self._outlines = cached = (n, np.stack([dx * r, dy * r], axis=1).reshape(len(self), n, 2))
        return cached[1][index]

    def local(self, points, index):
        """Points (m, 3) in the frames of holes `index` (m,): (across x, across y, along the axis),
        x and y turned into the shape's own frame by the hole's turn and divided by its scale."""
        d = points - self.centre[index]
        a = np.einsum("ij,ij->i", d, self.t1[index])
        b = np.einsum("ij,ij->i", d, self.t2[index])
        c, s = np.cos(self.turn[index]), np.sin(self.turn[index])
        return (a * c + b * s) / self.scale[index], (b * c - a * s) / self.scale[index], \
            np.einsum("ij,ij->i", d, self.normal[index])

    def outlines(self, index, n=OUTLINE, shrink=1.0):
        """3D points on the outlines of holes `index` (len(index), n, 3), in their tangent planes."""
        o = self.unit_outlines(index, n) * shrink
        c, s = np.cos(self.turn[index])[:, None], np.sin(self.turn[index])[:, None]
        k = self.scale[index][:, None]
        x, y = (o[..., 0] * c - o[..., 1] * s) * k, (o[..., 0] * s + o[..., 1] * c) * k
        return self.centre[index][:, None] + x[..., None] * self.t1[index][:, None] + y[..., None] * self.t2[index][:, None]

    def boxes(self):
        """Bounding box (lo, hi) of every hole's prism, (n, 3) each."""
        r = self.radius() * self.scale
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for z in (-self.depth, self.outer):
                    corners.append(self.centre + (sx * r)[:, None] * self.t1 + (sy * r)[:, None] * self.t2
                                   + z[:, None] * self.normal)
        corners = np.stack(corners)
        return corners.min(axis=0), corners.max(axis=0)

    def note(self, text, level="info"):
        self.lines.append((f"{self.title}: {text}", level))


# --- placing the holes ---------------------------------------------------------------------------
def place(mesh, chosen, shape, layout="staggered", spacing=6.0, align="as in CAD", angle=0.0,
          offset=(0.0, 0.0), margin=2.0, depth=0.0, size_scale=1.0, height=0.0, rotation=None, check=None,
          surface=False, randomness=0.0, seed=0, min_size=0.0, morph_to=None, morph=0.0, alternate="none",
          tilt=0.0, join="tangent"):
    """The holes of a pattern on CAD faces `chosen` (face numbers) of a tessellated body.
    join: "tangent" = faces joined smoothly share one pattern, unfolded flat; "no" = one per face.
    min_size: shapes narrower than this after Size scale (mm, inner width) are left out.
    morph_to: a second shape each hole blends into by `morph` (0..1, a number or a field read at its
    centre). randomness may be a field too (read where each centre would be without it).
    alternate: every other cell (or row) turned (pattern2d.ALTERNATE). tilt: degrees every hole's
    axis leans from the face's normal towards the pattern's x (louvres).
    depth: 0 = through holes, more = pockets that deep (one deeper than the wall goes through).
    height > 0: raised shapes that tall instead (for a union with the part; depth is not used).
    size_scale, depth and height may be fields, read at each centre (0 or less: no shape there);
    rotation: None (the shape's own) or a field of degrees added per shape. surface: the shapes of
    a Surface Pattern (no depth or height: see `relief`). randomness 0..1 moves every centre by
    up to that share of half a step (seed: another set); shape "voronoi" makes every centre's own
    cell of the layout, moved in by half the web (shape.size), its outline."""
    if layout not in P2.LAYOUTS:
        raise HoleError(f"layout must be one of {', '.join(P2.LAYOUTS)}")
    if alternate not in P2.ALTERNATE:
        raise HoleError(f"alternate must be one of {', '.join(P2.ALTERNATE)}")
    if not -80.0 <= float(tilt) <= 80.0:
        raise HoleError("Tilt must be between -80 and 80 degrees")
    if align not in ALIGN:
        raise HoleError(f"align must be one of {', '.join(ALIGN)}")
    if join not in JOIN_FACES:
        raise HoleError(f"Join Faces must be one of {', '.join(JOIN_FACES)}")
    if spacing <= 0:
        raise HoleError("spacing must be greater than 0")
    ids = np.asarray(mesh.face_attributes.get("cad_face", []))
    uv = mesh.face_attributes.get("cad_uv")
    if uv is None or len(ids) != len(mesh.faces) or len(uv) != len(mesh.faces):
        raise HoleError("this body has no CAD surface data: connect a body from Import STEP")
    surfaces = np.asarray(mesh.metadata.get("cad_surfaces", np.zeros((0, 3))))
    V = np.asarray(mesh.vertices, dtype=np.float64)
    Fc = np.asarray(mesh.faces, dtype=np.int64)
    holes = Holes(shape, mesh.bounds, "Surface Pattern" if surface else "Hole Pattern")
    chosen = sorted(int(f) for f in chosen)
    if not chosen:
        holes.note("no faces chosen")
        return holes
    raised = not surface and (isinstance(height, Field) or height > 0)
    pockets = not surface and not raised and (isinstance(depth, Field) or depth > 0)
    what = "shapes" if surface else "raised shapes" if raised else "pockets" if pockets else "holes"
    standing = surface or raised   # shapes, not holes: gaps between them rather than webs
    sx, sy = P2.steps(shape, spacing, layout, alternate)
    if morph_to is not None:  # room for both shapes
        sx, sy = (max(a, b) for a, b in zip((sx, sy), P2.steps(morph_to, spacing, layout, alternate)))
    shapes = [shape] + ([morph_to] if morph_to is not None else [])
    if isinstance(rotation, Field) and any(k.kind in P2.LONG for k in shapes):  # may point any way
        sx = sy = spacing + max(2 * k.circumradius() - k.size for k in shapes if k.kind in P2.LONG)
    cell_shape = next((k for k in shapes if k.kind in P2.CELLS), None)
    cells = cell_shape is not None
    inset = cell_shape.size / 2 + cell_shape.corner if cells else None
    if cells and spacing <= 2 * inset:
        raise HoleError(f"Spacing ({spacing:g} mm) leaves no room for cells with web {cell_shape.size:g} mm and "
                        f"corner radius {cell_shape.corner:g} mm")
    if not isinstance(randomness, Field):
        randomness = min(max(float(randomness), 0.0), 1.0)
    axis = None if align == "as in CAD" else np.eye(3)[ALIGN.index(align) - 1]

    parts = []
    for face, tri, corner_uv, period, name, flat in _frames(holes, V, Fc, ids, uv, surfaces, chosen, join, check):
        if check is not None:
            check()
        part = _face_holes(holes, face, V, Fc[tri], corner_uv, period, layout, sx, sy,
                           axis, angle, offset, randomness, seed, inset, check, alternate,
                           cell_shape.kind if cells else None, name=name, flat=flat)
        if part is not None:
            parts.append(part)
    if not parts:
        holes.note("no hole centre falls on the chosen faces (spacing larger than the faces?)", "error")
        return holes
    for name, i in (("centre", 0), ("normal", 1), ("t1", 2), ("t2", 3), ("turn", 4), ("face", 5)):
        setattr(holes, name, np.concatenate([p[i] for p in parts]))
    if cells:
        m = max(p[6].shape[1] for p in parts)
        holes.cells = np.concatenate([np.concatenate([p[6], np.repeat(p[6][:, -1:], m - p[6].shape[1], axis=1)], axis=1)
                                      for p in parts])
    found = len(holes)
    vanished = 0
    if cells:  # cells smaller than the web leave nothing
        empty = np.isnan(holes.cells).any(axis=(1, 2))
        vanished = int(empty.sum())
        holes.take(~empty)

    if morph_to is not None:  # each hole's share of the second shape
        holes.shape_b = morph_to
        if isinstance(morph, Field):
            share = np.clip(np.nan_to_num(read_at(morph, holes.centre, check)), 0.0, 1.0)
            holes.note(f"Morph from the field: {share.min():.2f} to {share.max():.2f}")
        else:
            share = np.full(len(holes), min(max(float(morph), 0.0), 1.0))
        holes.morph = share

    # size scale: a number, or a field read at every hole's centre
    if isinstance(size_scale, Field):
        holes.scale = np.nan_to_num(read_at(size_scale, holes.centre, check), nan=0.0)
    else:
        holes.scale = np.full(len(holes), float(size_scale))
    gone = (holes.scale < 0.02) if cells else (holes.scale * shape.size < 0.05)
    if gone.any():
        holes.take(~gone)
    none_scaled = int(gone.sum())
    tiny = holes.widths() < float(min_size)   # e.g. where a fade shrinks them to pinholes
    holes.take(~tiny)
    too_small = int(tiny.sum())
    if isinstance(rotation, Field) and len(holes) and not cells:
        turned = np.nan_to_num(read_at(rotation, holes.centre, check))
        holes.turn = holes.turn + np.radians(turned)
        holes.note(f"Rotation from the field: {turned.min():.0f} to {turned.max():.0f}°")
    holes.face_normal = holes.normal.copy()
    if tilt and len(holes):  # each axis leans towards the pattern's x: turn (n, t1) about t2
        a = math.radians(float(tilt))
        n, t1 = holes.normal, holes.t1
        holes.normal, holes.t1 = n * math.cos(a) + t1 * math.sin(a), t1 * math.cos(a) - n * math.sin(a)

    selected = np.isin(ids, chosen)
    border = _border_points(V, Fc, ids, selected, min(margin, shape.size))
    near_border = _near_border(holes, border, margin)
    holes.take(~near_border)
    scattered = (isinstance(randomness, Field) or randomness > 0) and not cells
    twice, too_close, crowded, overlaps = _crowded(holes, spacing, scattered=scattered)
    holes.take(~(twice | too_close | crowded))

    # each shape's pocket depth or height: a number, or a field read at its centre
    amount, unset, label = None, 0, "Height" if raised else "Depth"
    source = height if raised else depth if pockets else None
    if source is not None:
        if isinstance(source, Field):
            amount = np.nan_to_num(read_at(source, holes.centre, check))
        else:
            amount = np.full(len(holes), float(source))
        zero = amount <= 1e-3
        unset = int(zero.sum())
        holes.take(~zero)
        amount = amount[~zero]
        if isinstance(source, Field) and len(amount):
            holes.note(f"{label} from the field: {amount.min():.2f} to {amount.max():.2f} mm")
    if surface:  # the relief must not reach the far side of a thin wall
        no_wall = np.zeros(len(holes), dtype=bool)
        holes.wall = _walls(holes, V, Fc)
    else:
        no_wall = _depths(holes, V, Fc, pocket=amount if pockets else None, height=amount if raised else None)
        holes.take(~no_wall)

    n = len(holes)
    described = shape.describe() if morph_to is None else f"{shape.describe()} morphing into {morph_to.describe()}"
    msg = f"{n} {what} ({described}) on {len(chosen)} face{'s' if len(chosen) != 1 else ''}"
    dropped = [(int(near_border.sum()), f"near the border (Edge margin {margin:g} mm)"),
               (int(too_close.sum()), f"too close to {what} on a neighbouring face"),
               (int(crowded.sum()), "moved too close to a neighbour by Randomness"),
               (int(no_wall.sum()), "with no wall behind them"),
               (none_scaled, "with Size scale 0"),
               (too_small, f"under Min size {float(min_size):g} mm"),
               (unset, f"with {label} 0"),
               (vanished, "cells too small for the web")]
    dropped = [f"{k} {why}" for k, why in dropped if k]
    holes.note(msg + (f"; left out: {', '.join(dropped)}" if dropped else ""), "info" if n else "error")
    if cells and not isinstance(randomness, Field) and randomness == 0:
        holes.regular = REGULAR_CELLS.get((cell_shape.kind, layout))
        if holes.regular and n:
            holes.note(f"Randomness is 0, so the {cell_shape.kind} cells of the {layout} layout are all regular "
                       f"{holes.regular}: Randomness 0.6-0.9 gives organic cells")
    if overlaps:
        holes.note(f"{overlaps} pairs of {what} overlap: increase Spacing or reduce Size", "error")
    if holes.smallest_web is not None and n > 1:
        level = "error" if holes.smallest_web < MIN_WEB else "info"
        gap = "narrowest gap between shapes" if standing else "thinnest web between holes"
        holes.note(f"{gap} {holes.smallest_web:.2f} mm"
                   + (f" (under {MIN_WEB:g} mm: {'hard to print' if standing else 'fragile'})"
                      if level == "error" else ""), level)
    if n:
        sizes = holes.widths()
        small = int((sizes < MIN_HOLE).sum())
        if small:
            why = "fragile" if standing else "hard to print and to clear of powder"
            holes.note(f"{small} {what} are under {MIN_HOLE:g} mm across ({why})", "error")
        # the shapes under MIN_HOLE are reported above: a fade (Size scale) shrinking shapes to nothing
        # would otherwise ask for a voxel of a few hundredths of a mm
        printable = sizes[sizes >= MIN_HOLE]
        smallest = float(printable.min()) if len(printable) else float(sizes.min())
        holes.note(f"for clean edges, mesh with a voxel of at most {smallest / 6:.2f} mm "
                   f"(a sixth of the smallest shape{f' of {MIN_HOLE:g} mm or more' if small else ''})")
    return holes


def _frames(holes, V, F, ids, uv, surfaces, chosen, join, check=None):
    """What gets a pattern of its own: (face number, its triangles, (u, v) at their corners (k, 3, 2),
    the u and v periods, its name in the log, whether (u, v) is already in mm) per chosen face, or
    with join "tangent" per sheet of faces joined smoothly, unfolded flat (named by its first face)."""
    order = np.argsort(ids, kind="stable")
    starts = np.searchsorted(ids[order], chosen)
    ends = np.searchsorted(ids[order], chosen, side="right")
    own = {f: order[s:e] for f, s, e in zip(chosen, starts, ends) if e > s}
    out = []
    if join == "tangent" and len(own) > 1:
        for tri, local, source in unfold.sheets(V, F, ids, np.concatenate(list(own.values())), FOLD):
            faces = sorted(set(ids[tri].tolist()))
            if len(faces) < 2:
                continue
            name = "faces " + ", ".join(str(f) for f in faces[:6]) + (", ..." if len(faces) > 6 else "")
            flat = _unfolded(holes, name, V[source], local, V[F[tri]], ids[tri], np.asarray(uv[tri], dtype=np.float64),
                             check)
            if flat is not None:
                out.append((faces[0], tri, flat[local], np.zeros(2), name, True))
                for f in faces:
                    own.pop(f)
    for f, tri in own.items():
        period = surfaces[f, 1:3] if f < len(surfaces) else np.zeros(2)
        out.append((f, tri, np.asarray(uv[tri], dtype=np.float64), period, f"face {f}", False))
    return sorted(out, key=lambda frame: frame[0])


def _unfolded(holes, name, V, F, P, faces, UV, check=None):
    """The 2D positions (mm) of a sheet's vertices (V, F: its own numbering; P, faces, UV: the same
    triangles' corners, CAD faces and surface parameters), laid onto the CAD frame of its largest
    face, so that face's pattern stays where it was. None (and why, in the log) when it can't be."""
    why = unfold.closes_around(V, F)
    if why is None:
        X, stretch = unfold.flatten(V, F, check)
        area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        real = stretch[:, 1] != 0                                  # not degenerate
        worst = np.where(real, np.maximum(np.abs(stretch[:, 0]), 1 / np.where(real, np.abs(stretch[:, 1]), 1.0)) - 1, 0.0)
        usual = _quantile(worst, area, 0.95)
        folded = area[stretch[:, 0] < 0].sum() / area.sum()
        if folded > 1e-3 or usual > MAX_STRETCH:
            why = f"they stretch by {usual:.0%} or more unfolded flat"
    if why is not None:
        holes.note(f"{name}: {why}, so each face has its own pattern")
        return None
    holes.note(f"{name} unfolded into one sheet: the spacing varies by "
               + (f"up to {usual:.0%} (on 95% of the area, at most {worst.max():.0%}) where the faces curve "
                  f"both ways" if usual >= 0.01 else "less than 1%"))
    # the largest face's (u, v) in mm, as _face_holes scales it; the sheet turned (or mirrored) onto it
    biggest = np.bincount(faces, weights=area).argmax()
    ok, Ju, Jv, _, a = _jacobians(P, UV)
    ok &= faces == biggest
    if not ok.any():
        return X
    w = np.repeat(a[ok], 3)
    scale = [np.sum(a[ok] * np.linalg.norm(J[ok], axis=1)) / a[ok].sum() for J in (Ju, Jv)]
    src, dst = X[F[ok]].reshape(-1, 2), (UV[ok] * scale).reshape(-1, 2)
    src_mid, dst_mid = w @ src / w.sum(), w @ dst / w.sum()
    u, _, vt = np.linalg.svd((src - src_mid).T @ ((dst - dst_mid) * w[:, None]))
    return (X - src_mid) @ (u @ vt) + dst_mid


def _jacobians(P, UV):
    """Per triangle (corners P (k, 3, 3), surface parameters UV (k, 3, 2)): whether it is usable
    (finite, not degenerate), dP/du and dP/dv (k, 3), its edges' cross product and its area."""
    ok = np.isfinite(UV).all(axis=(1, 2))
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    d1, d2 = UV[:, 1] - UV[:, 0], UV[:, 2] - UV[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    ok &= (np.abs(det) > 1e-12 * np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)) & (area > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ju = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
        Jv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det[:, None]
    return ok, Ju, Jv, cross, area


def _face_holes(holes, face, V, corners, UV, period, layout, sx, sy, axis, angle, offset, randomness=0.0, seed=0,
                inset=None, check=None, alternate="none", cell_kind=None, name=None, flat=False):
    """Centres, axes and frames of the holes on one CAD face (corners: (k, 3) vertex numbers of its
    triangles, UV: (k, 3, 2) their surface parameters), and with `inset` (Voronoi) each one's cell
    (unit polygons in its frame), or None. name: what the log calls it; flat: UV is already in mm
    (an unfolded sheet of faces)."""
    name = name or f"face {face}"
    P = V[corners]
    if not np.isfinite(UV).all(axis=(1, 2)).any():
        holes.note(f"{name} has no surface parameters (left out)", "error")
        return None
    ok, Ju, Jv, cross, area = _jacobians(P, UV)
    if not ok.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        normal = cross / (2 * area[:, None])
    w = np.where(ok, area, 0.0)
    lu, lv = np.linalg.norm(Ju, axis=1), np.linalg.norm(Jv, axis=1)
    if flat:  # the sheet's stretch is reported where it is unfolded
        su = sv = 1.0
    else:
        su, sv = float(np.sum(w * np.nan_to_num(lu)) / w.sum()), float(np.sum(w * np.nan_to_num(lv)) / w.sum())
        lo = min(_quantile(lu / su, w, 0.05), _quantile(lv / sv, w, 0.05))
        hi = max(_quantile(lu / su, w, 0.95), _quantile(lv / sv, w, 0.95))
        if hi > 1.1 * lo:
            holes.note(f"cell size varies {lo:.2f}x to {hi:.2f}x across {name} (uneven CAD parameters)")

    eu = Ju / np.where(lu > 0, lu, 1.0)[:, None]
    side = np.cross(normal, eu)                       # in the face, 90° from u (seen from outside)
    hand = np.where(np.einsum("ij,ij->i", Jv, side) < 0, -1.0, 1.0)  # (u, v) mirrored seen from outside
    face_hand = 1.0 if np.sum(w * hand) >= 0 else -1.0
    phi = 0.0
    if axis is not None:
        au, aw = eu @ axis, side @ axis
        a = hand * np.arctan2(aw, au)
        weight = w * np.hypot(au, aw)
        vx, vy = np.sum(weight * np.cos(a)), np.sum(weight * np.sin(a))
        if math.hypot(vx, vy) < 0.3 * w.sum():
            holes.note(f"Align {'xyz'[int(np.argmax(axis))]} has no clear direction on {name} "
                       f"(kept as in CAD)")
        else:
            phi = math.atan2(vy, vx)
    phi += face_hand * math.radians(angle)
    # Periodic surfaces: a face covering (about) half the period or more, a cylinder or a half of
    # one, gets a whole number of cells in the period, so the pattern closes and both halves of a
    # split cylinder line up. A fillet (a quarter or less) is left as it is.
    span = np.ptp(UV[ok], axis=(0, 1))
    pu, pv = (float(p) if p > 0 and s > 0.45 * p else 0.0 for p, s in zip(period, span))
    closes = [p > 0 and s > 0.99 * p for p, s in zip((pu, pv), span)]
    if pu > 0 or pv > 0:  # only quarter turns keep the pattern closing
        snapped = round(phi / (math.pi / 2)) * (math.pi / 2)
        if abs(snapped - phi) > math.radians(1):
            holes.note(f"{name} runs around an axis, so the pattern only turns in steps of 90° "
                       f"there")
        phi = snapped
    c, s = round(math.cos(phi), 12), round(math.sin(phi), 12)

    # (x, y): the pattern's frame in mm
    Umm, Vmm = UV[..., 0] * su, UV[..., 1] * sv
    X, Y = Umm * c + Vmm * s, Vmm * c - Umm * s
    fsx, fsy = sx, sy
    wrap = [0, 0]   # cells around in x and in y (Randomness moves cell i and i + wrap alike)
    along_u = abs(c) > 0.5
    for p, scale, is_u, whole in ((pu, su, True, closes[0]), (pv, sv, False, closes[1])):
        if p <= 0 or layout in ("rings", "spiral"):
            continue
        length = p * scale
        if is_u == along_u:  # x runs around
            fsx = _fit(sx, length, even=alternate.startswith("checker"))
            count = length / fsx
            wrap[0] = int(round(count))
        else:
            fsy = _fit(sy, length, even=layout == "staggered" or alternate != "none")
            count = length / fsy
            wrap[1] = int(round(count))
        if whole:
            holes.note(f"{name} closes around: {int(round(count))} cells around "
                       f"({length:.1f} mm)")
    X, Y, tw = X[ok], Y[ok], np.repeat(w[ok], 3)
    box = (X.min(), Y.min(), X.max(), Y.max())
    centroid = (np.sum(tw * X.ravel()) / tw.sum(), np.sum(tw * Y.ravel()) / tw.sum())
    reach = 2 * max(fsx, fsy)   # a Voronoi cell at the face's edge needs its neighbours beyond it
    varies = isinstance(randomness, Field)   # then the centres move after the field is read where they are
    try:
        x, y, turn, li, lj = P2.lattice(layout, fsx, fsy, box, origin=offset,
                                        center=(centroid[0] + offset[0], centroid[1] + offset[1]),
                                        pad=(reach if inset is not None else 0.0) + (0.5 * reach if varies else 0.0),
                                        randomness=0.0 if varies else randomness, seed=seed, wrap=tuple(wrap))
    except ValueError as err:
        raise HoleError(f"{name}: {err}") from err
    if not len(x):
        return None

    def on_face(x, y):
        """(found mask, triangle of each found point, barycentric coordinates) in the face."""
        tri, bary = _locate(UV[ok], np.column_stack([(x * c - y * s) / su, (x * s + y * c) / sv]))
        found = tri >= 0
        return found, np.flatnonzero(ok)[tri[found]], bary[found]

    if varies:  # how random each centre is: the field where it would be; off the face, its nearest
        found, index, b = on_face(x, y)
        amount = np.zeros(len(x))
        if found.any():
            amount[found] = np.clip(np.nan_to_num(read_at(randomness, np.einsum("ij,ijk->ik", b, P[index]), check)),
                                    0.0, 1.0)
            if not found.all():
                near = cKDTree(np.column_stack([x[found], y[found]])).query(np.column_stack([x[~found], y[~found]]))[1]
                amount[~found] = amount[found][near]
        x, y = P2.jitter(layout, x, y, li, lj, amount, fsx, fsy, seed, tuple(wrap))
    if alternate != "none" and cell_kind is None:  # every other cell (or row) turned
        odd = ((lj if alternate == "rows 180" else li + lj) % 2).astype(np.float64)
        turn = turn + odd * (math.pi / 2 if alternate == "checker 90" else math.pi)
    cells = None
    if cell_kind == "delaunay":  # the triangles between the centres, each around its centroid
        x, y, cells = P2.delaunay_cells(x, y, inset)
        turn = np.zeros(len(x))
        if not len(x):
            return None
    elif inset is not None:  # each cell around its own centroid (a seed can lie outside its shrunk cell)
        cells = P2.voronoi_cells(x, y, inset, reach)
        middle = P2.centroids(cells)
        ok_cells = np.isfinite(middle).all(axis=1)
        x, y = np.where(ok_cells, x + middle[:, 0], x), np.where(ok_cells, y + middle[:, 1], y)
        cells = cells - np.where(ok_cells[:, None], middle, 0.0)[:, None, :]
    # back to (u, v), then into the face's triangles
    found, index, b = on_face(x, y)
    if not found.any():
        return None
    centre = np.einsum("ij,ijk->ik", b, P[index])
    # the axis: the face's own vertex normals (angle-weighted), interpolated. A triangle's normal
    # leans by up to half its angle on a curved face; this is exactly radial on a cylinder.
    local, inverse = np.unique(corners, return_inverse=True)
    vn = np.zeros((len(local), 3))
    unit = np.nan_to_num(cross / (2 * area[:, None]))
    np.add.at(vn, inverse.ravel(), (unit[:, None, :] * unfold.corner_angles(P)[..., None]).reshape(-1, 3))
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-300)[:, None]
    n = np.einsum("ij,ijk->ik", b, vn[inverse.reshape(-1, 3)[index]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    t1 = Ju[index] * (c / su) + Jv[index] * (s / sv)
    t1 -= np.einsum("ij,ij->i", t1, n)[:, None] * n
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(n, t1)
    ty = Jv[index] * (c / sv) - Ju[index] * (s / su)
    mirrored = np.einsum("ij,ij->i", ty, t2) < 0
    turn = np.where(mirrored, -turn[found], turn[found])
    if cells is not None:  # in the shapes' frames: y flips where the face's frame is mirrored
        cells = cells[found].copy()
        cells[mirrored, :, 1] *= -1
        cells[mirrored] = cells[mirrored, ::-1]     # counter-clockwise again
        turn = np.zeros(len(centre))
    return centre, n, t1, t2, turn, np.full(len(centre), face, dtype=np.int64), cells


def read_at(field, points, check=None):
    """A field's value at each point, read exactly (a one-sample grid per point, 0.1-0.8 ms each).
    A coarse grid over all the points, interpolated, misread a surface Face Mask by up to 0.45 on a
    320 mm part: the mask lives in a band thinner than the grid's cells."""
    out = np.empty(len(points))
    for i, p in enumerate(points):
        if check is not None and i % 256 == 0:
            check()
        out[i] = sample(field, p)
    return out


def _fit(step, length, even=False):
    """A step close to `step` that fits a whole number of times into `length` (even: an even
    number, so staggered rows close up)."""
    if even:
        return length / max(2, 2 * int(round(length / (2 * step))))
    return length / max(1, int(round(length / step)))


def _quantile(values, weights, q):
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return 1.0
    order = np.argsort(values[ok])
    v, w = values[ok][order], weights[ok][order]
    cum = np.cumsum(w)
    return float(v[min(np.searchsorted(cum, q * cum[-1]), len(v) - 1)])


def _locate(tri, q, eps=1e-9):
    """For 2D points q (m, 2): the triangle of `tri` (k, 3, 2) containing each (-1: none) and its
    barycentric coordinates (m, 3). Triangles are put in buckets of a grid by their boxes, so each
    point only tests the few triangles of its bucket."""
    m, k = len(q), len(tri)
    result, bary = np.full(m, -1, dtype=np.int64), np.zeros((m, 3))
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    box_lo, box_hi = lo.min(axis=0), hi.max(axis=0)
    n = max(1, min(int(math.sqrt(k)), 2048))
    size = np.maximum((box_hi - box_lo) / n, 1e-12)
    i0 = np.clip(((lo - box_lo) / size).astype(np.int64), 0, n - 1)
    i1 = np.clip(((hi - box_lo) / size).astype(np.int64), 0, n - 1)
    nx, ny = i1[:, 0] - i0[:, 0] + 1, i1[:, 1] - i0[:, 1] + 1
    count = nx * ny
    owner = np.repeat(np.arange(k), count)
    within = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    bx = i0[owner, 0] + within // ny[owner]
    by = i0[owner, 1] + within % ny[owner]
    bucket = bx * n + by
    sort = np.argsort(bucket, kind="stable")
    owner, bucket = owner[sort], bucket[sort]
    first = np.searchsorted(bucket, np.arange(n * n + 1))

    inside_box = np.all((q >= box_lo) & (q <= box_hi), axis=1)
    qi = np.flatnonzero(inside_box)
    if not len(qi):
        return result, bary
    cell = np.clip(((q[qi] - box_lo) / size).astype(np.int64), 0, n - 1)
    b = cell[:, 0] * n + cell[:, 1]
    cnt = first[b + 1] - first[b]
    point = np.repeat(qi, cnt)
    cand = owner[np.repeat(first[b], cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
    a, e1, e2 = tri[cand, 0], tri[cand, 1] - tri[cand, 0], tri[cand, 2] - tri[cand, 0]
    p = q[point] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = (p[:, 0] * e2[:, 1] - p[:, 1] * e2[:, 0]) / det
        b2 = (e1[:, 0] * p[:, 1] - e1[:, 1] * p[:, 0]) / det
    b0 = 1 - b1 - b2
    hit = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps) & np.isfinite(b1) & np.isfinite(b2)
    if hit.any():
        point, cand = point[hit], cand[hit]
        bb = np.column_stack([b0[hit], b1[hit], b2[hit]])
        _, at = np.unique(point, return_index=True)
        result[point[at]] = cand[at]
        bary[point[at]] = np.clip(bb[at], 0, 1) / np.clip(bb[at], 0, 1).sum(axis=1, keepdims=True)
    return result, bary


def _border_points(V, F, ids, selected, size):
    """Points along the selection's border: the edges between chosen and other triangles, and the
    sharp folds between two chosen faces."""
    segments = [border_segments(V, F, selected)] if selected.any() and not selected.all() else []
    m = trimesh.Trimesh(V, F, process=False)
    pairs, edges, bends = m.face_adjacency, m.face_adjacency_edges, m.face_adjacency_angles
    fold = selected[pairs[:, 0]] & selected[pairs[:, 1]] & (ids[pairs[:, 0]] != ids[pairs[:, 1]]) & (bends > FOLD)
    if fold.any():
        segments.append(V[edges[fold]])
    if not segments:
        return np.zeros((0, 3))
    spacing = float(np.clip(size / 8, 0.02, 0.25))
    return sample_segments(np.concatenate(segments), spacing)


def _near_border(holes, border, margin):
    """Holes whose outline comes closer than `margin` to the border (mask)."""
    out = np.zeros(len(holes), dtype=bool)
    if not len(holes) or not len(border):
        return out
    reach = holes.radius() * holes.scale + margin
    tree = cKDTree(border)
    found = tree.query_ball_point(holes.centre, reach, workers=-1)
    count = np.array([len(f) for f in found])
    if not count.sum():
        return out
    index = np.repeat(np.arange(len(holes)), count)
    x, y, _ = holes.local(border[np.concatenate(found).astype(np.int64)], index)
    d = holes.unit_sdf(x, y, index) * holes.scale[index]
    closest = np.full(len(holes), np.inf)
    np.minimum.at(closest, index, d)
    return closest < margin - 1e-9


def _crowded(holes, spacing, scattered=False):
    """(holes found twice, holes too close to a hole from another face, holes Randomness moved too
    close to a neighbour, pairs that overlap), the first three as masks of holes to drop. Found
    twice: on both ends of a cylinder's seam, or on two faces sharing one surface. Overlapping
    holes on one face are a setting to fix, so they stay (and are counted), unless `scattered`.
    Also sets holes.smallest_web."""
    n = len(holes)
    twice, close, crowded = (np.zeros(n, dtype=bool) for _ in range(3))
    if n < 2:
        return twice, close, crowded, 0
    r = holes.radius() * holes.scale
    pairs = cKDTree(holes.centre).query_pairs(2 * r.max() + max(spacing, MIN_WEB), output_type="ndarray")
    if not len(pairs):
        return twice, close, crowded, 0
    i, j = pairs[:, 0], pairs[:, 1]
    gap = np.linalg.norm(holes.centre[i] - holes.centre[j], axis=1)
    web = np.minimum(_web(holes, i, j), _web(holes, j, i))
    same = holes.face[i] == holes.face[j]
    dup = gap < 1e-3 * holes.shape.size
    within = web[same & ~dup]
    limit = min(MIN_WEB, float(within.min())) if len(within) else MIN_WEB
    why = np.where(dup, 0, np.where(same, 2, 1))     # which mask a dropped hole goes to
    bad = dup | (~same & (web < limit - 1e-9)) | (same & scattered & (web < MIN_WEB))
    masks = (twice, close, crowded)
    gone = np.zeros(n, dtype=bool)
    for a, b, k in sorted(zip(i[bad], j[bad], why[bad])):
        if not (gone[a] or gone[b]):
            masks[k][max(a, b)] = gone[max(a, b)] = True
    kept = ~gone[i] & ~gone[j]
    holes.smallest_web = float(web[kept].min()) if kept.any() else None
    return twice, close, crowded, int((kept & same & (web < 0)).sum())


def _web(holes, a, b):
    """Smallest distance from each hole a's outline to hole b (negative: they overlap)."""
    points = holes.outlines(a).reshape(-1, 3)
    index = np.repeat(b, OUTLINE)
    x, y, _ = holes.local(points, index)
    return (holes.unit_sdf(x, y, index) * holes.scale[index]).reshape(-1, OUTLINE).min(axis=1)


def _walls(holes, V, F):
    """Wall thickness straight under each shape's centre, mm (NaN where the ray finds no far side)."""
    from .checks import ray_thickness  # Warp: only needed here
    if not len(holes):
        return np.zeros(0)
    reach = float(np.linalg.norm(np.ptp(V, axis=0))) * 1.1 + 1.0
    return ray_thickness(V, F, holes.centre, holes.normal, reach).astype(np.float64)


def _depths(holes, V, F, pocket=None, height=None):
    """Sets each shape's depth below and reach above the face. pocket: per-shape pocket depths
    (None: through holes); height: per-shape heights of raised shapes (None: cut into the part).
    Returns the mask of through holes that found no wall behind them."""
    from .checks import ray_first_hit, ray_thickness  # Warp: only needed here
    n = len(holes)
    if not n:
        return np.zeros(0, dtype=bool)
    axis = holes.normal
    face_n = holes.face_normal if holes.face_normal is not None else axis
    # samples around the outline (in the plane across the axis), moved along the axis onto the
    # face's tangent plane: the same points unless the hole is tilted; `s` = how far along the axis
    around = holes.outlines(np.arange(n), RAYS, shrink=0.9)                    # (n, RAYS, 3)
    s = -np.einsum("nkd,nd->nk", around - holes.centre[:, None], face_n) / \
        np.einsum("nd,nd->n", axis, face_n)[:, None]
    around = around + s[..., None] * axis[:, None]
    up = np.repeat(axis, RAYS, axis=0)
    r = holes.radius() * holes.scale
    reach = float(np.linalg.norm(np.ptp(V, axis=0))) * 1.1 + 1.0
    if height is not None:  # raised shapes: into the wall as far as the face falls away under them, + JOIN
        hit, sign = ray_first_hit(V, F, around.reshape(-1, 3), -up, float(2 * r.max() + 1.0))
        drop = np.where((hit >= 0) & (sign > 0), hit, 0.0).reshape(n, RAYS)
        wall = ray_thickness(V, F, holes.centre, axis, reach)
        sink = np.maximum((drop - s).max(axis=1), 0.0) + JOIN
        holes.depth = np.where(np.isfinite(wall), np.minimum(sink, 0.9 * wall), sink)
        holes.outer = np.asarray(height, dtype=np.float64)
        return np.zeros(n, dtype=bool)
    hit, sign = ray_first_hit(V, F, around.reshape(-1, 3), up, float(2 * r.max() + OUTSIDE))
    rise = np.where((hit >= 0) & (sign < 0), hit, 0.0).reshape(n, RAYS)      # the face curves up there
    holes.outer = OUTSIDE + np.maximum((rise + s).max(axis=1), 0.0)
    starts = np.concatenate([holes.centre, around.reshape(-1, 3)])
    t = ray_thickness(V, F, starts, np.concatenate([axis, up]), reach)
    centre, ring = t[:n], t[n:].reshape(n, RAYS)
    deepest = np.fmax(centre, np.where(np.isfinite(ring), ring - s, -np.inf).max(axis=1))
    through = deepest + EXTRA
    ring_most = np.fmax(centre, np.where(np.isfinite(ring), ring, -np.inf).max(axis=1))
    if pocket is not None:  # a pocket as deep as the wall (or deeper) goes through, but no further
        pocket = np.asarray(pocket, dtype=np.float64)
        deeper = np.isfinite(deepest) & (pocket > deepest - 0.05)
        holes.depth = np.where(deeper, through, pocket)
        if deeper.any():
            holes.note(f"{int(deeper.sum())} pockets are as deep as the wall: cut through")
        return np.zeros(n, dtype=bool)
    holes.depth = np.where(np.isfinite(through), through, 0.0)
    # how much of each ring ray is wall: a sample in the tangent plane of a convex face starts in the
    # air (14 mm slots around a 30 mm cylinder: 0.7 mm above it), which is no thicker wall
    enter, side = ray_first_hit(V, F, around.reshape(-1, 3), -up, float(2 * r.max() + OUTSIDE))
    air = np.where((enter >= 0) & (side > 0), enter, 0.0).reshape(n, RAYS)
    wall = np.where(np.isfinite(ring), ring - air, np.nan)
    ring_most = np.fmax(centre, np.where(np.isfinite(wall), wall, -np.inf).max(axis=1))
    thinnest = np.fmin(centre, np.where(np.isfinite(wall), wall, np.inf).min(axis=1))
    varies = int(np.sum(np.isfinite(centre) & (ring_most > 1.2 * thinnest)))
    if varies:
        holes.note(f"{varies} holes go through a wall whose thickness varies more than 20% across "
                   f"the hole (cut to its thickest part)")
    return ~np.isfinite(centre)


# --- the cutter and the relief --------------------------------------------------------------------
class _Shapes:
    """The placed shapes as plain arrays (they travel inside worker tasks) and the grid samples
    around each one."""

    def __init__(self, holes):
        self.shape = holes.shape
        self.cells = None if holes.cells is None else np.ascontiguousarray(holes.cells, dtype=np.float64)
        self.shape_b = holes.shape_b
        self.morph = None if holes.morph is None else np.ascontiguousarray(holes.morph, dtype=np.float64)
        self.arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in
                       (holes.centre, holes.normal, holes.t1, holes.t2, holes.scale, holes.turn, holes.outer,
                        holes.depth)]
        self.lo, self.hi = holes.boxes() if len(holes) else (np.zeros((0, 3)), np.zeros((0, 3)))

    def __len__(self):
        return len(self.arrays[0])

    def key(self, *extra):
        second = None if self.shape_b is None else self.shape_b.key()
        h = hashlib.sha1(repr((self.shape.key(), second) + extra).encode())
        for a in self.arrays + [a for a in (self.cells, self.morph) if a is not None]:
            h.update(a.tobytes())
        return h.hexdigest()

    def _one(self, shape, k, u, v):
        if shape.kind in P2.CELLS:
            return P2.polygon_sdf(u, v, self.cells[k]) - shape.corner
        return shape.sdf(u, v)

    def _unit(self, k, u, v):
        """Shape k's 2D distance at scale 1 (as Holes.unit_sdf)."""
        first = self._one(self.shape, k, u, v)
        if self.shape_b is None:
            return first
        return first + self.morph[k] * (self._one(self.shape_b, k, u, v) - first)

    def bounds(self, default):
        return (self.lo.min(axis=0), self.hi.max(axis=0)) if len(self) else default

    def around(self, grid, reach=0.0):
        """For every shape whose box touches the grid: (shape number, slices of the grid, distance
        across (mm: the 2D shape's signed distance), distance along its axis), on the samples in its
        box plus PAD samples or `reach` mm, whichever is more. Positions come from the global sample
        index: the same in every block, bit for bit."""
        centre, normal, t1, t2, scale, turn, _, _ = self.arrays
        if not len(centre):
            return
        start = np.asarray(grid.start)
        pad = max(PAD * grid.voxel, float(reach))
        first = np.ceil((self.lo - pad - grid.base) / grid.voxel).astype(np.int64) - start   # sample indices
        last = np.floor((self.hi + pad - grid.base) / grid.voxel).astype(np.int64) - start + 1
        first, last = np.maximum(first, 0), np.minimum(last, np.asarray(grid.shape))
        axes = grid.axes()
        for k in np.flatnonzero(np.all(last > first, axis=1)):
            i0, i1 = first[k], last[k]
            x = axes[0][i0[0]:i1[0], None, None] - centre[k, 0]
            y = axes[1][None, i0[1]:i1[1], None] - centre[k, 1]
            z = axes[2][None, None, i0[2]:i1[2]] - centre[k, 2]
            a = x * t1[k, 0] + y * t1[k, 1] + z * t1[k, 2]
            b = x * t2[k, 0] + y * t2[k, 1] + z * t2[k, 2]
            along = x * normal[k, 0] + y * normal[k, 1] + z * normal[k, 2]
            c, s = math.cos(turn[k]), math.sin(turn[k])
            u, v = (a * c + b * s) / scale[k], (b * c - a * s) / scale[k]
            across = self._unit(k, u, v) * scale[k]
            yield k, tuple(slice(i0[d], i1[d]) for d in range(3)), across, along


def cutter(holes, reach=0.0):
    """Implicit body of the holes (negative inside): subtract it from the part (holes, pockets), or
    unite it with the part (raised shapes). Its distances are exact `reach` mm around every hole
    (at least PAD samples): a rounded or chamfered edge needs them as far out as its radius;
    `field.regrow(radius)` gives that cutter."""
    data = _Shapes(holes)
    outer, depth = data.arrays[6], data.arrays[7]
    reach = float(reach)

    def fn(ctx):
        out = np.full(ctx.grid.shape, FAR, dtype=np.float32)
        for k, box, across, along in data.around(ctx.grid, reach):
            value = np.maximum(across, np.maximum(along - outer[k], -depth[k] - along)).astype(np.float32)
            block = out[box]
            np.minimum(block, value, out=block)
        return ctx.xp.asarray(out)
    bounds = data.bounds(holes.bounds)
    field = Field(fn, _digest("holes", data.key(), reach), f"holes({len(data)})", bounds, cache=True)
    field.extent = bounds
    field.regrow = lambda radius: cutter(holes, max(reach, 1.25 * float(radius)))
    return field


def relief(holes, softness, band):
    """Field 0..1 for a Surface Pattern: 1 on each shape's top, falling to 0 at its outline over
    `softness` mm (a smooth step inside the outline, so the base keeps the shape's size; 0 = a
    sharp edge), the same all along the shape's axis from `band` mm above the face to `band` mm
    into it, but at most half the wall under the shape (holes.wall): an offset by it must not move
    the wall's far side too (2.5 mm wall, band 2, dimples 0.5 deep: the far side was touched). 0
    between the shapes and farther away. Overlapping shapes: the higher value."""
    n = len(holes)
    wall = holes.wall if holes.wall is not None else np.full(n, np.nan)
    holes.outer = np.full(n, float(band))
    holes.depth = np.where(np.isfinite(wall), np.minimum(float(band), 0.5 * np.nan_to_num(wall)), float(band))
    data = _Shapes(holes)
    soft, band = float(softness), float(band)
    inner = data.arrays[7]

    def fn(ctx):
        out = np.zeros(ctx.grid.shape, dtype=np.float32)
        for k, box, across, along in data.around(ctx.grid):
            if soft > 0:
                t = np.clip(-across / soft, 0.0, 1.0)
                value = t * t * (3 - 2 * t)
            else:
                value = (across < 0).astype(np.float64)
            value = np.where((along <= band) & (along >= -inner[k]), value, 0.0).astype(np.float32)
            block = out[box]
            np.maximum(block, value, out=block)
        return ctx.xp.asarray(out)
    field = Field(fn, _digest("relief", data.key(soft, band)), f"relief({len(data)})", cache=True, range=(0.0, 1.0))
    field.extent = data.bounds(holes.bounds)
    return field
