"""Copies of a CAD part laid out on picked faces of another body (node Pattern on Faces), with the
layouts, Join Faces, Align and Randomness of the face patterns (engine/holes.py). Each copy stands
on the surface: the part's Up along the face's normal and its own x along the pattern's rows
(turned by Rotation), at a size of Size scale (and of its cell, below).

The checks use the part's footprint: the convex outline of its vertices seen from above (along Up,
in its own x and y). Copies whose footprint comes closer than Edge margin to the border of the
chosen faces are left out whole, and so are copies found twice (on a seam) or crowding a copy on
another face.

Girth (holes.GIRTH), on a face that closes around an axis whose girth changes (a cone, a vase):
"same size" lays rows the same distance apart along the surface, each with as many copies as fit
around it (fewer where it is narrower); "same count" puts as many in every row (the copies keep
their size, the gaps grow where it is wider); "scale" puts as many in every row and the copies and
the rows grow and shrink with the girth, like the scales of a pinecone.

Fit to cells "voronoi": every copy gets the Voronoi cell of its centre among the others, drawn in
its own tangent plane (so cells work across faces and seams) and cut Edge margin in from the border
of the chosen faces where it faces the copy, moves to the middle of its cell and takes the largest
size whose footprint fits inside it, Gap / 2 from its edge, so neighbours stay at least Gap apart;
with Turn to fit it also turns to the angle (in TURNS steps) at which it fits largest. With
Randomness the cells, and so the copies, all differ in size. Fit (FITS) says how a copy fills its
cell: "scale" as drawn (uniformly); "stretch" also stretched (area kept, up to STRETCH) and moved to
where it grows largest (`_fit_shapes`: the best few turns and stretches at the cell's middle, then a
pattern search for the shift); "fill" bent so its outline is the cell (`_Warp`: the edge profile
keeps its width, the corners are rounded as wide as it), its triangles split where the bend needs
them (`_refine`) and, on a curved face, wrapped onto the surface."""
import math

import igl
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree

from . import holes as H
from . import pattern2d as P2
from .fields import Field
from .transform import transformed
from .unfold import corner_angles

UPS = ("auto", "x", "y", "z", "-x", "-y", "-z")
BASES = ("bottom centre", "origin")
CELLS = ("none", "voronoi")
FITS = ("scale", "stretch", "fill")              # how a copy fills its Voronoi cell
CROWDED = ("shrink", "leave out", "overlap")   # copies that would come closer than Gap
EDGES = ("leave out", "shrink")                # copies that would cross the edge of the chosen faces
EDGE_TRIM = 0.95     # ... are shrunk anyway when this much smaller fits (a fitted end row only just touching)
AUTO_SPACING = 1.1   # Spacing auto: the footprint's larger side times this
SIDES = 32           # a cell starts as this polygon round its copy, before its neighbours cut it
REACH = 0.6          # ... of radius (REACH + REACH_RANDOM x Randomness) x Spacing: how far a cell at the
REACH_RANDOM = 0.4   # edge of the pattern, with no neighbour beyond it, reaches
TURNS = 48           # Turn to fit tries every copy at this many angles all round (7.5° apart)
EDGE_FACING = 0.3    # a border cuts a cell only where its surface faces the copy's this much (cos)
STRETCH = 1.7        # Fit stretch: a copy gets at most this much longer one way than the other (its area kept)
STRETCH_STEPS = 6    # ... tried in this many steps from 1, along its cell's long axis and STRETCH_TILT
STRETCH_TILT = 20.0  # degrees either side
KEEP = 4             # the best turns and stretches of each copy whose place in its cell is then searched
SEARCH = 40          # ... in this many halving steps
DIRECTIONS = np.column_stack([np.cos(np.arange(16) * np.pi / 8), np.sin(np.arange(16) * np.pi / 8)])
CIRCLE = np.column_stack([np.cos(np.arange(96) * np.pi / 48), np.sin(np.arange(96) * np.pi / 48)])
CORNER = 0.5         # Fit fill: corners rounded as wide as the part's edge, at most this share of the cell's
                     # inner radius
BAND = 1.25          # ... the band whose distance from the outline is kept: the edge's width times this
BAND_MIN = 0.05      # ... plus this, mm
BAND_SHARE = 0.8     # ... at most this share of the inner radius of the footprint and of the cell
FILL_TOL = 0.02      # mm a filled copy's triangles may stray from the bent shape before they are split
FILL_ROUNDS = 8      # ... in at most this many rounds
FILL_SHORTEST = 0.1  # mm (seen from above, at the copy's size): shorter edges are not split


class CopyError(ValueError):
    pass


def frame(bodies, up="auto", base="bottom centre"):
    """Where and how the part stands: (its base point, the rows of its directions that become the
    pattern's x, y and the face's normal (right-handed), Up as chosen, why auto chose it or None)."""
    if up not in UPS:
        raise CopyError(f"Up must be one of {', '.join(UPS)}")
    if base not in BASES:
        raise CopyError(f"Base must be one of {', '.join(BASES)}")
    lo = np.min([b.bounds[0] for b in bodies], axis=0)
    hi = np.max([b.bounds[1] for b in bodies], axis=0)
    why = None
    if up == "auto":
        up, why = _auto_up(lo, hi)
    k, sign = "xyz".index(up[-1]), -1.0 if up.startswith("-") else 1.0
    e_up = sign * np.eye(3)[k]
    e1 = np.eye(3)[1 if k == 0 else 0]          # its x, or y when it stands along x
    point = np.zeros(3)
    if base == "bottom centre":
        point = (lo + hi) / 2
        point[k] = lo[k] if sign > 0 else hi[k]
    return point, np.array([e1, np.cross(e_up, e1), e_up]), up, why


def _auto_up(lo, hi):
    """The direction a part drawn standing on a plane through the origin stands up along: its box
    starts (or ends) at 0 along it; of several, the thinnest; of none, the thinnest side."""
    size = hi - lo
    tol = 0.02 * float(size.max()) + 1e-6
    options = []
    for k in range(3):
        if abs(lo[k]) <= tol and hi[k] > tol:
            options.append((size[k], "xyz"[k]))
        elif abs(hi[k]) <= tol and lo[k] < -tol:
            options.append((size[k], "-" + "xyz"[k]))
    if options:
        up = min(options)[1]
        return up, f"the part stands on {up[-1]} = 0"
    up = "xyz"[int(np.argmin(size))]
    return up, f"its thinnest side ({up}): it does not stand on a plane through the origin"


def footprint(bodies, point, basis):
    """The part's outline seen from above: the convex hull of its vertices in its own x, y around
    the base point, (m, 2) counter-clockwise."""
    flat = (np.concatenate([np.asarray(b.vertices) for b in bodies]) - point) @ basis[:2].T
    try:
        hull = ConvexHull(flat)
    except Exception:  # all in a line or a point: no area seen from above
        raise CopyError("the part has no area seen from above: is Up right?") from None
    return flat[hull.vertices]


def place(mesh, chosen, part, up="auto", base="bottom centre", layout="staggered", spacing=0.0, align="as in CAD",
          angle=0.0, offset=(0.0, 0.0), margin=0.0, cells="none", gap=1.0, randomness=0.0, seed=0,
          alternate="none", join="tangent", girth="same size", size_scale=1.0, rotation=0.0, lift=0.0,
          min_scale=0.2, turn_fit=False, crowded="shrink", edges="leave out", fit="scale", limit=1000,
          check=None):
    """Where copies of `part` (a list of bodies, placed together) go on the CAD faces `chosen` of
    the tessellated body `mesh`: a Placed, which unpacks as (matrices (n, 4, 4), each copy's size,
    log lines (text, level)) and makes the copies (Placed.bodies). spacing 0 = auto; size_scale and
    rotation (degrees) may be fields, read at each copy's centre; turn_fit: with cells, every copy
    turned to fit its cell best (from Rotation on); fit (FITS): how it fills its cell. crowded
    (CROWDED): copies closer than Gap shrink, are left out or overlap; edges (EDGES): copies
    crossing the edge of the faces are left out or shrink to fit inside."""
    for value, names, what in ((layout, P2.LAYOUTS, "Layout"), (alternate, P2.ALTERNATE, "Alternate"),
                               (align, H.ALIGN, "Align"), (join, H.JOIN_FACES, "Join Faces"),
                               (girth, H.GIRTH, "Girth"), (cells, CELLS, "Fit to cells"),
                               (crowded, CROWDED, "Crowded"), (edges, EDGES, "At edges"), (fit, FITS, "Fit")):
        if value not in names:
            raise CopyError(f"{what} must be one of {', '.join(names)}")
    if spacing < 0 or gap < 0 or min_scale < 0:
        raise CopyError("Spacing, Gap and Min scale must be 0 or more")
    ids = np.asarray(mesh.face_attributes.get("cad_face", []))
    uv = mesh.face_attributes.get("cad_uv")
    if uv is None or len(ids) != len(mesh.faces) or len(uv) != len(mesh.faces):
        raise CopyError("the surface has no CAD surface data: connect a body from Import STEP")
    point, basis, up_is, why = frame(part, up, base)
    hull = footprint(part, point, basis)
    width = float(np.ptp(hull, axis=0).max())
    spacing = float(spacing) or AUTO_SPACING * width
    inset = gap / 2 if cells == "voronoi" else None
    if inset is not None and spacing <= gap:
        raise CopyError(f"Gap ({gap:g} mm) leaves no room in cells {spacing:g} mm apart")

    # the checks' shape: every copy's footprint as its own outline (Holes.cells), at its size
    holes = H.Holes(P2.Shape("voronoi", width), mesh.bounds, "Pattern on Faces")
    if why:
        holes.note(f"Up: {up_is}, chosen as {why}")
    chosen = sorted(int(f) for f in chosen)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    Fc = np.asarray(mesh.faces, dtype=np.int64)
    surfaces = np.asarray(mesh.metadata.get("cad_surfaces", np.zeros((0, 3))))
    sx, sy = P2.steps(P2.Shape("round", width), spacing, layout, alternate)
    axis = None if align == "as in CAD" else np.eye(3)[H.ALIGN.index(align) - 1]
    if not isinstance(randomness, Field):
        randomness = min(max(float(randomness), 0.0), 1.0)
    # what reaches across the rows from each point (rows on a face that runs around are fitted between its ends)
    if isinstance(rotation, Field) or (inset is not None and turn_fit):
        a = 2 * np.pi * np.arange(SIDES) / SIDES
        reach = float(np.linalg.norm(hull, axis=1).max()) * np.column_stack([np.cos(a), np.sin(a)])
    else:
        c, s = math.cos(math.radians(float(rotation))), math.sin(math.radians(float(rotation)))
        reach = hull @ np.array([[c, s], [-s, c]])
    if isinstance(size_scale, Field):   # at most as large as the field gets
        most = max(1.0, float(size_scale.range[1])) if size_scale.range else 1.0
    else:
        most = max(float(size_scale), 0.0)
    tall = float(((np.concatenate([np.asarray(b.vertices) for b in part]) - point) @ basis[2]).max())
    height = max(tall * most + float(lift), 0.0)   # how far each copy stands out of the surface
    parts = []
    for face, tri, corner_uv, period, name, flat in H._frames(holes, V, Fc, ids, uv, surfaces, chosen, join, check):
        if check is not None:
            check()
        found = H._face_holes(holes, face, V, Fc[tri], corner_uv, period, layout, sx, sy, axis, angle, offset,
                              randomness, seed, None, check, alternate, None, name=name, flat=flat, girth=girth,
                              footprint=reach * most, margin=margin, height=height)
        if found is not None:
            parts.append(found)
    if not parts:
        holes.note("no copy falls on the chosen faces (Spacing larger than the faces?)", "error")
        return Placed(holes, point, basis, float(lift))
    for name, i in (("centre", 0), ("normal", 1), ("t1", 2), ("t2", 3), ("turn", 4), ("face", 5), ("scale", 7)):
        setattr(holes, name, np.concatenate([p[i] for p in parts]))
    holes.grow = holes.scale.copy()                  # Girth scale: the Gap grows and shrinks with the rest
    twice = _twice(holes.centre, 1e-3 * width)       # on both ends of a seam
    holes.take(~twice)
    if len(holes) > 4 * limit:
        raise CopyError(f"about {len(holes)} copies: at most {limit} (a larger Spacing, or fewer faces)")
    if isinstance(rotation, Field):
        turned = np.nan_to_num(H.read_at(rotation, holes.centre, check))
        holes.note(f"Rotation from the field: {turned.min():.0f} to {turned.max():.0f}°")
    else:
        turned = float(rotation)
    holes.turn = holes.turn + np.radians(turned)

    vanished, fill = 0, None
    if inset is not None:  # every copy as large as its cell allows
        scattered = float(randomness.range[1]) if isinstance(randomness, Field) and randomness.range else \
            1.0 if isinstance(randomness, Field) else randomness
        bound = (REACH + REACH_RANDOM * min(max(scattered, 0.0), 1.0)) * spacing
        surface = _Surface(V, Fc, ids, chosen)
        edge = _border_lines(V, Fc, ids, chosen, width)
        heights = None
        before = {name: getattr(holes, name).copy() for name in ("centre", "normal", "t1", "t2", "face", "turn")}
        for _ in range(3):   # again where copies lean over a neighbouring face: from the start, with their heights
            for name, value in before.items():
                setattr(holes, name, value.copy())
            polygons = _cells(holes, inset, bound, edge, float(margin), heights)
            polygons = _to_middles(holes, polygons, bound, surface)
            if fit == "scale":
                holes.scale = _fits(hull, holes.turn, polygons)
                if turn_fit:
                    start = holes.turn.copy()
                    for k in range(1, TURNS):
                        if check is not None and k % 8 == 0:
                            check()
                        turn = start + 2 * np.pi * k / TURNS
                        fits = _fits(hull, turn, polygons)
                        better = np.nan_to_num(fits, nan=-1.0) > np.nan_to_num(holes.scale, nan=-1.0) * (1 + 1e-9)
                        holes.scale = np.where(better, fits, holes.scale)
                        holes.turn = np.where(better, turn, holes.turn)
            else:
                turns = 2 * np.pi * np.arange(TURNS) / TURNS if turn_fit else np.zeros(1)
                ratios = np.linspace(1.0, STRETCH, STRETCH_STEPS) if fit == "stretch" else np.ones(1)
                size, holes.turn, stretch, shift = _fit_shapes(hull, holes.turn, polygons, turns, ratios, check)
                if fit == "stretch":
                    _shift(holes, polygons, shift, surface)
                    holes.scale, holes.stretch = size, stretch
                    holes.cells = np.einsum("nij,mj->nmi", stretch, hull)
                else:   # to the middle of its cell, then the cells again round the copies' new centres and frames
                    _shift(holes, polygons, _inner_circles(polygons)[1], surface)
                    polygons = _cells(holes, inset, bound, edge, float(margin), heights)
                    depth = _edge_depth(part, point, basis, hull)
                    holes.scale, holes.cells = _fill_outlines(holes, hull, polygons, depth)
                    fill = (hull, depth, surface)
            tall_now = np.maximum(tall * most * np.nan_to_num(holes.scale) + float(lift), 0.0)
            if heights is None and not _leaning(holes, edge, bound + max(float(margin), 0.0) + tall_now / 0.3):
                break
            if heights is not None and (tall_now <= heights + 1e-9).all():
                break
            heights = tall_now if heights is None else np.maximum(heights, tall_now)
        empty = ~np.isfinite(holes.scale)
        vanished = int(empty.sum())
        holes.take(~empty)
    if isinstance(size_scale, Field):
        holes.scale = holes.scale * np.nan_to_num(H.read_at(size_scale, holes.centre, check), nan=0.0)
    else:
        holes.scale = holes.scale * float(size_scale)
    if holes.cells is None:
        holes.cells = np.repeat(hull[None], len(holes), axis=0)

    border = H._border_points(V, Fc, ids, np.isin(ids, chosen), width)
    at_edges = _shrink_to_edges(holes, border, margin, edges == "shrink")
    near_border = H._near_border(holes, border, margin)
    holes.take(~near_border)
    crowding, shrunk = np.zeros(len(holes), dtype=bool), 0
    if inset is None and crowded != "overlap":   # (cells keep the Gap already)
        crowding, shrunk = _crowd(holes, gap, crowded == "leave out")
        holes.take(~crowding)
    small = holes.scale < max(float(min_scale), 1e-3)
    holes.take(~small)
    seam, too_close, _, overlaps = H._crowded(holes, spacing)
    holes.take(~(seam | too_close))
    n = len(holes)
    if n > limit:
        raise CopyError(f"{n} copies is too many (at most {limit}): a larger Spacing, or fewer faces")

    dropped = [(int(near_border.sum()), f"near the border (Edge margin {margin:g} mm)"),
               (int(too_close.sum()), "too close to copies on a neighbouring face"),
               (int(crowding.sum()), f"closer than the Gap ({gap:g} mm) to a neighbour"),
               (vanished, "cells too small for the Gap"),
               (int(small.sum()), f"smaller than Min scale {float(min_scale):g}x")]
    dropped = [f"{k} {text}" for k, text in dropped if k]
    holes.note(f"{n} copies on {len(chosen)} face{'s' if len(chosen) != 1 else ''}, Spacing {spacing:.3g} mm"
               + (f"; left out: {', '.join(dropped)}" if dropped else ""), "info" if n else "error")
    if n and np.ptp(holes.scale) > 1e-6:
        holes.note(f"sizes {holes.scale.min():.2f}x to {holes.scale.max():.2f}x")
    if shrunk:
        holes.note(f"{shrunk} copies shrunk to keep the Gap ({gap:g} mm) to their neighbours")
    if at_edges:
        holes.note(f"{at_edges} copies shrunk to fit inside the edge of the faces")
    if overlaps:
        nest = " (where a row has fewer copies than the next, staggered rows cannot nest: Layout grid keeps " \
               "rows a whole Spacing apart)" if holes.uneven_rows and layout == "staggered" else ""
        holes.note(f"{overlaps} pairs of copies overlap: increase Spacing or reduce Size scale{nest}", "warning")
    elif holes.smallest_web is not None and n > 1:
        holes.note(f"narrowest gap between copies {holes.smallest_web:.2f} mm")
    if holes.stretch is not None and n:
        ratio = np.linalg.svd(holes.stretch, compute_uv=False)
        holes.note(f"stretched up to {(ratio[:, 0] / ratio[:, 1]).max():.2f}:1")
    return Placed(holes, point, basis, float(lift), fill)


class Placed:
    """Where the copies go (place()): matrices (n, 4, 4) that move the part onto each copy (Fit fill:
    each copy's frame at its size, before it is bent onto its cell), each copy's size and the log's
    lines; unpacks as (matrices, sizes, lines). bodies() makes the copies."""

    def __init__(self, holes, point, basis, lift, fill=None):
        self.matrices = _matrices(holes, point, basis, lift)
        self.sizes, self.lines = holes.scale, holes.lines
        self._fill = None if fill is None else _Fill(holes, point, basis, lift, *fill)

    def __iter__(self):
        return iter((self.matrices, self.sizes, self.lines))

    def __getitem__(self, k):
        return (self.matrices, self.sizes, self.lines)[k]

    def __len__(self):
        return len(self.matrices)

    def bodies(self, part, check=None, progress=None):
        """Every copy's bodies (the part's, in its order), copy by copy. progress(done, total)."""
        if self._fill is None:
            return [transformed(b, m) for m in self.matrices for b in part]
        return self._fill.bodies(part, check, progress)


def _largest(fits, count, steps=22):
    """The largest factor 0..1 for each of `count` items at which fits(factors) holds (bisection;
    fits must hold for all smaller factors too)."""
    whole = fits(np.ones(count))
    lo, hi = np.zeros(count), np.ones(count)
    for _ in range(steps):
        mid = (lo + hi) / 2
        ok = fits(mid)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return np.where(whole, 1.0, lo)


def _in_frames(holes, points, k, factor):
    """The footprint's signed distance (mm) of points (m, 3) from copies k (m,) at their size times
    `factor` (m,), in each copy's tangent plane."""
    d = points - holes.centre[k]
    a, b = np.einsum("ij,ij->i", d, holes.t1[k]), np.einsum("ij,ij->i", d, holes.t2[k])
    c, s, size = np.cos(holes.turn[k]), np.sin(holes.turn[k]), holes.scale[k] * factor
    return holes.unit_sdf((a * c + b * s) / size, (b * c - a * s) / size, k) * size


def _shrink_to_edges(holes, border, margin, any_size):
    """Shrinks the copies whose footprint comes closer than `margin` to the border points, to the
    largest size that keeps it: all of them (any_size), else only those that need less than
    EDGE_TRIM. Returns how many shrank."""
    near = np.flatnonzero(H._near_border(holes, border, margin))
    if not len(near):
        return 0
    found = cKDTree(border).query_ball_point(holes.centre[near], holes.radius()[near] * holes.scale[near] + margin)
    owner = np.repeat(np.arange(len(near)), [len(f) for f in found])
    points = border[np.concatenate(found).astype(np.int64)]

    def fits(factor):
        closest = np.full(len(near), np.inf)
        np.minimum.at(closest, owner, _in_frames(holes, points, near[owner], factor[owner]))
        return closest >= margin - 1e-9
    factor = _largest(fits, len(near))
    take = (factor > 0) & (any_size | (factor >= EDGE_TRIM))
    holes.scale[near[take]] *= factor[take]
    return int(take.sum())


def _crowd(holes, gap, leave_out):
    """Copies whose footprints come closer than `gap` (times the smaller growth with the girth of the
    two, holes.grow): shrinks both of each such pair to the largest size that keeps it (each copy the
    most any of its pairs asks), or with leave_out marks the later of each pair to drop. Returns
    (copies to drop, copies shrunk)."""
    n = len(holes)
    drop = np.zeros(n, dtype=bool)
    if n < 2:
        return drop, 0
    r = holes.radius() * holes.scale
    pairs = cKDTree(holes.centre).query_pairs(2 * r.max() + gap, output_type="ndarray")
    if not len(pairs):
        return drop, 0
    i, j = pairs[:, 0], pairs[:, 1]
    near = (np.linalg.norm(holes.centre[i] - holes.centre[j], axis=1) < r[i] + r[j] + gap) & \
        (np.einsum("ij,ij->i", holes.normal[i], holes.normal[j]) > 0)     # not across a thin wall
    i, j = i[near], j[near]
    room = gap * np.minimum(holes.grow[i], holes.grow[j])

    def apart(factor):
        return np.minimum(_outline_gap(holes, i, j, factor), _outline_gap(holes, j, i, factor))
    bad = apart(np.ones(len(i))) < room - 1e-9
    i, j, room = i[bad], j[bad], room[bad]
    if not len(i):
        return drop, 0
    if leave_out:
        for a, b in sorted(zip(i, j)):
            if not (drop[a] or drop[b]):
                drop[max(a, b)] = True
        return drop, 0
    factor = _largest(lambda f: apart(f) >= room - 1e-9, len(i))
    each = np.ones(n)
    np.minimum.at(each, i, factor)
    np.minimum.at(each, j, factor)
    holes.scale = holes.scale * each
    return drop, int((each < 1).sum())


def _outline_gap(holes, a, b, factor):
    """How far points on the outlines of copies a come from copies b (negative: inside), both at
    their size times `factor`."""
    o = holes.unit_outlines(a, H.OUTLINE)
    c, s = np.cos(holes.turn[a])[:, None], np.sin(holes.turn[a])[:, None]
    size = (holes.scale[a] * factor)[:, None]
    x, y = (o[..., 0] * c - o[..., 1] * s) * size, (o[..., 0] * s + o[..., 1] * c) * size
    points = holes.centre[a][:, None] + x[..., None] * holes.t1[a][:, None] + y[..., None] * holes.t2[a][:, None]
    m = o.shape[1]
    d = _in_frames(holes, points.reshape(-1, 3), np.repeat(b, m), np.repeat(factor, m))
    return d.reshape(-1, m).min(axis=1)


def _twice(points, tol):
    """Points within `tol` of an earlier one (mask)."""
    out = np.zeros(len(points), dtype=bool)
    if len(points) > 1:
        for i, j in cKDTree(points).query_pairs(tol, output_type="ndarray"):
            if not out[min(i, j)]:
                out[max(i, j)] = True
    return out


def _cells(holes, inset, bound, edge=None, margin=0.0, heights=None):
    """Every copy's Voronoi cell among the others, drawn in its tangent plane (frame t1, t2, not
    turned) and moved in by `inset`: (n, m, 2) counter-clockwise round the copy, padded by repeating
    the last vertex; NaN where nothing is left. Copies facing the other way (the far side of a
    wall) are no neighbours. `edge` (_border_lines) cuts the cells `margin` in from the border where
    it faces the copy (a straight or outward-bent border exactly; one that bends in, round a cut-out,
    only where it faces the copy, the rest is left to the edge checks): along the plane of the face
    beyond it where that meets the chosen ones at a fold (an end of a vase), farther in by as much as
    a copy `heights` (n,) tall standing on its normal would lean over that plane."""
    a = 2 * np.pi * np.arange(SIDES) / SIDES
    start = bound * np.column_stack([np.cos(a), np.sin(a)])
    centre = holes.centre
    near = cKDTree(centre).query_ball_point(centre, 2 * bound)
    if edge is not None and len(edge[0]):
        points, way_out, facing, end = edge
        reach = bound + max(margin, 0.0) + (0.0 if heights is None else heights / 0.3)   # 0.3: see `length`
        near_edge = cKDTree(points).query_ball_point(centre, reach)
    else:
        near_edge = [[]] * len(centre)
    out = []
    for i, others in enumerate(near):
        others = np.array([j for j in others if j != i], dtype=np.int64)
        poly = start
        if len(others):
            others = others[holes.normal[others] @ holes.normal[i] > 0]
            d = centre[others] - centre[i]
            dx, dy = d @ holes.t1[i], d @ holes.t2[i]
            length = np.hypot(dx, dy)
            for k in np.argsort(length):
                if length[k] < 1e-9:
                    continue
                poly = P2._clip(poly, np.array([dx[k], dy[k]]) / length[k], length[k] / 2 - inset)
                if len(poly) < 3:
                    break
        k = np.asarray(near_edge[i], dtype=np.int64)
        k = k[facing[k] @ holes.normal[i] > EDGE_FACING] if len(k) else k
        if len(k) and len(poly) >= 3:
            v = way_out[k] @ np.stack([holes.t1[i], holes.t2[i]], axis=1)
            length = np.linalg.norm(v, axis=1)
            k, v, length = k[length > 0.3], v[length > 0.3], length[length > 0.3]   # not a border under the copy
            b = points[k] - centre[i]
            along = np.einsum("ij,ij->i", way_out[k], b) / length                # to the border's line, mm
            faces_it = (along > max(margin, 0.0) + 1e-9) & (along >= 0.5 * np.linalg.norm(b, axis=1))
            lean = np.zeros(len(k))
            if heights is not None:
                lean = np.where(end[k], np.maximum(np.maximum(way_out[k] @ holes.normal[i],
                                                              np.einsum("ij,ij->i", way_out[k], facing[k])), 0.0), 0.0)
                lean *= heights[i] / length
            lines = np.unique(np.round(np.column_stack([v / length[:, None], along - margin - lean])[faces_it], 9),
                              axis=0)
            for vx, vy, offset in lines:
                poly = P2._clip(poly, np.array([vx, vy]), offset)
                if len(poly) < 3:
                    break
        out.append(poly)
    m = max([len(p) for p in out] + [3])
    cells = np.full((len(out), m, 2), np.nan)
    for i, p in enumerate(out):
        if len(p) >= 3:
            cells[i, :len(p)] = p
            cells[i, len(p):] = p[-1]
    return cells


def _to_middles(holes, cells, bound, surface):
    """Moves every copy whose cell is closed by its neighbours (it no longer reaches `bound`) to the
    middle of its cell, back onto the surface; returns the cells round the copies' new centres."""
    middle = P2.centroids(cells)
    radius = np.linalg.norm(cells, axis=2)
    closed = np.isfinite(middle).all(axis=1) & ~(np.nan_to_num(radius) >= bound * (1 - 1e-9)).any(axis=1)
    if not closed.any():
        return cells
    return _shift(holes, cells, np.where(closed[:, None], middle, 0.0), surface)


def _shift(holes, cells, shift, surface):
    """Moves the copies by `shift` (n, 2) in their tangent planes and back onto the surface (their
    frames turned with its normal); returns the cells round the copies' new centres."""
    i = np.flatnonzero(np.isfinite(shift).all(axis=1) & (np.abs(shift) > 0).any(axis=1))
    if not len(i):
        return cells
    moved = holes.centre[i] + shift[i, :1] * holes.t1[i] + shift[i, 1:] * holes.t2[i]
    at, normal, face = surface.closest(moved)
    t1 = holes.t1[i] - np.einsum("ij,ij->i", holes.t1[i], normal)[:, None] * normal
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    holes.centre[i], holes.normal[i], holes.face[i] = at, normal, face
    holes.t1[i], holes.t2[i] = t1, np.cross(normal, t1)
    cells = cells.copy()
    cells[i] -= shift[i][:, None, :]
    return cells


def _halfplanes(polygons):
    """The edges of convex polygons (n, k, 2), counter-clockwise, as outward unit normals (n, k, 2)
    and offsets (n, k): inside is normal . q <= offset. Padding (repeated vertices) and NaN polygons
    give edges that hold nothing back."""
    e = np.roll(polygons, -1, axis=1) - polygons
    length = np.nan_to_num(np.linalg.norm(e, axis=2))
    real = length > 1e-9
    normal = np.stack([e[..., 1], -e[..., 0]], axis=2) / np.where(real, length, 1.0)[..., None]
    offset = np.einsum("nkd,nkd->nk", normal, polygons)
    return np.where(real[..., None], normal, [1.0, 0.0]), np.where(real, offset, np.inf)


def _homothets(shapes, normal, offset, size, steps=SEARCH):
    """For shapes (n, p, m, 2) round the origin, each in polygon n (normal, offset: _halfplanes): the
    largest factor s and the shift t (n, p, 2) that keep s shape + t inside it, by a pattern search
    from t = 0 (`size` (n,): how far it may look). The function is concave: the search may stop short
    of its top, never outside the polygon."""
    h = np.einsum("npmd,nkd->npkm", shapes, normal).max(axis=3)          # the shapes' reach along each normal
    t = np.zeros(shapes.shape[:2] + (2,))
    best = (offset[:, None] / h).min(axis=2)
    step = np.broadcast_to(0.25 * np.asarray(size, dtype=np.float64)[:, None], best.shape).copy()
    for _ in range(steps):
        tries = t[:, :, None] + step[..., None, None] * DIRECTIONS          # (n, p, 16, 2)
        value = ((offset[:, None, None] - np.einsum("npqd,nkd->npqk", tries, normal)) / h[:, :, None]).min(axis=3)
        k = value.argmax(axis=2)
        top = np.take_along_axis(value, k[..., None], axis=2)[..., 0]
        better = top > best
        t = np.where(better[..., None], np.take_along_axis(tries, k[..., None, None], axis=2)[:, :, 0], t)
        best, step = np.where(better, top, best), np.where(better, step, step / 2)
    return best, t


def _rotations(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)


def _fit_shapes(hull, turn, cells, turns, ratios, check=None):
    """Fit stretch (fill: only its turn): for every copy the footprint `hull` turned (its turn plus
    each of `turns`), stretched (each of `ratios`, area kept, along its cell's long axis and
    STRETCH_TILT either side) and moved to where it is largest inside its cell (n, k, 2, round the
    copy; NaN: none). The KEEP best at the cell's middle are searched (_homothets). Returns (size
    (n,), NaN for no cell; turn (n,); the stretch in the turned frame (n, 2, 2), det 1; the shift of
    the copy's centre (n, 2))."""
    n = len(cells)
    normal, offset = _halfplanes(cells)
    finite = np.isfinite(cells).all(axis=(1, 2))
    middle = P2.centroids(hull[None])[0]
    unit = hull - middle
    # each cell's long axis, from its edges' midpoints weighted by their length
    e = np.roll(cells, -1, axis=1) - cells
    w = np.nan_to_num(np.linalg.norm(e, axis=2))
    mid = np.nan_to_num(cells + e / 2)
    d = mid - ((w[..., None] * mid).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-300)[:, None])[:, None]
    cov = np.einsum("nk,nki,nkj->nij", w, d, d)
    axis = 0.5 * np.arctan2(2 * cov[:, 0, 1], cov[:, 0, 0] - cov[:, 1, 1])
    tilts = np.radians([-STRETCH_TILT, 0.0, STRETCH_TILT]) if len(ratios) > 1 else np.zeros(1)
    phi = (axis[:, None] + tilts[None])[:, :, None, None]                     # (n, tilt, ratio, turn)
    r = np.sqrt(np.asarray(ratios, dtype=np.float64))[None, None, :, None]
    theta = turn[:, None, None, None] + np.asarray(turns)[None, None, None, :]
    c, s = np.cos(phi), np.sin(phi)
    shape = np.broadcast_shapes(phi.shape, r.shape, theta.shape)
    S = np.empty(shape + (2, 2))
    S[..., 0, 0], S[..., 1, 1] = c * c * r + s * s / r, s * s * r + c * c / r
    S[..., 0, 1] = S[..., 1, 0] = c * s * (r - 1 / r)
    A = (S @ _rotations(np.broadcast_to(theta, shape))).reshape(n, -1, 2, 2)
    theta = np.broadcast_to(theta, shape).reshape(n, -1)
    q, m, k = A.shape[1], len(unit), normal.shape[1]
    first = np.empty((n, q))
    chunk = max(1, int(2e6 // (q * m * k)))
    for a in range(0, n, chunk):                         # every candidate with its middle on the copy's centre
        if check is not None:
            check()
        b = slice(a, a + chunk)
        h = np.einsum("nqmi,nki->nqkm", np.einsum("nqij,mj->nqmi", A[b], unit), normal[b]).max(axis=3)
        first[b] = (offset[b, None] / h).min(axis=2)
    pick = np.argsort(-np.nan_to_num(first, nan=-np.inf), axis=1)[:, :KEEP]
    chosen = np.take_along_axis(A, pick[..., None, None], axis=1)              # (n, p, 2, 2)
    size = np.where(finite, np.nan_to_num(np.ptp(cells, axis=1)).max(axis=1), 1.0)
    best, t = _homothets(np.einsum("npij,mj->npmi", chosen, unit), normal, offset, size)
    j = np.argmax(np.nan_to_num(best, nan=-np.inf), axis=1)
    size = np.take_along_axis(best, j[:, None], axis=1)[:, 0]
    A = np.take_along_axis(chosen, j[:, None, None, None], axis=1)[:, 0]
    turned = np.take_along_axis(theta, np.take_along_axis(pick, j[:, None], axis=1), axis=1)[:, 0]
    shift = np.take_along_axis(t, j[:, None, None], axis=1)[:, 0] - size[:, None] * (A @ middle)
    size = np.where(finite & (size > 0), size, np.nan)
    return size, turned, _rotations(-turned) @ A, np.where(np.isfinite(size)[:, None], shift, 0.0)


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _vertices(poly):
    """A padded polygon's own vertices (no NaN, no repeats)."""
    poly = poly[np.isfinite(poly).all(axis=1)]
    return poly[np.linalg.norm(poly - np.roll(poly, 1, axis=0), axis=1) > 1e-12]


def _inner_circles(cells):
    """The largest circle inside every cell (n, k, 2): (radius (n,), NaN for no cell; centre (n, 2),
    0 for none)."""
    finite = np.isfinite(cells).all(axis=(1, 2))
    normal, offset = _halfplanes(cells)
    size = np.where(finite, np.nan_to_num(np.ptp(cells, axis=1)).max(axis=1), 1.0)
    inner, middle = _homothets(np.broadcast_to(CIRCLE, (len(cells), 1) + CIRCLE.shape), normal, offset, size)
    return np.where(finite, inner[:, 0], np.nan), np.where(finite[:, None], middle[:, 0], 0.0)


def _fill_outlines(holes, hull, cells, depth):
    """Fit fill: every copy's outline is its cell, the corners rounded as wide as the part's edge
    (`depth`, at size 1): (sizes, at which the part's footprint has the cell's area, NaN for no cell;
    outlines (n, k, 2) at size 1 in the copies' turned frames, as Holes.cells)."""
    n = len(cells)
    inner, _ = _inner_circles(cells)
    sizes, outlines = np.full(n, np.nan), [None] * n
    footprint = _area(hull)
    for i in np.flatnonzero(np.nan_to_num(inner) > 0):
        poly = _vertices(cells[i])
        if len(poly) < 3 or _area(poly) <= 0:
            continue
        sizes[i] = math.sqrt(_area(poly) / footprint)
        poly = _round_corners(poly, min(depth * sizes[i], CORNER * inner[i]))
        outlines[i] = poly @ _rotations(holes.turn[i]) / sizes[i]          # turned back: R(-turn) q
    m = max([len(p) for p in outlines if p is not None] + [3])
    out = np.full((n, m, 2), np.nan)
    for i, p in enumerate(outlines):
        if p is not None:
            out[i, :len(p)], out[i, len(p):] = p, p[-1]
    return sizes, out


def _round_corners(poly, r, step=math.radians(10)):
    """The convex polygon with its corners rounded to radius r (moved in by r, then out again with
    arcs at most `step` apart)."""
    if r <= 1e-6:
        return poly
    normal, offset = _halfplanes(poly[None])
    inner = poly
    for nk, ck in zip(normal[0], offset[0]):
        inner = P2._clip(inner, nk, ck - r)
        if len(inner) < 3:
            return poly
    out = []
    for k in range(len(inner)):
        before, after = inner[k] - inner[k - 1], inner[(k + 1) % len(inner)] - inner[k]
        a0, a1 = math.atan2(-before[0], before[1]), math.atan2(-after[0], after[1])   # the edges' outward normals
        a1 = a0 + (a1 - a0) % (2 * np.pi)
        for a in np.linspace(a0, a1, max(1, math.ceil((a1 - a0) / step)) + 1):
            out.append(inner[k] + r * np.array([math.cos(a), math.sin(a)]))
    return _vertices(np.array(out))


def _edge_depth(part, point, basis, hull):
    """How far in from its outline seen from above the part's edge reaches (a fillet, a chamfer, a
    sloping side): the farthest vertex of a face that is not level, at size 1."""
    normal, offset = _halfplanes(hull[None])
    deepest = 0.0
    for b in part:
        local = (np.asarray(b.vertices) - point) @ basis.T
        F = np.asarray(b.faces)
        T = local[F]
        cross = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        sloped = np.abs(cross[:, 2]) < 0.999 * np.linalg.norm(cross, axis=1)
        if sloped.any():
            v = local[np.unique(F[sloped]), :2]
            deepest = max(deepest, float(np.clip((offset[0] - v @ normal[0].T).min(axis=1), 0, None).max()))
    return deepest


def _border_lines(V, F, ids, chosen, size):
    """The border of the chosen faces as points, each with the normal of the plane copies stay
    behind (the way out of the faces there, in the surface and square to the border; where another
    face meets them at a fold, that face's normal), the surface's normal and whether it is such a
    fold: (points, out, normal, end), at most size / 8 apart (as holes._border_points). The border is
    where a chosen triangle meets another one, or a sharp fold between two chosen faces (a point for
    either side)."""
    m = trimesh.Trimesh(V, F, process=False)
    pairs, edges, bends = m.face_adjacency, m.face_adjacency_edges, m.face_adjacency_angles
    sel = np.isin(ids, chosen)
    a, b = sel[pairs[:, 0]], sel[pairs[:, 1]]
    fold = a & b & (ids[pairs[:, 0]] != ids[pairs[:, 1]]) & (bends > H.FOLD)
    inside = np.concatenate([pairs[a & ~b, 0], pairs[~a & b, 1], pairs[fold, 0], pairs[fold, 1]])
    if not len(inside):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=bool)
    beyond = np.concatenate([pairs[a & ~b, 1], pairs[~a & b, 0]])
    end = np.r_[np.concatenate([bends[a & ~b], bends[~a & b]]) > H.FOLD, np.zeros(2 * int(fold.sum()), dtype=bool)]
    seg = V[np.concatenate([edges[a & ~b], edges[~a & b], edges[fold], edges[fold]])]

    def normals(tri):
        T = V[F[tri]]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None], T.mean(axis=1)
    normal, middle = normals(inside)
    out = np.cross(seg[:, 1] - seg[:, 0], normal)
    out /= np.maximum(np.linalg.norm(out, axis=1), 1e-300)[:, None]
    out *= np.sign(np.einsum("ij,ij->i", out, seg.mean(axis=1) - middle))[:, None]
    out[:len(beyond)][end[:len(beyond)]] = normals(beyond)[0][end[:len(beyond)]]
    spacing = float(np.clip(size / 8, 0.02, 0.25))
    count = np.maximum(np.ceil(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) / spacing).astype(np.int64), 1)
    k = np.repeat(np.arange(len(seg)), count + 1)
    t = (np.arange(len(k)) - np.repeat(np.cumsum(count + 1) - (count + 1), count + 1)) / np.repeat(count, count + 1)
    return seg[k, 0] + (seg[k, 1] - seg[k, 0]) * t[:, None], out[k], normal[k], end[k]


def _leaning(holes, edge, reach):
    """Whether a copy stands within `reach` (n,) of a border along a neighbouring face's plane
    (_border_lines) that its normal, or the surface's there, leans over."""
    points, out, facing, end = edge
    if not end.any() or not len(holes):
        return False
    for i, k in enumerate(cKDTree(points[end]).query_ball_point(holes.centre, reach)):
        if len(k) and (np.maximum(out[end][k] @ holes.normal[i],
                                  np.einsum("ij,ij->i", out[end][k], facing[end][k])) > 1e-6).any():
            return True
    return False


def _fits(hull, turn, cells):
    """The largest size of every copy whose footprint `hull` (m, 2), turned by `turn`, fits in its
    cell (n, k, 2: convex, round the copy): along the ray through each corner of the footprint, how
    far the cell's edge is, over how far the corner is; the smallest of those. NaN for no cell."""
    c, s = np.cos(turn)[:, None], np.sin(turn)[:, None]
    px = (hull[None, :, 0] * c - hull[None, :, 1] * s)[:, :, None]       # (n, m, 1)
    py = (hull[None, :, 0] * s + hull[None, :, 1] * c)[:, :, None]
    ax, ay = cells[:, None, :, 0], cells[:, None, :, 1]                 # (n, 1, k)
    e = np.roll(cells, -1, axis=1) - cells
    ex, ey = e[:, None, :, 0], e[:, None, :, 1]
    denom = px * ey - py * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ax * ey - ay * ex) / denom                                   # t p = a + u e
        u = (ax * py - ay * px) / denom
    hit = (np.abs(denom) > 1e-15) & (u >= -1e-9) & (u <= 1 + 1e-9) & (t > 0)
    fit = np.where(hit, t, np.inf).min(axis=2).min(axis=1)
    fit[~np.isfinite(cells).all(axis=(1, 2)) | ~np.isfinite(fit)] = np.nan
    return fit


class _Surface:
    """The chosen faces, for moving points back onto them: the closest point, the face's normal
    there (angle-weighted vertex normals of each CAD face, interpolated) and its CAD face."""

    def __init__(self, V, F, ids, chosen):
        sel = np.isin(ids, chosen)
        self.V, self.F, self.ids = V, np.ascontiguousarray(F[sel]), ids[sel]
        self.tree = igl.AABB()
        self.tree.init(self.V, self.F)
        P = V[self.F]
        cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        unit = np.nan_to_num(cross / np.linalg.norm(cross, axis=1)[:, None])
        key = self.F * (int(self.ids.max()) + 1) + self.ids[:, None]   # a vertex on each face apart
        corner, inverse = np.unique(key, return_inverse=True)
        vn = np.zeros((len(corner), 3))
        np.add.at(vn, inverse.ravel(), (unit[:, None, :] * corner_angles(P)[..., None]).reshape(-1, 3))
        vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-300)[:, None]
        self.normals = vn[inverse.reshape(-1, 3)]                         # (k, 3, 3) per corner

    def closest(self, points):
        _, tri, at = self.tree.squared_distance(self.V, self.F, np.ascontiguousarray(points, dtype=np.float64))
        P = self.V[self.F[tri]]
        v0, v1, v2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], at - P[:, 0]
        d00, d01, d11 = (np.einsum("ij,ij->i", p, q) for p, q in ((v0, v0), (v0, v1), (v1, v1)))
        d20, d21 = np.einsum("ij,ij->i", v2, v0), np.einsum("ij,ij->i", v2, v1)
        det = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-300, d00 * d11 - d01 * d01, 1.0)
        b1, b2 = (d11 * d20 - d01 * d21) / det, (d00 * d21 - d01 * d20) / det
        bary = np.clip(np.column_stack([1 - b1 - b2, b1, b2]), 0, 1)
        bary /= np.maximum(bary.sum(axis=1), 1e-300)[:, None]
        n = np.einsum("ij,ijk->ik", bary, self.normals[tri])
        return at, n / np.linalg.norm(n, axis=1)[:, None], self.ids[tri]


def _turned(holes):
    """The copies' own x and y directions: the tangent frame turned by each copy's turn."""
    c, s = np.cos(holes.turn)[:, None], np.sin(holes.turn)[:, None]
    return holes.t1 * c + holes.t2 * s, holes.t2 * c - holes.t1 * s


def _matrices(holes, point, basis, lift):
    """Every copy's 4 x 4 matrix: the part's base point to the copy's centre (Lift mm out along the
    normal), its directions (basis rows) to the copy's turned x, y (stretched, Fit stretch) and the
    normal, at its size."""
    x, y = _turned(holes)
    frame = np.stack([x, y, holes.normal], axis=2)
    if holes.stretch is not None:
        frame[:, :, :2] = frame[:, :, :2] @ holes.stretch
    turn = frame * holes.scale[:, None, None] @ basis
    out = np.tile(np.eye(4), (len(holes), 1, 1))
    out[:, :3, :3] = turn
    out[:, :3, 3] = holes.centre + lift * holes.normal - turn @ point
    return out


class _Fill:
    """Fit fill: makes every copy by bending the part (at the copy's size) so that its outline seen
    from above is the copy's outline (_Warp), standing on the copy's tangent plane or, where the
    face under it is curved, wrapped onto it: every point goes to the surface under it and stands
    out along the normal there. Triangles are split where the bend needs it (_refine)."""

    def __init__(self, holes, point, basis, lift, hull, depth, surface):
        self.centre, self.normal, self.scale, self.outlines = holes.centre, holes.normal, holes.scale, holes.cells
        self.x, self.y = _turned(holes)
        self.point, self.basis, self.lift = point, basis, lift
        self.hull, self.depth, self.surface = hull, depth, surface

    def bodies(self, part, check=None, progress=None):
        out = []
        for i in range(len(self.centre)):
            if check is not None:
                check()
            out.extend(self.copy(i, part))
            if progress is not None:
                progress(i + 1, len(self.centre))
        return out

    def copy(self, i, part):
        """Copy i's bodies."""
        s = float(self.scale[i])
        outline = _vertices(self.outlines[i]) * s
        image = self._image(i, _Warp(self.hull * s, outline, BAND * self.depth * s + BAND_MIN), outline)
        out = []
        for b in part:
            local = (np.asarray(b.vertices) - self.point) @ self.basis.T * s
            V, F, parent, bent = _refine(local, np.asarray(b.faces), image)
            out.append(trimesh.Trimesh(bent, F, process=False, face_attributes=_carried(b, local, V, F, parent),
                                       metadata=dict(b.metadata)))
        return out

    def _image(self, i, warp, outline):
        """Where points (m, 3) of the part (its x, y, up at the copy's size) go."""
        centre, x, y, n, lift = self.centre[i], self.x[i], self.y[i], self.normal[i], self.lift

        def plane(Q):
            q = warp(Q[:, :2])
            return centre + q[:, :1] * x + q[:, 1:] * y

        # flat under the copy: the outline's corners, the middles of its sides and its middle
        probe = np.concatenate([outline, (outline + np.roll(outline, -1, axis=0)) / 2, [[0.0, 0.0]]])
        p = centre + probe[:, :1] * x + probe[:, 1:] * y
        at, normal, _ = self.surface.closest(p)
        if np.abs(normal - n).max() < 1e-7 and np.abs((p - at) @ n).max() < 1e-7:
            return lambda Q: plane(Q) + (Q[:, 2:] + lift) * n

        def wrapped(Q):
            base = plane(Q)
            at, normal, _ = self.surface.closest(base)
            # onto the surface along its normal (past the faces' border: on along the tangent plane there)
            return base + np.einsum("ij,ij->i", at - base, normal)[:, None] * normal + (Q[:, 2:] + lift) * normal
        return wrapped


class _Warp:
    """Fit fill: maps a footprint (a convex polygon) onto a copy's outline (another), both round the
    origin. A point d from the footprint's edge, within the band, goes to the point d from the
    outline's edge in the same direction from the middle (the centre of the largest circle inside
    each), so an edge profile (a fillet, a chamfer) keeps its width all round; points farther in
    are scaled along those directions as much as the band's inner edge is."""

    def __init__(self, footprint, outline, band):
        found = []
        for poly in (footprint, outline):
            normal, offset = _halfplanes(poly[None])
            r, o = _homothets(CIRCLE[None, None], normal, offset, np.array([float(np.ptp(poly, axis=0).max())]))
            found.append((normal[0], offset[0], float(r[0, 0]), o[0, 0]))
        (self.nP, self.cP, rP, self.oP), (self.nC, self.cC, rC, self.oC) = found
        self.band = min(BAND_SHARE * min(rP, rC), band)

    @staticmethod
    def _reach(normal, offset, o, u, d):
        """How far from o along the unit directions u (m, 2) the polygon moved in by d (m,) ends."""
        along = u @ normal.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(along > 1e-12, (offset[None] - d[:, None] - (normal @ o)[None]) / along, np.inf)
        return t.min(axis=1)

    def __call__(self, q):
        d = np.clip((self.cP[None] - q @ self.nP.T).min(axis=1), 0, None)    # in from the footprint's edge
        v = q - self.oP
        rho = np.linalg.norm(v, axis=1)
        u = np.where(rho[:, None] > 1e-12, v / np.maximum(rho, 1e-300)[:, None], [1.0, 0.0])
        band = d <= self.band
        out = np.empty_like(q)
        out[band] = self.oC + self._reach(self.nC, self.cC, self.oC, u[band], d[band])[:, None] * u[band]
        inner = np.full(int((~band).sum()), self.band)
        ratio = self._reach(self.nC, self.cC, self.oC, u[~band], inner) / \
            self._reach(self.nP, self.cP, self.oP, u[~band], inner)
        out[~band] = self.oC + (rho[~band] * ratio)[:, None] * u[~band]
        return out


def _refine(V, F, image, tol=FILL_TOL, rounds=FILL_ROUNDS, shortest=FILL_SHORTEST):
    """Splits triangles (V: x, y, up) whose edges `image` (points (m, 3) -> (m, 3)) bends more than
    `tol` off the planes of their bent triangles (sliding within them, along a straight side or on
    a flat top, costs nothing), conforming (red-green: an edge is split in both its triangles).
    Returns (V, F, the part's triangle each one comes from, the bent V)."""
    parent = np.arange(len(F))
    bent = image(V)
    known, known_middle = np.zeros(0, dtype=np.int64), np.zeros((0, 3))     # edges kept from the round before
    for _ in range(rounds):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        E, inverse = np.unique(E, axis=0, return_inverse=True)
        inverse = inverse.reshape(3, -1).T                                   # each triangle's 3 edges
        key = E[:, 0].astype(np.int64) << 32 | E[:, 1]                       # sorted, as E is
        at = np.clip(np.searchsorted(known, key), 0, max(len(known) - 1, 0))
        seen = known[at] == key if len(known) else np.zeros(len(E), dtype=bool)
        middle = np.empty((len(E), 3))
        middle[seen] = known_middle[at[seen]]
        middle[~seen] = image((V[E[~seen, 0]] + V[E[~seen, 1]]) / 2)
        known, known_middle = key, middle
        T = bent[F]
        normal = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
        off = np.zeros(len(E))
        for k in range(3):
            np.maximum.at(off, inverse[:, k], np.abs(np.einsum("ij,ij->i", middle[inverse[:, k]] - T[:, 0], normal)))
        split = (off > tol) & (np.linalg.norm(V[E[:, 0], :2] - V[E[:, 1], :2], axis=1) > shortest)
        if not split.any():
            break
        new = np.full(len(E), -1)
        new[split] = len(V) + np.arange(int(split.sum()))
        V = np.concatenate([V, (V[E[split, 0]] + V[E[split, 1]]) / 2])
        bent = np.concatenate([bent, middle[split]])
        m = new[inverse]                                                     # the new vertex on edge k, or -1
        cut = m >= 0
        count = cut.sum(axis=1)
        out, came = [F[count == 0]], [parent[count == 0]]
        one = np.flatnonzero(count == 1)                                     # turned so edge 0 (v0-v1) is cut
        index = (np.argmax(cut[one], axis=1)[:, None] + np.arange(3)) % 3
        v, mm = np.take_along_axis(F[one], index, axis=1), np.take_along_axis(m[one], index, axis=1)[:, 0]
        out += [np.column_stack([v[:, 0], mm, v[:, 2]]), np.column_stack([mm, v[:, 1], v[:, 2]])]
        came += [parent[one]] * 2
        two = np.flatnonzero(count == 2)                                     # turned so edges 0 and 1 are cut
        index = ((np.argmin(cut[two], axis=1) + 1)[:, None] + np.arange(3)) % 3
        v, mm = np.take_along_axis(F[two], index, axis=1), np.take_along_axis(m[two], index, axis=1)
        out += [np.column_stack([mm[:, 0], v[:, 1], mm[:, 1]]), np.column_stack([v[:, 0], mm[:, 0], mm[:, 1]]),
                np.column_stack([v[:, 0], mm[:, 1], v[:, 2]])]
        came += [parent[two]] * 3
        three = np.flatnonzero(count == 3)
        v, mm = F[three], m[three]
        out += [np.column_stack([v[:, 0], mm[:, 0], mm[:, 2]]), np.column_stack([mm[:, 0], v[:, 1], mm[:, 1]]),
                np.column_stack([mm[:, 2], mm[:, 1], v[:, 2]]), np.column_stack([mm[:, 0], mm[:, 1], mm[:, 2]])]
        came += [parent[three]] * 4
        F, parent = np.concatenate(out), np.concatenate(came)
    return V, F, parent, bent


def _carried(body, before, after, F, parent):
    """The part's face attributes on its split triangles: per triangle from the triangle it was split
    from, per corner (the surface parameters, (n, 3, 2)) interpolated at the new corners. before,
    after: the vertices before and after splitting (the first are the same)."""
    T = before[np.asarray(body.faces)[parent]]
    v0, v1, v2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], after[F] - T[:, None, 0]
    d00, d01, d11 = (np.einsum("ij,ij->i", a, b) for a, b in ((v0, v0), (v0, v1), (v1, v1)))
    d20, d21 = np.einsum("icj,ij->ic", v2, v0), np.einsum("icj,ij->ic", v2, v1)
    det = d00 * d11 - d01 * d01
    ok = np.abs(det) > 1e-30 * np.maximum(d00 * d11, 1e-300)
    det = np.where(ok, det, 1.0)[:, None]
    b1, b2 = (d11[:, None] * d20 - d01[:, None] * d21) / det, (d00[:, None] * d21 - d01[:, None] * d20) / det
    bary = np.stack([1 - b1 - b2, b1, b2], axis=2)                              # (f, corner, parent's corner)
    bary[~ok] = np.eye(3)
    out = {}
    for key, value in body.face_attributes.items():
        value = np.asarray(value)
        if len(value) != len(body.faces):
            out[key] = value
        elif value.ndim >= 2 and value.shape[1] == 3 and value.dtype.kind == "f":
            out[key] = np.einsum("fcj,fj...->fc...", bary, value[parent]).astype(value.dtype)
        else:
            out[key] = value[parent]
    return out
