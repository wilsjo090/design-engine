"""Bars along an optimized shape (Topology Optimization's Members as: bars): the optimizer says where material
goes, and the members are drawn again as bars of a square section, flat-faced, with fillets where they meet
and their rims chamfered or rounded: the look of a designed frame instead of the optimizer's own surface.

`trace`: the shape's samples (members and what was kept solid) are thinned to their centre lines (a 3D
skeleton, which keeps every loop and every connection of the shape; what hangs loose shrinks away), the
lines inside the fixed bodies (Keep, the bosses) are dropped, and the rest is read as a network: joints where
lines meet, a branch between two joints. Short spurs go, joints a bar's width apart become one. Then either
(`follow`) each bar keeps its member's form: its centre line, smoothed, and its width along it (every line
sample's share of the material, over the thickness), cut into short straight pieces that stay within a small
share of the width of both; or each member is one or a few straight bars of one width, what its material needs
(its volume over its length). A width can be given instead.

`body`: every piece is a prism: a tapered stadium outline (a strip from one width to another, with round ends)
in the plane square to its `normal`, as thick as `thick` along it, so the two faces are flat and the sides
stand square on them. A member's pieces are joined plainly (they share their end circles), and the members'
outlines are blended where they meet (a G2 fillet, `fields.bulge`), on prisms taller than the bars so that the
blend never lifts the flat faces (a smooth union of the bars themselves pillows them at every joint), and the
faces are those of the bars' own slabs; the rims between the two are chamfered or rounded. A pure function of
position, member by member on the samples round each (as the pipes): the same in blocks and on the GPU.

`lofts`: a joint of a free part without fillets or a ball: every member is cut short of it, and between every
two members there is a loft, a bar along a curve from the one's cut end to the other's whose section goes
evenly from the one's to the other's. `body` draws the lofts (`curves`) as they are, on the samples' nearest
points of each curve. `fills`: the dimples and creases that a joint's lofts leave between them, filled by
letting the joint's surface flow outwards (`flow`) where it is hollow in every direction (a pit) or a groove
sharper than a bend; kept as a patch of samples per joint.
"""
import hashlib
import math
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from ..fields import Field, bulge, spline_values
from ..pipe import PAD, _window
from ..shared import Shared

SPUR = 2.0        # x its width: a loose end shorter than this is no bar
MERGE = 1.5       # x the width: joints nearer than this are one joint
STRAIGHT = 1.0    # x the width: a straight bar's pieces stay this near the member's centre line
CURVE = 0.02      # x the width: ... and a bar that follows its member this near its line and its width
SMOOTH = 3.0      # samples: the centre line and the width along it are smoothed by this before they are followed
WIDEST = 1.6      # x a member's median width: a bar is at most this wide (a gusset at a joint is the fillet's)
SPAN = 24         # samples: how far each way a member's thickness is measured through a line sample
INTO = 1.0        # x the width: a joint at a fixed body lies this far inside it
BEND = 2.5        # x the width: a bar that follows its member bends to a moved joint within this of it
ROUND = 3.0       # x the width: the radius of a straight bar's line where it turns a corner
ARC = 4.0         # degrees: a rounded corner's pieces turn at most this much each
TALL = 0.5        # x the thickness: the prisms the outlines are blended on stand this far past the faces
LOFT = 1.2        # x the larger bar's side: a loft leaves its two bars this far from their joint
LOFT_INNER = 0.8  # x the larger bar's side: the radius the inside of a loft's bend keeps between bars at a sharp angle
LOFT_MOST = 3.5   # x the larger bar's side: ... but a loft leaves its bars no farther from the joint than this
LOFT_OVER = 0.1   # x a loft's smaller side: how far it runs on into the bars it joins
LOFT_SHARE = 0.5  # of a member's length: what a loft at each end may take of it (the two then meet in its middle)
LOFT_ALONE = 0.8  # ... and of a member with a loft at one end only
FILL_TIME = 0.2   # x a joint's bars' side^2: how long a lofted joint's surface flows (`flow`): pits up to 1.3 sides across come out flat
FILL_SHARP = 0.3  # x the side: grooves sharper than this radius are rounded to it (a bend's inside is LOFT_INNER; a round opening wider than twice this stays: at 0.6 the openings 5 mm wide between short bars closed up)
FILL_CORE = 0.75  # x the side: a joint's fill reaches this far past its lofts' middles ...
FILL_FADE = 0.5   # x the side: ... and fades out over this much more
FILL_VOXEL = 0.4  # mm: the samples a joint's fill is worked out and kept on
FILL_BLUR = 1.0   # samples: the bars' values are smoothed by this before they flow
FILL_STEP = 0.1   # samples^2 of time an explicit step of the flow (the curvature is at most 1 / sample)
FILL_BAND = (3.0, 6.0)    # samples: the values within 3 of the surface flow fully, and none from 6 on
FILL_GATE = (0.4, 0.7)    # none where the values turn back within a sample: central over one-sided differences under 0.4 (all from 0.7; at 0.75-0.95 the bottom of a filled pit, where its creases met, stayed as a dot)
FILL_EVERY = 8    # steps between looking for the samples near the surface again
FILL_RIDGE = 0.0  # x the side: edges sharper than this radius where two lofts meet are rounded to it, material taken off (0 = none: off, it rounded the joints' rims and left the steps, FINDINGS "Sharp edges rounded too")
FILL_SEAM = (0.6, 1.2)    # mm: ... only where the surface of a second loft or bar comes within 0.6 of the sample (none from 1.2: a bar's own edge) ...
FILL_CROSS = (5.0, 15.0)  # degrees: ... turned from it by more than this (fully from 15: two lofts leaving one bar lie on each other and part slowly)
PATCH_TRUST = 0.5   # x the side: a joint's surface is kept where no other loft's or bar's surface lies within this (`patches`) ...
PATCH_BAND = 0.27   # x the side: ... the values this near it kept ...
PATCH_INNER = 0.42  # x the side: ... and those deeper inside than this, or along a bar's middle (the deepest within PATCH_MIDDLE), the rest fitted
PATCH_MIDDLE = 0.6  # x the side
PATCH_STEP = 2      # samples of FILL_VOXEL: the fit is solved on every second one
FILL_UNDER = 0.04 # mm: a fill lies this far under what the flow leaves (a face nothing was added to stays the bars' own)
LOFT_FOLD = 0.5   # x the larger bar's side: a loft whose curve is tighter than this folds over itself
LOFT_TURN = 12.0  # degrees: a loft's spans turn and twist at most this much each (its faces' normal goes evenly across a span)
LOFT_NODE = False # (on: try) a joint of three bars or more is one body: its lofts' outlines cut by the convex hull of its bars' cut ends, whose flat faces it has (FINDINGS "A node per joint")
LOFT_FACES = False # (on: try) a joint of three bars or more is drawn from its faces, each patched on its own (`body`'s joint_faces; FINDINGS "Face by face")
LOFT_FACE_NEAR = 0.4 # x a loft's side: the joint's faces lie within this of its lofts' own (its outline reaches no farther)
LOFT_FACE_POWER = 4  # Shepard's power: a bar's face weighs 1 / its distance^4 (with ^2 the joint's face left it within a mm or two of the cut: ledges)
LOFT_FACE_TILT = 0.7  # a bar whose faces all lie steeper than 45 degrees to the joint's plane gets none (a post on it)
LOFT_PLANE = False  # a joint's lofts share one plane through their middles, their faces its slab's there: off, the faces twisted and waved and most steps stayed (FINDINGS "One plane per joint")
LOFT_PLANE_SPREAD = 0.3   # x the bars' side: lofts' middles spread less than this across (in a line) give no plane
LOFT_PLANE_TILT = 0.7     # a loft whose middle runs steeper than 45 degrees out of the plane keeps its own section
LOFT_PLANE_OFF = 0.25     # x the joint's thickness: ... and so does one whose middle lies farther than this off the plane
CUBE = np.ones((3, 3, 3), dtype=bool)
VERSION = 37


def _chains(skeleton, anchor):
    """The skeleton as joints and chains: (positions (n, 3) in voxels, anchored (n,), chains), a chain =
    (joint a, joint b, voxels (m, 3) in order from a to b, joints not included)."""
    skeleton = np.pad(skeleton, 1)
    anchor = np.pad(anchor, 1)
    # (neighbours counted on the whole skeleton: a line that passes a fixed body and has a leg into it is a
    # joint there, held by that body, though the leg itself is not drawn)
    near = ndimage.convolve((skeleton | anchor).astype(np.int16), CUBE.astype(np.int16), mode="constant") - 1
    meeting = skeleton & (near >= 3)
    joints, count = ndimage.label(meeting, structure=CUBE)
    positions = [np.asarray(c) for c in ndimage.center_of_mass(meeting, joints, range(1, count + 1))]
    beside = ndimage.binary_dilation(anchor, structure=CUBE)
    anchored = [bool(v) for v in np.atleast_1d(ndimage.maximum(beside, joints, range(1, count + 1)))] if count else []
    runs, total = ndimage.label(skeleton & ~meeting, structure=CUBE)
    chains = []

    def around(v, array):
        return array[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]

    def loose(v):
        positions.append(np.asarray(v, dtype=float))
        anchored.append(bool(around(v, anchor).any()))
        return len(positions) - 1

    def end(v):
        """The joint a chain's end voxel belongs to: the one beside it, or a new one there (a loose end)."""
        beside = around(v, joints)
        return int(beside[beside > 0][0]) - 1 if beside.any() else loose(v)

    for k, box in enumerate(ndimage.find_objects(runs)):
        cells = np.argwhere(runs[box] == k + 1) + [s.start for s in box]
        index = {tuple(c): i for i, c in enumerate(cells)}
        links = [[index[n] for n in ((c[0] + i, c[1] + j, c[2] + l) for i in (-1, 0, 1) for j in (-1, 0, 1)
                                     for l in (-1, 0, 1) if (i, j, l) != (0, 0, 0)) if n in index] for c in cells]
        ends = [i for i, l in enumerate(links) if len(l) <= 1]
        at, order, seen = (ends[0] if ends else 0), [], set()
        while at is not None:   # (a chain: every voxel has at most two neighbours in it)
            order.append(at)
            seen.add(at)
            at = next((n for n in links[at] if n not in seen), None)
        path = cells[order]
        if len(path) == 1:      # one voxel: between two joints, or from a joint to a fixed body or to nothing
            beside = np.unique(around(path[0], joints))
            beside = beside[beside > 0] - 1
            if len(beside) >= 2:
                chains.append((int(beside[0]), int(beside[1]), path))
            elif len(beside) == 1:   # (as a loose end beside its own joint it was a ring, and the body's hold lost)
                chains.append((int(beside[0]), loose(path[0]), path))
            continue
        if not ends:            # a ring with no joint on it: one joint, where the walk began
            positions.append(path[0].astype(float))
            anchored.append(False)
            chains.append((len(positions) - 1, len(positions) - 1, path[1:]))
        else:
            chains.append((end(path[0]), end(path[-1]), path))
    return np.array(positions, dtype=float).reshape(-1, 3) - 1.0, np.array(anchored, dtype=bool), \
        [(a, b, path - 1) for a, b, path in chains]


def _simplify(points, tol):
    """Douglas-Peucker: the indices of `points` to keep so that the polyline stays within `tol` of all."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, d = points[i], points[j] - points[i]
        length = np.linalg.norm(d)
        rel = points[i + 1:j] - a
        if length < 1e-9:
            off = np.linalg.norm(rel, axis=1)
        else:
            t = np.clip(rel @ d / (length * length), 0.0, 1.0)
            off = np.linalg.norm(rel - t[:, None] * d, axis=1)
        k = int(np.argmax(off))
        if off[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return np.flatnonzero(keep)


def _smooth_line(points, sigma):
    """`points` ((n, 3), a line's samples) smoothed along the line by a Gaussian of `sigma` samples, its two ends
    where they were: the line is continued past each end by its own mirror image through that end. (Smoothed
    as it is and its ends put back, a steep line's ends moved by a sample, and putting them back moved its
    middle as far: a bowed member's bars ran a mm inside its bow.)"""
    n = min(len(points) - 1, int(4 * sigma) + 1)
    more = np.concatenate([2.0 * points[0] - points[n:0:-1], points, 2.0 * points[-1] - points[-2:-n - 2:-1]])
    return ndimage.gaussian_filter1d(more, sigma, axis=0, mode="nearest")[n:n + len(points)]


def _rounded(corners, radius):
    """The polyline `corners` ((n, 3)) with every corner turned in an arc of `radius` (shorter where a piece
    beside it is short: an arc takes at most 0.45 of each), as points no more than ARC degrees of turn apart.
    A bar bent at a sharp corner has a crease on the inside of the bend and, where its faces are chamfered,
    one across the outside too."""
    out = [corners[0]]
    for before, at, after in zip(corners[:-2], corners[1:-1], corners[2:]):
        back, on = before - at, after - at
        far_back, far_on = np.linalg.norm(back), np.linalg.norm(on)
        if far_back < 1e-9 or far_on < 1e-9:
            out.append(at)
            continue
        back, on = back / far_back, on / far_on
        turn = math.acos(float(np.clip(-(back @ on), -1.0, 1.0)))      # 0 = straight on
        reach = min(radius * math.tan(0.5 * turn), 0.45 * far_back, 0.45 * far_on)
        steps = int(math.ceil(math.degrees(turn) / ARC))
        if steps < 2 or reach < 1e-6 or turn > math.radians(175.0):   # (straight on, or a line doubling back)
            out.append(at)
            continue
        # the arc touching both pieces `reach` from the corner: about a centre on the corner's bisector
        middle = (back + on) / np.linalg.norm(back + on)
        centre = at + reach / math.sin(0.5 * turn) * middle
        from_, to = at + reach * back - centre, at + reach * on - centre
        for t in np.linspace(0.0, 1.0, steps + 1):
            out.append(centre + (math.sin((1.0 - t) * turn) * from_ + math.sin(t * turn) * to) / math.sin(turn))
    out.append(corners[-1])
    return np.array(out)


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def trace(solid, fixed, start, voxel, size=0.0, least=0.0, cancel=None, inward=None, thick=0.0, follow=False,
          normal=None, settle=None):
    """The bars of a shape. `solid`: bool samples (`voxel` mm apart from `start`) of the whole part, members
    and fixed bodies; `fixed`: those of the fixed bodies (Keep, the bosses), where no bars are drawn but bars
    may end; `inward(points)`: the unit directions into them at (n, 3) points (a joint at a fixed body is
    moved INTO x the width that way, so every bar ending there runs into the body, towards a boss's middle,
    and none past it), or `settle(points, reach)`: where such joints come to rest, at most `reach` ((n,) mm)
    from where they are (the middle of the body: every bar at a boss then ends on its axis, and the round ends
    of the bars and the boss are one eye, not three outlines). `size`: every bar's width, mm; 0 = as wide as the member is: its section over its
    thickness, which is measured through every line sample along `normal(points)` (unit vectors: the way the
    bars' faces will look), or is `thick` without one, or, with neither, the side of a square of the same
    section; at least `least`. `follow`: the bars keep their members' form, the smoothed centre line and (without a `size`)
    the width along it, as short pieces; else each member is straight bars of one width.
    A namespace: `a`, `b` ((n, 3) mm, each straight piece's ends), `width_a`, `width_b` (n,: its width at
    them), `width` (their mean), `section` (n,: the member's own mean section, mm2), `span` (n,: its mean
    thickness along `normal`, mm; 0 without one), `branch` (n,: which member each piece is of), `joints`
    ((m, 3) mm) and `members` (how many)."""
    from skimage.morphology import skeletonize
    step = cancel if cancel is not None else (lambda: None)
    solid, fixed = np.asarray(solid, dtype=bool), np.asarray(fixed, dtype=bool)
    start = np.asarray(start, dtype=np.float64)
    skeleton = skeletonize(solid).astype(bool)
    step()
    lines = skeleton & ~fixed
    positions, anchored, chains = _chains(lines, skeleton & fixed)
    # every line sample's share of the members' material: the free samples nearest it
    free = solid & ~fixed
    number = counts = None
    if lines.any() and free.any():
        number = np.full(solid.shape, -1, dtype=np.int32)
        number[lines] = np.arange(int(lines.sum()))
        nearest = ndimage.distance_transform_edt(~lines, return_distances=False, return_indices=True)
        counts = np.bincount(number[tuple(nearest)][free], minlength=int(lines.sum())).astype(float)
        del nearest
    spans = None
    if normal is not None and counts is not None:
        # how thick the member is through each line sample: the free samples in a row along the normal
        where = np.argwhere(lines).astype(np.float64)
        way = np.asarray(normal(where * voxel + start), dtype=np.float64)
        filled = free.astype(np.uint8)
        spans = np.ones(len(where))
        for sign in (1.0, -1.0):
            alive = np.ones(len(where), dtype=bool)
            for j in range(1, 2 * SPAN + 1):
                at = where + (sign * 0.5 * j) * way
                alive &= ndimage.map_coordinates(filled, at.T, order=0, mode="constant", cval=0) > 0
                if not alive.any():
                    break
                spans += 0.5 * alive
        spans *= voxel
    step()
    branches = []
    for a, b, path in chains:
        area = np.zeros(len(path))
        if counts is not None:   # mm2: a sample's material over the length of line it stands for
            # (of the line without the samples' steps: a line that zigzags a sample to and fro along a member
            # is 1.4 x as long as the member, and read its section, and so its width, 30% low)
            even = ndimage.gaussian_filter1d(path.astype(np.float64), 2.0, axis=0, mode="nearest") if len(path) > 2 else path
            gaps = np.linalg.norm(np.diff(even, axis=0), axis=1)
            along = np.ones(len(path))
            if len(path) > 1:
                along[0], along[-1] = gaps[0], gaps[-1]
                along[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
            area = counts[number[tuple(path.T)]] * voxel * voxel / along
        points = np.concatenate([positions[a][None], path, positions[b][None]]) * voxel + start
        span = spans[number[tuple(path.T)]] if spans is not None else np.zeros(len(path))
        branches.append({"a": a, "b": b, "points": points, "area": np.concatenate([area[:1], area, area[-1:]]),
                         "span": np.concatenate([span[:1], span, span[-1:]]),
                         "volume": float(area.sum() and (counts[number[tuple(path.T)]].sum() * voxel ** 3))})
    positions = positions * voxel + start
    anchored = list(anchored)

    def wide(area, span=None):
        """A section's width, mm: over the member's thickness there, or the bars', or the side of a square."""
        if span is not None and np.all(np.asarray(span) > 0):
            return np.asarray(area) / np.maximum(span, voxel)
        return np.asarray(area) / thick if thick > 0 else np.sqrt(area)

    def through(branch):
        """The member's mean thickness along the normal, mm (None when it was not measured)."""
        return float(np.mean(branch["span"])) if spans is not None and len(branch["span"]) else None

    def section(branch):
        return branch["volume"] / max(_length(branch["points"]), voxel)

    def gauge(branch):
        """The member's size for what counts as short or near, mm: the side of a square of its section."""
        return float(size) if size > 0 else max(math.sqrt(section(branch)), float(least), voxel)

    def width(branch):
        return float(size) if size > 0 else max(float(wide(section(branch), through(branch))), float(least), voxel)

    def follow_joints(branch):
        """The branch's line with its ends at its joints, where they are now, so a joint that was merged or
        moved into a body bends the member towards it and leaves no dog-leg at the member's old end. For
        straight bars every point moves by its share of what the ends moved (all at an end, none at the
        other). A bar that follows its member keeps the member's line and bends to the joint only near it,
        within BEND x its width: by shares, a chord whose two ends moved 13 mm down into the eyes they hang
        from moved 13 mm down all along, out of its member and onto the part's lower edge."""
        points = branch["points"]
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        if follow and along[-1] > 0:
            near = BEND * gauge(branch)
            first = np.clip(1.0 - along / near, 0.0, 1.0)[:, None] ** 2
            last = np.clip(1.0 - (along[-1] - along) / near, 0.0, 1.0)[:, None] ** 2
        else:
            last = (along / along[-1])[:, None] if along[-1] > 0 else np.linspace(0.0, 1.0, len(points))[:, None]
            first = 1.0 - last
        branch["points"] = points + first * (positions[branch["a"]] - points[0]) + last * (positions[branch["b"]] - points[-1])

    changed = True
    while changed:   # (a few dozen members: plain loops)
        changed = False
        degree = np.zeros(len(positions), dtype=int)
        for br in branches:
            degree[br["a"]] += 1
            degree[br["b"]] += 1
        # loose ends too short to be a bar
        for br in branches:
            loose = [n for n in (br["a"], br["b"]) if degree[n] == 1 and not anchored[n]]
            if loose and br["a"] != br["b"] and _length(br["points"]) < SPUR * gauge(br):
                branches.remove(br)
                changed = True
                break
        if changed:
            continue
        # a joint with two bars on it is no joint: one member running on
        for n in np.flatnonzero(degree == 2):
            pair = [br for br in branches if n in (br["a"], br["b"])]
            if len(pair) != 2 or anchored[n]:
                continue
            first, second = pair
            turn = slice(None) if first["b"] == n else slice(None, None, -1)
            on = slice(None) if second["a"] == n else slice(None, None, -1)
            joined = {"a": first["a"] if first["b"] == n else first["b"],
                      "b": second["b"] if second["a"] == n else second["a"],
                      "points": np.concatenate([first["points"][turn], second["points"][on][1:]]),
                      "area": np.concatenate([first["area"][turn], second["area"][on][1:]]),
                      "span": np.concatenate([first["span"][turn], second["span"][on][1:]]),
                      "volume": first["volume"] + second["volume"]}
            branches.remove(first)
            branches.remove(second)
            branches.append(joined)
            changed = True
            break
        if changed:
            continue
        # joints a bar's width apart are one
        for br in branches:
            a, b = br["a"], br["b"]
            if a == b or _length(br["points"]) >= MERGE * gauge(br) or (degree[a] < 3 and degree[b] < 3):
                continue
            if anchored[a] and anchored[b]:
                continue
            keep, gone = (a, b) if anchored[a] or not anchored[b] else (b, a)
            if not anchored[keep]:
                positions[keep] = 0.5 * (positions[a] + positions[b])
            branches.remove(br)
            for other in branches:
                for side in ("a", "b"):
                    if other[side] == gone:
                        other[side] = keep
            for other in list(branches):   # (their ends follow the joint; a member folded onto itself goes)
                follow_joints(other)
                if other["a"] == other["b"] and _length(other["points"]) < 2.0 * SPUR * gauge(other):
                    branches.remove(other)
            changed = True
            break
    step()
    if (inward is not None or settle is not None) and any(anchored):
        held = [n for n in range(len(positions)) if anchored[n] and any(n in (br["a"], br["b"]) for br in branches)]
        if held:
            reach = np.array([max(gauge(br) for br in branches if n in (br["a"], br["b"])) for n in held])
            if settle is not None:
                positions[held] = np.asarray(settle(positions[held], INTO * reach), dtype=np.float64)
            else:
                positions[held] = positions[held] + np.asarray(inward(positions[held])) * (INTO * reach)[:, None]
    a, b, first, last, member, sections, thicknesses = [], [], [], [], [], [], []
    for k, br in enumerate(branches):
        follow_joints(br)
        points, area = br["points"], br["area"]
        if follow:
            if len(points) > 4:      # the centre line without the samples' steps, its ends where they are
                points = _smooth_line(points, SMOOTH)
            if size > 0:
                widths = np.full(len(points), float(size))
            else:                    # as wide as the member is there, the samples' steps evened out
                even = lambda v: ndimage.gaussian_filter1d(np.asarray(v, dtype=np.float64), SMOOTH, mode="nearest")
                widths = wide(even(area), even(br["span"]) if spans is not None else None)
                widths = np.clip(widths, max(float(least), voxel), max(WIDEST * float(np.median(widths)), voxel))
                widths = even(widths)
            keep = _simplify(np.column_stack([points, widths]), CURVE * gauge(br))
        else:   # straight pieces, and where two of them meet the line turns in an arc
            corners = _rounded(points[_simplify(points, STRAIGHT * gauge(br))], ROUND * width(br))
            points, widths, keep = corners, np.full(len(corners), width(br)), np.arange(len(corners))
        corners, at = points[keep], widths[keep]
        if len(corners) < 2 or _length(corners) < 1e-6:
            continue
        for p, q, wp, wq in zip(corners[:-1], corners[1:], at[:-1], at[1:]):
            if np.linalg.norm(q - p) > 1e-6:
                a.append(p)
                b.append(q)
                first.append(wp)
                last.append(wq)
                member.append(k)
                sections.append(section(br))
                thicknesses.append(through(br) or 0.0)
    first, last = np.array(first, dtype=np.float64), np.array(last, dtype=np.float64)
    return SimpleNamespace(a=np.array(a, dtype=np.float64).reshape(-1, 3), b=np.array(b, dtype=np.float64).reshape(-1, 3),
                           width_a=first, width_b=last, width=0.5 * (first + last),
                           branch=np.array(member, dtype=np.int64), section=np.array(sections, dtype=np.float64),
                           span=np.array(thicknesses, dtype=np.float64),
                           joints=np.asarray(positions, dtype=np.float64), members=len(branches))


def mirrored(net, axis, at, near=1.0):
    """`net` made the same on both sides of the plane `axis` = `at` (Symmetry: the optimizer's shape is, its
    centre lines are not quite): the pieces on the low side and their mirror images; a piece across the plane
    runs from its low end to that end's image, one within `near` mm of the plane lies in it."""
    a, b = net.a.copy(), net.b.copy()
    first, last = net.width_a.copy(), net.width_b.copy()
    low = np.minimum(a[:, axis], b[:, axis])
    high = np.maximum(a[:, axis], b[:, axis])
    on = (np.abs(a[:, axis] - at) <= near) & (np.abs(b[:, axis] - at) <= near)
    across = ~on & (low < at - near) & (high > at + near)
    side = ~on & ~across & (0.5 * (a[:, axis] + b[:, axis]) < at)
    a[on, axis] = b[on, axis] = at
    flip = lambda p: np.concatenate([p[:, :axis], 2.0 * at - p[:, axis:axis + 1], p[:, axis + 1:]], axis=1)
    swap = across & (a[:, axis] > b[:, axis])          # (from the low end)
    a[swap], b[swap] = b[swap], a[swap].copy()
    first[swap], last[swap] = last[swap], first[swap].copy()
    b[across] = flip(a[across])
    last[across] = first[across]
    keep = on | across | side
    pick = lambda v: np.concatenate([v[keep], v[side]])
    first, last = pick(first), pick(last)
    return SimpleNamespace(a=np.concatenate([a[keep], flip(a[side])]), b=np.concatenate([b[keep], flip(b[side])]),
                           width_a=first, width_b=last, width=0.5 * (first + last), section=pick(net.section),
                           span=pick(net.span),
                           branch=np.concatenate([net.branch[keep], net.branch[side] + int(net.branch.max(initial=0)) + 1]),
                           joints=net.joints, members=len(set(net.branch[keep].tolist())) + len(set(net.branch[side].tolist())))


def normals(a, b, towards):
    """Each bar's normal: `towards` ((n, 3), e.g. out of the part's nearest face at its middle) made square
    to the bar; a bar that runs along it gets any direction square to itself."""
    axis = b - a
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    n = np.asarray(towards, dtype=np.float64) - (np.asarray(towards) * axis).sum(axis=1, keepdims=True) * axis
    size = np.linalg.norm(n, axis=1)
    for i in np.flatnonzero(size < 0.2):
        other = np.eye(3)[int(np.argmin(np.abs(axis[i])))]
        n[i] = other - (other @ axis[i]) * axis[i]
        size[i] = np.linalg.norm(n[i])
    return n / size[:, None]


def _reach(angle, big, small, edge):
    """How far from their joint the loft between two bars leaves them, mm: LOFT x the larger bar; farther where
    they lie at a sharp angle, so that the bend's inside keeps a radius (LOFT_INNER), but never beyond where the
    two bars' flat faces part (a hole between the bars and the bend), and at most LOFT_MOST x the larger bar."""
    half = 0.5 * angle
    sin, cos, tan = math.sin(half), math.cos(half), math.tan(half)
    out = max(LOFT * big, (LOFT_INNER + 0.5) * big / max(tan, 1e-6))
    if half < 0.25 * math.pi:
        flat = max(0.5 * small - edge, 0.1 * small)
        out = min(out, max(0.95 * flat * (1.0 + 1.0 / max(sin, 1e-6)) * cos / (1.0 - sin), LOFT * small))
    return min(out, LOFT_MOST * big)


def lofts(a, b, normal, width, width_b, thick, branch, joints, edge=0.0):
    """Lofted joints. `joints`: a list of joints, each the member ends that meet there, (member, 0 = its start
    or 1 = its end); the pieces of a member (`branch`) follow each other. Every member is cut short of its
    joints, square to its line, and between every two members of a joint there is a loft: a curve that leaves
    the one member's line where it was cut, in its direction, and arrives on the other's the same way (a cubic
    Bezier: a circle's arc for two straight bars cut as far from where their lines cross), along which the
    width, the thickness and the faces go evenly from the one member's to the other's (a section turned a
    quarter with its sides swapped is the same section, so a loft twists 45 degrees at most). No member runs
    through its joint, and none ends in it. A member with both ends in one joint (a stub between two joints
    that count as one) is left out: the lofts between the others are what joins them.
    Returns a namespace: the members' pieces a, b, normal, width, width_b, thick, piece (the piece each came
    from), flat_a and flat_b (the ends that are cut square for a loft), and `curves`, the lofts for `body`: each
    a namespace of ctrl (4, 3: the Bezier), tau (k + 1,: where its spans begin and end), normal (k + 1, 3: the
    faces' normal there), width and thick (2,: at its two ends); and `joints`, for
    `fills`: each a namespace of centre, size (its bars' side) and core (how far from the centre its fill reaches)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    unit = lambda v: v / max(float(np.linalg.norm(v)), 1e-12)
    square = lambda v, axis: unit(v - (v @ axis) * axis)
    ease = lambda t: t * t * (3.0 - 2.0 * t)
    lines = {}
    for m in dict.fromkeys(np.asarray(branch).tolist()):
        of = np.flatnonzero(branch == m)
        if len(of) > 1 and not np.allclose(b[of[:-1]], a[of[1:]], atol=1e-6):
            continue      # (not one line: left as it is)
        lines[m] = (of, np.concatenate([[0.0], np.cumsum(np.linalg.norm(b[of] - a[of], axis=1))]))

    def at(end, s):
        """The member `s` mm from its end: the point, the way it runs on from there (away from the joint), its
        width, thickness and normal."""
        (m, which), (of, run) = end, lines[end[0]]
        r = min(max(s, 0.0), float(run[-1]))
        r = r if which == 0 else float(run[-1]) - r
        k = int(np.clip(np.searchsorted(run, r, side="right") - 1, 0, len(of) - 1))
        i = of[k]
        f = (r - run[k]) / max(float(run[k + 1] - run[k]), 1e-12)
        axis = (b[i] - a[i]) / max(float(np.linalg.norm(b[i] - a[i])), 1e-12)
        return (a[i] + f * (b[i] - a[i]), axis if which == 0 else -axis, float(width[i] + f * (width_b[i] - width[i])),
                float(thick[i]), np.asarray(normal[i], dtype=np.float64))

    joints = [[tuple(end) for end in ends if end[0] in lines] for ends in joints]
    gone = set()
    for ends in joints:
        gone |= {m for m, _ in ends if (m, 0) in ends and (m, 1) in ends}
    joints = [[end for end in ends if end[0] not in gone] for ends in joints]
    joints = [ends for ends in joints if len(ends) > 1]
    busy = {}
    for ends in joints:
        for m, _ in ends:
            busy[m] = busy.get(m, 0) + 1

    def bezier(one, other, first, second):
        """The loft's curve between two member ends, `first` and `second` mm from them: its control points,
        the ends' ways, widths, thicknesses and normals, how far it turns, and its tightest radius."""
        p0, t0, w0, h0, n0 = at(one, first)
        p3, t3, w3, h3, n3 = at(other, second)
        t0 = -t0                  # (towards the joint)
        chord = float(np.linalg.norm(p3 - p0))
        if chord < 1e-6:
            return None
        turn = math.acos(float(np.clip(t0 @ t3, -1.0, 1.0)))
        handle = chord / (3.0 * max(math.cos(0.25 * turn) ** 2, 0.3))
        ctrl = np.array([p0, p0 + handle * t0, p3 - handle * t3, p3])
        s = np.linspace(0.0, 1.0, 65)[:, None]
        speed = 3 * (1 - s) ** 2 * (ctrl[1] - ctrl[0]) + 6 * (1 - s) * s * (ctrl[2] - ctrl[1]) + 3 * s ** 2 * (ctrl[3] - ctrl[2])
        bend = 6 * (1 - s) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * s * (ctrl[3] - 2 * ctrl[2] + ctrl[1])
        tight = np.linalg.norm(np.cross(speed, bend), axis=1) / np.maximum(np.linalg.norm(speed, axis=1) ** 3, 1e-12)
        return SimpleNamespace(ctrl=ctrl, ways=(t0, t3), wide=(w0, w3), tall=(h0, h3), normals=(n0, n3), turn=turn,
                               radius=1.0 / max(float(tight.max()), 1e-9))

    cut, pairs, of_joint, big, bars_at = {}, [], [], [], []
    for ends in joints:
        most, ways, sides, here = {}, {}, {}, []
        bars_at.append([at(end, 0.0) for end in ends])
        for end in ends:
            length = float(lines[end[0]][1][-1])
            _, _, wide, tall, _ = at(end, 0.0)
            sides[end] = (max(wide, tall), min(wide, tall))
            most[end] = (LOFT_SHARE if busy[end[0]] > 1 else LOFT_ALONE) * length
            way = at(end, min(LOFT * sides[end][0], most[end]))[0] - at(end, 0.0)[0]
            ways[end] = way / max(float(np.linalg.norm(way)), 1e-12)
        for i, one in enumerate(ends):
            for other in ends[i + 1:]:
                angle = math.acos(float(np.clip(ways[one] @ ways[other], -1.0, 1.0)))
                far = _reach(angle, max(sides[one][0], sides[other][0]), min(sides[one][1], sides[other][1]), float(edge))
                first, second = min(far, most[one]), min(far, most[other])
                # (a short member gives its loft less room than that: the loft then takes as much more of the
                # other member, where that one has it, and bends less sharply for it)
                first, second = min(2.0 * far - second, most[one]), min(2.0 * far - first, most[other])
                made = bezier(one, other, first, second)
                if made is not None:
                    made.folded = made.radius < LOFT_FOLD * max(sides[one][0], sides[other][0])
                    here.append((one, other, first, second, made))
        # a loft that would bend more tightly than its section is wide (two bars that leave the joint side by
        # side: a hairpin between them, folded over itself) is left out where both its bars have other lofts
        others = {end: sum(1 for one, other, _, _, made in here if end in (one, other) and not made.folded) for end in ends}
        for one, other, first, second, made in here:
            if made.folded and others[one] and others[other]:
                continue
            pairs.append((one, other, first, second, made))
            of_joint.append(len(big))
            cut[one], cut[other] = min(cut.get(one, np.inf), first), min(cut.get(other, np.inf), second)
        big.append(float(np.median([sides[end][0] for end in ends])))
    out = {name: [] for name in ("a", "b", "normal", "width", "width_b", "thick", "piece", "flat_a", "flat_b")}
    # the members, cut short of their joints
    for m in dict.fromkeys(np.asarray(branch).tolist()):
        if m in gone:
            continue
        of = np.flatnonzero(branch == m)
        run = lines[m][1] if m in lines else None
        low = cut.get((m, 0), 0.0) if run is not None else 0.0
        high = float(run[-1]) - cut.get((m, 1), 0.0) if run is not None else np.inf
        for k, i in enumerate(of):
            f0, f1 = 0.0, 1.0
            if run is not None:
                span = max(float(run[k + 1] - run[k]), 1e-12)
                f0, f1 = (max(float(run[k]), low) - run[k]) / span, (min(float(run[k + 1]), high) - run[k]) / span
                if f1 - f0 < 1e-6:
                    continue
            values = dict(a=a[i] + f0 * (b[i] - a[i]), b=a[i] + f1 * (b[i] - a[i]), normal=np.asarray(normal[i], dtype=np.float64),
                          width=float(width[i] + f0 * (width_b[i] - width[i])),
                          width_b=float(width[i] + f1 * (width_b[i] - width[i])), thick=float(thick[i]), piece=int(i),
                          flat_a=bool(run is not None and low > 0 and run[k] <= low), flat_b=bool(run is not None and run[k + 1] >= high and (m, 1) in cut))
            for name, value in values.items():
                out[name].append(value)
    # each joint's plane (LOFT_PLANE): through the middles of its lofts, and its thickness there, the median
    # of its bars' sides that lie across that plane
    planes = {}
    for number in range(len(big) if LOFT_PLANE else 0):
        mids = np.array([(made.ctrl[0] + 3.0 * made.ctrl[1] + 3.0 * made.ctrl[2] + made.ctrl[3]) / 8.0
                         for (_, _, _, _, made), joint in zip(pairs, of_joint) if joint == number])
        if len(mids) < 3:
            continue
        centre = mids.mean(axis=0)
        spread, vectors = np.linalg.eigh((mids - centre).T @ (mids - centre))
        if math.sqrt(max(float(spread[1]), 0.0)) < LOFT_PLANE_SPREAD * big[number]:
            continue       # (the middles in a line: no plane)
        flat = vectors[:, 0]
        across = [tall if abs(float(n @ flat)) >= abs(float(np.cross(n, way) @ flat)) else wide
                  for _, way, wide, tall, n in bars_at[number]]
        planes[number] = (centre, flat, float(np.median(across)))
    # each joint's node (LOFT_NODE): the convex hull of its bars' ends where they are cut, as planes (a, b, c, d:
    # a x + b y + c z + d <= 0 inside)
    hulls = {}
    for number, ends in enumerate(joints if LOFT_NODE else ()):
        corners = []
        for end in ends:
            if end not in cut:
                continue
            p_, way_, wide_, tall_, n_ = at(end, cut[end])
            up_ = square(n_, way_)
            side_ = np.cross(up_, way_)
            corners += [p_ + 0.5 * su * wide_ * side_ + 0.5 * sn * tall_ * up_ for su in (-1.0, 1.0) for sn in (-1.0, 1.0)]
        if sum(joint == number for joint in of_joint) < 2 or len(corners) < 8:
            continue
        try:
            from scipy.spatial import ConvexHull
            hulls[number] = np.ascontiguousarray(ConvexHull(np.array(corners)).equations, dtype=np.float64)
        except Exception:      # (flat: no hull)
            continue
    # each joint's faces (LOFT_FACES): the plane its bars lie in most (its normal the one across which their ways
    # spread least), and per bar the face of its four that looks most that way: where it is cut, its way on, that
    # face's normal and how far it lies from the bar's middle. A bar whose faces all lie steeper than LOFT_FACE_TILT
    # to that plane (a post standing on it) has none, and its lofts are drawn as they are (`free`)
    faced, free = {}, set()
    for number, ends in enumerate(joints if LOFT_FACES else ()):
        here = [end for end in ends if end in cut]
        if sum(joint == number for joint in of_joint) < 2 or len(here) < 3:
            continue
        cuts = [at(end, cut[end]) for end in here]
        spread, vectors = np.linalg.eigh(sum(np.outer(way_, way_) for _, way_, _, _, _ in cuts))
        if spread[1] > 0.6 * spread[2]:      # (not one plane: the bars' normals' mean)
            first_ = square(cuts[0][4], cuts[0][1])
            up = unit(sum(square(n_, way_) * (1.0 if square(n_, way_) @ first_ >= 0 else -1.0) for _, way_, _, _, n_ in cuts))
        else:
            up = vectors[:, 0]
        rows = []
        for end, (p_, way_, wide_, tall_, n_) in zip(here, cuts):
            n_ = square(n_, way_)
            best = max(((n_, 0.5 * tall_, 0.5 * wide_), (np.cross(n_, way_), 0.5 * wide_, 0.5 * tall_)), key=lambda c: abs(float(c[0] @ up)))
            if abs(float(best[0] @ up)) < LOFT_FACE_TILT:
                free.add(end)
                continue
            rows.append(np.concatenate([p_, way_, best[0] * np.sign(float(best[0] @ up)), [best[1], best[2]]]))
        if len(rows) >= 2:
            faced[number] = np.ascontiguousarray(rows, dtype=np.float64)
    # the lofts
    curves = []
    for (one, other, first, second, made), joint in zip(pairs, of_joint):
        ctrl, (t0, t3), (w0, w3), (h0, h3), (n0, n3), turn = made.ctrl, made.ways, made.wide, made.tall, made.normals, made.turn
        point = lambda s: ((1 - s) ** 3 * ctrl[0] + 3 * (1 - s) ** 2 * s * ctrl[1] + 3 * (1 - s) * s ** 2 * ctrl[2] + s ** 3 * ctrl[3])
        way = lambda s: unit(3 * (1 - s) ** 2 * (ctrl[1] - ctrl[0]) + 6 * (1 - s) * s * (ctrl[2] - ctrl[1])
                             + 3 * s ** 2 * (ctrl[3] - ctrl[2]))

        def carry(start, steps):
            """A normal carried along the curve over `steps` (its parameters), kept square to it."""
            out_ = [start]
            for s in steps:
                out_.append(square(out_[-1], way(float(s))))
            return out_

        def turned(level, end_way, now):
            """Which of a section's four sides (its normal `level` either way, or its side) is nearest `now`,
            whether that is a side, and the angle `now` turns about `end_way` to reach it."""
            side = np.cross(level, end_way)
            best = max(((level, False), (-level, False), (side, True), (-side, True)), key=lambda c: float(c[0] @ now))
            return best[0], best[1], math.atan2(float(best[0] @ np.cross(now, end_way)), float(best[0] @ now))
        # the first member's normal carried along the curve, and what it has to turn to become the other's; with
        # a plane, to become the plane's at the middle first (the faces of all the joint's lofts lie in its slab
        # there), then the other's
        fine = np.linspace(0.0, 1.0, 129)
        carried = carry(square(n0, t0), fine[1:])
        middle, shift = None, np.zeros(2)
        if joint in planes:
            centre, flat, across = planes[joint]
            m = way(0.5)
            cos = math.sqrt(max(1.0 - float(flat @ m) ** 2, 0.0))
            off = float(flat @ (point(0.5) - centre))
            if cos >= LOFT_PLANE_TILT and abs(off) <= LOFT_PLANE_OFF * across:
                level, sideways, twist_a = turned(square(flat, m), m, carried[64])
                axis_ = level if not sideways else np.cross(level, m)        # (the section's side across the plane)
                lean = float(axis_ @ flat)
                middle = (level, sideways, twist_a, across / abs(lean))
                shift[1 if sideways else 0] = -off / lean
        if middle is None:
            _, swap, twist = turned(square(n3, t3), t3, carried[-1])
            later = None
        else:
            later = carry(middle[0], fine[65:])
            _, swap, twist = turned(square(n3, t3), t3, later[-1])
        if swap:
            w3, h3 = h3, w3

        def frame(s):
            k = int(round(float(s) * 128))
            if middle is None:
                base, angle = carried[k], twist * ease(float(s))
            elif k <= 64:
                base, angle = carried[k], middle[2] * ease(2.0 * float(s))
            else:
                base, angle = later[k - 64], twist * ease(2.0 * float(s) - 1.0)
            return math.cos(angle) * base + math.sin(angle) * np.cross(base, way(float(s)))
        # as many spans as keep the curve's turn and the faces' within LOFT_TURN degrees a span (the faces turn
        # with the curve where it leaves its plane, and fastest in the loft's middle: 31 degrees in one of five
        # spans by the turn and twist as a whole); an even count, so that the middle is where two spans meet
        count = int(np.clip(math.ceil(math.degrees(turn) / LOFT_TURN), 4, 32))
        while True:
            count += count % 2 if middle is not None else 0
            tau = np.linspace(0.0, 1.0, count + 1)
            frames = [frame(s) for s in tau]
            most_ = max(math.degrees(math.acos(float(np.clip(one_ @ next_, -1.0, 1.0)))) for one_, next_ in zip(frames, frames[1:]))
            if most_ <= LOFT_TURN or count >= 32:
                break
            count = min(int(math.ceil(count * max(most_ / LOFT_TURN, 1.25))), 32)
        width_, thick_ = np.array([w0, w3]), np.array([h0, h3])
        if middle is not None:
            # in the middle the section's side across the plane is the joint's thickness (as measured across the
            # plane), the other side half way between the ends'
            width_ = np.array([w0, middle[3] if middle[1] else 0.5 * (w0 + w3), w3])
            thick_ = np.array([h0, 0.5 * (h0 + h3) if middle[1] else middle[3], h3])
        curves.append(SimpleNamespace(ctrl=ctrl, tau=tau, normal=np.array(frames), width=width_, thick=thick_, shift=shift, joint=joint,
                                      hull=hulls.get(joint), faces=faced.get(joint), free=bool({one, other} & free),
                                      points=np.array([point(float(s)) for s in fine])))
    # the joints, for their fills (`fills`): the middle of each one's lofts, its bars' side, and how far its lofts'
    # middles lie from that middle and a little more
    places = []
    for number, size in enumerate(big):
        mids = np.array([curve.points[len(curve.points) // 2] for curve, joint in zip(curves, of_joint) if joint == number])
        if len(mids):
            centre = mids.mean(axis=0)
            places.append(SimpleNamespace(centre=centre, size=size,
                                          core=float(np.linalg.norm(mids - centre, axis=1).max()) + FILL_CORE * size))
    kinds = {"piece": np.int64, "flat_a": bool, "flat_b": bool}
    made = {name: np.array(values, dtype=kinds.get(name, np.float64)) for name, values in out.items()}
    made["a"], made["b"], made["normal"] = (made[name].reshape(-1, 3) for name in ("a", "b", "normal"))
    return SimpleNamespace(curves=curves, joints=places, **made)


def flow(values, voxel, time, sharp, hold=None, cancel=None, ridge=None, shallow=None):
    """`values` (negative inside, about the distance near their zero surface, samples `voxel` mm apart, smooth)
    with that surface moved outwards, material added, for `time` mm2 by what tells a defect from a shape that is
    meant: by its larger principal curvature where that is hollow, which is a place hollow in every direction, a
    pit (it fills until it is straight one way: flat, for a pit with a rim), and by what its smaller principal
    curvature is hollower than 1 / `sharp`, a groove sharper than that radius (a crease between two faces, the
    foot of a step: it is rounded to that radius). Nothing else moves: a flat or hollow face, a bend between two
    bars (hollow one way, straight the other, and rounder than `sharp`), the wall of a round opening wider than
    2 x `sharp`, the walls of a slot. A level set flow, values -= dt x speed x |gradient|, in explicit steps of
    FILL_STEP samples^2 on the samples near the surface (found again every FILL_EVERY steps, when `cancel()` is
    called), every sample from the values before the step, and none where the values turn back within a sample
    (FILL_GATE: the middle of a gap). `hold` (0..1 per sample): how much of the motion each sample gets.
    `ridge`: the surface also moves inwards, material taken off, by what its larger principal curvature is
    rounder than 1 / `ridge`: an edge sharper than that radius (the top of a step, a line where one loft's
    face stands proud of another's) is rounded to it; `shallow` (0..1 per sample): how much of that each sample gets.
    (A ball rolled over the solid instead closes every opening it does not fit through and bridges gaps, and the
    rules that kept it out of those took most of its fills away; smoothing the solid lays a film in every bend.)"""
    steps = int(math.ceil(time / (FILL_STEP * voxel * voxel)))
    out = np.array(values, dtype=np.float32, order="C")
    if steps < 1:
        return out
    dt = np.float32(time / steps)
    flat = out.reshape(-1)
    held = None if hold is None else np.ascontiguousarray(hold, dtype=np.float32).reshape(-1)
    stride = (out.shape[1] * out.shape[2], out.shape[2], 1)
    inner = np.zeros(out.shape, dtype=bool)
    inner[1:-1, 1:-1, 1:-1] = True
    if hold is not None:
        inner &= np.asarray(hold) > 0
    first, second = np.float32(0.5 / voxel), np.float32(1.0 / (voxel * voxel))
    cap, groove = np.float32(1.0 / voxel), np.float32(1.0 / sharp)
    edge = None if not ridge else np.float32(1.0 / ridge)
    near, far = FILL_BAND
    low, high = FILL_GATE
    fade = lambda t: t * t * (3.0 - 2.0 * t)
    at = None
    for k in range(steps):
        if k % FILL_EVERY == 0:
            if cancel is not None:
                cancel()
            # (a sample's value changes by at most FILL_STEP of a sample a step)
            at = np.flatnonzero((np.abs(out) < (far + FILL_STEP * FILL_EVERY + 0.5) * voxel) & inner)
        if not at.size:
            break
        c = flat[at]
        g, d2 = [], {}
        sides = 0.0
        for a in range(3):
            lo, hi = flat[at - stride[a]], flat[at + stride[a]]
            g.append((hi - lo) * first)
            d2[a, a] = (hi - 2.0 * c + lo) * second
            sides = sides + (hi - c) * (hi - c) + (c - lo) * (c - lo)
        for a in range(3):
            for b in range(a + 1, 3):
                d2[a, b] = (flat[at + stride[a] + stride[b]] - flat[at + stride[a] - stride[b]]
                            - flat[at - stride[a] + stride[b]] + flat[at - stride[a] - stride[b]]) * (0.25 * second)
        x, y, z = g
        norm2 = np.maximum(x * x + y * y + z * z, 1e-12)
        norm = np.sqrt(norm2)
        # the two principal curvatures of the level set through each sample, from their sum and product (a
        # ball's are positive, a hollow's negative)
        total = ((d2[1, 1] + d2[2, 2]) * x * x + (d2[0, 0] + d2[2, 2]) * y * y + (d2[0, 0] + d2[1, 1]) * z * z
                 - 2.0 * (x * y * d2[0, 1] + x * z * d2[0, 2] + y * z * d2[1, 2])) / (norm2 * norm)
        gauss = (x * x * (d2[1, 1] * d2[2, 2] - d2[1, 2] * d2[1, 2]) + y * y * (d2[0, 0] * d2[2, 2] - d2[0, 2] * d2[0, 2])
                 + z * z * (d2[0, 0] * d2[1, 1] - d2[0, 1] * d2[0, 1])
                 + 2.0 * (x * y * (d2[0, 2] * d2[1, 2] - d2[0, 1] * d2[2, 2]) + y * z * (d2[0, 1] * d2[0, 2] - d2[1, 2] * d2[0, 0])
                          + x * z * (d2[0, 1] * d2[1, 2] - d2[0, 2] * d2[1, 1]))) / (norm2 * norm2)
        apart = np.sqrt(np.maximum(total * total - 4.0 * gauss, 0.0))
        speed = np.maximum(-0.5 * (total + apart), 0.0) + np.maximum(-0.5 * (total - apart) - groove, 0.0)
        if edge is not None:
            cut = np.maximum(0.5 * (total + apart) - edge, 0.0)
            speed -= cut if shallow is None else cut * shallow.reshape(-1)[at]
        np.clip(speed, -cap, cap, out=speed)
        weight = fade(np.clip((far - np.abs(c) / np.float32(voxel)) / np.float32(far - near), 0.0, 1.0))
        # 1 where the values run straight through the sample, 0 where they turn back in it
        weight *= fade(np.clip((norm / np.sqrt(np.maximum(sides * (0.5 * second), 1e-12)) - np.float32(low))
                               / np.float32(high - low), 0.0, 1.0))
        if held is not None:
            weight *= held[at]
        flat[at] = c - speed * norm * weight * dt
    return out


def fills(read, joints, cancel=None, seams=None):
    """The dimples and creases of lofted joints, filled. Three lofts round three bars cover the joint's middle
    each with its own faces, which turn from the one bar's to the other's: where they overlap the joint's face
    is several faces with grooves and small steps between them, and where the bars do not lie in one plane a
    dimple is left on the outside of the corner. What sets these apart from everything that is meant to be
    hollow is their form: a pit is hollow in every direction and a crease is sharp, where a bend between two bars
    is hollow one way only and round inside (LOFT_INNER), as an opening between bars is. For each joint (`lofts`'
    joints: centre, size, core) the bars' body is read on samples FILL_VOXEL apart round it (`read(grid)` -> its
    values there), smoothed over a sample (FILL_BLUR) and its surface moved out by `flow` for FILL_TIME x the
    joint's bars' side^2: pits fill flat, and grooves sharper than FILL_SHARP x the side are rounded to that.
    What that adds within the joint's core is kept as a patch for `body`: a namespace of samples (a
    `fields.sampled_body`'s: a cubic B-spline), start, voxel and the box lo, hi it is used in; it lies
    FILL_UNDER under what the flow leaves, so a face nothing was added to is the bars' own, and the flow fades
    out past the core."""
    from ..grid import Grid
    out = []
    for joint in joints:
        if cancel is not None:
            cancel()
        fade = FILL_FADE * joint.size
        reach_ = joint.core + fade
        centre = np.asarray(joint.centre, dtype=np.float64)
        wide = reach_ + (FILL_BAND[1] + 2.0) * FILL_VOXEL
        steps = int(math.ceil(2.0 * wide / FILL_VOXEL)) + 1
        start = centre - wide
        values = np.asarray(read(Grid.block(start, FILL_VOXEL, (0, 0, 0), (steps, steps, steps))), dtype=np.float32)
        x, y, z = ((np.float32(FILL_VOXEL) * np.arange(steps, dtype=np.float32) - np.float32(wide)).reshape(
            [-1 if a == k else 1 for a in range(3)]) for k in range(3))
        away = np.clip((np.sqrt(x * x + y * y + z * z) - np.float32(joint.core)) / np.float32(fade), 0.0, 1.0)
        away = away * away * (3.0 - 2.0 * away)
        smooth = ndimage.gaussian_filter(values, FILL_BLUR)
        shallow = None
        if FILL_RIDGE and seams is not None:
            # edges are rounded only where two lofts or bars meet at an angle (`seams(grid)`: the second nearest's
            # values): a step, a line where they cross. A bar's own edge, a rim of the joint, has no other near it,
            # and two lofts that leave one bar lie on each other there and part slowly, turned hardly at all
            other = np.asarray(seams(Grid.block(start, FILL_VOXEL, (0, 0, 0), (steps, steps, steps))), dtype=np.float32)
            ramp = lambda v, lo, hi: np.clip((v - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)
            one, two = np.stack(np.gradient(values)), np.stack(np.gradient(other))
            turned = 1.0 - (one * two).sum(axis=0) / np.maximum(np.linalg.norm(one, axis=0) * np.linalg.norm(two, axis=0), 1e-12)
            low, high = (1.0 - math.cos(math.radians(angle)) for angle in FILL_CROSS)
            # (|other| at the surface: where the second's surface comes through it, not one lying under it in the
            # body, as everywhere in a joint the lofts overlap)
            shallow = (1.0 - ramp(np.abs(other), *FILL_SEAM)) * ramp(turned, low, high)
            shallow = ndimage.gaussian_filter((shallow * shallow * (3.0 - 2.0 * shallow)).astype(np.float32), FILL_BLUR)
            del one, two, turned, other
        flowed = flow(smooth, FILL_VOXEL, FILL_TIME * joint.size * joint.size, FILL_SHARP * joint.size, 1.0 - away, cancel,
                      None if shallow is None else FILL_RIDGE * joint.size, shallow)
        patch = (flowed + np.float32(FILL_UNDER) + np.float32(joint.size) * away).astype(np.float32)
        if ((patch < 0) & (values > 0)).any():
            out.append(SimpleNamespace(samples=patch, start=start, voxel=FILL_VOXEL, lo=centre - reach_, hi=centre + reach_,
                                       cut=False))
        if shallow is not None:
            # what the flow took off, the same way: a patch the bars' values are raised to (`cut`)
            patch = (flowed - np.float32(FILL_UNDER) - np.float32(joint.size) * away).astype(np.float32)
            if ((patch > 0) & (values < 0)).any():
                out.append(SimpleNamespace(samples=patch, start=start, voxel=FILL_VOXEL, lo=centre - reach_, hi=centre + reach_,
                                           cut=True))
    return out


def _biharmonic(values, unknown):
    """`values` with those at `unknown` replaced so that their biharmonic (the Laplacian of the Laplacian, on the
    samples) is 0 there: a thin plate through the rest, as smooth as it can be where it meets them."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    at = np.argwhere(unknown)
    lo = np.maximum(at.min(axis=0) - 3, 0)
    hi = np.minimum(at.max(axis=0) + 4, np.array(values.shape))
    box = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    sub, free = values[box].astype(np.float64), unknown[box]
    shape, n = sub.shape, sub.size
    ids = np.arange(n).reshape(shape)
    rows, cols, vals = [ids.ravel()], [ids.ravel()], [np.full(n, -6.0)]
    for a in range(3):
        for step in (-1, 1):
            inside = np.ones(shape, dtype=bool)
            inside[(slice(None),) * a + (0 if step == 1 else -1,)] = False
            rows += [ids[inside], ids[~inside]]
            cols += [np.roll(ids, step, axis=a)[inside], ids[~inside]]     # (at the box's edge: the sample itself)
            vals += [np.ones(int(inside.sum())), np.ones(int((~inside).sum()))]
    lap = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    twice = (lap @ lap).tocsr()
    f = free.ravel()
    solved = sub.ravel().copy()
    solved[f] = spsolve(twice[f][:, f].tocsc(), -(twice[f][:, ~f] @ solved[~f]))
    out = np.array(values, dtype=np.float32)
    out[box] = solved.reshape(shape).astype(np.float32)
    return out


def patches(read, joints, seams, cancel=None):
    """Lofted joints patched, as Rhino's Patch skins a gap: each joint's surface is kept where it is one loft's or
    bar's alone (`seams(grid)`, the second nearest's values, lies PATCH_TRUST x the side beyond it), and only where
    such a surface runs on out of the joint (a patch of one loft stranded inside the joint made a lump); the rest
    of the joint's surface, where its lofts overlap with steps and lines between them, is fitted again: within the
    joint's core the values are those of a thin plate through the rest (`_biharmonic`, on every PATCH_STEP-th sample:
    it is smooth), free in the air (so the surface finds its own smooth form) and held deep inside the bars, deeper
    than PATCH_INNER x the side or along each bar's middle however thin (where the values are near the deepest of
    their PATCH_MIDDLE x the side: with the whole core free a joint's middle fell away into five pieces, held only
    PATCH_INNER deep a thin bar fell off). Each is a patch for `body`: samples, and `weight` 0..1, how much they
    replace the bars' values (`mode` "replace")."""
    from ..grid import Grid
    out = []
    for joint in joints:
        if cancel is not None:
            cancel()
        size = float(joint.size)
        trust, band, inner = PATCH_TRUST * size, PATCH_BAND * size, PATCH_INNER * size
        centre = np.asarray(joint.centre, dtype=np.float64)
        wide = float(joint.core) + band + 4.0 * PATCH_STEP * FILL_VOXEL
        steps = 2 * int(math.ceil(wide / (PATCH_STEP * FILL_VOXEL))) * PATCH_STEP + 1
        start = centre - 0.5 * (steps - 1) * FILL_VOXEL
        grid = Grid.block(start, FILL_VOXEL, (0, 0, 0), (steps, steps, steps))
        values = np.asarray(read(grid), dtype=np.float32)
        other = np.asarray(seams(grid), dtype=np.float32)
        axis = (FILL_VOXEL * np.arange(steps) - 0.5 * (steps - 1) * FILL_VOXEL).astype(np.float32)
        r = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
        kept = ndimage.binary_dilation((np.abs(values) < band) & (other - values > trust))
        label, _ = ndimage.label(kept, structure=CUBE)
        reach = np.unique(label[kept & (r >= joint.core)])
        kept = np.isin(label, reach[reach > 0])
        middle = ndimage.minimum_filter(values, size=2 * int(round(0.5 * PATCH_MIDDLE * size / FILL_VOXEL)) + 1)
        held = (values < -inner) | ((values < -0.1 * size) & (values <= 0.8 * middle))
        unknown = (r < joint.core) & ~kept & ~held
        if not unknown.any():
            continue
        coarse = (slice(None, None, PATCH_STEP),) * 3
        small = _biharmonic(values[coarse], unknown[coarse])
        back = np.meshgrid(*(np.arange(steps, dtype=np.float64) / PATCH_STEP,) * 3, indexing="ij")
        solved = np.where(unknown, ndimage.map_coordinates(small, back, order=3, mode="nearest"), values).astype(np.float32)
        weight = np.clip(1.5 * ndimage.gaussian_filter(unknown.astype(np.float32), 1.0), 0.0, 1.0).astype(np.float32)
        reach_ = float(joint.core) + 2.0 * FILL_VOXEL
        out.append(SimpleNamespace(samples=solved, weight=weight, start=start, voxel=FILL_VOXEL, lo=centre - reach_,
                                   hi=centre + reach_, cut=False, mode="replace"))
    return out


def body(a, b, normal, width, thick, blend=0.0, edge=0.0, edge_shape="round", reach=2.0, width_b=None, member=None,
         layer=None, knots=None, knot_size=0.0, knot_blend=0.0, knot_reach=1.0, knot_axis=None, knot_span=None, thick_b=None, twist=None, chain=None,
         flat_a=None, flat_b=None, curves=(), fills=(), seams=False):
    """The bars as an implicit body (see the module doc). `a`, `b`: (n, 3) mm, `normal`: (n, 3) vectors
    across the bars (made square to them), `width` (across, in the plane square to the normal; at `a`, and
    `width_b` at `b` when it differs: a taper) and `thick` (along the normal): (n,) mm. `member`: (n,) which
    member each piece is of (pieces of one member are joined plainly, members blended; none = every piece its
    own). `blend`: the width of the G2 blend where members meet (as wide as `nodes_fea.FILLET_SIZE` x a
    fillet's radius): one number, or (n,) per piece, a member's being the mean of its pieces'; the members are
    then joined widest first, each to what is there by its own blend (the narrower of two bars decides the
    fillet between them, as in the frames). `layer`: (n,) which layer each piece's member lies in (members
    in one plane: a wall's); only members of one layer are blended, the layers joined plainly (none = all one
    layer). A blend is a fillet in the bars' plane; between a bar in a front wall and one in the back wall it
    was a web from wall to wall, cut to the two slabs: a patch of skin beside every joint, and voids under it.
    `knots`: (m, 3) mm, the joints where bars of different layers meet (a frame in 3D): there the finished layers
    are joined with a blend in space, `knot_blend` wide ((m,) or one number) within `knot_reach` of the joint and
    fading to none at twice that, round a ball of radius `knot_size` at the joint, which the bars' ends lie in
    (joined plainly, the bars of two planes showed their round ends and corners beside each other); a knot with a
    `knot_axis` ((m, 3), a unit vector; zeros = the ball) is a round boss along it instead, `knot_size` its radius, from
    `knot_span[0]` to `knot_span[1]` mm of the joint along the axis (a joint at the part's face, which cuts it flush). `edge` / `edge_shape`: the rims' chamfer (its legs) or round (its radius), mm.
    `curves`: lofts (`lofts`), each a bar along a cubic Bezier whose width, thickness and faces go evenly from
    its one end's to its other's; each is worked out as it is, on the samples' nearest points of the curve (as
    straight pieces 6 degrees apart its faces showed every piece as a rib, and its tight bends slivers), a
    layer of its own, joined plainly. `flat_a` / `flat_b`: (n,) the piece ends cut square to the piece (where
    a loft takes over: a round end stood out of a loft that bends away across the bar's face). `fills`: the
    lofted joints' dimples filled (`fills`): each a patch of samples, joined plainly in its box. Values
    are right `reach` mm out and that beyond. `seams`: instead of the body, the second nearest of the layers
    and lofts (the second smallest of their values), for `fills`: where it is near the body's surface and turned
    from it, two meet at an angle."""
    a, b = np.ascontiguousarray(a, dtype=np.float64).reshape(-1, 3), np.ascontiguousarray(b, dtype=np.float64).reshape(-1, 3)
    if not len(a):
        raise ValueError("no bars")
    spread = lambda v: np.ascontiguousarray(np.broadcast_to(v, (len(a),)), dtype=np.float64)
    width, thick = spread(width), spread(thick)
    width_b = width if width_b is None else spread(width_b)
    thick_b = thick if thick_b is None else spread(thick_b)
    twist = np.zeros(len(a)) if twist is None else spread(twist)
    member = np.arange(len(a)) if member is None else np.asarray(member, dtype=np.int64)
    layer = np.zeros(len(a), dtype=np.int64) if layer is None else np.asarray(layer, dtype=np.int64)
    length = np.linalg.norm(b - a, axis=1)
    axis = (b - a) / length[:, None]
    normal = np.ascontiguousarray(normal, dtype=np.float64).reshape(-1, 3)
    normal = normal - (normal * axis).sum(axis=1, keepdims=True) * axis
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    across = np.cross(normal, axis)
    # the tapered outline's side: at `slope` to the bar's line (iq's uneven capsule, in the bar's plane)
    slope = np.clip(0.5 * (width - width_b) / length, -0.98, 0.98)
    level = np.sqrt(1.0 - slope * slope)
    blends = spread(blend)
    # pieces of one chain that follow each other meet on the plane halving their bend (a mitre), not in round
    # ends: a round end lies in its own piece's plane, and where the line leaves that plane by a few degrees it
    # stood out of the next piece's face, a rib across the bar at every piece
    mitre_a, mitre_b = np.zeros((len(a), 3)), np.zeros((len(a), 3))
    if chain is not None:
        chain = np.asarray(chain)
        for i in range(len(a) - 1):
            if chain[i] == chain[i + 1] and np.allclose(b[i], a[i + 1], atol=1e-6):
                halving = axis[i] + axis[i + 1]
                if float(np.linalg.norm(halving)) > 1.0:      # (not folded back on itself)
                    mitre_b[i] = mitre_a[i + 1] = halving / float(np.linalg.norm(halving))
    for ends, cuts in ((flat_a, mitre_a), (flat_b, mitre_b)):
        if ends is not None:
            cuts[np.asarray(ends, dtype=bool)] = axis[np.asarray(ends, dtype=bool)]
    edge, reach = float(edge), float(reach)
    extent = np.sqrt((0.5 * np.maximum(width, width_b)) ** 2 + (0.5 * np.maximum(thick, thick_b)) ** 2)   # from a bar's line to its farthest corner
    lo, hi = np.minimum(a, b) - extent[:, None], np.maximum(a, b) + extent[:, None]
    # the lofts: per span of each, the curve's points there (for the samples it is worked out on)
    patches = [(Shared(np.ascontiguousarray(fill.samples, dtype=np.float32)), tuple(float(v) for v in fill.start),
                float(fill.voxel), np.asarray(fill.lo, dtype=np.float64), np.asarray(fill.hi, dtype=np.float64),
                bool(getattr(fill, "cut", False)),
                None if getattr(fill, "weight", None) is None else Shared(np.ascontiguousarray(fill.weight, dtype=np.float32)))
               for fill in fills]
    lofts_ = []
    for curve in curves:
        ctrl, tau = np.asarray(curve.ctrl, dtype=np.float64), np.asarray(curve.tau, dtype=np.float64)
        fine = np.asarray(curve.points, dtype=np.float64)
        shift = np.asarray(getattr(curve, "shift", (0.0, 0.0)), dtype=np.float64)
        out_ = math.sqrt((0.5 * float(np.max(curve.width)) + abs(shift[1])) ** 2
                         + (0.5 * float(np.max(curve.thick)) + abs(shift[0])) ** 2)
        at_ = np.linspace(0.0, 1.0, len(fine))
        spans = []
        for s in range(len(tau) - 1):
            near = fine[(at_ >= tau[s] - 1e-9) & (at_ <= tau[s + 1] + 1e-9)]
            spans.append((near.min(axis=0) - out_, near.max(axis=0) + out_))
        lofts_.append((ctrl, tau, np.asarray(curve.normal, dtype=np.float64), np.asarray(curve.width, dtype=np.float64),
                       np.asarray(curve.thick, dtype=np.float64), spans, fine.min(axis=0) - out_, fine.max(axis=0) + out_, shift,
                       getattr(curve, "joint", None), getattr(curve, "hull", None), getattr(curve, "faces", None),
                       bool(getattr(curve, "free", False))))
    groups = [np.flatnonzero(member == m) for m in dict.fromkeys(member.tolist())]   # (in order of appearance)
    if np.ptp(blends) > 0:   # widest first (the order of appearance where they are as wide)
        groups.sort(key=lambda pieces: -float(np.mean(0.5 * (width[pieces] + width_b[pieces]))))
    own_blend = [float(blends[pieces].mean()) for pieces in groups]
    blend = max(own_blend)
    layers = [[k for k, pieces in enumerate(groups) if layer[pieces[0]] == which] for which in dict.fromkeys(layer.tolist())]
    knots = np.zeros((0, 3)) if knots is None else np.ascontiguousarray(knots, dtype=np.float64).reshape(-1, 3)
    several = lambda v: np.ascontiguousarray(np.broadcast_to(v, (len(knots),)), dtype=np.float64)
    knot_size, knot_blend, knot_reach = several(knot_size), several(knot_blend), several(knot_reach)
    knot_axis = np.zeros((len(knots), 3)) if knot_axis is None else np.ascontiguousarray(knot_axis, dtype=np.float64).reshape(-1, 3)
    knot_span = np.zeros((len(knots), 2)) if knot_span is None else np.ascontiguousarray(knot_span, dtype=np.float64).reshape(-1, 2)
    half = np.float32(math.sqrt(0.5))

    def fn(ctx):
        xp, grid = ctx.xp, ctx.grid
        far = np.float32(max(PAD * grid.voxel, reach))
        k = np.float32(blend)
        pad = float(far) + blend
        axes = grid.axes()
        soften = balls = None
        if len(knots):
            # how wide the blend between the layers is at each sample (0 away from the knots), and the knots' balls
            soften = xp.zeros(grid.shape, dtype=xp.float32)
            balls = xp.full(grid.shape, far, dtype=xp.float32)
            for j in range(len(knots)):
                out_ = max(2.0 * knot_reach[j], knot_size[j] + float(np.abs(knot_span[j]).max()) + float(far))
                first, last = _window(grid, knots[j] - out_, knots[j] + out_, 0.0)
                if np.any(last <= first):
                    continue
                x, y, z = (xp.asarray((axes[d][first[d]:last[d]] - knots[j][d]).astype(np.float32)).reshape(
                    [-1 if e == d else 1 for e in range(3)]) for d in range(3))
                d = xp.sqrt(x * x + y * y + z * z)
                w = xp.clip(np.float32(2.0) - d / np.float32(knot_reach[j]), 0.0, 1.0)
                box = tuple(slice(int(first[d_]), int(last[d_])) for d_ in range(3))
                block = soften[box]
                xp.maximum(block, np.float32(knot_blend[j]) * w * w * (np.float32(3.0) - np.float32(2.0) * w), out=block)
                block = balls[box]
                if np.any(knot_axis[j]):
                    # a pad: a round boss along the axis, from span[0] to span[1] of the joint
                    ux, uy, uz = (np.float32(c) for c in knot_axis[j])
                    h = x * ux + y * uy + z * uz
                    side = xp.sqrt(xp.maximum(d * d - h * h, 0.0)) - np.float32(knot_size[j])
                    ends = xp.maximum(np.float32(knot_span[j][0]) - h, h - np.float32(knot_span[j][1]))
                    xp.minimum(block, xp.minimum(xp.maximum(side, ends), 0.0)
                               + xp.sqrt(xp.maximum(side, 0.0) ** 2 + xp.maximum(ends, 0.0) ** 2), out=block)
                else:
                    xp.minimum(block, d - np.float32(knot_size[j]), out=block)

        def joined(one, other, soft):
            if soft is None:
                return xp.minimum(one, other)
            h = xp.maximum(1.0 - xp.abs(one - other) / xp.maximum(soft, np.float32(1e-6)), 0.0)
            return xp.minimum(one, other) - bulge(h, soft, 2)
        # (the start value lies a blend's width and more past `far`: no layer's values up to `far` blend with it)
        whole = xp.full(grid.shape, far + k + np.float32(1.0), dtype=xp.float32)
        second = xp.full(grid.shape, far + k + np.float32(1.0), dtype=xp.float32) if seams else None

        def behind(place, out):
            if second is not None:    # (the second smallest of the values so far)
                block = second[place]
                xp.minimum(block, xp.maximum(whole[place], out), out=block)
        for members in layers:
            # a layer is worked out on the samples round its own bars only (a loft is a layer of its own: with
            # every layer on the whole grid, a part with 40 lofts filled the grid 80 times)
            if not members:
                continue
            of = np.concatenate([groups[m] for m in members])
            origin, top = _window(grid, lo[of].min(axis=0), hi[of].max(axis=0), pad)
            if np.any(top <= origin):
                continue
            shape = tuple(int(n) for n in top - origin)
            # (the blended outlines start a blend's width past `far`: a bar's side would blend with the start value)
            sides = xp.full(shape, far + k, dtype=xp.float32)   # the blended outlines, on the taller prisms
            faces = xp.full(shape, far, dtype=xp.float32)       # the bars' own slabs
            for pieces, soft in ((groups[m], own_blend[m]) for m in members):
                # a slab reaches a little past its bar's outline: under the fillets by it (a blend adds at most a
                # sixth of its width, by the nearer bar), its sides out of the way of the rims' chamfer. (As far as
                # the whole blend, a slab met the taller prism of a bar in another plane: a tab beside every eye.)
                wider = np.float32(soft / 6.0 + edge + 1.0)
                start, end = _window(grid, lo[pieces].min(axis=0), hi[pieces].max(axis=0), pad)
                if np.any(end <= start):
                    continue
                own = xp.full(tuple(int(n) for n in end - start), far + k, dtype=xp.float32)   # this member's outline
                for i in pieces:
                    first, last = _window(grid, lo[i], hi[i], pad)
                    if np.any(last <= first):
                        continue
                    x, y, z = (xp.asarray((axes[d][first[d]:last[d]] - a[i][d]).astype(np.float32)).reshape(
                        [-1 if e == d else 1 for e in range(3)]) for d in range(3))
                    f = lambda v: tuple(np.float32(c) for c in v)
                    (ax, ay, az), (nx, ny, nz), (ux, uy, uz) = f(axis[i]), f(normal[i]), f(across[i])
                    t = x * ax + y * ay + z * az
                    run, rise, long = np.float32(level[i]), np.float32(slope[i]), np.float32(length[i])
                    tall = np.float32(thick[i])
                    if twist[i] != 0.0 or thick_b[i] != thick[i]:
                        # the section turns about the bar's line and its thickness changes evenly from a to b
                        share = xp.clip(t / long, 0.0, 1.0)
                        tall = tall + np.float32(thick_b[i] - thick[i]) * share
                        su, sn = x * ux + y * uy + z * uz, x * nx + y * ny + z * nz
                        turn = np.float32(twist[i]) * share
                        cs, sn_ = xp.cos(turn), xp.sin(turn)
                        rho, q = xp.abs(cs * su - sn_ * sn), xp.abs(cs * sn + sn_ * su) - np.float32(0.5) * tall
                        del share, su, sn, turn, cs, sn_
                    else:
                        rho = xp.abs(x * ux + y * uy + z * uz)
                        q = xp.abs(x * nx + y * ny + z * nz) - np.float32(0.5 * thick[i])
                    where = run * t - rise * rho
                    flank = run * rho + rise * t - np.float32(0.5 * width[i])
                    c, cut = flank, None
                    if np.any(mitre_b[i]):
                        mx, my, mz = f(mitre_b[i])
                        ex, ey, ez = f(b[i] - a[i])
                        cut = (x - ex) * mx + (y - ey) * my + (z - ez) * mz
                    else:
                        c = xp.where(where > run * long, xp.sqrt(rho * rho + (t - long) ** 2) - np.float32(0.5 * width_b[i]), c)
                    if np.any(mitre_a[i]):
                        mx, my, mz = f(mitre_a[i])
                        start_ = -(x * mx + y * my + z * mz)
                        cut = start_ if cut is None else xp.maximum(cut, start_)
                    else:
                        c = xp.where(where < 0, xp.sqrt(rho * rho + t * t) - np.float32(0.5 * width[i]), c)
                    box = tuple(slice(int(first[d] - origin[d]), int(last[d] - origin[d])) for d in range(3))
                    inner = tuple(slice(int(first[d] - start[d]), int(last[d] - start[d])) for d in range(3))
                    block = own[inner]
                    outline, slab = xp.maximum(c, q - np.float32(TALL) * tall), xp.maximum(q, c - wider)
                    if cut is None:
                        xp.minimum(block, outline, out=block)
                        block = faces[box]
                        xp.minimum(block, slab, out=block)
                    else:
                        # (a mitred piece gives its values between its two planes only: cut to them with max,
                        # the values by a plane were the distance to the plane, a seam in every blend and rim)
                        mine = cut <= 0
                        block[...] = xp.where(mine, xp.minimum(block, outline), block)
                        block = faces[box]
                        block[...] = xp.where(mine, xp.minimum(block, slab), block)
                block = sides[tuple(slice(int(start[d] - origin[d]), int(end[d] - origin[d])) for d in range(3))]
                if soft > 0:
                    h = xp.maximum(1.0 - xp.abs(block - own) / np.float32(soft), 0.0)
                    block[...] = xp.minimum(block, own) - bulge(h, np.float32(soft), 2)
                else:
                    xp.minimum(block, own, out=block)
                del own
            out = xp.maximum(sides, faces)
            if edge > 0 and edge_shape == "chamfer":
                out = xp.maximum(out, (sides + faces + np.float32(edge)) * half)
            elif edge > 0:    # (Mercury's rounded intersection: exact where the two meet at 90 degrees, as here)
                r = np.float32(edge)
                ua, ub = xp.maximum(r + sides, 0.0), xp.maximum(r + faces, 0.0)
                out = xp.minimum(-r, out) + xp.sqrt(ua * ua + ub * ub)
            del sides, faces
            place = tuple(slice(int(origin[d]), int(top[d])) for d in range(3))
            region = whole[place]
            behind(place, out)
            region[...] = joined(region, out, None if soften is None else soften[place])
        def loft_values(ctrl, tau, frames, wide_, tall_, spans, low, high, shift, origin, top):
            """A loft's outline (on the prisms taller than it), its slab, and its outline alone (how far a sample
            lies beside it, whatever its height), on the samples origin..top."""
            shape = tuple(int(n) for n in top - origin)
            sides = xp.full(shape, far + k, dtype=xp.float32)
            faces = xp.full(shape, far, dtype=xp.float32)
            beside = xp.full(shape, far + k, dtype=xp.float32)
            wider = np.float32(edge + 1.0)
            for s, (span_lo, span_hi) in enumerate(spans):
                first, last = _window(grid, span_lo, span_hi, float(far))
                if np.any(last <= first):
                    continue
                # every sample's nearest point of the span, from the span's middle by Gauss-Newton steps (a span
                # turns LOFT_TURN degrees at most); a sample whose nearest point lies past an end of the span is
                # the next span's (which reaches a tenth of a span over this one: none is left out between
                # them), or, past the loft's own end, in that end's round end, which lies in the bar there
                ta, tb = float(tau[s]), float(tau[s + 1])
                t_lo = ta - 0.1 * (tb - ta) if s > 0 else 0.0
                t_hi = tb + 0.1 * (tb - ta) if s < len(spans) - 1 else 1.0
                r_ = 1.0 - ta
                base = r_ ** 3 * ctrl[0] + 3 * r_ * r_ * ta * ctrl[1] + 3 * r_ * ta * ta * ctrl[2] + ta ** 3 * ctrl[3]
                x, y, z = (xp.asarray((axes[d][first[d]:last[d]] - base[d]).astype(np.float32)).reshape(
                    [-1 if e == d else 1 for e in range(3)]) for d in range(3))
                p = [tuple(np.float32(c) for c in point - base) for point in ctrl]
                v = [tuple(np.float32(3.0 * c) for c in ctrl[j + 1] - ctrl[j]) for j in range(3)]
                w = [tuple(np.float32(6.0 * c) for c in ctrl[j + 2] - 2.0 * ctrl[j + 1] + ctrl[j]) for j in range(2)]
                t = xp.full(tuple(int(n) for n in last - first), np.float32(0.5 * (ta + tb)), dtype=xp.float32)
                one, two, three = np.float32(1.0), np.float32(2.0), np.float32(3.0)

                def curve_at(t):
                    r = one - t
                    b0, b1, b2, b3 = r * r * r, three * r * r * t, three * r * t * t, t * t * t
                    d0, d1, d2 = r * r, two * r * t, t * t
                    return ([b0 * p[0][j] + b1 * p[1][j] + b2 * p[2][j] + b3 * p[3][j] for j in range(3)],
                            [d0 * v[0][j] + d1 * v[1][j] + d2 * v[2][j] for j in range(3)])
                for _ in range(5):
                    # (Newton's steps, with the curve's bend in them; where that would turn a step round,
                    # beyond the middle of the bend, the step without it)
                    (cx, cy, cz), (dx, dy, dz) = curve_at(t)
                    px, py, pz = x - cx, y - cy, z - cz
                    r = one - t
                    fast = dx * dx + dy * dy + dz * dz
                    bent = sum((px, py, pz)[j] * (r * w[0][j] + t * w[1][j]) for j in range(3))
                    t = xp.clip(t + (px * dx + py * dy + pz * dz) / xp.maximum(fast - bent, np.float32(0.25) * fast),
                                np.float32(t_lo), np.float32(t_hi))
                del fast, bent, r
                (cx, cy, cz), (dx, dy, dz) = curve_at(t)
                speed = xp.sqrt(dx * dx + dy * dy + dz * dz)
                dx, dy, dz = dx / speed, dy / speed, dz / speed
                px, py, pz = x - cx, y - cy, z - cz
                del cx, cy, cz, speed
                along = px * dx + py * dy + pz * dz
                # the faces' normal there: between the span's two ends', square to the curve
                share = xp.clip((t - np.float32(ta)) / np.float32(tb - ta), 0.0, 1.0)
                nx, ny, nz = (np.float32(frames[s][j]) + np.float32(frames[s + 1][j] - frames[s][j]) * share for j in range(3))
                lean = nx * dx + ny * dy + nz * dz
                nx, ny, nz = nx - lean * dx, ny - lean * dy, nz - lean * dz
                size_ = xp.sqrt(nx * nx + ny * ny + nz * nz)
                nx, ny, nz = nx / size_, ny / size_, nz / size_
                del share, lean, size_
                up = px * nx + py * ny + pz * nz
                rho = px * (ny * dz - nz * dy) + py * (nz * dx - nx * dz) + pz * (nx * dy - ny * dx)
                del px, py, pz, nx, ny, nz, dx, dy, dz
                ease = xp.clip(t, 0.0, 1.0)
                # (a loft through its joint's plane: its section there, the middle values, `bump` 1 in the
                # middle and 0 with no slope at the ends, so it leaves its bars as before)
                bump = None
                if len(wide_) > 2 or np.any(shift):
                    bump = ease * (one - ease) * np.float32(4.0)
                    bump = bump * bump
                if shift[0]:
                    up = up - np.float32(shift[0]) * bump
                rho = xp.abs(rho - np.float32(shift[1]) * bump) if shift[1] else xp.abs(rho)
                ease = ease * ease * (three - two * ease)
                tall = np.float32(tall_[0]) + np.float32(tall_[-1] - tall_[0]) * ease
                half_ = np.float32(0.5) * (np.float32(wide_[0]) + np.float32(wide_[-1] - wide_[0]) * ease)
                if len(wide_) > 2:
                    tall = tall + np.float32(tall_[1] - 0.5 * (tall_[0] + tall_[2])) * bump
                    half_ = half_ + np.float32(0.5 * (wide_[1] - 0.5 * (wide_[0] + wide_[2]))) * bump
                del bump
                q = xp.abs(up) - np.float32(0.5) * tall
                c = rho - half_
                # (past the loft's own end it runs straight on a little way into its bar and ends square: a
                # round end, half the width long, stood out of a bar that tapers or bends there, an arc on its face)
                # A sample is this span's only where the steps came to rest on the curve, square to it: beyond
                # the middle of a tight bend they do not, and what they stopped at made stripes of material
                # across the inside of the bend
                over = np.float32(LOFT_OVER * min(float(wide_.min()), float(tall_.min())))
                mine = xp.abs(along) <= over
                for end, own_end in (((t <= np.float32(t_lo)) & (along < 0), s == 0),
                                     ((t >= np.float32(t_hi)) & (along > 0), s == len(spans) - 1)):
                    if not own_end:
                        mine &= ~end
                del up, rho, ease, half_, along, t, end
                outline, slab = xp.maximum(c, q - np.float32(TALL) * tall), xp.maximum(q, c - wider)
                box = tuple(slice(int(first[d] - origin[d]), int(last[d] - origin[d])) for d in range(3))
                for target, values in ((sides, outline), (faces, slab)):
                    block = target[box]
                    block[...] = xp.where(mine, xp.minimum(block, values), block)
                block = beside[box]
                block[...] = xp.where(mine, xp.minimum(block, c), block)
            return sides, faces, beside

        one_ = np.float32(1.0)

        def rims(sides, faces):
            """An outline and faces as one body, the rims between them chamfered or rounded."""
            out = xp.maximum(sides, faces)
            if edge > 0 and edge_shape == "chamfer":
                out = xp.maximum(out, (sides + faces + np.float32(edge)) * half)
            elif edge > 0:
                r = np.float32(edge)
                ua, ub = xp.maximum(r + sides, 0.0), xp.maximum(r + faces, 0.0)
                out = xp.minimum(-r, out) + xp.sqrt(ua * ua + ub * ub)
            return out

        def joint_faces(members_, faced, origin, top):
            """A joint from its faces (LOFT_FACES), each patched on its own as Rhino's Patch would: its upper face
            one smooth surface through its bars' upper faces, its lower one through their lower faces (Shepard's
            mix of the faces' planes, each weighted by 1 / the distance to its bar past the cut squared, the bar's
            whole section: exactly the bar's faces in and past the cut, smooth between; weighted by the distance to
            its middle line a tilted bar's faces did not meet the joint's at the cut, and two joints came apart),
            its sides the lofts' outlines (the
            bends between bars), the rims between them as a bar's. A loft to a bar standing across the joint's
            plane (`free`) is drawn as it is."""
            shape = tuple(int(n) for n in top - origin)
            outline = xp.full(shape, far + k, dtype=xp.float32)
            alone = xp.full(shape, far + k + np.float32(1.0), dtype=xp.float32)
            for loft, (first, last) in members_:
                sides, faces, beside = loft_values(*loft[:6], *loft[6:9], first, last)
                box = tuple(slice(int(first[d] - origin[d]), int(last[d] - origin[d])) for d in range(3))
                if loft[12]:
                    block = alone[box]
                    xp.minimum(block, rims(sides, faces), out=block)
                else:    # (its outline within LOFT_FACE_NEAR x its side of its own slab: reaching to any height, or
                    # TALL past its faces, it crossed the faces' patch away from the bars, specks that made the part two pieces)
                    block = outline[box]
                    near_ = np.float32(LOFT_FACE_NEAR * float(min(loft[3].min(), loft[4].min())))
                    xp.minimum(block, xp.maximum(sides, faces - near_), out=block)
                del sides, faces, beside
            x, y, z = (xp.asarray(axes[d][int(origin[d]):int(top[d])].astype(np.float32)).reshape(
                [-1 if e == d else 1 for e in range(3)]) for d in range(3))
            total = xp.zeros(shape, dtype=xp.float32)
            upper = xp.zeros(shape, dtype=xp.float32)
            lower = xp.zeros(shape, dtype=xp.float32)
            for row in faced:
                (cx, cy, cz), (tx, ty, tz), (ux, uy, uz) = (tuple(np.float32(v) for v in row[i:i + 3]) for i in (0, 3, 6))
                vx, vy, vz = uy * tz - uz * ty, uz * tx - ux * tz, ux * ty - uy * tx     # (across the bar)
                px, py, pz = x - cx, y - cy, z - cz
                s = xp.maximum(px * tx + py * ty + pz * tz, 0.0)
                h = px * ux + py * uy + pz * uz
                a_ = xp.maximum(xp.abs(h) - np.float32(row[9]), 0.0)
                b_ = xp.maximum(xp.abs(px * vx + py * vy + pz * vz) - np.float32(row[10]), 0.0)
                back = xp.minimum(px * tx + py * ty + pz * tz, 0.0)      # (before the cut: towards the joint)
                w = a_ * a_ + b_ * b_ + back * back + np.float32(1e-6)
                w = one_ / (w * w) if LOFT_FACE_POWER == 4 else one_ / w
                total += w
                upper += w * (h - np.float32(row[9]))
                lower += w * (-h - np.float32(row[9]))
                del px, py, pz, s, h, a_, b_, back, w
            faces = xp.maximum(upper, lower) / total
            del upper, lower, total, x, y, z
            out = xp.minimum(rims(outline, faces), alone)
            del outline, faces, alone
            return out

        # A joint whose lofts have a node (LOFT_NODE) is one body: the lofts' outlines, whatever their height (so the
        # bends between bars stay), cut by the convex hull of the bars' ends, whose flat faces are the joint's (the
        # lofts' own faces, at different depths where they cross, left steps). Other lofts are joined plainly, also
        # near a knot: a loft lies on its bars where it leaves them, and its ends and theirs are cuts, steps in the
        # values, which a blend in space drew out into stripes
        sets = {}
        for i, loft in enumerate(lofts_):
            sets.setdefault(loft[9] if loft[10] is not None or loft[11] is not None else ("alone", i), []).append(loft)
        for members_ in sets.values():
            windows = [_window(grid, loft[6], loft[7], float(far)) for loft in members_]
            members_ = [(loft, w_) for loft, w_ in zip(members_, windows) if not np.any(w_[1] <= w_[0])]
            if not members_:
                continue
            origin = np.min([w_[0] for _, w_ in members_], axis=0)
            top = np.max([w_[1] for _, w_ in members_], axis=0)
            place = tuple(slice(int(origin[d]), int(top[d])) for d in range(3))
            hull, faced = members_[0][0][10], members_[0][0][11]
            if faced is not None:
                out = joint_faces(members_, faced, origin, top)
            elif hull is None:
                (loft, (first, last)), = members_
                sides, faces, _ = loft_values(*loft[:6], *loft[6:9], first, last)
                out = rims(sides, faces)
                del sides, faces
            else:
                shape = tuple(int(n) for n in top - origin)
                outline = xp.full(shape, far + k, dtype=xp.float32)
                for loft, (first, last) in members_:
                    _, _, beside = loft_values(*loft[:6], *loft[6:9], first, last)
                    box = tuple(slice(int(first[d] - origin[d]), int(last[d] - origin[d])) for d in range(3))
                    block = outline[box]
                    xp.minimum(block, beside, out=block)
                    del beside
                x, y, z = (xp.asarray(axes[d][int(origin[d]):int(top[d])].astype(np.float32)).reshape(
                    [-1 if e == d else 1 for e in range(3)]) for d in range(3))
                planes = xp.full(shape, -far, dtype=xp.float32)
                for a_, b_, c_, d_ in hull:
                    xp.maximum(planes, x * np.float32(a_) + y * np.float32(b_) + z * np.float32(c_) + np.float32(d_), out=planes)
                out = rims(outline, planes)
                del outline, planes, x, y, z
            region = whole[place]
            behind(place, out)
            xp.minimum(region, out, out=region)
            del out
        for samples, start_, step_, low, high, cut, weight in patches:
            first, last = _window(grid, low, high, 0.0)
            if np.any(last <= first):
                continue
            region = whole[tuple(slice(int(first[d]), int(last[d])) for d in range(3))]
            where_ = [axes[d][first[d]:last[d]] for d in range(3)]
            values = spline_values(samples.get(), start_, step_, where_, xp)
            if weight is not None:     # (a patch: its values replace the bars' by its weight)
                share = xp.clip(spline_values(weight.get(), start_, step_, where_, xp), 0.0, 1.0)
                region += share * (values - region)
                del share
            else:
                (xp.maximum if cut else xp.minimum)(region, values, out=region)
            del values
        if balls is not None:
            whole = joined(whole, balls, soften)
        if second is not None:
            return xp.minimum(second, far).astype(xp.float32)
        return xp.minimum(whole, far).astype(xp.float32)

    bounds = np.array([np.min([lo.min(axis=0)] + [loft[6] for loft in lofts_], axis=0),
                       np.max([hi.max(axis=0)] + [loft[7] for loft in lofts_], axis=0)])
    numbers = hashlib.sha1(b"".join(np.ascontiguousarray(v).tobytes()
                                    for v in [a, b, normal, width, width_b, thick, thick_b, twist, mitre_a, mitre_b, member, blends, layer, knots, knot_size,
                                              knot_blend, knot_reach, knot_axis, knot_span]
                                    + [part for loft in lofts_ for part in loft[:5] + (loft[8],) + (() if loft[10] is None else (loft[10],))
                                       + (() if loft[11] is None else (loft[11], np.array([loft[12]])))]
                                    + [np.array(patch[1] + (patch[2], float(patch[5]))) for patch in patches])).hexdigest()
    patch_keys = [patch[0].key if patch[6] is None else (patch[0].key, patch[6].key) for patch in patches]
    key = hashlib.sha1(repr(("bars", VERSION, numbers, blend, edge, edge_shape, reach, bool(seams), (bool(LOFT_NODE), bool(LOFT_FACES), LOFT_FACE_POWER) if lofts_ else (),
                             patch_keys)).encode()).hexdigest()[:20]
    field = Field(fn, key, f"bars({len(a)})", bounds, cache=True)
    field.extent = bounds
    return field
