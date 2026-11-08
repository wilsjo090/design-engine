"""A frame's bars as an implicit body.

One NVIDIA Warp kernel for every profile and joint (GPU, or Warp's CPU threads), like
`engine/segments.py`: a BVH over the bar pieces' boxes (grown by a margin); for each grid point the
pieces whose box holds it are measured exactly:
- round bars: a tube round each bar's curve (`_curves`: one quintic spline per continuous bar, straight
  on past its ends), its ends rounded as smoothly as the fillets (`kernel._ramp`);
- round tubes: the annulus round that curve, other profiles: their rectangles (holes as negative ones, as in
  `Profile._rectangles`), in the piece's local frame (z carried along the bar,
  `curves.orientation`), between its end planes. Pieces of one bar are mitred (cut at the plane
  halfway between them), so a curved square tube is one constant section with no bumps; a free
  end reaches half the section's diagonal past its node for plain joints.
- balls (joints = ball): a sphere at each free joint.
The kernel keeps the nearest *different* bars (`kernel.SLOTS`, each at its nearest piece) and blends
them thickest first, each into those before it with a polynomial smooth minimum of size k (joints =
fillet: k = FILLET x that bar, the thinner of each pair, or the node's Fillet in mm; plain: a hard
minimum) of order G1 / G2 / G3 (`fields.bulge`): one blend as smooth as its order however many bars
meet; pieces of one bar never blend (a curve would get rings). (Blending only the two nearest, the
fillet's size jumped where the second and third nearest swapped: a torn seam; the union of every pair's
fillet then creased where two fillets crossed.)
Outside every box the value is the margin (positive): only the surface matters there.

All of it is a pure function of position, so block meshing, slices and surface colours work. The
skin at supports and loads (bolt bosses, a pin's eye) is the region within `skin` mm of their
faces, joined to the bars with the same fillet; the design space crops everything, so holes stay
open. (First versions: capsules per size with Warp and prisms with numpy, plain or ball joints;
balls looked stuck on, and the bars met bosses and each other in sharp corners: 81-92 MPa there.)
"""
import numpy as np
import trimesh

from .. import fields as F
from ..fields import Field, _digest
from .curves import orientation
from .profiles import Profile

JOINTS = ("fillet", "plain", "ball")
FILLET = 0.6          # automatic fillet: this x the thinner bar's size
BALL = 0.6            # ball joint radius: this x the largest bar size (width or height) at the node
# The frame body's and the bosses' geometry version, in their keys: raise it with every change to the kernel,
# `_curves`, `skin_body` or `_cylinder_distance`. Their values and Static Analysis's displacements are kept on the
# SSD by key, and a change that kept the key loaded another shape's displacements (11000 MPa in a boss's cap)
GEOMETRY = 7
BLEND = f"ordered, curves {GEOMETRY}"   # (in the bars' keys)
CONTINUITY = ("G1", "G2", "G3")   # fillets: tangent, curvature continuous, and one derivative more
# G2 and G3 fillets are this much larger than G1's for the same size: the same depth in their middle (k/4,
# k/6, 3k/20 of their k), so about as much material at a joint, and a lower peak curvature (the bolted
# bracket's 5-bar joint: +27.6 and 27.3 mm3 against 25.4, peak mean curvature 5.0 against 5.6 /mm; at the
# same k a G2 fillet had half the material and a tighter middle, 6.1 /mm)
CONTINUITY_SIZE = {1: 1.0, 2: 1.5, 3: 5.0 / 3.0}

def frame_pieces(nodes, bars, chains, sizes, profile, joints, no_ball=None):
    """The kernel's arrays for a frame: per piece (and per ball) its kind, bar, size, geometry and
    box. nodes (n, 3), bars (k, 2), chains (k,), sizes (chains,) mm."""
    nodes = np.asarray(nodes, dtype=np.float64)
    bars, chains = np.asarray(bars), np.asarray(chains)
    profile = profile or Profile()
    width = np.asarray(sizes, dtype=np.float64)[chains]
    k = len(bars)
    sec = {float(s): profile.section(float(s)) for s in np.unique(width)}
    outer = np.array([float(np.hypot(sec[float(w)].width, sec[float(w)].height)) / 2 for w in width])
    zaxes, forward = orientation(nodes, bars, chains)
    a = np.where(forward[:, None], nodes[bars[:, 0]], nodes[bars[:, 1]])
    b = np.where(forward[:, None], nodes[bars[:, 1]], nodes[bars[:, 0]])
    ex = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
    ey = np.cross(zaxes, ex)
    n0, n1 = -ex.copy(), ex.copy()
    free = np.ones((k, 2), dtype=bool)
    start_at = {}
    for p in range(k):
        start_at.setdefault((int(chains[p]), tuple(np.round(a[p], 9))), []).append(p)
    following = {}
    for p in range(k):
        nxt = [q for q in start_at.get((int(chains[p]), tuple(np.round(b[p], 9))), []) if q != p]
        if nxt:   # mitre onto the next piece of the same bar
            q = nxt[0]
            following[p] = q
            m = ex[p] + ex[q]
            m /= np.linalg.norm(m)
            n1[p], n0[q] = m, -m
            free[p, 1] = free[q, 0] = False
    reach = outer if joints == "plain" else 0.25 * outer   # free ends: plain overlaps, a fillet or ball closes
    p0 = a - np.where(free[:, 0], reach, 0.0)[:, None] * ex
    p1 = b + np.where(free[:, 1], reach, 0.0)[:, None] * ex
    kind = np.full(k, 2 if not profile.round else (0 if profile.kind == "round" else 1))
    rad = width / 2
    curve, ends, tips, near = _curves(a, b, following, rad)
    wall = np.array([profile.thickness(w) if profile.kind == "round tube" else 0.0 for w in width])
    rects = np.zeros((k, 4, 4))
    rsign = np.zeros((k, 4), dtype=np.int64)
    cen = np.zeros((k, 2))
    rot = np.tile([1.0, 0.0], (k, 1))
    if not profile.round:
        for w in np.unique(width):
            r, _, _ = profile._rectangles(float(w))
            s = sec[float(w)]
            mine = width == w
            for x, (y0, y1, z0, z1, sg) in enumerate(r):
                rects[mine, x] = (y0, y1, z0, z1)
                rsign[mine, x] = 1 if sg > 0 else -1
            cen[mine] = s.centroid
            rot[mine] = (np.cos(s.turn), np.sin(s.turn))
    lo = np.minimum(p0, p1) - outer[:, None] * 1.5
    hi = np.maximum(p0, p1) + outer[:, None] * 1.5
    pieces = dict(kind=kind, chain=np.asarray(chains), width=width, a=a, b=b, ey=ey, ez=zaxes, p0=p0, n0=n0, p1=p1,
                  n1=n1, rad=rad, wall=wall, rects=rects, rsign=rsign, cen=cen, rot=rot, curve=curve, ends=ends,
                  tips=tips, near=near, lo=lo, hi=hi)
    if joints == "ball":
        extra = []
        n_chains = int(chains.max()) + 1
        for v in range(len(nodes)):
            if no_ball is not None and no_ball[v]:
                continue
            at = np.flatnonzero((bars == v).any(axis=1))
            if len(at) and (len(np.unique(chains[at])) > 1 or len(at) == 1):
                big = max(max(sec[float(w)].width, sec[float(w)].height) for w in width[at])
                extra.append((v, BALL * big))
        if extra:
            m = len(extra)
            c = np.array([nodes[v] for v, _ in extra])
            r = np.array([r for _, r in extra])
            add = dict(kind=np.full(m, 3), chain=n_chains + np.arange(m), width=2 * r, a=c, b=c, ey=np.zeros((m, 3)),
                       ez=np.zeros((m, 3)), p0=c, n0=np.zeros((m, 3)), p1=c, n1=np.zeros((m, 3)), rad=r,
                       wall=np.zeros(m), rects=np.zeros((m, 4, 4)), rsign=np.zeros((m, 4), dtype=np.int64),
                       cen=np.zeros((m, 2)), rot=np.tile([1.0, 0.0], (m, 1)), curve=np.zeros((m, 6, 3)),
                       ends=np.zeros((m, 2), dtype=np.int64), tips=np.zeros((m, 4, 3)),
                       near=np.zeros((m, 2), dtype=np.int64), lo=c - r[:, None], hi=c + r[:, None])
            pieces = {key: np.concatenate([pieces[key], add[key]]) for key in pieces}
    margin = 0.75 * float(pieces["width"].max()) + 1.0   # the fillet's reach, and the value outside every box
    pieces["lo"] = pieces["lo"] - margin
    pieces["hi"] = pieces["hi"] + margin
    pieces["cap"] = margin
    return pieces


def ramp(s, size, order):
    """A bar's end rounding (numpy; the kernel's `_ramp`): max(s, 0) made as smooth as the fillets over
    -size..size, s the distance past the end along the bar; the end is sqrt(rho^2 + ramp^2) = radius. Order
    1 a half ball (C1 where it meets the bar), 2 its slope a cubic smoothstep (C2), 3 a quintic one (C3)."""
    s = np.asarray(s, dtype=np.float64)
    if order <= 1:
        return np.maximum(s, 0.0)
    x = np.clip((s + size) / (2.0 * size), 0.0, 1.0)
    inside = 2.0 * size * (x ** 3 * (1.0 - 0.5 * x) if order == 2 else x ** 4 * (2.5 + x * (x - 3.0)))
    return np.where(s >= size, s, inside)


def fade(x):
    """How much of its fillet a bar x fillet sizes away blends with (numpy; the kernel's `_fade`): 1 up
    to 1, 0 from 2, a septic smoothstep between (C3)."""
    y = np.clip(2.0 - np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return y ** 4 * (35.0 + y * (-84.0 + y * (70.0 - 20.0 * y)))


def _curves(a, b, following, rad):
    """Round bars and tubes go round one curve per continuous bar, smooth to its fourth derivative: a
    quintic spline through its nodes (chord length knots; a closed loop periodic) whose second and third
    derivatives are 0 at its ends, where it goes on straight (so that join is smooth to the third
    derivative too, as G3 needs). A curved bar was a polyline of capsules, creased at every node (4
    degrees); a cubic spline would be G2 only. Per piece (pieces in `following` order): its quintic
    c0 + c1 t + ... + c5 t^5 (t 0..1, a to b), whether the bar ends at its start / end, the bar's end
    points with outward tangents, and whether it lies within two radii of either end (where the end's
    rounding acts: a bar curving back past its end must not be cut there)."""
    from scipy.interpolate import make_interp_spline
    k = len(a)
    curve = np.zeros((k, 6, 3))
    curve[:, 0], curve[:, 1] = a, b - a
    ends = np.zeros((k, 2), dtype=np.int64)
    tips = np.zeros((k, 4, 3))
    near = np.zeros((k, 2), dtype=np.int64)
    before = {q: p for p, q in following.items()}
    seen = np.zeros(k, dtype=bool)
    u = np.linspace(0.0, 1.0, 6)
    power = np.linalg.inv(np.vander(u, 6, increasing=True))   # samples at u -> c0..c5
    flat = [(2, np.zeros(3)), (3, np.zeros(3))]
    for first in [p for p in range(k) if p not in before] + list(range(k)):   # open bars, then loops
        if seen[first]:
            continue
        seq = [first]
        seen[first] = True
        while following.get(seq[-1]) is not None and not seen[following[seq[-1]]]:
            seq.append(following[seq[-1]])
            seen[seq[-1]] = True
        closed = len(seq) > 2 and following.get(seq[-1]) == seq[0]
        P = np.vstack([a[seq[0]]] + [b[p] for p in seq])
        h = np.linalg.norm(np.diff(P, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(h)])
        if len(seq) > 1 and h.min() > 1e-9:
            if closed:
                P[-1] = P[0]
            spline = make_interp_spline(t, P, k=5, bc_type="periodic" if closed else (flat, flat))
            for i, p in enumerate(seq):
                curve[p] = power @ spline(t[i] + u * h[i])
        if closed:
            continue
        start, end = curve[seq[0]], curve[seq[-1]]
        out0 = -start[1] / max(np.linalg.norm(start[1]), 1e-12)
        out1 = (np.arange(6)[:, None] * end).sum(axis=0)          # the curve's derivative at t = 1
        out1 = out1 / max(np.linalg.norm(out1), 1e-12)
        ends[seq[0], 0] = ends[seq[-1], 1] = 1
        reach = 2.0 * float(rad[seq[0]])
        for i, p in enumerate(seq):
            tips[p] = (P[0], out0, P[-1], out1)
            near[p] = (t[i] < reach, t[-1] - t[i + 1] < reach)
    return curve, ends, tips, near


def frame_solid(nodes, bars, chains, sizes, profile, joints, key, no_ball=None, fillet=0.0, continuity="G1"):
    """The bars of a frame as an implicit body: nodes (n, 3), bars (k, 2) pieces, chains (k,) the
    continuous bar of each piece, sizes (chains,) mm, a Profile, joints 'fillet', 'plain' or 'ball';
    no_ball: nodes that get no ball (at bolts and pins the boss covers the bar ends; a ball there
    bulged out); fillet: mm, 0 = automatic (FILLET x the thinner bar); continuity: CONTINUITY."""
    if joints not in JOINTS:
        raise ValueError(f"joints must be one of: {', '.join(JOINTS)}")
    data = frame_pieces(nodes, bars, chains, sizes, profile, joints, no_ball)
    return _solid(data, _digest("frame solid", key, profile, joints, float(fillet)), joints, fillet, continuity)


def order_of(continuity):
    """'G1' / 'G2' / 'G3' as the smooth minimum's order (`fields.bulge`)."""
    if continuity not in CONTINUITY:
        raise ValueError(f"fillet continuity must be one of: {', '.join(CONTINUITY)}")
    return CONTINUITY.index(continuity) + 1


def _solid(data, k, joints, fillet, continuity="G1"):
    order = order_of(continuity)
    k = _digest(k, BLEND, order)
    size = CONTINUITY_SIZE[order]
    blend = (0.0, 0.0) if joints == "plain" else ((size * float(fillet), 0.0) if fillet > 0 else (0.0, size * FILLET))
    # every bar within two of its fillet sizes is measured (the kernel fades its blend out there)
    data = dict(data)
    reach = data["rad"] + 2.0 * (blend[0] if blend[0] > 0 else blend[1] * data["width"]) + 1.0
    data["lo"] = np.minimum(data["lo"], np.minimum(data["a"], data["b"]) - reach[:, None])
    data["hi"] = np.maximum(data["hi"], np.maximum(data["a"], data["b"]) + reach[:, None])
    # the value outside every box, where the blends start from: more than any fillet reaches (else a bar
    # would blend with it)
    data["cap"] = max(float(data["cap"]), float((reach - data["rad"]).max()))

    def fn(ctx):
        from .kernel import evaluate
        return evaluate(data, ctx.grid, k, ctx.xp is not np, *blend, order)

    bounds = np.array([data["lo"].min(axis=0), data["hi"].max(axis=0)])
    return Field(fn, k, "frame", bounds, cache=True)


SEAM = 0.02   # mm a union of design spaces is grown by before it crops a part


def seamless(space):
    """A union of design spaces to crop with: where two only touch, the union's field is 0 along the
    shared face and cropping cut the part there (a joined frame came out in two pieces); grown by
    SEAM, the face is inside."""
    return F.offset(space, SEAM)


def join_body(frames, bodies=(), joints="fillet", fillet=0.0, continuity="G1"):
    """Frames (FrameRef) and CAD bodies as one part: the bars of all frames in one kernel, so where
    a bar of one meets a bar of another the joint is filleted exactly as inside one frame; each
    frame's skin; cropped by the frames' design spaces; bodies joined with the same fillet."""
    datas, offset = [], 0
    for f in frames:
        d = frame_pieces(f.nodes, f.bars, f.chains, f.sizes, f.profile, joints, _at_faces(f.conditions, f.nodes))
        d = dict(d)
        d["chain"] = np.asarray(d["chain"]) + offset
        offset = int(d["chain"].max()) + 1
        datas.append(d)
    if not datas:
        raise ValueError("connect at least one frame")
    data = {k: np.concatenate([d[k] for d in datas]) for k in datas[0] if k != "cap"}
    data["cap"] = max(d["cap"] for d in datas)
    key = _digest("joined frames", tuple(f.key for f in frames), tuple(f.profile for f in frames), joints, float(fillet))
    body = _solid(data, key, joints, fillet, continuity)
    sizes = np.concatenate([np.asarray(f.sizes, dtype=np.float64)[np.asarray(f.chains)] for f in frames])
    order = order_of(continuity)
    radius = CONTINUITY_SIZE[order] * (float(fillet) if fillet > 0 else FILLET * float(np.median(sizes)))
    join = (lambda a, b: F.union(a, b)) if joints == "plain" else (lambda a, b: F.smooth_union(a, b, radius, order))
    for f in frames:
        around = skin_body(f.conditions, f.skin, f.nodes, reach=radius + 1.0, order=order)
        if around is not None:
            body = join(body, around)
    space = frames[0].space
    for f in frames[1:]:
        space = F.union(space, f.space)
    if len(frames) > 1:
        space = seamless(space)
    body = F.intersect(space, body)
    for m in bodies:
        body = join(body, F.body_from_mesh(m, 2.0))
    return body


_TREES = {}   # piece key -> (V, F, igl AABB): built once per process


def _tree(key, tri):
    entry = _TREES.get(key)
    if entry is None:
        import igl
        mesh = trimesh.Trimesh(tri.reshape(-1, 3), np.arange(3 * len(tri)).reshape(-1, 3), process=True)
        V, Fc = np.ascontiguousarray(mesh.vertices, dtype=np.float64), np.ascontiguousarray(mesh.faces, dtype=np.int64)
        tree = igl.AABB()
        tree.init(V, Fc)
        entry = _TREES[key] = (V, Fc, tree)
        if len(_TREES) > 64:
            _TREES.pop(next(iter(_TREES)))
    return entry


def _cylinder(tri):
    """A piece of faces that is a cylinder, a bolt hole or a pin (its triangles' normals square to one
    axis, their corners on a circle round it: OpenCascade puts mesh nodes on the surface): dict(c, axis,
    e1, e2, r, z0, z1, ring, th0, span), else None. `ring`: all round (a hole, or a piece covering it),
    else the angles th0 .. th0 + span."""
    t = np.asarray(tri, dtype=np.float64).reshape(-1, 3, 3)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    ok = area > 1e-12
    if ok.sum() < 6:
        return None
    t, n, area = t[ok], n[ok] / (2 * area[ok, None]), area[ok]
    spread, frame = np.linalg.eigh((n * area[:, None]).T @ n)
    axis = frame[:, 0]
    # (normals square to the axis, and turning round it by ~30 degrees or more: a flat face's are square to
    # any axis in it)
    if np.abs(n @ axis).max() > 0.02 or spread[1] < 0.02 * area.sum():
        return None
    e1 = np.cross(axis, (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    pts = np.unique(t.reshape(-1, 3), axis=0)
    x, y = pts @ e1, pts @ e2
    a, b, c = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), x * x + y * y, rcond=None)[0]
    cx, cy = a / 2, b / 2
    r = float(np.sqrt(c + cx * cx + cy * cy))
    if not np.isfinite(r) or np.abs(np.hypot(x - cx, y - cy) - r).max() > 0.01 + 0.002 * r:
        return None
    th = np.sort(np.arctan2(y - cy, x - cx))
    gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
    widest = int(np.argmax(gaps))
    mid = t.mean(axis=1)
    radial = np.column_stack([mid @ e1 - cx, mid @ e2 - cy])
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    hole = float(((n @ e1) * radial[:, 0] + (n @ e2) * radial[:, 1]) @ area / area.sum()) < -0.9
    z = pts @ axis
    return dict(c=cx * e1 + cy * e2, axis=axis, e1=e1, e2=e2, r=r, z0=float(z.min()), z1=float(z.max()),
                ring=hole or gaps[widest] < np.radians(10.0), th0=float(th[(widest + 1) % len(th)]),
                span=float(2 * np.pi - gaps[widest]))


def _cylinder_distance(cyl, pts, order=1, size=1.0):
    """The distance from pts (k, 3) to a fitted cylinder (`_cylinder`): smooth wherever the nearest
    point is inside it, unlike the distance to its triangles (flat over each, round over each edge: the
    boss's curvature jumped every half millimetre). Past its ends rounded by `ramp` of `order` over
    `size` mm (order 1: exact, the boss's end a half torus meeting it with a curvature jump)."""
    rel = pts - cyl["c"]
    z, x, y = rel @ cyl["axis"], rel @ cyl["e1"], rel @ cyl["e2"]
    # the distance from the axis, rounded within half the radius of it (C3, an even polynomial: past the
    # ends every point on the axis is as far from the whole rim, a ridge a fillet over it picked up; only
    # the hole's inside and the axis past its ends change)
    rho = np.hypot(x, y)
    near = 0.5 * cyl["r"]
    u2 = np.minimum(rho / near, 1.0) ** 2
    rho = np.where(rho < near, near * (5.0 + u2 * (15.0 + u2 * (-5.0 + u2))) / 16.0, rho)
    dz2 = ramp(cyl["z0"] - z, size, order) ** 2 + ramp(z - cyl["z1"], size, order) ** 2
    d = np.sqrt((rho - cyl["r"]) ** 2 + dz2)
    if cyl["ring"]:
        return d
    phi = np.mod(np.arctan2(y, x) - cyl["th0"], 2 * np.pi)
    edge = np.where(phi - cyl["span"] < 2 * np.pi - phi, cyl["th0"] + cyl["span"], cyl["th0"])
    on_edge = cyl["c"] + cyl["r"] * (np.cos(edge)[:, None] * cyl["e1"] + np.sin(edge)[:, None] * cyl["e2"])         + np.clip(z, cyl["z0"], cyl["z1"])[:, None] * cyl["axis"]
    return np.where(phi <= cyl["span"], d, np.linalg.norm(pts - on_edge, axis=1))


def skin_body(conditions, skin, nodes=None, reach=0.0, order=1):
    """Material within `skin` mm of the faces of supports and loads (None for no skin): per connected
    piece of faces (a bolt hole, a pin), the exact distance to it, in that piece's box only, capped
    `reach` beyond the skin (at least 1 mm: as far as a fillet into the boss reaches, else the fillet
    creased where the cap began). A piece that is a cylinder (`_cylinder`: a hole gets a ring all round)
    is measured as one, so the boss is smooth (its ends rounded as the bars' of `order`); other pieces by
    their triangles. (The size of an open
    mesh's signed distance field was tried first: away from the faces it is interpolated with arbitrary
    signs and made sheets and specks.) With `nodes`, only pieces with a frame node in or at them (an
    unused bolt hole would be a loose boss)."""
    from .ground import _pieces
    if skin <= 0:
        return None
    tri = np.concatenate([np.asarray(c.triangles, dtype=np.float64).reshape(-1, 3, 3) for c in conditions])
    if not len(tri):
        return None
    skin = float(skin)
    cap = skin + max(float(reach), 1.0)
    pieces = _pieces(tri)
    if nodes is not None:
        nodes = np.asarray(nodes, dtype=np.float64)
        used = lambda p: bool(np.all((nodes >= p.reshape(-1, 3).min(axis=0) - 1.0)
                                     & (nodes <= p.reshape(-1, 3).max(axis=0) + 1.0), axis=1).any())
        pieces = [p for p in pieces if used(p)]
        if not pieces:
            return None
        tri = np.concatenate(pieces)
    shapes = []
    for p in pieces:
        cyl = _cylinder(p)
        lo, hi = p.reshape(-1, 3).min(axis=0), p.reshape(-1, 3).max(axis=0)
        if cyl is not None and cyl["ring"]:   # (a ring's box: the whole circle, whatever was chosen of it)
            ends = [cyl["c"] + z * cyl["axis"] for z in (cyl["z0"], cyl["z1"])]
            spread = cyl["r"] * np.sqrt(np.maximum(1.0 - cyl["axis"] ** 2, 0.0))
            lo = np.minimum(lo, np.minimum(*ends) - spread)
            hi = np.maximum(hi, np.maximum(*ends) + spread)
        shapes.append((p, cyl, lo - cap, hi + cap, _digest("skin piece", np.ascontiguousarray(p).tobytes())))

    def fn(ctx):
        axes = ctx.grid.axes()
        out = np.full(ctx.grid.shape, cap, dtype=np.float32)
        for tri_, cyl, lo, hi, key in shapes:
            sl = tuple(slice(int(np.searchsorted(ax, lo[i])), int(np.searchsorted(ax, hi[i], side="right")))
                       for i, ax in enumerate(axes))
            if any(s.start >= s.stop for s in sl):
                continue
            pts = np.stack(np.meshgrid(*[ax[s] for ax, s in zip(axes, sl)], indexing="ij"), axis=-1).reshape(-1, 3)
            if cyl is not None:
                dist = _cylinder_distance(cyl, pts, order, skin)
            else:
                V, Fc, tree = _tree(key, tri_)
                dist = np.sqrt(tree.squared_distance(V, Fc, np.ascontiguousarray(pts, dtype=np.float64))[0])
            block = np.minimum(dist, cap).astype(np.float32).reshape([s.stop - s.start for s in sl])
            out[sl] = np.minimum(out[sl], block)
        return ctx.xp.asarray(out - np.float32(skin))

    allv = np.concatenate([np.array([lo, hi]) for _, _, lo, hi, _ in shapes])
    return Field(fn, _digest("frame skin", np.ascontiguousarray(tri).tobytes(), skin, cap, GEOMETRY, order), "skin",
                 np.array([allv.min(axis=0) + cap - skin, allv.max(axis=0) - cap + skin]), cache=True)


def _at_faces(conditions, nodes):
    """Which nodes sit at a piece of a support's or load's faces (in its box): a bolt's or pin's node."""
    from .ground import _pieces
    nodes = np.asarray(nodes, dtype=np.float64)
    out = np.zeros(len(nodes), dtype=bool)
    for c in conditions:
        for p in _pieces(c.triangles):
            v = p.reshape(-1, 3)
            out |= np.all((nodes >= v.min(axis=0) - 1.0) & (nodes <= v.max(axis=0) + 1.0), axis=1)
    return out


def frame_body(space, nodes, bars, chains, sizes, key, conditions=(), skin=0.0, profile=None, joints="fillet",
               fillet=0.0, continuity="G1"):
    """The frame as a part: its bars and the skin at supports and loads (joined with the same
    fillet as the bars, unless the joints are plain), inside the design space."""
    face = _at_faces(conditions, nodes) if len(conditions) else None
    body = frame_solid(nodes, bars, chains, sizes, profile, joints, key, no_ball=face, fillet=fillet,
                       continuity=continuity)
    # bars into bosses: the fillet of the median bar (they ended in sharp corners: 81-92 MPa)
    order = order_of(continuity)
    radius = 0.0 if joints == "plain" else CONTINUITY_SIZE[order] * (
        float(fillet) if fillet > 0 else FILLET * float(np.median(np.asarray(sizes)[np.asarray(chains)])))
    around = skin_body(conditions, skin, nodes, reach=radius + 1.0, order=order)
    if around is not None:
        body = F.union(body, around) if radius == 0 else F.smooth_union(body, around, radius, order)
    return F.intersect(space, body)
