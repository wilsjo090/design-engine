"""A frame's bars as an implicit body.

One NVIDIA Warp kernel for every profile and joint (GPU, or Warp's CPU threads), like
`engine/segments.py`: a BVH over the bar pieces' boxes (grown by a margin); for each grid point the
pieces whose box holds it are measured exactly:
- round bars: capsules (rounded ends), so the pieces of a curved bar form one smooth tube;
- round tubes: the annulus, other profiles: their rectangles (holes as negative ones, as in
  `Profile._rectangles`), in the piece's local frame (z carried along the bar,
  `curves.orientation`), between its end planes. Pieces of one bar are mitred (cut at the plane
  halfway between them), so a curved square tube is one constant section with no bumps; a free
  end reaches half the section's diagonal past its node for plain joints.
- balls (joints = ball): a sphere at each free joint.
The kernel keeps the two nearest *different* bars and blends them with a polynomial smooth minimum
of size k (joints = fillet: k = FILLET x the thinner bar, or the node's Fillet in mm; plain: a
hard minimum): bars meet with a fillet, pieces of one bar never blend (a curve would get rings).
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
    for p in range(k):
        nxt = [q for q in start_at.get((int(chains[p]), tuple(np.round(b[p], 9))), []) if q != p]
        if nxt:   # mitre onto the next piece of the same bar
            q = nxt[0]
            m = ex[p] + ex[q]
            m /= np.linalg.norm(m)
            n1[p], n0[q] = m, -m
            free[p, 1] = free[q, 0] = False
    reach = outer if joints == "plain" else 0.25 * outer   # free ends: plain overlaps, a fillet or ball closes
    p0 = a - np.where(free[:, 0], reach, 0.0)[:, None] * ex
    p1 = b + np.where(free[:, 1], reach, 0.0)[:, None] * ex
    kind = np.full(k, 2 if not profile.round else (0 if profile.kind == "round" else 1))
    rad = width / 2
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
                  n1=n1, rad=rad, wall=wall, rects=rects, rsign=rsign, cen=cen, rot=rot, lo=lo, hi=hi)
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
                       cen=np.zeros((m, 2)), rot=np.tile([1.0, 0.0], (m, 1)), lo=c - r[:, None], hi=c + r[:, None])
            pieces = {key: np.concatenate([pieces[key], add[key]]) for key in pieces}
    margin = 0.75 * float(pieces["width"].max()) + 1.0   # the fillet's reach, and the value outside every box
    pieces["lo"] = pieces["lo"] - margin
    pieces["hi"] = pieces["hi"] + margin
    pieces["cap"] = margin
    return pieces


def frame_solid(nodes, bars, chains, sizes, profile, joints, key, no_ball=None, fillet=0.0):
    """The bars of a frame as an implicit body: nodes (n, 3), bars (k, 2) pieces, chains (k,) the
    continuous bar of each piece, sizes (chains,) mm, a Profile, joints 'fillet', 'plain' or 'ball';
    no_ball: nodes that get no ball (at bolts and pins the boss covers the bar ends; a ball there
    bulged out); fillet: mm, 0 = automatic (FILLET x the thinner bar)."""
    if joints not in JOINTS:
        raise ValueError(f"joints must be one of: {', '.join(JOINTS)}")
    data = frame_pieces(nodes, bars, chains, sizes, profile, joints, no_ball)
    return _solid(data, _digest("frame solid", key, profile, joints, float(fillet)), joints, fillet)


def _solid(data, k, joints, fillet):
    blend = (0.0, 0.0) if joints == "plain" else ((float(fillet), 0.0) if fillet > 0 else (0.0, FILLET))

    def fn(ctx):
        from .kernel import evaluate
        return evaluate(data, ctx.grid, k, ctx.xp is not np, *blend)

    bounds = np.array([data["lo"].min(axis=0), data["hi"].max(axis=0)])
    return Field(fn, k, "frame", bounds, cache=True)


SEAM = 0.02   # mm a union of design spaces is grown by before it crops a part


def seamless(space):
    """A union of design spaces to crop with: where two only touch, the union's field is 0 along the
    shared face and cropping cut the part there (a joined frame came out in two pieces); grown by
    SEAM, the face is inside."""
    return F.offset(space, SEAM)


def join_body(frames, bodies=(), joints="fillet", fillet=0.0):
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
    body = _solid(data, key, joints, fillet)
    sizes = np.concatenate([np.asarray(f.sizes, dtype=np.float64)[np.asarray(f.chains)] for f in frames])
    radius = float(fillet) if fillet > 0 else FILLET * float(np.median(sizes))
    join = (lambda a, b: F.union(a, b)) if joints == "plain" else (lambda a, b: F.smooth_union(a, b, radius))
    for f in frames:
        around = skin_body(f.conditions, f.skin, f.nodes)
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


def skin_body(conditions, skin, nodes=None):
    """Material within `skin` mm of the faces of supports and loads (None for no skin): the exact
    distance to each connected piece of faces (a bolt hole, a pin), in that piece's box only, capped
    a little beyond the skin. (The size of an open mesh's signed distance field was tried first:
    away from the faces it is interpolated with arbitrary signs and made sheets and specks.) With
    `nodes`, only pieces with a frame node in or at them (an unused bolt hole would be a loose boss)."""
    from .ground import _pieces
    if skin <= 0:
        return None
    tri = np.concatenate([np.asarray(c.triangles, dtype=np.float64).reshape(-1, 3, 3) for c in conditions])
    if not len(tri):
        return None
    skin = float(skin)
    cap = skin + 1.0
    pieces = _pieces(tri)
    if nodes is not None:
        nodes = np.asarray(nodes, dtype=np.float64)
        used = lambda p: bool(np.all((nodes >= p.reshape(-1, 3).min(axis=0) - 1.0)
                                     & (nodes <= p.reshape(-1, 3).max(axis=0) + 1.0), axis=1).any())
        pieces = [p for p in pieces if used(p)]
        if not pieces:
            return None
        tri = np.concatenate(pieces)
    pieces = [(p, _digest("skin piece", np.ascontiguousarray(p).tobytes())) for p in pieces]

    def fn(ctx):
        axes = ctx.grid.axes()
        out = np.full(ctx.grid.shape, cap, dtype=np.float32)
        for tri_, key in pieces:
            lo, hi = tri_.reshape(-1, 3).min(axis=0) - cap, tri_.reshape(-1, 3).max(axis=0) + cap
            sl = tuple(slice(int(np.searchsorted(ax, lo[i])), int(np.searchsorted(ax, hi[i], side="right")))
                       for i, ax in enumerate(axes))
            if any(s.start >= s.stop for s in sl):
                continue
            V, Fc, tree = _tree(key, tri_)
            pts = np.stack(np.meshgrid(*[ax[s] for ax, s in zip(axes, sl)], indexing="ij"), axis=-1).reshape(-1, 3)
            sq = tree.squared_distance(V, Fc, np.ascontiguousarray(pts, dtype=np.float64))[0]
            block = np.minimum(np.sqrt(sq), cap).astype(np.float32).reshape([s.stop - s.start for s in sl])
            out[sl] = np.minimum(out[sl], block)
        return ctx.xp.asarray(out - np.float32(skin))

    allv = tri.reshape(-1, 3)
    return Field(fn, _digest("frame skin", np.ascontiguousarray(tri).tobytes(), skin), "skin",
                 np.array([allv.min(axis=0) - skin, allv.max(axis=0) + skin]), cache=True)


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
               fillet=0.0):
    """The frame as a part: its bars and the skin at supports and loads (joined with the same
    fillet as the bars, unless the joints are plain), inside the design space."""
    face = _at_faces(conditions, nodes) if len(conditions) else None
    body = frame_solid(nodes, bars, chains, sizes, profile, joints, key, no_ball=face, fillet=fillet)
    around = skin_body(conditions, skin, nodes)
    if around is not None:
        if joints == "plain":
            body = F.union(body, around)
        else:   # bars into bosses: the fillet of the median bar (they ended in sharp corners: 81-92 MPa)
            radius = float(fillet) if fillet > 0 else FILLET * float(np.median(np.asarray(sizes)[np.asarray(chains)]))
            body = F.smooth_union(body, around, radius)
    return F.intersect(space, body)
