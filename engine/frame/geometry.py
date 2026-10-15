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
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh
import warp as wp

from .. import backend
from .. import fields as F
from ..fields import Field, _digest
from .curves import orientation
from .profiles import Profile

JOINTS = ("fillet", "plain", "ball")
FILLET = 0.6          # automatic fillet: this x the thinner bar's size
BALL = 0.6            # ball joint radius: this x the largest bar size (width or height) at the node

wp.config.quiet = True
wp.init()


@wp.func
def _box2(qy: float, qz: float, r: wp.vec4) -> float:
    dy = wp.abs(qy - (r[0] + r[1]) * 0.5) - (r[1] - r[0]) * 0.5
    dz = wp.abs(qz - (r[2] + r[3]) * 0.5) - (r[3] - r[2]) * 0.5
    oy = wp.max(dy, 0.0)
    oz = wp.max(dz, 0.0)
    return wp.sqrt(oy * oy + oz * oz) + wp.min(wp.max(dy, dz), 0.0)


@wp.kernel
def _frame_grid(bvh: wp.uint64, kind: wp.array(dtype=int), chain: wp.array(dtype=int), width: wp.array(dtype=float),
                a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), ey: wp.array(dtype=wp.vec3),
                ez: wp.array(dtype=wp.vec3), p0: wp.array(dtype=wp.vec3), n0: wp.array(dtype=wp.vec3),
                p1: wp.array(dtype=wp.vec3), n1: wp.array(dtype=wp.vec3), rad: wp.array(dtype=float),
                wall: wp.array(dtype=float), rects: wp.array(dtype=wp.vec4), rsign: wp.array(dtype=int),
                cen: wp.array(dtype=wp.vec2), rot: wp.array(dtype=wp.vec2), fillet: float, factor: float,
                cap: float, origin: wp.vec3, sx: int, sy: int, sz: int, voxel: float, ny: int, nz: int, first: int,
                out: wp.array(dtype=float)):
    tid = wp.tid() + first
    i = tid // (ny * nz)
    j = (tid // nz) % ny
    k = tid % nz
    p = origin + wp.vec3(float(i + sx), float(j + sy), float(k + sz)) * voxel
    d1 = cap
    c1 = int(-2)
    w1 = float(0.0)
    d2 = cap
    c2 = int(-2)
    w2 = float(0.0)
    query = wp.bvh_query_aabb(bvh, p, p)
    n = int(0)
    while wp.bvh_query_next(query, n):
        kd = kind[n]
        d = float(0.0)
        if kd == 0:  # round bar: capsule
            ab = b[n] - a[n]
            t = wp.clamp(wp.dot(p - a[n], ab) / wp.max(wp.dot(ab, ab), 1.0e-12), 0.0, 1.0)
            d = wp.length(p - (a[n] + t * ab)) - rad[n]
        elif kd == 3:  # ball
            d = wp.length(p - a[n]) - rad[n]
        else:
            rel = p - a[n]
            y = wp.dot(rel, ey[n])
            z = wp.dot(rel, ez[n])
            flat = float(0.0)
            if kd == 1:  # round tube
                rho = wp.sqrt(y * y + z * z)
                flat = wp.abs(rho - (rad[n] - wall[n] * 0.5)) - wall[n] * 0.5
            else:  # rectangles: union of the solid ones, minus the holes
                cs = rot[n]
                cc = cen[n]
                qy = cc[0] + y * cs[0] - z * cs[1]
                qz = cc[1] + y * cs[1] + z * cs[0]
                solid = float(1.0e30)
                hole = float(-1.0e30)
                for r in range(4):
                    sg = rsign[n * 4 + r]
                    if sg > 0:
                        solid = wp.min(solid, _box2(qy, qz, rects[n * 4 + r]))
                    if sg < 0:
                        hole = wp.max(hole, -_box2(qy, qz, rects[n * 4 + r]))
                flat = wp.max(solid, hole)
            along = wp.max(wp.dot(p - p0[n], n0[n]), wp.dot(p - p1[n], n1[n]))
            if flat > 0.0 and along > 0.0:
                d = wp.sqrt(flat * flat + along * along)
            else:
                d = wp.max(flat, along)
        cn = chain[n]
        wn = width[n]
        # the two nearest different bars (each at its nearest piece)
        if cn == c1:
            d1 = wp.min(d1, d)
        elif cn == c2:
            d2 = wp.min(d2, d)
            if d2 < d1:
                td = d1
                tc = c1
                tw = w1
                d1 = d2
                c1 = c2
                w1 = w2
                d2 = td
                c2 = tc
                w2 = tw
        elif d < d1:
            d2 = d1
            c2 = c1
            w2 = w1
            d1 = d
            c1 = cn
            w1 = wn
        elif d < d2:
            d2 = d
            c2 = cn
            w2 = wn
    res = d1
    if c2 != -2:
        kk = fillet
        if kk <= 0.0:
            kk = factor * wp.min(w1, w2)
        if kk > 0.0:
            h = wp.max(kk - wp.abs(d1 - d2), 0.0) / kk
            res = wp.min(d1, d2) - h * h * kk * 0.25
    out[tid] = res


_KERNEL_DATA = OrderedDict()  # (key, device) -> (wp.Bvh, its arrays): built once per process
PIECE = 4_000_000             # grid points per CPU launch


def _device_data(data, key, device):
    entry = _KERNEL_DATA.get((key, device))
    if entry is None:
        vec = lambda x, t=wp.vec3: wp.array(np.ascontiguousarray(x, dtype=np.float32), dtype=t, device=device)
        ints = lambda x: wp.array(np.ascontiguousarray(x, dtype=np.int32).reshape(-1), dtype=int, device=device)
        flts = lambda x: wp.array(np.ascontiguousarray(x, dtype=np.float32), dtype=float, device=device)
        arrays = [ints(data["kind"]), ints(data["chain"]), flts(data["width"]),
                  vec(data["a"]), vec(data["b"]), vec(data["ey"]), vec(data["ez"]), vec(data["p0"]), vec(data["n0"]),
                  vec(data["p1"]), vec(data["n1"]), flts(data["rad"]), flts(data["wall"]),
                  vec(data["rects"].reshape(-1, 4), wp.vec4), ints(data["rsign"]),
                  vec(data["cen"], wp.vec2), vec(data["rot"], wp.vec2)]
        lo, hi = vec(data["lo"]), vec(data["hi"])
        entry = (wp.Bvh(lo, hi), arrays, (lo, hi))
        _KERNEL_DATA[(key, device)] = entry
        while len(_KERNEL_DATA) > 8:
            _KERNEL_DATA.popitem(last=False)
    else:
        _KERNEL_DATA.move_to_end((key, device))
    return entry


def _evaluate(data, grid, key, gpu, fillet, factor):
    device = "cuda" if gpu else "cpu"
    bvh, arrays, _ = _device_data(data, key, device)
    common = [bvh.id, *arrays, float(fillet), float(factor), float(data["cap"]),
              wp.vec3(*[float(c) for c in grid.base]), *grid.start, float(grid.voxel), grid.shape[1], grid.shape[2]]
    if gpu:
        cp = backend.cupy()
        out = cp.empty(grid.size, dtype=cp.float32)
        cp.cuda.Device().synchronize()  # Warp runs on its own stream
        wp.launch(_frame_grid, dim=grid.size, inputs=[*common, 0, wp.from_dlpack(out)], device=device)
        wp.synchronize_device(device)
        return out.reshape(grid.shape)
    out = np.empty(grid.size, dtype=np.float32)
    target = wp.array(out, dtype=float, device="cpu", copy=False)
    threads = int(os.environ.get("IGL_NUM_THREADS") or os.cpu_count() or 4)
    step = max(1, min(PIECE, -(-grid.size // (4 * threads))))
    starts = range(0, grid.size, step)
    launch = lambda s: wp.launch(_frame_grid, dim=min(step, grid.size - s), inputs=[*common, s, target], device="cpu")
    if threads <= 1 or len(starts) == 1:
        for s in starts:
            launch(s)
    else:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(launch, starts))
    return out.reshape(grid.shape)


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
    blend = (0.0, 0.0) if joints == "plain" else ((float(fillet), 0.0) if fillet > 0 else (0.0, FILLET))
    k = _digest("frame solid", key, profile, joints, float(fillet))

    def fn(ctx):
        return _evaluate(data, ctx.grid, k, ctx.xp is not np, *blend)

    bounds = np.array([data["lo"].min(axis=0), data["hi"].max(axis=0)])
    return Field(fn, k, "frame", bounds, cache=True)


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
