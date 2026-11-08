import math

import numpy as np
import pytest

from engine import copies as C
from engine.cad_import import tessellate


def _box(x0, y0, z0, dx, dy, dz):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()


def _round(r1, r2, h):
    """A cylinder (r1 = r2) or a cone about z, from z = 0 to h."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    return (BRepPrimAPI_MakeCylinder(ax, r1, h) if r1 == r2 else BRepPrimAPI_MakeCone(ax, r1, r2, h)).Shape()


def mesh_of(shape):
    return tessellate(shape, 0.01, 0.5)


def side_face(mesh):
    ids = mesh.face_attributes["cad_face"]
    return int(np.bincount(ids[np.abs(mesh.face_normals[:, 2]) < 0.9]).argmax())


def top_face(mesh):
    ids = mesh.face_attributes["cad_face"]
    return int(np.bincount(ids[mesh.face_normals[:, 2] > 0.99]).argmax())


@pytest.fixture(scope="module")
def stud():
    """A 6 x 4 x 2 mm block drawn standing on y = 0 (SolidWorks' Top plane, extruded up)."""
    return mesh_of(_box(-3, 0, -2, 6, 2, 4))


def base_points(M, point=(0.0, 0.0, 0.0)):
    return M[:, :3, :3] @ np.asarray(point) + M[:, :3, 3]


def test_up_and_base_follow_how_the_part_is_drawn(stud):
    point, basis, up, why = C.frame([stud])
    assert up == "y" and "y = 0" in why and np.allclose(point, 0)
    assert np.allclose(basis, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])                  # right-handed, x kept
    hanging = mesh_of(_box(-2, -2, -5, 4, 4, 5))                                    # ends at z = 0
    point, basis, up, _ = C.frame([hanging])
    assert up == "-z" and np.allclose(point, [0, 0, 0]) and np.allclose(basis[2], [0, 0, -1])
    loose = mesh_of(_box(10, 10, 10, 8, 2, 6))                                      # nowhere near the origin
    point, _, up, why = C.frame([loose])
    assert up == "y" and "thinnest" in why and np.allclose(point, [14, 10, 13])
    point, _, up, _ = C.frame([loose], "z", "origin")
    assert up == "z" and np.allclose(point, 0)
    assert np.allclose(sorted(np.ptp(C.footprint([stud], *C.frame([stud])[:2]), axis=0)), [4, 6])


def test_copies_stand_on_a_cylinder(stud):
    tube = mesh_of(_round(20, 20, 40))
    M, sizes, lines = C.place(tube, [side_face(tube)], [stud], spacing=8.0, layout="grid")
    base = base_points(M)
    radius = np.hypot(base[:, 0], base[:, 1])
    assert len(M) > 50 and np.allclose(radius, 20, atol=0.01) and np.allclose(sizes, 1)
    up = M[:, :3, :3] @ np.array([0, 1.0, 0])                   # the part's up: straight out of the surface
    assert np.allclose(up[:, :2], base[:, :2] / radius[:, None], atol=1e-3) and np.allclose(up[:, 2], 0, atol=1e-6)
    assert np.allclose(np.linalg.det(M[:, :3, :3]), 1)            # turned, never mirrored
    rows = np.round(base[:, 2], 2)
    for z in set(rows):                                           # a whole number in every row, evenly round
        a = np.sort(np.arctan2(base[rows == z, 1], base[rows == z, 0]))
        assert np.allclose(np.diff(np.r_[a, a[0] + 2 * np.pi]), 2 * np.pi / round(2 * np.pi * 20 / 8), atol=1e-3)
    # the rows spread evenly from end to end: the 4 mm deep footprints of the first and the last row
    # just clear the bottom and top edges (Edge margin 0 + the slack for flat footprints on a curve)
    heights = np.array(sorted(set(rows)))
    assert heights[0] == pytest.approx(2.02, abs=0.01) and heights[-1] == pytest.approx(37.98, abs=0.01)
    assert len(heights) == 5 and np.allclose(np.diff(heights), (37.98 - 2.02) / 4, atol=0.02)   # >= Spacing 8
    assert any("16 around in 5 rows" in text for text, _ in lines)


@pytest.mark.parametrize("girth", ["same size", "same count", "scale"])
def test_copies_follow_a_cones_girth(stud, girth):
    cone = mesh_of(_round(30, 8, 50))
    M, sizes, lines = C.place(cone, [side_face(cone)], [stud], spacing=8.0, layout="grid", girth=girth)
    base = base_points(M)
    rows = np.round(base[:, 2], 2)
    heights = sorted(set(rows))
    counts = [int((rows == z).sum()) for z in heights]
    radius = np.hypot(base[:, 0], base[:, 1])
    assert len(heights) >= 5 and not any("overlap" in text for text, _ in lines)
    if girth == "same size":       # fewer round the narrower rows, but only now and then: 0.9-1.25 Spacings apart
        assert counts == sorted(counts, reverse=True) and counts[0] > 2 * counts[-1]
        for z, k in zip(heights, counts):
            assert 0.9 * 8.0 - 0.05 <= 2 * np.pi * radius[rows == z][0] / k <= 1.25 * 8.0 + 0.05
        assert np.allclose(sizes, 1)
    else:                          # as many in every row
        assert len(set(counts)) == 1
        if girth == "same count":  # the same size, but where the cone squeezes them they shrink to keep the Gap
            assert np.allclose(sizes[rows == heights[0]], 1) and sizes[rows == heights[-1]].max() < 0.7
            assert any("shrunk to keep the Gap" in text for text, _ in lines)
            gap = next(float(t.split()[-2]) for t, _ in lines if "narrowest gap" in t)
            assert gap >= 1.0 - 0.02
        else:                      # and everything grows with the girth: size / radius stays the same
            ratio = sizes / radius
            assert np.ptp(sizes) > 0.5 and np.ptp(ratio) < 0.01 * ratio.mean()
            gaps = np.diff(heights)              # rows farther apart where the cone is wider
            assert gaps[0] > 1.5 * gaps[-1]


@pytest.mark.parametrize("r1, r2, girth", [(20, 30, "same size"), (30, 8, "same size"), (30, 8, "scale")],
                         ids=["flaring", "narrowing", "narrowing-scale"])
def test_copies_stay_between_the_planes_of_the_ends(stud, r1, r2, girth):
    # where the face leans out over an end (a flaring foot, a narrowing neck), the copies lean with it:
    # the end rows sit just far enough in that nothing pokes past the end's plane, and no farther
    cone = mesh_of(_round(r1, r2, 40))
    M, sizes, _ = C.place(cone, [side_face(cone)], [stud], spacing=8.0, girth=girth)
    z = np.einsum("nij,kj->nki", M[:, :3, :3], stud.vertices)[..., 2] + M[:, None, 2, 3]
    assert z.min() >= -1e-6 and z.max() <= 40 + 1e-6
    assert z.min() < 0.5 and z.max() > 39.5
    assert np.allclose(sizes, 1) if girth == "same size" else np.ptp(sizes) > 0.5


def test_rows_keep_their_count_while_they_can(stud):
    # a gentle cone: rows keep the count of the row below until the copies would be 25% too far apart
    cone = mesh_of(_round(30, 21, 60))
    M, sizes, lines = C.place(cone, [side_face(cone)], [stud], spacing=8.0)
    rows = np.round(base_points(M)[:, 2], 2)
    counts = [int((rows == z).sum()) for z in sorted(set(rows))]
    assert len(counts) >= 6 and 1 < len(set(counts)) <= len(counts) // 2
    assert counts == sorted(counts, reverse=True)
    assert not any("overlap" in text for text, _ in lines) and np.allclose(sizes, 1)


def test_crowded_copies_shrink_or_are_left_out(stud):
    cone = mesh_of(_round(30, 8, 50))
    face = side_face(cone)
    shrink, leave, overlap = (C.place(cone, [face], [stud], spacing=8.0, layout="grid", girth="same count",
                                      crowded=mode) for mode in ("shrink", "leave out", "overlap"))
    assert len(shrink[0]) == len(overlap[0]) > len(leave[0])
    assert np.allclose(overlap[1], 1) and np.allclose(leave[1], 1) and shrink[1].min() < 0.7
    assert any("overlap" in t for t, _ in overlap[2])
    assert not any("overlap" in t for t, _ in leave[2] + shrink[2])


def test_copies_at_an_edge_shrink_to_fit_inside():
    plate = mesh_of(_box(0, 0, 0, 47, 33, 3))            # not a whole number of Spacings either way
    block = mesh_of(_box(-3, -2, 0, 6, 4, 2))
    out = C.place(plate, [top_face(plate)], [block], spacing=8.0, layout="grid", edges="leave out")
    fit = C.place(plate, [top_face(plate)], [block], spacing=8.0, layout="grid", edges="shrink")
    assert len(fit[0]) > len(out[0]) and np.allclose(out[1], 1) and fit[1].min() < 0.95
    for M in (out[0], fit[0]):
        feet = footprints(M)[..., :2]
        assert feet[..., 0].min() >= -1e-6 and feet[..., 0].max() <= 47 + 1e-6
        assert feet[..., 1].min() >= -1e-6 and feet[..., 1].max() <= 33 + 1e-6
    assert any("shrunk to fit inside the edge" in t for t, _ in fit[2])


def footprints(M):
    """The 6 x 4 block's footprint under every copy (its underside's corners, as drawn: standing on
    z = 0), in the plate's plane."""
    corners = np.array([[-3.0, -2, 0], [3, -2, 0], [3, 2, 0], [-3, 2, 0]])
    return np.einsum("nij,kj->nki", M[:, :3, :3], corners) + M[:, None, :3, 3]


def polygon_gap(a, b):
    """Distance between two convex polygons in the xy plane (negative: they overlap)."""
    def inside(p, poly):
        e = np.roll(poly, -1, axis=0) - poly
        q = p - poly
        cross = e[:, 0] * q[:, 1] - e[:, 1] * q[:, 0]
        return np.all(cross >= 0) or np.all(cross <= 0)

    def to_edges(points, poly):
        s0, s1 = poly, np.roll(poly, -1, axis=0)
        d = s1 - s0
        t = np.clip(np.einsum("pkj,kj->pk", points[:, None] - s0, d) / np.einsum("kj,kj->k", d, d), 0, 1)
        return np.linalg.norm(points[:, None] - (s0 + t[..., None] * d), axis=2).min()
    if any(inside(p, b) for p in a) or any(inside(p, a) for p in b):
        return -1.0
    return min(to_edges(a, b), to_edges(b, a))


def test_voronoi_cells_give_every_copy_its_own_size():
    plate = mesh_of(_box(0, 0, 0, 80, 60, 3))
    block = mesh_of(_box(-3, -2, 0, 6, 4, 2))                     # stands on z = 0
    M, sizes, lines = C.place(plate, [top_face(plate)], [block], spacing=8.0, cells="voronoi", gap=1.0,
                              randomness=0.8, seed=2)
    assert len(M) > 40 and np.ptp(sizes) > 0.3                    # cells of many sizes, copies too
    assert np.allclose(base_points(M)[:, 2], 3, atol=1e-6)        # on the top
    feet = footprints(M)[..., :2]
    assert feet[..., 0].min() >= -1e-6 and feet[..., 0].max() <= 80 + 1e-6    # inside the face
    assert feet[..., 1].min() >= -1e-6 and feet[..., 1].max() <= 60 + 1e-6
    middle = feet.mean(axis=1)
    from scipy.spatial import cKDTree
    gaps = [polygon_gap(feet[i], feet[j]) for i, j in cKDTree(middle).query_pairs(20.0)]
    assert min(gaps) >= 1.0 - 1e-6                                # never closer than the Gap
    turned, larger, _ = C.place(plate, [top_face(plate)], [block], spacing=8.0, cells="voronoi", gap=1.0,
                                randomness=0.8, seed=2, turn_fit=True)
    assert larger.mean() > 1.05 * sizes.mean()                    # each turned to fit its cell best
    feet = footprints(turned)[..., :2]
    gaps = [polygon_gap(feet[i], feet[j]) for i, j in cKDTree(feet.mean(axis=1)).query_pairs(20.0)]
    assert min(gaps) >= 1.0 - 1e-6
    regular, s, _ = C.place(plate, [top_face(plate)], [block], spacing=8.0, cells="voronoi", gap=1.0)
    inner = np.all((footprints(regular)[..., :2] > 10) & (footprints(regular)[..., :2] < 50), axis=(1, 2))
    assert inner.sum() > 10 and np.ptp(s[inner]) < 1e-6           # regular cells: all alike
    # honeycomb cells 8 mm apart less the 1 mm gap: 3.5 mm from the middle to every side. The block's
    # corner (3, 2) meets the side whose normal is 60° from its x first
    assert s[inner][0] == pytest.approx(3.5 / (3 * 0.5 + 2 * math.sqrt(3) / 2), rel=1e-3)


def test_voronoi_cells_stop_at_the_edge():
    # cells are cut at the plate's edge (Edge margin in), so the copies along it fill up to it
    plate = mesh_of(_box(0, 0, 0, 80, 60, 3))
    block = mesh_of(_box(-3, -2, 0, 6, 4, 2))
    M, sizes, _ = C.place(plate, [top_face(plate)], [block], spacing=8.0, cells="voronoi", gap=1.0,
                          randomness=0.8, seed=2, margin=0.5)
    feet = footprints(M)[..., :2]
    assert feet[..., 0].min() >= 0.5 - 1e-6 and feet[..., 0].max() <= 79.5 + 1e-6
    assert feet[..., 1].min() >= 0.5 - 1e-6 and feet[..., 1].max() <= 59.5 + 1e-6
    assert feet[..., 0].min() < 1.0 and feet[..., 1].max() > 59.0             # some reach right up to it


def _gaps(outlines):
    """Every outline's gap to its nearest neighbour."""
    from scipy.spatial import cKDTree
    middle = np.array([o.mean(axis=0) for o in outlines])
    nearest = np.full(len(outlines), np.inf)
    for i, j in cKDTree(middle).query_pairs(20.0):
        nearest[[i, j]] = np.minimum(nearest[[i, j]], polygon_gap(outlines[i], outlines[j]))
    return nearest


def test_fit_stretch_fills_cells_further():
    plate = mesh_of(_box(0, 0, 0, 80, 60, 3))
    block = mesh_of(_box(-3, -2, 0, 6, 4, 2))
    kw = dict(spacing=8.0, cells="voronoi", gap=1.0, randomness=0.8, seed=2, turn_fit=True)
    scaled, s, _ = C.place(plate, [top_face(plate)], [block], fit="scale", **kw)
    stretched, t, lines = C.place(plate, [top_face(plate)], [block], fit="stretch", **kw)
    assert (t ** 2).sum() > 1.15 * (s ** 2).sum()                 # the footprints' area: fuller cells
    assert np.allclose(np.linalg.det(stretched[:, :3, :3]), t ** 3)  # area kept by the stretch, height times size
    ratio = np.linalg.svd(stretched[:, :3, :3], compute_uv=False)
    assert (ratio[:, 0] / ratio[:, 2]).max() > 1.3 and (ratio[:, 0] / ratio[:, 2]).max() <= C.STRETCH + 1e-9
    feet = footprints(stretched)[..., :2]
    assert np.allclose(footprints(stretched)[..., 2], 3, atol=1e-6)
    assert feet[..., 0].min() >= -1e-6 and feet[..., 0].max() <= 80 + 1e-6
    assert feet[..., 1].min() >= -1e-6 and feet[..., 1].max() <= 60 + 1e-6
    assert min(_gaps(list(feet))) >= 1.0 - 1e-6
    assert any("stretched up to" in text for text, _ in lines)


def test_fit_fill_gives_every_copy_its_cell():
    from scipy.spatial import ConvexHull
    plate = mesh_of(_box(0, 0, 0, 80, 60, 3))
    block = mesh_of(_box(-3, -2, 0, 6, 4, 2))
    kw = dict(spacing=8.0, cells="voronoi", gap=1.0, randomness=0.8, seed=2, turn_fit=True)
    _, s, _ = C.place(plate, [top_face(plate)], [block], **kw)
    placed = C.place(plate, [top_face(plate)], [block], fit="fill", **kw)
    bodies = placed.bodies([block])
    assert len(bodies) == len(placed) > 40
    outlines = []
    for b, size in zip(bodies, placed.sizes):
        assert b.is_watertight and b.volume > 0                     # split conformingly, still outwards
        assert set(np.unique(b.face_attributes["cad_face"])) == set(np.unique(block.face_attributes["cad_face"]))
        z = np.asarray(b.vertices)[:, 2]
        assert z.min() == pytest.approx(3) and z.max() == pytest.approx(3 + 2 * size)   # height times its size
        xy = np.asarray(b.vertices)[:, :2]
        outlines.append(xy[ConvexHull(xy).vertices])
    area = sum(abs(C._area(o)) for o in outlines)
    assert area > 1.5 * (s ** 2).sum() * 24                       # far fuller than as drawn (6 x 4 footprints)
    gaps = _gaps(outlines)
    assert min(gaps) >= 1.0 - 1e-6 and np.median(gaps) < 1.05     # the Gap, all round
    lo = np.min([o.min(axis=0) for o in outlines], axis=0)
    hi = np.max([o.max(axis=0) for o in outlines], axis=0)
    assert np.allclose(lo, 0, atol=1e-6) and np.allclose(hi, [80, 60], atol=1e-6)   # right up to the edge


def test_fill_warp_keeps_the_edge_profile():
    # a point d in from the footprint's edge goes to a point d in from the outline's edge (within the band)
    footprint = np.array([[-3.0, -2], [3, -2], [3, 2], [-3, 2]])
    a = np.radians(np.arange(0, 360, 60) + 10)
    outline = np.column_stack([5 * np.cos(a), 4 * np.sin(a)]) + [0.3, -0.2]
    warp = C._Warp(footprint, outline, 1.0)
    rng = np.random.default_rng(1)
    q = rng.uniform([-3, -2], [3, 2], (2000, 2))
    normal, offset = C._halfplanes(footprint[None])
    d = np.clip((offset[0] - q @ normal[0].T).min(axis=1), 0, None)
    band = d <= warp.band
    from engine import pattern2d as P2
    after = -P2.polygon_sdf(*warp(q[band]).T, outline)
    assert band.sum() > 500 and np.allclose(after, d[band], atol=1e-9)
    inside = -P2.polygon_sdf(*warp(q[~band]).T, outline)            # the rest stays farther in
    assert inside.min() >= warp.band - 1e-9


def test_fit_fill_bends_copies_onto_a_cylinder(stud):
    tube = mesh_of(_round(20, 20, 40))
    placed = C.place(tube, [side_face(tube)], [stud], spacing=8.0, cells="voronoi", gap=1.0, randomness=0.7,
                     fit="fill")
    bodies = placed.bodies([stud])
    assert len(bodies) > 30
    for b, M in zip(bodies, placed.matrices):
        V = np.asarray(b.vertices)
        r = np.hypot(V[:, 0], V[:, 1])
        assert r.min() == pytest.approx(20, abs=0.02) and V[:, 2].min() >= -1e-6 and V[:, 2].max() <= 40 + 1e-6
        height = (r.max() - 20) / np.linalg.norm(M[:3, 0])        # the 2 mm stud at the copy's size
        assert height == pytest.approx(2, abs=0.05)


@pytest.mark.parametrize("fit", ["scale", "fill"])
def test_fitted_copies_stay_between_the_planes_of_the_ends(stud, fit):
    # the cone narrows towards its foot: copies standing on its normal lean down over the foot's plane,
    # so the cells there stop as far in as the copies are tall times that lean. (Filled copies bent onto
    # the surface sit on its triangles, which lie up to the tessellation's 0.01 mm inside the cone.)
    cone = mesh_of(_round(20, 30, 40))
    placed = C.place(cone, [side_face(cone)], [stud], spacing=8.0, cells="voronoi", gap=1.0, randomness=0.7, fit=fit)
    z = np.concatenate([np.asarray(b.vertices)[:, 2] for b in placed.bodies([stud])])
    tol = 1e-6 if fit == "scale" else 0.01
    assert z.min() >= -tol and z.max() <= 40 + tol and z.min() < 1.5


def test_pattern_on_faces_node(tmp_path, stud):
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from engine.graph import Evaluator, Graph, GraphError, link
    for name, shape in (("stud", _box(-3, 0, -2, 6, 2, 4)), ("tube", _round(20, 20, 40))):
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(str(tmp_path / f"{name}.step"))
    g = Graph()
    g.add("stud", "input.import_step", path=str(tmp_path / "stud.step"))
    g.add("tube", "input.import_step", path=str(tmp_path / "tube.step"))
    tube = Evaluator(g, log=None).run()["tube"]["bodies"][0]
    g.add("copies", "cad.pattern_faces", body=link("stud.bodies"), cad=link("tube.bodies"),
          faces=str(side_face(tube)), spacing=8.0, cells="voronoi", randomness=0.7)
    events = []
    out = Evaluator(g, log=None, on_event=events.append).run()["copies"]["body"]
    note = next(e["note"] for e in events if e["type"] == "node_done" and e["node"] == "copies")
    assert note.startswith(f"{len(out)} copies, ") and "x" in note
    assert len(out) > 40 and all(np.array_equal(c.face_attributes["cad_face"], stud.face_attributes["cad_face"])
                                 for c in out)                    # faces picked on the part stay picked
    text = [e["text"] for e in events if e["type"] == "message"]
    assert any(t.startswith("Pattern on Faces:") and "copies on 1 face" in t for t in text)
    g.set("copies", "fit", "fill")                                # every copy bent onto its cell
    events = []
    filled = Evaluator(g, log=None, on_event=events.append).run()["copies"]["body"]
    assert len(filled) == len(out) and all(len(c.faces) > len(stud.faces) for c in filled)
    assert all(set(c.face_attributes["cad_face"]) == set(stud.face_attributes["cad_face"]) for c in filled)
    assert any("triangles in the filled copies" in e["text"] for e in events if e["type"] == "message")
    g.set("copies", "faces", "")
    with pytest.raises(GraphError, match="pick the faces"):
        Evaluator(g, log=None).run()
