import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from engine import fields as F
from engine import holes as H
from engine import pattern2d as P2
from engine.cad_import import tessellate
from engine.graph.nodes_pattern import card_note
from engine.grid import Grid
from engine.meshing import export, field_to_mesh
from engine.transform import scale_about


# --- test parts (OpenCascade solids) -------------------------------------------------------------
def _box(x0, y0, z0, dx, dy, dz):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()


def _cylinder(x, y, z, r, h):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1)), r, h).Shape()


def _cut(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    return BRepAlgoAPI_Cut(a, b).Shape()


def mesh_of(shape):
    return tessellate(shape, 0.01, 0.5)


def face_where(mesh, test):
    """The CAD face most of whose triangles pass test(centres, normals)."""
    ids = mesh.face_attributes["cad_face"]
    return int(np.bincount(ids[test(mesh.triangles_center, mesh.face_normals)]).argmax())


def top_face(mesh):
    z = mesh.bounds[1][2]
    return face_where(mesh, lambda c, n: (np.abs(c[:, 2] - z) < 1e-6) & (n[:, 2] > 0.99))


def cut_mesh(part, holes, voxel, padding=1.5, raised=False):
    combine = F.union if raised else F.subtract
    body = combine(F.body_from_mesh(part, 2.0), H.cutter(holes))
    grid = Grid.from_bounds(part.bounds, voxel, padding)
    return field_to_mesh(F.Context(grid).eval_full(body), grid)


def reloaded(mesh, tmp_path, name="out.stl"):
    export(mesh, tmp_path / name)
    return trimesh.load(tmp_path / name)


def genus(mesh):
    return (2 * len(mesh.split()) - mesh.euler_number) // 2


def section_loops(mesh, z):
    """Closed loops of a horizontal section: (areas, centres), outer outline first."""
    path = mesh.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
    down = np.eye(4)
    down[2, 3] = -z
    flat, _ = path.to_2D(to_2D=down)
    loops = [l[:-1] if np.allclose(l[0], l[-1]) else l for l in flat.discrete]
    areas = np.array([abs(0.5 * np.sum(l[:, 0] * np.roll(l[:, 1], -1) - np.roll(l[:, 0], -1) * l[:, 1]))
                      for l in loops])
    centres = np.array([l.mean(axis=0) for l in loops])
    order = np.argsort(-areas)
    return areas[order], centres[order]


# --- milestone 1: surface parameters from the import ---------------------------------------------
def _nurbs(shape):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
    return BRepBuilderAPI_NurbsConvert(shape, True).Shape()


def _moved(shape):
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    rotate, shift = gp_Trsf(), gp_Trsf()
    rotate.SetRotation(gp_Ax1(gp_Pnt(1, 2, 3), gp_Dir(1, 1, 0.3)), 0.7)
    shift.SetTranslation(gp_Vec(10.5, -3.25, 7.125))
    return shape.Moved(TopLoc_Location(shift.Multiplied(rotate)))


@pytest.mark.parametrize("make", [
    lambda: _box(0, 0, 0, 30, 20, 10),
    lambda: _cylinder(0, 0, 0, 10, 20),
    lambda: _nurbs(_cylinder(0, 0, 0, 10, 20)),     # B-spline faces
    lambda: _moved(_cylinder(0, 0, 0, 10, 20)),     # placed as in an assembly
    lambda: _cut(_box(0, 0, 0, 40, 30, 20), _box(2, 2, 2, 36, 26, 16)),  # reversed inner faces
], ids=["box", "cylinder", "bspline", "located", "hollow"])
def test_corner_uvs_evaluate_back_to_the_corners(make):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    shape = make()
    mesh = mesh_of(shape)
    ids, uv = mesh.face_attributes["cad_face"], mesh.face_attributes["cad_uv"]
    assert uv.shape == (len(mesh.faces), 3, 2) and uv.dtype == np.float32 and np.isfinite(uv).all()
    corners = mesh.vertices[mesh.faces]
    exp, f, worst = TopExp_Explorer(shape, TopAbs_FACE), 0, 0.0
    while exp.More():
        surface = BRep_Tool.Surface_s(TopoDS.Face(exp.Current()))  # with the face's location applied
        exp.Next()
        tri = np.flatnonzero(ids == f)
        for t in tri[:: max(1, len(tri) // 40)]:
            for k in range(3):
                p = surface.Value(float(uv[t, k, 0]), float(uv[t, k, 1])).Coord()
                worst = max(worst, float(np.linalg.norm(np.array(p) - corners[t, k])))
        f += 1
    assert worst < 1e-4  # mm; the tessellation tolerance is 0.01
    scaled = scale_about(mesh, np.zeros(3), 1.2)
    assert np.array_equal(scaled.face_attributes["cad_uv"], uv)
    assert np.array_equal(scaled.metadata["cad_surfaces"], mesh.metadata["cad_surfaces"])


def test_surface_types_and_periods_are_kept():
    mesh = mesh_of(_cylinder(0, 0, 0, 10, 20))
    side = face_where(mesh, lambda c, n: np.abs(n[:, 2]) < 0.1)
    kinds = mesh.metadata["cad_surfaces"]
    assert kinds[side][0] == 1 and kinds[side][1] == pytest.approx(2 * np.pi) and kinds[side][2] == 0  # cylinder
    assert kinds[top_face(mesh)][0] == 0 and kinds[top_face(mesh)][1] == 0                              # plane
    assert mesh_of(_nurbs(_cylinder(0, 0, 0, 10, 20))).metadata["cad_surfaces"][0][0] == 6              # B-spline


# --- milestone 3: shapes -------------------------------------------------------------------------
SHAPES = [P2.Shape("round", 4), P2.Shape("square", 4, corner=0.5), P2.Shape("hexagon", 4, corner=0.4),
          P2.Shape("triangle", 3, rotation=10), P2.Shape("octagon", 5), P2.Shape("polygon", 4, sides=7),
          P2.Shape("slot", 3, 9, rotation=30), P2.Shape("diamond", 4, 7, corner=0.3), P2.Shape("cross", 2, 8),
          P2.Shape("cross", 2.5, 8, corner=0.4, rotation=15), P2.Shape("rectangle", 3, 7, corner=0.5, rotation=20),
          P2.Shape("star", 3, 8, sides=5), P2.Shape("star", 3, 8, sides=6, corner=0.3),
          P2.Shape("chevron", 1.5, 8, corner=0.3, rotation=15)]


def _segments_distance(points, loop):
    a, b = loop, np.roll(loop, -1, axis=0)
    ab = b - a
    best = np.full(len(points), np.inf)
    for s in range(0, len(a), 4000):
        pa = points[:, None] - a[None, s:s + 4000]
        t = np.clip(np.einsum("ijk,jk->ij", pa, ab[s:s + 4000]) / np.einsum("jk,jk->j", ab[s:s + 4000], ab[s:s + 4000]),
                    0, 1)
        best = np.minimum(best, np.linalg.norm(pa - t[..., None] * ab[None, s:s + 4000], axis=2).min(axis=1))
    return best


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.kind)
def test_shapes_are_exact_distances_with_the_right_area(shape):
    # the area of the inside matches the closed form, and the values are distances to the outline
    r = shape.circumradius()
    n = 1200
    axis = np.linspace(-1.3 * r, 1.3 * r, n)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    d = shape.sdf(x, y)
    cell = (axis[1] - axis[0]) ** 2
    assert (d < 0).sum() * cell == pytest.approx(shape.area(), rel=4e-3)
    assert np.abs(shape.outline(512)).max() <= r * (1 + 1e-6)
    rng = np.random.default_rng(1)
    points = rng.uniform(-1.5 * r, 1.5 * r, (3000, 2))
    true = _segments_distance(points, shape.outline(20000))
    value = shape.sdf(points[:, 0], points[:, 1])
    outside = value > 0
    assert np.abs(value[outside] - true[outside]).max() < 2e-3
    if shape.kind == "cross" or (shape.kind in ("star", "chevron") and shape.corner > 0):
        # the union of two boxes, or a polygon grown round by the corner radius: never more than the
        # true distance inside (the star's centre: 1.43 instead of 1.58), exact outside
        assert np.all(-value[~outside] <= true[~outside] + 2e-3)
    else:
        assert np.abs(-value[~outside] - true[~outside]).max() < 2e-3


def test_shape_sizes_and_orientation():
    hexagon = P2.Shape("hexagon", 4)
    assert hexagon.sdf(2.0, 0.0) == pytest.approx(0, abs=1e-12)                    # a flat faces +x
    assert hexagon.sdf(0.0, 4 / math.sqrt(3)) == pytest.approx(0, abs=1e-12)       # a corner points +y
    # inner circle 2 mm; centred half way between the flat and the tip (turned 180° it fills the same band)
    triangle = P2.Shape("triangle", 2)
    assert triangle.sdf(0.0, -1.5) == pytest.approx(0, abs=1e-12) and triangle.sdf(0.0, 1.5) == pytest.approx(0, abs=1e-12)
    slot = P2.Shape("slot", 2, 10, rotation=90)
    assert slot.sdf(0.0, 5.0) == pytest.approx(0, abs=1e-12) and slot.sdf(1.0, 0.0) == pytest.approx(0, abs=1e-12)
    diamond = P2.Shape("diamond", 4, 6)
    assert diamond.sdf(2.0, 0.0) == pytest.approx(0, abs=1e-12) and diamond.sdf(0.0, 3.0) == pytest.approx(0, abs=1e-12)
    square = P2.Shape("square", 4, corner=1.0)
    assert square.sdf(2.0, 0.0) == pytest.approx(0, abs=1e-12)                     # rounding keeps the size
    assert square.sdf(2.0, 2.0) == pytest.approx(math.sqrt(2) - 1, abs=1e-12)      # corner rounded with r = 1
    assert P2.steps(P2.Shape("slot", 2, 10), 4, "grid") == (12, 4)                 # end webs = side webs = 2
    assert P2.steps(P2.Shape("round", 4), 6, "staggered") == pytest.approx((6, 6 * math.sqrt(3) / 2))


def test_convex_polygon_distance_is_exact():
    rng = np.random.default_rng(4)
    a = np.sort(rng.uniform(0, 2 * np.pi, 7))
    poly = np.column_stack([np.cos(a), np.sin(a)]) * rng.uniform(2, 3, 7)[:, None]   # convex, counter-clockwise
    dense = np.concatenate([np.linspace(poly[k], poly[(k + 1) % 7], 400, endpoint=False) for k in range(7)])
    poly = np.concatenate([poly, poly[-1:], poly[-1:]])                              # padded as cells are
    points = rng.uniform(-4, 4, (2000, 2))
    value = P2.polygon_sdf(points[:, 0], points[:, 1], poly)
    assert np.abs(np.abs(value) - _segments_distance(points, dense)).max() < 1e-3
    edge, rel = np.roll(poly[:7], -1, 0) - poly[:7], points[:, None] - poly[:7]
    hull_side = np.all(edge[:, 0] * rel[..., 1] - edge[:, 1] * rel[..., 0] >= 0, axis=1)
    assert np.array_equal(value < 0, hull_side)


def test_voronoi_cells_of_a_square_grid_are_squares():
    x, y, _, _, _ = P2.lattice("grid", 5, 5, (0, 0, 20, 20), pad=10)
    cells = P2.voronoi_cells(x, y, 0.5, 10)
    middle = np.flatnonzero((x == 10) & (y == 10))[0]
    assert np.allclose(np.abs(cells[middle]).max(axis=0), [2.0, 2.0])   # 5 mm cells, 1 mm web: 4 mm squares
    assert P2.polygon_sdf(0.0, 0.0, cells[middle]) == pytest.approx(-2.0)


def test_layouts():
    x, y, turn, _, _ = P2.lattice("staggered", 2, 3, (0, 0, 10, 9))
    rows = sorted(set(np.round(y, 6)))
    assert rows == [0, 3, 6, 9]
    assert np.allclose(sorted(x[np.isclose(y, 3)]), [1, 3, 5, 7, 9])               # odd rows moved half a step
    x, y, turn, ring, _ = P2.lattice("rings", 5, 5, (-20, -20, 20, 20))
    assert np.sum(ring == 0) == 1 and np.sum(ring == 1) == 6 and np.sum(ring == 2) == 13  # round(2 pi k)
    assert np.allclose(np.arctan2(y, x)[ring == 1] % (2 * np.pi), turn[ring == 1])


# --- milestone 4: holes --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plate():
    return mesh_of(_box(0, 0, 0, 40, 30, 3))


def test_holes_in_a_plate_are_whole_and_measure_right(plate, tmp_path):
    holes = H.place(plate, [top_face(plate)], P2.Shape("round", 4), "grid", 6.0, align="x", margin=2.0)
    # centres every 6 mm from the plane's origin, at least margin + radius (4 mm) from the edges
    assert len(holes) == 6 * 4
    assert np.allclose(np.unique(np.round(holes.centre[:, 0], 6)), [6, 12, 18, 24, 30, 36])
    assert np.allclose(holes.normal, [0, 0, 1]) and np.allclose(holes.depth, 3.5, atol=1e-4)
    assert holes.smallest_web == pytest.approx(2.0, abs=1e-3)
    out = reloaded(cut_mesh(plate, holes, 0.2), tmp_path)
    assert out.is_watertight and genus(out) == 24
    areas, centres = section_loops(out, 1.5)
    assert len(areas) == 25
    diameters = 2 * np.sqrt(areas[1:] / np.pi)
    assert np.abs(diameters - 4.0).max() < 0.08
    assert np.abs(np.sort(centres[1:, 0]) - np.sort(holes.centre[:, 0])).max() < 0.05


def test_pockets_sink_the_shapes_into_the_face(plate, tmp_path):
    holes = H.place(plate, [top_face(plate)], P2.Shape("round", 4), "grid", 6.0, align="x", depth=1.0)
    assert len(holes) == 24 and np.allclose(holes.depth, 1.0)
    out = reloaded(cut_mesh(plate, holes, 0.2), tmp_path)
    assert out.is_watertight and len(out.split()) == 1 and genus(out) == 0
    assert out.volume == pytest.approx(40 * 30 * 3 - 24 * np.pi * 4, rel=4e-3)
    areas, _ = section_loops(out, 2.5)                                     # through the pockets
    assert len(areas) == 25 and np.abs(2 * np.sqrt(areas[1:] / np.pi) - 4.0).max() < 0.08
    assert len(section_loops(out, 1.5)[0]) == 1                            # below them: solid


def test_raised_shapes_stand_on_the_face(plate, tmp_path):
    shape = P2.Shape("hexagon", 3, corner=0.3)
    holes = H.place(plate, [top_face(plate)], shape, "staggered", 5.0, align="x", height=1.5)
    assert len(holes) > 20 and np.allclose(holes.outer, 1.5) and np.allclose(holes.depth, H.JOIN, atol=1e-4)
    assert any("raised shapes" in t for t, _ in holes.lines)
    out = reloaded(cut_mesh(plate, holes, 0.15, raised=True), tmp_path)
    assert out.is_watertight and len(out.split()) == 1 and genus(out) == 0
    assert out.volume == pytest.approx(40 * 30 * 3 + len(holes) * shape.area() * 1.5, rel=4e-3)
    areas, centres = section_loops(out, 3.75)                              # through the raised shapes
    assert len(areas) == len(holes) and np.allclose(areas, shape.area(), rtol=0.03)
    assert out.section(plane_origin=(0, 0, 4.7), plane_normal=(0, 0, 1)) is None   # nothing above them
    assert out.bounds[0][2] == pytest.approx(0, abs=0.01)                 # nothing pokes out below


def test_fields_set_each_pockets_depth_studs_height_and_slots_turn(plate, tmp_path):
    top = top_face(plate)
    # pockets 0.2 to 3.2 mm deep across x: the column at x = 36 is as deep as the 3 mm wall, so it goes
    # through (and no further: wall + 0.5)
    holes = H.place(plate, [top], P2.Shape("round", 4), "grid", 6.0, align="x", depth=F.X / 10 - 0.4)
    x = holes.centre[:, 0]
    assert len(holes) == 24 and np.allclose(holes.depth, np.where(x > 33, 3.5, x / 10 - 0.4), atol=1e-4)
    assert any("Depth from the field: 0.20 to 3.20 mm" in t for t, _ in holes.lines)
    out = reloaded(cut_mesh(plate, holes, 0.2), tmp_path)
    assert out.is_watertight and genus(out) == 4
    # studs as tall as the field says; where it is 0 or less, none
    studs = H.place(plate, [top], P2.Shape("hexagon", 3), "grid", 6.0, align="x", height=F.Y / 10 - 1.0)
    y = studs.centre[:, 1]
    assert np.all(y > 10) and np.allclose(studs.outer, y / 10 - 1.0, atol=1e-6)
    assert any("with Height 0" in t for t, _ in studs.lines)
    # slots turned by the field at their centres, spaced to fit in any direction
    slots = H.place(plate, [top], P2.Shape("slot", 1.5, 5), "grid", 6.0, align="x", margin=1.0, rotation=F.X * 3)
    long_axis = slots.t1 * np.cos(slots.turn)[:, None] + slots.t2 * np.sin(slots.turn)[:, None]
    seen = np.degrees(np.arctan2(long_axis[:, 1], long_axis[:, 0])) % 180
    assert len(slots) > 4 and np.allclose(seen, (3 * slots.centre[:, 0]) % 180, atol=1e-6)
    assert np.allclose(np.diff(np.unique(np.round(slots.centre[:, 0], 6))), 6 + 5 - 1.5)
    assert slots.smallest_web > 0


def test_voronoi_cells_keep_the_web(plate, tmp_path):
    # staggered: hexagonal cells 6 mm across, 1 mm web; with Randomness organic cells, the web the same
    honeycomb = H.place(plate, [top_face(plate)], P2.Shape("voronoi", 1.0), "staggered", 6.0, align="x", margin=1.0)
    assert len(honeycomb) > 10 and honeycomb.cells.shape[1] >= 6
    assert np.allclose(honeycomb.widths(), 5.0, atol=1e-6) and honeycomb.smallest_web == pytest.approx(1.0, abs=1e-3)
    # regular cells look exactly like Shape hexagon: the log and the card say so, and what makes them organic
    assert honeycomb.regular == "hexagons" and any("Randomness 0.6-0.9" in t for t, _ in honeycomb.lines)
    assert card_note(honeycomb, "holes").endswith("all regular hexagons: Randomness makes organic cells")
    rough = H.place(plate, [top_face(plate)], P2.Shape("voronoi", 1.0, corner=0.4), "staggered", 6.0, align="x",
                    margin=1.0, randomness=0.9, seed=5)
    assert rough.regular is None and card_note(rough, "holes") == f"{len(rough)} holes"
    assert len(rough) > 10 and rough.smallest_web == pytest.approx(1.0, abs=0.02)
    assert not np.allclose(rough.widths(), rough.widths()[0])            # cells of different sizes
    again = H.place(plate, [top_face(plate)], P2.Shape("voronoi", 1.0, corner=0.4), "staggered", 6.0, align="x",
                    margin=1.0, randomness=0.9, seed=5)
    other = H.place(plate, [top_face(plate)], P2.Shape("voronoi", 1.0, corner=0.4), "staggered", 6.0, align="x",
                    margin=1.0, randomness=0.9, seed=6)
    assert np.array_equal(again.centre, rough.centre) and not np.array_equal(other.centre[:5], rough.centre[:5])
    out = reloaded(cut_mesh(plate, rough, 0.1), tmp_path)
    assert out.is_watertight and genus(out) == len(rough)


def test_random_cells_close_around_a_tube():
    # cells i and i + (cells around) move alike, so the cells across the seam still keep the web
    tube = mesh_of(_cut(_cylinder(0, 0, 0, 15, 30), _cylinder(0, 0, -1, 12, 32)))
    outer = face_where(tube, lambda c, n: np.hypot(c[:, 0], c[:, 1]) > 14.9)
    cells = H.place(tube, [outer], P2.Shape("voronoi", 1.0), "staggered", 5.0, margin=1.0, randomness=1.0, seed=1)
    angle = np.arctan2(cells.centre[:, 1], cells.centre[:, 0])
    assert len(cells) > 40 and np.ptp(angle) > 6.0                       # all the way round
    assert cells.smallest_web == pytest.approx(1.0, abs=0.05)            # (cells drawn in their tangent planes)


def test_scattered_dots_keep_their_distance(plate):
    dots = H.place(plate, [top_face(plate)], P2.Shape("round", 2.5), "staggered", 4.0, align="x", margin=1.0,
                   randomness=1.0, seed=3)
    assert len(dots) > 20 and dots.smallest_web >= H.MIN_WEB - 1e-9
    assert any("too close" in t for t, _ in dots.lines)


def test_spiral_spreads_evenly_and_turns_its_shapes():
    disc = mesh_of(_cylinder(0, 0, 0, 25, 3))
    dots = H.place(disc, [top_face(disc)], P2.Shape("round", 1.5), "spiral", 4.0, margin=1.0)
    gaps = cKDTree_gaps(dots.centre)
    # neighbours 4 mm apart; a sunflower that even is a little less dense than a honeycomb (~1 per 18 mm²)
    assert len(dots) > 85 and np.median(gaps) == pytest.approx(4.0, rel=0.06) and gaps.min() > 3.3
    sun = H.place(disc, [top_face(disc)], P2.Shape("slot", 1.0, 2.5), "spiral", 4.0, margin=1.0)
    r = np.hypot(sun.centre[:, 0], sun.centre[:, 1])
    out = r > 1e-6
    long_axis = sun.t1 * np.cos(sun.turn)[:, None] + sun.t2 * np.sin(sun.turn)[:, None]
    radial = sun.centre[:, :2] / np.where(r > 0, r, 1)[:, None]
    assert np.allclose(np.abs(np.einsum("ij,ij->i", long_axis[out, :2], radial[out])), 1, atol=1e-6)


def cKDTree_gaps(points):
    from scipy.spatial import cKDTree
    return cKDTree(points).query(points, 2)[0][:, 1]


def test_raised_shapes_join_a_curved_wall(tmp_path):
    tube = mesh_of(_cut(_cylinder(0, 0, 0, 15, 30), _cylinder(0, 0, -1, 12, 32)))
    outer = face_where(tube, lambda c, n: np.hypot(c[:, 0], c[:, 1]) > 14.9)
    holes = H.place(tube, [outer], P2.Shape("round", 4), "grid", 7.0, margin=1.0, height=1.0)
    drop = 15 - math.sqrt(15 ** 2 - (0.9 * 2) ** 2)   # the wall falls away under the outline
    assert np.allclose(holes.depth, drop + H.JOIN, atol=0.012)   # the triangles: within 0.01 of the cylinder
    out = reloaded(cut_mesh(tube, holes, 0.2, raised=True), tmp_path)
    assert out.is_watertight and len(out.split()) == 1 and genus(out) == 1


def relief_mesh(part, field, amplitude, voxel):
    body = F.offset(F.body_from_mesh(part, 2.0), field * amplitude)
    grid = Grid.from_bounds(part.bounds, voxel, 1.5 + amplitude)
    return field_to_mesh(F.Context(grid).eval_full(body), grid)


def test_surface_pattern_raises_whole_studs(plate, tmp_path):
    shapes = H.place(plate, [top_face(plate)], P2.Shape("round", 4), "grid", 6.0, align="x", surface=True)
    assert len(shapes) == 24 and shapes.lines[0][0].startswith("Surface Pattern: 24 shapes")
    relief = H.relief(shapes, 0.5, 2.0)
    assert relief.range == (0.0, 1.0)
    out = reloaded(relief_mesh(plate, relief, 0.8, 0.15), tmp_path)
    assert out.is_watertight and len(out.split()) == 1 and genus(out) == 0
    assert out.bounds[1][2] == pytest.approx(3.8, abs=0.02)                 # the tops, 0.8 mm up
    areas, centres = section_loops(out, 3.4)   # half way up the side: 0.25 mm inside the 4 mm outline
    assert len(areas) == 24 and np.abs(2 * np.sqrt(areas / np.pi) - 3.5).max() < 0.1
    assert np.abs(np.sort(centres[:, 0]) - np.sort(shapes.centre[:, 0])).max() < 0.05
    assert len(section_loops(out, 3.05)[0]) == 24                            # nothing raised between them


def test_dimples_leave_the_far_side_of_a_thin_wall_alone(tmp_path):
    # the relief stops half way into the wall: with Band 2 it reached the far side of this 2 mm plate,
    # and 0.6 mm dimples on top dented the bottom too
    thin = mesh_of(_box(0, 0, 0, 40, 30, 2))
    shapes = H.place(thin, [top_face(thin)], P2.Shape("round", 3), "staggered", 4.5, align="x", margin=1.0,
                     surface=True)
    assert np.allclose(shapes.wall, 2.0, atol=1e-3)
    out = reloaded(relief_mesh(thin, H.relief(shapes, 0.5, 2.0), -0.6, 0.1), tmp_path)
    assert out.is_watertight and len(out.split()) == 1
    areas, _ = section_loops(out, 0.3)
    assert len(areas) == 1 and areas[0] == pytest.approx(40 * 30, rel=2e-3)   # the bottom: flat, whole
    assert len(section_loops(out, 1.7)[0]) == 1 + len(shapes)                  # the dimples, near the top


def test_surface_pattern_inverted_raises_the_web(plate, tmp_path):
    from engine.graph.nodes_pattern import surface_pattern
    web = surface_pattern(cad=plate, faces=str(top_face(plate)), apply="chosen faces", shape="hexagon", sides=6,
                          size=4.0, length=8.0, corner_radius=0.0, rotation=0.0, layout="staggered", spacing=5.0,
                          align="x", angle=0.0, offset_u=0.0, offset_v=0.0, margin=2.0, softness=0.4,
                          invert="yes", band=2.0, size_scale=1.0)
    count = int(web.note.split()[0])
    assert count > 20 and web.range == (0.0, 1.0)
    out = reloaded(relief_mesh(plate, web, 0.6, 0.15), tmp_path)
    assert out.is_watertight and len(out.split()) == 1
    areas, _ = section_loops(out, 3.3)          # through the raised web: its outline and every cell
    assert len(areas) == 1 + count
    assert out.bounds[0][0] == pytest.approx(0, abs=0.05) and out.bounds[1][0] == pytest.approx(40, abs=0.05)


def test_through_holes_stop_at_the_wall(tmp_path):
    # a closed hollow box: holes in its top must not reach the bottom 16 mm below
    box = mesh_of(_cut(_box(0, 0, 0, 40, 30, 20), _box(2, 2, 2, 36, 26, 16)))
    holes = H.place(box, [top_face(box)], P2.Shape("hexagon", 3), "staggered", 5.0, margin=2.0)
    assert len(holes) > 10 and np.allclose(holes.depth, 2.5, atol=1e-4)
    cut = H.cutter(holes)
    for c in holes.centre:
        assert F.sample(cut, c - (0, 0, 1.0)) < 0                    # inside the hole, in the wall
        assert F.sample(cut, c - (0, 0, 19.0)) > 0                   # the bottom wall below it
    out = reloaded(cut_mesh(box, holes, 0.25), tmp_path)
    assert out.is_watertight and genus(out) == len(holes) - 1  # outer and inner shell joined by the holes
    areas, _ = section_loops(out, 1.0)                                    # through the bottom wall
    assert len(areas) == 1


def test_holes_on_a_tube_close_around_and_point_radially(tmp_path):
    tube = mesh_of(_cut(_cylinder(0, 0, 0, 15, 30), _cylinder(0, 0, -1, 12, 32)))
    outer = face_where(tube, lambda c, n: np.hypot(c[:, 0], c[:, 1]) > 14.9)
    holes = H.place(tube, [outer], P2.Shape("round", 3), "staggered", 5.0, margin=2.0)
    radial = holes.centre[:, :2] / np.linalg.norm(holes.centre[:, :2], axis=1)[:, None]
    assert np.allclose(holes.normal[:, :2], radial, atol=1e-3) and np.allclose(holes.normal[:, 2], 0, atol=1e-6)
    rows = np.round(holes.centre[:, 2], 3)
    around = 2 * np.pi * 15 / 5.0
    for z in set(rows):  # a whole number of holes in every row, evenly spaced all the way round
        a = np.sort(np.arctan2(holes.centre[rows == z, 1], holes.centre[rows == z, 0]))
        gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
        assert len(a) == round(around) and np.allclose(gaps, 2 * np.pi / len(a), atol=1e-3)
    assert np.all(holes.depth > 3.5) and np.all(holes.depth < 3.6)  # wall 3 mm + 0.5, a little more around
    out = reloaded(cut_mesh(tube, holes, 0.25), tmp_path)
    assert out.is_watertight and genus(out) == 1 + len(holes)


def test_holes_keep_clear_of_a_cut_out():
    # square holes every 4 mm: the one at (42, 20) has a flat side 3.3 - 1.5 = 1.8 mm from the edge
    # (kept), the one at (38, 28) a corner 1.2 mm from it (dropped, though its centre is farther)
    part = mesh_of(_cut(_box(0, 0, 0, 60, 40, 3), _cylinder(30, 20, -1, 8.7, 5)))
    holes = H.place(part, [top_face(part)], P2.Shape("square", 3), "grid", 4.0, align="x", margin=1.5)
    gap = np.hypot(holes.centre[:, 0] - 30, holes.centre[:, 1] - 20) - 8.7
    corner = P2.Shape("square", 3).circumradius()
    assert len(holes) > 50 and np.all(gap >= 1.5 + 1.5 - 1e-6)     # the flat side may face the edge
    assert np.sum(gap < 1.5 + corner) > 0                            # the outline is tested, not a circle
    assert not np.any(np.all(np.isclose(holes.centre[:, :2], (38, 28)), axis=1))
    # the whole outline keeps the margin (within the tessellation's 0.01 mm and the border's sampling)
    assert _outline_gaps(holes, (30, 20), 8.7).min() >= 1.5 - 0.02


def _outline_gaps(holes, centre, radius):
    points = holes.outlines(np.arange(len(holes)), 256)
    return np.hypot(points[..., 0] - centre[0], points[..., 1] - centre[1]).min(axis=1) - radius


def test_size_scale_grades_the_holes(plate):
    scale = F.clamp(F.remap(F.X, 10, 30, 0, 1.2), 0, 1.2)
    holes = H.place(plate, [top_face(plate)], P2.Shape("round", 4), "grid", 6.0, align="x", margin=1.0,
                    size_scale=scale)
    assert np.all(holes.centre[:, 0] > 10)                           # scale 0: no hole
    expected = np.clip((holes.centre[:, 0] - 10) / 20 * 1.2, 0, 1.2)
    assert np.allclose(holes.scale, expected, atol=1e-5)  # read exactly at each centre
    assert any("Hole Pattern" in t and "Size scale 0" in t for t, _ in holes.lines)


def test_morph_blends_two_shapes_across_the_face(plate, tmp_path):
    # round 4 mm holes turning into 4 mm squares from x = 6 to x = 36: the blended outline at 45° sits
    # where (1 - t)(r - 2) + t (r / sqrt 2 - 2) = 0, and the flats stay at 2 mm all the way
    t_field = F.clamp(F.remap(F.X, 6, 36, 0, 1), 0, 1)
    holes = H.place(plate, [top_face(plate)], P2.Shape("round", 4), "grid", 6.0, align="x", margin=1.0,
                    morph_to=P2.Shape("square", 4), morph=t_field)
    t = holes.morph
    assert len(holes) > 10 and t.min() < 0.1 and t.max() > 0.9
    assert np.allclose(t, np.clip((holes.centre[:, 0] - 6) / 30, 0, 1), atol=1e-5)
    every = np.arange(len(holes))
    r = 2 / ((1 - t) + t / math.sqrt(2))
    assert np.allclose(holes.unit_sdf(r / math.sqrt(2), r / math.sqrt(2), every), 0, atol=1e-9)
    assert np.allclose(holes.unit_sdf(np.full(len(t), 2.0), np.zeros(len(t)), every), 0, atol=1e-9)
    assert any("morphing into square" in text for text, _ in holes.lines)
    out = reloaded(cut_mesh(plate, holes, 0.15), tmp_path)
    assert out.is_watertight and genus(out) == len(holes)


def test_morph_into_voronoi_cells_and_regular_into_organic(plate):
    # round holes becoming their staggered (hexagonal) Voronoi cells
    to_cells = H.place(plate, [top_face(plate)], P2.Shape("round", 3), "staggered", 6.0, align="x", margin=1.0,
                       morph_to=P2.Shape("voronoi", 1.0), morph=1.0)
    assert to_cells.cells is not None and np.allclose(to_cells.widths(), 5.0, atol=1e-6)   # whole cells
    half = H.place(plate, [top_face(plate)], P2.Shape("round", 3), "staggered", 6.0, align="x", margin=1.0,
                   morph_to=P2.Shape("voronoi", 1.0), morph=0.5)
    assert np.allclose(half.widths(), 4.0, atol=1e-6)                     # half way between 3 and 5 across
    # Randomness from a field: regular hexagonal cells on the left, organic ones on the right
    rough = F.clamp(F.remap(F.X, 12, 28, 0, 1), 0, 1)
    cells = H.place(plate, [top_face(plate)], P2.Shape("voronoi", 1.0), "staggered", 5.0, align="x", margin=1.0,
                    randomness=rough, seed=2)
    x, w = cells.centre[:, 0], cells.widths()
    # cells whose neighbours are all still regular (the ramp starts at x = 12) are exact hexagons
    assert (x < 7).sum() >= 2 and np.ptp(w[x < 7]) < 1e-6 and np.ptp(w[x > 30]) > 0.5
    assert cells.smallest_web == pytest.approx(1.0, abs=0.01)


def test_min_size_leaves_out_pinholes(plate):
    # a fade shrinks holes towards nothing: those that would be under Min size are left out whole
    from engine.graph.core import REGISTRY
    scale = F.clamp(F.remap(F.X, 10, 30, 0, 1.2), 0, 1.2)
    every = H.place(plate, [top_face(plate)], P2.Shape("round", 4), "grid", 6.0, align="x", margin=1.0,
                    size_scale=scale)
    kept = H.place(plate, [top_face(plate)], P2.Shape("round", 4), "grid", 6.0, align="x", margin=1.0,
                   size_scale=scale, min_size=1.5)
    small = every.widths() < 1.5
    assert small.sum() > 0 and len(kept) == len(every) - small.sum() and kept.widths().min() >= 1.5
    assert any("under Min size 1.5 mm" in t for t, _ in kept.lines)  # (counted before the edge check)
    assert REGISTRY["pattern.holes"].input("min_size").default == 1.0      # what hardly prints
    assert REGISTRY["pattern.surface"].input("min_size").default == 0.0    # small bumps are fine


def test_a_face_mask_grades_the_holes_on_a_big_part():
    # a surface Face Mask lives within its band (2 mm) of the surface: read from a coarse grid over a
    # 200 mm part it came out up to 0.45 off; read at each centre it is exact
    from engine import facemask
    big = mesh_of(_box(0, 0, 0, 200, 150, 3))
    top = top_face(big)
    mask = facemask.face_mask(big, facemask.cad_faces(big), [top], 30.0, "inside", 2.0, "surface")
    holes = H.place(big, [top], P2.Shape("round", 6), "grid", 15.0, align="x", margin=1.0, size_scale=mask)
    x, y = holes.centre[:, 0], holes.centre[:, 1]
    t = np.clip(np.minimum.reduce([x, 200 - x, y, 150 - y]) / 30.0, 0, 1)   # along the top to its edge
    assert len(holes) > 80 and holes.scale.min() < 0.6 and holes.scale.max() == pytest.approx(1.0)
    assert np.allclose(holes.scale, t * t * (3 - 2 * t), atol=1e-3)


def test_rings_on_a_disc_turn_slots_with_the_ring():
    disc = mesh_of(_cylinder(0, 0, 0, 20, 3))
    holes = H.place(disc, [top_face(disc)], P2.Shape("slot", 1.5, 3), "rings", 4.0, margin=1.0)
    r = np.hypot(holes.centre[:, 0], holes.centre[:, 1])
    assert np.sum(r < 1e-6) == 1                                      # one at the centre
    rings = np.round(r / r[r > 1e-6].min()).astype(int)
    assert set(rings) >= {0, 1, 2, 3}
    outer = r > 1e-6
    # each slot's long axis (x of its turned frame) points away from the centre
    c, s = np.cos(holes.turn), np.sin(holes.turn)
    long_axis = holes.t1 * c[:, None] + holes.t2 * s[:, None]
    radial = holes.centre[:, :2] / np.where(r > 0, r, 1)[:, None]
    assert np.allclose(np.abs(np.einsum("ij,ij->i", long_axis[outer, :2], radial[outer])), 1, atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES[:9] + SHAPES[10:], ids=lambda s: s.kind)
def test_every_shape_cuts_its_own_area(plate, shape):
    holes = H.place(plate, [top_face(plate)], shape, "grid", 2 * shape.circumradius() + 2, align="y",
                    angle=20, margin=1.0)
    assert len(holes) >= 2
    voxel = 0.04
    grid = Grid.block((plate.bounds[0][0], plate.bounds[0][1], 1.5), voxel, (0, 0, 0),
                      (int(40 / voxel) + 1, int(30 / voxel) + 1, 1))
    inside = F.Context(grid).eval_full(H.cutter(holes)) < 0
    assert inside.sum() * voxel ** 2 / len(holes) == pytest.approx(shape.area(), rel=0.03)


def test_hole_pattern_node(tmp_path):
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from engine.graph import Evaluator, Graph, GraphError, link
    writer = STEPControl_Writer()
    writer.Transfer(_cut(_box(0, 0, 0, 40, 30, 20), _box(2, 2, 2, 36, 26, 16)), STEPControl_AsIs)
    writer.Write(str(tmp_path / "box.step"))
    g = Graph()
    g.add("import", "input.import_step", path=str(tmp_path / "box.step"))
    found = Evaluator(g, log=None).run()["import"]["bodies"][0]
    g.add("holes", "pattern.holes", cad=link("import.bodies"), faces=str(top_face(found)), shape="slot",
          size=2.0, length=6.0, layout="grid", spacing=4.0)
    g.add("part", "implicit.from_cad", cad=link("import.bodies"), band=2.0)
    g.add("cut", "implicit.subtract", a=link("part.out"), b=link("holes.out"))
    events = []
    out = Evaluator(g, log=None, on_event=events.append).run()
    assert isinstance(out["cut"]["out"][0], F.Field)
    text = [e["text"] for e in events if e["type"] == "message"]
    assert any(t.startswith("Hole Pattern:") and "slot 2 x 6 mm" in t for t in text)
    g.add("x", "field.coordinate", axis="x")
    g.add("turn", "math.multiply", a=link("x.out"), b=4.0)
    g.set("holes", "rotation", link("turn.out"))                       # a field into Rotation
    events.clear()
    Evaluator(g, log=None, on_event=events.append).run()
    assert any("Rotation from the field" in e["text"] for e in events if e["type"] == "message")
    g.set("holes", "morph_to", "round")                                   # slots morphing into round holes
    g.set("holes", "morph", link("x.out"))
    events.clear()
    Evaluator(g, log=None, on_event=events.append).run()
    text = [e["text"] for e in events if e["type"] == "message"]
    assert any("morphing into round" in t for t in text) and any("Morph from the field" in t for t in text)
    g.set("holes", "faces", "999")
    with pytest.raises(GraphError, match="does not exist"):
        Evaluator(g, log=None).run()
    g.set("holes", "faces", "lid")
    with pytest.raises(GraphError, match="no face selection named"):
        Evaluator(g, log=None).run()


# --- more standard patterns ----------------------------------------------------------------------
def test_alternating_triangles_tile_with_one_web(plate):
    # triangles on a grid, every other one turned 180°: the rows are spaced so the web between rows
    # equals the web between neighbours in a row (sx cos 30° - 1.5 r)
    tri = H.place(plate, [top_face(plate)], P2.Shape("triangle", 4), "grid", 6.0, align="x", margin=1.0,
                  alternate="checker 180")
    web = 6.0 * math.cos(math.pi / 6) - 3.0
    assert len(tri) > 10 and tri.smallest_web == pytest.approx(web, abs=1e-3)
    up = np.isclose(np.cos(tri.turn), 1.0)
    assert 0 < up.sum() < len(tri)                                        # both ways round
    from scipy.spatial import cKDTree
    pairs = cKDTree(tri.centre).query_pairs(6.0 + 1e-6, output_type="ndarray")
    webs = np.minimum(H._web(tri, pairs[:, 0], pairs[:, 1]), H._web(tri, pairs[:, 1], pairs[:, 0]))
    assert np.allclose(webs, web, atol=0.01)                              # every neighbour, both directions


def test_delaunay_cells_are_shrunk_triangles(plate, tmp_path):
    # a staggered layout's Delaunay triangles are equilateral (side 6): moved in by half the 1 mm web,
    # each is 2 x (6 / (2 sqrt 3) - 0.5) across its inscribed circle
    cells = H.place(plate, [top_face(plate)], P2.Shape("delaunay", 1.0), "staggered", 6.0, align="x", margin=1.0)
    assert len(cells) > 20 and cells.cells.shape[1] == 3
    assert np.allclose(cells.widths(), 2 * (6 / (2 * math.sqrt(3)) - 0.5), atol=1e-6)
    assert cells.smallest_web == pytest.approx(1.0, abs=1e-3)
    out = reloaded(cut_mesh(plate, cells, 0.12), tmp_path)
    assert out.is_watertight and genus(out) == len(cells)


def test_tilted_holes_go_straight_through_at_an_angle(tmp_path):
    thick = mesh_of(_box(0, 0, 0, 40, 30, 4))
    holes = H.place(thick, [top_face(thick)], P2.Shape("round", 3), "grid", 8.0, align="x", margin=2.0, tilt=30.0)
    assert np.allclose(holes.normal, [math.sin(math.pi / 6), 0, math.cos(math.pi / 6)], atol=1e-9)
    # along a 30° axis the 4 mm plate is 4 / cos 30° long; the outline's far side reaches the face later
    assert np.allclose(holes.depth, 4 / math.cos(math.pi / 6) + 0.9 * 1.5 * math.tan(math.pi / 6) + H.EXTRA, atol=1e-3)
    assert not any("varies" in t for t, _ in holes.lines)
    out = reloaded(cut_mesh(thick, holes, 0.12), tmp_path)
    assert out.is_watertight and genus(out) == len(holes)
    high, low = section_loops(out, 3.5)[1][1:], section_loops(out, 0.5)[1][1:]
    shift = np.sort(high[:, 0]) - np.sort(low[:, 0])                         # the holes lean towards +x
    assert np.allclose(shift, 3.0 * math.tan(math.pi / 6), atol=0.08)


@pytest.mark.parametrize("edge, at_depth", [("round", 0.6 - math.sqrt(0.6 ** 2 - 0.3 ** 2)), ("chamfer", 0.3)])
def test_rounded_and_chamfered_hole_edges(tmp_path, edge, at_depth):
    # one 4 mm hole with a 0.6 mm edge: 0.3 mm below the face the hole is wider by the fillet (or the
    # chamfer); half way through the plate it is 4 mm
    small = mesh_of(_box(0, 0, 0, 12, 12, 4))
    holes = H.place(small, [top_face(small)], P2.Shape("round", 4), "grid", 20.0, offset=(6.0, 6.0), margin=1.0)
    assert len(holes) == 1
    body = F.subtract_edged(F.body_from_mesh(small, 2.0), H.cutter(holes).regrow(0.6), edge, 0.6)
    grid = Grid.from_bounds(small.bounds, 0.05, 1.0)
    out = reloaded(field_to_mesh(F.Context(grid).eval_full(body), grid), tmp_path)
    assert out.is_watertight and genus(out) == 1
    diameter = lambda z: 2 * math.sqrt(section_loops(out, z)[0][1] / math.pi)
    assert diameter(2.0) == pytest.approx(4.0, abs=0.05)
    assert diameter(4 - 0.3) == pytest.approx(4.0 + 2 * at_depth, abs=0.06)
    assert diameter(0.3) == pytest.approx(4.0 + 2 * at_depth, abs=0.06)   # the far side of the wall too


def test_a_picture_lies_on_the_face_and_wraps_round_ones():
    from engine import picture as PIC
    from PIL import Image, ImageDraw
    image = Image.new("L", (200, 100), 255)
    ImageDraw.Draw(image).rectangle((0, 0, 99, 99), fill=0)                 # the left half black
    darkness = np.asarray(image, dtype=np.float32)
    darkness = 1 - darkness / 255
    plate40 = mesh_of(_box(0, 0, 0, 40, 30, 3))
    place = PIC.Placement(plate40, [top_face(plate40)])
    field = PIC.picture_field(place, darkness, 20.0)                        # 20 x 10 mm, centred at (20, 15)
    assert field.range == (0.0, 1.0) and place.mapping == "flat"
    assert F.sample(field, (15, 15, 3)) == pytest.approx(1.0) and F.sample(field, (25, 15, 3)) == pytest.approx(0.0)
    assert F.sample(field, (15, 15, 0.5)) == 0.0 and F.sample(field, (5, 15, 3)) == 0.0   # beyond Band, off the picture
    turned = PIC.picture_field(PIC.Placement(plate40, [top_face(plate40)], angle=180), darkness, 20.0)
    assert F.sample(turned, (25, 15, 3)) == pytest.approx(1.0)
    tube = mesh_of(_cut(_cylinder(0, 0, 0, 15, 30), _cylinder(0, 0, -1, 12, 32)))
    outer = face_where(tube, lambda c, n: np.hypot(c[:, 0], c[:, 1]) > 14.9)
    around = PIC.Placement(tube, [outer])
    assert around.mapping == "around" and around.radius == pytest.approx(15, abs=0.01)
    wrapped = PIC.picture_field(around, darkness, 20.0)
    # it faces the front (-y), half way up: seen from there, 5 mm to the left (-x) along the
    # circumference is the black half, 5 mm to the right white, and nothing on the back
    mid = 15.0
    left, right = (-15 * math.sin(5 / 15), -15 * math.cos(5 / 15)), (15 * math.sin(5 / 15), -15 * math.cos(5 / 15))
    assert F.sample(wrapped, (*left, mid)) == pytest.approx(1.0) and F.sample(wrapped, (*right, mid)) == pytest.approx(0.0)
    assert F.sample(wrapped, (0.0, 15.0, mid)) == 0.0


def test_text_and_picture_nodes(tmp_path):
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from engine.graph import Evaluator, Graph, GraphError, link
    writer = STEPControl_Writer()
    writer.Transfer(_box(0, 0, 0, 60, 30, 3), STEPControl_AsIs)
    writer.Write(str(tmp_path / "plate.step"))
    g = Graph()
    g.add("import", "input.import_step", path=str(tmp_path / "plate.step"))
    body = Evaluator(g, log=None).run()["import"]["bodies"][0]
    top = str(top_face(body))
    g.add("words", "pattern.text", cad=link("import.bodies"), faces=top, text="HI", font="arialbd", height=10.0)
    g.add("logo", "pattern.picture", cad=link("import.bodies"), faces=top, width=20.0,
          path=str(Path(__file__).resolve().parents[1] / "samples" / "logo.png"))
    out = Evaluator(g, log=None).run()
    words, logo = out["words"]["out"][0], out["logo"]["out"][0]
    assert words.range == (0.0, 1.0) and "mm, flat" in words.note
    xs = np.linspace(20, 40, 81)
    on_letters = [F.sample(words, (x, 15, 3)) for x in xs]
    assert max(on_letters) > 0.9 and min(on_letters) < 0.1                 # strokes and the gaps between
    assert F.sample(logo, (30, 15, 3)) > 0.9                                # the logo's wave crosses its centre
    g.set("words", "font", "no-such-font")
    with pytest.raises(GraphError, match="no font"):
        Evaluator(g, log=None).run()


def test_edges_are_node_inputs():
    from engine.graph.core import REGISTRY
    for kind in ("implicit.subtract", "implicit.union"):
        assert REGISTRY[kind].input("edge").choices == ["sharp", "round", "chamfer"]
    assert REGISTRY["pattern.holes"].input("tilt").default == 0.0
    assert REGISTRY["pattern.surface"].input("alternate").choices == list(P2.ALTERNATE)


# --- Join Faces: one pattern across faces joined smoothly ----------------------------------------
def _fillet(shape, radius, pick):
    """The shape with every edge whose midpoint passes pick(x, y, z) rounded."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    maker = BRepFilletAPI_MakeFillet(shape)
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge(exp.Current())
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        middle = props.CentreOfMass()
        if pick(middle.X(), middle.Y(), middle.Z()):
            maker.Add(radius, edge)
        exp.Next()
    return maker.Shape()


def _halves(shape):
    """The shape with every closed face (a cylinder's side) split into two halves, as many STEP
    exports have them."""
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
    divide = ShapeUpgrade_ShapeDivideClosed(shape)
    divide.SetNbSplitPoints(1)
    divide.Perform()
    return divide.Result()


def test_tangent_faces_share_one_pattern():
    # a box with its top front edge rounded (r 5): the front, the fillet and the top unfold into one
    # flat strip, so the rows run on across both borders 5 mm apart, measured along the surface
    part = mesh_of(_fillet(_box(0, 0, 0, 40, 30, 20), 5.0, lambda x, y, z: abs(y) < 1e-6 and abs(z - 20) < 1e-6))
    chosen = [top_face(part), face_where(part, lambda c, n: (np.abs(c[:, 1]) < 1e-6) & (n[:, 1] < -0.99)),
              face_where(part, lambda c, n: (n[:, 1] < -0.01) & (n[:, 2] > 0.01))]
    joined = H.place(part, chosen, P2.Shape("round", 2), "grid", 5.0, align="x", margin=1.0)
    apart = H.place(part, chosen, P2.Shape("round", 2), "grid", 5.0, align="x", margin=1.0, join="no")
    assert any("unfolded into one sheet: the spacing varies by less than 1%" in t for t, _ in joined.lines)

    def along(c):  # mm along the surface from the bottom front edge
        return np.where(c[:, 1] < 1e-6, c[:, 2], np.where(c[:, 2] > 20 - 1e-6, 15 + 2.5 * math.pi + c[:, 1] - 5,
                                                          15 + 5 * np.arctan2(c[:, 2] - 15, 5 - c[:, 1])))

    def off_rows(c):
        d = (along(c) - along(c)[0]) % 5.0
        return np.minimum(d, 5.0 - d).max()
    rows = np.unique(np.round(along(joined.centre), 1))
    assert len(rows) == 9 and off_rows(joined.centre) < 0.01      # 3 rows on the front, 1 on the fillet, 5 on top
    assert off_rows(apart.centre) > 1.0                             # a pattern per face: out of step at the borders
    assert joined.smallest_web > 2.5 and apart.smallest_web < 1.0
    # the largest face (the top) keeps the holes it has without joining
    from scipy.spatial import cKDTree
    top_joined = joined.centre[joined.centre[:, 2] > 20 - 1e-6]
    top_apart = apart.centre[apart.centre[:, 2] > 20 - 1e-6]
    assert len(top_joined) == len(top_apart) == 35 and cKDTree(top_joined).query(top_apart)[0].max() < 1e-5


def test_folds_cut_the_sheet_open():
    # all four top edges rounded: the top, the fillets and the four sides are one sheet, cut open at
    # the box's vertical edges (90° folds), so it unfolds with hardly any stretch
    box = mesh_of(_fillet(_box(0, 0, 0, 40, 30, 20), 5.0, lambda x, y, z: abs(z - 20) < 1e-6))
    ids = box.face_attributes["cad_face"]
    upper = np.unique(ids[box.triangles_center[:, 2] > 1e-6]).tolist()
    holes = H.place(box, upper, P2.Shape("round", 2), "grid", 5.0, align="x", margin=1.0)
    assert len(upper) == 9 and len(holes) > 100 and holes.smallest_web > 1.5
    assert any("unfolded into one sheet: the spacing varies by less than 1%" in t for t, _ in holes.lines)
    # a plain box's top and sides only meet at folds: a pattern per face, as without joining
    plain = mesh_of(_box(0, 0, 0, 40, 30, 20))
    ids = plain.face_attributes["cad_face"]
    upper = np.unique(ids[plain.triangles_center[:, 2] > 1e-6]).tolist()
    joined = H.place(plain, upper, P2.Shape("round", 2), "grid", 5.0, align="x", margin=1.0)
    apart = H.place(plain, upper, P2.Shape("round", 2), "grid", 5.0, align="x", margin=1.0, join="no")
    assert np.array_equal(joined.centre, apart.centre) and not any("unfolded" in t for t, _ in joined.lines)


def test_halves_of_a_tube_keep_a_pattern_each():
    # the halves of a cylinder can't be unfolded flat: each keeps its own pattern, which closes around
    tube = mesh_of(_halves(_cut(_cylinder(0, 0, 0, 15, 30), _cylinder(0, 0, -1, 12, 32))))
    ids = tube.face_attributes["cad_face"]
    outer = np.unique(ids[np.hypot(tube.triangles_center[:, 0], tube.triangles_center[:, 1]) > 14.9]).tolist()
    joined = H.place(tube, outer, P2.Shape("round", 3), "staggered", 5.0, margin=2.0)
    apart = H.place(tube, outer, P2.Shape("round", 3), "staggered", 5.0, margin=2.0, join="no")
    assert len(outer) == 2 and len(joined) > 100
    assert any("close around (like the halves of a cylinder)" in t for t, _ in joined.lines)
    assert np.array_equal(joined.centre, apart.centre)
    # a hole in a flat sheet is no reason not to unfold it
    from engine import unfold
    part = mesh_of(_cut(_box(0, 0, 0, 60, 40, 3), _cylinder(30, 20, -1, 8.7, 5)))
    top = part.face_attributes["cad_face"] == top_face(part)
    (_, local, source), = unfold.sheets(part.vertices, part.faces, part.face_attributes["cad_face"],
                                         np.flatnonzero(top), H.FOLD)
    assert unfold.closes_around(np.asarray(part.vertices)[source], local) is None


def test_join_faces_is_a_node_input():
    from engine.graph.core import REGISTRY
    for kind in ("pattern.holes", "pattern.surface"):
        port = REGISTRY[kind].input("join")
        assert port.default == "tangent" and port.choices == list(H.JOIN_FACES)
