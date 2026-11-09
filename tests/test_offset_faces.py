import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from engine import fields as F
from engine import offset_faces as OF
from engine import tiles
from engine.checks import mesh_report
from engine.graph import Evaluator, Graph, GraphError, link
from engine.grid import Grid
from engine.meshing import field_to_mesh, signed_volume

TEST_STEP = Path(__file__).resolve().parents[1] / "samples" / "test_part.step"
SIDES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
TOP, PAD = 4, 6


def cad_box(extents=(20, 20, 20)):
    """A box whose triangles carry CAD face numbers like an imported body: one face per side."""
    box = trimesh.creation.box(extents)
    box.face_attributes["cad_face"] = np.argmax(box.face_normals @ SIDES.T, axis=1).astype(np.int32)
    return box


def pad_box(half=10.0, pad=4.0):
    """A 20 mm box whose top has the square |x|, |y| <= pad in its middle as a CAD face of its own
    (PAD); the rest of the top is face TOP, the sides 0-3, the bottom 5."""
    h, p = half, pad
    V = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h), (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
         (-p, -p, h), (p, -p, h), (p, p, h), (-p, p, h)]
    Fc = [(0, 2, 1), (0, 3, 2), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
          (4, 5, 9), (4, 9, 8), (5, 6, 10), (5, 10, 9), (6, 7, 11), (6, 11, 10), (7, 4, 8), (7, 8, 11),
          (8, 9, 10), (8, 10, 11)]
    mesh = trimesh.Trimesh(np.array(V, dtype=float), np.array(Fc), process=False)
    ids = np.argmax(mesh.face_normals @ SIDES.T, axis=1).astype(np.int32)
    ids[-2:] = PAD
    mesh.face_attributes["cad_face"] = ids
    return mesh


def half_box(half=10.0):
    """A 20 mm box whose top is split at x = 0: the half x > 0 is face PAD, the other half TOP."""
    h = half
    V = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h), (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
         (0, -h, h), (0, h, h)]
    Fc = [(0, 2, 1), (0, 3, 2), (0, 1, 5), (0, 5, 8), (0, 8, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 9), (2, 9, 6),
          (3, 0, 4), (3, 4, 7), (4, 8, 9), (4, 9, 7), (8, 5, 6), (8, 6, 9)]
    mesh = trimesh.Trimesh(np.array(V, dtype=float), np.array(Fc), process=False)
    ids = np.argmax(mesh.face_normals @ SIDES.T, axis=1).astype(np.int32)
    ids[-2:] = PAD
    mesh.face_attributes["cad_face"] = ids
    return mesh


def moved(mesh, faces, distance, low=None, high=None, invert=False, **kw):
    patch = OF.Patch(mesh, mesh.face_attributes["cad_face"], faces, invert)
    plain = not isinstance(distance, F.Field)
    low = float(distance) if plain and low is None else low
    high = float(distance) if plain and high is None else high
    caps = OF.walls(mesh, patch, max(high, 0.0), max(-low, 0.0))
    return OF.offset_faces(F.body_from_mesh(mesh, 3.0), patch, distance, low, high, caps=caps, **kw)


def crossing(field, start, end, steps=40):
    """Where the field changes sign between start (inside) and end (outside), by bisection."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    fa, fb = F.sample(field, a), F.sample(field, b)
    assert fa < 0 < fb, f"no surface between {start} ({fa:.4f}) and {end} ({fb:.4f})"
    for _ in range(steps):
        m = (a + b) / 2
        if F.sample(field, m) < 0:
            a = m
        else:
            b = m
    return (a + b) / 2


def height_at(field, x, y=0.0, lo=5.0, hi=16.0):
    return crossing(field, (x, y, lo), (x, y, hi))[2]


def side_at(field, z, y=0.0, lo=0.0, hi=16.0):
    return crossing(field, (lo, y, z), (hi, y, z))[0]


# --- 1. a box's top raised and lowered: the taller and shorter box, no bulge at its edges ----------
@pytest.mark.parametrize("distance", [2.0, -2.0])
def test_box_top_moves_to_the_taller_or_shorter_box(distance):
    box = cad_box()
    field = moved(box, [TOP], distance, foot="round", foot_size=1.0, edge="sharp")
    top = 10 + distance
    exact = F.box((0, 0, distance / 2), (20, 20, 20 + distance))
    for x in (0.0, 5.0, 9.5, 9.95):
        assert height_at(field, x, 0.3 * x) == pytest.approx(top, abs=1e-4)
    # the sides run straight on to the new top: no bulge where the top's edge was, with Foot round
    for z in (top - 1.5, top - 0.3, top - 0.05, 9.99, 10.0, 10.01, 5.0):
        if z < top:
            assert side_at(field, z) == pytest.approx(10.0, abs=1e-4)
    # outside, near the surface, the distances are the box's own; above a top that was cut down they are a
    # lower bound (the old box's sides and the cut's, which continue them, still count), as with Boolean
    # Subtract
    grid = Grid((-12, -12, 6), (12, 12, 14), 0.25)
    values = F.Context(grid).eval_full(field)
    ref = F.Context(grid).eval_full(exact)
    removed = (grid.coords()[2] > top) & (distance < 0)
    near = (ref > 0.02) & (ref < 1.0) & ~removed
    assert near.sum() > 1000
    assert np.abs(values[near] - ref[near]).max() < 2e-3
    assert np.all(np.sign(values[np.abs(ref) > 0.02]) == np.sign(ref[np.abs(ref) > 0.02]))
    assert np.all(values[removed] <= ref[removed] + 2e-3)


def test_corners_of_a_raised_top_are_square_and_round_sides_overhang():
    box = cad_box()
    field = moved(box, [TOP], 2.0)
    corner = crossing(field, (9, 9, 11), (11, 11, 11))
    assert corner[:2] == pytest.approx([10, 10], abs=1e-4)
    rounded = moved(box, [TOP], 2.0, sides="round")
    # the faces' own offset: the top 2 mm up, its border a quarter round reaching 2 mm out
    assert height_at(rounded, 0.0) == pytest.approx(12.0, abs=1e-4)
    assert side_at(rounded, 10.0) == pytest.approx(12.0, abs=1e-4)
    assert side_at(rounded, 11.0) == pytest.approx(10 + math.sqrt(3.0), abs=1e-4)


# --- 2. a pad in the middle of the top: a step with vertical walls, Edge and Foot shaped ------------
def test_raised_pad_makes_a_step_with_straight_walls():
    mesh = pad_box()
    field = moved(mesh, [PAD], 2.0)
    assert height_at(field, 0.0) == pytest.approx(12.0, abs=1e-4)
    assert height_at(field, 3.9) == pytest.approx(12.0, abs=1e-4)
    assert height_at(field, 4.1) == pytest.approx(10.0, abs=1e-4)   # the face around it stays
    assert height_at(field, 9.0) == pytest.approx(10.0, abs=1e-4)
    for z in (10.05, 11.0, 11.95):
        assert side_at(field, z) == pytest.approx(4.0, abs=1e-4)


@pytest.mark.parametrize("edge, size, x, z", [
    ("chamfer", 0.5, 3.75, 11.75),                         # 45°, 0.5 mm along both faces
    ("round", 1.0, 3.5, 11 + math.sqrt(0.75)),             # a quarter circle round (3, 11)
])
def test_edge_shapes_only_the_rim(edge, size, x, z):
    field = moved(pad_box(), [PAD], 2.0, edge=edge, edge_size=size)
    assert height_at(field, x) == pytest.approx(z, abs=1e-4)
    assert height_at(field, 4 - size - 0.1) == pytest.approx(12.0, abs=1e-4)  # the top beyond it
    assert side_at(field, 12 - size - 0.1) == pytest.approx(4.0, abs=1e-4)    # the wall below it
    assert height_at(field, 4.05) == pytest.approx(10.0, abs=1e-4)            # no foot
    assert side_at(field, 10.05) == pytest.approx(4.0, abs=1e-4)


@pytest.mark.parametrize("foot, size, x, z", [
    ("chamfer", 0.5, 4.25, 10.25),
    ("round", 1.0, 4.5, 11 - math.sqrt(0.75)),             # a quarter circle round (5, 11)
])
def test_foot_shapes_only_the_inside_corner(foot, size, x, z):
    field = moved(pad_box(), [PAD], 2.0, foot=foot, foot_size=size)
    assert height_at(field, x) == pytest.approx(z, abs=1e-4)
    assert height_at(field, 4 + size + 0.1) == pytest.approx(10.0, abs=1e-4)
    assert side_at(field, 10 + size + 0.1) == pytest.approx(4.0, abs=1e-4)
    assert height_at(field, 3.95) == pytest.approx(12.0, abs=1e-4)            # a sharp rim
    assert side_at(field, 11.95) == pytest.approx(4.0, abs=1e-4)


def bisector_depth(field, corner, inward):
    """How far along the corner's bisector the surface lies from the corner (x, z plane, y = 0)."""
    c, d = np.array([corner[0], 0.0, corner[1]]), np.array([inward[0], 0.0, inward[1]]) / np.hypot(*inward)
    inside, outside = (c + 2 * d, c - 0.5 * d) if F.sample(field, c) > 0 else (c - 0.5 * d, c + 2 * d)
    return float(np.linalg.norm(crossing(field, inside, outside) - c))


def test_smooth_is_as_deep_as_round_and_reaches_farther():
    mesh = pad_box()
    for which, corner, inward in (("edge", (4, 12), (-1, -1)), ("foot", (4, 10), (1, 1))):
        round_ = moved(mesh, [PAD], 2.0, **{which: "round", f"{which}_size": 1.0})
        smooth = moved(mesh, [PAD], 2.0, **{which: "smooth", f"{which}_size": 1.0})
        assert bisector_depth(round_, corner, inward) == pytest.approx(math.sqrt(2) - 1, abs=1e-4)
        assert bisector_depth(smooth, corner, inward) == pytest.approx(math.sqrt(2) - 1, abs=1e-4)
    # beyond round's 1 mm the smooth edge still bends the top a little
    smooth = moved(mesh, [PAD], 2.0, edge="smooth", edge_size=1.0)
    round_ = moved(mesh, [PAD], 2.0, edge="round", edge_size=1.0)
    assert height_at(round_, 2.9) == pytest.approx(12.0, abs=1e-5)
    assert 11.98 < height_at(smooth, 2.9) < 12.0 - 1e-4
    assert height_at(smooth, 4 - OF.SMOOTH_SIZE - 0.05) == pytest.approx(12.0, abs=1e-5)


def test_foot_runs_out_where_the_border_meets_a_face_that_folds_away():
    # half the top raised: along x = 0 the face beside it is level (a foot), along the box's edges the
    # sides run on up (none); where the two meet the fillet runs out instead of bulging past the side
    field = moved(half_box(), [PAD], 2.0, foot="round", foot_size=1.0)
    for y in (0.0, 5.0, -8.0):
        assert height_at(field, -0.5, y) == pytest.approx(11 - math.sqrt(0.75), abs=1e-4)
    assert side_at(field, 10.5, lo=5.0) == pytest.approx(10.0, abs=1e-4)               # no foot on the +x side
    assert crossing(field, (5, 0, 10.5), (5, 13, 10.5))[1] == pytest.approx(10.0, abs=1e-4)
    grid = Grid((-1.5, 10.02, 10.02), (0.5, 11.5, 11.5), 0.05)                         # past the side: air
    assert (F.Context(grid).eval_full(field) > 0).all()
    fillet = [crossing(field, (-0.05, 0, 10.05), (-0.05, 12, 10.05))[1], crossing(field, (-0.05, 0, 10.05),
                                                                                    (-0.05, -12, 10.05))[1]]
    assert 10 - OF.RUNOUT < fillet[0] < 10.0 and -10.0 < fillet[1] < -10 + OF.RUNOUT


def test_round_sides_sunk_make_a_cove():
    field = moved(pad_box(), [PAD], -1.0, sides="round", edge="round", edge_size=0.3)
    assert height_at(field, 0.0) == pytest.approx(9.0, abs=1e-4)
    assert height_at(field, 4.5) == pytest.approx(10 - math.sqrt(0.75), abs=1e-4)   # a quarter round (4, 10)
    assert height_at(field, 5.5) == pytest.approx(10.0, abs=1e-4)
    assert height_at(field, 4.9) == pytest.approx(10 - math.sqrt(0.19), abs=1e-4)
    assert 9.8 < height_at(field, 5.1) < 9.99                                        # the rim (5, 10), rounded
    assert height_at(field, 5.35) == pytest.approx(10.0, abs=1e-4)


def test_sizes_are_cut_to_the_step():
    field = moved(pad_box(), [PAD], 0.5, edge="round", edge_size=2.0, foot="round", foot_size=2.0)
    assert height_at(field, 0.0) == pytest.approx(10.5, abs=1e-4)     # the top is not lifted by the fillets
    assert height_at(field, 3.0) == pytest.approx(10.5, abs=1e-4)
    assert height_at(field, 5.0) == pytest.approx(10.0, abs=1e-4)


# --- 3. the pad sunk: a pocket ---------------------------------------------------------------------
def test_sunk_pad_makes_a_pocket():
    mesh = pad_box()
    field = moved(mesh, [PAD], -2.0)
    assert height_at(field, 0.0) == pytest.approx(8.0, abs=1e-4)
    assert height_at(field, 4.1) == pytest.approx(10.0, abs=1e-4)
    assert crossing(field, (5, 0, 9), (0, 0, 9))[0] == pytest.approx(4.0, abs=1e-4)
    shaped = moved(mesh, [PAD], -2.0, edge="round", edge_size=0.5, foot="round", foot_size=0.5)
    assert height_at(shaped, 4.25) == pytest.approx(9.5 + math.sqrt(0.1875), abs=1e-4)   # the rim, round (4.5, 9.5)
    assert height_at(shaped, 3.75) == pytest.approx(8.5 - math.sqrt(0.1875), abs=1e-4)   # the floor, round (3.5, 8.5)
    assert height_at(shaped, 2.0) == pytest.approx(8.0, abs=1e-4)
    assert crossing(shaped, (5, 0, 9), (0, 0, 9))[0] == pytest.approx(4.0, abs=1e-4)


def test_lowered_box_top_keeps_a_clean_rim_with_foot_and_edge():
    # the walls continue the box's sides: no foot (it left a knife-thin lip), the edge rounds the new rim
    field = moved(cad_box(), [TOP], -2.0, edge="round", edge_size=0.5, foot="round", foot_size=0.5)
    assert height_at(field, 0.0) == pytest.approx(8.0, abs=1e-4)
    assert height_at(field, 9.75) == pytest.approx(7.5 + math.sqrt(0.1875), abs=1e-4)
    assert side_at(field, 7.0) == pytest.approx(10.0, abs=1e-4)
    grid = Grid((8, -1, 7), (11, 1, 11), 0.05)
    assert (F.Context(grid).eval_full(field)[:, :, grid.axes()[2] > 8.01] > 0).all()   # nothing above the rim


# --- 4. a field distance ---------------------------------------------------------------------------
def test_a_field_distance_varies_the_height():
    mesh = pad_box()
    ramp = 1.0 + 0.1 * F.X
    field = moved(mesh, [PAD], ramp, low=-2.0, high=2.0)
    for x in (-3.0, 0.0, 3.0):
        assert height_at(field, x) == pytest.approx(11 + 0.1 * x, abs=2e-3)
    # positive on one side and negative on the other: raised there, a pocket here, no crack between
    tilt = 0.5 * F.X
    field = moved(mesh, [PAD], tilt, low=-2.0, high=2.0, foot="round", foot_size=0.5)
    assert height_at(field, 2.0) == pytest.approx(11.0, abs=2e-3)
    assert height_at(field, -2.0) == pytest.approx(9.0, abs=2e-3)
    assert height_at(field, 0.3) == pytest.approx(10.15, abs=2e-3)
    grid = Grid((0.1, -3, 10.0), (3.9, 3, 10.0), 0.05)     # the old face's level under the raised half
    assert (F.Context(grid).eval_full(field) < 0).all()


# --- 5. all other faces ----------------------------------------------------------------------------
def test_all_other_faces_move_and_the_picked_one_stays():
    field = moved(pad_box(), [PAD], 1.0, invert=True)
    assert height_at(field, 0.0) == pytest.approx(10.0, abs=1e-4)      # the pad stays: now a pocket
    assert height_at(field, 7.0) == pytest.approx(11.0, abs=1e-4)
    assert side_at(field, 0.0, hi=14) == pytest.approx(11.0, abs=1e-4)
    assert crossing(field, (0, 0, 0), (0, 0, -14))[2] == pytest.approx(-11.0, abs=1e-4)
    assert crossing(field, (6, 0, 10.5), (0, 0, 10.5))[0] == pytest.approx(4.0, abs=1e-4)


# --- 6. blocks: the same values and the same mesh as the whole grid --------------------------------
def test_blocks_compute_the_same_values_as_the_whole_grid():
    field = moved(pad_box(), [PAD], 1.5, edge="chamfer", edge_size=0.4, foot="smooth", foot_size=0.5)
    whole = Grid.from_bounds(field.bounds, 0.25, 1.5)
    ref = F.Context(whole).eval_full(field)[24:73, 24:73, 48:97]
    block = F.Context(Grid.block(whole.base, 0.25, (24, 24, 48), (49, 49, 49))).eval_full(field)
    near = np.abs(ref) < 1.0
    assert near.sum() > 1000
    assert np.array_equal(block[near], ref[near])


def test_the_field_travels_to_the_workers():
    import cloudpickle
    field = moved(pad_box(), [PAD], 1.5, edge="round", edge_size=0.5, foot="smooth", foot_size=0.5)
    copy = cloudpickle.loads(cloudpickle.dumps(field))
    grid = Grid((-6, -6, 8), (6, 6, 13), 0.25)
    assert np.array_equal(F.Context(grid).eval_full(copy), F.Context(grid).eval_full(field))


def test_tiled_mesh_matches_the_whole_grid(monkeypatch):
    monkeypatch.setattr(tiles, "BLOCK", 48)
    monkeypatch.setattr(tiles, "TILE_MIN_VOXELS", 0)
    field = moved(pad_box(), [PAD], 1.5, edge="round", edge_size=0.5, foot="round", foot_size=0.5)
    grid = Grid.from_bounds(field.bounds, 0.25, 1.5)
    whole = field_to_mesh(F.Context(grid).eval_full(field), grid)
    mesh, computed, _ = tiles.mesh_tiled(field, field.bounds, 0.25, 1.5, 1.0,
                                         lambda fn, args, **_: [fn(**a) for a in args])
    assert computed > 1 and len(mesh.faces) == len(whole.faces)
    assert signed_volume(mesh.vertices, mesh.faces) == pytest.approx(signed_volume(whole.vertices, whole.faces),
                                                                     rel=1e-6)


# --- meshes: closed, one piece, the volume they should have ----------------------------------------
@pytest.mark.parametrize("distance, kw, volume", [
    (2.0, {}, 8000 + 64 * 2),
    (-2.0, {}, 8000 - 64 * 2),
    (2.0, {"edge": "chamfer", "edge_size": 0.5, "foot": "round", "foot_size": 1.0},
     8000 + 64 * 2 - 4 * 8 * 0.125 + 4 * 8 * (1 - math.pi / 4)),
])
def test_meshes_are_closed_and_hold_the_right_volume(distance, kw, volume):
    field = moved(pad_box(), [PAD], distance, **kw)
    grid = Grid.from_bounds(field.bounds, 0.2, 1.0)
    mesh = field_to_mesh(F.Context(grid).eval_full(field), grid)
    report = mesh_report(mesh.vertices, mesh.faces)
    assert report["open_edges"] == 0 and report["bodies"] == 1 and report["cavities"] == 0
    assert signed_volume(mesh.vertices, mesh.faces) == pytest.approx(volume, rel=4e-3)


@pytest.fixture(scope="module")
def part():
    from engine.cad_import import load_step, tessellate
    return tessellate(load_step(TEST_STEP)[0])


@pytest.mark.parametrize("distance, kw", [
    (1.5, {"edge": "chamfer", "edge_size": 0.4, "foot": "smooth", "foot_size": 0.6}),
    (-1.0, {"edge": "round", "edge_size": 0.4, "foot": "round", "foot_size": 0.4}),
    (1.0, {"sides": "round", "foot": "round", "foot_size": 0.5}),
], ids=["raised", "sunk", "round"])
def test_the_top_of_a_part_with_fillets_and_a_hole_meshes_closed(part, distance, kw):
    # face 10: the top, fillets all round (they run on tangentially: a step with a foot), a hole in the
    # middle (its wall runs on up or down: no foot there, no sheet of zeros along it)
    field = moved(part, [10], distance, **kw)
    grid = Grid.from_bounds(field.bounds, 0.25, 1.0)
    mesh = field_to_mesh(F.Context(grid).eval_full(field), grid)
    report = mesh_report(mesh.vertices, mesh.faces)
    assert report["open_edges"] == 0 and report["bodies"] == 1 and report["cavities"] == 0
    before, after = signed_volume(part.vertices, part.faces), signed_volume(mesh.vertices, mesh.faces)
    if kw.get("sides") == "round":  # the top's area 1 mm up, and less than a half round along its border
        assert before + 1635.02 < after < before + 1635.02 + (176 + 16 * math.pi) * math.pi / 2
    else:
        assert after == pytest.approx(before + 1635.02 * distance, rel=2e-3)


# --- textures and holes on moved faces -------------------------------------------------------------
def test_a_face_mask_finds_the_moved_face_when_its_band_reaches_it():
    from engine import facemask
    mesh = pad_box()
    ids = mesh.face_attributes["cad_face"]
    top = (0.0, 0.0, 12.0)                                   # the pad's top, raised 2 mm
    assert F.sample(facemask.face_mask(mesh, ids, [PAD], 1.0, band=3.0), top) == pytest.approx(1.0)
    assert F.sample(facemask.face_mask(mesh, ids, [PAD], 1.0, band=1.5), top) == 0.0


def test_holes_start_from_the_moved_surface():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from engine import holes as H
    from engine import pattern2d as P2
    from engine.cad_import import tessellate
    plate = tessellate(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 40, 30, 10).Shape(), 0.01, 0.5)
    ids = plate.face_attributes["cad_face"]
    top = int(np.bincount(ids[(plate.triangles_center[:, 2] > 9.99) & (plate.face_normals[:, 2] > 0.99)]).argmax())
    shape = P2.Shape("round", 4)

    def pattern(moved_by, **kw):
        return H.place(plate, [top], shape, "grid", 10.0, align="x", margin=2.0, moved=moved_by, **kw)

    for d in (2.0, -1.0):
        part = moved(plate, [top], d)
        holes = pattern(d)
        assert len(holes) == 6
        x, y, _ = holes.centre[0]
        through = F.subtract(part, H.cutter(holes))
        assert all(F.sample(through, (x, y, z)) > 0 for z in np.linspace(-0.4, 10 + d + 0.4, 25))
        pockets = F.subtract(part, H.cutter(pattern(d, depth=1.0)))        # 1 mm deep from the moved face
        assert crossing(pockets, (x, y, 0.5), (x, y, 15))[2] == pytest.approx(10 + d - 1, abs=1e-3)
        studs = F.union(part, H.cutter(pattern(d, height=1.0)))            # 1 mm tall on it
        assert crossing(studs, (x, y, 0.5), (x, y, 15))[2] == pytest.approx(10 + d + 1, abs=1e-3)
    # without it a raised face plugs the holes' tops
    plugged = F.subtract(moved(plate, [top], 2.0), H.cutter(pattern(0.0)))
    assert F.sample(plugged, (x, y, 11.5)) < 0


# --- the node --------------------------------------------------------------------------------------
def test_offset_faces_node_on_a_step_file():
    g = Graph()
    g.add("import", "input.import_step", path=str(TEST_STEP))
    g.add("part", "implicit.from_cad", cad=link("import.bodies"), band=3.0)
    g.add("raise", "implicit.offset_faces", body=link("part.out"), cad=link("import.bodies"), faces="10",
          distance=1.5, edge="chamfer", edge_size=0.4, foot="smooth", foot_size=0.6)
    out = Evaluator(g, log=None).run()["raise"]["out"]
    assert len(out) == 1
    field = out[0]
    assert "1 face raised 1.5 mm" in field.note
    assert "no foot along" in field.note                 # the hole's wall runs on up through the top
    assert height_at(field, 15.0, 10.0, lo=10.0, hi=20.0) == pytest.approx(16.5, abs=1e-3)
    # the 16 mm hole in the middle stays open, its wall straight up to the new top
    assert crossing(field, (30 + 9.0, 20, 16.0), (30 + 7.0, 20, 16.0))[0] == pytest.approx(38.0, abs=0.02)
    g.set("raise", "faces", "999")
    with pytest.raises(GraphError, match="does not exist"):
        Evaluator(g, log=None).run()
    g.set("raise", "faces", "10")
    g.add("x", "field.coordinate", axis="x")
    g.set("raise", "distance", link("x.out"))
    with pytest.raises(GraphError, match="Max distance"):
        Evaluator(g, log=None).run()


def test_blend_sides_are_a_soft_ramp():
    g = Graph()
    g.add("import", "input.import_step", path=str(TEST_STEP))
    g.add("part", "implicit.from_cad", cad=link("import.bodies"), band=3.0)
    g.add("raise", "implicit.offset_faces", body=link("part.out"), cad=link("import.bodies"), faces="10",
          distance=1.0, sides="blend", width=3.0)
    field = Evaluator(g, log=None).run()["raise"]["out"][0]
    assert height_at(field, 15.0, 15.0, lo=10.0, hi=20.0) == pytest.approx(16.0, abs=1e-3)
    assert 15.0 < height_at(field, 4.5, 15.0, lo=10.0, hi=20.0) < 16.0      # on the ramp
