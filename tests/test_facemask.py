from pathlib import Path

import numpy as np
import pytest
import trimesh

from engine import facemask
from engine import fields as F
from engine.graph import Evaluator, Graph, GraphError, link
from engine.grid import Grid
from engine.transform import scale_about

TEST_STEP = Path(__file__).resolve().parents[1] / "samples" / "test_part.step"
SIDES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
TOP = 4


def cad_box(extents):
    """A box whose triangles carry CAD face numbers like an imported body: one face per side."""
    box = trimesh.creation.box(extents)
    box.face_attributes["cad_face"] = np.argmax(box.face_normals @ SIDES.T, axis=1).astype(np.int32)
    return box


def mask_at(mesh, points, faces=(TOP,), fade=2.0, position="inside"):
    field = facemask.face_mask(mesh, facemask.cad_faces(mesh), faces, fade, position)
    return np.array([F.sample(field, p) for p in points])


def test_import_keeps_cad_faces():
    from engine.cad_import import load_step, tessellate
    body = load_step(TEST_STEP)[0]
    mesh = tessellate(body)
    ids = facemask.cad_faces(mesh)
    assert ids is not None and len(ids) == len(mesh.faces)
    assert ids.min() == 0 and len(np.unique(ids)) > 10
    scaled = scale_about(mesh, np.zeros(3), 1.2)
    assert np.array_equal(facemask.cad_faces(scaled), ids)


@pytest.mark.parametrize("position, edge_inside, edge_outside", [
    ("inside", (0.0, 0.5), 0.0),       # fades in over 2 mm inside the top face; the sides stay 0
    ("center", (0.4, 0.95), (0.05, 0.6)),
    ("outside", 1.0, (0.2, 0.9)),      # top face fully on; spills 2 mm down the sides
])
def test_fade_position(position, edge_inside, edge_outside):
    box = cad_box((20, 20, 20))
    middle, inside, outside = mask_at(box, [(0, 0, 10.05), (9.5, 0, 10.05), (10.05, 0, 9.5)], position=position)
    assert middle == pytest.approx(1.0)
    assert mask_at(box, [(10.05, 0, 0)], position=position)[0] == pytest.approx(0.0)  # middle of a side face
    for value, expected in ((inside, edge_inside), (outside, edge_outside)):
        if isinstance(expected, tuple):
            assert expected[0] <= value <= expected[1]
        else:
            assert value == pytest.approx(expected, abs=1e-6)


def test_thin_wall_far_side_is_not_selected():
    # 1 mm plate, top chosen: the top's middle is 10 mm from the selection's border, so fully on
    # even with a 2 mm fade; just under the plate (1 mm from the top face) is off
    plate = cad_box((20, 20, 1))
    top, under = mask_at(plate, [(0, 0, 0.55), (0, 0, -0.55)])
    assert top == pytest.approx(1.0) and under == pytest.approx(0.0)


def test_hard_edge_and_no_faces():
    box = cad_box((20, 20, 20))
    assert mask_at(box, [(9.9, 0, 10.05), (10.05, 0, 9.9)], fade=0) == pytest.approx([1.0, 0.0])
    assert mask_at(box, [(0, 0, 10.05)], faces=())[0] == 0.0


def test_blocks_compute_the_same_mask_as_the_whole_grid():
    box = cad_box((20, 20, 20))
    field = facemask.face_mask(box, facemask.cad_faces(box), [TOP, 0], 3.0, "center")
    whole = Grid.from_bounds(box.bounds, 0.25, 2.0)
    ref = F.Context(whole).eval_full(field)[24:73, 24:73, 48:97]
    block = F.Context(Grid.block(whole.base, 0.25, (24, 24, 48), (49, 49, 49))).eval_full(field)
    assert (ref > 0).sum() > 100 and np.array_equal(block, ref)


def test_smoothly_joined_faces_are_grouped():
    box = cad_box((20, 20, 20))
    assert len(set(facemask.tangent_groups(box, facemask.cad_faces(box)))) == 6  # 90° edges: all apart
    from engine.cad_import import load_step, tessellate
    part = tessellate(load_step(TEST_STEP)[0])
    ids = facemask.cad_faces(part)
    groups = facemask.tangent_groups(part, ids)
    assert len(set(groups)) < len(np.unique(ids))  # its fillets join their neighbours


def test_face_mask_node_maps_over_bodies_and_checks_faces(tmp_path):
    g = Graph()
    g.add("import", "input.import_step", path=str(TEST_STEP))
    g.add("mask", "field.face_mask", cad=link("import.bodies"), faces="0, 1", fade=1.0)
    out = Evaluator(g, log=None).run()["mask"]["out"]
    assert len(out) == 1 and isinstance(out[0], F.Field)
    g.set("mask", "faces", "9999")
    with pytest.raises(GraphError, match="does not exist"):
        Evaluator(g, log=None).run()


def volume_mask(box, faces=(TOP,), fade=10.0, position="inside"):
    return facemask.face_mask(box, facemask.cad_faces(box), faces, fade, position, reach="volume")


def test_volume_reach_fades_with_distance_from_the_chosen_faces():
    # 20 mm box, top chosen, fade 10 mm: 1 at the top, 0.5 at 5 mm, 0 from 10 mm on
    box = cad_box((20, 20, 20))
    field = volume_mask(box)
    assert F.sample(field, (0, 0, 10)) == pytest.approx(1.0)
    assert F.sample(field, (0, 0, 5)) == pytest.approx(0.5)
    assert F.sample(field, (9, 0, 7)) == pytest.approx(1 - 0.3 ** 2 * (3 - 0.6))  # the side nearby doesn't matter
    assert F.sample(field, (0, 0, -1)) == pytest.approx(0.0)
    # a chosen side of a flat board reaches as far as Fade says (the old version stopped halfway
    # to the top and bottom faces)
    board = cad_box((200, 200, 30))
    side = volume_mask(board, faces=(0,), fade=50.0)  # face 0 = the +x side
    assert F.sample(side, (100 - 25, 0, 0)) == pytest.approx(0.5, abs=1e-3)


def test_volume_reach_is_smooth_and_the_same_in_blocks():
    box = cad_box((20, 20, 20))
    field = volume_mask(box, fade=8.0)
    whole = Grid.block((-12, -12, -12), 0.25, (0, 0, 0), (97, 97, 97))
    values = F.Context(whole).eval_full(field)
    column = values[48, 48, 8:89]                               # down through the middle, inside the box
    assert np.all(np.diff(column) >= -1e-6)                     # rises steadily towards the top
    assert np.abs(np.diff(column)).max() < 0.1
    block = F.Context(Grid.block((-12, -12, -12), 0.25, (24, 36, 48), (49, 49, 49))).eval_full(field)
    assert np.array_equal(block, values[24:73, 36:85, 48:97])


def test_volume_mask_fades_a_lattice():
    from engine import lattice
    box = cad_box((20, 20, 20))
    struts = lattice.struts("cubic", 5.0, volume_mask(box) * 1.2)
    assert F.sample(struts, (0.3, 0, 7.5)) < 0    # 0.3 mm from a strut's axis near the chosen top: inside it
    assert F.sample(struts, (0.3, 0, -7.5)) > 0   # same spot near the bottom: the strut has thinned away


def test_body_qualified_faces_only_apply_to_that_body():
    from engine.facemask import faces_for_body, parse_faces
    assert faces_for_body("3, 1:7, 2:9", 1) == [3, 7] and faces_for_body("3, 1:7, 2:9", 2) == [3, 9]
    assert faces_for_body("3, 1:7", 0) == [3]
    assert parse_faces("4, 2:9") == [4, 9]


def test_masks_of_different_bodies_are_shown_on_their_own_body(tmp_path):
    # a face mask has no bounds (it is defined everywhere), but a list of them (one per body) must
    # be drawn per body: the cylinder's chosen top once showed body 0's mask (0)
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from engine.graph.nodes import face_mask, import_step
    from engine.paint import field_at_points
    comp, b = TopoDS_Compound(), BRep_Builder()
    b.MakeCompound(comp)
    b.Add(comp, BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 30, 20, 10).Shape())
    b.Add(comp, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(60, 10, 0), gp_Dir(0, 0, 1)), 8, 25).Shape())
    writer = STEPControl_Writer()
    writer.Transfer(comp, STEPControl_AsIs)
    writer.Write(str(tmp_path / "two.step"))
    box, cyl = import_step(str(tmp_path / "two.step"), 0.01, 1, 0.5)
    assert (box.metadata["cad_body"], cyl.metadata["cad_body"]) == (0, 1)
    top_of = lambda m, z: int(np.bincount(m.face_attributes["cad_face"][np.abs(m.triangles_center[:, 2] - z) < 1e-6]).argmax())
    faces = f"0:{top_of(box, 10)}, 1:{top_of(cyl, 25)}"
    masks = [face_mask(m, faces, 2.0, "inside", 2.0, "surface") for m in (box, cyl)]
    points = np.array([[15, 10, 10], [15, 10, 0], [60, 10, 25], [60, 10, 0]], dtype=float)
    assert np.allclose(field_at_points(masks, points, 96), [1, 0, 1, 0], atol=0.05)


def _wide_fade_mask(part):
    ids = facemask.cad_faces(part)
    top = int(np.bincount(ids[part.triangles_center[:, 2] > part.bounds[1][2] - 1e-6]).argmax())
    return facemask.face_mask(part, ids, (top,), 6.0, "inside", 2.0, "surface")


def test_wide_fades_are_sampled_coarser_and_stay_close(monkeypatch):
    # a fade of 6 mm is smooth: sampled 6x across it on a coarser grid and interpolated, it is the
    # same as sampling every voxel to within ~1%
    from engine.cad_import import load_step, tessellate
    part = tessellate(load_step(TEST_STEP)[0])
    grid = Grid.from_bounds(part.bounds, 0.2, 2.0)
    assert facemask.sample_factor(6.0, grid.voxel, grid.size) == 3
    sampled = F.Context(grid).eval_full(_wide_fade_mask(part))
    monkeypatch.setattr(facemask, "SURFACE_FACTORS", ())
    full = F.Context(grid).eval_full(_wide_fade_mask(part))
    sdf = F.Context(grid).eval_full(F.body_from_mesh(part, 2.0))
    near = np.abs(sdf) < 1.5   # within the band (2 mm), where a texture's surface can be
    diff = np.abs(sampled - full)[near]
    # worst spots (0.05 outside, 0.10 inside the part) are where the closest face switches at an
    # edge (the mask itself jumps there); beyond the band the
    # full-resolution mask is simply not computed (0), the sampled one reaches a little further
    assert diff.max() < 0.12 and np.quantile(diff, 0.999) < 0.04 and diff.mean() < 0.002


def test_sampled_masks_are_the_same_in_blocks():
    from engine.cad_import import load_step, tessellate
    part = tessellate(load_step(TEST_STEP)[0])
    mask = _wide_fade_mask(part)
    base = part.bounds[0] - 2.0
    whole = Grid.block(base, 0.2, (0, 0, 0), (121, 97, 85))
    ref = F.Context(whole).eval_full(mask)[24:97, 36:85, 12:73]
    block = F.Context(Grid.block(base, 0.2, (24, 36, 12), (73, 49, 61))).eval_full(mask)
    assert facemask.sample_factor(6.0, 0.2, block.size) > 1
    assert np.array_equal(block, ref)


def test_all_other_faces_leaves_the_chosen_ones_plain():
    # "Apply to: all other faces": pick the few faces that should stay plain instead of the many
    # that get the texture; the fade then runs inside the faces that get it
    box = cad_box((40, 40, 20))
    ids = facemask.cad_faces(box)
    at = lambda field, p: F.sample(field, p)
    plain_top = facemask.face_mask(box, ids, (TOP,), 4.0, "inside", 2.0, "surface", invert=True)
    assert at(plain_top, (0, 0, 10)) == pytest.approx(0.0)        # the picked top stays plain
    assert at(plain_top, (20, 0, -5)) == pytest.approx(1.0)       # a side, away from the top edge
    assert 0 < at(plain_top, (20, 0, 8.5)) < 1                    # fading in below the top edge
    everything = facemask.face_mask(box, ids, (), 4.0, "inside", 2.0, "surface", invert=True)
    assert at(everything, (0, 0, 10)) == pytest.approx(1.0) and at(everything, (20, 0, 0)) == pytest.approx(1.0)
    vol = facemask.face_mask(box, ids, (TOP,), 4.0, reach="volume", invert=True)
    assert at(vol, (0, 0, 10)) < 0.05 and at(vol, (20, 0, -5)) == pytest.approx(1.0, abs=0.02)


def test_apply_to_is_a_node_input():
    from engine.graph.core import REGISTRY
    port = REGISTRY["field.face_mask"].input("apply")
    assert port.default == "chosen faces" and port.choices == ["chosen faces", "all other faces"]
