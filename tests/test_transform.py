from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.ndimage import map_coordinates

from engine import fields as F
from engine.facemask import cad_body as facemask_body
from engine.grid import Grid, upsample_linear
from engine.graph import Evaluator, Graph, GraphError, Many, link
from engine.graph.nodes import mirror_bodies, move_bodies, pattern_bodies, rotate_bodies, scale_bodies
from engine.meshing import signed_volume
from engine.noise import fractal, fractal_on_grid
from engine.sdf import MeshQuery
from engine.transform import anchor_point

ROOT = Path(__file__).resolve().parents[1]


def cup_on_pins():
    """A square cup (truncated pyramid, side planes meeting at the origin) resting on 4 pins
    whose tips touch its sloped sides at z = 2."""
    bottom = [(x, y, 1.0) for x in (-1, 1) for y in (-1, 1)]
    top = [(x, y, 3.0) for x in (-3, 3) for y in (-3, 3)]
    cup = trimesh.convex.convex_hull(np.array(bottom + top, dtype=float))
    pins = []
    for tip in [(2, 0, 2), (-2, 0, 2), (0, 2, 2), (0, -2, 2)]:
        tip = np.array(tip, float)
        out = tip * np.array([1, 1, 0]) / 2  # base of the pin, away from the cup
        base = [out + d for d in ([0, 0, -1.5], [0.3, 0.3, -1.5], [-0.3, 0.3, -1.5])]
        pins.append(trimesh.convex.convex_hull(np.array([tip] + base)))
    return cup, trimesh.util.concatenate(pins)


def test_upsample_matches_map_coordinates():
    rng = np.random.default_rng(0)
    coarse = rng.normal(size=(7, 6, 5)).astype(np.float32)
    shape = (25, 21, 17)
    ours = upsample_linear(coarse, 4, shape)
    idx = np.indices(shape).reshape(3, -1) / 4.0
    ref = map_coordinates(coarse, idx, order=1, mode="nearest").reshape(shape)
    assert np.allclose(ours, ref, atol=1e-5)


def test_coarse_sampled_noise_is_accurate():
    grid = Grid((0, 0, 0), (12, 10, 6), 0.1)        # fine grid, smooth noise -> coarse path
    noise = fractal_on_grid(grid, scale=40, octaves=2, seed=5)
    pts = grid.points()[::1009]
    direct = fractal(*pts.T.astype(np.float32), scale=40, octaves=2, seed=5)
    idx = np.unravel_index(np.arange(grid.size)[::1009], grid.shape)
    assert np.abs(noise[idx] - direct).max() < 2e-4  # x140 density -> < 0.03 rad of phase


def test_anchor_is_the_tip_of_the_contact_faces():
    cup, pins = cup_on_pins()
    anchor, how = anchor_point(cup, pins)
    assert how == "apex"
    assert np.allclose(anchor, [0, 0, 0], atol=1e-6)
    assert anchor_point(cup)[1] == "center"


@pytest.mark.parametrize("factor", [0.8, 1.0, 1.3, 1.6])
def test_scaled_body_stays_seated(factor):
    cup, pins = cup_on_pins()
    scaled = scale_bodies(cup, factor, pins)
    q = MeshQuery(scaled.vertices, scaled.faces)
    tips = np.array([(2, 0, 2), (-2, 0, 2), (0, 2, 2), (0, -2, 2)], float)
    assert np.abs(q.signed(tips)).max() < 1e-6                    # pins still touch the sides
    assert np.isclose(scaled.extents[0], cup.extents[0] * factor)  # and it really got bigger


def test_scale_can_come_from_a_field():
    cup, pins = cup_on_pins()
    # anchor is at the origin, so a field 1.25 + X samples to exactly 1.25 there
    scaled = scale_bodies(cup, F.X + 1.25, pins)
    assert np.isclose(scaled.extents[0], cup.extents[0] * 1.25)
    with pytest.raises(Exception, match="greater than 0"):
        scale_bodies(cup, F.X - 1.0, pins)


# --- Move, Rotate, Mirror and Pattern Bodies ---------------------------------------------------
def _box_body(size=(10, 6, 4)):
    """A box with what Import STEP keeps: a CAD face per side (x- 0, x+ 1, y- 2, y+ 3, z- 4, z+ 5) and
    surface parameters at every triangle corner."""
    box = trimesh.creation.box(size)
    n = box.face_normals
    axis = np.abs(n).argmax(axis=1)
    box.face_attributes["cad_face"] = (2 * axis + (n[np.arange(len(n)), axis] > 0)).astype(np.int32)
    box.face_attributes["cad_uv"] = np.arange(len(box.faces) * 6, dtype=np.float32).reshape(-1, 3, 2)
    box.metadata["cad_body"] = 0
    box._cache.clear()
    return box


def test_move_and_rotate_keep_the_faces():
    box = _box_body()
    moved = move_bodies(box, "5, 0, -2")
    assert np.allclose(moved.vertices, box.vertices + [5, 0, -2])
    assert np.array_equal(moved.face_attributes["cad_face"], box.face_attributes["cad_face"])
    assert move_bodies(box, "0, 0, 0") is box                         # nothing to do: the same body
    turned = rotate_bodies(moved, 90, "z", "0, 0, 1", "centre", "0, 0, 0")
    assert np.allclose(turned.bounds.mean(axis=0), [5, 0, -2]) and np.allclose(turned.extents, [6, 10, 4])
    assert np.allclose(rotate_bodies(moved, 90, "z", "", "origin", "").bounds.mean(axis=0), [0, 5, -2])
    tilted = rotate_bodies(box, 90, "custom", "1, 0, 0", "point", "0, 0, 10")
    assert np.allclose(tilted.bounds.mean(axis=0), [0, 10, 10]) and np.allclose(tilted.extents, [10, 4, 6])
    pair = rotate_bodies(Many([box, moved]), 180, "z", "", "centre", "")   # a list turns as one
    assert isinstance(pair, Many) and np.allclose(pair[0].bounds.mean(axis=0), [5, 0, 0])
    assert signed_volume(turned.vertices, turned.faces) > 0


def test_mirror_keeps_bodies_closed_and_facing_out():
    box = move_bodies(_box_body(), "10, 0, 0")
    both = mirror_bodies(box, "YZ", 0.0, "", "yes")
    assert isinstance(both, Many) and len(both) == 2 and both[0] is box
    image = both[1]
    assert np.allclose(image.bounds.mean(axis=0), [-10, 0, 0])
    assert np.isclose(signed_volume(image.vertices, image.faces), 240)          # still outward
    assert np.array_equal(image.face_attributes["cad_uv"], box.face_attributes["cad_uv"][:, ::-1])
    only = mirror_bodies(box, "XY", 3.0, "", "no")
    assert not isinstance(only, list) and np.allclose(only.bounds.mean(axis=0), [10, 0, 6])
    # about its x+ face (at x = 15): the mirror image sits next to it, a whole part from half of one
    whole = mirror_bodies(box, "face", 0.0, "1", "yes")
    assert np.allclose(whole[1].bounds.mean(axis=0), [20, 0, 0])
    assert np.allclose(mirror_bodies(box, "face", 0.0, "0:1", "no").bounds.mean(axis=0), [20, 0, 0])
    with pytest.raises(GraphError, match="not flat"):
        mirror_bodies(box, "face", 0.0, "1, 3", "no")
    with pytest.raises(GraphError, match="pick the face"):
        mirror_bodies(box, "face", 0.0, "", "no")


def test_pattern_copies_in_rows_and_round_an_axis():
    box = _box_body()
    grid = pattern_bodies(box, "linear", 3, "20, 0, 0", 2, "0, 30, 0", "z", "", "", 360)
    assert len(grid) == 6 and grid[0] is box
    centres = np.array([b.bounds.mean(axis=0) for b in grid])
    assert np.allclose(centres[:, :2], [[0, 0], [20, 0], [40, 0], [0, 30], [20, 30], [40, 30]])
    away = move_bodies(box, "50, 0, 0")
    ring = pattern_bodies(away, "circular", 4, "", 1, "", "z", "", "0, 0, 0", 360)
    assert np.allclose([b.bounds.mean(axis=0)[:2] for b in ring], [[50, 0], [0, 50], [-50, 0], [0, -50]])
    half = pattern_bodies(away, "circular", 3, "", 1, "", "z", "", "0, 0, 0", 180)
    assert np.allclose(half[-1].bounds.mean(axis=0)[:2], [-50, 0])
    two = pattern_bodies(Many([box, away]), "linear", 2, "0, 0, 10", 1, "", "z", "", "", 360)
    assert [round(b.bounds.mean(axis=0)[0]) for b in two] == [0, 50, 0, 50]   # originals first, then copies
    assert {facemask_body(b) for b in two} == {0}
    with pytest.raises(GraphError, match="too many"):
        pattern_bodies(box, "linear", 2000, "1, 0, 0", 1, "0, 1, 0", "z", "", "", 360)


def test_moved_bodies_flow_through_the_graph():
    g = Graph()
    g.add("import", "input.import_step", path=str(ROOT / "samples" / "two_blocks.step"))
    g.add("ring", "cad.pattern", body=link("import.bodies"), kind="circular", count=3, point="0, 0, 0")
    g.add("flip", "cad.mirror", cad=link("ring.body"), plane="XZ", keep="no")
    g.add("implicit", "implicit.from_cad", cad=link("flip.body"))
    assert g.problems() == []
    out = Evaluator(g, log=None).run(["implicit"])
    assert len(out["ring"]["body"]) == 6 and len(out["implicit"]["out"]) == 6
