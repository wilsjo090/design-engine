"""Field formulas: boxes that grow with offsets, and no NaN from equal edges or a zero radius."""
import math

import numpy as np
import pytest
import trimesh

from engine import fields as F
from engine import tiles
from engine.graph.nodes import _mesh
from engine.grid import Grid
from engine.meshing import field_to_mesh

BOX = trimesh.creation.box((20, 20, 10))  # -10..10, -10..10, -5..5


@pytest.fixture(scope="module")
def body():
    return F.body_from_mesh(BOX, band=2.0)


def sequential(fn, arg_list, need=0, gpu=False, gpu_need=0):
    return [fn(**a) for a in arg_list]


def nan_above_x5(body):
    """The body, but not a number where x > 5 (like a Divide by 0 upstream)."""
    hole = F.Field(lambda ctx: np.where(ctx.axis(0) > 5, np.float32(np.nan), np.float32(0)), "nan-x", "nan")
    return body + hole


def test_offsets_grow_the_box_by_their_reach(body):
    lo, hi = BOX.bounds
    texture = F.remap(F.noise(5), -1, 1, -0.5, 1.5)  # -0.5 .. 1.5 mm
    assert np.allclose(F.offset(body, 3).bounds, [lo - 3, hi + 3])
    assert np.allclose(F.offset(body, texture).bounds, [lo - 1.5, hi + 1.5])
    assert np.allclose((body + -texture).bounds, [lo - 1.5, hi + 1.5])  # Add / Subtract nodes too
    assert np.allclose(F.shell(body, 4).bounds, [lo - 2, hi + 2])
    assert np.allclose(F.offset(body, -2).bounds, BOX.bounds)      # shrinking keeps the box
    assert np.allclose(F.offset(body, F.X * 0.1).bounds, BOX.bounds)  # how far is not known
    assert np.allclose((body - body).bounds, BOX.bounds)           # two bodies: not an offset


def test_offsets_within_the_padding_keep_the_grid(body):
    # trying out a texture's amplitude must not move the grid: the distance field stays cached
    for amplitude in (0.3, 0.6, 1.2):
        textured = F.smoothen(F.offset(body, F.noise(5) * amplitude), 2, 0.2)
        box, room = F.mesh_box(textured, 1.5)
        assert np.array_equal(box, body.bounds) and room == 1.5
    box, room = F.mesh_box(F.subtract(F.offset(body, 4), body), 1.5)  # reaches further: more room
    assert np.allclose(box, BOX.bounds) and room == 4


def test_an_offset_beyond_the_padding_is_not_cut_off(body):
    grown = F.offset(body, 4)
    box, room = F.mesh_box(grown, 0.0)
    mesh = _mesh(grown, box, 0.5, room, 1.0)
    assert np.allclose(mesh.bounds, [BOX.bounds[0] - 4, BOX.bounds[1] + 4], atol=0.3)
    assert not mesh.metadata["clipped"]


def test_a_body_cut_off_by_its_box_is_detected(body):
    grid = Grid.from_bounds(BOX.bounds, 0.5, 1.0)  # 2 mm around the box
    cut = F.offset(body, F.X * 0 + 2.6)  # 2.6 mm, but the formula hides how far
    assert np.allclose(cut.bounds, BOX.bounds)
    assert field_to_mesh(F.Context(grid).eval_full(cut), grid).metadata["clipped"]
    assert not field_to_mesh(F.Context(grid).eval_full(F.offset(body, 0.3)), grid).metadata["clipped"]


def test_nan_samples_are_meshed_as_outside(body):
    grid = Grid.from_bounds(BOX.bounds, 0.5, 1.0)
    mesh = field_to_mesh(F.Context(grid).eval_full(nan_above_x5(body)), grid)
    assert mesh.metadata["nonfinite"] > 0
    assert np.isfinite(mesh.vertices).all() and mesh.is_watertight
    assert 4.9 < mesh.bounds[1][0] < 5.6  # the half that was NaN counts as outside


def test_blocks_report_nan_and_cut_off_bodies(body, monkeypatch):
    monkeypatch.setattr(tiles, "BLOCK", 48)
    monkeypatch.setattr(tiles, "TILE_MIN_VOXELS", 0)
    mesh, computed, _ = tiles.mesh_tiled(nan_above_x5(body), BOX.bounds, 0.25, 1.0, 1.0, sequential)
    assert computed > 1 and mesh.metadata["nonfinite"] > 0 and not mesh.metadata["clipped"]
    assert np.isfinite(mesh.vertices).all() and mesh.is_watertight
    cut = F.offset(body, F.X * 0 + 2)  # 1.5 mm around the box: cut off at the faces, not the corners
    mesh, _, _ = tiles.mesh_tiled(cut, cut.bounds, 0.25, 1.0, 1.0, sequential)
    assert mesh.metadata["clipped"] and mesh.metadata["nonfinite"] == 0


def test_smoothstep_with_equal_edges_is_a_step():
    step = F.smoothstep(F.X, 2, 2)
    assert [F.sample(step, (x, 0, 0)) for x in (1.9, 2.0, 2.1)] == [0, 1, 1]
    assert step.range == (0, 1)
    fade_out = F.smoothstep(F.X, 1, 0)
    assert F.sample(fade_out, (0, 0, 0)) == 1 and F.sample(fade_out, (1, 0, 0)) == 0
    by_field = F.smoothstep(F.X, F.Y, F.Y + F.Z)  # the edges meet where z = 0
    assert F.sample(by_field, (1, 1, 0)) == 1 and F.sample(by_field, (0, 1, 0)) == 0
    assert F.sample(by_field, (1, 0, 2)) == pytest.approx(0.5)


def test_smooth_union_with_radius_zero_is_a_plain_union(body):
    other = F.offset(body, 1)
    assert F.smooth_union(body, other, 0).key == F.union(body, other).key
    with pytest.raises(ValueError, match="0 or more"):
        F.smooth_union(body, other, -1)


def test_acos_range_of_values_outside_minus_one_to_one():
    assert F.acos(F.noise(5) * 10 + 20).range == (0, 0)      # clipped to 1
    assert F.acos(F.noise(5) - 5).range == pytest.approx((math.pi, math.pi))
    assert F.asin(F.noise(5) + 9).range == pytest.approx((math.pi / 2, math.pi / 2))
