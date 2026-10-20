import numpy as np
import pytest
import trimesh
from scipy.ndimage import map_coordinates

from engine import fields as F
from engine.grid import Grid, upsample_linear
from engine.graph.nodes import scale_bodies
from engine.noise import fractal, fractal_on_grid
from engine.sdf import MeshQuery
from engine.transform import anchor_point


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
