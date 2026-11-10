import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from engine import fields as F
from engine import pipe as PIPE
from engine.graph.core import REGISTRY, GraphError
from engine.grid import Grid

CURVE = np.array([[0, 0, 0], [15, 10, 5], [30, 0, 12], [40, -10, 0]], float)


def values_at(field, points, voxel=0.5):
    """The field at `points` (each one a grid sample: they are snapped to a grid of `voxel`)."""
    lo = points.min(axis=0) - 2 * voxel
    grid = Grid(lo, points.max(axis=0) + 2 * voxel, voxel)
    v = F.Context(grid).eval_full(field)
    idx = np.rint((points - lo) / voxel).astype(int)
    return v[idx[:, 0], idx[:, 1], idx[:, 2]], lo + idx * voxel


def sweep_samples(field, radius, end_radius, n=3001):
    joints = field.joints
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(joints, axis=0), axis=1))])
    t = np.union1d(np.linspace(0, length[-1], n), length)   # (the corners themselves too)
    c = np.stack([np.interp(t, length, joints[:, k]) for k in range(3)], axis=1)
    end = end_radius or radius
    return c, radius + (end - radius) * t / length[-1]


def unit_teardrop(roof):
    """A dense sampling of the radius-1 teardrop's outline in its meridian plane (rho, y), and an inside test."""
    c, s = math.cos(math.radians(roof)), math.sin(math.radians(roof))
    a = np.linspace(-math.pi / 2, math.pi / 2 - math.radians(roof), 40000)
    arc = np.c_[np.cos(a), np.sin(a)]
    line = np.linspace(0, 1, 40000)[:, None] * (np.array([0, 1 / c]) - arc[-1]) + arc[-1]
    inside = lambda rho, y: (rho * rho + y * y <= 1) | ((y >= c) & (s * rho + c * y <= 1))
    return cKDTree(np.r_[arc, line]), inside


def brute_teardrop(points, centres, radii, up, roof):
    tree, inside = unit_teardrop(roof)
    up = np.asarray(up, float) / np.linalg.norm(up)
    apex = radii / math.cos(math.radians(roof))
    out = np.full(len(points), np.inf)
    for i, p in enumerate(points):
        w = p - centres
        near = np.linalg.norm(w, axis=1) - apex < 1.0   # (a section is no nearer than that: farther ones can't count)
        if not near.any():
            continue
        w, r = w[near], radii[near]
        y = w @ up
        rho = np.linalg.norm(np.cross(w, up), axis=1)
        d, _ = tree.query(np.c_[rho / r, y / r])
        out[i] = np.min(r * np.where(inside(rho / r, y / r), -d, d))
    return out


def test_a_round_pipe_is_the_exact_sweep_of_a_ball():
    rng = np.random.default_rng(0)
    for points, smooth, r, end in ((CURVE[:2], False, 3.0, 1.5), (CURVE, True, 2.0, 0.0), (CURVE, False, 2.0, 3.0)):
        field = PIPE.pipe(points, r, end, "round", smooth=smooth)
        c, radii = sweep_samples(field, r, end)
        near = c[rng.integers(0, len(c), 400)] + rng.normal(0, 2.5, (400, 3))
        got, at = values_at(field, near)
        ref = np.min(np.linalg.norm(at[:, None, :] - c[None], axis=2) - radii[None], axis=1)
        outside = (ref > 0) & (ref < 0.9)
        assert outside.sum() > 50
        assert np.abs(got - ref)[outside].max() < 2e-3
        assert np.all(np.sign(got[np.abs(ref) > 0.05]) == np.sign(ref[np.abs(ref) > 0.05]))


@pytest.mark.parametrize("roof, up", [(45.0, (0, 0, 1)), (55.0, (0, 0.3, 1))])
def test_a_teardrop_pipe_is_the_exact_sweep_of_a_teardrop(roof, up):
    rng = np.random.default_rng(1)
    field = PIPE.pipe(CURVE, 2.5, 1.5, "teardrop", up=up, roof=roof)
    c, radii = sweep_samples(field, 2.5, 1.5, n=2001)
    near = c[rng.integers(0, len(c), 150)] + rng.normal(0, 2.5, (150, 3))
    got, at = values_at(field, near)
    ref = brute_teardrop(at, c, radii, up, roof)
    outside = (ref > 0.02) & (ref < 0.9)
    assert outside.sum() > 15
    assert np.abs(got - ref)[outside].max() < 3e-3   # (the outline's sampling)
    inside = ref < -0.05
    assert np.all(got[inside] < 0) and np.all(got[ref > 0.05] > 0)


def test_a_curved_horizontal_channel_keeps_its_pointed_roof_all_along():
    flat = np.array([[0, 0, 0], [20, 15, 0], [40, 0, 0], [60, 20, 0]], float)
    field = PIPE.pipe(flat, 2.0, 0, "teardrop")
    apex = 2.0 / math.cos(math.radians(45))
    for p in flat:   # the curve goes through every point
        assert F.sample(field, p) == pytest.approx(-2.0, abs=1e-4)
    along = field.joints[:-1] + 0.37 * np.diff(field.joints, axis=0)
    under = [F.sample(field, p + (0, 0, apex - 0.01)) for p in along]
    over = [F.sample(field, p + (0, 0, apex + 0.01)) for p in along]
    assert max(under) < 0 < min(over)   # no notches between the pieces


def test_a_vertical_teardrop_pipe_is_round():
    field = PIPE.pipe(np.array([[0, 0, 0], [0, 0, 20]], float), 2.0, 0, "teardrop", smooth=False)
    assert F.sample(field, (3, 0, 10)) == pytest.approx(1.0, abs=1e-5)
    assert F.sample(field, (0, -3, 10)) == pytest.approx(1.0, abs=1e-5)


def test_pipe_blocks_compute_the_same_values_as_the_whole_grid():
    field = PIPE.pipe(CURVE, 2.5, 1.0, "teardrop", up=(0, 0.2, 1), ends="flat")
    whole = Grid.from_bounds(field.bounds, 0.3, 1.5)
    ref = F.Context(whole).eval_full(field)
    block = F.Context(Grid.block(whole.base, 0.3, (24, 12, 12), (40, 37, 29))).eval_full(field)
    assert ref.dtype == np.float32
    assert np.array_equal(block, ref[24:64, 12:49, 12:41])


def test_flat_ends_are_cut_square_and_round_ends_are_round():
    line = np.array([[0, 0, 0], [20, 0, 0]], float)
    flat = PIPE.pipe(line, 3.0, 0, "round", smooth=False, ends="flat")
    round_ = PIPE.pipe(line, 3.0, 0, "round", smooth=False)
    assert F.sample(flat, (-0.5, 0, 0)) == pytest.approx(0.5, abs=1e-5)
    assert F.sample(round_, (-0.5, 0, 0)) == pytest.approx(-2.5, abs=1e-5)
    assert F.sample(flat, (0.25, 0, 0)) == pytest.approx(-0.25, abs=1e-5)
    assert F.sample(flat, (10, 4, 0)) == pytest.approx(1.0, abs=1e-5)
    assert F.sample(flat, (20.5, 0, 0)) == pytest.approx(0.5, abs=1e-5)


def test_one_point_is_a_ball_or_a_drop():
    ball = PIPE.pipe(np.array([[1.0, 2, 3]]), 2.0)
    assert F.sample(ball, (1, 2, 6)) == pytest.approx(1.0, abs=1e-5)
    drop = PIPE.pipe(np.array([[0.0, 0, 0]]), 1.0, section="teardrop")
    assert F.sample(drop, (0, 0, math.sqrt(2) + 0.5)) == pytest.approx(0.5, abs=1e-5)
    assert F.sample(drop, (0, 0, -1.5)) == pytest.approx(0.5, abs=1e-5)


def test_values_are_exact_within_reach_and_regrow_reaches_farther():
    line = np.array([[0, 0, 0], [20, 0, 0]], float)
    pipe = PIPE.pipe(line, 1.0, 0, smooth=False)
    far = max(PIPE.PAD * 1.0, PIPE.REACH)   # (F.sample's grid has 1 mm voxels)
    assert F.sample(pipe, (10, 6, 0)) == pytest.approx(far)
    assert F.sample(pipe.regrow(5.0), (10, 6, 0)) == pytest.approx(5.0, abs=1e-5)
    assert pipe.regrow(5.0).key != pipe.key
    grown = REGISTRY["implicit.offset"].fn(body=pipe, distance=5.0)
    assert F.sample(grown, (10, 6, 0)) == pytest.approx(0.0, abs=1e-5)
    wall = REGISTRY["implicit.shell"].fn(body=pipe, thickness=8.0)
    assert F.sample(wall, (10, 5, 0)) == pytest.approx(0.0, abs=1e-5)


def test_the_pipe_node_reads_its_points_and_checks_its_inputs():
    node = REGISTRY["implicit.pipe"].fn
    args = dict(radius=2.0, end_radius=0.0, path="smooth", section="round", up="0, 0, 1", roof=45.0, ends="round")
    field = node(points="0, 0, 0; 10, 0, 5\n20, 0, 0;", **args)
    assert F.sample(field, (10, 0, 5)) == pytest.approx(-2.0, abs=1e-4)
    with pytest.raises(GraphError, match="three numbers"):
        node(points="0, 0, 0; 10, 0", **args)
    with pytest.raises(GraphError, match="at least one point"):
        node(points=" ; ", **args)
    with pytest.raises(GraphError, match="more than 0"):
        node(points="0, 0, 0", **{**args, "radius": 0})
    with pytest.raises(GraphError, match="roof"):
        node(points="0, 0, 0", **{**args, "section": "teardrop", "roof": 85})
    with pytest.raises(GraphError, match="section"):
        node(points="0, 0, 0", **{**args, "section": "square"})
