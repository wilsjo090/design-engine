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


def sequential(fn, arg_list, need=0, gpu=False, gpu_need=0, cpu_need=None):
    return [fn(**a) for a in arg_list]


def nan_above_x5(body):
    """The body, but not a number where x > 5 (like a Divide by 0 upstream)."""
    hole = F.Field(lambda ctx: np.where(ctx.axis(0) > 5, np.float32(np.nan), np.float32(0)), "nan-x", "nan")
    return body + hole


def test_texture_node_is_the_wired_chain(body):
    """Texture = Remap (pattern's range to 0..1) x Height x Fade into Offset Body, in one node."""
    from engine.graph.nodes import texture
    grid = Grid((-12, -12, -7), (12, 12, 7), 1.0)
    noise, fade = F.noise(5), F.smoothstep(F.X, -5.0, 5.0)
    values = lambda f: F.Context(grid).eval_full(f)
    wired = F.offset(body, F.remap(noise, -1, 1, 0, 0.8) * fade)
    assert np.allclose(values(texture(body, noise, 0.8, fade)), values(wired), atol=1e-6)
    inverted = F.offset(body, F.remap(noise, -1, 1, 0, 0.8) * (1 - fade))
    assert np.allclose(values(texture(body, noise, 0.8, fade, invert_fade="yes")), values(inverted), atol=1e-6)
    assert np.allclose(values(texture(body, noise, -0.5, 1.0, stretch="as is")), values(F.offset(body, noise * -0.5)),
                       atol=1e-6)
    lo, hi = BOX.bounds
    assert np.allclose(texture(body, noise, 0.8, 1.0).bounds, [lo - 0.8, hi + 0.8])  # grown by the height


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


@pytest.mark.parametrize("edge, grows", [("round", 0.29), ("chamfer", 0.5)])
def test_an_edged_union_leaves_the_faces_two_bodies_share_alone(edge, grows):
    # two boxes side by side share four faces. Mercury's rounded (chamfered) union assumes the surfaces
    # meet at 90 degrees: where they run along each other it moved them out by 0.29 (0.5) x the radius
    # (+555 mm3 on these boxes with round 2 mm)
    a, b = F.box((0, 0, 0), (20, 20, 20)), F.box((10, 0, 0), (20, 20, 20))
    joined = F.union_edged(a, b, edge, 2.0)
    voxel = 0.25
    grid = Grid((-13, -13, -13), (23, 13, 13), voxel)
    values = F.Context(grid).eval_full(joined)
    plain = F.Context(grid).eval_full(F.union(a, b))
    x = np.broadcast_to(grid.coords()[0], values.shape)
    # away from where one box's edge lies hidden on the other's face (x = 0 and x = 10), exactly the union
    surface = (np.abs(plain) < 0.3) & (np.abs(x) > 2.5) & (np.abs(x - 10) > 2.5)
    assert surface.sum() > 10_000 and np.abs(values[surface] - plain[surface]).max() < 1e-5
    # along those hidden edges a body's values fan round its edge: what is left is under a voxel high
    assert not ((values < 0) & (plain > voxel)).any()
    mesh = field_to_mesh(values, grid)
    assert np.abs(mesh.bounds - [[-10, -10, -10], [20, 10, 10]]).max() < voxel
    assert mesh.volume == pytest.approx(30 * 20 * 20, rel=4e-3)
    for point in [(5.0, 10.0 + 0.5 * grows, 0.0), (5.0, 0.0, -10.2), (-10.2, 3.0, 3.0)]:   # also read at a point
        assert F.sample(joined, point) > 0, point


@pytest.mark.parametrize("degrees", [90, 120, 150])
def test_an_edged_union_fills_a_corner_as_a_rolling_ball_would_at_any_angle(degrees):
    # two half-spaces whose faces meet at `degrees` (seen from outside): the fillet is where a ball of the
    # radius touches both, the chamfer its chord. (Mercury's formula is that only at 90 degrees: at 150 it
    # filled 0.83 mm deep where the ball leaves 0.07.)
    radius, g = 2.0, -math.cos(math.radians(degrees))
    second = np.array([math.sqrt(1 - g * g), 0.0, g])
    a, b = F.Z * 1.0, F.X * float(second[0]) + F.Z * float(second[2])
    along = (np.array([0.0, 0.0, 1.0]) + second) / math.sqrt(2 + 2 * g)   # out of the corner, between the faces
    centre = radius / math.sqrt((1 + g) / 2)
    for edge, depth in (("round", centre - radius), ("chamfer", centre - radius * math.sqrt((1 + g) / 2))):
        joined = F.union_edged(a, b, edge, radius)
        assert F.sample(joined, (depth - 0.02) * along) < 0 < F.sample(joined, (depth + 0.02) * along), (edge, depth)


def test_an_edged_union_of_an_l_keeps_its_fillet_and_its_outside():
    # a slab and a block standing on it: the inside corner at x = 0, z = 10 is filled, the faces they
    # share (x = -20, y = +-10) stay where they are
    slab, block = F.box((0, 0, 0), (40, 20, 20)), F.box((-10, 0, 17.5), (20, 20, 25))
    voxel = 0.25
    grid = Grid((-23, -13, -13), (23, 13, 33), voxel)
    joined = F.union_edged(slab, block, "round", 2.0)
    mesh = field_to_mesh(F.Context(grid).eval_full(joined), grid)
    assert np.abs(mesh.bounds - [[-20, -10, -10], [20, 10, 30]]).max() < voxel
    fillet = (1 - math.pi / 4) * 2.0 ** 2 * 20          # the corner's fillet, 20 mm long
    assert mesh.volume == pytest.approx(40 * 20 * 20 + 20 * 20 * 20 + fillet, rel=3e-3)
    depth = (math.sqrt(2) - 1) * 2.0 / math.sqrt(2)
    assert F.sample(joined, (depth - 0.03, 0.0, 10 + depth - 0.03)) < 0 < F.sample(joined, (depth + 0.03, 0.0, 10 + depth + 0.03))


def test_an_edged_union_is_the_same_in_blocks():
    # it reads its neighbours (the two bodies' gradients), so blocks compute a margin around themselves
    a, b = F.box((0, 0, 0), (20, 20, 20)), F.ball((9, 3, 8), 7.0)
    joined = F.union_edged(a, b, "round", 2.0)
    assert F.halo(joined, 0.25) == 1 and F.halo(F.union(a, b), 0.25) == 0
    whole = Grid((-14.5, -14.5, -14.5), (21.5, 14.5, 20.3), 0.25)
    ref = F.Context(whole).eval_full(joined)
    block = F.Context(Grid.block(whole.base, 0.25, (35, 23, 35), (51, 51, 51))).eval_full(joined)
    assert np.array_equal(block[1:-1, 1:-1, 1:-1], ref[36:85, 24:73, 36:85])
    assert (ref[36:85, 24:73, 36:85] != np.minimum(*(F.Context(whole).eval_full(f) for f in (a, b)))[36:85, 24:73, 36:85]).sum() > 1000


def test_an_edged_intersection_rounds_where_surfaces_cross_and_nowhere_else():
    # Topology Optimization's Edges: a member cut off by a face of the design space. A rod through the
    # top of a box: the rim is an arc of the radius, the faces beside it stay
    voxel, radius, rod = 0.2, 1.5, 3.0
    box, through = F.box((0, 0, 0), (16, 16, 16)), F.bar((0, 0, -20), (0, 0, 20), 3.0)
    grid = Grid((-6, -6, 2), (6, 6, 10), voxel)
    cut = F.intersect_edged(box, through, radius)
    values, plain = F.Context(grid).eval_full(cut), F.Context(grid).eval_full(F.intersect(box, through))
    assert np.all(values >= plain)
    solid = lambda a, voxel=voxel: float(np.clip(0.5 - a / voxel, 0, 1).sum()) * voxel ** 3
    spandrel = (1 - math.pi / 4) * radius ** 2                 # what a round takes off a square corner, mm2
    assert solid(plain) - solid(values) == pytest.approx(2 * math.pi * (rod - 0.2234 * radius) * spandrel, rel=0.02)
    for degrees in (0, 20, 45, 70, 90):
        t = math.radians(degrees)
        rho, z = rod - radius + radius * math.cos(t), 8 - radius + radius * math.sin(t)
        for point in ((rho, 0.0, z), (rho * math.sqrt(0.5), -rho * math.sqrt(0.5), z)):
            assert abs(F.sample(cut, point)) < 0.01, (degrees, point)
    assert F.sample(cut, (0.0, 0.0, 8.0)) == pytest.approx(0.0, abs=1e-6)       # the cut face's middle
    assert F.sample(cut, (3.0, 0.0, 5.0)) == pytest.approx(0.0, abs=1e-6)       # the rod's side below the round
    # a radius out and beyond it is the plain intersection again (in between a lower bound of the distance)
    far = plain >= radius
    assert far.any() and np.array_equal(values[far], plain[far])
    assert F.intersect_edged(box, through, 0).key == F.intersect(box, through).key
    # the faces two bodies share stay where they are, and so do the edges both have (a part and the box it
    # is cropped by: Mercury's rounded intersection sinks such faces by 0.29 x the radius)
    grid = Grid((-10, -10, -10), (10, 10, 10), 0.25)
    own = F.Context(grid).eval_full(box)
    same = F.Context(grid).eval_full(F.intersect_edged(box, F.box((0, 0, 0), (16, 16, 16.0)), radius))
    assert np.abs(same - own)[np.abs(own) < 3].max() < 1e-5
    # a part of the second body that is kept as it is (a Keep plate with a strut under it): its edges that
    # lie on the box's faces stay sharp (they read as crossing surfaces on one side only: half a round and a
    # step along the edge), nothing is taken off it, and the strut's rims are rounded
    slab, strut = F.box((0, 0, 5), (16, 16, 6)), F.bar((0, -20, -2), (0, 20, -2), 2.5)
    both = F.union(slab, strut)
    got = F.Context(grid).eval_full(F.intersect_edged(box, both, radius, keep=slab))
    want = F.Context(grid).eval_full(F.intersect(box, both))
    kept = F.Context(grid).eval_full(slab)
    assert np.all(got >= want) and not ((got > 0) & (kept < 0) & (want < 0)).any()
    z = np.broadcast_to(grid.coords()[2], got.shape)
    assert np.abs(got - want)[(z > 2.5) & (np.abs(want) < 1.0)].max() < 1e-5
    rims = 2 * 2 * math.pi * (2.5 - 0.2234 * radius) * spandrel
    assert solid(want, 0.25) - solid(got, 0.25) == pytest.approx(rims, rel=0.05)
    half = F.Context(grid).eval_full(F.intersect_edged(box, both, radius))         # (without `keep`: the step)
    assert np.abs(half - want)[(z > 2.5) & (np.abs(want) < 1.0)].max() > 0.1
    # a body minus another: the same, with the first one's bounds
    minus = F.intersect_edged(box, -through, radius, bounds="first")
    assert np.array_equal(minus.bounds, box.bounds)
    assert F.sample(minus, (rod + radius * (1 - math.sqrt(0.5)), 0.0, 8 - radius * (1 - math.sqrt(0.5)))) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("degrees", [90, 120, 150])
def test_an_edged_intersection_rounds_an_edge_as_a_ball_inside_it_at_any_angle(degrees):
    # two half-spaces whose normals are `180 - degrees` apart: the round is where a ball of the radius inside
    # both touches both
    radius, g = 2.0, -math.cos(math.radians(degrees))
    second = np.array([math.sqrt(1 - g * g), 0.0, g])
    a, b = F.Z * 1.0, F.X * float(second[0]) + F.Z * float(second[2])
    into = -(np.array([0.0, 0.0, 1.0]) + second) / math.sqrt(2 + 2 * g)   # into the edge, between the faces
    depth = radius / math.sqrt((1 + g) / 2) - radius
    cut = F.intersect_edged(a, b, radius)
    assert F.sample(cut, (depth - 0.02) * into) > 0 > F.sample(cut, (depth + 0.02) * into), depth


def test_an_edged_intersection_can_chamfer():
    # a square bar up through a box's top face (z = 10): a flat cut with legs of the size on both faces
    box, bar, size = F.box((0, 0, 0), (20, 20, 20)), F.box((0, 0, 8), (6, 6, 30)), 2.0
    cut = F.intersect_edged(box, bar, size, edge="chamfer")
    plain, arc = F.intersect(box, bar), F.intersect_edged(box, bar, size)
    assert cut.key != arc.key and F.halo(cut, 0.25) == 1
    for t in (0.5, 1.0, 1.5):        # the cut's plane runs from (x 3, z 8) to (x 1, z 10)
        assert F.sample(cut, (3.0 - t, 0.0, 8.0 + t - 0.03)) < 0 < F.sample(cut, (3.0 - t, 0.0, 8.0 + t + 0.03))
    # the faces themselves beyond the legs, and the bar's corner edges (two of its own faces), stay
    for at in ((0.5, 0.0, 10.0), (3.0, 0.0, 7.5), (0.0, 0.0, 10.0)):
        assert F.sample(cut, at) == pytest.approx(F.sample(plain, at), abs=1e-5)
    # it takes more than the arc: half the leg's square against (1 - pi / 4) of it
    grid = Grid((-4.0, -4.0, 6.0), (4.0, 4.0, 11.0), 0.1)
    taken = lambda f: float(((F.Context(grid).eval_full(plain) < 0) & (F.Context(grid).eval_full(f) >= 0)).sum()) * 1e-3
    assert taken(cut) > 2.0 * taken(arc) > 0
    assert F.intersect_edged(box, bar, size, edge="sharp").key == plain.key


def test_an_edged_intersection_is_the_same_in_blocks():
    a, b = F.box((0, 0, 0), (20, 20, 20)), F.ball((9, 3, 8), 7.0)
    cut = F.intersect_edged(a, b, 2.0)
    assert F.halo(cut, 0.25) == 1
    whole = Grid((-14.5, -14.5, -14.5), (21.5, 14.5, 20.3), 0.25)
    ref = F.Context(whole).eval_full(cut)
    block = F.Context(Grid.block(whole.base, 0.25, (60, 30, 50), (51, 51, 51))).eval_full(cut)
    assert np.array_equal(block[1:-1, 1:-1, 1:-1], ref[61:110, 31:80, 51:100])
    assert (ref[61:110, 31:80, 51:100] != np.maximum(*(F.Context(whole).eval_full(f) for f in (a, b)))[61:110, 31:80, 51:100]).sum() > 1000


def test_a_sampled_body_is_a_smooth_function_of_its_samples():
    # Topology Optimization's body: the smoothing's samples as the control values of a cubic B-spline
    from scipy import ndimage
    voxel, start, shape = 0.5, np.array([-8.0, -7.0, -6.0]), (33, 29, 25)
    X, Y, Z = np.meshgrid(*[start[k] + voxel * np.arange(n) for k, n in enumerate(shape)], indexing="ij")
    grid = Grid(start + 1.0, start + voxel * (np.array(shape) - 1) - 1.0, 0.13)   # (between the samples)
    gx, gy, gz = np.meshgrid(*grid.axes(), indexing="ij")
    # a plane comes back exactly
    plane = F.sampled_body((0.3 * X - 0.2 * Y + 0.5 * Z + 1.0).astype(np.float32), start, voxel)
    assert np.abs(F.Context(grid).eval_full(plane) - (0.3 * gx - 0.2 * gy + 0.5 * gz + 1.0)).max() < 1e-5
    # a ball's distance: the surface moves in by the spline's own blur, (voxel^2 / 3) / 2 x 2 / radius
    ball = F.sampled_body((np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 4.0).astype(np.float32), start, voxel)
    exact = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) - 4.0
    values = F.Context(grid).eval_full(ball)
    near = np.abs(exact) < 1.0
    assert (values - exact)[near].mean() == pytest.approx(voxel ** 2 / 3 / 4.0, abs=0.004)
    assert np.abs(values - exact)[near].max() < 0.03
    # at a sample it is the samples' [1, 4, 1] / 6 average, and it never leaves their range: no ripples
    # between them (a spline through the samples rings, and left loose islands by every hard edge)
    rough = np.random.default_rng(1).uniform(-1.0, 1.0, shape).astype(np.float32)
    noisy = F.sampled_body(rough, start, voxel)
    blurred = rough
    for k in range(3):
        blurred = ndimage.convolve1d(blurred, np.array([1, 4, 1], np.float32) / 6, axis=k)
    at = F.Context(Grid.block(start, voxel, (2, 3, 4), (20, 15, 10))).eval_full(noisy)
    assert np.abs(at - blurred[2:22, 3:18, 4:14]).max() < 1e-5
    between = F.Context(Grid.block(start, voxel / 4, (9, 9, 9), (90, 80, 70))).eval_full(noisy)
    assert -1.0 <= between.min() and between.max() <= 1.0 and between.std() > 0.1
    # beyond its samples: the value at their edge plus the distance out
    out = F.Context(Grid((-30, -8, -6), (-20, -7, -5), 1.0)).eval_full(ball)
    assert out.min() > 15 and np.all(np.diff(out, axis=0) < 0)
    assert F.sample(ball, (0.0, 0.0, 3.5)) < 0 < F.sample(ball, (0.0, 0.0, 4.5))   # (a grid of one sample)
    with pytest.raises(ValueError, match="at least 4 samples"):
        F.sampled_body(np.zeros((3, 8, 8), np.float32), start, voxel)


def test_a_sampled_body_is_the_same_in_blocks_and_in_a_worker():
    import pickle
    import cloudpickle
    voxel, start, shape = 0.4, np.array([-10.0, -8.0, -6.0]), (51, 41, 31)
    X, Y, Z = np.meshgrid(*[start[k] + voxel * np.arange(n) for k, n in enumerate(shape)], indexing="ij")
    body = F.sampled_body((np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 4.0 + 0.2 * np.sin(3 * X)).astype(np.float32), start, voxel)
    assert F.halo(body, 0.11) == 0
    whole = Grid((-12.3, -6, -5.5), (6, 6, 5.5), 0.11)       # (it starts before the samples do)
    ref = F.Context(whole).eval_full(body)
    for lo in ((0, 0, 0), (36, 24, 12), (60, 48, 36), (100, 50, 40)):
        n = (40, 50, 45)
        block = F.Context(Grid.block(whole.base, whole.voxel, lo, n)).eval_full(body)
        assert np.array_equal(block, ref[lo[0]:lo[0] + n[0], lo[1]:lo[1] + n[1], lo[2]:lo[2] + n[2]]), lo
    again = pickle.loads(cloudpickle.dumps(body))            # (as a task's field arrives in a worker)
    assert again.key == body.key and np.array_equal(F.Context(whole).eval_full(again), ref)
    assert F.sampled_body(np.asarray(ref[:20, :20, :20]), start, voxel).key != body.key
