"""Members as bars (engine/fea/bars.py): an optimized shape's centre lines traced into straight bars, and the
bars as a body with flat faces, filleted joints and broken rims."""
import math

import numpy as np
import pytest

from engine import fields as F
from engine.fea import bars as B
from engine.grid import Grid


def _portal():
    """A plate (fixed) with two legs under it and a beam between their feet, 5 samples square, and a stub."""
    solid = np.zeros((60, 15, 45), dtype=bool)
    fixed = np.zeros_like(solid)
    fixed[:, :, 36:] = True                  # the plate, z from 36
    solid[8:13, 5:10, 6:36] = True           # a leg at x 8..12
    solid[46:51, 5:10, 6:36] = True          # and one at x 46..50
    solid[8:51, 5:10, 6:11] = True           # the beam, z 6..10
    solid[28:31, 6:9, 11:15] = True          # a stub on the beam: no bar
    return solid | fixed, fixed


def test_a_shape_is_traced_into_bars_along_its_members():
    solid, fixed = _portal()
    up = lambda points: np.tile([0.0, 0.0, 1.0], (len(points), 1))
    net = B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0, inward=up)
    # one member, from the plate down a leg, along the beam and up the other, no stub: three straight bars
    # whose two corners are turned in arcs, as pieces that each turn a little
    assert net.members == 1 and len(net.a) > 3 and np.abs(net.b[:-1] - net.a[1:]).max() < 1e-9
    ends = np.concatenate([net.a, net.b])
    assert np.abs(ends[:, 1] - 7.0).max() < 1.0                              # in the members' middle plane
    way = (net.b - net.a) / np.linalg.norm(net.b - net.a, axis=1, keepdims=True)
    turns = np.degrees(np.arccos(np.clip((way[:-1] * way[1:]).sum(axis=1), -1.0, 1.0)))
    assert turns.max() < B.ARC + 0.5 and turns.sum() == pytest.approx(180.0, abs=12.0)
    long = np.linalg.norm(net.b - net.a, axis=1) > 8.0                       # the straight stretches left
    assert long.sum() == 3
    # the line stays by the members': down x 10, along z 8, up x 48 (a corner cut by its arc)
    off = np.minimum.reduce([np.abs(ends[:, 0] - 10.0), np.abs(ends[:, 2] - 8.0), np.abs(ends[:, 0] - 48.0)])
    assert off.max() < 7.5                   # (an arc of 3 x the width cuts a square corner by 0.41 of its radius)
    # as wide as the members' material needs (5 x 5 samples), and into the plate by their width
    assert net.width == pytest.approx(5.0, abs=0.6) and net.section == pytest.approx(25.0, rel=0.2)
    tops = ends[ends[:, 2] > 30]
    assert len(tops) == 2 and np.all(tops[:, 2] > 36 + 3.0) and np.all(tops[:, 2] < 36 + 7.0)
    # a given size is every bar's width
    assert np.all(B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0, size=3.0).width == 3.0)
    # in mm from the samples' start
    moved = B.trace(solid, fixed, (10.0, -5.0, 2.0), 0.5, inward=up)
    assert np.abs(moved.a - (net.a * 0.5 + [10.0, -5.0, 2.0])).max() < 1e-9
    # nothing to trace: no bars, no error
    assert len(B.trace(fixed, fixed, (0.0, 0.0, 0.0), 1.0).a) == 0


def test_bars_end_in_the_body_that_holds_them_through_a_single_sample():
    # a joint one sample from a fixed body: the link was taken for a ring on the joint, and the bars ended short
    # of the boss (the hanger's outer shaft eyes hung on their back struts alone)
    lines = np.zeros((9, 9, 9), dtype=bool)
    lines[1:8, 4, 4] = True                   # a line along x ...
    lines[4, 4, 5:7] = True                   # ... with a leg up from its middle: its first sample is of the joint,
    anchor = np.zeros_like(lines)             # its second a link of one sample
    anchor[4, 4, 7:9] = True                  # to where a fixed body's part of the skeleton goes on
    positions, anchored, chains = B._chains(lines, anchor)
    held = [k for k in range(len(positions)) if anchored[k]]
    assert len(held) == 1 and tuple(positions[held[0]]) == (4.0, 4.0, 6.0)
    assert any(held[0] in (a, b) and a != b for a, b, _ in chains)


def test_mirrored_bars_are_the_same_on_both_sides():
    solid, fixed = _portal()
    net = B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0)
    both = B.mirrored(net, 0, 29.0)
    flip = lambda p: p * [-1.0, 1.0, 1.0] + [58.0, 0.0, 0.0]
    have = {tuple(np.round(np.concatenate(sorted([p, q], key=tuple)), 6)) for p, q in zip(both.a, both.b)}
    assert have == {tuple(np.round(np.concatenate(sorted([flip(p), flip(q)], key=tuple)), 6)) for p, q in zip(both.a, both.b)}
    assert len(both.a) == len(both.width) == len(both.width_a) > 3


def _tee(blend=0.0, edge=0.0, edge_shape="round"):
    a = np.array([[-20.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    return B.body(a, b, np.tile([0.0, 0.0, 1.0], (2, 1)), 6.0, 4.0, blend, edge, edge_shape, reach=4.0)


def test_bars_have_flat_faces_square_sides_and_filleted_joints():
    plain, filleted = _tee(), _tee(blend=5.0)
    for body in (plain, filleted):
        for at in ((0.0, 0.0), (-10.0, 0.0), (0.0, 10.0), (2.0, 2.5), (3.5, 3.5)):
            # the two faces, also over the joint and its fillet: flat (a smooth union of the bars pillows them)
            assert F.sample(body, (*at, 2.0)) == pytest.approx(0.0, abs=1e-4) or F.sample(body, (*at, 1.9)) > 0
            assert F.sample(body, (*at, 2.3)) > 0.25
        assert F.sample(body, (-10.0, 3.0, 0.0)) == pytest.approx(0.0, abs=1e-4)     # a side, 3 mm from the line
        assert F.sample(body, (-10.0, 2.5, 1.5)) == pytest.approx(-0.5, abs=1e-4)
        assert F.sample(body, (23.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-4)      # a round end
    corner = (3.5, 3.5, 0.0)                 # in the joint's inside corner, half a mm from both bars
    assert F.sample(plain, corner) == pytest.approx(0.5, abs=1e-4) and F.sample(filleted, corner) < 0
    assert F.sample(filleted, (3.5, 3.5, 1.9)) < 0 < F.sample(filleted, (3.5, 3.5, 2.1))   # ... as thick as the bars
    assert F.sample(filleted, (-12.0, 3.0, 0.0)) == pytest.approx(0.0, abs=1e-3)           # and none far from it
    assert plain.key != filleted.key and np.allclose(plain.bounds, [[-23.6, -3.6, -4.0], [23.6, 23.6, 4.0]], atol=0.7)


def test_the_bars_rims_are_chamfered_or_rounded():
    cut, round_ = _tee(edge=1.0, edge_shape="chamfer"), _tee(edge=1.0)
    rim = np.array([-10.0, 3.0, 2.0])        # on a bar's sharp rim
    for body in (cut, round_):
        assert F.sample(body, rim) > 0 and F.sample(body, (-10.0, 3.0, 0.0)) == pytest.approx(0.0, abs=1e-4)
        assert F.sample(body, (-10.0, 0.0, 2.0)) == pytest.approx(0.0, abs=1e-4)
    # the chamfer runs from 1 mm in on the face to 1 mm down the side; the round is an arc between the same lines
    assert F.sample(cut, rim + [0.0, -0.5, -0.5]) == pytest.approx(0.0, abs=1e-4)
    arc = 1.0 - math.sqrt(0.5)
    assert F.sample(round_, rim + [0.0, -arc, -arc]) == pytest.approx(0.0, abs=1e-4)


def test_bars_are_the_same_in_blocks():
    body = _tee(blend=5.0, edge=1.0, edge_shape="chamfer")
    whole = Grid((-26.0, -6.0, -5.0), (26.0, 26.0, 5.0), 0.25)
    ref = F.Context(whole).eval_full(body)
    block = F.Context(Grid.block(whole.base, 0.25, (84, 12, 6), (60, 60, 24))).eval_full(body)
    assert np.array_equal(block, ref[84:144, 12:72, 6:30])
    assert float(ref.min()) < -1.9 and float(ref.max()) == pytest.approx(4.0)


def _frame():
    """Two fixed blocks joined by a straight member that widens along x and by one that runs in a curve."""
    x, y, z = np.meshgrid(np.arange(60), np.arange(75), np.arange(16), indexing="ij")
    fixed = (x < 8) | (x >= 52)
    deep = (z >= 5) & (z < 10)                                   # both members 5 samples thick
    taper = deep & (np.abs(y - 10) <= 2.0 + 3.5 * (x - 8) / 44.0)   # 5 samples wide at x 8, 12 at x 52
    curve = deep & (np.abs(y - _bow(x)) <= 3.0)                  # 7 wide, bowed 15 samples
    return fixed | taper | curve, fixed


def _bow(x):
    return 45.0 + 15.0 * (1.0 - ((np.asarray(x, dtype=float) - 30.0) / 22.0) ** 2)


def test_bars_that_follow_keep_their_members_width_and_curve():
    solid, fixed = _frame()
    up = lambda points: np.tile([0.0, 0.0, 1.0], (len(points), 1))
    net = B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0, follow=True, normal=up)
    middle = 0.5 * (net.a + net.b)
    low = middle[:, 1] < 25                                      # the widening member's pieces
    assert net.members == 2 and low.sum() >= 2 and (~low).sum() >= 4
    # as wide as the member is where each piece is (its section over its thickness, measured along the normal)
    wide = lambda at: 2.0 * np.floor(2.0 + 3.5 * (np.clip(at, 8, 52) - 8) / 44.0) + 1.0    # (in whole samples)
    for at, width in ((net.a[low, 0], net.width_a[low]), (net.b[low, 0], net.width_b[low])):
        clear = (at > 11) & (at < 49)          # (at its ends a member's line has the joint's material too)
        assert clear.sum() >= 2 and np.abs(width - wide(at))[clear].max() < 1.3
    assert net.width_b[low].max() - net.width_a[low].min() > 3.0
    assert net.span[low] == pytest.approx(5.0, abs=0.6) and net.span[~low] == pytest.approx(5.0, abs=0.6)
    assert net.width[~low] == pytest.approx(5.4, abs=1.6)       # (6 to 7 samples up and down, 4 to 6.5 across it)
    # along the member's own line: the curved one's pieces lie on its curve, a piece's ends are its neighbours'
    ends = np.concatenate([net.a[~low], net.b[~low]])
    inside = (ends[:, 0] > 10) & (ends[:, 0] < 50)
    assert np.abs(ends[inside, 1] - _bow(ends[inside, 0])).max() < 1.0
    order = np.flatnonzero(~low)
    assert np.abs(net.b[order[:-1]] - net.a[order[1:]]).max() < 1e-9
    assert np.abs(net.width_b[order[:-1]] - net.width_a[order[1:]]).max() < 1e-9
    # straight: one width for each member, and the curve cut short by its chords
    plain = B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0, normal=up)
    for member in np.unique(plain.branch):
        of = plain.branch == member
        assert np.ptp(plain.width_a[of]) < 1e-9 and np.all(plain.width_a[of] == plain.width_b[of])
    bowed = 0.5 * (plain.a + plain.b)[:, 1] > 25
    assert np.abs(0.5 * (plain.a + plain.b)[bowed, 1] - _bow(0.5 * (plain.a + plain.b)[bowed, 0])).max() > 1.0
    # a given size is every bar's width, also along a curve
    sized = B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0, size=4.0, follow=True, normal=up)
    assert np.all(sized.width_a == 4.0) and np.all(sized.width_b == 4.0)
    # mirrored: the widths go with their ends
    both = B.mirrored(net, 0, 30.0)
    assert len(both.width_a) == len(both.a) == len(both.span) and np.all(both.width == 0.5 * (both.width_a + both.width_b))


def test_a_bar_can_taper_and_a_members_pieces_join_without_a_bulge():
    a, b, up = np.array([[0.0, 0.0, 0.0]]), np.array([[30.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
    cone = B.body(a, b, up, 8.0, 4.0, width_b=2.0, reach=4.0)
    # 4 mm from the line at its wide end, 1 mm at the narrow one, straight sides between, round ends
    slope = (4.0 - 1.0) / 30.0
    level = math.sqrt(1.0 - slope * slope)
    for t in (5.0, 15.0, 25.0):
        side = (4.0 - slope * t) / level        # where the side is, square out from the line at t
        assert F.sample(cone, (t, side, 0.0)) == pytest.approx(0.0, abs=2e-3)
    assert F.sample(cone, (-4.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-4)
    assert F.sample(cone, (31.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-4)
    assert F.sample(cone, (15.0, 0.0, 2.0)) == pytest.approx(0.0, abs=1e-4)      # its faces flat all along
    assert cone.key != B.body(a, b, up, 8.0, 4.0, reach=4.0).key
    # two pieces of one member, at a bend: joined as they are; as two members, blended (a fillet in the bend)
    a = np.array([[-20.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0], [18.0, 9.0, 0.0]])
    ups = np.tile([0.0, 0.0, 1.0], (2, 1))
    sharp = B.body(a, b, ups, 6.0, 4.0, 0.0, reach=4.0)
    one = B.body(a, b, ups, 6.0, 4.0, 5.0, reach=4.0, member=[0, 0])
    two = B.body(a, b, ups, 6.0, 4.0, 5.0, reach=4.0, member=[0, 1])
    grid = Grid((-24.0, -6.0, -3.0), (22.0, 14.0, 3.0), 0.25)
    plain, same, apart = (F.Context(grid).eval_full(body) for body in (sharp, one, two))
    near = np.abs(plain) < 1.5
    assert np.abs(same - plain)[near].max() < 1e-5
    assert (plain - apart)[near].max() > 0.3 and np.all(apart <= plain + 1e-5)


def test_held_joints_settle_where_they_are_told():
    # `settle`: where the joints at a fixed body come to rest (the node: the middle of the body), no farther
    # than INTO x the bars' width
    solid, fixed = _portal()
    asked = {}

    def settle(points, reach):
        asked["reach"] = np.array(reach)
        return points + [0.0, 0.0, 2.5]
    plain = B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0)
    net = B.trace(solid, fixed, (0.0, 0.0, 0.0), 1.0, settle=settle)
    top = lambda n: np.sort(np.concatenate([n.a, n.b])[:, 2])[-2:]
    assert top(net) == pytest.approx(top(plain) + 2.5, abs=1e-9)
    assert len(asked["reach"]) == 2 and asked["reach"] == pytest.approx(B.INTO * 5.0, abs=0.7)


def test_fillets_go_by_the_narrower_bar_and_only_between_bars_of_one_layer():
    a = np.array([[-20.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    ups = np.tile([0.0, 0.0, 1.0], (2, 1))
    grid = Grid((-24.0, -6.0, -3.0), (24.0, 24.0, 3.0), 0.25)
    read = lambda body: F.Context(grid).eval_full(body)
    # a blend of its own for each bar: the narrower bar's is the fillet between the two
    each = B.body(a, b, ups, [8.0, 4.0], 4.0, [9.0, 3.0], reach=4.0)
    narrow, wide = B.body(a, b, ups, [8.0, 4.0], 4.0, 3.0, reach=4.0), B.body(a, b, ups, [8.0, 4.0], 4.0, 9.0, reach=4.0)
    near = np.abs(read(narrow)) < 1.5
    assert np.abs(read(each) - read(narrow))[near].max() < 1e-5
    assert np.abs(read(each) - read(wide))[near].max() > 0.3 and each.key not in (narrow.key, wide.key)
    # the order the bars are given in does not matter: the widest is laid first
    turned = B.body(a[::-1], b[::-1], ups, [4.0, 8.0], 4.0, [3.0, 9.0], reach=4.0)
    assert np.abs(read(turned) - read(each))[near].max() < 1e-5
    # bars of two layers are not blended: each as it is alone
    apart = B.body(a, b, ups, [8.0, 4.0], 4.0, 9.0, reach=4.0, layer=[0, 1])
    alone = [B.body(a[k:k + 1], b[k:k + 1], ups[k:k + 1], [8.0, 4.0][k], 4.0, 9.0, reach=4.0) for k in range(2)]
    assert np.abs(read(apart) - np.minimum(read(alone[0]), read(alone[1]))).max() < 1e-5
    assert (read(apart) - read(wide))[near].max() > 0.3 and apart.key != wide.key


def test_a_bar_deep_in_the_part_has_one_plane():
    # an S-shaped strut along the middle of a cube, as far from four of its faces: the design space's nearest face
    # is another one every few pieces, and with each piece's faces laid as that face the pieces were turned up to 80
    # degrees against each other, shards standing out of the bar all along it
    from types import SimpleNamespace
    from scipy.spatial import cKDTree
    from engine.graph import nodes_fea as NF
    x = np.linspace(-14.0, 14.0, 400)
    line = np.column_stack([x, 3.0 * np.sin(math.pi * x / 14.0), 0.0 * x])   # an S in the plane z = 0
    axis = np.arange(-20.0, 20.25, 0.5)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    values = (cKDTree(line).query(points)[0] - 2.5).reshape((len(axis),) * 3).astype(np.float32)
    made = SimpleNamespace(values=values, start=np.full(3, -20.0), voxel=0.5)
    ends = F.union(F.ball(tuple(line[0]), 4.0), F.ball(tuple(line[-1]), 4.0))
    seen = {}
    body = B.body

    def keep(a, b, normal, *args, **kwargs):
        seen.update(a=a, b=b, normal=normal, member=kwargs["member"])
        return body(a, b, normal, *args, **kwargs)
    B.body = keep
    try:
        field, about = NF._bars(made, F.box((0.0, 0.0, 0.0), (40.0, 40.0, 40.0)), None, ends, 0.0, 2.0, 0.0, 0.0, 0.0,
                                "round")
    finally:
        B.body = body
    assert field is not None and about.bars == 1 and len(seen["a"]) > 3
    way = seen["b"] - seen["a"]
    normal = seen["normal"] - (seen["normal"] * way).sum(axis=1, keepdims=True) * way / (way * way).sum(axis=1, keepdims=True)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    assert (normal @ normal[0]).min() > 0.9999                # one normal all along: no piece askew of the next
    assert abs(normal[0] @ [0.0, 0.0, 1.0]) > 0.98   # across the plane the bar bends in
    assert np.abs((seen["a"] - seen["a"][0]) @ normal[0]).max() < 1e-6   # and the bar lies in that plane


def test_bars_of_two_planes_flow_together_at_their_joint():
    # a bar lying flat and one standing on edge, in two layers, meeting at the origin: joined plainly but for the
    # knot there, a ball their ends lie in and a blend in space round it that fades out along the bars
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.array([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0]])
    ways = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    make = lambda **knot: B.body(a, b, ways, 6.0, 3.0, 4.0, reach=4.0, layer=[0, 1], **knot)
    plain, knot = make(), make(knots=[[0.0, 0.0, 0.0]], knot_size=1.5, knot_blend=4.0, knot_reach=6.0)
    whole = Grid((-8.0, -8.0, -8.0), (34.0, 34.0, 8.0), 0.25)
    read = lambda body: F.Context(whole).eval_full(body)
    x, y, z = np.meshgrid(*whole.axes(), indexing="ij")
    far = np.sqrt(x * x + y * y + z * z) > 12.5
    one, two = read(plain), read(knot)
    assert knot.key != plain.key and np.array_equal(one[far], two[far])     # the same beyond twice the reach
    assert (one - two).min() > -1e-6 and (one - two).max() > 0.3             # material in the corner between them
    corner = [2.0, 3.4, 0.0]                                                 # outside both bars, in their corner
    assert F.sample(plain, corner) > 0.2 and F.sample(knot, corner) < 0.0
    block = F.Context(Grid.block(whole.base, 0.25, (12, 24, 6), (60, 48, 36))).eval_full(knot)
    assert np.array_equal(block, two[12:72, 24:72, 6:42])                    # the same in blocks


def test_a_joint_at_the_face_is_a_boss_along_its_axis():
    # the knot as a round boss along z from 2 mm under the joint to 5 mm over it, radius 4: a flat pad for the
    # part's face to cut, where a ball would be cut to a disc beside the bars
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.array([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0]])
    ways = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    make = lambda **knot: B.body(a, b, ways, 6.0, 3.0, 0.0, reach=4.0, layer=[0, 1], knots=[[0.0, 0.0, 0.0]],
                                 knot_size=4.0, knot_blend=0.0, knot_reach=6.0, **knot)
    ball, boss = make(), make(knot_axis=[[0.0, 0.0, 1.0]], knot_span=[[-2.0, 5.0]])
    assert ball.key != boss.key
    beside = [-3.0, -2.0, 4.0]          # 3.6 mm from the axis, 4 mm over the joint: beside both bars
    assert F.sample(ball, beside) > 0.5 and abs(F.sample(boss, beside) - (math.hypot(3.0, 2.0) - 4.0)) < 1e-4
    assert abs(F.sample(boss, [-3.0, -2.0, 6.0]) - 1.0) < 1e-4       # 1 mm over its top
    assert F.sample(boss, [-3.0, -2.0, -3.5]) > 0.5                   # and nothing under its foot


def _star(ways, normals, width, thick, long=40.0):
    """Straight bars from the origin, a piece each, lofted at the origin: the lofts and their body."""
    ways = np.array(ways, dtype=float)
    ways /= np.linalg.norm(ways, axis=1, keepdims=True)
    a, b = np.zeros_like(ways), long * ways
    width, thick = np.broadcast_to(width, (len(a),)).astype(float), np.broadcast_to(thick, (len(a),)).astype(float)
    made = B.lofts(a, b, B.normals(a, b, np.array(normals, dtype=float)), width, width, thick, np.arange(len(a)),
                   [[(m, 0) for m in range(len(a))]], edge=0.8)
    body = B.body(made.a, made.b, made.normal, made.width, made.thick, 0.0, 0.8, "chamfer", reach=4.0, width_b=made.width_b,
                  flat_a=made.flat_a, flat_b=made.flat_b, curves=made.curves)
    return made, body


def test_a_joints_bars_are_cut_short_and_lofted_into_each_other():
    # three bars 6 x 6 in one plane, 120 degrees apart
    turn = math.sqrt(0.75)
    ways = np.array([[1.0, 0.0, 0.0], [-0.5, turn, 0.0], [-0.5, -turn, 0.0]])
    made, body = _star(ways, np.tile([0.0, 0.0, 1.0], (3, 1)), 6.0, 6.0)
    # every bar ends 1.2 x its side short of the joint, cut square, and there is a loft between every two
    assert np.allclose(made.a, 7.2 * ways) and made.flat_a.all() and not made.flat_b.any() and len(made.curves) == 3
    unit = lambda v: v / np.linalg.norm(v)
    for curve in made.curves:
        # a loft leaves its one bar where that was cut, along it, and arrives on the other the same way
        first, last = curve.ctrl[0], curve.ctrl[3]
        assert min(np.abs(ways @ unit(first) - 1.0).min(), np.abs(ways @ unit(last) - 1.0).min()) < 1e-9
        assert np.allclose(unit(curve.ctrl[1] - first), -unit(first)) and np.allclose(unit(last - curve.ctrl[2]), unit(last))
        assert np.allclose(curve.width, 6.0) and np.allclose(curve.thick, 6.0)
        assert np.abs(curve.normal @ [0.0, 0.0, 1.0]).min() > 1.0 - 1e-9          # its faces stay the bars'
        assert np.abs(np.linalg.norm(curve.points - 14.4 * unit(first + last), axis=1) - 7.2 * math.sqrt(3.0)).max() < 0.2   # an arc
    # the joint is solid (its middle lies a mm inside each loft), and as thick as its bars: flat faces, no ball
    for at in ((0.0, 0.0), (2.0, 1.0), (-3.0, 0.0), (6.0, 0.0), (9.0, 0.0)):
        assert F.sample(body, (*at, 0.0)) < -0.9 and F.sample(body, (*at, 2.9)) < 0 < F.sample(body, (*at, 3.1))
    # between two bars the loft's inside is the joint's fillet: their sharp corner would lie 3.5 mm from the joint
    between = np.array([0.5, turn, 0.0])
    assert F.sample(body, 4.4 * between) < 0 < F.sample(body, 5.4 * between)
    assert F.sample(body, (20.0, 3.0, 0.0)) == pytest.approx(0.0, abs=1e-4)      # the bars as they were
    # one piece, and the same in blocks
    whole = Grid((-24.0, -24.0, -5.0), (24.0, 24.0, 5.0), 0.25)
    ref = F.Context(whole).eval_full(body)
    from scipy import ndimage
    assert ndimage.label(ref < 0)[1] == 1 and float(ref.min()) < -2.9
    block = F.Context(Grid.block(whole.base, 0.25, (60, 72, 6), (84, 72, 24))).eval_full(body)
    assert np.array_equal(block, ref[60:144, 72:144, 6:30])


def test_a_loft_takes_its_section_from_one_bar_to_the_other():
    # a bar 8 wide and 4 thick lying flat, and one standing on edge: across the bend the loft is 8 wide at the one
    # and 4 at the other, with no twist (a section turned a quarter with its sides swapped is the same section)
    made, body = _star([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 8.0, 4.0)
    curve, = made.curves
    assert np.allclose(curve.width, [8.0, 4.0]) and np.allclose(curve.thick, [4.0, 8.0])
    assert np.abs(curve.normal @ [0.0, 0.0, 1.0]).min() > 1.0 - 1e-6
    start, end = curve.ctrl[0], curve.ctrl[3]
    for point, across, up in ((start, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), (end, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])):
        wide, tall = (8.0, 4.0) if point is start else (4.0, 8.0)
        point = point * (1.0 - 1.0 / np.linalg.norm(point))           # (1 mm along the loft from its end)
        assert F.sample(body, point + np.multiply(across, 0.5 * wide - 0.3)) < 0 < F.sample(body, point + np.multiply(across, 0.5 * wide + 0.4))
        assert F.sample(body, point + np.multiply(up, 0.5 * tall - 0.3)) < 0 < F.sample(body, point + np.multiply(up, 0.5 * tall + 0.4))
    # the bend is a bar's bend: nothing in the sharp corner the two bars would have made
    assert F.sample(body, (-3.0, -1.5, 0.0)) > 1.0


def test_a_surface_flows_out_in_pits_and_sharp_grooves_only():
    # samples 0.4 mm apart, as a joint's fill, and the time and the sharpest groove of a joint of 6 mm bars
    from scipy import ndimage
    h = 0.4
    axis = h * np.arange(-30, 31)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    time, sharp = B.FILL_TIME * 36.0, B.FILL_SHARP * 6.0
    smooth = lambda values: ndimage.gaussian_filter(values.astype(np.float32), B.FILL_BLUR)

    def crossing(values, line):
        """Where the values along a line of samples cross 0, in samples from its start."""
        v = values[line]
        k = int(np.flatnonzero((v[:-1] < 0) != (v[1:] < 0))[0])
        return k + v[k] / (v[k] - v[k + 1])
    # a flat face stays as it is, exactly
    face = z.astype(np.float32)
    assert np.array_equal(B.flow(face, h, time, sharp), face)
    # a dimple 1.5 mm deep and 6 mm wide in a flat face is hollow in every direction: it fills from the bottom,
    # three quarters of the way in this time (the joints' own are half as wide and fill flat), the face round it stays
    pit = B.flow(smooth(np.maximum(z, 4.0 - np.sqrt(x * x + y * y + (z - 2.5) ** 2))), h, time, sharp)
    assert axis[0] + h * crossing(pit, (30, 30, slice(None))) == pytest.approx(-0.37, abs=0.08)
    assert abs(axis[0] + h * crossing(pit, (45, 30, slice(None)))) < 0.01
    # the inside of a bend and a round opening 5.5 mm wide (solid outside a cylinder: hollow one way, straight the
    # other, and rounder than `sharp`) stay where they are
    for radius in (5.0, 2.75):
        hole = smooth(radius - np.sqrt(x * x + y * y))
        line = (slice(30, None), 30, 30)
        assert crossing(B.flow(hole, h, time, sharp), line) == pytest.approx(crossing(hole, line), abs=0.01)
    # two walls meeting in an inside corner (solid where x < 0 or y < 0) are a groove sharper than `sharp`: it is
    # rounded to an arc of that radius, and the walls away from the corner stay
    groove = B.flow(smooth(np.minimum(x, y)), h, time, sharp)
    diagonal = (np.arange(30, 61), np.arange(30, 61), np.full(31, 30))
    assert h * math.sqrt(2.0) * crossing(groove, diagonal) == pytest.approx(sharp * (math.sqrt(2.0) - 1.0), abs=0.06)
    assert abs(axis[0] + h * crossing(groove, (45, slice(None), 30))) < 0.01
    # with `ridge`, an edge sharper than that (solid where x < 0 and y < 0) is rounded to an arc of that radius,
    # material taken off, the walls away from it staying; with `shallow` 0 it is not touched
    ridge = 3.0
    edge = smooth(np.maximum(x, y))
    inward = (np.arange(30, -1, -1), np.arange(30, -1, -1), np.full(31, 30))
    rounded = B.flow(edge, h, time, sharp, ridge=ridge)
    assert h * math.sqrt(2.0) * crossing(rounded, inward) == pytest.approx(ridge * (math.sqrt(2.0) - 1.0), abs=0.08)
    assert abs(axis[0] + h * crossing(rounded, (15, slice(None), 30))) < 0.01
    assert np.array_equal(B.flow(face, h, time, sharp, ridge=ridge), face)
    assert np.array_equal(B.flow(edge, h, time, sharp, ridge=ridge, shallow=np.zeros(x.shape, dtype=np.float32)),
                          B.flow(edge, h, time, sharp))
    # and with `hold` 0 nothing moves
    assert np.array_equal(B.flow(smooth(np.minimum(x, y)), h, time, sharp, hold=np.zeros(x.shape, dtype=np.float32)),
                          smooth(np.minimum(x, y)))


def test_the_dimple_where_lofts_of_three_planes_meet_is_filled():
    # three bars 6 x 6 along x, y and down: a corner in space, whose three lofts leave a dimple on its outside
    made, plain = _star([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 6.0, 6.0)
    joint, = made.joints
    assert np.linalg.norm(joint.centre - [1.5, 1.5, -1.5]) < 1.5 and joint.size == pytest.approx(6.0)
    patches = B.fills(lambda grid: F.Context(grid).eval_full(plain), made.joints)
    filled = B.body(made.a, made.b, made.normal, made.width, made.thick, 0.0, 0.8, "chamfer", reach=4.0, width_b=made.width_b,
                    flat_a=made.flat_a, flat_b=made.flat_b, curves=made.curves, fills=patches)
    assert [patch.cut for patch in patches] == [False] and filled.key != plain.key
    whole = Grid((-12.0, -12.0, -24.0), (24.0, 24.0, 12.0), 0.25)
    before, after = F.Context(whole).eval_full(plain), F.Context(whole).eval_full(filled)
    x, y, z = np.meshgrid(*whole.axes(), indexing="ij")
    away = np.sqrt((x - joint.centre[0]) ** 2 + (y - joint.centre[1]) ** 2 + (z - joint.centre[2]) ** 2)
    added, taken = (before >= 0) & (after < 0), (before < 0) & (after >= 0)
    # material is added, in the dimple on the outside of the corner (and in the creases inside it), and only
    # round the joint; nothing is taken off
    assert added.sum() * 0.25 ** 3 > 2.0 and not taken.any()
    assert ((x + y - z)[added] < -0.5).sum() > 20 and away[added | taken].max() < joint.core + 0.5 * joint.size + 0.5
    assert np.array_equal(before[away > joint.core + 0.5 * joint.size + 1.0], after[away > joint.core + 0.5 * joint.size + 1.0])
    # the bars' flat faces by the joint are theirs: nothing lies on them
    for point in ((7.0, 0.0, 3.05), (0.0, 7.0, 3.05), (7.0, -3.05, 0.0), (-3.05, 0.0, -7.0)):
        assert F.sample(plain, point) == pytest.approx(0.05, abs=0.01) and F.sample(filled, point) == F.sample(plain, point)
    # and in blocks the same
    block = F.Context(Grid.block(whole.base, 0.25, (24, 24, 60), (72, 72, 60))).eval_full(filled)
    assert np.array_equal(block, after[24:96, 24:96, 60:120])


def test_a_joints_dimple_is_filled_flat_and_its_bends_stay():
    # the corner in space again. Its dimple, a pit with sharp creases where three lofts' faces meet, is filled
    # flat; so is the sharp inside of the corner, where the three bends meet; the bend between two bars, round
    # inside and straight across, is not touched, nor a flat face or a bar's side
    made, plain = _star([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 6.0, 6.0)
    patches = B.fills(lambda grid: F.Context(grid).eval_full(plain), made.joints)
    filled = B.body(made.a, made.b, made.normal, made.width, made.thick, 0.0, 0.8, "chamfer", reach=4.0, width_b=made.width_b,
                    flat_a=made.flat_a, flat_b=made.flat_b, curves=made.curves, fills=patches)
    out = np.array([-1.0, -1.0, 1.0]) / math.sqrt(3.0)          # the way out of the corner
    side = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    other = np.cross(out, side)

    def reach(body, start, way, lo=0.0, hi=12.0):
        for _ in range(30):         # how far from `start` along `way` the body goes
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if F.sample(body, start + mid * way) < 0 else (lo, mid)
        return lo
    spots = [u * side + v * other for u, v in ((0.0, 0.0), (0.6, 0.0), (-0.6, 0.0), (0.0, 0.6), (0.0, -0.6))]
    low, high = (np.array([reach(body, at, out, -1.0, 6.0) for at in spots]) for body in (plain, filled))
    assert low.min() < -0.2 and np.ptp(low) > 0.3                # a dimple with creases in it
    assert high.min() > low.max() + 0.3 and np.ptp(high) < 0.03  # filled, and flat
    for at in ((4.0, -1.0, 3.3), (10.0, 3.3, 0.0)):              # over a flat face, beside a bar
        assert F.sample(plain, at) > 0.2 and F.sample(filled, at) == F.sample(plain, at)
    bend, inside = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0), np.array([1.0, 1.0, -1.0]) / math.sqrt(3.0)
    assert reach(filled, np.zeros(3), bend) == reach(plain, np.zeros(3), bend)
    assert reach(filled, np.array([3.0, 3.0, -3.0]), inside) > reach(plain, np.array([3.0, 3.0, -3.0]), inside) + 0.5


def test_a_joint_is_drawn_face_by_face(monkeypatch):
    # three bars in one plane: the joint's faces are the bars' own (z = +-3), its middle as solid as before
    from scipy import ndimage
    monkeypatch.setattr(B, "LOFT_FACES", True)
    turn = math.sqrt(0.75)
    made, body = _star([[1.0, 0.0, 0.0], [-0.5, turn, 0.0], [-0.5, -turn, 0.0]], np.tile([0.0, 0.0, 1.0], (3, 1)), 6.0, 6.0)
    assert all(curve.faces is not None and not curve.free for curve in made.curves)
    for at in ((0.0, 0.0), (2.0, 1.0), (-3.0, 0.0), (6.0, 0.0)):
        assert F.sample(body, (*at, 2.9)) < 0 < F.sample(body, (*at, 3.1)) and F.sample(body, (*at, -2.9)) < 0 < F.sample(body, (*at, -3.1))
    # three bars tilted out of one plane: one piece, the bars past their cuts as they were, and in blocks the same
    ways, normals = [[1.0, 0.0, 0.3], [-0.5, turn, 0.0], [-0.5, -turn, -0.3]], [[0.0, 0.3, 1.0], [0.2, 0.0, 1.0], [0.0, 0.0, 1.0]]
    made, body = _star(ways, normals, 6.0, 6.0)
    saved, B.LOFT_FACES = B.LOFT_FACES, False
    try:
        _, plain = _star(ways, normals, 6.0, 6.0)
    finally:
        B.LOFT_FACES = saved
    assert body.key != plain.key
    whole = Grid((-24.0, -24.0, -12.0), (24.0, 24.0, 12.0), 0.25)
    values, before = F.Context(whole).eval_full(body), F.Context(whole).eval_full(plain)
    assert ndimage.label(values < 0)[1] == 1
    x, y, z = np.meshgrid(*whole.axes(), indexing="ij")
    far = np.sqrt(x * x + y * y + z * z) > 16.0
    assert np.array_equal(values[far], before[far])
    block = F.Context(Grid.block(whole.base, 0.25, (60, 60, 24), (72, 72, 48))).eval_full(body)
    assert np.array_equal(block, values[60:132, 60:132, 24:72])


def test_a_thin_plate_fit_keeps_what_it_is_given_and_is_smooth():
    # a quadratic's biharmonic is 0, and the samples' own is exact for it: the fit gives it back where it was taken out
    axis = np.arange(-12, 13, dtype=np.float64)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    values = (0.1 * x * x - 0.05 * y * z + 0.3 * z).astype(np.float32)
    hole = (x * x + y * y + z * z) < 36
    gone = np.where(hole, np.float32(7.0), values)
    got = B._biharmonic(gone, hole)
    assert np.abs(got - values).max() < 1e-3 and np.array_equal(got[~hole], values[~hole])


def test_a_joint_is_patched_where_its_lofts_overlap():
    # the corner of three bars: its middle, where the three lofts overlap, is fitted again as one smooth surface; the
    # bars away from the joint are as they were, it stays one piece, and in blocks the same
    from scipy import ndimage
    made, plain = _star([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 6.0, 6.0)
    seams = B.body(made.a, made.b, made.normal, made.width, made.thick, 0.0, 0.8, "chamfer", reach=4.0, width_b=made.width_b,
                   flat_a=made.flat_a, flat_b=made.flat_b, curves=made.curves, seams=True)
    patches = B.patches(lambda grid: F.Context(grid).eval_full(plain), made.joints, lambda grid: F.Context(grid).eval_full(seams))
    patched = B.body(made.a, made.b, made.normal, made.width, made.thick, 0.0, 0.8, "chamfer", reach=4.0, width_b=made.width_b,
                     flat_a=made.flat_a, flat_b=made.flat_b, curves=made.curves, fills=patches)
    assert len(patches) == 1 and patches[0].mode == "replace" and patched.key != plain.key
    whole = Grid((-12.0, -12.0, -24.0), (24.0, 24.0, 12.0), 0.25)
    before, after = F.Context(whole).eval_full(plain), F.Context(whole).eval_full(patched)
    x, y, z = np.meshgrid(*whole.axes(), indexing="ij")
    joint, = made.joints
    away = np.sqrt((x - joint.centre[0]) ** 2 + (y - joint.centre[1]) ** 2 + (z - joint.centre[2]) ** 2)
    changed = np.abs(before - after) > 1e-3
    assert changed.any() and away[changed].max() < joint.core + 2.0 and np.array_equal(before[away > joint.core + 3.0], after[away > joint.core + 3.0])
    assert ndimage.label(after < 0)[1] == 1
    assert abs(float((after < 0).sum() - (before < 0).sum())) * 0.25 ** 3 < 0.1 * float((before < 0).sum()) * 0.25 ** 3
    block = F.Context(Grid.block(whole.base, 0.25, (24, 24, 60), (72, 72, 60))).eval_full(patched)
    assert np.array_equal(block, after[24:96, 24:96, 60:120])


def test_a_stub_between_two_joints_that_count_as_one_is_left_out():
    # two bars from the origin, a stub 5 mm long down from it, and a bar on from the stub's end: with both the
    # stub's ends in the joint, the three bars are lofted into each other and the stub is gone
    turn = math.sqrt(0.75)
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -5.0, 0.0]])
    b = np.array([[40.0, 0.0, 0.0], [-20.0, 40.0 * turn, 0.0], [0.0, -5.0, 0.0], [0.0, -45.0, 0.0]])
    up = np.tile([0.0, 0.0, 1.0], (4, 1))
    six = np.full(4, 6.0)
    made = B.lofts(a, b, up, six, six, six, np.arange(4), [[(0, 0), (1, 0), (2, 0), (2, 1), (3, 0)]], edge=0.8)
    assert sorted(set(made.piece.tolist())) == [0, 1, 3] and len(made.curves) == 3
    body = B.body(made.a, made.b, made.normal, made.width, made.thick, 0.0, 0.8, "chamfer", reach=4.0, width_b=made.width_b,
                  flat_a=made.flat_a, flat_b=made.flat_b, curves=made.curves)
    from scipy import ndimage
    values = F.Context(Grid((-24.0, -30.0, -5.0), (30.0, 30.0, 5.0), 0.25)).eval_full(body)
    assert ndimage.label(values < 0)[1] == 1
