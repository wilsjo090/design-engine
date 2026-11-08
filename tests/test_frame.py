"""Frame (truss) optimization: profiles, the layout LP and member adding against hand solutions,
the frame check against beam theory and Euler buckling, sizing, and a whole run on a box."""
import json
import math
import re

import numpy as np
import pytest

from engine import fields as F
from engine.fea.materials import PRESETS
from engine.fea.model import Condition, ConditionError, Conditions
from engine.frame import beam, ground as G, joints as J, layout as LY, sizing as S
from engine.frame.optimize import optimize_frame, unbraced_lengths
from engine.frame.profiles import Profile
from test_fea import _box_step, _face_id, _face_triangles

PLA = PRESETS["PLA"]


# --- profiles ------------------------------------------------------------------------------------
def _raster(mask_fn, b, h, n=1400):
    """Area and principal second moments of a shape drawn as a pixel mask (an independent reference)."""
    y = (np.arange(n) + 0.5) / n * b
    z = (np.arange(n) + 0.5) / n * h
    Y, Z = np.meshgrid(y, z, indexing="ij")
    m = mask_fn(Y, Z)
    da = (b / n) * (h / n)
    A = m.sum() * da
    cy, cz = (Y * m).sum() * da / A, (Z * m).sum() * da / A
    Iyy = (((Z - cz) ** 2) * m).sum() * da
    Izz = (((Y - cy) ** 2) * m).sum() * da
    Iyz = (((Y - cy) * (Z - cz)) * m).sum() * da
    w = np.linalg.eigvalsh([[Izz, Iyz], [Iyz, Iyy]])
    return A, w[1], w[0]


@pytest.mark.parametrize("profile, size, mask", [
    (Profile("rectangle", ratio=2.0), 10.0, lambda Y, Z: np.ones_like(Y, dtype=bool)),
    (Profile("square tube", wall=0.1), 20.0, lambda Y, Z: ~((Y > 2) & (Y < 18) & (Z > 2) & (Z < 18))),
    (Profile("rect tube", ratio=1.5, wall=0.1), 20.0, lambda Y, Z: ~((Y > 2) & (Y < 18) & (Z > 2) & (Z < 28))),
    (Profile("I", ratio=1.0, flange=0.1, web=0.1), 20.0,
     lambda Y, Z: (Z < 2) | (Z > 18) | ((Y > 9) & (Y < 11))),
    (Profile("angle", wall=0.15), 20.0, lambda Y, Z: (Z < 3) | (Y < 3)),
])
def test_section_properties_match_a_pixel_integration(profile, size, mask):
    sec = profile.section(size)
    A, I1, I2 = _raster(mask, sec.width, sec.height)
    assert sec.A == pytest.approx(A, rel=2e-3)
    assert sec.Iy == pytest.approx(I1, rel=5e-3) and sec.Iz == pytest.approx(I2, rel=5e-3)
    assert sec.Iy >= sec.Iz


def test_round_and_rectangle_formulas():
    r = Profile("round").section(10.0)
    assert r.A == pytest.approx(math.pi * 25) and r.Iy == pytest.approx(math.pi * 10 ** 4 / 64)
    t = Profile("round tube", wall=0.1).section(10.0)
    assert t.A == pytest.approx(math.pi * (100 - 64) / 4)
    rect = Profile("rectangle", ratio=2.0).section(10.0)
    assert (rect.Iy, rect.Iz, rect.Wy, rect.Wz) == pytest.approx((10 * 20 ** 3 / 12, 20 * 10 ** 3 / 12,
                                                                  10 * 20 ** 2 / 6, 20 * 10 ** 2 / 6))
    assert Profile("round tube", wall=0.01, min_wall=0.8).thickness(10.0) == 0.8  # SLS minimum wall


# --- layout ---------------------------------------------------------------------------------------
def _ground(nodes, bars, fixed_nodes, loads, spacing=1.0):
    nodes = np.asarray(nodes, dtype=np.float64)
    bars = np.asarray(bars)
    fixed = np.zeros((len(nodes), 3), dtype=bool)
    fixed[fixed_nodes] = True
    L = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    owner = np.full(len(nodes), -1)
    return G.Ground(nodes, bars, L, fixed, np.asarray(loads, dtype=np.float64), spacing, owner)


def test_two_and_three_bar_trusses_match_the_hand_solution():
    # load F down at B = (L, 0, 0); supports A = (0,0,0), C = (0,0,h), D = (0,0,-h)
    F_, L, h, sigma = 100.0, 40.0, 30.0, 25.0
    nodes = [(0, 0, 0), (L, 0, 0), (0, 0, h), (0, 0, -h)]
    loads = np.zeros((4, 3))
    loads[1, 2] = -F_
    two = LY.optimize(_ground(nodes[:3], [(0, 1), (2, 1)], [0, 2], loads[:3]), sigma, member_adding=False)
    sin = h / math.hypot(L, h)
    assert two.volume == pytest.approx(F_ * (2 * L * L + h * h) / (h * sigma), rel=1e-6)
    # B's forces: the diagonal from C pulls (tension F / sin), the bar to A pushes
    by_bar = {tuple(sorted(map(tuple, two.nodes[b]))): q for b, q in zip(two.bars, two.forces)}
    assert by_bar[tuple(sorted([(0., 0., h), (L, 0., 0.)]))] == pytest.approx(F_ / sin, rel=1e-6)
    assert by_bar[tuple(sorted([(0., 0., 0.), (L, 0., 0.)]))] == pytest.approx(-F_ * L / h, rel=1e-6)
    # with D as well the symmetric pair wins: +-F / (2 sin), volume F (L^2 + h^2) / (h sigma)
    three = LY.optimize(_ground(nodes, [(0, 1), (2, 1), (3, 1)], [0, 2, 3], loads), sigma, member_adding=False)
    assert three.volume == pytest.approx(F_ * (L * L + h * h) / (h * sigma), rel=1e-6)
    assert sorted(three.forces) == pytest.approx([-F_ / (2 * sin), F_ / (2 * sin)], rel=1e-6)


def _wall_ground(L=8, H=8, reach=3, overlap=False):
    """A vertical lattice x = 0..L, z = -H..H (y = 0) held along x = 0, a unit load down at (L, 0)."""
    ij = np.array([(i, k) for i in range(L + 1) for k in range(-H, H + 1)])
    nodes = np.column_stack([ij[:, 0], np.zeros(len(ij)), ij[:, 1]]).astype(float)
    pairs = []
    for a in range(len(ij)):
        for b in range(a + 1, len(ij)):
            d = np.abs(ij[b] - ij[a])
            if np.hypot(*d) <= reach + 1e-9 and (overlap or math.gcd(int(d[0]), int(d[1])) == 1):
                pairs.append((a, b))
    loads = np.zeros((len(nodes), 3))
    loads[np.flatnonzero((ij[:, 0] == L) & (ij[:, 1] == 0))[0], 2] = -1.0
    return _ground(nodes, pairs, np.flatnonzero(ij[:, 0] == 0), loads)


def test_a_load_off_a_long_wall_takes_the_45_degree_truss():
    # Hemp / Michell: with the whole wall available two bars at +-45 degrees are optimal: V = 2 F L / sigma
    lay = LY.optimize(_wall_ground(), 1.0)
    assert lay.volume == pytest.approx(2 * 8.0, rel=1e-6)
    used = lay.nodes[lay.bars]
    slopes = np.abs((used[:, 1, 2] - used[:, 0, 2]) / (used[:, 1, 0] - used[:, 0, 0]))
    assert np.allclose(slopes, 1.0)
    assert len(np.unique(lay.chains)) == 2          # two straight bars (chains of 45 degree pieces)


def test_member_adding_finds_the_full_lps_optimum():
    g = _wall_ground(L=6, H=3, reach=3)   # the wall too short for 45 degrees: a real Michell-like layout
    full = LY.optimize(g, 1.0, member_adding=False)
    added = LY.optimize(g, 1.0, member_adding=True)
    assert added.volume == pytest.approx(full.volume, rel=1e-6)
    assert added.candidates < len(g.bars)
    assert full.volume > 2 * 6.0          # dearer than with a long wall


def test_solver_leftovers_that_reach_no_support_are_dropped():
    # a frame of the connect-to demo kept a lone 0.1 N bar (its partners fell below the drop share):
    # the frame check was singular. Bars ending loose, and pieces reaching no support, go.
    nodes = np.array([[0, 0, 0], [10, 0, 0], [10, 5, 0], [10, 10, 0], [20, 0, 0], [20, 5, 0]], float)
    bars = np.array([[0, 1], [1, 2], [2, 3], [4, 5]])
    fixed = np.zeros((6, 3), bool)
    fixed[0] = True
    loads = np.zeros((6, 3))
    loads[1, 2] = -10.0
    g = G.Ground(nodes, bars, np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1), fixed, loads,
                 5.0, np.full(6, -1))
    assert LY._held_bars(g, np.arange(4)).tolist() == [0]


def test_an_lp_is_stopped_from_inside_highs_and_reports_its_iterations(monkeypatch):
    # an LP of a fine spacing runs for seconds: Stop (cancel raising) ends it from HiGHS's own callback, in this
    # process, and its iterations go to `tick` while it solves; `show` gets every LP's numbers
    import psutil
    g = _wall_ground()
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    monkeypatch.setattr(LY, "TICK", 0.0)
    ticks, shown = [], []
    lay = LY.optimize(g, 1.0, member_adding=False, cancel=lambda: None, tick=ticks.append, min_area=0.5,
                      show=lambda what, q, numbers: shown.append((what, numbers)))
    assert lay.volume == pytest.approx(2 * 8.0, rel=1e-6)
    assert ticks and all(re.fullmatch(r"LP \d+, interior point iteration \d+", t) for t in ticks)
    assert shown[0][0] == "the lightest layout" and shown[0][1]["lp"] == 1
    assert shown[0][1]["in_lp"] == shown[0][1]["candidates"] == len(g.bars)
    assert shown[0][1]["volume"] == pytest.approx(16.0, rel=1e-6)

    class Stopped(Exception):
        pass

    calls = []

    def later():   # once before the solve, then from inside it
        calls.append(1)
        if len(calls) > 2:
            raise Stopped
    with pytest.raises(Stopped):
        LY.optimize(g, 1.0, member_adding=False, cancel=later)
    assert len(calls) == 3
    assert not {p.pid for p in psutil.Process().children(recursive=True)} - before


def test_without_scipys_highs_bindings_linprog_gives_the_same_layout_and_stops_apart(monkeypatch):
    # (scipy's private HiGHS bindings may go): every LP is then a linprog, one over APART columns in a process of
    # its own, which Stop ends
    import psutil
    g = _wall_ground()
    kept = LY.optimize(g, 1.0, min_area=0.5)
    monkeypatch.setattr(LY, "_Highs", None)
    plain = LY.optimize(g, 1.0, min_area=0.5)
    assert plain.volume == pytest.approx(kept.volume, rel=1e-6) and len(plain.bars) == len(kept.bars)
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    monkeypatch.setattr(LY, "APART", 0)
    assert LY.optimize(g, 1.0, member_adding=False, cancel=lambda: None).volume == pytest.approx(2 * 8.0, rel=1e-6)

    class Stopped(Exception):
        pass

    def stop():
        raise Stopped
    with pytest.raises(Stopped):
        LY.optimize(g, 1.0, member_adding=False, cancel=stop)
    calls = []

    def later():   # while the LP's process runs (it takes a second to start)
        calls.append(1)
        if len(calls) > 2:
            raise Stopped
    with pytest.raises(Stopped):
        LY.optimize(g, 1.0, member_adding=False, cancel=later)
    assert not {p.pid for p in psutil.Process().children(recursive=True)} - before


def test_joint_length_gives_fewer_longer_bars():
    # with joint costs bars may pass over nodes (overlap); each bar then costs a little extra
    plain = LY.optimize(_wall_ground(L=6, H=3, reach=7), 1.0)
    costly = LY.optimize(_wall_ground(L=6, H=3, reach=7, overlap=True), 1.0, joint_length=3.0)
    assert len(costly.bars) < len(plain.bars)
    assert costly.lengths.mean() > plain.lengths.mean()
    assert costly.volume >= plain.volume - 1e-9


def test_max_size_shares_a_load_among_more_bars():
    # the 45 degree pair carries 0.71 each; bounded to 0.3 (in compression less over longer bars, as buckling
    # does), the load goes over more bars, none above its bound, and the layout is no lighter
    g = _wall_ground()
    limit = lambda L: (np.full(len(L), 0.3), 0.3 * np.minimum(1.0, 2.0 / np.asarray(L)))
    plain = LY.optimize(g, 1.0)
    capped = LY.optimize(g, 1.0, limit=limit)
    t, c = limit(capped.lengths)
    assert np.abs(plain.forces).max() > 0.7
    assert np.all(capped.forces <= t * (1 + 1e-6)) and np.all(-capped.forces <= c * (1 + 1e-6))
    assert len(capped.bars) > len(plain.bars) and capped.volume >= plain.volume - 1e-9


def test_a_smallest_size_gives_fewer_fuller_bars():
    # real volume: every bar at least min_area; the plain layout spreads the load over many weak bars
    # (a short wall: the plain layout is a Michell-like fan of 77 pieces; 50 and 23% lighter when written)
    g = _wall_ground(L=12, H=3, reach=3)
    real = lambda lay, a: float((lay.lengths * np.maximum(np.abs(lay.forces), a)).sum())
    plain = LY.optimize(g, 1.0)
    sparse_ = LY.optimize(g, 1.0, min_area=0.5)
    assert len(sparse_.bars) < 0.8 * len(plain.bars)
    assert real(sparse_, 0.5) < 0.85 * real(plain, 0.5)
    assert sparse_.volume >= plain.volume - 1e-9      # the plain LP is still the least plastic volume


def test_pieces_running_straight_through_a_node_are_one_bar():
    nodes = np.array([(0, 0, 0), (10, 0, 0), (20, 0, 0), (10, 0, 10), (10, 10, 0)], dtype=float)
    bars = np.array([(0, 1), (1, 2), (1, 3), (3, 4)])
    chains = LY.continuous_bars(nodes, bars)
    assert chains[0] == chains[1] and len(np.unique(chains)) == 3
    # the long bar is held sideways at its middle (another bar meets it there)
    span = unbraced_lengths(nodes, bars, chains, np.zeros(len(nodes), dtype=bool))
    assert span[chains[0]] == pytest.approx(10.0)


# --- moving joints ---------------------------------------------------------------------------------
class _Space:
    """A box, less another box (`hole`), as the signed distance `ground.inside` is."""
    step = 0.25

    def __init__(self, lo, hi, hole=None):
        self.boxes = [(np.asarray(lo, float), np.asarray(hi, float))] + \
            ([tuple(np.asarray(v, float) for v in hole)] if hole else [])

    def __call__(self, p):
        d = [np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
             for q in (np.abs(p - 0.5 * (lo + hi)) - 0.5 * (hi - lo) for lo, hi in self.boxes)]
        return d[0] if len(d) == 1 else np.maximum(d[0], -d[1])


def _bent_truss(space, joint):
    """Load F = 100 N down at B = (40, 0, 0), held at A = (0, 0, 30) and C = (0, 0, -30); the bar from A
    reaches B through a free joint J off the straight line, which a bar from C holds sideways."""
    nodes = np.array([(0, 0, 30), (0, 0, -30), (40, 0, 0), joint], float)   # A, C, B, J
    loads = np.zeros((4, 3))
    loads[2, 2] = -100.0
    g = _ground(nodes, [(0, 3), (2, 3), (1, 2), (1, 3)], [0, 1], loads, spacing=4.0)
    g.inside, g.free = space, np.zeros(4)
    return g, LY.optimize(g, 25.0, member_adding=False)


def test_moved_joints_straighten_a_bent_bar_into_the_hand_solution():
    # sequential LPs move J onto the line A-B, the bar from C goes idle and J is taken out of the straight
    # bar: the symmetric two-bar truss, V = F (L^2 + h^2) / (h sigma) (as in the three-bar test above)
    g, lay = _bent_truss(_Space((-5, -5, -35), (45, 5, 35)), (14, 0, 30))
    moved, report = J.move_joints(lay, g, 25.0, 25.0)
    assert lay.volume > 1.2 * 100.0 * (40 ** 2 + 30 ** 2) / (30 * 25.0)
    assert moved.volume == pytest.approx(100.0 * (40 ** 2 + 30 ** 2) / (30 * 25.0), rel=1e-3)
    assert len(moved.bars) == 2 and report["steps"] and report["dropped"] == report["straightened"] == 1
    assert sorted(map(tuple, moved.nodes[moved.bars].reshape(-1, 3).round(6))) == \
        sorted([(0, 0, 30), (40, 0, 0), (0, 0, -30), (40, 0, 0)])
    r = (LY.equilibrium_matrix(moved.nodes, moved.bars) @ moved.forces).reshape(-1, 3) - moved.loads
    assert np.abs(r[~moved.fixed.any(axis=1)]).max() < 1e-6                     # it carries the load


def test_moved_joints_keep_their_bars_inside_the_design_space():
    # a block between J and the straight line: J moves only as far as its bars stay clear of it
    space = _Space((-5, -5, -35), (45, 5, 35), hole=((26, -6, 5), (32, 6, 9)))
    g, lay = _bent_truss(space, (24, 0, 4))
    assert G._stays_inside(lay.nodes, lay.bars, np.zeros(len(lay.nodes)), space, 2.0).all()
    moved, report = J.move_joints(lay, g, 25.0, 25.0)
    assert G._stays_inside(moved.nodes, moved.bars, np.zeros(len(moved.nodes)), space, 2.0).all()
    assert report["joints"] == 1 and moved.volume < lay.volume


# --- frame check ---------------------------------------------------------------------------------
def test_cantilever_matches_beam_theory():
    P, L, d = 10.0, 100.0, 6.0
    sec = Profile("round").section(d)
    G_ = PLA.E / (2 * (1 + PLA.nu))
    nodes = [(0, 0, 0), (L, 0, 0)]
    loads = np.array([(0, 0, 0), (0, 0, -P)], dtype=float)
    for shear, extra in ((False, 0.0), (True, P * L / (0.9 * sec.A * G_))):
        res = beam.check(nodes, [(0, 1)], [sec], PLA, [0], loads, ["round"], shear=shear, buckling=False)
        tip = P * L ** 3 / (3 * PLA.E * sec.Iy) + extra
        assert -res.displacements[1, 2] == pytest.approx(tip, rel=1e-9)
        assert abs(res.reactions[0, 4]) == pytest.approx(P * L, rel=1e-9)   # fixed-end moment
        assert res.reactions[0, 2] == pytest.approx(P, rel=1e-9)
    assert res.stress[0] == pytest.approx(P * L / sec.Wy, rel=1e-9)


def test_euler_buckling_of_a_column():
    # clamped at the bottom, free at the top: P_cr = pi^2 E I / (2 L)^2
    L, d, n = 120.0, 4.0, 16
    sec = Profile("round").section(d)
    nodes = [(0, 0, L * i / n) for i in range(n + 1)]
    els = [(i, i + 1) for i in range(n)]
    loads = np.zeros((n + 1, 3))
    loads[-1, 2] = -1.0
    res = beam.check(nodes, els, [sec] * n, PLA, [0], loads, ["round"] * n, shear=False)
    assert res.buckling == pytest.approx(math.pi ** 2 * PLA.E * sec.Iy / (2 * L) ** 2, rel=0.01)


def test_reactions_balance_the_loads():
    nodes = np.array([(0, 0, 0), (0, 40, 0), (30, 20, 10), (30, 20, 40), (60, 10, 20)], dtype=float)
    els = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (0, 3), (1, 3)]
    sec = Profile("square tube").section(8.0)
    loads = np.zeros((5, 3))
    loads[4] = (3.0, -2.0, -50.0)
    res = beam.check(nodes, els, [sec] * len(els), PLA, [0, 1], loads, ["square tube"] * len(els))
    assert np.allclose(res.reactions[:, :3].sum(axis=0), -loads.sum(axis=0), atol=1e-8)
    moment = np.cross(nodes, res.reactions[:, :3]).sum(axis=0) + res.reactions[:, 3:].sum(axis=0) + \
        np.cross(nodes, loads).sum(axis=0)
    assert np.allclose(moment, 0, atol=1e-6)


# --- sizing ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("force", [500.0, -500.0, -50.0])
def test_required_size_passes_and_a_hair_less_does_not(force):
    prof, L = Profile("round tube"), 80.0
    s = S.required_size(prof, force, L, PLA, 2.0)
    t, c = S.capacity(prof.section(s), L, PLA, 2.0)
    t2, c2 = S.capacity(prof.section(s * 0.995), L, PLA, 2.0)
    assert (force <= t) if force > 0 else (-force <= c)
    assert (force > t2) if force > 0 else (-force > c2)


def test_rounding_never_goes_down_and_groups_are_optimal():
    assert S.round_up(3.01, 2.0, 0.5) == 3.5 and S.round_up(1.0, 2.0, 0.5) == 2.0 and S.round_up(3.0, 2, 0.5) == 3.0
    assert S.round_up(4.2, 2.0, sizes=[4, 6, 8]) == 6.0
    sizes = np.array([2.0, 2.5, 3.0, 6.0, 6.5, 9.0])
    lengths = np.array([10, 10, 10, 50, 50, 5.0])
    grouped = S.group(sizes, lengths, Profile("round"), 3)
    assert len(np.unique(grouped)) <= 3 and np.all(grouped >= sizes)
    # brute force over every way to cut the sorted sizes into 3 runs
    best = min(sum(Profile("round").section(sizes[j - 1]).A * lengths[i:j].sum() for i, j in zip((0, a, b), (a, b, 6)))
               for a in range(1, 5) for b in range(a + 1, 6))
    got = sum(Profile("round").section(g).A * l for g, l in zip(grouped, lengths))
    assert got == pytest.approx(best)


def test_sizes_never_go_above_max_size():
    assert S.round_up(3.2, 2.0, 0.5, largest=4.0) == 3.5
    assert S.round_up(3.8, 2.0, 1.5, largest=4.0) == 4.0     # (a Max size off the size step)
    assert S.next_size(3.5, 1.0, largest=4.0) == 4.0
    with pytest.raises(ConditionError, match="Max size"):
        S.round_up(4.2, 2.0, 0.5, largest=4.0)
    with pytest.raises(ConditionError, match="Max size"):
        S.next_size(4.0, 0.5, largest=4.0)


def test_johnson_and_euler_meet_at_the_transition():
    sec = Profile("round").section(5.0)
    Cc = math.pi * math.sqrt(2 * PLA.E / PLA.strength)
    L = Cc * sec.r_min
    below = S.capacity(sec, L * (1 - 1e-9), PLA, 1.0)[1]
    above = S.capacity(sec, L * (1 + 1e-9), PLA, 1.0)[1]
    assert below == pytest.approx(above, rel=1e-6)


# --- ground structure ----------------------------------------------------------------------------
def test_holes_get_one_node_each_and_loads_on_one_pin_add_up():
    # the bolted wall bracket: 4 bolt holes (one Fixed Faces) -> 4 nodes at their axes; a bearing
    # load and a sideways force on the pin hole -> one node carrying both
    from pathlib import Path
    from engine.graph.nodes import import_step
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_bolts.step"), 0.01, 1, 0.5)[0]
    ids = mesh.face_attributes["cad_face"]
    pin = mesh.triangles[ids == 10]
    conds = Conditions((Condition("fixed", mesh.triangles[np.isin(ids, [5, 6, 7, 8])]),
                        Condition("load", pin, load="bearing", amount=200.0, direction="-Z"),
                        Condition("load", pin, amount=80.0, direction="+Y")))
    g = G.build(F.body_from_mesh(mesh, 2.0), conds, min_size=2.0)
    face = np.flatnonzero(g.owner >= 0)
    held, loaded = face[g.fixed[face].all(axis=1)], face[np.abs(g.loads[face]).sum(axis=1) > 0]
    assert len(held) == 4 and len(loaded) == 1 and len(face) == 5
    assert np.allclose(g.loads[loaded[0]], [0, 80, -200], atol=1.0)
    assert np.allclose(g.nodes[loaded[0]][[0, 2]], [50, 40], atol=0.05)      # on the pin's axis
    assert (np.bincount(g.bars.ravel(), minlength=len(g.nodes))[face] > 100).all()   # long bars from them


def test_supports_on_one_line_are_held_along_their_bolts():
    # the bolted bracket held by its two lower bolts only: one node per hole put the supports on one line,
    # which bars on pin joints can turn about, and no frame carried the pin's load. Each hole is then held
    # by two nodes on its axis; with four held holes nothing changes (one node each, as above)
    from pathlib import Path
    from engine.graph.nodes import import_step
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_bolts.step"), 0.01, 1, 0.5)[0]
    ids = mesh.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", mesh.triangles[np.isin(ids, [5, 7])]),
                        Condition("load", mesh.triangles[ids == 10], load="bearing", amount=200.0, direction="-Z")))
    g = G.build(F.body_from_mesh(mesh, 2.0), conds, min_size=2.0)
    held = g.nodes[g.fixed.all(axis=1)]
    assert len(held) == 4 and any("one line" in n for n in g.notes)
    assert np.allclose(np.sort(held[:, 1]), [7, 7, 23, 23], atol=0.05) and np.allclose(held[:, 2], 8, atol=0.05)
    assert np.ptp(held[:, 0]) > 3.0 and not G._on_a_line(held, g.spacing)   # along the bolts, 0..8 mm


def test_light_bars_go_where_the_frame_passes_without_them():
    # two bars, each in two pieces, from supports to a load, and a brace between their middles. Hanging (both
    # in tension) the brace carries nothing and goes (Light bars: remove); the bars that carry the load stay.
    # With the lower bar in compression the brace halves its buckling length: without it that bar would
    # have to be thicker, the frame heavier, so it stays
    from engine.frame.optimize import LIGHT_SHARE, _checked, _drop_light

    def run(nodes):
        nodes = np.asarray(nodes, float)
        bars = np.array([(0, 3), (3, 2), (1, 4), (4, 2), (3, 4)])
        chains = np.array([0, 0, 1, 1, 2])
        loads = np.zeros((5, 3))
        loads[2, 2] = -100.0
        supports = np.array([0, 1])
        braced = np.isin(np.arange(5), [0, 1, 2])
        sizer = lambda b, c, s: _checked(nodes, b, c, s, Profile("round"), PRESETS["PLA"], 2.0, supports, loads,
                                         braced, 0.5, None, 0, 1.0)
        current, res, util, _ = sizer(bars, chains, np.full(3, 2.0))
        assert util[2] < LIGHT_SHARE
        return _drop_light(nodes, bars, chains, current, res, util, Profile("round"), braced & (loads[:, 2] != 0),
                           supports, sizer)
    out_bars, _, sizes, _, util, taken = run([(0, 0, 40), (40, 0, 40), (20, 0, 0), (10, 0, 20), (30, 0, 20)])
    assert taken == 1 and len(sizes) == 2 and util.max() <= 1.0
    assert {tuple(b) for b in out_bars} == {(0, 3), (3, 2), (1, 4), (4, 2)}
    _, _, _, _, _, taken = run([(0, 0, 0), (0, 0, 40), (60, 0, 20), (30, 0, 10), (30, 0, 30)])   # a wall bracket
    assert taken == 0


def test_holes_on_one_axis_are_held_round_their_walls():
    # two lugs on one axle (holes in line, the user's bracket_mid): their axis nodes are on one line too,
    # and no frame carried the loads; each hole is then held at nodes round its wall, and a frame is found
    space = F.subtract(F.box((0.0, 0.0, 0.0), (60.0, 20.0, 40.0)), F.bar((-40.0, 0.0, -12.0), (40.0, 0.0, -12.0), 4.0))
    lugs = np.concatenate([_tube(4.0, 6.0) + (x, 0.0, -12.0) for x in (-30.0, 24.0)])   # x -30..-24 and 24..30
    load = _tube(4.0, 20.0)[:, :, [1, 0, 2]] * (1.0, 1.0, 1.0) + (0.0, -10.0, 12.0)    # a hole along y, up top
    conds = Conditions((Condition("fixed", lugs),
                        Condition("load", load, load="bearing", amount=100.0, direction="+X")))
    g = G.build(space, conds, min_size=2.0)
    held = g.nodes[g.fixed.all(axis=1)]
    assert any("one axis" in n for n in g.notes) and not G._on_a_line(held, g.spacing)
    assert len(held) >= 6 and np.all(np.abs(np.hypot(held[:, 1], held[:, 2] + 12.0) - 4.0) > 0.5)   # in the material
    frame = optimize_frame(space, conds, PRESETS["PETG"], Profile("round"), min_size=2.0)
    assert len(frame.bars)


def _rect(origin, u, v):
    """Two triangles of the rectangle origin + s u + t v (0 <= s, t <= 1)."""
    o, u, v = (np.asarray(a, float) for a in (origin, u, v))
    return np.array([(o, o + u, o + u + v), (o, o + u + v, o + v)])


def test_automatic_spacing_gets_finer_when_a_load_gets_no_bar():
    # a 4 mm fin out of a block, loaded at its tip, held under the block: the automatic lattice (7.8 mm) has
    # no layer in the fin, so no bar reached the load (the table bracket's 8 mm plate, 13 mm). Automatic: a
    # finer lattice, and the log says so; a spacing that was set stays and says what to do
    block = F.box((20.0, 20.0, 2.0), (40.0, 40.0, 44.0))
    fin = F.box((69.0, 20.0, 2.0), (62.0, 40.0, 4.0))
    space = F.union(block, fin)
    conds = Conditions((Condition("fixed", _rect((0, 0, -20), (40, 0, 0), (0, 40, 0))),
                        Condition("load", _rect((100, 0, 0), (0, 40, 0), (0, 0, 4)), amount=100.0, direction="-Z")))
    auto = G.tidy(float(np.linalg.norm(np.subtract(*space.bounds[::-1]))) / G.AUTO_NODES)
    with pytest.raises(G.ConditionError, match=r"no bar reaches its point at \(100, 20, 2\) mm .* 7\.8 mm apart"):
        G.build(space, conds, auto, min_size=1.0)
    g = G.build(space, conds, min_size=1.0)
    assert g.spacing < auto and any("7.8 mm apart no bar reached load" in n for n in g.notes)
    assert (g.bars == np.flatnonzero(g.owner == 1)[0]).any()
    assert g.coarser == auto

    # Show: candidate nodes draws them: the fin without dots and the load's point no bar reaches at 7.8 mm, and
    # the finer lattice Optimize then uses
    from engine.graph.nodes_frame import _ground_view
    stuck = _ground_view(space, conds, auto, 1.0, 0.0, [])
    assert stuck["spacing"] == auto and not stuck["auto"] and "no bar reaches" in stuck["error"]
    assert [stuck["kinds"][i] for i in stuck["lonely"]] == ["load"]
    assert stuck["points"][stuck["lonely"][0]] == pytest.approx([100, 20, 2], abs=0.5)
    fin_dots = [p for p in stuck["lattice"] if p[0] > 40.0]
    assert not any(0.5 <= p[2] <= 3.5 for p in fin_dots)
    used = _ground_view(space, conds, 0.0, 1.0, 0.0, [])
    assert used["auto"] and used["coarser"] == auto and used["spacing"] == g.spacing and not used["lonely"]
    assert used["error"] is None and used["bars"] == len(g.bars) and len(used["lattice"]) == int((g.owner < 0).sum())
    assert set(used["kinds"]) == {"fixed", "load"} and len(used["points"]) == len(used["labels"])


def _end_conditions(length, w, h, amount=100.0, direction="-Z"):
    """A box design space 0..length x 0..w x 0..h held at x = 0 and loaded at x = length."""
    return Conditions((Condition("fixed", _rect((0, 0, 0), (0, w, 0), (0, 0, h))),
                       Condition("load", _rect((length, 0, 0), (0, w, 0), (0, 0, h)), amount=amount,
                                 direction=direction)))


def test_a_spacing_far_too_fine_is_refused_before_the_design_space_is_sampled():
    # 0.01 mm on a 60 mm block sampled the design space at a third of it first: 341 GB asked for; now the lattice
    # is counted first and the error says what spacing would do. A typed spacing never samples more than MAX_INSIDE
    import time
    space = F.box((30, 10, 15), (60, 20, 30))
    t = time.perf_counter()
    with pytest.raises(ConditionError, match=r"node spacing 0\.01 mm gives too many nodes here: use [\d.]+ mm or more"):
        G.build(space, _end_conditions(60, 20, 30), 0.01)
    assert time.perf_counter() - t < 5
    assert G._Inside(space, 0.002).values.size <= 1.1 * G.MAX_INSIDE


def test_min_size_thicker_than_the_design_space_is_refused():
    # a 1 mm plate: the lattice's 1.5 mm step missed it, the thickness is measured again round the deepest sample
    plate = F.box((30, 0.5, 15), (60, 1.0, 30))
    assert G._Inside(plate, 1.5).thickest() == pytest.approx(1.0, abs=0.05)
    with pytest.raises(ConditionError, match=r"Min size 2 mm: no bar that thick fits in the design space, which is at "
                                             r"most (0\.9\d|1) mm thick"):
        G.build(plate, _end_conditions(60, 1, 30), min_size=2.0)


def test_loads_on_their_own_support_or_cancelling_are_refused():
    # a load on the held face (clustered apart from the support, it became a 1.3 g frame of short bars beside it),
    # two that cancel (a frame that carried nothing) and a 0 N load: each says what it is
    space = F.box((30, 10, 15), (60, 20, 30))
    held, end = _rect((0, 0, 0), (0, 20, 0), (0, 0, 30)), _rect((60, 0, 0), (0, 20, 0), (0, 0, 30))
    load = lambda tri, amount=100.0, direction="-Z": Condition("load", tri, amount=amount, direction=direction)
    for conds, words in [((Condition("fixed", held), load(held)), "all of the load sits on Fixed Faces"),
                         ((Condition("fixed", held), load(end), load(end, direction="+Z")), "cancel each other out"),
                         ((Condition("fixed", held), load(end, 0.0)), "the loads are 0")]:
        with pytest.raises(ConditionError, match=words):
            G.build(space, Conditions(conds), 6.0)
    # half of a load on the support: that half goes straight into it, the note says so
    g = G.build(space, Conditions((Condition("fixed", held), load(np.concatenate([held, end])))), 6.0)
    assert any("50% of the load sits on Fixed Faces" in n for n in g.notes)
    assert np.abs(g.loads.sum(axis=0)) == pytest.approx([0, 0, 50.0], abs=1e-6)


def test_a_load_cut_off_by_a_neck_thinner_than_min_size_says_so():
    # two halves joined by a 1 mm neck: 2 mm bars cannot cross it (the LP said "a load not connected to a support?")
    space = F.union(F.union(F.box((14.5, 10, 15), (29, 20, 30)), F.box((45.5, 10, 15), (29, 20, 30))),
                    F.box((30, 10, 0.5), (4, 20, 1)))
    with pytest.raises(G.Unreached, match=r"no chain of bars inside the design space joins its point .* too thin for "
                                          r"bars of Min size 2 mm") as err:
        G.build(space, _end_conditions(60, 20, 30), 4.0, min_size=2.0)
    assert len(err.value.lonely) and err.value.ground is not None


def test_nodes_on_one_line_get_a_spacing_across_the_design_space():
    # a 300 x 10 x 10 mm bar: its automatic 20 mm lattice is one row down the middle, which carries no load across
    # it; automatic spacing gets finer until nodes lie across, a typed one says so; along the row it is fine
    space = F.box((150, 5, 5), (300, 10, 10))
    with pytest.raises(G.Flat, match=r"all candidate nodes lie on one line .*\(about 4 mm\)"):
        G.build(space, _end_conditions(300, 10, 10, 1.0), 20.0, min_size=2.0)
    g = G.build(space, _end_conditions(300, 10, 10, 1.0), min_size=2.0)
    assert g.spacing < 20.0 and any("all lay on one line" in n for n in g.notes)
    assert np.ptp(g.nodes[:, 1]) > 4.0 and np.ptp(g.nodes[:, 2]) > 4.0
    G.build(space, _end_conditions(300, 10, 10, 1.0, "+X"), 20.0, min_size=2.0)
    # 4 mm across: 2 mm bars fill it, no frame fits (1 mm apart they made an LP of 4 minutes and 3.9 GB); with 1 mm
    # bars it is a rod, 75 times as long as thick (1.5 mm apart: 6 GB)
    rod = F.box((150, 2, 2), (300, 4, 4))
    with pytest.raises(G.Flat, match=r"too thin for a frame of 2 mm bars \(that needs about 6 mm across\)"):
        G.build(rod, _end_conditions(300, 4, 4, 1.0), min_size=2.0)
    with pytest.raises(G.Flat, match=r"too slender for a frame \(75 times as long as it is thick\)"):
        G.build(rod, _end_conditions(300, 4, 4, 1.0), min_size=1.0)
    with pytest.raises(G.Flat, match=r"too thin for a frame of 2 mm bars"):   # (short: thin, not slender)
        G.build(F.box((20, 2, 2), (40, 4, 4)), _end_conditions(40, 4, 4, 1.0), 10.0, min_size=2.0)


def test_the_faces_of_a_small_part_are_reached_whatever_the_spacing():
    # a 5 mm cube with 2 mm bars: bars from a face node may leave the design space within 0.75 x Min size of it, as
    # far as a bar starting on the face needs to get 1 mm in (half a 0.58 mm spacing reached no face at all)
    space = F.box((2.5, 2.5, 2.5), (5, 5, 5))
    for spacing in (0.0, 0.4):
        g = G.build(space, _end_conditions(5, 5, 5), spacing, min_size=2.0)
        assert (g.free[g.owner >= 0] >= 1.5 - 1e-9).all()


def test_connect_to_a_body_out_of_reach_says_so():
    # beyond its grid the design space's distance grows (its edge value had put a body 500 mm away within reach)
    space = F.box((30, 10, 15), (60, 20, 30))
    g = G.build(space, _end_conditions(60, 20, 30), 6.0, attach=[_rect((500, 0, 0), (0, 20, 0), (0, 0, 30))])
    assert any("connect to #1: nothing of it lies within 6 mm" in n for n in g.notes)
    assert g.inside(np.array([[560.0, 10.0, 15.0]]))[0] > 450


def test_failing_bars_grow_at_least_two_percent_a_round():
    # a 0.01 mm Size step grew a bar 0.3 mm in 30 rounds and the frame did not pass; a 0.5 mm step is one step a
    # round as before for bars up to 25 mm
    from engine.frame.optimize import _grown
    assert _grown(2.0, 0.01, None, 0.0) == pytest.approx(2.04)
    assert [_grown(s, 0.5, None, 0.0) for s in (2.0, 4.5, 25.0)] == [2.5, 5.0, 25.5]
    assert _grown(30.0, 0.5, None, 0.0) == 31.0 and _grown(30.0, 0.5, None, 30.3) == 30.3
    assert _grown(3.0, 0.5, [3.0, 3.01, 4.0], 0.0) == 3.01                 # (a list: the next listed size)


def test_sizes_that_do_not_fit_the_design_space_are_refused():
    # the largest size whose section fits the thickest spot; a size step, list or profile height that cannot
    from engine.frame.optimize import _room_size
    round_bar = Profile("round")
    assert _room_size(round_bar, 20.0, 2.0, 0.5, None) == 20.0 and _room_size(round_bar, 20.3, 2.0, 0.5, None) == 20.0
    assert _room_size(round_bar, 20.0, 2.0, 0.5, [3.0, 15.0, 25.0]) == 15.0
    assert _room_size(round_bar, 0.0, 2.0, 0.5, None) == 0.0                # (not known)
    assert _room_size(Profile("rectangle", ratio=10.0), 20.0, 2.0, 0.5, None) == 20.0   # (turned: its 2 mm side)
    for args, words in [((round_bar, 20.0, 2.0, 50.0, None), r"Min size rounded up to the Size step \(50 mm\)"),
                        ((round_bar, 20.0, 2.0, 0.5, [100.0]), r"the smallest size in the list \(100 mm\)"),
                        ((Profile("rectangle", ratio=0.5), 20.0, 50.0, 0.5, None),
                         r"Min size \(50 mm, 25 mm across its thinner side\)")]:
        with pytest.raises(ConditionError, match=words + " does not fit in the design space, which is at most 20 mm"):
            _room_size(*args)


def test_a_hole_spread_around_takes_its_load_at_several_nodes_in_the_material():
    # the 600 N bearing load on the bolted bracket's bolt hole 8: one node on the axis, or (spread) three
    # or more on the side it presses, set into the material, carrying the same total
    from pathlib import Path
    from engine.graph.nodes import import_step
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_bolts.step"), 0.01, 1, 0.5)[0]
    ids = mesh.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", mesh.triangles[np.isin(ids, [5, 6, 7])]),
                        Condition("load", mesh.triangles[ids == 8], load="bearing", amount=600.0, direction="-Z")))
    space = F.body_from_mesh(mesh, 2.0)
    one = G.build(space, conds, min_size=2.0)
    spread = G.build(space, conds, min_size=2.0, spread=(1,))
    a, b = np.flatnonzero(one.owner == 1), np.flatnonzero(spread.owner == 1)
    assert len(a) == 1 and len(b) >= 3
    assert np.allclose(spread.loads[b].sum(axis=0), one.loads[a[0]], atol=1e-6)
    assert all(F.sample(space, p) < 0 for p in spread.nodes[b])                  # in the material
    assert (spread.nodes[b][:, 2] < 42).all()                                    # under the pin (it presses down)
    assert all((spread.bars == i).any() for i in b)                              # bars reach every one


def test_a_weak_body_by_a_hole_is_laid_out_again_with_it_spread():
    # `_spread_where_weak`: laid out again only when the weak spot is by a support's or load's faces; the
    # safer body is kept, and it stops when spreading does not help
    from engine.graph import nodes_frame as NF
    near = Condition("load", np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float), amount=10.0, direction="-Z")
    far = Condition("fixed", np.array([[[50, 0, 0], [51, 0, 0], [50, 1, 0]]], float))
    conds = Conditions((far, near))
    state = lambda s, at=(0.5, 0.3, 0.5): {"check": json.dumps({"safety": s, "at": list(at)}), "tag": s}
    calls = []

    def lay_out(spread=()):
        calls.append(spread)
        return {"spread": spread}

    for better, kept in ((1.8, 1.8), (1.2, 1.5)):
        calls.clear()
        out = NF._spread_where_weak(state(1.5), lay_out, lambda st: state(better, (30, 0, 0)), conds, 2.0, 2.0)
        assert calls == [(1,)] and out["tag"] == kept
    calls.clear()
    out = NF._spread_where_weak(state(1.5, (20, 20, 20)), lay_out, lambda st: state(1.9), conds, 2.0, 2.0)
    assert calls == [] and out["tag"] == 1.5                                      # weak away from any hole
    out = NF._spread_where_weak(state(2.1), lay_out, lambda st: state(1.9), conds, 2.0, 2.0)
    assert calls == [] and out["tag"] == 2.1                                      # safe enough already


# --- a whole run --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def block(tmp_path_factory):
    from engine.graph.nodes import import_step
    return import_step(_box_step(tmp_path_factory.mktemp("frame") / "block.step", (60, 20, 30)), 0.01, 1, 0.5)[0]


@pytest.mark.parametrize("move", [False, True])
def test_a_block_held_at_one_end_becomes_a_checked_truss(block, move):
    conds = Conditions((Condition("fixed", _face_triangles(block, 0, 0.0)),
                        Condition("load", _face_triangles(block, 0, 60.0), amount=200.0, direction="-Z")))
    space = F.body_from_mesh(block, 2.0)
    frame = optimize_frame(space, conds, PLA, Profile("round"), min_size=2.0, safety=2.0, spacing=5.0,
                           move_joints=move)
    r = frame.report
    if move:   # (joints moved off the lattice, the frame no heavier)
        assert r["moves"]["volume_after"] <= r["moves"]["volume_before"] * (1 + 1e-9)
    assert r["utilisation"] <= 1.0 + 1e-6 and r["buckling"] >= 2.0
    # all bars inside the design space (their centre lines), the reactions balance the load
    pts = frame.nodes[frame.bars].mean(axis=1)
    assert max(F.sample(space, p) for p in pts) < 0
    assert np.allclose(frame.check.reactions[:, :3].sum(axis=0), [0, 0, 200.0], rtol=1e-6)
    # sized bars weigh at least what the plastic layout needs, and not absurdly more
    assert frame.layout_volume * PLA.density / 1000 <= frame.mass < 20 * frame.layout_volume * PLA.density / 1000
    assert np.all(frame.sizes >= 2.0) and np.all(np.round(frame.sizes / 0.5, 6) % 1 == 0)
    assert r["seconds"] < 60


def test_max_size_keeps_every_bar_at_most_that_thick(block):
    # unlimited, the block's frame has 4.5 mm bars; at Max size 4 mm none is thicker and it still passes its check
    conds = Conditions((Condition("fixed", _face_triangles(block, 0, 0.0)),
                        Condition("load", _face_triangles(block, 0, 60.0), amount=200.0, direction="-Z")))
    space = F.body_from_mesh(block, 2.0)
    frame = optimize_frame(space, conds, PLA, Profile("round"), min_size=2.0, safety=2.0, spacing=5.0, max_size=4.0)
    assert frame.sizes.max() <= 4.0 and frame.report["utilisation"] <= 1.0 + 1e-6 and frame.report["buckling"] >= 2.0
    assert np.allclose(frame.check.reactions[:, :3].sum(axis=0), [0, 0, 200.0], rtol=1e-6)
    with pytest.raises(ConditionError, match="at least Min size"):
        optimize_frame(space, conds, PLA, Profile("round"), min_size=2.0, max_size=1.5)


def test_bars_at_max_size_that_fail_lay_the_frame_out_again_at_a_lower_share(monkeypatch):
    from engine.frame import optimize as FO
    # the check: a two-bar truss whose bars need more than Max size cannot pass (unlimited, it is made thicker)
    nodes = np.array([(0, 0, 0), (40, 0, 0), (0, 0, 30)], dtype=float)
    bars, chains = np.array([(0, 1), (2, 1)]), np.array([0, 1])
    loads = np.zeros((3, 3))
    loads[1, 2] = -1000.0
    braced = np.array([True, True, True])
    args = (nodes, bars, chains, np.array([2.0, 2.0]), Profile("round"), PLA, 2.0, [0, 2], loads, braced, 0.5, None,
            0, 1.0)
    assert FO._checked(*args)[0].max() > 3.0
    with pytest.raises(FO.MaxSizeShort, match="Max size"):
        FO._checked(*args, largest=3.0)
    # the run: each failing share is followed by the next; the frame says which share it took
    tried = []

    def attempt(*a, share, **k):
        tried.append(share)
        if share == FO.MAX_SIZE_SHARES[0]:
            raise FO.MaxSizeShort("bars at Max size (3 mm) do not pass the frame check")
        return FO.Frame(*[None] * 11, notes=[])
    monkeypatch.setattr(FO, "_frame", attempt)
    frame = FO.optimize_frame("space", "conditions", PLA, max_size=3.0)
    assert tried == list(FO.MAX_SIZE_SHARES[:2])
    assert any(f"at most {FO.MAX_SIZE_SHARES[1]:.0%} of what a 3 mm bar carries" in n for n in frame.notes)
    monkeypatch.setattr(FO, "_frame", lambda *a, share, **k: (_ for _ in ()).throw(FO.MaxSizeShort("short")))
    with pytest.raises(FO.MaxSizeShort):
        FO.optimize_frame("space", "conditions", PLA, max_size=3.0)


def test_bars_thicker_than_the_design_space_are_laid_out_again_within_it(monkeypatch):
    # safety factor 100 made 113 mm bars in a 20 mm block ("4.85 kg of bars"; the body was the block): bars at
    # the largest size that fits fail, and it is laid out again with that as Max size; if no frame then passes,
    # the error says the bars would have to be thicker than the design space
    from engine.frame import optimize as FO
    calls = []

    def attempt(*a, max_size=0.0, share=FO.MAX_SIZE_SHARES[0], **k):
        calls.append((max_size, share))
        if not max_size:
            raise FO.MaxSizeShort("bars at the design space's thickness (19.5 mm) do not pass", 19.5, 155.0, 20.0)
        return FO.Frame(*[None] * 11, notes=[])
    monkeypatch.setattr(FO, "_frame", attempt)
    frame = FO.optimize_frame("space", "conditions", PLA)
    assert calls == [(0.0, FO.MAX_SIZE_SHARES[0]), (19.5, FO.MAX_SIZE_SHARES[0])]
    assert any("bars of 155 mm were needed, more than fits the design space (at most 20 mm thick)" in n
               for n in frame.notes)
    monkeypatch.setattr(FO, "_frame", lambda *a, max_size=0.0, **k: (_ for _ in ()).throw(
        FO.MaxSizeShort("short", 19.5, 155.0, 20.0) if not max_size else FO.MaxSizeShort("short")))
    with pytest.raises(ConditionError, match=r"thicker than the design space: 155 mm where it is at most 20 mm thick "
                                             r"\(they would be cut off\), and no frame of bars up to 19\.5 mm passes"):
        FO.optimize_frame("space", "conditions", PLA)


def test_loads_no_frame_in_the_design_space_can_carry_are_refused_at_once():
    # 1e5 N on a 60 x 20 x 30 mm PLA block: the lightest truss needs ~10 times the block; it was laid out again at the
    # largest size that fits, three times, for 15 minutes, before it said no frame passes
    import time
    space = F.box((30, 10, 15), (60, 20, 30))
    t = time.perf_counter()
    with pytest.raises(ConditionError, match=r"the loads need at least [\d.]+ cm3 of material .* more than twice what "
                                             r"the design space holds \(3[5-7](\.\d)? cm3\)"):
        optimize_frame(space, _end_conditions(60, 20, 30, 1e5), PLA, Profile("round"), min_size=2.0, spacing=6.0)
    assert time.perf_counter() - t < 30


def test_a_tiny_load_gets_a_frame_of_the_smallest_bars():
    # 1e-9 N: the LPs' absolute tolerances took every force for noise and the joint moves crashed on no bars at all;
    # loads and allowed stress are scaled together for them, and the frame is all Min size bars
    space = F.box((30, 10, 15), (60, 20, 30))
    frame = optimize_frame(space, _end_conditions(60, 20, 30, 1e-9), PLA, Profile("round"), min_size=2.0,
                           spacing=10.0, mid_bar=False, move_joints=True)
    assert np.all(frame.sizes == 2.0) and frame.report["utilisation"] < 1e-6
    assert np.allclose(frame.check.reactions[:, :3].sum(axis=0), [0, 0, 1e-9], rtol=1e-6)


# --- geometry and the node -------------------------------------------------------------------------
def test_bars_are_an_exact_union_of_round_bars():
    # plain joints (no fillet): exactly the union of the capsules, where it matters (outside every
    # bar's box the kernel returns its margin)
    from engine.frame.geometry import frame_solid
    from engine.grid import Grid
    nodes = np.array([(0, 0, 0), (20, 0, 0), (20, 15, 5), (5, 10, 12)], dtype=float)
    bars = np.array([(0, 1), (1, 2), (2, 3), (0, 3)])
    d = np.array([3.0, 5.0, 2.0, 3.0])
    grid = Grid((-4, -4, -4), (24, 19, 16), 0.7)
    body = frame_solid(nodes, bars, np.arange(4), d, Profile("round"), "plain", "test")
    got = F.Context(grid).eval_full(body)
    P = np.stack(np.meshgrid(*grid.axes(), indexing="ij"), axis=-1).reshape(-1, 3)
    ref = np.full(len(P), np.inf)
    for (a, b), di in zip(bars, d):
        A, B = nodes[a], nodes[b]
        t = np.clip(((P - A) @ (B - A)) / ((B - A) @ (B - A)), 0, 1)
        ref = np.minimum(ref, np.linalg.norm(P - (A + t[:, None] * (B - A)), axis=1) - di / 2)
    near = ref < 2.0
    assert np.allclose(got.ravel()[near], ref[near], atol=1e-4) and (got.ravel()[~near] > 1.0).all()


def test_the_node_optimizes_on_request_and_previews_read_the_frame(block, tmp_path_factory):
    # previews never start the job; an "optimize" job does and saves the frame, which the next
    # preview (a new stamp) shows as a body: bars plus the skin at the supports, in the block
    from engine.graph import Evaluator
    from engine.server import build_run_graph
    step = _box_step(tmp_path_factory.mktemp("frame_node") / "block.step", (60, 20, 30))
    link = lambda ref: {"link": ref}
    recipe = {"version": 1, "nodes": {
        "import": {"type": "input.import_step", "inputs": {"path": step}},
        "body": {"type": "input.pick_body", "inputs": {"bodies": link("import.bodies")}},
        "pla": {"type": "sim.material", "inputs": {"preset": "PLA"}},
        "fixed": {"type": "sim.fixed", "inputs": {"cad": link("body.body"), "faces": str(_face_id(block, 0, 0.0))}},
        "load": {"type": "sim.load", "inputs": {"cad": link("body.body"), "faces": str(_face_id(block, 0, 60.0)),
                                                "amount": 200.0, "direction": "-Z", "add_to": link("fixed.conditions")}},
        "frame": {"type": "sim.frame", "inputs": {"space": link("body.body"), "material": link("pla.material"),
                                                  "conditions": link("load.conditions"), "spacing": 6.0}},
    }}

    def run(mode, stamp, optimize=None):
        graph, _, _, _ = build_run_graph(recipe, mode, "draft", ["frame"], optimize, stamp)
        events = []
        out = Evaluator(graph, workers=1, log=None, on_event=events.append).run(["frame"])["frame"]
        return events, out["body"]

    _, body = run("preview", 0)
    assert body.note.startswith("not optimized")
    events, body = run("optimize", 0, optimize="frame")
    assert any(e["type"] == "message" and "Frame Optimization done" in e["text"] for e in events)
    # while it works the editor is shown the frame after every step (live_frame), stage by stage
    from engine.frame.optimize import STAGES
    live = [e["frame"] for e in events if e["type"] == "live_frame"]
    stages = [f["stage"] for f in live]
    assert stages[0] == "layout" and "body check" in stages and "moving joints" in stages, stages
    assert [STAGES.index(s) for s in stages] == sorted(STAGES.index(s) for s in stages)
    for f in live:
        bars = np.asarray(f["bars"])
        assert bars.max() < len(f["nodes"]) and len(f["sizes"]) == max(f["chains"]) + 1 == len(f["lengths"])
        assert len(f["axial"]) == len(bars) and min(f["sizes"]) >= 2.0
    assert not live[0]["sized"] and live[-1]["sized"] and len(live[-1]["spot"]) == 3
    # ... with what each LP did (the calculation panel) and the bars' mass as drawn
    first = live[0]["numbers"]
    assert first["lp"] >= 1 and 0 < first["in_lp"] <= first["candidates"] and 0 < first["volume"] <= first["real"]
    assert all(f["mass"] > 0 for f in live)
    progress = [e for e in events if e["type"] == "progress" and e.get("step") in STAGES]
    assert [e["done"] for e in progress] == sorted(e["done"] for e in progress) and progress[0]["total"] == len(STAGES)
    events, body = run("preview", 1)
    assert "bars" in body.note and " g " in body.note
    # Optimize has nothing left to do: the card says so at once (and offers no Optimize that does nothing)
    assert body.finished.startswith("Laid out and sized for these settings")
    assert any(e["type"] == "finished" and e["text"] == body.finished for e in events)
    # the body itself is checked: it reaches the safety factor (2), or the node says it does not
    found = re.search(r"body \(medium Static Analysis\): safety factor (>?[\d.]+)", body.note)
    assert found, body.note
    warned = any(e["type"] == "message" and "the body reaches safety factor" in e["text"] for e in events)
    assert found.group(1).startswith(">") or float(found.group(1)) >= 2.0 or warned
    assert F.sample(body, (0.5, 10.0, 15.0)) < 0        # the skin on the held face
    from engine.grid import Grid
    inside = F.Context(Grid((0.5, 0.5, 0.5), (59.5, 19.5, 29.5), 1.0)).eval_full(body) < 0
    assert 0.01 < inside.mean() < 0.45                   # a frame, not the block (curved bars: ~35%)
    # Show: nothing for shape, else the body check's fields (safety factor = strength / stress)
    from engine.graph.core import REGISTRY
    assert REGISTRY["sim.frame"].paint == "result"
    shown = {}
    for show in ("shape", "stress", "safety factor"):
        recipe["nodes"]["frame"]["inputs"]["show"] = show
        graph, _, _, _ = build_run_graph(recipe, "preview", "draft", ["frame"], None, 1)
        shown[show] = Evaluator(graph, workers=1, log=None).run(["frame"])["frame"]["result"]
    assert shown["shape"] is None and shown["stress"].unit == "MPa"
    at = (55.0, 10.0, 15.0)                                 # in a bar near the loaded face
    stress = F.sample(shown["stress"], at)
    from engine.fea.analysis import SAFETY_CAP
    want = min(PRESETS["PLA"].strength / stress, SAFETY_CAP)          # (shown up to the cap)
    assert stress > 0.5 and F.sample(shown["safety factor"], at) == pytest.approx(want, rel=0.2)


def test_the_body_check_thickens_the_bars_at_the_most_stressed_spot(monkeypatch):
    # the beam model passed, the body (joints, bosses) did not: the bars meeting at its most stressed
    # spot get thicker until the body reaches the safety factor; the others stay as they were
    import json
    from engine.graph import nodes_frame as NF
    built = []
    monkeypatch.setattr(NF, "frame_body", lambda crop, nodes, bars, chains, sizes, *a, **k:
                        built.append(np.array(sizes)) or object())

    def analysis(body, conditions, material, detail):   # the bar at the spot: stress goes as 1 / size^3
        sf = (built[-1][1] / 2.0) ** 3
        return {"min_safety": sf, "max_away_at": (18.0, 0.0, 0.0), "max_displacement": 0.1}, f"safety factor {sf:.3g}", []
    monkeypatch.setattr(NF, "check_result", analysis)
    state = {"nodes": np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0]]), "bars": np.array([[0, 1], [1, 2]]),
             "chains": np.array([0, 1]), "beam_sizes": np.array([2.0, 2.0]),
             "report": np.array(json.dumps({"bars": 2, "pieces": 2, "joints": 0, "utilisation": 0.9,
                                            "deflection": 0.1, "buckling": 3.0}))}
    out = NF._size_for_body(None, state, (), PRESETS["PLA"], Profile(), {"skin": 2.0, "joints": "fillet", "fillet": 0.0},
                            "key", 2.0, 2.0, None, 0.5, 0)
    assert out["sizes"].tolist() == [2.0, 3.0]           # 2 x 1.02 x 2^(1/3) = 2.57, up to the next 0.5 mm
    assert len(built) == 2 and json.loads(str(out["check"]))["safety"] >= 2.0


@pytest.mark.parametrize("safeties, rounds, kept", [((1.24, 1.19, 1.40, 1.60), 4, 1.60),   # the wall bracket
                                                    ((1.20, 1.10, 1.15, 1.30), 3, 1.20)])
def test_the_body_check_takes_one_round_that_dips_and_keeps_the_safest(monkeypatch, safeties, rounds, kept):
    # thicker bars at the spot may first draw more load there (a dip), then pass; two rounds in a row
    # without gain stop it, and the safest body of all is kept
    import json
    from engine.graph import nodes_frame as NF
    built = []
    monkeypatch.setattr(NF, "frame_body", lambda crop, nodes, bars, chains, sizes, *a, **k:
                        built.append(np.array(sizes)) or object())
    monkeypatch.setattr(NF, "check_result", lambda *a: ({"min_safety": safeties[len(built) - 1], "max_away_at": (18.0, 0, 0),
                                                         "max_displacement": 0.1}, "", []))
    state = {"nodes": np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0]]), "bars": np.array([[0, 1], [1, 2]]),
             "chains": np.array([0, 1]), "beam_sizes": np.array([2.0, 2.0]),
             "report": np.array(json.dumps({"bars": 2, "pieces": 2, "joints": 0, "utilisation": 0.9,
                                            "deflection": 0.1, "buckling": 3.0}))}
    out = NF._size_for_body(None, state, (), PRESETS["PLA"], Profile(), {"skin": 2.0, "joints": "fillet", "fillet": 0.0},
                            "key", 1.43, 2.0, None, 0.5, 0)
    assert len(built) == rounds and json.loads(str(out["check"]))["safety"] == kept
    assert out["sizes"].tolist() == built[safeties.index(kept)].tolist()


def test_the_skin_is_the_exact_distance_to_the_faces():
    # a 10 x 10 face at z = 0 and a second, separate one: within the cap the field is distance - skin,
    # beyond it positive (no stray material: the open faces' signed field made sheets and specks)
    from engine.frame.geometry import skin_body
    from engine.grid import Grid
    square = lambda x0: np.array([[(x0, 0, 0), (x0 + 10, 0, 0), (x0 + 10, 10, 0)], [(x0, 0, 0), (x0 + 10, 10, 0), (x0, 10, 0)]])
    conds = Conditions((Condition("fixed", square(0.0)), Condition("load", square(30.0), amount=1.0)))
    grid = Grid((-5, -5, -5), (45, 15, 5), 0.5)
    got = F.Context(grid).eval_full(skin_body(conds, 2.0)).ravel()
    P = np.stack(np.meshgrid(*grid.axes(), indexing="ij"), axis=-1).reshape(-1, 3)
    dist = lambda x0: np.linalg.norm(P - np.clip(P, (x0, 0, 0), (x0 + 10, 10, 0)), axis=1)
    exact = np.minimum(dist(0.0), dist(30.0))
    near = exact < 3.0
    assert np.allclose(got[near], exact[near] - 2.0, atol=1e-5)
    assert (got[~near] > 0).all()


def _tube(r, length, segments=48, start=0.0, sweep=2 * np.pi, inward=True):
    """Triangles of a cylinder of radius r along x (0..length), as a CAD mesher makes them: corners on the
    surface; normals into it (a hole) or out (a pin's outside)."""
    th = start + np.linspace(0.0, sweep, segments + 1)
    ring = lambda x: np.column_stack([np.full(len(th), x), r * np.cos(th), r * np.sin(th)])
    a, b = ring(0.0), ring(length)
    tri = []
    for i in range(segments):
        quad = [(a[i], b[i], b[i + 1]), (a[i], b[i + 1], a[i + 1])]
        tri += [q if inward else q[::-1] for q in quad]   # (as listed, the normals point at the axis)
    return np.array(tri)


def test_a_hole_s_boss_is_the_exact_ring_round_its_fitted_cylinder():
    # a bolt hole's triangles gave a boss flat over each and round over each edge (its curvature jumped
    # every half millimetre); fitted, it is the ring r .. r + skin all round, and not capped within reach
    from engine.frame.geometry import _cylinder, skin_body
    from engine.grid import Grid
    hole = _tube(3.2, 8.0, sweep=np.radians(354), inward=True)   # (one segment left out, as a mesher may)
    cyl = _cylinder(hole)
    assert cyl is not None and cyl["ring"] and abs(abs(cyl["axis"][0]) - 1) < 1e-9 and abs(cyl["r"] - 3.2) < 1e-6
    body = skin_body(Conditions((Condition("fixed", hole),)), 2.0, reach=4.0)
    grid = Grid((1.0, -9.0, -9.0), (6.0, 9.0, 9.0), 0.25)
    got = F.Context(grid).eval_full(body)
    P = np.stack(np.meshgrid(*grid.axes(), indexing="ij"), axis=-1)
    rho = np.hypot(P[..., 1], P[..., 2])
    want = np.abs(rho - 3.2) - 2.0
    inside = want < 4.0                                          # (the cap: skin + reach)
    exact = inside & (rho >= 1.6)                                # (within half the radius of the axis: rounded)
    assert np.abs(got - want)[exact].max() < 1e-5 and (got[~inside] >= 4.0 - 1e-5).all()
    pin_side = _tube(4.0, 10.0, sweep=np.pi, inward=False)       # half a pin's outside: its edges stay
    half = _cylinder(pin_side)
    assert half is not None and not half["ring"] and abs(half["span"] - np.pi) < 1e-6
    assert _cylinder(np.array([[(0, 0, 0), (1, 0, 0), (0, 1, 0)]] * 8, float)) is None   # flat: not a cylinder


# --- supports the layout leaves out --------------------------------------------------------------
def test_a_left_out_support_is_joined_to_the_frame():
    # the 45 degree truss off the wall uses two wall nodes; the one in the middle of the wall is
    # joined by the shortest bars to a node of the frame that is not held (it then shares the load)
    g = _wall_ground()
    lay = LY.optimize(g, 1.0)
    middle = int(np.flatnonzero((g.nodes[:, 0] == 0) & (g.nodes[:, 2] == 0))[0])
    assert middle not in lay.ground_index
    joined, where = LY.connect(lay, g, [middle])
    assert middle in joined.ground_index and np.allclose(where[0], g.nodes[middle])
    assert len(joined.bars) > len(lay.bars) and np.all(joined.forces[len(lay.bars):] == 0)
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    n = len(joined.nodes)
    graph = coo_matrix((np.ones(len(joined.bars)), (joined.bars[:, 0], joined.bars[:, 1])), shape=(n, n))
    assert connected_components(graph, directed=False)[0] == 1        # one piece
    end = joined.bars[len(lay.bars):].ravel()
    assert (~joined.fixed[end].any(axis=1)).any()                      # it ends on a node that is not held


def test_unused_support_pieces_get_no_skin():
    from engine.frame.geometry import skin_body
    from engine.grid import Grid
    square = lambda x0: np.array([[(x0, 0, 0), (x0 + 10, 0, 0), (x0 + 10, 10, 0)], [(x0, 0, 0), (x0 + 10, 10, 0), (x0, 10, 0)]])
    conds = Conditions((Condition("fixed", np.concatenate([square(0.0), square(30.0)])),))
    body = skin_body(conds, 2.0, nodes=np.array([(5.0, 5.0, 0.0)]))   # a frame node on the first only
    v = F.Context(Grid((-5, -5, -5), (45, 15, 5), 0.5)).eval_full(body)
    assert v[:20].min() < 0 and v[60:].min() > 0                       # skin at x < 5, none at x >= 25


def test_every_bolt_of_the_bracket_gets_bars_also_curved_square_tubes():
    # the layout needs only three of the four bolts; the fourth is joined so it shares the load
    from pathlib import Path
    from engine.graph.nodes import import_step
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_bolts.step"), 0.01, 1, 0.5)[0]
    ids = mesh.face_attributes["cad_face"]
    pin = mesh.triangles[ids == 10]
    conds = Conditions((Condition("fixed", mesh.triangles[np.isin(ids, [5, 6, 7, 8])]),
                        Condition("load", pin, load="bearing", amount=200.0, direction="-Z"),
                        Condition("load", pin, amount=80.0, direction="+Y")))
    frame = optimize_frame(F.body_from_mesh(mesh, 2.0), conds, PRESETS["Nylon PA12"], Profile("square tube"),
                           curving=0.5)
    assert len(frame.supports) == 4
    assert frame.report["curved"] >= 1                     # (curved square tubes: checked as they are)
    degree = np.bincount(frame.bars.ravel(), minlength=len(frame.nodes))
    assert (degree[frame.supports] >= 1).all()
    assert any("was not needed by the layout" in n for n in frame.notes)
    assert frame.report["utilisation"] <= 1.0 + 1e-6 and frame.report["buckling"] >= 2.0
    assert np.abs(frame.check.reactions[frame.supports, :3]).sum(axis=1).min() > 1.0   # each one carries some


# --- profiles in the body, curving ----------------------------------------------------------------
@pytest.mark.parametrize("profile", [Profile("round tube"), Profile("square"), Profile("square tube"),
                                     Profile("rectangle", ratio=2.0), Profile("rectangle", ratio=0.5),
                                     Profile("rect tube", ratio=1.5), Profile("I"), Profile("angle")])
def test_profile_distance_encloses_the_section(profile):
    # the region distance < 0 has the section's area, its centroid at the bar's axis and its second
    # moments about the local axes (strong one about y: integral of z^2)
    size, n = 10.0, 900
    sec = profile.section(size)
    half = 0.8 * max(sec.width, sec.height)
    t = (np.arange(n) + 0.5) / n * 2 * half - half
    Y, Z = np.meshgrid(t, t, indexing="ij")
    inside = profile.distance(size, Y, Z) < 0
    da = (2 * half / n) ** 2
    assert inside.sum() * da == pytest.approx(sec.A, rel=0.01)
    assert abs((Y * inside).sum() * da) < 0.02 * sec.A * size and abs((Z * inside).sum() * da) < 0.02 * sec.A * size
    assert ((Z ** 2) * inside).sum() * da == pytest.approx(sec.Iy, rel=0.02)
    assert ((Y ** 2) * inside).sum() * da == pytest.approx(sec.Iz, rel=0.02)
    # on the outline it is 0: half the width out along y for a solid square
    if profile.kind == "square":
        assert float(profile.distance(size, np.array([5.0]), np.array([0.0]))[0]) == pytest.approx(0.0, abs=1e-12)


def test_a_path_of_bars_becomes_one_smooth_curve():
    # two bars through a free node: one curved bar from end to end, passing the node, with no kink
    # anywhere (the tangent turns a few degrees from piece to piece, also at the node)
    from engine.frame import curves as CV
    nodes = np.array([(0, 0, 0), (40, 0, 0), (70, 0, 30)], dtype=float)   # turns by 45 degrees at node 1
    bars = np.array([(0, 1), (1, 2)])
    fixed = np.zeros((3, 3), dtype=bool)
    fixed[0] = True
    loads = np.zeros((3, 3))
    loads[2] = (0, 0, -10)
    n2, b2, f2, c2, fx2, ld2 = CV.smooth_bars(nodes, bars, np.array([-5.0, -7.0]), np.array([0, 1]), fixed, loads, 1.0)
    assert len(np.unique(c2)) == 1 and sorted(set(f2)) == [-7.0, -5.0]     # one bar; each span keeps its force
    assert np.all(np.isclose(n2[:3], nodes))                               # it passes all three nodes
    d = n2[b2[:, 1]] - n2[b2[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    turns = np.degrees(np.arccos(np.clip([d[i] @ d[j] for i, j in _consecutive(b2)], -1, 1)))
    assert len(turns) == len(b2) - 1 and turns.max() < 6.0                 # smooth all along, also at node 1
    span0 = n2[np.unique(b2[f2 == -5.0])]
    assert np.abs(span0[:, 2]).max() > 1.0                                 # curved along its whole length
    assert np.all(n2[:, 1] == 0) and fx2.any(axis=1).sum() == 1 and np.abs(ld2).sum() == 10.0
    loaded = np.vstack([loads[:1], [(0, 0, 1)], loads[2:]])
    assert len(CV.smooth_bars(nodes, bars, np.array([-5.0, -7.0]), np.array([0, 1]), fixed, loaded, 1.0)[1]) == 2
    straight = CV.smooth_bars(nodes, bars, np.array([-5.0, -7.0]), np.array([0, 1]), fixed, loads, 1.0,
                              inside=lambda p: np.where(p[:, 2] < -0.5, 1.0, -10.0))   # no room below z = -0.5
    assert np.all(straight[0][:, 2] >= -0.5)                               # flatter where it would leave


def _consecutive(bars):
    """Pairs of pieces sharing a node, in walk order (piece i ends where piece j starts)."""
    out = []
    for i, (a, b) in enumerate(bars):
        for j, (c, d) in enumerate(bars):
            if i != j and b == c:
                out.append((i, j))
    return out


def test_square_bars_keep_their_orientation_round_a_curve_without_bumps():
    from engine.frame import curves as CV
    from engine.frame.geometry import frame_solid
    from engine.grid import Grid
    nodes = np.array([(0, 0, 0), (40, 0, 0), (60, 20, 0)], dtype=float)  # a bend in the x-y plane
    fixed = np.zeros((3, 3), dtype=bool)
    out = CV.smooth_bars(nodes, np.array([(0, 1), (1, 2)]), np.array([1.0, 1.0]), np.array([0, 1]), fixed,
                           np.zeros((3, 3)), 1.0)
    n2, b2, _, c2 = out[:4]
    z, forward = CV.orientation(n2, b2, c2)
    assert np.allclose(z, [0, 0, 1])                       # z stays up all the way round (no twist)
    body = frame_solid(n2, b2, c2, np.array([6.0]), Profile("square"), "plain", "bend")
    grid = Grid((-5, -10, -5), (65, 25, 5), 0.25)
    v = F.Context(grid).eval_full(body)
    # on the bar's top face (z = 3) along the whole curve: the field is 0, no bumps at the mitres
    for p in n2:
        assert F.sample(body, p + (0, 0, 3.0)) == pytest.approx(0.0, abs=0.03)
    assert v.min() == pytest.approx(-3.0, abs=0.13)


def test_a_bend_with_a_third_bar_curves_through_the_joint():
    # the two bars that go on most nearly straight become one curve through the node, where the
    # third bar still lands; the curve is smooth there (its tangent is the chord's direction)
    from engine.frame import curves as CV
    nodes = np.array([(0, 0, 0), (40, 0, 0), (70, 0, 30), (40, 0, -30)], dtype=float)
    bars = np.array([(0, 1), (1, 2), (1, 3)])
    fixed = np.zeros((4, 3), dtype=bool)
    fixed[[0, 3]] = True
    n2, b2, f2, c2, _, _ = CV.smooth_bars(nodes, bars, np.array([1.0, 2.0, -3.0]), np.array([0, 1, 2]), fixed,
                                             np.zeros((4, 3)), 1.0)
    joint = int(np.flatnonzero(np.all(np.isclose(n2, nodes[1]), axis=1))[0])
    at = np.flatnonzero((b2 == joint).any(axis=1))
    assert len(at) == 3                                     # two curve pieces and the third bar
    assert len(np.unique(c2[at])) == 2                      # the curve is one bar, the third another
    d = [n2[b[1]] - n2[b[0]] if b[0] == joint else n2[b[0]] - n2[b[1]] for b in b2[at]]
    curve = [x / np.linalg.norm(x) for x, c in zip(d, c2[at]) if (c2[at] == c).sum() == 2]
    assert curve[0] @ curve[1] < -np.cos(np.radians(7.0))         # smooth through the joint (pieces turn < 7 deg)


@pytest.mark.parametrize("kind", ["round tube", "square", "square tube", "rectangle", "rect tube", "I", "angle"])
def test_the_kernel_draws_each_profile_exactly(kind):
    # a straight bar along x: at its middle the kernel's value is the profile's own 2D distance
    from engine.frame.geometry import frame_solid
    from engine.grid import Grid
    prof = Profile(kind, ratio=1.5 if kind in ("rectangle", "rect tube", "I") else 1.0)
    body = frame_solid(np.array([(0, 0, 0), (60, 0, 0)], float), np.array([(0, 1)]), np.array([0]), np.array([10.0]),
                       prof, "plain", f"kernel-{kind}")
    grid = Grid((30, -12, -12), (30, 12, 12), 0.2)
    got = F.Context(grid).eval_full(body)[0]
    Yy, Zz = np.meshgrid(grid.axes()[1], grid.axes()[2], indexing="ij")
    z = np.array([0.0, 0.0, 1.0])
    ex = np.array([1.0, 0, 0])
    ey = np.cross(z, ex)                       # local y of a bar along x with z up: world -... as the frame check
    ref = prof.distance(10.0, Yy * ey[1], Zz)  # (ey is (0, 1, 0) here)
    near = np.abs(ref) < 3.0
    assert np.allclose(got[near], ref[near], atol=2e-4)


def test_fillets_join_different_bars_and_never_pieces_of_one_bar():
    # two round bars meeting at 90 degrees: the fillet adds material in the inner corner only; a
    # straight bar cut into two pieces of one bar gets no bump where they meet
    from engine.frame.geometry import frame_solid
    nodes = np.array([(0, 0, 0), (20, 0, 0), (20, 20, 0), (40, 0, 0)], float)
    corner = np.array([(0, 1), (1, 2)])
    plain = frame_solid(nodes, corner, np.array([0, 1]), np.array([4.0, 4.0]), Profile("round"), "plain", "c-plain")
    fill = frame_solid(nodes, corner, np.array([0, 1]), np.array([4.0, 4.0]), Profile("round"), "fillet", "c-fillet")
    inner = (17.4, 2.6, 0.0)                   # in the inner corner, just off both bars
    assert F.sample(plain, inner) > 0 and F.sample(fill, inner) < F.sample(plain, inner) - 0.1
    far = (8.0, 0.0, 2.0)                       # on bar 0's surface, far from the joint
    assert F.sample(fill, far) == pytest.approx(F.sample(plain, far), abs=1e-5)
    one = frame_solid(nodes, np.array([(0, 1), (1, 3)]), np.array([0, 0]), np.array([4.0]), Profile("round"),
                      "fillet", "one-bar")
    assert F.sample(one, (20.0, 0.0, 2.0)) == pytest.approx(0.0, abs=1e-5)  # on the surface at the seam


@pytest.mark.parametrize("continuity", ["G1", "G2", "G3"])
def test_bars_of_different_sizes_meet_without_a_seam(continuity):
    # five bars of three sizes at one node: blending only the two nearest, the fillet's size jumped where
    # the second and third nearest swapped (a torn seam); the field must not jump between samples
    from engine.frame.geometry import frame_solid
    from engine.grid import Grid
    ends = [(20, 0, 0), (0, 20, 0), (0, 0, 20), (-14, -14, 2), (-3, 12, -15)]
    nodes = np.array([(0, 0, 0)] + ends, float)
    bars = np.array([(0, i + 1) for i in range(len(ends))])
    body = frame_solid(nodes, bars, np.arange(len(ends)), np.array([7.0, 3.0, 2.0, 3.0, 2.0]), Profile("round"),
                       "fillet", "seam", continuity=continuity)
    voxel = 0.1
    steep = []
    for step, n in ((voxel, 141), (voxel / 2, 281)):   # a jump's step stays as the spacing halves: twice as steep
        v = F.Context(Grid.block(np.full(3, -7.0), step, (0, 0, 0), (n, n, n))).eval_full(body)
        steep.append(max(float(np.abs(np.diff(v, axis=a)).max()) for a in range(3)) / step)
    # (a bar fading out of the blend makes the field a little steeper than a distance, ~2.5 mm off the surface)
    assert steep[0] < 1.5 and steep[1] < steep[0] + 0.05


def test_g2_and_g3_fillets_are_the_smooth_union_of_that_order_one_size_up():
    # the kernel's blend is fields.bulge; G2 / G3 fillets are CONTINUITY_SIZE larger, so a joint keeps
    # about the material of a G1 one
    from engine.frame.geometry import CONTINUITY_SIZE, frame_solid
    from engine.grid import Grid
    nodes = np.array([(0, 0, 0), (20, 0, 0), (20, 20, 0)], float)
    one = lambda b, c, g: frame_solid(nodes, np.array([b]), np.array([0]), np.array([c]), Profile("round"), "plain",
                                      f"one-{c}", continuity=g)    # (its ends rounded alike)
    grid = Grid.block(np.array([12.0, -6.0, -4.0]), 0.25, (0, 0, 0), (45, 45, 33))
    for continuity, order in (("G2", 2), ("G3", 3)):
        both = frame_solid(nodes, np.array([(0, 1), (1, 2)]), np.array([0, 1]), np.array([4.0, 3.0]), Profile("round"),
                           "fillet", "pair", fillet=1.5, continuity=continuity)
        bar0, bar1 = one((0, 1), 4.0, continuity), one((1, 2), 3.0, continuity)
        alone = F.smooth_union(bar0, bar1, CONTINUITY_SIZE[order] * 1.5, order)
        a, b = F.Context(grid).eval_full(both), F.Context(grid).eval_full(alone)
        # (outside a piece's box a body is a constant, its own for each: compare where both bars are measured;
        # and where bar 1 is within its fillet size: farther, off the surface, it fades out of the kernel's blend)
        d1 = F.Context(grid).eval_full(bar1)
        near = (np.abs(a) < 1.0) & (F.Context(grid).eval_full(bar0) < 3.0) & (d1 < CONTINUITY_SIZE[order] * 1.5)
        assert near.sum() > 1000 and np.abs(a - b)[near].max() < 1e-5


def test_bar_curves_are_smooth_to_the_fourth_derivative_and_straight_at_their_ends():
    # a bent bar of several pieces is one quintic spline: its pieces agree up to the fourth derivative
    # where they meet (a polyline of capsules creased there), and at its ends the second and third
    # derivatives are 0, so the straight run past them joins smoothly
    from math import factorial
    from engine.frame.geometry import frame_pieces
    nodes = np.array([(0, 0, 0), (5, 1, 0), (10, 3, 1), (15, 6, 1), (20, 10, 2), (24, 15, 2)], float)
    bars = np.array([(i, i + 1) for i in range(5)])
    data = frame_pieces(nodes, bars, np.zeros(5, int), np.array([3.0]), Profile("round"), "fillet")
    c, a, b = data["curve"], data["a"], data["b"]
    h = np.linalg.norm(b - a, axis=1)

    def derivative(i, u, n):   # n-th derivative by the chord length at u of piece i
        return sum(c[i, j] * factorial(j) / factorial(j - n) * u ** (j - n) for j in range(n, 6)) / h[i] ** n
    nxt = {i: j for i in range(5) for j in range(5) if np.allclose(b[i], a[j])}
    assert len(nxt) == 4
    for i, j in nxt.items():
        for n in range(5):
            assert np.allclose(derivative(i, 1.0, n), derivative(j, 0.0, n), atol=1e-6 * 10 ** n)
    first, last = (set(range(5)) - set(nxt.values())).pop(), (set(range(5)) - set(nxt)).pop()
    for i, u in ((first, 0.0), (last, 1.0)):
        assert np.abs(derivative(i, u, 2)).max() < 1e-9 and np.abs(derivative(i, u, 3)).max() < 1e-9
    assert data["ends"][first, 0] == 1 and data["ends"][last, 1] == 1


def test_bar_ends_and_fading_have_the_continuity_of_their_order():
    # a derivative is continuous where its change across x shrinks with the step (it halves), and jumps
    # where it stays
    from engine.frame.geometry import fade, ramp

    def settles(f, x, n):
        change = [abs(_nth(f, x - 4 * e, n, e) - _nth(f, x + 4 * e, n, e)) for e in (1e-3, 5e-4)]
        return change[1] < 0.6 * change[0] + 1e-3
    for order, smooth_to in ((1, 1), (2, 2), (3, 3)):   # (the end is sqrt(rho^2 + ramp^2): its square counts)
        r = lambda s: ramp(s, 1.0, order) ** 2
        for edge in (-1.0, 1.0) if order > 1 else (0.0,):
            assert all(settles(r, edge, n) for n in range(1, smooth_to + 1))
            if smooth_to < 3 and edge >= 0:   # (where it turns into the half ball's s, the next one jumps)
                assert not settles(r, edge, smooth_to + 1)
    for edge in (1.0, 2.0):
        assert all(settles(fade, edge, n) for n in range(1, 4))


def _nth(f, x, n, e):
    """The n-th derivative of f at x by central differences of step e."""
    w = {1: [(-1, -0.5), (1, 0.5)], 2: [(-1, 1.0), (0, -2.0), (1, 1.0)],
         3: [(-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)]}[n]
    return sum(c * float(f(x + k * e)) for k, c in w) / e ** n


@pytest.mark.parametrize("continuity", ["G1", "G2", "G3"])
def test_the_kernel_rounds_ends_and_folds_bars_as_the_formulas_say(continuity):
    # a lone bar's end: sqrt(rho^2 + ramp(s)^2) - r; three bars of three sizes at one node: thickest
    # first, each blended into those before it with fields.bulge, its fillet faded out by its distance
    from engine.frame.geometry import CONTINUITY_SIZE, FILLET, fade, frame_solid, order_of, ramp
    from engine.grid import Grid
    order = order_of(continuity)
    lone = frame_solid(np.array([(0, 0, 0), (20, 0, 0)], float), np.array([(0, 1)]), np.array([0]), np.array([4.0]),
                       Profile("round"), "fillet", f"lone-{continuity}", continuity=continuity)
    grid = Grid.block(np.array([15.0, -4.0, -4.0]), 0.2, (0, 0, 0), (50, 41, 41))
    P = np.stack(np.meshgrid(*[grid.base[i] + grid.voxel * np.arange(grid.shape[i]) for i in range(3)],
                             indexing="ij"), -1)
    rho = np.sqrt(P[..., 1] ** 2 + P[..., 2] ** 2)
    want = np.sqrt(rho ** 2 + ramp(P[..., 0] - 20.0, 2.0, order) ** 2 + ramp(-P[..., 0], 2.0, order) ** 2) - 2.0
    inside = want < 3.5          # (farther, outside the bar's box, the kernel returns a constant)
    assert np.abs(F.Context(grid).eval_full(lone) - want)[inside].max() < 1e-4
    ends = [(20, 0, 0), (0, 18, 3), (-12, -10, 6)]
    nodes = np.array([(0, 0, 0)] + ends, float)
    widths = [3.0, 6.0, 2.0]
    body = frame_solid(nodes, np.array([(0, 1), (0, 2), (0, 3)]), np.arange(3), np.array(widths), Profile("round"),
                       "fillet", f"three-{continuity}", continuity=continuity)
    alone = [frame_solid(nodes, np.array([(0, i + 1)]), np.array([0]), np.array([widths[i]]), Profile("round"),
                         "fillet", f"one-{i}-{continuity}", continuity=continuity) for i in range(3)]   # (one bar)
    grid = Grid.block(np.full(3, -6.0), 0.2, (0, 0, 0), (61, 61, 61))
    got = F.Context(grid).eval_full(body)
    d = [F.Context(grid).eval_full(b).astype(np.float64) for b in alone]
    res = d[1]                                    # the 6 mm bar first, then 3, then 2
    for i in (0, 2):
        kk = FILLET * CONTINUITY_SIZE[order] * widths[i] * fade(d[i] / (FILLET * CONTINUITY_SIZE[order] * widths[i]))
        h = np.where(kk > 0, np.maximum(kk - np.abs(res - d[i]), 0) / np.maximum(kk, 1e-12), 0.0)
        res = np.minimum(res, d[i]) - F.bulge(h, kk, order)
    # (outside a bar's box, two of its fillets out, its own body is a constant)
    measured = np.all([x < 2 * FILLET * CONTINUITY_SIZE[order] * w for x, w in zip(d, widths)], axis=0)
    assert measured.sum() > 10000 and np.abs(got - res)[measured].max() < 1e-4


def test_fillet_profiles_have_the_continuity_they_are_named_for():
    # g(d) = smooth min - (a + b) / 2 as a function of d = a - b: G1 has a curvature jump where the blend
    # ends (|d| = k), G2 none there, but its third derivative jumps in the middle (d = 0); G3 neither
    k, e = 1.0, 1e-3
    for order in (1, 2, 3):
        g = lambda d: -np.abs(d) / 2 - F.bulge(np.maximum(1 - np.abs(d) / k, 0.0), k, order)
        second = lambda d: (g(d + e) - 2 * g(d) + g(d - e)) / e ** 2
        third = lambda d: (g(d + 2 * e) - 2 * g(d + e) + 2 * g(d - e) - g(d - 2 * e)) / (2 * e ** 3)
        jump2 = abs(second(k - 3 * e) - second(k + 3 * e))
        jump3_mid = abs(third(4 * e) - third(-4 * e))
        jump3_edge = abs(third(k - 4 * e) - third(k + 4 * e))
        if order == 1:
            assert jump2 > 0.4
        else:
            assert jump2 < 0.02
        assert (jump3_mid > 1.5) == (order == 2)
        if order > 1:   # (G1's third derivative is a spike at the edge, not a step)
            assert (jump3_edge > 0.5) == (order == 2)


def test_frame_blocks_are_bit_identical_to_the_whole_grid():
    # block meshing: a block of the grid must give exactly the values of the whole grid there
    from engine.frame.geometry import frame_solid
    from engine.grid import Grid
    nodes = np.array([(0, 0, 0), (30, 0, 5), (30, 20, 10), (5, 15, 20)], float)
    bars = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    body = frame_solid(nodes, bars, np.arange(5), np.array([3.0, 4.0, 3.5, 3.0, 5.0]), Profile("square tube"),
                       "fillet", "blocks")
    whole = Grid.block(np.array([-5.0, -5.0, -5.0]), 0.5, (0, 0, 0), (80, 60, 60))
    part = Grid.block(np.array([-5.0, -5.0, -5.0]), 0.5, (24, 12, 36), (20, 24, 12))
    a = F.Context(whole).eval_full(body)[24:44, 12:36, 36:48]
    b = F.Context(part).eval_full(body)
    assert np.array_equal(a, b)


# --- mid-bar joints, joining frames ----------------------------------------------------------------
def _box_at(path, lo, size):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from engine.graph.nodes import import_step
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(gp_Pnt(*lo), *size).Shape(), STEPControl_AsIs)
    writer.Write(str(path))
    return import_step(str(path), 0.01, 1, 0.5)[0]


@pytest.fixture(scope="module")
def two_blocks(tmp_path_factory):
    """Block A (0..40, 0..20, 0..24) and block B (40..80, 0..20, 0..24), touching at x = 40."""
    d = tmp_path_factory.mktemp("frames")
    return _box_at(d / "a.step", (0, 0, 0), (40, 20, 24)), _box_at(d / "b.step", (40, 0, 0), (40, 20, 24))


def _through_joints(frame):
    """Joints where a bar runs on through (two pieces of one bar meet where a third bar does)."""
    count = 0
    for v in range(len(frame.nodes)):
        at = np.flatnonzero((frame.bars == v).any(axis=1))
        if len(at) > 2 and len(np.unique(frame.chains[at])) < len(at):
            count += 1
    return count


def test_mid_bar_joints_off_means_bars_end_at_every_joint(block):
    conds = Conditions((Condition("fixed", _face_triangles(block, 0, 0.0)),
                        Condition("load", _face_triangles(block, 0, 60.0), amount=200.0, direction="-Z")))
    space = F.body_from_mesh(block, 2.0)
    off = optimize_frame(space, conds, PLA, Profile("round"), spacing=6.0, mid_bar=False)
    on = optimize_frame(space, conds, PLA, Profile("round"), spacing=6.0, mid_bar=True)
    assert _through_joints(off) == 0
    assert any("added along the bars used" in n for n in on.notes)
    for frame in (off, on):     # one size per bar, and the frame holds
        assert frame.report["utilisation"] <= 1 + 1e-6 and len(frame.sizes) == int(frame.chains.max()) + 1


def test_two_design_spaces_make_one_frame(two_blocks):
    # held at A's far end, loaded at B's: one frame across both, joined where they touch
    a, b = two_blocks
    conds = Conditions((Condition("fixed", _face_triangles(a, 0, 0.0)),
                        Condition("load", _face_triangles(b, 0, 80.0), amount=150.0, direction="-Z")))
    frame = optimize_frame(F.union(F.body_from_mesh(a, 2.0), F.body_from_mesh(b, 2.0)), conds, PLA, Profile("round"))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(frame.nodes)
    graph = coo_matrix((np.ones(len(frame.bars)), (frame.bars[:, 0], frame.bars[:, 1])), shape=(n, n))
    assert connected_components(graph, directed=False)[0] == 1
    assert (frame.nodes[:, 0] < 30).any() and (frame.nodes[:, 0] > 50).any()
    assert frame.report["utilisation"] <= 1 + 1e-6


def test_a_frame_connected_to_another_passes_its_load_into_it_and_joins(two_blocks):
    # frame A: held at x = 0, loaded at x = 40; frame B: loaded at x = 80, held only by frame A
    # (Connect to): it ends on A's bars and puts its whole load on A; Join Frames makes one part
    from engine.frame.geometry import join_body
    from engine.frame.optimize import FrameRef
    from engine.grid import Grid
    from engine.meshing import field_to_mesh
    a, b = two_blocks
    space_a, space_b = F.body_from_mesh(a, 2.0), F.body_from_mesh(b, 2.0)
    conds_a = Conditions((Condition("fixed", _face_triangles(a, 0, 0.0)),
                          Condition("load", _face_triangles(a, 0, 40.0), amount=100.0, direction="-Z")))
    fa = optimize_frame(space_a, conds_a, PLA, Profile("round"))
    ref_a = FrameRef(fa.nodes, fa.bars, fa.chains, fa.sizes, fa.profile, space_a, tuple(conds_a), 1.0, "frame-a")
    conds_b = Conditions((Condition("load", _face_triangles(b, 0, 80.0), amount=120.0, direction="-Z"),))
    fb = optimize_frame(space_b, conds_b, PLA, Profile("round"), attach=[ref_a])
    note = next(n for n in fb.notes if n.startswith("connect to #1"))
    push = np.array([float(x) for x in note.split("(")[1].split(")")[0].split(",")])
    assert np.allclose(push, [0, 0, -120.0], atol=0.5)             # all of B's load goes into A
    assert fb.report["utilisation"] <= 1 + 1e-6
    ref_b = FrameRef(fb.nodes, fb.bars, fb.chains, fb.sizes, fb.profile, space_b, tuple(conds_b), 1.0, "frame-b")
    body = join_body([ref_a, ref_b])
    grid = Grid((-1, -1, -1), (81, 21, 25), 0.5)
    mesh = field_to_mesh(F.Context(grid).eval_full(body).astype(np.float32), grid)
    solids = [p for p in mesh.split(only_watertight=False) if p.volume > 1.0]
    assert len(solids) == 1                                          # one part across both blocks
