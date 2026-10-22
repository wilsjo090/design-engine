"""Frame (truss) optimization: profiles, the layout LP and member adding against hand solutions,
the frame check against beam theory and Euler buckling, sizing, and a whole run on a box."""
import math
import re

import numpy as np
import pytest

from engine import fields as F
from engine.fea.materials import PRESETS
from engine.fea.model import Condition, Conditions
from engine.frame import beam, ground as G, layout as LY, sizing as S
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


def test_joint_length_gives_fewer_longer_bars():
    # with joint costs bars may pass over nodes (overlap); each bar then costs a little extra
    plain = LY.optimize(_wall_ground(L=6, H=3, reach=7), 1.0)
    costly = LY.optimize(_wall_ground(L=6, H=3, reach=7, overlap=True), 1.0, joint_length=3.0)
    assert len(costly.bars) < len(plain.bars)
    assert costly.lengths.mean() > plain.lengths.mean()
    assert costly.volume >= plain.volume - 1e-9


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


# --- a whole run --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def block(tmp_path_factory):
    from engine.graph.nodes import import_step
    return import_step(_box_step(tmp_path_factory.mktemp("frame") / "block.step", (60, 20, 30)), 0.01, 1, 0.5)[0]


def test_a_block_held_at_one_end_becomes_a_checked_truss(block):
    conds = Conditions((Condition("fixed", _face_triangles(block, 0, 0.0)),
                        Condition("load", _face_triangles(block, 0, 60.0), amount=200.0, direction="-Z")))
    space = F.body_from_mesh(block, 2.0)
    frame = optimize_frame(space, conds, PLA, Profile("round"), min_size=2.0, safety=2.0, spacing=5.0)
    r = frame.report
    assert r["utilisation"] <= 1.0 + 1e-6 and r["buckling"] >= 2.0
    # all bars inside the design space (their centre lines), the reactions balance the load
    pts = frame.nodes[frame.bars].mean(axis=1)
    assert max(F.sample(space, p) for p in pts) < 0
    assert np.allclose(frame.check.reactions[:, :3].sum(axis=0), [0, 0, 200.0], rtol=1e-6)
    # sized bars weigh at least what the plastic layout needs, and not absurdly more
    assert frame.layout_volume * PLA.density / 1000 <= frame.mass < 20 * frame.layout_volume * PLA.density / 1000
    assert np.all(frame.sizes >= 2.0) and np.all(np.round(frame.sizes / 0.5, 6) % 1 == 0)
    assert r["seconds"] < 60


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
    _, body = run("preview", 1)
    assert "bars" in body.note and " g " in body.note
    # the body itself is checked: it reaches the safety factor (2), or the node says it does not
    found = re.search(r"body \(medium Static Analysis\): safety factor (>?[\d.]+)", body.note)
    assert found, body.note
    warned = any(e["type"] == "message" and "the body reaches safety factor" in e["text"] for e in events)
    assert found.group(1).startswith(">") or float(found.group(1)) >= 2.0 or warned
    assert F.sample(body, (0.5, 10.0, 15.0)) < 0        # the skin on the held face
    from engine.grid import Grid
    inside = F.Context(Grid((0.5, 0.5, 0.5), (59.5, 19.5, 29.5), 1.0)).eval_full(body) < 0
    assert 0.01 < inside.mean() < 0.45                   # a frame, not the block (curved bars: ~35%)


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
