"""Topology optimization: the volume is held, stiffness improves, the result is black and white,
runs are repeatable and resumable, and fixed regions stay fixed."""
import numpy as np
import pytest

from engine import fields as F
from engine.fea import topology as T
from engine.fea.materials import PRESETS
from engine.fea.model import Condition, ConditionError, Conditions
from test_fea import _box_step, _face_id


@pytest.fixture(scope="module")
def box_step(tmp_path_factory):
    return _box_step(tmp_path_factory.mktemp("topo") / "box.step", (30, 10, 10))


@pytest.fixture(scope="module")
def box(box_step):
    """A 30 x 10 x 10 mm design space."""
    from engine.graph.nodes import import_step
    return import_step(box_step, 0.01, 1, 0.5)[0]


def _cantilever(box, h=1.0, **kw):
    ids = box.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", box.triangles[ids == _face_id(box, 0, 0.0)]),
                        Condition("load", box.triangles[ids == _face_id(box, 0, 30.0)], amount=50.0, direction="-Z")))
    kw = {"keep": 30.0, "skin": 1.0, **kw}
    return T.prepare(F.body_from_mesh(box, 2.0), conds, PRESETS["PLA"], h=h, **kw)


def test_volume_is_held_and_the_result_is_stiffer_and_black_and_white(box):
    p = _cantilever(box)
    opt = T.Optimizer(p)
    start = opt.design_compliance()                    # the uniform 30% start
    while not opt.done:
        c, v, _ = opt.step()
        assert v == pytest.approx(0.30, abs=0.005)
    assert opt.design_compliance() < 0.5 * start     # far stiffer than the uniform start
    d = opt.density()[np.asarray(p.domain) > 0]
    # densities end near 0 or 1, but for a rim at the members' faces: the filter radius is the min
    # member (3 elements here), so the filtered densities fall off over more elements (7% of this box)
    assert ((d > 0.1) & (d < 0.9)).mean() < 0.1
    # settled: the stiffness of the eroded design creeps for long in this small box (8% over the last 10
    # iterations when written), the design does not (4 of 3000 elements differed at 200 iterations)
    assert opt.history[-10][0] < 1.1 * opt.history[-1][0]


def test_members_thinner_than_min_thickness_carry_almost_nothing(box):
    # the stiffness is the eroded design's: at full sharpness a plate thinner than the filter radius
    # (= the min thickness) keeps only THIN_SHARE of its density, a thick one all of it in its middle
    p = _cantilever(box, min_thickness=3.0)
    assert p.radius == pytest.approx(3.0)
    opt = T.Optimizer(p)
    opt.beta = T.BETA_MAX
    x = np.zeros(p.grid.shape, dtype=p.dtype)
    x[8:10] = 1.0    # a 2 mm plate across the box
    x[18:22] = 1.0   # a 4 mm one
    xt, xb = opt.physical(x)
    carrying = np.asarray(opt.carrying(p.filter_empty(x), xb))
    assert np.asarray(xb)[9, 5, 5] > 0.9 and carrying[9, 5, 5] < T.THIN_SHARE + 0.01   # in the design, not carrying
    assert carrying[20, 5, 5] > 0.95


def _across(solid, *at):
    """Thickness in samples of the solid along the first axis through (y, z) = at."""
    return int(np.count_nonzero(solid[:, at[0], at[1]]))


def test_thin_members_are_thickened_to_min_thickness():
    from scipy import ndimage
    v, t = 0.25, 3.0                                      # samples 0.25 mm apart, 3 mm min thickness
    n = (80, 60, 60)
    smooth = lambda a: ndimage.gaussian_filter(a, 2.0)   # values run smoothly, as density_mesh's do
    values = np.zeros(n, np.float32)
    values[30:36, 5:55, 5:55] = 1.0                      # a plate (1.5 mm after the blur)
    values[50:75, 5:55, 5:55] = 1.0                      # a block, 6.25 mm
    values = smooth(values)
    out, added = T.thicken(values, 0.5, np.ones(n, bool), v, t)
    solid = out >= 0.5
    assert added > 0 and t <= _across(solid[:45], 30, 30) * v <= t + 2 * v
    assert _across(solid[45:], 30, 30) == _across(values[45:] >= 0.5, 30, 30)   # thick members stay as they are
    band = np.zeros(n, np.float32)
    band[0:40, 26:34, 5:55] = 1.0                        # a 2 mm band across a 5 mm wall (x 10..30)
    wall = np.zeros(n, bool)
    wall[10:30] = True
    out, _ = T.thicken(smooth(band), 0.5, wall, v, t)
    solid = (out >= 0.5) & wall
    assert np.count_nonzero(solid[10, :, 30]) * v >= t and np.count_nonzero(solid[20, :, 30]) * v >= t  # up to the faces


def test_smoothing_never_takes_a_member_away():
    # a strut of the thinnest member the optimizer makes (3 elements) between two blocks: blurring the
    # densities at Smoothing 5 took it away; blurring the distance to the surface keeps it, and with
    # the min thickness it stays at least that thick
    density = np.zeros((60, 18, 18), np.float32)
    density[:14] = 1.0
    density[46:] = 1.0
    density[14:46, 7:10, 7:10] = 1.0
    phi = np.ones_like(density)
    kept = T.density_mesh(density, phi, (0, 0, 0), 1.0, blur=5.0, min_thickness=3.0)
    assert len(kept.split(only_watertight=False)) == 1
    from engine import checks
    th = checks.wall_thickness(np.asarray(kept.vertices, np.float32), np.asarray(kept.faces))
    middle = np.abs(kept.vertices[:, 0] - 30) < 8          # on the strut
    assert np.percentile(th[middle & np.isfinite(th)], 5) >= 2.9


def test_a_new_strength_or_density_keeps_the_optimized_result(box):
    from engine.fea.materials import Material
    ids = box.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", box.triangles[ids == _face_id(box, 0, 0.0)]),
                        Condition("load", box.triangles[ids == _face_id(box, 0, 30.0)], amount=50.0, direction="-Z")))
    space, pla = F.body_from_mesh(box, 2.0), PRESETS["PLA"]
    key = lambda m: T.topology_key(space, conds, m, 30.0, 0.0, 1.0, None, None, "draft")
    assert key(pla) == key(Material("PLA, other spool", pla.E, pla.nu, 1.30, 40.0))
    assert key(pla) != key(Material("stiffer", 2 * pla.E, pla.nu, pla.density, pla.strength))


def test_the_skin_is_never_thinner_than_min_thickness(box):
    p = _cantilever(box, skin=1.0, min_thickness=3.0)
    solid = np.asarray(p.solid)
    assert solid[:3].all() and not solid[3].any()   # 3 mm at the fixed end (elements of 1 mm)
    assert any("skin 1 mm made as thick as the min thickness" in n for n in p.notes)


def test_a_solid_wall_thinner_than_max_thickness_is_not_too_full(box):
    # outside the design space counts as empty: the whole 10 mm box, solid, is thinner than 12 mm
    # (counted as a share of the design space it was full, and every thin wall got holes)
    p = _cantilever(box, max_thickness=12.0)
    assert float(np.max(p.local(np.ones(p.grid.shape, dtype=p.dtype)))) < T.LOCAL_SHARE


def test_big_filters_by_fft_match_the_direct_ones():
    rng = np.random.default_rng(1)
    a = rng.random((30, 12, 20))
    o = np.arange(-5, 6)
    kernel = np.maximum(5.5 - np.sqrt(o[:, None, None] ** 2 + o[None, :, None] ** 2 + o[None, None, :] ** 2), 0)
    assert kernel.size >= T.FFT_TAPS
    assert np.allclose(T._convolve(np, kernel.size)(a, kernel), T._convolve(np, 0)(a, kernel), atol=1e-9)


def test_same_inputs_give_the_same_design(box):
    a, b = T.Optimizer(_cantilever(box)), T.Optimizer(_cantilever(box))
    for _ in range(8):
        a.step()
        b.step()
    assert np.array_equal(np.asarray(a.x), np.asarray(b.x))


@pytest.mark.parametrize("max_thickness", [0.0, 4.5])
def test_a_stopped_run_continues_exactly(box, max_thickness):
    p = _cantilever(box, max_thickness=max_thickness)
    straight = T.Optimizer(p)
    for _ in range(14):
        straight.step()
    first = T.Optimizer(p)
    for _ in range(9):
        first.step()
    state = {k: np.array(v) for k, v in first.state(stopped=True).items()}   # as saved and loaded
    assert "power_0" in state   # the smoothers' warm lmax estimates go on from the same vectors
    resumed = T.Optimizer(p, state)
    for _ in range(5):
        resumed.step()
    assert resumed.iteration == 14 and resumed.beta == straight.beta
    assert np.allclose(np.asarray(resumed.x), np.asarray(straight.x), atol=1e-6)


def test_max_thickness_keeps_every_neighbourhood_open(box):
    # at 55% the free design has a region fuller than LOCAL_SHARE (0.65 when written: outside the
    # 10 mm box counts as empty, so leaner designs never are); with a max thickness no neighbourhood
    # of that size is, and the volume is still Keep %
    p = _cantilever(box, keep=55.0, max_thickness=4.5)
    fullest = lambda d: float(np.max(p.local(d)[np.asarray(p.free)]))
    free_run = T.Optimizer(_cantilever(box, keep=55.0)).run()
    capped = T.Optimizer(p).run()
    d = capped.density()
    assert fullest(free_run.density()) > T.LOCAL_SHARE
    assert fullest(d) <= 1.02 * T.LOCAL_SHARE
    assert p.volume(d) == pytest.approx(0.55, abs=0.005)
    assert capped.design_compliance() < 1.5 * free_run.design_compliance()   # about as stiff


def test_struts_only_leaves_no_room_for_plates(box):
    # max thickness measured on round struts: plates may be at most 0.62 of it, and members must be
    # at least 0.65 of it (the filter radius is the thinnest member), so no plate fits
    from scipy.special import erfinv
    t = 6.0
    p = _cantilever(box, max_thickness=t, members="struts only")
    sigma = p.local_sigma * p.grid.h
    assert 1 - np.exp(-t * t / (8 * sigma * sigma)) == pytest.approx(T.LOCAL_SHARE, rel=1e-3)  # a strut of t fills 0.6
    widest_plate = 2 * np.sqrt(2) * sigma * erfinv(T.LOCAL_SHARE)
    assert widest_plate < p.radius == pytest.approx(T.STRUT_MIN * t)
    plates = _cantilever(box, max_thickness=t)   # plates allowed: measured on a plate, a wider neighbourhood
    assert plates.radius < p.radius and plates.local_sigma > p.local_sigma


def test_too_thin_a_max_thickness_is_raised_with_a_note(box):
    p = _cantilever(box, max_thickness=1.0)   # 1 mm elements: members are at least ~3 mm
    assert any("max thickness 1 mm is too thin" in n for n in p.notes)
    assert p.local_sigma * p.grid.h / T.LOCAL_SIGMA == pytest.approx(4.5)


def test_skin_keep_and_avoid_are_respected(box):
    keep = F.sqrt((F.X - 15) * (F.X - 15) + (F.Y - 5) * (F.Y - 5) + (F.Z - 9) * (F.Z - 9)) - 1.6   # a ball near the top
    avoid = F.sqrt((F.X - 15) * (F.X - 15) + (F.Y - 5) * (F.Y - 5) + (F.Z - 2) * (F.Z - 2)) - 2.5  # one near the bottom
    p = _cantilever(box, keep_body=keep, avoid_body=avoid)
    opt = T.Optimizer(p)
    for _ in range(6):
        opt.step()
    d = opt.density()
    solid, phi = np.asarray(p.solid), np.asarray(p.phi)
    assert solid.any() and np.all(d[solid] == 1.0)
    assert d[0, :, :].min() == 1.0                    # the skin at the fixed end (x = 0..1 mm)
    centre = lambda x, y, z: (int(x / p.grid.h), int(y / p.grid.h), int(z / p.grid.h))
    assert d[centre(15, 5, 9)] == 1.0 and phi[centre(15, 5, 2)] == 0.0 and d[centre(15, 5, 2)] == 0.0


def test_the_node_optimizes_only_on_request_and_previews_read_the_result(box_step, box, monkeypatch):
    # previews never start the minutes-long job; an "optimize" job does (showing shapes as it goes)
    # and saves the result, which the next preview (a new stamp) reads
    from engine.fea import model
    from engine.graph import Evaluator
    from engine.server import build_run_graph
    monkeypatch.setitem(model.DETAIL, "draft", (3000, 400_000))  # 1 mm elements: a quick test
    link = lambda ref: {"link": ref}
    recipe = {"version": 1, "nodes": {
        "import": {"type": "input.import_step", "inputs": {"path": box_step}},
        "body": {"type": "input.pick_body", "inputs": {"bodies": link("import.bodies")}},
        "pla": {"type": "sim.material", "inputs": {"preset": "PLA"}},
        "fixed": {"type": "sim.fixed", "inputs": {"cad": link("body.body"), "faces": str(_face_id(box, 0, 0.0))}},
        "load": {"type": "sim.load", "inputs": {"cad": link("body.body"), "faces": str(_face_id(box, 0, 30.0)),
                                                "amount": 50.0, "direction": "-Z", "add_to": link("fixed.conditions")}},
        "topo": {"type": "sim.topology", "inputs": {"space": link("body.body"), "material": link("pla.material"),
                                                    "conditions": link("load.conditions"), "skin": 1.0}},
    }}

    def run(mode, stamp, optimize=None):
        graph, targets, _, _ = build_run_graph(recipe, mode, "draft", ["topo"], optimize, stamp)
        events = []
        out = Evaluator(graph, workers=1, log=None, on_event=events.append).run(["topo"])["topo"]
        return graph, events, out

    graph, events, out = run("preview", 0)
    assert graph.nodes["topo"]["inputs"]["compute"] == 0
    assert out["body"].note.startswith("not optimized")
    graph, events, out = run("optimize", 0, optimize="topo")
    assert graph.nodes["topo"]["inputs"]["compute"] == 1
    assert any(e["type"] == "live_view" for e in events)
    assert any(e["type"] == "message" and "done in" in e["text"] for e in events)
    graph, events, out = run("preview", 1)
    assert "iterations" in out["body"].note and not out["body"].note.startswith("stopped")
    assert "safety factor" in out["body"].note and "max deflection" in out["body"].note   # the shape analysed
    assert F.sample(out["body"], (0.5, 5.0, 5.0)) < 0     # solid at the clamped end
    _, targets, _, _ = build_run_graph(recipe, "optimize", "draft", [], "topo", 2)
    assert targets == ["topo"]


def test_ball_bar_and_box_are_exact_distances():
    from engine.graph.core import REGISTRY, GraphError
    assert F.sample(F.ball((1, 2, 3), 2), (1, 2, 6)) == pytest.approx(1.0)
    assert F.sample(F.ball((1, 2, 3), 2), (1, 2, 3)) == pytest.approx(-2.0)
    rod = F.bar((0, 0, 0), (10, 0, 0), 1)
    assert F.sample(rod, (5, 3, 0)) == pytest.approx(2.0) and F.sample(rod, (13, 0, 0)) == pytest.approx(2.0)
    assert F.sample(F.box((0, 0, 0), (10, 6, 4)), (8, 0, 0)) == pytest.approx(3.0)
    assert F.sample(F.box((0, 0, 0), (10, 6, 4)), (0, 0, 0)) == pytest.approx(-2.0)
    assert F.sample(F.box((0, 0, 0), (10, 6, 4), 1), (6, 4, 0)) == pytest.approx(np.sqrt(8) - 1, abs=1e-5)  # rounded
    assert np.allclose(F.bar((0, 0, 0), (10, 0, 0), 1).bounds, [(-1, -1, -1), (11, 1, 1)])
    ball = REGISTRY["implicit.ball"].fn(centre="1, 2, 3", radius=2)
    assert F.sample(ball, (1, 2, 6)) == pytest.approx(1.0)
    with pytest.raises(GraphError, match="three numbers"):
        REGISTRY["implicit.bar"].fn(start="1, 2", end="0, 0, 0", radius=1)
    with pytest.raises(GraphError, match="more than 0"):
        REGISTRY["implicit.box"].fn(centre="0, 0, 0", size="10, 0, 4", rounding=0)


def test_prefer_puts_material_in_a_guide_and_a_negative_one_keeps_it_out(box):
    # the cantilever's middle (x 15, centre of the cross section) is empty without a guide, the top flange
    # there full; a Prefer ball fills the middle (Keep % held), a negative one empties the flange
    run = lambda **kw: T.Optimizer(_cantilever(box, **kw)).run()
    plain = run().density()
    assert plain[15, 5, 5] < 0.1 and plain[14:17, 4:7, 8:10].mean() > 0.9
    guided = run(prefer_body=F.ball((15, 5, 5), 1.5), prefer_strength=0.5)
    assert guided.density()[14:17, 4:7, 4:7].mean() > 0.9
    assert guided.p.volume(guided.density()) == pytest.approx(0.30, abs=0.005)
    kept_out = run(prefer_body=F.ball((15, 5, 9), 1.5), prefer_strength=-1.0).density()
    assert kept_out[14:17, 4:7, 8:10].mean() < 0.1
    far = _cantilever(box, prefer_body=F.ball((100, 5, 5), 2), prefer_strength=1.0)
    assert far.prefer is None and any("outside the design space" in n for n in far.notes)
    args = (F.body_from_mesh(box, 2.0), Conditions(()), PRESETS["PLA"], 30, 0, 1, None, None, "draft")
    ball = F.ball((15, 5, 5), 1.5)
    assert T.topology_key(*args) == T.topology_key(*args, prefer_body=ball, prefer_strength=0.0)
    assert T.topology_key(*args) != T.topology_key(*args, prefer_body=ball, prefer_strength=0.5)


def test_designs_move_to_a_finer_grid_exactly_where_they_are_linear():
    from types import SimpleNamespace
    old = SimpleNamespace(origin=np.array([0.0, 0.0, 0.0]), h=2.0, shape=(8, 6, 5))
    new = SimpleNamespace(origin=np.array([0.0, 0.0, 0.0]), h=0.8, shape=(20, 15, 12))
    centres = lambda g: np.meshgrid(*[g.origin[a] + (np.arange(g.shape[a]) + 0.5) * g.h for a in range(3)],
                                    indexing="ij")
    f = lambda x, y, z: 0.02 * x + 0.03 * y - 0.01 * z + 0.4
    moved = T._transfer(f(*centres(old)).astype(np.float32), old, new)
    x, y, z = centres(new)
    inner = (x > 1) & (x < 15) & (y > 1) & (y < 11) & (z > 1) & (z < 9)   # between the old centres
    assert np.allclose(moved[inner], f(x, y, z)[inner], atol=1e-6)


def test_a_coarse_start_runs_the_coarser_details_that_hold_the_max_thickness(box):
    from engine.fea.analysis import plan_grid
    space, volume = F.body_from_mesh(box, 2.0), 3000.0
    assert T.levels(space, "fine", True, volume) == ["draft", "medium", "fine"]
    assert T.levels(space, "fine", False, volume) == ["fine"] and T.levels(space, "draft", True, volume) == ["draft"]
    h = {d: plan_grid(space, d, volume).h for d in ("draft", "medium")}
    between = 0.5 * (T._thinnest_max(h["draft"], 0.0) + T._thinnest_max(h["medium"], 0.0))
    assert T.levels(space, "fine", True, volume, max_thickness=between) == ["medium", "fine"]   # draft would raise it
    args = (space, Conditions(()), PRESETS["PLA"], 30, 0, 1, None, None)
    assert T.levels(space, "medium", True, volume) == ["medium"]     # (quick anyway: nothing gained)
    for detail in ("draft", "medium"):    # results saved before stay valid
        assert T.topology_key(*args, detail) == T.topology_key(*args, detail, coarse=True)
    assert T.topology_key(*args, "fine") != T.topology_key(*args, "fine", coarse=True)


def test_a_coarse_level_hands_its_design_over(box):
    coarse = T.Optimizer(_cantilever(box, h=2.0)).run(hand_over=4.0)
    assert coarse.beta == 4.0 and not coarse.done
    fine = T.Optimizer(_cantilever(box, h=1.0)).take_over(coarse.handover())
    assert fine.iteration == coarse.iteration and fine.beta == 4.0 and fine.since_beta == 0
    _, v, _ = fine.step()                   # (moved, the design is 25% here: the next update holds Keep %)
    assert v == pytest.approx(0.30, abs=0.005)
    fine.run()
    direct = T.Optimizer(_cantilever(box, h=1.0)).run()
    # on 2 mm elements the 10 mm box is 5 across, its layout cruder than the direct run's (1.25 x the
    # compliance when written); the brackets, 20 and more across, came out as stiff (FINDINGS.md)
    assert fine.design_compliance() < 1.35 * direct.design_compliance()


def test_the_node_starts_coarse_and_continues_after_a_stop(box_step, box, monkeypatch):
    # fine on 1 mm elements, starting on draft's 2 mm and medium's 1.4 mm ones; stopped on draft, Optimize
    # continues there
    from engine.fea import model
    from engine.graph import Evaluator, nodes_fea
    from engine.graph.core import Cancelled
    from engine.server import build_run_graph
    monkeypatch.setitem(model.DETAIL, "draft", (375, 400_000))
    monkeypatch.setitem(model.DETAIL, "medium", (1000, 400_000))
    monkeypatch.setitem(model.DETAIL, "fine", (3000, 400_000))
    link = lambda ref: {"link": ref}
    recipe = {"version": 1, "nodes": {
        "import": {"type": "input.import_step", "inputs": {"path": box_step}},
        "body": {"type": "input.pick_body", "inputs": {"bodies": link("import.bodies")}},
        "pla": {"type": "sim.material", "inputs": {"preset": "PLA"}},
        "fixed": {"type": "sim.fixed", "inputs": {"cad": link("body.body"), "faces": str(_face_id(box, 0, 0.0))}},
        "load": {"type": "sim.load", "inputs": {"cad": link("body.body"), "faces": str(_face_id(box, 0, 30.0)),
                                                "amount": 50.0, "direction": "-Z", "add_to": link("fixed.conditions")}},
        "topo": {"type": "sim.topology", "inputs": {"space": link("body.body"), "material": link("pla.material"),
                                                    "conditions": link("load.conditions"), "skin": 1.0,
                                                    "detail": "fine"}},
    }}

    def run(mode, stamp, optimize=None):
        graph, _, _, _ = build_run_graph(recipe, mode, "draft", ["topo"], optimize, stamp)
        events = []
        out = Evaluator(graph, workers=1, log=None, on_event=events.append).run(["topo"])["topo"]
        return events, out

    progress = nodes_fea.report_progress

    def stop_at_10(done, total):
        if done == 10:
            raise Cancelled("stopped")
        progress(done, total)

    monkeypatch.setattr(nodes_fea, "report_progress", stop_at_10)
    with pytest.raises(Cancelled):
        run("optimize", 0, optimize="topo")
    monkeypatch.setattr(nodes_fea, "report_progress", progress)
    _, out = run("preview", 1)
    assert out["body"].note.startswith("stopped at iteration 10 on draft elements")
    events, out = run("optimize", 2, optimize="topo")
    texts = [e["text"] for e in events if e["type"] == "message"]
    assert any("continuing at iteration 10" in t for t in texts)
    assert any(t.startswith("draft done at iteration") for t in texts)
    assert any(t.startswith("medium done at iteration") for t in texts)
    assert any("(fine, " in t and "on finer elements" in t for t in texts)
    _, out = run("preview", 3)
    assert "iterations" in out["body"].note and not out["body"].note.startswith("stopped")
    assert F.sample(out["body"], (0.5, 5.0, 5.0)) < 0     # solid at the clamped end


def test_a_bearing_load_keeps_skin_only_on_its_contact_side():
    # the pin pulls down in the wall bracket's hole (x = 50, z = 40, radius 4, along Y): the solid
    # skin stays below the hole; above it the optimizer decides (it was kept all round before)
    from pathlib import Path
    from engine.graph.nodes import import_step
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_space.step"), 0.01, 1, 0.5)[0]
    ids = mesh.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", mesh.triangles[ids == 0]),
                        Condition("load", mesh.triangles[ids == 6], load="bearing", amount=200.0, direction="-Z")))
    p = T.prepare(F.body_from_mesh(mesh, 2.0), conds, PRESETS["PETG"], keep=25.0, skin=2.0, h=1.0)
    solid = np.asarray(p.solid)
    cell = lambda x, y, z: tuple(int(v) for v in (np.array([x, y, z]) - p.grid.origin) // p.grid.h)
    assert solid[cell(50.5, 15.5, 34.5)]          # 1.5 mm under the hole
    assert not solid[cell(50.5, 15.5, 45.5)]      # 1.5 mm over it
    force_only = Conditions((conds[0], Condition("load", conds[1].triangles, amount=200.0, direction="-Z")))
    q = T.prepare(F.body_from_mesh(mesh, 2.0), force_only, PRESETS["PETG"], keep=25.0, skin=2.0, h=1.0)
    assert np.asarray(q.solid)[cell(50.5, 15.5, 45.5)]   # a plain force keeps the whole ring
    assert T.topology_key(F.body_from_mesh(mesh, 2.0), conds, PRESETS["PETG"], 25, 0, 2, None, None, "draft") != \
        T.topology_key(F.body_from_mesh(mesh, 2.0), force_only, PRESETS["PETG"], 25, 0, 2, None, None, "draft")


def test_a_gap_keeps_material_off_the_design_space_faces(box):
    # 2 mm on 1 mm elements: the two outer layers stay empty, except within skin + gap (3 mm) of the
    # fixed and loaded ends; the volume is still 30% of the whole box. Less stiff (2.5 x the
    # compliance when written: in a 10 mm beam the outer 2 mm are the best place for material)
    p = _cantilever(box, gap=2.0)
    gapped, free = T.Optimizer(p).run(), T.Optimizer(_cantilever(box)).run()
    d = gapped.density()
    outer = np.ones((10, 10), dtype=bool)
    outer[2:-2, 2:-2] = False
    assert d[4:26][:, outer].max() == 0.0 and d[:4][:, outer].max() == 1.0
    assert p.volume(d) == pytest.approx(0.30, abs=0.005)
    assert any("gap 2 mm" in n for n in p.notes)
    assert gapped.full_compliance == pytest.approx(free.full_compliance, rel=1e-3)   # the same reference
    assert free.design_compliance() < gapped.design_compliance() < 3 * free.design_compliance()
    body = F.intersect(F.body_from_mesh(box, 2.0), F.body_from_mesh(
        T.density_mesh(d, np.asarray(p.phi), p.grid.origin, p.grid.h, space=p.space), 2.0))
    assert F.sample(body, (15.0, 5.0, 9.8)) > 0       # nothing against the top face half way
    args = (F.body_from_mesh(box, 2.0), Conditions(()), PRESETS["PLA"], 30, 0, 1, None, None, "draft")
    assert T.topology_key(*args) == T.topology_key(*args, gap=0.0) != T.topology_key(*args, gap=2.0)
    with pytest.raises(ConditionError, match="lower Keep"):
        _cantilever(box, keep=80.0, gap=2.0)


def test_smoothness_and_symmetry_are_in_the_key_only_when_set(box):
    args = (F.body_from_mesh(box, 2.0), Conditions(()), PRESETS["PLA"], 30, 0, 1, None, None, "draft")
    plain = T.topology_key(*args)
    assert plain == T.topology_key(*args, smoothness=0.0, symmetry="none")   # earlier results stay valid
    assert len({plain, T.topology_key(*args, smoothness=0.3), T.topology_key(*args, symmetry="x")}) == 3
    with pytest.raises(ConditionError, match="symmetry must be one of"):
        _cantilever(box, symmetry="w")


def test_the_outline_gradient_is_exact(box):
    opt = T.Optimizer(_cantilever(box, smoothness=0.3))
    x = np.random.default_rng(1).random(opt.p.grid.shape)
    _, grad = opt.outline(x)
    for at in [(5, 3, 4), (15, 0, 9), (29, 9, 0)]:   # inside, on a face, in a corner
        up, down = x.copy(), x.copy()
        up[at] += 1e-4
        down[at] -= 1e-4
        assert grad[at] == pytest.approx((opt.outline(up)[0] - opt.outline(down)[0]) / 2e-4, rel=1e-5)


def test_smoothness_gives_fewer_holes_and_less_outline_for_the_same_stiffness(tmp_path_factory):
    # a deep thin cantilever (40 x 4 x 20 mm, held at one end, pulled down at the other) makes a truss
    # with 6 openings; Smoothness 0.5 makes it with 3, 5% less outline and the same stiffness (when
    # written; the outline penalty never shrank Keep %)
    from scipy import ndimage
    from engine.graph.nodes import import_step
    plate = import_step(_box_step(tmp_path_factory.mktemp("plate") / "plate.step", (40, 4, 20)), 0.01, 1, 0.5)[0]
    ids = plate.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", plate.triangles[ids == _face_id(plate, 0, 0.0)]),
                        Condition("load", plate.triangles[ids == _face_id(plate, 0, 40.0)], amount=50.0,
                                  direction="-Z")))
    run = lambda s: T.Optimizer(T.prepare(F.body_from_mesh(plate, 2.0), conds, PRESETS["PLA"], keep=30.0, skin=1.0,
                                          h=1.0, smoothness=s)).run()
    plain, smooth = run(0.0), run(0.5)
    openings = lambda o: ndimage.label(o.density()[:, 2, :] < 0.5)[1]
    assert openings(smooth) < openings(plain)
    assert plain.outline(smooth.x)[0] < 0.97 * plain.outline(plain.x)[0]   # (both at full sharpness)
    assert smooth.p.volume(smooth.density()) == pytest.approx(0.30, abs=0.005)
    assert smooth.design_compliance() < 1.1 * plain.design_compliance()


def test_material_that_carries_nothing_never_grows_when_keep_does_not_fit(box):
    # Max thickness may leave no room for Keep %: material was then free, and elements carrying nothing
    # grew (loose pieces cut off from the bolted bracket with Smoothness 1 stayed and grew)
    p = _cantilever(box, max_thickness=4.5)
    opt = T.Optimizer(p)
    free = np.asarray(p.free)
    x = np.asarray(opt.x).copy()
    useless = np.zeros_like(free)
    useless[12:16, 3:7, 3:7] = True
    useless &= free
    dc = np.where(useless, -1e-7, -1.0) * free           # carries nothing there
    dv = free / free.sum()
    dg = np.where(useless, 0.0, 1.0) * free              # and crowds no neighbourhood
    xn = np.asarray(opt._update_two(x, dc, dv, dg, p.volume(opt.physical(x)[1]), 200.0))   # far too full
    assert p.volume(opt.physical(xn)[1]) < p.target - 0.01   # Keep % does not fit
    assert np.all(xn[useless] < x[useless])


def _sideways(box):
    """The cantilever's end pulled down and sideways (+Y): its stiffest shape is not symmetric across y."""
    ids = box.face_attributes["cad_face"]
    return Conditions((Condition("fixed", box.triangles[ids == _face_id(box, 0, 0.0)]),
                       Condition("load", box.triangles[ids == _face_id(box, 0, 30.0)], amount=50.0,
                                 direction="custom", vector=(0.0, 0.6, -1.0))))


def test_symmetry_makes_exact_mirror_images(box):
    space, conds = F.body_from_mesh(box, 2.0), _sideways(box)
    run = lambda **kw: T.Optimizer(T.prepare(space, conds, PRESETS["PLA"], keep=30.0, skin=1.0, h=1.0, **kw))
    mirrored, leaning = run(symmetry="y and z"), run()
    for _ in range(10):
        mirrored.step()
        leaning.step()
    d = mirrored.density()
    assert np.array_equal(d, d[:, ::-1]) and np.array_equal(d, d[:, :, ::-1])
    assert mirrored.p.volume(d) == pytest.approx(0.30, abs=0.005)
    d = leaning.density()
    assert not np.allclose(d, d[:, ::-1], atol=0.05)
    # 0.8 mm elements, 12.5 across the box: 13 laid symmetrically about its middle (y = 5)
    p = T.prepare(space, conds, PRESETS["PLA"], keep=30.0, skin=1.0, h=0.8, symmetry="y")
    assert p.mirror == ((1, 13),) and p.grid.origin[1] == pytest.approx(-0.2) and p.grid.origin[0] == 0.0
    assert any("mirror images across y = 5 mm" in n for n in p.notes)
    # a Keep ball on one side has no mirror image: said in the note
    ball = F.sqrt((F.X - 10) * (F.X - 10) + (F.Y - 5) * (F.Y - 5) + (F.Z - 5) * (F.Z - 5)) - 2.0
    p = _cantilever(box, symmetry="x", keep_body=ball)
    assert any("skin and Keep has no mirror image" in n for n in p.notes)


def test_the_surface_stays_out_of_avoid_regions():
    # a full block of 20^3 elements around an empty 6^3 Avoid pocket: material is extended past the
    # design space only, not into the pocket (with phi > 0 taken as the design space it was filled)
    d = np.ones((20, 20, 20), dtype=np.float32)
    d[7:13, 7:13, 7:13] = 0.0
    space = np.ones(d.shape, dtype=bool)
    for given, filled in ((space, False), (None, True)):
        mesh = T.density_mesh(d, d, np.zeros(3), 1.0, space=given)
        assert (F.sample(F.body_from_mesh(mesh, 2.0), (10.0, 10.0, 10.0)) < 0) == filled


def test_bits_on_their_own_inside_the_design_space_are_dropped():
    # a small bit by a face that joins the part only through the half-filled elements at that face (where
    # the crop to the design space cuts it off): emptied, the part kept (flakes on the wall bracket's pin eye)
    d = np.zeros((30, 20, 22), dtype=np.float32)
    phi = np.ones_like(d)
    phi[:, :, 17:] = 0.3                # the design space's face runs through these elements
    d[2:18, 4:16, 4:20] = 1.0           # the part, up to the face
    d[21:24, 8:11, 16:20] = 1.0         # a bit by the face
    d[17:21, 8:11, 18:20] = 1.0         # joined to the part in the face's elements only
    body = F.body_from_mesh(T.density_mesh(d, phi, np.zeros(3), 1.0, blur=0.0, space=np.ones(d.shape, bool)), 2.0)
    assert F.sample(body, (22.5, 9.5, 16.5)) > 0 and F.sample(body, (10.0, 10.0, 10.0)) < 0


def test_too_little_to_keep_is_refused(box):
    with pytest.raises(ConditionError, match="raise Keep"):
        _cantilever(box, keep=3.0, skin=4.0)


def test_densities_become_a_closed_smooth_body(box):
    p = _cantilever(box)
    opt = T.Optimizer(p)
    opt.run()
    mesh = T.density_mesh(opt.density(), np.asarray(p.phi), p.grid.origin, p.grid.h)
    assert mesh.is_watertight and mesh.split(only_watertight=False).__len__() == 1
    topo = F.body_from_mesh(mesh, 2.0)
    body = F.intersect(F.body_from_mesh(box, 2.0), topo)
    # the material at the clamped end runs past the design space, so cropping by it gives back the
    # CAD face itself; the volume is about the kept share of the box (surface at 0.5, smoothed)
    assert F.sample(topo, (0.0, 5.0, 5.0)) < -0.5
    assert F.sample(body, (0.0, 5.0, 5.0)) == pytest.approx(0.0, abs=0.02)
    assert F.sample(body, (0.5, 5.0, 5.0)) < 0
    from engine.grid import Grid
    inside = F.Context(Grid((0.25, 0.25, 0.25), (29.75, 9.75, 9.75), 0.5)).eval_full(body) < 0
    assert inside.mean() == pytest.approx(0.30, abs=0.06)
    # Smoothing rounds the element steps away (less surface in the design space), keeps the volume and
    # never closes the beam's open sides (growing past the design space by dilation once did)
    shares, areas = [], []
    for b in (1, 2, 3):
        m = T.density_mesh(opt.density(), np.asarray(p.phi), p.grid.origin, p.grid.h, blur=b)
        # (a strong blur can take thin members away; with the min thickness, as the node always gives it,
        # thin members keep the light smoothing and nothing is lost)
        kept = T.density_mesh(opt.density(), np.asarray(p.phi), p.grid.origin, p.grid.h, blur=b,
                              min_thickness=T.MIN_MEMBER * p.grid.h)
        assert len(kept.split(only_watertight=False)) == 1
        cropped = F.intersect(F.body_from_mesh(box, 2.0), F.body_from_mesh(m, 2.0))
        shares.append(float((F.Context(Grid((0.25, 0.25, 0.25), (29.75, 9.75, 9.75), 0.5)).eval_full(cropped) < 0).mean()))
        c = m.triangles_center
        areas.append(float(m.area_faces[np.all((c > 0.5) & (c < [29.5, 9.5, 9.5]), axis=1)].sum()))
    assert max(shares) - min(shares) < 0.015, shares
    assert areas[0] > areas[1] and areas[0] > areas[2], areas   # (smoothing the distance shrinks little)
