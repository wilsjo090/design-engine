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


def test_the_surface_falls_back_to_the_cpu_when_the_gpu_runs_out_of_memory(monkeypatch):
    # (the card's free memory changes with other programs between the check and the work)
    density = np.zeros((16, 12, 12), np.float32)
    density[3:13, 4:8, 4:8] = 1.0
    phi = np.ones_like(density)
    plain = T.density_mesh(density, phi, (0, 0, 0), 1.0, blur=1.5)
    real, tried = T._density_mesh, []
    gpu = object()   # stands for CuPy

    def on(*args, xp=np, error="out of memory allocating 2 GB"):
        if xp is gpu:
            tried.append(1)
            raise MemoryError(error)
        return real(*args, xp=xp)
    monkeypatch.setattr(T, "_smoothing_module", lambda samples: gpu)
    monkeypatch.setattr(T, "_density_mesh", on)
    monkeypatch.setattr(T.backend, "release_gpu_memory", lambda: None)
    assert tried == [] and np.array_equal(T.density_mesh(density, phi, (0, 0, 0), 1.0, blur=1.5).vertices,
                                          plain.vertices) and tried == [1]
    monkeypatch.setattr(T, "_density_mesh", lambda *a, xp=np: on(*a, xp=xp, error="a bug"))
    with pytest.raises(MemoryError, match="a bug"):   # (only running out of memory goes to the CPU)
        T.density_mesh(density, phi, (0, 0, 0), 1.0, blur=1.5)


def test_making_the_surface_can_be_stopped_between_its_steps():
    # (Stop: on a fine grid it is ~20 s in one call)
    density = np.zeros((16, 12, 12), np.float32)
    density[3:13, 4:8, 4:8] = 1.0
    density[7:9, 4:5, 4:8] = 0.6
    phi = np.ones_like(density)
    calls = []
    plain = T.density_mesh(density, phi, (0, 0, 0), 1.0, blur=1.5, min_thickness=3.0)
    checked = T.density_mesh(density, phi, (0, 0, 0), 1.0, blur=1.5, min_thickness=3.0, cancel=lambda: calls.append(1))
    assert len(calls) >= 5 and np.array_equal(plain.vertices, checked.vertices)

    class Stopped(Exception):
        pass

    def stop():
        raise Stopped
    with pytest.raises(Stopped):
        T.density_mesh(density, phi, (0, 0, 0), 1.0, blur=1.5, min_thickness=3.0, cancel=stop)


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
    p = _cantilever(box, max_thickness=9.5)
    assert p.local_sigma and float(np.max(p.local(np.ones(p.grid.shape, dtype=p.dtype)))) < 1.02 * T.LOCAL_SHARE
    # and past that the limit cannot bind, so it is off, and says so
    p = _cantilever(box, max_thickness=12.0)
    assert not p.local_sigma and any("no effect" in n for n in p.notes)


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


@pytest.mark.parametrize("limits", [{}, {"max_thickness": 4.5}, {"max_section": 30.0}])
def test_a_stopped_run_continues_exactly(box, limits):
    p = _cantilever(box, **limits)
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


def test_a_run_stopped_by_max_iterations_goes_on_when_it_is_raised(box):
    p = _cantilever(box)
    straight = T.Optimizer(p, max_iterations=20).run()
    first = T.Optimizer(p, max_iterations=12).run()
    assert first.done and not first.settled and first.iteration == 12
    state = {k: np.array(v) for k, v in first.state().items()}   # as saved and loaded
    assert T.finished(state, 12) and not T.finished(state, 20) and not T.settled(state)
    resumed = T.Optimizer(p, state, max_iterations=20)
    assert not resumed.done
    resumed.run()
    assert resumed.iteration == 20 and np.allclose(np.asarray(resumed.x), np.asarray(straight.x), atol=1e-6)
    # states from before the setting: done under 125 means settled, at 125 stopped there
    old = {"done": np.array(True), "iteration": np.array(125)}
    assert not T.settled(old) and not T.finished(old, 200) and T.finished({**old, "iteration": np.array(90)}, 200)


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


def test_max_section_reads_a_members_area_whatever_its_shape():
    # across long members along Z the Gaussian neighbourhood is 2D: section area / (2 pi sigma^2) at the axis
    from scipy.ndimage import gaussian_filter
    from scipy.special import erf
    area, thinnest, h = 50.0, 4.0, 0.25
    sigma, share = T.section_limit(area, thinnest)
    o = (np.arange(800) - 399.5) * h
    x, y = np.meshgrid(o, o, indexing="ij")

    def fill(solid):
        return float(gaussian_filter(solid.astype(float), sigma / h, mode="constant", truncate=6.0)[400, 400])

    def strip(width, thick):
        return (np.abs(x) < width / 2) & (np.abs(y) < thick / 2)
    assert fill(x * x + y * y < area / np.pi) == pytest.approx(share, rel=0.02)    # a round one just fits
    assert fill(strip(np.sqrt(area), np.sqrt(area))) == pytest.approx(share, rel=0.03)
    assert fill(strip(area / 4, 4.0)) == pytest.approx(share, rel=0.1)              # a blade as wide as ~2 sigma
    assert fill(strip(1.5 * area / 4, 4.0)) > 1.15 * share                          # half as much again does not
    plate = lambda t: erf(t / (2 * np.sqrt(2) * sigma))
    assert plate(T.SECTION_PLATE * thinnest) == pytest.approx(share, rel=1e-6)   # the thickest plate that fits
    assert plate(thinnest) > 1.3 * share                                         # one that would carry does not


def test_max_section_sets_the_members_range(box):
    p = _cantilever(box, max_section=30.0)
    across = T.section_diameter(30.0)
    assert p.radius == pytest.approx(max(3.0, T.SECTION_MIN * across))
    assert p.section_sigma * p.grid.h == pytest.approx(T.section_limit(30.0, p.radius)[0])
    assert any("max section 30 mm2" in n for n in p.notes)
    # a thick Struts only cannot ask for members that do not fit in the section
    tight = _cantilever(box, max_section=20.0, max_thickness=12.0, members="struts only")
    assert tight.radius <= T.section_diameter(20.0) / 1.5 < T.STRUT_MIN * 12.0
    small = _cantilever(box, max_section=5.0)   # 1 mm elements: members are at least ~3 mm
    assert any("max section is too small" in n for n in small.notes)
    assert small.section_share > 0


def test_max_section_counts_designed_material_only(box):
    # the skin at supports and loads cannot get thinner: counted, the members meeting it got no room at all
    p = _cantilever(box, max_section=30.0)
    assert float(np.max(p.section(np.asarray(p.solid, dtype=p.dtype)))) == 0.0
    assert float(np.max(p.section(np.asarray(p.free, dtype=p.dtype)))) > 0.0


def test_max_section_holds_on_the_design(box):
    p = _cantilever(box, keep=40.0, max_section=30.0)
    fullest = lambda d: float(np.max(p.section(d)[np.asarray(p.free)]))
    free_run = T.Optimizer(_cantilever(box, keep=40.0)).run()
    capped = T.Optimizer(p).run()
    assert fullest(free_run.density()) > 1.1 * p.section_share
    assert fullest(capped.density()) <= 1.05 * p.section_share
    assert capped.section_max > 0 and capped.local_max is None


def test_a_design_variable_that_turns_back_gets_a_smaller_move_limit(box):
    # (the wide neighbourhood's linearization overshoots: bracket, compliance 5458 <-> 1915 every other iteration)
    opt = T.Optimizer(_cantilever(box, max_section=30.0))
    shape = opt.x.shape
    up, down, still = np.full(shape, 0.1, dtype=opt.p.dtype), np.full(shape, -0.1, dtype=opt.p.dtype), np.zeros(shape, dtype=opt.p.dtype)
    opt._track_moves(up)
    assert np.all(opt.moves == T.MOVE)                                     # the first step has nothing to compare with
    opt._track_moves(down)
    assert np.allclose(opt.moves, T.MOVE * T.TURN_SHRINK)
    for _ in range(40):
        opt._track_moves(-opt.last_step)
    assert np.allclose(opt.moves, T.TURN_FLOOR)                            # never quite frozen
    opt._track_moves(still)                                                # saturated at 0 or 1: free to move again
    assert np.allclose(opt.moves, T.TURN_FLOOR * T.TURN_GROW)
    for _ in range(40):
        opt._track_moves(opt.last_step + 0.1)
    assert np.allclose(opt.moves, T.MOVE)
    assert T.Optimizer(_cantilever(box)).moves is None                     # results without the limit do not change


def test_the_move_limits_are_followed_only_with_a_max_section(box):
    plain = T.Optimizer(_cantilever(box, max_thickness=4.5))
    plain.step()
    assert plain.moves is None
    section = T.Optimizer(_cantilever(box, max_section=30.0))
    for _ in range(3):
        before = np.array(section.x)
        limits = np.array(section.moves) if section.moves is not None else T.MOVE
        section.step()
        assert np.all(np.abs(np.array(section.x) - before) <= limits + 1e-6)


def test_max_thickness_and_max_section_hold_together(box):
    p = _cantilever(box, keep=40.0, max_section=30.0, max_thickness=4.5)
    d = T.Optimizer(p).run().density()
    assert float(np.max(p.section(d)[np.asarray(p.free)])) <= 1.05 * p.section_share
    assert float(np.max(p.local(d)[np.asarray(p.free)])) <= 1.05 * T.LOCAL_SHARE


def test_max_section_is_in_the_key_only_when_set(box):
    ids = box.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", box.triangles[ids == _face_id(box, 0, 0.0)]),
                        Condition("load", box.triangles[ids == _face_id(box, 0, 30.0)], amount=50.0, direction="-Z")))
    args = F.body_from_mesh(box, 2.0), conds, PRESETS["PLA"], 30.0, 0.0, 1.0, None, None, "draft"
    assert T.topology_key(*args) == T.topology_key(*args, max_section=0.0)
    assert T.topology_key(*args) != T.topology_key(*args, max_section=30.0) != T.topology_key(*args, max_section=40.0)


def test_a_coarse_start_skips_levels_too_coarse_for_the_section(box):
    space = F.body_from_mesh(box, 2.0)
    h = {d: T.plan_grid(space, d, 3000.0).h for d in T.LEVELS}
    assert T.levels(space, "fine", True, 3000.0) == list(T.LEVELS)
    fits_medium = np.pi / 4 * (1.01 * T._thinnest_max(h["medium"], 0.0)) ** 2
    assert T.levels(space, "fine", True, 3000.0, max_section=fits_medium) == ["medium", "fine"]


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
    # nothing left to do: the card says so before the result is read (minutes on a fine grid), and with the
    # cached node (this small part stops at Max iterations; the wall bracket settles at 94)
    told = [e for e in events if e["type"] == "finished"]
    assert len(told) == 1 and told[0]["node"] == "topo" and told[0]["text"].startswith("Stopped at Max iterations")
    assert [e for e in events if e["type"] == "node_done" and e["node"] == "topo"][0]["finished"] == told[0]["text"]
    # and Optimize then only says so: no iterations (this looked as if the button did nothing)
    graph, events, out = run("optimize", 1, optimize="topo")
    assert graph.nodes["topo"]["inputs"]["compute"] == 1
    assert any(e["type"] == "finished" for e in events) and out["body"].finished == told[0]["text"]
    assert not any(e["type"] == "progress" and e.get("step") == "iteration" for e in events)
    assert any(e["type"] == "message" and told[0]["text"] in e["text"] for e in events)
    assert not any("does not reach" in e.get("text", "") for e in events)   # (a real result: no false alarm)
    assert out["result"] is None     # Show: shape, plain
    # Show: stress colours the part by that same analysis' stress
    recipe["nodes"]["topo"]["inputs"]["show"] = "stress"
    _, _, views, _ = build_run_graph(recipe, "preview", "draft", ["topo"], None, 1)
    assert views["topo"].paint == ("topo", "result")
    graph, events, out = run("preview", 1)
    assert out["result"].name == "von Mises stress" and F.sample(out["result"], (2.0, 5.0, 5.0)) > 0
    _, targets, _, _ = build_run_graph(recipe, "optimize", "draft", [], "topo", 2)
    assert targets == ["topo"]


def test_the_shape_says_which_loads_it_leaves_in_the_air(box):
    # a 30 x 10 x 10 box held at x = 0. Arrows hung in the air on a real bracket, with no word on the card
    ids = box.face_attributes["cad_face"]
    held = Condition("fixed", box.triangles[ids == _face_id(box, 0, 0.0)])
    far = Condition("load", box.triangles[ids == _face_id(box, 0, 30.0)], amount=50.0, direction="-Z", label="far end")
    near = Condition("load", box.triangles[ids == _face_id(box, 2, 10.0)], amount=50.0, direction="-Z", label="top")
    phi = np.ones((30, 10, 10), dtype=np.float32)
    meshed = lambda d, *conds: T.density_mesh(d, phi, np.zeros(3), 1.0, blur=0.0, conditions=conds).metadata["unreached"]

    whole = np.ones_like(phi)
    assert meshed(whole, held, far, near) == []
    # half of it gone: the far end is off the shape; half of the top is too, and that is over REACH_MISS as well
    half = whole.copy()
    half[15:] = 0.0
    missed = dict(meshed(half, held, far, near))
    assert missed["far end"] == pytest.approx(1.0) and missed["top"] == pytest.approx(0.5, abs=0.1)
    # there is material under the far end, but a loose piece of it: nothing holds it
    loose = half.copy()
    loose[24:] = 1.0
    assert dict(meshed(loose, held, far))["far end"] == pytest.approx(1.0)
    # (a piece under SPECK goes anyway, and the load with it)
    speck = half.copy()
    speck[29, 4:6, 4:6] = 1.0
    assert dict(meshed(speck, held, far))["far end"] == pytest.approx(1.0)
    # most of a load reached is no alarm: under REACH_MISS
    most = whole.copy()
    most[29, 0:1] = 0.0
    assert meshed(most, held, far) == []


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

    def stop_at_10(done, total, step=None):
        if done == 10:
            raise Cancelled("stopped")
        progress(done, total, step)

    monkeypatch.setattr(nodes_fea, "report_progress", stop_at_10)
    with pytest.raises(Cancelled):
        run("optimize", 0, optimize="topo")
    monkeypatch.setattr(nodes_fea, "report_progress", progress)
    _, out = run("preview", 1)
    assert out["body"].note.startswith("stopped at iteration 10 on draft elements")
    assert not out["body"].finished   # (a stopped run goes on: Optimize stays on the card)
    events, out = run("optimize", 2, optimize="topo")
    texts = [e["text"] for e in events if e["type"] == "message"]
    assert any("continuing at iteration 10" in t for t in texts)
    assert any(t.startswith("draft done at iteration") for t in texts)
    assert any(t.startswith("medium done at iteration") for t in texts)
    assert any("(fine, " in t and "on finer elements" in t for t in texts)
    steps = [e["step"] for e in events if e["type"] == "progress"]
    after = steps[steps.index("smoothing the shape"):]    # what the card says once the iterations are done
    assert "iteration" not in after and "analysing the shape" in after
    # the check's solver on the card (a result's check once ran 10 minutes with nothing moving there)
    assert any(s and s.startswith("analysing the shape (solver iteration ") for s in after), after
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


def test_the_surface_runs_straight_out_of_the_crop_and_leaves_no_flakes():
    # material outside the design space hanging over a stretch of its face that is empty inside dips in by
    # a fraction of a sample, so the crop left a paper-thin loose shell there (bolted bracket: on the
    # unloaded side of its bolt and pin holes); given the crop, the surface runs straight out of it
    d = np.zeros((30, 20, 22), dtype=np.float32)
    phi = np.ones_like(d)
    phi[:, :, 17] = 0.3                 # the face, z = 17.3, runs through these elements
    phi[:, :, 18:] = 0.0
    d[2:18, 4:16, 4:20] = 1.0           # the part, through the face
    d[2:26, 4:16, 18:20] = 1.0          # outside the design space ...
    d[20:26, 4:16, 17] = 1.0            # ... hanging over its face beside the part
    crop = F.Z - 17.3
    flake, part, face = (23.0, 10.0, 17.28), (10.0, 10.0, 17.2), (10.0, 10.0, 17.3)
    for given, loose in ((None, True), (crop, False)):
        mesh = T.density_mesh(d, phi, np.zeros(3), 1.0, blur=0.0, space=np.ones(d.shape, bool), crop=given)
        body = F.intersect(crop, F.body_from_mesh(mesh, 2.0))
        assert (F.sample(body, flake) < 0) == loose
        assert F.sample(body, part) < 0 and F.sample(body, face) == pytest.approx(0.0, abs=1e-3)


def test_thin_tips_are_cut_and_thin_members_kept():
    # as thin as each other: a needle hanging off a block by one patch goes (its end rounded); a strut joining
    # two blocks is a member and a plate on one has far more section (a skin), so both stay (cutting by thickness
    # alone took 8% of a key box's skins); the blocks stay as they are, sample for sample
    from scipy import ndimage
    v, t = 0.25, 2.0
    solid = np.zeros((120, 110, 40), dtype=bool)
    solid[10:50, 10:70, 10:30] = True       # a block
    solid[80:110, 10:40, 10:30] = True      # another
    solid[50:80, 20:24, 18:22] = True       # a 1 mm strut between them
    solid[50:90, 56:60, 18:22] = True       # a 1 mm needle off the first block
    solid[10:50, 70:105, 18:22] = True      # a 1 mm plate off it
    values = ((ndimage.distance_transform_edt(solid) - ndimage.distance_transform_edt(~solid)) * v).astype(np.float32)
    inside = np.ones(solid.shape, dtype=bool)
    out, tips = T.cut_tips(values, 0.0, inside, v, t)
    kept = out >= 0.0
    assert tips == 1
    assert kept[50:80, 20:24, 18:22].all() and kept[10:50, 70:105, 18:22].all()
    assert kept[50, 58, 20] and not kept[58:, 56:60, :].any()
    assert np.array_equal(out[:44], values[:44]) and np.array_equal(out[80:, :40], values[80:, :40])
    thin = np.zeros_like(solid)
    thin[10:50, 20:24, 18:22] = True        # nothing thick to hang on: left alone
    values = ((ndimage.distance_transform_edt(thin) - ndimage.distance_transform_edt(~thin)) * v).astype(np.float32)
    out, tips = T.cut_tips(values, 0.0, inside, v, t)
    assert tips == 0 and np.array_equal(out, values)


def _signed(solid, v):
    from scipy import ndimage
    return ((ndimage.distance_transform_edt(solid) - ndimage.distance_transform_edt(~solid)) * v).astype(np.float32)


def test_a_thin_spot_in_a_plate_is_no_tip(monkeypatch):
    # a thin patch closed in by thicker material hangs by one (ring-shaped) patch and is small, but cutting it makes a
    # hole (a bracket of its own, at medium: skins came apart)
    v, t = 0.25, 3.3
    solid = np.zeros((100, 100, 44), dtype=bool)
    solid[20:80, 20:80, 12:32] = True       # a block...
    solid[32:68, 32:68, 12:32] = False      # ...round a 9 mm square of a 0.5 mm plate
    solid[32:68, 32:68, 21:23] = True
    values, inside = _signed(solid, v), np.ones(solid.shape, dtype=bool)
    out, tips = T.cut_tips(values, 0.0, inside, v, t)
    assert tips == 0 and np.array_equal(out, values)
    monkeypatch.setattr(T, "_free_all_round", lambda region, empty: True)
    assert T.cut_tips(values, 0.0, inside, v, t)[1] == 1   # (it is that rule that keeps it)


def test_a_sharp_edge_is_no_tip(monkeypatch):
    # a plate meeting a hole at a slant has a sharp rim, out of every ball that fits: cut in patches, it came out wavy
    v, t = 0.25, 2.0
    x, y, z = np.mgrid[0:130, 0:40, 0:32]
    solid = (x >= 10) & (y >= 4) & (y < 36) & (z >= 12) & (z < 20) & (x < 90 + (z - 12) * 0.6)   # (a 59 degree edge)
    solid |= (x >= 10) & (x < 60) & (y >= 4) & (y < 36) & (z >= 4) & (z < 28)
    values, inside = _signed(solid, v), np.ones(solid.shape, dtype=bool)
    out, tips = T.cut_tips(values, 0.0, inside, v, t)
    assert tips == 0 and np.array_equal(out, values)
    monkeypatch.setattr(T, "TIP_OUT", 1.0)
    assert T.cut_tips(values, 0.0, inside, v, t)[1] > 0    # (it is that rule that keeps it)


def test_a_tip_that_a_load_stands_on_stays():
    # a 4 mm pin at draft's 6 mm: it went with its load, and the safety factor from 10 to 0.81
    from types import SimpleNamespace
    v, t = 0.25, 2.0

    def load(lo, hi):
        lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        tri = np.array([[lo, (hi[0], lo[1], lo[2]), hi], [lo, hi, (lo[0], hi[1], lo[2])]])
        return SimpleNamespace(triangles=tri, presses=lambda normals: np.ones(len(normals), dtype=bool))
    solid = np.zeros((90, 80, 40), dtype=bool)
    solid[10:50, 10:70, 10:30] = True       # a block
    solid[50:80, 38:42, 18:22] = True       # a 1 mm needle off it
    values, inside = _signed(solid, v), np.ones(solid.shape, dtype=bool)
    start = np.array((-3.0, 1.0, 2.0))
    on_needle = load(start + (15.0, 9.5, 5.5), start + (19.0, 10.5, 5.5))
    on_block = load(start + (3.0, 3.0, 2.5), start + (12.0, 17.0, 2.5))
    over_both = load(start + (2.5, 2.5, 5.5), start + (20.0, 17.5, 5.5))   # (a corner of a big load's faces)
    assert T.cut_tips(values, 0.0, inside, v, t, start=start)[1] == 1
    assert T.cut_tips(values, 0.0, inside, v, t, start=start, conditions=(on_block, over_both))[1] == 1
    out, tips = T.cut_tips(values, 0.0, inside, v, t, start=start, conditions=(on_block, on_needle))
    assert tips == 0 and np.array_equal(out, values)


def test_tips_are_measured_against_what_was_asked_for():
    # the grid's own limit is no reason to cut: at draft Min is 6 mm for lack of elements, whatever the skin is
    d = np.zeros((36, 30, 12), dtype=np.float32)
    d[:, :20, :] = 1.0
    crop = F.subtract(F.box((18.0, 15.0, 6.0), (36.0, 30.0, 8.0)), F.bar((18.0, 8.4, -20.0), (18.0, 8.4, 30.0), 12.0))

    def tips(**settings):
        return T.density_mesh(d, np.ones_like(d), np.zeros(3), 1.0, blur=1.0, space=np.ones(d.shape, bool),
                              min_thickness=3.0, crop=crop, **settings).metadata["tips"]
    assert tips() == 2 and tips(tips_under=3.0) == 2 and tips(tips_under=0.2) == 0
    assert tips(tips_under=0.0) == 0        # (no Min and no Skin asked for: nothing to go by)


def test_the_point_the_crop_leaves_where_an_edge_runs_into_a_hole_is_cut(monkeypatch):
    # a plate's free edge running into a hole at a shallow angle: the strip between them narrows to a point, which
    # thickening cannot help, since the crop takes off what it adds (bolted bracket: the skin of a face the hole cut)
    d = np.zeros((36, 30, 12), dtype=np.float32)
    d[:, :20, :] = 1.0
    crop = F.subtract(F.box((18.0, 15.0, 6.0), (36.0, 30.0, 8.0)), F.bar((18.0, 8.4, -20.0), (18.0, 8.4, 30.0), 12.0))

    def body():
        mesh = T.density_mesh(d, np.ones_like(d), np.zeros(3), 1.0, blur=1.0, space=np.ones(d.shape, bool),
                              min_thickness=3.0, crop=crop)
        return mesh, F.intersect(crop, F.body_from_mesh(mesh, 2.0))
    points = (13.0, 19.6, 6.0), (23.0, 19.6, 6.0)      # under 1 mm wide here
    plate = (3.0, 10.0, 6.0), (33.0, 10.0, 6.0)
    mesh, shape = body()
    assert mesh.metadata["tips"] == 2 and len(mesh.split(only_watertight=False)) == 1
    assert all(F.sample(shape, p) > 0 for p in points) and all(F.sample(shape, p) < 0 for p in plate)
    monkeypatch.setattr(T, "TIP_SECTION", 0.0)         # (no tip then: the points were there)
    mesh, shape = body()
    assert mesh.metadata["tips"] == 0 and all(F.sample(shape, p) < 0 for p in points)


def test_dead_ends_are_found_and_working_or_kept_material_is_not():
    # a beam (stress 10) with: a stub hanging off it that carries nothing (a dead end), a lightly loaded
    # strut joining it to a second beam at both ends (kept), the same strut idle (goes), a core of the
    # beam at its neutral axis that reaches no surface (kept), and a stub by a load's face (kept)
    from types import SimpleNamespace
    d = np.zeros((40, 30, 20), dtype=np.float32)
    s = np.zeros_like(d)
    d[2:38, 2:8, 8:14] = 1.0          # beam 1
    d[2:38, 20:26, 8:14] = 1.0        # beam 2
    s[d > 0] = 10.0
    s[4:36, 4:6, 10:12] = 0.1         # beam 1's core
    d[10:13, 8:20, 10:13] = 1.0       # strut between them
    s[10:13, 8:20, 10:13] = 0.5
    d[20:23, 4:7, 14:19] = 1.0        # stub up from beam 1
    s[20:23, 4:7, 14:19] = 0.01
    d[33:36, 4:7, 14:17] = 1.0        # stub up to the load's face, within the skin
    s[33:36, 4:7, 14:17] = 0.01
    face = np.array([[[33.0, 4.0, 17.0], [36.0, 4.0, 17.0], [36.0, 7.0, 17.0]],
                     [[33.0, 4.0, 17.0], [36.0, 7.0, 17.0], [33.0, 7.0, 17.0]]])
    load = SimpleNamespace(triangles=face)
    s[5:15, 25, 8:14] = 0.01          # a quiet strip of beam 2's surface: sticks out nowhere, stays
    assert T.dead_ends(d, s, np.zeros(3), 1.0, [load], 3.0)[1] == 2   # (counted without a reach)
    found, count = T.dead_ends(d, s, np.zeros(3), 1.0, [load], 3.0, reach=1.5)
    assert count == 1 and (found[20:23, 4:7, 14:19] > 0).all() and (found > 0).sum() == 45
    s[10:13, 8:20, 10:13] = 0.01      # the strut carries nothing: it goes too
    found, count = T.dead_ends(d, s, np.zeros(3), 1.0, [load], 3.0, reach=1.5)
    assert count == 2 and (found[10:13, 9:19, 10:13] > 0).all() and not (found[4:36, 4:6, 10:12] > 0).any()


def test_the_optimizer_keeps_dead_ends_it_finds_empty(box):
    # a beam from the clamped end to the load with a finger up from it: a solve with the design's own
    # stiffness shows the finger carries nothing past its root; beyond Min thickness (1.5 mm here) from what
    # stays that is emptied and may not grow back, nearer it is left where it can, and a resumed run knows
    # it. (A short wide stub on the beam is no dead end: it works as a flange.)
    p = _cantilever(box, h=0.5, prune=True)
    opt = T.Optimizer(p)
    opt.beta = T.BETA_MAX
    x = np.zeros(p.grid.shape, dtype=p.dtype)
    x[:, 6:14, :8] = 1.0          # the beam, 4 x 4 mm
    x[26:30, 6:14, 8:] = 1.0      # the finger, 2 mm thick, 6 mm up
    opt.x = x
    assert opt._prune() >= 1
    pruned = np.asarray(opt.pruned)
    assert pruned[26:30, 6:14, 14:].any() and not np.asarray(opt.x)[pruned].any() and not pruned[:, :, :8].any()
    finger = np.asarray(opt.x)[26:30, 6:14, 8:]
    left = (finger > 0) & (finger <= T.PRUNE_REGROW)
    assert left.any() and ((finger == 1) | (finger <= T.PRUNE_REGROW)).all() and not pruned[26:30, 6:14, 8:][left].any()
    opt.step()
    assert not np.asarray(opt.x)[pruned].any()
    state = {k: np.array(v) for k, v in opt.state(stopped=True).items()}
    resumed = T.Optimizer(p, state)
    assert (np.asarray(resumed.pruned) == pruned).all() and resumed.dead_ends == opt.dead_ends
    # pruning is in the key only when asked for (earlier results stay valid)
    args = (F.body_from_mesh(box, 2.0), Conditions(()), PRESETS["PLA"], 30.0, 0.0, 1.0, None, None, "draft")
    assert T.topology_key(*args) == T.topology_key(*args, prune=False) != T.topology_key(*args, prune=True)


def test_load_cases_are_optimized_for_together(box):
    # pushed down in one case, sideways in the other: one solve per case, compliances added (stiff in each),
    # and a stopped run continues exactly with both cases' displacements
    import dataclasses
    ids = box.face_attributes["cad_face"]
    fixed = Condition("fixed", box.triangles[ids == _face_id(box, 0, 0.0)])
    end = box.triangles[ids == _face_id(box, 0, 30.0)]
    down = Condition("load", end, amount=50.0, direction="-Z", case="down")
    side = Condition("load", end, amount=50.0, direction="+Y", case="side")
    space = F.body_from_mesh(box, 2.0)
    both = T.prepare(space, Conditions((fixed, down, side)), PRESETS["PLA"], keep=30.0, skin=1.0, h=1.0)
    assert both.force.shape[0] == 2 and any("2 load cases (down, side)" in n for n in both.notes)
    opt = T.Optimizer(both)
    xb = opt.physical(opt.x)[1]
    single = [T.prepare(space, Conditions((fixed, dataclasses.replace(c, case=None))), PRESETS["PLA"], keep=30.0,
                        skin=1.0, h=1.0) for c in (down, side)]
    parts = [T.Optimizer(p)._compliance_of(xb)[0] for p in single]
    assert opt._compliance_of(xb)[0] == pytest.approx(sum(parts), rel=1e-6)
    for _ in range(3):
        opt.step()
    state = {k: np.array(v) for k, v in opt.state(stopped=True).items()}
    assert state["u"].shape[0] == 2
    resumed = T.Optimizer(both, state)
    opt.step()
    resumed.step()
    assert np.allclose(np.asarray(resumed.x), np.asarray(opt.x), atol=1e-6)


def test_too_little_to_keep_is_refused(box):
    # and it names what made the skin that thick
    with pytest.raises(ConditionError, match="raise Keep.*make Skin thinner"):
        _cantilever(box, keep=3.0, skin=4.0)
    with pytest.raises(ConditionError, match="lower Min thickness.*6 mm"):
        _cantilever(box, skin=1.0, min_thickness=6.0, keep=30.0)
    with pytest.raises(ConditionError, match="finer Detail"):
        _cantilever(box, skin=0.0, keep=1.0)


def test_limits_past_the_design_space_are_off_and_ask_for_no_big_filters(box):
    # on this 10 mm thick box they asked for gigabytes of filter in seconds, and 1000 mm2 for 18 mm members:
    # a grey blob that reached no load, where a value that big should change nothing
    for limit in ({"max_section": 1e6}, {"max_section": 300.0}, {"max_thickness": 400.0, "members": "struts only"},
                  {"max_thickness": 17.0, "members": "struts only"}, {"max_thickness": 400.0}):
        p = _cantilever(box, **limit)
        assert not p.section_sigma and not p.local_sigma and any("no effect" in n for n in p.notes)
        assert p.radius == 3.0 and p.kernel.shape == (7, 7, 7)


def test_limits_near_the_design_spaces_size_ask_for_members_that_fit(box):
    p = _cantilever(box, max_section=250.0)    # (17.8 mm across, on a 10 mm box)
    assert p.section_sigma and p.radius == 5.0 and p.section_sigma * p.grid.h <= 15.0
    assert 0 < p.section_share < 1 and any("members 5 to 10 mm" in n for n in p.notes)
    assert _cantilever(box, max_thickness=12.0, members="struts only").radius == pytest.approx(6.5)


def test_the_design_spaces_room_is_measured():
    room = T._room(np.ones((30, 10, 10)), 1.0)
    assert (room.thickest, room.across, room.longest, room.largest) == (10.0, 10.0, 30.0, 300.0)
    sheet = T._room(np.ones((60, 40, 1)), 0.5)      # thin, but members have room across it
    assert (sheet.across, sheet.longest, sheet.largest) == (20.0, 30.0, 600.0) and sheet.thickest <= 1.0


def test_a_min_thickness_nothing_can_be_is_refused(box):
    with pytest.raises(ConditionError, match="no member that thick fits"):
        _cantilever(box, min_thickness=11.0, skin=0.0)


def test_a_load_that_all_sits_on_the_support_is_refused(box):
    ids = box.face_attributes["cad_face"]
    end = box.triangles[ids == _face_id(box, 0, 0.0)]
    conds = Conditions((Condition("fixed", end), Condition("load", end, amount=50.0, direction="-Z")))
    with pytest.raises(ConditionError, match="nothing bends"):
        T.prepare(F.body_from_mesh(box, 2.0), conds, PRESETS["PLA"], h=1.0, skin=1.0)


def test_a_design_space_cut_in_two_says_what_cut_it(box):
    with pytest.raises(ConditionError, match="not connected.*does Avoid cut it off"):
        _cantilever(box, avoid_body=F.ball((15.0, 5.0, 5.0), 9.0))
    with pytest.raises(ConditionError, match="not on the part.*does Avoid cut it off"):
        _cantilever(box, avoid_body=F.ball((0.0, 5.0, 5.0), 12.0))
    with pytest.raises(ConditionError, match="not connected.*does Gap to faces cut it off"):
        _cantilever(box, gap=4.9)


def test_a_keep_body_outside_the_design_space_is_noted(box):
    assert any("Keep body is outside" in n for n in _cantilever(box, keep_body=F.ball((100.0, 5.0, 5.0), 3.0)).notes)
    assert not any("Keep body" in n for n in _cantilever(box, keep_body=F.ball((15.0, 5.0, 5.0), 3.0)).notes)


def test_dead_ends_are_cut_only_with_their_mirror_image(box, monkeypatch):
    # stress is no exact mirror image (a load off the plane, rounding): cut on one side only, the halves stopped
    # being images, and the other side's fin may be working
    p = _cantilever(box, prune=True, symmetry="y")
    opt = T.Optimizer(p)
    opt.beta = T.BETA_MAX
    x = np.zeros(p.grid.shape, dtype=p.dtype)
    x[:, 3:7, :5] = 1.0         # a beam
    x[10:14, 3:5, 5:] = x[10:14, 5:7, 5:] = 1.0     # fins, both sides
    x[20:24, 3:5, 5:] = x[20:24, 5:7, 5:] = 1.0     # fins, one found dead
    opt.x = x

    def found(density, *args, **kw):
        out = np.zeros(p.grid.shape, dtype=np.int32)
        out[10:14, 3:7, 5:] = 1
        out[20:24, 3:5, 5:] = 2
        return out, 2
    monkeypatch.setattr(T, "dead_ends", found)
    assert opt._prune() == 2
    left = np.asarray(opt.x)
    assert (left == np.flip(left, 1)).all() and np.asarray(opt.pruned).any()
    assert left[10:14, 3:7, 7:].max() <= T.PRUNE_REGROW and left[20:24, 3:7, 5:].min() == 1.0


def test_max_iterations_counts_every_level_of_a_coarse_start(box):
    coarse = T.Optimizer(_cantilever(box, h=2.0), max_iterations=5).run(hand_over=8.0)
    fine = T.Optimizer(_cantilever(box, h=1.0), max_iterations=5).take_over(coarse.handover()).run()
    assert coarse.iteration == fine.iteration == 5 and fine.done and not fine.settled


def test_a_float32_multigrid_that_gives_up_hands_over_to_float64(box, monkeypatch):
    # on the GPU a sheet or a slender bar stops converging beside soft elements: float64 goes on, for the rest of
    # the run, also after a Stop
    import dataclasses
    from engine.fea import solver
    p = dataclasses.replace(_cantilever(box), dtype=np.float32)   # (float32 stands in for the GPU's multigrid)
    real, seen = solver.Hierarchy.solve, []

    def solve(self, *args, **kw):
        seen.append(self.dtype)
        if self.dtype == np.float32:
            raise solver.NotConverged("gave up", residual=1.0)
        return real(self, *args, **kw)
    monkeypatch.setattr(solver.Hierarchy, "solve", solve)
    opt = T.Optimizer(p, max_iterations=3).run()
    assert seen[0] == np.float32 and set(seen[1:]) == {np.float64} and opt.precision == np.float64
    assert opt.iteration == 3 and np.isfinite(np.asarray(opt.x)).all()
    state = {k: np.array(v) for k, v in opt.state(stopped=True).items()}
    assert "wide" in state and T.Optimizer(p, state).precision == np.float64


def test_a_float64_solve_near_the_tolerance_counts_and_a_failed_one_does_not(box, monkeypatch):
    from engine.fea import solver
    p = _cantilever(box)
    real = solver.Hierarchy.solve
    left = {"residual": 0.05}

    def solve(self, f, u0=None, tol=1e-6, maxiter=1000, callback=None):
        u, n, _ = real(self, f, u0, tol, maxiter, callback)
        raise solver.NotConverged("stopped", u, maxiter, left["residual"])
    monkeypatch.setattr(solver.Hierarchy, "solve", solve)
    opt = T.Optimizer(p, max_iterations=2)
    assert opt.full_compliance > 0
    left["residual"] = 1.0
    with pytest.raises(solver.NotConverged):
        T.Optimizer(p, max_iterations=2)


def test_static_analysis_retries_in_float64(box, monkeypatch):
    from engine.fea import analysis, solver
    ids = box.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", box.triangles[ids == _face_id(box, 0, 0.0)]),
                        Condition("load", box.triangles[ids == _face_id(box, 0, 30.0)], amount=50.0, direction="-Z")))
    model = analysis.build_model(F.body_from_mesh(box, 2.0), conds, h=2.0)
    real, seen = solver.Hierarchy.solve, []

    def solve(self, *args, **kw):
        seen.append(self.dtype)
        if self.dtype == np.float32:
            raise solver.NotConverged("gave up")
        return real(self, *args, **kw)
    monkeypatch.setattr(solver.Hierarchy, "solve", solve)
    us, its, _ = analysis.solve_cases(model, PRESETS["PLA"])
    assert seen == [np.float32, np.float64] and its > 0 and len(us) == 1


def test_a_damaged_saved_state_counts_as_no_saved_state(tmp_path, monkeypatch):
    # (it failed the node on every run until the file was found and deleted)
    from engine import cache
    monkeypatch.setattr(cache, "default_cache_dir", lambda: tmp_path)
    cache.save_topology("damaged", x=np.ones(1000))
    file = tmp_path / cache.TOPOLOGY_FOLDER / "damaged.npz"
    assert cache.load_topology("damaged") is not None
    file.write_bytes(file.read_bytes()[:500])
    assert cache.load_topology("damaged") is None


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
