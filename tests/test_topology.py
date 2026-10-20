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
    first = None
    while not opt.done:
        c, v, _ = opt.step()
        first = first or c
        assert v == pytest.approx(0.30, abs=0.005)
    assert opt.history[-1][0] < 0.5 * first          # the uniform 30% start is far less stiff
    d = opt.density()[np.asarray(p.domain) > 0]
    assert ((d > 0.1) & (d < 0.9)).mean() < 0.02     # densities end near 0 or 1
    assert opt.iteration < T.MAX_ITERATIONS          # stopped because it settled


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
    # at 40% the free design has a region fuller than LOCAL_SHARE (0.64 when written); with a max
    # thickness no neighbourhood of that size is, and the volume is still Keep % (at 50% the two
    # cannot both hold in this 10 mm thin box: the max thickness wins, 46% was kept)
    p = _cantilever(box, keep=40.0, max_thickness=4.5)
    fullest = lambda d: float(np.max(p.local(d)[np.asarray(p.free)]))
    free_run = T.Optimizer(_cantilever(box, keep=40.0)).run()
    capped = T.Optimizer(p).run()
    d = capped.density()
    assert fullest(free_run.density()) > T.LOCAL_SHARE
    assert fullest(d) <= 1.02 * T.LOCAL_SHARE
    assert p.volume(d) == pytest.approx(0.40, abs=0.005)
    assert capped.history[-1][0] < 1.5 * free_run.history[-1][0]   # about as stiff


def test_struts_only_leaves_no_room_for_plates(box):
    # max thickness measured on round struts: plates may be at most 0.62 of it, and members must be
    # at least 0.65 of it, so no plate fits
    from scipy.special import erfinv
    t = 6.0
    p = _cantilever(box, max_thickness=t, members="struts only")
    sigma = p.local_sigma * p.grid.h
    assert 1 - np.exp(-t * t / (8 * sigma * sigma)) == pytest.approx(T.LOCAL_SHARE, rel=1e-3)  # a strut of t fills 0.6
    widest_plate = 2 * np.sqrt(2) * sigma * erfinv(T.LOCAL_SHARE)
    assert widest_plate < 2 * p.radius == pytest.approx(T.STRUT_MIN * t)
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
    assert F.sample(out["body"], (0.5, 5.0, 5.0)) < 0     # solid at the clamped end
    _, targets, _, _ = build_run_graph(recipe, "optimize", "draft", [], "topo", 2)
    assert targets == ["topo"]


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
    assert free.history[-1][0] < gapped.history[-1][0] < 3 * free.history[-1][0]
    body = F.intersect(F.body_from_mesh(box, 2.0), F.body_from_mesh(
        T.density_mesh(d, np.asarray(p.phi), p.grid.origin, p.grid.h, space=p.space), 2.0))
    assert F.sample(body, (15.0, 5.0, 9.8)) > 0       # nothing against the top face half way
    args = (F.body_from_mesh(box, 2.0), Conditions(()), PRESETS["PLA"], 30, 0, 1, None, None, "draft")
    assert T.topology_key(*args) == T.topology_key(*args, gap=0.0) != T.topology_key(*args, gap=2.0)
    with pytest.raises(ConditionError, match="lower Keep"):
        _cantilever(box, keep=80.0, gap=2.0)


def test_the_surface_stays_out_of_avoid_regions():
    # a full block of 20^3 elements around an empty 6^3 Avoid pocket: material is extended past the
    # design space only, not into the pocket (with phi > 0 taken as the design space it was filled)
    d = np.ones((20, 20, 20), dtype=np.float32)
    d[7:13, 7:13, 7:13] = 0.0
    space = np.ones(d.shape, dtype=bool)
    for given, filled in ((space, False), (None, True)):
        mesh = T.density_mesh(d, d, np.zeros(3), 1.0, space=given)
        assert (F.sample(F.body_from_mesh(mesh, 2.0), (10.0, 10.0, 10.0)) < 0) == filled


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
        assert len(m.split(only_watertight=False)) == 1
        cropped = F.intersect(F.body_from_mesh(box, 2.0), F.body_from_mesh(m, 2.0))
        shares.append(float((F.Context(Grid((0.25, 0.25, 0.25), (29.75, 9.75, 9.75), 0.5)).eval_full(cropped) < 0).mean()))
        c = m.triangles_center
        areas.append(float(m.area_faces[np.all((c > 0.5) & (c < [29.5, 9.5, 9.5]), axis=1)].sum()))
    assert max(shares) - min(shares) < 0.015, shares
    assert areas[0] > areas[1] > areas[2], areas
