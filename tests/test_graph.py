import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from engine import sdf
from engine.graph import ARRAY_CACHE, Evaluator, Graph, GraphError, Many, link, node_catalog

ROOT = Path(__file__).resolve().parents[1]
TEST_STEP = ROOT / "samples" / "test_part.step"


@pytest.fixture(autouse=True)
def fresh_cache():
    ARRAY_CACHE.clear()


def wave_graph(tmp_path, amplitude=0.3, voxel=0.5):
    g = Graph()
    g.add("import", "input.import_step", path=str(TEST_STEP))
    g.add("part", "input.pick_body", bodies=link("import.bodies"))
    g.add("sdf", "implicit.from_cad", cad=link("part.body"), band=1.5)
    g.add("x", "field.coordinate", axis="x")
    g.add("cos", "math.cos", x=link("x.out"))
    g.add("tri", "math.acos", x=link("cos.out"))
    g.add("amp", "input.number", value=amplitude)
    g.add("wave", "math.multiply", a=link("tri.out"), b=link("amp.out"))
    g.add("offset", "implicit.offset", body=link("sdf.out"), distance=link("wave.out"))
    g.add("mesh", "mesh.from_implicit", body=link("offset.out"), voxel=voxel, padding=1.5)
    return g


def count_sdf_calls(monkeypatch):
    calls = []
    real = sdf.mesh_to_sdf

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr("engine.fields.mesh_to_sdf", spy)
    return calls


def test_runs_and_produces_watertight_mesh(tmp_path):
    out = Evaluator(wave_graph(tmp_path), log=None).run()
    mesh = out["mesh"]["mesh"]
    assert mesh.is_watertight
    assert mesh.volume > 60 * 40 * 15 * 0.8  # offset only grows the part


def test_exported_stl_stays_watertight(tmp_path):
    # STL has no connectivity; readers (and slicers) re-merge vertices by position.
    g = wave_graph(tmp_path, voxel=0.4)
    g.add("export", "output.export", mesh=link("mesh.mesh"), path=str(tmp_path / "out.stl"))
    Evaluator(g, log=None).run()
    assert trimesh.load(tmp_path / "out.stl").is_watertight


def test_parameter_change_reuses_upstream_work(tmp_path, monkeypatch):
    calls = count_sdf_calls(monkeypatch)
    g = wave_graph(tmp_path)
    ev = Evaluator(g, log=None)
    first = ev.run()["mesh"]["mesh"].volume
    assert len(calls) == 1

    g.set("amp", "value", 0.6)
    second = ev.run()["mesh"]["mesh"].volume
    assert "import" not in ev.computed          # STEP file not re-read
    assert set(ev.computed) == {"amp", "wave", "offset", "mesh"}
    assert len(calls) == 1                       # distance field reused from the array cache
    assert second > first

    ev.run()
    assert ev.computed == []                     # nothing changed, nothing recomputed


def test_json_round_trip(tmp_path):
    g = wave_graph(tmp_path)
    path = tmp_path / "recipe.json"
    g.save(path)
    loaded = Graph.load(path)
    assert loaded.to_dict() == json.loads(json.dumps(g.to_dict()))


def test_validation_reports_problems(tmp_path):
    g = wave_graph(tmp_path)
    g.set("offset", "body", link("import.bodies"))   # cad into implicit
    g.set("wave", "a", link("nowhere.out"))
    g.add("bad", "math.nonexistent")
    with pytest.raises(GraphError) as err:
        g.validate()
    msg = str(err.value)
    assert "cannot connect cad" in msg and "nowhere" in msg and "unknown node type" in msg


def test_cycle_detected():
    g = Graph()
    g.add("a", "math.add", a=link("b.out"), b=1)
    g.add("b", "math.add", a=link("a.out"), b=1)
    with pytest.raises(GraphError, match="cycle"):
        g.validate()


def test_lists_map_through_nodes(tmp_path):
    boxes = Many(trimesh.creation.box((4, 4, 4)).apply_translation((10 * i, 0, 0)) for i in range(3))
    g = Graph()
    g.add("sdf", "implicit.from_cad", cad=link("src.bodies"), band=1.0)
    g.add("off", "implicit.offset", body=link("sdf.out"), distance=0.5)
    g.add("mesh", "mesh.from_implicit", body=link("off.out"), voxel=0.25, padding=1.0)
    # Feed a list directly by caching a fake source node.
    g.add("src", "input.import_step", path="unused")
    ev = Evaluator(g, log=None)
    original = ev._call

    def fake_call(node_id, ntype, kwargs):
        if ntype.type_id == "input.import_step":
            return {"bodies": boxes}
        return original(node_id, ntype, kwargs)
    ev._call = fake_call
    try:
        meshes = ev.run()["mesh"]["mesh"]   # 3 items -> runs in worker processes
    finally:
        ev.close()
    assert isinstance(meshes, Many) and len(meshes) == 3
    for i, m in enumerate(meshes):
        assert m.is_watertight
        assert np.allclose(m.centroid[0], 10 * i, atol=0.05)
        assert 5 ** 3 * 0.9 < m.volume < 5 ** 3  # 4 mm box grown 0.5 mm, rounded edges


def test_preview_of_many_bodies_gets_the_detail_of_the_group(monkeypatch):
    # 144 petals at Final once got 512 voxels each (finer than the export) and ran out of memory
    import engine.graph.nodes as nodes
    boxes = Many(trimesh.creation.box((4, 4, 4)).apply_translation((10 * i, 0, 0)) for i in range(20))
    voxels = []
    real = nodes._mesh
    monkeypatch.setattr(nodes, "_mesh", lambda body, bounds, voxel, *a: voxels.append(voxel) or real(body, bounds, voxel, *a))
    g = Graph()
    g.add("src", "input.import_step", path="unused")
    g.add("sdf", "implicit.from_cad", cad=link("src.bodies"), band=1.0)
    g.add("view", "mesh.view", body=link("sdf.out"), cells=128, padding=2.0)
    ev = Evaluator(g, workers=1, log=None)
    original = ev._call
    ev._call = lambda node_id, ntype, kwargs: ({"bodies": boxes} if ntype.type_id == "input.import_step"
                                               else original(node_id, ntype, kwargs))
    meshes = ev.run()["view"]["mesh"]
    assert len(meshes) == 20 and all(m.is_watertight for m in meshes)
    assert voxels == [pytest.approx(8.0 / nodes.VIEW_MIN_CELLS)] * 20  # 40 per box, not 128 each

    voxels.clear()  # a single body: `cells` along its own longest side, as before
    g.add("pick", "input.pick_body", bodies=link("src.bodies"))
    g.add("sdf1", "implicit.from_cad", cad=link("pick.body"), band=1.0)
    g.add("view1", "mesh.view", body=link("sdf1.out"), cells=128, padding=2.0)
    ev.run(["view1"])
    assert voxels == [pytest.approx(8.0 / 128)]


def test_catalog_is_complete():
    types = {t["type"] for t in node_catalog()}
    for required in ["input.import_step", "implicit.offset", "implicit.smoothen", "mesh.from_implicit",
                     "output.export", "math.acos"]:
        assert required in types


def test_recipes_are_valid():
    for recipe in (ROOT / "recipes").glob("*.json"):
        Graph.load(recipe).validate()


def test_events_report_progress_and_cache(tmp_path):
    events = []
    g = wave_graph(tmp_path)
    ev = Evaluator(g, log=None, on_event=events.append)
    ev.run()
    assert {"type": "node_start", "node": "mesh"} in events
    assert any(e["type"] == "node_done" and e["node"] == "mesh" and not e["cached"] for e in events)
    events.clear()
    ev.run()
    assert all(e["type"] == "node_done" and e["cached"] for e in events)


def test_cancel_stops_run(tmp_path):
    g = wave_graph(tmp_path)
    ev = Evaluator(g, log=None)
    ev.on_event = lambda e: ev.cancel() if e.get("node") == "sdf" else None
    with pytest.raises(GraphError, match="cancelled"):
        ev.run()
    assert "mesh" not in ev.computed


def test_unfinished_node_does_not_block_other_targets(tmp_path):
    g = wave_graph(tmp_path)
    g.add("half_wired", "implicit.offset")          # required 'body' not connected
    assert any(n == "half_wired" for n, _ in g.problems())
    Evaluator(g, log=None).run(["mesh"])            # still runs


def test_node_errors_name_the_node(tmp_path):
    g = wave_graph(tmp_path)
    g.set("mesh", "voxel", 0.0001)                  # absurd grid
    with pytest.raises(GraphError) as err:
        Evaluator(g, log=None).run(["mesh"])
    assert err.value.problems[0][0] == "mesh" and "too large" in err.value.problems[0][1]


def test_smoothen_looks_the_same_at_any_resolution():
    box = trimesh.creation.box((10, 10, 10))
    from engine import fields as F
    from engine.grid import Grid
    from engine.meshing import field_to_mesh
    body = F.smoothen(F.body_from_mesh(box, band=2.0), iterations=3, grid_size=0.5)
    vols = []
    for voxel in (0.2, 0.4):
        grid = Grid.from_bounds(box.bounds, voxel, 1.0)
        vols.append(field_to_mesh(F.Context(grid).eval_full(body), grid).volume)
    assert abs(vols[0] - vols[1]) / vols[0] < 0.01
    assert vols[0] < 1000 * 0.995                   # corners actually got rounded


def test_value_ranges_follow_the_formulas():
    from engine import fields as F
    wave = F.remap(F.acos(F.cos(F.noise(55, 3, 0.45, 3) * 140)), 0, 3.141593, 0, 1)
    height = F.remap(F.noise(20, 2, 0.5, 7), -1, 1, 0.6, 1.4)
    assert wave.range == pytest.approx((0, 1), abs=1e-5)
    assert (wave * (height * 0.63)).range == pytest.approx((0, 0.882), abs=1e-5)
    assert F.smoothstep(F.X, 0, 1).range == (0, 1)          # clamped, whatever goes in
    assert F.X.range is None and (1 / F.noise(5)).range is None  # depends on the part / divides by 0


def test_remap_auto_uses_the_input_range():
    g = Graph()
    g.add("noise", "field.noise", scale=10)
    g.add("auto", "math.remap", x=link("noise.out"), auto="yes", to_low=2, to_high=4)
    g.add("manual", "math.remap", x=link("noise.out"), from_low=-1, from_high=1, to_low=2, to_high=4)
    g.add("x", "field.coordinate", axis="x")
    events = []
    out = Evaluator(g, log=None, on_event=events.append).run(["auto", "manual"])
    assert out["auto"]["out"].range == out["manual"]["out"].range == (2, 4)
    done = {e["node"]: e for e in events if e["type"] == "node_done"}
    assert done["noise"]["range"] == [-1, 1] and done["auto"]["range"] == [2, 4]   # shown on the cards
    g.add("bad", "math.remap", x=link("x.out"), auto="yes")
    with pytest.raises(GraphError, match="cannot tell"):
        Evaluator(g, log=None).run(["bad"])


def _wave_group():
    # $in.x * amp (a Number node: the group's setting) -> output "out"
    return {"nodes": {"amp": {"type": "input.number", "inputs": {"value": 0.5}},
                      "m": {"type": "math.multiply", "inputs": {"a": {"link": "$in.x"}, "b": {"link": "amp.out"}}}},
            "inputs": [{"name": "x", "type": "scalar"}],
            "outputs": [{"name": "out", "type": "scalar", "link": "m.out"}]}


def _value(result):
    from engine.fields import Field, sample
    return sample(result, (0.0, 0.0, 0.0)) if isinstance(result, Field) else float(result)


def test_groups_expand_into_their_nodes():
    recipe = {"version": 1, "groups": {"wave": _wave_group()}, "nodes": {
        "n": {"type": "input.number", "inputs": {"value": 3.0}},
        "g": {"type": "group:wave", "inputs": {"x": {"link": "n.out"}, "amp": 2.0}},   # setting overridden
        "g2": {"type": "group:wave", "inputs": {"x": {"link": "n.out"}}},               # setting at its default
        "sum": {"type": "math.add", "inputs": {"a": {"link": "g.out"}, "b": {"link": "g2.out"}}}}}
    g = Graph.from_dict(recipe)
    assert "g" not in g.nodes and "g__m" in g.nodes and "g2__amp" in g.nodes
    assert g.nodes["sum"]["inputs"]["a"] == {"link": "g__m.out"}
    assert g.owners["g__m"] == "g" and g.aliases["g.out"] == "g__m.out" and g.groups["g"] == [("out", "scalar")]
    results = Evaluator(g, log=None).run(["sum"])
    assert _value(results["sum"]["out"]) == pytest.approx(3 * 2 + 3 * 0.5)


def test_groups_nest_and_unknown_groups_are_reported():
    outer = {"nodes": {"inner": {"type": "group:wave", "inputs": {"x": {"link": "$in.x"}, "amp": 4.0}}},
             "inputs": [{"name": "x", "type": "scalar"}],
             "outputs": [{"name": "out", "type": "scalar", "link": "inner.out"}]}
    recipe = {"version": 1, "groups": {"wave": _wave_group(), "outer": outer}, "nodes": {
        "n": {"type": "input.number", "inputs": {"value": 1.5}},
        "o": {"type": "group:outer", "inputs": {"x": {"link": "n.out"}}},
        "use": {"type": "math.add", "inputs": {"a": {"link": "o.out"}, "b": 0.0}},
        "bad": {"type": "group:missing"}}}
    g = Graph.from_dict(recipe)
    assert g.nodes["use"]["inputs"]["a"] == {"link": "o__inner__m.out"} and g.owners["o__inner__m"] == "o"
    assert any(n == "bad" and "unknown node type" in m for n, m in g.problems())
    assert _value(Evaluator(g, log=None).run(["use"])["use"]["out"]) == pytest.approx(6.0)


def test_face_selections_are_resolved_by_name():
    recipe = {"version": 1, "selections": {"top": "10, 11", "rim": "3"}, "nodes": {
        "m": {"type": "field.face_mask", "inputs": {"faces": "top, rim 7"}},
        "u": {"type": "field.face_mask", "inputs": {"faces": "nope"}}}}
    g = Graph.from_dict(recipe)
    assert g.nodes["m"]["inputs"]["faces"] == "10, 11, 3 7"
    assert g.nodes["u"]["inputs"]["faces"] == "nope"           # the node reports unknown names
    assert recipe["nodes"]["m"]["inputs"]["faces"] == "top, rim 7"   # the recipe itself is untouched
