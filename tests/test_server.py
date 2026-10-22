import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.graph.core import GROUP_SEP
from engine.server import build_run_graph, create_app

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    g = json.loads((ROOT / "recipes" / "wave_offset.json").read_text())
    g["nodes"]["import"]["inputs"]["path"] = str(ROOT / "samples" / "test_part.step")
    return g


def run_and_collect(client, ws, body):
    run_id = client.post("/api/run", json=body).json()["run"]
    events = []
    while True:
        e = ws.receive_json()
        if e.get("run") != run_id:
            continue
        events.append(e)
        if e["type"] == "run_done":
            return events


def decode(blob):
    nv, nf = np.frombuffer(blob[:8], dtype=np.uint32)
    verts = np.frombuffer(blob[8:8 + nv * 12], dtype=np.float32).reshape(-1, 3)
    faces = np.frombuffer(blob[8 + nv * 12:], dtype=np.uint32).reshape(-1, 3)
    assert len(faces) == nf and faces.max() < nv
    return verts, faces


def test_build_run_graph_applies_quality_and_views():
    g, targets, views, slices = build_run_graph(recipe(), "preview", "draft", ["mesh", "implicit_body_0", "result_0"])
    assert g.nodes["mesh"]["inputs"]["voxel"] == pytest.approx(0.4)   # 4x coarser draft
    assert "__view__implicit_body_0" in targets                        # bodies are meshed for display
    assert "result_0" not in views and "result_0" in slices          # plain fields: shown as a slice
    assert not any(g.nodes[t]["type"].startswith("output.") for t in targets)


def test_guide_shapes_are_checked_but_not_meshed_for_the_view():
    # the editor draws Ball, Bar and Box itself (live while dragged); the engine still runs them for errors
    g = recipe()
    g["nodes"]["guide"] = {"type": "implicit.ball", "inputs": {"centre": "1, 2, 3", "radius": 4}}
    _, targets, views, _ = build_run_graph(g, "preview", "draft", ["guide"])
    assert "guide" in targets and "guide" not in views and "__view__guide" not in targets


def test_preview_run_streams_events_and_serves_mesh():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        cat = client.get("/api/catalog").json()
        assert any(n["type"] == "implicit.offset" for n in cat["nodes"])
        assert not any(n["type"] == "mesh.view" for n in cat["nodes"])   # hidden helper

        events = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                              "visible": ["mesh", "implicit_body_6"]})
        done = events[-1]
        assert done["ok"], done
        ready = {e["node"]: e for e in events if e["type"] == "view_ready"}
        assert set(ready) == {"mesh", "implicit_body_6"}
        assert any(e["type"] == "node_done" and e["node"] == "mesh" for e in events)

        verts, faces = decode(client.get("/api/view/mesh").content)
        assert len(faces) == ready["mesh"]["faces"] > 1000
        assert verts[:, 0].min() < 0 and verts[:, 0].max() > 60      # the part plus its wave

        # Second identical run: everything cached
        events = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                              "visible": ["mesh"]})
        assert all(e.get("cached", True) for e in events if e["type"] == "node_done")


def test_field_nodes_are_shown_as_slices():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {
            "graph": recipe(), "mode": "preview", "quality": "draft", "visible": ["wave"],
            "slice": {"axis": "z", "position": 10.0, "lo": [0, 0, 0], "hi": [60, 40, 15]}})
        assert events[-1]["ok"], events[-1]
        ready = [e for e in events if e["type"] == "slice_ready"]
        assert len(ready) == 1 and ready[0]["node"] == "wave"
        assert ready[0]["rect"] == [0.0, 0.0, 60.0, 40.0]
        assert 0 <= ready[0]["min"] < 0.1 and 1.0 < ready[0]["max"] <= 0.35 * np.pi + 1e-3  # 0.35*acos(..)
        blob = client.get("/api/view/slice:wave").content
        w, h = np.frombuffer(blob[:8], dtype=np.uint32)
        img = np.frombuffer(blob[8:], dtype=np.float32).reshape(h, w)
        assert w == 601 and h == 401 and np.isfinite(img).all()
        assert np.allclose(img[0], img[-1])            # the wave only varies along x


def test_errors_are_attached_to_nodes():
    g = recipe()
    g["nodes"]["mesh"]["inputs"]["voxel"] = 0.00001
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        done = run_and_collect(client, ws, {"graph": g, "mode": "preview", "quality": "final",
                                            "visible": ["mesh"]})[-1]
        assert not done["ok"] and done["problems"][0][0] == "mesh"


def test_a_failing_node_does_not_hide_the_rest_of_the_preview():
    # another STEP file under supports picked on the previous one (the wall bracket's back face and
    # pin hole): the new part is still shown
    link = lambda ref: {"link": ref}
    g = {"version": 1, "nodes": {
        "import": {"type": "input.import_step", "inputs": {"path": str(ROOT / "samples" / "bracket.step")}},
        "space": {"type": "input.pick_body", "inputs": {"bodies": link("import.bodies"), "index": 0}},
        "petg": {"type": "sim.material", "inputs": {"preset": "PETG"}},
        "wall": {"type": "sim.fixed", "inputs": {"cad": link("space.body"), "faces": "0"}},
        "hang": {"type": "sim.load", "inputs": {"cad": link("space.body"), "faces": "6", "kind": "bearing",
                                                "amount": 200.0, "direction": "-Z", "add_to": link("wall.conditions")}},
        "topology": {"type": "sim.topology", "inputs": {"space": link("space.body"), "material": link("petg.material"),
                                                        "conditions": link("hang.conditions"), "keep": 25.0}}}}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": g, "mode": "preview", "quality": "draft",
                                              "visible": ["import", "topology"]})
    done = events[-1]
    assert not done["ok"] and [n for n, _ in done["problems"]] == ["hang"]
    assert any(e["type"] == "node_error" and e["node"] == "hang" for e in events)
    assert [e["node"] for e in events if e["type"] == "view_ready"] == ["import"]
    assert done["left_out"] == ["hang", "topology"]   # the editor stops showing their old results


def test_gpu_setting_can_be_switched():
    from engine import backend
    with TestClient(create_app()) as client:
        s = client.get("/api/settings").json()
        assert set(s) == {"gpu", "gpu_available"}
        assert client.post("/api/settings", json={"gpu": False}).json()["gpu"] is False
        on = client.post("/api/settings", json={"gpu": True}).json()["gpu"]
        assert on == backend.gpu_available()          # only turns on when a GPU exists
        backend.set_gpu(False)                         # leave the suite on the CPU path
        mem = client.get("/api/memory").json()
        assert {"gpu_free", "gpu_total"} <= set(mem)


def test_file_api_round_trip(tmp_path):
    with TestClient(create_app()) as client:
        path = tmp_path / "r.json"
        assert client.post("/api/file", params={"path": str(path)}, json=recipe()).status_code == 200
        assert client.get("/api/file", params={"path": str(path)}).json()["nodes"]["wave"]["type"] == "math.multiply"
        listing = client.get("/api/browse", params={"path": str(tmp_path), "exts": ".json"}).json()
        assert [e["name"] for e in listing["entries"]] == ["r.json"]
        assert client.post("/api/file", params={"path": str(tmp_path / "x.txt")}, json=recipe()).status_code == 400


def decode_faces(blob):
    """/api/faces -> (face per triangle, body per triangle, copies of one part?, group lists)."""
    b = np.frombuffer(blob, dtype=np.uint32)
    n = int(b[0])
    ids, bodies = b[1:1 + n], b[1 + n:1 + 2 * n]
    same, lists = bool(b[1 + 2 * n]), int(b[2 + 2 * n])
    pos, groups = 3 + 2 * n, []
    for _ in range(lists):
        groups.append(b[pos + 1:pos + 1 + b[pos]])
        pos += 1 + int(b[pos])
    assert pos == len(b)
    return ids, bodies, same, groups


def test_cad_views_tell_which_face_each_triangle_is():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                              "visible": ["import", "mesh"]})
        assert events[-1]["ok"], events[-1]
        _, faces = decode(client.get("/api/view/import").content)
        ids, bodies, same, groups = decode_faces(client.get("/api/faces/import").content)
        assert len(ids) == len(faces) and (bodies == 0).all() and same and len(groups) == 1
        assert ids.max() + 1 == len(groups[0])                    # one face number per triangle
        assert len(set(groups[0])) < len(groups[0])               # fillets grouped with their neighbours
        assert client.get("/api/faces/mesh").status_code == 404   # not a CAD body


def test_face_picking_tells_different_bodies_apart():
    import trimesh as tm
    from engine.server import pack_faces
    def body(mesh, k):
        mesh.face_attributes["cad_face"] = np.repeat(np.arange(6), 2).astype(np.int32)  # a box: 6 faces
        mesh.metadata["cad_body"] = k
        return mesh
    cube, slab = tm.creation.box((10, 10, 10)), tm.creation.box((40, 10, 2))
    _, bodies, same, groups = decode_faces(pack_faces([body(cube, 3), body(slab, 4)], 24))
    assert not same and len(groups) == 2 and list(bodies[:12]) == [3] * 12 and list(bodies[12:]) == [4] * 12
    copies = [body(tm.creation.box((10, 10, 10)), 0), body(tm.creation.box((13, 13, 13)), 1)]   # a scaled copy
    _, _, same, groups = decode_faces(pack_faces(copies, 24))
    assert same and len(groups) == 1


def test_group_nodes_are_shown_and_report_on_the_group(tmp_path):
    g = recipe()
    # the wave chain as a group: x coordinate -> ... -> wave, with the density Number as setting
    inner = {k: g["nodes"].pop(k) for k in ["implicit_body_1", "implicit_body_2", "result_14", "result_13",
                                            "result_15", "result_0", "result_16", "wave"]}
    g["groups"] = {"wave_group": {"nodes": inner, "inputs": [],
                                  "outputs": [{"name": "out", "type": "scalar", "link": "wave.out"}]}}
    g["nodes"]["waves"] = {"type": "group:wave_group", "inputs": {"implicit_body_2": 1.909859}}
    g["nodes"]["implicit_body_0"]["inputs"]["distance"] = {"link": "waves.out"}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": g, "mode": "preview", "quality": "draft",
                                              "visible": ["mesh", "waves"],
                                              "slice": {"axis": "z", "position": 10.0, "lo": [0, 0, 0], "hi": [60, 40, 15]}})
        assert events[-1]["ok"], events[-1]
        assert any(e["type"] == "slice_ready" and e["node"] == "waves" for e in events)   # the group's field
        done = [e for e in events if e["type"] == "node_done" and e["node"] == "waves"]
        assert len(done) == 1 and done[0]["inner"] == "wave"       # reported once, when its output is ready
        assert not any(GROUP_SEP in e.get("node", "") for e in events)


def test_fields_can_colour_the_surfaces():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                              "visible": ["mesh", "implicit_body_1"], "slice": {"mode": "surface"}})
        assert events[-1]["ok"], events[-1]
        assert not any(e["type"] == "slice_ready" for e in events)
        paint = next(e for e in events if e["type"] == "paint_ready")
        view = next(e for e in events if e["type"] == "view_ready" and e["node"] == "mesh")
        assert paint["node"] == "mesh" and paint["field"] == "implicit_body_1"
        assert paint["view_version"] == view["version"]
        values = np.frombuffer(client.get("/api/view/paint:mesh").content, dtype=np.float32)
        verts, _ = decode(client.get("/api/view/mesh").content)
        assert len(values) == len(verts)
        assert np.abs(values - verts[:, 0]).max() < 1e-3     # the x coordinate, at every vertex


def test_print_check_reports_the_shown_mesh():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft", "visible": ["mesh"]})
        r = client.post("/api/check", json={"node": "mesh", "density": 1.0, "thickness": True, "min_wall": 1.0}).json()
        assert r["bodies"] == 1 and r["cavities"] == 0 and r["open_edges"] == 0
        assert r["weight"] == pytest.approx(r["volume"] / 1000)
        assert 40 < r["size"][0] < 80
        assert 0 < r["thickness"]["thinnest"] <= r["thickness"]["median"]
        values = np.frombuffer(client.get("/api/view/paint:mesh").content, dtype=np.float32)
        assert len(values) == r["vertices"]
        assert client.post("/api/check", json={"node": "nothing"}).status_code == 404


def test_autosave_and_presets_round_trip(tmp_path, monkeypatch):
    from engine import server
    monkeypatch.setattr(server, "PRESETS", tmp_path / "presets")
    with TestClient(create_app()) as client:
        path = str(tmp_path / "r.json")
        assert client.get("/api/autosave", params={"path": path}).json() == {}
        client.post("/api/autosave", json={"path": path, "recipe": recipe()})
        saved = client.get("/api/autosave", params={"path": path}).json()
        assert saved["recipe"]["nodes"]["wave"]["type"] == "math.multiply" and saved["time"] > 0
        assert client.get("/api/autosave").json() == {}               # a new recipe has its own slot
        client.delete("/api/autosave", params={"path": path})
        assert client.get("/api/autosave", params={"path": path}).json() == {}

        clip = {"kind": "design-engine/nodes", "version": 1, "nodes": [{"id": "n", "type": "input.number",
                                                                        "inputs": {"value": 2}, "links": {}, "pos": [0, 0]}]}
        assert client.post("/api/presets/my wave", json=clip).status_code == 200
        assert client.get("/api/presets").json() == [{"name": "my wave", "nodes": 1, "doc": ""}]
        assert client.get("/api/presets/my wave").json()["nodes"][0]["type"] == "input.number"
        assert client.post("/api/presets/bad*name", json=clip).status_code == 400
        client.delete("/api/presets/my wave")
        assert client.get("/api/presets").json() == []


def test_exports_queue_up_and_previews_wait(tmp_path):
    g = recipe()
    g["nodes"]["export"]["inputs"]["path"] = str(tmp_path / "a.stl")
    g["nodes"]["preview"]["inputs"]["path"] = str(tmp_path / "a.png")
    g2 = json.loads(json.dumps(g))
    g2["nodes"]["export"]["inputs"]["path"] = str(tmp_path / "b.stl")
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        client.post("/api/run", json={"graph": g, "mode": "export", "quality": "draft", "label": "first"})
        client.post("/api/run", json={"graph": g2, "mode": "export", "quality": "draft", "label": "second"})
        client.post("/api/run", json={"graph": g, "mode": "preview", "quality": "draft", "visible": ["mesh"]})
        queued = client.get("/api/queue").json()
        assert [e["label"] for e in queued["exports"]] == ["second"] and queued["preview_waiting"]
        finished = []
        while len(finished) < 3:
            e = ws.receive_json()
            if e["type"] == "run_done":
                assert e["ok"], e
                finished.append(e.get("label"))
        assert finished == ["first", "second", "preview"]            # nothing was cancelled
        assert (tmp_path / "a.stl").exists() and (tmp_path / "b.stl").exists()
        state = client.get("/api/queue").json()
        assert state["current"] is None and state["exports"] == [] and not state["preview_waiting"]
