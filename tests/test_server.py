import copy
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


def test_views_know_what_they_are_made_of_and_keep_what_the_editor_shows_finer():
    g = recipe()
    shown = ["mesh", "part_0", "implicit_body_0"]
    _, _, views, _ = build_run_graph(g, "preview", "draft", shown)
    key = views["mesh"].key
    assert views["part_0"].exact and not views["mesh"].exact and not views["implicit_body_0"].exact
    assert build_run_graph(g, "preview", "final", shown)[2]["mesh"].key == key    # the same at every quality
    moved = copy.deepcopy(g)
    moved["nodes"]["mesh"]["pos"] = [5, 5]                                        # editor data is not part of it
    assert build_run_graph(moved, "preview", "draft", shown)[2]["mesh"].key == key
    changed = copy.deepcopy(g)
    changed["nodes"]["wave"]["inputs"]["b"] = 0.3
    other = build_run_graph(changed, "preview", "draft", shown)[2]
    assert other["mesh"].key != key and other["part_0"].key == views["part_0"].key

    have = {"mesh": {"key": key, "level": "medium"},
            "implicit_body_0": {"key": views["implicit_body_0"].key, "level": "draft"}}
    _, targets, kept, _ = build_run_graph(g, "preview", "medium", ["mesh", "implicit_body_0"], have=have)
    assert kept["mesh"].kept and "mesh" not in targets                   # shown at medium already
    assert not kept["implicit_body_0"].kept and "__view__implicit_body_0" in targets   # only at draft
    _, targets, kept, _ = build_run_graph(g, "preview", "final", ["mesh"], have=have)
    assert not kept["mesh"].kept and "mesh" in targets                   # finer than shown
    _, targets, kept, _ = build_run_graph(g, "export", "final", ["mesh"], have={"mesh": {"key": key, "level": "final"}})
    assert not kept["mesh"].kept and "mesh" in targets                   # exports make everything


def test_refining_previews_compute_only_what_is_not_shown_finer_yet():
    shown = ["mesh", "part_0"]
    have = lambda events: {e["node"]: {"key": e["key"], "level": e["level"], "version": e["version"]}
                           for e in events if e["type"] == "view_ready"}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        first = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft", "visible": shown,
                                             "tag": {"version": 7}})
        start = next(e for e in first if e["type"] == "run_start")
        assert start["tag"] == {"version": 7} and start["keep_status"] == []
        ready = {e["node"]: e for e in first if e["type"] == "view_ready"}
        assert ready["mesh"]["level"] == "draft" and not ready["mesh"]["exact"]
        assert ready["part_0"]["level"] == "final" and ready["part_0"]["exact"]          # a CAD body: no finer view
        assert first[-1]["ok"] and first[-1]["sharper"] and first[-1]["shown"] == "draft"
        assert any(e["type"] == "phase" and "mesh" in e["text"] for e in first)        # the steps after the nodes
        shown_now = have(first)

        # the same graph at Medium: the CAD body is kept, the mesh is made finer
        second = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "medium",
                                              "visible": shown, "have": shown_now})
        ready = {e["node"]: e for e in second if e["type"] == "view_ready"}
        assert set(ready) == {"mesh"} and ready["mesh"]["level"] == "medium"
        assert second[-1]["ok"] and second[-1]["sharper"] and second[-1]["shown"] == "medium"
        shown_now.update(have(second))

        # a Draft run after a change elsewhere (here: nothing changed) keeps both, computes nothing and
        # leaves their nodes' cards alone; the view still shows Medium
        third = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                             "visible": shown, "have": shown_now})
        assert not any(e["type"] in ("view_ready", "node_start", "node_done") for e in third)
        assert third[-1]["shown"] == "medium"
        keep = next(e for e in third if e["type"] == "run_start")["keep_status"]
        assert {"mesh", "implicit_body_3", "import", "part_0"} <= set(keep)

        # a view the server does not have (any more) is made again
        stale = {n: dict(h, version=h["version"] + 1000) for n, h in shown_now.items()}
        fourth = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                              "visible": shown, "have": stale})
        assert {e["node"] for e in fourth if e["type"] == "view_ready"} == {"mesh", "part_0"}
        cad = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                           "visible": ["part_0"]})[-1]
        assert not cad["sharper"] and cad["shown"] is None                              # nothing gets finer


def test_a_detail_shows_part_of_a_body_finer_than_its_view():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": recipe(), "mode": "preview", "quality": "draft",
                                              "visible": ["implicit_body_0", "mesh", "part_0"]})
        ready = {e["node"]: e for e in events if e["type"] == "view_ready"}
        body, mesh, cad = ready["implicit_body_0"], ready["mesh"], ready["part_0"]
        assert body["voxel"] > 0.2 and body["floor"] == pytest.approx(0.01)     # a body's view: as fine as wanted
        assert mesh["voxel"] == pytest.approx(0.4) and mesh["floor"] == pytest.approx(0.1)   # a mesh: its own voxel
        assert cad["voxel"] is None and cad["floor"] is None                     # a CAD body is exact
        view, _ = decode(client.get("/api/view/implicit_body_0").content)
        lo, hi = view.min(axis=0).astype(float), view.max(axis=0).astype(float)
        box_lo = (lo + 0.25 * (hi - lo)).tolist()
        box_hi = [hi[0] - 0.25 * (hi[0] - lo[0]), hi[1] - 0.25 * (hi[1] - lo[1]), hi[2] + 1.0]   # the top's middle

        def detail(node, version, voxel, lo=box_lo, hi=box_hi):
            r = client.post("/api/detail", json={"node": node, "version": version, "lo": lo, "hi": hi, "voxel": voxel,
                                                 "tag": {"version": 3}})
            if r.status_code != 200:
                return r.status_code, []
            run_id, out = r.json()["run"], []
            while True:
                e = ws.receive_json()
                if e.get("run") != run_id:
                    continue
                out.append(e)
                if e["type"] == "run_done":
                    return 200, out

        status, got = detail("implicit_body_0", body["version"], 0.1)
        start = next(e for e in got if e["type"] == "run_start")
        assert status == 200 and start["mode"] == "detail" and start["tag"] == {"version": 3}
        assert got[-1]["ok"], got[-1]
        assert not any(e["type"] == "node_done" and e.get("cached") for e in got)   # cards keep their state
        done = next(e for e in got if e["type"] == "detail_ready")
        assert done["node"] == "implicit_body_0" and done["view_version"] == body["version"]
        assert 0.1 <= done["voxel"] < body["voxel"]
        verts, faces = decode(client.get("/api/view/detail:implicit_body_0").content)
        assert len(faces) == done["faces"] > 1000
        margin = 4 * done["voxel"]  # (triangles reaching into the box from up to 2 voxels outside)
        assert (verts >= np.array(box_lo) - margin).all() and (verts <= np.array(box_hi) + margin).all()
        edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert (counts == 1).any()                                  # open where it leaves the box
        assert (verts[:, 2] > hi[2] - 1.0).any() and verts[:, 2].max() <= view[:, 2].max() + body["voxel"]  # the top

        # a Mesh from Implicit Body's detail stops at its own voxel; CAD bodies and old views get none
        status, got = detail("mesh", mesh["version"], 0.01)
        assert status == 200 and next(e for e in got if e["type"] == "detail_ready")["voxel"] >= 0.1 - 1e-9
        assert detail("part_0", cad["version"], 0.1)[0] == 409
        assert detail("implicit_body_0", body["version"] - 1000, 0.1)[0] == 409
        assert detail("implicit_body_0", body["version"], 0.1, lo=[0, 0, 0], hi=[0, 1, 1])[0] == 422


def test_kept_views_are_coloured_again_only_when_the_field_changes():
    g = recipe()
    g["nodes"]["fy"] = {"type": "field.coordinate", "inputs": {"axis": "y"}}   # not part of the mesh
    body = lambda graph, have=None: {"graph": graph, "mode": "preview", "quality": "draft", "visible": ["mesh", "fy"],
                                     "slice": {"mode": "surface"}, "have": have or {}}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        first = run_and_collect(client, ws, body(g))
        view = next(e for e in first if e["type"] == "view_ready")
        paint = next(e for e in first if e["type"] == "paint_ready")
        assert paint["key"] and paint["level"] == "draft"
        have = {"mesh": {"key": view["key"], "level": view["level"], "version": view["version"],
                         "paint": {"key": paint["key"], "level": paint["level"]}}}
        again = run_and_collect(client, ws, body(g, have))
        assert not any(e["type"] in ("view_ready", "paint_ready") for e in again)    # the same colours
        g["nodes"]["fy"]["inputs"]["axis"] = "z"
        changed = run_and_collect(client, ws, body(g, have))
        assert not any(e["type"] == "view_ready" for e in changed)                     # the body is kept,
        repaint = next(e for e in changed if e["type"] == "paint_ready")              # coloured by the new field
        assert repaint["view_version"] == view["version"] and repaint["key"] != paint["key"]
        values = np.frombuffer(client.get("/api/view/paint:mesh").content, dtype=np.float32)
        verts, _ = decode(client.get("/api/view/mesh").content)
        assert np.abs(values - verts[:, 2]).max() < 1e-3                               # z at every vertex


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


def test_face_info_describes_a_face_of_a_cad_view():
    graph = {"version": 1, "nodes": {"import": {"type": "input.import_step", "inputs": {
        "path": str(ROOT / "samples" / "wall_bracket_space.step")}}}}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": graph, "mode": "preview", "quality": "draft",
                                              "visible": ["import"]})
        assert events[-1]["ok"], events[-1]
        ask = lambda node="import", **at: client.get(f"/api/faceinfo/{node}", params=at)
        bore = ask(body=0, face=6).json()                     # the pin hole: radius 4 mm, along Y
        assert set(bore) == {"type", "area", "centre", "normal", "axis", "radius", "hole", "triangles"}
        assert bore["type"] == "cylinder" and bore["hole"] is True and bore["normal"] is None
        assert bore["radius"] == pytest.approx(4.0, rel=1e-3) and abs(bore["axis"][1]) == pytest.approx(1.0)
        side = ask(body=0, face=0).json()
        assert side["type"] == "plane" and side["normal"] == pytest.approx([-1, 0, 0]) and side["axis"] is None
        assert ask(body=0, face=99).status_code == 404        # no such face
        assert ask(body=1, face=6).status_code == 404         # no such body
        assert ask("nowhere", body=0, face=6).status_code == 404   # not a CAD view
        assert ask(body=0).status_code == 422                 # face is required


def bracket_graph(**more):
    """The wall bracket (face 0: the plane x = 0; face 6: the pin hole, radius 4 mm at x = 50, z = 40)
    as an implicit body, and an offset of it by 0.5 mm."""
    link = lambda ref: {"link": ref}
    return {"version": 1, "nodes": {
        "import": {"type": "input.import_step", "inputs": {"path": str(ROOT / "samples" / "wall_bracket_space.step")}},
        "space": {"type": "input.pick_body", "inputs": {"bodies": link("import.bodies"), "index": 0}},
        "solid": {"type": "implicit.from_cad", "inputs": {"cad": link("space.body"), "band": 2.0}},
        "thick": {"type": "implicit.offset", "inputs": {"body": link("solid.out"), "distance": 0.5}}, **more}}


def test_face_at_finds_the_cad_face_behind_an_implicit_view():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ask = lambda node, x, y, z: client.get(f"/api/face_at/{node}", params={"x": x, "y": y, "z": z})
        assert ask("solid", 0, 0, 0).status_code == 404                    # nothing has run yet
        events = run_and_collect(client, ws, {"graph": bracket_graph(), "mode": "preview", "quality": "draft",
                                              "visible": ["import", "solid", "thick"]})
        assert events[-1]["ok"], events[-1]

        found = ask("solid", -0.5, 15, 25)                                 # just off the plane x = 0
        assert found.status_code == 200, found.text
        found = found.json()
        assert set(found) == {"source", "port", "face", "body", "distance", "same", "bodies", "info"}
        assert (found["source"], found["port"]) == ("space", "body")     # Pick Body's output, not the import
        assert (found["face"], found["body"], found["bodies"]) == (0, 0, 1) and found["same"] is True
        assert found["distance"] == pytest.approx(0.5, abs=1e-3)
        assert isinstance(found["info"]["type"], str) and found["info"]["type"] == "plane"
        assert found["info"]["area"] == pytest.approx(1500.0, rel=1e-3) and found["info"]["normal"][0] == pytest.approx(-1)

        bumpy = ask("thick", -1.0, 15, 25).json()                          # on the offset body's own surface
        assert (bumpy["source"], bumpy["face"]) == ("space", 0) and bumpy["distance"] == pytest.approx(1.0, abs=1e-3)
        bore = ask("solid", 52, 15, 40).json()                             # inside the pin hole, 2 mm from its wall
        assert bore["face"] == 6 and bore["distance"] == pytest.approx(2.0, abs=0.05)
        assert bore["info"]["type"] == "cylinder" and bore["info"]["hole"] is True and bore["info"]["area"] > 0

        cad = ask("import", -0.5, 15, 25).json()                           # a CAD view is its own source
        assert (cad["source"], cad["port"], cad["face"]) == ("import", "bodies", 0)

        assert ask("nowhere", 0, 0, 0).status_code == 404                  # no such view
        assert client.get("/api/face_at/solid", params={"x": 0, "y": 0}).status_code == 422   # z is required


def test_face_at_leaves_out_bodies_inside_groups_and_goes_through_group_inputs():
    link = lambda ref: {"link": ref}
    inner = {"solid": {"type": "implicit.from_cad", "inputs": {"cad": link("$in.cad"), "band": 2.0}}}
    g = bracket_graph(skin={"type": "group:skin", "inputs": {"cad": link("space.body")}})
    g["groups"] = {"skin": {"nodes": inner, "inputs": [], "outputs": [{"name": "out", "type": "implicit",
                                                                       "link": "solid.out"}]}}
    # the whole chain inside a group: its bodies cannot be linked to from outside
    chain = {k: g["nodes"][k] for k in ("import", "space", "solid")}
    g["groups"]["all"] = {"nodes": chain, "inputs": [], "outputs": [{"name": "out", "type": "implicit",
                                                                     "link": "solid.out"}]}
    g["nodes"]["packed"] = {"type": "group:all", "inputs": {}}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": g, "mode": "preview", "quality": "draft",
                                              "visible": ["skin", "packed"]})
        assert events[-1]["ok"], events[-1]
        ask = lambda node: client.get(f"/api/face_at/{node}", params={"x": -0.5, "y": 15, "z": 25})
        through = ask("skin").json()                                       # the group is fed from outside
        assert (through["source"], through["port"], through["face"]) == ("space", "body", 0)
        assert ask("packed").status_code == 404


def test_node_at_names_the_node_that_made_the_surface():
    link = lambda ref: {"link": ref}
    g = bracket_graph(
        flat={"type": "implicit.offset", "inputs": {"body": link("solid.out"), "distance": 0.0}},
        ball={"type": "implicit.ball", "inputs": {"centre": "10, 15, 25", "radius": 4.0}},
        cut={"type": "implicit.subtract", "inputs": {"a": link("solid.out"), "b": link("ball.out")}})
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ask = lambda node, x, y, z, **more: client.get(f"/api/node_at/{node}", params={"x": x, "y": y, "z": z, **more})
        assert ask("thick", 0, 0, 0).status_code == 404                    # nothing has run yet
        events = run_and_collect(client, ws, {"graph": g, "mode": "preview", "quality": "draft",
                                              "visible": ["import", "thick", "flat", "cut"]})
        assert events[-1]["ok"], events[-1]

        bump = ask("thick", -0.5, 15, 25).json()                           # the offset's own surface, 0.5 off the plane
        assert set(bump) == {"node", "inner", "type", "path"}
        assert (bump["node"], bump["inner"], bump["type"]) == ("thick", None, "implicit.offset")
        assert [s["node"] for s in bump["path"]] == ["thick"]              # the body below is 0.5 mm away
        assert bump["path"][0]["value"] == pytest.approx(0, abs=1e-4)

        plain = ask("flat", 0, 15, 25).json()                              # an offset by nothing: the part's face
        assert (plain["node"], plain["inner"], plain["type"]) == ("space", None, "input.pick_body")
        assert [s["node"] for s in plain["path"]] == ["flat", "solid", "space"]
        assert [s["value"] for s in plain["path"][:2]] == pytest.approx([0, 0], abs=1e-4) and plain["path"][2]["value"] is None

        wall = ask("cut", 14, 15, 25).json()                               # inside the part: the wall of the cut
        assert (wall["node"], wall["type"]) == ("ball", "implicit.ball")
        assert [s["node"] for s in wall["path"]] == ["cut", "ball"]
        outer = ask("cut", 0, 15, 25).json()                               # the part's outer face, away from the ball
        assert (outer["node"], outer["type"]) == ("space", "input.pick_body")
        assert [s["node"] for s in outer["path"]] == ["cut", "solid", "space"]

        cad = ask("import", 0, 15, 25).json()                              # a CAD view is made by its own node
        assert (cad["node"], cad["type"], cad["path"]) == ("import", "input.import_step", [{"node": "import", "value": None}])

        for node, name in (("thick", "thick"), ("flat", "space")):         # a vertex of the shown mesh, which is
            verts, _ = decode(client.get(f"/api/view/{node}").content)     # off the surface by up to its resolution
            found = ask(node, *map(float, verts[np.argmin(verts[:, 0])]), tol=0.2).json()
            assert found["node"] == name

        assert ask("nowhere", 0, 0, 0).status_code == 404                  # no such view
        assert client.get("/api/node_at/thick", params={"x": 0, "y": 0}).status_code == 422   # z is required
        assert ask("thick", 0, 0, "nan").status_code == 422
        assert ask("thick", -0.5, 15, 25, tol=0).json()["node"] == "thick"  # tol is at least 0.001


def test_node_at_reports_nodes_inside_a_group_as_the_group():
    link = lambda ref: {"link": ref}
    def skin(distance):
        inner = {"solid": {"type": "implicit.from_cad", "inputs": {"cad": link("$in.cad"), "band": 2.0}},
                 "fat": {"type": "implicit.offset", "inputs": {"body": link("solid.out"), "distance": distance}}}
        return {"nodes": inner, "inputs": [], "outputs": [{"name": "out", "type": "implicit", "link": "fat.out"}]}
    g = bracket_graph(textured={"type": "group:skin", "inputs": {"cad": link("space.body")}},
                      plain={"type": "group:flat", "inputs": {"cad": link("space.body")}})
    g["groups"] = {"skin": skin(0.5), "flat": skin(0.0)}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": g, "mode": "preview", "quality": "draft",
                                              "visible": ["textured", "plain"]})
        assert events[-1]["ok"], events[-1]
        ask = lambda node, x: client.get(f"/api/node_at/{node}", params={"x": x, "y": 15, "z": 25}).json()
        added = ask("textured", -0.5)                                      # made by the group's Offset Body
        assert (added["node"], added["inner"], added["type"]) == ("textured", "fat", "implicit.offset")
        assert [s["node"] for s in added["path"]] == ["textured"]
        through = ask("plain", 0)                                          # the part's face: the group's steps are one entry
        assert (through["node"], through["inner"], through["type"]) == ("space", None, "input.pick_body")
        assert [s["node"] for s in through["path"]] == ["plain", "space"]


def test_node_at_follows_the_matching_body_of_a_list():
    link = lambda ref: {"link": ref}
    g = {"version": 1, "nodes": {   # two blocks: x 0..40 and 40..90, y 0..20, z 0..24
        "import": {"type": "input.import_step", "inputs": {"path": str(ROOT / "samples" / "two_blocks.step")}},
        "solid": {"type": "implicit.from_cad", "inputs": {"cad": link("import.bodies"), "band": 2.0}},
        "bump": {"type": "implicit.offset", "inputs": {"body": link("solid.out"), "distance": 1.0}},
        "flat": {"type": "implicit.offset", "inputs": {"body": link("solid.out"), "distance": 0.0}}}}
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        events = run_and_collect(client, ws, {"graph": g, "mode": "preview", "quality": "draft",
                                              "visible": ["bump", "flat"]})
        assert events[-1]["ok"], events[-1]
        ask = lambda node, z: client.get(f"/api/node_at/{node}", params={"x": 65, "y": 10, "z": z}).json()
        grown = ask("bump", 25)                                            # 1 mm above the second block
        assert (grown["node"], grown["type"]) == ("bump", "implicit.offset") and len(grown["path"]) == 1
        top = ask("flat", 24)
        assert (top["node"], top["type"]) == ("import", "input.import_step")
        assert [s["node"] for s in top["path"]] == ["flat", "solid", "import"]


def test_find_creator_reads_the_same_item_of_every_list_and_only_items_that_reach_the_point(monkeypatch):
    from engine import creator, fields as F
    from engine.graph import Graph
    from engine.graph.core import Many
    link = lambda ref: {"link": ref}
    g = Graph.from_dict({"version": 1, "nodes": {
        "balls": {"type": "implicit.ball", "inputs": {}},
        "grown": {"type": "implicit.offset", "inputs": {"body": link("balls.out"), "distance": 0.0}},
        "__view__grown": {"type": "mesh.view", "inputs": {"body": link("grown.out")}}}})
    find = lambda outs, point: creator.find_creator(g, outs.get, "__view__grown", "mesh", point, 0.05)

    # a small ball grown by 2 mm has its surface where the big ball's is: the offset made it, the
    # big ball (another item of the list) is not what it was made from
    balls = Many([F.ball((0, 0, 0), 5), F.ball((8, 0, 0), 1)])
    outs = {"balls": {"out": balls}, "grown": {"out": Many([F.offset(balls[0], 0.5), F.offset(balls[1], 2.0)])}}
    assert find(outs, (5, 0, 0)) == ("grown", [("grown", pytest.approx(0, abs=1e-6))])

    balls = Many(F.ball((100.0 * i, 0, 0), 5) for i in range(50))
    calls = []
    real = creator.sample
    monkeypatch.setattr(creator, "sample", lambda f, p: calls.append(p) or real(f, p))
    outs = {"balls": {"out": balls}, "grown": {"out": Many(F.offset(b, 1.0) for b in balls)}}
    assert find(outs, (3106, 0, 0))[0] == "grown" and len(calls) == 2      # the 31st only: its bounds, its ball
    calls.clear()
    outs["grown"] = {"out": Many(F.offset(b, 0.0) for b in balls)}
    assert find(outs, (3105, 0, 0))[0] == "balls" and len(calls) == 2


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
