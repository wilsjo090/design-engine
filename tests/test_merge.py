import io

import numpy as np
import pytest
import trimesh

from engine.graph import Evaluator, Graph, Many, link
from engine.meshing import fuse, snap_float32
from engine.server import build_run_graph


def box(x=0.0, size=10.0):
    return trimesh.creation.box((size, size, size)).apply_translation((x, 0, 0))


def run_with_bodies(graph, bodies, targets=None, events=None):
    """Run a graph whose Import STEP nodes return the given bodies (a list for the node `src`,
    or a dict node id -> list)."""
    if not isinstance(bodies, dict):
        bodies = {"src": bodies}
    ev = Evaluator(graph, log=None, on_event=events.append if events is not None else None)
    original = ev._call

    def fake_call(node_id, ntype, kwargs):
        if ntype.type_id == "input.import_step":
            return {"bodies": Many(bodies[node_id])}
        return original(node_id, ntype, kwargs)
    ev._call = fake_call
    try:
        return ev.run(targets)
    finally:
        ev.close()


def test_fuse_joins_overlapping_parts():
    mesh, pieces, specks = fuse([box(0), box(8)])
    assert pieces == 1 and specks == 0
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(1000 + 1000 - 2 * 10 * 10, rel=1e-6)


def test_fuse_keeps_parts_that_do_not_overlap_apart():
    mesh, pieces, _ = fuse([box(0), box(15)])
    assert pieces == 2
    assert mesh.volume == pytest.approx(2000, rel=1e-6)


def test_fuse_drops_specks():
    # a 0.8 mm speck floating next to the part (textures can pinch off such bits)
    mesh, pieces, specks = fuse([box(0), box(8), box(20, 0.8)])
    assert pieces == 1 and specks == 1
    assert mesh.volume == pytest.approx(1800, rel=1e-6)


def test_fuse_rejects_open_mesh():
    holed = box(0)
    holed.update_faces(list(range(1, len(holed.faces))))  # drop one triangle
    with pytest.raises(ValueError, match="not closed"):
        fuse([box(8), holed])


def test_float32_slivers_are_removed(tmp_path):
    # The fused lamp had 4 triangles whose corners coincide in float32: fine in memory, but the
    # reloaded STL had edges shared by 4 triangles. Here: a box face split at a point 1e-9 from a corner.
    m = box(0)
    a, b, c = m.faces[0]
    d = len(m.vertices)
    sliver = trimesh.Trimesh(np.vstack([m.vertices, m.vertices[a] + 1e-9]),
                             np.vstack([m.faces[1:], [[a, b, d], [d, b, c], [a, d, c]]]), process=False)
    assert sliver.is_watertight
    sliver.export(tmp_path / "raw.stl")
    assert not trimesh.load(tmp_path / "raw.stl").is_watertight
    snap_float32(sliver).export(tmp_path / "clean.stl")
    assert trimesh.load(tmp_path / "clean.stl").is_watertight


def test_merge_node_fuses_and_reports(tmp_path):
    g = Graph()
    g.add("src", "input.import_step", path="unused")
    g.add("cad_mesh", "mesh.from_cad", cad=link("src.bodies"))
    g.add("merge", "mesh.merge", a=link("cad_mesh.mesh"))
    g.add("export", "output.export", mesh=link("merge.mesh"), path=str(tmp_path / "fused.stl"))
    events = []
    run_with_bodies(g, [box(0), box(8), box(16)], events=events)
    reloaded = trimesh.load(tmp_path / "fused.stl")
    assert reloaded.is_watertight and len(reloaded.split()) == 1
    assert reloaded.volume == pytest.approx(3000 - 2 * 200, rel=1e-5)
    assert any(e["type"] == "message" and "one solid" in e["text"] for e in events)


def test_merge_without_fuse_keeps_every_part(tmp_path):
    g = Graph()
    g.add("src", "input.import_step", path="unused")
    g.add("cad_mesh", "mesh.from_cad", cad=link("src.bodies"))
    g.add("merge", "mesh.merge", a=link("cad_mesh.mesh"), fuse="no")
    out = run_with_bodies(g, [box(0), box(8)])
    assert out["merge"]["mesh"].volume == pytest.approx(2000, rel=1e-6)  # overlap counted twice


def test_fused_marching_cubes_meshes_export_watertight(tmp_path):
    # Real pipeline: implicit bodies meshed (and decimated, as in exports), then fused with a CAD mesh.
    g = Graph()
    g.add("src", "input.import_step", path="unused")
    g.add("sdf", "implicit.from_cad", cad=link("src.bodies"), band=1.0)
    g.add("off", "implicit.offset", body=link("sdf.out"), distance=0.3)
    g.add("mesh", "mesh.from_implicit", body=link("off.out"), voxel=0.25, padding=1.0, keep=0.5)
    g.add("center", "input.import_step", path="unused")
    g.add("center_mesh", "mesh.from_cad", cad=link("center.bodies"))
    g.add("merge", "mesh.merge", a=link("mesh.mesh"), b=link("center_mesh.mesh"))
    g.add("export", "output.export", mesh=link("merge.mesh"), path=str(tmp_path / "lamp.stl"))
    # two small "petals", each reaching into the "center" box
    run_with_bodies(g, {"src": [box(-6, 4), box(6, 4)], "center": [box(0, 10)]})
    reloaded = trimesh.load(tmp_path / "lamp.stl")
    assert reloaded.is_watertight
    assert len(reloaded.split()) == 1
    assert 1000 < reloaded.volume < 1000 + 2 * 4.6 ** 3


def test_previews_below_final_do_not_fuse():
    g = {"version": 1, "nodes": {"merge": {"type": "mesh.merge", "inputs": {"fuse": "yes"}}}}
    for quality, expected in [("draft", "no"), ("medium", "no"), ("final", "yes")]:
        graph, *_ = build_run_graph(g, "preview", quality, [])
        assert graph.nodes["merge"]["inputs"]["fuse"] == expected


def test_decimation_reaches_keep_and_stays_closed():
    # the collapse threshold is scaled to the voxel size: unscaled it stopped at ~half the triangles
    from engine import fields as F
    from engine.grid import Grid
    from engine.meshing import decimate, field_to_mesh
    ball = F.body_from_mesh(trimesh.creation.icosphere(subdivisions=4, radius=8.0), 2.0)
    body = F.offset(ball, F.noise(3.0, octaves=2) * 0.4)
    for voxel in (0.1, 0.4):
        grid = Grid((-10, -10, -10), (10, 10, 10), voxel)
        full = field_to_mesh(F.Context(grid).eval_full(body), grid)
        small = decimate(full.copy(), 0.3, voxel)
        assert len(small.faces) == pytest.approx(0.3 * len(full.faces), rel=0.02)
        reloaded = trimesh.load(io.BytesIO(trimesh.exchange.stl.export_stl(small)), file_type="stl")
        assert reloaded.is_watertight
        assert abs(small.volume - full.volume) < 2e-3 * full.volume


def test_reduce_keeps_detail_and_flattens_flat_faces():
    # tolerance-based: a flat box ends up as a few big triangles, a textured ball keeps its detail
    from engine import fields as F
    from engine.grid import Grid
    from engine.meshing import field_to_mesh, reduce
    box = F.body_from_mesh(trimesh.creation.box((20, 20, 20)), 2.0)
    ball = F.offset(F.body_from_mesh(trimesh.creation.icosphere(subdivisions=4, radius=8.0).apply_translation((30, 0, 0)), 2.0),
                    F.noise(3.0, octaves=2) * 0.5)
    grid = Grid((-12, -12, -12), (40, 12, 12), 0.25)
    mesh = field_to_mesh(F.Context(grid).eval_full(F.union(box, ball)), grid)
    V, Fc = reduce(mesh.vertices, mesh.faces, 0.01)
    reduced = trimesh.Trimesh(V, Fc, process=False)
    on_box = lambda m: int((m.triangles_center[:, 0] < 15).sum())
    assert on_box(reduced) < 0.05 * on_box(mesh)
    assert len(Fc) - on_box(reduced) > 0.3 * (len(mesh.faces) - on_box(mesh))
    reloaded = trimesh.load(io.BytesIO(trimesh.exchange.stl.export_stl(reduced)), file_type="stl")
    assert reloaded.is_watertight
    assert abs(reduced.volume - mesh.volume) < 1e-3 * mesh.volume


def test_reduce_removes_folded_triangles():
    from engine.meshing import _clean, _edge_problems
    box = trimesh.creation.box((1, 1, 1))
    fin = np.array([[0, 1, 7], [0, 7, 1]])                       # two triangles back to back: a fold
    v, f = _clean(np.asarray(box.vertices, float), np.concatenate([box.faces, fin]))
    assert len(f) == 12 and not any(map(len, _edge_problems(v, f)))
    triple = np.concatenate([box.faces, box.faces[:1], box.faces[:1, ::-1]])  # a surface triangle + a fold on it
    assert any(map(len, _edge_problems(box.vertices, triple)))
    v, f = _clean(np.asarray(box.vertices, float), triple)
    assert len(f) == 12 and not any(map(len, _edge_problems(v, f)))   # the surface keeps its triangle


def test_reduce_mesh_node_runs_in_a_graph():
    g = Graph()
    g.add("src", "input.import_step", path="unused")
    g.add("m", "mesh.from_cad", cad=link("src.bodies"))
    g.add("r", "mesh.reduce", mesh=link("m.mesh"), tolerance=0.01)
    box = trimesh.creation.box((10, 10, 10)).subdivide().subdivide()   # 768 triangles on flat faces
    out = run_with_bodies(g, [box], targets=["r"])["r"]["mesh"]
    out = out[0] if isinstance(out, list) else out
    assert len(out.faces) == 12 and out.is_watertight                  # a box needs 12
