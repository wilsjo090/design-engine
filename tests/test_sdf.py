from pathlib import Path

import igl
import numpy as np
import pytest
import trimesh

from engine import load_step, tessellate
from engine.grid import Grid
from engine.sdf import MeshQuery, mesh_to_sdf

ROOT = Path(__file__).resolve().parents[1]


def reference(mesh, points):
    """libigl's exact signed distance with (exact, slow) winding-number signs. Note: its
    FAST_WINDING_NUMBER mode returns approximate distances (errors ~0.02 mm), so not that."""
    return igl.signed_distance(np.asarray(points, dtype=np.float64), np.asarray(mesh.vertices),
                               np.asarray(mesh.faces, dtype=np.int64),
                               sign_type=igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)[0]


@pytest.fixture(scope="module")
def cad_part():
    # filleted box with a hole: has T-junction cracks at the fillet corners
    return tessellate(load_step(ROOT / "samples" / "test_part.step")[0], tolerance=0.01)


@pytest.mark.parametrize("which", ["cad", "sphere"])
def test_signed_distance_matches_reference(which, cad_part):
    mesh = cad_part if which == "cad" else trimesh.creation.icosphere(4, radius=10)
    rng = np.random.default_rng(1)
    lo, hi = mesh.bounds
    # random points plus points very close to the surface (where sign errors would show)
    pts = rng.uniform(lo - 3, hi + 3, size=(60_000, 3))
    surf, _ = trimesh.sample.sample_surface(mesh, 20_000, seed=2)
    pts = np.vstack([pts, surf + rng.normal(scale=0.02, size=surf.shape)])
    ours = MeshQuery(mesh.vertices, mesh.faces).signed(pts)
    ref = reference(mesh, pts)
    assert np.abs(np.abs(ours) - np.abs(ref)).max() < 1e-4
    significant = np.abs(ref) > 1e-6
    assert (np.sign(ours[significant]) != np.sign(ref[significant])).sum() == 0


def test_grid_sdf_all_paths_match_reference(cad_part):
    band = 1.5
    tiny = Grid.from_bounds(cad_part.bounds, 2.5, band)           # exact everywhere
    medium = Grid.from_bounds(cad_part.bounds, 0.5, band)         # coarse pass at 2x + band
    big = Grid.from_bounds(cad_part.bounds, 0.25, band)           # coarse pass at 4x + band
    assert tiny.size <= 20_000 < medium.size < 2_000_000 <= big.size
    rng = np.random.default_rng(3)
    for grid in (tiny, medium, big):
        sdf = mesh_to_sdf(cad_part, grid, band, key="test").ravel()
        pick = rng.choice(grid.size, size=min(grid.size, 100_000), replace=False)  # exact ref is slow
        ref = reference(cad_part, grid.points()[pick])
        near = np.abs(ref) < band
        assert np.abs(sdf[pick] - ref)[near].max() < 1e-4
        assert (np.sign(sdf[pick]) != np.sign(ref))[np.abs(ref) > 1e-6].sum() == 0


def _tessellate_the_old_way(body):
    """Lists of tuples, one OpenCascade call per node and per triangle."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    verts, faces, offset = [], [], 0
    exp = TopExp_Explorer(body, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        exp.Next()
        trsf = loc.Transformation()
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            verts.append((p.X(), p.Y(), p.Z()))
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            if face.Orientation() == TopAbs_REVERSED:
                b, c = c, b
            faces.append((a - 1 + offset, b - 1 + offset, c - 1 + offset))
        offset = len(verts)
    old = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
    old.merge_vertices()
    if old.volume < 0:
        old.invert()
    return old


def test_tessellation_is_unchanged_and_can_be_cancelled():
    # the faster import must give exactly the old mesh: cached distance fields are keyed by it
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    from engine.cad_import import load_step, tessellate
    body = load_step(Path(__file__).resolve().parents[1] / "samples" / "test_part.step")[0]
    mesh = tessellate(body, 0.01)
    old = _tessellate_the_old_way(body)
    assert np.array_equal(mesh.vertices, old.vertices) and np.array_equal(mesh.faces, old.faces)

    # a placed body (as in an assembly): its faces carry a rotation + translation
    rotate, shift = gp_Trsf(), gp_Trsf()
    rotate.SetRotation(gp_Ax1(gp_Pnt(1, 2, 3), gp_Dir(1, 1, 0.3)), 0.7)
    shift.SetTranslation(gp_Vec(10.5, -3.25, 7.125))
    for trsf in (shift, shift.Multiplied(rotate)):
        placed = body.Moved(TopLoc_Location(trsf))
        mesh, old = tessellate(placed, 0.01), _tessellate_the_old_way(placed)
        assert np.array_equal(mesh.vertices, old.vertices) and np.array_equal(mesh.faces, old.faces)

    def stop():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        tessellate(body, 0.01, check=stop)


def test_step_import_is_kept_on_the_ssd(tmp_path, monkeypatch):
    # opening the same file again reads the SSD copy (same meshes, OpenCascade not used); another
    # setting or a changed file imports afresh
    import os
    import shutil
    from engine import cad_import
    from engine.fields import mesh_fingerprint
    from engine.graph.nodes import import_step
    monkeypatch.setenv("DESIGN_ENGINE_CACHE", str(tmp_path / "cache"))
    step = tmp_path / "part.step"
    shutil.copy(ROOT / "samples" / "test_part.step", step)
    original, reads = cad_import.load_step, []
    monkeypatch.setattr(cad_import, "load_step", lambda p: reads.append(p) or original(p))

    first = import_step(str(step), 0.01, 1, 0.5)
    again = import_step(str(step), 0.01, 1, 0.5)
    assert len(reads) == 1
    assert [mesh_fingerprint(m) for m in again] == [mesh_fingerprint(m) for m in first]
    assert np.array_equal(again[0].face_attributes["cad_face"], first[0].face_attributes["cad_face"])
    assert again[0].faces.dtype == np.int64
    assert first[0].metadata["cad_body"] == again[0].metadata["cad_body"] == 0   # for "body:face" selections

    import_step(str(step), 0.02, 1, 0.5)
    st = os.stat(step)
    os.utime(step, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # saved again from the CAD program
    import_step(str(step), 0.01, 1, 0.5)
    assert len(reads) == 3
