from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from engine import fields as F
from engine import tiles
from engine.cad_import import load_step, tessellate
from engine.dualcontour import dual_contour
from engine.graph import Evaluator, Graph, link
from engine.grid import Grid
from engine.meshing import FIELD_NORMAL, NORMAL_SCALE, field_to_mesh

TEST_STEP = Path(__file__).resolve().parents[1] / "samples" / "test_part.step"
VOXEL = 0.25


@pytest.fixture(scope="module")
def part():
    return tessellate(load_step(TEST_STEP)[0])


@pytest.fixture
def small_blocks(monkeypatch):
    # 48-voxel blocks and tiling for any size, so a small part spans several blocks
    monkeypatch.setattr(tiles, "BLOCK", 48)
    monkeypatch.setattr(tiles, "TILE_MIN_VOXELS", 0)


def textured(part, smooth=False):
    """Like the lamp: a dense wave on noise (noise sampled 3x coarser and interpolated)."""
    body = F.body_from_mesh(part, 1.5)
    wave = F.acos(F.cos(F.noise(40, 1, 0.5, 3) * 60)) * (0.3 / np.pi)
    out = F.offset(body, wave)
    return F.smoothen(out, 3, 0.2) if smooth else out


def sequential(fn, arg_list, need=0, gpu=False, gpu_need=0, cpu_need=None):
    return [fn(**a) for a in arg_list]


def whole_grid_mesh(field):
    grid = Grid.from_bounds(field.bounds, VOXEL, 1.5)
    return field_to_mesh(F.Context(grid).eval_full(field), grid)


def reload(mesh, tmp_path):
    mesh.export(tmp_path / "out.stl")
    return trimesh.load(tmp_path / "out.stl")


def test_block_coordinates_match_the_whole_grid():
    whole = Grid((-3.3, 1.7, 0.1), (20, 30, 12), 0.1)
    block = Grid.block(whole.base, 0.1, (24, 36, 12), (13, 13, 13))
    for a, b, s in zip(whole.axes(), block.axes(), block.start):
        assert np.array_equal(a[s:s + 13], b)
    assert np.array_equal(whole.coarser(4).axes()[0][6:9], block.coarser(4).axes()[0][:3])
    with pytest.raises(ValueError):
        Grid.block(whole.base, 0.1, (5, 0, 0), (13, 13, 13)).coarser(4)


def test_blocks_compute_the_same_values_as_the_whole_grid(part):
    field = textured(part)
    whole = Grid.from_bounds(field.bounds, VOXEL, 1.5)
    ref = F.Context(whole).eval_full(field)[48:97, 48:97, 24:73]
    block = F.Context(Grid.block(whole.base, VOXEL, (48, 48, 24), (49, 49, 49))).eval_full(field)
    near = np.abs(ref) < 1.0  # exact distances; farther out both are interpolated (differently)
    assert near.sum() > 1000
    assert np.array_equal(block[near], ref[near])


def test_tiled_mesh_matches_the_whole_grid(part, small_blocks, tmp_path):
    field = textured(part)
    whole = whole_grid_mesh(field)
    mesh, computed, total = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential)
    assert computed > 1
    assert len(mesh.faces) == len(whole.faces)
    assert mesh.volume == pytest.approx(whole.volume, rel=1e-6)
    assert reload(mesh, tmp_path).is_watertight


def unit_normals(mesh):
    return mesh.vertex_attributes[FIELD_NORMAL] / NORMAL_SCALE


def facet_normals(mesh):
    """Area-weighted vertex normals of the triangles (as the editor's computeVertexNormals), unit length."""
    V, T = mesh.vertices, mesh.faces
    cross = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, T[:, k], cross)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


@pytest.mark.parametrize("smooth", [False, True])   # (smoothing: blocks with a field halo)
def test_block_normals_match_the_whole_grid(part, small_blocks, smooth):
    # views carry the field's normals, central differences at the samples of each vertex's edge. They need a sample
    # past a block's face, so each block's cells move one up (blockmesh.normals_block): a seam's vertex gets the same
    # normal in both blocks, the whole grid's, and the triangles are the same as without normals
    field = textured(part, smooth)
    grid = Grid.from_bounds(field.bounds, VOXEL, 1.5)
    whole = field_to_mesh(F.Context(grid).eval_full(field), grid, normals=True)
    mesh, computed, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential, normals=True)
    plain, _, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential)
    assert computed > 1 and len(mesh.faces) == len(whole.faces) == len(plain.faces)
    assert FIELD_NORMAL not in plain.vertex_attributes and same_vertices(mesh, plain)
    dist, nearest = cKDTree(whole.vertices).query(mesh.vertices)
    assert dist.max() < 1e-4 and len(np.unique(nearest)) == len(mesh.vertices) == len(whole.vertices)
    ours, theirs = (m.vertex_attributes[FIELD_NORMAL].astype(np.int32) for m in (mesh, whole))
    assert np.abs(ours - theirs[nearest]).max() <= 2   # (int16 steps of 3e-5: the positions' last float32 bits)
    # unit length, and out of the body (the triangles' own normals agree, but for a few in the texture's creases)
    n = unit_normals(mesh)
    assert np.abs(np.linalg.norm(n, axis=1) - 1).max() < 1e-4
    agree = (n * facet_normals(mesh)).sum(axis=1)
    assert (agree > 0).mean() > 0.999 and np.median(agree) > 0.99
    # Keep or Tolerance would move the vertices: no normals then
    reduced, _, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 0.5, sequential, normals=True)
    assert FIELD_NORMAL not in reduced.vertex_attributes


def test_normals_of_a_sphere_are_true_and_point_out():
    grid = Grid((-6, -6, -6), (6, 6, 6), 0.25)
    X, Y, Z = grid.coords()
    mesh = field_to_mesh(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - np.float32(4.3), grid, normals=True)
    radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    angle = lambda n: np.degrees(np.arctan2(np.linalg.norm(np.cross(n, radial), axis=1), (n * radial).sum(axis=1)))
    # (0.15° at most, 0.02° on average: samples within 1% of a voxel of zero are moved off it, see extract_surface)
    assert angle(unit_normals(mesh)).max() < 0.25 < 1.0 < angle(facet_normals(mesh)).max()   # (the triangles' 2.7°)
    # a hollow (the body is cut flat by the box, closed by its pad): out of the body is towards the middle
    hollow = field_to_mesh(np.float32(4.3) - np.sqrt(X ** 2 + Y ** 2 + Z ** 2), grid, normals=True)
    n, r = unit_normals(hollow), np.linalg.norm(hollow.vertices, axis=1)
    radial = hollow.vertices / r[:, None]
    assert angle(-n)[np.abs(r - 4.3) < 0.1].max() < 0.25
    assert (n * facet_normals(hollow)).sum(axis=1).min() > 0.5   # (on the box's faces too: the triangles agree)


def test_empty_space_is_skipped(small_blocks, tmp_path):
    # a thin-walled ring like the lamp center: the hole and the corners of its box are empty
    ring = trimesh.creation.annulus(30, 32, 20)
    field = F.offset(F.body_from_mesh(ring, 1.5), F.noise(10, 2, 0.5, 1) * 0.3)
    whole = whole_grid_mesh(field)
    mesh, computed, total = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential)
    assert computed <= total - 8  # at least the middle of the ring (2 x 2 blocks on 2 layers)
    assert len(mesh.faces) == len(whole.faces)
    assert reload(mesh, tmp_path).is_watertight


def test_smoothing_is_seamless(part, small_blocks, tmp_path):
    field = textured(part, smooth=True)
    whole = whole_grid_mesh(field)
    mesh, _, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential)
    assert len(mesh.faces) == len(whole.faces)
    assert mesh.volume == pytest.approx(whole.volume, rel=1e-6)
    assert reload(mesh, tmp_path).is_watertight


def test_simplified_blocks_stay_closed(part, small_blocks, tmp_path):
    field = textured(part)
    whole = whole_grid_mesh(field)
    mesh, _, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 0.3, sequential)
    assert len(mesh.faces) < 0.6 * len(whole.faces)
    assert mesh.volume == pytest.approx(whole.volume, rel=2e-3)
    reloaded = reload(mesh, tmp_path)
    assert reloaded.is_watertight and len(reloaded.split()) == 1


def test_blocks_the_probe_missed_are_added(part, small_blocks, monkeypatch):
    field = textured(part)
    full, computed, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential)
    found = tiles.find_surface
    monkeypatch.setattr(tiles, "find_surface", lambda f, plan: [b for b in found(f, plan)
                                                                if tiles._mesh_block(f, plan, b, 1.0, np)[1] is not None][:1])
    mesh, _, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential)
    assert len(mesh.faces) == len(full.faces)


DUAL_VOXEL = 0.4  # (dual contouring in blocks of 48 with their halo: ~4x the samples)


def whole_box_dual(field):
    """Dual contouring on one grid over the box the blocks cover."""
    plan = tiles.Plan(field, field.bounds, DUAL_VOXEL, 1.5)
    grid = Grid.block(plan.base, DUAL_VOXEL, (0, 0, 0), [n * plan.block + 1 for n in plan.blocks])
    return dual_contour(F.Context(grid).eval_full(field), grid)


def same_vertices(a, b, within=1e-5):  # (blocks are snapped to float32)
    return all(cKDTree(q.vertices).query(p.vertices)[0].max() < within for p, q in ((a, b), (b, a)))


@pytest.mark.parametrize("make", [lambda part: F.body_from_mesh(part, 1.5), textured,
                                  lambda part: F.subtract(F.body_from_mesh(part, 1.5),
                                                          F.bar((0, -30, 7.5), (0, 30, 7.5), 3.0))])
def test_dual_contouring_in_blocks_matches_the_whole_grid(part, small_blocks, make, tmp_path):
    # blocks keep the vertices of their own cells and the quads of their own edges, computed with a halo
    # that holds everything they depend on, and join by vertex names: the whole grid's mesh
    field = make(part)
    whole = whole_box_dual(field)
    mesh, computed, _ = tiles.mesh_tiled(field, field.bounds, DUAL_VOXEL, 1.5, 1.0, sequential,
                                         method="dual contouring")
    assert computed > 1
    assert len(mesh.faces) == len(whole.faces) and same_vertices(mesh, whole)
    assert mesh.volume == pytest.approx(whole.volume, rel=1e-6)
    assert reload(mesh, tmp_path).is_watertight


def test_dual_contouring_blocks_the_probe_missed_are_added(part, small_blocks, monkeypatch):
    field = F.body_from_mesh(part, 1.5)
    full, _, _ = tiles.mesh_tiled(field, field.bounds, DUAL_VOXEL, 1.5, 1.0, sequential, method="dual contouring")
    found = tiles.find_surface
    monkeypatch.setattr(tiles, "find_surface", lambda f, plan: [b for b in found(f, plan)
                                                                if tiles._mesh_block(f, plan, b, 1.0, np)[1] is not None][:1])
    mesh, _, _ = tiles.mesh_tiled(field, field.bounds, DUAL_VOXEL, 1.5, 1.0, sequential, method="dual contouring")
    assert len(mesh.faces) == len(full.faces) and same_vertices(mesh, full)


def test_dual_contouring_blocks_close_a_body_cut_by_the_box(part, small_blocks, tmp_path):
    field = F.offset(F.body_from_mesh(part, 1.5), F.X * 0 + 2.0)  # reaches past the box, unknown to the mesher
    mesh, _, _ = tiles.mesh_tiled(field, part.bounds, DUAL_VOXEL, 1.0, 1.0, sequential, method="dual contouring")
    assert mesh.metadata["clipped"] and reload(mesh, tmp_path).is_watertight
    lighter, _, _ = tiles.mesh_tiled(field, part.bounds, DUAL_VOXEL, 1.0, 0.3, sequential, method="dual contouring")
    assert len(lighter.faces) < 0.4 * len(mesh.faces) and reload(lighter, tmp_path).is_watertight


def test_big_body_is_meshed_in_blocks_by_the_workers(small_blocks, tmp_path):
    g = Graph()
    g.add("import", "input.import_step", path=str(TEST_STEP))
    g.add("part", "input.pick_body", bodies=link("import.bodies"))
    g.add("sdf", "implicit.from_cad", cad=link("part.body"), band=1.5)
    g.add("noise", "field.noise", scale=40, octaves=1, seed=3)
    g.add("wave", "math.multiply", a=link("noise.out"), b=0.3)
    g.add("offset", "implicit.offset", body=link("sdf.out"), distance=link("wave.out"))
    g.add("mesh", "mesh.from_implicit", body=link("offset.out"), voxel=VOXEL, padding=1.5, keep=0.5)
    g.add("export", "output.export", mesh=link("mesh.mesh"), path=str(tmp_path / "big.stl"))
    events = []
    ev = Evaluator(g, workers=2, log=None, on_event=events.append)
    try:
        ev.run()
    finally:
        ev.close()
    assert any(e["type"] == "message" and "blocks" in e["text"] for e in events)
    assert any(e["type"] == "progress" and e["node"] == "mesh" for e in events)
    reloaded = trimesh.load(tmp_path / "big.stl")
    assert reloaded.is_watertight and len(reloaded.split()) == 1


@pytest.mark.parametrize("settings", [dict(keep=0.5), dict(tolerance=0.01), dict(method="dual contouring", voxel=0.4)])
def test_two_stage_blocks_give_the_same_mesh(small_blocks, settings, monkeypatch):
    # blocks whose meshing reduces (or dual contours) are meshed by CPU workers from the samples the GPU
    # workers computed (engine/stages.py; forced on here on the CPU): the same mesh bit for bit
    from engine.graph import core
    monkeypatch.setattr(core, "CPU_WORKERS_MAX", 2)
    meshes = {}
    for split in (False, True):
        g = Graph()
        g.add("import", "input.import_step", path=str(TEST_STEP))
        g.add("part", "input.pick_body", bodies=link("import.bodies"))
        g.add("sdf", "implicit.from_cad", cad=link("part.body"), band=1.5)
        g.add("noise", "field.noise", scale=40, octaves=1, seed=3)
        g.add("wave", "math.multiply", a=link("noise.out"), b=0.3)
        g.add("offset", "implicit.offset", body=link("sdf.out"), distance=link("wave.out"))
        g.add("mesh", "mesh.from_implicit", body=link("offset.out"), **{"voxel": VOXEL, "padding": 1.5, **settings})
        ev = Evaluator(g, workers=2, log=None)
        ev.split = split
        try:
            meshes[split] = ev.run()["mesh"]["mesh"]
            assert (ev._pool.cpu_started() > 0) == split
        finally:
            ev.close()
    assert np.array_equal(meshes[False].vertices, meshes[True].vertices)
    assert np.array_equal(meshes[False].faces, meshes[True].faces)


def vented(part):
    """The test part with hexagonal holes through its top face (a cutter computed hole by hole)."""
    from engine import holes as H
    from engine import pattern2d as P2
    ids = part.face_attributes["cad_face"]
    top = int(np.bincount(ids[part.face_normals[:, 2] > 0.999]).argmax())
    holes = H.place(part, [top], P2.Shape("hexagon", 3.0, corner=0.3, rotation=10), "staggered", 5.0, angle=7,
                    margin=1.0)
    assert len(holes) > 30
    return H.cutter(holes), F.subtract(F.body_from_mesh(part, 1.5), H.cutter(holes))


def test_hole_cutter_is_the_same_in_blocks(part, small_blocks, tmp_path):
    cutter, field = vented(part)
    whole = Grid.from_bounds(field.bounds, VOXEL, 1.5)
    ref = F.Context(whole).eval_full(cutter)[48:97, 48:97, 24:73]
    block = F.Context(Grid.block(whole.base, VOXEL, (48, 48, 24), (49, 49, 49))).eval_full(cutter)
    assert (ref < 0).sum() > 1000 and np.array_equal(block, ref)
    full = whole_grid_mesh(field)
    mesh, computed, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential)
    assert computed > 1 and len(mesh.faces) == len(full.faces)
    assert mesh.volume == pytest.approx(full.volume, rel=1e-6)
    assert reload(mesh, tmp_path).is_watertight


def test_voronoi_cutter_is_the_same_in_blocks(part):
    from engine import holes as H
    from engine import pattern2d as P2
    ids = part.face_attributes["cad_face"]
    top = int(np.bincount(ids[part.face_normals[:, 2] > 0.999]).argmax())
    cells = H.place(part, [top], P2.Shape("voronoi", 0.8, corner=0.3), "staggered", 4.0, margin=1.0, randomness=0.8)
    cutter = H.cutter(cells)
    whole = Grid.from_bounds(part.bounds, VOXEL, 1.5)
    ref = F.Context(whole).eval_full(cutter)[48:97, 48:97, 24:73]
    block = F.Context(Grid.block(whole.base, VOXEL, (48, 48, 24), (49, 49, 49))).eval_full(cutter)
    assert (ref < 0).sum() > 1000 and np.array_equal(block, ref)


def test_morph_cutter_is_the_same_in_blocks(part):
    from engine import holes as H
    from engine import pattern2d as P2
    ids = part.face_attributes["cad_face"]
    top = int(np.bincount(ids[part.face_normals[:, 2] > 0.999]).argmax())
    holes = H.place(part, [top], P2.Shape("round", 2.5), "staggered", 4.0, margin=1.0,
                    morph_to=P2.Shape("voronoi", 0.8, corner=0.3), morph=F.clamp(F.remap(F.X, 10, 50, 0, 1), 0, 1),
                    randomness=F.clamp(F.remap(F.Y, 5, 35, 0, 1), 0, 1))
    cutter = H.cutter(holes)
    whole = Grid.from_bounds(part.bounds, VOXEL, 1.5)
    ref = F.Context(whole).eval_full(cutter)[48:97, 48:97, 24:73]
    block = F.Context(Grid.block(whole.base, VOXEL, (48, 48, 24), (49, 49, 49))).eval_full(cutter)
    assert (ref < 0).sum() > 1000 and np.array_equal(block, ref)


def test_new_patterns_are_the_same_in_blocks(part):
    # tilted, alternating stars cut with rounded edges, and a picture: all pure functions of position
    from engine import holes as H
    from engine import pattern2d as P2
    from engine import picture as PIC
    ids = part.face_attributes["cad_face"]
    top = int(np.bincount(ids[part.face_normals[:, 2] > 0.999]).argmax())
    stars = H.place(part, [top], P2.Shape("star", 2.5, 5, sides=5, corner=0.2), "grid", 5.0, margin=1.0,
                    alternate="rows 180", tilt=20.0)
    body = F.subtract_edged(F.body_from_mesh(part, 1.5), H.cutter(stars).regrow(0.4), "round", 0.4)
    logo = PIC.load_image(Path(__file__).resolve().parents[1] / "samples" / "logo.png")
    picture = PIC.picture_field(PIC.Placement(part, [top]), logo, 30.0)
    whole = Grid.from_bounds(part.bounds, VOXEL, 1.5)
    for field, part_distance in ((body, True), (picture, False)):
        ref = F.Context(whole).eval_full(field)[48:97, 48:97, 24:73]
        block = F.Context(Grid.block(whole.base, VOXEL, (48, 48, 24), (49, 49, 49))).eval_full(field)
        near = np.abs(ref) < 1.0 if part_distance else np.ones(ref.shape, dtype=bool)  # (see above)
        assert near.sum() > 1000 and np.array_equal(block[near], ref[near])


def test_surface_pattern_is_the_same_in_blocks(part):
    from engine import holes as H
    from engine import pattern2d as P2
    ids = part.face_attributes["cad_face"]
    top = int(np.bincount(ids[part.face_normals[:, 2] > 0.999]).argmax())
    shapes = H.place(part, [top], P2.Shape("diamond", 2.5, 4, corner=0.3), "staggered", 4.0, angle=12, margin=1.0,
                     surface=True)
    relief = H.relief(shapes, 0.6, 1.5)
    whole = Grid.from_bounds(part.bounds, VOXEL, 1.5)
    ref = F.Context(whole).eval_full(relief)[48:97, 48:97, 24:73]
    block = F.Context(Grid.block(whole.base, VOXEL, (48, 48, 24), (49, 49, 49))).eval_full(relief)
    assert ((ref > 0) & (ref < 1)).sum() > 1000 and np.array_equal(block, ref)


@pytest.mark.parametrize("make", [lambda part: F.body_from_mesh(part, 1.5), textured])
def test_blocks_reduced_to_a_tolerance_stay_closed(part, small_blocks, tmp_path, make):
    # the reduction runs in each block (borders kept for the neighbours), the seams after stitching
    from engine.meshing import reduce
    field = make(part)
    whole = whole_grid_mesh(field)
    mesh, _, _ = tiles.mesh_tiled(field, field.bounds, VOXEL, 1.5, 1.0, sequential, tolerance=0.01)
    reloaded = reload(mesh, tmp_path)
    assert reloaded.is_watertight and len(reloaded.split()) == 1
    assert mesh.volume == pytest.approx(whole.volume, rel=1e-3)
    V, Fc = reduce(whole.vertices, whole.faces, 0.01)                  # the whole mesh in one go
    assert len(mesh.faces) < 1.3 * len(Fc) + 2000                      # seams cost little extra
    assert len(mesh.faces) < 0.8 * len(whole.faces)
