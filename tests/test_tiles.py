from pathlib import Path

import numpy as np
import pytest
import trimesh

from engine import fields as F
from engine import tiles
from engine.cad_import import load_step, tessellate
from engine.graph import Evaluator, Graph, link
from engine.grid import Grid
from engine.meshing import field_to_mesh

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


def sequential(fn, arg_list, need=0, gpu=False, gpu_need=0):
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
