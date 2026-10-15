"""GPU path (CuPy + NVIDIA Warp) must give the same results as the CPU path."""
import io
from pathlib import Path

import numpy as np
import pytest
import trimesh

from engine import backend
from engine import fields as F
from engine import load_step, tessellate
from engine.grid import Grid
from engine.noise import fractal_on_grid, wave3d_on_grid
from engine.sdf import MeshQuery

pytestmark = pytest.mark.skipif(not backend.gpu_available(), reason="no CUDA GPU")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gpu():
    backend.set_gpu(True)
    yield backend.cupy()
    backend.set_gpu(False)


@pytest.fixture(scope="module")
def cad_part():
    return tessellate(load_step(ROOT / "samples" / "test_part.step")[0], tolerance=0.01)


@pytest.mark.parametrize("voxel", [0.8, 0.25])   # one-pass and narrow-band paths
def test_gpu_distance_field_matches_cpu(gpu, cad_part, voxel):
    from engine.sdf_gpu import mesh_to_sdf_gpu
    band = 1.5
    grid = Grid.from_bounds(cad_part.bounds, voxel, band)
    sdf = backend.to_host(mesh_to_sdf_gpu(cad_part.vertices, cad_part.faces, grid, band, key="t"))
    exact = MeshQuery(cad_part.vertices, cad_part.faces).signed(grid.points()).reshape(grid.shape)
    near = np.abs(exact) < band
    assert np.abs(sdf - exact)[near].max() < 2e-3                      # float32 on the GPU
    assert (np.sign(sdf) != np.sign(exact))[np.abs(exact) > 1e-3].sum() == 0   # cracks included


def test_gpu_sign_is_right_on_a_hole_axis(gpu, cad_part):
    """A point on the axis of the 8 mm hole is equally close to the whole cylinder wall; a
    single-face normal test gets the sign wrong there, the pseudonormal test must not."""
    from engine.sdf_gpu import mesh_to_sdf_gpu
    grid = Grid((30, 20, 1), (30, 20, 14), 0.5)            # a line of points on the axis
    sdf = backend.to_host(mesh_to_sdf_gpu(cad_part.vertices, cad_part.faces, grid, 1.5, key="axis"))
    assert np.all(sdf > 7.9)                                # outside the part, ~8 mm from the wall


def test_gpu_noise_and_waves_match_cpu(gpu):
    grid = Grid((0, 0, 0), (20, 15, 10), 0.1)
    cpu = fractal_on_grid(grid, 12, 3, 0.5, 4)
    assert np.allclose(backend.to_host(fractal_on_grid(grid, 12, 3, 0.5, 4, xp=gpu)), cpu, atol=1e-5)
    cpu = wave3d_on_grid(grid, "ridges", 3.0, warp=1.0, seed=2)
    assert np.allclose(backend.to_host(wave3d_on_grid(grid, "ridges", 3.0, warp=1.0, seed=2, xp=gpu)), cpu, atol=1e-4)


def lamp_like_body(cad_part):
    body = F.body_from_mesh(cad_part, 1.5)
    wave = F.remap(F.acos(F.cos(F.noise(30, 3, 0.45, 3) * 60)), 0, np.pi, 0, 1)
    textured = F.smoothen(F.offset(body, wave * 0.5 * F.smoothstep(F.Z, 2, 4)), 3, 0.1)
    return F.subtract(textured, F.offset(F.body_from_mesh(trimesh.creation.box((20, 20, 4)), 1.0), 0.1))


def test_gpu_field_evaluation_matches_cpu(gpu, cad_part):
    grid = Grid.from_bounds(cad_part.bounds, 0.3, 1.5)
    body = lamp_like_body(cad_part)
    cpu = F.Context(grid).eval_full(body)
    out = backend.to_host(F.Context(grid, xp=gpu).eval_full(body))
    near = np.abs(cpu) < 1.0  # what shapes the part; farther out both interpolate
    assert np.abs(out - cpu)[near].max() < 3e-3
    assert (np.sign(out) != np.sign(cpu))[np.abs(cpu) > 3e-3].sum() == 0


def test_gpu_out_of_memory_falls_back_to_cpu(gpu, cad_part, monkeypatch):
    import engine.graph.nodes as nodes
    real = nodes._mesh_on
    used = []

    def flaky(body, grid, xp):
        used.append(xp.__name__)
        if xp is not np:
            raise gpu.cuda.memory.OutOfMemoryError(1 << 30, 0)   # as if another program took it
        return real(body, grid, xp)
    monkeypatch.setattr(nodes, "_mesh_on", flaky)
    mesh = nodes._mesh(F.body_from_mesh(cad_part, 1.5), cad_part.bounds, 0.3, 1.5, 1.0)  # 2M voxels: GPU
    assert used == ["cupy", "numpy"] and mesh.is_watertight


def test_small_grids_stay_on_the_cpu(gpu):
    from engine.graph.nodes import array_module
    assert array_module(backend.GPU_MIN_VOXELS - 1) is np      # no CUDA context for Draft-size work
    assert array_module(backend.GPU_MIN_VOXELS) is gpu


def test_gpu_mesh_node_is_watertight_after_stl_reload(gpu, cad_part):
    from engine.graph.nodes import _mesh, array_module
    body = lamp_like_body(cad_part)
    assert array_module(1_000_000) is gpu
    mesh = _mesh(body, cad_part.bounds, 0.3, 1.5, 1.0)
    backend.set_gpu(False)
    reference = _mesh(body, cad_part.bounds, 0.3, 1.5, 1.0)
    reloaded = trimesh.load(io.BytesIO(trimesh.exchange.stl.export_stl(mesh)), file_type="stl")
    assert reloaded.is_watertight
    assert abs(mesh.volume - reference.volume) / reference.volume < 1e-3


def test_gpu_marching_cubes_matches_cpu(gpu, cad_part):
    from engine.meshing import field_to_mesh
    body = lamp_like_body(cad_part)
    grid = Grid.from_bounds(cad_part.bounds, 0.2, 1.5)
    values = F.Context(grid).eval_full(body)
    cpu = field_to_mesh(values, grid)
    on_gpu = field_to_mesh(gpu.asarray(values), grid)     # Warp marching cubes
    assert len(on_gpu.vertices) == len(cpu.vertices) and len(on_gpu.faces) == len(cpu.faces)
    assert on_gpu.is_watertight and on_gpu.is_winding_consistent
    assert abs(on_gpu.volume - cpu.volume) < 1e-5 * cpu.volume
    assert np.abs(np.sort(on_gpu.vertices, axis=0) - np.sort(cpu.vertices, axis=0)).max() < 1e-4


def test_gpu_blocks_stitch_into_a_closed_mesh(gpu, cad_part, monkeypatch, tmp_path):
    """Big bodies are meshed in blocks on the GPU; GPU values must agree between blocks (shared
    samples computed from the same global index, also inside the Warp kernel)."""
    from engine import tiles
    from engine.meshing import field_to_mesh
    monkeypatch.setattr(tiles, "BLOCK", 48)
    body = F.body_from_mesh(cad_part, 1.5)
    field = F.smoothen(F.offset(body, F.acos(F.cos(F.noise(40, 1, 0.5, 3) * 60)) * (0.3 / np.pi)), 3, 0.2)
    grid = Grid.from_bounds(field.bounds, 0.25, 1.5)
    cpu = field_to_mesh(F.Context(grid).eval_full(field), grid)
    mesh, computed, _ = tiles.mesh_tiled(field, field.bounds, 0.25, 1.5, 1.0,
                                         lambda fn, args, **_: [fn(**a) for a in args])
    assert computed > 1
    assert mesh.volume == pytest.approx(cpu.volume, rel=1e-4)
    mesh.export(tmp_path / "gpu_blocks.stl")
    reloaded = trimesh.load(tmp_path / "gpu_blocks.stl")
    assert reloaded.is_watertight and len(reloaded.split()) == 1


def test_gpu_face_mask_matches_cpu(gpu, cad_part):
    from engine import facemask
    ids = facemask.cad_faces(cad_part)
    faces = list(np.unique(ids)[:6])
    field = facemask.face_mask(cad_part, ids, faces, 1.5, "center")
    grid = Grid.from_bounds(cad_part.bounds, 0.3, 2.0)
    cpu = F.Context(grid).eval_full(field)
    on_gpu = backend.to_host(F.Context(grid, xp=gpu).eval_full(field))
    assert cpu.max() == 1.0 and (cpu > 0).sum() > 1000
    # float32 closest points on the GPU: values agree except a few right at face borders
    assert np.mean(np.abs(on_gpu - cpu) < 1e-3) > 0.999


@pytest.mark.parametrize("make", ["tpms", "struts"])
def test_gpu_lattices_match_cpu(gpu, make):
    from engine import lattice
    field = lattice.tpms("gyroid", 5.0, 0.6) if make == "tpms" else lattice.struts("octet", 5.0, 0.8, blend=0.5)
    grid = Grid((0.3, -1.1, 2.0), (14.0, 12.0, 16.0), 0.1)   # 2.7M voxels: big enough for the GPU path
    cpu = F.Context(grid).eval_full(field)
    on_gpu = backend.to_host(F.Context(grid, xp=gpu).eval_full(field))
    # float32 trig differs slightly between GPU and CPU; far from the walls (where TPMS distances
    # are only an estimate anyway) that grows with the value, near them it is ~1e-6 mm
    assert (np.abs(on_gpu - cpu) < 1e-4 * np.maximum(1.0, np.abs(cpu))).all()


def test_gpu_voronoi_matches_cpu(gpu):
    from engine import lattice
    field = lattice.voronoi(np.array([[0.0, 0, 0], [30, 25, 20]]), 6.0, 1.2, randomness=1.0, seed=3)
    grid = Grid((-1.0, -1.0, -1.0), (31.0, 26.0, 21.0), 0.2)   # 2.2M voxels: the GPU path
    cpu = F.Context(grid).eval_full(field)
    on_gpu = backend.to_host(F.Context(grid, xp=gpu).eval_full(field))
    assert np.abs(on_gpu - cpu).max() < 1e-4                         # exact on both, float32 on the GPU


def test_gpu_wall_thickness_matches_cpu(gpu, cad_part):
    from engine import checks
    on_gpu = checks.wall_thickness(cad_part.vertices, cad_part.faces)
    backend.set_gpu(False)
    cpu = checks.wall_thickness(cad_part.vertices, cad_part.faces)
    both = np.isfinite(on_gpu) & np.isfinite(cpu)
    assert both.mean() > 0.99 and np.abs(on_gpu - cpu)[both].max() < 1e-3   # the same kernel on both devices


def test_gpu_static_analysis_matches_cpu(gpu):
    # float32 multigrid, float64 conjugate gradients: all-float32 CG bent this bracket 3.60 mm
    # instead of 4.21 while reporting convergence (see engine/fea/solver.py, MIXED)
    from engine.fea import analysis
    from engine.fea.analysis import static_analysis
    from engine.fea.materials import PRESETS
    from engine.fea.model import Condition, Conditions
    mesh = tessellate(load_step(ROOT / "samples" / "bracket.step")[0], tolerance=0.01)
    ids = mesh.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", mesh.triangles[np.isin(ids, [8, 9])]),
                        Condition("load", mesh.triangles[ids == 17], amount=50.0, direction="+X")))
    part = F.body_from_mesh(mesh, 2.0)
    cpu = static_analysis(part, conds, PRESETS["PETG"], "draft", xp=np)
    analysis._RESULTS.clear()  # else the same inputs give back the CPU result
    on_gpu = static_analysis(part, conds, PRESETS["PETG"], "draft", xp=gpu)
    assert on_gpu.report["device"] == "GPU"
    for k in ("max_displacement", "max_stress"):
        assert on_gpu.report[k] == pytest.approx(cpu.report[k], rel=1e-4), k
    assert on_gpu.report["imbalance"] < 1e-5


@pytest.mark.parametrize("kind, joints", [("round", "fillet"), ("square tube", "fillet"), ("I", "ball"),
                                          ("round tube", "plain")])
def test_gpu_frame_bars_match_cpu(gpu, kind, joints):
    # the frame kernel (profiles, mitres, fillets between different bars) on the GPU and on Warp's CPU
    from engine.frame import curves as CV
    from engine.frame.geometry import frame_solid
    from engine.frame.profiles import Profile
    nodes = np.array([(0, 0, 0), (30, 0, 5), (45, 20, 10), (5, 15, 20)], float)
    bars = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    fixed = np.zeros((4, 3), dtype=bool)
    n2, b2, _, c2, _, _ = CV.smooth_bars(nodes, bars, np.ones(5), np.arange(5), fixed, np.zeros((4, 3)), 1.0)
    sizes = np.linspace(3.0, 6.0, int(c2.max()) + 1)
    body = frame_solid(n2, b2, c2, sizes, Profile(kind, ratio=1.5 if kind == "I" else 1.0), joints, f"gpu-{kind}")
    grid = Grid((-6, -6, -6), (52, 27, 27), 0.4)
    cpu = F.Context(grid, None, np).eval_full(body)
    on_gpu = backend.to_host(F.Context(grid, None, gpu).eval_full(body))
    assert np.abs(on_gpu - cpu).max() < 1e-4                           # float32 both, same kernel
