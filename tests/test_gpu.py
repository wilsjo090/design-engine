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

    def flaky(body, grid, xp, cut_hint, method="marching cubes"):
        used.append(xp.__name__)
        if xp is not np:
            raise gpu.cuda.memory.OutOfMemoryError(1 << 30, 0)   # as if another program took it
        return real(body, grid, xp, cut_hint, method)
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


def test_gpu_topology_surface_matches_cpu(gpu, monkeypatch):
    # density_mesh's distance transforms, filters, labels and marching cubes on the GPU (a fine result's two
    # smoothings took ~3 minutes on the CPU): thickened, a tip cut, a speck dropped, cropped, as on the CPU
    from scipy.spatial import cKDTree
    from engine.fea import topology as T
    d = np.zeros((44, 30, 30), dtype=np.float32)
    d[4:40, 12:18, 12:18] = 1.0          # a strut ...
    d[20:24, 4:26, 14:16] = 1.0          # ... a thin plate across it (thickened)
    d[30:32, 15:16, 18:27] = 1.0         # a needle off it (a tip, cut)
    d[2:4, 2:4, 2:4] = 1.0               # a speck on its own (dropped)
    phi = np.ones_like(d)
    crop = F.X - 41.0                    # the design space ends before the strut does
    meshes = {}
    for name, xp in (("cpu", np), ("gpu", gpu)):
        monkeypatch.setattr(T, "_smoothing_module", lambda samples, xp=xp: xp)
        meshes[name] = T.density_mesh(d, phi, np.zeros(3), 1.0, blur=2.0, space=np.ones(d.shape, bool),
                                      min_thickness=3.0, crop=crop)
    cpu, on_gpu = meshes["cpu"], meshes["gpu"]
    assert cpu.metadata["thickened"] > 0
    for key in ("tips", "unreached"):
        assert on_gpu.metadata[key] == cpu.metadata[key], key
    assert on_gpu.metadata["thickened"] == pytest.approx(cpu.metadata["thickened"], rel=1e-3)
    assert on_gpu.volume == pytest.approx(cpu.volume, rel=1e-4)
    apart = max(cKDTree(cpu.vertices).query(on_gpu.vertices)[0].max(), cKDTree(on_gpu.vertices).query(cpu.vertices)[0].max())
    assert apart < 1e-3, apart     # (mm; samples 0.5 mm apart)
    # thin tips (thickening grows the needle above, so here directly): a needle off a block goes, a strut
    # between two blocks and a plate stay (tests/test_topology.py), on the GPU as on the CPU
    from scipy import ndimage
    v = 0.25
    solid = np.zeros((120, 110, 40), dtype=bool)
    solid[10:50, 10:70, 10:30] = solid[80:110, 10:40, 10:30] = True
    solid[50:80, 20:24, 18:22] = solid[50:90, 56:60, 18:22] = solid[10:50, 70:105, 18:22] = True
    values = ((ndimage.distance_transform_edt(solid) - ndimage.distance_transform_edt(~solid)) * v).astype(np.float32)
    inside = np.ones(solid.shape, dtype=bool)
    out, tips = T.cut_tips(values, 0.0, inside, v, 2.0)
    out_gpu, tips_gpu = T.cut_tips(gpu.asarray(values), 0.0, gpu.asarray(inside), v, 2.0)
    assert tips == tips_gpu == 1
    assert np.allclose(backend.to_host(out_gpu), out, atol=1e-5)


@pytest.mark.parametrize("radius", [3, 10])   # direct and by FFT (topology's big struts-only filters)
def test_gpu_topology_filter_matches_cpu(gpu, radius):
    from engine.fea import topology as T
    rng = np.random.default_rng(2)
    a = rng.random((60, 24, 40)).astype(np.float32)
    o = np.arange(-radius, radius + 1)
    kernel = np.maximum(radius + 0.25 - np.sqrt(o[:, None, None] ** 2 + o[None, :, None] ** 2 + o[None, None, :] ** 2),
                        0).astype(np.float32)
    cpu = T._convolve(np, 0)(a.astype(np.float64), kernel.astype(np.float64))
    on_gpu = backend.to_host(T._convolve(gpu, kernel.size)(gpu.asarray(a), gpu.asarray(kernel)))
    assert np.allclose(on_gpu, cpu, atol=1e-5 * np.abs(cpu).max())


def test_gpu_topology_outline_and_symmetry_match_cpu(gpu):
    # Smoothness's outline and its gradient, and Symmetry's mirror average, in float32 on the GPU
    from engine.fea import topology as T
    rng = np.random.default_rng(3)
    shape = (30, 20, 16)
    x = rng.random(shape)
    domain = np.ones(shape)
    domain[:, :, 12:] = 0.0        # a design space face inside the grid
    free = rng.random(shape) > 0.1
    solid = np.zeros(shape, dtype=bool)
    solid[:2] = True
    o = np.arange(-2, 3)
    kernel = np.maximum(2.5 - np.sqrt(o[:, None, None] ** 2 + o[None, :, None] ** 2 + o[None, None, :] ** 2), 0)
    found = []
    for xp, dtype in ((np, np.float64), (gpu, np.float32)):
        D, K = xp.asarray(domain, dtype=dtype), xp.asarray(kernel, dtype=dtype)
        N = T._convolve(xp, K.size)(D, K)
        norm = xp.where(N > 0, 1.0 / xp.where(N > 0, N, 1.0), 0.0).astype(dtype)
        p = T.Problem(None, D, xp.asarray(solid), xp.asarray(free & ~solid), None, None, K, norm, D, 1.0, 0.3, 2.5,
                      None, xp, dtype, mirror=((0, 30), (1, 19)))
        opt = object.__new__(T.Optimizer)   # (no solve: only the outline)
        opt.p, opt.beta = p, 4.0
        area, grad = opt.outline(xp.asarray(x, dtype=dtype))
        found.append((area, backend.to_host(grad), backend.to_host(p.symmetric(xp.asarray(x, dtype=dtype)))))
    (area, grad, mirrored), (area_gpu, grad_gpu, mirrored_gpu) = found
    assert area_gpu == pytest.approx(area, rel=1e-4)
    assert np.allclose(grad_gpu, grad, atol=1e-4 * np.abs(grad).max())
    assert np.allclose(mirrored_gpu, mirrored, atol=1e-6)


def test_gpu_max_section_matches_cpu(gpu):
    # Max thickness and Max section pooled, their gradient and the move limits, in float32 on the GPU
    from engine.fea import topology as T
    rng = np.random.default_rng(4)
    shape = (30, 20, 16)
    xb = rng.random(shape)
    domain = np.ones(shape)
    domain[:, :, 12:] = 0.0
    solid = np.zeros(shape, dtype=bool)
    solid[:2] = True
    free = (domain > 0) & ~solid
    phi = domain * rng.uniform(0.7, 1.0, shape)
    steps = rng.uniform(-0.1, 0.1, (2,) + shape)
    found = []
    for xp, dtype in ((np, np.float64), (gpu, np.float32)):
        D = xp.asarray(domain, dtype=dtype)
        p = T.Problem(None, xp.asarray(phi, dtype=dtype), xp.asarray(solid), xp.asarray(free), None, None,
                      xp.asarray(np.ones((1, 1, 1)), dtype=dtype), D, D, float(phi.sum()), 0.3, 2.0, None, xp, dtype,
                      local_sigma=1.5, section_sigma=3.0, section_share=0.15)
        opt = object.__new__(T.Optimizer)   # (no solve: only the constraint)
        opt.p, opt.local_scale, opt.moves, opt.last_step = p, 0.0, None, None
        g, dg = opt._local_constraint(xp.asarray(xb, dtype=dtype))
        for step in steps:
            opt._track_moves(xp.asarray(step, dtype=dtype))
        found.append((g, opt.section_max, opt.local_max, backend.to_host(dg), backend.to_host(opt.moves)))
    (g, section, local, dg, moves), (g_gpu, section_gpu, local_gpu, dg_gpu, moves_gpu) = found
    assert g_gpu == pytest.approx(g, rel=1e-3)
    assert section_gpu == pytest.approx(section, rel=1e-3)
    assert local_gpu == pytest.approx(local, rel=1e-3)
    assert np.allclose(dg_gpu, dg, atol=1e-3 * np.abs(dg).max())
    assert np.allclose(moves_gpu, moves, atol=1e-6)


def test_gpu_solver_kernels_match_cupy_and_cpu(gpu, monkeypatch):
    # the fused Warp kernels (engine/fea/gpu.py) against the CuPy code they replace, on every kind of
    # level (an L-shaped part: its fine and first coarse levels go element by element), and the solve
    # against the CPU in float64
    from engine.fea import solver as S
    from engine.fea.element import stiffness
    shape = (24, 12, 16)
    rng = np.random.default_rng(4)
    scale = rng.uniform(0.5, 1.0, shape)
    scale[6:, :, 4:] = 0.0                         # an L: 44% of the box
    scale[4:6, 4:8, :] = 1e-6                      # and a nearly empty slot (stiff next to soft)
    fixed = np.zeros(S.node_shape(shape) + (3,), dtype=bool)
    fixed[0] = True
    force = np.zeros(S.node_shape(shape) + (3,))
    force[-1, :, :5, 2] = -1.0
    H = S.Hierarchy(gpu.asarray(scale, dtype=np.float32), stiffness(0.3), fixed, (3, 2, 3), gpu, np.float32)
    assert H.levels[0].cells is not None and [lv.kind for lv in H.levels] == ["fine", "children", "stored", "stored"]

    def both(fn):
        monkeypatch.setattr(S, "GPU_KERNELS", False)
        a = backend.to_host(fn())
        monkeypatch.setattr(S, "GPU_KERNELS", True)
        return a, backend.to_host(fn())

    def close(a, b, rel=1e-5):
        return np.abs(a - b).max() <= rel * np.abs(a).max()
    for level in H.levels:
        u = gpu.asarray(rng.standard_normal(level.nodes + (3,)), dtype=np.float32)
        assert close(*both(lambda: level.apply(u))), level.kind
        assert close(*both(lambda: level.apply(u, masked=False))), level.kind
        if level.direct is None:
            r = u * level.free
            x = gpu.asarray(rng.standard_normal(level.nodes + (3,)), dtype=np.float32) * level.free
            assert close(*both(lambda: level.smooth(r))), level.kind
            assert close(*both(lambda: level.smooth(r, x.copy()))), level.kind
    u = gpu.asarray(rng.standard_normal(H.exact.nodes + (3,)))
    assert close(*both(lambda: H.exact.apply(u)), 1e-12)
    r = gpu.asarray(rng.standard_normal(H.levels[0].nodes + (3,)), dtype=np.float32) * H.free
    assert close(*both(lambda: H.precondition(r)))
    on_gpu, its, _ = H.solve(force, tol=1e-8)
    cpu, its_cpu, _ = S.Hierarchy(scale, stiffness(0.3), fixed, (3, 2, 3)).solve(force, tol=1e-8)
    assert close(cpu, backend.to_host(on_gpu), 1e-6)
    assert abs(its - its_cpu) <= 2


@pytest.mark.parametrize("shape, coarsenings", [((32, 32, 2), (3, 3, 0)), ((64, 3, 3), (4, 0, 0)), ((3, 24, 16), (0, 3, 3))])
def test_gpu_solver_kernels_on_grids_not_halved_along_every_axis(gpu, monkeypatch, shape, coarsenings):
    # a sheet or a bar: the level under the fine one has 4 or 2 children per element, not 8. The kernels took 8
    # as given and read past each element's row: a plain 200 x 200 x 1 sheet gave nan on the GPU, also in Static
    # Analysis, while the CPU solved it
    from engine.fea import solver as S
    from engine.fea.element import stiffness
    rng = np.random.default_rng(5)
    scale = rng.uniform(0.5, 1.0, shape)
    fixed = np.zeros(S.node_shape(shape) + (3,), dtype=bool)
    axis = int(np.argmax(shape))
    fixed[(slice(None),) * axis + (0,)] = True
    force = np.zeros(S.node_shape(shape) + (3,))
    force[(slice(None),) * axis + (-1,) + (slice(None),) * (2 - axis) + (2 if axis != 2 else 0,)] = -1.0
    H = S.Hierarchy(gpu.asarray(scale, dtype=np.float32), stiffness(0.3), fixed, coarsenings, gpu, np.float32)
    assert H.levels[1].kind == "children" and H.levels[1].const.shape[0] == 2 ** sum(k > 0 for k in coarsenings)

    def both(fn):
        monkeypatch.setattr(S, "GPU_KERNELS", False)
        a = backend.to_host(fn())
        monkeypatch.setattr(S, "GPU_KERNELS", True)
        return a, backend.to_host(fn())

    def close(a, b, rel=1e-5):
        return np.abs(a - b).max() <= rel * np.abs(a).max()
    level = H.levels[1]
    u = gpu.asarray(rng.standard_normal(level.nodes + (3,)), dtype=np.float32)
    assert close(*both(lambda: level.apply(u)))
    assert close(*both(lambda: level.apply(u, masked=False)))
    r = gpu.asarray(rng.standard_normal(H.levels[0].nodes + (3,)), dtype=np.float32) * H.free
    assert close(*both(lambda: H.precondition(r)))
    on_gpu, its, _ = H.solve(force, tol=1e-8)
    cpu, its_cpu, _ = S.Hierarchy(scale, stiffness(0.3), fixed, coarsenings).solve(force, tol=1e-8)
    assert close(cpu, backend.to_host(on_gpu), 1e-6)
    assert abs(its - its_cpu) <= 3


def test_gpu_topology_runs_on_a_sheet_and_a_slender_bar(gpu, tmp_path):
    # both stopped with "the solver did not converge" on the GPU: the kernels' children, then float32 rounding
    # beside soft elements (the sheet at iteration 1, the bar at 50), which float64 takes over from
    from engine import fields as F
    from engine.fea import topology as T
    from engine.fea.materials import PRESETS
    from engine.fea.model import Condition, Conditions
    from engine.graph.nodes import import_step
    from test_fea import _box_step, _face_id
    for size, iterations in (((200, 200, 1), 8), ((500, 5, 5), 60)):
        mesh = import_step(_box_step(tmp_path / f"{size[0]}.step", size), 0.01, 1, 0.5)[0]
        ids = mesh.face_attributes["cad_face"]
        end = lambda at: mesh.triangles[ids == _face_id(mesh, 0, at)]
        conds = Conditions((Condition("fixed", end(0.0)), Condition("load", end(float(size[0])), amount=50.0, direction="-Z")))
        p = T.prepare(F.body_from_mesh(mesh, 2.0), conds, PRESETS["PLA"], keep=30.0, skin=0.25, detail="draft", xp=gpu)
        opt = T.Optimizer(p, max_iterations=iterations).run()
        assert opt.iteration == iterations and np.isfinite(backend.to_host(opt.x)).all()
        assert opt.history[-1][1] == pytest.approx(0.3, abs=0.01)


def test_gpu_solver_kernels_let_their_hierarchy_go(gpu):
    # the kernels' data refers to its level weakly: a reference back kept every topology iteration's
    # hierarchy (GPU memory) until Python's cycle collector ran, and a fine result's check no longer fit;
    # the coarsest level's direct solver must not refer to its level either
    import gc
    import weakref
    from engine.fea import solver as S
    from engine.fea.element import stiffness
    shape = (16, 8, 8)
    fixed = np.zeros(S.node_shape(shape) + (3,), dtype=bool)
    fixed[0] = True
    force = np.zeros(S.node_shape(shape) + (3,))
    force[-1, :, :, 2] = -1.0
    H = S.Hierarchy(gpu.ones(shape, dtype=np.float32), stiffness(0.3), fixed, (2, 1, 1), gpu, np.float32)
    H.solve(force)
    assert all(level._gpu is not None for level in H.levels[:-1])
    levels = [weakref.ref(level) for level in H.levels] + [weakref.ref(H.exact)]
    gc.disable()
    try:
        del H
        assert all(r() is None for r in levels)
    finally:
        gc.enable()


def test_gpu_meshes_are_bounded_by_size(gpu, monkeypatch):
    # every new shape is a new key: the GPU mesh cache keeps at most MAX_MESH_BYTES, the newest always
    from collections import OrderedDict

    from engine import sdf_gpu
    meshes = [trimesh.creation.icosphere(subdivisions=3, radius=r) for r in (1.0, 2.0, 3.0)]
    monkeypatch.setattr(sdf_gpu, "_MESHES", OrderedDict())
    one = sdf_gpu._mesh(meshes[0].vertices, meshes[0].faces, None).nbytes
    assert one > 0
    monkeypatch.setattr(sdf_gpu, "MAX_MESH_BYTES", int(2.5 * one))
    for i, m in enumerate(meshes):
        sdf_gpu._mesh(m.vertices, m.faces, f"shape {i}")
    assert list(sdf_gpu._MESHES) == ["shape 1", "shape 2"]
    monkeypatch.setattr(sdf_gpu, "MAX_MESH_BYTES", 1)
    sdf_gpu._mesh(meshes[0].vertices, meshes[0].faces, "big")
    assert list(sdf_gpu._MESHES) == ["big"]


@pytest.mark.parametrize("radius", [2, 6])   # the volume kernel convolves, or the filter goes by FFT
def test_gpu_fused_oc_update_matches_cupy(gpu, monkeypatch, radius):
    # the optimality criteria's root finding in fused kernels (gpu.Volume), with and without Max thickness
    from engine.fea import solver as S
    from engine.fea import topology as T
    rng = np.random.default_rng(5)
    shape = (30, 20, 16)
    domain = np.ones(shape)
    domain[:, :, 12:] = 0.0
    solid = np.zeros(shape, dtype=bool)
    solid[:2] = True
    free = (domain > 0) & ~solid
    phi = domain * rng.uniform(0.7, 1.0, shape)
    o = np.arange(-radius, radius + 1)
    kernel = np.maximum(radius + 0.5 - np.sqrt(o[:, None, None] ** 2 + o[None, :, None] ** 2 + o[None, None, :] ** 2), 0)
    assert (kernel.size >= T.FFT_TAPS) == (radius == 6)
    D, K = gpu.asarray(domain, dtype=np.float32), gpu.asarray(kernel, dtype=np.float32)
    N = T._convolve(gpu, K.size)(D, K)
    norm = gpu.where(N > 1e-4, 1.0 / gpu.where(N > 1e-4, N, 1.0), 0.0).astype(np.float32)
    p = T.Problem(None, gpu.asarray(phi, dtype=np.float32), gpu.asarray(solid), gpu.asarray(free), None, None, K, norm,
                  D, float(phi.sum()), 0.3, float(radius), None, gpu, np.float32)
    opt = object.__new__(T.Optimizer)
    opt.p, opt.beta, opt.local_scale, opt.pruned, opt.moves = p, 4.0, 0.0, None, None
    f32 = lambda a: gpu.asarray(a, dtype=np.float32)
    x = f32(np.where(solid, 1.0, rng.uniform(0.1, 0.6, shape)) * domain)
    dc, dv, dg = f32(-rng.random(shape)), f32(rng.uniform(0.5, 1.0, shape)), f32(rng.random(shape) * 1e-3)
    moves = f32(rng.uniform(0.01, 0.2, shape))

    def limited():   # (Max section's move limits, per design variable)
        opt.moves = moves
        try:
            return opt._update_two(x, dc, dv, dg, 0.3, 0.02)
        finally:
            opt.moves = None

    for update in (lambda: opt._update(x, dc, dv), lambda: opt._update_two(x, dc, dv, dg, 0.3, 0.02), limited):
        monkeypatch.setattr(S, "GPU_KERNELS", False)
        a = update()
        monkeypatch.setattr(S, "GPU_KERNELS", True)
        b = update()
        assert float(gpu.abs(a - b)[p.free].max()) < 1e-4
        assert p.volume(opt.physical(b)[1]) == pytest.approx(p.volume(opt.physical(a)[1]), abs=1e-5)


@pytest.mark.parametrize("kind, joints, continuity", [("round", "fillet", "G1"), ("square tube", "fillet", "G2"),
                                                      ("I", "ball", "G3"), ("round tube", "plain", "G1")])
def test_gpu_frame_bars_match_cpu(gpu, kind, joints, continuity):
    # the frame kernel (profiles, mitres, fillets between different bars) on the GPU and on Warp's CPU
    from engine.frame import curves as CV
    from engine.frame.geometry import frame_solid
    from engine.frame.profiles import Profile
    nodes = np.array([(0, 0, 0), (30, 0, 5), (45, 20, 10), (5, 15, 20)], float)
    bars = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    fixed = np.zeros((4, 3), dtype=bool)
    n2, b2, _, c2, _, _ = CV.smooth_bars(nodes, bars, np.ones(5), np.arange(5), fixed, np.zeros((4, 3)), 1.0)
    sizes = np.linspace(3.0, 6.0, int(c2.max()) + 1)
    body = frame_solid(n2, b2, c2, sizes, Profile(kind, ratio=1.5 if kind == "I" else 1.0), joints, f"gpu-{kind}",
                       continuity=continuity)
    grid = Grid((-6, -6, -6), (52, 27, 27), 0.4)
    cpu = F.Context(grid, None, np).eval_full(body)
    on_gpu = backend.to_host(F.Context(grid, None, gpu).eval_full(body))
    assert np.abs(on_gpu - cpu).max() < 1e-4                           # float32 both, same kernel


def test_gpu_hole_depths_match_cpu(gpu, cad_part):
    # wall thickness under each hole and the face's rise around it: Warp ray casts on the GPU and
    # on Warp's CPU device give the same holes
    from engine import holes as H
    from engine import pattern2d as P2
    top = int(np.bincount(cad_part.face_attributes["cad_face"][cad_part.face_normals[:, 2] > 0.999]).argmax())
    place = lambda: H.place(cad_part, [top, 3, 14], P2.Shape("hexagon", 3.0), "staggered", 4.0, margin=1.0)
    on_gpu = place()
    backend.set_gpu(False)
    cpu = place()
    assert len(cpu) == len(on_gpu) > 50
    assert np.allclose(on_gpu.depth, cpu.depth, atol=1e-4) and np.allclose(on_gpu.outer, cpu.outer, atol=1e-4)
