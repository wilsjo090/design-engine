"""Finite element analysis: element, matrix-free products, multigrid, and results checked against
textbook answers (uniform tension, a cantilever beam)."""
import numpy as np
import pytest

from engine import fields as F
from engine.fea import element as EL
from engine.fea.analysis import Model, build_model, element_stresses, solve, static_analysis
from engine.fea.materials import PRESETS, Material
from engine.fea.model import (Condition, ConditionError, Conditions, FEGrid, apply_conditions, plan_coarsenings,
                              sample_triangles)
from engine.fea.solver import Hierarchy, prolong, restrict


def test_element_stiffness_is_symmetric_with_six_rigid_modes():
    K = EL.stiffness(0.3)
    assert np.allclose(K, K.T)
    eig = np.linalg.eigvalsh(K)
    assert (np.abs(eig[:6]) < 1e-10).all() and (eig[6:] > 1e-3).all()
    assert np.allclose(K.sum(axis=1).reshape(8, 3).sum(axis=0), 0)   # a translation costs nothing


def _assembled(scale, KE):
    """Dense stiffness matrix of a grid, the plain way."""
    nx, ny, nz = scale.shape
    X, Y, Z = nx + 1, ny + 1, nz + 1
    K = np.zeros((X * Y * Z * 3,) * 2)
    for i, j, k in np.ndindex(scale.shape):
        nodes = [((i + cx) * Y + (j + cy)) * Z + (k + cz) for cx, cy, cz in EL.CORNERS]
        dofs = np.array([3 * n + d for n in nodes for d in range(3)])
        K[np.ix_(dofs, dofs)] += scale[i, j, k] * KE
    return K


@pytest.fixture(params=["slabs", "elements"])
def layout(request, monkeypatch):
    """Run a test with the whole-grid layout and with the element-by-element one."""
    from engine.fea import solver
    monkeypatch.setattr(solver, "DENSE_FILL", 0.0 if request.param == "slabs" else 1.01)
    return request.param


def test_matrix_free_product_equals_the_assembled_matrix(layout):
    rng = np.random.default_rng(1)
    scale = rng.uniform(0.1, 2.0, (3, 4, 2))
    scale[1, 2, 0] = 0.0   # an empty element
    KE = EL.stiffness(0.35)
    fixed = np.zeros((4, 5, 3, 3), dtype=bool)
    fixed[0, :, :, :] = True
    fixed[2, 1, 1, 2] = True
    H = Hierarchy(scale, KE, fixed, (0, 0, 0))
    u = rng.standard_normal((4, 5, 3, 3)) * H.free
    expect = (_assembled(scale, KE) @ u.ravel()).reshape(u.shape) * H.free
    assert np.allclose(H.apply(u), expect, rtol=1e-12, atol=1e-12)


def test_coarse_levels_are_galerkin_operators_of_the_fixed_fine_one(layout):
    rng = np.random.default_rng(2)
    scale = rng.uniform(0.01, 1.0, (4, 4, 8))
    scale[:2, :2, :3] = 0.0
    fixed = np.zeros((5, 5, 9, 3), dtype=bool)
    fixed[:, 0, :, 1] = True
    fixed[3, 3, 5, :] = True   # an odd node: only dilated coarse nodes can hold it
    H = Hierarchy(scale, EL.stiffness(0.3), fixed, (1, 2, 2))
    assert [lv.kind for lv in H.levels] == ["fine", "children", "stored"]
    for fine, coarse in zip(H.levels, H.levels[1:]):
        vc = rng.standard_normal(coarse.nodes + (3,)) * coarse.free
        galerkin = restrict(fine.apply(prolong(vc, fine.factors)), fine.factors) * coarse.free
        assert np.allclose(coarse.apply(vc), galerkin, rtol=1e-10, atol=1e-10 * np.abs(galerkin).max())


def test_coarsening_plan_pads_little():
    from engine.fea.model import MAX_COARSE, MAX_PADDING
    for shape in ((100, 10, 10), (203, 61, 33), (30, 10, 10)):
        ks, padded = plan_coarsenings(shape)
        assert all(n <= p <= MAX_PADDING * n and p % 2 ** k == 0 for p, n, k in zip(padded, shape, ks))
        assert np.prod([p // 2 ** k for p, k in zip(padded, ks)]) <= MAX_COARSE


def test_uniform_tension_is_exact():
    # a block pulled at x = L, held by sliding supports on three faces: uniform stress sigma, stretch
    # sigma L / E and contraction -nu sigma W / E, exactly, for hex8 at any element size
    h, shape, E, nu, sigma = 0.5, (8, 4, 4), 1000.0, 0.3, 2.0
    nodes = tuple(n + 1 for n in shape)
    fixed = np.zeros(nodes + (3,), dtype=bool)
    fixed[0, :, :, 0] = fixed[:, 0, :, 1] = fixed[:, :, 0, 2] = True
    force = np.zeros(nodes + (3,))
    for j in range(shape[1]):
        for k in range(shape[2]):
            for dj in (0, 1):
                for dk in (0, 1):
                    force[-1, j + dj, k + dk, 0] += sigma * h * h / 4
    ks, padded = plan_coarsenings(shape)
    model = Model(FEGrid(np.zeros(3), h, shape, ks), np.ones(shape), fixed, force)
    material = Material("test", E, nu, 1.0, 100.0)
    u, its, res = solve(model, material)
    L, W = shape[0] * h, shape[1] * h
    assert np.allclose(u[-1, :, :, 0], sigma * L / E, rtol=1e-6)
    assert np.allclose(u[:, -1, :, 1], -nu * sigma * W / E, rtol=1e-6)
    assert np.allclose(element_stresses(model, material, u), sigma, rtol=1e-6)


def test_multigrid_converges_with_holes_and_a_million_to_one_contrast():
    # solid and nearly empty blobs (as in a topology optimization), solid plates where it is held
    # and loaded. With damped Jacobi smoothing this stalled: lmax(D^-1 K) is 3.1 on a uniform part
    # but 4.2-4.9 where stiff meets soft (see solver.DEGREE)
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    blobs = gaussian_filter(rng.standard_normal((32, 32, 32)), 2.0)
    scale = np.where(blobs > np.median(blobs), 1.0, 1e-6)
    scale[:2] = scale[-2:] = 1.0
    fixed = np.zeros((33, 33, 33, 3), dtype=bool)
    fixed[0] = True
    force = np.zeros((33, 33, 33, 3))
    force[-1, :, :, 2] = -1.0
    H = Hierarchy(scale, EL.stiffness(0.3), fixed, plan_coarsenings((32, 32, 32))[0])
    x, y = rng.standard_normal((2,) + force.shape) * H.free
    assert np.isclose(np.vdot(y, H.precondition(x)), np.vdot(x, H.precondition(y)), rtol=1e-9)  # symmetric
    u, its, res = H.solve(force, tol=1e-6)
    assert res <= 1e-6 and its < 80, its   # 56 when written
    r = (force * H.free - H.apply(u))
    assert np.linalg.norm(r) <= 1.01e-6 * np.linalg.norm(force * H.free)


def test_a_float32_multigrid_stays_float32_and_cg_stops_when_converged(monkeypatch):
    # lmax was a NumPy float64 scalar, which turns float32 arrays into float64 ones (NumPy 2, CuPy
    # 14): the GPU's float32 preconditioner ran in float64
    rng = np.random.default_rng(4)
    scale = rng.uniform(0.5, 2.0, (16, 8, 8))
    fixed = np.zeros((17, 9, 9, 3), dtype=bool)
    fixed[0] = True
    force = np.zeros((17, 9, 9, 3))
    force[-1, :, :, 2] = -1.0
    H = Hierarchy(scale, EL.stiffness(0.3), fixed, (2, 1, 1), np, np.float32)
    assert all(type(level.lmax) is float for level in H.levels[:-1])
    assert H.precondition((force * H.free).astype(np.float32)).dtype == np.float32
    cycles = []
    vcycle = H.precondition
    monkeypatch.setattr(H, "precondition", lambda r: cycles.append(1) or vcycle(r))
    u, its, res = H.solve(force, tol=1e-6)
    assert res <= 1e-6 and len(cycles) == its   # none for the residual that already meets the tolerance


# --- the whole analysis on CAD geometry ------------------------------------------------------------
def _box_step(path, size):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), *size).Shape(), STEPControl_AsIs)
    writer.Write(str(path))
    return str(path)


@pytest.fixture(scope="module")
def beam_step(tmp_path_factory):
    return _box_step(tmp_path_factory.mktemp("fea") / "beam.step", (100, 10, 10))


@pytest.fixture(scope="module")
def beam(beam_step):
    """A 100 x 10 x 10 mm box as Import STEP gives it."""
    from engine.graph.nodes import import_step
    return import_step(beam_step, 0.01, 1, 0.5)[0]


def _face_id(mesh, axis, value):
    """The CAD face lying in the plane coordinate[axis] == value."""
    ids = mesh.face_attributes["cad_face"]
    on = np.abs(mesh.triangles_center[:, axis] - value) < 1e-6
    return int(np.bincount(ids[on]).argmax())


def _face_triangles(mesh, axis, value):
    return mesh.triangles[mesh.face_attributes["cad_face"] == _face_id(mesh, axis, value)]


def test_tessellated_faces_point_outward(beam):
    _, _, normals = sample_triangles(_face_triangles(beam, 2, 10.0), 5.0)
    assert np.allclose(normals, [0, 0, 1])
    _, _, normals = sample_triangles(_face_triangles(beam, 0, 0.0), 5.0)
    assert np.allclose(normals, [-1, 0, 0])


def test_loads_add_up_to_the_amount(beam):
    part = F.body_from_mesh(beam, 2.0)
    top = _face_triangles(beam, 2, 10.0)
    conds = Conditions((Condition("fixed", _face_triangles(beam, 0, 0.0)),
                        Condition("load", top, load="pressure", amount=0.5),
                        Condition("load", top, load="force", amount=30.0, direction="custom", vector=(1, 0, 1))))
    model = build_model(part, conds, h=1.0)
    total = model.force.reshape(-1, 3).sum(axis=0)
    # the fixed end's nodes take the load that sits on them: none of the top face's load is lost
    # there except the strip within one element of the end (1 of 100 mm)
    expect = np.array([0, 0, -0.5 * 100 * 10]) + 30 * np.array([1, 0, 1]) / np.sqrt(2)
    assert np.allclose(total, expect, rtol=0.02)


def test_a_bearing_load_presses_only_on_the_contact_side():
    # a pin pulled down in the wall bracket's hole (axis along Y at x = 50, z = 40, radius 4): the
    # 200 N go on the bottom half only, square to the face, adding up to 200 N straight down
    from engine.fea.model import markers
    from engine.graph.nodes import import_step
    from pathlib import Path
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_space.step"),
                       0.01, 1, 0.5)[0]
    ids = mesh.face_attributes["cad_face"]
    hole = Condition("load", mesh.triangles[ids == 6], load="bearing", amount=200.0, direction="-Z")
    pts, areas, normals = sample_triangles(hole.triangles, 0.5)
    t = hole.traction(normals)
    assert np.allclose((t * areas[:, None]).sum(axis=0), [0, 0, -200], atol=2.0)   # within 1%
    assert np.all(np.linalg.norm(t[pts[:, 2] > 40.01], axis=1) == 0)                # nothing on the top half
    bottom = np.argmin(pts[:, 2])
    assert t[bottom] @ [0, 0, -1] > 0.95 * np.linalg.norm(t, axis=1).max()        # most at the very bottom
    marker_points, _, sizes = markers(hole)
    assert len(marker_points) and np.all(marker_points[:, 2] <= 40.0 + 1e-6)
    low = np.argmax(marker_points[:, 2])                                          # nearest the hole's middle:
    assert sizes[low] < 0.5 < sizes.max() <= 1.0                                  # a short arrow
    with pytest.raises(ConditionError, match="needs a direction"):
        Condition("load", hole.triangles, load="bearing", amount=1.0, direction="into the face").traction(normals)


def test_a_load_partly_off_the_part_keeps_its_total():
    # a topology result often lacks the unloaded side of a pin hole: put back by area, the 200 N
    # bearing load became 276 N there; by what each point carries it stays 200 N
    from engine.graph.nodes import import_step
    from pathlib import Path
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_space.step"),
                       0.01, 1, 0.5)[0]
    ids = mesh.face_attributes["cad_face"]
    conds = Conditions((Condition("fixed", mesh.triangles[ids == 0]),
                        Condition("load", mesh.triangles[ids == 6], load="bearing", amount=200.0, direction="-Z")))
    lower = F.intersect(F.body_from_mesh(mesh, 2.0), F.Z - 40.0)   # nothing above the pin's axis
    model = build_model(lower, conds, "draft")
    assert np.allclose(model.force.reshape(-1, 3).sum(axis=0), [0, 0, -200], atol=2.0)
    load_notes = [n for n in model.notes if n.startswith("load")]
    assert not load_notes, load_notes                               # the half that is gone carried nothing


def test_long_thin_triangles_get_points_by_area():
    # a 29.9 x 0.2 mm strip, as along a hole: the m x m split put 120 x 120 points on it, nearly all
    # across its width; the grid stretched to it 120 x 4, with the same area and centroid. Supports
    # keep the old split (which nodes lie near a face depends on how densely it is covered)
    t = np.array([[[0, 0, 0], [29.9, 0, 0], [0, 0.2, 0]]], dtype=float)
    pts, areas, _ = sample_triangles(t, 0.25)
    assert len(pts) == 120 * 4 and areas.sum() == pytest.approx(2.99, rel=1e-12)
    assert np.allclose((pts * areas[:, None]).sum(axis=0) / areas.sum(), t[0].mean(axis=0), atol=1e-12)
    assert len(sample_triangles(t, 0.25, stretch=False)[0]) == 120 * 120


def test_a_torque_turns_about_its_axis_and_adds_up_to_no_force():
    # 2 N m about +Y on the wall bracket's pin hole (axis along Y at x = 50, z = 40, radius 4):
    # tangential everywhere, no net force, exactly the moment; right-hand rule: the bottom moves -X
    from engine.graph.nodes import import_step
    from pathlib import Path
    mesh = import_step(str(Path(__file__).resolve().parents[1] / "samples" / "wall_bracket_space.step"),
                       0.01, 1, 0.5)[0]
    hole = Condition("load", mesh.triangles[mesh.face_attributes["cad_face"] == 6], load="torque", amount=2.0,
                     direction="+Y")
    pts, areas, normals = sample_triangles(hole.triangles, 0.25)
    t = hole.traction(normals, pts)
    f = t * areas[:, None]
    centre = hole.geometry[1]
    assert np.allclose(centre, [50, 15, 40], atol=1e-6)
    assert np.allclose(f.sum(axis=0), 0, atol=1e-6)
    assert np.allclose(np.cross(pts - centre, f).sum(axis=0), [0, 2000, 0], atol=0.1)
    size = np.linalg.norm(t, axis=1)
    assert np.allclose(size, size.mean(), rtol=0.01)                                # the same all round
    assert np.all(np.abs(np.einsum("ij,ij->i", t, normals)) < 0.06 * size)          # along the (faceted) wall
    assert np.all(np.abs(t[:, 1]) < 1e-9) and t[np.argmin(pts[:, 2]), 0] < 0
    with pytest.raises(ConditionError, match="needs the axis"):                    # a round face has no "into"
        Condition("load", hole.triangles, load="torque", amount=1.0).check()


def test_a_force_with_an_offset_acts_as_if_it_were_there(beam):
    # 100 N down, 50 mm beyond the beam's end face: the face carries the force and its moment, so
    # about the point where the force acts the face's load turns nothing
    end = Condition("load", _face_triangles(beam, 0, 100.0), amount=100.0, direction="-Z", offset=(50.0, 0, 0))
    pts, areas, normals = sample_triangles(end.triangles, 0.5)
    f = end.traction(normals, pts) * areas[:, None]
    there = end.geometry[1] + [50, 0, 0]
    assert np.allclose(f.sum(axis=0), [0, 0, -100], atol=1e-9)
    assert np.allclose(np.cross(pts - there, f).sum(axis=0), 0, atol=10)         # of 5000 N mm (4 when written)
    plain = Condition("load", end.triangles, amount=100.0, direction="-Z")
    assert plain.key == Condition("load", end.triangles, amount=100.0, direction="-Z", offset=(0, 0, 0)).key
    assert plain.key != end.key
    assert Condition("load", end.triangles, load="pressure", amount=1.0, offset=(5, 0, 0)).offset_moment() is None


def test_an_offset_force_bends_a_cantilever_like_the_force_and_its_moment(beam):
    # Timoshenko tip deflection plus that of the end moment M = F e: F L^3/(3EI) + F L/(kGA) + M L^2/(2EI)
    F_, L, b, e = 100.0, 100.0, 10.0, 50.0
    pla = PRESETS["PLA"]
    I = b * b ** 3 / 12
    G = pla.E / (2 * (1 + pla.nu))
    tip = F_ * L ** 3 / (3 * pla.E * I) + F_ * L / (5 / 6 * G * b * b) + F_ * e * L ** 2 / (2 * pla.E * I)
    conds = Conditions((Condition("fixed", _face_triangles(beam, 0, 0.0)),
                        Condition("load", _face_triangles(beam, 0, 100.0), amount=F_, direction="-Z",
                                  offset=(e, 0, 0))))
    result = static_analysis(F.body_from_mesh(beam, 2.0), conds, pla, h=1.0)
    assert F.sample(result.fields["displacement"], (100.0, 5.0, 5.0)) == pytest.approx(tip, rel=0.02)
    assert any("also carry a moment of 5 N·m" in n for n in result.report["notes"])


def test_a_torque_twists_a_square_bar_as_saint_venant(beam):
    # twist per length T / (G J), J = 0.1406 a^4 for a square bar; measured between 10 and 90 mm
    # (away from the clamped end and the loaded one) on the middle of the top face, where the
    # section does not warp, so it moves by the twist angle x 5 mm
    T_, a = 5.0, 10.0
    pla = PRESETS["PLA"]
    G = pla.E / (2 * (1 + pla.nu))
    rate = 1000 * T_ / (G * 0.1406 * a ** 4)
    conds = Conditions((Condition("fixed", _face_triangles(beam, 0, 0.0)),
                        Condition("load", _face_triangles(beam, 0, 100.0), load="torque", amount=T_,
                                  direction="+X")))
    model = build_model(F.body_from_mesh(beam, 2.0), conds, h=1.0)
    total = model.force.reshape(-1, 3)
    nodes = model.grid.origin + np.stack(np.meshgrid(*[np.arange(n) for n in model.grid.nodes], indexing="ij"),
                                         axis=-1).reshape(-1, 3) * model.grid.h
    assert np.allclose(total.sum(axis=0), 0, atol=1e-6)
    assert np.allclose(np.cross(nodes - [100, 5, 5], total).sum(axis=0), [1000 * T_, 0, 0], rtol=1e-3)
    result = static_analysis(F.body_from_mesh(beam, 2.0), conds, pla, h=1.0)
    turn = lambda x: F.sample(result.fields["displacement"], (x, 5.0, 10.0)) / 5.0
    assert (turn(90.0) - turn(10.0)) / 80.0 == pytest.approx(rate, rel=0.03)
    assert result.report["imbalance"] < 1e-6


def test_cantilever_matches_beam_theory(beam):
    # Timoshenko: tip deflection F L^3 / (3 E I) + F L / (k G A); bending stress M y / I
    F_, L, b = 100.0, 100.0, 10.0
    pla = PRESETS["PLA"]
    I = b * b ** 3 / 12
    G = pla.E / (2 * (1 + pla.nu))
    tip = F_ * L ** 3 / (3 * pla.E * I) + F_ * L / (5 / 6 * G * b * b)
    part = F.body_from_mesh(beam, 2.0)
    conds = Conditions((Condition("fixed", _face_triangles(beam, 0, 0.0)),
                        Condition("load", _face_triangles(beam, 0, 100.0), amount=F_, direction="-Z")))
    result = static_analysis(part, conds, pla, h=1.0)
    deflection = F.sample(result.fields["displacement"], (100.0, 5.0, 5.0))
    # 1.1% short when written (the plain hex8 1.8%, and 8.8% before the grid started at the part's
    # corner: half-filled elements on the flat faces); the clamped end makes up most of it
    assert deflection == pytest.approx(tip, rel=0.02)
    assert result.report["imbalance"] < 1e-6
    # bending stress at the centres of the elements next to the top face, half way along
    model = build_model(part, conds, h=1.0)
    u, _, _ = solve(model, pla)
    vm = element_stresses(model, pla, u)
    i, j, k = 50, 5, 9
    centre = model.grid.origin + (np.array([i, j, k]) + 0.5) * model.grid.h
    expect = F_ * (L - centre[0]) * (centre[2] - b / 2) / I
    assert vm[i, j, k] == pytest.approx(expect, rel=0.01)


def test_kept_displacements_serve_another_density_strength_and_stiffness(beam, tmp_path):
    # only Poisson's ratio changes the shape of the displacements; E scales them, and stresses not at all
    from engine.cache import DiskCache
    store = DiskCache(tmp_path)
    store.MIN_BYTES = 0
    part = F.body_from_mesh(beam, 2.0)
    conds = Conditions((Condition("fixed", _face_triangles(beam, 0, 0.0)),
                        Condition("load", _face_triangles(beam, 0, 100.0), amount=50.0, direction="-Z")))
    pla = PRESETS["PLA"]
    first = static_analysis(part, conds, pla, h=2.0, store=store).report
    other = static_analysis(part, conds, Material("other", 2 * pla.E, pla.nu, 7.8, 300.0), h=2.0, store=store).report
    assert not first["cached"] and other["cached"] and other["iterations"] == 0
    assert other["max_displacement"] == pytest.approx(first["max_displacement"] / 2, rel=1e-6)
    assert other["max_away"] == pytest.approx(first["max_away"], rel=1e-6)
    assert other["mass"] == pytest.approx(first["mass"] * 7.8 / pla.density)
    softer = static_analysis(part, conds, Material("softer", pla.E, 0.25, 1.0, 50.0), h=2.0, store=store).report
    assert not softer["cached"]


def test_the_safety_factor_colours_on_a_fixed_scale(beam):
    # safe everywhere (steel, 1 N): the field is 10 everywhere, and on its own range (10..10) it was
    # painted in the "bad" colour; a failing part's scale reaches down to its smallest safety factor
    part = F.body_from_mesh(beam, 2.0)
    conds = lambda amount: Conditions((Condition("fixed", _face_triangles(beam, 0, 0.0)),
                                       Condition("load", _face_triangles(beam, 0, 100.0), amount=amount,
                                                 direction="-Z")))
    safe = static_analysis(part, conds(1.0), PRESETS["Steel S355"], h=2.0)
    assert safe.report["min_safety"] > 10 and safe.fields["safety"].display_range == (1.0, 10.0)
    failing = static_analysis(part, conds(2000.0), PRESETS["PLA"], h=2.0)
    lo, hi = failing.fields["safety"].display_range
    assert lo == pytest.approx(failing.report["min_safety"]) and lo < 1.0 and hi == 10.0


def test_a_part_without_support_is_refused(beam):
    part = F.body_from_mesh(beam, 2.0)
    conds = Conditions((Condition("load", _face_triangles(beam, 0, 100.0), amount=10.0, direction="-Z"),))
    with pytest.raises(ConditionError, match="not held"):
        build_model(part, conds, h=2.0)


def test_loose_pieces_are_left_out(beam):
    part = F.union(F.body_from_mesh(beam, 2.0), _ball((50, 40, 5), 4.0))
    conds = Conditions((Condition("fixed", _face_triangles(beam, 0, 0.0)),
                        Condition("load", _face_triangles(beam, 0, 100.0), amount=10.0, direction="-Z")))
    model = build_model(part, conds, h=1.0, bounds=np.array([[0, 0, 0], [100, 45, 10]]))
    assert any("loose piece" in n for n in model.notes)


def _ball(centre, radius):
    c = np.asarray(centre, dtype=float)
    return F.sqrt((F.X - c[0]) * (F.X - c[0]) + (F.Y - c[1]) * (F.Y - c[1]) + (F.Z - c[2]) * (F.Z - c[2])) - radius


def test_conditions_away_from_the_part_are_reported(beam):
    far = _face_triangles(beam, 0, 0.0) + np.array([0.0, 0.0, 50.0])
    with pytest.raises(ConditionError, match="not on the part"):
        apply_conditions(*_grid_and_fill(beam), Conditions((Condition("fixed", far),)))


def _grid_and_fill(beam):
    model_part = F.body_from_mesh(beam, 2.0)
    from engine.fea.model import element_fill, make_grid
    grid = make_grid(beam.bounds, 2.0)
    return grid, element_fill(model_part, grid)


# --- the nodes -------------------------------------------------------------------------------------
def _recipe(beam_step, beam, show="stress"):
    end, tip = _face_id(beam, 0, 0.0), _face_id(beam, 0, 100.0)
    link = lambda ref: {"link": ref}
    return {"version": 1, "nodes": {
        "import": {"type": "input.import_step", "inputs": {"path": beam_step}},
        "body": {"type": "input.pick_body", "inputs": {"bodies": link("import.bodies")}},
        "pla": {"type": "sim.material", "inputs": {"preset": "PLA"}},
        "fixed": {"type": "sim.fixed", "inputs": {"cad": link("body.body"), "faces": str(end)}},
        "load": {"type": "sim.load", "inputs": {"cad": link("body.body"), "faces": str(tip), "amount": 100.0,
                                                "direction": "-Z", "add_to": link("fixed.conditions")}},
        "static": {"type": "sim.static", "inputs": {"part": link("body.body"), "material": link("pla.material"),
                                                    "conditions": link("load.conditions"), "detail": "draft",
                                                    "show": show}},
    }}


def test_the_nodes_analyse_a_part_and_report(beam_step, beam):
    from engine.graph import Evaluator, Graph
    events = []
    ev = Evaluator(Graph.from_dict(_recipe(beam_step, beam)), workers=1, log=None, on_event=events.append)
    out = ev.run(["static"])["static"]
    lines = [e["text"] for e in events if e["type"] == "message"]
    assert any(t.startswith("Static analysis (draft") for t in lines), lines
    assert any("max stress" in t and "safety factor" in t for t in lines), lines
    stress = out["result"]
    assert stress.unit == "MPa" and stress.range[1] > 10       # bending stress near the clamp ~60 MPa
    assert F.sample(out["displacement"], (100.0, 5.0, 5.0)) == pytest.approx(11.4, rel=0.08)
    done = {e["node"]: e for e in events if e["type"] == "node_done"}
    fixed_markers, load_markers = done["fixed"]["markers"], done["load"]["markers"]
    assert fixed_markers[0]["kind"] == "fixed" and len(fixed_markers[0]["points"]) > 0
    assert load_markers[0]["kind"] == "load" and np.allclose(load_markers[0]["dirs"], [0, 0, -1])
    assert all(p[0] == pytest.approx(100.0) for p in load_markers[0]["points"])


def test_previews_show_supports_and_colour_the_analysed_part(beam_step, beam):
    from engine.server import build_run_graph
    _, targets, views, _ = build_run_graph(_recipe(beam_step, beam), "preview", "draft", ["static"])
    assert {"fixed", "load"} <= set(targets)                         # markers even when nothing is shown
    assert views["static"][3] == ("static", "result")                # coloured by its own result
    half_wired = _recipe(beam_step, beam)
    del half_wired["nodes"]["fixed"]["inputs"]["cad"]
    _, targets, _, _ = build_run_graph(half_wired, "preview", "draft", [])
    assert "fixed" not in targets and "load" not in targets         # would fail the run
