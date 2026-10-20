import numpy as np
import pytest
import trimesh
from skimage.measure import marching_cubes

from engine import fields as F
from engine import lattice as L
from engine.graph import Evaluator, Graph, link
from engine.grid import Grid

CELL = 5.0


def values(field, grid):
    return F.Context(grid).eval_full(field)


def brute_strut_distance(points, kind, cell):
    """Distance to the struts of the 27 cells around each point: no folding, no clipping."""
    segs = [(np.asarray(a) + o, np.asarray(b) + o) for a, b in L.STRUTS[kind] for o in np.ndindex(3, 3, 3)]
    q = points / cell
    best = np.full(len(points), np.inf)
    base = np.floor(q) - 1
    for a, b in segs:
        a, b = base + a, base + b
        ab = b - a
        t = np.clip(((q - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        best = np.minimum(best, np.linalg.norm(q - a - t[:, None] * ab, axis=1))
    return best * cell


@pytest.mark.parametrize("kind", list(L.STRUTS))
def test_folded_struts_give_the_true_distance(kind):
    grid = Grid((-3.1, 0.4, 2.2), (9.0, 11.3, 13.7), 0.7)
    got = values(L.struts(kind, CELL, 0.0), grid).ravel()
    expect = brute_strut_distance(grid.points(), kind, CELL)
    assert np.abs(got - expect).max() < 1e-4


def test_strut_thickness_and_blend():
    on_edge = F.sample(L.struts("cubic", CELL, 1.0), (2.5, 0, 0))
    assert on_edge == pytest.approx(-0.5)
    # a node where three struts meet: blending adds material there (more negative)
    sharp = F.sample(L.struts("cubic", CELL, 1.0), (0.6, 0.6, 0))
    blended = F.sample(L.struts("cubic", CELL, 1.0, blend=1.0), (0.6, 0.6, 0))
    assert blended < sharp


@pytest.mark.parametrize("kind", ["gyroid", "schwarz_p", "diamond"])
def test_tpms_solid_fills_half(kind):
    grid = Grid((0, 0, 0), (2 * CELL - 0.1, 2 * CELL - 0.1, 2 * CELL - 0.1), 0.1)
    inside = (values(L.tpms(kind, CELL, 0.0, "solid"), grid) < 0).mean()
    assert inside == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("kind", list(L.TPMS))
def test_tpms_gradients_are_exact(kind):
    rng = np.random.default_rng(1)
    x, y, z = rng.uniform(-7, 7, (3, 2000))
    h = 1e-6
    exact = np.array(L.TPMS_GRADIENT[kind](x, y, z))
    f = L.TPMS[kind]
    numeric = np.array([(f(x + h, y, z) - f(x - h, y, z)), (f(x, y + h, z) - f(x, y - h, z)),
                        (f(x, y, z + h) - f(x, y, z - h))]) / (2 * h)
    assert np.abs(exact - numeric).max() < 1e-6


def test_tpms_sheet_thickness_is_in_mm():
    t, voxel = 0.4, 0.05
    grid = Grid((0, 0, 0), (CELL - voxel, CELL - voxel, CELL - voxel), voxel)
    raw = L.TPMS["gyroid"](*[2 * np.pi / CELL * np.asarray(a) for a in np.meshgrid(*grid.axes(), indexing="ij")])
    v, f, _, _ = marching_cubes(raw.astype(np.float32), 0.0, spacing=(voxel,) * 3)
    area = trimesh.Trimesh(v, f, process=False).area
    sheet_volume = (values(L.tpms("gyroid", CELL, t), grid) < 0).sum() * voxel ** 3
    assert sheet_volume == pytest.approx(area * t, rel=0.06)


def test_voronoi_nearest_struts_are_exact_and_repeatable():
    bounds = np.array([[0, 0, 0], [20, 15, 10]], float)
    field = L.voronoi(bounds, 4.0, 0.0, randomness=1.0, seed=3)
    grid = Grid(bounds[0], bounds[1], 0.9)
    got = values(field, grid).ravel()
    segs = L.voronoi_struts(bounds, 4.0, 1.0, 3)
    p = grid.points()
    a, ab = segs[:, 0], segs[:, 1] - segs[:, 0]
    best = np.full(len(p), np.inf)
    for i in range(len(segs)):
        t = np.clip(((p - a[i]) * ab[i]).sum(1) / (ab[i] @ ab[i]), 0, 1)
        best = np.minimum(best, np.linalg.norm(p - a[i] - t[:, None] * ab[i], axis=1))
    assert np.abs(got - best).max() < 1e-4
    assert np.array_equal(got, values(L.voronoi(bounds, 4.0, 0.0, 1.0, 3), grid).ravel())


def test_fill_keeps_a_skin():
    box = trimesh.creation.box((20, 20, 20))
    body = F.body_from_mesh(box, 2.0)
    open_lattice = L.struts("cubic", CELL, 0.5)
    filled = L.fill(body, open_lattice, skin=1.5)
    assert F.sample(filled, (9.0, 1.3, 1.7)) < 0     # 1 mm under the surface: in the skin
    assert F.sample(filled, (3.0, 1.3, 1.7)) > 0     # deep inside, between struts: empty
    assert F.sample(L.fill(body, open_lattice), (9.0, 1.3, 1.7)) > 0  # no skin


def test_lattice_nodes_mesh_a_filled_part(tmp_path):
    g = Graph()
    g.add("box", "input.import_step", path="unused")
    g.add("sdf", "implicit.from_cad", cad=link("box.bodies"), band=2.0)
    g.add("tpms", "lattice.tpms", pattern="gyroid", cell=6.0, thickness=0.8)
    g.add("fill", "lattice.fill", body=link("sdf.out"), lattice=link("tpms.out"), skin=1.0)
    g.add("mesh", "mesh.from_implicit", body=link("fill.out"), voxel=0.3, padding=1.0)
    g.add("export", "output.export", mesh=link("mesh.mesh"), path=str(tmp_path / "filled.stl"))
    ev = Evaluator(g, workers=1, log=None)
    original = ev._call
    ev._call = lambda node_id, ntype, kwargs: ({"bodies": trimesh.creation.box((20, 20, 20))}
                                               if ntype.type_id == "input.import_step" else original(node_id, ntype, kwargs))
    ev.run()
    mesh = trimesh.load(tmp_path / "filled.stl")
    assert mesh.is_watertight
    assert 0.2 * 8000 < mesh.volume < 0.8 * 8000   # a skin and a lattice, not solid, not empty


@pytest.mark.parametrize("make", [lambda: L.tpms("gyroid", CELL, 0.6),
                                  lambda: L.struts("octet", CELL, 0.8, blend=0.5),
                                  lambda: L.voronoi([[0, 0, 0], [30, 30, 30]], 4.0, 0.8, seed=1),
                                  lambda: L.voronoi([[0, 0, 0], [30, 30, 30]], 4.0, F.noise(8.0) * 0.3 + 0.6,
                                                    seed=1, units="density")])
def test_lattices_are_the_same_in_blocks(make):
    field = make()
    whole = Grid.block((0, 0, 0), 0.25, (0, 0, 0), (97, 97, 97))
    ref = values(field, whole)[24:73, 36:85, 12:61]
    block = values(field, Grid.block((0, 0, 0), 0.25, (24, 36, 12), (49, 49, 49)))
    assert np.array_equal(block, ref)


@pytest.mark.parametrize("make", [
    lambda d: L.voronoi(np.array([[0, 0, 0], [40, 40, 24.0]]), 5.0, d, units="density"),
    lambda d: L.struts("octet", CELL, d, blend=0.4, units="density"),
    lambda d: L.tpms("gyroid", CELL, d, "sheet", units="density"),
    lambda d: L.tpms("diamond", CELL, 0.0, "solid (other side)", d, units="density")])
def test_density_is_the_share_of_solid_material(make):
    # in mm the solid share saturates (a 5 mm foam is solid from ~7 mm struts on); density is
    # the share itself, so fades change evenly
    grid = Grid((7.3, 6.1, 5.2), (32.9, 31.7, 18.8), 0.2)
    for d in (0.0, 0.25, 0.5, 0.8, 1.0):
        assert (values(make(d), grid) < 0).mean() == pytest.approx(d, abs=0.04)


def test_density_range_follows_the_input():
    foam = L.voronoi(np.array([[0, 0, 0], [30, 30, 30.0]]), 5.0, F.smoothstep(F.noise(5.0), 0, 1), units="density")
    thickness = foam.children[0]
    lo, hi = thickness.range
    assert lo == pytest.approx(0.0, abs=0.2) and 5.0 < hi < 9.0   # 0 = no struts, 1 = solid (~7 mm)
