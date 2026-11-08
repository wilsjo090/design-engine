from pathlib import Path

import igl
import numpy as np
import pytest
import trimesh

from engine import fields as F
from engine import tiles
from engine.cad_import import load_step, tessellate
from engine.dualcontour import COUNT, TABLE, dual_contour
from engine.graph import GraphError
from engine.graph.core import REGISTRY
from engine.grid import Grid
from engine.meshing import field_to_mesh

TURN = np.array([[0.8660254, -0.3535534, 0.3535534],   # a box turned off the grid's axes
                 [0.5, 0.6123724, -0.6123724],
                 [0.0, 0.7071068, 0.7071068]])
HALF = np.array([2.0, 1.4, 1.0])


def box(points, half=HALF, turn=TURN, centre=(0.0, 0.0, 0.0)):
    """Exact distance to a turned box."""
    q = np.abs((points - np.asarray(centre)) @ turn) - half
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def sample(fn, lo, hi, voxel):
    grid = Grid(lo, hi, voxel)
    X, Y, Z = grid.coords()
    return fn(np.stack([X, Y, Z], axis=-1).astype(np.float64)).astype(np.float32), grid


def distance_to(points, mesh):
    d2, _, _ = igl.point_mesh_squared_distance(np.asarray(points, float), np.asarray(mesh.vertices, float),
                                               np.asarray(mesh.faces, np.int64))
    return np.sqrt(d2)


def box_features(half=HALF, turn=TURN, per_edge=20):
    """The box's corners and points along its 12 edges (inside the edges' ends)."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    corners = signs * half
    edge_points = []
    for a, b in [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(signs[i] != signs[j]) == 1]:
        t = np.linspace(0.1, 0.9, per_edge)[:, None]
        edge_points.append(corners[a] + t * (corners[b] - corners[a]))
    return corners @ turn.T, np.concatenate(edge_points) @ turn.T


def closed(mesh):
    return mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0


def test_each_cell_case_gets_pieces_of_surface():
    assert (COUNT[0] == 0).all() and (COUNT[255] == 0).all()
    for c in range(8):  # one corner in, or one corner out: one piece of three crossed edges
        for case in (1 << c, 255 ^ (1 << c)):
            assert (COUNT[case] == 1).all() and ((TABLE[case] == 0).sum(axis=1) == 3).all()
    # corners 0 and 3 in, diagonal on the bottom face (face 4): kept apart, two pieces; joined across
    # the middle, one; the same face the other way round
    assert COUNT[0b1001, 0] == 2 and COUNT[0b1001, 1 << 4] == 1
    assert COUNT[255 ^ 0b1001, 0] == 1 and COUNT[255 ^ 0b1001, 1 << 4] == 2
    assert (COUNT[0b10000001] == 2).all()                         # two opposite corners: no face between
    for case in range(256):
        crossed = sum(((case >> c0) & 1) != ((case >> c1) & 1) for c0, c1 in
                      [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)])
        assert ((TABLE[case] >= 0).sum(axis=1) == crossed).all()


def test_a_box_keeps_its_corners_and_edges_where_marching_cubes_cuts_them_off():
    voxel = 0.3
    values, grid = sample(box, (-3.3, -3.1, -3.2), (3.2, 3.3, 3.1), voxel)
    dc, mc = dual_contour(values, grid), field_to_mesh(values, grid)
    assert closed(dc)
    assert dc.volume == pytest.approx(8 * np.prod(HALF), rel=0.005)
    corners, edges = box_features()
    dc_corners, mc_corners = distance_to(corners, dc), distance_to(corners, mc)
    dc_edges, mc_edges = distance_to(edges, dc), distance_to(edges, mc)
    # (measured: corners 0.09 voxel on average, at most 0.16; edges 0.03, at most 0.10)
    assert dc_corners.mean() < 0.12 * voxel and dc_corners.max() < 0.25 * voxel
    assert dc_edges.mean() < 0.05 * voxel and dc_edges.max() < 0.15 * voxel
    assert mc_corners.mean() > 4 * dc_corners.mean() and mc_edges.mean() > 4 * dc_edges.mean()
    assert dc.metadata["voxel"] == voxel and not dc.metadata["clipped"]


def test_a_sphere_stays_on_its_radius():
    values, grid = sample(lambda p: np.linalg.norm(p, axis=-1) - 2.0, (-2.6,) * 3, (2.6,) * 3, 0.25)
    mesh = dual_contour(values, grid)
    assert closed(mesh)
    assert np.abs(np.linalg.norm(mesh.vertices, axis=1) - 2.0).max() < 0.02
    assert mesh.volume == pytest.approx(4 / 3 * np.pi * 8, rel=0.01)


def test_sheets_closer_than_a_voxel_stay_closed_and_manifold():
    # two plates 0.12 apart (under half a voxel), turned so the gap cuts through cells: where a row of
    # samples misses the gap the plates are joined (for marching cubes too), and cells hold a piece that
    # crosses a face twice, beside one that does too, whose two vertices would share four quads
    plate = dict(half=np.array([2.0, 2.0, 0.6]), turn=TURN)
    both = lambda p: np.minimum(box(p, centre=TURN @ [0, 0, -0.66], **plate), box(p, centre=TURN @ [0, 0, 0.66], **plate))
    values, grid = sample(both, (-3.5,) * 3, (3.5,) * 3, 0.3)
    info = {}
    mesh = dual_contour(values, grid, debug=info)
    assert info["tunnels"] > 0                                        # (the case is there)
    assert closed(mesh)                                               # every edge in exactly two triangles
    mc = field_to_mesh(values, grid)
    assert mesh.euler_number == mc.euler_number                       # the same shape topologically
    assert len(mesh.split(only_watertight=False)) == len(mc.split(only_watertight=False))


def test_a_crease_between_two_bodies_is_kept():
    # an L of two boxes: the inside corner where they meet is a concave crease
    a = dict(half=np.array([2.0, 0.5, 1.0]), turn=TURN)
    b = dict(half=np.array([0.5, 2.0, 1.0]), turn=TURN)
    ell = lambda p: np.minimum(box(p, centre=TURN @ [0, -1.5, 0], **a), box(p, centre=TURN @ [-1.5, 0, 0], **b))
    values, grid = sample(ell, (-4.0,) * 3, (4.0,) * 3, 0.3)
    dc, mc = dual_contour(values, grid), field_to_mesh(values, grid)
    assert closed(dc)
    crease = (np.array([-1.0, -1.0, 0.0]) + np.linspace(-0.8, 0.8, 30)[:, None] * [0, 0, 1]) @ TURN.T
    assert distance_to(crease, dc).max() < 0.1 * 0.3
    assert distance_to(crease, mc).mean() > 4 * distance_to(crease, dc).mean()


def test_a_dense_texture_stays_closed_in_a_float32_file(tmp_path):
    # ridges finer than the voxel fold triangles over; vertices put back into their cells once met a
    # neighbour's on their shared corner: one vertex in an STL, its edges shared by four triangles
    part = tessellate(load_step(Path(__file__).resolve().parents[1] / "samples" / "test_part.step")[0])
    field = F.offset(F.body_from_mesh(part, 1.5), F.acos(F.cos(F.noise(8, 3, 0.45, 3) * 140)) / np.pi * 0.6)
    grid = Grid.from_bounds(field.bounds, 0.4, 1.5)
    mesh = dual_contour(F.Context(grid).eval_full(field), grid)
    assert mesh.metadata["folded"] > 0                                  # (the case is there)
    mesh.export(tmp_path / "out.stl")
    assert trimesh.load(tmp_path / "out.stl").is_watertight


def test_mesh_from_implicit_meshes_by_the_chosen_method(monkeypatch):
    centre, size = np.array([0.07, 0.11, 0.13]), np.array([4.0, 2.8, 2.0])  # faces between the samples
    mesh = REGISTRY["mesh.from_implicit"].fn
    args = dict(body=F.box(centre, size), voxel=0.3, padding=1.0, domain=None, keep=1.0, tolerance=0.0, coarser=1.0)
    mc, dc = mesh(method="marching cubes", **args), mesh(method="dual contouring", **args)
    assert closed(dc)
    _, edges = box_features(half=size / 2, turn=np.eye(3))
    assert distance_to(edges + centre, mc).mean() > 4 * distance_to(edges + centre, dc).mean()
    with pytest.raises(GraphError, match="Method"):
        mesh(method="splines", **args)
    # a body meshed in blocks: dual contouring too
    monkeypatch.setattr(tiles, "BLOCK", 12)
    monkeypatch.setattr(tiles, "TILE_MIN_VOXELS", 1000)
    blocks = mesh(method="dual contouring", **args)
    assert closed(blocks) and distance_to(edges + centre, blocks).mean() < 2 * distance_to(edges + centre, dc).mean()
