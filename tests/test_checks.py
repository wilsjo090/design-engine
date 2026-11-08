"""Measure's member section (`checks.member_section`) on shapes whose sections are known."""
import math

import numpy as np
import pytest
import trimesh

from engine import checks


def _rotated(mesh, *turns):
    m = mesh.copy()
    for angle, axis in turns:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(angle), axis))
    return m


def _side_point(mesh, outward):
    """(point, the normal of a triangle it lies on): the first hit coming in along -outward towards the middle."""
    outward = np.asarray(outward, dtype=np.float64)
    start = mesh.bounds.mean(axis=0) + 1000.0 * outward
    t, _ = checks.ray_first_hit(mesh.vertices, mesh.faces, start[None], -outward[None], 2000.0)
    point = start - float(t[0]) * outward
    a, b, c = (mesh.vertices[mesh.faces[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # a triangle whose plane holds the point and whose edges all have it on their inner side
    on = np.abs(np.einsum("ij,ij->i", point - a, n)) < 1e-4
    for p, q in ((a, b), (b, c), (c, a)):
        on &= np.einsum("ij,ij->i", np.cross(q - p, point - p), n) >= -1e-6
    return point, n[np.flatnonzero(on)[0]]


def _section(mesh, outward=(1.0, 0.0, 0.0)):
    point, normal = _side_point(mesh, outward)
    return checks.member_section(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32),
                                 point, normal)


def _polygon_area(r, sides):
    return 0.5 * sides * r * r * math.sin(2 * math.pi / sides)


def test_round_bar():
    bar = trimesh.creation.cylinder(radius=2.0, height=30.0, sections=64)
    got = _section(bar)
    assert got["area"] == pytest.approx(_polygon_area(2.0, 64), rel=1e-3)
    assert got["diameter"] == pytest.approx(4.0, rel=2e-3)
    assert got["narrowest"] == pytest.approx(4.0, rel=5e-3) and got["widest"] == pytest.approx(4.0, rel=5e-3)
    assert abs(got["normal"][2]) == pytest.approx(1.0, abs=1e-3)   # cut square to the bar
    assert got["thickness"] == pytest.approx(4.0, rel=3e-3)        # (straight in from a facet: off the axis a little)
    assert len(got["outline"]) == 128 and got["holes"] == []   # (each side face is two triangles)


def test_leaning_bar_and_unwelded_mesh():
    turns = ((35, [1, 0, 0]), (20, [0, 0, 1]))
    bar = _rotated(trimesh.creation.cylinder(radius=1.5, height=40.0, sections=48), *turns)
    axis = _rotated(trimesh.Trimesh([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [], process=False), *turns).vertices
    axis = axis[1] - axis[0]
    out = np.cross(axis, [0.0, 0.0, 1.0])
    got = _section(bar, out / np.linalg.norm(out))
    assert got["area"] == pytest.approx(_polygon_area(1.5, 48), rel=1e-4)
    assert abs(np.dot(got["normal"], axis)) == pytest.approx(1.0, abs=1e-4)
    soup = trimesh.Trimesh(bar.vertices[bar.faces].reshape(-1, 3), np.arange(3 * len(bar.faces)).reshape(-1, 3),
                           process=False)
    again = _section(soup, out / np.linalg.norm(out))
    assert again["area"] == pytest.approx(got["area"], rel=1e-6)


def test_tube_takes_off_its_hole():
    tube = trimesh.creation.annulus(r_min=1.0, r_max=2.0, height=30.0, sections=64)
    got = _section(tube)
    assert got["area"] == pytest.approx(_polygon_area(2.0, 64) - _polygon_area(1.0, 64), rel=2e-3)
    assert len(got["holes"]) == 1


def test_flat_bar():
    bar = trimesh.creation.box(extents=(2.0, 6.0, 40.0))
    got = _section(bar)   # on the wide face: 2 mm deep
    assert got["thickness"] == pytest.approx(2.0, rel=1e-4)
    assert got["area"] == pytest.approx(12.0, rel=1e-6)
    assert got["narrowest"] == pytest.approx(2.0, rel=1e-3) and got["widest"] == pytest.approx(math.hypot(2, 6), rel=1e-3)


def test_plate_is_open_and_no_wall_is_none():
    plate = trimesh.creation.box(extents=(200.0, 200.0, 2.0))
    got = _section(plate, (0.0, 0.0, 1.0))
    assert got["open"] and got["thickness"] == pytest.approx(2.0)
    point, normal = _side_point(plate, (0.0, 0.0, 1.0))
    assert checks.member_section(np.asarray(plate.vertices, np.float32), np.asarray(plate.faces, np.uint32),
                                 point, -normal) is None
