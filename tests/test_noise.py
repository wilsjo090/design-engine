import numpy as np
import pytest

from engine import fields as F
from engine.grid import Grid
from engine.noise import fractal, fractal_on_grid, perlin, wave3d_on_grid


def test_perlin_is_smooth_bounded_and_deterministic():
    x = np.linspace(0, 20, 20001, dtype=np.float32)
    zeros = np.zeros_like(x)
    a = perlin(x, zeros + 0.3, zeros + 0.7, seed=1)
    assert np.array_equal(a, perlin(x, zeros + 0.3, zeros + 0.7, seed=1))
    assert not np.array_equal(a, perlin(x, zeros + 0.3, zeros + 0.7, seed=2))
    assert np.abs(a).max() <= 1.0
    assert np.abs(np.diff(a)).max() < 0.01        # no jumps between neighbouring points
    assert a.std() > 0.1                          # but it does vary


def test_grid_noise_matches_pointwise_and_slabbing():
    grid = Grid((0, 0, 0), (30, 20, 10), 0.25)    # >2M points would slab; also check small
    full = fractal_on_grid(grid, scale=6, octaves=3, seed=4)
    pts = grid.points()[::997]
    direct = fractal(*pts.T.astype(np.float32), scale=6, octaves=3, seed=4)
    idx = np.unravel_index(np.arange(grid.size)[::997], grid.shape)
    assert np.allclose(full[idx], direct, atol=1e-5)


@pytest.mark.parametrize("pattern", ["gyroid", "schwarz", "diamond"])
def test_wave3d_range_and_period(pattern):
    grid = Grid((0, 0, 0), (10, 10, 10), 0.25)
    w = wave3d_on_grid(grid, pattern, period=5.0)
    assert -1.05 < w.min() < -0.6 and 0.6 < w.max() < 1.05
    # one period along x repeats exactly (20 voxels = 5 mm)
    assert np.allclose(w[:20], w[20:40], atol=1e-4)


def test_ridges_are_the_original_triangle_wave():
    grid = Grid((0, 0, 0), (10, 10, 10), 0.25)
    w = wave3d_on_grid(grid, "ridges", period=3.0, axis="y")
    y = grid.axes()[1].astype(np.float32)
    expected = np.arccos(np.cos(2 * np.pi / 3.0 * y)) * (2 / np.pi) - 1
    assert np.allclose(w[3, :, 5], expected, atol=1e-5)
    assert np.allclose(w[0], w[17])               # constant across x and z without warp
    bent = wave3d_on_grid(grid, "ridges", period=3.0, axis="y", warp=1.0, seed=2)
    assert not np.allclose(bent[0], bent[17], atol=0.05)  # with warp the ridges bend in x


def test_warp_breaks_the_repetition_but_stays_smooth():
    grid = Grid((0, 0, 0), (20, 20, 20), 0.25)
    w = wave3d_on_grid(grid, "gyroid", period=5.0, warp=1.5, warp_scale=8.0, seed=3)
    assert not np.allclose(w[:20], w[20:40], atol=0.05)
    assert np.abs(np.diff(w, axis=0)).max() < 0.5


def test_fields_cache_keys_depend_on_parameters():
    assert F.noise(10, 3, 0.5, 1).key == F.noise(10, 3, 0.5, 1).key
    assert F.noise(10, 3, 0.5, 1).key != F.noise(10, 3, 0.5, 2).key
    assert F.wave3d("gyroid", 5, 1, 8, 0).key != F.wave3d("gyroid", 5, 2, 8, 0).key
    with pytest.raises(ValueError):
        F.wave3d("nope")
