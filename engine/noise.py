"""3D gradient noise (Ken Perlin's improved noise) and fractal noise, vectorised.

Runs on numpy or CuPy arrays (pass `xp`, or it follows the input arrays). Deterministic for a
given seed, so every worker process (and the GPU) produces the same pattern.
"""
import numpy as np

from .backend import xp_of
from .grid import upsample_linear

_GRADS = np.array([[1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
                   [1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1],
                   [0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1]], dtype=np.float32)
_CHUNK = 2_000_000  # points per slab on the CPU: bounds the temporaries on big grids
_GPU_CHUNK = 16_000_000
_TABLES = {}


def _tables(seed, xp):
    """Permutation table and gradient components for a seed, on xp's device."""
    key = (int(seed), xp.__name__)
    if key not in _TABLES:
        p = np.random.default_rng(int(seed)).permutation(256)
        perm = np.concatenate([p, p]).astype(np.int32)
        _TABLES[key] = (xp.asarray(perm), *(xp.asarray(g) for g in _GRADS))
    return _TABLES[key]


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(x, y, z, seed=0):
    """Gradient noise at the given coordinates (same-shape arrays), roughly in -1..1."""
    xp = xp_of(x)
    perm, gx, gy, gz = _tables(seed, xp)
    fx, fy, fz = xp.floor(x), xp.floor(y), xp.floor(z)
    xf, yf, zf = (x - fx).astype(xp.float32), (y - fy).astype(xp.float32), (z - fz).astype(xp.float32)
    xi, yi, zi = fx.astype(xp.int32) & 255, fy.astype(xp.int32) & 255, fz.astype(xp.int32) & 255
    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    # Perlin's hashing: 14 table lookups for the 8 corners of the cell
    a, b = perm[xi] + yi, perm[xi + 1] + yi
    aa, ab, ba, bb = perm[a] + zi, perm[a + 1] + zi, perm[b] + zi, perm[b + 1] + zi

    def grad(h, dx, dy, dz):
        h = perm[h] % 12
        return gx[h] * (xf - dx) + gy[h] * (yf - dy) + gz[h] * (zf - dz)

    c000, c100, c010, c110 = grad(aa, 0, 0, 0), grad(ba, 1, 0, 0), grad(ab, 0, 1, 0), grad(bb, 1, 1, 0)
    c001, c101, c011, c111 = grad(aa + 1, 0, 0, 1), grad(ba + 1, 1, 0, 1), grad(ab + 1, 0, 1, 1), grad(bb + 1, 1, 1, 1)
    x00 = c000 + u * (c100 - c000)
    x10 = c010 + u * (c110 - c010)
    x01 = c001 + u * (c101 - c001)
    x11 = c011 + u * (c111 - c011)
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return (y0 + w * (y1 - y0)).astype(xp.float32)


def fractal(x, y, z, scale, octaves=3, roughness=0.5, seed=0):
    """Fractal (fBm) noise: `octaves` layers of perlin noise, each twice as fine and `roughness`
    times as strong as the last. `scale` is the size of the largest features in mm."""
    xp = xp_of(x)
    total = xp.zeros(xp.broadcast(x, y, z).shape, dtype=xp.float32)
    amp, freq, norm = 1.0, 1.0 / float(scale), 0.0
    for octave in range(int(octaves)):
        shift = 37.1 * octave  # decorrelate the octaves
        total += amp * perlin(x * freq + shift, y * freq - shift, z * freq + 2 * shift, seed + octave)
        norm += amp
        amp *= roughness
        freq *= 2.0
    return total / max(norm, 1e-9)


def _slabs(grid, xp=np):
    """Yield (index slice, X, Y, Z) blocks of full coordinate arrays covering the grid."""
    ax, ay, az = [xp.asarray(a.astype(np.float32)) for a in grid.axes()]
    chunk = _CHUNK if xp is np else _GPU_CHUNK
    slab = max(1, chunk // (grid.shape[1] * grid.shape[2]))
    for i in range(0, grid.shape[0], slab):
        X, Y, Z = xp.broadcast_arrays(ax[i:i + slab, None, None], ay[None, :, None], az[None, None, :])
        yield slice(i, i + slab), X, Y, Z


def fractal_on_grid(grid, scale, octaves=3, roughness=0.5, seed=0, xp=np):
    """Fractal noise for every voxel of `grid`.

    The noise is smooth (its finest layer has features of scale / 2^(octaves-1)), so on fine grids
    it is computed on a coarser grid and interpolated: with ~48 samples per finest feature the
    error is ~1e-4, small enough even when the noise drives a dense wave."""
    finest = float(scale) / 2 ** max(int(octaves) - 1, 0)
    factor = int(min(4 * grid.voxel, finest / 48) // grid.voxel)
    target = grid.coarser(factor) if factor >= 2 else grid
    out = xp.empty(target.shape, dtype=xp.float32)
    for s, X, Y, Z in _slabs(target, xp):
        out[s] = fractal(X, Y, Z, scale, octaves, roughness, seed)
    return upsample_linear(out, factor, grid.shape) if target is not grid else out


# Patterns, each scaled to -1..1. k = 2*pi / period. "ridges" is the triangle wave acos(cos(k*axis))
# (the original nTop wave) along one axis; the others are triply periodic. (np.sin etc. are ufuncs:
# on CuPy arrays they run on the GPU.)
PATTERNS = {
    "ridges": None,  # handled in wave3d_on_grid: needs the axis
    "gyroid": lambda x, y, z: (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)) / 1.5,
    "schwarz": lambda x, y, z: (np.cos(x) + np.cos(y) + np.cos(z)) / 3.0,
    "diamond": lambda x, y, z: (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
                                + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z)) / 1.42,
}


def wave3d_on_grid(grid, pattern, period, warp=0.0, warp_scale=8.0, seed=0, axis="y", xp=np):
    """A wave pattern with a `period` mm repeat: "ridges" (triangle wave along `axis`) or a
    triply periodic one (gyroid, schwarz, diamond). With `warp` > 0 the sample points are pushed
    around in 3D by up to ~`warp` mm of smooth noise (blobs about `warp_scale` mm), so the ridges
    bend and meander and never repeat exactly."""
    k = 2 * np.pi / float(period)
    out = xp.empty(grid.shape, dtype=xp.float32)
    for s, X, Y, Z in _slabs(grid, xp):
        if warp:
            # three independent noise fields displace x, y and z
            X, Y, Z = (X + warp * fractal(X, Y, Z, warp_scale, 2, 0.5, seed),
                       Y + warp * fractal(X, Y, Z, warp_scale, 2, 0.5, seed + 101),
                       Z + warp * fractal(X, Y, Z, warp_scale, 2, 0.5, seed + 202))
        if pattern == "ridges":
            c = {"x": X, "y": Y, "z": Z}[axis]
            out[s] = xp.arccos(xp.cos(k * c)) * (2 / np.pi) - 1
        else:
            out[s] = PATTERNS[pattern](k * X, k * Y, k * Z)
    return out
