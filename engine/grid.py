"""Regular voxel grid that every field is evaluated on."""
import numpy as np


def upsample_linear(coarse, factor, shape):
    """Trilinear upsampling of values on a grid `factor` times coarser (same origin) to `shape`.
    Done one axis at a time with plain array ops, which is much faster than map_coordinates.
    Works on numpy or CuPy arrays (the result stays on the same device)."""
    from .backend import xp_of
    xp = xp_of(coarse)
    out = coarse
    for axis in range(3):
        pos = np.arange(shape[axis], dtype=np.float64) / factor
        lo = np.minimum(pos.astype(np.int64), out.shape[axis] - 1)
        hi = np.minimum(lo + 1, out.shape[axis] - 1)
        w = xp.asarray((pos - lo).astype(np.float32).reshape([-1 if a == axis else 1 for a in range(3)]))
        lo, hi = xp.asarray(lo), xp.asarray(hi)
        out = xp.take(out, lo, axis=axis) * (1 - w) + xp.take(out, hi, axis=axis) * w
    return xp.ascontiguousarray(out, dtype=xp.float32)


class Grid:
    """Sample (i, j, k) sits at base + (start + (i, j, k)) * voxel. A plain grid has base = origin
    and start = 0; a block of a bigger grid (see `block`) shares the big grid's base, so every
    block computes a shared sample's position the same way, bit for bit."""

    def __init__(self, bbox_min, bbox_max, voxel_size):
        self.voxel = float(voxel_size)
        self.origin = np.asarray(bbox_min, dtype=np.float64)
        extent = np.asarray(bbox_max, dtype=np.float64) - self.origin
        self.shape = tuple(int(np.ceil(e / self.voxel)) + 1 for e in extent)
        self.base = self.origin
        self.start = (0, 0, 0)

    @classmethod
    def block(cls, base, voxel, start, shape):
        """The part of a bigger grid (origin `base`) beginning at integer sample index `start`."""
        g = cls.__new__(cls)
        g.voxel = float(voxel)
        g.base = np.asarray(base, dtype=np.float64)
        g.start = tuple(int(s) for s in start)
        g.shape = tuple(int(n) for n in shape)
        g.origin = g.base + np.asarray(g.start, dtype=np.float64) * g.voxel
        return g

    def coarser(self, factor):
        """A grid with `factor` times larger voxels whose nodes coincide with every factor-th
        sample of this one, covering it plus one node past the end (for interpolation)."""
        if any(s % factor for s in self.start):
            raise ValueError(f"block start {self.start} is not a multiple of {factor}")
        shape = [(n - 1 + factor - 1) // factor + 2 for n in self.shape]
        return Grid.block(self.base, self.voxel * factor, [s // factor for s in self.start], shape)

    @classmethod
    def around(cls, mesh, voxel_size, padding):
        """Grid covering a mesh's bounding box plus `padding` mm on every side."""
        return cls.from_bounds(mesh.bounds, voxel_size, padding)

    @classmethod
    def from_bounds(cls, bounds, voxel_size, padding):
        lo, hi = np.asarray(bounds, dtype=np.float64)
        pad = padding + 2 * voxel_size
        return cls(lo - pad, hi + pad, voxel_size)

    @property
    def key(self):
        return (tuple(np.round(self.origin, 6)), self.voxel, self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def axes(self):
        return [self.base[i] + self.voxel * (self.start[i] + np.arange(n)) for i, n in enumerate(self.shape)]

    def coords(self):
        """X, Y, Z coordinate arrays (float32, shape == self.shape)."""
        return np.meshgrid(*[a.astype(np.float32) for a in self.axes()], indexing="ij")

    def points(self, mask=None):
        """(N, 3) float64 world points, optionally only where `mask` is True."""
        idx = np.argwhere(mask) if mask is not None else np.indices(self.shape).reshape(3, -1).T
        return self.base + (np.asarray(self.start) + idx) * self.voxel

    def describe(self):
        mb = self.size * 4 / 1e6
        return f"{self.shape[0]}x{self.shape[1]}x{self.shape[2]} voxels @ {self.voxel} mm (~{mb:.0f} MB per field)"
