"""Planar slices of scalar fields, for looking at noise, waves and other fields."""
import numpy as np

from .fields import Context, Field, display_extent
from .grid import Grid

AXES = "xyz"


def slice_plane(axis, position, lo, hi, res):
    """Grid lying in the plane `axis` = `position`, covering lo..hi on the other two axes with
    about `res` samples along the longer side. Returns (grid, (u, v)) with u, v the in-plane axes."""
    a = AXES.index(axis)
    u, v = [i for i in range(3) if i != a]
    lo, hi = np.asarray(lo, dtype=np.float64).copy(), np.asarray(hi, dtype=np.float64).copy()
    extent = max(hi[u] - lo[u], hi[v] - lo[v], 1e-6)
    lo[a] = hi[a] = position
    return Grid(lo, hi, extent / max(int(res), 2)), (u, v)


def field_slice(field, axis, position, lo, hi, res=512):
    """Values of `field` on a plane. Returns (image, rect): image[row, col] with columns along the
    first in-plane axis and rows along the second (x/y for a z slice, x/z for y, y/z for x);
    rect = (u0, v0, u1, v1) is the area covered, in mm. Pixels a field does not cover are NaN.

    A list of fields (one per body) is composited: each body's field is drawn inside its own box."""
    grid, (u, v) = slice_plane(axis, position, lo, hi, res)
    nu, nv = grid.shape[u], grid.shape[v]
    image = np.full((nv, nu), np.nan, dtype=np.float32)
    items = field if isinstance(field, list) else [field]
    for item in items:
        if not isinstance(item, Field):
            image[:] = float(item)
            continue
        sub, (i0, j0) = grid, (0, 0)
        box = display_extent(item)
        if box is not None and len(items) > 1:
            blo, bhi = box
            a = AXES.index(axis)
            if not blo[a] - 1 <= position <= bhi[a] + 1:
                continue
            i0 = max(0, int(np.floor((blo[u] - 1 - grid.origin[u]) / grid.voxel)))
            j0 = max(0, int(np.floor((blo[v] - 1 - grid.origin[v]) / grid.voxel)))
            i1 = min(nu, int(np.ceil((bhi[u] + 1 - grid.origin[u]) / grid.voxel)) + 1)
            j1 = min(nv, int(np.ceil((bhi[v] + 1 - grid.origin[v]) / grid.voxel)) + 1)
            if i1 <= i0 or j1 <= j0:
                continue
            sub_lo = grid.origin.copy()
            sub_lo[u] += i0 * grid.voxel
            sub_lo[v] += j0 * grid.voxel
            sub_hi = sub_lo.copy()
            sub_hi[u] += (i1 - i0 - 1) * grid.voxel
            sub_hi[v] += (j1 - j0 - 1) * grid.voxel
            sub = Grid(sub_lo, sub_hi, grid.voxel)
        values = Context(sub).eval_full(item)
        plane = np.moveaxis(values, (u, v), (1, 0)).reshape(values.shape[v], values.shape[u])
        h, w = plane.shape
        target = image[j0:j0 + h, i0:i0 + w]
        target[...] = np.where(np.isnan(target), plane[:target.shape[0], :target.shape[1]], target)
        if len(items) > 1 and box is None:
            break  # identical copies of a body-independent field: one is enough
    covered_hi = grid.origin + grid.voxel * (np.asarray(grid.shape) - 1)
    rect = (float(grid.origin[u]), float(grid.origin[v]), float(covered_hi[u]), float(covered_hi[v]))
    return image, rect
