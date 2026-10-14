"""A field's values on the surface of displayed bodies (to colour them by the field)."""
import numpy as np
from scipy.ndimage import map_coordinates

from .backend import to_host
from .fields import Context, Field, display_extent
from .grid import Grid


def field_at_points(field, points, cells, cache=None, array_module=None, max_voxels=None):
    """Values of `field` at `points` (n, 3), as float32. Sampled on a grid with about `cells`
    samples along the longest side of the points' box and interpolated (trilinear): fields are
    smooth at that scale, and a grid is the only thing fields can be evaluated on. A list of
    fields (one per body) covers the points inside each body's own box, as slices do; points no
    field covers are NaN. `array_module(voxels)` picks numpy or cupy for a grid size; `max_voxels`
    makes the grid coarser if it would be bigger."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.full(len(pts), np.nan, dtype=np.float32)
    items = field if isinstance(field, list) else [field]
    for item in items:
        todo = np.isnan(out)
        if not todo.any():
            break
        if not isinstance(item, Field):
            out[todo] = float(item)
            continue
        inside = todo
        box = display_extent(item)
        if box is not None and len(items) > 1:
            lo, hi = box
            inside = todo & np.all((pts >= lo - 1.0) & (pts <= hi + 1.0), axis=1)
        if not inside.any():
            continue
        sel = pts[inside]
        lo, hi = sel.min(axis=0), sel.max(axis=0)
        voxel = max(float((hi - lo).max()) / max(int(cells), 2), 1e-3)
        if max_voxels:
            voxel = max(voxel, (float(np.prod(hi - lo + 2 * voxel)) / max_voxels) ** (1 / 3))
        grid = Grid(lo - voxel, hi + voxel, voxel)
        xp = array_module(grid.size) if array_module is not None else np
        values = to_host(Context(grid, cache, xp).eval_full(item))
        values = np.broadcast_to(np.asarray(values, dtype=np.float32), grid.shape)
        out[inside] = map_coordinates(values, ((sel - grid.origin) / voxel).T, order=1, mode="nearest")
        if len(items) > 1 and box is None:
            break  # identical copies of a body-independent field: that one covered every point
    return out
