"""Static stress analysis: how a part deforms under its loads, and the stresses in it.

Stresses are computed at element centres with the material's full stiffness (the stress in the
material, also in partly filled elements) and averaged to the nodes, weighted by how full the
elements are. Result fields interpolate those nodal values trilinearly: pure functions of position,
so slices, surface colours and block meshing treat them like any other field. Near sharp edges and
fillets voxel stresses are approximate, and the surface value comes from element centres half an
element inside.
"""
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..backend import to_host, xp_of
from ..fields import Field, _digest
from ..shared import Shared
from .element import CORNERS, elasticity, stiffness, strain_matrix, von_mises
from .model import (MAX_COARSE, MAX_COARSE_STATIC, ConditionError, active_nodes, apply_cases, element_fill,
                    element_size, estimate_volume, make_grid)
from .solver import Hierarchy, Level, NotConverged, _slabs, gather

FEA_VERSION = 2      # bump when the numerics change: results kept on the SSD are then ignored
SAFETY_CAP = 10.0    # safety factors above this are shown as this
SUPPORT_PEAK = 1.2   # a peak at a support this much above the stress elsewhere is called artificial
TOLERANCE = 1e-6     # relative residual a static analysis solves to
# Nonzeros a static analysis' factorization of its coarsest multigrid level may keep (Hierarchy's direct_fill):
# it takes as fine a level as this allows. On the CPU, where an iteration takes 1-3 s, DIRECT_FILL or
# DIRECT_FILL_PER_ELEMENT per box element, whichever is more (18M took 1.2 s and 0.2 GB; on 8.85M elements 44M
# saved 32 iterations of 2.7 s for 8 s). On the GPU an iteration takes milliseconds and every one solves the
# coarsest level on the host: 25M cost more than it saved (a strut web 2.7 -> 4.1 s, the bracket at medium
# 0.5 -> 2.2 s), one per box element less (2.7 -> 1.7, 0.5 -> 0.4) (FINDINGS, "The result's check").
DIRECT_FILL = 25_000_000
DIRECT_FILL_PER_ELEMENT = 5
DIRECT_FILL_PER_ELEMENT_GPU = 1
_RESULTS = OrderedDict()  # analysis key -> Result, so switching what is shown is instant
KEEP_RESULTS = 4


@dataclass
class Model:
    grid: object
    phi: np.ndarray       # (nx, ny, nz) how full each element is
    fixed: np.ndarray     # (X, Y, Z, 3) bool
    force: np.ndarray     # (X, Y, Z, 3) N: the first load case's (without Load Case nodes the only one)
    notes: list = field(default_factory=list)
    forces: np.ndarray = None   # (cases, X, Y, Z, 3) N: every load case's
    cases: list = None          # their names ([None] without Load Case nodes)

    def __post_init__(self):
        if self.forces is None:
            self.forces, self.cases = self.force[None], [None]


def _drop_loose(phi, fixed, force):
    """Pieces of the part that touch no support: an error if loaded, else left out (they would make
    the system singular)."""
    labels, count = ndimage.label(phi > 0, structure=np.ones((3, 3, 3), dtype=bool))
    if count <= 1:
        return phi, []
    nx, ny, nz = phi.shape
    node_label = np.zeros((nx + 1, ny + 1, nz + 1), dtype=labels.dtype)
    for cx, cy, cz in CORNERS:
        view = node_label[cx:cx + nx, cy:cy + ny, cz:cz + nz]
        np.maximum(view, labels, out=view)
    held = np.unique(node_label[fixed.any(axis=-1)])
    loaded = np.unique(node_label[np.abs(force).sum(axis=-1) > 0])
    if len(np.setdiff1d(loaded[loaded > 0], held)):
        raise ConditionError("a piece of the part carries a load but is not connected to any Fixed Faces")
    loose = np.setdiff1d(np.arange(1, count + 1), held)
    if not len(loose):
        return phi, []
    phi = phi.copy()
    phi[np.isin(labels, loose)] = 0.0
    return phi, [f"{len(loose)} loose piece{'s' if len(loose) > 1 else ''} without a support left out"]


def plan_grid(part, detail="medium", volume=None, h=None, bounds=None, max_coarse=MAX_COARSE):
    """The element grid for a part: the element size comes from `detail` and the part's volume
    (estimated if not given), or is `h`; `max_coarse`: elements on the coarsest multigrid level."""
    bounds = part.bounds if bounds is None else bounds
    if bounds is None:
        raise ConditionError("the part has no size: connect a CAD body or an implicit body made from one")
    if h is None:
        volume = estimate_volume(part, bounds) if volume is None else volume
        if volume <= 0:
            raise ConditionError("the part has no volume")
        h = element_size(bounds, volume, detail)
    return make_grid(bounds, h, max_coarse)


def build_model(part, conditions, detail="medium", xp=np, cache=None, volume=None, h=None, bounds=None, grid=None):
    """The FE model of an implicit body `part` with `conditions` (supports and loads)."""
    grid = grid or plan_grid(part, detail, volume, h, bounds, MAX_COARSE_STATIC)
    phi = element_fill(part, grid, xp, cache)
    if not (phi > 0).any():
        raise ConditionError("no material found in the part")
    fixed, forces, cases, notes = apply_cases(grid, phi, conditions)
    phi, more = _drop_loose(phi, fixed, np.abs(forces).sum(axis=0))   # (loaded in any case)
    return Model(grid, phi, fixed, forces[0], notes + more, forces, cases)


def solve(model, material, xp=np, dtype=np.float32, tol=TOLERANCE, callback=None):
    """Displacements (X, Y, Z, 3, float64) on xp's device; the multigrid works in `dtype`, conjugate
    gradients and the fine product in float64 (solver.py, MIXED): as accurate, and on the CPU too
    faster than all-float64 (bracket, medium: 25 -> 16 s). Returns (u, iterations, relative residual)
    of the first load case (`solve_cases`: all of them)."""
    (u,), its, res = solve_cases(model, material, xp, dtype, tol, callback, [0])
    return u, its, res


def solve_cases(model, material, xp=np, dtype=np.float32, tol=TOLERANCE, callback=None, cases=None, direct_fill=None):
    """`solve` for the load cases numbered in `cases` (default all) with one multigrid hierarchy:
    ([u], iterations in all, the largest relative residual). direct_fill: Hierarchy's (default: see DIRECT_FILL;
    0 = the grid's own levels)."""
    scale = material.E * model.grid.h * model.phi
    if direct_fill is None:
        direct_fill = (max(DIRECT_FILL, DIRECT_FILL_PER_ELEMENT * scale.size) if xp is np else
                       DIRECT_FILL_PER_ELEMENT_GPU * scale.size)
    for precision in (dtype, np.float64):   # (float32 rounding can stop a slender part's multigrid: see Optimizer._solve)
        H = Hierarchy(scale, stiffness(material.nu), model.fixed, model.grid.coarsenings, xp, precision,
                      direct_fill=direct_fill)
        us, its, res = [], 0, 0.0
        try:
            for k in (range(len(model.forces)) if cases is None else cases):
                u, n, r = H.solve(model.forces[k], tol=tol, callback=callback)
                us.append(u)
                its, res = its + n, max(res, float(r))
            return us, its, res
        except NotConverged:
            if precision == np.float64:
                raise


def element_stresses(model, material, u):
    """von Mises stress at every element centre (0 where empty), MPa, on the host."""
    xp = xp_of(u)
    grid = model.grid
    S = xp.asarray(elasticity(material.E, material.nu) @ strain_matrix() / grid.h, dtype=u.dtype)  # (6, 24)
    vm = xp.zeros(grid.shape, dtype=u.dtype)
    for x0, x1 in _slabs(grid.shape, 24 * 8 * u.itemsize, xp):
        U = gather(u, x0, x1, grid.shape).reshape(-1, 24)
        vm[x0:x1] = von_mises(U @ S.T).reshape(x1 - x0, grid.shape[1], grid.shape[2])
    return np.asarray(to_host(vm), dtype=np.float64) * (model.phi > 0)


def _near_supports(model, reach=2):
    """Elements touching a fixed node, and those within `reach` elements of them."""
    nx, ny, nz = model.phi.shape
    fixed = model.fixed.any(axis=-1)
    touch = np.zeros(model.phi.shape, dtype=bool)
    for cx, cy, cz in CORNERS:
        touch |= fixed[cx:cx + nx, cy:cy + ny, cz:cz + nz]
    return ndimage.binary_dilation(touch, structure=np.ones((3, 3, 3), dtype=bool), iterations=reach)


def _neighbour_fill(values, valid, passes=2):
    """Give nodes without a value the mean of their neighbours' (a margin outside the part, so
    interpolating right at the surface does not mix in zeros)."""
    values, valid = values.copy(), valid.copy()
    for _ in range(passes):
        known = values * valid
        total, count = np.zeros_like(values), np.zeros(values.shape, dtype=np.uint8)
        for axis in range(3):
            for src, dst in ((slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None))):
                s, d = [slice(None)] * 3, [slice(None)] * 3
                s[axis], d[axis] = src, dst
                total[tuple(d)] += known[tuple(s)]
                count[tuple(d)] += valid[tuple(s)]
        new = ~valid & (count > 0)
        values[new] = total[new] / count[new]
        valid |= new
    return values


def nodal_average(values, phi):
    """Element values averaged onto the nodes, weighted by how full the elements are."""
    nx, ny, nz = phi.shape
    num = np.zeros((nx + 1, ny + 1, nz + 1))
    den = np.zeros_like(num)
    weighted = phi * values
    for cx, cy, cz in CORNERS:
        num[cx:cx + nx, cy:cy + ny, cz:cz + nz] += weighted
        den[cx:cx + nx, cy:cy + ny, cz:cz + nz] += phi
    valid = den > 0
    return _neighbour_fill(np.where(valid, num / np.where(valid, den, 1), 0.0), valid)


def nodal_field(values, grid, key, name, unit, value_range, extent):
    """A Field interpolating nodal values (X, Y, Z) trilinearly (constant beyond the grid)."""
    shared = Shared(np.ascontiguousarray(values, dtype=np.float32), key=key)
    origin, h = np.asarray(grid.origin, dtype=np.float64), float(grid.h)

    def fn(ctx):
        xp = ctx.xp
        out = xp.asarray(shared.get())
        for axis, coords in enumerate(ctx.grid.axes()):
            n = out.shape[axis]
            pos = (coords - origin[axis]) / h
            lo = np.clip(np.floor(pos), 0, n - 2).astype(np.int64)
            w = np.clip(pos - lo, 0.0, 1.0).astype(np.float32).reshape([-1 if a == axis else 1 for a in range(3)])
            lo, w = xp.asarray(lo), xp.asarray(w)
            out = xp.take(out, lo, axis=axis) * (1 - w) + xp.take(out, lo + 1, axis=axis) * w
        return xp.ascontiguousarray(out, dtype=xp.float32)
    f = Field(fn, _digest("fea-field", key), name, range=value_range)
    f.extent = extent
    f.unit = unit
    return f


@dataclass
class Result:
    key: str
    grid: object
    report: dict
    fields: dict    # stress, safety, displacement

    def summary(self):
        r = self.report
        point = lambda p: "(" + ", ".join(f"{v:.1f}" for v in p) + ")"
        lines = [f"Static analysis ({r['detail']}, {r['grid']}, {r['elements']:,} with material, {r['device']}, "
                 f"{r['time']:.1f} s{', from the SSD cache' if r['cached'] else ''})",
                 f"max stress {r['max_away']:.3g} MPa at {point(r['max_away_at'])} · smallest safety factor "
                 f"{r['min_safety']:.2g} ({r['material']}, strength {r['strength']:g} MPa) · max deflection "
                 f"{r['max_displacement']:.3g} mm · mass {r['mass']:.3g} g"
                 + (f" (the worst of {len(r['cases'])} load cases)" if r.get("cases") else "")]
        for c in r.get("cases") or []:
            lines.append(f"{c['name']}: max stress {c['max_away']:.3g} MPa at {point(c['max_away_at'])} · safety factor "
                         f"{c['min_safety']:.2g} · max deflection {c['max_displacement']:.3g} mm")
        if r["peak_at_support"]:
            lines.append(f"note: {r['max_stress']:.3g} MPa right at the fixed faces {point(r['max_stress_at'])}: faces "
                         f"held perfectly rigid make an artificial peak at their edge (it grows with finer detail), "
                         f"so it is left out above and of the colour scale. Real bolts and clamps give a little.")
        if r.get("imbalance") is not None and r["imbalance"] > 1e-3:
            lines.append(f"warning: the support reactions differ from the loads by {100 * r['imbalance']:.1f}%")
        lines += [f"note: {n}" for n in r["notes"]]
        return "\n".join(lines)


def static_analysis(part, conditions, material, detail="medium", xp=np, cache=None, store=None, callback=None,
                    volume=None, h=None, extent=None, cancel=None):
    """Analyse an implicit body under `conditions`. `xp`: numpy, cupy, or a function giving one of
    them for a number of elements. `store` (a DiskCache) keeps displacements on the SSD;
    callback(iteration, residual) reports solver progress (and may raise to cancel); `cancel()` is
    called between the other steps (it may raise to stop)."""
    key = _digest("fea", FEA_VERSION, part.key, conditions.key, material.key, detail, h)
    if key in _RESULTS:
        _RESULTS.move_to_end(key)
        return _RESULTS[key]
    t0 = time.perf_counter()
    step = cancel if cancel is not None else (lambda: None)
    grid = plan_grid(part, detail, volume, h, max_coarse=MAX_COARSE_STATIC)
    if callable(xp):
        xp = xp(grid.size)
    step()
    model = build_model(part, conditions, xp=xp, cache=cache, grid=grid)
    step()
    dtype = np.float64 if xp is np else np.float32
    # displacements depend on the material only through E (as 1 / E) and Poisson's ratio: they are
    # kept for E = 1, so a new density, strength or E does not solve again (one array per load case)
    base = _digest("fea-u", FEA_VERSION, part.key, conditions.key, float(material.nu), detail, h)
    u_keys = [("fea-u", base if len(model.cases) == 1 else _digest(base, "case", k)) for k in range(len(model.cases))]
    us = [store.load(k) if store is not None else None for k in u_keys]
    us = [xp.asarray(u, dtype=dtype) / material.E if u is not None and u.shape == grid.nodes + (3,) else None for u in us]
    missing = [k for k, u in enumerate(us) if u is None]
    iterations = 0
    if missing:
        solved, iterations, _ = solve_cases(model, material, xp, callback=callback, cases=missing)
        for k, u in zip(missing, solved):
            us[k] = u
            if store is not None:
                store.save(u_keys[k], (to_host(u) * material.E).astype(np.float32))

    act = active_nodes(model.phi)
    near = _near_supports(model)
    level = Level(grid.shape, "fine", xp.asarray(material.E * grid.h * model.phi, dtype=dtype),
                  xp.asarray(stiffness(material.nu), dtype=dtype), xp)
    vm_e = size = None
    cases, imbalance = [], 0.0
    for name, u, force in zip(model.cases, us, model.forces):
        step()
        stress_k = element_stresses(model, material, u)
        # equilibrium check: what the supports push back with should balance the loads
        Ku = to_host(level.apply(u, masked=False))
        applied = force.reshape(-1, 3).sum(axis=0)
        reaction = np.array([np.asarray(Ku[..., d], dtype=np.float64)[model.fixed[..., d]].sum() for d in range(3)])
        # relative to the total force, or for loads adding up to (nearly) none (a torque) to their size
        scale = max(np.linalg.norm(applied), 0.1 * np.linalg.norm(force.reshape(-1, 3), axis=1).sum(), 1e-30)
        imbalance = max(imbalance, float(np.linalg.norm(reaction + applied) / scale))
        del Ku
        size_k = np.linalg.norm(np.asarray(to_host(u), dtype=np.float64), axis=-1)
        away_k = np.where(near, 0.0, stress_k)
        i_k = np.unravel_index(int(np.argmax(away_k)), away_k.shape)
        top = float(away_k[i_k])
        cases.append({"name": name, "max_away": top, "max_away_at": tuple(grid.origin + (np.asarray(i_k) + 0.5) * grid.h),
                      "min_safety": material.strength / max(top, 1e-9),
                      "max_displacement": float(np.where(act, size_k, 0.0).max())})
        # with load cases: the worst of them everywhere (what the part must stand)
        vm_e = stress_k if vm_e is None else np.maximum(vm_e, stress_k)
        size = size_k if size is None else np.maximum(size, size_k)
        del stress_k, size_k
    del level, us
    step()
    disp = _neighbour_fill(np.where(act, size, 0.0), act)
    stress = nodal_average(vm_e, model.phi)
    safety = np.minimum(material.strength / np.maximum(stress, 1e-9), SAFETY_CAP)

    i_max = np.unravel_index(int(np.argmax(vm_e)), vm_e.shape)
    max_stress = float(vm_e[i_max])
    # Faces held perfectly rigid make an artificial stress peak at their edge, which grows with finer
    # detail (bracket: 12.8, 14.0, 21.8 MPa at draft, medium, fine). Report the highest stress
    # away from the supports too, and use it for the safety factor and the colour scale.
    away = np.where(near, 0.0, vm_e)
    i_away = np.unravel_index(int(np.argmax(away)), away.shape)
    max_away = float(away[i_away]) if (~near & (model.phi > 0)).any() else max_stress
    n_disp = np.where(act, size, 0.0)
    elements = int((model.phi > 0).sum())
    partial = int(((model.phi > 0) & (model.phi < 0.99)).sum())
    notes = list(model.notes)
    if elements and partial > 0.5 * elements:
        notes.append(f"the part is thin compared with the {grid.h:g} mm elements: results are rough, use a finer detail")
    report = {
        "detail": detail, "grid": grid.describe(), "elements": elements, "h": grid.h,
        "device": "GPU" if xp is not np else "CPU", "iterations": iterations, "cached": not missing,
        "time": time.perf_counter() - t0, "material": material.name, "strength": material.strength,
        "max_stress": max_stress, "max_stress_at": tuple(grid.origin + (np.asarray(i_max) + 0.5) * grid.h),
        "max_away": max_away, "max_away_at": tuple(grid.origin + (np.asarray(i_away) + 0.5) * grid.h),
        "peak_at_support": bool(near[i_max]) and max_stress > SUPPORT_PEAK * max_away,
        "min_safety": material.strength / max(max_away, 1e-9), "max_displacement": float(n_disp.max()),
        "mass": float(model.phi.sum() * grid.h ** 3 * material.density / 1000.0),
        "load": tuple(model.forces[0].reshape(-1, 3).sum(axis=0)), "imbalance": imbalance, "notes": notes,
        # per load case, when there are Load Case nodes (the rest of the report: the worst of them)
        "cases": cases if len(cases) > 1 else [],
    }
    extent = extent if extent is not None else part.bounds
    stress_max = float(stress[act].max()) if act.any() else 0.0
    fields = {
        "stress": nodal_field(stress, grid, key + "-vm", "von Mises stress", "MPa", (0.0, stress_max), extent),
        "safety": nodal_field(safety, grid, key + "-sf", "safety factor", "", (float(safety[act].min()), SAFETY_CAP),
                              extent),
        "displacement": nodal_field(disp, grid, key + "-u", "displacement", "mm", (0.0, float(n_disp.max())), extent),
    }
    if report["peak_at_support"]:  # a colour scale that the peak at a support does not wash out
        fields["stress"].display_range = (0.0, max_away)
    # the safety factor on a fixed scale, from 1 (or lower, where the part fails away from the supports)
    # up to the cap: on its own range a part safe everywhere showed 10..10 in the "bad" colour, and a
    # peak at a support turned the range backwards (17..10)
    fields["safety"].display_range = (min(1.0, report["min_safety"]), SAFETY_CAP)
    result = Result(key, grid, report, fields)
    _RESULTS[key] = result
    while len(_RESULTS) > KEEP_RESULTS:
        _RESULTS.popitem(last=False)
    return result
