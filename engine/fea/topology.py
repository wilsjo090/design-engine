"""Topology optimization: where to keep material so a part is as stiff as possible for its mass.

Density method (SIMP) on the FE grid of the static analysis: every element of the design space
gets a density x (0..1). Each iteration:
  x -> density filter (cone of radius `min thickness`)
    -> Heaviside projection (pushes densities to 0 or 1; sharper every BETA_EVERY iterations) at 0.5:
       the design, and at ERODED: the eroded design, in which members thinner than the filter
       radius vanish
    -> fixed regions (solid skin at supports and loads, Keep / Avoid bodies)
    -> stiffness of the eroded design E = Emin + rho^3 (E - Emin) -> solve -> compliance (how much
       the loads move): a member too thin to be in the eroded design costs mass and gives nothing
    -> sensitivities back through projection and filter (plus an outline penalty with Smoothness, a
       reward in a Prefer guide body; averaged with their mirror images with Symmetry) -> optimality-
       criteria update with the
       design's volume (Keep %) held exactly by bisection.
The state (x, the iteration, the projection sharpness, the last displacement) is saved every few
iterations, so a stopped run continues where it stopped and the result survives restarts.
See docs/TOPOLOGY_PLAN.md and docs/FINDINGS.md.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..backend import gaussian_filter, to_host, xp_of
from ..fields import Context, _digest
from .analysis import _drop_loose, plan_grid
from .element import stiffness
from .model import ConditionError, apply_conditions, element_fill, make_grid, sample_triangles
from .solver import Hierarchy, NotConverged

TOPO_VERSION = 2      # bump when the method changes: saved results are then ignored
PENAL = 3.0           # SIMP penalty: intermediate densities give little stiffness for their mass
EMIN = 1e-6           # stiffness of empty elements, relative (keeps the system solvable)
ETA = 0.5             # projection threshold
# Min thickness: the filter and a projection at 0.5 alone do not stop thin members (a ridge of the
# filtered density just over 0.5 is one): with a 3-element radius a key box came out with members
# of 1-2 elements. The stiffness is therefore that of the design projected at ERODED; with the cone
# filter only plates at least the radius thick (struts 1.3 x) reach it. Its filter counts the design
# space's faces as empty (`filter_empty`), so plates lying on them need that thickness too (no stiffer
# for it on the brackets; see FINDINGS.md, "Min thickness" and "The optimizers check their own results").
ERODED = 0.8
# What erosion removes still carries THIN_SHARE of its density (THIN_SHARE^3 = 0.3% of its stiffness):
# the eroded design alone lost its load path when Keep % left no room for members that thick (a 30 x
# 10 x 10 mm box at 20% fell apart), and thin members still cost far more than they give.
THIN_SHARE = 0.15
BETA_START, BETA_MAX, BETA_EVERY = 1.0, 16.0, 25
MOVE = 0.2            # largest change of a density per iteration (0.1 once the projection is sharp)
MAX_ITERATIONS = 125
CHANGE_TOL = 0.01     # done when no density changes more than this (at the sharpest projection) ...
PLATEAU = 1e-3        # ... or the compliance changed less than this over the last PLATEAU_WINDOW
PLATEAU_WINDOW = 5    # iterations (a few elements can flip back and forth for ever: 3D cantilever)
# Inexact solves, warm-started from the last one: the 3D cantilever came out the same (compliance
# 242.1) with 1e-5, 1e-3 and 1e-2, at 2.8, 1.6 and 1.1 s per CPU iteration.
SOLVE_TOL = 1e-2
# The multigrid smoothers' lmax estimates start from the last iteration's vectors after this many
# iterations (solver.WARM_POWER_ITERATIONS); in the first ones the densities change too much (warm
# estimates from iteration 1 on came out too low and CG blew up). A warm solve that does not
# converge in WARM_MAXITER iterations (~3 are usual) is done again with fresh estimates.
WARM_LMAX_AFTER = 4
WARM_MAXITER = 100
MIN_MEMBER = 3.0      # thinnest member (= filter radius), elements, when Min thickness is 0 or less
FFT_TAPS = 1000       # filter kernels with this many entries or more are applied by FFT (from radius 5)
# Max thickness: a local volume constraint (as Wu et al. 2018, bone-like infill): no neighbourhood
# may be more than LOCAL_SHARE full, so thick members split into more, thinner ones. Gaussian
# neighbourhoods (separable, so as cheap at any size): at the middle of a plate of thickness t the
# local density is erf(t / (2 sqrt(2) sigma)), which is LOCAL_SHARE (0.6) for t = 1.68 sigma.
LOCAL_SHARE = 0.6
LOCAL_SIGMA = 1 / 1.68     # sigma per mm of max thickness
LOCAL_P = 16               # p-mean of the local densities, scaled to their maximum (adaptively)
# "struts only": plates (the top flange of a bracket) pass a thickness limit however wide they are.
# Measuring Max thickness on a round strut instead (a strut of diameter d fills
# 1 - exp(-d^2 / 8 sigma^2) at its axis: LOCAL_SHARE for d = 2.71 sigma) allows plates of at most
# 2 sqrt(2) erfinv(0.6) / 2.71 = 0.62 of it; with members at least STRUT_MIN of it no plate fits.
# Max thickness may leave no room for Keep %: the volume's multiplier (in units of the mean sensitivities)
# then went down to 1e-9, material was free, and elements that carry nothing (1e-7 of the mean) grew
# wherever no neighbourhood was full: pieces cut off from the part stayed and grew (bolted bracket with
# Smoothness 1: 10 loose pieces). Material is never cheaper than this.
PRICE_FLOOR = 1e-3
# Prefer: a guide body where material is wanted (strength > 0) or not (< 0), soft: the design's material
# in it, fading out over GUIDE_FADE x the thinnest member, is rewarded in the sensitivities (strength x
# the mean stiffness sensitivity), and the optimizer shapes and smooths it as everything else (no union).
# Wall bracket, a 5 mm ball in its open triangle: 0.5 filled it with a web at no cost (27.2% against
# 27.0%), 1 grew a hub with branches (26.4%). See FINDINGS.md, "Soft guides".
GUIDE_FADE = 2.0
# Start coarse: fine runs start on the coarser details and move on (the design variables
# moved trilinearly) when the projection sharpness reaches HAND_OVER[level], so a first shape shows within
# seconds and most iterations are cheap. Wall bracket, fine: 58 s instead of 110, 29.0% stiffness instead
# of 28.3%; bolted bracket (Max 4) from medium: 118 s instead of 180, 20.2% instead of 21.2%. A level too
# coarse for the Max thickness is skipped: raised there (draft: 4 -> 5.9 mm), its few thick members were
# later thinned instead of split (12.7%). See FINDINGS.md, "Coarse-to-fine".
LEVELS = ("draft", "medium", "fine")
HAND_OVER = {"draft": 4.0, "medium": 8.0}
MEMBERS = ("plates and struts", "struts only")
STRUT_SIGMA = 1 / 2.707
STRUT_MIN = 0.65
# Smoothness: the design minimizes compliance x outline^smoothness, the outline being the projected
# design's (sum of |grad|, about its surface area), so fewer, rounder members and smoother faces. The
# filtered densities' outline (tried first) did not shrink the real surface (key box: 261 -> 264 cm2
# at 1, worse beyond): it rewards thin, grey members. Off by default: the commercial tools smooth
# afterwards instead (see FINDINGS.md, "Smoothness while optimizing").
OUTLINE_EPS = 1e-2         # |grad| is sqrt(... + eps^2): differentiable where the densities are flat
# where the penalty outweighs the stiffness an element keeps this share of its own sensitivity instead
# of none: with none, densities everywhere there only shrank and Keep % could not be held (a cantilever
# ended at 9% instead of 30%). A share of the mean sensitivity made useless elements more attractive than
# they were: with Max thickness the bolted bracket filled Keep % with scattered thin material (+73% surface)
OUTLINE_FLOOR = 1e-3
# Symmetry: mirror planes through the design space's middle (the grid is then laid symmetrically
# about it, so every element has its mirror image), the sensitivities averaged with their images
SYMMETRY = ("none", "x", "y", "z", "x and y", "x and z", "y and z", "x, y and z")


def _thinnest_max(h, min_thickness):
    """The thinnest Max thickness elements of h mm can hold: 1.5 x the thinnest member, and a
    neighbourhood of an element."""
    return max(1.5 * max(float(min_thickness), MIN_MEMBER * h), h / LOCAL_SIGMA)


def levels(space, detail, coarse=True, volume=None, min_thickness=0.0, max_thickness=0.0):
    """The details an optimization runs on, coarsest first: with `coarse` and fine detail, the coarser
    ones whose elements can hold the Max thickness, then `detail`. (Medium gained nothing from it: its
    iterations are as quick as draft's on the GPU; key box 52 s either way.)"""
    if not coarse or detail != LEVELS[-1]:
        return [detail]
    out = []
    for level in LEVELS[:LEVELS.index(detail)]:
        h = plan_grid(space, level, volume).h
        if not max_thickness or max_thickness >= _thinnest_max(h, min_thickness):
            out.append(level)
    return out + [detail]


def _transfer(x, old, new):
    """Values per element of grid `old` at the element centres of grid `new` (trilinear; host arrays)."""
    at = [(new.origin[a] + (np.arange(new.shape[a]) + 0.5) * new.h - old.origin[a]) / old.h - 0.5 for a in range(3)]
    return ndimage.map_coordinates(x, np.meshgrid(*at, indexing="ij"), order=1, mode="nearest").astype(np.float32)


def mirror_axes(symmetry):
    if symmetry not in SYMMETRY:
        raise ConditionError(f"symmetry must be one of: {', '.join(SYMMETRY)}")
    return [k for k, a in enumerate("xyz") if a in symmetry] if symmetry != "none" else []


@dataclass
class Problem:
    """Everything that stays the same during an optimization (arrays on xp's device unless noted)."""
    grid: object
    phi: object            # (nx, ny, nz) how full each element is (design space, keep, avoid)
    solid: object          # bool: always full (skin at supports and loads, Keep)
    free: object           # bool: densities to optimize
    fixed: np.ndarray      # host (X, Y, Z, 3)
    force: np.ndarray      # host (X, Y, Z, 3)
    kernel: object         # filter weights
    norm: object           # 1 / filtered domain (0 outside)
    domain: object         # float 0/1: elements that can hold material
    total: float           # sum of phi over the domain (the full design space's volume in elements)
    target: float          # volume fraction to keep
    radius: float          # filter radius, mm
    material: object
    xp: object
    dtype: object
    notes: list = field(default_factory=list)
    local_sigma: float = 0.0  # max thickness: neighbourhood size in elements (0 = no max thickness)
    empty_norm: float = 1.0    # 1 / the kernel's sum (filter_empty)
    space: object = None       # host bool: the design space (and Keep), Avoid and the gap included
    full_phi: object = None    # with a gap: phi without it (the "full design space" stiffness)
    smoothness: float = 0.0    # weight of the outline penalty (0 = off)
    mirror: tuple = ()         # (axis, elements across the part): symmetry planes in their middle
    prefer: object = None      # Prefer: 0..1 per element, how much of the guide body it is in (None = none)
    prefer_strength: float = 0.0

    def filter(self, x):
        return _convolve(self.xp, self.kernel.size)(x * self.domain, self.kernel) * self.norm

    def filter_transpose(self, g):
        return self.domain * _convolve(self.xp, self.kernel.size)(g * self.norm, self.kernel)

    def filter_empty(self, x):
        """The filter with everything outside the design space counted as empty, as it is in the part
        (`filter` averages over the design space only, so its faces keep their material exactly; for
        the eroded design that made a plate lying on a face carry at half the thickness of one inside,
        and 40% of a key box's material came out as such plates)."""
        return _convolve(self.xp, self.kernel.size)(x * self.domain, self.kernel) * self.empty_norm

    def filter_empty_transpose(self, g):
        return self.domain * _convolve(self.xp, self.kernel.size)(g * self.empty_norm, self.kernel)

    def _gauss(self, a):
        return gaussian_filter(self.xp)(a, sigma=self.local_sigma, mode="constant", cval=0.0, truncate=3.0)

    def local(self, xb):
        """How full each element's neighbourhood is (0..1). Outside the design space counts as
        empty, as it is in the part: counted as a share of the design space, a 6 mm wall of a
        thin-walled design space was full whatever the max thickness, and every wall got holes."""
        return self._gauss(xb * self.phi * self.domain)

    def local_transpose(self, w):
        return self.domain * self.phi * self._gauss(w)

    def volume(self, xb):
        return float(self.xp.sum(self.phi * xb)) / self.total

    def symmetric(self, a):
        """`a` averaged with its mirror image across each symmetry plane where both elements are free
        (exactly the same on both sides then; fixed regions and what has no image stay as they are)."""
        xp = self.xp
        for axis, n in self.mirror:
            cut = (slice(None),) * axis + (slice(0, n),)
            both = self.free[cut] & xp.flip(self.free[cut], axis)
            part = a[cut]
            mean = xp.where(both, 0.5 * (part + xp.flip(part, axis)), part)
            a = a.copy()
            a[cut] = mean
        return a


def _convolve(xp, taps):
    """Convolution with a (symmetric) kernel of `taps` entries, zero outside the grid: big kernels by
    FFT (struts only at Max thickness 15 mm, 21^3 taps on 0.96 mm elements: 14.6 -> 0.9 ms a filter on
    the GPU, which runs ~100 an iteration); FFT leaves ~1e-6 of the largest value where it is 0."""
    if taps >= FFT_TAPS:
        if xp is np:
            from scipy.signal import fftconvolve
        else:
            from cupyx.scipy.signal import fftconvolve
        return lambda a, kernel: fftconvolve(a, kernel, mode="same")
    if xp is np:
        return lambda a, kernel: ndimage.convolve(a, kernel, mode="constant")
    from cupyx.scipy.ndimage import convolve
    return lambda a, kernel: convolve(a, kernel, mode="constant")


def project(xt, beta, eta=ETA):
    xp = xp_of(xt)
    a, b = math.tanh(beta * eta), math.tanh(beta * (1 - eta))
    return (a + xp.tanh(beta * (xt - eta))) / (a + b)


def project_slope(xt, beta, eta=ETA):
    xp = xp_of(xt)
    a, b = math.tanh(beta * eta), math.tanh(beta * (1 - eta))
    return beta * (1 - xp.tanh(beta * (xt - eta)) ** 2) / (a + b)


def _decreasing_root(f, lo=1e-9, hi=1e9, ftol=1e-5, rtol=1e-4, maxit=60):
    """A root of a decreasing function on [lo, hi], searched in log space (Illinois regula falsi):
    the smallest value found with f <= 0 (lo if f(lo) <= 0 already, hi if f(hi) > 0)."""
    a, b = math.log(lo), math.log(hi)
    fa = f(lo)
    if fa <= 0:
        return lo
    fb = f(hi)
    if fb > 0:
        return hi
    kept = 0  # which end the last two steps kept (Illinois: halve the other end's value then)
    for _ in range(maxit):
        c = (a * fb - b * fa) / (fb - fa)
        if not a < c < b:
            c = 0.5 * (a + b)
        fc = f(math.exp(c))
        if fc > 0:
            a, fa = c, fc
            if kept == 1:
                fb *= 0.5
            kept = 1
        else:
            b, fb = c, fc
            if kept == -1:
                fa *= 0.5
            kept = -1
            if -fc <= ftol:
                break
        if b - a < rtol:
            break
    return math.exp(b)


def _skin_points(condition, spacing):
    """Points of a condition's faces the solid skin is kept around: all of them, except for a load
    acting on part of its faces only (a bearing load: the contact side), where the rest is left to
    the optimizer (the whole pin eye stayed solid although only its bottom carries the pin)."""
    pts, _, normals = sample_triangles(condition.triangles, spacing, stretch=False)  # a near-face test
    return pts[condition.presses(normals)] if len(pts) else pts


def _gap_cells(space, grid, phi, conditions, keep_fill, gap, reach, xp, cache):
    """Elements within `gap` mm of the design space's faces, except around the faces of supports and
    loads (within reach + gap, so material gets to them from inside) and Keep (within gap): kept
    empty, so the shape is not cut flat by the design space's faces."""
    depth = -to_host(Context(grid.centres(), cache, xp).eval_full(space)).astype(np.float64)  # mm inside
    near = (depth < gap) & (phi > 0)
    cells = np.flatnonzero(near)
    if not len(cells):
        return near
    centres = grid.origin + (np.column_stack(np.unravel_index(cells, phi.shape)) + 0.5) * grid.h
    pts = np.concatenate([sample_triangles(c.triangles, grid.h / 2, stretch=False)[0] for c in conditions])  # all faces
    exempt = np.zeros(phi.shape, dtype=bool)
    exempt.ravel()[cells[cKDTree(pts).query(centres, distance_upper_bound=reach + gap + 1e-9)[0] <= reach + gap]] = True
    if keep_fill is not None:
        exempt |= ndimage.distance_transform_edt(keep_fill <= 0.5, sampling=grid.h) <= gap
    return near & ~exempt


def topology_key(space, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, detail, h=None,
                 max_thickness=0.0, members=MEMBERS[0], gap=0.0, smoothness=0.0, symmetry="none", coarse=False,
                 prefer_body=None, prefer_strength=0.0):
    """Fingerprint of everything an optimization's result depends on (without a max thickness, a
    gap, smoothness, symmetry, a coarse start or a guide the same as before they existed, so earlier
    results stay valid)."""
    extra = (float(max_thickness),) if max_thickness else ()
    if max_thickness and members != MEMBERS[0]:
        extra += (members,)
    if gap:
        extra += (("gap", float(gap), "members along it as thick as elsewhere"),)
    if smoothness:
        extra += (("smoothness", float(smoothness)),)
    if symmetry != "none":
        extra += (("symmetry", symmetry),)
    if coarse and detail == LEVELS[-1]:
        extra += ("started coarse",)
    if prefer_body is not None and prefer_strength:
        extra += (("prefer", prefer_body.key, float(prefer_strength), GUIDE_FADE),)
    if any(c.kind == "load" and c.load == "bearing" for c in conditions):
        extra += ("bearing skin on the contact side",)  # results from before that change are not reused
    # of the material only its stiffness and Poisson's ratio (a new strength or density kept nothing)
    stiffness = _digest("material", float(material.E), float(material.nu))
    return _digest("topology", TOPO_VERSION, space.key, conditions.key, stiffness, float(keep),
                   float(min_thickness), float(skin), getattr(keep_body, "key", None),
                   getattr(avoid_body, "key", None), detail, h, *extra)


def prepare(space, conditions, material, keep=30.0, min_thickness=0.0, skin=2.0, keep_body=None, avoid_body=None,
            detail="draft", xp=np, cache=None, volume=None, h=None, max_thickness=0.0, members=MEMBERS[0],
            gap=0.0, smoothness=0.0, symmetry="none", prefer_body=None, prefer_strength=0.0):
    """The optimization problem. `xp` may be a function giving numpy or cupy for a number of elements."""
    if not 0 < keep < 100:
        raise ConditionError("Keep % must be between 0 and 100")
    if gap < 0:
        raise ConditionError("the gap must be 0 or more")
    if smoothness < 0:
        raise ConditionError("smoothness must be 0 or more")
    axes = mirror_axes(symmetry)
    grid = plan_grid(space, detail, volume, h)
    across = {}
    if axes:  # the same number of elements on both sides of each plane (the part's faces in them alike)
        lo, hi = (np.array(b, dtype=float) for b in space.bounds)
        for a in axes:
            across[a] = max(1, int(math.ceil((hi[a] - lo[a]) / grid.h - 1e-9)))
            mid = 0.5 * (lo[a] + hi[a])
            lo[a], hi[a] = mid - 0.5 * across[a] * grid.h, mid + 0.5 * across[a] * grid.h
        grid = make_grid((lo, hi), grid.h)
    if callable(xp):
        xp = xp(grid.size)
    h = grid.h
    phi = element_fill(space, grid, xp, cache)
    keep_fill = element_fill(keep_body, grid, xp, cache) if keep_body is not None else None
    if keep_fill is not None:
        phi = np.maximum(phi, keep_fill)
    region = phi > 0  # the surface is only extended past this (not into Avoid or the gap)
    if avoid_body is not None:
        avoid = element_fill(avoid_body, grid, xp, cache) > 0.5
        if keep_fill is not None:
            avoid &= keep_fill <= 0.5
        phi[avoid] = 0.0
    if not (phi > 0).any():
        raise ConditionError("the design space is empty")
    # solid skin at the faces with supports and loads: at least the elements those faces run through,
    # and never thinner than a Min thickness (a 2 mm skin on a key box's rim, under a bearing load, was
    # most of what came out thinner than its 3 mm)
    reach = max(float(skin), h)
    if skin > 0 and min_thickness > reach:
        reach = float(min_thickness)
    full, emptied = None, None
    if gap > 0:
        emptied = _gap_cells(space, grid, phi, conditions, keep_fill, float(gap), reach, xp, cache)
        full = phi.copy()
        phi = np.where(emptied, 0.0, phi)
    fixed, force, notes = apply_conditions(grid, phi, conditions)
    phi, more = _drop_loose(phi, fixed, force)
    notes = notes + more
    if full is not None:
        full, _ = _drop_loose(full, fixed, force)
        left = float(phi.sum()) / float(full.sum())
        if keep / 100 > 0.97 * left:
            raise ConditionError(f"with a gap of {gap:g} mm only {100 * left:.0f}% of the design space is left: "
                                 f"lower Keep % or the gap")
        notes.append(f"gap {gap:g} mm: {100 * (1 - left):.0f}% of the design space along its faces is kept empty")
    if reach > max(float(skin), h):
        notes.append(f"skin {skin:g} mm made as thick as the min thickness, {reach:g} mm")

    # solid skin: elements within `reach` mm of the faces with supports and loads, so bolt holes and
    # contact faces stay
    pts = np.concatenate([_skin_points(c, min(h, reach) / 2) for c in conditions])
    cells = np.flatnonzero(phi > 0)
    centres = grid.origin + (np.column_stack(np.unravel_index(cells, phi.shape)) + 0.5) * h
    near = cKDTree(pts).query(centres, distance_upper_bound=reach + 1e-9)[0] <= reach
    solid = np.zeros(phi.shape, dtype=bool)
    solid.ravel()[cells[near]] = True
    if keep_fill is not None:
        solid |= keep_fill > 0.5
    solid &= phi > 0
    domain = phi > 0
    total = float((phi if full is None else full).sum())  # Keep % stays a share of the whole design space
    target = keep / 100.0
    solid_share = float(phi[solid].sum()) / total
    if solid_share >= target:
        raise ConditionError(f"the solid skin at supports and loads (and Keep) is already {100 * solid_share:.0f}% "
                             f"of the design space: raise Keep % above that, or make Skin thinner")

    radius = max(float(min_thickness), MIN_MEMBER * h)  # the thinnest member (see ERODED)
    if min_thickness > 0 and min_thickness < MIN_MEMBER * h:
        notes.append(f"min thickness {min_thickness:g} mm is under {MIN_MEMBER:g} elements of {h:g} mm: using "
                     f"{radius:.3g} mm (choose a finer detail for thinner members)")
    if members not in MEMBERS:
        raise ConditionError(f"members must be one of: {', '.join(MEMBERS)}")
    sigma = 0.0
    if max_thickness > 0:
        thinnest = _thinnest_max(h, min_thickness)
        if max_thickness < thinnest:
            notes.append(f"max thickness {max_thickness:g} mm is too thin for {h:g} mm elements: using "
                         f"{thinnest:.3g} mm (choose a finer detail, or a smaller min thickness)")
            max_thickness = thinnest
        if members == "struts only":
            # measured on a round strut, so a plate may be at most PLATE_SHARE of it; and members at
            # least STRUT_MIN of it: thinner than any plate allowed, so plates cannot form at all
            sigma = max_thickness * STRUT_SIGMA / h
            radius = max(radius, STRUT_MIN * max_thickness)
        else:
            sigma = max_thickness * LOCAL_SIGMA / h
    R = int(math.floor(radius / h + 1e-9))
    o = np.arange(-R, R + 1)
    d = h * np.sqrt(o[:, None, None] ** 2 + o[None, :, None] ** 2 + o[None, None, :] ** 2)
    kernel = np.maximum(radius - d, 0.0)
    dtype = np.float64 if xp is np else np.float32
    D = xp.asarray(domain, dtype=dtype)
    kern = xp.asarray(kernel, dtype=dtype)
    # the filter averages over the gap as over empty material (not only over the domain, as at the
    # design space's faces): members along the gap are then as thick as anywhere (they were 1-2
    # elements, plates the gap's edge, which Smoothing then erased)
    N = _convolve(xp, kern.size)(D if emptied is None else xp.asarray(domain | emptied, dtype=dtype), kern)
    reached = N > 1e-4 * radius  # (in the domain N is at least the kernel's middle, the radius)
    norm = xp.where(reached, 1.0 / xp.where(reached, N, 1.0), 0.0).astype(dtype)
    problem = Problem(grid, xp.asarray(phi, dtype=dtype), xp.asarray(solid), xp.asarray(domain & ~solid), fixed,
                      force, kern, norm, D, total, target, radius, material, xp, dtype, notes)
    problem.space = region
    problem.empty_norm = 1.0 / float(kernel.sum())
    if full is not None:
        problem.full_phi = xp.asarray(full, dtype=dtype)
    problem.local_sigma = sigma
    problem.smoothness = float(smoothness)
    problem.mirror = tuple(across.items())
    if prefer_body is not None and prefer_strength:
        d = to_host(Context(grid.centres(), cache, xp).eval_full(prefer_body)).astype(np.float64)  # mm, < 0 inside
        guide = np.clip(1.0 - d / (GUIDE_FADE * radius), 0.0, 1.0) * domain
        if guide.any():
            problem.prefer = xp.asarray(guide, dtype=dtype)
            problem.prefer_strength = float(prefer_strength)
        else:
            problem.notes.append("the Prefer body is outside the design space: it has no effect")
    for a, n in problem.mirror:
        cut = (slice(None),) * a + (slice(0, n),)
        mid = grid.origin[a] + 0.5 * n * h
        note = f"symmetry: mirror images across {'xyz'[a]} = {mid:.4g} mm"
        lone = []
        for what, cells in (("design space", domain), ("skin and Keep", solid)):
            share = float((cells[cut] & ~np.flip(cells[cut], a)).sum()) / max(int(cells.sum()), 1)
            if share >= 0.05:
                lone.append(f"{100 * share:.0f}% of the {what}")
        if lone:  # only the free elements are kept symmetric
            note += f" ({' and '.join(lone)} has no mirror image: the shape is symmetric only elsewhere)"
        problem.notes.append(note)
    return problem


class Optimizer:
    """Runs the iterations of a Problem; `state()` / a `state` argument save and resume."""

    def __init__(self, problem, state=None):
        p = self.p = problem
        xp = p.xp
        self.KE = stiffness(problem.material.nu)
        self.local_max = None    # max thickness: the fullest neighbourhood in the last iteration
        self.power = []          # the last iteration's lmax power vectors per multigrid level (warm starts)
        self._design = None      # (iteration, compliance of the design itself)
        self.level = None        # with a coarse start: the detail of this problem (saved with the state)
        if state is not None:
            self.x = xp.asarray(state["x"], dtype=p.dtype)
            self.iteration = int(state["iteration"])
            self.beta = float(state["beta"])
            self.since_beta = int(state["since_beta"])
            self.u = xp.asarray(state["u"], dtype=np.float64) if state["u"].size else None
            self.history = [tuple(r) for r in np.asarray(state["history"]).reshape(-1, 4)]
            self.full_compliance = float(state["full_compliance"])
            self.done = bool(state["done"])
            self.local_scale = float(state["local_scale"]) if "local_scale" in state else 0.0
            levels = sorted((int(k[6:]), k) for k in state if k.startswith("power_"))
            self.power = [xp.asarray(state[k]) for _, k in levels]   # (older states: fresh estimates once)
            if "compliance" in state:
                self._design = (self.iteration, float(state["compliance"]))
        else:
            self.local_scale = 0.0   # max thickness: max / p-mean of the local densities (0 = not yet)
            # start uniform at the target volume (solid parts full)
            free_share = (p.target * p.total - float(xp.sum(p.phi * p.solid))) / float(xp.sum(p.phi * p.free))
            self.x = (xp.where(p.solid, 1.0, 0.0) + p.free * min(max(free_share, 0.0), 1.0)).astype(p.dtype)
            self.iteration, self.beta, self.since_beta = 0, BETA_START, 0
            self.u, self.history, self.done = None, [], False
            # the stiffness of the whole design space filled (a gap filled too: what it costs shows)
            self.full_compliance = self._compliance_of(xp.ones_like(self.x), phi=p.full_phi)[0]

    # --- the pieces ------------------------------------------------------------------------------
    def physical(self, x):
        """(filtered, projected-with-fixed-regions) densities of design variables x."""
        p = self.p
        xt = p.filter(x)
        xb = project(xt, self.beta)
        xb = p.xp.where(p.solid, 1.0, xb) * p.domain
        return xt, xb.astype(p.dtype)

    def carrying(self, xe_filtered, xb):
        """What carries the loads, of densities filtered with the faces counted as empty
        (`filter_empty`) and the design xb: the eroded design (fixed regions full), plus THIN_SHARE of
        what erosion takes away."""
        p = self.p
        xe = p.xp.where(p.solid, 1.0, project(xe_filtered, self.beta, ERODED)) * p.domain
        return (xe + THIN_SHARE * (xb - xe)).astype(p.dtype)

    def outline(self, x):
        """Smoothness: (the design's outline, its gradient by x). The outline is the sum of |grad xb|
        at every element corner (forward differences, everything outside the design space and the
        grid empty, so the part's whole surface counts), about its surface area in elements^2."""
        p = self.p
        xp = p.xp
        xt, xb = self.physical(x)
        padded = xp.pad(xb, 1)
        n = padded.shape
        corners = tuple(slice(0, m - 1) for m in n)
        aheads = [tuple(slice(1, m) if a == k else slice(0, m - 1) for a, m in enumerate(n)) for k in range(3)]
        steps = [padded[ahead] - padded[corners] for ahead in aheads]
        size = xp.sqrt(steps[0] ** 2 + steps[1] ** 2 + steps[2] ** 2 + OUTLINE_EPS ** 2)
        grad = xp.zeros_like(padded)
        for step, ahead in zip(steps, aheads):
            unit = step / size
            grad[corners] -= unit
            grad[ahead] += unit
        inner = tuple(slice(1, m - 1) for m in n)
        grad = xp.where(p.solid, 0.0, grad[inner] * project_slope(xt, self.beta))   # through the projection
        area = float(xp.sum(size)) - size.size * OUTLINE_EPS   # (without what eps adds where it is flat)
        return area, p.filter_transpose(grad * p.domain).astype(p.dtype)

    def _scale(self, xb, phi=None):
        """Element stiffness factors (E h; 0 outside the design space, so the solver skips it)."""
        p = self.p
        E = p.material.E
        phi = p.phi if phi is None else phi
        rho = phi * xb
        domain = p.domain if phi is p.phi else (phi > 0)
        return p.grid.h * (EMIN * E + rho ** PENAL * (E - EMIN * E)) * domain, rho

    def _solve(self, scale, u0=None, callback=None, power=None):
        p = self.p
        H = Hierarchy(scale, self.KE, p.fixed, p.grid.coarsenings, p.xp, p.dtype, power=power)
        u, its, _ = H.solve(p.force, u0=u0, tol=SOLVE_TOL, callback=callback, maxiter=WARM_MAXITER if power else 1000)
        return H, u, its

    def _compliance_of(self, xb, callback=None, phi=None):
        scale, _ = self._scale(xb, phi)
        H, u, its = self._solve(scale, callback=callback)
        level = H.exact or H.levels[0]
        energy = level.unit_energies(u)
        return float(self.p.xp.sum(self.p.xp.asarray(scale, dtype=np.float64) * energy)), u

    def _update(self, x, dc, dv):
        """Optimality criteria: x * sqrt(-dc / (lambda dv)) within the move limit, lambda found by
        bisection so the projected volume is the target."""
        p = self.p
        xp = p.xp
        move = MOVE if self.beta < 4 else MOVE / 2
        free = p.free
        dc = xp.minimum(dc, 0.0)
        scale_c = float(xp.mean(xp.abs(dc[free]))) or 1.0
        scale_v = float(xp.mean(dv[free])) or 1.0
        ratio = xp.where(free, -dc / scale_c, 0.0) / xp.where(free, xp.maximum(dv / scale_v, 1e-30), 1.0)
        lo_x, hi_x = xp.maximum(x - move, 0.0), xp.minimum(x + move, 1.0)
        lo, hi = 1e-9, 1e9
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            xn = xp.where(free, xp.clip(x * xp.sqrt(ratio / mid), lo_x, hi_x), x).astype(p.dtype)
            if p.volume(self.physical(xn)[1]) > p.target:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1 + 1e-5:
                break
        return xn

    def _local_constraint(self, xb):
        """Max thickness: g = max local density / LOCAL_SHARE - 1 (<= 0 when no neighbourhood is too
        full), as an adaptively scaled p-mean so it has a gradient; returns (g, dg/dxb, max local)."""
        p = self.p
        xp = p.xp
        free = p.free
        rho = p.local(xb)
        r = rho[free]
        n = max(int(r.size), 1)
        top = float(xp.max(r)) if r.size else 0.0
        mean_p = float(xp.mean(r ** LOCAL_P)) if r.size else 0.0
        pmean = mean_p ** (1 / LOCAL_P) if mean_p > 0 else 0.0
        if pmean <= 0:
            return -1.0, xp.zeros_like(xb), top
        scale = top / pmean
        self.local_scale = scale if self.local_scale <= 0 else 0.5 * (self.local_scale + scale)
        g = self.local_scale * pmean / LOCAL_SHARE - 1.0
        w = xp.where(free, (self.local_scale / LOCAL_SHARE) * pmean ** (1 - LOCAL_P) * rho ** (LOCAL_P - 1) / n, 0.0)
        return g, p.local_transpose(w.astype(p.dtype)), top

    def _update_two(self, x, dc, dv, dg, volume, g):
        """Optimality criteria with two constraints, the volume (= target, exact: filtered and
        projected) and max thickness (g <= 0, linearized): x * sqrt(-dc / (l1 dv + l2 dg)) within
        the move limit, l1 and l2 by nested root finding (l2 = 0 when the max thickness holds anyway).
        A linearized volume ended 3.5 points under Keep % on the wall bracket (the projection is
        steep), and nested bisection took ~300 evaluations per iteration (Illinois: ~60)."""
        p = self.p
        xp = p.xp
        move = MOVE if self.beta < 4 else MOVE / 2
        free = p.free
        dc = xp.where(free, xp.minimum(dc, 0.0), 0.0)
        sc = float(xp.mean(xp.abs(dc[free]))) or 1.0
        sv = float(xp.mean(dv[free])) or 1.0
        sg = float(xp.mean(xp.abs(dg[free]))) or 1.0
        ndc, ndv, ndg = -dc / sc, xp.maximum(dv / sv, 0.0), xp.maximum(dg / sg, 0.0)
        lo_x, hi_x = xp.maximum(x - move, 0.0), xp.minimum(x + move, 1.0)

        def new_x(l1, l2):
            return xp.where(free, xp.clip(x * xp.sqrt(ndc / xp.maximum(l1 * ndv + l2 * ndg, 1e-30)), lo_x, hi_x),
                            x).astype(p.dtype)

        def volume_ok(l2):  # l1 so the volume is the target (PRICE_FLOOR if Keep % does not fit)
            l1 = _decreasing_root(lambda l1: p.volume(self.physical(new_x(l1, l2))[1]) - p.target, lo=PRICE_FLOOR,
                                  ftol=1e-5)
            return new_x(l1, l2)

        xn = volume_ok(0.0)
        if g + float(xp.sum(dg * (xn - x))) <= 0:
            return xn
        held = {}  # the root found is the last l2 tried where the max thickness held: kept, not found again

        def thickness(l2):
            xn = volume_ok(l2)
            value = g + float(xp.sum(dg * (xn - x)))
            if value <= 0:
                held.clear()
                held[l2] = xn
            return value
        l2 = _decreasing_root(thickness, ftol=1e-4)
        return held[l2] if l2 in held else volume_ok(l2)

    # --- one iteration -------------------------------------------------------------------------
    def step(self, solver_callback=None):
        """One iteration. Returns (compliance, volume fraction, largest change)."""
        p = self.p
        xp = p.xp
        xt, xb = self.physical(self.x)
        xv = p.filter_empty(self.x)
        scale, rho = self._scale(self.carrying(xv, xb))
        warm = self.power if self.iteration >= WARM_LMAX_AFTER else []
        try:
            H, u, its = self._solve(scale, self.u, solver_callback, warm)
        except NotConverged:
            if not warm:
                raise
            # a warm lmax estimate too low (the smoothers then amplify what they should damp)
            H, u, its = self._solve(scale, self.u, solver_callback, [])
        self.power = H.power
        self.u = u
        level = H.exact or H.levels[0]
        energy = level.unit_energies(u).astype(p.dtype)
        compliance = float(xp.sum(xp.asarray(scale, dtype=np.float64) * energy))
        E = p.material.E
        dc_rho = -PENAL * rho ** (PENAL - 1) * (E - EMIN * E) * p.grid.h * energy
        free = p.free
        slope = project_slope(xt, self.beta)
        dc_x = xp.where(free, dc_rho * p.phi, 0.0)
        dc = ((1 - THIN_SHARE) * p.filter_empty_transpose(dc_x * project_slope(xv, self.beta, ERODED))
              + THIN_SHARE * p.filter_transpose(dc_x * slope))
        dv = p.filter_transpose(xp.where(free, p.phi / p.total, 0.0) * slope)
        if p.smoothness and bool(xp.any(free)):
            # minimizes compliance x outline^smoothness: relative changes of both are weighed
            area, ds = self.outline(self.x)
            dc = xp.minimum(dc + (p.smoothness * compliance / max(area, 1e-30)) * ds, OUTLINE_FLOOR * dc)
        if p.prefer is not None:
            # the design's material in the guide, rewarded (or with a negative strength made dearer) at
            # strength x the mean stiffness sensitivity there
            reward = p.filter_transpose(p.prefer * p.phi * slope)
            near = (p.prefer > 0) & free
            if bool(xp.any(near)):
                scale = float(xp.mean(xp.abs(dc[free]))) / max(float(xp.mean(xp.abs(reward[near]))), 1e-30)
                dc = dc - (p.prefer_strength * scale) * reward
        if p.mirror:
            dc, dv = p.symmetric(dc), p.symmetric(dv)
        if p.local_sigma:
            g, dg_xb, self.local_max = self._local_constraint(xb)
            dg = p.filter_transpose(xp.where(free, dg_xb, 0.0) * slope)
            if p.mirror:
                dg = p.symmetric(dg)
            x_new = self._update_two(self.x, dc, dv, dg, p.volume(xb), g)
        else:
            x_new = self._update(self.x, dc, dv)
        change = float(xp.max(xp.abs(x_new - self.x)[free])) if bool(xp.any(free)) else 0.0
        self.x = x_new
        self.iteration += 1
        self.since_beta += 1
        volume = p.volume(self.physical(self.x)[1])
        self.history.append((compliance, volume, change, self.beta))
        settled = change < CHANGE_TOL or self._plateau()
        if self.beta < BETA_MAX and (self.since_beta >= BETA_EVERY or settled):
            self.beta = min(2 * self.beta, BETA_MAX)
            self.since_beta = 0
        elif self.beta >= BETA_MAX and settled:
            self.done = True
        if self.iteration >= MAX_ITERATIONS:
            self.done = True
        return compliance, volume, change

    def _plateau(self):
        """The compliance has (nearly) stopped changing at this projection sharpness."""
        if self.since_beta <= PLATEAU_WINDOW:  # the window must lie within this sharpness
            return False
        first, last = self.history[-PLATEAU_WINDOW - 1][0], self.history[-1][0]
        return abs(first - last) <= PLATEAU * abs(last)

    def run(self, on_iteration=None, solver_callback=None, hand_over=None):
        """Iterate until done, or until the projection sharpness reaches `hand_over` (a coarser level of
        a coarse start). on_iteration(optimizer, compliance, volume, change, seconds) after every
        iteration (progress, live view, checkpoints; it may raise to stop)."""
        while not self.done and (hand_over is None or self.beta < hand_over):
            t = time.perf_counter()
            c, v, ch = self.step(solver_callback)
            if on_iteration is not None:
                on_iteration(self, c, v, ch, time.perf_counter() - t)
        return self

    def handover(self):
        """What the next, finer level continues from (`take_over`), on the host."""
        return {"x": np.asarray(to_host(self.x), dtype=np.float32), "grid": self.p.grid, "beta": self.beta,
                "iteration": self.iteration, "history": list(self.history)}

    def take_over(self, handover):
        """Continue a coarser level's run: its design moved to this grid, its projection sharpness,
        iteration count and history (the sharpness schedule starts over at it)."""
        p = self.p
        xp = p.xp
        x = xp.asarray(_transfer(handover["x"], handover["grid"], p.grid), dtype=p.dtype)
        x = xp.where(p.free, xp.clip(x, 0.0, 1.0), self.x).astype(p.dtype)
        self.x = p.symmetric(x) if p.mirror else x    # (interpolation may round one side differently)
        self.beta, self.since_beta = handover["beta"], 0
        self.iteration, self.history = handover["iteration"], list(handover["history"])
        return self

    def density(self):
        """Final densities per element (0..1, host), solid regions included."""
        return np.asarray(to_host(self.physical(self.x)[1]), dtype=np.float32)

    def design_compliance(self):
        """The compliance of the design itself, one solve per iteration asked for. The history has
        the eroded design's, which is less stiff (key box, struts only: 58% of the full design space
        where the design has 77%)."""
        if self._design is None or self._design[0] != self.iteration:
            self._design = (self.iteration, self._compliance_of(self.physical(self.x)[1])[0])
        return self._design[1]

    def state(self, stopped=False):
        p = self.p
        # the lmax vectors too, while the run can continue: it then continues exactly
        extra = {} if self.done else {f"power_{i}": np.asarray(to_host(v)) for i, v in enumerate(self.power)}
        if self._design is not None and self._design[0] == self.iteration:  # when it was worked out
            extra["compliance"] = np.array(self._design[1])
        if self.level is not None:   # the detail a coarse start was at
            extra["level"] = np.array(self.level)
        return {**extra, "x": np.asarray(to_host(self.x), dtype=np.float32), "iteration": np.array(self.iteration),
                "beta": np.array(self.beta), "since_beta": np.array(self.since_beta),
                # float64: a resumed run then gives exactly the uninterrupted one's result
                "u": np.asarray(to_host(self.u), dtype=np.float64) if self.u is not None else np.zeros(0),
                "history": np.asarray(self.history, dtype=np.float64).reshape(-1, 4),
                "full_compliance": np.array(self.full_compliance), "done": np.array(self.done),
                "local_scale": np.array(self.local_scale),
                "stopped": np.array(stopped), "density": self.density(),
                "grid": np.array([*p.grid.origin, p.grid.h, *p.grid.shape], dtype=np.float64),
                "phi": np.asarray(to_host(p.phi), dtype=np.float32),
                "space": np.asarray(p.space if p.space is not None else to_host(p.phi) > 0, dtype=bool)}

    def summary(self):
        return summary(self.history, self.full_compliance, self.iteration,
                       self.design_compliance() if self.history else None)


def summary(history, full_compliance, iteration, compliance=None):
    """`compliance`: the design's (else the last iteration's, of the eroded design)."""
    if not len(history):
        return "no iterations yet"
    c, v = history[-1][0] if compliance is None else compliance, history[-1][1]
    return (f"{iteration} iterations · {100 * v:.1f}% of the design space kept · stiffness "
            f"{100 * full_compliance / max(c, 1e-300):.0f}% of the full design space")


# --- from densities to a body ------------------------------------------------------------------
GROW_OUTSIDE = 2      # elements the densities are extended past the design space (cropped by it)
UPSAMPLE = 2          # the surface is found on a grid this much finer (trilinear), then smoothed
SMOOTH_ITERATIONS = 10
# Gaussian blur of the distance to the surface, sigma in elements (the node's Smoothing). An optimized
# result is nearly black and white per element, so gently sloped faces show its element steps as
# terraces (rings) and members' edges waves; more passes of mesh smoothing did not help. A blur of
# the densities themselves (until 2026-10-22) also thinned members: at 2.0 the bolted bracket with a
# 3 mm gap came out 25% less stiff, with stubs where members had joined; blurring the distance, it
# is as stiff at 3 as at 1 and the key box's band edges are straight from 2. See FINDINGS.md, "Smoother
# surfaces", "Smoothing 2 was weaker" and "Lumpy surfaces".
SMOOTHING = 2.0
THICKEN_PASSES = 6    # rounds of moving the surface of thin members out (each corrects the last)
LIGHT_BLUR = 1.0      # elements: the most smoothing thin members get, whatever Smoothing is
SPECK = 0.01          # pieces smaller than this share of the largest are dropped


def thicken(values, level, inside, voxel, thickness, passes=THICKEN_PASSES):
    """`values` (samples `voxel` mm apart, solid where >= level) with every member at least
    `thickness` thick, by moving the surface out where a member is thinner, as smoothly as the values
    run: a member's half thickness near a point is the deepest point of the solid within reach (the
    design space's faces counted as empty, as they are in the part); what it lacks of thickness / 2,
    blurred, is added to the values as that many mm along their gradient, a few passes. Thick members
    are left alone. (Balls on the middle of thin members and filling in from the faces, done on the
    samples, left steps of a sample and knobs all over the surface.) Plates lying on a face grow on
    both sides, so only half inwards: the optimizer's eroded design counts the faces as empty, so it
    does not make them thinner than elsewhere. Returns the new values and how many samples became
    solid inside."""
    r = thickness / 2
    # within a whole thickness, so a point by a thick member sees its depth (with r, a block beside a
    # thin plate grew by 1 mm)
    reach = int(math.ceil(thickness / voxel))     # samples
    before = int(((values >= level) & inside).sum())
    out = values.astype(np.float32)
    for _ in range(passes):
        solid = (out >= level) & inside
        # a sample's distance runs to the middle of the nearest sample outside, half a sample past the surface
        depth = ndimage.distance_transform_edt(solid, sampling=voxel) - 0.5 * voxel
        half = ndimage.maximum_filter(depth, size=2 * reach + 1)
        lack = np.where(half > 0, np.clip(r - half, 0.0, r), 0.0).astype(np.float32)
        del depth, half
        if float(lack.max()) < 0.05 * voxel:
            break
        lack = ndimage.gaussian_filter(lack, reach / 4)
        grad = np.sqrt(sum(d * d for d in np.gradient(out, voxel)))   # values per mm
        # half of it a pass: the values flatten out away from the surface, so a full step overshoots
        # (a 2 mm band became 4 mm instead of 3)
        out = out + 0.5 * grad * lack
        del grad, lack
    return out.astype(values.dtype), int(((out >= level) & inside).sum()) - before


def _surface_distance(values, voxel):
    """Signed distance (mm, negative inside) to the 0.5 surface of densities sampled `voxel` apart:
    by distance transforms, and within a sample of the surface from the densities themselves, so the
    surface is not stepped by the samples."""
    solid = values >= 0.5
    dist = np.where(solid, 0.5 * voxel - ndimage.distance_transform_edt(solid, sampling=voxel),
                    ndimage.distance_transform_edt(~solid, sampling=voxel) - 0.5 * voxel).astype(np.float32)
    grad = np.sqrt(sum(d * d for d in np.gradient(values, voxel)))
    near = (np.abs(dist) < voxel) & (grad > 1e-3 / voxel)
    dist[near] = np.clip((0.5 - values[near]) / grad[near], -voxel, voxel)
    return dist


def density_mesh(density, phi, origin, h, upsample=UPSAMPLE, smooth=SMOOTH_ITERATIONS, blur=SMOOTHING, space=None,
                 min_thickness=0.0):
    """The surface of per-element densities as a closed mesh (world coordinates, host arrays).
    Densities are extended outward past the design space (`space`, bool per element; phi > 0 when
    not given), so where material reaches it the surface runs past it, and intersecting with the
    design space gives back its exact CAD faces. Avoid regions and a gap inside it stay empty (with
    phi > 0 as the design space they were filled from the side, 2-4 elements deep). `blur`:
    Gaussian sigma in elements applied to the distance to the densities' 0.5 surface (rounds element
    steps and waves, members keep their thickness; thin members get at most LIGHT_BLUR).
    `min_thickness` (mm): what is still thinner than that is thickened (`thicken`)."""
    import trimesh
    from ..grid import Grid, upsample_linear
    from ..meshing import field_to_mesh
    g = GROW_OUTSIDE + int(math.ceil(blur))  # the blur must not round off faces cut by the design space
    # room to grow into: the FE grid starts right at the part's lowest corner
    region = np.asarray(space, dtype=bool) if space is not None else np.asarray(phi) > 0
    inside = np.pad(region, g, mode="constant", constant_values=False)
    ext = np.pad(np.asarray(density, dtype=np.float32), g, mode="constant")
    # outside the design space each element takes the density of the nearest one inside: material
    # runs straight out through the boundary, empty stretches of it stay empty (growing by dilation
    # spread it sideways too: 3 elements of that closed a beam's open sides into a cavity)
    distance, nearest = ndimage.distance_transform_edt(~inside, return_indices=True)
    ext = np.where(distance <= g, ext[tuple(nearest)], 0.0).astype(np.float32)
    del nearest
    values = np.pad(ext, 1, mode="constant", constant_values=0.0)  # a ring of "outside"
    start = np.asarray(origin, dtype=np.float64) + h / 2 - (g + 1) * h  # element centres, less the pads
    shape = tuple((n - 1) * upsample + 1 for n in values.shape)
    up = (lambda a: upsample_linear(a, upsample, shape)) if upsample > 1 else (lambda a: a)
    values = up(values)
    voxel = h / upsample
    level = 0.5
    if blur > 0:
        # Smoothing blurs the distance to the surface, not the densities: across a member it runs
        # straight, so the blur hardly moves the surface. (Blurring the densities thinned members and took
        # thin ones away; held back to a light blur on thin members, the edges of bands inside a thin wall
        # stayed wavy at the elements' scale.) A member only goes once its half thickness is under ~0.8 of
        # the blur's sigma (at Smoothing 5 the key box lost a loaded boss), so members under half the sigma
        # get only LIGHT_BLUR, blended up to the full blur at a sigma (`thicken` then brings back what that
        # thinned below the min thickness; with the light blur up to twice the sigma, a thin-walled part's
        # edges stayed as wavy as at Smoothing 1).
        dist = _surface_distance(values, voxel)
        sigma = blur * h
        values = -ndimage.gaussian_filter(dist, blur * upsample)        # mm, positive inside
        if blur > LIGHT_BLUR:
            light = -ndimage.gaussian_filter(dist, LIGHT_BLUR * upsample)
            half = ndimage.maximum_filter(np.maximum(light, 0.0), size=2 * int(math.ceil(sigma / voxel)) + 1)
            weight = ndimage.gaussian_filter(np.clip(2.0 * half / sigma - 1.0, 0.0, 1.0), 0.5 * sigma / voxel)
            values = weight * values + (1.0 - weight) * light
            del light, half, weight
        values = values.astype(np.float32)
        # the surface where the material in the design space stays as much as the densities give (a
        # strong blur rounds members off: the bolted bracket lost 10% at Smoothing 3), within half an element
        space = up(np.pad(inside, 1, mode="constant").astype(np.float32)) >= 0.5
        flat = values[space]
        count = int(np.count_nonzero(dist[space] <= 0.0))
        del dist
        level = 0.0
        if 0 < count < flat.size:
            level = float(np.clip(np.partition(flat, flat.size - count)[flat.size - count], -0.5 * h, 0.5 * h))
    thickened = 0.0
    # inside: where the design space fills half an element or more (no Avoid, no gap)
    inside_fine = up(np.pad(np.asarray(phi, dtype=np.float32), g + 1, mode="constant")) >= 0.5
    if min_thickness > 0:
        had = int(((values >= level) & inside_fine).sum())
        values, added = thicken(values, level, inside_fine, voxel, float(min_thickness))
        thickened = added / max(had, 1)
    # bits on their own inside the design space, under SPECK of the largest, are emptied there: flakes by
    # its faces that only joined the part outside it, where the crop to the design space cuts them off
    # (the wall bracket's pin eye at Smoothness 0.5: three of 0.3-1.8 mm3, 0.5 mm thin)
    pieces, count = ndimage.label((values >= level) & inside_fine)
    if count > 1:
        sizes = np.bincount(pieces.ravel())[1:]
        small = np.flatnonzero(sizes < SPECK * sizes.max()) + 1
        if len(small):
            values = np.where(np.isin(pieces, small), np.minimum(values, level - voxel), values).astype(np.float32)
    del pieces
    values = (level - values).astype(np.float32)
    grid = Grid.block(start, h / upsample, (0, 0, 0), values.shape)
    if float(values.min()) >= 0:
        raise ConditionError("no material left (every density is below 0.5)")
    mesh = field_to_mesh(values, grid)
    if smooth:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth)
    pieces = mesh.split(only_watertight=False)
    if len(pieces) > 1:
        # (a cavity's surface faces inwards: negative volume, never a speck; by size alone an Avoid
        # pocket's surface was dropped as one and the pocket filled)
        volumes = np.array([p.volume for p in pieces])
        kept = [p for p, v in zip(pieces, volumes) if v < 0 or v >= SPECK * volumes.max()]
        mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    mesh.metadata["thickened"] = thickened  # share of material added for the min thickness
    return mesh


class LiveSurface:
    """The shape shown while optimizing: density_mesh's raw 0.5 surface (upsample 1, no smoothing,
    no blur) of one design space, with what stays the same between iterations done once (which
    element each one outside the design space copies) and the pieces found by labelling vertices
    instead of splitting the mesh (wall bracket, medium: 135 -> 50 ms). The same triangles as
    density_mesh, in another order."""

    def __init__(self, phi, origin, h, space=None):
        region = np.asarray(space, dtype=bool) if space is not None else np.asarray(phi) > 0
        self.g = GROW_OUTSIDE
        inside = np.pad(region, self.g, mode="constant", constant_values=False)
        distance, nearest = ndimage.distance_transform_edt(~inside, return_indices=True)
        self.reach = distance <= self.g
        self.nearest = np.ravel_multi_index(tuple(nearest), inside.shape)
        self.start = np.asarray(origin, dtype=np.float64) + h / 2 - (self.g + 1) * h
        self.h = h

    def __call__(self, density):
        import trimesh
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from ..grid import Grid
        from ..meshing import field_to_mesh
        ext = np.pad(np.asarray(density, dtype=np.float32), self.g, mode="constant")
        ext = np.where(self.reach, ext.ravel()[self.nearest], 0.0).astype(np.float32)
        values = (0.5 - np.pad(ext, 1, mode="constant", constant_values=0.0)).astype(np.float32)
        if float(values.min()) >= 0:
            raise ConditionError("no material left (every density is below 0.5)")
        mesh = field_to_mesh(values, Grid.block(self.start, self.h, (0, 0, 0), values.shape))
        faces = mesh.faces
        n = len(mesh.vertices)
        edges = coo_matrix((np.ones(2 * len(faces), dtype=np.int8), (faces[:, :2].ravel(), faces[:, 1:].ravel())),
                           shape=(n, n))
        count, labels = connected_components(edges, directed=False)
        if count > 1:  # pieces under SPECK of the largest dropped, as in density_mesh
            tri = mesh.vertices[faces]
            piece = labels[faces[:, 0]]
            volumes = np.abs(np.bincount(piece, weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])),
                                         minlength=count))
            keep = volumes[piece] >= SPECK * volumes.max()
            if not keep.all():
                mesh = trimesh.Trimesh(mesh.vertices, faces[keep], process=False)
                mesh.remove_unreferenced_vertices()
        return mesh
