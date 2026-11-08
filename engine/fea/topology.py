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
from types import SimpleNamespace
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.special import erfinv

from .. import backend
from ..backend import gaussian_filter, to_host, xp_of
from ..fields import Context, _digest
from .analysis import _drop_loose, plan_grid
from .element import stiffness
from .model import ConditionError, apply_cases, element_fill, make_grid, sample_triangles
from . import solver
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
MAX_ITERATIONS = 125  # the node's Max iterations by default; a run stopped by it continues when it is raised
CHANGE_TOL = 0.01     # done when no density changes more than this (at the sharpest projection) ...
PLATEAU = 1e-3        # ... or the compliance changed less than this over the last PLATEAU_WINDOW
PLATEAU_WINDOW = 5    # iterations (a few elements can flip back and forth for ever: 3D cantilever)
# Inexact solves, warm-started from the last one: the 3D cantilever came out the same (compliance
# 242.1) with 1e-5, 1e-3 and 1e-2, at 2.8, 1.6 and 1.1 s per CPU iteration.
SOLVE_TOL = 1e-2
SOLVE_NEAR = 10.0     # a float64 solve stopped by its iteration limit is used when within this x SOLVE_TOL (see _solve)
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
# Max section: "struts only" still leaves plates at exactly its thickest plate (bolted bracket, Max 10: half the
# solid 6.7 mm thick, 40-62% of it plate). A neighbourhood wider than the members reads their area instead: along
# a long member the Gaussian integrates to 1, so at its axis the local density is its section area / (2 pi sigma^2)
# (1 - exp of minus that on a round one), whatever the section's shape, up to about 2 sigma wide. Wider than that it
# is a plate of thickness t, filling erf(t / (2 sqrt 2 sigma)): sigma is chosen so the thickest plate that fits is
# SECTION_PLATE of the thinnest member (`section_limit`), and the plates that fit carry nothing. Only designed
# material counts: the skin at supports and loads cannot get thinner, and a joint of n members holds n / 2 sections.
# At 0.7 the bracket (Max section 35) kept a gusset, at 0.5 it is 90% strut. See FINDINGS.md, "Max section".
SECTION_PLATE = 0.5
SECTION_MIN = 0.5     # thinnest member, as a share of the diameter of a round member of Max section
# With a neighbourhood this wide the constraint's linearization overshoots at a sharp projection: material near the
# limit went and came back every other iteration, a load path with it (bracket: compliance 5458 <-> 1915 for 200
# iterations, every step at the move limit, the result depending on where it stopped). So, as MMA does with its
# asymptotes, a design variable that turns back gets a smaller move limit, and one that keeps going a larger one.
TURN_SHRINK = 0.7
TURN_GROW = 1.2
TURN_FLOOR = 0.002
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


def section_diameter(area):
    """The diameter of a round member of this section, mm."""
    return 2.0 * math.sqrt(area / math.pi)


def section_limit(area, thinnest, widest=math.inf):
    """Max section: (sigma in mm, how full that neighbourhood may be), so a round member of `area` mm2 just
    fits on its axis and the thickest plate that fits is SECTION_PLATE x `thinnest` (see SECTION_PLATE). Sigma
    stays under `widest` mm: a neighbourhood wider than the part reads nothing and costs a long filter, so a big
    area with a thin member gets thicker plates instead."""
    def share(sigma):
        return -math.expm1(-area / (2.0 * math.pi * sigma * sigma))

    def plate(sigma):
        return 2.0 * math.sqrt(2.0) * sigma * float(erfinv(share(sigma)))
    target = SECTION_PLATE * thinnest
    sigma = min(brentq(lambda s: plate(s) - target, 0.1 * math.sqrt(area), area / target), widest)
    return sigma, share(sigma)


def _room(phi, h):
    """How much room the design space has, mm: its thickest spot (the largest ball that fits, doubled), the
    second longest side of its box (what a member can be across, in a sheet too), the longest, and its largest
    section square to an axis (mm2). Limits past these cannot bind, and read as sizes they asked for filters of
    gigabytes (a Max section of 1e6 mm2 on a 30 mm box: 6 GB in 5 s)."""
    inside = phi > 0
    thickest = 2.0 * h * float(ndimage.distance_transform_edt(np.pad(inside, 1)).max())
    planes = ((1, 2), (0, 2), (0, 1))
    sides = sorted(int(inside.any(axis=a).sum()) * h for a in planes)
    largest = max(float(phi.sum(axis=a).max()) for a in planes) * h * h
    return SimpleNamespace(thickest=thickest, across=sides[1], longest=sides[2], largest=largest)


def levels(space, detail, coarse=True, volume=None, min_thickness=0.0, max_thickness=0.0, max_section=0.0):
    """The details an optimization runs on, coarsest first: with `coarse` and fine detail, the coarser
    ones whose elements can hold the Max thickness and Max section, then `detail`. (Medium gained nothing
    from it: its iterations are as quick as draft's on the GPU; key box 52 s either way.)"""
    if not coarse or detail != LEVELS[-1]:
        return [detail]
    limits = [t for t in (max_thickness, section_diameter(max_section)) if t]
    out = []
    for level in LEVELS[:LEVELS.index(detail)]:
        h = plan_grid(space, level, volume).h
        if all(t >= _thinnest_max(h, min_thickness) for t in limits):
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
    force: np.ndarray      # host (cases, X, Y, Z, 3): the forces of each load case
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
    section_sigma: float = 0.0 # max section: neighbourhood size in elements (0 = no max section)
    section_share: float = 0.0 # ... and how full it may be
    max_member: float = 0.0    # the Max thickness in use, mm (raised to what the elements hold; 0 when it cannot bind)
    section_area: float = 0.0  # the Max section in use, mm2 (the same)
    empty_norm: float = 1.0    # 1 / the kernel's sum (filter_empty)
    space: object = None       # host bool: the design space (and Keep), Avoid and the gap included
    full_phi: object = None    # with a gap: phi without it (the "full design space" stiffness)
    smoothness: float = 0.0    # weight of the outline penalty (0 = off)
    mirror: tuple = ()         # (axis, elements across the part): symmetry planes in their middle
    prefer: object = None      # Prefer: 0..1 per element, how much of the guide body it is in (None = none)
    prefer_strength: float = 0.0
    prune: bool = False        # Dead ends: remove: dead ends kept empty while optimizing (`Optimizer._prune`)
    conditions: object = None  # (for that: supports and loads, and the skin around them, stay)
    skin: float = 0.0
    min_member: float = 0.0    # Min thickness, mm (at least MIN_MEMBER elements; `radius` without struts only)

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

    def _gauss(self, a, sigma):
        return gaussian_filter(self.xp)(a, sigma=sigma, mode="constant", cval=0.0, truncate=3.0)

    def local(self, xb):
        """How full each element's neighbourhood is (0..1). Outside the design space counts as
        empty, as it is in the part: counted as a share of the design space, a 6 mm wall of a
        thin-walled design space was full whatever the max thickness, and every wall got holes."""
        return self._gauss(xb * self.phi * self.domain, self.local_sigma)

    def local_transpose(self, w):
        return self.domain * self.phi * self._gauss(w, self.local_sigma)

    def section(self, xb):
        """How full each element's wide neighbourhood is with designed material (`free`, see SECTION_PLATE)."""
        return self._gauss(xb * self.phi * self.free, self.section_sigma)

    def section_transpose(self, w):
        return self.free * self.phi * self._gauss(w, self.section_sigma)

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


def _face_tree(pts):
    """A KD-tree of points sampled on faces, with its boxes not shrunk to the points (`compact_nodes`):
    on flat faces those boxes are flat, and queries with a distance bound then search far too much (the
    fine bolted bracket's skin test took 25 s instead of 1.5 s, with the same distances)."""
    return cKDTree(pts, compact_nodes=False, balanced_tree=False)


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
    exempt.ravel()[cells[_face_tree(pts).query(centres, distance_upper_bound=reach + gap + 1e-9)[0] <= reach + gap]] = True
    if keep_fill is not None:
        exempt |= ndimage.distance_transform_edt(keep_fill <= 0.5, sampling=grid.h) <= gap
    return near & ~exempt


def topology_key(space, conditions, material, keep, min_thickness, skin, keep_body, avoid_body, detail, h=None,
                 max_thickness=0.0, members=MEMBERS[0], gap=0.0, smoothness=0.0, symmetry="none", coarse=False,
                 prefer_body=None, prefer_strength=0.0, prune=False, max_section=0.0):
    """Fingerprint of everything an optimization's result depends on (without a max thickness, a max section, a
    gap, smoothness, symmetry, a coarse start, a guide or pruning the same as before they existed, so
    earlier results stay valid)."""
    extra = (float(max_thickness),) if max_thickness else ()
    if max_thickness and members != MEMBERS[0]:
        extra += (members,)
    if max_section:
        extra += (("max section", float(max_section), SECTION_PLATE, SECTION_MIN),)
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
    if prune:
        extra += (("dead ends pruned", PRUNE_EVERY, PRUNE_BETA, DEAD_SHARE, DEAD_IDLE, PRUNE_REGROW),)
    if any(c.kind == "load" and c.load == "bearing" for c in conditions):
        extra += ("bearing skin on the contact side",)  # results from before that change are not reused
    # of the material only its stiffness and Poisson's ratio (a new strength or density kept nothing)
    stiffness = _digest("material", float(material.E), float(material.nu))
    return _digest("topology", TOPO_VERSION, space.key, conditions.key, stiffness, float(keep),
                   float(min_thickness), float(skin), getattr(keep_body, "key", None),
                   getattr(avoid_body, "key", None), detail, h, *extra)


def prepare(space, conditions, material, keep=30.0, min_thickness=0.0, skin=2.0, keep_body=None, avoid_body=None,
            detail="draft", xp=np, cache=None, volume=None, h=None, max_thickness=0.0, members=MEMBERS[0],
            gap=0.0, smoothness=0.0, symmetry="none", prefer_body=None, prefer_strength=0.0, prune=False,
            max_section=0.0):
    """The optimization problem. `xp` may be a function giving numpy or cupy for a number of elements.
    `prune`: dead ends are kept empty as they show (`Optimizer._prune`)."""
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
    try:
        fixed, force, cases, notes = apply_cases(grid, phi, conditions)
        loaded = np.abs(force).sum(axis=0)   # (in any load case)
        if not loaded.any():   # (a compliance of 0: no sensitivities, and the design melts away)
            raise ConditionError("all of the load sits on Fixed Faces, so nothing bends and there is nothing to "
                                 "optimize: put the load on other faces")
        phi, more = _drop_loose(phi, fixed, loaded)
    except ConditionError as err:
        cut = [name for name, on in (("Avoid", avoid_body is not None), ("Gap to faces", gap > 0)) if on]
        if cut and "Fixed Faces, so" not in str(err):
            raise ConditionError(f"{err} (does {' or '.join(cut)} cut it off?)") from err
        raise
    notes = notes + more
    if keep_fill is not None and not (keep_fill > 0.5).any():
        notes.append("the Keep body is outside the design space: it stays in the result, but nothing joins it")
    if len(cases) > 1:
        notes.append(f"{len(cases)} load cases ({', '.join(cases)}): stiff for each, one solve per case and iteration")
    if full is not None:
        full, _ = _drop_loose(full, fixed, loaded)
        left = float(phi.sum()) / float(full.sum())
        if keep / 100 > 0.97 * left:
            raise ConditionError(f"with a gap of {gap:g} mm only {100 * left:.0f}% of the design space is left: "
                                 f"lower Keep % or the gap")
        notes.append(f"gap {gap:g} mm: {100 * (1 - left):.0f}% of the design space along its faces is kept empty")
    if reach > max(float(skin), h):
        notes.append(f"skin {skin:g} mm made as thick as the min thickness, {reach:g} mm")
    room = _room(phi, h) if min_thickness or max_thickness or max_section else None
    if room is not None and min_thickness > room.across:
        raise ConditionError(f"Min thickness {min_thickness:g} mm: no member that thick fits, the design space is "
                             f"{room.across:.3g} mm across")

    # solid skin: elements within `reach` mm of the faces with supports and loads, so bolt holes and
    # contact faces stay
    pts = np.concatenate([_skin_points(c, min(h, reach) / 2) for c in conditions])
    cells = np.flatnonzero(phi > 0)
    centres = grid.origin + (np.column_stack(np.unravel_index(cells, phi.shape)) + 0.5) * h
    near = _face_tree(pts).query(centres, distance_upper_bound=reach + 1e-9)[0] <= reach
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
        cure = (f", or lower Min thickness (the skin is never thinner than it, {reach:g} mm)" if reach > max(skin, h)
                else ", or make Skin thinner" if skin > h else ", or use a finer Detail (it is one element thick)")
        raise ConditionError(f"the solid skin at supports and loads (and Keep) is already {100 * solid_share:.0f}% "
                             f"of the design space: raise Keep % above that{cure}")

    radius = max(float(min_thickness), MIN_MEMBER * h)  # the thinnest member (see ERODED)
    if min_thickness > 0 and min_thickness < MIN_MEMBER * h:
        notes.append(f"min thickness {min_thickness:g} mm is under {MIN_MEMBER:g} elements of {h:g} mm: using "
                     f"{radius:.3g} mm (choose a finer detail for thinner members)")
    if members not in MEMBERS:
        raise ConditionError(f"members must be one of: {', '.join(MEMBERS)}")
    sigma = 0.0
    per_mm = STRUT_SIGMA if members == "struts only" else LOCAL_SIGMA    # (sigma per mm of max thickness)
    # the thickest plate that fits is this x the max thickness: when the whole design space passes, it cannot bind
    if max_thickness > 0 and 2.0 * math.sqrt(2.0) * per_mm * float(erfinv(LOCAL_SHARE)) * max_thickness >= room.thickest:
        notes.append(f"max thickness {max_thickness:g} mm has no effect: the design space is at most "
                     f"{room.thickest:.3g} mm thick")
        max_thickness = 0.0
    if max_thickness > 0:
        thinnest = _thinnest_max(h, min_thickness)
        if max_thickness < thinnest:
            notes.append(f"max thickness {max_thickness:g} mm is too thin for {h:g} mm elements: using "
                         f"{thinnest:.3g} mm (choose a finer detail, or a smaller min thickness)")
            max_thickness = thinnest
        sigma = max_thickness * per_mm / h
        if members == "struts only":
            # measured on a round strut, so a plate may be at most PLATE_SHARE of it; and members at
            # least STRUT_MIN of it: thinner than any plate allowed, so plates cannot form at all
            radius = max(radius, STRUT_MIN * min(max_thickness, room.thickest))
    if max_section >= (room.largest if room is not None else math.inf):
        notes.append(f"max section {max_section:g} mm2 has no effect: the design space's largest section is "
                     f"{room.largest:.3g} mm2")
        max_section = 0.0
    section_sigma, section_share = 0.0, 0.0
    if max_section > 0:
        round_d = section_diameter(max_section)
        thinnest = _thinnest_max(h, min_thickness)
        if round_d < thinnest:
            max_section = math.pi / 4 * thinnest * thinnest
            notes.append(f"max section is too small for {h:g} mm elements: using {max_section:.3g} mm2, {thinnest:.3g} mm "
                         f"across (choose a finer detail, or a smaller min thickness)")
            round_d = thinnest
        # members at least SECTION_MIN of it (of what a Max thickness and the design space leave: 1000 mm2 on a 10 mm
        # thick box asked for 18 mm members, a grey blob that reached no load), and no thicker than one that still
        # carries at this section: a strut needs 1.3 x the filter's radius (see ERODED)
        widest = min(round_d, max_thickness or round_d, room.thickest)
        radius = min(max(radius, SECTION_MIN * widest), round_d / 1.5)
        section_sigma, section_share = section_limit(max_section, radius, 0.5 * room.longest)
        notes.append(f"max section {max_section:g} mm2: members {radius:.2g} to {widest:.2g} mm across, neighbourhoods "
                     f"of {section_sigma:.2g} mm at most {100 * section_share:.0f}% full")
        section_sigma /= h
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
    problem.local_sigma, problem.max_member = sigma, float(max_thickness)
    problem.section_sigma, problem.section_share, problem.section_area = section_sigma, section_share, float(max_section)
    problem.smoothness = float(smoothness)
    problem.prune, problem.conditions, problem.skin = bool(prune), conditions, float(skin)
    problem.min_member = max(float(min_thickness), MIN_MEMBER * h)
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

    def __init__(self, problem, state=None, max_iterations=MAX_ITERATIONS, callback=None):
        """`callback(iteration, residual)`: called by the solve of the full design space (it may raise to stop)."""
        p = self.p = problem
        xp = p.xp
        self.KE = stiffness(problem.material.nu)
        self.precision = p.dtype # the multigrid's (see _solve)
        self.local_max = None    # max thickness: the fullest neighbourhood in the last iteration
        self.section_max = None  # max section: the same of its neighbourhoods
        self.moves = None        # max section: each design variable's move limit (TURN_SHRINK)
        self.last_step = None    # ... and its last step
        self.power = []          # the last iteration's lmax power vectors per multigrid level (warm starts)
        self._design = None      # (iteration, compliance of the design itself)
        self.level = None        # with a coarse start: the detail of this problem (saved with the state)
        self.pruned = None       # bool: dead ends kept empty (problem.prune), None while there are none
        self.dead_ends = 0       # how many were pruned
        self.max_iterations = int(max_iterations)
        self.settled = False     # done because the design stopped changing (not at max_iterations)
        if state is not None:
            self.x = xp.asarray(state["x"], dtype=p.dtype)
            self.iteration = int(state["iteration"])
            self.beta = float(state["beta"])
            self.since_beta = int(state["since_beta"])
            u = np.asarray(state["u"])   # one load case: (X, Y, Z, 3); several: one of those per case
            self.u = None if not u.size else [xp.asarray(v, dtype=np.float64) for v in (u[None] if u.ndim == 4 else u)]
            self.history = [tuple(r) for r in np.asarray(state["history"]).reshape(-1, 4)]
            self.full_compliance = float(state["full_compliance"])
            self.done = bool(state["done"])
            self.local_scale = float(state["local_scale"]) if "local_scale" in state else 0.0
            levels = sorted((int(k[6:]), k) for k in state if k.startswith("power_"))
            self.power = [xp.asarray(state[k]) for _, k in levels]   # (older states: fresh estimates once)
            if "moves" in state and tuple(state["moves"].shape) == tuple(self.x.shape):
                self.moves = xp.asarray(state["moves"], dtype=p.dtype)
                self.last_step = xp.asarray(state["last_step"], dtype=p.dtype)
            if "compliance" in state:
                self._design = (self.iteration, float(state["compliance"]))
            if "pruned" in state and tuple(state["pruned"].shape) == tuple(self.x.shape):
                self.pruned = xp.asarray(state["pruned"], dtype=bool)
                self.dead_ends = int(state["dead_ends"])
            self.settled = settled(state)
            if "wide" in state:
                self.precision = np.float64
            if self.done and not self.settled and self.iteration < self.max_iterations:
                self.done = False   # stopped by a lower Max iterations: on from there
        else:
            self.local_scale = 0.0   # max thickness: max / p-mean of the local densities (0 = not yet)
            # start uniform at the target volume (solid parts full)
            free_share = (p.target * p.total - float(xp.sum(p.phi * p.solid))) / float(xp.sum(p.phi * p.free))
            self.x = (xp.where(p.solid, 1.0, 0.0) + p.free * min(max(free_share, 0.0), 1.0)).astype(p.dtype)
            self.iteration, self.beta, self.since_beta = 0, BETA_START, 0
            self.u, self.history, self.done = None, [], False
            # the stiffness of the whole design space filled (a gap filled too: what it costs shows)
            self.full_compliance = self._compliance_of(xp.ones_like(self.x), callback, phi=p.full_phi)[0]

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
        """One multigrid hierarchy for the stiffness `scale` and a solve per load case on it (warm
        from the list `u0`): (hierarchy, [u], iterations in all). On the GPU a float32 multigrid that gives up
        hands over to float64 for the rest of the run: its rounding stops it on slender parts and sheets beside
        soft elements (a 200 x 200 x 1 sheet at iteration 1, a 500 x 5 x 5 bar at 50), and float64 ran all 125
        iterations about as fast per iteration. And float64 stopped by the iteration limit counts when within
        SOLVE_NEAR x the tolerance (a 300 x 2 x 2 bar, 2.5e-2 after 1000 iterations: enough for a step, and the
        next one starts from it)."""
        p = self.p

        def solved(H, k, force):
            try:
                return H.solve(force, u0=None if u0 is None else u0[k], tol=SOLVE_TOL, callback=callback,
                               maxiter=WARM_MAXITER if power else 1000)
            except NotConverged as err:
                if power or H.dtype != np.float64 or not err.residual <= SOLVE_NEAR * SOLVE_TOL:
                    raise
                return err.u, err.iterations, err.residual

        while True:
            H = Hierarchy(scale, self.KE, p.fixed, p.grid.coarsenings, p.xp, self.precision, power=power)
            us, its = [], 0
            try:
                for k, force in enumerate(p.force):
                    u, n, _ = solved(H, k, force)
                    us.append(u)
                    its += n
                return H, us, its
            except NotConverged:
                if power or self.precision == np.float64:   # (warm: `step` tries again from fresh estimates)
                    raise
                self.precision = np.float64

    def _fused_volume(self, x, ndc, ndv, ndg, lo_x, hi_x):
        """On the GPU: the OC root finding's designs and volumes in fused kernels (gpu.Volume), else None."""
        p = self.p
        if p.xp is np or not solver.GPU_KERNELS:
            return None
        from .gpu import Volume
        taps = int(p.kernel.size)
        return Volume(p, x, ndc, ndv, ndg, lo_x, hi_x, self.beta, ETA,
                      fft=_convolve(p.xp, taps) if taps >= FFT_TAPS else None)

    def _compliance_of(self, xb, callback=None, phi=None):
        """(compliance, summed over the load cases; [u])."""
        scale, _ = self._scale(xb, phi)
        H, us, its = self._solve(scale, callback=callback)
        level = H.exact or H.levels[0]
        energy = sum(level.unit_energies(u) for u in us)
        return float(self.p.xp.sum(self.p.xp.asarray(scale, dtype=np.float64) * energy)), us

    def _upper(self, x, move):
        """The most each design variable may grow to this iteration (pruned dead ends: nothing)."""
        hi = self.p.xp.minimum(x + move, 1.0)
        return hi if self.pruned is None else self.p.xp.where(self.pruned, 0.0, hi)

    def _prune(self):
        """Dead ends of the design as it is, by a solve with its own stiffness (its sensitivities come
        from the eroded design, where a fin strains with the member it hangs on), quiet patches of member
        surfaces included: what lies farther than Min thickness from what stays is emptied and kept empty
        from now on; nearer, it is left at PRUNE_REGROW, so members may still grow there (to Min thickness,
        or where load comes later). Returns how many."""
        p = self.p
        xp = p.xp
        xb = self.physical(self.x)[1]
        scale, rho = self._scale(xb)
        H, us, _ = self._solve(scale)
        level = H.exact or H.levels[0]
        # strain energy density E rho^p u^T KE u (per h^3): its square root goes as the stress; with load
        # cases the most of any (material that works in one case is no dead end)
        energy = level.unit_energies(us[0])
        for u in us[1:]:
            energy = xp.maximum(energy, level.unit_energies(u))
        stress = np.sqrt(np.maximum(to_host(energy * xp.asarray(scale, dtype=energy.dtype)), 0.0))
        del H, us, energy
        density = np.asarray(to_host(rho), dtype=np.float32)
        found, count = dead_ends(density, stress, p.grid.origin, p.grid.h, p.conditions, p.skin)
        cut = (found > 0) & np.asarray(to_host(p.free), dtype=bool)
        if p.mirror:   # (only with its mirror image: the halves stay images, and the other one may be working)
            cut = to_host(p.symmetric(xp.asarray(cut, dtype=p.dtype))) >= 1.0 - 1e-6
            count = count if cut.any() else 0
        if count:
            far = ndimage.distance_transform_edt(~((density >= 0.5) & ~cut), sampling=p.grid.h) > p.min_member
            lock = xp.asarray(cut & far)
            self.pruned = lock if self.pruned is None else self.pruned | lock
            near = xp.asarray(cut & ~far)
            self.x = xp.where(self.pruned, 0.0, xp.where(near, xp.minimum(self.x, PRUNE_REGROW), self.x))
            self.x = self.x.astype(p.dtype)
            self.dead_ends += count
        return count

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
        lo_x, hi_x = xp.maximum(x - move, 0.0), self._upper(x, move)
        fused = self._fused_volume(x, ratio, xp.ones_like(ratio), xp.zeros_like(ratio), lo_x, hi_x)
        lo, hi = 1e-9, 1e9
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            if fused is not None:   # (the same design: sqrt(ratio / max(mid * 1 + 0 * 0, 1e-30)))
                volume = fused.volume(mid, 0.0)
            else:
                xn = xp.where(free, xp.clip(x * xp.sqrt(ratio / mid), lo_x, hi_x), x).astype(p.dtype)
                volume = p.volume(self.physical(xn)[1])
            if volume > p.target:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1 + 1e-5:
                break
        return fused.design(mid, 0.0) if fused is not None else xn

    def _local_constraint(self, xb):
        """Max thickness and Max section: g = max (local density / its limit) - 1 (<= 0 when no neighbourhood
        is too full), as an adaptively scaled p-mean of both so it has a gradient; returns (g, dg/dxb)."""
        p = self.p
        xp = p.xp
        free = p.free
        parts = []    # (local densities / their limit, the limit, the transpose)
        if p.local_sigma:
            parts.append((p.local(xb) / LOCAL_SHARE, LOCAL_SHARE, p.local_transpose))
        if p.section_sigma:
            parts.append((p.section(xb) / p.section_share, p.section_share, p.section_transpose))
        free_r = [rho[free] for rho, _, _ in parts]
        tops = [float(xp.max(r)) if r.size else 0.0 for r in free_r]
        if p.local_sigma:
            self.local_max = tops[0] * LOCAL_SHARE
        if p.section_sigma:
            self.section_max = tops[-1] * p.section_share
        n = max(sum(int(r.size) for r in free_r), 1)
        mean_p = sum(float(xp.sum(r ** LOCAL_P)) for r in free_r) / n
        pmean = mean_p ** (1 / LOCAL_P) if mean_p > 0 else 0.0
        if pmean <= 0:
            return -1.0, xp.zeros_like(xb)
        scale = max(tops) / pmean
        self.local_scale = scale if self.local_scale <= 0 else 0.5 * (self.local_scale + scale)
        g = self.local_scale * pmean - 1.0
        dg = 0.0
        for rho, limit, transpose in parts:
            w = xp.where(free, (self.local_scale / limit) * pmean ** (1 - LOCAL_P) * rho ** (LOCAL_P - 1) / n, 0.0)
            dg = dg + transpose(w.astype(p.dtype))
        return g, dg

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
        if self.moves is not None:
            move = xp.minimum(self.moves, move)
        lo_x, hi_x = xp.maximum(x - move, 0.0), self._upper(x, move)
        fused = self._fused_volume(x, ndc, ndv, ndg, lo_x, hi_x)

        def new_x(l1, l2):
            if fused is not None:
                return fused.design(l1, l2)
            return xp.where(free, xp.clip(x * xp.sqrt(ndc / xp.maximum(l1 * ndv + l2 * ndg, 1e-30)), lo_x, hi_x),
                            x).astype(p.dtype)

        def volume(l1, l2):
            if fused is not None:
                return fused.volume(l1, l2)
            return p.volume(self.physical(new_x(l1, l2))[1])

        def volume_ok(l2):  # l1 so the volume is the target (PRICE_FLOOR if Keep % does not fit)
            l1 = _decreasing_root(lambda l1: volume(l1, l2) - p.target, lo=PRICE_FLOOR, ftol=1e-5)
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

    def _track_moves(self, step):
        """Max section: smaller move limits where a design variable turned back (see TURN_SHRINK)."""
        xp = self.p.xp
        if self.moves is None:
            self.moves = xp.full_like(step, MOVE)
        else:
            turned = step * self.last_step < 0
            moves = xp.where(turned, self.moves * TURN_SHRINK, self.moves * TURN_GROW)
            self.moves = xp.clip(moves, TURN_FLOOR, MOVE).astype(self.p.dtype)
        self.last_step = step

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
        # load cases: the sum of their compliances is minimized (stiff in each), so their energies add
        energy = sum(level.unit_energies(uk) for uk in u).astype(p.dtype)
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
        if p.local_sigma or p.section_sigma:
            g, dg_xb = self._local_constraint(xb)
            dg = p.filter_transpose(xp.where(free, dg_xb, 0.0) * slope)
            if p.mirror:
                dg = p.symmetric(dg)
            x_new = self._update_two(self.x, dc, dv, dg, p.volume(xb), g)
        else:
            x_new = self._update(self.x, dc, dv)
        if p.section_sigma:
            self._track_moves(x_new - self.x)
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
            self.done = self.settled = True
        # (also in the last iteration Max iterations allows: raised, the run goes on as if it had been all along)
        if p.prune and not self.settled and self.beta >= PRUNE_BETA and self.iteration % PRUNE_EVERY == 0:
            self._prune()
        if self.iteration >= self.max_iterations:
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
        self.done = self.iteration >= self.max_iterations     # (Max iterations is for all levels together)
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
        # the lmax vectors too, while the run can continue (also at Max iterations): it then continues exactly
        extra = {} if self.settled else {f"power_{i}": np.asarray(to_host(v)) for i, v in enumerate(self.power)}
        if self._design is not None and self._design[0] == self.iteration:  # when it was worked out
            extra["compliance"] = np.array(self._design[1])
        if self.level is not None:   # the detail a coarse start was at
            extra["level"] = np.array(self.level)
        if self.pruned is not None:
            extra["pruned"] = np.asarray(to_host(self.pruned), dtype=bool)
            extra["dead_ends"] = np.array(self.dead_ends)
        if self.precision != p.dtype:
            extra["wide"] = np.array(True)
        if self.moves is not None and not self.settled:
            extra["moves"] = np.asarray(to_host(self.moves), dtype=np.float32)
            extra["last_step"] = np.asarray(to_host(self.last_step), dtype=np.float32)
        return {**extra, "x": np.asarray(to_host(self.x), dtype=np.float32), "iteration": np.array(self.iteration),
                "beta": np.array(self.beta), "since_beta": np.array(self.since_beta),
                # float64: a resumed run then gives exactly the uninterrupted one's result
                "u": (np.zeros(0) if self.u is None else np.asarray(to_host(self.u[0]), dtype=np.float64)
                      if len(self.u) == 1 else np.stack([np.asarray(to_host(u), dtype=np.float64) for u in self.u])),
                "history": np.asarray(self.history, dtype=np.float64).reshape(-1, 4),
                "full_compliance": np.array(self.full_compliance), "done": np.array(self.done),
                "settled": np.array(self.settled), "local_scale": np.array(self.local_scale),
                "stopped": np.array(stopped), "density": self.density(),
                "grid": np.array([*p.grid.origin, p.grid.h, *p.grid.shape], dtype=np.float64),
                "phi": np.asarray(to_host(p.phi), dtype=np.float32),
                "space": np.asarray(p.space if p.space is not None else to_host(p.phi) > 0, dtype=bool)}

    def summary(self):
        return summary(self.history, self.full_compliance, self.iteration,
                       self.design_compliance() if self.history else None)


def settled(state):
    """Whether a saved run ended because its design stopped changing, not at its Max iterations (states
    from before there was a setting: done under the fixed MAX_ITERATIONS of then, 125)."""
    if "settled" in state:
        return bool(state["settled"])
    return bool(state["done"]) and int(state["iteration"]) < 125


def finished(state, max_iterations):
    """A saved run has nothing more to do with this Max iterations."""
    return bool(state["done"]) and (settled(state) or int(state["iteration"]) >= int(max_iterations))


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
# Thin tips (`cut_tips`), as shares of Min thickness: no ball of TIP_SHARE fits in them (not 1: `thicken` stops
# just short of it), and their mean cross-section is under TIP_SECTION x Min^2. The bracket's tips reach 0.34 (the
# same tip on the other side of a hole 0.31, joined to a rib: how a region falls moves it, so leave a margin); the
# first plate as thin is at 0.49, flanges and shells at 0.6-2. See FINDINGS.md, "Thin tips".
TIP_SHARE = 0.8
TIP_SECTION = 0.45
# How far past the balls a tip reaches, in their radii. Points and the loose bits by them: 4.1-7.1 on three parts; sharp
# edges, as a slanted plate's rim at a hole, and strips of skin: up to 2.44 (taken at 1, the rims came out wavy).
TIP_OUT = 3.0
# The most of a support's or load's faces a tip may take: a tip of a skin takes a corner of them, but a pin or a tab that
# a load stands on is as thin (a 4 mm pin at draft's 6 mm: the load went with it, safety factor 10 -> 0.81).
TIP_FACE_SHARE = 0.1
# Dead ends: branches and fins that end in nothing carry next to no load, but the optimizer keeps them when
# material is cheap (Max thickness leaving no room for Keep %: bolted bracket, 22 of them, 3% of the part;
# a dearer floor price took working members as fast: 0.3 kept 6.5% instead of 11.2%, 9.6 mm deflection
# instead of 2.5). Regions of the part whose von Mises stress is under DEAD_SHARE of the median that reach
# its surface and meet the rest of it through one patch only are emptied, then it is smoothed again. The
# optimizer's own strains do not show them: its stiffness is the eroded design's, so thin fins there
# strain with their members (the bolted bracket's biggest fin at 1.6 x the median energy). A region that
# meets the rest in more places goes too when nothing passes through it (90% of it under DEAD_IDLE of the
# median: a 34 mm3 finger touching another member along its side). Cutting a branch leaves its root as
# a new, smaller dead end: DEAD_PASSES rounds of analysing and trimming.
DEAD_SHARE = 0.1
DEAD_IDLE = 0.02
DEAD_PASSES = 2
# The optimizer prunes them too (Dead ends: remove): every PRUNE_EVERY iterations once the projection is
# PRUNE_BETA sharp, a solve with the design's own stiffness (projected densities, SIMP: not the eroded
# design its sensitivities come from) shows them, and they are kept empty from then on, so the material
# goes to members that carry. Bolted bracket: every 10 from 8, 11.7% kept, deflection 2.45 mm; every 5
# from 4, 12.3%, 1.90 mm, safety 0.45 (unpruned 11.2%, 2.48 mm, 0.11). It keeps growing new stubs (30-70
# at each check late in the run), so `dead_ends` trims what the last iterations left.
PRUNE_EVERY = 5
PRUNE_BETA = 4.0
# It prunes to the last iteration. With Keep % out of reach the design never settles, and runs past 125
# iterations came out worse (bolted bracket, Min 1.5: 2.12 mm deflection at 125, 2.57 at 200; pruning only
# until 25 iterations at the sharpest projection, 2.76: the stubs grow back; unpruned 2.49 / 2.46).
# See FINDINGS.md, "Max iterations".
# A design variable at exactly 0 never grows again in the OC update (x times a factor): pruned material within
# Min thickness of a member is left at this instead, invisible but free to grow back. Locking it all (zeroing
# whole dead ends and quiet patches of member surfaces) made the bolted bracket stiffer at 125 iterations
# (1.90 mm deflection against 2.48) but worse the longer it ran (2.10 / 2.30 / 2.35 mm at 125 / 200 / 300 with
# a band beside thin members): 3871 of the 7123 locked elements lay against members, hemming them in.
PRUNE_REGROW = 1e-3
LIGHT_BLUR = 1.0      # elements: the most smoothing thin members get, whatever Smoothing is
# GPU memory density_mesh takes per sample of its fine grid (UPSAMPLE x finer than the elements), for
# `_smoothing_module`: a fine strut web's 39.7M samples peaked at 56 B in CuPy's pool (1.6 GB in use, during a
# distance transform) and 59 B on the card, with the pool's free blocks given back between the steps (88 B without)
SMOOTH_GPU_BYTES = 64
SPECK = 0.01          # pieces smaller than this share of the largest are dropped
# More than this share of a load's force on no piece joined to a support: the card says the shape does not reach it.
# (By force: a bearing load's unloaded sides are often left empty. A load on two holes has half of it on each.)
REACH_MISS = 0.2


def dead_ends(density, stress, origin, h, conditions, skin, share=DEAD_SHARE, idle=DEAD_IDLE, reach=0.0):
    """The elements of the design (density >= 0.5) that carry next to nothing and lead nowhere, as
    region numbers 1..count (0 elsewhere), and count. `stress`: per element, e.g. a Static Analysis of
    the body made of them sampled at their centres. A region of elements whose stress is under `share` x
    the median of the design's counts when it sticks out farther than `reach` mm (Min thickness) from the
    rest of the design and meets it through one connected patch (or none: a branch or fin hanging off a
    member), or when 90% of it is under `idle` x the median (nothing passes through it). A core around a
    member's neutral axis reaches no surface; a lightly loaded member meets the rest at both ends; a
    quiet patch of a member's surface sticks out nowhere (counted, the optimizer shaved members: 3871 of
    the 7123 elements it had locked after 300 iterations lay against them). Elements within the skin (at
    least an element) and an element more of a support's or load's faces stay."""
    solid = np.asarray(density) >= 0.5
    if not solid.any():
        return np.zeros(solid.shape, dtype=np.int32), 0
    stress = np.asarray(stress)
    median = float(np.median(stress[solid]))
    low = solid & (stress < share * median)
    if conditions is not None and len(conditions) and low.any():
        faces = max(float(skin), h) + h
        pts = np.concatenate([sample_triangles(c.triangles, h / 2, stretch=False)[0] for c in conditions])
        cells = np.argwhere(low)
        near = _face_tree(pts).query(np.asarray(origin) + (cells + 0.5) * h, distance_upper_bound=faces)[0] <= faces
        low[tuple(cells[near].T)] = False
    rest = solid & ~low
    out_from_rest = ndimage.distance_transform_edt(~rest, sampling=h) if reach > 0 and rest.any() else None
    cube = np.ones((3, 3, 3), dtype=bool)
    pieces, _ = ndimage.label(low, cube)
    out = np.zeros(solid.shape, dtype=np.int32)
    count = 0
    for i, box in enumerate(ndimage.find_objects(pieces), start=1):
        box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
        region = pieces[box] == i
        if out_from_rest is not None and float(out_from_rest[box][region].max()) <= reach:
            continue
        around = ndimage.binary_dilation(region, cube)
        if not (around & ~solid[box]).any():
            continue
        if (ndimage.label(around & rest[box], cube)[1] <= 1     # (none: a loose bit of its own)
                or np.percentile(stress[box][region], 90) < idle * median):
            count += 1
            out[box][region] = count
    return out, count


def _modules(a):
    """(numpy or CuPy, scipy.ndimage or cupyx.scipy.ndimage) for array `a`: the smoothing's helpers run on
    the device their arrays are on (density_mesh, `_smoothing_module`)."""
    xp = xp_of(a)
    if xp is np:
        return np, ndimage
    import cupyx.scipy.ndimage as nd
    return xp, nd


def _gradient_norm(a, voxel):
    """|grad a| (per mm), one axis at a time: the same numbers as sqrt(sum(d * d for d in gradient(a))),
    without its three gradients at once."""
    xp = xp_of(a)
    total = None
    for axis in range(a.ndim):
        d = xp.gradient(a, voxel, axis=axis)
        d *= d
        if total is None:
            total = d
        else:
            total += d
        del d
    return xp.sqrt(total, out=total)


def _label_max(values, labels, count):
    """The largest of `values` on each of the labels 1..count, on the host. On the GPU one scatter pass:
    cupyx's ndimage.maximum took 95 ms for 4M samples and 4k labels, growing with both, and a GPU kernel
    running for seconds makes Windows reset the display driver."""
    xp = xp_of(values)
    if xp is np:
        return ndimage.maximum(values, labels, np.arange(1, count + 1))
    import cupyx
    top = xp.full(count + 1, -np.inf, dtype=values.dtype)
    cupyx.scatter_max(top, labels.ravel(), values.ravel())
    return to_host(top[1:])


def thicken(values, level, inside, voxel, thickness, passes=THICKEN_PASSES, cancel=None):
    """`values` (samples `voxel` mm apart, solid where >= level) with every member at least
    `thickness` thick, by moving the surface out where a member is thinner, as smoothly as the values
    run: a member's half thickness near a point is the deepest point of the solid within reach (the
    design space's faces counted as empty, as they are in the part); what it lacks of thickness / 2,
    blurred, is added to the values as that many mm along their gradient, a few passes. Thick members
    are left alone. (Balls on the middle of thin members and filling in from the faces, done on the
    samples, left steps of a sample and knobs all over the surface.) Plates lying on a face grow on
    both sides, so only half inwards: the optimizer's eroded design counts the faces as empty, so it
    does not make them thinner than elsewhere. Returns the new values and how many samples became
    solid inside. `cancel()` is called before each pass (it may raise to stop)."""
    xp, nd = _modules(values)
    r = thickness / 2
    # within a whole thickness, so a point by a thick member sees its depth (with r, a block beside a
    # thin plate grew by 1 mm)
    reach = int(math.ceil(thickness / voxel))     # samples
    before = int(((values >= level) & inside).sum())
    out = values.astype(np.float32)
    for _ in range(passes):
        if cancel is not None:
            cancel()
        solid = (out >= level) & inside
        # a sample's distance runs to the middle of the nearest sample outside, half a sample past the surface
        # (in place, the arrays let go as soon as they are used: on the GPU memory is short)
        depth = nd.distance_transform_edt(solid, sampling=voxel)
        del solid
        depth -= 0.5 * voxel
        half = nd.maximum_filter(depth, size=2 * reach + 1)
        del depth
        lack = r - half
        xp.clip(lack, 0.0, r, out=lack)
        lack[half <= 0] = 0.0
        del half
        lack = lack.astype(np.float32)
        if float(lack.max()) < 0.05 * voxel:
            break
        lack = nd.gaussian_filter(lack, reach / 4)
        grad = _gradient_norm(out, voxel)   # values per mm
        # half of it a pass: the values flatten out away from the surface, so a full step overshoots
        # (a 2 mm band became 4 mm instead of 3)
        out = out + 0.5 * grad * lack
        del grad, lack
    return out.astype(values.dtype), int(((out >= level) & inside).sum()) - before


def _free_all_round(region, empty):
    """Whether what is empty around `region` is one space. With empty on two sides that do not meet it around
    the region, the region is a thin spot in a plate: taking it off makes a hole."""
    cube = np.ones((3, 3, 3), dtype=bool)
    return ndimage.label(ndimage.binary_dilation(region, cube, iterations=2) & empty, cube)[1] <= 1


def _around(pieces, points, start, voxel):
    """The numbers of the `pieces` in the 27 samples around each of `points` (mm): (n, 27)."""
    at = np.rint((np.asarray(points) - start) / voxel).astype(np.int64)
    top = np.array(pieces.shape) - 1
    return np.stack([pieces[tuple(np.clip(at + (x, y, z), 0, top).T)]
                     for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)], axis=1).astype(np.int64)


def _shares(pieces, count, points, start, voxel):
    """The share of `points` (mm) on each of the numbered `pieces` (or a sample from one), index = number."""
    if not len(points):
        return np.zeros(count + 1)
    on = _around(pieces, points, start, voxel)
    pairs = np.unique(np.arange(len(on))[:, None] * (count + 1) + on)   # (a point counts once for each piece)
    return np.bincount(pairs % (count + 1), minlength=count + 1) / len(on)


def _unreached(pieces, count, dropped, conditions, start, voxel):
    """[(label, share of its force)] of the loads the numbered `pieces` leave in the air: over REACH_MISS of the
    force on no piece joined to a support (a piece that stays, not one of `dropped`). A load on a loose piece has
    its material and is carried by nothing, and a load's own skin, cut off from the rest, went as a speck: all that
    was left of it were arrows in the air, and at most an analysis that could not be done."""
    stays = np.ones(count + 1, dtype=bool)
    stays[[0, *np.asarray(dropped, dtype=np.int64).tolist()]] = False
    held = np.zeros(count + 1, dtype=bool)
    for c in conditions:
        pts = sample_triangles(c.triangles, voxel, stretch=False)[0] if c.kind == "fixed" else ()
        if len(pts):
            held[np.unique(_around(pieces, pts, start, voxel))] = True
    held &= stays
    if not held.any() and count:   # (no support reached either: then the main piece is the shape)
        sizes = np.bincount(pieces.ravel(), minlength=count + 1)
        held[int(np.argmax(np.where(stays, sizes, 0)))] = True
        held &= stays
    missed = []
    for c in conditions:
        if c.kind == "fixed" or not len(c.triangles):
            continue
        pts, areas, normals = sample_triangles(c.triangles, voxel, stretch=False)
        force = np.linalg.norm(c.traction(normals, pts) * areas[:, None], axis=1)
        if force.sum() > 0:
            share = 1.0 - float(force[held[_around(pieces, pts, start, voxel)].any(axis=1)].sum() / force.sum())
            if share > REACH_MISS:
                missed.append((c.label or "a load", share))
    return missed


def cut_tips(values, level, inside, voxel, thickness, per_mm=1.0, start=np.zeros(3), conditions=(), cancel=None):
    """`values` (samples `voxel` mm apart, solid where >= level) without the thin tips the design space's
    faces leave (`inside`: the samples in it): a skin or a plate whose edge runs into a hole at a shallow
    angle ends in a point, a skin wrapped round a fold in a rib. `thicken` cannot help those, since the crop
    takes off what it adds. A tip: solid farther than a little from every ball of TIP_SHARE x `thickness`
    that fits in it, at least TIP_OUT of the ball's radii out (a sharp edge is not), of a small cross-section
    (TIP_SECTION; a plate as thin is a skin, kept), hanging on the rest by one patch at most (with two it joins two
    thicker places: a thin member, kept), with empty space all round it, so that taking it off never makes a hole
    in a plate, and with little of a support's or load's faces on it (TIP_FACE_SHARE; `conditions`, `start`: the
    first sample, mm). It is cut along that distance, so its stump is round. `per_mm`:
    how fast the values run per mm. Returns the new values and how many tips went; `cancel()` is called
    between the steps."""
    xp, nd = _modules(values)
    step = cancel if cancel is not None else (lambda: None)
    r = TIP_SHARE * thickness / 2
    solid = (values >= level) & inside
    # (a sample's distance runs half a sample past the surface, as in `thicken`)
    core = nd.distance_transform_edt(solid, sampling=voxel) - 0.5 * voxel >= r
    if not bool(core.any()):
        return values, 0
    step()
    out = (nd.distance_transform_edt(~core, sampling=voxel) - r).astype(np.float32)
    del core
    reach = max(0.5 * r, 1.1 * voxel)  # (a right angle's corner is 0.41 r out of the balls: edges stay)
    far = solid & (out > reach)
    cube = np.ones((3, 3, 3), dtype=bool)
    pieces, count = nd.label(far, xp.asarray(cube))
    count = int(count)
    if not count:
        return values, 0
    step()
    highest = _label_max(out, pieces, count)
    rest = solid & ~far
    del far
    # (each piece is looked at in its own box, on the host: ndimage's find_objects and small labellings)
    pieces_h, rest_h, solid_h = to_host(pieces), to_host(rest), to_host(solid)
    del rest, solid
    tips = np.zeros(count + 1, dtype=bool)
    face = ndimage.generate_binary_structure(3, 1)
    for i, box in enumerate(ndimage.find_objects(pieces_h), start=1):
        if highest[i - 1] < TIP_OUT * r:
            continue
        box = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in box)
        region = pieces_h[box] == i
        cells = np.argwhere(region).astype(np.float64) * voxel
        # along its longest axis (a curved one looks shorter, so thicker: left alone rather than cut)
        length = math.sqrt(12.0 * max(float(np.linalg.eigvalsh(np.cov(cells.T))[-1]), 0.0)) if len(cells) > 3 else 0.0
        if len(cells) * voxel ** 3 / max(length, voxel) > TIP_SECTION * thickness ** 2:
            continue
        if ndimage.label(ndimage.binary_dilation(region, cube) & rest_h[box], cube)[1] > 1:
            continue
        tips[i] = _free_all_round(region, ~solid_h[box])
    del rest_h, solid_h
    if tips.any():
        for condition in conditions or ():
            tips[_shares(pieces_h, count, _skin_points(condition, voxel), start, voxel) > TIP_FACE_SHARE] = False
    del pieces_h
    if not tips.any():
        return values, 0
    step()
    sigma = 1.0                        # samples: the distance is stepped by them, and the stump a surface
    grow = int(math.ceil(reach / voxel + 1 + 3 * sigma))
    taken = xp.asarray(tips)[pieces]
    # (cupyx dilates repeatedly only by brute force: one short kernel an iteration)
    near = nd.binary_dilation(taken, xp.asarray(face), iterations=grow, **({} if xp is np else {"brute_force": True}))
    near &= taken | (pieces == 0)
    cut = nd.gaussian_filter(reach + 0.5 * voxel - out, sigma)
    return xp.where(near, xp.minimum(values, level + per_mm * cut), values).astype(values.dtype), int(tips.sum())


def _surface_distance(values, voxel):
    """Signed distance (mm, negative inside) to the 0.5 surface of densities sampled `voxel` apart:
    by distance transforms, and within a sample of the surface from the densities themselves, so the
    surface is not stepped by the samples."""
    xp, nd = _modules(values)
    solid = values >= 0.5
    # (one distance transform at a time, each cast to float32 where it is used: the same numbers as one where())
    deep = nd.distance_transform_edt(solid, sampling=voxel)
    dist = (0.5 * voxel - deep).astype(np.float32)
    del deep
    out = nd.distance_transform_edt(~solid, sampling=voxel)
    out -= 0.5 * voxel
    dist[~solid] = out.astype(np.float32)[~solid]
    del out, solid
    grad = _gradient_norm(values, voxel)
    near = (xp.abs(dist) < voxel) & (grad > 1e-3 / voxel)
    dist[near] = xp.clip((0.5 - values[near]) / grad[near], -voxel, voxel)
    return dist


def density_mesh(density, phi, origin, h, upsample=UPSAMPLE, smooth=SMOOTH_ITERATIONS, blur=SMOOTHING, space=None,
                 min_thickness=0.0, crop=None, cancel=None, conditions=(), tips_under=None):
    """The surface of per-element densities as a closed mesh (world coordinates, host arrays).
    Densities are extended outward past the design space (`space`, bool per element; phi > 0 when
    not given), so where material reaches it the surface runs past it, and intersecting with the
    design space gives back its exact CAD faces. Avoid regions and a gap inside it stay empty (with
    phi > 0 as the design space they were filled from the side, 2-4 elements deep). `blur`:
    Gaussian sigma in elements applied to the distance to the densities' 0.5 surface (rounds element
    steps and waves, members keep their thickness; thin members get at most LIGHT_BLUR).
    `min_thickness` (mm): what is still thinner than that is thickened (`thicken`), and the thin tips
    the design space's faces leave are cut (`cut_tips`), thin measured against `tips_under` (mm) when given.
    `conditions`: the supports and loads, whose faces those tips must leave and which the mesh's metadata says the
    shape does not reach (`unreached`). `crop`: the implicit body the mesh will be
    intersected with (the design space less Avoid): tips and loose bits are then found in it exactly,
    and outside it the surface runs straight out (see below). `cancel()` is called between the steps
    (it may raise to stop). Its array work (distance transforms, filters, labelling, marching cubes) runs on
    the GPU when there is room for it (`_smoothing_module`; a fine grid's 35M samples took ~95 s on the CPU,
    one distance transform 7.2 s there and 0.02 s on the GPU, the same to 1e-16), else on the CPU."""
    args = (density, phi, origin, h, upsample, smooth, blur, space, min_thickness, crop, cancel, conditions, tips_under)
    xp = _smoothing_module(smoothing_samples(np.shape(density), blur, upsample))
    if xp is not np:
        try:
            return _density_mesh(*args, xp=xp)
        except Exception as err:   # (out of memory after all: other programs took some meanwhile)
            if not backend.is_out_of_memory(err):
                raise
        finally:
            backend.release_gpu_memory()
    return _density_mesh(*args, xp=np)


def smoothing_samples(shape, blur=SMOOTHING, upsample=UPSAMPLE):
    """Samples of density_mesh's fine grid for densities of `shape` (padded by GROW_OUTSIDE + the blur and a
    ring, then UPSAMPLE x finer): its GPU memory is SMOOTH_GPU_BYTES each."""
    g = GROW_OUTSIDE + int(math.ceil(blur))
    return int(np.prod([(n + 2 * g + 2 - 1) * upsample + 1 for n in shape]))


def _smoothing_module(samples):
    """CuPy for density_mesh when the GPU is on and has SMOOTH_GPU_BYTES per sample free beside GPU_HEADROOM
    (this process's pooled blocks given back first), else numpy."""
    if not backend.use_gpu_for(samples):
        return np
    backend.release_gpu_memory()
    free, _ = backend.gpu_memory()
    return backend.cupy() if samples * SMOOTH_GPU_BYTES + backend.GPU_HEADROOM < free else np


def _density_mesh(density, phi, origin, h, upsample, smooth, blur, space, min_thickness, crop, cancel, conditions,
                  tips_under, xp=np):
    """density_mesh on numpy or CuPy arrays (the mesh and its metadata on the host either way)."""
    import trimesh
    from ..grid import Grid, upsample_linear
    from ..meshing import field_to_mesh
    xp, nd = _modules(xp.zeros(1))
    if xp is not np:   # between the steps CuPy's pool gives back the blocks it keeps: those that do not fit the
        def cancel(stop=cancel):   # next step's arrays doubled what the GPU held (3.3 GB for 1.6 in use)
            if stop is not None:
                stop()
            backend.release_gpu_memory()
    step = cancel if cancel is not None else (lambda: None)
    g = GROW_OUTSIDE + int(math.ceil(blur))  # the blur must not round off faces cut by the design space
    # room to grow into: the FE grid starts right at the part's lowest corner
    region = np.asarray(space, dtype=bool) if space is not None else np.asarray(phi) > 0
    inside = xp.pad(xp.asarray(region), g, mode="constant", constant_values=False)
    ext = xp.pad(xp.asarray(density, dtype=np.float32), g, mode="constant")
    # outside the design space each element takes the density of the nearest one inside: material
    # runs straight out through the boundary, empty stretches of it stay empty (growing by dilation
    # spread it sideways too: 3 elements of that closed a beam's open sides into a cavity)
    distance, nearest = nd.distance_transform_edt(~inside, return_indices=True)
    ext = xp.where(distance <= g, ext[tuple(nearest)], 0.0).astype(np.float32)
    del nearest, distance
    values = xp.pad(ext, 1, mode="constant", constant_values=0.0)  # a ring of "outside"
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
        step()
        dist = _surface_distance(values, voxel)
        step()
        sigma = blur * h
        values = -nd.gaussian_filter(dist, blur * upsample)        # mm, positive inside
        if blur > LIGHT_BLUR:
            light = -nd.gaussian_filter(dist, LIGHT_BLUR * upsample)
            half = nd.maximum_filter(xp.maximum(light, 0.0), size=2 * int(math.ceil(sigma / voxel)) + 1)
            weight = nd.gaussian_filter(xp.clip(2.0 * half / sigma - 1.0, 0.0, 1.0), 0.5 * sigma / voxel)
            values = weight * values + (1.0 - weight) * light
            del light, half, weight
        values = values.astype(np.float32)
        step()
        # the surface where the material in the design space stays as much as the densities give (a
        # strong blur rounds members off: the bolted bracket lost 10% at Smoothing 3), within half an element
        space = up(xp.pad(inside, 1, mode="constant").astype(np.float32)) >= 0.5
        flat = values[space]
        count = int(xp.count_nonzero(dist[space] <= 0.0))
        del dist
        level = 0.0
        if 0 < count < flat.size:
            level = float(xp.clip(xp.partition(flat, flat.size - count)[flat.size - count], -0.5 * h, 0.5 * h))
        del flat, space
    thickened = 0.0
    # inside: where the design space fills half an element or more (no Avoid, no gap)
    inside_fine = up(xp.pad(xp.asarray(phi, dtype=np.float32), g + 1, mode="constant")) >= 0.5
    if min_thickness > 0:
        had = int(((values >= level) & inside_fine).sum())
        values, added = thicken(values, level, inside_fine, voxel, float(min_thickness), cancel=cancel)
        thickened = added / max(had, 1)
    step()
    grid = Grid.block(start, voxel, (0, 0, 0), values.shape)
    # bits on their own inside the design space, under SPECK of the largest, are emptied there: flakes by
    # its faces that only joined the part outside it, where the crop to the design space cuts them off
    # (the wall bracket's pin eye at Smoothness 0.5: three of 0.3-1.8 mm3, 0.5 mm thin)
    kept_in = inside_fine if crop is None else Context(grid, xp=xp).eval_full(crop) < 0
    del inside_fine
    tips = 0
    under = min_thickness if tips_under is None else tips_under
    if under > 0:
        values, tips = cut_tips(values, level, kept_in, voxel, float(under),
                                per_mm=1.0 if blur > 0 else 1.0 / h, start=start, conditions=conditions, cancel=cancel)
    pieces, count = nd.label((values >= level) & kept_in)
    count = int(count)
    small = np.zeros(0, dtype=np.int64)
    if count > 1:
        sizes = xp.bincount(pieces.ravel())[1:]
        small = to_host(xp.flatnonzero(sizes < SPECK * sizes.max()) + 1)
        if len(small):
            values = xp.where(xp.isin(pieces, xp.asarray(small)), xp.minimum(values, level - voxel),
                              values).astype(np.float32)
    unreached = _unreached(to_host(pieces), count, small, conditions, start, voxel)
    del pieces
    if crop is not None:
        # outside the crop no more solid than the sample of it nearest by, so the surface runs straight out
        # through its faces: the blur and the thickening bulge it past them, and where such a bulge hangs
        # over a face that is empty inside, the crop left paper-thin loose shells (bolted bracket: on the
        # unloaded side of its bolt and pin holes, 0.1 mm thin and 4 mm long, a 3 mm3 bit at Draft)
        nearest = nd.distance_transform_edt(~kept_in, return_distances=False, return_indices=True)
        values = xp.minimum(values, values[tuple(nearest)])
        del nearest
    del kept_in
    step()
    values = (level - values).astype(np.float32)
    if float(values.min()) >= 0:
        raise ConditionError("no material left (every density is below 0.5)")
    # (on the host: Warp's marching cubes puts the same vertices but splits cells along other diagonals than
    # scikit-image's, 9715 of 12340 triangles, and the Taubin smoothing then moved vertices up to half a sample
    # apart: the shape, and its analysis, would depend on whether the GPU had room)
    mesh = field_to_mesh(to_host(values), grid)
    del values
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
    mesh.metadata["tips"] = tips            # thin tips cut off
    mesh.metadata["unreached"] = unreached  # loads the shape leaves in the air (`_unreached`)
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
