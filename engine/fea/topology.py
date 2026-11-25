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

from .. import backend, trace
from ..backend import gaussian_filter, to_host, xp_of
from ..fields import Context, Field, _digest
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
# Extrude along: every member runs straight through the design space along one axis, with flat sides (a web
# between plates, a part cut from sheet): the free densities of each grid line along the axis are one design
# variable. Their sensitivities are averaged along the line (the chain rule's sums, as Symmetry's with the
# mirror image), and everything that looks at neighbours looks in the plane only: the filter (a member's
# thickness is its width in the plane; through the depth it is as thick as the design space, and the eroded
# design must not lose the layers at the two faces), and Max thickness's neighbourhoods.
EXTRUDE = ("none", "x", "y", "z")


def extrude_axis(extrude):
    if extrude not in EXTRUDE:
        raise ConditionError(f"extrude along must be one of: {', '.join(EXTRUDE)}")
    return None if extrude == "none" else "xyz".index(extrude)


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


def _room(phi, h, extrude=None):
    """How much room the design space has, mm: its thickest spot (the largest ball that fits, doubled), the
    second longest side of its box (what a member can be across, in a sheet too), the longest, and its largest
    section square to an axis (mm2). Limits past these cannot bind, and read as sizes they asked for filters of
    gigabytes (a Max section of 1e6 mm2 on a 30 mm box: 6 GB in 5 s). `extrude` (an axis): members run through
    the depth along it, so the room is that of the design space seen along it (the largest circle, the
    shorter and the longer side of its outline)."""
    inside = phi > 0
    planes = ((1, 2), (0, 2), (0, 1))
    largest = max(float(phi.sum(axis=a).max()) for a in planes) * h * h
    if extrude is not None:
        seen = inside.any(axis=extrude)
        thickest = 2.0 * h * float(ndimage.distance_transform_edt(np.pad(seen, 1)).max())
        sides = sorted(int(seen.any(axis=a).sum()) * h for a in (1, 0))
        return SimpleNamespace(thickest=thickest, across=sides[0], longest=sides[1], largest=largest)
    thickest = 2.0 * h * float(ndimage.distance_transform_edt(np.pad(inside, 1)).max())
    sides = sorted(int(inside.any(axis=a).sum()) * h for a in planes)
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
    extrude: object = None     # Extrude along: the axis (0, 1, 2) members run straight through along, or None

    def along(self, a):
        """Extrude along: `a` with the mean of the free elements of each grid line along the axis in each of
        them (the same value all along a line then; fixed regions stay as they are)."""
        xp, free = self.xp, self.free
        count = xp.sum(free, axis=self.extrude, keepdims=True)
        mean = xp.sum(xp.where(free, a, 0.0), axis=self.extrude, keepdims=True) / xp.maximum(count, 1)
        return xp.where(free, mean, a).astype(a.dtype)

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
        if self.extrude is not None:   # neighbourhoods in the plane only (a member is as deep as the design space)
            sigma = tuple(0.0 if k == self.extrude else sigma for k in range(3))
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


# Elements the members' field is carried into the fixed regions (the skin at supports and loads, Keep), from the
# nearest free element: members then run on into the exact bodies of those regions and overlap them.
CARRY = 3
MEMBERS_FIELD = 1   # bump when `members_field` gives other values: saved ones are then worked out again (`saved_members`)


def saved_members(state):
    """The members' field a state was saved with, or None when it has none or one of another version."""
    if "members" in state and "members_version" in state and int(state["members_version"]) == MEMBERS_FIELD:
        return state["members"]
    return None


def _carried(problem):
    """(fixed elements that copy a free one, the flat index of that one), on the host: `members_field`'s
    carrying into fixed regions, the same for every design of a problem."""
    free = to_host(problem.free).astype(bool)
    fixed = (to_host(problem.domain) > 0) & ~free
    if not fixed.any() or not free.any():
        return np.zeros(free.shape, dtype=bool), np.zeros(0, dtype=np.int64)
    distance, nearest = ndimage.distance_transform_edt(~free, return_indices=True)
    carried = fixed & (distance <= CARRY)
    return carried, np.ravel_multi_index(tuple(n[carried] for n in nearest), free.shape)


def members_field(problem, x, carry=None):
    """What the optimizer designed, apart from what was fixed: per element (0..1, host float32) the filtered
    design variables `x` on the free elements, 0.5 where a member's surface is (the densities are these,
    projected: solid in the same free elements). The node's body is this (smoothed, `density_field`) joined
    with the exact bodies of the fixed regions, the skin at supports and loads (`skin_body`) and Keep. The
    densities have those regions in them as whole elements: stepped bosses, a Keep plate up to half an
    element thicker than the CAD one, and no telling a member's surface from theirs, which a fillet between
    the two needs (the rolling ball's corner is between a member and the plate, not inside one body). The
    densities are also nearly 0 or 1, so their 0.5 surface follows the elements' faces, while this runs
    smoothly across a member's surface (a cone as wide as the thinnest member) and places it between the
    elements. Into the fixed regions the field is carried CARRY elements from the nearest free element, so
    members overlap the exact bodies and nothing else of a fixed region is in it; empty outside the design
    space. (Filtered over the free elements only, without the fixed regions' full elements, it lost what the
    optimizer puts next to them, where those count: members necked at their roots, the bolted bracket 30%
    less stiff and the wall bracket 8% lighter.)"""
    p = problem
    xp = p.xp
    out = np.asarray(to_host(xp.where(p.free, p.filter(xp.asarray(x, dtype=p.dtype)), 0.0)), dtype=np.float32)
    np.clip(out, 0.0, 1.0, out=out)
    carried, source = _carried(p) if carry is None else carry
    if len(source):
        out[carried] = out.ravel()[source]
    return out


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


def skin_reach(skin, min_thickness, h):
    """How far from the faces of supports and loads the solid skin reaches, mm: Skin, at least an element, and
    with a Skin never less than a set Min thickness."""
    reach = max(float(skin), h)
    if skin > 0 and min_thickness > reach:
        reach = float(min_thickness)
    return reach


SKIN_BEYOND = 1.0   # mm past the skin its body's values are worked out at least (a fillet's reach more)


def skin_body(conditions, reach, beyond=0.0):
    """The solid skin as a body of its own: everything within `reach` mm of the faces the skin is kept around
    (`_skin_points`' rule on the triangles: all of a support's or load's, the contact side of a bearing load),
    as an exact distance, so that a boss round a hole is a true cylinder and not the elements the optimizer
    kept solid, smoothed (those stayed stepped at every Smoothing). Per connected piece of faces, in its box
    only: a piece of a cylinder (`frame.geometry._cylinder`) is measured as that cylinder: a hole's wall as a
    ring all round, as the frames' bosses are, also when the optimizer keeps only the side a pin presses on
    (with that side alone the eye came out open: the hanger's shaft holes as slots with a notch at each end
    of the arc, 17% more stress); a pin's wall as the arc it is; other pieces by their triangles. Values are
    exact up to `beyond` mm past the skin (at least
    SKIN_BEYOND); farther out they are at least that and grow with the distance from the pieces' boxes, so
    an offset of the part does not fill the room. None when no face is left. The design space crops it."""
    from ..frame.geometry import _cylinder, _cylinder_distance, _tree
    from ..frame.ground import _pieces
    reach, cap = float(reach), float(reach) + max(float(beyond), SKIN_BEYOND)
    shapes, every = [], []
    for c in conditions:
        tri = np.asarray(c.triangles, dtype=np.float64).reshape(-1, 3, 3)
        if not len(tri):
            continue
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        size = np.linalg.norm(normal, axis=1)
        ok = size > 1e-12
        tri, normal = tri[ok], normal[ok] / size[ok, None]
        tri = tri[c.presses(normal)]
        if not len(tri):
            continue
        for piece in _pieces(tri):
            cyl = _cylinder(piece)
            lo, hi = piece.reshape(-1, 3).min(axis=0), piece.reshape(-1, 3).max(axis=0)
            if cyl is not None and cyl["ring"]:   # (a ring's box: the whole circle, whatever was chosen of it)
                ends = [cyl["c"] + z * cyl["axis"] for z in (cyl["z0"], cyl["z1"])]
                spread = cyl["r"] * np.sqrt(np.maximum(1.0 - cyl["axis"] ** 2, 0.0))
                lo = np.minimum(lo, np.minimum(*ends) - spread)
                hi = np.maximum(hi, np.maximum(*ends) + spread)
            shapes.append((piece, cyl, lo - cap, hi + cap, _digest("skin piece", np.ascontiguousarray(piece).tobytes())))
            every.append(piece)
    if not shapes:
        return None

    def fn(ctx):
        axes = ctx.grid.axes()
        out = None
        for _, _, lo, hi, _ in shapes:   # outside every box: at least the cap, and the way to the nearest box
            off = [np.maximum(np.maximum(lo[i] - ax, ax - hi[i]), 0.0).astype(np.float32).reshape(
                [-1 if a == i else 1 for a in range(3)]) for i, ax in enumerate(axes)]
            far = np.sqrt(off[0] * off[0] + off[1] * off[1] + off[2] * off[2])
            out = far if out is None else np.minimum(out, far)
        out = np.ascontiguousarray(out + np.float32(cap), dtype=np.float32)
        for piece, cyl, lo, hi, key in shapes:
            sl = tuple(slice(int(np.searchsorted(ax, lo[i])), int(np.searchsorted(ax, hi[i], side="right")))
                       for i, ax in enumerate(axes))
            if any(part.start >= part.stop for part in sl):
                continue
            pts = np.stack(np.meshgrid(*[ax[part] for ax, part in zip(axes, sl)], indexing="ij"), axis=-1).reshape(-1, 3)
            if cyl is not None:
                dist = _cylinder_distance(cyl, pts)
            else:
                V, Fc, tree = _tree(key, piece)
                dist = np.sqrt(tree.squared_distance(V, Fc, np.ascontiguousarray(pts, dtype=np.float64))[0])
            block = np.minimum(dist, cap).astype(np.float32).reshape([part.stop - part.start for part in sl])
            out[sl] = np.minimum(out[sl], block)
        return ctx.xp.asarray(out - np.float32(reach))

    box = np.concatenate([np.array([lo, hi]) for _, _, lo, hi, _ in shapes])
    key = _digest("topology skin", np.ascontiguousarray(np.concatenate(every)).tobytes(), reach, cap)
    return Field(fn, key, "skin", np.array([box.min(axis=0) + cap - reach, box.max(axis=0) - cap + reach]), cache=True)


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
                 prefer_body=None, prefer_strength=0.0, prune=False, max_section=0.0, extrude="none"):
    """Fingerprint of everything an optimization's result depends on (without a max thickness, a max section, a
    gap, smoothness, symmetry, a coarse start, a guide, pruning or an extrude direction the same as before they
    existed, so earlier results stay valid)."""
    extra = (float(max_thickness),) if max_thickness else ()
    if extrude != "none":   # (members and max section do nothing then: not in the key)
        extra += (("extrude", extrude),)
        members, max_section = MEMBERS[0], 0.0
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
            max_section=0.0, extrude="none"):
    """The optimization problem. `xp` may be a function giving numpy or cupy for a number of elements.
    `prune`: dead ends are kept empty as they show (`Optimizer._prune`). `extrude`: see EXTRUDE."""
    if not 0 < keep < 100:
        raise ConditionError("Keep % must be between 0 and 100")
    if gap < 0:
        raise ConditionError("the gap must be 0 or more")
    if smoothness < 0:
        raise ConditionError("smoothness must be 0 or more")
    axes = mirror_axes(symmetry)
    along = extrude_axis(extrude)
    extruded = []   # notes about what Extrude along turns off
    if along is not None:
        if max_thickness and members != MEMBERS[0]:
            extruded.append(f"Members: {members} has no effect with Extrude along: every member is a wall through "
                            f"the depth, and Max thickness is its width")
            members = MEMBERS[0]
        if max_section:
            extruded.append("Max section has no effect with Extrude along (every member runs through the whole "
                            "depth): use Max thickness for the members' width")
            max_section = 0.0
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
    reach = skin_reach(skin, min_thickness, h)
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
    notes = notes + extruded
    room = _room(phi, h, along) if min_thickness or max_thickness or max_section else None
    if room is not None and min_thickness > room.across:
        raise ConditionError(f"Min thickness {min_thickness:g} mm: no member that thick fits, the design space is "
                             f"{room.across:.3g} mm across" + (f" seen along {extrude}" if along is not None else ""))

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
    if along is not None:
        # in the plane only: the cone's middle layer, in an array of the same (cubic) shape, which is what
        # the convolutions and the GPU's volume kernel (its taps are the non-zero entries) expect
        middle = (o == 0).reshape([-1 if k == along else 1 for k in range(3)])
        kernel = kernel * middle
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
    problem.extrude = along
    if along is not None:
        free = domain & ~solid
        lines = free.any(axis=along)
        whole = (free.sum(axis=along) == domain.sum(axis=along))[lines]
        note = f"extruded along {extrude}: every member runs straight through the design space that way, with flat sides"
        if lines.any() and not whole.all():
            note += (f" (the skin and Keep stay as they are: {100 * (1 - float(whole.mean())):.0f}% of the lines "
                     f"pass through them)")
        problem.notes.append(note)
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
        self._carry = None       # (`members_field`: which free element each fixed one copies)
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
        whole = lambda cells: to_host(p.along(xp.asarray(cells, dtype=p.dtype))) >= 1.0 - 1e-6
        if p.extrude is not None:   # (only whole lines: a line is one design variable)
            cut = whole(cut)
            count = count if cut.any() else 0
        if count:
            far = ndimage.distance_transform_edt(~((density >= 0.5) & ~cut), sampling=p.grid.h) > p.min_member
            if p.extrude is not None:
                far = whole(cut & far)
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
        with trace.span("solve", xp=xp):
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
        if p.extrude is not None:
            dc, dv = p.along(dc), p.along(dv)
        with trace.span("oc update", xp=xp):
            if p.local_sigma or p.section_sigma:
                g, dg_xb = self._local_constraint(xb)
                dg = p.filter_transpose(xp.where(free, dg_xb, 0.0) * slope)
                if p.mirror:
                    dg = p.symmetric(dg)
                if p.extrude is not None:
                    dg = p.along(dg)
                x_new = self._update_two(self.x, dc, dv, dg, p.volume(xb), g)
            else:
                x_new = self._update(self.x, dc, dv)
        if p.extrude is not None:   # (the same along every line already, but for rounding: kept exactly so)
            x_new = p.along(x_new)
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
            with trace.span("iteration", item=self.iteration + 1, xp=self.p.xp):
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
        x = p.symmetric(x) if p.mirror else x    # (interpolation may round one side differently)
        self.x = p.along(x) if p.extrude is not None else x   # (and gives a line's ends other values)
        self.beta, self.since_beta = handover["beta"], 0
        self.iteration, self.history = handover["iteration"], list(handover["history"])
        self.done = self.iteration >= self.max_iterations     # (Max iterations is for all levels together)
        return self

    def density(self):
        """Final densities per element (0..1, host), solid regions included."""
        return np.asarray(to_host(self.physical(self.x)[1]), dtype=np.float32)

    def members_field(self):
        """`members_field` of this design (what it carries into fixed regions worked out once)."""
        if self._carry is None:
            self._carry = _carried(self.p)
        return members_field(self.p, self.x, self._carry)

    def design_compliance(self):
        """The compliance of the design itself, one solve per iteration asked for. The history has
        the eroded design's, which is less stiff (key box, struts only: 58% of the full design space
        where the design has 77%)."""
        if self._design is None or self._design[0] != self.iteration:
            self._design = (self.iteration, self._compliance_of(self.physical(self.x)[1])[0])
        return self._design[1]

    def state(self, stopped=False):
        p = self.p
        xt, xb = self.physical(self.x)
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
                "stopped": np.array(stopped), "density": np.asarray(to_host(xb), dtype=np.float32),
                # (states saved before 2026-11-12 have none; MEMBERS_FIELD: which `members_field` made it)
                "members": self.members_field(), "members_version": np.array(MEMBERS_FIELD),
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
# `density_field`'s samples are the control values of a cubic B-spline (`fields.sampled_body`), which at a sample
# is their [1, 4, 1] / 6 average along each axis: a blur of this variance (samples^2), taken off the Smoothing's own.
SPLINE_VARIANCE = 1.0 / 3.0
# Its values are made distances again, smoothly. The blurred distance runs slower than 1 per mm across members
# thinner than the blur (0.64 at a strut 3 elements across at Smoothing 2), so a Texture 0.5 mm high would stand
# 0.8 mm off them: the values are divided by their own slope (to first order the distance to their zero surface,
# which stays where it is), the slope taken as at least SLOPE_MIN (it is 0 along the middle of a member) and at
# most SLOPE_MAX (steps left by pushed-out specks). Far from the surface the values are the distance to the nearest
# sample of the other side, blurred by FAR_BLUR samples: specks pushed out, tips cut and the clamp outside the crop
# leave "just outside" far from any surface, and an Offset Body of 1 mm brought them back as blobs. That distance is
# stepped by the samples: within FAR_SAMPLES of the surface the smooth values stay as they are (the steps showed as
# stripes there), blended over to the blurred distance by twice that. (Held to within a sample of the unblurred
# distance, the first version, a fillet into a Keep plate came out lumpy: a blend reads both bodies well away from
# their surfaces. Left alone but for what was 30% off, values 2 mm out read 2.65.)
SLOPE_MIN = 0.35
SLOPE_MAX = 2.0
FAR_SAMPLES = 2.5
FAR_BLUR = 1.5
# Even out (`even_out`, step 4 of docs/ORGANIC_PARTS_PLAN.md): the smoothed surface moves inwards by its smaller
# principal curvature (outwards where that is negative) for EVEN_TIME x Even out elements^2 of time. Along a
# straight strut and across a plate that curvature is 0, so they keep their thickness (the mean curvature thins a
# strut at 1 / r), while bulges and necks level out, the corners of openings and the joints between members get
# round and loose ends pull back. With Extrude along it is the curvature of the outline seen along the axis: a
# flat-sided member's smaller curvature is 0 wherever its side bulges, and only the necks would fill.
EVEN = 1.0                       # Even out's default
# (The prototype's 4 elements^2 closes every hole under 2.8 elements' radius, R^2 falls by 2 x the time, and
# halves one of 3.3: the hanger's two 5 mm openings, at 2.5% more deflection. With 2 the ripples at the elements'
# scale are as gone, a wave 4 elements long falls to 1%, and holes over 3 elements' radius shrink by under 12%.)
EVEN_TIME = 2.0
EVEN_STEP = 0.1                  # samples^2 of time an explicit step (the curvature is at most 1 / sample)
# Where two principal curvatures are closer than this (per element) they count as equal, and their mean moves the
# surface: the hard minimum diffuses along one direction only, a random one where they are equal, and wrinkled
# plates within 40 steps.
EVEN_SOFT = 0.15
# Only what is near the surface moves, fading out between these distances (samples), and not where the values
# turn back within a sample: half way between two members, and along the middle of one, they have a ridge the
# samples cannot resolve, and a flow over it pulled flakes out of the gaps (443 on the bolted bracket). Told by
# how straight the values run through a sample, the central difference over the one-sided ones' (1 on a slope,
# 0 on a ridge's top, 0.71 beside a ridge that falls between two samples), fading in between EVEN_GATE. (The
# prototype stopped the flow where the slope was under 0.5 per mm, fading in to 0.8: the smoothed distance runs
# at 0.66-0.74 at a member's surface, 0.2-0.5 on thin ones, so most of the surface moved at part speed and 5-27%
# of it not at all, thin members least. On the bolted bracket that moved 2.5% of the material, this 8.6%.)
EVEN_NEAR, EVEN_FAR = 3.0, 6.0
EVEN_GATE = (0.75, 0.95)
EVEN_PINHOLE = (0.3, 0.7)
EVEN_BAND = 8                    # steps between looking for the samples near the surface again
EVEN_CHUNK = 1 << 21             # samples worked on at a time (the temporaries are 30 times this)
EVEN_HOLD = 3.0                  # samples: within this of a fixed body (a boss, Keep) the motion fades to none


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


def thicken(values, level, inside, voxel, thickness, passes=THICKEN_PASSES, cancel=None, backing=None):
    """`values` (samples `voxel` mm apart, solid where >= level) with every member at least
    `thickness` thick, by moving the surface out where a member is thinner, as smoothly as the values
    run: a member's half thickness near a point is the deepest point of the solid within reach (the
    design space's faces counted as empty, as they are in the part); what it lacks of thickness / 2,
    blurred, is added to the values as that many mm along their gradient, a few passes. Thick members
    are left alone. (Balls on the middle of thin members and filling in from the faces, done on the
    samples, left steps of a sample and knobs all over the surface.) Plates lying on a face grow on
    both sides, so only half inwards: the optimizer's eroded design counts the faces as empty, so it
    does not make them thinner than elsewhere. Returns the new values and how many samples became
    solid inside. `cancel()` is called before each pass (it may raise to stop). `backing` (bool): samples that are
    material whatever the values (the exact bodies joined to them afterwards)."""
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
        if backing is not None:
            solid |= backing & inside
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


def cut_tips(values, level, inside, voxel, thickness, per_mm=1.0, start=np.zeros(3), conditions=(), cancel=None,
             backing=None):
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
    between the steps. `backing` (bool): samples that are material whatever the values."""
    xp, nd = _modules(values)
    step = cancel if cancel is not None else (lambda: None)
    r = TIP_SHARE * thickness / 2
    solid = (values >= level) & inside
    if backing is not None:
        solid |= backing & inside
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
    return _on_gpu_or_cpu(args, smoothing_samples(np.shape(density), blur, upsample))


def density_field(density, phi, origin, h, upsample=UPSAMPLE, blur=SMOOTHING, space=None, min_thickness=0.0,
                  crop=None, cancel=None, conditions=(), tips_under=None, fixed=None, even=0.0, axis=None,
                  avoid=None):
    """density_mesh's smoothed samples themselves instead of a mesh of them, for `fields.sampled_body`: a
    namespace with `values` (host float32, mm, negative inside, 0 at the surface), `start` (the first sample,
    mm), `voxel`, `bounds` (the box of what is solid) and `metadata` (density_mesh's). The body is then a smooth
    function of them (a cubic B-spline), not the distance to a mesh at their spacing, whose every facet showed in
    the shading as stripes along the members. `density`: the densities, or better the members' own field
    (`members_field`, the state's `members`) with `fixed`, the exact body of what the optimizer kept solid (the
    skin, Keep): it is joined to the result afterwards, so here it only counts as material when pieces are told
    apart (members that meet through a boss are one piece; a load on a boss is reached), and a design with no
    member at all is no error. The same steps as density_mesh, but: the Smoothing's blur is
    narrower by the spline's own (SPLINE_VARIANCE; at Smoothing 0 the spline's 0.3 elements remain), no mesh
    smoothing, and the values are made distances in mm again (`_as_distances`, `_far_distances`). `even`: Even
    out (`even_out` for EVEN_TIME x even elements^2, after the thickening; nothing moves in or by `fixed`, and the
    material in the design space stays what it was), `axis`: Extrude along's, for the outline's curvature.
    `avoid`: the Avoid body, which `crop` leaves out and the caller cuts away exactly afterwards: the members run
    straight on into it, as they do through the design space's faces, so that cut leaves them a flat face.
    (Without it the surface pulled back from Avoid by the smoothing: in a hollow design space, Walls, every
    member was flat outside and round inside.)"""
    args = (density, phi, origin, h, upsample, 0, blur, space, min_thickness, crop, cancel, conditions, tips_under)
    return _on_gpu_or_cpu(args, smoothing_samples(np.shape(density), blur, upsample), as_field=True, fixed=fixed,
                          even=float(even), axis=axis, avoid=avoid)


def _on_gpu_or_cpu(args, samples, **more):
    xp = _smoothing_module(samples)
    if xp is not np:
        try:
            return _density_mesh(*args, xp=xp, **more)
        except Exception as err:   # (out of memory after all: other programs took some meanwhile)
            if not backend.is_out_of_memory(err):
                raise
        finally:
            backend.release_gpu_memory()
    return _density_mesh(*args, xp=np, **more)


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


def _as_distances(values, voxel):
    """`values` (negative inside, samples `voxel` mm apart) divided by their slope, so that near their zero
    surface they are the distance to it in mm (SLOPE_MIN, SLOPE_MAX)."""
    xp = xp_of(values)
    slope = _gradient_norm(values, voxel)
    xp.clip(slope, SLOPE_MIN, SLOPE_MAX, out=slope)
    return (values / slope).astype(np.float32)


def _far_distances(values, voxel):
    """`values` (mm, negative inside, samples `voxel` apart) as they are within FAR_SAMPLES of the surface, and
    from twice that the distance to the nearest sample of the other side (less half a sample: the surface is
    between the two), blurred by FAR_BLUR samples; blended between. Nowhere farther from 0 than that sample is
    away: where the slope is near 0 (the middle of a thin member) or the values are stepped (Smoothing 0), the
    division by it (`_as_distances`) leaves them too large: -1.24 where the surface was 1 mm away, which a
    Shell reads, and at Smoothing 0 the surface itself moved in, 12% of a thin test shape's material."""
    xp, nd = _modules(values)
    solid = values < 0
    signed = nd.distance_transform_edt(~solid, sampling=voxel).astype(np.float32)
    signed -= nd.distance_transform_edt(solid, sampling=voxel).astype(np.float32)
    values = xp.where(solid, xp.maximum(values, signed), xp.minimum(values, signed))
    signed -= xp.where(solid, -0.5 * voxel, 0.5 * voxel).astype(np.float32)
    del solid
    weight = xp.clip(xp.abs(signed) / (FAR_SAMPLES * voxel) - 1.0, 0.0, 1.0)
    weight *= weight * (3.0 - 2.0 * weight)
    signed = nd.gaussian_filter(signed, FAR_BLUR)
    return (values + weight * (signed - values)).astype(np.float32)


def _fade(t):
    """0 where t <= 0, 1 where t >= 1, a smooth step between (in place)."""
    xp = xp_of(t)
    xp.clip(t, 0.0, 1.0, out=t)
    t *= t * (3.0 - 2.0 * t)
    return t


def _second_largest(h):
    """Of the symmetric 2 x 2 or 3 x 3 matrices `h` (a nested list of arrays) the second largest eigenvalue where
    it is positive and within reach of the largest (fading in from EVEN_PINHOLE[0] to [1] of it), else 0."""
    xp = xp_of(h[0][0])
    if len(h) == 2:
        mean = 0.5 * (h[0][0] + h[1][1])
        half = xp.sqrt(0.25 * (h[0][0] - h[1][1]) ** 2 + h[0][1] * h[0][1])
        second, largest = mean - half, mean + half
    else:
        (a, d, e), (_, b, f), (_, _, c) = h
        mean = (a + b + c) / 3.0
        a, b, c = a - mean, b - mean, c - mean
        p = xp.sqrt(xp.maximum((a * a + b * b + c * c + 2.0 * (d * d + e * e + f * f)) / 6.0, 1e-20))
        det = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)
        angle = xp.arccos(xp.clip(det / (2.0 * p * p * p), -1.0, 1.0)) / 3.0
        largest = mean + 2.0 * p * xp.cos(angle)
        smallest = mean + 2.0 * p * xp.cos(angle + np.float32(2.0 * math.pi / 3.0))
        second = 3.0 * mean - largest - smallest
    low, high = EVEN_PINHOLE
    share = second / xp.maximum(largest, 1e-12)
    return xp.maximum(second, 0.0) * _fade((share - np.float32(low)) / np.float32(high - low))


def even_out(values, voxel, time, axis=None, hold=None, soft=0.0, cancel=None):
    """`values` (positive inside, 0 at the surface, samples `voxel` mm apart) with their zero surface moved
    inwards by its smaller principal curvature (1 / mm; out where it is negative) for `time` mm2: a level set
    flow, values -= dt x speed x |gradient|, in explicit steps of EVEN_STEP samples^2. `axis`: the outline's
    curvature in the planes across that axis instead (Extrude along). `soft` (1 / mm): the two curvatures'
    soft minimum (EVEN_SOFT), exact where they are far apart. `hold` (0..1 per sample): how much of the motion
    each sample gets. Only samples near the surface are worked on (found again every EVEN_BAND steps), all of
    them from the values before the step, and none where the values turn back within a sample (EVEN_GATE), but
    for holes a sample or two wide, which are filled (EVEN_PINHOLE). The values should be smooth (blurred): on
    a raw distance every sharp edge is such a ridge. The same code on numpy and CuPy arrays; `cancel()` is
    called every EVEN_BAND steps."""
    xp = xp_of(values)
    steps = int(math.ceil(time / (EVEN_STEP * voxel * voxel)))
    if steps < 1:
        return values
    dt = np.float32(time / steps)
    out = xp.array(values, dtype=np.float32, order="C")
    flat = out.reshape(-1)
    held = None if hold is None else xp.ascontiguousarray(hold, dtype=np.float32).reshape(-1)
    stride = (out.shape[1] * out.shape[2], out.shape[2], 1)
    across = [k for k in range(3) if k != axis]
    inner = xp.zeros(out.shape, dtype=bool)
    inner[1:-1, 1:-1, 1:-1] = True
    first, second = np.float32(0.5 / voxel), np.float32(1.0 / (voxel * voxel))
    cap, soft2 = np.float32(1.0 / voxel), np.float32(soft * soft)
    low, high = EVEN_GATE
    at = fresh = None
    for k in range(steps):
        if k % EVEN_BAND == 0:
            if cancel is not None:
                cancel()
            # (a sample's value changes by at most 0.1 of a sample a step)
            at = xp.flatnonzero((xp.abs(out) < (EVEN_FAR + 0.1 * EVEN_BAND + 0.5) * voxel) & inner)
            fresh = xp.empty(at.size, dtype=np.float32)
        for start in range(0, int(at.size), EVEN_CHUNK):
            i = at[start:start + EVEN_CHUNK]
            c = flat[i]
            g, d2 = {}, {}
            sides = 0.0
            for a in across:
                lo, hi = flat[i - stride[a]], flat[i + stride[a]]
                g[a] = (hi - lo) * first
                d2[a, a] = (hi - 2.0 * c + lo) * second
                sides = sides + (hi - c) * (hi - c) + (c - lo) * (c - lo)
            for n, a in enumerate(across):
                for b in across[n + 1:]:
                    d2[a, b] = (flat[i + stride[a] + stride[b]] - flat[i + stride[a] - stride[b]]
                                - flat[i - stride[a] + stride[b]] + flat[i - stride[a] - stride[b]]) * (0.25 * second)
            norm2 = sum(g[a] * g[a] for a in across)
            xp.maximum(norm2, 1e-12, out=norm2)
            norm = xp.sqrt(norm2)
            if axis is None:
                x, y, z = across
                total = -((d2[y, y] + d2[z, z]) * g[x] * g[x] + (d2[x, x] + d2[z, z]) * g[y] * g[y]
                          + (d2[x, x] + d2[y, y]) * g[z] * g[z]
                          - 2.0 * (g[x] * g[y] * d2[x, y] + g[x] * g[z] * d2[x, z] + g[y] * g[z] * d2[y, z])) / (norm2 * norm)
                gauss = (g[x] * g[x] * (d2[y, y] * d2[z, z] - d2[y, z] * d2[y, z])
                         + g[y] * g[y] * (d2[x, x] * d2[z, z] - d2[x, z] * d2[x, z])
                         + g[z] * g[z] * (d2[x, x] * d2[y, y] - d2[x, y] * d2[x, y])
                         + 2.0 * (g[x] * g[y] * (d2[x, z] * d2[y, z] - d2[x, y] * d2[z, z])
                                  + g[y] * g[z] * (d2[x, y] * d2[x, z] - d2[y, z] * d2[x, x])
                                  + g[x] * g[z] * (d2[x, y] * d2[y, z] - d2[x, z] * d2[y, y]))) / (norm2 * norm2)
                # (k1 - k2)^2 = total^2 - 4 gauss. The soft minimum: their mean where they are closer than `soft`,
                # the smaller one beyond. ((k1 + k2 + soft - sqrt((k1 - k2)^2 + soft^2)) / 2, the prototype's, is
                # soft / 2 too large on every strut, and a form that is soft^2 / 4 (k1 - k2) too large still
                # thinned a 4 mm strut by 0.27 mm in 4 mm2 of time.)
                apart = total * total - 4.0 * gauss
                xp.maximum(apart, 0.0, out=apart)
                if soft2 > 0:
                    speed = 0.5 * (total - xp.sqrt(apart) * (1.0 - xp.exp(apart / -soft2)))
                else:
                    speed = 0.5 * (total - xp.sqrt(apart))
                del total, gauss, apart
            else:
                x, y = across
                speed = -(d2[x, x] * g[y] * g[y] - 2.0 * g[x] * g[y] * d2[x, y] + d2[y, y] * g[x] * g[x]) / (norm2 * norm)
            del g, norm2
            xp.clip(speed, -cap, cap, out=speed)
            weight = _fade((EVEN_FAR - xp.abs(c) / np.float32(voxel)) / np.float32(EVEN_FAR - EVEN_NEAR))
            if held is not None:
                weight *= held[i]
            # 1 where the values run straight through the sample, 0 where they turn back in it (a ridge)
            straight = _fade((norm / xp.sqrt(xp.maximum(sides * (0.5 * second), 1e-12)) - np.float32(low))
                             / np.float32(high - low))
            del sides
            speed *= norm
            speed *= weight
            speed *= straight
            speed *= dt
            new = c - speed
            del speed, norm
            # A hole that has shrunk to a sample or two is all ridge, and stayed as a pit or a thread of empty
            # space inside. What the flow adds to the values there is the second largest eigenvalue of their
            # second derivatives: large along the middle of a hole (two directions across it), about 0 along
            # the middle of a gap between two members (one direction across it), which has to stay open
            # (EVEN_PINHOLE: how near the largest it must be). Filled to half a sample above 0: the
            # level that keeps the material moves by a few hundredths of a mm afterwards, and a hole filled to
            # just over 0 came back as a pit.
            pit = xp.flatnonzero((c < np.float32(0.5 * voxel)) & (straight < 1.0) & (weight > 0))
            if pit.size:
                fill = _second_largest([[d2[min(a, b), max(a, b)][pit] for b in across] for a in across])
                fill *= (1.0 - straight[pit]) * weight[pit] * dt
                new[pit] += fill
                del fill
            fresh[start:start + EVEN_CHUNK] = new
            del c, new, weight, straight, d2, pit
        flat[at] = fresh
    return out.astype(values.dtype, copy=False)


def _level_keeping(values, space, count, limit):
    """The level at which `count` of the `space` samples of `values` are at or above it, within +-`limit` of 0."""
    xp = xp_of(values)
    flat = values[space]
    if not 0 < count < flat.size:
        return 0.0
    return float(xp.clip(xp.partition(flat, flat.size - count)[flat.size - count], -limit, limit))


def _density_mesh(density, phi, origin, h, upsample, smooth, blur, space, min_thickness, crop, cancel, conditions,
                  tips_under, xp=np, as_field=False, fixed=None, even=0.0, axis=None, avoid=None):
    """density_mesh (or, with `as_field`, density_field) on numpy or CuPy arrays (the result on the host either way)."""
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
    if avoid is not None:
        # Avoid's elements (empty, their centre in the body) count as outside too: what lies by them runs on into
        # them, and the exact cut by the Avoid body, which the caller makes, leaves a flat face (not the gap's
        # elements: nothing cuts those)
        centres = Grid.block(np.asarray(origin, dtype=np.float64) + h / 2, h, (0, 0, 0), region.shape)
        region = region & ~((np.asarray(phi) <= 0) & to_host(Context(centres, xp=xp).eval_full(avoid) < 0))
    inside = xp.pad(xp.asarray(region), g, mode="constant", constant_values=False)
    ext = xp.pad(xp.asarray(density, dtype=np.float32), g, mode="constant")
    # outside the design space each element takes the density of the nearest one inside: material
    # runs straight out through the boundary, empty stretches of it stay empty (growing by dilation
    # spread it sideways too: 3 elements of that closed a beam's open sides into a cavity)
    distance, nearest = nd.distance_transform_edt(~inside, return_indices=True)
    ext = xp.where(distance <= g, ext[tuple(nearest)], 0.0).astype(np.float32)
    # (Even out: where the extension ends, an element in from its end, nothing moves: its cut-off ends would pull
    # back into the design space like any loose end)
    free = xp.clip(g - 0.5 - distance, 0.0, 1.0).astype(np.float32) if even > 0 else None
    del nearest, distance
    values = xp.pad(ext, 1, mode="constant", constant_values=0.0)  # a ring of "outside"
    start = np.asarray(origin, dtype=np.float64) + h / 2 - (g + 1) * h  # element centres, less the pads
    shape = tuple((n - 1) * upsample + 1 for n in values.shape)
    up = (lambda a: upsample_linear(a, upsample, shape)) if upsample > 1 else (lambda a: a)
    values = up(values)
    voxel = h / upsample
    level = 0.5
    # the spline's own blur is taken off the Smoothing's (samples)
    spread = (lambda sigma: math.sqrt(max(sigma * sigma - SPLINE_VARIANCE, 0.0))) if as_field else (lambda sigma: sigma)
    if as_field and not blur > 0:
        step()
        values = -_surface_distance(values, voxel)   # mm, positive inside: the same surface
        level = 0.0
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
        values = -nd.gaussian_filter(dist, spread(blur * upsample))        # mm, positive inside
        if blur > LIGHT_BLUR:
            light = -nd.gaussian_filter(dist, spread(LIGHT_BLUR * upsample))
            half = nd.maximum_filter(xp.maximum(light, 0.0), size=2 * int(math.ceil(sigma / voxel)) + 1)
            weight = nd.gaussian_filter(xp.clip(2.0 * half / sigma - 1.0, 0.0, 1.0), 0.5 * sigma / voxel)
            values = weight * values + (1.0 - weight) * light
            del light, half, weight
        values = values.astype(np.float32)
        step()
        # the surface where the material in the design space stays as much as the densities give (a
        # strong blur rounds members off: the bolted bracket lost 10% at Smoothing 3), within half an element
        space = up(xp.pad(inside, 1, mode="constant").astype(np.float32)) >= 0.5
        level = _level_keeping(values, space, int(xp.count_nonzero(dist[space] <= 0.0)), 0.5 * h)
        del dist, space
    thickened = 0.0
    # inside: where the design space fills half an element or more (no Avoid, no gap)
    inside_fine = up(xp.pad(xp.asarray(phi, dtype=np.float32), g + 1, mode="constant")) >= 0.5
    grid = Grid.block(start, voxel, (0, 0, 0), values.shape)
    # the exact bodies of the fixed regions (joined to the members afterwards) count as material where thin
    # members, tips and pieces are told: a member lying half in a boss is not thin, and what the members' field
    # carries into a Keep plate is no tip (ten "thin tips cut off" on the hanger, all of them inside its plate)
    backing = anchored = None
    if fixed is not None:
        away = Context(grid, xp=xp).eval_full(fixed)
        backing = away < 0
        if even > 0:
            # Even out moves nothing in a fixed body, and by it less and less: EVEN_HOLD. (Not farther than the
            # skin's body is exact whatever the fillet, SKIN_BEYOND: with the values' cap as "far", the flow
            # ran at 2 / 3 round every boss at Fillet 0, and the shape depended on the fillet.)
            anchored = xp.clip(away / min(EVEN_HOLD * voxel, SKIN_BEYOND), 0.0, 1.0).astype(np.float32)
        del away
    if min_thickness > 0:
        had = int(((values >= level) & inside_fine).sum())
        values, added = thicken(values, level, inside_fine, voxel, float(min_thickness), cancel=cancel, backing=backing)
        thickened = added / max(had, 1)
    if even > 0:
        # after the thickening, whose own ripples and necks it evens (it thins no straight member), and before
        # the tips, which are told by the design space's faces; the material in the design space stays as it was
        step()
        space = up(xp.pad(inside, 1, mode="constant").astype(np.float32)) >= 0.5
        if backing is not None:   # (what a member has inside a boss or Keep is no material of the part's: counted,
            space &= ~backing     # holes that closed were paid for there, and the part grew by 0.5-1%)
        count = int(xp.count_nonzero(values[space] >= level))
        hold = up(xp.pad(free, 1, mode="constant"))
        if anchored is not None:
            hold *= anchored
        with trace.span("even out", xp=xp):
            values = even_out(values - np.float32(level), voxel, EVEN_TIME * even * h * h, axis, hold, EVEN_SOFT / h,
                              cancel)
        del hold
        level = _level_keeping(values, space, count, 0.5 * h)
        del space
    del free, anchored
    step()
    # bits on their own inside the design space, under SPECK of the largest, are emptied there: flakes by
    # its faces that only joined the part outside it, where the crop to the design space cuts them off
    # (the wall bracket's pin eye at Smoothness 0.5: three of 0.3-1.8 mm3, 0.5 mm thin)
    kept_in = inside_fine if crop is None else Context(grid, xp=xp).eval_full(crop) < 0
    del inside_fine
    tips = 0
    under = min_thickness if tips_under is None else tips_under
    if under > 0:
        values, tips = cut_tips(values, level, kept_in, voxel, float(under),
                                per_mm=1.0 if blur > 0 or as_field else 1.0 / h, start=start, conditions=conditions,
                                cancel=cancel, backing=backing)
    material = (values >= level) & kept_in
    if backing is not None:
        material |= backing & kept_in
    pieces, count = nd.label(material)
    del material, backing
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
    if float(values.min()) >= 0 and fixed is None:
        raise ConditionError("no material left (every density is below 0.5)")
    metadata = {"thickened": thickened,   # share of material added for the min thickness
                "tips": tips,             # thin tips cut off
                "unreached": unreached}   # loads the shape leaves in the air (`_unreached`)
    if as_field:
        step()
        values = _far_distances(_as_distances(values, voxel), voxel)
        solid = [to_host(xp.flatnonzero((values < 0).any(axis=tuple(a for a in range(3) if a != k))))
                 for k in range(3)]
        if not len(solid[0]):   # (no member at all: only what was fixed)
            solid = [np.array([n // 2]) for n in values.shape]
        bounds = np.array([[start[k] + voxel * (int(solid[k][0]) - 1) for k in range(3)],
                           [start[k] + voxel * (int(solid[k][-1]) + 1) for k in range(3)]])
        return SimpleNamespace(values=to_host(values), start=start, voxel=voxel, bounds=bounds, metadata=metadata)
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
    mesh.metadata.update(metadata)
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
