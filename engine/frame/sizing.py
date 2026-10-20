"""Sizing: the size of each bar from its force, length and the profile.

A bar in tension needs A >= SF N / strength. A bar in compression also must not buckle over its
unbraced length: Euler sigma_cr = pi^2 E / (K L / r)^2 for slender bars, Johnson's parabola
sigma_cr = strength (1 - (K L / r)^2 / (2 Cc^2)) below the transition slenderness
Cc = pi sqrt(2 E / strength). Sizes are rounded up to the size step (or the next size of a list),
never below the smallest size; *max sizes* groups the bars into that many sizes with the least
mass (optimal 1D grouping by dynamic programming; a group takes its largest bar's size).
"""
import math

import numpy as np

from ..fea.model import ConditionError


def capacity(section, length, material, safety, K=1.0):
    """(tension, compression) the bar may carry, N, with the safety factor."""
    sy, E = material.strength, material.E
    tension = section.A * sy / safety
    slender = K * length / section.r_min
    Cc = math.pi * math.sqrt(2 * E / sy)
    if slender < Cc:
        scr = sy * (1 - slender ** 2 / (2 * Cc ** 2))
    else:
        scr = math.pi ** 2 * E / slender ** 2
    return tension, section.A * scr / safety


def required_size(profile, force, length, material, safety, K=1.0, smallest=1e-3):
    """The smallest (continuous) size that carries the force (+ tension, - compression)."""
    def ok(size):
        t, c = capacity(profile.section(size), length, material, safety, K)
        return (force <= t) if force >= 0 else (-force <= c)

    lo = smallest
    if ok(lo):
        return lo
    hi = max(2 * lo, 1.0)
    while not ok(hi):
        hi *= 2
        if hi > 1e5:
            raise ConditionError(f"no {profile.kind} bar carries {abs(force):.3g} N over {length:.3g} mm")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        if hi - lo < 1e-4 * hi:
            break
    return hi


def round_up(size, min_size, step=0.5, sizes=None):
    """Up to the size step or the next listed size, at least min_size."""
    size = max(size, min_size)
    if sizes:
        above = [s for s in sorted(sizes) if s >= size * (1 - 1e-9)]
        if not above:
            raise ConditionError(f"a bar needs {size:.3g} mm: larger than the largest size in the list "
                                 f"({max(sizes):g} mm)")
        return float(above[0])
    return float(math.ceil(size / step - 1e-9) * step)


def next_size(size, step=0.5, sizes=None):
    """The next size up."""
    if sizes:
        above = [s for s in sorted(sizes) if s > size * (1 + 1e-9)]
        if not above:
            raise ConditionError(f"a bar needs more than the largest size in the list ({max(sizes):g} mm)")
        return float(above[0])
    return float(round(size / step) * step + step)


def group(sizes, weights, profile, groups):
    """Sizes per bar so that at most `groups` different sizes are used, each bar at least its own
    size, with the least total weight x area (weights: the bars' lengths)."""
    sizes = np.asarray(sizes, dtype=np.float64)
    levels = np.unique(sizes)
    if groups <= 0 or len(levels) <= groups:
        return sizes.copy()
    w = np.array([float(np.asarray(weights)[sizes == s].sum()) for s in levels])
    area = np.array([profile.section(s).A for s in levels])
    n = len(levels)
    prefix = np.concatenate([[0.0], np.cumsum(w)])
    # best[g][j]: least cost of levels[:j] in g groups, the last group ending at level j - 1
    best = np.full((groups + 1, n + 1), np.inf)
    cut = np.zeros((groups + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for g in range(1, groups + 1):
        for j in range(1, n + 1):
            costs = best[g - 1, :j] + (prefix[j] - prefix[:j]) * area[j - 1]
            i = int(np.argmin(costs))
            best[g, j], cut[g, j] = costs[i], i
    g = int(np.argmin(best[1:, n])) + 1
    out_level = np.empty(n)
    j = n
    while j > 0:
        i = cut[g, j]
        out_level[i:j] = levels[j - 1]
        j, g = i, g - 1
    return out_level[np.searchsorted(levels, sizes)]
