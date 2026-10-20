"""The trilinear 8-node hexahedron (hex8) on a cube.

Local node n = 4*cx + 2*cy + cz sits at corner (cx, cy, cz) of the cube; its displacement DOFs
are 3*n + d (d = x, y, z). Matrices are for a unit cube with E = 1: a cube with side h and
stiffness E has stiffness matrix E * h * stiffness(nu) (strain ~ 1/h, volume h^3), and strain
strain_matrix() / h.
"""
from functools import lru_cache

import numpy as np

CORNERS = np.array([(cx, cy, cz) for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)], dtype=np.int64)


def elasticity(E, nu):
    """6x6 elasticity matrix, Voigt order xx, yy, zz, yz, xz, xy (engineering shear strains)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def strain_matrix(point=(0.5, 0.5, 0.5)):
    """6x24 matrix giving the strain at `point` (unit-cube coordinates) from the 24 DOFs."""
    p = np.asarray(point, dtype=np.float64)
    B = np.zeros((6, 24))
    for n, c in enumerate(CORNERS):
        f = np.where(c == 1, p, 1 - p)       # the shape function's factor along each axis
        df = np.where(c == 1, 1.0, -1.0)     # and its derivative
        gx, gy, gz = df[0] * f[1] * f[2], f[0] * df[1] * f[2], f[0] * f[1] * df[2]
        i = 3 * n
        B[0, i], B[1, i + 1], B[2, i + 2] = gx, gy, gz
        B[3, i + 1], B[3, i + 2] = gz, gy   # yz
        B[4, i], B[4, i + 2] = gz, gx       # xz
        B[5, i], B[5, i + 1] = gy, gx       # xy
    return B


def _mode_strain_matrix(point):
    """6x9 strain of the incompatible modes (1 - xi_a^2) in direction d (column 3a + d)."""
    xi = 2 * np.asarray(point, dtype=np.float64) - 1
    G = np.zeros((6, 9))
    rows = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
    for a in range(3):
        for d in range(3):
            G[rows[(d, a)], 3 * a + d] = -4 * xi[a]   # d(1 - xi^2)/dx = -2 xi * dxi/dx, dxi/dx = 2
    return G


@lru_cache(maxsize=16)
def _stiffness(nu, incompatible):
    D = elasticity(1.0, nu)
    g = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
    Kuu, Kua, Kaa = np.zeros((24, 24)), np.zeros((24, 9)), np.zeros((9, 9))
    for x in g:
        for y in g:
            for z in g:  # 2x2x2 Gauss points, weight 1/8 each on the unit cube
                B, G = strain_matrix((x, y, z)), _mode_strain_matrix((x, y, z))
                Kuu += B.T @ D @ B / 8
                Kua += B.T @ D @ G / 8
                Kaa += G.T @ D @ G / 8
    K = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T) if incompatible else Kuu
    K = (K + K.T) / 2
    K.flags.writeable = False
    return K


def stiffness(nu, incompatible=True):
    """24x24 stiffness matrix of a unit cube with E = 1.

    incompatible: Wilson's incompatible modes (as Abaqus C3D8I), condensed out. A plain hex8 is
    too stiff in bending (shear locking: a 100 x 10 x 10 mm cantilever bent 3.9% too little with
    2 mm elements, 1.8% with 1 mm; with the modes 1.3% and 1.1%, the rest being the fully clamped
    end, and the bending stress is exact). On identical cubes the
    condensed matrix is the same for every element, so it costs nothing; uniform strain (the patch
    test) is unchanged, and the modes add no strain at the element centre, where stresses are read."""
    return _stiffness(float(nu), bool(incompatible))


def child_interpolation(offset, factors):
    """24x24 matrix giving a child element's DOFs from its parent's (trilinear interpolation).
    The parent is split `factors[a]` times (1 or 2) along axis a; `offset` is the child's position."""
    mats = []
    for a, f in zip(offset, factors):
        if f == 1:
            mats.append(np.eye(2))
        else:
            pos = (a + np.array([0.0, 1.0])) / 2  # the child's two nodes, in the parent's coordinates
            mats.append(np.stack([1 - pos, pos], axis=1))
    return np.kron(np.kron(np.kron(mats[0], mats[1]), mats[2]), np.eye(3))


def child_offsets(factors):
    """Positions of the children of a coarse element, in the order `children()` uses."""
    return [(a, b, c) for a in range(factors[0]) for b in range(factors[1]) for c in range(factors[2])]


def von_mises(s):
    """von Mises stress of Voigt stresses s (..., 6)."""
    xx, yy, zz, yz, xz, xy = (s[..., i] for i in range(6))
    return np.sqrt(np.maximum(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                              + 3 * (yz * yz + xz * xz + xy * xy), 0))
