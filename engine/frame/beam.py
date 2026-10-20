"""Frame check: a 3D frame of beam elements with rigid joints (a printed part is one piece).

Each element has 12 degrees of freedom (u, v, w and rotations about x, y, z at both ends).
Bending in each principal plane uses the exact cubic beam (Euler-Bernoulli), with Timoshenko's
shear term (Phi = 12 E I / (G k A L^2)) for short thick bars. Supports are clamped (a bolted or
glued face). The linear buckling factor is the smallest lambda with (K + lambda K_G) phi = 0, K_G
the geometric stiffness from the elements' axial forces under the loads.

Stresses: the axial stress plus the bending stresses at the end with the larger moment, and the
torsional shear, combined as von Mises (round bars: the resultant moment over W).
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

SHEAR_K = {"round": 0.9, "round tube": 0.5}   # shear area share (others ~ 5/6)
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Check:
    displacements: np.ndarray   # (n, 6): u, v, w (mm), rotations (rad)
    end_forces: np.ndarray      # (e, 12) local end forces on the element
    axial: np.ndarray           # (e,) N, + tension
    stress: np.ndarray          # (e,) von Mises at the worst end, MPa
    reactions: np.ndarray       # (n, 6) at supports (forces N, moments N mm)
    buckling: float             # load factor at which the frame buckles (inf: no compression)


def local_axes(p0, p1, up=UP):
    """Rows ex (along the bar), ey, ez: ez is `up` made square to the bar (world X for a bar
    along `up`)."""
    ex = (p1 - p0) / np.linalg.norm(p1 - p0)
    ref = up if abs(ex @ up) < 0.999 else np.array([1.0, 0.0, 0.0])
    ez = ref - (ref @ ex) * ex
    ez /= np.linalg.norm(ez)
    return np.array([ex, np.cross(ez, ex), ez])


def _plane(EI, L, phi):
    """Bending stiffness for (v1, t1, v2, t2) with t = dv/dx."""
    c = EI / (L ** 3 * (1 + phi))
    return c * np.array([[12, 6 * L, -12, 6 * L],
                         [6 * L, (4 + phi) * L * L, -6 * L, (2 - phi) * L * L],
                         [-12, -6 * L, 12, -6 * L],
                         [6 * L, (2 - phi) * L * L, -6 * L, (4 + phi) * L * L]])


def _plane_geometric(N, L):
    return N / (30 * L) * np.array([[36, 3 * L, -36, 3 * L],
                                    [3 * L, 4 * L * L, -3 * L, -L * L],
                                    [-36, -3 * L, 36, -3 * L],
                                    [3 * L, -L * L, -3 * L, 4 * L * L]])


# dof indices of the two bending planes; the x-z plane's rotation is t = dw/dx = -theta_y
_XY = np.array([1, 5, 7, 11])
_XZ = np.array([2, 4, 8, 10])
_SIGN_XZ = np.array([1.0, -1.0, 1.0, -1.0])


def element_stiffness(L, E, G, sec, shear_k=5 / 6, shear=True):
    k = np.zeros((12, 12))
    a = E * sec.A / L
    k[np.ix_([0, 6], [0, 6])] = a * np.array([[1, -1], [-1, 1]])
    t = G * sec.J / L
    k[np.ix_([3, 9], [3, 9])] = t * np.array([[1, -1], [-1, 1]])
    As = shear_k * sec.A
    phi_z = 12 * E * sec.Iz / (G * As * L * L) if shear else 0.0   # bending in x-y uses Iz
    phi_y = 12 * E * sec.Iy / (G * As * L * L) if shear else 0.0
    k[np.ix_(_XY, _XY)] = _plane(E * sec.Iz, L, phi_z)
    k[np.ix_(_XZ, _XZ)] = _SIGN_XZ[:, None] * _plane(E * sec.Iy, L, phi_y) * _SIGN_XZ[None, :]
    return k


def element_geometric(L, N, sec):
    g = np.zeros((12, 12))
    p = _plane_geometric(N, L)
    g[np.ix_(_XY, _XY)] = p
    g[np.ix_(_XZ, _XZ)] = _SIGN_XZ[:, None] * p * _SIGN_XZ[None, :]
    ip = (sec.Iy + sec.Iz) / sec.A
    g[np.ix_([3, 9], [3, 9])] = N * ip / L * np.array([[1, -1], [-1, 1]])
    return g


def _rotation(R):
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return T


def _assemble(n, elements, blocks):
    rows, cols, vals = [], [], []
    for (a, b), K in zip(elements, blocks):
        dofs = np.concatenate([6 * a + np.arange(6), 6 * b + np.arange(6)])
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        vals.append(K.ravel())
    return sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(6 * n, 6 * n))


def check(nodes, elements, sections, material, supports, loads, kinds=None, up=UP, shear=True, buckling=True,
          zaxes=None):
    """Analyse a frame. nodes (n, 3); elements (e, 2); sections: one Section per element; supports:
    indices of clamped nodes; loads (n, 3) forces or (n, 6) with moments; kinds: profile kind per
    element (shear areas); zaxes: each element's local z (its section's strong axis), else `up`
    made square to it."""
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements)
    n = len(nodes)
    E, nu = material.E, material.nu
    G = E / (2 * (1 + nu))
    Ts, Ks, lengths = [], [], []
    for i, (a, b) in enumerate(elements):
        L = float(np.linalg.norm(nodes[b] - nodes[a]))
        T = _rotation(local_axes(nodes[a], nodes[b], up if zaxes is None else np.asarray(zaxes[i], dtype=np.float64)))
        k = element_stiffness(L, E, G, sections[i], SHEAR_K.get(kinds[i] if kinds else "", 5 / 6), shear)
        Ts.append(T)
        Ks.append(T.T @ k @ T)
        lengths.append(L)
    K = _assemble(n, elements, Ks)
    f = np.zeros(6 * n)
    loads = np.asarray(loads, dtype=np.float64)
    f.reshape(n, 6)[:, :loads.shape[1]] = loads
    fixed = np.zeros(6 * n, dtype=bool)
    for s in np.atleast_1d(supports):
        fixed[6 * int(s):6 * int(s) + 6] = True
    free = np.flatnonzero(~fixed)
    Kff = K[free][:, free].tocsc()
    u = np.zeros(6 * n)
    lu = splinalg.splu(Kff)
    u[free] = lu.solve(f[free])
    reactions = (K @ u - f).reshape(n, 6)
    reactions[~fixed.reshape(n, 6).any(axis=1)] = 0.0

    end = np.zeros((len(elements), 12))
    axial = np.zeros(len(elements))
    stress = np.zeros(len(elements))
    for i, (a, b) in enumerate(elements):
        dofs = np.concatenate([6 * a + np.arange(6), 6 * b + np.arange(6)])
        L = lengths[i]
        k = element_stiffness(L, E, G, sections[i], SHEAR_K.get(kinds[i] if kinds else "", 5 / 6), shear)
        fl = k @ (Ts[i] @ u[dofs])
        end[i] = fl
        axial[i] = fl[6]
        stress[i] = von_mises(fl, sections[i], round_bar=bool(kinds) and kinds[i] in SHEAR_K)

    factor = np.inf
    if buckling and (axial < 0).any():
        Gs = [Ts[i].T @ element_geometric(lengths[i], axial[i], sections[i]) @ Ts[i] for i in range(len(elements))]
        KG = _assemble(n, elements, Gs)[free][:, free].tocsc()
        factor = _buckling_factor(Kff, KG, lu)
    return Check(u.reshape(n, 6), end, axial, stress, reactions, factor)


def von_mises(fl, sec, round_bar=False):
    """Largest von Mises stress at the two ends of an element from its local end forces."""
    N = fl[6]
    T = abs(fl[9])
    worst = 0.0
    for my, mz in ((fl[4], fl[5]), (fl[10], fl[11])):
        if round_bar:
            bend = float(np.hypot(my, mz)) / sec.Wy
        else:
            bend = abs(my) / sec.Wy + abs(mz) / sec.Wz
        sigma = abs(N) / sec.A + bend
        worst = max(worst, float(np.sqrt(sigma ** 2 + 3 * (T / sec.Wt) ** 2)))
    return worst


def _buckling_factor(K, KG, lu):
    """Smallest positive lambda of (K + lambda KG) phi = 0: the largest mu of -KG phi = mu K phi,
    lambda = 1 / mu."""
    n = K.shape[0]
    if n <= 600:
        from scipy.linalg import eigh
        mu = eigh(-KG.toarray(), K.toarray(), eigvals_only=True)
        top = float(mu.max())
    else:
        op = splinalg.LinearOperator((n, n), matvec=lambda x: lu.solve(-(KG @ x)))   # K^-1 (-KG)
        top = float(np.max(np.real(splinalg.eigs(op, k=1, which="LR", return_eigenvectors=False))))
    return 1.0 / top if top > 1e-12 else np.inf
