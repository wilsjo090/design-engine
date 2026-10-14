"""Matrix-free stiffness products and a conjugate gradient solver with a multigrid preconditioner.

Elements are the cells of a regular grid, shape (nx, ny, nz); nodes are its corners, shape
(nx+1, ny+1, nz+1), and a displacement is a (nx+1, ny+1, nz+1, 3) array. The stiffness matrix is
never assembled: K u = sum over elements of K_e u_e, where gathering an element's 24 values is 8
array slices (or, for a part filling little of its box, an index per element with material).
Everything works on numpy or CuPy arrays.

Multigrid (MGCG, Amir et al. 2014): coarser grids by a factor 2 per axis (or 1 where an axis is
too thin), trilinear prolongation P, Galerkin coarse operators P^T K P. A coarse element's matrix
is exactly the sum over its children of P_c^T K_child P_c, so the first coarse level stays
matrix-free (8 fixed matrices M_c = P_c^T KE P_c weighted by the children's stiffness); deeper
levels store their element matrices; the coarsest is factorized directly (scipy, CPU). A coarse
node is fixed where any fine node it interpolates to is fixed: then the coarse operator is exactly
the Galerkin operator of the fixed (masked) fine one, so the V-cycle is symmetric positive
definite, as conjugate gradients needs. The answer is always that of the exact fine operator.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..backend import to_host, xp_of
from .element import CORNERS, child_interpolation, child_offsets

SLAB_BYTES = 64 * 2**20  # temporary memory per batch of elements in a product
# Smoothing: Chebyshev polynomials of the Jacobi-preconditioned operator D^-1 K, aimed at its
# eigenvalues from lmax / EIG_RATIO to lmax. Plain damped Jacobi diverged where stiff and soft
# material meet: lmax rose from 3.1 (uniform part) to 4.2-4.9 (1 : 1e6 contrast), so a damping
# that suits one part blows up on another. lmax is estimated per level by power iteration, with a
# margin, and never above the Gershgorin bound (which is always safe).
DEGREE = 3
EIG_RATIO = 10.0
POWER_ITERATIONS = 12
EIG_MARGIN = 1.15
DIAG_EPS = 1e-12         # a DOF with a diagonal below this (relative) has no material: left out
DENSE_FILL = 0.5         # levels at least this full go in slabs of the whole grid, emptier ones element by element


class NotConverged(RuntimeError):
    pass


def node_shape(shape):
    return tuple(n + 1 for n in shape)


def _slabs(shape, bytes_per_element):
    per = max(1, int(SLAB_BYTES // max(1, bytes_per_element * shape[1] * shape[2])))
    for x0 in range(0, shape[0], per):
        yield x0, min(shape[0], x0 + per)


def gather(u, x0, x1, shape):
    """Values at the 8 corners of elements x0..x1 (all y, z): (x1-x0, ny, nz, 8, *rest)."""
    xp = xp_of(u)
    ny, nz = shape[1], shape[2]
    return xp.stack([u[x0 + cx:x1 + cx, cy:cy + ny, cz:cz + nz] for cx, cy, cz in CORNERS], axis=3)


def scatter(y, Y, x0, x1, shape):
    """Add per-corner values Y (x1-x0, ny, nz, 8, *rest) to the nodes y."""
    ny, nz = shape[1], shape[2]
    for n, (cx, cy, cz) in enumerate(CORNERS):
        y[x0 + cx:x1 + cx, cy:cy + ny, cz:cz + nz] += Y[:, :, :, n]


def connectivity(shape, cells, xp=np):
    """Node numbers (m, 8) of the elements with flat numbers `cells`, corners in CORNERS order."""
    i, j, k = xp.unravel_index(cells, shape)
    X, Y, Z = node_shape(shape)
    c = xp.asarray(CORNERS)
    return ((i[:, None] + c[:, 0]) * Y + (j[:, None] + c[:, 1])) * Z + (k[:, None] + c[:, 2])


def _scatter_add(flat, conn, V):
    """flat[conn[e, c]] += V[e, c] (flat: (N, 3) nodes, V: (m, 8, 3)), repeated nodes summed."""
    xp = xp_of(flat)
    if xp is not np:
        import cupyx
        cupyx.scatter_add(flat, conn.ravel(), V.reshape(-1, 3))
        return
    idx = conn.ravel()
    lo, hi = int(idx.min()), int(idx.max())  # a batch's elements are in order: their nodes are a short range
    idx = idx - lo
    for d in range(3):
        flat[lo:hi + 1, d] += np.bincount(idx, weights=V[..., d].ravel(), minlength=hi - lo + 1)


def children(a, factors):
    """Group a per-element array (nx, ny, nz, *rest) by coarse element: (nx/fx, ny/fy, nz/fz,
    fx*fy*fz, *rest), children in `element.child_offsets` order."""
    fx, fy, fz = factors
    nx, ny, nz = a.shape[:3]
    rest = a.shape[3:]
    b = a.reshape(nx // fx, fx, ny // fy, fy, nz // fz, fz, *rest)
    b = b.transpose(0, 2, 4, 1, 3, 5, *range(6, 6 + len(rest)))
    return b.reshape(nx // fx, ny // fy, nz // fz, fx * fy * fz, *rest)


def _along(a, axis, fn):
    """Apply fn to `a` with `axis` moved to the front (and moved back afterwards)."""
    xp = xp_of(a)
    return xp.moveaxis(fn(xp.moveaxis(a, axis, 0)), 0, axis)


def prolong(uc, factors):
    """Trilinear interpolation of coarse nodal values to the fine nodes."""
    xp = xp_of(uc)
    out = uc
    for axis, f in enumerate(factors):
        if f == 1:
            continue

        def up(c):
            fine = xp.empty((2 * c.shape[0] - 1,) + c.shape[1:], dtype=c.dtype)
            fine[0::2] = c
            fine[1::2] = 0.5 * (c[:-1] + c[1:])
            return fine
        out = _along(out, axis, up)
    return out


def restrict(uf, factors):
    """The transpose of `prolong` (sums fine values onto coarse nodes)."""
    out = uf
    for axis, f in enumerate(factors):
        if f == 1:
            continue

        def down(v):
            c = v[0::2].copy()
            odd = 0.5 * v[1::2]
            c[:-1] += odd
            c[1:] += odd
            return c
        out = _along(out, axis, down)
    return out


def coarsen_fixed(fixed, factors):
    """Fixed DOFs on the coarser nodes: where any fine node a coarse node interpolates to is fixed."""
    out = fixed
    for axis, f in enumerate(factors):
        if f == 1:
            continue

        def down(v):
            c = v[0::2].copy()
            c[:-1] |= v[1::2]
            c[1:] |= v[1::2]
            return c
        out = _along(out, axis, down)
    return out


def dot(a, b):
    xp = xp_of(a)
    if xp is np and a.dtype == np.float64:
        return float(np.dot(a.ravel(), b.ravel()))
    return float(xp.sum(a * b, dtype=xp.float64))


class Level:
    """One grid of the multigrid hierarchy.

    kind "fine": K_e = per[e] * const (const = KE, per = stiffness factor); "children": K_e =
    sum_c per[e, c] * const[c] (const = the matrices M_c); "stored": K_e = per[e] (24 x 24).
    A level filled at least DENSE_FILL is processed in slabs of the whole grid (`per` in grid
    layout); an emptier one element by element, only where there is material (`cells`, `conn`;
    `per` holds those elements' rows): an L bracket filling 17% of its box went ~5x faster."""

    def __init__(self, shape, kind, per, const, xp):
        self.shape = tuple(int(n) for n in shape)
        self.kind, self.const, self.xp = kind, const, xp
        n = int(np.prod(self.shape))
        filled = xp.abs(per.reshape(n, -1)).max(axis=1) > 0
        self.count = int(filled.sum())
        if self.count >= DENSE_FILL * n:
            self.cells = self.conn = None
            self.per = per
        else:
            self.cells = xp.flatnonzero(filled)
            self.per = xp.ascontiguousarray(per.reshape(n, *per.shape[3:])[self.cells])
            self.conn = connectivity(self.shape, self.cells, xp)
        self.factors = None   # coarsening to the next level
        self.free = None      # (X, Y, Z, 3): 1 where the DOF is solved for, 0 where fixed or empty
        self.inv_diag = None
        self.lmax = None      # largest eigenvalue of D^-1 K (an upper estimate), for smoothing
        self.direct = None    # coarsest level: solves exactly

    @property
    def nodes(self):
        return node_shape(self.shape)

    def full(self):
        """`per` in grid layout (nx, ny, nz, ...), zeros where there is no material."""
        if self.cells is None:
            return self.per
        out = self.xp.zeros((int(np.prod(self.shape)),) + self.per.shape[1:], dtype=self.per.dtype)
        out[self.cells] = self.per
        return out.reshape(self.shape + self.per.shape[1:])

    def _products(self, U, per):
        """K_e u_e for rows U (m, 24) with the elements' `per` rows."""
        if self.kind == "fine":
            return (U @ self.const) * per[:, None]
        if self.kind == "children":
            Y = None
            for c in range(self.const.shape[0]):
                part = (U @ self.const[c]) * per[:, c, None]
                Y = part if Y is None else Y + part
            return Y
        return self.xp.matmul(per, U[..., None])[..., 0]

    def _diagonals(self, per, absolute_rows=False):
        """Diagonals of the element matrices (absolute_rows: sums of absolute values per row, for
        the Gershgorin bound), (m, 24)."""
        xp = self.xp
        if self.kind == "fine":
            return per[:, None] * (xp.abs(self.const).sum(axis=1) if absolute_rows else xp.diagonal(self.const))
        if self.kind == "children":
            M = self.const
            return per @ (xp.abs(M).sum(axis=2) if absolute_rows else xp.diagonal(M, axis1=1, axis2=2))
        return xp.abs(per).sum(axis=-1) if absolute_rows else xp.diagonal(per, axis1=-2, axis2=-1)

    def _accumulate(self, element_values, out, row_bytes):
        """out += per-corner values of every element, element_values(per rows, where) giving them
        as (m, 24) for the slab (x0, x1) or the node numbers (m, 8) of a batch of elements."""
        if self.cells is None:
            ny, nz = self.shape[1], self.shape[2]
            for x0, x1 in _slabs(self.shape, row_bytes):
                per = self.per[x0:x1].reshape(-1, *self.per.shape[3:])
                V = element_values(per, (x0, x1))
                scatter(out, V.reshape(x1 - x0, ny, nz, 8, 3), x0, x1, self.shape)
            return out
        if not out.flags.c_contiguous:
            raise ValueError("the result array must be C-contiguous (its flat view is written to)")
        flat = out.reshape(-1, 3)
        step = max(1, int(SLAB_BYTES // row_bytes))
        for i0 in range(0, self.count, step):
            conn = self.conn[i0:i0 + step]
            _scatter_add(flat, conn, element_values(self.per[i0:i0 + step], conn).reshape(-1, 8, 3))
        return out

    def apply(self, u, masked=True):
        """K u (masked: only the free DOFs, and u is taken as 0 on the others)."""
        # C order: `u` may be a transposed view (prolong), and zeros_like would copy its layout, so
        # the flat view written to below would be a copy (the element path then returned zeros)
        y = self.xp.zeros(u.shape, dtype=u.dtype)
        if self.cells is None:
            def values(per, slab):
                return self._products(gather(u, *slab, self.shape).reshape(-1, 24), per)
        else:
            flat = u.reshape(-1, 3)

            def values(per, conn):
                return self._products(flat[conn].reshape(-1, 24), per)
        self._accumulate(values, y, 3 * 24 * 8 * u.itemsize)
        if masked:
            y *= self.free
        return y

    def diagonal(self, dtype, absolute_rows=False):
        d = self.xp.zeros(self.nodes + (3,), dtype=dtype)
        return self._accumulate(lambda per, _: self._diagonals(per, absolute_rows), d, 2 * 24 * 8 * d.itemsize)

    def cell_matrices(self):
        """(node numbers (m, 8), element matrices (m, 24, 24)) of the elements with material, on
        the host (for the direct solver of a small level)."""
        xp = self.xp
        if self.cells is None:
            n = int(np.prod(self.shape))
            per = self.per.reshape(n, *self.per.shape[3:])
            cells = xp.flatnonzero(xp.abs(per.reshape(n, -1)).max(axis=1) > 0)
            per, conn = per[cells], connectivity(self.shape, cells, xp)
        else:
            per, conn = self.per, self.conn
        if self.kind == "fine":
            K = per[:, None, None] * self.const
        elif self.kind == "children":
            K = xp.einsum("mc,cij->mij", per, self.const)
        else:
            K = per
        return np.asarray(to_host(conn)), np.asarray(to_host(K), dtype=np.float64)

    def estimate_lmax(self, dtype, seed=0):
        """Upper estimate of the largest eigenvalue of D^-1 K on the free DOFs."""
        xp = self.xp
        rows = self.diagonal(dtype, absolute_rows=True)
        bound = float(xp.max(rows * self.inv_diag)) if self.free.size else 1.0
        rng = np.random.default_rng(seed)
        v = xp.asarray(rng.standard_normal(self.nodes + (3,)), dtype=dtype) * self.free
        lam = 0.0
        for _ in range(POWER_ITERATIONS):
            norm = np.sqrt(dot(v, v))
            if norm == 0:
                break
            v = v / norm
            w = self.inv_diag * self.apply(v)
            lam = np.sqrt(dot(w, w))
            v = w
        return max(min(EIG_MARGIN * lam, bound), 1e-12)

    def smooth(self, r, x=None):
        """A Chebyshev smoothing step for K x = r (from x = 0 when x is None)."""
        beta = self.lmax
        alpha = beta / EIG_RATIO
        theta, delta = (beta + alpha) / 2, (beta - alpha) / 2
        sigma = theta / delta
        rho = 1 / sigma
        res = r if x is None else r - self.apply(x)
        d = self.inv_diag * res / theta
        x = d if x is None else x + d
        for _ in range(DEGREE - 1):
            res = res - self.apply(d)
            rho_new = 1 / (2 * sigma - rho)
            d = (rho_new * rho) * d + (2 * rho_new / delta) * (self.inv_diag * res)
            rho = rho_new
            x = x + d
        return x


def _direct_solver(level, dtype):
    """Factorize the level's free part with scipy; returns r -> K^-1 r."""
    xp = level.xp
    conn, Ke = level.cell_matrices()
    dofs = (3 * conn[:, :, None] + np.arange(3)).reshape(-1, 24)
    n = int(np.prod(level.nodes)) * 3
    rows = np.broadcast_to(dofs[:, :, None], Ke.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], Ke.shape).ravel()
    A = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    free = np.flatnonzero(to_host(level.free).ravel() > 0)
    if len(free) == 0:
        return lambda r: xp.zeros_like(r)
    lu = splu(A[free][:, free].tocsc())

    def solve(r):
        x = np.zeros(n)
        x[free] = lu.solve(np.asarray(to_host(r), dtype=np.float64).ravel()[free])
        return xp.asarray(x.reshape(level.nodes + (3,)), dtype=dtype)
    return solve


class Hierarchy:
    """The stiffness operator of a grid of elements and its multigrid preconditioner.

    scale: (nx, ny, nz) stiffness factor per element (E_e * h; 0 = empty); KE: the unit element
    matrix; fixed: (X, Y, Z, 3) bool; coarsenings: how often each axis is halved (its element count
    must be divisible by 2 that often)."""

    def __init__(self, scale, KE, fixed, coarsenings, xp=np, dtype=np.float64):
        self.xp, self.dtype = xp, dtype
        shape = tuple(int(n) for n in scale.shape)
        for n, k in zip(shape, coarsenings):
            if n % (2 ** k):
                raise ValueError(f"{shape} elements cannot be halved {coarsenings} times")
        scale = xp.asarray(scale, dtype=dtype)
        KE = np.asarray(KE, dtype=np.float64)
        fixed = np.asarray(to_host(fixed), dtype=bool)
        self.levels = [Level(shape, "fine", scale, xp.asarray(KE, dtype=dtype), xp)]
        fixed_by_level = [fixed]
        for depth in range(max(coarsenings, default=0)):
            prev = self.levels[-1]
            factors = tuple(2 if depth < k else 1 for k in coarsenings)
            prev.factors = factors
            coarse = tuple(n // f for n, f in zip(prev.shape, factors))
            P = [child_interpolation(o, factors) for o in child_offsets(factors)]
            if prev.kind == "fine":
                M = np.stack([p.T @ KE @ p for p in P])
                level = Level(coarse, "children", children(prev.full(), factors), xp.asarray(M, dtype=dtype), xp)
            elif prev.kind == "children":
                Mh = to_host(prev.const).astype(np.float64)
                Q = np.stack([p.T @ m @ p for p in P for m in Mh]).reshape(len(P) * len(Mh), 576)
                E2 = children(prev.full(), factors).reshape(*coarse, -1)
                level = Level(coarse, "stored", (E2 @ xp.asarray(Q, dtype=dtype)).reshape(*coarse, 24, 24), None, xp)
            else:
                Pst = xp.asarray(np.stack(P), dtype=dtype)
                K = xp.einsum("cji,...cjk,ckl->...il", Pst, children(prev.full(), factors), Pst)
                level = Level(coarse, "stored", K, None, xp)
            self.levels.append(level)
            fixed_by_level.append(coarsen_fixed(fixed_by_level[-1], factors))

        for level, fx in zip(self.levels, fixed_by_level):
            d = level.diagonal(dtype)
            dmax = float(d.max()) if d.size else 0.0
            has = d > DIAG_EPS * max(dmax, 1e-300)
            level.free = (has & xp.asarray(~fx)).astype(dtype)
            level.inv_diag = xp.where(level.free > 0, 1.0 / xp.where(has, d, 1.0), 0.0).astype(dtype)
        for level in self.levels[:-1]:
            level.lmax = level.estimate_lmax(dtype)
        self.levels[-1].direct = _direct_solver(self.levels[-1], dtype)

    @property
    def free(self):
        return self.levels[0].free

    def apply(self, u):
        return self.levels[0].apply(u)

    def _vcycle(self, depth, r):
        level = self.levels[depth]
        if level.direct is not None:
            return level.direct(r)
        x = level.smooth(r)
        rc = restrict(r - level.apply(x), level.factors) * self.levels[depth + 1].free
        x = x + prolong(self._vcycle(depth + 1, rc), level.factors) * level.free
        return level.smooth(r, x)

    def precondition(self, r):
        return self._vcycle(0, r)

    def solve(self, f, u0=None, tol=1e-6, maxiter=1000, callback=None):
        """Solve K u = f on the free DOFs (u = 0 on the others). callback(iteration, relative
        residual) is called every iteration (progress, cancelling). Returns (u, iterations, residual)."""
        xp = self.xp
        f = xp.asarray(f, dtype=self.dtype) * self.free
        u = xp.zeros_like(f) if u0 is None else xp.asarray(u0, dtype=self.dtype) * self.free
        bnorm = np.sqrt(dot(f, f))
        if bnorm == 0:
            return u, 0, 0.0
        r = f - self.apply(u)
        z = self.precondition(r)
        p = z.copy()
        rz = dot(r, z)
        res = np.sqrt(dot(r, r)) / bnorm
        for it in range(1, maxiter + 1):
            if res <= tol:
                return u, it - 1, res
            q = self.apply(p)
            alpha = rz / dot(p, q)
            u += alpha * p
            r -= alpha * q
            res = np.sqrt(dot(r, r)) / bnorm
            if callback is not None:
                callback(it, res)
            if not np.isfinite(res):
                break
            z = self.precondition(r)
            rz, rz_old = dot(r, z), rz
            p = z + (rz / rz_old) * p
        if res <= tol:
            return u, maxiter, res
        raise NotConverged(f"the solver did not converge (residual {res:.1e} after {maxiter} iterations)")
