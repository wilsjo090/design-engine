"""Moving and resizing CAD bodies (triangle meshes) before they become implicit bodies."""
import igl
import numpy as np
import trimesh

from .sdf import MeshQuery


def contact_points(body, support, tol=0.05):
    """Points of `support` (e.g. the part a body is clipped onto) lying on `body`'s surface,
    with the normal of the body's face at each. Returns (points, normals), possibly empty."""
    lo, hi = body.bounds
    sv = np.asarray(support.vertices)
    near = sv[np.all((sv > lo - 0.5) & (sv < hi + 0.5), axis=1)]
    if len(near) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    V = np.ascontiguousarray(body.vertices, dtype=np.float64)
    F = np.ascontiguousarray(body.faces, dtype=np.int64)
    tree = igl.AABB()
    tree.init(V, F)
    sq, face, _ = tree.squared_distance(V, F, np.ascontiguousarray(near, dtype=np.float64))
    touching = np.sqrt(sq) < tol
    return near[touching], igl.per_face_normals(V, F)[face[touching]]


def anchor_point(body, support=None, tol=0.05):
    """The point a body should be scaled about so it stays seated on `support`.

    Where the body rests on the support across several sloped faces (like a petal on fins), the
    contact faces' planes meet in one point (the tip of the cone/pyramid they form). Scaling about
    that point maps every one of those planes onto itself, so the contacts are kept exactly.
    With a single flat seat, the middle of the contact area is used instead; without a support,
    the body's bounding-box centre.
    """
    if support is None:
        return body.bounds.mean(axis=0), "center"
    pts, normals = contact_points(body, support, tol)
    if len(pts) == 0:
        # not touching: keep the point closest to the support fixed
        q = MeshQuery(support.vertices, support.faces)
        d = np.abs(q.signed(np.asarray(body.vertices)))
        return np.asarray(body.vertices)[np.argmin(d)], "closest"
    # least-squares point on all contact planes  n . x = n . p
    A = normals
    b = np.einsum("ij,ij->i", normals, pts)
    x, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    if rank == 3 and sv[-1] / sv[0] > 0.05:
        return x, "apex"
    return pts.mean(axis=0), "seat"


def scale_about(body, anchor, factor):
    """Uniformly scaled copy of `body`; `anchor` stays where it is."""
    if factor == 1.0:
        return body
    v = anchor + (np.asarray(body.vertices) - anchor) * float(factor)
    return trimesh.Trimesh(v, body.faces.copy(), process=False, face_attributes=dict(body.face_attributes),
                           metadata=dict(body.metadata))


def transformed(body, matrix):
    """Copy of `body` moved by a 4 x 4 `matrix` (a move, a turn, a mirror), keeping its CAD face numbers,
    surface parameters and metadata. A mirror turns the triangles inside out, so their corners are
    reversed, and the per-corner attributes (the surface parameters, (n, 3, 2)) with them."""
    m = np.asarray(matrix, dtype=np.float64)
    if np.allclose(m, np.eye(4), rtol=0, atol=1e-12):
        return body
    v = np.asarray(body.vertices) @ m[:3, :3].T + m[:3, 3]
    faces, attrs = np.array(body.faces), dict(body.face_attributes)
    if np.linalg.det(m[:3, :3]) < 0:
        faces = np.ascontiguousarray(faces[:, ::-1])
        attrs = {k: np.ascontiguousarray(a[:, ::-1]) if np.ndim(a) == 3 and np.shape(a)[1] == 3 else a
                 for k, a in attrs.items()}
    return trimesh.Trimesh(v, faces, process=False, face_attributes=attrs, metadata=dict(body.metadata))


def face_plane(bodies, faces_of):
    """(point, unit normal) of the plane the chosen faces lie on; `faces_of(body)` gives each body's
    chosen triangles (a boolean mask or None). Raises ValueError when there are none or they are not flat."""
    tris = [np.asarray(b.vertices)[np.asarray(b.faces)[sel]] for b in bodies
            if (sel := faces_of(b)) is not None and np.any(sel)]
    if not tris:
        raise ValueError("none of the chosen faces is on these bodies")
    tri = np.concatenate(tris)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # 2 x area along the normal
    normal = cross.sum(axis=0)
    area = np.linalg.norm(cross, axis=1)
    if np.linalg.norm(normal) < 1e-9 * max(area.sum(), 1e-30):
        raise ValueError("the chosen faces are not flat")
    normal /= np.linalg.norm(normal)
    centre = (tri.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
    off = np.abs((tri.reshape(-1, 3) - centre) @ normal).max()
    size = np.ptp(tri.reshape(-1, 3), axis=0).max()
    if off > 0.01 + 1e-4 * size:
        raise ValueError(f"the chosen faces are not flat (they bend {off:.2g} mm out of their plane)")
    return centre, normal
