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
