"""Load STEP files with OpenCascade and turn B-rep bodies into triangle meshes."""
import io
from pathlib import Path

import numpy as np
import trimesh
from OCP.BinTools import BinTools_ShapeSet
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Identity, gp_PntMirror, gp_Scale, gp_Translation
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Face

from .meshing import signed_volume


def load_step(path):
    """Read a STEP file and return the list of solid bodies it contains (units: mm)."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(Path(path))) != IFSelect_RetDone:
        raise IOError(f"Could not read STEP file: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    bodies = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        bodies.append(TopoDS.Solid(exp.Current()))
        exp.Next()
    if not bodies:
        raise ValueError(f"No solid bodies found in {path}")
    return bodies


def tessellate(body, tolerance=0.01, angular_tolerance=0.1, check=None):
    """Triangulate a B-rep body. `tolerance` is the max chord deviation in mm. `check()` is
    called after every face (it may raise to cancel)."""
    BRepMesh_IncrementalMesh(body, tolerance, False, angular_tolerance, True)

    found = []  # (CAD face number in the body's own order, triangulation, location, reversed)
    exp = TopExp_Explorer(body, TopAbs_FACE)
    face_index = -1
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        exp.Next()
        face_index += 1
        if tri is not None:
            found.append((face_index, tri, loc, face.Orientation() == TopAbs_REVERSED))
        if check is not None:
            check()
    arrays = _read_triangulations([f[1] for f in found])

    # Collected as one small array per face: Python lists of tuples for the whole body took
    # ~10x the memory (a part with 9.4M vertices needed several GB just for the lists).
    verts, faces, face_ids, offset = [], [], [], 0
    for (index, tri, loc, reversed_), read in zip(found, arrays or [None] * len(found)):
        points, tris = read if read is not None else _read_one(tri)
        if not loc.IsIdentity():
            points = _transformed(tri, points, loc.Transformation())
        tris = tris.astype(np.int64)
        if reversed_:
            tris = tris[:, [0, 2, 1]]
        verts.append(points)
        faces.append(tris - 1 + offset)
        face_ids.append(np.full(len(tris), index, dtype=np.int32))
        offset += len(points)
        if check is not None and read is None:
            check()
    del arrays

    mesh = trimesh.Trimesh(np.concatenate(verts) if verts else np.zeros((0, 3)),
                           np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64), process=True)
    # which CAD face (in the body's own order) each triangle belongs to, for face selections
    mesh.face_attributes["cad_face"] = np.concatenate(face_ids) if face_ids else np.zeros(0, dtype=np.int32)
    del verts, faces, face_ids
    # trimesh's .volume and edge tables cost +3.9 and +2.7 GB on a 12M-triangle part; the sign of
    # the volume and an edge count are all that is needed here
    if signed_volume(mesh.vertices, mesh.faces) < 0:
        mesh.invert()
    # Small cracks (e.g. T-junctions at fillet corners) are harmless: the fast winding number used
    # for inside/outside tolerates them. Only warn when a meaningful part of the surface is open.
    open_edges, edges = _open_edges(mesh.faces, len(mesh.vertices))
    if open_edges > 0.001 * edges:
        print(f"  warning: tessellation has {open_edges} open edges; inside/outside may be unreliable")
    mesh._cache.clear()  # don't keep derived arrays alive with the imported body
    return mesh


def _read_triangulations(tris):
    """Nodes (float64, untransformed) and 1-based triangles of every triangulation, read in one
    go through OpenCascade's binary shape writer. Asking for them one node and one triangle at a
    time from Python took ~3.5 us per triangle (49 s for a 12M-triangle part); this is ~15x faster
    and gives the same numbers bit for bit. None if the data doesn't look as expected (the caller
    then reads face by face)."""
    if not tris:
        return []
    shapes = BinTools_ShapeSet()
    shapes.SetWithTriangles(True)
    shapes.SetWithNormals(False)
    builder = BRep_Builder()
    for tri in tris:
        # a bare face holding only the triangulation: adding the real face would also add its
        # edges, whose polygons register the neighbouring faces' triangulations out of order
        holder = TopoDS_Face()
        builder.MakeFace(holder, tri)
        shapes.Add(holder)
    stream = io.BytesIO()
    shapes.WriteTriangulation(stream)
    del shapes
    data = stream.getbuffer()
    try:
        head = bytes(data[:64]).split(b"\n", 1)[0]
        if not head.startswith(b"Triangulations ") or int(head.split()[1]) != len(tris):
            return None  # e.g. two faces sharing one triangulation
        flags = 2 if BinTools_ShapeSet().FormatNb() >= 4 else 1  # format 4 added a "has normals" byte
        pos, out = len(head) + 1, []
        for tri in tris:
            nn, nt = (int(v) for v in np.frombuffer(data, "<i4", 2, pos))
            if nn != tri.NbNodes() or nt != tri.NbTriangles():
                return None
            has_uv, has_normals = data[pos + 8], (data[pos + 9] if flags == 2 else 0)
            pos += 8 + flags + 8  # counts, flags, deflection
            points = np.frombuffer(data, "<f8", 3 * nn, pos).reshape(-1, 3).copy()
            pos += 24 * nn + (16 * nn if has_uv else 0)
            out.append((points, np.frombuffer(data, "<i4", 3 * nt, pos).reshape(-1, 3).copy()))
            pos += 12 * nt + (12 * nn if has_normals else 0)
        return out if pos == len(data) else None
    finally:
        data.release()


def _read_one(tri):
    """One triangulation the slow way (node by node)."""
    n = tri.NbNodes()
    points = np.fromiter((c for p in map(tri.Node, range(1, n + 1)) for c in p.Coord()),
                         np.float64, 3 * n).reshape(-1, 3)
    tris = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64).reshape(-1, 3)
    return points, tris


def _transformed(tri, points, trsf):
    """`points` moved by the face's location the way gp_Pnt.Transformed does it, checked against
    OpenCascade on a few nodes (node by node if the arithmetic doesn't match exactly)."""
    form, t = trsf.Form(), trsf.TranslationPart().Coord()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if form == gp_Identity:
        moved = points.copy()
    elif form == gp_Translation:
        moved = np.column_stack([x + t[0], y + t[1], z + t[2]])
    elif form == gp_Scale:
        s = trsf.ScaleFactor()
        moved = np.column_stack([x * s + t[0], y * s + t[1], z * s + t[2]])
    elif form == gp_PntMirror:
        moved = np.column_stack([-x + t[0], -y + t[1], -z + t[2]])
    else:
        m, s = trsf.HVectorialPart(), trsf.ScaleFactor()
        rows = [m.Value(r, 1) * x + m.Value(r, 2) * y + m.Value(r, 3) * z for r in (1, 2, 3)]
        if s != 1.0:
            rows = [v * s for v in rows]
        moved = np.column_stack([rows[r] + t[r] for r in range(3)])
    sample = np.unique(np.linspace(0, len(points) - 1, min(len(points), 16)).astype(int))
    exact = np.array([tri.Node(int(i) + 1).Transformed(trsf).Coord() for i in sample]).reshape(-1, 3)
    if np.array_equal(moved[sample], exact):
        return moved
    n = tri.NbNodes()
    return np.fromiter((c for i in range(1, n + 1) for c in tri.Node(i).Transformed(trsf).Coord()),
                       np.float64, 3 * n).reshape(-1, 3)


def _open_edges(faces, count):
    """(edges used by only one triangle, distinct edges)."""
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, uses = np.unique(e[:, 0] * np.int64(count) + e[:, 1], return_counts=True)
    return int((uses == 1).sum()), len(uses)
