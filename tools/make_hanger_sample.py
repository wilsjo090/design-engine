"""Write the tool hanger sample to STEP: a screwed-on plate with a web under it that holds three shafts.

    python tools/make_hanger_sample.py samples/hanger_web.step

Two solids: 0 = the design space (plate and web as one body, 203 x 28 x 57 mm), 1 = the plate alone (the Keep
body of recipes/topo_hanger_extruded.json). The plate has rounded corners and top edges and countersunk screw
holes; the web under it is a wedge, as deep as the plate where they meet and 9 mm at the bottom, its front
flush with the plate's, with sloped ends, round bottom corners and rounded outer edges: what Topology
Optimization cannot round, since these edges are the design space's own (its Round edges rounds the rims of
the openings it makes). Prints the CAD faces of the screw holes and the shaft holes on the design space, as
the recipe's Fixed Faces and Loads name them.
"""
import sys
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS

LENGTH, DEPTH, HEIGHT = 203.0, 28.0, 57.0
PLATE = 6.0                      # the plate's thickness (its top is the part's top)
WEB_BOTTOM = 9.0                 # the web's thickness at z = 0 (from its front, y = 0); the plate's depth at its top
ENDS = (16.0, 4.0)               # the web's ends slope: x at z = 0 and under the plate (mirrored at the other end)
SCREWS = [(22.0 + 31.8 * i, DEPTH / 2) for i in range(6)]                    # (x, y), 4.8 mm, along z
SHAFTS = [(27.0, 11.0), (LENGTH / 2, 11.0), (LENGTH - 27.0, 11.0)]           # (x, z), 3.7 mm, along y
SCREW_D, SHAFT_D, SINK_D = 4.8, 3.7, 10.5
PLATE_CORNER, PLATE_TOP = 9.0, 3.5       # the plate's rounded corners (seen from above) and top edges
WEB_CORNER, WEB_EDGE = 8.0, 2.0          # the web's bottom corners (seen from the front) and outer edges
UNDER = HEIGHT - PLATE                   # the plate's underside
TOP = UNDER + 1.0                        # the web's top: 1 mm into the plate, so they fuse


def _box(lo, hi):
    return BRepPrimAPI_MakeBox(gp_Pnt(*lo), gp_Pnt(*hi)).Shape()


def _cut(shape, tool):
    return BRepAlgoAPI_Cut(shape, tool).Shape()


def _prism(points, vector):
    """The polygon `points` (3D, in one plane) swept along `vector`."""
    polygon = BRepBuilderAPI_MakePolygon()
    for p in points:
        polygon.Add(gp_Pnt(*p))
    polygon.Close()
    return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(polygon.Wire()).Face(), gp_Vec(*vector)).Shape()


def _fillet(shape, radius, chosen):
    """`shape` with the edges for which chosen(first end, last end) holds rounded."""
    edges, explorer = [], TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():       # (every edge comes once per face it borders)
        edge = TopoDS.Edge(explorer.Current())
        if not any(edge.IsSame(e) for e in edges):
            edges.append(edge)
        explorer.Next()
    maker, count = BRepFilletAPI_MakeFillet(shape), 0
    for edge in edges:
        a, b = (BRep_Tool.Pnt_s(v) for v in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)))
        if chosen(np.array([a.X(), a.Y(), a.Z()]), np.array([b.X(), b.Y(), b.Z()])):
            maker.Add(radius, edge)
            count += 1
    if not count:
        raise ValueError("no edge to round")
    return maker.Shape()


def _back(z):
    """The y of the web's sloped back face at height z."""
    return WEB_BOTTOM + (DEPTH - WEB_BOTTOM) * z / TOP


def make_parts():
    plate = _box((0, 0, UNDER), (LENGTH, DEPTH, HEIGHT))
    plate = _fillet(plate, PLATE_CORNER, lambda a, b: abs(a[2] - b[2]) > 1)               # the four upright edges
    plate = _fillet(plate, PLATE_TOP, lambda a, b: min(a[2], b[2]) > HEIGHT - 1e-6)
    holes = []
    for x, y in SCREWS:
        holes.append(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, UNDER - 1), gp_Dir(0, 0, 1)),
                                              SCREW_D / 2, PLATE + 2).Shape())
        deep = (SINK_D - SCREW_D) / 2            # a 90 degree countersink
        holes.append(BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, y, HEIGHT - deep), gp_Dir(0, 0, 1)),
                                          SCREW_D / 2, SINK_D / 2 + 0.01, deep + 0.01).Shape())
    for hole in holes:
        plate = _cut(plate, hole)
    # the web: a wedge along x, its ends cut off on a slope
    web = _prism([(0, 0, 0), (0, _back(0), 0), (0, _back(TOP), TOP), (0, 0, TOP)], (LENGTH, 0, 0))
    lean = (ENDS[0] - ENDS[1]) / UNDER           # x per z along a sloped end
    for sign, x0 in ((1.0, 0.0), (-1.0, LENGTH)):
        at = lambda z: x0 + sign * (ENDS[0] - lean * z)
        web = _cut(web, _prism([(x0 - sign, -1, -1), (at(-1), -1, -1), (at(TOP + 1), -1, TOP + 1),
                                (x0 - sign, -1, TOP + 1)], (0, DEPTH + 2, 0)))
    web = _fillet(web, WEB_CORNER, lambda a, b: abs(a[1] - b[1]) > 1 and max(a[2], b[2]) < 1e-6)   # bottom corners
    # the edges round its front and its back, up to the plate
    on_face = lambda p: abs(p[1]) < 1e-6 or abs(p[1] - _back(p[2])) < 1e-6
    web = _fillet(web, WEB_EDGE, lambda a, b: on_face(a) and on_face(b) and abs(a[1] - b[1]) < 1e-3
                  and min(a[2], b[2]) < TOP - 1)
    for x, z in SHAFTS:
        web = _cut(web, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, -1, z), gp_Dir(0, 1, 0)),
                                                 SHAFT_D / 2, DEPTH + 2).Shape())
    space = BRepAlgoAPI_Fuse(plate, web).Shape()
    unify = ShapeUpgrade_UnifySameDomain(space, True, True, False)   # (the plate's front and the web's: one face)
    unify.Build()
    space = unify.Shape()
    for hole in holes:   # (the web under a screw hole: the hole goes on down 1 mm)
        space = _cut(space, hole)
    return space, plate


def faces_round(mesh, axis, centres, radius):
    """Per centre the CAD faces of `mesh` whose triangles lie on the cylinder of `radius` round it along `axis`."""
    ids = mesh.face_attributes["cad_face"]
    corners = mesh.vertices[mesh.faces]          # (n, 3, 3): every corner on the cylinder
    across = [k for k in range(3) if k != axis]
    out = []
    for c in centres:
        d = np.hypot(corners[:, :, across[0]] - c[0], corners[:, :, across[1]] - c[1])
        out.append(sorted({int(i) for i in ids[(np.abs(d - radius) < 0.02).all(axis=1)]}))
    return out


def faces_on_sinks(mesh):
    """Per screw the CAD faces of `mesh` that are its countersink's cone."""
    ids = mesh.face_attributes["cad_face"]
    corners = mesh.vertices[mesh.faces]
    start = HEIGHT - (SINK_D - SCREW_D) / 2      # where the cone meets the hole
    out = []
    for x, y in SCREWS:
        r, z = np.hypot(corners[:, :, 0] - x, corners[:, :, 1] - y), corners[:, :, 2]
        on = (np.abs(r - (SCREW_D / 2 + z - start)) < 0.03) & (z > start - 1e-3)
        out.append(sorted({int(i) for i in ids[on.all(axis=1)]}))
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/hanger_web.step")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    for shape in make_parts():
        writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(out))
    print(f"wrote {out}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine.cad_import import load_step, tessellate
    from engine.meshing import signed_volume
    solids = load_step(str(out))
    space, plate = (tessellate(s, 0.01, 0.5) for s in solids[:2])
    print("design space", space.bounds.round(3).tolist(), "faces", int(space.face_attributes["cad_face"].max()) + 1,
          "volume", round(float(signed_volume(space.vertices, space.faces)), 1),
          "plate", round(float(signed_volume(plate.vertices, plate.faces)), 1))
    print("screw holes:", faces_round(space, 2, SCREWS, SCREW_D / 2))
    print("their countersinks (Fixed Faces of recipes/topo_hanger.json):", faces_on_sinks(space))
    print("shaft holes (Loads):", faces_round(space, 1, SHAFTS, SHAFT_D / 2))
