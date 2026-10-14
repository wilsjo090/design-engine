"""Write a simple filleted bracket to STEP so the pipeline can be tested without real parts.

    python tools/make_test_part.py samples/test_part.step
"""
import sys
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


def make_part():
    box = BRepPrimAPI_MakeBox(60.0, 40.0, 15.0).Shape()
    fillet = BRepFilletAPI_MakeFillet(box)
    exp = TopExp_Explorer(box, TopAbs_EDGE)
    while exp.More():
        fillet.Add(3.0, TopoDS.Edge(exp.Current()))
        exp.Next()
    shape = fillet.Shape()
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(30, 20, -1), gp_Dir(0, 0, 1)), 8.0, 17.0).Shape()
    return BRepAlgoAPI_Cut(shape, hole).Shape()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/test_part.step")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(make_part(), STEPControl_AsIs)
    writer.Write(str(out))
    print(f"wrote {out}")
