"""Off-screen PNG render of a mesh (VTK), for quick checks without a slicer.

    python -m engine.preview part_wave.stl
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def _to_polydata(mesh):
    """VTK copy of the mesh, kept lean for big meshes (millions of triangles): float32 points,
    triangle lists without per-cell counts, and smooth normals without splitting vertices."""
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float32), deep=True))
    n = len(mesh.faces)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * n + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(mesh.faces, dtype=np.int64).ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SplittingOff()
    normals.ConsistencyOff()
    normals.ComputeCellNormalsOff()
    normals.Update()
    return normals.GetOutput()


def render(mesh, png, size=(1400, 1000)):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(_to_polydata(mesh))
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.72, 0.75, 0.80)
    prop.SetSpecular(0.4)
    prop.SetSpecularPower(30)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0.12, 0.13, 0.15)
    ren.SetBackground2(0.30, 0.32, 0.36)
    ren.GradientBackgroundOn()
    cam = ren.GetActiveCamera()
    cam.SetPosition(1, -1.4, 1.1)
    cam.SetFocalPoint(0, 0, 0)
    cam.SetViewUp(0, 0, 1)
    ren.ResetCamera()
    cam.Zoom(1.3)

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(ren)
    win.SetSize(*size)
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(png))
    writer.SetInputConnection(grab.GetOutputPort())
    writer.Write()


if __name__ == "__main__":
    src = Path(sys.argv[1])
    render(trimesh.load(src), src.with_suffix(".png"))
    print(f"wrote {src.with_suffix('.png')}")
