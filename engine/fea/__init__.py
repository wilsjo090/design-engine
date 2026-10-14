"""Finite element analysis on voxel grids (static stress analysis; topology optimization later).

The part is cut into cube elements on a regular grid (8-node hexahedra), partly filled elements
are as stiff as the share of them inside the part, and the displacement is solved matrix-free with
conjugate gradients and a geometric multigrid preconditioner, on numpy or CuPy arrays. Units: mm,
N, MPa (N/mm^2). See docs/TOPOLOGY_PLAN.md.
"""
