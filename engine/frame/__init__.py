"""Frame (truss) optimization: parts built from bars of one profile, joined at nodes.

Ground structure (`ground`) -> minimum-volume layout LP with member adding (`layout`) -> sizes
with buckling (`sizing`) -> 3D frame check (`beam`), orchestrated by `optimize.optimize_frame`.
Plan and status: docs/FRAME_PLAN.md.
"""
from .optimize import Frame, optimize_frame
from .profiles import KINDS, Profile, Section

__all__ = ["Frame", "optimize_frame", "Profile", "Section", "KINDS"]
