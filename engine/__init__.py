"""The design engine. The names below are imported when first used (`from engine import load_step`), not
when the package is: a CPU worker (engine/stages.py) needs only meshing's modules, and importing them used to
load OpenCascade, libigl and the field library too (~0.2 GB in every such process)."""
import importlib

_EXPORTS = {
    "ArrayCache": "cache", "load_step": "cad_import", "tessellate": "cad_import", "Grid": "grid",
    "decimate": "meshing", "export": "meshing", "field_to_mesh": "meshing",
    **{name: "fields" for name in (
        "Angle", "Context", "Field", "Radius", "X", "Y", "Z", "abs_", "acos", "angle", "asin", "body_from_mesh",
        "clamp", "cos", "intersect", "maximum", "minimum", "offset", "radial", "shell", "sin", "smooth_union",
        "smoothen", "smoothstep", "sqrt", "subtract", "tan", "union")},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
