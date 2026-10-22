"""The node library. Each node is a small function registered with its typed ports.

Port types: number (plain value), int, scalar (number or field), implicit (body field),
cad (tessellated CAD body), mesh (output triangles), path, text.
"""
import re
from pathlib import Path

import numpy as np
import trimesh

from .. import backend
from .. import facemask
from .. import fields as F
from .. import lattice as LAT
from .. import tiles
from ..cache import (ArrayCache, DiskCache, available_memory, load_step_meshes, ram_cache_budget,
                     save_step_meshes)
from ..grid import Grid
from ..meshing import decimate, export, field_to_mesh
from ..meshing import reduce as reduce_mesh
from ..meshing import fuse as fuse_meshes
from ..transform import anchor_point, scale_about
from .core import GraphError, Many, Port, check_cancel, node, parallel_map, report_message, report_progress

# Expensive field arrays (distance fields, noise) kept between graph runs: a small RAM tier in
# this process, with the shared SSD cache behind it (see engine/cache.py).
ARRAY_CACHE = ArrayCache(max_bytes=ram_cache_budget()[0], disk=DiskCache())

S = "scalar"
HUGE_TESSELLATION = 3_000_000  # triangles per body: warn, everything downstream gets slow and heavy


# --- inputs ------------------------------------------------------------------------------
@node("input.number", "Number", "Inputs",
      [Port("value", "number", 0.0, doc="The number. Type it, or nudge it with the arrow keys (Shift = x10).")],
      [Port("out", "number", doc="The value, for any number or field input (an amplitude, a density...).")])
def number(value):
    """A constant value. Use one Number for a setting you want to change in one place, and connect
    it to every node that needs it."""
    return float(value)


@node("input.import_step", "Import STEP", "Inputs",
      [Port("path", "path", doc="The STEP file to read. SolidWorks: File > Save As > STEP (AP214)."),
       Port("tolerance", "number", 0.01, doc="How closely the triangles follow the CAD surfaces, mm: the "
                                             "accuracy. 0.01 is very accurate; 0.02-0.05 is plenty for 0.1 mm voxels."),
       Port("skip_duplicates", "int", 1, doc="1 = drop bodies that sit exactly on top of another (some exports "
                                             "contain doubles); 0 = keep every body."),
       Port("angle", "number", 0.5, doc="Largest angle between neighbouring triangles, radians. The tolerance "
                                        "already sets the accuracy; a small angle only adds triangles (0.1 gave "
                                        "12 million on an intricate part, 0.5 gave 1.9 million).")],
      [Port("bodies", "cad", doc="Every solid in the file, as a list. Nodes after it run once per body "
                                   "automatically; use Pick Body to work on just one.")])
def import_step(path, tolerance, skip_duplicates, angle):
    """Read all solid bodies from a STEP file. The triangulated bodies are kept on the SSD, so
    opening the same file (unchanged, same settings) again takes about a second."""
    if not path or not Path(path).exists():
        raise GraphError(f"STEP file not found: {path}")
    found = load_step_meshes(path, tolerance, angle)
    if found is None:
        # imported here: OpenCascade is large, and worker processes never need it
        from ..cad_import import load_step, tessellate
        bodies, found = load_step(path), []
        for i, body in enumerate(bodies):
            report_progress(i, len(bodies))
            found.append(tessellate(body, tolerance=tolerance, angular_tolerance=angle, check=check_cancel))
        save_step_meshes(path, tolerance, angle, found)
    meshes, seen = Many(), set()
    for i, mesh in enumerate(found):
        if len(mesh.faces) > HUGE_TESSELLATION:
            report_message(f"Import STEP: body {i + 1} has {len(mesh.faces) / 1e6:.1f} million triangles "
                           f"(tolerance {tolerance:g} mm, angle {angle:g}). Everything after it gets slow and "
                           f"heavy; a larger angle (e.g. 0.5-1.0) or tolerance gives far fewer.", "error")
        key = tuple(np.round(mesh.bounds.ravel(), 2))
        if skip_duplicates and key in seen:
            continue
        seen.add(key)
        mesh.metadata["cad_body"] = len(meshes)  # its number in the list (face selections "body:face")
        meshes.append(mesh)
    return meshes


@node("input.pick_body", "Pick Body", "Inputs",
      [Port("bodies", "cad", many=True, doc="A list of bodies, e.g. from Import STEP."),
       Port("index", "int", 0, doc="Which body: 0 = the first, 1 = the second... Negative counts from the end "
                                   "(-1 = the last).")],
      [Port("body", "cad", doc="That one body.")])
def pick_body(bodies, index):
    """One body from a list."""
    items = bodies if isinstance(bodies, list) else [bodies]
    if not -len(items) <= int(index) < len(items):
        raise GraphError(f"body index {index} out of range (file has {len(items)} bodies)")
    return items[int(index)]


# --- transforms --------------------------------------------------------------------------
_ANCHORS = {}  # (body, support) fingerprints -> anchor; finding contacts is the slow part


@node("cad.scale", "Scale Bodies", "Transform",
      [Port("body", "cad", label="Bodies", doc="The bodies to resize (one or a list)."),
       Port("scale", S, 1.0, doc="Size factor (1 = unchanged, 1.3 = 30% bigger). Connect a field to give each "
                                 "body its own size: the field is read at the body's anchor point."),
       Port("support", "cad", many=True, optional=True,
            doc="Optional: the part the bodies sit on (e.g. the lamp center). Each body is then scaled about "
                "the point where its contact faces meet, so it stays seated exactly.")],
      [Port("body", "cad", label="Bodies", doc="The resized bodies, in the same order.")])
def scale_bodies(body, scale, support):
    """Resize bodies in place. With a support connected, each body is scaled about the point where
    its contact faces with the support meet, so it stays seated exactly; otherwise about its centre."""
    sup = None
    if support is not None:
        items = support if isinstance(support, list) else [support]
        sup = items[0] if len(items) == 1 else trimesh.util.concatenate(list(items))
    key = (F.mesh_fingerprint(body), F.mesh_fingerprint(sup) if sup is not None else None)
    if key not in _ANCHORS:
        _ANCHORS[key] = anchor_point(body, sup)[0]
    anchor = _ANCHORS[key]
    factor = F.sample(scale, anchor)
    if factor <= 0:
        raise GraphError(f"scale must be greater than 0 (got {factor:.3g})")
    return scale_about(body, anchor, factor)


# --- fields ------------------------------------------------------------------------------
@node("field.coordinate", "Coordinate", "Fields",
      [Port("axis", "text", "x", choices=["x", "y", "z"], doc="Which coordinate: x, y or z.")],
      [Port("out", S, doc="The position along that axis at every point, mm. Patterns built on it change "
                          "along that direction (Cos of it gives straight waves).")])
def coordinate(axis):
    """The X, Y or Z position, in mm."""
    if axis not in F.AXES:
        raise GraphError(f"axis must be x, y or z, not '{axis}'")
    return F.AXES[axis]


@node("field.cylindrical", "Cylindrical Coordinate", "Fields",
      [Port("axis", "text", "z", choices=["x", "y", "z"], doc="The axis to measure around (the lamp's axis is y)."),
       Port("component", "text", "radius", choices=["radius", "angle"],
            doc="radius = distance from the axis, mm; angle = angle around it, radians (-3.14..3.14).")],
      [Port("out", S, doc="The radius (mm) or angle (radians) at every point.")])
def cylindrical(axis, component):
    """Distance from (radius, mm) or angle around (radians) a coordinate axis."""
    if axis not in F.AXES:
        raise GraphError(f"axis must be x, y or z, not '{axis}'")
    return F.radial(axis) if component == "radius" else F.angle(axis)


@node("field.noise", "Noise", "Fields",
      [Port("scale", "number", 10.0, doc="Size of the largest blobs, mm. Bigger = calmer, larger shapes."),
       Port("octaves", "int", 3, doc="Layers of finer detail added on top. 1-2 = smooth, 4+ = wrinkly."),
       Port("roughness", "number", 0.5, doc="How strong each finer layer is, 0..1. Higher = rougher."),
       Port("seed", "int", 0, doc="Any whole number. Change it for a different pattern with the same character.")],
      [Port("out", S, doc="Smooth random values, roughly -1..1. Remap it to the range you need, or multiply "
                          "it by a density and wave it for contour lines.")])
def noise(scale, octaves, roughness, seed):
    """Smooth random variation in 3D, roughly -1..1. Same seed = same pattern."""
    if scale <= 0:
        raise GraphError("scale must be greater than 0")
    return F.noise(scale, octaves, roughness, seed)


@node("field.face_mask", "Face Mask", "Fields",
      [Port("cad", "cad", label="Body", doc="The CAD bodies the effect goes on: the same bodies the texture is "
                                              "applied to (for the lamp: scaled.body)."),
       Port("faces", "faces", "", doc="The CAD faces where the effect applies. Click Pick, then in the 3D view: "
                                      "click a face to add/remove it, Ctrl+drag to paint over several, Shift+click "
                                      "for every face joined to it tangentially. Face numbers, or names of saved "
                                      "selections (the list button next to Pick)."),
       Port("fade", "number", 2.0, doc="Surface reach: width of the fade at the edge of the selection, mm (0 = hard "
                                       "edge). Volume reach: how far from the chosen faces the mask fades to 0, mm."),
       Port("position", "text", "inside", choices=list(facemask.POSITIONS),
            doc="Surface reach only: where the fade happens: inside the chosen faces (the effect never reaches the others), "
                "centred on their edge, or outside it (the effect spills onto the neighbours)."),
       Port("band", "number", 2.0, doc="Surface reach only: how far from the surface the mask is computed, mm. "
                                       "At least the largest offset."),
       Port("apply", "text", "chosen faces", choices=list(facemask.APPLY), label="Apply to",
            doc="chosen faces = the effect goes on the faces you pick. all other faces = it goes everywhere "
                "except the faces you pick: easier when only a few faces should stay plain (with no faces "
                "picked, everything gets it). Fade and Position then refer to the faces that get the effect."),
       Port("reach", "text", "surface", choices=list(facemask.REACHES),
            doc="surface = for textures: the mask lives on the surface (also right on thin walls). volume = "
                "for lattices and other effects inside the part: 1 at the chosen faces, fading to 0 at Fade mm "
                "from them, straight through the part (Position and Band do not apply). E.g. Face Mask -> "
                "Remap (Auto) -> a lattice's Thickness: solid at the chosen faces, opening into the lattice.")],
      [Port("out", S, doc="1 on the chosen faces, 0 elsewhere, smooth in between. Multiply a texture's "
                          "distance (or a lattice's thickness) by it.")])
def face_mask(cad, faces, fade, position, band, reach, apply="chosen faces"):
    """1 on the chosen CAD faces, 0 elsewhere, fading smoothly at the edge of the selection.
    Multiply a texture's distance by it to put the texture only on those faces. With a list of
    bodies a plain face number applies to every body (identical bodies like the petals), and
    "body:face" (e.g. 2:15) to that body only."""
    ids = facemask.cad_faces(cad)
    if ids is None:
        raise GraphError("this body has no CAD faces: connect a body from Import STEP")
    unknown = re.findall(r"[A-Za-z_][\w-]*", str(faces or ""))
    if unknown:
        raise GraphError(f"no face selection named '{unknown[0]}' in this recipe")
    body = facemask.cad_body(cad)
    chosen = facemask.faces_for_body(faces, body)
    wrong = [f for f in chosen if f > ids.max()]
    if wrong:
        raise GraphError(f"face {wrong[0]} does not exist (body {body} has faces 0-{ids.max()})")
    if fade < 0:
        raise GraphError("fade must be 0 or more")
    if apply not in facemask.APPLY:
        raise GraphError(f"Apply to must be one of: {', '.join(facemask.APPLY)}")
    return facemask.face_mask(cad, ids, chosen, fade, position, band, reach, invert=apply == "all other faces")


@node("field.wave3d", "3D Wave", "Fields",
      [Port("pattern", "text", "ridges", choices=list(F.PATTERNS),
            doc="ridges = straight-flanked waves along an axis; gyroid, schwarz, diamond = 3D cell patterns."),
       Port("axis", "text", "y", choices=["x", "y", "z"], doc="The direction the ridges repeat in (ridges only)."),
       Port("period", "number", 5.0, doc="Distance between ridges (or the cell size), mm."),
       Port("warp", "number", 0.0, doc="How far the ridges are bent in 3D by noise, mm. 0 = straight."),
       Port("warp_scale", "number", 8.0, doc="Size of the bends, mm."),
       Port("seed", "int", 0, doc="Change it for a different random bending.")],
      [Port("out", S, doc="The pattern, -1..1. Remap it to 0..1 and multiply by an amplitude for an offset.")])
def wave3d(pattern, axis, period, warp, warp_scale, seed):
    """A wave -1..1 whose ridges bend in all three directions (warp). Remap it to 0..1 for offsets."""
    if period <= 0 or warp_scale <= 0:
        raise GraphError("period and warp scale must be greater than 0")
    return F.wave3d(pattern, period, warp, warp_scale, seed, axis)


@node("math.remap", "Remap", "Math",
      [Port("x", S, doc="The value or field to remap."),
       Port("auto", "text", "no", choices=["no", "yes"],
            doc="yes = From Low and From High are taken from X's range (shown on the card of the node "
                "connected to X, e.g. 0 ... 1 for a Face Mask), so only To Low and To High need setting."),
       Port("from_low", S, -1.0, doc="The input value that becomes To Low (ignored with Auto)."),
       Port("from_high", S, 1.0, doc="The input value that becomes To High (ignored with Auto)."),
       Port("to_low", S, 0.0, doc="The output where the input equals From Low."),
       Port("to_high", S, 1.0, doc="The output where the input equals From High.")],
      [Port("out", S, doc="The remapped value. Not clamped: inputs beyond From Low..From High give outputs "
                          "beyond To Low..To High.")])
def remap(x, auto, from_low, from_high, to_low, to_high):
    """Stretch a range: From Low..From High becomes To Low..To High (e.g. noise -1..1 to 0..1). Every
    input takes a number or a field. With Auto, From Low/High come from X's own range."""
    x = _field(x)
    if auto == "yes":
        if x.range is None:
            raise GraphError("Auto cannot tell X's range (it depends on the part, e.g. a distance or a "
                             "coordinate): set From Low and From High yourself")
        from_low, from_high = x.range
        if from_high - from_low < 1e-9:
            raise GraphError(f"X is constant ({from_low:g}), there is no range to stretch")
    elif not isinstance(from_low, F.Field) and not isinstance(from_high, F.Field) and from_low == from_high:
        raise GraphError(f"From Low and From High are both {from_low:g}: there is no range to stretch")
    return F.remap(x, from_low, from_high, to_low, to_high)


def _binary(type_id, title, fn, doc, a, b, out):
    node(type_id, title, "Math", [Port("a", S, doc=a), Port("b", S, doc=b)], [Port("out", S, doc=out)],
         doc=doc)(fn)


def _unary(type_id, title, fn, doc, x, out):
    node(type_id, title, "Math", [Port("x", S, doc=x)], [Port("out", S, doc=out)], doc=doc)(lambda x: fn(x))


MIX = " Numbers and fields can be mixed: with a field the result is a field, point by point."


def _field(v):
    return v if isinstance(v, F.Field) else F._lift(v)


_binary("math.add", "Add", lambda a, b: _field(a) + b, "A + B." + MIX,
        "First value or field.", "Second value or field.", "A + B.")
_binary("math.subtract", "Subtract", lambda a, b: _field(a) - b, "A - B." + MIX,
        "The value or field to subtract from.", "The value or field to subtract.", "A - B.")
_binary("math.multiply", "Multiply", lambda a, b: _field(a) * b,
        "A x B." + MIX + " Multiplying by a 0..1 field (Smooth Step, Face Mask) fades an effect in and out.",
        "First factor.", "Second factor.", "A x B.")
_binary("math.divide", "Divide", lambda a, b: _field(a) / b, "A / B." + MIX,
        "The value or field to divide.", "What to divide by (not 0).", "A / B.")
_binary("math.min", "Min", lambda a, b: F.minimum(a, b),
        "The smaller of A and B at every point. On two bodies this is their union.",
        "First value, field or body.", "Second value, field or body.", "The smaller of A and B.")
_binary("math.max", "Max", lambda a, b: F.maximum(a, b),
        "The larger of A and B at every point. On two bodies this is their intersection.",
        "First value, field or body.", "Second value, field or body.", "The larger of A and B.")
_unary("math.sin", "Sin", F.sin, "Sine. A full wave every 2 x pi (6.28) of input.",
       "Angle in radians: multiply a coordinate (mm) by 2 x pi / period for waves period mm apart.", "sin(X), -1..1.")
_unary("math.cos", "Cos", F.cos, "Cosine. A full wave every 2 x pi (6.28) of input.",
       "Angle in radians: multiply a coordinate (mm) by 2 x pi / period for waves period mm apart.", "cos(X), -1..1.")
_unary("math.acos", "Acos", F.acos, "Arc cosine. acos(cos(x)) is a triangle wave: ridges with straight flanks "
       "instead of round ones.", "A value -1..1 (values outside are clipped).", "The angle, 0..pi (0..3.14).")
_unary("math.abs", "Abs", F.abs_, "Absolute value: drops the sign.", "Value or field.",
       "The value without its sign (always 0 or more).")
_unary("math.sqrt", "Sqrt", F.sqrt, "Square root.", "Value or field, 0 or more (negative values count as 0).",
       "The square root.")


@node("math.clamp", "Clamp", "Math",
      [Port("x", S, doc="Value or field."), Port("low", S, 0.0, doc="The smallest value allowed."),
       Port("high", S, 1.0, doc="The largest value allowed.")],
      [Port("out", S, doc="X limited to Low..High.")])
def clamp(x, low, high):
    """Limit x to the range low..high."""
    return F.clamp(_field(x), low, high)


@node("math.smoothstep", "Smooth Step", "Math",
      [Port("x", S, doc="Value or field, e.g. a distance from a body."),
       Port("edge0", S, 0.0, doc="Where the output starts rising: 0 at and below this."),
       Port("edge1", S, 1.0, doc="Where the output reaches 1: 1 at and above this. Equal to Edge0 = a sharp "
                                 "step; smaller than Edge0 = a fade out instead of in.")],
      [Port("out", S, doc="0..1 with a smooth S-curve between Edge0 and Edge1. Multiply an effect by it to fade "
                          "it in.")])
def smoothstep(x, edge0, edge1):
    """0 below edge0, 1 above edge1, smooth in between. Use it to fade an effect in or out."""
    return F.smoothstep(x, edge0, edge1)


# --- implicit bodies -----------------------------------------------------------------------
@node("implicit.from_cad", "Implicit Body from CAD Body", "Implicit",
      [Port("cad", "cad", doc="A CAD body (one or a list), from Import STEP, Pick Body or Scale Bodies."),
       Port("band", "number", 2.0, doc="Distances are exact within this many mm of the surface. Keep it at "
                                       "least the largest offset you apply.")],
      [Port("out", "implicit", doc="The body as a distance field: negative inside, positive outside, 0 on the "
                                   "surface. Feed it to Offset, the Booleans, Mesh...")])
def from_cad(cad, band):
    """Signed distance field of a CAD body (negative inside)."""
    return F.body_from_mesh(cad, band)


@node("implicit.offset", "Offset Body", "Implicit",
      [Port("body", "implicit", doc="The body to grow or shrink."),
       Port("distance", S, 0.0, doc="How far, mm: positive grows, negative shrinks. Connect a field (e.g. a "
                                    "wave) for a textured surface.")],
      [Port("out", "implicit", doc="The offset body.")])
def offset(body, distance):
    """Grow (positive) or shrink (negative) a body. Drive the distance with a field for textures."""
    return F.offset(body, distance)


@node("implicit.shell", "Shell", "Implicit",
      [Port("body", "implicit", doc="The body to hollow out."),
       Port("thickness", S, 1.0, doc="Wall thickness, mm (a number or a field).")],
      [Port("out", "implicit", doc="A wall of that thickness, centred on the original surface.")])
def shell(body, thickness):
    """Hollow wall of the given thickness, centred on the surface."""
    return F.shell(body, thickness)


_EDGE_PORTS = [
    Port("edge", "text", "sharp", choices=list(F.EDGES),
         doc="sharp = as the bodies meet. round = the edges where they meet rounded with Radius (exactly that "
             "radius where the surfaces meet at 90°). chamfer = cut at 45°, Radius along each surface."),
    Port("radius", "number", 0.5, doc="Round or chamfer only: the size of the edge, mm.")]


def _regrown(body, edge, radius):
    """A Hole Pattern computes its holes only a little way around them: grown as far as the edge
    reaches when it is rounded or chamfered."""
    if edge != "sharp" and radius > 0 and callable(getattr(body, "regrow", None)):
        return body.regrow(radius)
    return body


@node("implicit.union", "Boolean Union", "Implicit",
      [Port("a", "implicit", doc="First body."), Port("b", "implicit", doc="Second body."), *_EDGE_PORTS],
      [Port("out", "implicit", doc="Both bodies together.")])
def union(a, b, edge="sharp", radius=0.5):
    """Both bodies combined. Edge round or chamfer fills the inner edges where they meet (e.g. a
    fillet at the foot of raised shapes from a Hole Pattern)."""
    if edge not in F.EDGES:
        raise GraphError(f"Edge must be one of: {', '.join(F.EDGES)}")
    return F.union_edged(_regrown(a, edge, radius), _regrown(b, edge, radius), edge, radius)


@node("implicit.subtract", "Boolean Subtract", "Implicit",
      [Port("a", "implicit", doc="The body to keep."), Port("b", "implicit", doc="The body to cut away from it."),
       *_EDGE_PORTS],
      [Port("out", "implicit", doc="A with B removed.")])
def subtract(a, b, edge="sharp", radius=0.5):
    """A with B cut away. Edge round or chamfer softens the edges the cut leaves (e.g. the rims of a
    Hole Pattern's holes, on both sides of the wall)."""
    if edge not in F.EDGES:
        raise GraphError(f"Edge must be one of: {', '.join(F.EDGES)}")
    return F.subtract_edged(a, _regrown(b, edge, radius), edge, radius)


@node("implicit.intersect", "Boolean Intersect", "Implicit",
      [Port("a", "implicit", doc="First body."), Port("b", "implicit", doc="Second body.")],
      [Port("out", "implicit", doc="Only the volume that is inside both.")])
def intersect(a, b):
    """Only where both bodies overlap."""
    return F.intersect(a, b)


@node("implicit.smooth_union", "Smooth Union", "Implicit",
      [Port("a", "implicit", doc="First body."), Port("b", "implicit", doc="Second body."),
       Port("radius", "number", 1.0, doc="Size of the rounded blend where the bodies meet, mm (0 = a sharp "
                                         "union).")],
      [Port("out", "implicit", doc="Both bodies joined, with a fillet-like blend between them.")])
def smooth_union(a, b, radius):
    """Union with a fillet-like blend of about `radius` mm."""
    return F.smooth_union(a, b, radius)


@node("implicit.smoothen", "Smoothen Body", "Implicit",
      [Port("body", "implicit", doc="The body to round off."),
       Port("iterations", "int", 3, doc="Smoothing passes. More = rounder, softer edges."),
       Port("grid_size", "number", 0.1, doc="mm. Together with Iterations it sets how far the rounding "
                                            "reaches.")],
      [Port("out", "implicit", doc="The smoothed body.")])
def smoothen(body, iterations, grid_size):
    """Round off creases. Looks the same at any preview quality."""
    return F.smoothen(body, int(iterations), float(grid_size))


def _point(text, name):
    from ..fea.model import ConditionError, parse_vector
    try:
        return parse_vector(text)
    except ConditionError:
        raise GraphError(f"{name} must be three numbers x, y, z in mm, like 10, 0, 25 (got '{text}')") from None


def _size(value, name):
    if not float(value) > 0:
        raise GraphError(f"{name} must be more than 0 mm")
    return float(value)


_SHAPE_USE = (" Coordinates in mm, as in the CAD part (the 3D view's axes). Use it as Topology Optimization's "
              "Prefer (a guide), Keep or Avoid body, or with the Booleans.")


@node("implicit.ball", "Ball", "Implicit",
      [Port("centre", "text", "0, 0, 0", doc="Where the middle is: x, y, z in mm."),
       Port("radius", "number", 5.0, doc="mm.")],
      [Port("out", "implicit", doc="The ball.")],
      doc="A ball at a point." + _SHAPE_USE)
def ball(centre, radius):
    return F.ball(_point(centre, "centre"), _size(radius, "radius"))


@node("implicit.bar", "Bar", "Implicit",
      [Port("start", "text", "0, 0, 0", doc="One end: x, y, z in mm (the middle of its round end)."),
       Port("end", "text", "20, 0, 0", doc="The other end: x, y, z in mm."),
       Port("radius", "number", 3.0, doc="mm: half the bar's thickness.")],
      [Port("out", "implicit", doc="The bar.")],
      doc="A round bar between two points, with round ends: e.g. a guide along the path a member should "
          "take." + _SHAPE_USE)
def bar(start, end, radius):
    return F.bar(_point(start, "start"), _point(end, "end"), _size(radius, "radius"))


@node("implicit.box", "Box", "Implicit",
      [Port("centre", "text", "0, 0, 0", doc="Where the middle is: x, y, z in mm."),
       Port("size", "text", "20, 20, 20", doc="Size along x, y, z in mm."),
       Port("rounding", "number", 0.0, doc="mm: the edges rounded by this much (0 = sharp).")],
      [Port("out", "implicit", doc="The box.")],
      doc="A box, its edges optionally rounded." + _SHAPE_USE)
def box(centre, size, rounding):
    dims = _point(size, "size")
    if min(dims) <= 0:
        raise GraphError("size must be more than 0 mm along x, y and z")
    return F.box(_point(centre, "centre"), dims, float(rounding))


# --- lattices ------------------------------------------------------------------------------
_DOMAIN_DOC = ("Optional: a body whose box the lattice fills, so it can be previewed on its own. To put the "
               "lattice into a part, use Fill Body with Lattice.")
_SPACE_DOC = ("The lattice as an implicit body that fills all of space. Put it into a part with Fill Body with "
              "Lattice (or Boolean Intersect).")


def _fills_box(field):
    """A lattice previewed in a Domain's box fills that box: being cut off at its edges is the
    point, not something to warn about."""
    field.fills_box = True
    return field


def _units_port(what):
    return Port("units", "text", "mm", choices=list(LAT.UNITS),
                doc=f"mm = {what} in mm. density = {what} as the share of solid material instead, 0-1: 1 = completely "
                    "solid, 0.3 = 30% solid (worked out from this lattice's own shape). Use density for fades with a "
                    "Face Mask: the lattice then opens up evenly, while in mm most of the change happens in a short "
                    "stretch (thicker struts than the lattice can hold all look the same: solid).")


@node("lattice.tpms", "TPMS Lattice", "Lattice",
      [Port("pattern", "text", "gyroid", choices=list(LAT.TPMS),
            doc="The surface: gyroid (smooth twisting channels), schwarz_p (round tubes along x, y and z), "
                "diamond (stiff, four-way), neovius (Schwarz P with extra openings), lidinoid (finer, more complex)."),
       Port("cell", "number", 5.0, doc="Cell size: how often the pattern repeats, mm."),
       Port("mode", "text", "sheet", choices=list(LAT.TPMS_MODES),
            doc="sheet = a thin wall following the surface (both channel systems stay open); solid = all the "
                "material on one side of the surface; solid (other side) = the other side."),
       Port("thickness", S, 0.6, doc="Sheet mode: wall thickness, mm. Connect a field for walls that get thicker "
                                     "or thinner across the part (graded)."),
       Port("offset", S, 0.0, doc="Solid modes: moves the surface, mm. Positive = more material."),
       Port("domain", "cad", optional=True, doc=_DOMAIN_DOC),
       _units_port("Thickness (sheet) or Offset (solid modes)")],
      [Port("out", "implicit", doc=_SPACE_DOC)])
def tpms_lattice(pattern, cell, mode, thickness, offset, domain, units):
    """A triply periodic minimal surface (TPMS) lattice: smooth walls with no
    sharp nodes, good for printing."""
    if cell <= 0:
        raise GraphError("cell size must be greater than 0")
    return _fills_box(LAT.tpms(pattern, cell, thickness, mode, offset, domain.bounds if domain is not None else None,
                               units))


@node("lattice.struts", "Strut Lattice", "Lattice",
      [Port("pattern", "text", "bcc", choices=list(LAT.STRUTS),
            doc="cubic = struts along the cell edges; bcc = diagonals through the cell centre; fcc = diagonals "
                "across the cell faces; octet = fcc plus an octahedron (very stiff); cubic + bcc = both."),
       Port("cell", "number", 5.0, doc="Cell size, mm."),
       Port("thickness", S, 0.8, doc="Strut diameter, mm. Connect a field for struts that get thicker or "
                                     "thinner across the part (graded)."),
       Port("blend", "number", 0.0, doc="Rounds the nodes where struts meet, mm. 0 = sharp joints."),
       Port("domain", "cad", optional=True, doc=_DOMAIN_DOC),
       _units_port("Thickness")],
      [Port("out", "implicit", doc=_SPACE_DOC)])
def strut_lattice(pattern, cell, thickness, blend, domain, units):
    """A beam lattice repeating in every direction."""
    if cell <= 0:
        raise GraphError("cell size must be greater than 0")
    return _fills_box(LAT.struts(pattern, cell, thickness, blend, domain.bounds if domain is not None else None, units))


@node("lattice.voronoi", "Voronoi Lattice", "Lattice",
      [Port("domain", "cad", doc="The body to fill: the foam is grown in its box. With a list of bodies "
                                 "each gets its own foam."),
       Port("cell", "number", 5.0, doc="Average cell size, mm."),
       Port("thickness", S, 0.8, doc="Strut diameter, mm. Connect a field for graded struts."),
       Port("randomness", "number", 1.0, doc="0 = regular box-like cells, 1 = a fully irregular foam."),
       Port("seed", "int", 0, doc="Change it for a different random foam."),
       _units_port("Thickness")],
      [Port("out", "implicit", doc="The foam as an implicit body (inside the domain's box). Put it into the "
                                   "part with Fill Body with Lattice.")])
def voronoi_lattice(domain, cell, thickness, randomness, seed, units):
    """An organic foam of struts along the edges of random (Voronoi) cells."""
    if cell <= 0:
        raise GraphError("cell size must be greater than 0")
    try:
        return _fills_box(LAT.voronoi(domain.bounds, cell, thickness, randomness, seed, units))
    except ValueError as err:
        raise GraphError(str(err)) from err


@node("lattice.fill", "Fill Body with Lattice", "Lattice",
      [Port("body", "implicit", doc="The part to fill, as an implicit body (e.g. from Implicit Body from CAD Body)."),
       Port("lattice", "implicit", doc="The lattice: from TPMS, Strut or Voronoi Lattice."),
       Port("skin", S, 0.0, doc="A solid skin along the part's surface, mm thick. 0 = no skin: the lattice is "
                                "just cut to the part's shape.")],
      [Port("out", "implicit", doc="The part filled with the lattice (and its skin). Mesh it like any body.")])
def fill_with_lattice(body, lattice, skin):
    """Put a lattice inside a part, optionally keeping a solid skin (like a Boolean of a lattice
    with a shelled part)."""
    return LAT.fill(_field(body), _field(lattice), skin)


# --- mesh / output -------------------------------------------------------------------------
MAX_VOXELS = 300_000_000  # ~1.2 GB per field array; beyond this a typo would freeze the PC
PEAK_ARRAYS = 12          # grid-sized float32 arrays alive at once while meshing (measured ~8-10)
TILED_ESTIMATE = int(1.5 * 2**30)  # a body meshed in blocks: one block at a time plus its result


def grid_memory(voxels):
    """Bytes needed to mesh a grid with this many voxels (arrays + distance-query temporaries)."""
    return int(voxels * 4 * PEAK_ARRAYS)


def array_module(voxels):
    """cupy (GPU) for big grids when enabled and the job fits in the card's free memory, else
    numpy (CPU)."""
    if backend.use_gpu_for(voxels):
        free, _ = backend.gpu_memory()
        if grid_memory(voxels) < 0.7 * free:
            return backend.cupy()
    return np


def _finish(mesh, keep, voxel, tolerance):
    """Reduce a whole body's mesh: to a tolerance if one is given, else keep a share."""
    if tolerance > 0:
        V, F = reduce_mesh(mesh.vertices, mesh.faces, float(tolerance), check=check_cancel)
        return trimesh.Trimesh(V, F, process=False)
    return decimate(mesh, keep, voxel)


def _report_mesh(mesh, body, cut_hint):
    """Warnings about a body just meshed (see meshing.field_to_mesh): samples that were not numbers,
    and a body cut off by its box (`cut_hint`: what to do about it; None = cutting is intended)."""
    bad = mesh.metadata.get("nonfinite", 0)
    if bad:
        report_message(f"{bad:,} samples were not a number (NaN) or infinite and were meshed as outside "
                       "(inside for minus infinity). Usually a Divide by 0 upstream.", "warning")
    if mesh.metadata.get("clipped") and cut_hint and not getattr(body, "fills_box", False):
        report_message(f"The body reaches the edge of the box it is meshed in and is cut flat there: {cut_hint}",
                       "warning")


def _mesh(body, bounds, voxel, padding, keep, tolerance=0.0, cut_hint=None):
    if voxel <= 0:
        raise GraphError("voxel size must be greater than 0")
    grid = Grid.from_bounds(bounds, voxel, padding)
    if grid.size > tiles.TILE_MIN_VOXELS:  # big body: in blocks, only where its surface is
        if grid.size > tiles.MAX_TILED_VOXELS:
            raise GraphError(f"grid too large ({grid.size / 1e9:,.0f} billion voxels); increase the voxel size")
        try:
            mesh, computed, total = tiles.mesh_tiled(body, bounds, voxel, padding, keep, parallel_map, tolerance)
        except ValueError as err:
            raise GraphError(str(err)) from err
        _report_mesh(mesh, body, cut_hint)
        report_message(f"Meshed in blocks: {computed} of {total} blocks have surface "
                       f"({grid.size / 1e6:,.0f} million voxels in the box)")
        return mesh
    if grid.size > MAX_VOXELS:
        raise GraphError(f"grid too large ({grid.size / 1e6:,.0f} million voxels); increase the voxel size")
    xp = array_module(grid.size)
    if xp is not np:
        try:
            return _finish(_mesh_on(body, grid, xp, cut_hint), keep, voxel, tolerance)
        except Exception as err:
            if not backend.is_out_of_memory(err):
                raise
            # another program (or worker) took the GPU memory: do this body on the CPU instead
    need, free = grid_memory(grid.size), available_memory()
    if need > 0.8 * free:
        raise GraphError(f"this needs about {need / 2**30:.1f} GB of memory but only {free / 2**30:.1f} GB is "
                         f"free; use a larger voxel size or close other programs")
    return _finish(_mesh_on(body, grid, np, cut_hint), keep, voxel, tolerance)


def _mesh_on(body, grid, xp, cut_hint):
    try:
        values = F.Context(grid, ARRAY_CACHE, xp).eval_full(body)
        mesh = field_to_mesh(values, grid)
        _report_mesh(mesh, body, cut_hint)
        return mesh
    finally:
        values = None
        if xp is not np:
            backend.release_gpu_memory()  # hand GPU memory back between bodies


def _mesh_estimate(body=None, voxel=0.1, padding=2.0, domain=None, **_):
    """Memory one mesh.from_implicit call needs (used to decide how many run in parallel)."""
    body = _field(body)
    bounds, padding = (domain.bounds, padding) if domain is not None else F.mesh_box(body, padding, voxel)
    if bounds is None or voxel <= 0:
        return 0
    size = Grid.from_bounds(bounds, voxel, padding).size
    return TILED_ESTIMATE if size > tiles.TILE_MIN_VOXELS else grid_memory(size)


VIEW_MIN_CELLS = 40  # in a preview of many bodies, each still gets this many voxels along its longest side


@node("mesh.from_implicit", "Mesh from Implicit Body", "Mesh",
      [Port("body", "implicit", doc="The implicit body (one or a list) to turn into triangles."),
       Port("voxel", "number", 0.1, doc="Grid size, mm: smaller = finer detail but slower. Use about a third of "
                                        "the finest feature. Draft and Medium previews use 4x and 2x this."),
       Port("padding", "number", 2.0, doc="Extra room around the body's box, mm. Offsets by a number or a "
                                          "field of known size (shown on its card) are already included; "
                                          "you are warned when the surface is cut off."),
       Port("domain", "cad", optional=True, doc="Optional: mesh inside this body's box instead of the body's own."),
       Port("tolerance", "number", 0.0,
            doc="Smart reduction: how far the surface may move, mm (e.g. 0.01); 0 = off. Triangles are only "
                "removed where the shape does not need them (flat faces become a few big ones, textures keep "
                "theirs), done in the worker processes while meshing, so it is much faster than a Reduce Mesh "
                "node afterwards. Replaces Keep when set. Used at Final quality and for exports only."),
       Port("keep", "number", 1.0, doc="Share of triangles kept after simplifying: 1 = all, 0.3 = 30%. Smaller "
                                       "files; used at Final quality and for exports only.")],
      [Port("mesh", "mesh", doc="A watertight triangle mesh (a list if the body was a list).")],
      parallel=True, estimate=_mesh_estimate)
def mesh_from_implicit(body, voxel, padding, domain, keep, tolerance):
    """Turn an implicit body into a watertight triangle mesh (marching cubes)."""
    body = _field(body)
    if domain is not None:  # meshed inside the domain's box: cutting the body there is the point
        return _mesh(body, domain.bounds, voxel, padding, keep, tolerance)
    if body.bounds is None:
        raise GraphError("cannot tell where this body is; connect a Domain")
    bounds, room = F.mesh_box(body, padding, voxel)
    return _mesh(body, bounds, voxel, room, keep, tolerance, f"increase Padding (now {padding:g} mm)")


@node("mesh.view", "View Mesh", "Mesh",
      [Port("body", "implicit", many=True), Port("cells", "int", 160), Port("padding", "number", 2.0)],
      [Port("mesh", "mesh")], hidden=True)
def view_mesh(body, cells, padding):
    """Used by the editor to preview an implicit body, or a list of them: about `cells` voxels
    along the longest side of the whole group (so 144 petals together get the detail of one body
    their size, not 144 times that), but at least VIEW_MIN_CELLS per body."""
    bodies = [_field(b) for b in _items(body)]
    if any(b.bounds is None for b in bodies):
        raise GraphError("this has no size of its own (a lattice fills all of space): connect a Domain, or "
                         "show the result of Fill Body with Lattice")
    boxes = [F.mesh_box(b, padding) for b in bodies]  # offsets that reach further than padding get more
    room = max(r for _, r in boxes)
    lo = np.min([b[0] for b, _ in boxes], axis=0)
    hi = np.max([b[1] for b, _ in boxes], axis=0)
    group = float(np.max(hi - lo)) + 2 * room
    largest = max(float(np.max(b[1] - b[0])) for b, _ in boxes) + 2 * room
    voxel = max(min(group / cells, largest / VIEW_MIN_CELLS), 1e-3)
    args = [dict(body=b, voxel=voxel, padding=padding) for b in bodies]
    need = max(_mesh_estimate(**a) for a in args)
    meshes = parallel_map(_view_one, args, need=need, gpu=need >= grid_memory(backend.GPU_MIN_VOXELS),
                          title="View Mesh")
    return Many(meshes) if isinstance(body, list) else meshes[0]


def _view_one(body, voxel, padding):
    bounds, room = F.mesh_box(body, padding, voxel)
    return _mesh(body, bounds, voxel, room, 1.0, cut_hint=(
        f"it reaches more than {padding:g} mm beyond its own box (an offset by a field whose size is not known?). "
        "Mesh from Implicit Body needs at least that much Padding."))


@node("mesh.from_cad", "Mesh from CAD", "Mesh",
      [Port("cad", "cad", label="Body", doc="A CAD body (one or a list).")],
      [Port("mesh", "mesh", doc="Its triangles, as a mesh for Merge Meshes or Export.")])
def mesh_from_cad(cad):
    """The CAD body's own triangles from Import STEP, without remeshing: exact and instant. Use it
    for parts that get no texture."""
    return cad


def _items(mesh):
    return list(mesh) if isinstance(mesh, list) else [mesh]


def _merged(mesh):
    items = _items(mesh)
    return items[0] if len(items) == 1 else trimesh.util.concatenate(items)


REDUCE_BYTES = 700  # per triangle while reducing (pyfqmr's tables plus the copies around it)


@node("mesh.reduce", "Reduce Mesh", "Mesh",
      [Port("mesh", "mesh", doc="The mesh to make lighter (one or a list), e.g. from Mesh from Implicit Body."),
       Port("tolerance", "number", 0.01,
            doc="How far the surface may move, mm. Triangles are only removed where that keeps the shape: flat "
                "faces become a few big triangles, textures and curves keep theirs. 0.01 kept 19% of the key "
                "cabinet's triangles (surface within 0.006 mm); larger values remove more (on curved detail the "
                "surface can move up to ~2x the value).")],
      [Port("mesh", "mesh", doc="The lighter mesh: watertight, every edge shared by exactly two triangles.")],
      parallel=True, estimate=lambda mesh=None, **_: REDUCE_BYTES * sum(len(m.faces) for m in _items(mesh)) if mesh is not None else 0)
def reduce_mesh_node(mesh, tolerance):
    """Fewer triangles only where they are not needed (unlike Keep in Mesh from Implicit Body,
    which removes a share everywhere): a smart reduction for smaller files."""
    if tolerance <= 0:
        raise GraphError("tolerance must be greater than 0")
    before = len(mesh.faces)
    V, F = reduce_mesh(mesh.vertices, mesh.faces, float(tolerance), check=check_cancel)
    if len(F) == before:
        report_message(f"Reduce Mesh: nothing could be removed safely within {tolerance:g} mm", "warning")
    return trimesh.Trimesh(V, F, process=False)


@node("mesh.merge", "Merge Meshes", "Mesh",
      [Port("a", "mesh", many=True, doc="Meshes to combine (one or a list)."),
       Port("b", "mesh", many=True, optional=True, doc="Optional: more meshes, e.g. the lamp center."),
       Port("fuse", "text", "yes", choices=["yes", "no"],
            doc="yes = Boolean union into one solid (parts must overlap to join, touching is not enough); "
                "no = only put them in one file. Draft and Medium previews never fuse.")],
      [Port("mesh", "mesh", doc="One mesh with every part (one solid when fused).")])
def merge_meshes(a, b, fuse):
    """Combine meshes into one, e.g. the textured petals and the center they sit on."""
    items = _items(a) + (_items(b) if b is not None else [])
    if fuse != "yes":
        return _merged(items)
    try:
        mesh, pieces, specks = fuse_meshes(items)
    except ValueError as err:
        raise GraphError(str(err)) from err
    note = f" (removed {specks} specks and bubbles under 1 mm³)" if specks else ""
    if pieces == 1:
        report_message(f"Merge Meshes: fused {len(items)} meshes into one solid{note}")
    else:
        report_message(f"Merge Meshes: {len(items)} meshes gave {pieces} separate solids; parts that do not "
                       f"overlap stay separate{note}", "error")
    return mesh


@node("output.export", "Export Mesh", "Output",
      [Port("mesh", "mesh", many=True, doc="The mesh to save. A list goes into one file."),
       Port("path", "path", "out.stl", doc="Where to save: .stl, .3mf, .obj or .ply. Missing folders are created.")],
      [Port("path", "path", doc="The file that was written.")], side_effect=True)
def export_mesh(mesh, path):
    """Write the mesh (a list is merged into one file)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    export(_merged(mesh), path)
    return str(path)


@node("output.preview", "Preview Image", "Output",
      [Port("mesh", "mesh", many=True, doc="The mesh to render. A list is shown together."),
       Port("path", "path", "preview.png", doc="The PNG file to write.")],
      [Port("path", "path", doc="The file that was written.")], side_effect=True)
def preview_image(mesh, path):
    """Render a PNG of the mesh."""
    from ..preview import render
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    render(_merged(mesh), path)
    return str(path)
