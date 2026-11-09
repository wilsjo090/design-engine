"""The node library. Each node is a small function registered with its typed ports.

Port types: number (plain value), int, scalar (number or field), implicit (body field),
cad (tessellated CAD body), mesh (output triangles), path, text.
"""
import os
import re
from pathlib import Path

import numpy as np
import trimesh

from .. import backend
from .. import facemask
from .. import fields as F
from .. import lattice as LAT
from .. import offset_faces as OF
from .. import tiles
from .. import trace
from ..cache import (LIMIT_OF_TOTAL, ArrayCache, DiskCache, load_step_meshes, load_step_names, memory_room,
                     physical_memory, ram_cache_budget, save_step_meshes, save_step_names)
from ..dualcontour import dual_contour
from ..grid import Grid
from .. import meshing
from ..meshing import export, extract_surface, field_to_mesh
from ..meshing import reduce as reduce_mesh
from ..meshing import fuse as fuse_meshes
from ..stages import cpu_stage, splitting
from ..transform import anchor_point, face_plane, scale_about, transformed
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
       Port("tolerance", "number", 0.01, unit="mm", doc="How closely the triangles follow the CAD surfaces, mm: the "
                                             "accuracy. 0.01 is very accurate; 0.02-0.05 is plenty for 0.1 mm voxels."),
       Port("skip_duplicates", "int", 1, switch=True,
            doc="On = drop bodies that sit exactly on top of another (some exports contain doubles); off = keep "
                "every body."),
       Port("angle", "number", 0.5, unit="rad", label="Max angle",
            doc="Largest angle between neighbouring triangles, radians. The tolerance already sets the accuracy; a "
                "small angle only adds triangles (0.1 gave 12 million on an intricate part, 0.5 gave 1.9 million).")],
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
        (bodies, names), found = load_step(path, names=True), []
        for i, body in enumerate(bodies):
            report_progress(i, len(bodies), "body")
            found.append(tessellate(body, tolerance=tolerance, angular_tolerance=angle, check=check_cancel))
        save_step_meshes(path, tolerance, angle, found)
        save_step_names(path, tolerance, angle, names)
    else:  # None for imports kept before names were: step_names reads them when the editor asks
        names = load_step_names(path, tolerance, angle)
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
        mesh.metadata["cad_solid"] = i  # its solid in the file (names count the skipped doubles too)
        if names is not None and i < len(names):
            mesh.metadata["cad_name"] = names[i]
        meshes.append(mesh)
    return meshes


def step_names(path, tolerance, angle):
    """The names of the file's solids (cad_import.load_step), kept beside its meshes; read from the file when
    an import kept before names were has none yet (the whole file again: up to half a minute for a big one)."""
    names = load_step_names(path, tolerance, angle)
    if names is None:
        from ..cad_import import load_step
        names = load_step(path, names=True)[1]
        save_step_names(path, tolerance, angle, names)
    return names


@node("input.pick_body", "Pick Body", "Inputs",
      [Port("bodies", "cad", many=True, doc="A list of bodies, e.g. from Import STEP."),
       Port("index", "int", 0, pick="body",
            doc="Which body: 0 = the first, 1 = the second... Negative counts from the end (-1 = the last). Click "
                "the field for a list of the bodies with their names from the STEP file.")],
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
       Port("scale", S, 1.0, unit="×", doc="Size factor (1 = unchanged, 1.3 = 30% bigger). Connect a field to give "
                                           "each body its own size: the field is read at the body's anchor point."),
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


# Move, Rotate, Mirror and Pattern take the whole list (many=True), so a list turns about the middle of all
# of it, as one; a single body gives a single body back, a list a list.
_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_TURN_DOC = ("The axis to turn about: x, y, z, or custom (a direction of your own). Positive angles turn "
             "anticlockwise looking down the axis towards its start (right-hand rule).")
MAX_COPIES = 1000


def _listed(body):
    return list(body) if isinstance(body, list) else [body]


def _returned(body, items):
    return Many(items) if isinstance(body, list) or len(items) != 1 else items[0]


def _axis(axis, vector):
    d = np.asarray(_AXES[axis] if axis in _AXES else _point(vector, "vector"), dtype=np.float64)
    if np.linalg.norm(d) < 1e-12:
        raise GraphError("vector must not be 0, 0, 0: it gives the axis' direction")
    return d / np.linalg.norm(d)


def _middle(bodies):
    """The middle of the bodies' box, all of them together."""
    return (np.min([b.bounds[0] for b in bodies], axis=0) + np.max([b.bounds[1] for b in bodies], axis=0)) / 2


@node("cad.move", "Move Bodies", "Transform",
      [Port("body", "cad", label="Bodies", many=True, doc="The bodies to move (one or a list)."),
       Port("offset", "text", "0, 0, 0", doc="How far, along x, y and z, mm.")],
      [Port("body", "cad", label="Bodies", doc="The moved bodies, in the same order.")])
def move_bodies(body, offset):
    """Move bodies by a distance along x, y and z. Their faces keep their numbers, so faces picked on them
    (Face Mask, patterns, supports and loads) still apply."""
    m = trimesh.transformations.translation_matrix(_point(offset, "offset"))
    return _returned(body, [transformed(b, m) for b in _listed(body)])


@node("cad.rotate", "Rotate Bodies", "Transform",
      [Port("body", "cad", label="Bodies", many=True, doc="The bodies to turn (one or a list: a list turns as one)."),
       Port("angle", "number", 90.0, unit="°", doc="How far to turn, degrees."),
       Port("axis", "text", "z", choices=["x", "y", "z", "custom"], doc=_TURN_DOC),
       Port("vector", "text", "0, 0, 1", when={"axis": ["custom"]}, doc="The custom axis' direction: x, y, z."),
       Port("about", "text", "centre", choices=["centre", "origin", "point"],
            doc="Where the axis passes: through the middle of the bodies (of all of them together), through the "
                "origin (0, 0, 0), or through a point of your own."),
       Port("point", "text", "0, 0, 0", when={"about": ["point"]}, doc="A point on the axis: x, y, z in mm.")],
      [Port("body", "cad", label="Bodies", doc="The turned bodies, in the same order.")])
def rotate_bodies(body, angle, axis, vector, about, point):
    """Turn bodies about an axis. Their faces keep their numbers, so faces picked on them still apply."""
    bodies = _listed(body)
    through = (_middle(bodies) if about == "centre" else np.zeros(3) if about == "origin"
               else _point(point, "point"))
    m = trimesh.transformations.rotation_matrix(np.radians(float(angle)), _axis(axis, vector), through)
    return _returned(body, [transformed(b, m) for b in bodies])


@node("cad.mirror", "Mirror Bodies", "Transform",
      [Port("cad", "cad", label="Bodies", many=True, doc="The bodies to mirror (one or a list)."),
       Port("plane", "text", "YZ", choices=["YZ", "XZ", "XY", "face"],
            doc="The mirror plane: YZ (mirrors x), XZ (mirrors y), XY (mirrors z), or a flat face of the bodies."),
       Port("position", "number", 0.0, unit="mm", when={"plane": ["YZ", "XZ", "XY"]},
            doc="Where the plane is: how far from the origin along x (YZ), y (XZ) or z (XY)."),
       Port("faces", "faces", "", label="Face", when={"plane": ["face"]},
            doc="A flat face of the bodies, picked in the 3D view: the mirror plane lies on it. Mirroring half a "
                "part about its cut face gives the whole part."),
       Port("keep", "text", "yes", choices=["no", "yes"], label="Keep original",
            doc="yes = the originals and the mirrored bodies (originals first); no = only the mirrored bodies.")],
      [Port("body", "cad", label="Bodies", doc="The mirrored bodies (after the originals with Keep original).")])
def mirror_bodies(cad, plane, position, faces, keep):
    """Mirror bodies about a plane, keeping the originals or not. The copies' faces keep their numbers, so
    a face picked on a body is the same face on its mirror image. Where the original and its mirror image
    touch (half a part mirrored about its cut face) they are still two bodies: join them with Merge Meshes
    (fuse) or Boolean Union."""
    bodies = _listed(cad)
    if plane == "face":
        if not facemask.parse_faces(faces):
            raise GraphError("pick the face to mirror about (Face: Pick)")

        def chosen(b):
            ids = facemask.cad_faces(b)
            wanted = facemask.faces_for_body(faces, facemask.cad_body(b))
            return None if ids is None or not wanted else np.isin(ids, wanted)
        try:
            point, normal = face_plane(bodies, chosen)
        except ValueError as err:
            raise GraphError(f"Face: {err}") from None
    else:
        normal = np.asarray(_AXES["xyz"[["YZ", "XZ", "XY"].index(plane)]])
        point = normal * float(position)
    m = trimesh.transformations.reflection_matrix(point, normal)
    mirrored = [transformed(b, m) for b in bodies]
    return _returned(cad, (bodies if keep == "yes" else []) + mirrored)


@node("cad.pattern", "Pattern Bodies", "Transform",
      [Port("body", "cad", label="Bodies", many=True, doc="The bodies to repeat (one or a list)."),
       Port("kind", "text", "linear", choices=["linear", "circular"],
            doc="linear = copies in a row (or rows: more settings); circular = copies round an axis."),
       Port("count", "int", 3, doc="How many in all, the original included."),
       Port("spacing", "text", "20, 0, 0", when={"kind": ["linear"]},
            doc="From one copy to the next: x, y, z in mm."),
       Port("rows", "int", 1, when={"kind": ["linear"]}, advanced=True,
            doc="A second direction: how many rows of copies (1 = one row)."),
       Port("row_spacing", "text", "0, 20, 0", when={"kind": ["linear"]}, advanced=True,
            doc="From one row to the next: x, y, z in mm."),
       Port("axis", "text", "z", choices=["x", "y", "z", "custom"], when={"kind": ["circular"]}, doc=_TURN_DOC),
       Port("vector", "text", "0, 0, 1", when={"kind": ["circular"], "axis": ["custom"]},
            doc="The custom axis' direction: x, y, z."),
       Port("point", "text", "0, 0, 0", label="Through", when={"kind": ["circular"]},
            doc="A point on the axis: x, y, z in mm."),
       Port("angle", "number", 360.0, unit="°", when={"kind": ["circular"]},
            doc="How far the copies spread: 360 = evenly all round (360 / Count apart); less = the first and the "
                "last copy are this far apart.")],
      [Port("body", "cad", label="Bodies",
            doc="All bodies as a list: the originals first, then each copy of them in turn. Nodes after it run once "
                "per body; Merge Meshes joins them into one file.")])
def pattern_bodies(body, kind, count, spacing, rows, row_spacing, axis, vector, point, angle):
    """Copies of bodies in a row, a grid or round an axis, like a linear or circular pattern in CAD. The
    copies' faces keep their numbers, so a face picked on the original is picked on every copy."""
    bodies, count = _listed(body), int(count)
    rows = int(rows) if kind == "linear" else 1
    if count < 1 or rows < 1:
        raise GraphError("Count and Rows must be at least 1")
    if count * rows * len(bodies) > MAX_COPIES:
        raise GraphError(f"{count * rows * len(bodies)} bodies is too many (at most {MAX_COPIES})")
    if kind == "linear":
        step = np.asarray(_point(spacing, "spacing"))
        row_step = np.asarray(_point(row_spacing, "row spacing")) if rows > 1 else np.zeros(3)
        moves = [trimesh.transformations.translation_matrix(i * step + j * row_step)
                 for j in range(rows) for i in range(count)]
    else:
        total = float(angle)
        apart = total / count if abs(total) >= 360 - 1e-9 else total / max(count - 1, 1)
        d, through = _axis(axis, vector), _point(point, "point")
        moves = [trimesh.transformations.rotation_matrix(np.radians(i * apart), d, through) for i in range(count)]
    return _returned(body, [transformed(b, m) for m in moves for b in bodies])


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
      [Port("scale", "number", 10.0, unit="mm", doc="Size of the largest blobs, mm. Bigger = calmer, larger shapes."),
       Port("octaves", "int", 3, doc="Layers of finer detail added on top. 1-2 = smooth, 4+ = wrinkly."),
       Port("roughness", "number", 0.5, slider=(0, 1), doc="How strong each finer layer is, 0..1. Higher = rougher."),
       Port("seed", "int", 0, doc="Any whole number. Change it for a different pattern with the same character.")],
      [Port("out", S, doc="Smooth random values, roughly -1..1. Connect it to a Texture for bumps, Remap it to "
                          "the range you need, or multiply it by a density and wave it for contour lines.")])
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
       Port("fade", "number", 2.0, unit="mm", zero="hard edge",
            doc="Surface reach: width of the fade at the edge of the selection, mm (0 = hard edge). Volume reach: how "
                "far from the chosen faces the mask fades to 0, mm."),
       Port("position", "text", "inside", choices=list(facemask.POSITIONS), when={"reach": ["surface"]},
            doc="Surface reach only: where the fade happens: inside the chosen faces (the effect never reaches the "
                "others), centred on their edge, or outside it (the effect spills onto the neighbours)."),
       Port("band", "number", 2.0, when={"reach": ["surface"]}, unit="mm",
            doc="Surface reach only: how far from the surface the mask is computed, mm. At least the largest offset."),
       Port("apply", "text", "chosen faces", choices=list(facemask.APPLY), label="Apply to",
            doc="chosen faces = the effect goes on the faces you pick. all other faces = it goes everywhere "
                "except the faces you pick: easier when only a few faces should stay plain (with no faces "
                "picked, everything gets it). Fade and Position then refer to the faces that get the effect."),
       Port("reach", "text", "surface", choices=list(facemask.REACHES),
            doc="surface = for textures: the mask lives on the surface (also right on thin walls). volume = "
                "for lattices and other effects inside the part: 1 at the chosen faces, fading to 0 at Fade mm "
                "from them, straight through the part (Position and Band do not apply). E.g. Face Mask -> "
                "Remap (Auto) -> a lattice's Thickness: solid at the chosen faces, opening into the lattice.")],
      [Port("out", S, doc="1 on the chosen faces, 0 elsewhere, smooth in between. Connect it to a Texture's Fade "
                          "(or multiply a lattice's thickness by it).")])
def face_mask(cad, faces, fade, position, band, reach, apply="chosen faces"):
    """1 on the chosen CAD faces, 0 elsewhere, fading smoothly at the edge of the selection.
    Connect it to a Texture's Fade to put the texture only on those faces (Invert fade: everywhere
    but there). With a list of
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
       Port("period", "number", 5.0, unit="mm", doc="Distance between ridges (or the cell size), mm."),
       Port("warp", "number", 0.0, unit="mm", zero="straight",
            doc="How far the ridges are bent in 3D by noise, mm. 0 = straight."),
       Port("warp_scale", "number", 8.0, unit="mm", label="Warp size", doc="Size of the bends, mm."),
       Port("seed", "int", 0, doc="Change it for a different random bending.")],
      [Port("out", S, doc="The pattern, -1..1. Connect it to a Texture (it stretches -1..1 to 0..Height).")])
def wave3d(pattern, axis, period, warp, warp_scale, seed):
    """A wave -1..1 whose ridges bend in all three directions (warp). Put it on a part with Texture."""
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
                          "beyond To Low..To High.")], numeric=True)
def remap(x, auto, from_low, from_high, to_low, to_high):
    """Stretch a range: From Low..From High becomes To Low..To High (e.g. noise -1..1 to 0..1). Every
    input takes a number or a field. With Auto, From Low/High come from X's own range."""
    if _plain(x, from_low, from_high, to_low, to_high) and auto != "yes":
        lo, hi = float(from_low), float(from_high)
        if lo == hi:
            raise GraphError(f"From Low and From High are both {lo:g}: there is no range to stretch")
        return _number((float(x) - lo) * ((float(to_high) - float(to_low)) / (hi - lo)) + float(to_low), "Remap")
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


def _plain(*values):
    """Whether all of a Math node's values are plain numbers (no field among them)."""
    return not any(isinstance(v, F.Field) for v in values)


def _number(value, title):
    """A Math node's result from plain numbers: a plain number (float64, so 0.1 + 0.2 stays 0.3 to the
    eye), which can drive number inputs (a Voxel size, a Spacing)."""
    value = float(value)
    if not np.isfinite(value):
        raise GraphError(f"{title} gives {value:g} from these numbers")
    return value


def _binary(type_id, title, fn, plain, doc, a, b, out):
    def call(a, b):
        return _number(plain(float(a), float(b)), title) if _plain(a, b) else fn(a, b)
    node(type_id, title, "Math", [Port("a", S, doc=a), Port("b", S, doc=b)], [Port("out", S, doc=out)],
         doc=doc, numeric=True)(call)


def _unary(type_id, title, fn, plain, doc, x, out):
    def call(x):
        return _number(plain(float(x)), title) if _plain(x) else fn(x)
    node(type_id, title, "Math", [Port("x", S, doc=x)], [Port("out", S, doc=out)], doc=doc, numeric=True)(call)


MIX = (" Numbers and fields can be mixed: with a field the result is a field, point by point. With numbers only "
       "it is a number, which can set any number input (spacing = 1.5 x size).")


def _field(v):
    return v if isinstance(v, F.Field) else F._lift(v)


def _divide(a, b):
    if b == 0:
        raise GraphError("Divide: B is 0")
    return a / b


_binary("math.add", "Add", lambda a, b: _field(a) + b, lambda a, b: a + b, "A + B." + MIX,
        "First value or field.", "Second value or field.", "A + B.")
_binary("math.subtract", "Subtract", lambda a, b: _field(a) - b, lambda a, b: a - b, "A - B." + MIX,
        "The value or field to subtract from.", "The value or field to subtract.", "A - B.")
_binary("math.multiply", "Multiply", lambda a, b: _field(a) * b, lambda a, b: a * b,
        "A x B." + MIX + " Multiplying by a 0..1 field (Smooth Step, Face Mask) fades an effect in and out.",
        "First factor.", "Second factor.", "A x B.")
_binary("math.divide", "Divide", lambda a, b: _field(a) / b, _divide, "A / B." + MIX,
        "The value or field to divide.", "What to divide by (not 0).", "A / B.")
_binary("math.min", "Min", lambda a, b: F.minimum(a, b), min,
        "The smaller of A and B at every point. On two bodies this is their union.",
        "First value, field or body.", "Second value, field or body.", "The smaller of A and B.")
_binary("math.max", "Max", lambda a, b: F.maximum(a, b), max,
        "The larger of A and B at every point. On two bodies this is their intersection.",
        "First value, field or body.", "Second value, field or body.", "The larger of A and B.")
_unary("math.sin", "Sin", F.sin, np.sin, "Sine. A full wave every 2 x pi (6.28) of input.",
       "Angle in radians: multiply a coordinate (mm) by 2 x pi / period for waves period mm apart.", "sin(X), -1..1.")
_unary("math.cos", "Cos", F.cos, np.cos, "Cosine. A full wave every 2 x pi (6.28) of input.",
       "Angle in radians: multiply a coordinate (mm) by 2 x pi / period for waves period mm apart.", "cos(X), -1..1.")
_unary("math.acos", "Acos", F.acos, lambda x: np.arccos(np.clip(x, -1, 1)),
       "Arc cosine. acos(cos(x)) is a triangle wave: ridges with straight flanks instead of round ones.",
       "A value -1..1 (values outside are clipped).", "The angle, 0..pi (0..3.14).")
_unary("math.abs", "Abs", F.abs_, abs, "Absolute value: drops the sign.", "Value or field.",
       "The value without its sign (always 0 or more).")
_unary("math.sqrt", "Sqrt", F.sqrt, lambda x: np.sqrt(max(x, 0.0)), "Square root.",
       "Value or field, 0 or more (negative values count as 0).", "The square root.")


@node("math.clamp", "Clamp", "Math",
      [Port("x", S, doc="Value or field."), Port("low", S, 0.0, doc="The smallest value allowed."),
       Port("high", S, 1.0, doc="The largest value allowed.")],
      [Port("out", S, doc="X limited to Low..High.")], numeric=True)
def clamp(x, low, high):
    """Limit x to the range low..high."""
    if _plain(x, low, high):
        return _number(min(max(float(x), float(low)), float(high)), "Clamp")
    return F.clamp(_field(x), low, high)


@node("math.smoothstep", "Smooth Step", "Math",
      [Port("x", S, doc="Value or field, e.g. a distance from a body."),
       Port("edge0", S, 0.0, label="0 at", doc="Where the output starts rising: 0 at and below this."),
       Port("edge1", S, 1.0, label="1 at", doc="Where the output reaches 1: 1 at and above this. Equal to 0 at = a "
                                               "sharp step; smaller than 0 at = a fade out instead of in.")],
      [Port("out", S, doc="0..1 with a smooth S-curve between 0 at and 1 at. Multiply an effect by it to fade "
                          "it in.")], numeric=True)
def smoothstep(x, edge0, edge1):
    """0 below edge0, 1 above edge1, smooth in between. Use it to fade an effect in or out."""
    if _plain(x, edge0, edge1):
        x, e0, e1 = float(x), float(edge0), float(edge1)
        t = float(x >= e0) if e0 == e1 else min(max((x - e0) / (e1 - e0), 0.0), 1.0)
        return _number(t * t * (3 - 2 * t), "Smooth Step")
    return F.smoothstep(x, edge0, edge1)


# --- implicit bodies -----------------------------------------------------------------------
@node("implicit.from_cad", "Implicit Body from CAD Body", "Implicit",
      [Port("cad", "cad", doc="A CAD body (one or a list), from Import STEP, Pick Body or Scale Bodies."),
       Port("band", "number", 2.0, unit="mm", doc="Distances are exact within this many mm of the surface. Keep it at "
                                       "least the largest offset you apply.")],
      [Port("out", "implicit", doc="The body as a distance field: negative inside, positive outside, 0 on the "
                                   "surface. Feed it to Offset, the Booleans, Mesh...")])
def from_cad(cad, band):
    """Signed distance field of a CAD body (negative inside)."""
    return F.body_from_mesh(cad, band)


@node("implicit.offset", "Offset Body", "Implicit",
      [Port("body", "implicit", doc="The body to grow or shrink."),
       Port("distance", S, 0.0, unit="mm", doc="How far, mm: positive grows, negative shrinks. Connect a field (e.g. a "
                                               "wave) for a textured surface.")],
      [Port("out", "implicit", doc="The offset body.")])
def offset(body, distance):
    """Grow (positive) or shrink (negative) a body. Drive the distance with a field for textures
    (the Texture node does the height, the range and the fade of a texture in one)."""
    return F.offset(body, distance)


TEXTURE_RANGES = ("auto", "as is")


@node("implicit.texture", "Texture", "Implicit",
      [Port("body", "implicit", doc="The part to put the texture on."),
       Port("pattern", S, doc="The texture: a Surface Pattern, Text or Picture (0..1), Noise or a 3D Wave (-1..1), "
                              "or any field."),
       Port("height", S, 0.5, unit="mm", doc="How far the texture stands out at its highest, mm (raised shapes, "
                                             "bumps). Negative sinks it into the part instead (dimples, engraved "
                                             "text). A field varies it over the part."),
       Port("fade", S, 1.0, label="Fade", slider=(0, 1),
            doc="Where the texture goes: connect a Face Mask (1 on its faces, fading to 0 off them), a Smooth Step or "
                "any 0..1 field; the height is multiplied by it. 1 = wherever the pattern is."),
       Port("invert_fade", "text", "no", choices=["no", "yes"], label="Invert fade",
            doc="yes = the texture goes where Fade is 0 instead (1 - Fade): e.g. it stays off a Face Mask's faces "
                "and fades in away from them."),
       Port("stretch", "text", "auto", choices=list(TEXTURE_RANGES), label="Pattern range", advanced=True,
            doc="auto = the pattern's own range (shown on its card: -1 ... 1 for Noise and waves) is stretched to "
                "0..1 first, so the surface stays where it was at the pattern's lowest points and Height is how far "
                "its highest stand out (a Surface Pattern's 0..1 stays as it is). as is = the pattern's values "
                "times Height (Noise then goes both in and out).")],
      [Port("out", "implicit", doc="The textured part. Mesh it, or texture it again.")])
def texture(body, pattern, height, fade, invert_fade="no", stretch="auto"):
    """A texture on the part in one node: the pattern (0..1) times Height, faded by Fade, added to the
    surface. The same as Multiply, Multiply and Offset Body, without the wiring."""
    if stretch not in TEXTURE_RANGES:
        raise GraphError(f"Pattern range must be one of: {', '.join(TEXTURE_RANGES)}")
    if invert_fade not in ("no", "yes"):
        raise GraphError("Invert fade must be no or yes")
    p = _field(pattern)
    if stretch == "auto":
        if p.range is None:
            report_message("Texture: the pattern's range cannot be told (it depends on the part): it is used as "
                           "it is. Remap it to 0..1, or set Pattern range to 'as is' to hide this.", "warning")
        elif p.range[1] - p.range[0] > 1e-9 and tuple(p.range) != (0.0, 1.0):
            p = F.remap(p, p.range[0], p.range[1], 0.0, 1.0)
    if invert_fade == "yes":
        fade = 1.0 - fade
    distance = p * height
    if isinstance(fade, F.Field) or fade != 1.0:
        distance = distance * fade
    return F.offset(body, distance)


_STEP_SIDES = ["straight", "round"]
_SHAPED = ["chamfer", "round", "smooth"]


@node("implicit.offset_faces", "Offset Faces", "Implicit",
      [Port("body", "implicit", doc="The part: Implicit Body from CAD Body of the CAD body below (its Band at least "
                                    "Distance plus the Foot size), or the part after other changes."),
       Port("cad", "cad", label="Faces of", doc="The CAD body the faces are on, the one the part is made from (e.g. "
                                                 "the Import STEP's bodies): Pick shows it."),
       Port("faces", "faces", "", doc="The CAD faces to move. Click Pick, then in the 3D view: click a face to add or "
                                      "remove it, Ctrl+drag to paint over several, Shift+click for every face joined to "
                                      "it tangentially. Face numbers, or names of saved selections."),
       Port("distance", S, 1.0, unit="mm",
            doc="How far the faces move, mm: positive raises them (a panel, a label area, a thicker lip), negative sinks "
                "them (a recess for an inlay, a thinner wall; deeper than the wall makes a hole). A field moves every "
                "part of the faces by its value there (a ramp, a bulge; positive in one place and negative in another "
                "works too)."),
       Port("sides", "text", "straight", choices=list(OF.SIDES),
            doc="How the moved faces meet the faces around them. straight = walls along the faces' normals, as if the "
                "faces were pushed out of (or into) the part; where a neighbouring face folds away (a box's sides when "
                "its top is raised, a hole's wall) the walls simply continue it. round = the faces' own offset, as "
                "Offset Body: the side is a quarter circle Distance wide, reaching that far past the faces' border. "
                "blend = a soft ramp Width wide just inside the border, no step (a Face Mask under a Texture). Where "
                "chosen faces meet at an outward fold they stay joined, rounded as Offset Body does."),
       Port("width", "number", 2.0, unit="mm", when={"sides": ["blend"]},
            doc="Blend only: how wide the ramp is, mm, inside the faces' border."),
       Port("edge", "text", "sharp", choices=list(OF.EDGES), when={"sides": _STEP_SIDES},
            doc="The step's outside edge: a raised face's top rim, a recess's rim. sharp; chamfer = cut at 45°, Edge "
                "size along both faces; round = a circular fillet of radius Edge size, tangent to both (G1); smooth = "
                "curvature continuous (G2): it starts more gently, reaches farther along the faces and is as deep in "
                "its middle as round. Chamfers stay crisp with Mesh from Implicit Body's dual contouring; a smooth "
                "edge needs a voxel of about Edge size / 4 to show. Left out where the step has no such edge (a "
                "pocket's floor raised or deepened: its walls continue the pocket's). Round sides have no top rim."),
       Port("edge_size", "number", 0.5, unit="mm", when={"sides": _STEP_SIDES, "edge": _SHAPED},
            doc="mm: the chamfer's legs or the fillet's radius. At most the step's height (larger is cut to it)."),
       Port("foot", "text", "sharp", choices=list(OF.EDGES), when={"sides": _STEP_SIDES},
            doc="The step's inside edge: where a raised face's walls meet the part, or a recess's floor meets its "
                "walls. The same choices as Edge, e.g. a chamfered top with a smooth foot. Left out where the walls just "
                "continue a neighbouring face (a box's top raised: its sides run on up, and a fillet would bulge out "
                "there); it runs out over 1.5 mm where the border meets such a place. A recess with round sides has "
                "no floor corner."),
       Port("foot_size", "number", 0.5, unit="mm", when={"sides": _STEP_SIDES, "foot": _SHAPED},
            doc="mm: the chamfer's legs or the fillet's radius. At most the step's height (larger is cut to it)."),
       Port("apply", "text", "chosen faces", choices=list(facemask.APPLY), label="Apply to", advanced=True,
            doc="chosen faces = the faces you pick move. all other faces = every face except those moves (the part "
                "thickened all round except where you picked)."),
       Port("max_distance", "number", 0.0, unit="mm", zero="auto", advanced=True,
            doc="Only for a field Distance: the largest distance in or out it gives, mm (larger values are cut to it). "
                "auto = from the field's range, shown on its card; set it when the range cannot be told (a field "
                "built from a Coordinate or a distance).")],
      [Port("out", "implicit", doc="The part with the faces moved. Texture it, cut holes in it, mesh it.")],
      doc="Move picked CAD faces out or in by a distance, as one piece: a raised panel or label area, a thicker lip, a "
          "recess for an inlay, a thinner wall. Straight sides stand on the faces' border; Edge and Foot shape the "
          "step's rim and its inside corner. Put it before textures and holes: a Face Mask or Surface Pattern on "
          "the same faces still finds the moved faces when its Band is at least Distance (plus the texture's "
          "height), and a Hole Pattern on them needs Faces moved by = Distance.")
def offset_faces(body, cad, faces, distance, sides, width, edge, edge_size, foot, foot_size, apply="chosen faces",
                 max_distance=0.0):
    ids = facemask.cad_faces(cad)
    if ids is None:
        raise GraphError("this body has no CAD faces: connect a body from Import STEP")
    unknown = re.findall(r"[A-Za-z_][\w-]*", str(faces or ""))
    if unknown:
        raise GraphError(f"no face selection named '{unknown[0]}' in this recipe")
    number = facemask.cad_body(cad)
    chosen = facemask.faces_for_body(faces, number)
    wrong = [f for f in chosen if f > ids.max()]
    if wrong:
        raise GraphError(f"face {wrong[0]} does not exist (body {number} has faces 0-{ids.max()})")
    for name, value, choices in (("Sides", sides, OF.SIDES), ("Edge", edge, OF.EDGES), ("Foot", foot, OF.EDGES),
                                 ("Apply to", apply, facemask.APPLY)):
        if value not in choices:
            raise GraphError(f"{name} must be one of: {', '.join(choices)}")
    if edge_size < 0 or foot_size < 0 or max_distance < 0:
        raise GraphError("Edge size, Foot size and Max distance must be 0 or more")
    if isinstance(distance, F.Field):
        if max_distance > 0:
            low, high = -float(max_distance), float(max_distance)
        elif distance.range is not None:
            low, high = distance.range
        else:
            raise GraphError("Distance is a field whose range cannot be told (it depends on the part): set Max "
                             "distance (under the card's more settings) to the largest distance it gives")
    else:
        low = high = float(distance)
    invert = apply == "all other faces"
    if max(abs(low), abs(high)) < 1e-9 or (not chosen and not invert):
        return body  # nothing moves
    if sides == "blend":
        if width <= 0:
            raise GraphError("Width must be more than 0 mm")
        mask = facemask.face_mask(cad, ids, chosen, width, "inside", max(abs(low), abs(high)) + 0.5, "surface",
                                  invert=invert)
        return F.offset(body, mask * distance)
    patch = OF.Patch(cad, ids, chosen, invert)
    if not len(patch):
        return body
    caps = OF.walls(cad, patch, max(high, 0.0), max(-low, 0.0))
    out = OF.offset_faces(body, patch, distance, low, high, sides, edge, edge_size, foot, foot_size, caps)
    out.note = _offset_note(patch, low, high, sides, edge, edge_size, foot, foot_size)
    return out


def _offset_note(patch, low, high, sides, edge, edge_size, foot, foot_size):
    """The card's note: what moved, and where Edge or Foot are left out or cut to the step."""
    count = len(set(patch.ids.tolist()))
    what = f"{count} face{'s' if count != 1 else ''}"
    if low == high:
        lines = [f"{what} {'raised' if high > 0 else 'sunk'} {abs(high):g} mm"]
    else:
        lines = [f"{what} moved {low:g} to {high:g} mm"]
    step = max(abs(low), abs(high))
    for name, kind, size, test, why in (
            ("foot", foot, foot_size, lambda s: s > 0.99, "the walls continue a neighbour"),
            ("edge", edge, edge_size, lambda s: s < -0.99, "the walls continue a neighbour")):
        if kind == "sharp" or sides != "straight" or size <= 0:
            continue
        share = patch.border_share(test)
        if share > 0.005:
            lines.append(f"no {name} along {share:.0%} of the border: {why}")
        most = step / OF.SMOOTH_SIZE if kind == "smooth" else step   # a smooth blend reaches 1.76 x its size
        if size > most + 1e-9:
            lines.append(f"{name} size cut to {most:.2g} mm (the step's height{' / 1.76 for smooth' * (kind == 'smooth')})")
    if patch.unmatched:
        lines.append(f"{patch.unmatched} border edges without a neighbouring face (cracks in the mesh)")
    return "\n".join(lines)


@node("implicit.shell", "Shell", "Implicit",
      [Port("body", "implicit", doc="The body to hollow out."),
       Port("thickness", S, 1.0, unit="mm", doc="Wall thickness, mm (a number or a field).")],
      [Port("out", "implicit", doc="A wall of that thickness, centred on the original surface.")])
def shell(body, thickness):
    """Hollow wall of the given thickness, centred on the surface."""
    return F.shell(body, thickness)


_EDGE_PORTS = [
    Port("edge", "text", "sharp", choices=list(F.EDGES),
         doc="sharp = as the bodies meet. round = the edges where they meet rounded with Radius (exactly that "
             "radius where the surfaces meet at 90°). chamfer = cut at 45°, Radius along each surface."),
    Port("radius", "number", 0.5, unit="mm", when={"edge": ["round", "chamfer"]},
         doc="Round or chamfer only: the size of the edge, mm.")]


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
       Port("radius", "number", 1.0, unit="mm", zero="sharp",
            doc="Size of the rounded blend where the bodies meet, mm (0 = a sharp union).")],
      [Port("out", "implicit", doc="Both bodies joined, with a fillet-like blend between them.")])
def smooth_union(a, b, radius):
    """Union with a fillet-like blend of about `radius` mm."""
    return F.smooth_union(a, b, radius)


@node("implicit.smoothen", "Smoothen Body", "Implicit",
      [Port("body", "implicit", doc="The body to round off."),
       Port("iterations", "int", 3, doc="Smoothing passes. More = rounder, softer edges."),
       Port("grid_size", "number", 0.1, unit="mm", doc="mm. Together with Iterations it sets how far the rounding "
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
       Port("radius", "number", 5.0, unit="mm", doc="mm.")],
      [Port("out", "implicit", doc="The ball.")],
      doc="A ball at a point." + _SHAPE_USE)
def ball(centre, radius):
    return F.ball(_point(centre, "centre"), _size(radius, "radius"))


@node("implicit.bar", "Bar", "Implicit",
      [Port("start", "text", "0, 0, 0", doc="One end: x, y, z in mm (the middle of its round end)."),
       Port("end", "text", "20, 0, 0", doc="The other end: x, y, z in mm."),
       Port("radius", "number", 3.0, unit="mm", doc="mm: half the bar's thickness.")],
      [Port("out", "implicit", doc="The bar.")],
      doc="A round bar between two points, with round ends: e.g. a guide along the path a member should "
          "take." + _SHAPE_USE)
def bar(start, end, radius):
    return F.bar(_point(start, "start"), _point(end, "end"), _size(radius, "radius"))


@node("implicit.box", "Box", "Implicit",
      [Port("centre", "text", "0, 0, 0", doc="Where the middle is: x, y, z in mm."),
       Port("size", "text", "20, 20, 20", doc="Size along x, y, z in mm."),
       Port("rounding", "number", 0.0, unit="mm", zero="sharp", doc="mm: the edges rounded by this much (0 = sharp).")],
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
    return Port("units", "text", "mm", choices=list(LAT.UNITS), advanced=True,
                doc=f"mm = {what} in mm. density = {what} as the share of solid material instead, 0-1: 1 = completely "
                    "solid, 0.3 = 30% solid (worked out from this lattice's own shape). Use density for fades with a "
                    "Face Mask: the lattice then opens up evenly, while in mm most of the change happens in a short "
                    "stretch (thicker struts than the lattice can hold all look the same: solid).")


@node("lattice.tpms", "TPMS Lattice", "Lattice",
      [Port("pattern", "text", "gyroid", choices=list(LAT.TPMS),
            doc="The surface: gyroid (smooth twisting channels), schwarz_p (round tubes along x, y and z), "
                "diamond (stiff, four-way), neovius (Schwarz P with extra openings), lidinoid (finer, more complex)."),
       Port("cell", "number", 5.0, unit="mm", doc="Cell size: how often the pattern repeats, mm."),
       Port("mode", "text", "sheet", choices=list(LAT.TPMS_MODES),
            doc="sheet = a thin wall following the surface (both channel systems stay open); solid = all the "
                "material on one side of the surface; solid (other side) = the other side."),
       Port("thickness", S, 0.6, when={"mode": ["sheet"]}, unit={"units": {"mm": "mm", "density": "solid"}},
            doc="Sheet mode: wall thickness, mm. Connect a field for walls that get thicker or thinner across the part "
                "(graded)."),
       Port("offset", S, 0.0, when={"mode": [m for m in LAT.TPMS_MODES if m != "sheet"]},
            unit={"units": {"mm": "mm", "density": "solid"}},
            doc="Solid modes: moves the surface, mm. Positive = more material."),
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
       Port("cell", "number", 5.0, unit="mm", doc="Cell size, mm."),
       Port("thickness", S, 0.8, unit={"units": {"mm": "mm", "density": "solid"}},
            doc="Strut diameter, mm. Connect a field for struts that get thicker or thinner across the part (graded)."),
       Port("blend", "number", 0.0, unit="mm", zero="sharp",
            doc="Rounds the nodes where struts meet, mm. 0 = sharp joints."),
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
       Port("cell", "number", 5.0, unit="mm", doc="Average cell size, mm."),
       Port("thickness", S, 0.8, unit={"units": {"mm": "mm", "density": "solid"}},
            doc="Strut diameter, mm. Connect a field for graded struts."),
       Port("randomness", "number", 1.0, slider=(0, 1), doc="0 = regular box-like cells, 1 = a fully irregular foam."),
       Port("seed", "int", 0, advanced=True, doc="Change it for a different random foam."),
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
       Port("skin", S, 0.0, unit="mm", zero="none", doc="A solid skin along the part's surface, mm thick. 0 = no skin: "
                                                        "the lattice is just cut to the part's shape.")],
      [Port("out", "implicit", doc="The part filled with the lattice (and its skin). Mesh it like any body.")])
def fill_with_lattice(body, lattice, skin):
    """Put a lattice inside a part, optionally keeping a solid skin (like a Boolean of a lattice
    with a shelled part)."""
    return LAT.fill(_field(body), _field(lattice), skin)


# --- mesh / output -------------------------------------------------------------------------
MAX_VOXELS = 300_000_000  # ~1.2 GB per field array; beyond this a typo would freeze the PC
PEAK_ARRAYS = 12          # grid-sized float32 arrays alive at once while meshing (measured ~8-10)
TILED_ESTIMATE = int(1.5 * 2**30)  # a body meshed in blocks: one block at a time plus its result
# how Mesh from Implicit Body finds the surface between the samples
MESHERS = {"marching cubes": field_to_mesh, "dual contouring": dual_contour}


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


def _cut_hint(body, cut_hint):
    """What to do about a body cut off by its box (see meshing.report_mesh), None when cutting is intended."""
    return None if getattr(body, "fills_box", False) else cut_hint


def _mesh(body, bounds, voxel, padding, keep, tolerance=0.0, cut_hint=None, method="marching cubes", normals=False):
    """A body's mesh (in blocks when big). normals: with the field's normals at its vertices
    (meshing.FIELD_NORMAL; the editor's views only, never a node's mesh: marching cubes, no Keep or Tolerance)."""
    if voxel <= 0:
        raise GraphError("voxel size must be greater than 0")
    if method not in MESHERS:
        raise GraphError(f"Method must be one of: {', '.join(MESHERS)}")
    grid = Grid.from_bounds(bounds, voxel, padding)
    hint = _cut_hint(body, cut_hint)
    if grid.size > tiles.TILE_MIN_VOXELS:  # big body: in blocks, only where its surface is
        if grid.size > tiles.MAX_TILED_VOXELS:
            raise GraphError(f"grid too large ({grid.size / 1e9:,.0f} billion voxels); increase the voxel size")
        try:
            mesh, computed, total = tiles.mesh_tiled(body, bounds, voxel, padding, keep, parallel_map, tolerance,
                                                     method=method, normals=normals)
        except ValueError as err:
            raise GraphError(str(err)) from err
        meshing.report_mesh(mesh, hint)
        report_message(f"Meshed in blocks: {computed} of {total} blocks have surface "
                       f"({grid.size / 1e6:,.0f} million voxels in the box)")
        return mesh
    if grid.size > MAX_VOXELS:
        raise GraphError(f"grid too large ({grid.size / 1e6:,.0f} million voxels); increase the voxel size")
    xp = array_module(grid.size)
    finish = dict(grid=grid, method=method, cut_hint=hint, keep=keep, voxel=voxel, tolerance=tolerance)
    # (normals only for what is not reduced afterwards: a view)
    more = {"normals": True} if normals and method == "marching cubes" and keep >= 1.0 and tolerance <= 0 else {}
    if xp is not np:
        try:
            sampled = _mesh_on(body, grid, xp, method, **more)
        except Exception as err:
            if not backend.is_out_of_memory(err):
                raise
            # another program (or worker) took the GPU memory: do this body on the CPU instead
        else:
            return _finish_body(sampled, finish)
    # (also in a worker, which has no run's limit: what the PC can still give out, and the engine's share of RAM)
    need, room = grid_memory(grid.size), min(memory_room(), LIMIT_OF_TOTAL * physical_memory())
    if need > room:
        raise GraphError(f"this needs about {need / 2**30:.1f} GB of memory but only {max(room, 0) / 2**30:.1f} GB "
                         f"can be used; use a larger voxel size or close other programs")
    return _finish_body(_mesh_on(body, grid, np, method, **more), finish)


def _mesh_on(body, grid, xp, method="marching cubes", normals=False):
    """The first stage of meshing a body, where the field is computed (the GPU for CuPy): its samples,
    and for marching cubes the surface through them (normals: and the field's normals at its vertices, computed
    beside the samples). Plain host arrays, for meshing.finish_body."""
    try:
        with trace.span("field", xp=xp, voxels=grid.size):
            values = F.Context(grid, ARRAY_CACHE, xp).eval_full(body)
        if method == "marching cubes":
            return {"surface": extract_surface(values, grid, normals)}
        return {"values": backend.to_host(values)}
    finally:
        values = None
        if xp is not np:
            backend.release_gpu_memory()  # hand GPU memory back between bodies


def _finish_body(sampled, finish):
    """The rest of meshing a body (meshing.finish_body): in a CPU worker when the map splits (engine/stages.py)
    and there is real work left (dual contouring, Keep, Tolerance); marching cubes alone only makes the mesh."""
    if finish["method"] == "marching cubes" and finish["keep"] >= 1.0 and finish["tolerance"] <= 0:
        return meshing.finish_body(**sampled, **finish)
    # (a CPU worker has no run to stop: Stop skips the stages still queued)
    return cpu_stage(meshing.finish_body, **sampled, **finish, check=None if splitting() else check_cancel)


def _mesh_estimate(body=None, voxel=0.1, padding=2.0, domain=None, coarser=1.0, **_):
    """Memory one mesh.from_implicit call needs (used to decide how many run in parallel)."""
    body, voxel = _field(body), voxel * coarser
    bounds, padding = (domain.bounds, padding) if domain is not None else F.mesh_box(body, padding, voxel)
    if bounds is None or voxel <= 0:
        return 0
    size = Grid.from_bounds(bounds, voxel, padding).size
    return TILED_ESTIMATE if size > tiles.TILE_MIN_VOXELS else grid_memory(size)


VIEW_MIN_CELLS = 40  # in a preview of many bodies, each still gets this many voxels along its longest side


@node("mesh.from_implicit", "Mesh from Implicit Body", "Mesh",
      [Port("body", "implicit", doc="The implicit body (one or a list) to turn into triangles."),
       Port("voxel", "number", 0.1, unit="mm", label="Voxel size",
            doc="Grid size, mm: smaller = finer detail but slower. Use about a third of the finest feature. Draft and "
                "Medium previews use 4x and 2x this."),
       Port("method", "text", "marching cubes", choices=list(MESHERS),
            doc="How the surface is found between the grid's samples. Marching cubes: fast, but cuts sharp edges "
                "off about a voxel wide. Dual contouring: keeps CAD edges, corners and the rims of holes sharp "
                "(6-20x closer at the same voxel, so 2-3x the voxel will do), takes 2-4x longer to mesh "
                "(10-15x with many holes), and gains little on textures (CAD fillets smaller than a voxel come "
                "out sharp; ridges finer than the voxel leave a few tiny folded-over triangles). Big bodies "
                "are meshed in blocks either way; with dual contouring, "
                "Tolerance and Keep then reduce the whole mesh at the end (slower, more memory)."),
       Port("padding", "number", 2.0, unit="mm", doc="Extra room around the body's box, mm. Offsets by a number or a "
                                          "field of known size (shown on its card) are already included; "
                                          "you are warned when the surface is cut off."),
       Port("domain", "cad", optional=True, doc="Optional: mesh inside this body's box instead of the body's own."),
       Port("tolerance", "number", 0.0, unit="mm", zero="off",
            doc="Smart reduction: how far the surface may move, mm (e.g. 0.01); 0 = off. Triangles are only removed "
                "where the shape does not need them (flat faces become a few big ones, textures keep theirs), done in "
                "the worker processes while meshing, so it is much faster than a Reduce Mesh node afterwards. Replaces "
                "Keep when set. Used at Final quality and for exports only."),
       Port("keep", "number", 1.0, slider=(0, 1), label="Keep triangles",
            doc="Share of triangles kept after simplifying: 1 = all, 0.3 = 30%. Smaller files; used at Final quality "
                "and for exports only."),
       # set by the server for Draft (4) and Medium (2) when Voxel size is linked (a typed one is scaled itself)
       Port("coarser", "number", 1.0, hidden=True)],
      [Port("mesh", "mesh", doc="A watertight triangle mesh (a list if the body was a list).")],
      parallel=True, estimate=_mesh_estimate)
def mesh_from_implicit(body, voxel, method, padding, domain, keep, tolerance, coarser):
    """Turn an implicit body into a watertight triangle mesh (marching cubes, or dual contouring
    for sharp CAD edges)."""
    body, voxel = _field(body), float(voxel) * float(coarser)
    if domain is not None:  # meshed inside the domain's box: cutting the body there is the point
        return _mesh(body, domain.bounds, voxel, padding, keep, tolerance, method=method)
    if body.bounds is None:
        raise GraphError("cannot tell where this body is; connect a Domain")
    bounds, room = F.mesh_box(body, padding, voxel)
    return _mesh(body, bounds, voxel, room, keep, tolerance, f"increase Padding (now {padding:g} mm)", method)


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
                          title="View Mesh", parts="mesh")
    return Many(meshes) if isinstance(body, list) else meshes[0]


def _view_one(body, voxel, padding):
    bounds, room = F.mesh_box(body, padding, voxel)
    return _mesh(body, bounds, voxel, room, 1.0, cut_hint=(
        f"it reaches more than {padding:g} mm beyond its own box (an offset by a field whose size is not known?). "
        "Mesh from Implicit Body needs at least that much Padding."), normals=view_normals())


def view_normals():
    """Whether views and details carry the field's normals (meshing.field_normals; the editor then draws with them
    instead of working out the triangles' own): on unless DESIGN_ENGINE_VIEW_NORMALS=0 (for comparing; the workers
    read it from their environment)."""
    return os.environ.get("DESIGN_ENGINE_VIEW_NORMALS", "1") != "0"


DETAIL_MAX_BLOCKS = 40   # blocks in one detail piece (a few seconds with the workers); the voxel grows to fit
DETAIL_MAX_BOX = 4096    # blocks in the boxes probed for the surface (each probe is 9^3 samples)
DETAIL_MIN_VOXEL = 0.01  # mm: finer shows nothing a print or the fields themselves have


@node("mesh.detail", "Detail Mesh", "Mesh",
      [Port("body", "implicit", many=True), Port("lo", "text"), Port("hi", "text"), Port("voxel", "number", 0.1)],
      [Port("mesh", "mesh")], hidden=True)
def detail_mesh(body, lo, hi, voxel):
    """Used by the editor to show the part of a shown body that is on screen finer than its view: the
    surface inside the box lo..hi (open where it leaves the box), meshed in blocks near the surface
    only. The voxel grows until the surface takes at most DETAIL_MAX_BLOCKS blocks; mesh.metadata
    ["voxel"] is the one used. An empty mesh when no surface passes through the box."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    voxel = max(float(voxel), DETAIL_MIN_VOXEL)
    boxes = []  # (body, the part of the box it can reach)
    for b in (_field(x) for x in _items(body)):
        if b.bounds is None:
            continue
        (blo, bhi), room = F.mesh_box(b, 0.0, voxel)
        box = (np.maximum(lo, np.asarray(blo) - room), np.minimum(hi, np.asarray(bhi) + room))
        if np.all(box[1] > box[0]):
            boxes.append((b, box))
    while True:  # a box too big to even look for the surface in at this voxel: coarser
        count = sum(tiles.Plan(b, box, voxel, 0.0).count for b, box in boxes)
        if count <= DETAIL_MAX_BOX:
            break
        voxel *= max(1.1, (count / DETAIL_MAX_BOX) ** (1 / 3))
    for _ in range(8):
        check_cancel()
        found = [tiles.find_surface(b, tiles.Plan(b, box, voxel, 0.0)) for b, box in boxes]
        total = sum(len(f) for f in found)
        if total <= DETAIL_MAX_BLOCKS:
            break
        voxel *= max(1.2, (total / DETAIL_MAX_BLOCKS) ** 0.5)  # blocks near a surface go with its area
    pieces = []
    for (b, box), todo in zip(boxes, found):
        if not todo:
            continue
        try:
            mesh, _, _ = tiles.mesh_tiled(b, box, voxel, 0.0, 1.0, parallel_map, close=False, surface=todo,
                                          normals=view_normals())
        except ValueError:  # no surface after all
            continue
        pieces.append(mesh)
    # (concatenate keeps the field's normals when every piece has them)
    out = trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0] if pieces else trimesh.Trimesh()
    if len(out.faces):  # whole blocks reach past the box: only what the editor shows (it clips at the box)
        near = np.all((out.vertices >= lo - 2 * voxel) & (out.vertices <= hi + 2 * voxel), axis=1)
        out = trimesh.Trimesh(out.vertices, out.faces[near[out.faces].any(axis=1)], process=False,
                              vertex_attributes=dict(out.vertex_attributes))
        out.remove_unreferenced_vertices()
    out.metadata["voxel"] = voxel
    return out


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
       Port("tolerance", "number", 0.01, unit="mm", doc="How far the surface may move, mm. Triangles are only removed "
                                                        "where that keeps the shape: flat faces become a few big "
                                                        "triangles, textures and curves keep theirs. 0.01 kept 19% of "
                                                        "the key cabinet's triangles (surface within 0.006 mm); larger "
                                                        "values remove more (on curved detail the surface can move up "
                                                        "to ~2x the value)."),
       # set by the server below Final: previews show the mesh unreduced (it looks the same, and saves the wait)
       Port("skip", "int", 0, hidden=True)],
      [Port("mesh", "mesh", doc="The lighter mesh: watertight, every edge shared by exactly two triangles.")],
      parallel=True, estimate=lambda mesh=None, skip=0, **_:
          REDUCE_BYTES * sum(len(m.faces) for m in _items(mesh)) if mesh is not None and not skip else 0)
def reduce_mesh_node(mesh, tolerance, skip):
    """Fewer triangles only where they are not needed (unlike Keep in Mesh from Implicit Body,
    which removes a share everywhere): a smart reduction for smaller files. Previews below Final
    show the mesh as it comes in."""
    if tolerance <= 0:
        raise GraphError("tolerance must be greater than 0")
    if skip:
        return mesh
    before = len(mesh.faces)
    V, F = reduce_mesh(mesh.vertices, mesh.faces, float(tolerance), check=check_cancel)
    if len(F) == before:
        report_message(f"Reduce Mesh: nothing could be removed safely within {tolerance:g} mm", "warning")
    return trimesh.Trimesh(V, F, process=False)


FUSE_BYTES = 650   # per input triangle while fusing: the solids and their union at once (measured 640 B on
                   # three overlapping spheres, 1M triangles; the peak is inside manifold3d's union)
MERGE_BYTES = 100  # per triangle when only put into one mesh (trimesh's concatenate, measured 96 B)


def _merge_estimate(a=None, b=None, fuse="yes", **_):
    triangles = sum(len(m.faces) for m in _items(a) + (_items(b) if b is not None else []) if m is not None)
    return (FUSE_BYTES if fuse == "yes" else MERGE_BYTES) * triangles


@node("mesh.merge", "Merge Meshes", "Mesh",
      [Port("a", "mesh", many=True, doc="Meshes to combine (one or a list)."),
       Port("b", "mesh", many=True, optional=True, doc="Optional: more meshes, e.g. the lamp center."),
       Port("fuse", "text", "yes", choices=["yes", "no"],
            doc="yes = Boolean union into one solid (parts must overlap to join, touching is not enough); "
                "no = only put them in one file. Draft and Medium previews never fuse.")],
      [Port("mesh", "mesh", doc="One mesh with every part (one solid when fused).")],
      estimate=_merge_estimate)
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
