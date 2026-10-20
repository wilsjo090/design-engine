"""Face pattern nodes: holes (and later raised or sunk textures) laid out on picked CAD faces
(engine/holes.py, shapes and layouts in engine/pattern2d.py)."""
import re

from .. import facemask
from .. import fields as F
from .. import holes as H
from .. import pattern2d as P2
from .core import GraphError, Port, check_cancel, node, report_message

PATTERNS = "Patterns"
S = "scalar"


def chosen_faces(cad, faces, apply):
    """Face numbers a pattern goes on: the picked ones, or all others (Apply to)."""
    ids = facemask.cad_faces(cad)
    if ids is None:
        raise GraphError("this body has no CAD faces: connect a body from Import STEP")
    unknown = re.findall(r"[A-Za-z_][\w-]*", str(faces or ""))
    if unknown:
        raise GraphError(f"no face selection named '{unknown[0]}' in this recipe")
    if apply not in facemask.APPLY:
        raise GraphError(f"Apply to must be one of: {', '.join(facemask.APPLY)}")
    body = facemask.cad_body(cad)
    picked = facemask.faces_for_body(faces, body)
    count = int(ids.max()) + 1 if len(ids) else 0
    wrong = [f for f in picked if f >= count]
    if wrong:
        raise GraphError(f"face {wrong[0]} does not exist (body {body} has faces 0-{count - 1})")
    if apply == "all other faces":
        return sorted(set(range(count)) - set(picked))
    return picked


def second_shape(morph_to, sides, size, length, corner_radius, rotation, web):
    """The shape to morph into, or None."""
    if morph_to in (None, "", "none"):
        return None
    if morph_to not in P2.SHAPES:
        raise GraphError(f"Morph To must be none or one of: {', '.join(P2.SHAPES)}")
    return shape_from(morph_to, sides, size, length, corner_radius, rotation, web)


def card_note(found, what):
    """The note under the node's card: how many shapes, and that regular cells are all alike."""
    if not len(found):
        return "none placed"
    note = f"{len(found)} {what}"
    if found.regular:
        note += f"\nall regular {found.regular}: Randomness makes organic cells"
    return note


def shape_from(shape, sides, size, length, corner_radius, rotation, web=1.0):
    """The shape; for Voronoi cells its size is the web between them."""
    try:
        if shape in P2.CELLS:
            if web <= 0:
                raise ValueError("Web must be greater than 0")
            return P2.Shape(shape, web, corner=corner_radius)
        return P2.Shape(shape, size, length, sides, corner_radius, rotation)
    except ValueError as err:
        raise GraphError(str(err)) from err


_JOIN_PORT = Port("join", "text", "tangent", choices=list(H.JOIN_FACES), label="Join Faces",
                  doc="tangent = chosen faces that meet smoothly (a floor, the fillets around it and the walls they "
                      "run into) are unfolded together into one flat sheet and get one pattern that runs across "
                      "their borders without a seam, lined up with the largest face. Where the surface curves both "
                      "ways the sheet stretches a little, so the spacing varies a few % there (the log says how "
                      "much). Faces that close around (the two halves of a cylinder) keep a pattern each, which "
                      "closes around them. no = every face gets its own pattern.")

_SHAPE_PORTS = [
    Port("shape", "text", "round", choices=list(P2.SHAPES),
         doc="round, square, hexagon, triangle, octagon, polygon (any number of sides), slot (rounded ends), "
             "diamond, cross, rectangle (Size x Length), star (Sides points; Size across the valleys, Length "
             "across the tips), chevron (Size thick, Length tip to tip), voronoi (every centre's own cell of the "
             "layout, with Web between the cells; with Randomness 0 they are regular: staggered gives hexagons, "
             "grid squares. Randomness 0.6-0.9 gives organic cells), delaunay (the triangles between the "
             "centres, with Web between them; with Randomness a faceted, low-poly look)."),
    Port("sides", "int", 6, doc="Polygon: the number of sides. Star: the number of points."),
    Port("size", "number", 4.0, doc="mm: the diameter of a round hole; across the flats of a polygon (odd sides: "
                                    "the circle that fits inside); the width of a slot, a diamond or a cross's "
                                    "arms. Always the largest round pin that fits through."),
    Port("length", "number", 8.0, doc="Slot, diamond, cross, rectangle and chevron: overall length, mm. Star: "
                                      "the diameter across its tips."),
    Port("web", "number", 1.0, doc="Voronoi and delaunay: the wall between neighbouring cells, mm."),
    Port("morph_to", "text", "none", choices=["none"] + list(P2.SHAPES), label="Morph To",
         doc="A second shape every hole blends into by Morph (same Size, Length, Sides, corner and rotation): "
             "e.g. round holes turning into hexagons, or into Voronoi cells, across the part. none = off."),
    Port("morph", S, 0.5, doc="How far each hole has turned into Morph To: 0 = Shape, 1 = Morph To. Connect a "
                              "field to morph across the part (Coordinate -> Remap -> Clamp 0..1, a Face Mask, "
                              "a stress). Read at each hole's centre."),
    Port("corner_radius", "number", 0.0, doc="Rounds the corners of polygons, diamonds, crosses and Voronoi cells, "
                                             "mm (the size stays). 0.3-0.5 mm gives cleaner printed edges."),
    Port("rotation", S, 0.0, doc="Turns every hole on the face, degrees (counter-clockwise seen from outside). "
                                 "With Layout rings or spiral, holes also turn with their angle around the "
                                 "centre. A field turns each hole by its value at the hole's centre (slots, "
                                 "diamonds and crosses are then spaced so they fit in any direction). Not used "
                                 "for Voronoi cells."),
    Port("layout", "text", "staggered", choices=list(P2.LAYOUTS),
         doc="grid = rows and columns. staggered = every other row moved half a step (round and hexagonal holes "
             "and Voronoi cells pack as a honeycomb: every neighbour Spacing away). rings = a hole at the face's "
             "centre and rings around it, for round grilles. spiral = a sunflower around the face's centre "
             "(evenly spread, no rows), also for round grilles."),
    Port("spacing", "number", 6.0, doc="Centre to centre, mm (for rings: also between rings). Slots, diamonds "
                                       "and crosses are spaced farther apart along their length, so the web "
                                       "between their ends equals the web between their sides (Spacing - Size)."),
    Port("align", "text", "as in CAD", choices=list(H.ALIGN),
         doc="The pattern's direction on each face: as the CAD surface runs, or with its rows along the x, y or "
             "z axis (as seen on the face). On faces that close around (a cylinder) only steps of 90° are "
             "possible, so the pattern closes without a seam."),
    Port("angle", "number", 0.0, doc="Turns the whole pattern on each face, degrees."),
    Port("offset_u", "number", 0.0, doc="Moves the pattern along its rows, mm."),
    Port("offset_v", "number", 0.0, doc="Moves the pattern across its rows, mm."),
    Port("margin", "number", 2.0, label="Edge Margin",
         doc="Material kept between every hole and the edge of the chosen faces, mm. Holes that would come "
             "closer are left out whole: a hole is never cut by an edge."),
    Port("randomness", S, 0.0, doc="0 = a regular pattern. Up to 1 = every centre moved at random by up to "
                                   "that share of half a step: organic Voronoi cells, scattered dots (shapes "
                                   "that come too close to a neighbour are left out). A field (0..1) makes the "
                                   "pattern regular in one place and organic in another."),
    Port("seed", "int", 0, doc="Change it for a different random pattern (with Randomness)."),
    Port("alternate", "text", "none", choices=list(P2.ALTERNATE),
         doc="Every other shape turned: checker 90 = in a checkerboard by 90° (rectangles: a parquet); checker "
             "180 = by 180° (triangles on a grid tile, every other one pointing down); rows 180 = every other "
             "row turned 180° (chevrons pointing both ways)."),
]


@node("pattern.holes", "Hole Pattern", PATTERNS,
      [Port("cad", "cad", label="Body", doc="The CAD bodies to put holes in (the same bodies the part is made "
                                              "from)."),
       Port("faces", "faces", "", doc="The CAD faces the holes go on. Click Pick, then click faces in the 3D "
                                      "view (Ctrl+drag paints over several, Shift+click adds every face joined "
                                      "tangentially). Face numbers or names of saved selections."),
       Port("apply", "text", "chosen faces", choices=list(facemask.APPLY), label="Apply to",
            doc="chosen faces = holes on the faces you pick; all other faces = on every face except those."),
       _JOIN_PORT,
       *_SHAPE_PORTS,
       Port("depth", S, 0.0, doc="0 = through: each hole is cut as deep as the wall behind it (measured under "
                                 "the hole). More than 0 = a pocket this deep, mm: subtracted, the shapes are "
                                 "sunk into the face (debossed); a pocket as deep as the wall goes through. A "
                                 "field gives each pocket its own depth, read at its centre (0 there: no pocket). "
                                 "Not used with Height."),
       Port("height", S, 0.0, doc="0 = holes or pockets (Boolean Subtract the result). More than 0 = raised shapes "
                                  "this tall instead, mm, measured from the face at each shape's centre (embossed, "
                                  "flat-topped, straight sides): Boolean Union the result with the part. They reach "
                                  "just far enough into the wall to join it. A field gives each its own height, "
                                  "read at its centre (0 there: no shape)."),
       Port("size_scale", S, 1.0, doc="Every hole's size times this, read at the hole's centre: a number, or a "
                                      "field for graded vents (e.g. a Remap of Noise, a Coordinate or a Face "
                                      "Mask). Where it is 0 there is no hole."),
       Port("min_size", "number", 1.0, label="Min Size",
            doc="Holes that would come out narrower than this, mm, are left out: no pinholes where Size scale "
                "fades them (1 mm: smaller ones hardly print or clear of powder). 0 = keep them all."),
       Port("tilt", "number", 0.0, doc="Leans every hole's axis away from the face's normal, degrees, towards the "
                                       "pattern's rows (Align sets their direction): louvres that keep rain or "
                                       "eyes out. The holes stay straight and are cut through the slanted wall.")],
      [Port("out", "implicit", doc="The shapes as a body (each a straight plug along the face's normal). Holes "
                                   "and pockets: Boolean Subtract it from the part (A = the part, B = this). "
                                   "Raised shapes (Height): Boolean Union it with the part.")])
def hole_pattern(cad, faces, apply, shape, sides, size, length, corner_radius, rotation, layout, spacing, align,
                 angle, offset_u, offset_v, margin, depth, size_scale, height=0.0, web=1.0, randomness=0.0, seed=0,
                 min_size=1.0, morph_to="none", morph=0.5, alternate="none", tilt=0.0, join="tangent"):
    """Whole holes in a chosen shape, laid out on picked CAD faces: vents and grilles. Each hole's
    axis is the face's normal at its centre; holes that would be cut by the edge of the chosen
    faces are left out. Subtract the result from the part (Boolean Subtract). Depth makes pockets
    instead (sunk shapes), Height raised shapes (unite them with the part). For different hole
    types on one part, use one Hole Pattern each and subtract them one after another."""
    chosen = chosen_faces(cad, faces, apply)
    if margin < 0 or min_size < 0:
        raise GraphError("Edge Margin and Min Size must be 0 or more")
    if any(not isinstance(v, F.Field) and v < 0 for v in (depth, height)):
        raise GraphError("Depth and Height must be 0 or more")
    turned = rotation if isinstance(rotation, F.Field) else None
    outline = shape_from(shape, sides, size, length, corner_radius, 0.0 if turned is not None else rotation, web)
    second = second_shape(morph_to, sides, size, length, corner_radius, 0.0 if turned is not None else rotation, web)
    try:
        found = H.place(cad, chosen, outline, layout, spacing, align, angle, (offset_u, offset_v), margin, depth,
                        size_scale, height, turned, check=check_cancel, randomness=randomness, seed=seed,
                        min_size=min_size, morph_to=second, morph=morph, alternate=alternate, tilt=tilt,
                        join=join)
    except H.HoleError as err:
        raise GraphError(str(err)) from err
    for text, level in found.lines:
        report_message(text, level)
    out = H.cutter(found)
    raised = isinstance(height, F.Field) or height > 0
    pockets = not raised and (isinstance(depth, F.Field) or depth > 0)
    out.note = card_note(found, "raised" if raised else "pockets" if pockets else "holes")
    return out


@node("pattern.surface", "Surface Pattern", PATTERNS,
      [Port("cad", "cad", label="Body", doc="The CAD bodies the texture goes on (the same bodies the part is made "
                                              "from)."),
       Port("faces", "faces", "", doc="The CAD faces the shapes go on. Click Pick, then click faces in the 3D "
                                      "view (Ctrl+drag paints over several, Shift+click adds every face joined "
                                      "tangentially). Face numbers or names of saved selections."),
       Port("apply", "text", "chosen faces", choices=list(facemask.APPLY), label="Apply to",
            doc="chosen faces = shapes on the faces you pick; all other faces = on every face except those."),
       _JOIN_PORT,
       *_SHAPE_PORTS,
       Port("softness", "number", 0.5, doc="Width of each shape's sloped side, mm, inside its outline: the base "
                                           "keeps the shape's Size, the top is this much smaller all round. 0 = "
                                           "straight sides (they come out as steps of one voxel)."),
       Port("invert", "text", "no", choices=["no", "yes"],
            doc="yes = 1 between the shapes and 0 inside them, on the chosen faces only (fading in over "
                "Softness at their edge): a raised web with sunk cells (positive amplitude) or raised "
                "shapes with the face around them lowered (negative)."),
       Port("band", "number", 2.0, doc="How far from the surface the pattern is computed, mm. At least the "
                                       "amplitude you use it with."),
       Port("size_scale", S, 1.0, doc="Every shape's size times this, read at its centre: a number, or a field "
                                      "(e.g. a Face Mask: shapes shrink towards the faces' edges). Where it is "
                                      "0 there is no shape."),
       Port("min_size", "number", 0.0, label="Min Size",
            doc="Shapes that would come out narrower than this, mm, are left out (where Size scale fades them). "
                "0 = keep them all.")],
      [Port("out", S, doc="0..1: 1 on the shapes, 0 between them (Invert: the other way round). Multiply by an "
                          "amplitude in mm and connect to Offset Body's Distance: positive raises the shapes "
                          "(studs, tiles), negative sinks them (dimples).")])
def surface_pattern(cad, faces, apply, shape, sides, size, length, corner_radius, rotation, layout, spacing, align,
                    angle, offset_u, offset_v, margin, softness, invert, band, size_scale, web=1.0, randomness=0.0,
                    seed=0, min_size=0.0, morph_to="none", morph=0.5, alternate="none", join="tangent"):
    """A texture of whole shapes laid out on picked CAD faces: studs, tiles, dimples, a raised
    honeycomb. Gives 0..1; multiply it by an amplitude (mm) and offset the part by that (Offset
    Body). Same shapes and layouts as Hole Pattern, so the two line up; shapes that would be cut
    by the edge of the chosen faces are left out."""
    chosen = chosen_faces(cad, faces, apply)
    if margin < 0 or softness < 0 or min_size < 0:
        raise GraphError("Edge Margin, Softness and Min Size must be 0 or more")
    if band <= 0:
        raise GraphError("Band must be greater than 0")
    if invert not in ("no", "yes"):
        raise GraphError("Invert must be no or yes")
    turned = rotation if isinstance(rotation, F.Field) else None
    outline = shape_from(shape, sides, size, length, corner_radius, 0.0 if turned is not None else rotation, web)
    second = second_shape(morph_to, sides, size, length, corner_radius, 0.0 if turned is not None else rotation, web)
    try:
        found = H.place(cad, chosen, outline, layout, spacing, align, angle, (offset_u, offset_v), margin,
                        size_scale=size_scale, rotation=turned, check=check_cancel, surface=True,
                        randomness=randomness, seed=seed, min_size=min_size, morph_to=second, morph=morph,
                        alternate=alternate, join=join)
    except H.HoleError as err:
        raise GraphError(str(err)) from err
    for text, level in found.lines:
        report_message(text, level)
    out = H.relief(found, softness, band)
    if invert == "yes":  # the web: 1 on the chosen faces (fading in at their edge) except on the shapes
        ids = facemask.cad_faces(cad)
        on_faces = facemask.face_mask(cad, ids, chosen, softness, "inside", band, "surface")
        out = on_faces * (1 - out)
        out.extent = on_faces.extent
    out.note = card_note(found, "shapes")
    return out


# --- pictures and text on faces ------------------------------------------------------------------
def _placement(cad, faces, apply, mapping, angle):
    from .. import picture as PIC
    if mapping not in PIC.MAPPINGS:
        raise GraphError(f"Mapping must be one of: {', '.join(PIC.MAPPINGS)}")
    try:
        return PIC.Placement(cad, chosen_faces(cad, faces, apply), mapping, angle)
    except PIC.PictureError as err:
        raise GraphError(str(err)) from err


_PLACE_PORTS = [
    Port("mapping", "text", "auto", choices=["auto", "flat", "around"],
         doc="flat = projected straight onto the face (along its normal). around = wrapped around the axis of a "
             "round face (a cylinder or cone), keeping its proportions on the surface. auto = around on "
             "cylinders and cones, flat otherwise."),
    Port("offset_u", "number", 0.0, doc="Moves it to the right on the face, mm (around: along the circumference)."),
    Port("offset_v", "number", 0.0, doc="Moves it up on the face, mm (around: along the axis)."),
    Port("angle", "number", 0.0, doc="Turns it on the face, degrees (counter-clockwise seen from outside)."),
    Port("invert", "text", "no", choices=["no", "yes"], doc="no = dark parts are 1, light parts 0. yes = the other "
                                                           "way round."),
    Port("band", "number", 2.0, doc="How far from the face it reaches, mm: at least the height you emboss it by "
                                    "(it is cut off beyond, so it never reaches the far side of the part)."),
]
_PICTURE_OUT = [Port("out", S, doc="0..1 on the face: e.g. into a Hole or Surface Pattern's Size scale (a "
                                   "halftone), or x a height into Offset Body (embossed; negative: engraved).")]


@node("pattern.picture", "Picture", PATTERNS,
      [Port("cad", "cad", label="Body", doc="The CAD bodies the picture goes on."),
       Port("faces", "faces", "", doc="The face (or faces) it goes on: its centre and direction place the picture. "
                                      "Click Pick, then click the face in the 3D view."),
       Port("apply", "text", "chosen faces", choices=list(facemask.APPLY), label="Apply to",
            doc="chosen faces = placed on the faces you pick; all other faces = on all the others."),
       Port("path", "path", "", doc="A picture file: PNG, JPG or BMP. Dark = 1, light = 0 (transparent counts as "
                                    "white), as gray levels."),
       Port("width", "number", 30.0, doc="How wide the picture is on the face, mm; its height follows its "
                                         "proportions."),
       *_PLACE_PORTS],
      _PICTURE_OUT)
def picture(cad, faces, apply, path, width, mapping, offset_u, offset_v, angle, invert, band):
    """A picture (a logo, a photo) laid onto a face as a field: dark = 1. Make a halftone of it with a
    pattern's Size scale, or emboss or engrave it with Offset Body."""
    from pathlib import Path as _Path
    from .. import picture as PIC
    if not path or not _Path(path).is_file():
        raise GraphError(f"picture file not found: {path}")
    if width <= 0 or band <= 0:
        raise GraphError("Width and Band must be greater than 0")
    place = _placement(cad, faces, apply, mapping, angle)
    try:
        darkness = PIC.load_image(path)
    except PIC.PictureError as err:
        raise GraphError(str(err)) from err
    out = PIC.picture_field(place, darkness, width, (offset_u, offset_v), invert == "yes", band, key=str(path))
    out.note = f"{darkness.shape[1]} x {darkness.shape[0]} px, {width:g} x {width * darkness.shape[0] / darkness.shape[1]:.1f} mm, {place.mapping}"
    return out


@node("pattern.text", "Text", PATTERNS,
      [Port("cad", "cad", label="Body", doc="The CAD bodies the text goes on."),
       Port("faces", "faces", "", doc="The face (or faces) it goes on: its centre and direction place the text. "
                                      "Click Pick, then click the face in the 3D view."),
       Port("apply", "text", "chosen faces", choices=list(facemask.APPLY), label="Apply to",
            doc="chosen faces = placed on the faces you pick; all other faces = on all the others."),
       Port("text", "text", "Text", doc="The text (one line)."),
       Port("font", "text", "arialbd", doc="A font in the Windows Fonts folder by its file name (arial, arialbd, times, "
                                           "timesbd, consola, segoeui, segoeuib, ...) or a full path to a .ttf/.otf."),
       Port("height", "number", 8.0, doc="Letter height (the font size), mm. The text is as wide as it comes out."),
       *_PLACE_PORTS],
      _PICTURE_OUT)
def text(cad, faces, apply, text, font, height, mapping, offset_u, offset_v, angle, invert, band):
    """Lettering on a face as a field: 1 on the letters. Multiply by a height for Offset Body to
    emboss it (negative: engraved), or use it as a pattern's Size scale."""
    from .. import picture as PIC
    if height <= 0 or band <= 0:
        raise GraphError("Height and Band must be greater than 0")
    place = _placement(cad, faces, apply, mapping, angle)
    try:
        darkness, pixels = PIC.render_text(text, font)
    except PIC.PictureError as err:
        raise GraphError(str(err)) from err
    width = darkness.shape[1] * height / pixels
    out = PIC.picture_field(place, darkness, width, (offset_u, offset_v), invert == "yes", band,
                            key=f"text:{text}:{font}")
    out.note = f"{width:.1f} x {darkness.shape[0] * height / pixels:.1f} mm, {place.mapping}"
    return out
