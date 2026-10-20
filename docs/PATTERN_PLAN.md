# Face patterns and holes plan

Plan of 2026-10-16, with the status of the work below it.

## Status (2026-10-18)

| Milestone | State |
|---|---|
| 1. UVs in the import | Done: `cad_uv` per triangle corner, `cad_surfaces` per face, `STEP_VERSION` 2 |
| 2. Surface Coordinates node | Not started; no longer needed for holes (see below) |
| 3. 2D shape library | Done: `engine/pattern2d.py` |
| 4. Hole Pattern node | Done: `engine/holes.py`, node `pattern.holes`, recipe `recipes/vent_holes.json` on `samples/vent_case.step` |
| 5. Surface Pattern node | Done: `holes.relief`, node `pattern.surface`, recipe `recipes/hex_studs.json` |
| 6. Editor touches | Category colour only; no shape icons or hole markers yet |

**Change from the plan: the cutter does not use Surface Coordinates.** The plan computed U/V and
depth per voxel from the closest point on the chosen faces. Built that way, a hole near the border
between two chosen faces is cut in half (voxels on the other side of the border read the other
face's frame), and every voxel near the faces costs a mesh query. Instead each hole is a straight
prism along the face normal at its centre (like a drilled hole), evaluated only on the grid
samples around it. Holes are whole across tangent borders between chosen faces, the cutter costs
~0.5 s for 211 holes on 46M voxels, and it is still a pure function of position (blocks match the
whole grid bit for bit). On a curved face the hole is straight instead of following the normals
through the wall; for vent-sized holes the difference is far below a voxel. Surface Coordinates
(U/V as fields for math) can still be added for textures; Surface Pattern can use the same per-shape
evaluation as holes.

Other decisions made while building:

- Size is always the shape's inner width (the largest pin that fits): diameter, across flats, a
  slot's or a cross arm's width, a diamond's width (its Length is the other diagonal). For odd
  polygons: the diameter of the inscribed circle. Diamond also takes Length.
- Slots, diamonds and crosses are spaced farther apart along their length (bounding box of the
  turned shape), so the web between their ends equals the web between their sides.
- Rings turn every hole with its angle around the centre (rotation 0 = a slot points outward).
  The centre is each face's area centroid plus Offset.
- Depth: 0 = through (deepest of 9 rays: centre and 8 points around the outline, + 0.5 mm);
  more = a pocket that deep (sunk / debossed shapes). The cutter starts 1 mm above the face, more
  where the face curves up.
- Height (added 2026-10-18 on request): > 0 = raised shapes that tall (embossed: flat top at Height
  above the face at the shape's centre, straight sides), for Boolean Union with the part. They reach
  into the wall as far as the face falls away under their outline + 0.3 mm (at most 90% of the
  wall), so they join curved walls without poking out behind thin ones. Soft-sided bumps and
  dimples that follow the surface are Surface Pattern's job (milestone 5).
- Periodic faces: a whole number of cells around for faces covering at least ~45% of the period
  (a full cylinder, or one half of a split one, so the halves line up); fillets are left alone.
  Only 90° turns there. "Closes around" is reported only for faces that really close.
- Not done yet: Edge radius (smooth subtract), the "under 6 voxels" check (the node does not know
  the mesh voxel; it reports the largest voxel for clean edges instead), report of holes whose
  wall varies > 20% is there (cut to the thickest part).

## Goal and scope

The plan adds three nodes: **Surface Coordinates** (U/V in mm across picked CAD faces), **Hole
Pattern** (a cutter body of whole holes in a chosen shape, cut with Boolean Subtract) and
**Surface Pattern** (the same shapes as a 0..1 texture height). One shared library of 2D shapes
covers both nodes, so every shape works as a hole and as a raised or sunk texture.

- **Hole shapes:** round, square, hexagon, triangle, octagon, any regular polygon (n sides), slot
  (rounded ends), diamond and cross. Every polygon takes a corner radius and a rotation.
- **Layouts:** square grid, staggered (honeycomb spacing), and rings around a point for round grilles.
- **Different hole types on one part:** each Hole Pattern node has one shape. Several nodes on
  different faces (or with different spacing on the same faces) are joined with Boolean Union, or
  subtracted one after another.
- **Holes follow the face:** each hole's axis follows the face normal, on flat, cylindrical,
  conical and freeform faces.
- **Whole holes only:** a hole is kept whole or dropped, never cut by a face border, and keeps a
  set margin to the border.
- **Through or blind:** a through hole is cut to the measured wall thickness under that hole. A
  blind hole (a pocket) is cut to a set depth.
- **Out of scope for now:** text, logos and image heightmaps. They fit the same Surface
  Coordinates later.

SLS needs no supports and the powder empties out of through-holes, so the only print limits are a
minimum hole size and a minimum web between holes. The node checks both and reports them.

## How it fits the pipeline

The cutter is an ordinary implicit body that depends only on position, so block meshing, caching,
previews and the GPU path need no changes. A recipe with vent holes looks like this:

1. Import STEP → Implicit Body from CAD Body (the part).
2. Hole Pattern: `cad` = the same bodies, `faces` = picked in the CAD view, exactly as for Face Mask.
3. Boolean Subtract: A = the part (or the textured part), B = the Hole Pattern.
4. Mesh from Implicit Body as today.

## Surface Coordinates (milestone 2, not started)

Every point near a chosen face gets the U/V (in mm) of its closest point on the chosen faces only,
plus its signed depth below that face. The node is exposed on its own, so U and V can feed Cos,
Remap and the other Math nodes.

- U/V come from the per-corner `cad_uv` of the closest triangle, weighted by barycentric
  coordinates; each face's mm scales and Align/Angle as in `holes._face_holes` (reuse it).
- The closest point is searched among the chosen faces' triangles alone (a sub-mesh AABB, as
  `facemask._subset_distance` builds for the volume mask), so U/V stay constant going down into a
  wall. The depth's sign comes from the closest triangle's normal.
- Cost: one closest-point query per voxel within reach of the chosen faces, the same work as a
  Face Mask with a hard edge.

## Hole Pattern node (milestone 4, done)

Inputs as built: Body, Faces, Apply to, Shape, Sides, Size, Length, Corner radius, Rotation,
Layout, Spacing, Align, Angle, Offset U/V, Edge margin (2 mm), Depth (0 = through), Height (0 =
holes; more = raised shapes for a union), Size scale. Size scale, Depth, Height and Rotation take a
number or a field (e.g. a Face Mask, a Coordinate, Noise), read exactly at each shape's centre;
where a field gives 0 there is no shape. With a field Rotation, slots, diamonds and crosses are
spaced to fit in any direction.

Report lines: holes kept and why others were left out (border, found twice, too close to another
face's holes, no wall behind, Size scale 0), overlapping holes, thinnest web (error under 0.8 mm),
holes under 1 mm, the largest voxel for clean edges (a sixth of the smallest hole), cell size
spread on faces with uneven parameters, walls whose thickness varies more than 20% under a hole.

## Voronoi, Randomness, spiral and presets (added 2026-10-18)

- **Voronoi** is a shape: every centre's own cell of the layout (`pattern2d.voronoi_cells`: half-plane
  clipping of a bounding square by each Qhull neighbour's bisector, moved in by Web/2 + Corner
  radius, then rounded back by Corner radius), so every hole has its own convex outline
  (`Holes.cells`, exact distance `polygon_sdf`). Staggered gives hexagons, grid squares,
  Randomness organic cells. Each cell is centred on its own centroid: with Randomness two seeds can
  land so close that a seed lies outside its shrunk cell (the web read 0.84 instead of 1.0).
  Neighbours beyond the face (pad 2 steps) are generated so edge cells are right; on a cylinder the
  wrapped copies get the same jitter, so cells close across the seam.
- **Randomness** (0..1) + **Seed** on every layout: each centre moves up to that share of half a
  step, hashed per lattice cell (splitmix64 of i, j, seed; i wraps with the cells around), so it is
  repeatable and closes around cylinders. Non-Voronoi shapes moved too close (web < 0.8 mm) are
  dropped ("moved too close to a neighbour by Randomness").
- **spiral**: a sunflower (r = c sqrt(k), angle k x 137.5°, c = spacing / 1.68, seeds 2 and 3
  skipped), shapes turned with their angle like rings. Neighbours are Spacing apart, which makes it
  a little less dense than a honeycomb.
- **Presets** (`presets/*.json`, in the node palette): Honeycomb vents, Voronoi vents, Slot vents,
  Round grille, Raised hexagons, Knurl texture, Hex stud texture, Voronoi web texture, Dimples,
  Scattered dots. Each runs on `samples/vent_case.step` to one watertight body
  (`docs/pattern_presets.png`).
- **Fading with a Face Mask** (`docs/pattern_fade.png`): Face Mask into Size scale shrinks whole
  shapes towards the mask's edge (0 = left out); Surface Pattern x Face Mask keeps the shapes'
  size and fades their height. The voxel hint ignores shapes under 1 mm (reported separately), or
  a fade would ask for a voxel of a few hundredths of a mm.
- **Morph** (`docs/pattern_morph.png`): Morph To (a second shape with the same Size, Length,
  Sides, corner and rotation) and Morph (0..1, a number or a field read at each centre). Each hole's
  outline is (1 - t) d_A + t d_B, a convex blend of two distances (no steeper than either, so the
  checks and the cutter work unchanged; the shape in between is not an exact distance and can look
  lens-like, e.g. dots into slots). Either shape may be Voronoi (cells are then built, and the other
  shape is centred on each cell's centroid). The layout's steps fit both shapes.
- **Randomness from a field**: the field is read where each centre would be without it (off the
  face: its nearest centre on the face), then each centre moves by its own amount (`pattern2d.jitter`,
  same hash and wrap-around as before): regular cells in one place, organic in another.
- **Any field drives them** (`docs/pattern_wave.png`): 3D Wave (-1..1) -> Remap (Auto) -> Size
  scale gives bands of big and small holes, gyroid-varied Voronoi openings, halftone dots.
  Embossed and embossed inverted (Surface Pattern, Invert) versions: `docs/pattern_emboss_invert.png`.
  Height faded from the wall's top and bottom to its middle (`docs/pattern_height_fade.png`): Face
  Masks of the neighbouring faces (position outside, fade 18 mm) -> Max -> x Surface Pattern; 1 - that
  for the other way round.
- **Min Size**: shapes whose inner width after Size scale is under it are left out (no pinholes
  where a fade or FEA shrinks them; checked before the edge margin). Hole Pattern 1 mm by default
  (what hardly prints or clears of powder), Surface Pattern 0 (off).
- **Driven by FEA** (`recipes/fea_holes.json`, `docs/pattern_fea.png`): Static Analysis' stress,
  safety and displacement outputs are fields (trilinear over the analysis grid), so they feed
  Size scale, Depth, Height and Rotation like any field. Stress -> Remap (1 MPa: 1.3, 6 MPa: 0) ->
  Clamp -> Size scale puts holes where the material barely works; a second Static Analysis of the
  vented body checks it (bracket: 6% lighter, max stress 11.6 -> 15.4 MPa from the holes' own stress
  concentrations). Remap's Auto would use the stress range including the artificial peak at the
  fixed faces (13.2 instead of 11.6 MPa): set From Low/High by hand.
- Surface Pattern's relief stops at half the wall under each shape (ray cast): with Band 2 on the
  case's 2.5 mm wall, 0.5 mm dimples reached the far side exactly (zero-volume slivers), a little
  deeper would have dimpled the inside too.

## More standard patterns (added 2026-10-19)

- **Shapes**: rectangle (Size x Length, corner radius), star (Sides points; Size across the valleys,
  Length across the tips) and chevron (Size thick, Length tip to tip, arms 30° below the horizontal,
  square ends; centred in the middle of its apex, since the centre of its box lies in the notch).
  Stars and chevrons are simple polygons (`pattern2d.simple_polygon_sdf`, exact; rounded by growing a
  smaller polygon, which is exact outside and short of the true distance inside). Odd polygons
  (triangles) are now centred half way between their flat side and their tip, so turned 180° they
  fill the same band. Stars keep Spacing as it is; chevrons only space wider sideways (rows nest).
- **Alternate** (`pattern2d.ALTERNATE`): checker 90 (a parquet of rectangles; spacing fits both
  ways), checker 180 (triangles on a grid tile: rows `sx cos 30° + 1.5 r` apart, so the web between
  rows equals the web in a row), rows 180. On faces that close around, an even number of cells.
- **delaunay** cells (`pattern2d.delaunay_cells`): the Delaunay triangles of the layout's points, each
  shrunk about its incentre by Web/2 + corner and centred on its centroid (the holes sit on the
  triangles, not on the points). Staggered gives equilateral triangles; Randomness a low-poly look.
- **Tilt** (Hole Pattern, degrees): the axis leans towards the pattern's x, turning (normal, t1)
  about t2 (louvres). Depths are measured from where each sample's axis line meets the face, and the
  cutter's ends reach past the slanted face at every sample (`_depths`, `face_normal`).
- **Edges** (Boolean Subtract and Union: Edge sharp / round / chamfer, Radius): exact fillet and
  chamfer blends for surfaces meeting at 90° (hg_sdf's fOpIntersectionRound / Chamfer and the union
  versions). A Hole Pattern's cutter is exact only a few voxels around each hole, so the Booleans
  ask it to `regrow(radius)` (exact 1.25 x radius around).
- **Picture and Text** (`engine/picture.py`, nodes `pattern.picture`, `pattern.text`): a grayscale
  image (dark = 1; transparent = white; scaled to at most 1600 px) or text drawn with a Windows font
  (PIL), projected onto the picked face: flat along the face's mean normal (right = up x normal,
  up = world z), or around the axis of a cylinder or cone (axis from the normals' covariance, circle
  fit for the radius; a face all the way round faces -y). Only within Band of the face. For Size
  scale (halftones) or x a height into Offset Body (embossed, engraved).
- **Presets** (25 in all): the second set adds Bricks, Tiles, Parquet, Triangle tiles, Chevron grip,
  Star texture, Low-poly facets, Louvre vents, Rounded Voronoi vents, Labyrinth (noise contours, the
  lamp's look), Halftone picture, Embossed logo, Engraved text (`docs/pattern_presets_2.png`).

## Join Faces (added 2026-10-19)

Asked for on the key box (`nyckellada.STEP`): a floor and twelve B-spline faces rising to the rim,
picked together, got a pattern per face, with a band without studs at every border and uneven cells
on the B-spline faces. Join Faces (Hole and Surface Pattern, `tangent` by default, `no` = as before):

- The chosen triangles are split into sheets joined across every edge except folds between two
  faces (bend > 30°); vertices along a fold get a copy per side, so a box corner is cut open
  (`unfold.sheets`). A sheet of two or more faces is unfolded flat: libigl LSCM as the start, then
  as-rigid-as-possible steps (Liu et al. 2008; cotangent Laplacian factorised once, closest rotation
  per triangle in closed form), `unfold.flatten`, kept per process.
- The sheet is turned (and mirrored if the CAD frame is) and moved onto the (u, v) frame in mm of
  its largest face, so that face's pattern is the same as without joining; Align, Angle, Offset,
  rings and spiral work on the sheet as on one face (centre: the sheet's area centroid).
- Not unfolded (a pattern per face, said in the log): sheets that close around (a border loop that
  is not the outline or a hole of a flat sheet, told apart by its turning on the surface: +-2 pi
  versus ~0 at a tube's end), and sheets whose stretch would exceed 25% on 5% of the area or that
  fold over. The log gives each sheet's stretch (key box: up to 2% on 95% of the area, 8% at most).
- Holes on one sheet count as on one face (overlaps reported, not dropped); the sheet's spacing
  changes with the stretch, the shapes keep their size.

## Surface Pattern (milestone 5, done)

Same shapes, layouts and placement as Hole Pattern (`holes.place(..., surface=True)`), output a
0..1 height: 1 on each shape's top, 0 between; Surface Pattern × amplitude into Offset Body
(positive = raised studs or tiles, negative = dimples). Inputs: as Hole Pattern without Depth and
Height, plus Softness (the sloped side, inside the outline: the base keeps Size, the top is
Softness smaller all round; a smooth step, 0 = straight), Invert and Band (how far from the face
it is computed). Evaluated per shape on the samples around each, like the cutter (`holes.relief`,
the same `_Shapes.around` loop): local coordinates in the tangent plane at each centre, constant
along the axis within Band. Invert = a surface Face Mask of the chosen faces (fading in over
Softness) × (1 - relief), so only the chosen faces get the raised web. Size scale and Rotation take
fields as on Hole Pattern.

## Editor touches (milestone 6)

Shape icons in the Shape dropdown, and hole centres drawn as markers on the CAD view while the
faces are picked (like Fixed Faces / Load markers).

## Risks and open questions

- **Seams between faces.** Solved for faces joined smoothly by Join Faces (below). Faces meeting at
  a fold, and faces that close around (split cylinders, which line up through their shared surface),
  still have a frame each; holes from another face that come too close are dropped.
- **Sharp hole edges.** Square and hex holes mesh with corners rounded to about one voxel. Use a
  Final voxel at most a sixth of the hole size, or the sharp-edges meshing planned in the dual
  contouring plan.
- **Very thin or curved walls.** Depth is the deepest of 9 rays + 0.5 mm; holes whose wall varies
  more than 20% are reported.

Open questions:

- [ ] Mixing shapes in one node (alternating round and hex)? The node has one shape; use several.
- [ ] Is the default edge margin of 2 mm right for your parts?
- [ ] Should text, logos or image heightmaps follow right after, or later?
