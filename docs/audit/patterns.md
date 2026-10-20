# Audit: patterns (Hole Pattern, Surface Pattern, Picture, Text, presets)

Scripts and outputs: `SCRATCH/audit/patterns/` (common.py = in-process runner; parts/ = STEP test parts I generated with OCP;
out/ = logs and the preset contact sheet `out/sheet_tp_top.png`). All runs through guard.py, CPU, voxel 0.1-0.5 mm.
Nothing in the repo was changed. Findings are appended as I go (ranked at the end, see "Ranking").

---

### [BUG] Picture / Text: the field's key ignores Angle (and the face's axes), so two nodes that differ only by angle collapse into one
Severity: medium · Status: VERIFIED (`t2_keys.py`)
Evidence: `engine/picture.py:174-176` builds the key from `key, image digest, width, ox, oy, invert, band, mapping, centre` only.
`angle` (`Placement.right/up/turn`), the face normal, and for `around` the axis/radius/toward are not in it. `Context._memo` and
`_plan` are keyed by `Field.key` (`engine/fields.py:60-111`), so a second field with the same key reuses the first one's array.
Repro: two Text nodes (text "H", same face) with angle 0 and angle 90, combined with Max:
```
keys equal: True
a (angle 0)  ink box w 5.69 x h 7.12
b (angle 90) ink box w 7.12 x h 5.69
c = max(a,b) ink box w 5.69 x h 7.12   <- identical to a; b has been replaced by a
```
Same with Picture (halftone.png, angle 0 vs 90: keys equal, max(a,b) == a). Different faces give different keys (centre is in the key), so
the collision needs the same face/centre.
Why it matters to the user: e.g. lettering at 0 and 90 degrees on one face, or a logo placed twice with different angles, shows the
first placement twice (or only once). It can also surface later if any cache=True field ever takes a Picture as a child (stale array after
changing Angle); today only the in-run memo is affected because Picture itself is not cache=True and the evaluator's node cache is
signature based (angle is in the node signature, so editing Angle does recompute).
Suggestion: put `angle`, `placement.normal/right/up` (flat) or `axis/origin/radius/toward/turn/height` (around) into the digest (a tuple
of the rounded placement numbers is enough).

### [BUG] Hole Pattern / Surface Pattern: shapes on the opposite face of a thin wall are dropped as "too close to ... on a neighbouring face"
Severity: medium · Status: VERIFIED (`t4_twosided.py`)
Evidence: `holes._crowded` (holes.py:726-756) asks every pair within `2*r.max()+max(spacing, MIN_WEB)` (3D centre distance) and measures the
web with `_web`, which only uses the in-plane (x, y) of the second hole in the first hole's frame and ignores `along` (holes.py:759-764).
On a 6 mm plate, faces "5,4" (top + bottom), round 4 mm, spacing 6:
```
pockets faces=5      -> 120 pockets
pockets faces=4      -> 120 pockets
pockets faces=5,4    -> 120 pockets ... left out: 84 near the border, 120 too close to pockets on a neighbouring face
   volume change: top only -1501.9 mm3, top+bottom -1499.7 mm3  (the bottom pockets simply are not there)
raised shapes 5,4    -> 120 raised ... 120 too close to raised shapes on a neighbouring face
Surface Pattern 5,4  -> 120 shapes ... 120 too close to shapes on a neighbouring face
```
Why it matters: texturing/embossing/pocketing both sides of a panel (or the inside and outside of a tube/lamp wall, wall < about 2r + spacing)
in one node silently gives one side only; the log line says "neighbouring face", which is not what the user did (opposite face). For
through holes the dedupe is right, for everything else it is wrong.
Suggestion: skip the pair test when the two axes point against each other (n_i . n_j < -0.5) or when the separation along the axis exceeds
the web; or only drop for through holes. At least word the message "on the opposite/another face".

### [BUG] Picture / Text: with Band larger than the wall, the effect also lands on the far side of a thin wall
Severity: medium · Status: VERIFIED by volume (`t8_picture_leak.py`; far-side vertex check being redone, see update below)
Evidence: `picture_field` returns the picture for every point with |off| <= band on the plane through the face's centroid (picture.py:163),
independent of which surface is there. Port doc (nodes_pattern.py:282): "it is cut off beyond, so it never reaches the far side of the part".
That only holds when Band < wall thickness (default Band 2 mm). Surface Pattern limits itself to half the wall (`holes.wall`), Picture does not.
Repro: 'H' (height 14) engraved -0.4 mm into a 1.5 mm plate, voxel 0.1:
```
band 2.0 : volume change -38.27 mm3 (letter engraved on both faces)
band 0.5 : volume change -19.14 mm3 (one face)
```
Why it matters: lettering/logo on a thin shell, lid, or lamp wall appears on both sides (or dents the inside), with no warning.
Suggestion: same wall/2 rule as Surface Pattern (one ray per sample is not needed: the plane's distance to the part's other surface can be
read from a few rays), or tell the user in the Band doc/log; default Band could be min(2, wall/2) where the wall is known.

---

Update (`t8b.py`, interior vertices only, 1.5 mm plate, 'H' engraved -0.4): Band 2.0 -> bottom face z 0.00..0.40 (the letter is dented into the
underside too); Band 0.5 -> bottom 0.00..0.00 (untouched). Status now fully VERIFIED.

### [UX] Picture / Text: no feedback when the picture does not land on the chosen face (centre of a face with a hole, L-shaped face, ...)
Severity: medium · Status: VERIFIED (`out/sheet_tp_top.png`, preset "Engraved text" on samples/test_part.step face 10)
Evidence: `Placement.centre` is the area centroid of the chosen triangles (picture.py:83). samples/test_part.step's top face has a big central
hole, so its centroid lies in the hole: the "Engraved text" preset on it produced dvol -2.25 mm3 (vs about -18 expected) and the height map
shows no lettering at all; the card note just says "18.4 x 7.8 mm, flat" and the log has no line. Same for any ring-shaped/annular face.
Why it matters: it looks like "the node does nothing" (exactly the user's complaint), and the user has to guess to use Offset U/V.
Suggestion: after placing, count how much of the picture rectangle (or of the ink) falls on the chosen faces (project the face triangles onto
the placement plane and sample the picture over them) and report "x% of the lettering is on the face" as an error-level line when 0; and/or
pick the centre as the point of the face nearest the centroid (pole of inaccessibility) instead of the raw centroid.

### [UX] Text: card note size includes the bitmap's padding; "Height" is the font em size (capitals are 71% of it)
Severity: low · Status: VERIFIED (`t1_text.py`)
Evidence: render_text pads the bitmap by `pixels//8` on every side (picture.py:61) and `text()` reports `darkness.shape * height / pixels`
(nodes_pattern.py:343-346). Measured ink for Text "H", Height 10, arialbd: 5.7 x 7.1 mm (note says "9.7 x 9.7 mm"); "HELLO": ink 32.7 x 7.4 mm
(note "36.4 x 9.9 mm"); lowercase "x" is 5.1 mm tall. Port doc says "Letter height (the font size)".
Why it matters: someone from SolidWorks typing 10 expects 10 mm capitals and sizes the nameplate from the note.
Suggestion: report the ink box in the note; either rename the port "Font size" or scale so that Height is the cap height (render a reference
"H" to get the cap-height ratio per font). Text is centred on its own ink box, so "H" and "Hg" do not share a baseline (fine, but worth a doc line).

### [IMPROVEMENT] Hole / Surface Pattern: the lattice is anchored at the CAD face's own (u, v) origin, not centred on the face
Severity: low-medium · Status: VERIFIED (`t6_anchor.py`)
Evidence: on a 40 x 30 box face, grid spacing 6, round 3: centres x = 6..36, so margins to the edges are 4.5 mm left and 2.5 mm right
(30 x 30 plate: symmetric only because 30 = 5 x 6). On a box moved to (103.3, -41.7) the pattern moved with it (anchored at the face's plane
origin = the box corner here; in a SolidWorks STEP the plane origin can be anywhere). Rings/spiral are centred on the face centroid, grid and
staggered are not (holes.py:540-547 passes `origin=offset`).
Why it matters: the pattern never looks centred on a panel; the only fix is trial-and-error Offset U/V.
Suggestion: an "Anchor" option (face origin / centre of face / centre of the sheet bounding box) or a "Centre pattern" checkbox that picks
offsets so the margins left/right and top/bottom are equal.

### [UX] Diamond: Size is the diagonal across x, not the largest pin; the port doc and the "under 1 mm" checks assume it is the pin
Severity: low · Status: VERIFIED (numbers from `Shape.sdf`)
Evidence: `-2*sdf(centre)` for diamond Size 4 / Length 8 is 3.58, for the Knurl preset's 1.6 x 1.6 diamonds 1.13; the Size port doc
(nodes_pattern.py:84-86) says "Always the largest round pin that fits through". `Holes.widths()` (holes.py:130-136) returns `scale*size` for
non-cell shapes, so Min Size and the "under 1 mm across" report use 1.6 for a diamond whose real inner width is 1.13.
Other shapes agree with the doc (cross, slot, rectangle, hexagon, octagon, odd polygons, star 'across the valleys' all match).
Suggestion: say "diamond: Size = its width across x" in the doc, or use the apothem form in `widths()`.

### [UX] Face list parser accepts ranges and junk silently ("4-8" is faces 4 and 8; "-1" is 1; "5.5" is 5)
Severity: low · Status: VERIFIED (`t3_params.py`, facemask.py:46)
Evidence: `faces_for_body` uses `re.findall(r"(\d+)\s*:\s*(\d+)|(\d+)")`. "4-5" -> faces 4,5 (coincidentally right), "2-4" would give 2 and 4;
"-1" -> face 1 ("0 holes ... 18 near the border" on the plate's end face instead of an error); "5.5" -> faces 5,5. Names are checked
(`chosen_faces` rejects letters) but other characters are not.
Suggestion: reject unknown characters, or support "a-b" ranges (CAD users type them).

### [UX] Surface Pattern: Band, wall/2 and Softness silently cap the relief; none of it is in the port docs
Severity: low-medium · Status: VERIFIED (`t5_relief.py`, `t5b.py`)
Evidence (plate 6 mm, round 4 mm stud, softness 0.5, voxel 0.1, mesh padding 5):
- amplitude +3, Band 2 -> stud top +2.03 (cap = Band); Band 4 -> +3.00. Implicit Body from CAD's band does not matter here.
- amplitude -3, Band 2 -> dimple -2.03; amplitude -5, Band 8 -> -3.04 (cap = half the 6 mm wall).
- 1.5 mm plate, amplitude -1.0 / -1.4 -> dimple only 0.73-0.75 deep (wall/2), watertight, no message.
- Softness 5 on a 4 mm stud: relief peaks at 0.35 (a low dome), no message. Softness 2: 1.0 at r=0, 0.84 at r=0.5, 0.5 at r=1.0.
- Mesh from Implicit Body with padding 2 (default) clips anything taller than padding + 2 voxels above the part (stud top 8.22 = 6 + 2 + 0.2 with
  amplitude 3, Band 4) without a message. (That belongs to the mesh node, noted here because every texture preset goes through it.)
Why it matters: "I set amplitude 3 and get 2" is exactly the "something isn't working" feeling; the Band doc says only "at least the amplitude".
Suggestion: Surface Pattern's Band doc: "also limited to half the wall and by Mesh padding"; log the number of shapes whose relief is cut by
the wall rule (the data is in `holes.wall`); consider warning when Softness > Size/2. Offset Body can read the distance field's `range`
(0..amp) and warn when it exceeds the pattern's band/mesh padding.

### [UX] Several presets raise red "error" log lines (and one fragments) with their own defaults; "Parquet" is not a parquet
Severity: low-medium · Status: VERIFIED (`test_presets.py` on samples/test_part.step face 10 at 0.3 mm and samples/vent_case.step face 0 at 0.5 mm)
All 23 pattern presets (the 25 minus "Static analysis" and "Topology", which I did not run: FEA is not in my area) ran to one watertight mesh
on both parts. 0.2-3 s each for placement plus meshing at these voxels. Holes/relief did appear everywhere (volume change 0.1-33%),
except where noted. Preset defaults that the node itself flags:
- Halftone picture: "thinnest web between holes 0.30 mm (under 0.8 mm: fragile)" at full darkness (Size 3, Spacing 3.3). On the vent case wall
  at 0.5 mm voxel the result is 7 separate bodies (webs of 0.3 mm cannot exist at that voxel; they are below any printable web too).
- Knurl texture: "narrowest gap 0.10 mm (under 0.8 mm: hard to print)" (diamonds 1.6 at spacing 1.7: this is the design, but it shows as an error).
- Low-poly facets: gap 0.79-0.80 mm error plus "3-15 shapes are under 1 mm across (fragile)" (Delaunay + Randomness 0.7 makes some tiny triangles).
- Voronoi web texture: gap 0.79-0.80 error (web 0.8 on a 0.8 mm limit).
- Round grille (spiral) on a closed cylinder wall (vent_case face 0): "2 pairs of holes overlap", web -0.87 mm: the spiral's centre is the
  unrolled face's centroid and the sunflower is cut at the seam. The preset text says "for round faces", so fine on a flat disc; but there is no
  warning that spiral/rings on a cylinder is unsupported.
- Parquet texture: rectangles 2.5 x 8, Spacing 4, checker 90 gives a 9.5 mm pitch both ways (`steps()` sets sx = sy = max(sx, sy)); on the 60 x 40 face
  only 10 slats (see the contact sheet): isolated slats in a checkerboard, no tiling. A parquet/basket weave needs the cells to pack (pitch about
  Length + web, 2-3 slats per block).
- Embossed logo on test_part gives bodies = 2 (logo.png is a ring; it sits around the through hole, and the picture slab reaches into the hole's wall,
  leaving a separate sliver). Not chased further.
Suggestion: pick defaults that are clean (Halftone Size 2.6/Spacing 3.4, Knurl accepts it but could set the doc "web 0.1 by design"), make the
fragile-web check relative (informational for textures: MIN_WEB is a hole rule), and rebuild Parquet as a real tiling.

### [UX] Silent clamps and interplay: corner radius, length < size, Randomness/Morph out of range, Alternate on shapes it does not suit
Severity: low · Status: VERIFIED (`t3_params.py`)
- hexagon Corner radius 5 (inner 2): clamped to 0.999 x inner, hexagon turns into a circle, no message. Slot Length 2 < Size 4: becomes 4 x 4 slot (the
  summary line does say "slot 4 x 4 mm"). Round/slot ignore Corner radius; Voronoi ignores Size and Rotation (documented).
- Randomness -1 -> 0, 5 -> 1, Morph 3 -> 1 silently (docs say 0..1).
- Alternate checker 90 on staggered round holes: steps become sx = sy so the honeycomb packing is lost (120 -> 108 holes) although turning a round hole
  does nothing. Triangle + staggered + checker 180 overlaps (132 overlapping pairs, web -0.00 mm; works only with grid, which the doc does not say).
- Overlap warnings are good: "1353 pairs of holes overlap: increase Spacing or reduce Size", margin 100 -> "0 holes ... left out: 162 near the border"
  (error level), Depth > wall -> "N pockets are as deep as the wall: cut through", Tilt 85 -> clear error, Size 0 / negative -> clear error.
- faces empty -> Hole/Surface Pattern: info-level log "no faces chosen" and card "none placed" (Picture/Text give a proper error "pick the face to put it on").
  A selection of "1:5" on a one-body import also silently selects nothing.
Suggestion: make "no faces chosen" and body-qualified selections that match no body warnings; list clamped values in the log once.


Additional evidence for the two-sided finding (`t9_faces.py`, tube R20/R17.5, wall 2.5 mm): faces "0,3" (outer + inner cylinder) ->
"126 holes ... 111 too close to holes on a neighbouring face" (the inner cylinder's holes are all but gone). Via a lamp-like double-walled
body the same thing happens to a texture on both sides.

### [IMPROVEMENT] Patterns on a cone: cell size varies 0.71x-1.28x and the web drops to 0.65 mm (requested 2.0); single developable faces are never unfolded
Severity: medium (conical lamp shades are the user's kind of part) · Status: VERIFIED (`t9_faces.py`, generated cone R20 -> R10, height 30)
Evidence: Hole Pattern round 4 mm, spacing 6, staggered on the cone face: log "cell size varies 0.71x to 1.28x across face 0 (uneven CAD parameters)",
"thinnest web between holes 0.65 mm (under 0.8 mm: fragile)" (nominal web 2.0 mm). Rings on the cone: "4 pairs of holes overlap ... web -1.80 mm".
The frame of a face is one mean (u, v) -> mm scale (holes.py:481); the cone's circumference changes with height, so the spacing follows u.
Join Faces only unfolds sheets of two or more faces (holes.py:392 `len(own) > 1`, and `if len(faces) < 2: continue`), so one cone (or a cone plus
its fillet) never gets the exact development a cone allows. Cylinders are fine (closes around, 21 cells at 125.7 mm).
Why it matters: textures on a tapered body look compressed at the narrow end, and a web that is meant to be printable ends up fragile.
Suggestion: unfold single cone faces (surface type 2 in `cad_surfaces`) like a sheet: the development is a circular sector, exact; or at least
use the local (u, v) Jacobian per triangle instead of the per-face mean when the log already knows the spread is > 10%.


### [BUG] Text / Picture "around" mapping is mirrored when seen from the inside of a concave cylinder (a tube bore, a cup's inner wall)
Severity: medium-low · Status: VERIFIED (`t11_orient.py`, `t11b_inner.py`)
Evidence: flat mapping is correct on all six faces of a block (right x up . outward normal = +1 everywhere, Angle turns counter-clockwise seen from outside).
`Placement` for `around` always unrolls with `side = axis x toward` (picture.py:141-143) whatever way the face's normals point. Text "FR" on the tube's
outer face reads correctly from outside; on the inner face (face 3, R17.5) the glyphs seen from inside the tube (screen-right = -x) are mirrored
(F's arms point right, R's leg runs down-left; ASCII dump in `out/` run). Both faces report "mapping around" and the same fitted axis.
Why it matters: lettering or a logo on the inside of a cup, a tube bore or a lamp shade interior reads backwards, and Angle/Offset U go the wrong way.
Suggestion: flip the sign of s (or of `side`) when the mean face normal points toward the axis (concave), and say so in the note.


### [IMPROVEMENT] Hole Pattern: memory grows with overlapping pairs (about 11 KB each) before the overlap warning is even computed
Severity: low-medium (CLAUDE.md: a RAM overrun freezes the PC) · Status: VERIFIED (`t12_scale.py`, peak RSS from guard.py)
Evidence: 80 x 60 plate, round 4 mm, staggered, Spacing s (overlap = Spacing < Size), min_size 0, margin 1:
```
spacing 3.0: 500 holes, 1411 overlapping pairs,  0.43 GB peak, 0.6 s
spacing 2.0: 1162 holes, 6598 pairs,             0.36 GB, 0.8 s
spacing 1.5: 2029 holes, 28595 pairs,            0.75 GB, 1.6 s
spacing 1.2: 3224 holes, 55029 pairs,            0.95 GB, 2.6 s   (base process 0.34 GB -> about 11 KB per pair)
```
`_crowded` builds `_web(holes, i, j)` for ALL pairs at once: `holes.outlines(a)` gives pairs x 64 x 3 float64 plus the local-frame copies
(holes.py:759-764). A face 6x bigger (200 x 150 mm) with Spacing 1.5 and Size 4 would need about 5-6 GB. Non-overlapping dense layouts are fine:
Voronoi web 0.6 at Spacing 1.0 -> 5115 cells, 6.9 s, 0.54 GB (cells are the slow part: about 1.3 ms each, linear).
Why it matters: a typo (Spacing 1.5 for Size 4, or Size scale 2 from a field) on a big face can run the PC out of RAM instead of showing the
"pairs overlap" error.
Suggestion: evaluate `_web` in chunks of about 20k pairs (it only needs the minimum), and/or fail early when the pair count estimate
(n x neighbours) exceeds a bound; or compute overlap from centre distance vs radius first.


### [UX] Picture / Text act on every surface inside the picture's slab, not only on the chosen face
Severity: low · Status: VERIFIED (`t14_logo.py`, `t15_overhang.py`)
Evidence: (1) a solid-black 50 mm picture centred on a 30 x 30 mm top face, +0.5 emboss, Band 2: 40,612 vertices end up outside the part's
footprint (x -0.50..30.50, z 3.92..6.49): the side walls within Band of the top edge are pushed out. With a 20 mm picture: 0. (2) "Embossed logo" on
samples/test_part.step: the logo (a ring) lands around the central through hole; the slab also acts on the hole's upper 2 mm wall and leaves a
second body of 0.005 mm3 (64 faces, bounds x 22.9, z 13.0-15.1) in the mesh (bodies = 2, euler 2): a loose sliver in the STL.
Why it matters: it is documented as "projected onto the face", and the Face selection reads as a restriction. Slivers are what the Surface Pattern
code already guards against (wall/2 rule).
Suggestion: multiply the picture by a surface Face Mask of the chosen faces (like Surface Pattern's Invert does) or at least say in the doc that
the whole slab is affected; consider dropping sub-0.01 mm3 components in the mesh step.


### [UX] Text: the Font input takes Windows file names only ("Arial Bold", "Times New Roman", "Segoe UI" fail); per-user fonts are not searched
Severity: low · Status: VERIFIED (`t17_fonts.py`)
Evidence: `render_text` looks for `C:\Windows\Fonts\<font>.ttf/.otf` (picture.py:54-57). "Arial Bold", "Segoe UI", "Times New Roman" -> "no font ... (try arial,
arialbd, times, consola, segoeui, or a full path)"; "arial", "ARIALBD", "arialbd.ttf", "Calibri", "Impact", "comic" work. Fonts installed "for me only" live in
%LOCALAPPDATA%\Microsoft\Windows\Fonts and are never searched. In the editor the port is a free text box (no font list).
Why it matters: a SolidWorks user types the font name they see in Word. The error does at least list working names.
Suggestion: map display names through the registry (HKLM and HKCU `...\Fonts`: "Arial Bold (TrueType)" -> arialbd.ttf), search the per-user folder too, and offer
the installed fonts as a dropdown/datalist in the editor.

### [OK] Checked and fine (no action): direction conventions, error messages, edges, fields, caching
- Angle turns counter-clockwise and Offset U/V move along "right"/"up" as seen from outside, on top, bottom and side faces (`t16_dirs.py`, `t16b.py`: top and bottom
  faces, slot at Rotation 30 lies at 30 deg CCW on every face). Flat Picture/Text handedness correct on all six block faces (`t11_orient.py`).
- Boolean Subtract/Union Edge round/chamfer through `regrow` (`t7_edges.py`, 6 mm plate, 4 mm hole, voxel 0.1): sharp 4.00 mm all the way; round 0.6 -> 4.31 at
  0.2 mm below the face, 4.07 at 0.4; chamfer 0.6 -> 4.80 at 0.2 mm; same on top and bottom; raised studs with a round union fillet 4.11 at 0.2 above the base;
  pocket rims rounded, pocket floor edge left sharp (concave corner, as the maths of the fillet gives); tilted holes and Voronoi cells with Edge round stay watertight, 1 body.
- Fields (`t13_fields.py`): a Coordinate ramp as Size scale shrinks holes and reports "6 with Size scale 0", "36 under Min size 1 mm", "33 holes are under 1 mm across";
  Morph, Randomness, Depth, Height, Rotation as fields are read per centre and logged ("Depth from the field: 0.38 to 1.00 mm"); a field Depth <= 0 means no pocket.
- Re-running one Evaluator after an edit (`t13_fields.py`): seed/shape change recomputes only the pattern node and gives a new field key (holes(81) -> holes(91) -> holes(86)); an
  unchanged re-run is cached.
- Errors are clear and on the right node: Size 0/negative, Web 0, "Spacing leaves no room for cells with web 7 mm", polygon < 3 sides, Tilt > 80, Edge Margin < 0, Depth/Height < 0,
  face 99 "does not exist (body 0 has faces 0-5)", unknown selection name, bad Apply/Alternate/Morph To choices, missing picture file, unreadable picture, empty text, missing font.
- Join Faces tangent on samples/test_part.step top + fillets: "faces 3, 10, 14, 19, 20 unfolded into one sheet: the spacing varies by less than 1%", 94 holes vs 87 with join "no"
  (19 dropped as too close at the borders): matches the docs.
- Body lists: "0:5" / "1:5" select per body, "1:9" gives "face 9 does not exist (body 1 has faces 0-5)".
- Diamond/cross/slot/rectangle/hexagon/octagon/star Sizes match their docs except the diamond (see above).


### [BUG] Picture: 16-bit grayscale images saturate to white, and EXIF-rotated photos load sideways
Severity: low-medium · Status: VERIFIED (`t18_images.py`)
Evidence: `load_image` does `image.convert("L")` (picture.py:40). For a PIL "I;16" image that clips at 255 instead of scaling: a 16-bit black -> white gradient loads with darkness 1.0 / 0.0 / 0.0 at
left / middle / right columns (the 8-bit control gives 1.0 / 0.498 / 0.0); only the darkest 255/65535 of the range survives. 16-bit grayscale PNG is the usual format of exported
heightmaps and depth maps. An 80 x 40 JPEG with EXIF Orientation = 6 (a phone photo held upright) loads as (40, 80) rows x cols, i.e. unrotated: PIL does not apply the tag, every other viewer does
(`ImageOps.exif_transpose`). The Picture doc advertises "a logo, a photo".
Why it matters: heightmap/photos silently give a thresholded or sideways pattern with no message.
Suggestion: `if image.mode.startswith("I"): image = image.point(lambda v: v / 256)` or load through numpy and divide by the dtype's max; call `ImageOps.exif_transpose(image)` first.


### [IMPROVEMENT] Patterns on a sphere / dome bunch up at the poles (neighbour distance 0.25-2 mm at spacing 6); torus 0.68x-1.24x
Severity: medium for domes (lamp shades) · Status: VERIFIED (`t20_sphere.py`, generated sphere R20 and torus R20/r6)
Evidence: Hole Pattern round 4, staggered, spacing 6 on a full sphere (one face, u = longitude, v = latitude): "cell size varies 0.39x to 1.27x across face 0 (uneven CAD
parameters)", "84 pairs of holes overlap" (grid: 272 pairs, spiral: 117). Nearest-neighbour distance by latitude: -30..+30 deg median 6.30 mm; 30..60 deg 4.76 mm; 60..90 deg 2.10 mm
(minimum 0.25 mm). Torus: cells 0.68x-1.24x, 75 overlapping pairs with hexagons at spacing 5.2. A `RuntimeWarning: invalid value encountered in divide` comes from the degenerate pole
triangles (`holes.py:593`, handled by nan_to_num). The log does say the sizes vary, but it cannot be fixed with any setting; the overlaps are reported as "increase Spacing".
Why it matters: domes, balls, and shades are exactly where perforation patterns go; the result is a dense clump at the top.
Suggestion: for doubly curved faces place centres by a geodesic/Poisson-disc or a sunflower on the surface area (equal-area), or at least warn "doubly curved face: spacing is only kept near the equator" and
suggest rings/spiral (still (u, v) based). Same remedy as the cone item: lay the pattern out from true surface distances, not from the average (u, v) scale.


### [BUG] Picture / Text on a cone (auto = around): the picture is cut off where the cone leaves the fitted cylinder's Band
Severity: medium (cones are advertised: "auto = around on cylinders and cones") · Status: VERIFIED (`t21_cone_pic.py`, generated cone R20 -> R10, height 30)
Evidence: `Placement` fits ONE circle (mean radius 15 here) and `coordinates()` measures `off = sqrt(a^2 + b^2) - radius` from that cylinder (picture.py:145). A solid-black 40 x 40 mm
picture on the cone face, field sampled on the cone's surface along its front line: Band 2 (default) -> 1.0 only for z = 8..18, 0.0 for z 0..6 and 20..30 (the picture silently ends
where the cone is more than 2 mm away from the fitted cylinder); Band 6 -> 1.0 along the whole height. The note says "40 x 40.0 mm, around" with no hint.
Why it matters: lettering or a logo on a tapered shade or a cone is truncated at the top and bottom unless Band is raised; and the way to know that is a guess. The picture's proportions also
only hold at the mean radius (a cone needs the development, a circular sector).
Suggestion: for cones (surface type 2) measure `off` from the cone (use the half angle: off = (rho - r(h)) cos(alpha)) and unroll with the cone's true development; at least widen Band to the radius
difference over the picture's height.


### [UX] Status vs docs/PATTERN_PLAN.md and CLAUDE.md: small mismatches that explain odd expectations
Severity: low · Status: VERIFIED by reading
- PATTERN_PLAN "Status (2026-10-18)" still lists "Not done yet: Edge radius (smooth subtract)"; it was done on 2026-10-19 (Boolean Subtract/Union Edge round/chamfer; verified above).
- "Presets (25 in all)" (PATTERN_PLAN, CLAUDE.md): 23 are pattern presets; "Static analysis" and "Topology" are FEA clips in the same folder (not run in this audit).
- Milestone 6 (shape icons, hole markers on the CAD view) is still open, as the plan says: the editor (frontend/src) has no pattern-specific UI beyond a file filter for Picture
  (`EngineNode.tsx:164`: .png/.jpg/.jpeg/.bmp); Hole Pattern shows all about 28 inputs for every shape (Web, Sides, Length, Size... whether they apply or not). Combined with the silent
  clamps above, "inputs that do nothing" is a real source of the "something isn't working" feeling. Suggestion: hide or grey Web/Sides/Length/Corner/Rotation by shape (the shape table is
  already in code: `LONG`, `CELLS`, `SIDES`) and put the recommended voxel ("for clean edges, mesh with a voxel of at most 0.67 mm", currently only in the log) on the card.
- The voxel hint is informational only; nothing connects it to Mesh from Implicit Body's voxel or to the preview quality (Draft = voxel x 4 makes small studs one blob, documented in FINDINGS).

---

## Ranking (bugs first)

1. [BUG, medium] Shapes on the opposite face of a thin wall are silently dropped (pockets, raised shapes, textures; tube inner + outer): VERIFIED.
2. [BUG, medium] Picture/Text with Band >= wall also engrave/emboss the far side of thin walls: VERIFIED.
3. [BUG, medium] Picture/Text field key ignores Angle (and the placement frame): two nodes differing only by angle collapse into one in a tree: VERIFIED.
4. [BUG, medium] Picture/Text on a cone: cut off where the cone leaves Band of the fitted cylinder: VERIFIED.
5. [BUG, medium-low] Text/Picture "around" is mirrored on concave (inner) cylinder faces: VERIFIED.
6. [BUG, low-medium] Picture: 16-bit grayscale saturates to white; EXIF rotation ignored: VERIFIED.
7. [UX/IMPROVEMENT, medium] Picture/Text give no warning when the picture misses the chosen face (centroid in a hole): VERIFIED.
8. [IMPROVEMENT, medium] Cone/sphere/torus faces: cell sizes 0.39x-1.27x, web 0.65 mm instead of 2.0, bunching at poles; single developable faces are never unfolded: VERIFIED.
9. [IMPROVEMENT, low-medium] Memory per overlapping pair about 11 KB in `_crowded` (spacing < size on a big face can run the PC out of RAM before the overlap warning): VERIFIED to 0.95 GB at 55k pairs.
10. [IMPROVEMENT, low-medium] Lattice anchored at the CAD face origin: uneven margins (4.5 mm vs 2.5 mm): VERIFIED; add a "centre on face" option.
11. [UX] Surface Pattern silent caps (Band, wall/2, Softness > Size/2; Mesh padding clips tall studs); Text note includes padding; Height is em size; diamond Size; face list parser; font names; preset defaults that
    log errors (Halftone 0.3 mm web, Knurl 0.1 mm gap); Parquet preset is not a tiling; silent clamps; no-face selections only info-level.

## What I could not or did not test
- Presets "Static analysis" and "Topology" (FEA, other agent's area). GPU path (CPU only by rule). The key box / lamp files on OneDrive (rule: never). Block meshing (grids > 32M voxels) of patterns: runs stayed
  under 12M voxels; `tests/test_tiles.py` and `tests/test_holes.py::test_hole_pattern_node` cover bit-exactness (the targeted tests I ran, test_text_and_picture_nodes,
  test_a_picture_lies_on_the_face_and_wraps_round_ones, test_tangent_faces_share_one_pattern, test_hole_pattern_node, pass).
- The editor UI itself (only code-read: EngineNode.tsx, types.ts); the UI agent owns that. Face picking in the 3D view not exercised.
- Join Faces on large B-spline sheets beyond the test_part fillets (the key box is on OneDrive); performance of the as-rigid-as-possible loop at 100k+ triangles is untested (300 iterations x 2 sparse solves, factorised once).
