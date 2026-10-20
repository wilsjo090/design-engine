# Audit: implicit-lattice (Implicit + Lattice nodes)

Scripts: `SCRATCH/audit/implicit-lattice/s00..s13*.py` (run through guard.py via `runit.sh <script>`; CPU only,
voxels 0.04-0.8 mm, small boxes plus samples/test_part.step and two_blocks.step).
Nodes: Implicit Body from CAD Body, Offset Body, Shell, Boolean Union/Subtract/Intersect, Smooth Union,
Smoothen Body, TPMS / Strut / Voronoi Lattice, Fill Body with Lattice.

What was checked and is fine (so it is not repeated below): Boolean round / chamfer edges are exact (fitted
fillet radius 3.000 for r=3 at a 90 degree corner, union and subtract; chamfer legs exact); bounds of union /
subtract / intersect are right; Many lists map element-wise and a length mismatch gives a clear message;
lattices, Smoothen (with `fields.halo`) are bit-identical in a block of a bigger grid (s10); `units="density"`
hits the stated solid fraction within 1 percentage point for TPMS sheet/solid, all strut patterns and Voronoi
(randomness >= 0.5); the Fill Body skin is accurate (inner surface at the requested depth +-0.01 mm, also
beyond the band); strut diameter is exact.

Ranking (bugs first; entries below are in the order found, severity is on each):
1. BUG high: Offset / Shell bounds not grown -> flat cut in the preview beyond ~2.8 mm, "no sign change" in Mesh
2. BUG med-high: TPMS thickness / offset in mm wrong beyond thin walls (neovius, lidinoid +-50%)
3. BUG med: Strut Lattice blend does nothing on bcc (the default) and fattens other patterns a lot
4. BUG med: Smooth Union radius 0 -> NaN garbage mesh, negative radius -> intersection
5. BUG med: Intersect of non-overlapping bodies crashes ("negative dimensions"); Union with unbounded lattice silently a lattice box
6. BUG med: Voronoi foam build can need ~14 GB / 160 s at its 2M-point cap
7. UX: textured Offset then Shell/skin is thinner by 1/|grad|; Shell centred (part grows t/2); Smoothen erodes thin walls;
   lattice previews under-fill at Draft/Medium; band silent; one generic "no sign change" message; lattice anchoring/origin.

---------------------------------------------------------------------------------------------------

### [BUG] Offset Body / Shell: bounds are not grown, so the editor preview and Mesh from Implicit Body silently cut the result flat (or fail) beyond ~2.8 mm
Severity: high · Status: VERIFIED (s02_padding.py)
Evidence: `fields.offset` is `_lift(body) - distance` and `_binary(..., bounds="first")` keeps the body's box
(fields.py:200, 357-365); `shell` also keeps it. The grid is `bounds + padding + 2*voxel` and the editor's hidden
`mesh.view` node always uses `padding=2.0` (nodes.py:609, server.py:153), which the user cannot change.
two_blocks box 40x20x24, voxel 0.4:
```
offset 5.0: editor preview: bounds [[-3.05,-3.05,-3.05],[43.41,23.24,27.28]]   (expected -5 .. 45)  -> flat cut
offset 5.0: Mesh from Implicit Body (padding 2): ERROR "Field has no surface inside the grid (no sign change)"
offset 3.0: preview bounds [[-2.77..]] / mesh [[-2.84..]]   (expected -3)
offset 1.5: exact.   shell 8.0: outer bounds -2.8 instead of -4 (cut flat); shell 3.0 fine
```
Why it matters to the user: an Offset of 3 mm or more (or a Shell of 6 mm or more) looks wrong in the preview of the
Offset node itself, with no message: the part is sliced flat at the edge of the grid, and in the preview the user
cannot fix it (padding is hidden). The explicit Mesh node fails with a message that does not mention padding.
Suggestion: let `offset` / `shell` / `smooth_union` / round edges return bounds grown by the largest outward
reach (number: max(d, 0); field: `distance.range[1]` when known; Shell: thickness/2), so grids are sized
correctly in views and meshes. For fields whose range is unknown, fall back to `body.bounds` plus a warning
via `report_message`. Independently, `field_to_mesh`'s message should say which case it is (see the "no sign
change" entry).

### [BUG] TPMS Lattice: `thickness` and `offset` in mm are only right for thin walls; neovius and lidinoid are far off
Severity: medium-high (printing: min wall) · Status: VERIFIED (s08_tpms.py)
Evidence: lattice.py:143-151 uses the first-order estimate `f/|grad f|` as a distance. Measured as twice the
distance of the sheet's surface from the f=0 surface, cell 5 mm, voxel 0.1 mm (mean / min / max):
```
requested t      0.3                0.6                1.0                1.5
gyroid     0.295/0.291/0.302  0.569/0.562/0.579  0.877/0.863/0.896  1.167/1.138/1.206
schwarz_p  0.299/0.286/0.309  0.590/0.542/0.629  0.957/0.830/1.051  1.372/1.110/1.533
diamond    0.294/0.284/0.302  0.558/0.540/0.576  0.842/0.801/0.891  1.100/1.026/1.198
neovius    0.300/0.244/0.346  0.614/0.407/0.757  1.104/0.515/1.444  1.575/0.621/2.163
lidinoid   0.290/0.208/0.359  0.529/0.311/0.691  0.708/0.385/0.931  0.838/0.520/1.075
solid mode, requested offset 0.5 / 1.0 / 1.5 -> real shift of the surface (gyroid): 0.438 / 0.687 / 0.817 mm
```
The FINDINGS claim "thickness really is in mm" holds only for walls <= ~10% of the cell. A 1.0 mm gyroid wall is
12% thin, a 1.5 mm diamond wall 27% thin, a 0.6 mm neovius wall varies from 0.41 to 0.76 mm, a 0.6 mm lidinoid
wall from 0.31 to 0.69 mm. Solid-mode offsets saturate (offset 1.5 moves the surface only 0.8 mm).
Why it matters to the user: he sets "0.6 mm" for a printable wall and gets 0.3-0.4 mm in places, or thicker walls
than asked for in others; graded lattices (thickness driven by a field) inherit the same error.
Suggestion: replace the first-order estimate by a better distance: (a) 2-3 Newton projection steps
(x <- x - f grad f / |grad f|^2) and take |x - x*| (3x the trig cost, still a pure function of position), or
(b) a periodic table of the true distance to the f=0 surface sampled on one cell (like `_density_curve` already
samples one cell: scipy EDT on 64^3 samples, sub-voxel refined with f/|grad f| at the surface, trilinear lookup of
the folded position). Until then say in the port doc: "exact for thickness below ~10% of the cell; thicker is
thinner than requested", and keep the density mode (which is correct) as the recommended way.

### [BUG] Strut Lattice: `blend` does nothing for the default pattern (bcc); elsewhere it adds much more material than "rounding"
Severity: medium · Status: VERIFIED (s09_struts.py)
Evidence: lattice.py:263-267 smooth-mins the distances to the struts folded into one octant. bcc has exactly one
folded strut, so there is nothing to blend. Max change of the field when blend goes 0 -> 1.5:
```
bcc 0.0000 mm (folded struts: 1)   cubic 0.586   fcc 0.586   octet 0.901   cubic+bcc 0.725
solid fraction, thickness 0.8, cell 5: octet blend 0 -> 0.26, blend 0.5 -> 0.29, blend 1.5 -> 0.79 ; cubic 0.053 -> 0.107
```
Why it matters: the node's default is bcc, and the blend input silently has no effect there (looks like a broken
control). For the other patterns the blend is `h^2*k/4` of distance removed wherever two struts are within k,
which at 1.5 mm triples the mass of an octet lattice, not a "rounded node".
Suggestion: for bcc (and any pattern with one folded strut) either blend against the mirrored strut (add the
neighbouring strut that shares the node, i.e. use two folded copies) or disable the input / say in its doc
"has no effect on bcc". Document the mass added by blend (about blend^3 per node) or scale it so that blend
is the fillet radius at the joint of two struts.

### [UX] Offset / Shell with a field distance (texture): the result is no longer a distance field, so Shell, Skin and round Booleans are off
Severity: medium · Status: VERIFIED (s03_shell_texture.py)
Evidence: `phi = sdf - g(x)` has |grad phi| = sqrt(1 + |grad g|^2). Shell(1.0) of an offset-textured box, wall
thickness measured along the vertex normals with `checks.wall_thickness` (voxel 0.1):
```
no texture:             0.998
amp 0.4 period 4.0 (flank slope 0.20): min 0.980   (predicted 1/|grad| = 0.981)
amp 0.8 period 3.0 (slope 0.53):       min 0.882   (predicted 0.882)
amp 1.2 period 2.5 (slope 0.96):       min 0.721   (predicted 0.721)  max 2.45 on crests
```
Why it matters: a textured part that is then shelled (or Fill Body with Lattice with a skin) gets thinner walls on
the flanks exactly by the factor 1/sqrt(1+slope^2), up to 28% thin at a slope of 1. The user cannot see this
in the preview.
Suggestion: a "Shell: keep thickness normal to the surface" option that divides by |grad phi| (central
differences on the grid; it needs a halo of 1 voxel declared like `smoothen.blur`), or an Offset Body that
renormalises (`phi/|grad phi|`) when the distance is a field. At minimum note it in the Shell and Offset docs.

### [UX] Shell is centred on the surface: the part grows by t/2 and is a closed, undrainable cavity
Severity: medium · Status: VERIFIED (s02: shell 3.0 of the 40x20x24 box -> outer bounds -1.5..41.5; plain wall 0.998)
Evidence: `shell = abs(body) - t/2` (fields.py:362). The wall thickness itself is exact (0.998 for 1.0, 1.998 for
2.0 at voxel 0.1 on a flat face).
Why it matters: in SolidWorks, Shell removes material inward (the outside dimension stays); other implicit tools' Shell has a
direction choice (inside / outside / both, I believe). With "centred" the part's outer size changes by t/2, which
is a surprise for anyone who dimensions the outside. There is also no way to leave a face open, so the result is
always a sealed void (powder / resin cannot drain).
Suggestion: add `direction`: inside (default for a designer), outside, centred (current). Inside = `max(body, -body - t)`
(exactly what Fill Body's skin builds). A separate "open faces" step can be done with Boolean Subtract today, say so in the doc.

### [BUG] Boolean Intersect of bodies that do not overlap crashes with "negative dimensions are not allowed"
Severity: medium · Status: VERIFIED (s04_booleans.py)
Evidence: `_merge_bounds(mode="intersect")` returns lo > hi (fields.py:35-41): bounds [[100,0,0],[20,20,10]].
`Mesh from Implicit Body` and the editor's mesh.view then raise `ValueError: negative dimensions are not allowed`
from `Grid`. Bodies that only touch (a shared plane) mesh to a zero-volume, "watertight" sheet.
Why it matters: an intersect that comes out empty is a normal modelling mistake; the error names no cause.
Suggestion: in `intersect` (and `lattice.fill`) detect lo > hi and give the field empty-body bounds / raise
GraphError("Boolean Intersect: the two bodies do not overlap").

### [UX] Boolean Union (and Smooth Union) with an unbounded lattice silently returns a lattice-filled box
Severity: medium · Status: VERIFIED (s04: union(A, gyroid) -> bounds -2.84..22.84 around a 20x20x10 part)
Evidence: `_merge_bounds` drops None, so Union(part, lattice-without-domain) takes the part's box plus padding
and meshes the lattice in it (a box full of lattice with the part inside). Subtract and Intersect with a lattice
are legitimately bounded by the part and work.
Why it matters: Union instead of Fill Body with Lattice is a very natural mistake; the mesh looks like a
lattice cube with no explanation, instead of the "no size of its own" message you get for the lattice alone.
Suggestion: when exactly one operand of a Union / Smooth Union has no bounds, return bounds None (so the
existing "connect a Domain / use Fill Body" message appears). Same for Offset of it (already None).

### [UX] Smoothen Body erodes thin walls and shrinks the part a lot; settings do not map to a radius
Severity: medium · Status: VERIFIED (s07_smoothen.py)
Evidence: 10 mm cube, volume 1000, voxel 0.1: (iterations 3, grid 0.1, sigma 0.14 mm) -0.26%; (3, 0.5; 0.71 mm) -5.5%;
(3, 1.0; 1.41 mm) -17.1%; (10, 1.0; 2.58 mm) -44.7%. Corner cut back along the diagonal 0.15 / 0.87 / 1.91 / 3.29 mm,
i.e. an effective corner radius of ~1.8 sigma. A 1.0 mm shell wall smoothened with (3, 0.5): volume 594 -> 73 mm3
(the wall is gone); with (3, 1.0) the whole body disappears and Mesh fails with "no sign change". The default
(3, 0.1) is harmless (wall stays 0.998).
Why it matters: "Round off creases" is expected to leave walls and struts alone; a lattice or shell that goes through
Smoothen with typical numbers loses its thin parts. Convex edges also shrink the whole part, not only the edge.
Suggestion: add a warning when sigma (= grid_size*sqrt(2n/3)) exceeds ~1/3 of the smallest feature (known for
lattices / shells) or simply document "removes features thinner than ~3 sigma". Consider an alternative
"Round Edges (radius)" node using offset-out/offset-in (closing / opening), which rounds concave / convex edges with an
exact radius and does not erode thin walls.

### [UX] Smooth Union and Boolean Union `edge = round` are two fillets with different meanings
Severity: low · Status: VERIFIED (s06_edges.py)
Evidence: 90 degree inner corner, plates of equal height: round union r=3 -> circular fillet of radius 3.000;
smooth union radius 3 -> blend of fitted radius 2.43 (0.81 x radius), blend zone starts 2.5 mm from the corner;
radius 1 -> 0.78. Gap bridging: radius 2 bridges a gap of up to 1.1 mm (round) vs 0.9-1.0 mm (smooth). Smooth union
also rounds the free corners differently (polynomial smooth-min, not an exact distance) and has no `regrow`
treatment for Hole Pattern bodies (nodes.py:354) and no chamfer option.
Why it matters: two nodes for nearly the same job with different radius conventions.
Suggestion: keep Smooth Union (it blends any two bodies) but say "radius = blend width, fillet radius is ~0.8 x
this" in the port doc and point to Boolean Union round for an exact fillet; or fold it into Boolean Union as
`edge = smooth`.

### [UX] Lattice previews under-fill thin struts / walls at Draft and Medium quality
Severity: medium · Status: VERIFIED (s12_resolution.py)
Evidence: share of the true volume in the mesh at a given voxel (cell 5 mm):
```
voxel            0.8   0.5   0.4   0.25  0.125  0.08
gyroid 0.6        61%   97%  100%   100%   100%   100%
gyroid 0.3         8%   22%   56%    99%   100%   100%
bcc 0.8           51%   59%   83%    91%    98%   100%
octet 0.5         19%  102%   69%    77%    97%   100%
```
The editor's view voxel is longest-side/128 (Draft), /256 (Medium), /512 (Final): about 0.5, 0.25, 0.125 mm for a
60 mm part. A bcc lattice therefore looks 40% thinner in Draft than in the export.
Why it matters: previews of lattices look broken or too light; the user will think the thickness setting is wrong.
Suggestion: give lattice fields a `feature` hint (thickness / 2) that `view_mesh` uses to cap the voxel
(within the memory estimate), or show a card note "preview voxel 0.5 mm is coarser than the struts: thin parts
are lost, use Final".

### [UX] Implicit Body from CAD Body: `band` is silent about being too small (errors measured, no warning)
Severity: low · Status: VERIFIED (s01_band.py)
Evidence: voxel 0.4, default band 2 vs band 10, max error of the offset surface (vertices vs true distance to the CAD):
test_part: offset +0.5/+2/+5 -> 0.034/0.010/0.029 mm; -0.5/-2 -> 0.036/0.060; -5 -> 0.21 mm (p99 0.12) with band 2 vs
0.083 with band 10. At voxel 0.8 (coarse pass 4x coarser) offset -5: 0.47 mm (box: 0.60) and +5: 0.11 mm. Error sits at
ridges / medial axes beyond the exact zone (`band + coarse_voxel*sqrt(3)`); outward offsets are fine because linear
interpolation of a distance field is accurate away from kinks.
Why it matters: smaller than feared. Large inward offsets, shells with thickness > 2*band, skins beyond the band
(skin was exact up to 5 mm in s11) and coarse previews are off by up to 0.2-0.6 mm at corners, with no warning.
Suggestion: have `body_from_mesh` record `f.band = band`; Offset / Shell / Fill (skin) / subtract_edged then call
`report_message(...)` when the requested reach exceeds `band` (reach is known for numbers and for fields with a
range). Cheap and removes the need to remember the rule in the port doc.

### [UX] "Field has no surface inside the grid (no sign change)" is the only message for three different causes
Severity: low-medium · Status: VERIFIED (s02, s07)
Evidence: the same text appears when (1) the body is empty (offset -50, Shell thickness <= 0, Subtract removed
everything, Smoothen ate the part), (2) the whole grid is inside (Offset/Shell larger than the padding), (3) an
intersect/fill came out empty. `meshing.field_to_mesh` raises it as a plain ValueError.
Suggestion: test `values.min() >= 0` ("the body is empty: nothing is inside") and `values.max() <= 0` ("the whole
box is solid: increase Padding or shrink the offset") separately.

### [BUG] Smooth Union with radius 0 gives a NaN field and a garbage mesh; a negative radius turns it into an intersection
Severity: medium · Status: VERIFIED (s14_bad_inputs.py, s15_smooth0_voronoi_cost.py)
Evidence: `smooth_union` divides by `radius` with no check (fields.py:423). Radius 0: 701 of 7161 voxels are NaN, Mesh from
Implicit Body returns a 52,284-face mesh with volume nan, not watertight, no error. Radius -1: no error, the result
is the overlap of the two boxes (volume 935 instead of 4959). Radius 1e-6 is fine. `Boolean Union` checks its `edge`
value ("Edge must be one of...") but Smooth Union checks nothing; Boolean round/chamfer with radius <= 0 silently
become sharp (acceptable).
Suggestion: `radius <= 0` -> plain union (or GraphError "radius must be greater than 0"), as the other nodes do for `cell <= 0`.

### [UX] Smoothen Body: zero/negative values are silent or cryptic
Severity: low · Status: VERIFIED (s14)
Evidence: iterations 0 or grid_size <= 0 silently do nothing; iterations -1 -> `ValueError: math domain error`
(node failed: math domain error) from `math.sqrt` (fields.py:434). Offset -50 / Shell thickness <= 0 / TPMS thickness <= 0 /
strut thickness <= 0 all end in the generic "Field has no surface inside the grid (no sign change)" at the mesh node,
not on the node that caused it.
Suggestion: GraphError("Shell: thickness must be greater than 0") etc. on the node itself; iterations < 0 -> clear message.

### [BUG] Voronoi Lattice: building the foam is unbounded in memory and time (cap is 2 million seed points)
Severity: medium (PC freeze risk) · Status: PARTLY MEASURED, rest extrapolated (s13, s15)
Evidence: `voronoi_struts` runs scipy `Voronoi` (Qhull) in the main process over (box + 3 cells) / cell seed points,
capped at 2,000,000 (lattice.py:280). Measured: 12k points 1.6 s, 36k 3.9 s, 80k 6.1 s, 133k 10.6 s with the guard's peak RSS
1.24 GB for the 133k run (about 7 KB per point at peak, Qhull plus ridge sets); the cost is linear, so the cap would
take ~160 s and ~14 GB, which is more than this PC has free (CLAUDE.md: 13 GB free, page file on a hard disk).
The Memory watchdog cannot help while Qhull is in C code. A 100x100x60 mm part at cell 1.5 mm is already 270k
points. It is also recomputed on every change of domain, cell, randomness or seed (not cached on the SSD).
Suggestion: lower the cap to ~300k points (about 2 GB peak), name the estimate in the error ("N seed points ~ X GB: use
a cell size of at least Y"), and build the foam in tiles / cache the strut list on the SSD keyed by (bounds, cell,
randomness, seed).

### [UX] Voronoi foam is anchored to the domain's lower corner; TPMS and Strut lattices to the world origin; no origin / rotation input
Severity: low-medium · Status: VERIFIED (s10, s16)
Evidence: same seed, domain box moved by 0.5 mm -> solid regions overlap only 13% (IoU 0.13; seed 0 vs seed 1: 0.05),
i.e. a completely different foam. TPMS and strut fields are pure functions of the world position (periodic with the
cell, check in s16): the `domain` port only supplies bounds (lattice.py:152, 269), so the cell grid starts at (0,0,0)
whatever the part does. No input to shift (phase) or rotate a lattice; no transform nodes for implicit bodies exist.
Why it matters: moving or resizing the part in SolidWorks (or Scale Bodies) re-rolls the Voronoi foam and shifts where
TPMS / strut cells are cut at the part's edges. Users expect cells to start at the part's corner or at a chosen
origin. A lattice cannot be aligned to a hole or edge, or rotated 45 degrees.
Suggestion: `origin` input (x, y, z, default 0 or "domain corner") added to the coordinates before the lattice; an
`angle` about an axis later. For Voronoi, anchor the jittered seed points to a world lattice (hash of the integer cell
index, like pattern2d.jitter does) so the foam no longer depends on the domain and block meshing sees one infinite foam.

### [UX] Voronoi `units = density` with randomness 0 misses the target; micro-struts from the 0.02 jitter floor
Severity: low · Status: VERIFIED (s10, s10b)
Evidence: density 0.1 / 0.3 / 0.5 -> measured 0.121 / 0.249 / 0.485 at randomness 0 (0.102 / 0.307 / 0.512 at 1.0, 0.102 / 0.303 /
0.503 at 0.5). The calibration samples the foam at cell/8 spacing (`_foam_samples`), which for a regular lattice lands
exactly on the strut axes and over-counts small distances. The regular case (`max(randomness, 0.02)`) also produces
6118 struts shorter than 0.5 mm out of 13,766 (Qhull splits each 8-cell vertex), harmless but slow.
Suggestion: offset the calibration lattice by a non-aligned fraction of the cell (e.g. cell*0.37) or use random sample points.

### [UX] TPMS "solid" at offset 0 is not "about half the volume" for lidinoid
Severity: low · Status: VERIFIED (s16)
Evidence: solid 0.385 / other side 0.615 for lidinoid (0.500 for the other four) because of the +0.15 constant (lattice.py:75).
The `lattice.tpms` docstring says offset 0 = about half. Density mode corrects for it.

### [IMPROVEMENT] No way to union a list of bodies into one body
Severity: low · Status: VERIFIED (s05_lists.py)
Evidence: Boolean Union/Subtract/Intersect/Smooth Union take `a`, `b` and map over lists element by element
(union(list2, single) -> list of 2, each united with the single body; union(list2, list1) -> "list inputs have different lengths
[1, 2]", a clear message). Fill Body with Lattice with a Voronoi built from a list domain gives one foam per domain, and a single
body then yields as many results as foams (the second result had bounds x 40..40, a degenerate sliver).
Why it matters: a Boolean Union elsewhere takes a list; here a multi-body STEP cannot be turned into one implicit body before
Offset / Shell / Smoothen (Merge Meshes fuses meshes only after meshing).
Suggestion: a "Union All" (many=True port reducing with minimum) or `a` as a `many` port with a note.

### [UX] Implicit Body from CAD Body: coarse-pass factor depends on the grid size, so "exact within band" differs between preview and block meshing
Severity: low · Status: SUSPECTED (code: sdf.py:154 factor 2 below 2M voxels else 4; tiles blocks are 97^3 = 912k voxels)
Evidence: the exactly-computed zone is `band + coarse_voxel*sqrt(3)`; a Draft/Medium preview (factor 2 or 4), a mid-size export
(factor 4) and a tiled export (blocks always factor 2) therefore differ outside the band by the interpolation error
measured under the band entry (0.1-0.6 mm at kinks, only for offsets/shells/skins beyond `band`).

