# Audit: inputs-fields-math

Nodes: Number, Import STEP, Pick Body, Scale Bodies, Coordinate, Cylindrical Coordinate, Noise, Face Mask, 3D Wave,
Add/Subtract/Multiply/Divide/Min/Max/Abs/Sqrt/Sin/Cos/Acos/Remap/Clamp/Smooth Step.
Scripts and raw outputs: `SCRATCH/audit/inputs-fields-math/` (all runs through guard.py, CPU, private cache, peak < 1.2 GB).
Test parts: `samples/test_part.step` (60x40x15 plate, fillets r = 3, centre hole r = 8, CAD faces 0-26, top = face 10),
`samples/two_blocks.step`, and STEP files I generated in `step_files/`.

Findings are appended as they are confirmed; bugs first, then UX/improvements (a short "checked and fine" list is at the end).

## BUGS

### [BUG] Smooth Step / Divide / Remap: NaN or inf in a field gives a mesh with NaN vertices and an STL full of NaN triangles, and the run reports success
Severity: high · Status: VERIFIED
Evidence: `nan_e2e.py`, `nan_export2.py`. test_part.step -> Implicit Body -> Offset Body (distance = SmoothStep(X, 30, 30) * 0.5) -> Mesh from Implicit Body, voxel 0.5:
`verts=82242 faces=164484 watertight=True volume=nan bounds_x=(nan,nan)`; 48913 of 82242 vertices are NaN although NaN exists only on
the samples with x == 30 (a single plane). Exported through Export Mesh: the raw STL has 98268 of 164484 triangles with NaN corners
and NaN normals; the node returns the path with no warning. (trimesh hides it when reloading, it silently drops those faces; a slicer will not.)
Other cases: offset = 0/0 everywhere -> "No surface found at the given iso value."; offset = 1/0 -> "Field has no surface inside the grid (no sign change)" (both misleading).
Cause: `fields.smoothstep` divides by `(edge1 - edge0)` (fields.py:238) with no guard; `fields.remap` divides by `(in_hi - in_lo)` (fields.py:279); Divide is a plain np.divide.
`meshing.field_to_mesh` (meshing.py:12-26) never checks `np.isfinite`: `float(values.min()) >= 0` is False for NaN, `max(nan, voxel)` pads the grid with NaN,
and skimage's marching cubes then returns NaN vertices instead of failing.
Why it matters to the user: SmoothStep with Edge0 == Edge1 is an easy state to reach (a Number drives Edge0, "Edge0 + width" drives Edge1, width -> 0 for a hard step; or both edges share one Number).
Only the grid samples exactly on the edge are NaN, so the preview may look almost fine while the exported file is broken, and the only symptom is a strange print.
Suggestion: (1) in `field_to_mesh` (and the paint/slice paths) raise a clear GraphError "the field contains N NaN/inf values (division by zero? Smooth Step with equal edges?)"; cheap: `np.isfinite(values).all()`.
(2) Make the nodes safe: Smooth Step with edge1 == edge0 -> hard step (`t = clip((x-e0)/max(e1-e0, tiny))`, or `where(x >= e0)`); Remap with from_low == from_high -> GraphError with a clear text (numbers) or a safe divide (fields);
Divide: `np.divide(a, b, out=0, where=b != 0)` or document that it yields inf/NaN and make the mesher report it.

### [BUG] Acos: a value or field that lies entirely outside -1..1 crashes with "math domain error", although the port says "values outside are clipped"
Severity: medium · Status: VERIFIED
Evidence: `edge_math.py`. `Acos(Number 2.0)`, `Acos(Number -2.0)`, `Acos(Number 1.0000001)`, `Acos(noise + 5)`, `Acos(noise - 5)` all fail with
`GraphError: node 'f' (Acos) failed: math domain error`. Partly-outside ranges (`Acos(noise*3)`, `Acos(X)`) work and are clipped as documented.
Cause: `_unary_range("acos")` (fields.py:156-158): when the input range lies fully outside [-1,1] then `lo > hi` after clamping and `math.acos(max(lo, -1.0))` is called with lo = 2 (or `acos(-2)`) -> ValueError.
The range code runs when the Field is built, not when it is evaluated, so any constant expression beyond +-1 trips it. Float32 rounding makes it reachable without typing a "wrong" number
(e.g. a constant `0.1 * 10` is lifted as float32(0.1) -> 1.0000000149 -> crash).
Why it matters: the error text on the card means nothing to a designer, and a field such as Cos(x) + 1.2 fed into Acos (a triangle-wave recipe with an offset) dies instead of clipping.
Suggestion: in `_unary_range` clamp both ends into [-1, 1] first (`lo = min(max(r[0],-1),1); hi = min(max(r[1],-1),1)`); the result is then a constant 0 or pi, which is what np.arccos(np.clip(v,-1,1)) really gives.

### [BUG] Noise: octaves <= 0 gives a silent constant 0; negative roughness gives values up to 1e9; negative seed and huge octaves crash with raw Python errors
Severity: medium · Status: VERIFIED
Evidence: `noise_params.py` (through `nodes.noise`, which only validates scale):
- `octaves=0` or `-3` -> field is exactly 0 everywhere (`total / max(norm, 1e-9)` with no layers), range tag still says -1..1.
- `roughness=-1, octaves=2` -> min -1.36e9, max 1.16e9 (norm = 1 + (-1) = 0 -> divide by 1e-9, noise.py:56). `roughness=-0.5, octaves=3` -> up to +-1.4 (outside the documented/tagged -1..1). The port doc says 0..1 but nothing enforces it.
- `seed=-1` -> `ValueError: expected non-negative integer` (numpy `default_rng`, noise.py:23); the port doc says "Any whole number". Same for 3D Wave with warp > 0 (with warp == 0 the seed is unused, so no error: inconsistent).
- `octaves=1100` -> `OverflowError: int too large to convert to float` (frequency doubles every octave). octaves=30 works but is pointless (each octave is a full pass; layers above the grid's Nyquist are only aliasing).
Why it matters: the UI lets you type anything into these boxes (int box, scrub with the mouse past 0); a field of +-1e9 added to a distance field quietly wrecks the mesh.
Suggestion: validate in the `noise` node: octaves 1..~12 (GraphError), roughness clipped to 0..1 or error, seed `abs(int(seed))` (or `& 0xFFFFFFFF`) in `_tables`.

### [BUG] Import STEP: Skip duplicates (on by default) drops bodies that merely share a bounding box
Severity: medium · Status: VERIFIED
Evidence: `import_tests.py`. A STEP with a 20x20x24 box and a cylinder (r 10, h 24) in the same bounding box:
`skip_duplicates=1` -> 1 body (the cylinder, volume 7535, is gone); `skip_duplicates=0` -> 2 bodies. A true duplicate (same box twice) is removed correctly.
Cause: nodes.py:75-78: the duplicate key is only `np.round(mesh.bounds.ravel(), 2)`.
Why it matters: silent loss of a body, with the default setting; downstream index-based picks ("2:15") also shift. Also the rounding to 0.01 mm is boundary-sensitive: two identical bodies whose bounds straddle a x.xx5 rounding boundary are not seen as duplicates.
Suggestion: key on bounds + volume (+ triangle count or centroid) with tolerances, or compare the vertex sets; and say in the log how many bodies were skipped ("Import STEP: skipped 1 duplicate body").

### [BUG] Face Mask: a negative Band makes the texture vanish; Band 0 / a Band smaller than the offset flattens or removes it silently (default band 2.0 vs textures taller than 2 mm)
Severity: medium · Status: VERIFIED
Evidence: `band_e2e.py`, test_part, top face masked, ridges 0..3 mm: no mask -> highest z = 18.00; Face Mask band 2.0 (default) -> 16.94 (flat tops 2 mm above the face);
band 3.5 -> 18.00; band -3.5 -> 15.00 (no texture at all, no error). `facemask_semantics.py`: at band=0 the mask is 1 only exactly on the surface.
Cause: `values_on` keeps points with `sq <= max_dist**2` and the coarse pass uses `d < max_dist + ...` (facemask.py:98-105); a negative max_dist squares to a valid radius in one test and fails in the other. Node `face_mask` validates fade (>= 0) but not band.
Why it matters: "the texture is missing/flat on the masked faces" is exactly the "something isn't working" feeling. The doc says "at least the largest offset" but the default (2.0) is smaller than a typical 3 mm amplitude, and nothing tells you where the cut is.
Suggestion: GraphError for band <= 0; and either default band larger (e.g. 5), or let `band` be taken automatically from the Offset Body that uses the mask (hard), or at least show a node `note`
("mask is 0 farther than 2 mm from the surface") and warn at mesh time when the mesh's surface touches the band limit.

### [BUG] Face Mask: `faces` text like "1-5" is read as faces 1 and 5; a body number that does not exist ("5:10") silently gives an all-zero mask
Severity: low-medium · Status: VERIFIED
Evidence: `facemask_parse.py`: `faces_for_body("1-5")` -> [1, 5]; "1..5" -> [1, 5]; "-3" -> [3]; "1.5" -> [1, 5]. On test_part `5:10` -> `face_mask(0 faces)`, no error, and on two_blocks `2:5` and `5:1` are `0 faces` for both bodies.
Cause: `faces_for_body` just regex-extracts every integer (facemask.py:46).
Why it matters: a designer who types "1-5" (the usual way to write a range) gets two faces and no message; a mistyped body index makes the texture disappear.
Suggestion: support `a-b` ranges (or reject text containing '-' / '.' with "write 1, 2, 3"), and when the selection resolves to no face on ANY body of the list raise/emit "no face of the selection exists on this body" (a per-body empty result inside a list is legitimate, so only warn when all are empty).

### [BUG] Remap: From Low == From High (plain numbers) gives the raw Python message "float division by zero"; with fields it gives NaN everywhere
Severity: low-medium · Status: VERIFIED
Evidence: `edge_math.py`: `Remap(X, 1, 1, 0, 1)` -> `GraphError: ... failed: float division by zero`; `Remap(X, from_low=Y, from_high=Y)` -> 68921 of 68921 samples non-finite.
Cause: `fields.remap` (fields.py:279) computes `(out_hi - out_lo) / (in_hi - in_lo)` in Python.
Suggestion: check in the node (`remap`) for numbers: "From Low and From High are equal"; same helper as the NaN item above.

## UX / IMPROVEMENTS

### [UX] Noise: the real range is about +-0.7 (not -1..1) and changes with Octaves/Roughness, so "Remap -1..1 -> 0..1" and Remap Auto never reach 0 and 1
Severity: medium · Status: VERIFIED
Evidence: `noise_params.py`, 121^3 samples, scale 10: octaves 1: min -0.95 max 0.94 (std 0.28); octaves 2: -0.79..0.74 (std 0.21); octaves 3 (default): -0.71..0.65 (std 0.18, 1st..99th percentile -0.40..0.40);
octaves 6: -0.65..0.59 (std 0.16). roughness 1.0: -0.65..0.69. `ranges.py`: Remap(noise) with From -1..1 -> To 0..1 gives 0.13..0.84; Remap Auto does the same because `noise()` declares `range=(-1, 1)` (fields.py:287) and tags the card "-1 ... 1".
Why it matters: the user's amplitude is never reached (a "0.5 mm texture" is really ~0.2-0.35 mm over most of the surface), and changing Octaves or Roughness changes the strength as well as the roughness, which is surprising when comparing looks.
Suggestion: normalise the output so that about the 99th percentile is +-1 (divide by ~0.45-0.5 for the default, or by a per-(octaves, roughness) constant measured once, then clip/soft-clip to +-1), or declare the honest range and say "typically +-0.5" in the port doc.
Related: `ranges.py` also shows the interval-arithmetic limit "same variable twice": `noise*noise` is tagged -1..1 (real 0..0.55), `noise-noise` -2..2 (real 0), so Remap Auto after such an expression is far off. Worth one sentence in the Remap Auto doc.

### [UX] Cylindrical Coordinate (angle) has no range, so Remap Auto refuses it even though the range is exactly -pi..pi
Severity: low · Status: VERIFIED
Evidence: `edge_math.py`: `Remap(angle, auto=yes)` -> "Auto cannot tell X's range (it depends on the part...)". `fields.angle` (fields.py:260) creates the Field without `range`; the port doc itself says -3.14..3.14. `radial` is also without the (0, inf) lower bound (Remap Auto cannot be used, fine, but `Sqrt`/`Acos` chains lose information).
Suggestion: `range=(-math.pi, math.pi)` for the angle field; `(0, inf)` is not needed.
Also (low): the handedness differs per axis: `angle("y")` is `arctan2(z, x)` (x towards z = clockwise seen from +y), while x -> arctan2(z, y) and z -> arctan2(y, x) are right-handed. A pattern copied from the z-axis recipe to the lamp's y-axis runs the other way round. Suggest `arctan2(x, z)` for y, or document.

### [UX] Math nodes always output a field, so Number -> Add/Multiply -> (any plain "number" port such as Noise scale, 3D Wave period, Face Mask fade, Mesh voxel) is refused as "cannot connect scalar output to number input"
Severity: low-medium · Status: VERIFIED
Evidence: `edge_math.py`: `Number(2) -> Add(+3) -> Noise.scale` -> `invalid graph: z: scale: cannot connect scalar output 'f.out' to number input`. `_COMPATIBLE["number"] = {"number"}` (core.py:118), math nodes return `Field` even for two plain numbers (`Add(2,3)` -> Field range (5,5)).
Why it matters: the Number node's own doc encourages "one Number for a setting, connect to every node that needs it", but you cannot derive a second setting from it (period = 2 x pitch, scale = k x thickness).
Suggestion: allow scalar -> number links, and have the evaluator collapse a constant Field (range lo == hi) to a float at the port; otherwise raise "this input needs a plain number, not a field".

### [UX] Clamp with Low > High silently returns High; int ports truncate a Number link (1.9 -> 1, 2.7 octaves -> 2) while typing rounds
Severity: low · Status: VERIFIED
Evidence: `edge_math.py`: `Clamp(X, 5, -5)` -> constant -5. `Number(2.7) -> Noise.octaves` -> `noise(10.0, 2, ...)`; Pick Body index 1.9 -> body 1. The editor rounds when typing into an int box (EngineNode.tsx:31, `Math.round`), the engine truncates (`int(index)`).
Suggestion: swap or error on Low > High; `round` instead of `int` in the engine for int ports.

### [UX] Face Mask range tag says 0..1 when nothing is selected (constant 0); "Apply to" and surface/volume behave as documented
Severity: low · Status: VERIFIED
Evidence: `facemask_parse.py`: faces='' -> range (0.0, 1.0) but the field is the constant 0 (surface reach) while the volume branch tags (0, 0) (facemask.py:191 vs 231). Remap Auto of an empty mask therefore does not hit the "constant" error but maps 0 -> 0.
Numerics checked against the docstring (`facemask_semantics.py`, test_part top face): inside/center/outside with fade 1 and 3 match the smoothstep formula to <= 0.013; hard edge (fade 0) correct; volume reach matches `1 - smoothstep(depth/fade)` exactly (0.844/0.5/0.156/0 at 0.75/1.5/2.25/3 mm); "all other faces" matches the formula to 0.01.

### [BUG] Scale Bodies with a Support: a body that sits on a bigger support (a stud on a plate) is scaled about one corner of its base, not about the middle of the contact area
Severity: medium · Status: VERIFIED
Evidence: `scale_tests.py`, 100x100x10 plate + 10x10x10 stud on its top centre (stud base x,y 40..50), scale 2:
no support -> bounds x,y 35..55 (grows about its centre, as documented); WITH the plate as Support -> bounds [[40,40,10],[60,60,30]]: the base stays on z = 10 (seated) but the stud has slid so that it grows off one corner (40,40).
With `two_blocks.step` (body 1 against body 0, shared face) the "seat" case works: x stays at 40.
Cause: `transform.contact_points` (transform.py:9-23) only looks at the SUPPORT's vertices that lie on the body's surface. A plate's top face has vertices only at its own corners, so none lies on the stud: `anchor_point` falls to the "closest" branch (transform.py:38-42) and uses one body vertex (`argmin` of |distance|, a tie between the four base corners -> the first one).
The docstring promises "With a single flat seat, the middle of the contact area is used".
Why it matters: textured lamps/parts whose bodies get scaled by a field (each body its own size) end up displaced instead of seated in place whenever the support face is larger than the body's foot (the normal case on a plain base plate); only supports that have vertices exactly on the foot (fins whose edges coincide with the petal foot) work.
Suggestion: do the contact search from the other side as well: take the BODY's vertices (and a few surface samples) whose distance to the support surface is < tol; their normals (the support's, or the body's) and mean give the apex / seat. I.e. use `MeshQuery(support).signed(body.vertices)` instead of querying the support's vertices against the body.

### [BUG] Scale Bodies: a NaN or infinite scale field is accepted (`factor <= 0` is False for NaN) and produces a body with NaN/inf vertices
Severity: low · Status: VERIFIED
Evidence: `scale_tests.py`: scale = Divide(0, 0) -> "no error; bounds nan"; Divide(1, 0) -> bounds -inf..inf. A scale field that is exactly 0 at the body's anchor gives "scale must be greater than 0 (got 0)" (fine, clear). Scale 1e-6 collapses the body to a point silently.
Suggestion: `if not (factor > 0 and np.isfinite(factor))` -> error; optionally warn below ~1e-3.

(Acos addendum, `acos_float.py`: `Acos(Multiply(0.1, 10))` also fails with "math domain error" because the constant is lifted as float32(0.1) x 10 = 1.0000000149; `Acos(Cos(X*0.1) + 2.5)`, range 1.5..3.5, fails too; `Cos(...) + 1.2`, range 0.2..2.2, is fine.)

### [BUG] Subtract (and Offset-style `a - b`) loses the body's bounds when A is a number or a field: "cannot tell where this body is; connect a Domain"
Severity: low · Status: VERIFIED
Evidence: `misc_tests.py`: `Subtract(0, body)` and `Subtract(X, body)` -> `bounds None`; meshing `Subtract(0, body)` -> `GraphError: ... cannot tell where this body is; connect a Domain`. `Multiply(body, -1)`, `Add(body, X)`, `Min`, `Max`, `Abs`, `Clamp`, `Remap`, `SmoothStep` of a body all keep the bounds.
Cause: `Field.__sub__`/`__rsub__` use `bounds="first"` (fields.py:200-201): "body - distance keeps the body's bounds", but `first` is the constant's (None) when the body is B.
Suggestion: for "first" fall back to the other operand's bounds when the first has none. (`Max` also unions the bounds although it is the intersection: only a bigger grid, not wrong.)

### [UX] Sin / Cos report -1..1 even for a narrow input, so Remap Auto after them is wrong
Severity: low · Status: VERIFIED
Evidence: `ranges.py`: `Sin(Number 0.5)` tagged -1..1 (real 0.479 constant); `Cos(0)` tagged -1..1 (real 1). A realistic chain `Cos(noise * 0.3)` (real 0.95..1) -> `Remap(auto=yes)` is stretched by -1..1, not by 0.95..1. The propagated ranges of Abs, Sqrt, Acos, Min, Max, Clamp, Smooth Step, +, -, x, / are correct (never narrower than the sampled values in 30 cases; see `ranges.py` output; the only over-wide cases are the same-variable cases and sin/cos).
Suggestion: `_unary_range` for sin/cos: if `hi - lo >= 2*pi` -> (-1, 1); else evaluate at both ends plus any multiple of pi/2 inside the interval (a dozen lines).

### [UX] Number box (editor) accepts "0,35", but invalid text is ignored without any sign; the engine itself does not accept a comma
Severity: low · Status: SUSPECTED for the editor (code reading), VERIFIED for the engine
Evidence: EngineNode.tsx:29-31: `commit` only calls `onCommit` when `Number(raw.replace(',', '.'))` is finite; "abc", "1.000,5", "5mm" leave the previous value in the graph while the box keeps showing the typed text (no red border, no revert on blur). `Number("0x10")` is 16 (hex is accepted). In an int box "0.5" is rewritten to "1" while typing (Math.round + the effect on line 27).
Engine side (`misc_tests.py`): a recipe/`run_graph --set amplitude.value=0,5` (json.loads fails, falls back to the string) gives `could not convert string to float: '0,5'` on the Number node; a text value in any other numeric port (Noise.scale = "10,5") gives `'<=' not supported between instances of 'str' and 'int'`.
Suggestion: mark the box invalid (class `bad`) when the text does not parse and revert to the last value on blur; in `nodes.number` accept `str.replace(",", ".")`.

### [UX] Noise Scale: what it means in mm
Severity: info · Status: VERIFIED
Evidence: `misc_tests.py` (1 octave, 6 seeds x 3 lines of 2000 mm): scale 10 gives positive/negative lobes about 10 mm wide (mean zero-crossing spacing 10.2 mm) and peak-to-peak about 14.6 mm; with the default 3 octaves the peak-to-peak spacing drops to 5.2 mm because the finest layer dominates the bumps. So Scale is the lattice spacing of the coarsest layer; with default octaves the visible features are about half of Scale. 3D Wave Period is exact (ridges peak every `period` mm along the chosen axis, value -1 at 0; checked for x, y, z); Warp displaces sample points by a maximum of about 0.72 x Warp (std 0.5 x Warp), not by Warp.
Suggestion: in the Noise doc say "Scale = spacing of the biggest layer; with 3 octaves bumps are about Scale/2 apart"; in the Warp doc say "up to about 0.7 x Warp".

### [UX] Import STEP details
Severity: low · Status: VERIFIED
Evidence (`import_tests.py`, `import_extremes.py`):
- Tolerance <= 0 or Angle <= 0: raw OpenCascade text "BRepMesh_IncrementalMesh::initParameters : invalid parameter value". No lower bound: tolerance 0.0001 on the tiny test_part gives 311,548 triangles and 1.15 GB peak (0.01 -> 3,852 triangles, 0.3 GB); the "more than 3 million triangles" warning only comes after tessellation, i.e. after the memory is spent. A user trying "finer = 0.001" on the lamp gets the runaway import the memory notes describe.
- The "tessellation has N open edges; inside/outside may be unreliable" warning of `cad_import.tessellate` is a `print` (cad_import.py:107), so the editor log never shows it (printed for test_part: 8 open edges). Use `report_message`.
- Body numbers: `cad_body` is assigned after duplicates are skipped (nodes.py:79) while the huge-mesh message uses the pre-skip index (`i + 1`), and toggling Skip duplicates renumbers the bodies, so a face selection "2:15" can point at another body. Say so in the doc or number by the position in the file.
- File replaced/edited: the evaluator signature includes the file mtime for literal path inputs (core.py:477-479) and the SSD cache key includes size + mtime_ns: replacing the file re-imports at once (verified, 1 -> new bounds). Only a same-size edit with preserved mtime stays stale (not a realistic case). A path that arrives through a link would not get the mtime (no node produces one today).
- Empty path: "STEP file not found: " (empty after the colon); say "choose a STEP file".
- Wrong extension with STEP content works; a directory/other file -> "Could not read STEP file: <path>" (fine).
Suggestion: GraphError for tolerance < 0.001 or angle <= 0 ("tolerance must be at least 0.001 mm") before calling OpenCascade, and tell the user in the doc what 0.001 costs.

### [UX] Pick Body
Severity: info · Status: VERIFIED
Evidence: two_blocks, index 0, 1, -1, -2 correct; 2, -3, 5 -> "body index N out of range (file has 2 bodies)" on the right node (clear). A single body (not a list) is accepted as a list of one. Index 1.9 from a Number link silently becomes 1 (see Clamp item). Nothing to fix.

## Checked and fine (no finding)
- Noise and 3D Wave are pure functions of position: whole grid vs aligned sub-blocks bit-identical for voxel 0.1/0.25/0.3/0.5, octaves 1-4, scale 6-100, all four wave patterns with and without warp, also with the coarse-interpolation path (`purity.py`). Block starts that are not multiples of 12 also matched for the cases tried. Preview vs final voxel differ by at most 4e-4 (noise 40/1 coarse path), negligible.
- Face Mask is block-exact too (surface fade 3/1.2/0/6 outside, volume, invert; coarse factors 1-4; `facemask_blocks.py`, 3.2 M voxels).
- 3D Wave/Noise parameter checks that fail with clear messages: Noise scale <= 0, Wave period <= 0, warp scale <= 0, bad axis. Wave ranges: ridges/schwarz reach +-1.000, gyroid/diamond +-0.98/0.96 on a coarse grid (true maxima 1, within sampling).
- Face Mask text parsing of "15", "2:15", "1, 2,3", " 4 ,5 ,", "1 2 3", "2 : 15", empty -> OK; face numbers that don't exist -> "face 27 does not exist (body 0 has faces 0-26)"; unknown names -> "no face selection named 'abc'"; a list of bodies: plain number applies to every body, "1:5" only to body 1 (two_blocks) as documented.
- Error text for sqrt of negatives: Sqrt clamps to 0 as documented (a negative constant gives constant 0).
- Evaluator caching for the nodes in this area (Number -> downstream recompute, Import STEP mtime) behaves.

### [UX] Scale Bodies driven by a surface-reach Face Mask always fails with "scale must be greater than 0 (got 0)"; Smooth Step with Edge0 > Edge1 works as a fade-out but is undocumented
Severity: low · Status: VERIFIED
Evidence: `many_tests.py`: Scale Bodies(two_blocks list, scale = Face Mask of face 5) -> error for body 1: the scale field is read at the body's anchor (the bounding-box centre), which lies deeper than Band from the surface, so a surface mask is 0 there. A volume-reach mask or a Remap(mask) with a floor would work, but the message does not say so.
`edge_math.py`: `SmoothStep(X, 5, -5)` gives 1 below Edge1 and 0 above Edge0 with no NaN (a handy "fade out"), not mentioned in the node doc.
Math nodes over lists (`many_tests.py`): Multiply/Remap/Min/Offset over a Many of masks or bodies map item by item and broadcast a single field against a list correctly; mismatching list lengths give "list inputs have different lengths".
Suggestion: in the scale error say "the Scale field is 0 at the body's anchor point (a surface Face Mask is 0 away from the surface: use reach = volume)"; add the fade-out sentence to Smooth Step.
