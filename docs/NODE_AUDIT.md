# Node audit (2026-10-20)

All 54 node types, the evaluator/server and the editor were audited by eight agents in parallel:
inputs/fields/math, implicit/lattice, mesh/output, face patterns, simulation, graph core, a static
review of the editor code, and a live test of the editor (test instance on port 8766, small sample
parts, Draft). Every run was in-process on the CPU under a memory kill switch (peak 2.2 GB).
173 findings; the detailed reports with repro scripts and numbers are in [docs/audit/](audit/).
**Checked again by the coordinator** = read in the code and/or re-run independently after the
agent reported it.

Not covered: GPU paths, worker pools with several workers (killed by the memory guard), Final
exports of big parts, the lamp files, real Topology/Frame optimizations beyond one small case,
native dialogs and Shift/Ctrl clicks in the live test.

## Why it "feels like something isn't working"

Most of the odd behaviour comes from a few patterns, not from one broken node:

1. **Results are cut or emptied silently.** Offset Body and Shell never grow the body's box, so
   anything that reaches past the 2 mm preview padding is cut flat, at a different depth in
   Draft (3.1 mm) and Final (2.3 mm). Face Mask's default Band of 2 mm flattens textures taller
   than 2 mm. A NaN in a field (Smooth Step with equal edges, Divide by 0, Remap with From Low =
   From High, Smooth Union radius 0) makes a mesh and an STL full of NaN triangles, and the run
   says "done".
2. **The editor stops telling you things.** Node warnings appear only on the first run (a cache
   hit replays nothing) and never from worker processes (lists of bodies). After a failed run the
   old mesh stays with no stale marker, and one failing visible node stops every other view
   from refreshing. After an engine restart the open tab ignores every event until it is reloaded.
3. **Previews look different from the export.** The preview voxel is never shown: a 0.35 mm
   texture or a 0.6 mm wall shrinks or vanishes in Draft (a 0.6 mm shell: Draft 1146 mm³, Final
   4525 mm³). A voxel wired from a Number node is never coarsened, so that Draft runs at full
   resolution.
4. **Inputs mean something other than what they say.** TPMS thickness in mm is only right for
   thin walls, Strut blend does nothing on bcc, Noise really spans about ±0.7 rather than ±1, and
   "10-15" in a face list means faces 10 and 15. A wire dropped on a new node connects to the
   first compatible input: a Number dropped on Offset Body becomes its *Body*.

## P1: wrong results, lost work, frozen UI

✅ = fixed. 7, 16-20 on 2026-10-21 (safety factor on a fixed colour scale from 1 to 10; Topology and Frame Optimization analyse their own result: safety factor and deflection in the note, warnings below 1 and for large deflections; Frame thickens the bars at the body's most stressed spot until it reaches its Safety factor at medium; Min thickness holds after Smoothing). 1-3 on 2026-10-20:
- **1:** Offsets by a known amount grow the field's bounds (`Field.grown`, `fields.mesh_box`). The grid stays as it was while Padding covers the offset, so the cached distance fields are reused. A body cut by its box gets a warning, except inside a Domain or a lattice's own box.
- **2:** Smooth Step with equal edges is a step. Remap with equal From values is an error. Smooth Union with radius 0 is a plain union. Acos/Asin ranges are clipped. NaN and inf samples are meshed as outside/inside, with a warning (`meshing.finite_values`, also in blocks).
- **3:** Warnings and errors are kept with a node's cached result and re-sent on every run (`node_done.messages`). They are shown on the card, including the mesher's warnings for the 3D view and the warnings of a group's inner nodes. Lines from worker processes come back with the item's result, and the same line from many list items is reported once with "(xN)".

| # | Where | Finding | Status | Fix |
|---|---|---|---|---|
| 1 ✅ | Offset Body, Shell, Mesh from Implicit, View Mesh | Bounds are not grown by the distance, and preview padding is fixed at 2 mm: offsets above ~2.3–3.1 mm are cut flat (depth depends on quality); Mesh from Implicit fails with "no sign change". Textured offsets are cut the same way. `fields.py:200`, `nodes.py:611` | Checked again (offset 4 mm: mesh ends at ±3.09 Draft, ±2.29 Final) | Grow bounds by the distance's upper range (`Field.range`), and warn when the field is negative on the grid's outer shell |
| 2 ✅ | Smooth Step, Divide, Remap, Smooth Union, `field_to_mesh` | NaN/inf reach marching cubes: 48,913 of 82,242 vertices NaN, exported STL with 98k NaN triangles, run reports success. `fields.py:238,279,423` | Checked again (code) | Guard the divisions (edge0 == edge1 becomes a step), `isfinite` check before meshing with a clear error naming the node |
| 3 ✅ | `report_message`, store | Warnings ("9024 pairs of holes overlap", "N separate solids", "nothing could be removed") show once; cached runs replay nothing; messages from worker processes are dropped (`_current.ctx` is unset there). `core.py:104` | Checked again (code) | Store messages with the node's cached result and re-emit them; forward worker messages through the result queue; show the latest warning on the card |
| 4 | Store, server | After an engine restart run ids and view versions start at 1 again; `store.ts:392,475` drop everything lower than the old ones; the catalog is also never refetched | Checked again (code) | Server sends a session id/`hello` on connect; the store resets `run` and view versions (and refetches the catalog) when it changes |
| 5 | Autosave, recovery banner | The first edit after opening a recipe overwrites its crash-recovery file 2 s later while the banner still offers Restore (from page memory only). Discard deletes without confirmation. The banner survives New. The "untitled" slot is shared by all new graphs, written but never offered back. `Toolbar.tsx:77-89` | Checked again (code) | Don't autosave a recipe while a recovery offer for it is pending (or write to a second slot); confirm Discard; offer the untitled slot on New |
| 6 | Join Frames, Connect to | The load one frame puts on the other (120 N in frame_join) is not added to it; the joined body has no check or note: both frames "pass" while A carries 240 N | Agent-verified | Beam-check the combined frame, or at least report the extra load and utilisation |
| 7 ✅ | Static Analysis colouring | Safety ≥ 10 everywhere gives the range 10..10, and `normalise` + reverse paints the whole part the "bad" red; the display range can be inverted (17.3..10) | Checked again (code) | Clamp the display range to `lo < hi`, with a degenerate range drawn as the good end |
| 8 | Hole / Surface Pattern | Shapes on the opposite face of a thin wall are dropped as "too close to a neighbouring face": `_web` measures in the 2D plane and ignores axial distance and depth. 120 pockets under a 6 mm plate vanish. `holes.py:759` | Checked again (code) | Count two holes as clashing only if their prisms (depth/height along the axis) actually meet |
| 9 | Import STEP | Skip duplicates (on by default) compares only the rounded bounding box: a cylinder sharing a box's bbox is dropped and body numbers shift | Agent-verified | Compare volume + area + vertex hash (or triangle count) as well |
| 10 | Face Mask | Default Band 2 mm silently flattens a 3 mm texture (top at 16.94 instead of 18 mm); a negative Band removes it | Agent-verified | Validate Band > 0; warn when a downstream offset's range exceeds Band (or derive Band from it) |
| 11 | Picture / Text | `Field.key` ignores Angle (two angles collapse to one); Band ≥ wall leaks onto the far side; on a cone the picture is cut where it leaves the fitted cylinder's Band; "around" is mirrored inside a bore; 16-bit PNGs saturate, EXIF rotation ignored. `picture.py:174` | Key checked again (code) | Put angle/axes in the key; limit Band to half the wall like Surface Pattern; cone mapping; flip for concave faces; `ImageOps.exif_transpose` + 16-bit scaling |
| 12 | Views, run thread | One failing or half-wired visible node stops all other views refreshing; an exception after evaluation kills the run thread without `run_done` ("Updating…" forever); an unwired Export blocks the whole export | Agent-verified | Evaluate visible targets independently; `try/finally` that always sends `run_done`; skip unwired side-effect nodes with a warning |
| 13 | Draft/Medium quality | A Voxel wired from a Number or group setting is not scaled (`server.py:111`); Reduce Mesh still runs in Draft; exports also build 512-cell views of every visible body | Checked again (code) | Insert a multiply on linked voxels; skip Reduce below Final; don't build views in export runs |
| 14 | Memory | Voronoi foam is built in the main process with a 2 M seed cap (extrapolated ~14 GB, ~160 s); Hole Pattern `_crowded` uses ~11 KB per overlapping pair (0.95 GB at 55k pairs); the watchdog halves the array cache every 0.5 s and never restores it; the evaluator keeps results of deleted nodes; the server keeps packed views of every node ever shown | Agent-verified (Voronoi partly extrapolated) | Estimate and refuse big foams; cap pairs / warn before; restore the cache limit when memory recovers; prune `_cache` and `Session.views` to the current graph |
| 15 | Node editor | Dropping a wire on the canvas and picking a node connects to the *first* compatible input (`NodePalette.tsx:14`): a Number goes to Offset Body's Body, Hole Pattern's Sides, Mesh from Implicit's Body. A Number can be wired into a body input; refused wires snap back silently | Checked again (code) | Prefer an exact type match, then inputs with a matching default; give number→implicit links a lower priority; show why a wire is refused |
| 16 ✅ | Topology Optimization | Changing only the material's strength or density throws away a finished optimization (they are in the key, but only E matters) | Agent-verified | Key on E and Poisson only |
| 17 ✅ | Topology Optimization | Min thickness not honoured: members of 1-2 mm at Min 3, and a 2 mm skin; Max thickness holed every thin wall | Measured on the shown body | Eroded-design stiffness, skin ≥ Min thickness, Max thickness counts outside as empty (FINDINGS, "Min thickness, skin and load arrows") |
| 18 ✅ | Static Analysis, Load | A load partly off the part was put back by area: a topology result without the unloaded side of a pin hole got 276 N of a 200 N bearing load (its stiffness looked 44% lower than the optimizer's) | Measured | Put back by what each point carries |
| 19 ✅ | Frame Optimization | Safety factor 2 is not reached by the body: Static Analysis gives 1.57 (medium) / 1.42 (fine) at the junctions of bars with the bosses, which the beam model cannot see | Measured (frame_bracket) | Check the body with Static Analysis after sizing and thicken the bars there, or at least report that safety factor |
| 20 ✅ | Topology / Frame Optimization | Only relative numbers ("stiffness 76% of the full design space"): the key box held by one 4 mm boss came out at safety factor 0.16 and 189 mm deflection with nothing said | Measured | Analyse the result and put its safety factor and deflection in the note; warn below 1 or at large deflections |

## P2: crashes, cryptic or wrong messages

- Boolean Intersect of bodies that do not overlap: "negative dimensions are not allowed" (`fields.py:40`); say "the bodies do not overlap".
- Acos with an input entirely outside -1..1: "math domain error", although the port says values are clipped (`fields.py:157`).
- Noise: octaves ≤ 0 gives a constant 0, negative roughness gives values up to 1e9, a negative seed or huge octaves give raw Python errors. Add min/max to the ports.
- Remap with From Low = From High: "float division by zero". Clamp with Low > High silently returns High.
- Export Mesh `.3mf` always fails ("No module named 'lxml'"): add lxml to requirements, or drop .3mf from the picker. A wrong extension gives the garbled `('%s exporter not available!', 'step')`. Relative paths land in the server's working directory and are overwritten silently, and the log doesn't show the full path or size.
- "Field has no surface inside the grid (no sign change)" is the only message for three causes: an empty body, a body thinner than the voxel, and a body clipped by Padding.
- Boolean Union / Smooth Union with an unbounded lattice silently gives a lattice-filled box instead of "connect a Domain".
- Smooth Union with a negative radius turns into an intersection. Smoothen Body with 0 or negative values is silent or cryptic.
- Custom Material: "must be below 0.5", yet 0.499 is rejected.
- Static Analysis repeats the "N bodies analysed as one piece" note on every re-run (it appends to a memoized report).
- Topology / Frame previews accept a missing Fixed Faces, a zero load or Keep 150%; the error comes only after Optimize.
- Error texts name internal ports ("space", "conditions", "cad", "type") instead of the card labels.
- `Graph.problems()` doesn't check literals against `choices`; a saved enum value that is no longer valid is shown as the first option.
- Scale Bodies accepts a NaN scale; Subtract(number, body) loses the bounds ("connect a Domain").
- Reduce Mesh returns meshes with open edges unchanged (a float32-snapped vertex is compared with float64 edge keys).
- Several presets log red errors on their own defaults (Halftone 0.3 mm web, Knurl 0.1 mm gap). "Parquet" is not a real parquet tiling.

## P3: UX on the cards and in the editor

- **Card size and relevance.** Hole/Surface Pattern have 28 inputs (~970 px tall), all always visible; only Remap's Auto greys anything out. About 20 inputs depend on another one (Sides on shape, Ratio/Wall on profile, Fillet on joints, Vector on direction…). Grey them out or hide them, and group inputs (Shape / Layout / Depth).
- **Units and ranges are invisible.** `Port` has no unit/min/max; Angle is radians in Import STEP but degrees in patterns; Keep means three different things; Scale is mm in Noise and a factor in Scale Bodies. Show the unit after the field and clamp to min/max.
- **Lists are invisible.** `PortSpec.many` is never used: Import STEP doesn't say how many bodies it found. Show "×150" on list outputs. On lists, an unqualified face number in Fixed Faces/Load applies to every body silently (11× too stiff in the two-blocks case).
- **Number fields.** Invalid or empty text stays on screen while the old value is used, and every keystroke is a run. A Number made by dragging from an input starts at 0 instead of the input's value, which zeroes the effect. The "Sizes" text reads `2,5` as [2, 5], while number fields read it as 2.5.
- **Face lists.** "10-15", "1.5", "-1" and "5.5" are read as separate face numbers without a warning. Either support ranges or reject anything else. A non-existent body ("5:10") gives an all-zero mask without a warning.
- **Math to number ports.** Math nodes always output a field, so Number → Multiply → Noise Scale / Voxel / Spacing is refused: derived parameters (spacing = 1.5 × size) are impossible. Make Math output `number` when all its inputs are numbers.
- **Previews.** Show the preview voxel ("Draft: 0.4 mm"), warn when a wall or strut is thinner than ~2 voxels, and mark stale views (failed run, Auto off).
- **Palette.** Search ranks description matches above titles ("number" puts Face Mask first, "honeycomb" puts the preset third), and 33 common words find nothing (gyroid, stl, simplify, fuse, box, rotate). Rank by title, then synonyms (names other CAD tools use).
- **Static Analysis.** The numbers live only in the expanded log; the card's range tag (22.1) disagrees with the legend (17.54) and the log (20.4 away from supports). Put max stress / smallest safety factor / max deflection in the card note; print the safety factor as "≥ 10" instead of "5e+10"; warn when the deflection is large compared with the part, or when the safety factor is below 1.
- **Layout.** The toolbar status is clipped to one letter below ~1350 px; the editor's Fit control sits under the log panel; presets paste on top of existing nodes; a kept variant survives New; the log of the previous recipe stays after New/Open; the camera doesn't refit when the body changes size; picking faces turns the CAD view on for good (z-fighting) and marks the recipe dirty.
- **Undo** folds visibility toggles and node moves into the neighbouring step; there are no undo/redo buttons or shortcut help.
- **Names.** Non-ASCII group/selection names (övre, fäste) are refused with a wrong message. Enum spelling is inconsistent (yes/no vs on/off, `schwarz` vs `schwarz_p`).

## Node improvements suggested

**Fields and math**
- Noise: normalise to a true -1..1 (the span is ±0.7 and shrinks with octaves, so Remap -1..1 → 0..1 never reaches 0 or 1).
- Cylindrical angle: declare the range -π..π.
- Sin/Cos: narrow their range from the input.
- Smooth Step with Edge0 > Edge1 already works as a fade-out; document it.

**Import and transform**
- Import STEP: show the body count and the open-edges warning on the card (it is console-only now), and refuse tolerances that explode the triangle count (0.0001 gave 1.15 GB for a small part).
- Scale Bodies with a support: scale about the middle of the actual contact area (a stud on a bigger plate slides sideways now).

**Implicit**
- Shell: add a direction option (inside / centred / outside), as in SolidWorks. Inside should be the default for "hollow out", since centred grows the part by t/2.
- Smoothen: give the setting as a radius in mm, and preserve volume: a 1 mm wall vanishes at 3 × 0.5, and a cube loses 17%.
- Add a node that unions a whole list of bodies into one.
- Make Smooth Union and Union "round" one concept with one name.

**Lattices**
- TPMS: a true wall thickness. Gyroid 1.5 mm gives 1.17 mm; neovius and lidinoid vary ±35% across a cell. Density mode is accurate.
- Strut `blend`: make it work for bcc. Elsewhere it adds far more than a rounding (octet: 26% → 79% solid).
- Add origin and rotation inputs to all lattices. Moving a Voronoi domain by 0.5 mm re-rolls the whole foam.

**Mesh**
- Mesh nodes: show triangles, volume, size and watertightness on the card.
- Mesh from CAD: warn about cracks (8 bad edges on test_part).
- Merge Meshes: make the 1 mm³ speck limit an input, and log intentional separate solids as info, not error.

**Patterns**
- Centre the lattice on the face. Today it is anchored at the CAD origin, so margins come out 4.5 vs 2.5 mm.
- Use the local (u, v) scale on cones, spheres and tori instead of the average. On a cone the cell size varies 0.71–1.28×; on a sphere the shapes bunch at the poles. This matters for lamp shades.
- Unfold single developable faces.
- Warn when a Picture or Text lands outside its face. The "Engraved text" preset on test_part's top face does nothing, because the face centroid is in the hole.
- Text: give Height as cap height, and accept font display names ("Arial Bold").
- Diamond: make Size the pin, like the other shapes.

**Simulation**
- Topology `density` is a constant 0 before Optimize, so Remap Auto fails. Say "not optimized yet" instead.
- Let Keep and Avoid take CAD bodies.
- Skin 0 should mean none.
- Warn before an export runs or resumes an optimization.
- More material presets, and a layer-strength factor for prints.

**Missing basics** a CAD user reaches for first:
- primitives (box, sphere, cylinder)
- Move / Rotate / Mirror / Array of bodies
- a list Union

## Checked and fine

- Static Analysis matches beam theory to within 1.5–2.7% (cantilever, round bar, two blocks). It is linear in the load and scales with 1/E. Pressure equals force, and the material presets and units are right.
- Block meshing equals whole-grid meshing exactly. Noise, 3D Wave, Face Mask, lattices and Smoothen are bit-identical in blocks.
- Exported STLs reload closed. Tolerance and Keep do what they say. Marching-cubes volume is within 0.03% at 0.3 mm.
- The evaluator recomputes exactly the right nodes for these changes:
  - literal edits
  - rewired links
  - a STEP file replaced on disk
  - group setting edits and edited group definitions
  - redefined face selections
  - list length changes
- Errors name the right node, also inside groups. The export queue order is right. Group create/edit/ungroup and copy/paste keep their links.
- Booleans: round and chamfer edges are exact, and bounds propagation is right. Face Mask fade/position/reach/"all other faces" match their formulas. Lattice density mode hits the target within ~1 point.
- All 23 pattern presets run to a watertight mesh on test_part and vent_case. Directions, Edge round through `regrow`, and field-driven inputs work.

## Note for testing

The test editor (port 8766) reopens its own `lastRecipe` from the browser's localStorage at
start-up. During this audit that was `recipes/hex_studs.json`, so its recovery banner offered
the user's unsaved hex_studs changes, and the autosave was deleted. It was restored from the
backup. Before testing, back up `%LOCALAPPDATA%\DesignEngine\autosave`, and clear `lastRecipe`
in the test tab.
