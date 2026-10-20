# Simulation nodes audit (Material, Custom Material, Fixed Faces, Load, Static Analysis, Topology Optimization, Frame Optimization, Join Frames)

All scripts are in `SCRATCH/audit/simulation/` (common.py, s1..s16). Every run went through guard.py (CPU, private cache, peak RSS 0.3-0.85 GB).
Not tested: the GPU path, picking faces in the 3D view, the frontend colour behaviour in a live browser (code read only), `fine` detail, long topology runs (only a 30x10x10 mm design space with DETAIL patched in-process).

## What works (so you know what was checked)

- Static Analysis against beam theory (s1, s14, s6b): 80x12x6 PLA cantilever, 20 N: tip 4.415 mm at draft (-2.65% vs Timoshenko 4.536), 4.455 at medium (-1.77%); D10x60 round bar: -1.8% draft, -1.45% medium; two touching blocks (Many) -2.0%. Load 40 N gives exactly 2.0000x the displacement; steel vs PLA scales with 1/E (and nu). Pressure 0.5 MPa on a 72 mm2 face = force 36 N, identical results, and 0.01141 mm = sigma L / E. Mass is exact (7.15 g = 5760 mm3 x 1.24).
- Material preset values (E, nu, density, strength) match datasheet-typical values: PLA 3500/0.36/1.24/50, PETG 2000/0.38/1.27/45, ABS 2200/0.35/1.04/40, PA12 1700/0.39/1.01/48 (SLS PA2200), Al 6061-T6 69000/0.33/2.70/276, S355 210000/0.30/7.85/355. Units MPa and g/cm3 are right everywhere (report, mass, labels).
- Vector parsing: "0, 0, -1", "0,0,-1", "0 0 -1", "0;0;-1", "1.0e0" all parse; Swedish decimal commas ("0,5 0 -1") are rejected with an error, never silently misread.
- Error messages are clear and sit on the right node for: unknown face, empty faces, bearing without a direction, bearing against a flat face, no Fixed Faces, no Load, zero amount, Custom Material E<=0 / density<=0 / strength<=0, frame sizes too small, min size 0, safety<1, curving>1.
- Two loads on the same face add up exactly (tip 8.83 = 2 x 4.415). A load on the fixed face gives zero stress plus a note.
- Topology: Smoothing changes do NOT re-optimize (0.1 s vs 9 s, no messages), Optimize on a finished state does nothing, Keep/Avoid bodies are respected (inside/outside checks), Gap, Max thickness, Struts only, Min thickness all run; Keep % below the skin share gives a good error.
- Frame Optimization: all 8 profiles, sizes list, max sizes, min size, curving 0, mid-bar off, joint cost 0, spacing 40, load 1e5 N run in 0.6-2.5 s at spacing 8 mm and 11 s at the default spacing (peak 0.37 GB); Join Frames and Connect to work.

---

## Findings

### [BUG] Static Analysis: the "N bodies analysed as one piece" note is added again on every re-run
Severity: low · Status: VERIFIED (s16_notes_dup.py)
Evidence: `engine/graph/nodes_fea.py:288-289` appends to `result.report["notes"]`, but `FA.static_analysis` returns the *memoized* Result object (`analysis.py:205-207`, `_RESULTS`). Output of three runs of the two-blocks case that only change Show: run 1: one "note: 2 bodies analysed as one piece", run 2: two, run 3: three.
Why it matters to the user: switching Show (stress / safety / displacement), which is instant because of the memo, makes the log fill with repeated notes; the longer the session the longer the report block.
Suggestion: add the note inside `FA.static_analysis` (it knows the part count if passed in) or copy the report before appending (`notes = report["notes"] + [...]` into the summary only).

### [BUG] Safety factor colouring: a part that is safe everywhere (SF >= 10) is painted with the "bad" colour; the legend/range can be inverted
Severity: medium · Status: engine side VERIFIED (s1_static.py), frontend side from code reading (SUSPECTED)
Evidence: steel S355, 20 N, draft: `safety` Field range = (10.0, 10.0). `server._paint` (server.py:517-530) gets lo = hi = 10 from the vertex values; the frontend `normalise` (SliceView.tsx:41) uses `span = max - min || 1` so v = 10 gives t = 0, and `paintColors` (Viewport.tsx:49-53) then reverses it (t = 1 - t = 1) which is the red end of the rainbow. The legend reads "10 ... 10". Second case, medium detail, steel 20 N: `analysis.py:280` sets `display_range = (strength / max_away, SAFETY_CAP)` = (17.28, 10.0), i.e. low > high; values near the support then map to t >= 1 (clamped) and the legend reads "17 ... 10".
Why it matters to the user: the first thing a designer tries is a stiff material or a light load; the part then shows the same red as "will break". Looks exactly like "something isn't working".
Suggestion: clamp `display_range` for safety to `(min(strength/max_away, SAFETY_CAP), SAFETY_CAP)`; in the paint, when lo >= hi fall back to a fixed range such as (1, 10) for the safety factor (so 10 is blue); or always use a fixed safety scale 0..10 (it is capped at 10 anyway).

### [BUG] Custom Material: Poisson's ratio message contradicts the check (0.499 is rejected with "must be below 0.5")
Severity: low · Status: VERIFIED (s8_custom.py: nu = 0.499 -> "Poisson's ratio must be below 0.5 (rubber-like) and above -1")
Evidence: `engine/fea/materials.py:37` accepts only -0.99 < nu < 0.499 but the message (line 38) says below 0.5 and above -1.
Why it matters: someone typing rubber (0.499 or 0.4995) is told a value below 0.5 is wrong.
Suggestion: say "between -0.99 and 0.499" (and note that near 0.5 results get stiff/inaccurate: at nu 0.49 the cantilever was already 4.0% short against 2.7% at 0.36).

### [BUG] Topology Optimization: a finished result is thrown away when only Strength or Density (or E) of the material changes
Severity: medium · Status: VERIFIED (s3_topology.py)
Evidence: `topology_key` (topology.py:199) uses `material.key` = (E, nu, density, strength). The optimization only depends on nu (E scales the compliance only; loads are fixed forces; density and strength are never used). Run: Optimize with PLA (9 s), then a Custom Material with identical E/nu/density and strength 55 -> preview says "not optimized for these settings yet: press Optimize" and the density output is 0 again. Same for switching PLA -> a custom PLA with another density. (Frame Optimization really needs strength and density, so its key is right.)
Why it matters to the user: on a real medium/fine run (minutes) changing the safety-related strength or the density label in the material, or trying PLA vs PETG with the same shape intent, silently loses the result and everything downstream (Smoothen, textures, meshes) recomputes on the full design space.
Suggestion: key on `nu` only (plus E only if you want compliance numbers kept), or at least drop density and strength.

### [UX] Topology / Frame Optimization: invalid settings are only reported when Optimize is pressed
Severity: medium · Status: VERIFIED (s13_preview_validation.py, s3)
Evidence: with compute=0 these all return "not optimized for these settings yet: press Optimize" with no error: no Fixed Faces; a Load with amount 0; Keep % = 150. With compute=1 they fail: "the part is not held anywhere", "there is no load", "Keep % must be between 0 and 100", frame: "the loads add up to nothing". `nodes_fea.py:453-464` and `nodes_frame.py:112-131` validate only part of it; `_validate` (nodes_fea.py:306) checks that conditions have faces but not that there is a support and a load.
Why it matters: the user wires everything, sees a calm note and a green node, presses Optimize, waits for the queue and only then learns that the setup was wrong. Static Analysis reports the same problems instantly.
Suggestion: run `apply_conditions` (cheap, 0.1-1 s) or at least the "has a support / has a load with amount != 0 / 0 < keep < 100" checks in the preview path and raise there.

### [UX] Static Analysis report: the headline numbers disagree with each other and with a hand calculation
Severity: medium · Status: VERIFIED (s1_static.py)
Evidence (80x12x6 PLA cantilever, 20 N, theory: root stress 22.2 MPa, SF 2.25): at medium the log says "max stress 20.4 MPa ... smallest safety factor 2.4" (the maximum *away from supports*, `analysis.py:249-252`), the node's range tag and the field range say 0..27.8 MPa (includes the artificial support peak), the colour bar says 0..20.4 and "clipped", the safety field's own minimum is 1.80 (display range 2.45..10). At draft the log's SF is 2.4 while the safety field's minimum is 2.22. Stress theory at the position of the reported maximum (x = 3 mm) is 21.4, so the numbers are right in themselves: it is the choice of what to call "max" that confuses. For a clamped beam the true maximum IS at the support.
Why it matters: the designer who checks the tool against a hand calculation sees a safety factor 6-10% higher than theirs (non-conservative), and three different "max stress" numbers depending on where they look. `Remap` with Auto on the stress output uses the field range (0..27.8), so a stress-driven lattice is scaled to the artificial peak, not to the 20.4 that is shown.
Suggestion: print both in the log ("2.4 away from supports, 1.8 at the fixed faces"), put the same range in `Field.range` as in `display_range` (or document it), and optionally show the report on the node card (see next item).

### [IMPROVEMENT] Static Analysis: put the result summary on the node card (`note`), format the safety factor sanely
Severity: low · Status: VERIFIED
Evidence: `static_analysis` sets no `note` (Topology and Frame do); the only place for max stress / SF / deflection / mass is the log panel. Safety factor is printed with `:.2g`: "5e+10" (load on the fixed face), "1.7e+02", "1.9e+02", "4.7e-08".
Suggestion: `result.note = "20 MPa max · SF 2.4 · 4.4 mm · 7.1 g"`; print SF as ">10" above the cap and with 3 significant digits ("170").

### [UX] Fixed Faces / Load with a list of bodies: an unqualified face number silently applies to every body
Severity: medium · Status: VERIFIED (s6b_two_blocks.py)
Evidence: samples/two_blocks.step (two touching blocks, same face numbering). Fixed "0:0" + Load "1:1" 24 N -Z: tip 0.0750 mm (theory 0.0765). Fixed "0" + Load "1" (what you get if you type or pick on a different body): fixed face 0 exists on both bodies (x = 0 and x = 40, the glued interface), load face 1 on both (x = 40 and x = 90): tip 0.0068 mm, 11x too stiff, no error, only the generic "note: 2 bodies analysed as one piece". This is by design for copies (`facemask.faces_for_body`), but the picked-area and body count are never reported.
Why it matters: a plausible-looking but wrong analysis; the user thinks the load is on one end.
Suggestion: when a condition's faces land on more than one body (and the bodies are not identical copies) say so in the Load/Fixed node note ("faces 0 found on 2 bodies, 480 mm2") and always show the picked area ("face 1: 72 mm2") on the card or in the log.

### [UX] Faces text: "1.5" means faces 1 and 5, "-1" means face 1, silently
Severity: low · Status: VERIFIED (s2b.py: "1.5" -> areas [72, 1032]; "-1" -> face 1)
Evidence: `facemask.faces_for_body` regex `(\d+)\s*:\s*(\d+)|(\d+)`.
Suggestion: reject anything that is not digits, commas, spaces, colons and semicolons.

### [UX] Load: Offset errors talk about "the vector"; irrelevant inputs always visible; Amount has no unit
Severity: low · Status: VERIFIED (messages), SUSPECTED (UI)
Evidence: Offset "10,0" -> "the vector must be three numbers like 0, 0, -1 (got '10,0')" (`model.parse_vector` reused for Offset, `nodes_fea.py:185`). Vector (custom only), Offset (force/bearing only) and Direction (not for pressure) stay on the card for every kind; the Amount unit (N / MPa / N·m) changes with Kind but the label is just "Amount" (the unit is only in the tooltip). `OPTIMIZERS`/`PortEditor` have no conditional visibility.
Suggestion: pass a name to `parse_vector` ("offset must be three numbers like 10, 0, 0"); grey out ports that do not apply; label the amount dynamically or put the unit in the Load card note (the label already has it: "100 N force on ...").

### [IMPROVEMENT] Static Analysis does not warn when the linear assumption is broken
Severity: medium · Status: VERIFIED (s8_custom.py)
Evidence: Custom Material E = 0.05 MPa, 20 N: "max deflection 3.11e+05 mm", SF 0.00047, no warning that the analysis is meaningless. Less extreme: the 80 mm PLA cantilever at 40 N (SF 1.2) deflects 8.9 mm (11% of its length) and is reported as exact. Also nothing is said when SF < 1 beyond the number.
Suggestion: a note when max deflection > ~5% of the part's size ("small-deformation analysis: results are not reliable") and when SF < 1 ("beyond yield; the stress shown is what a linear material would carry").

### [IMPROVEMENT] Topology Optimization before Optimize: `density` is a constant 0 with range (0, 0); Keep/Avoid cannot take a CAD body
Severity: low · Status: VERIFIED (s3: "density range=(0.0, 0.0)", Remap Auto raises "X is constant (0), there is no range to stretch" at `nodes.py:248`)
Evidence: `nodes_fea.py:471-475` returns `F._lift(0.0)`. `core.py:115-129`: port type `implicit` does not accept `cad`, while Design space (`part`) does, so a Keep or Avoid solid imported from STEP needs an Implicit Body from CAD node in between.
Suggestion: before Optimize return density with range (0, 1) (or the Keep % as a uniform value) so a downstream Remap does not fail with a message that never mentions Optimize; make Keep/Avoid ports `part` and convert through `_part`.

### [IMPROVEMENT] Topology: Skin 0 does not mean "none"; Skin below the element size is the same as the element size
Severity: low · Status: VERIFIED (s12_topo_opts.py)
Evidence: `topology.py:246` `reach = max(float(skin), h)`. Skin 0 and Skin 1.0 at 1.1 mm elements gave the identical result (116 iterations, 33%), yet are two different keys so both are optimized. The port doc says "(0 = none)".
Suggestion: say "at least one element layer", and key on the effective reach; log "skin: 1.1 mm (one element)" with the element size.

### [IMPROVEMENT] Join Frames / Connect to: the load one frame puts on the other is not added to the other frame, and nothing checks the joined result
Severity: high (safety) · Status: VERIFIED (s11_join.py, recipes/frame_join.json)
Evidence: `frame_b` log: "connect to #1: joined at 4 points, this frame puts (0, 0, -120) N on it". Frame A was optimized for its own 120 N (6.17 g, utilisation 0.93) and now carries 240 N where B lands; `Join Frames` returns a body with no note, no mass and no check (`note` is None). The port doc admits "which is not added to the other frame" but the frame is shown as passed (utilisation 0.93).
Why it matters: the designer sees two frames that each "pass" and a nicely filleted joined part that is in fact overloaded.
Suggestion: Join Frames should run the beam check on the combined frame (or at least print "frame A sees 240 N at the joint, its bars were sized for 120 N, utilisation x") and the joined body should carry a note with total mass.

### [IMPROVEMENT] Frame Optimization: the "Supports" option shares its name with the "Supports & loads" input; ratio/wall always visible
Severity: low · Status: SUSPECTED (UI)
Evidence: ports `conditions` (label "Supports & loads") and `supports` (label "Supports", choices connect all / only needed). Ratio only applies to rectangle, rect tube and I; Wall to the four tube/angle profiles (`nodes_frame.py:133-134`); Fillet only to fillet/ball joints. Skin is not part of the key (unlike the docs for Joints and Fillet it does not say "changing it does not optimize again"; verified that changing it does not).
Suggestion: rename the option "Unused supports" (join them / leave out); hide or grey irrelevant inputs; add the same remark to Skin.

### [IMPROVEMENT] Frame Optimization: check against the voxel Static Analysis of its own body (known, still worth showing)
Severity: low · Status: VERIFIED (s5_frame_fe.py)
Evidence: recipes/frame_bracket.json at spacing 8: frame check says 3.88 g, utilisation 0.95 (allowable stress 24 MPa at safety 2), deflection 0.96 mm. Draft Static Analysis of the frame body (with bosses and fillets): 8.49 g, max 29.9 MPa (SF 1.6, not 2), deflection 0.79 mm. The mass difference is the skin/bosses at holes and the fillets (not in the beam model). Consistent with FINDINGS ("the frame check cannot see joints"), but the node note quotes only the beam-model numbers.
Suggestion: say "beam model" in the note ("3.9 g of bars, 8.5 g with bosses and fillets") or report the body mass (the mesh/body volume is available) next to the bar mass.

### [IMPROVEMENT] Materials: short list, no print-orientation knockdown, Custom Material has no name
Severity: low · Status: VERIFIED
Evidence: `materials.py` has six presets. Missing for this user: stainless 316L, Ti-6Al-4V, polycarbonate, TPU, PA12-CF/GF, resin, CF-PETG. Layer strength is only mentioned in the tooltip; the report always says "(custom, strength 50 MPa)". Default Poisson of Custom Material is 0.35 while PLA is 0.36.
Suggestion: more presets; an optional "Z strength factor" or two strengths (in-plane / between layers) with the lower one used by the safety factor; a Name on Custom Material.

### [IMPROVEMENT] Simulation features CAD users reach for and that are missing
Severity: low · Status: observation
Evidence: only fully fixed supports (no roller/slider, no radial-only bolt support), no self weight/gravity, no signed or per-axis displacement (Displacement is a magnitude, so a load of -20 N and +20 N look identical), no deformed-shape view, one load case per run. The plan lists them as phase 3.
Suggestion: order by value: gravity (cheap: body force from density), a "sliding/frictionless" support, signed displacement component, deformed view.

### [UX] Exports silently run (or finish) every Topology/Frame optimization that is missing
Severity: low · Status: SUSPECTED (code: `server._set_compute`, server.py:87-95, `mode == "export"` -> compute = 1)
Evidence: an export sets `compute=1` on every Topology and Frame node in the graph; a stopped optimization is continued to the end (`nodes_fea.py:468` runs while `not done`).
Why it matters: pressing Export with a medium-detail Topology node in a changed state starts several minutes of optimization with no confirmation, and a "stopped" shape the user liked is replaced by a continued one.
Suggestion: warn before queueing an export that will optimize ("Topology Optimization 'x' is not optimized for these settings: this export will run it first"), and do not continue a stopped state in an export, export what is saved.
