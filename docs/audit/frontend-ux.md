# frontend-ux audit: how nodes are presented and edited in the React editor

Scope: static review of `frontend/src` (EngineNode, NodeEditor, NodePalette, Toolbar, Viewport, SliceView, HelpPanel,
LogPanel, FileBrowser, store, graphConvert, styles, types, api) plus the engine catalog that drives the cards
(`engine/graph/core.py`, the `@node` definitions). The editor was **not run** (the UI agent does that); everything
marked VERIFIED was reproduced with in-process engine scripts, a node one-liner, a catalog simulation or React's own source.
Scripts and dumps: `SCRATCH/audit/frontend-ux/` (`catalog.json`, `docs_dump.txt`, `conn_cases.py`, `msg_replay.py`,
`palette_search.py`, `sizes_comma.py`, `analyze_catalog.py`). No repo file was touched.

Catalog facts used below: 53 nodes, 258 visible inputs; input types: number 80, text 56 (all enums are `text` with
`choices`), scalar 46, cad 16, implicit 16, int 16, faces 7, mesh 5, frame 5, conditions 5, path 4, material 3, part 3.
Output types: `number` exists on exactly one node (Number). Every input has a `doc` (good), but `Port` has no unit/min/max.

Ranked: bugs first, then UX, then improvements.

---

### [BUG] Node warnings (`report_message`) are shown once, only as a log line, and are gone on the next run
Severity: high · Status: VERIFIED (`frontend-ux/msg_replay.py`)
Evidence:
- `core.py:104-111` `report_message` emits a `message` event; `core.py:643-648` a cache hit emits only `node_done`
  (with range/note), never the node's earlier messages. `store.ts:439-440` puts the text in the log and drops `e.node`.
- `LogPanel.tsx:8` shows only the **last** log line while collapsed (`log.slice(-1)`).
- Repro: Hole Pattern on `vent_case.step` face 0 with spacing 2 < size 4, evaluated 3 times with one Evaluator:
  run 1 = 5 message events, two of them level `error` ("9024 pairs of holes overlap: increase Spacing or reduce Size",
  "thinnest web between holes -1.99 mm (under 0.8 mm: fragile)"), and the **last** one is an info line ("for clean edges,
  mesh with a voxel of at most 0.67 mm"), so the collapsed log shows the harmless line and hides both errors.
  Run 2 (nothing changed) and run 3 (only the downstream Boolean Subtract changed): **0 messages**; the Hole Pattern card
  note stays "1598 holes" (`nodes_pattern.py:193`, `card_note` only says count / "none placed" / "all regular").
- Same pattern for other warnings that matter: Merge Meshes "N meshes gave K separate solids" (`nodes.py:697-698`, only
  at Final/export, followed by green "Exported ..." and "Export finished" lines, and a desktop notification that says
  "Export finished" regardless), Import STEP "body N has X million triangles" (`nodes.py:72`), Topology/Frame `note:` lines
  (`nodes_fea.py:332,365`, `nodes_frame.py:152`), Reduce Mesh "nothing could be removed" (`nodes.py:673`).
- Messages use `level="error"` for non-fatal warnings (only `info|error` exist; `LogEntry.level` has info/error/success),
  so "error" in the log does not mean the run failed.
- SUSPECTED extra: `report_message` returns silently when `_current.ctx` is None (`core.py:107-109`), which is the case
  inside the worker pool. Reduce Mesh is `parallel=True` and calls it (`nodes.py:673`), so for a list of >= 2 meshes the message is lost.
Why it matters: this is exactly "something isn't working as it should": the node produced a warning-worthy result (overlapping
holes, loose solids in the STL) and the user sees at most one transient grey line.
Suggestion: keep a per-node `messages` list in the evaluator cache and replay it on cache hits (or store it on the value like `note`);
send `node` with each message; add a `warn` level; show warnings as an amber block on the card (below `enode-note`), with a count
badge; make log lines clickable (select + zoom to the node); open log shows newest at the bottom.

### [BUG] Dragging a wire onto the canvas and picking a node connects to the wrong input
Severity: medium · Status: VERIFIED by simulating the code on the real catalog (not clicked in the UI)
Evidence: `NodePalette.tsx:14-16` `compatibleInput` returns the **first** input whose compat list contains the output type,
with no preference for an exact type match; `NodeEditor.tsx:244-253` uses it. With `compat["implicit"] = [implicit, number, scalar]`
and `compat["int"] = [int, number]` the first input wins. Simulated over the catalog:
- Dragging a **Number** output: 44 nodes are offered, **27 connect to a questionable input**: Offset Body -> `Body` (should be Distance),
  Hole/Surface Pattern -> `Sides` (int) instead of Size/Spacing, Mesh from Implicit Body -> `Body` instead of Voxel,
  Smoothen Body -> `Body` instead of Grid Size, Boolean Union/Subtract -> `A` instead of Radius.
- Dragging a **scalar** (field) output: Offset Body -> `Body` (not Distance), Shell -> `Body` (not Thickness), Fill Body with Lattice -> `Body` (not Skin).
Why it matters: the user picks "Offset Body" precisely to drive the distance; the wire lands on the body input, the card turns
red or silently makes a nonsense graph (see the next entry) and the wire has to be deleted and redone.
Suggestion: prefer exact type, then first compat input; skip inputs already linked; when several fit, let the palette show
"Offset Body -> Distance" as separate rows (a port-aware search), or pick the one the dragged type matches by name hint.

### [BUG] Connection rules: a Number can drive a body input, the refusal gives no reason, and the error shows up on a different node
Severity: medium · Status: VERIFIED (`frontend-ux/conn_cases.py`)
Evidence:
- `core.py:115-129` `_COMPATIBLE["implicit"] = {implicit, scalar, number}`. Repro: Number(5) -> Offset Body.body -> Mesh from Implicit Body:
  `problems()` is empty, the run fails on the **Mesh** node with "cannot tell where this body is; connect a Domain".
  Nothing marks the Number-to-body wire (nonsense for a designer; a field-as-body is meaningful, a bare number is not).
- Refused wires: `NodeEditor.tsx:214-222` returns false silently. There is no `.valid`/`.connectingto` styling in `styles.css`
  (React Flow's own CSS has none either), no tooltip on handles (`EngineNode.tsx:304,315` have no `title`), and input rows do not show their
  type (only outputs have `type-tag`). A user who drags a field onto a `number` input (80 of the 258 inputs are `number`) just sees the wire snap back.
- Dropping onto an incompatible handle does nothing at all (`NodeEditor.tsx:225`: `state.toHandle` -> return).
- Int ports accept fractions through a link: Number(2.5) -> Noise.octaves runs without complaint; Number(1.5) -> Pick Body.index gives the
  misleading "body index 1.5 out of range (file has 1 bodies)". The typed field rounds (`EngineNode.tsx:31`), the linked path does not.
Suggestion: drop `number` from `compat["implicit"]` (keep scalar); highlight valid targets while dragging and dim invalid ones; on drop onto an invalid handle
show a toast/log line "A Number cannot go into a body input (needs: implicit)"; add `title="<label>: <type>, accepts a number or a field"` on handles;
round or reject non-integers at `int` ports in the engine.

### [BUG] After a failed run the 3D view keeps showing the old geometry with no marker; one failing node stops independent branches; Auto-off has no "out of date" cue
Severity: medium · Status: SUSPECTED (from code)
Evidence:
- `store.ts:394-398` `run_start` sets `status: {}` (so every card loses its range tag, note and green dot at the start of every run, and flickers while a slow upstream node
  runs) but keeps `views`. `Viewport.tsx:1134-1139` legend: a node with an old view shows "N k tris" even if its node now has `state === 'error'`; "error" is only shown when there is no old view.
- `core.py:656-661` the first exception aborts the run: independent visible branches after it in topological order are never computed, their views stay old.
  `server.py:416` publishes problems for every node (so a new, half-wired node is red at once) and `core.py:620` validates all needed nodes, so one visible half-wired node makes the whole preview "Failed".
- Toolbar says only "Failed" (`store.ts:452`); the reason is in the log (last line) and on the card, which can be off-screen in a big graph.
- With Auto off (`Toolbar.tsx:267`), nothing says the preview no longer matches the graph (grep: no stale/outdated state outside the Print check).
Suggestion: tint the legend row / dim the view of a node whose status is error or not-yet-updated; show "Failed: <node id>: <message>" in the toolbar and make it clickable (select and fit the node);
keep notes/ranges of the previous run until the node reports again; a small "preview out of date" chip when Auto is off and `graphVersion` moved.

### [BUG] Number fields: empty or invalid text stays on screen while the old value is used; every keystroke is a real edit
Severity: medium-low · Status: parse table VERIFIED (node one-liner), display divergence SUSPECTED (`EngineNode.tsx:23-48`)
Evidence: `NumberInput.commit` ignores anything that is not a finite number, and nothing reverts or flags it on blur. Parse table (same expression as the code):
`"0,35"`->0.35, `"1e-3"`->0.001, `"1,5e2"`->150, `"4."`->4, `"+4"`->4, `"0x10"`->16; ignored (value unchanged, text stays): `""`, `"-"`, `"5mm"`, `"5 mm"`, `"10%"`, `"1/3"`, `"2*3"`, `"1 000"`, `"1.000,5"`, `"abc"`.
So: clearing a field leaves it blank while the engine still uses the old value (the card looks like "unset"); pasting "12 mm" (a SolidWorks habit) is silently dropped; there are no expressions; no min/max
(scrub and arrow keys can make Voxel, Cell, Spacing negative; the engine then errors on the card with "must be 0 or more", `nodes_pattern.py:175`).
Typed "12mm" key by key works by accident (the "12" prefix was committed).
Every keystroke commits (`EngineNode.tsx:46`, `store.ts:331`), so with Auto on and a pause of > 450 ms (`App.tsx:18`) a prefix such as "1" of "12" Spacing starts a real preview
(and "0" of "0.5" Scale/Voxel gives an error flash). A Hole Pattern with spacing 1 on a big face is a heavy compute; the memory watchdog protects the PC but the user sees a long stall.
Suggestion: commit on Enter/blur (keep live preview only while scrubbing), revert to the last good value on blur, red border + tooltip for unparseable text, strip trailing `mm`/`°`/`%`, optional tiny expression parser ("1.5*4"),
and take `min`/`max`/`unit` from the port (see the units entry). Arrow-key step for an integer-looking "100" is 0.1 (`EngineNode.tsx:38`): use 1 (or 10% of magnitude).

### [BUG] A saved enum value that is no longer in `choices` is displayed as the first choice
Severity: low-medium · Status: VERIFIED from source (React `updateOptions`, `react-dom.development.js:2135-2154`; `EngineNode.tsx:173-178`)
Evidence: for a controlled `<select>` whose value matches no option, React marks the **first** option selected. The stored value is untouched, the engine then fails with e.g.
"pattern must be one of ..." (repro: `field.wave3d` pattern `schwarz_p`, `lattice.tpms` pattern `foo` in `conn_cases.py`), while the dropdown shows a valid-looking choice.
This happens whenever an enum value is renamed in the engine and an older recipe is opened (the project renames choices from time to time).
Suggestion: add the unknown value as an extra red option "foo (not valid)" and mark the row.

### [BUG] Frame Optimization "Sizes": a decimal comma is silently read as a list separator
Severity: medium-low (the user types Swedish decimals: Number fields accept "0,35") · Status: VERIFIED (`frontend-ux/sizes_comma.py`)
Evidence: `nodes_frame.py:32` replaces `,` and `;` by spaces: `"2,5"` -> sizes [2.0, 5.0]; `"2,5; 3"` -> [2.0, 3.0, 5.0]. No error. (Load `vector`/`offset` fail loudly instead: "the vector must be three numbers like 0, 0, -1 (got '0,5 0 -1')", `model.py:326`.)
Suggestion: treat `,` as separator only when followed by a space, or reject numbers with a comma and a digit on both sides with a clear message, and echo the parsed list on the card ("sizes 2, 5 mm").

### [BUG] Face lists: "10-15" and "10..15" mean faces 10 and 15
Severity: low-medium · Status: VERIFIED (`facemask.faces_for_body("10-15", 0) == [10, 15]`, same regex in `types.ts:236`)
Evidence: the parser takes every digit group, ignores everything else ("1;2" -> 1,2; "face 4" -> 4), no range syntax, no warning. Face numbers also start at 0 (Face Mask error text: "body 0 has faces 0-29"), which the card never says.
Suggestion: support `a-b` ranges, warn on unparsed characters, show "0-based" in the placeholder/tooltip.

### [BUG] Error texts name inputs the card does not show
Severity: low · Status: VERIFIED (`conn_cases.py`; `core.py:428-430`; `lattice.py:125,251`)
Evidence: "required input 'cad' is not set" (the row says "Body"), 'space' vs "Design space", 'conditions' vs "Supports & loads", 'a' vs "Frame A". TPMS and Strut Lattice report "type must be one of gyroid, schwarz_p, ..." for the input labelled **Pattern**
(there is no "type" input). 3D Wave says "pattern must be ..." correctly. 46 ports have an explicit label different from the name-derived one (`analyze_catalog.py labels`).
Suggestion: build all messages from `port.label`; for enum errors use one helper that takes the Port.

### [BUG] Log panel usability
Severity: low-medium · Status: SUSPECTED (`LogPanel.tsx`, 20 lines)
Evidence: open shows `log.slice(-60)` top-to-bottom in a `max-height: 50%` scroller with no scroll-to-bottom, so the newest lines are below the fold; the whole panel toggles on any click (`onClick` on the container), which also collapses it when the user clicks to select/copy an error text;
lines carry no node reference; 200-line cap.
Suggestion: scroll to the newest on open, toggle only via a header button, make lines clickable, add a "copy" button.

---

## UX findings

### [UX] Hole Pattern / Surface Pattern: 28 inputs, always all visible, ~970 px tall, nothing explains which ones apply
Severity: medium-high · Status: layout computed from CSS (30 px rows + 1 px border x 30 rows + header), SUSPECTED visually
Evidence: `EngineNode.tsx:309-326` renders every non-hidden input; `.enode-row min-height 30px` (`styles.css:152-157`). Hole Pattern = 1 output + 28 inputs (~965 px at zoom 1), Surface Pattern 28 inputs, Frame Optimization 20, Topology 13.
`fitView` caps at zoom 1, so such a card is cut off or shrunk to ~0.6 on a 1080p editor pane. Only one "ignored" rule exists, hard-coded for Remap/Auto (`EngineNode.tsx:17-18`). Inputs that only matter for some settings:
- Hole/Surface: `sides` (polygon/star), `length` (slot, diamond, cross, rectangle, chevron, star), `web` (voronoi/delaunay), `corner_radius` (polygons, diamonds, crosses, cells), `morph` (only if Morph To != none), `seed` (only if Randomness != 0), `rotation` (not Voronoi), `depth` (not used with Height), `size` changes meaning per shape.
- Boolean Union/Subtract `radius` (edge != sharp); Load `vector` (direction = custom), `offset` (force/bearing only), `direction` (not pressure); TPMS `thickness` (sheet) vs `offset` (solid modes); Face Mask `position`/`band` (reach = surface); Frame `ratio` (rect/I), `wall` (tubes/angle), `size_step` (sizes = any); Topology `members` (needs Max thickness); Mesh from Implicit `keep` (replaced by `tolerance`, both only at Final/export).
Also the order differs between the two sibling nodes (Hole: ... depth, height, size_scale, min_size, tilt; Surface: ... softness, invert, band, size_scale, min_size) and related inputs are far apart (`rotation` vs `angle`; `depth`/`height`, which decide hole vs pocket vs boss, sit at position 24-25).
Suggestion: add optional port metadata `group` ("Faces", "Shape", "Layout", "Variation", "Depth") and `when: {"shape": [...]}`; the card collapses groups and dims (or hides, with a "show all" toggle) inputs whose `when` is false (generalise `ignoredByAuto`); put `depth/height` next to Shape.

### [UX] Units, ranges and "what does this number mean" are only in hover tooltips / the Help panel
Severity: medium · Status: VERIFIED (catalog: `Port.describe()` has no unit/min/max; `analyze_catalog.py units`)
Evidence: card rows show only the label and a bare input; `p.doc` is the row's `title` (`EngineNode.tsx:314`) and in the Help panel (only for exactly one selected node, `HelpPanel.tsx:31`; it does not list `choices`, `many`).
Worst offenders (meaning or unit not visible, or the same word with different units):
- `Angle`: Import STEP = **radians** (0.5), Hole/Surface/Picture/Text Angle, Rotation, Tilt = **degrees**, Cylindrical Coordinate "angle" = radians, Sin/Cos x = radians.
- `Keep`: Mesh from Implicit = share 0..1 of triangles (1.0); Topology `Keep %` = 30 (percent of mass); Topology `Keep` (`keep_body`) = a body. Three meanings, two nodes.
- `Scale`: Noise = size of blobs in mm (bigger = calmer); Scale Bodies = factor.
- Load `Amount`: N, MPa or N*m depending on `Kind` (card shows none; doc does).
- Frame: `Wall` 0.15 = share of width (not mm), `Joint cost` = share of the space's diagonal, `Curving` 0..1, `Safety factor`, `Node spacing` 0 = auto, `Height / width`.
- `Density`: Custom Material = g/cm3; lattice `Units: density` = 0..1 solid share; Topology `Density` output 0..1; Print check density g/cm3.
- `Tolerance`: Import STEP (accuracy, mm), Mesh from Implicit (0 = off, smart reduction, replaces Keep), Reduce Mesh (mm). `Smoothen Grid Size` "mm".
- `Skip Duplicates` is an int 0/1 used as a boolean and shown as a free number field; `Randomness`, `Roughness`, `Morph`, `Blend` have an implied 0..1.
Suggestion: extend `Port` with `unit` ("mm", "deg", "%", "MPa", ...), `min`, `max`, `step`, `kind="bool"`; show the unit as a dim suffix in the input, clamp/flag out-of-range, render booleans as toggles; where the unit depends on another input (Load Amount) let `unit` be a function of `kind`.

### [UX] Same concept, different name / spelling / defaults
Severity: low-medium · Status: VERIFIED (`catalog.json`)
- Enum spelling: 3D Wave `schwarz` vs TPMS `schwarz_p` (same surface); yes/no vs on/off vs tangent/no (`fuse`, `auto`, `invert`, `mid_bar`, `join`); axis `x|y|z` (Coordinate, Align) vs `+X ... -Z` (Load); `Detail` draft|medium|fine vs toolbar Draft|Medium|Final; raw values like `schwarz_p`, `cubic + bcc`, `checker 90`, `I`, `rect tube` are shown as stored.
- Label style: Simulation nodes use sentence case ("Min size", "Max sizes", "Size step", "Safety factor", "Node spacing", "Design space", "Gap to faces"), everything else Title Case ("Min Size", "Edge Margin", "Size Scale"). Two nodes both show "Min Size" with different meaning (Hole: drop holes narrower than this; Frame: thinnest bar).
- Auto labels: `implicit.from_cad` input shows "Cad" (every sibling says "Body"); "Edge0"/"Edge1"; every single-output node's output is "Out".
- Same idea, many words: band (Implicit from CAD, Face Mask, Picture/Text/Surface Pattern: three different meanings, all "at least the largest offset"), padding (Mesh), margin (Hole), skin (Fill, Frame, Topology), gap (Topology), fade (Face Mask), softness (Surface), wall/thickness/web; scale/size/cell/period/spacing/voxel/grid size for lengths.
- Defaults: Boolean `radius` 0.5 vs Smooth Union 1.0; `detail` medium (Static) vs draft (Topology); strut/foam thickness 0.8 vs TPMS 0.6.
- Overlapping nodes: Smooth Union vs Boolean Union with `edge=round` (different radius semantics: "about" vs "exactly"), two nodes titled "Subtract" (Math and "Boolean Subtract"), three "Smooth*" (Union, Smoothen Body, Smooth Step).
Suggestion: one helper for enum labels/values, a single casing rule, an `angle` unit rule (degrees everywhere in the UI), `Body` label everywhere.

### [UX] Lists (`Many`) are invisible
Severity: medium · Status: VERIFIED (grep: `PortSpec.many` is never used in the frontend; `_output_extras` `core.py:463-474` sends markers/note/frames only)
Evidence: after Import STEP nothing says how many bodies were read; a node mapped over a list shows no "x12"; `many` ports (Pick Body, Merge, Export, Fixed Faces, Load, Static/Topology/Frame `space`) look the same as single ports; length mismatches only appear as "list inputs have different lengths [..]" at run time.
Suggestion: add `count` to `node_done` for list outputs and show a small "x12" badge on the output tag and on linked rows ("<- import (12)"); Import STEP `note` "3 bodies, 41 faces"; mark `many` ports ("list ok").

### [IMPROVEMENT] Field vs number inputs, and no derived parameters
Severity: medium · Status: VERIFIED (catalog)
Evidence: 80 inputs are `number`, only `input.number` outputs `number`; Math nodes output `scalar`, and `scalar` cannot feed `number`. So `Spacing = 1.5 x Size`, `Cell = 2 x thickness`, `Voxel = Size / 6` are impossible, even from two Number nodes. The `linked` row shows only `<- source_id`, not the value or port (`EngineNode.tsx:322`), and there is no cue which inputs accept a field (handle colour only: scalar green vs number grey).
Suggestion: let Math nodes return a plain number when all inputs are numbers (declare the output `number|scalar`) or add a "Number Math / Expression" node; show a field-capable badge (f) on `scalar` inputs; show the linked Number's value in the row.

### [UX] Palette search is substring-only, unranked and misses the words a designer types
Severity: medium-low · Status: VERIFIED (`frontend-ux/palette_search.py`, replicates `NodePalette.tsx:45-66`)
Evidence: the match is `title + category + type + node doc` as one substring; port docs are not searched. Zero hits: `gyroid`, `schwarz`, `stl`, `save`, `simplify`, `decimate`, `fuse`, `group`, `weight`, `perforat`, `fea`, `constraint`, `bracket`, `thicken`, `array`, `mirror`, `translate`, `rotate`, `box`, `sphere`, `cylinder`, `primitive`, `extrude`, `gradient`, `graded`, `parameter`.
No ranking: title matches are not first (query `text`: the Text node is 5th of 6, `hole`: Hole Pattern 3rd behind Boolean Subtract/Union, `number`: Number 2nd behind Face Mask). Categories appear alphabetically (Fields, Implicit, **Inputs**, Lattice, Math, ...), so Import STEP is 3rd. Multi-word queries must match as one substring ("union smooth" finds nothing).
Suggestion: token-AND search over title, aliases and port names/labels with title matches first; add an `aliases` field per node (gyroid -> TPMS Lattice, STL/save -> Export Mesh, simplify/decimate -> Reduce Mesh, perforation/vent/grille -> Hole Pattern, constraint -> Fixed Faces, array/pattern -> Hole/Surface Pattern, cut -> Boolean Subtract, hollow -> Shell, thicken/grow -> Offset Body, fillet/blend -> Smooth Union); order categories by workflow (Inputs, Fields, Patterns, Implicit, Lattice, Math, Mesh, Output, Simulation).

### [UX] Missing editor affordances that make the tool feel unfinished
Severity: low-medium · Status: VERIFIED by grep
- No undo/redo buttons (Ctrl+Z/Y exist, `App.tsx:78-83`, but no tooltip mentions them); no shortcut list. Undocumented: Ctrl+D (duplicate with inputs kept), Ctrl+Shift+V (paste keeping outer links), Ctrl+Shift+Z, F (fit, only when the 3D view has focus), arrow keys in number fields.
- No node rename (ids are `offset_2`, `number_3`; links show `<- number_3`), no comment/sticky note, no node colour. A 40-node recipe with six Number nodes is hard to read.
- No reset-to-default and no indication which values differ from the default; right-click on a node/edge gives the browser menu (only the pane opens the palette).
- Tab inside the editor opens the palette unless the target is INPUT/SELECT/TEXTAREA (`NodeEditor.tsx:276`), so Tab from a focused button (Pick, browse) cannot move focus on.
- Hint text "Click ○ on a node to show it here" (`Viewport.tsx:1126`) refers to the old icon (it is an eye now).
- The 3D legend lists node ids, not titles.

### [UX] Hazards with no warning
Severity: low-medium · Status: SUSPECTED (code)
- Export sets `compute=1` on Topology/Frame nodes (`server.py:94`; `nodes_fea.py:~464` `if compute and not done`), so pressing Export on a recipe whose optimization was never run starts the full optimization inside the export; the node doc only says "Previews show the last result".
- Toolbar "Final" (`Toolbar.tsx:261`) re-runs the whole preview at full resolution with one click and no estimate; there is no "voxel N mm / grid N M voxels" shown on the Mesh node, so a pattern or texture that vanishes at Draft (voxel x4) is not explained.
- Export with no Export Mesh node only logs "Nothing to export" (`store.ts:458`); the button stays enabled. A relative export path (`out.stl`) is logged as typed, not as the absolute file written.
- Windows "Copy as path" pastes quotes into path fields; `Path('"C:\...step"').exists()` is false and the node says "STEP file not found: ..." (strip quotes and spaces in the UI).

### [UX] Face picking flow
Severity: low-medium · Status: SUSPECTED (code)
Good: Pick button disabled with a reason when no body is linked, ghosted other bodies, Ctrl+drag paint, Shift+click tangent group, saved selections, banner with counts, `body:face` for different bodies, Esc.
Gaps: no face number/area/type shown on hover (the text field is the only place numbers appear), so "which face is 17?" can only be answered by clicking; the picker needs a drawn CAD view first ("loading ..." until a run produced it); picking toggles the source node visible and leaves it visible; the placeholder "no faces" does not tell that 0 means the first face; no way to type a range (see the face-list entry).

### [UX] Accessibility basics and layout robustness
Severity: low · Status: SUSPECTED (code/CSS)
- Colour-only signals: status dot idle/done/cached/error (`styles.css:145-149`; done #7fcf4f vs cached #5f8a4a are nearly the same), type colours (number = int, path = text, cad = faces, implicit = part share colours), scalar vs implicit hues.
- Contrast: `.placeholder` ("connect...", "Optional") #6b7078 on #2d2f34 is about 2.7:1 at 12 px italic; `.log-time` similar.
- No `aria-label` on icon buttons (eye has `title` only), `<select>`/inputs not associated with their labels, no `aria-live` for run status.
- Layout: card width 250-340 px; label column is `flex: 0 0 auto; min-width: 70px`, so long labels ("Supports & loads", "Mid-bar joints") squeeze the value box; long enum values are cut inside the select; `.enode-message` blocks are unbounded in height; node docs keep their source indentation in the `title` tooltip.
- Group ports are limited to `scalar, implicit, cad, mesh, number` (`EngineNode.tsx:348`): a group cannot pass faces, part, material, conditions or frame (so a "Static Analysis" group cannot take its Supports & loads from outside).

### [IMPROVEMENT] Basic nodes a CAD user looks for first are missing
Severity: low-medium · Status: VERIFIED (zero-hit search above)
No Move/Rotate/Mirror/Array of bodies, no primitives (box, sphere, cylinder), no Shell by faces, no Fillet on a CAD edge. `Scale Bodies` is the only Transform node. Because the catalog is the product, the palette "No matching nodes" for `box` or `rotate` is the first thing a new design hits.

### Misc low-severity notes
- Surface Pattern port docs say "hole" (shared text with Hole Pattern: `morph`, `rotation`, `corner_radius`).
- `New` recipe starts with Import STEP `path: ''` and fails with "STEP file not found: " (empty).
- `HelpPanel` and `RunStatus` subscribe to whole arrays (`nodes`), so they re-render on every store change (cosmetic).
- Print check number fields use `<input type=number>` (locale-dependent) while node cards use the custom comma-tolerant field.
- `literal values kept while linked` works within a session (`types.ts:65`, unlinking restores the typed value) but `flowToRecipe` (`graphConvert.ts:85-88`) overwrites the literal with the link when saving, so after save/reload the old literal is gone and the default returns.

### [BUG] The node catalog is fetched once: after an engine restart an open tab keeps showing the old nodes
Severity: medium-low (this is the normal dev loop: edit a node, restart `app.py`, keep the tab) · Status: SUSPECTED (code)
Evidence: `App.tsx:59-60` calls `api.catalog()` once at startup; `api.ts:183-187` on WebSocket (re)connect only sets `connected` and refreshes the queue; the server sends no version or hello (`server.py:586-590` `/api/catalog`, `server.py:786-788` `/ws` accepts and streams events).
After a restart that renamed/added/removed an input or changed `choices`, cards still show the old inputs, the recipe POSTs old keys (engine: "implicit.offset has no input 'x'" on the card) and new inputs are invisible until a manual page reload. Combined with the unknown-enum entry above, a renamed choice shows the first option while the engine rejects the stored one.
Suggestion: put a catalog hash in `/api/catalog` and in a `hello` WebSocket event; on reconnect compare and either refetch the catalog (`setCatalog` + rebuild `specs`) or show a "Engine was updated: reload" banner.

### [UX] Field slices are coloured per plane, and the section/slice position cannot be typed
Severity: low · Status: SUSPECTED (code)
Evidence: `server.py:487-489` computes `min/max` of each slice image; `SliceView.tsx:47` normalises that slice to its own min..max. Moving the plane slider, or looking at two fields in turn, re-stretches the colours every time; a Noise whose true range is -1..1 but whose slice spans -0.6..0.7 is drawn with full contrast, so a "weak" region looks as strong as a strong one. The declared `Field.range` (shown as the range tag on the card) is not used for the colour scale. Also, the slice and Section sliders (`SliceView.tsx:178`, `Viewport.tsx:359`) have no numeric box, so "cut at 12.5 mm" is a drag; only one field can be shown at a time, silently (`store.ts:372-378` replaces the previous one without a message).
Suggestion: option "fixed range" (use the card's range, or the min/max over the whole box) and a number input next to each slider.

### [UX] Names with non-ASCII letters are refused with a wrong message
Severity: low (Swedish user) · Status: SUSPECTED (code)
Evidence: group and selection names are checked with JS `/^[A-Za-z_][\w-]*$/` (`store.ts:594,784`, `EngineNode.tsx:121`); JS `\w` is ASCII-only, so "övre" or "fäste" fails with "A name starts with a letter and has no spaces" although it does start with a letter. The engine's `_SELECTION_NAME` (`core.py:304`) has the same ASCII start class, so allowing them in the UI alone would not work.
Suggestion: say "letters a-z, digits, _ and - only" in the message (or accept Unicode letters in both places).

## What worked well (keep)
Every input has a doc; scrub-to-edit labels, comma decimals, range tag from interval arithmetic, Remap Auto, node `note` lines for patterns/FEA, per-node error blocks with group-inner prefixes, undo grouping of rapid edits, autosave recovery, drag-to-palette with type filter, copy/paste across tabs, face-picking banner, export queue, memory meter.

## Not tested / limits
Editor never run (no live layout measurements, no real click-through of drag-to-create or log scrolling); heights are computed from CSS; worker-pool message loss, Export-triggers-optimization and the Final-quality guard are read from code only; the unknown-enum display is verified from React's source, not in a browser; `tsc` was not run.
