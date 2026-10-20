# ui-live audit: the real editor, driven in the browser pane (test editor on port 8766)

Method: Claude built-in browser pane at 1000-1260 px wide viewports (and the pane's own 1171 px),
Draft/Medium preview, small samples only (test_part.step, vent_case.step). Screenshots are in the
tool-results folder of this session (paths quoted per finding where useful). Running log; the newest
findings are at the bottom. Harness quirks that are NOT editor bugs: node drags move half the
dragged distance, modifier-clicks do not reach React Flow (Ctrl was injected through a keydown
event to multi-select), native prompt()/confirm() were stubbed in the page for Group.

## What worked as intended (so it is not re-tested)
- Palette (right-click / + Node / drag from a port to empty canvas = filtered palette that auto-connects the new node), preset insertion, copy/paste (Ctrl+C/V, "Pasted 1 node"), duplicate, delete node and delete edge (the input falls back to its typed number), undo/redo of number edits (typing merges into one step), of node deletion with its links, Ctrl+S, Save as dialog (typed folder path, filename), Open dialog for STEP files.
- Number fields accept "0,35" (comma) and apply immediately; ranges tags on cards ("0 ... 1.5"); progress text in the toolbar ("Updating: Mesh from Implicit Body 0.3 s") and Cancel button; typed error "voxel size must be greater than 0" appears on the right node and in the log.
- Face picking on the CAD view (blue highlight, face number written into the node), Face Mask shown as slice and as colours on bodies, section plane (X/Y/Z slider, Flip), print Check panel (size, volume, weight, bodies, cavities, wall thickness colouring), wireframe, Draft/Medium switch (re-mesh 0.2 s on the small part), group create / open / add a Number node (becomes a group setting) / Done.
- Time to first preview after typing a STEP path in a fresh graph: under 1 s (0.6 s "Done") at Draft.

## Findings (newest at the bottom)

### [BUG] Recovery banner: survives New / other recipes, cannot be dismissed, Discard deletes the autosave with no confirmation
Severity: medium · Status: VERIFIED (behaviour) + code
Evidence: frontend/src/components/Toolbar.tsx:18-39 (offer set only in openRecipe), :92-110 (RecoveryBanner: only Restore / Discard), :113-ish `loadRecipe(STARTER, null)` for New does not touch `useRecovery`. In the live editor the banner "Unsaved changes to hex_studs from 21:27 were kept" stayed on screen after I clicked New and kept offering a recipe that was no longer open. It sits at top centre over the viewport toolbar (Fit / Wire buttons) and over the header of the node help panel, so the help panel's close button (X) could not be clicked (the click landed on the banner). Discard (line 107) calls `api.dropAutosave(offer.path)` immediately: the only copy of the unsaved work is deleted irrevocably; Restore (line 103) replaces the open graph without checking `dirty`.
Why it matters to the user: it is exactly the kind of overlay that gets clicked away; "Discard" is a one-click, no-undo delete of days of unsaved work, and the stale offer can overwrite a different graph the user is now working on.
Suggestion: show the banner only while the offered recipe is the open one (clear on New/Open), add a neutral "Later"/X that keeps the autosave, confirm Discard (or move the autosave to a `.bak` instead of deleting), and make Restore ask when the open graph is dirty. Place it in the flow (e.g. top of the node pane) so it does not cover the help panel / viewport buttons.

### [BUG] Autosave of an untitled recipe is written but never offered back (slot is also shared and silently overwritten)
Severity: medium · Status: VERIFIED by file evidence + code
Evidence: server.py:557-560 `_autosave_file(None)` uses key "untitled" (one slot for every new recipe). The slot file 9f273e321a1d1ea4.json (path null, dated Oct 14) existed at startup but only the hex_studs autosave was offered: `openRecipe(path)` (Toolbar.tsx:23-40) is the only reader of `api.autosave` and it is never called with null, so the "a new recipe" branch of RecoveryBanner (line 95) is dead code. Editing any new graph rewrites the slot after 2 s (useAutosave, Toolbar.tsx:75-90), so an earlier unsaved untitled design is overwritten without a word; Save-as from untitled deletes the slot (Toolbar.tsx:58 `dropAutosave(before)` with before = null).
Why it matters to the user: a crash while designing in a new graph can never be recovered, and the slot the user thinks is a safety net is not.
Suggestion: on startup also read the untitled slot and offer it; key untitled autosaves by a per-session id (or timestamp) instead of one shared slot.

### [BUG] Node editor "Fit view" control is hidden under the log panel
Severity: low-medium · Status: VERIFIED
Evidence: `document.elementFromPoint` at the centre of `.react-flow__controls-fitview` returns the log line (`log-time`); fit button rect (15,847,26,26) vs log panel (12,846,147x30). The log panel (LogPanel.tsx, pane bottom-left) sits on top of React Flow's Controls whenever the log has a line. Zoom +/- are visible, the third button is not; after a mis-pan there is no mouse way to bring the graph back (the toolbar "Fit" is the 3D view's).
Suggestion: move the log panel to the right of the Controls (left: 60px) or put the Controls above it; bind F / a toolbar button to fit the node graph.

### [UX] Toolbar status ("Done in 0.5s" / "Failed") is clipped to one letter below ~1350 px; toolbar overflows below ~1240 px
Severity: low-medium · Status: VERIFIED
Evidence: at 1260 px `.run-status` is 10 px wide with `overflow:hidden` (scrollWidth 33): the toolbar shows "D" or "F". At the pane's own 1171 px nothing of the status or the memory text is visible; at 1000 px the Update / Export buttons fall off the right edge (`.toolbar` scrollWidth 1238 at 1120 px).
Why it matters: "did my edit fail?" is the most important answer on screen; on a laptop-width window it is invisible.
Suggestion: let `.run-status` shrink last (flex-shrink 0 / min-width), hide the memory meter text first, wrap or collapse the Quality/GPU/Auto group into an overflow menu.

### [UX] Palette search ranks matches in the description text above the node's own title, so Enter adds the wrong node
Severity: medium · Status: VERIFIED
Evidence: typing "number" highlights Face Mask first (its help text says "face number"), Number is second; typing "honeycomb" lists Surface Pattern first (doc mentions a raised honeycomb), then the preset "Hex stud texture", and the exact-title preset "Honeycomb vents" third. Typing "pick" does select Pick Body (title match + doc match).
Why it matters: the palette is used for every node; Enter on an exact name must give that node.
Suggestion: rank exact / prefix title matches first, then title substring, then description (NodePalette.tsx scoring).

### [UX] Invalid text in a number field stays in the field with no marker while the engine keeps the old value; intermediate keystrokes are committed
Severity: low-medium · Status: VERIFIED (invalid text), code (intermediate values)
Evidence: typing "abc" after "0,15" left "0,15abc" in the Voxel field after blur; the value in the graph stayed 0.15 (EngineNode.tsx:23-48 `commit` ignores non-numbers and nothing resets or flags the text). Each keystroke that parses is committed: "-0,1" commits 0 / -0 first, "0,15" commits 0 then 0.1; with Auto on this starts runs with Voxel 0 and flashes "voxel size must be greater than 0" on the way. Negative voxel: clear error on the right node.
Suggestion: on blur reset the text to the last valid value (or mark the field red); commit on Enter/blur, or only after the 450 ms debounce, instead of per keystroke.

### [UX] After a failed run the 3D view keeps showing the previous mesh with no "stale" marker
Severity: low-medium · Status: VERIFIED
Evidence: Voxel -0.1 -> node red, log "node 'mesh' failed: voxel size must be greater than 0", toolbar "F..."; the viewport still shows the old box exactly as if it were current. Same when the graph becomes invalid ("invalid graph: pattern: required input 'cad' is not set / cut: required input 'a' is not set" in the log; mesh unchanged).
Suggestion: dim the view or add a "out of date" chip in the viewport legend while the last run failed.

### [UX] A Number node created by dragging from an input starts at 0 instead of the value it replaces, which silently zeroes the effect
Severity: low-medium · Status: VERIFIED
Evidence: Remap "To High" was 1.5; dragging from that input to empty canvas and choosing Number connected a Number node with value 0, and the texture amplitude became 0 (smooth surface in the preview) until I typed 1,5 again. (NodeEditor.tsx addFromPalette connects but does not seed the value.)
Suggestion: when the new node is a Number (or a constant-like node) created from an input, seed it with that input's current value.

### [UX] Groups: only Number nodes become settings, named "number_1"; an inner value cannot be promoted
Severity: low · Status: VERIFIED
Evidence: after Group (Noise + Remap -> "noise_wave") the node has no settings at all; the amplitude (Remap To High) is only reachable by opening the group, adding a Number and wiring it; the new setting is labelled `number_1` on the group card (the Number's id) and cannot be renamed there.
Suggestion: right-click an input in group edit mode -> "Expose as setting" with a name; show the label instead of the id.

### [UX] The preview voxel is never shown, so features below it vanish without a hint
Severity: medium · Status: VERIFIED
Evidence: Noise amplitude 0.35 mm on a part with Mesh Voxel 0.2 at Draft (x4 = 0.8 mm effective) gives a perfectly smooth top face; raising To High to 1.5 or switching to Medium shows the texture. Nothing on the Mesh card or in the toolbar says "preview voxel 0.8 mm" (the log only warns for patterns: "mesh with a voxel of at most ...").
Suggestion: show the effective voxel on the Mesh from Implicit Body card and in the toolbar quality tooltip, and warn when an Offset Body's range (amplitude) is smaller than 2 voxels.

### [UX] Starting to pick faces turns the body's view on for good and marks the recipe dirty; the CAD view then z-fights with the mesh
Severity: low-medium · Status: VERIFIED
Evidence: store.ts:240-243 `setPicking` calls `toggleVisible(source)` (dirty: true, line 381); after Done the Pick Body eye stays on. In the viewport the CAD body and the offset mesh share the same surface where the mask is 0 and show a black/yellow speckle pattern ("pick_body_1" legend entry) until the eye is turned off by hand.
Suggestion: restore the previous visibility when picking ends and do not set dirty for it.

### [UX] Error text uses internal port names and a new graph starts in the "Failed" state
Severity: low · Status: VERIFIED
Evidence: Face Mask / Hole Pattern: "required input 'cad' is not set" for the port the card calls "Body". New recipe: Import STEP is red with "STEP file not found: " (empty path) and the toolbar says Failed before the user did anything.
Suggestion: use port labels in messages; say "Choose a STEP file" when the path is empty; do not run the starter graph until it is complete.

### [UX] Several panels cover the working area
Severity: low · Status: VERIFIED
Evidence: the minimap (bottom right of the node pane) covers nodes placed there (my Multiply node was hidden under it, and a drag aimed at the node panned the graph through the minimap instead); the selected node's help panel (top centre) covers the neighbouring nodes at narrow widths; palette-created nodes and inserted presets land on top of existing nodes (the Honeycomb vents preset landed over Mesh from Implicit Body).
Suggestion: place new nodes at a free spot (right of the selection / of the last node), let the minimap be collapsed.

### [UX] The log of the previous recipe stays in the panel after New / Open
Severity: low · Status: VERIFIED
Evidence: after New the log still showed the hex_studs "Surface Pattern: faces 8, 9, 10 ..." lines; the first line of the new graph is mixed into it.
Suggestion: clear (or separate with a header) on recipe load.

### [UX] Save dialog: no overwrite warning; new recipes default into the repo's recipes folder as new_recipe.json
Severity: medium · Status: SUSPECTED (code; not exercised on a user file)
Evidence: FileBrowser.tsx `finish()` and the double-click handler in save mode close with the existing file's path with no confirmation; server.py `write_file` overwrites. Toolbar.tsx:55 default `recipes\\new_recipe.json`.
Suggestion: confirm "Replace X?" when the target exists.

### [UX] 3D camera does not refit when the shown body changes size
Severity: low · Status: VERIFIED
Evidence: after opening vent_case.step in a graph that previously showed the 60 mm test part, the cup was cut off by the viewport until Fit was pressed.

### [UX] Print check readout is hard to interpret on a solid part
Severity: low · Status: VERIFIED
Evidence: Check on the textured block: "Thinnest wall 0.40 mm (1%: 12.04)" and "Typical wall 16.36 mm" with black patches on one end face; the "(1%: 12.04)" part is unexplained, and for a solid block 0.40 mm (rays grazing the rim) reads like a problem. The panel also shows decimal commas ("1,24 g/cm3", "0,8") while the node cards use dots (locale formatting).
Suggestion: explain the percentile, ignore edge-grazing hits, use one decimal separator.

### [BUG] The first edit after opening a recipe overwrites the pending crash-recovery copy on disk (the banner still says Restore)
Severity: medium · Status: VERIFIED (probe recipe in SCRATCH, simulated crash autosave)
Evidence: I wrote an autosave slot for SCRATCH/.../probe_a.json (Remap To High 0.9, Voxel 0.2, "time" one hour old), loaded the editor: banner "Unsaved changes to probe_a from 03:06 were kept". I typed Voxel 0,25 without touching the banner; 2 s later (`useAutosave`, Toolbar.tsx:75-90) the slot file on disk held Voxel 0.25 and the Remap link (the 0.9 copy was gone; mtime 04:06:41). The banner stayed on screen, still offering Restore. Restore still worked, but only because the offer lives in page memory: after a reload or crash the earlier unsaved work would be unrecoverable. Restore then replaced the graph without a warning (Voxel 0.25 -> 0.2, i.e. my newer edit was thrown away) and `loadRecipe` clears the undo history (store.ts:257 `past: []`), so Ctrl+Z cannot bring the newer edit back.
Why it matters to the user: the recovery banner looks like a safe choice that can be made later; one tweak of a number destroys it.
Suggestion: keep the recovered copy in a separate `.recovered` file until the banner is answered (autosave writes elsewhere), and make Restore push an undo snapshot.

### [FINDING] Which action deleted the autosave c78f2276fec46548.json (hex_studs), and how a normal action can delete an unrelated autosave
Status: analysis from code + the live session; the exact click that deleted c78f... cannot be proven.
- The only code that deletes an autosave file is the server's `DELETE /api/autosave` (server.py:641-644), called from exactly three places in the editor: (1) the banner's Discard button (Toolbar.tsx:107, deletes `offer.path`), (2) `confirmDiscard()` (Toolbar.tsx:66-72, deletes the slot of the recipe that is OPEN, after the native "Discard unsaved changes?" confirm, used by New / Open / the recipe dropdown when the graph is dirty), (3) `saveRecipe()` (Toolbar.tsx:57-58: the saved path's slot AND the slot of the path before Save as, which is the shared untitled slot when the graph was new).
- My recorded actions never invoked (1)-(3) for hex_studs: hex_studs was never dirty (no dot on the toolbar when I pressed New, so `confirmDiscard` returned true without deleting), never saved (recipes/hex_studs.json mtime stays 2026-10-19 20:29), and I never pressed the banner's Discard (my only click near it was the hidden help-panel close button, which landed on the banner text). I checked in this harness that a native confirm() is auto-cancelled (New while dirty did nothing and deleted nothing), so (2) cannot have been triggered either.
- In the live page the banner disappeared (and the viewport scale changed because the pane was resized) between two of my screenshots while I was clicking elsewhere; the most likely explanation is a human click on Discard in the banner, or in the user's own editor on port 8765 (same autosave folder, same banner at startup, port not listening when I checked at 04:03). Whoever it was, the banner makes it easy: the offer for hex_studs stayed visible after New, sits at the top centre of the work area, and Discard is a single click with no confirmation.
- A normal action that silently deletes an UNRELATED unsaved design: Save as (or Ctrl+S) of ANY new graph deletes the single shared "untitled" slot (`dropAutosave(before)` with before = null). I did exactly that at 23:35 when I saved my test graph: the user's Oct 14 untitled autosave (9f273e321a1d1ea4.json) was deleted and later re-created by my next New graph. Two editors (or two browser tabs) on the same recipe share one slot too: Save in one deletes the other's recovery copy, and each overwrites the other's every 2 s.
Suggestion: key the untitled slot per session (e.g. `untitled-<timestamp>`), never delete other slots on Save as (only the slot of the recipe that was saved), and confirm Discard.

### [UX] Static Analysis: the numbers live only in the log, the card's range tag disagrees with the legend, and surfaces z-fight
Severity: medium · Status: VERIFIED (bracket.step, Fixed Faces on face 10, 100 N Load on face 1, detail draft, 1.6 s)
Evidence: result log lines (expanded log only): "max stress 17.5 MPa at (10.2, 20.0, 6.7) · smallest safety factor 2.9 (PLA, strength 50 MPa) · max deflection 1.85 mm · mass 34.9 g" and "note: 22.1 MPa right at the fixed faces ...". The card shows no text result, its range tag reads "0 ... 22.1" (no unit) while the viewport legend says "0.00 MPa ... 17.54 MPa"; the collapsed one-line log shows only the last line ("Update finished in 1.6s"). With the picked body's view (Import STEP eye, left on by picking), the Mesh view and the analysis view all on, the three coincident surfaces z-fight into a blue/white speckle that looks like noisy stress.
Why it matters: the first thing a designer wants from a static analysis is "max stress / safety factor / deflection" at a glance; the speckle looks like a solver artefact.
Suggestion: set the summary line as the node's `note` (the card already shows `note` for topology), put the unit in the range tag and use the same (peak-free) range as the legend, hide the CAD/mesh views while a result view is shown, or offset them slightly.

### [UX] Presets are pasted on top of existing nodes and on top of each other
Severity: low-medium · Status: VERIFIED
Evidence: "Static analysis" preset: Fixed Faces, Load, Material and Static Analysis overlap each other and cover Implicit Body / Mesh / Export of the open graph (screenshot, 4 selected nodes); "Honeycomb vents" preset landed on Mesh from Implicit Body. Each preset node has to be dragged apart before a port can be reached (the import handle was hidden under the pattern node).
Suggestion: place a pasted clip at a free spot right of the existing graph (bounding box of the graph + gap), keep the preset's own relative layout with a minimum node spacing.

### [UX] Inputs that do not apply are not greyed out (only Remap's Auto does it)
Severity: low-medium · Status: VERIFIED
Evidence: Boolean Subtract with Edge = sharp still shows an editable Radius (0.5); Load shows Vector while Direction = "into the face"; Hole Pattern (27 rows, ~340 px tall at zoom 1) shows Sides with shape hexagon, Length and Web, Morph with Morph To = none, Tilt, Depth/Height... EngineNode.tsx:17-18 has the one special case `ignoredByAuto`.
Suggestion: a generic `applies_when` on Port (shape, edge, direction, morph_to ...) to grey or hide the rows, which keeps cards with many inputs readable.

### [UX] The one-line log keeps showing an old red error after the problem is fixed
Severity: low · Status: VERIFIED
Evidence: after "spacing must be greater than 0" was fixed (successful run, 1.1 s) the collapsed log still showed the red "node 'pattern' failed: spacing must be greater than 0" because runs under 1 s add no "Update finished" line (store.ts:459 `if (e.time > 1)`); same with "STEP file not found" under a working bracket graph.
Suggestion: always add a short success line, or clear errors of nodes that now run fine.

### [UX] Pick buttons can sit under the 3D view
Severity: low · Status: VERIFIED
Evidence: with the usual split the Load node's Pick / clear / saved-selections buttons are clipped by the 3D pane (the node is wider than the editor); a click on the button's reported position hit the Faces text field instead. Nodes wider than the pane need a pan first.

### [BUG] A kept variant survives "New" and is shown next to the new graph ("Variant A, 7 changes")
Severity: low-medium · Status: VERIFIED
Evidence: after Keep variant on the two_blocks graph and then New (and a different STEP + a preset), the viewport still listed "Variant A  3 changes" (later 7, 8 changes) and drew the two-block mesh to the left of the new part; store.ts:255 `variants: path === s.recipePath ? s.variants : []` keeps them when both paths are null (every untitled graph). The variant row's X sat under the face-picking panel, so it could not be removed while picking.
Suggestion: clear variants whenever `path` is null or a New is made (compare a recipe id, not the path).

### [UX] Nothing says the preview is out of date (Auto off, or after a failed run)
Severity: low-medium · Status: VERIFIED
Evidence: Auto unticked, Pick Body Index edited: the toolbar still reads "Done in 0.2 s", Update is not highlighted, the old mesh stays; unticking back runs the pending edit at once (57k tris, 0.2 s), which is good. RunStatus (Toolbar.tsx:152-172) never compares the graph version with the run that produced the view.
Suggestion: show "Out of date: press Update" (and highlight the button) when `graphVersion` is newer than the shown run; same chip after a failed run.

### [UX] Export gives no confirmation of what was written; the default path is inside the repo
Severity: low-medium · Status: VERIFIED
Evidence: Export (Mesh voxel 0.6 on the gyroid-filled test part) ran 1.2 s, wrote a 17.35 MB STL to the typed SCRATCH path and logged only "Export finished in 1.2s" (tab title gets a check mark). No path, size, triangle count or watertight statement; the Export Mesh node's default path is `samples/out.stl` (relative to the repo root) so an unedited Export writes into the repo's samples folder.
Suggestion: log "Wrote <full path> (N triangles, X MB, watertight)" with the path clickable; default to a per-recipe output folder or ask on first export.

### [UX] Error text names the internal input ("space", "conditions", "cad") instead of the label on the card
Severity: low · Status: VERIFIED (more cases)
Evidence: Topology Optimization card: "required input 'space' is not set" for "Design space", "required input 'conditions' is not set" for "Supports & loads"; Hole Pattern / Face Mask: 'cad' for "Body". (Same finding as above, now with three node types.)

### [UX] Smaller things seen
- Pressing Escape does not close the node palette once focus left the search box (after clicking a preset's delete X); the next click outside is used up closing it (my first New click did nothing).
- Typing a number and scrubbing the label both work well (scrub 60 px took 2 -> 2.29, one run); only the label has the scrub hint (tooltip), nothing on the field.
- Multi-select needs Ctrl (React Flow default); Shift+click does nothing (SUSPECTED for a real keyboard: my harness drops modifiers, I injected a Control keydown). Users from SolidWorks expect Shift+click and Shift+drag box select; add `multiSelectionKeyCode={['Control','Meta','Shift']}`.
- Dragging a connection from an already linked input to empty canvas opens the palette (filtered) and the new node's output replaces the old link only after you choose a node; cancelling leaves the old link (good).
- Static analysis / Hole Pattern previews on the small samples were fast: Hole Pattern 216 holes + Boolean Subtract at Medium 1.1-1.4 s; Static Analysis draft (42k elements, GPU) 1.6 s; Surface Pattern studs 0.1-0.2 s.

## Screenshots (SCRATCH/audit/ui-live/screenshots/)
01 stale recovery banner over the help panel and viewport buttons right after startup; 02 / 03 palette search ranking ("number", "honeycomb"); 04 node pane with Controls and log panel at the bottom left; 05 print Check panel; 06 / 07 / 11 presets pasted on top of existing nodes; 08 Static Analysis with the CAD, mesh and result views z-fighting; 09 stale variant after New; 10 toolbar status clipped to one letter.

## What I could not test or only partly tested
- Frame Optimization / Join Frames / Topology Optimization runs (only that the cards and the Optimize button appear; no optimization was started, memory).
- Final-quality export of a big part, block meshing and the lamp recipes (memory safety); Final was used only with Mesh voxel 0.6 on the 60 mm test part.
- The GPU toggle (left on; grids stayed small), the real native confirm()/prompt() dialogs (stubbed in the page for Group / Save as preset / Delete preset; in this harness a native confirm is auto-cancelled).
- Modifier clicks (Shift/Ctrl) and the real mouse wheel are not delivered reliably by the harness; multi-select was done with an injected Control keydown, node drags moved half the dragged distance (I used a small JS helper to place nodes). Everything about the graph itself (links, values, runs) was done through real clicks and typing.
- Picture / Text nodes, Merge Meshes, Preview Image, Smooth Union, Shell, Smoothen, the Voronoi / Strut lattices (other audit areas cover them from code).
- The user's own recipes were never opened (all work started from New or from files in SCRATCH); the autosave folder was restored from the backup at the end (hashes verified, see below).

## Cleanup record
- Test editor on 8766 stopped with preview_stop; its tab closed.
- Autosave folder restored from autosave-backup/ (the three originals, byte-identical, test slots removed).
- presets/: one test preset (zz_ui_audit_test) was created and deleted again through the palette's delete button; 25 files as before. `git status` clean.
- Files left in SCRATCH/audit/ui-live/: ui_graph1.json, probe_a.json, probe_setup.py, screenshots/, the autosave backups.
