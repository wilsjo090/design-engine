# UI audit (2026-10-26)

The question: does the editor work as expected, and what can be simpler? Two concrete pains: a
surface texture took Surface Pattern → Multiply (height) → Multiply (Face Mask) → Offset Body, and
topology optimization took ~10 clicks in three different menus with no prompt for Keep %, material or
detail.

Tried in the test editor (port 8766) as a user would: opening recipes, right-click menus on faces and
bodies, Fix this face / Add load / Analyse ▸ Topology optimization / Optimize, Texture ▸ on a face,
picking faces by clicking, the quick-edit box, undo, a recipe with two bodies.

## Built

- **Texture node** (`implicit.texture`, Implicit): Body, Pattern, Height (mm), Fade (0..1), Invert fade,
  and Pattern range under "more". A pattern's own range is stretched to 0..1 (Noise -1..1 → 0..Height),
  so Remap is not needed either. A texture is now *Pattern → Texture*; with a fade, *Pattern → Texture ←
  Face Mask*. The 15 texture presets use it (2 nodes instead of 4; Labyrinth 6 instead of 10), and so
  does the 3D view's *Texture ▸* menu (3 nodes instead of 5 for noise). Old recipes are unchanged:
  Multiply and Offset Body still work. The user's `kakformsgrej.json` would go from 8 nodes to 5:
  surface_1 → Texture (Fade ← face_mask_1, Invert fade yes, no Remap), surface_2 → Texture (Height 0.5).
- ***Texture ▸ Shapes*** in the face menu: round studs, hex studs, dimples, tiles, bricks, knurl,
  Voronoi web, scattered dots, low-poly facets (the presets' settings), each one click.
- **Shorter cards.** `Port.when` hides inputs that do not apply (Sides only for polygons and stars,
  Length for long shapes, Web for Voronoi / Delaunay, Morph only with a Morph To, a Load's Vector only for
  a custom direction, Face Mask's Position and Band only for surface reach, Start coarse only at fine
  detail); `Port.advanced` puts rarely changed inputs under "▾ N more settings" (shown anyway once changed
  or linked). Surface Pattern with round shapes: 12 of 28 inputs (497 px instead of ~970); Topology
  Optimization 12 + 8 under "more"; Frame Optimization with round bars 12 of 24 (Height / width and Wall
  only for the profiles that have them, Fillet and its continuity only for fillet and ball joints, also in
  Join Frames); TPMS Thickness only in sheet mode and Offset only in the solid ones; the lattices' Units
  and the foam's Seed under "more". Static Analysis has 5 inputs that always apply: nothing to hide, but its
  card now shows safety factor, deflection, stress and mass (they were only in the log) and warns below a
  safety factor of 1. The quick-edit box follows the same rules.
- **Analysis wizard** (the 3D view's *Analyse* button, or right-click a body ▸ *Analyse ▸ Wizard ▸*):
  Find (topology optimization, frame optimization or static analysis) and Part → Held (click faces) →
  Loads (click faces, kind, amount, direction, + Load) → Settings (material plus the goal's few: Keep % and
  min thickness; profile, min size, safety factor, joints and curving; detail) → Optimize or Analyse
  (progress, Stop, the result's note and warnings, Colour by, Mesh and export…). It makes the normal nodes
  (one Undo takes them away), so everything stays editable in the graph. Checked on the bolted wall
  bracket: topology 117 iterations, safety factor 2.02, 0.38 mm.

## Clearer settings (2026-10-27)

The question: can the settings be more intuitive? Going through all 232 settings of the node library:
no number showed its unit (it was only in the hover text), two dozen inputs used 0 for "automatic", "no limit",
"sharp" or "through" without saying so, on/off was a yes/no list, points were typed as "0, 0, 0", and
choices like a hole's shape were a list of words. Now, on cards, in the quick-edit box and in the wizard
alike (`components/PortInputs.tsx`):

- **Units in the field**: 4 mm, 15 °, 100 N (MPa for a pressure, N·m for a torque: the Amount's unit
  follows Kind), 30 %, 1.24 g/cm³, × for factors; a lattice's thickness says mm or solid (density units).
- **What 0 means**, shown in place of 0: Min thickness *auto*, Max thickness *no limit*, Depth *through*,
  Corner radius *sharp*, Mesh Tolerance *off*... Empty the field to go back to it.
- **Sliders** under 0..1 values and percentages (Randomness, Morph, Fade, Roughness, Curving, Keep,
  Poisson's ratio, Keep triangles): drag them, or type beyond them.
- **Switches** for yes/no and on/off (Invert, Fuse, Start coarse, Move joints...).
- **Buttons side by side** for a few short choices: x / y / z, sharp / round / chamfer, draft / medium /
  fine, fillet / plain / ball, inside / center / outside.
- **Pictures** for a pattern's Shape and Morph To (14 shapes), Layout (grid, staggered, rings, spiral) and a
  frame's bar Profile (8 cross-sections).
- **X / Y / Z fields** for Centre, Size, Start, End, a Load's Vector and Offset.
- Clearer names: Smooth Step's Edge0 / Edge1 are *0 at* / *1 at*; Voxel size, Keep triangles, Warp size,
  Max angle (rad); a Boolean's Radius only shows for round and chamfer edges.

## Easier and faster: the first three (2026-10-27)

The question: what else would make the editor easier or faster to use? Done first:

- **Move, Rotate, Mirror and Pattern Bodies** (Transform, next to Scale Bodies): on CAD bodies, before they
  become implicit bodies. Faces keep their numbers, so Face Masks, patterns, supports and loads picked on the
  original apply to the moved body and to every copy. Rotate turns about the bodies' middle (a list as one),
  the origin or a point; Mirror about YZ / XZ / XY at a position, or a flat face (Keep original: yes gives
  both); Pattern in a row, rows (more settings), or round an axis (360° = evenly all round). In the 3D view:
  right-click a CAD body ▸ *Transform ▸ Move… / Rotate… / Mirror… / Pattern…* (the node goes in after the
  body, what used the body uses the result, the quick-edit box opens), or a flat face ▸ *Mirror about this
  face* (half a part made whole).
- **Math gives numbers**: Add, Multiply, Remap... with only numbers in give a number out, which can set any
  number input: Number → Multiply (x 1.5) → Spacing. A wire from Math that depends on a field into a number
  input is refused on the card, and a field reaching one later (a Noise wired into the Math) is named.
- **An engine restart no longer leaves the open tab deaf**: the tab used to drop every new result (the new
  engine numbered them from 1 again) until it was reloaded; it now sees the restart, makes everything again
  and reads the node library again.
- **The recovery banner keeps what it offers**: the first edit after opening a recipe overwrote the kept
  changes 2 s later, while the banner still offered them. Now nothing is written over them until Restore or
  Discard (the banner says so once you edit); Discard asks first; New offers a new recipe that was never
  saved.

## Speed fixes (2026-10-27)

- **A voxel size from a Number or Math is coarsened in Draft and Medium** like a typed one (Mesh from
  Implicit Body's hidden `coarser`): before, such a Draft meshed at the export's full detail. On the test
  part, Number 0.15 x 2: Draft at 1.2 mm, Medium 0.6 mm, Final 0.3 mm.
- **Reduce Mesh waits for Final**: Draft and Medium show the mesh unreduced (it looks the same); Refine then
  reduces it at Final (test part: 46k → 13k triangles, 0.2 s).
- **Exports make no views**: they meshed every shown implicit body again at 512 cells after writing the
  files. The 3D view keeps what it showed.
- **The preview voxel is on the legend**: "Draft · voxel 1.2 mm · 12k tris", and its hover text says what
  that hides (walls, bumps and gaps under ~2 voxels look thinner or vanish; Final or zooming in shows them).
  A Reduce Mesh's Final view shows no voxel (its triangles are no longer the grid's).

## Fixed on the way

- An edit that also shows a node (every Texture, Modify, the wizard's Pick Body) lost the Undo button on
  its toast at once: showing bumped the graph version. `applyEdit` now re-stamps the toast.
- *Again: …* in the face menu survived opening another recipe.
- The wizard kept the previous recipe's part after Open and waited forever for its faces (found while
  testing it).

## Found, not fixed

1. The log keeps the previous recipe's last message after Open/New (NODE_AUDIT P3 "Layout", still so).
2. On a narrow graph pane the minimap covers a third of the graph and the log box sits on the zoom
   buttons; the pick banner (max 70% wide) turns into a tall narrow column when the 3D view is narrow.
3. Shape textures are not scaled to the face like noise is: on a narrow face a preset's Edge margin and
   Spacing leave 0 shapes (the card says so). The menu's hex studs now use a 1 mm margin.
4. `recipes/kakformsgrej.json` and `recipes/testlamp2.json` have a Mesh node without a body, so
   `tests/test_graph.py::test_recipes_are_valid` fails.
5. The wizard's progress lives in memory: after a reload it starts at Part (its nodes stay in the graph).
   Frame Optimization takes one load case only, so the wizard's loads all act together.
6. ~~NODE_AUDIT P1 4 and 5~~: fixed 2026-10-27 (above).

## Next simplifications worth doing

- Load cases (+ Case) and Keep / Avoid regions in the wizard; checking a topology or frame result with
  the static analysis goal in one click.
- From the 2026-10-27 list, not done yet: node search with CAD words (gyroid, stl, simplify, fuse, rotate,
  hollow); a wire dropped on a new node going to the right input, and a Number dragged out of an input
  starting at its value; cards saying what came out (body count, "x150" on lists, a mesh's triangles and
  volume); Shell inside / centred / outside; shape textures scaled to the face; Undo / Redo buttons and a
  shortcut list. Faster: the engine work in FINDINGS ("Performance": the CPU sign test and noise as kernels,
  the VTK thumbnail) and warm workers after opening; a warning when a wall is thinner than ~2 preview voxels
  (the legend's hover text only says so).
- Transforms of implicit bodies: a move is a shifted grid (exact in blocks), but a turn needs points off
  the axis-aligned grid; for now transforms act on CAD bodies.
