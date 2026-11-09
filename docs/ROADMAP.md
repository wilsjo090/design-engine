# Roadmap

Draft, 2026-11-08. A list of ideas to choose from, not a promise. Items that already have a plan
keep it in their own doc (see the end); this file says what to build next and why.

Sizes are rough: **S** about a day, **M** a few days, **L** a week or more.

## 1. The user's list

_To fill in: features and annoyances from using the editor._

- [Offset Faces](#offset-faces): pick a few faces and move them out or in as one piece (M-L).
- [GPU and CPU together](#gpu-and-cpu): split the work so neither waits for the other (M).
- [Implicit body viewer](#viewer): faster to show, sharper when shown (M).

### <a id="offset-faces"></a>Offset Faces

Pick faces and move them out (or in) by a distance, as one piece. A flat face stays flat, chosen
faces that meet stay joined, and the faces around them stretch or shrink to meet the moved faces, like
Move Face or Offset Surface in SolidWorks. Uses: raise a panel or a label area, thicken a lip or a
mounting face, recess a face for an inlay, thin one wall.

**What exists today and why it is not enough.** A Texture with a plain number as pattern and a Face
Mask as Fade offsets the chosen faces, but only with a soft ramp at the selection's edge (the Fade
width). At Fade 0 the step follows wherever the nearest face changes, which outside an edge is not the
side face's own plane: a box's top face raised that way does not get its vertical sides continued up
(not tried yet; expected from how the mask picks its face).

**Settings (proposal):**

| Setting | Choices | Note |
|---|---|---|
| Body, Faces | Pick, as on Face Mask | Shift+click already adds tangent faces |
| Distance | mm, + out / - in; a number or a field | A field gives a varying offset (a ramp, thicker in the middle) |
| Sides | extend (default) / straight / taper / round / blend | How the moved faces meet the faces around them. extend: the neighbours continue in their own direction (CAD-like, sharp). straight: walls along the moved faces' normals. taper: walls at an Angle (draft). round: a rolling-ball edge, as Offset Body. blend: a smooth ramp over Width (today's Texture + Face Mask) |
| Corners | round / sharp | Where two chosen faces fold: round as Offset Body does, or the moved faces extended until they meet |
| Edge, Size | sharp / chamfer / round (G1) / smooth (G2) | The outside edge of the step: a raised face's top rim, a recess's rim. chamfer: 45°, Size along both faces. round: a circular fillet, tangent (G1), as the Booleans. smooth: curvature continuous (G2), the cubic blend the frames use (`fields.bulge` order 2, grown by `CONTINUITY_SIZE` so it is as deep as round at the same Size) |
| Foot, Size | the same choices | The inside edge of the step: where a raised face's sides meet the part, or a recess's floor meets its walls. Separate from Edge, since a chamfered top with a smooth foot is a common pair. Default: as Edge |
| Apply to | chosen faces / all other faces | As on Face Mask |
| Direction (advanced) | normals / a vector | A vector moves the faces instead (pull a boss up 5 mm) |

**How it could work.**

**Decided (2026-11-08): implicit.** The result does not need to be a CAD body. Textures and holes must
still work on it.

- *Implicit, in the field math (chosen).* Distance to the chosen faces' own triangles, kept only
  where the closest point lies inside the selection (the space over the faces), then joined to the body
  (out) or cut from it (in). straight, taper, round and blend come out of that directly. extend and
  sharp corners need each face's surface continued past its edges: planes, cylinders, cones, spheres
  and tori from OpenCascade at import. Free-form (B-spline) faces fall back to straight. It works on any
  part, it is a pure function of position (blocks stay exact), and the result is a field, so textures and
  holes go on after it. A Face Mask on the original faces still finds the moved ones when its Band is at
  least the distance.
- *B-rep, with OpenCascade.* `BRepOffset_MakeOffset` with a value per face gives an exact CAD body with
  new face numbers, so every node after it sees a normal CAD body. But OpenCascade's offsets often fail
  on fillets and blends, with unhelpful errors, and the body must be tessellated again. Not needed.

**Edges, as built in the engine already:** chamfer and round are the Booleans' `subtract_edged` /
`union_edged` (`_round_max` / `_round_min`, Mercury's rounded intersection). smooth needs a smooth max
with `bulge` order 2 for the outside edge (the frames only have the smooth min). Chamfers look crisp only
with Mesh from Implicit Body's Method dual contouring, which keeps the chamfer's two edges sharp;
marching cubes rounds them by about a voxel. A smooth edge needs a voxel of about Size / 4 to show its
shape.

**Steps.** O1: the node with Sides straight / round / blend, Corners round, Edge and Foot (all four
kinds), a field Distance.
O2: extend, taper and sharp corners from the faces' surfaces. O3: in the 3D view, *Offset faces…* on
picked faces (Ctrl+click, then right-click), and an arrow on the face that is dragged to set the
distance.

**Questions:**

1. Mostly out (raise, thicken), mostly in (recess, thin), or both equally?
2. Should extend (sharp, CAD-like) be the default, or a soft blend?
3. ~~Must the result be a CAD body afterwards?~~ No: textures and holes working on it is enough.

**Must work after it:** Surface Pattern and Texture with a Face Mask on the moved faces (their Band at
least the distance), and Hole Pattern through the moved faces. A Hole Pattern places its holes on the
original CAD faces, measures depth on the CAD mesh, and starts its cutter 1 mm above the face
(PATTERN_PLAN). On a face raised more than that, holes would not reach the new surface, and through
holes would stop at the old wall's depth. So Hole Pattern has to learn the offset: an extra start
height and depth, or the cutter reading the offset field. O1 should check this on a raised face and a
recessed one.

### <a id="gpu-and-cpu"></a>GPU and CPU together

Make the GPU and the CPU cores work at the same time, instead of one waiting for the other.

**What is measured already** (FINDINGS, "Where the cores sit idle", 2026-11-01; 32 logical cores, 12 GB card):

- Block and per-body meshing uses 5-8 cores and 25-35% of the GPU. 16 workers instead of 8: lamp
  meshing 18.2 -> 18.1 s.
- Per full-resolution petal: fields 0.13 s on the GPU, then decimation 0.59 s on the CPU (FINDINGS, GPU).
  A GPU worker spends most of each item on CPU work.
- Why more workers do not help: when the card's free memory limits the GPU workers, the whole map runs
  on only those few (`core.py`, `active = n_gpu`: 9-10 at most with an empty card, about 5 in normal use).
  The other workers sit idle, and the GPU waits while its workers decimate.
- One-core tails after the parallel part: `tiles.stitch` ~5 s of hex studs' 23.7 s mesh step; Merge
  Meshes ~4 s on one core after manifold3d's parallel union; the last petals on 1-2 workers.
- Static Analysis on the CPU (when the card is full) uses 2.2 cores on average. Frame Optimization uses
  one core (its LPs depend on each other).

**More found in the code (2026-11-08):**

- The evaluator runs nodes strictly one after another (`Evaluator._run`: `for node_id in order`). Only the
  items inside one node run in parallel. While a node's one-core tail runs (seams, the fuse), the GPU and
  the other workers wait, even when an independent node could already start.
- Topology's optimizer thread does CPU work between GPU iterations: every 3 iterations it meshes the
  densities for the live view (`LiveSurface`, ~50 ms at medium) and packs them, and every 5 it writes a
  checkpoint (`np.savez`). A medium iteration is ~154 ms on the GPU, so the live view alone takes
  roughly 10% (the checkpoints are not measured).
- Each finished node's view is packed in the run's thread (`on_result` -> `Session._publish_view` ->
  `pack_meshes`: 0.1-0.2 s for a Final view of 145 MB) before the next node starts.
- Each GPU worker prepares every CAD mesh itself before its first query: splits it to 2 mm edges, finds
  cracks and normals (MEMORY_AUDIT C8, ~290 B per triangle, up to 8 workers at once), all on the CPU.

**Ideas, best first** (gains are estimates until measured):

| # | Idea | Size | Expected gain | Where |
|---|---|---|---|---|
| 1 | **Two stages for meshing.** The few GPU workers only sample the field and run marching cubes, then hand the raw mesh to CPU-only workers (no CUDA context: ~0.3 GB each instead of ~1 GB) for decimation, reduction or dual contouring's extraction. The hand-over through shared memory or the SSD `shared` folder, not a pickle through the pipe | M | Lamp meshing 18 -> 8-10 s (GPU stage ~0.15 s per petal over 5 workers ≈ 4 s, CPU stage ~0.6 s over 10-12 workers ≈ 7-9 s); blocks alike | `core._map_in_workers`, `tiles` |
| 2 | ~~**Topology: live view and checkpoints in a background thread.**~~ Dropped 2026-11-09: they cost under 3% of an iteration (measured), and the editor run was no faster | S | none measured | `nodes_fea.on_iteration` |
| 3 | **Views packed off the run's thread.** A publisher thread packs and announces views, so the next node starts at once | S | 0.1-0.2 s per Final view | `Evaluator.on_result`, `Session._publish_view` |
| 4 ✅ | **GPU mesh preparation once, on the CPU, ahead of time.** Done 2026-11-09 with MEMORY_AUDIT #11 / #12 (see Progress). Done once per mesh (in the main process or a CPU worker, while the GPU is busy with the node before) and kept in the SSD `shared` folder; GPU workers only upload. Also skip the split when no edge is over 2 mm, and the crack search when the import found no open edges | S-M | Measured 2026-11-09: 0.95 s for the key cabinet (109k triangles, 294k after the split), 0.42 s for the vent case, 0.02 s per lamp petal, once per worker and mesh (kept after). Only first runs gain, so low priority: worth it for the memory (MEMORY_AUDIT C8), not the speed | `sdf_gpu._GpuMesh` |
| 5 | **A faster CPU fallback for FEA.** The GPU's Warp kernels (`fea/gpu.py`) on Warp's CPU device in threads, as the Voronoi foam does | M | Static Analysis on a full card: 2.2 cores -> most of 32 (bracket_mid's frame body checks took 312 s each) | `fea/solver.py` |
| 6 | **Parallel tails.** Each block seam reduced in its own process (pyfqmr holds the GIL, threads gain only 1.5x); profile Merge Meshes' one-core end | S-M | Part of hex studs' 5 s seam step; Merge Meshes' 4 s one-core end | `tiles.stitch`, `mesh.merge` |
| 7 | **Independent nodes at the same time.** Nodes whose inputs are ready run together, so one node's CPU tail overlaps another's GPU work. The memory planning (`_parallel_slots`, `_make_room`, GPU slots) then has to count every running node | L | The lamp: the center's blocks beside the petals; seams and fuses overlapped | `Evaluator._run` |

Not worth it: idle workers taking whole items on the CPU path. A CPU distance field is ~50x slower (2.1 s
against 0.04 s per petal) and would finish last. The CPU workers take only CPU stages (idea 1).

Smaller ones: the frames' body check overlapping the next layout's LPs where they don't depend on each
other; the editor's right-click search tree (3.7 s on the main thread for 12M triangles) in a Web Worker
(see the viewer section).

**How to look into it:**

1. **A timeline first** (S). An env flag (`DESIGN_ENGINE_TRACE=1`) makes each process write spans: process,
   node, item, stage (field, marching cubes, decimate, reduce, transfer, wait), device, start, end. A sampler
   records GPU use and memory (NVML) and each process's CPU (psutil) every 0.1 s. A small page draws it as one
   row per worker plus GPU and CPU curves. Run it on the lamp export, hex studs Final, the key cabinet,
   fea_bracket at fine on a full card, and frame_bracket. That shows where one side waits for the other,
   which the 2026-11-01 sampling only showed in total.
2. **Prototype idea 1 on the petals** (M). The petals are the simplest case: per body, no seams. Compare
   with the 18.2 s baseline. Measure the hand-over cost and the extra RAM (commit room, not free RAM).
3. **The small ones** (2, 3, 4) can go in alongside, since each is a contained change.
4. **Blocks next**, then the tails, the FEA fallback and independent nodes, by what the timeline shows.

**Testing in the editor.** Every change is done only when it works and is faster in the editor itself, not
just in tests or headless runs:

- Measured in the test editor (port 8766, the same app and code; CLAUDE.md, "Testing the editor": back up
  the autosave folder first). Each recipe runs before and after the change, from the same cache state
  (cold: an empty `DESIGN_ENGINE_CACHE`; warm: the second run), with the timeline on.

  | Recipe | What it exercises |
  |---|---|
  | `lamp_noise.json`: Draft -> Final, then Export | 144 petals per body, the center in blocks, Merge Meshes' fuse |
  | `hex_studs.json`: Final | Blocks and the seam step |
  | `vent_holes.json`: Final with dual contouring | Hole Pattern, dual contouring's CPU extraction |
  | `topo_wall_bracket_bolted.json`: Optimize at medium, Stop, Optimize again | Live view, checkpoints, continuing |
  | `fea_bracket.json`: Static Analysis at fine, with the GPU and with `DESIGN_ENGINE_GPU=0` | The solver and the CPU fallback |
  | `frame_bracket.json`: Optimize | Frame LPs and the body check |

- Checked each time: the same result (triangle counts and view keys; exported STLs reloaded and compared,
  as CLAUDE.md says); "done in" on the status bar and the timeline; peak memory of the process tree and
  the commit room; GPU memory from `/api/memory`. In the editor: an edit during a Final run drops back to
  Draft at once, Stop works mid-node, pieces still stream in, details still appear, the export queue
  still runs, a topology Stop continues exactly.
- `tests/test_gpu.py --gpu` and the full suite stay green.
- Last, the user tries it in their own editor after restarting it with start.bat (never restarted for
  them: it may hold unsaved work).

**Rules while measuring** (CLAUDE.md): the user's editor stopped, or say that it ran (an idle editor holds
~6 GB of the card and makes GPU timings ~10x slow); a kill switch on the whole process tree; check
`/api/memory` first. More processes means more commit: each CPU-only worker needs its BLAS threads capped,
as `StickyPool._start` does.

### <a id="viewer"></a>Implicit body viewer: speed and detail

Show an implicit body sooner after an edit, and with more detail once shown.

**How it works now:**

- Each shown implicit body gets a hidden `mesh.view` node: marching cubes at 128 / 256 / 512 voxels along
  the longest side of all shown bodies (Draft / Medium / Final; each body of a list gets at least 40).
  Bodies over 32M voxels are meshed in blocks near the surface. Draft grids under 1M voxels run on the CPU.
- Refine: Draft at once, then Medium and Final, each after `REFINE_DELAY` (800 ms) of quiet.
- A view is sent as float32 positions and uint32 indices, with no normals. A Final view is ~145 MB
  (8.2M triangles), and packing it takes 0.1-0.2 s.
- In the browser, `computeVertexNormals` runs on the main thread (0.7 s for 12M triangles). The
  right-click search tree is built on the first right-click (3.7 s).
- On-screen detail when the camera rests: the visible part is meshed finer (a voxel of ~1.25 pixels, at
  least 1.5x finer than the view), at most 40 blocks per piece. A whole part asked at 0.02 mm came back
  at 0.17 mm.
- Views always use marching cubes, never dual contouring.

**Speed:**

| # | Idea | Size | Gain |
|---|---|---|---|
| V1 | **Normals from the field.** The engine takes each vertex's normal from the samples' gradient (the grid is in memory already, on the GPU for GPU grids) and sends it as two int16 (4 bytes a vertex) | S-M | No `computeVertexNormals` in the browser (0.7 s at 12M triangles). Also a detail gain (D1) |
| V2 | **Smaller downloads.** Positions as uint16 inside each view's box (steps of ~3 µm on a 200 mm box), indices with meshoptimizer's encoding (three.js has the decoder) | S-M | Roughly 145 -> 30-50 MB for a Final view. Measure first: on one PC, packing and parsing may cost more than the download |
| V3 | **Browser work off the main thread.** The right-click search tree built in a Web Worker as soon as a view arrives, not on the first right-click; normals there too if V1 is not done | S | 3.7 s of frozen view on the first right-click at 12M triangles |
| V4 | **A refine pause that adapts.** Start the next level at once when the last one was quick and nothing is being dragged; keep the pause while a slider moves | S | Hex studs: Medium and Final take 2.25 s, and the pauses add 1.6 s |
| V5 | **What is on screen first.** At Final, mesh the blocks and list items in view first (the camera is known, as DetailProbe uses it); stream a single body's blocks as list items already stream | M | The visible part sharpens first |
| V6 | **Keep unchanged blocks.** An edit that only touches one area (a texture faded by a Face Mask) meshes again only the blocks the changed field reaches, from the fields' bounds and extents | L | Final after a local edit in a fraction of the time |

**Detail:**

| # | Idea | Size | Gain |
|---|---|---|---|
| D1 | **Normals from the field** (V1). True surface normals instead of facet averages | S-M | Smoother shading. Textures read better at the same voxel |
| D2 | **Sharp edges with dual contouring.** A view setting, Sharp edges: auto / on / off (auto: bodies with CAD faces whose field is a distance field). Details too (DUAL_CONTOURING_PLAN, "Later") | M | CAD edges and hole rims about 6x sharper at the same voxel. Extraction is 2-4x slower (10-15x with many holes), on CPU workers (GPU and CPU together, idea 1) |
| D3 | **Finer detail at rest, step by step.** When a detail is done and the camera still rests, ask for the next finer one, until a voxel spans ~1.25 pixels or `DETAIL_MIN_VOXEL`, instead of one piece capped at 40 blocks | S-M | Zoomed-in views reach full detail; a whole small part too |
| D4 | **Each body's own voxel.** Pick each body's voxel from its size on screen (DetailProbe has it), not from the longest side of all shown bodies | M | A small body beside a big one is no longer coarse |
| D5 | **Explore: draw the field instead of meshing it.** Sphere-trace sparse bricks of samples in the browser | L | No meshing wait, pixel-exact against the samples. Only if meshing stays the bottleneck after the rest |

**How to look into it:**

1. **A timing line per view** (S). A debug option in the editor shows, for each view: engine (fields and
   meshing), packing, download, parsing, normals, upload to the graphics card, first frame. It comes from the
   run's events plus `performance.mark` in the browser. Run Draft and Final of `lamp_noise.json`,
   `hex_studs.json`, `vent_holes.json` and a lattice recipe. That shows which of V1-V6 pays.
2. **Detail changes judged by eye.** Screenshots before and after from the same camera (the standard views,
   Space and 1-7), close up and whole, for the user to compare.
3. **Tested in the editor** as for GPU and CPU together: the test editor first, with the same checks (an edit
   mid-run, Stop, pieces, details, right-click menus, picking, section and box cuts on the new geometry),
   then the user's own editor.

**Questions:**

1. What feels slowest: the wait after an edit, the wait for Final, turning a big view, or the first
   right-click?
2. Where is detail missing most: CAD edges and holes, fine textures, or zoomed in close?

## 2. Suggested first picks

| Idea | Size | Why now |
|---|---|---|
| [Live STEP link](#live-step-link) | S | Save in SolidWorks, see the result here: no re-import step |
| [Keep picked faces when the part changes](#keep-picked-faces) | M-L | Today an edit in SolidWorks can move picks, supports and loads to other faces without a word |
| [Overhang view and build direction](#overhang-view) | S-M | Print checks have walls and cavities but not the main reason prints fail |
| [Backlit view for lamps](#backlit-view) | M | The lamp is the main design. Wall thickness decides how it glows, and today that is only known after printing |
| [Surface Coordinates](#surface-coordinates) | M | Textures that follow a face (around a handle, along a petal), not the world axes |
| [Design study](#design-study) | M-L | The tool already measures mass, safety factor and print checks: compare a dozen versions at once |

## 3. New ideas

### Working with SolidWorks

- <a id="live-step-link"></a>**Live STEP link** (S). Import STEP watches its file. When SolidWorks
  saves it again, the editor re-imports and previews (the STEP cache already keys on mtime). A
  switch on the card, on by default.
- <a id="keep-picked-faces"></a>**Keep picked faces when the part changes** (M-L). Face numbers are the
  B-rep's order, and a SolidWorks edit can renumber faces. On re-import, match old faces to new ones
  by surface type, area, centre and normal (`face_info` has most of this). Picks follow the match, and
  faces with no clear match are flagged on the card instead of silently pointing elsewhere.
- **Open in slicer** (S). A button on Export (and on the export toast) that opens the file in the
  chosen slicer (PrusaSlicer, Bambu Studio, OrcaSlicer: path set once).

### Printing and making

- <a id="overhang-view"></a>**Overhang view and build direction** (S-M). Pick the build direction
  (a face menu item, "print on this face", or ±X/Y/Z) and colour the body by overhang angle. The
  check panel lists the area over 45° and the lowest points that need support. Topology's overhang
  limit (below) uses the same direction later.
- **Suggest an orientation** (M). Try directions on a sphere and rank them by support area, height
  and flat area on the plate. Show the best three.
- **Drain and powder holes** (M). Print checks already find cavities. Offer to add escape holes at
  each cavity's lowest point for the build direction, as a Hole Pattern the user can move.
- **Mold maker** (M). A body becomes a two-part mold: box minus body, split on a plane or the
  parting line for a pull direction, registration keys and a pour channel. Implicit modelling makes
  this cheap, and it fits the cake mold recipe.
- **Draft-safe textures** (M). An option on Texture to limit the height so the textured face has no
  undercut against a pull direction. Molded and cast parts can then take textures too.
- **Print estimate** (S). Mass, material and a rough print time (volume, area, height) on the
  Export card. The FEA notes already compute mass.

### Lamp and light

- <a id="backlit-view"></a>**Backlit view** (M). A view mode with a light inside the shade: each pixel's
  brightness falls with the wall thickness along the view ray (Warp ray casts, as `checks` uses). Shows
  hot spots, dark seams and how a texture reads when the lamp is on.
- **Lithophane preset** (S). Picture → wall thickness on a face. Mostly a preset over Picture and
  Texture, with the light and thickness range set for white PLA.

### Textures and patterns

- <a id="surface-coordinates"></a>**Surface Coordinates** (M). U, V, "along" and "around" as fields
  from `cad_uv` and the unfolded sheets. Waves, knurls and spirals then follow the face. PATTERN_PLAN
  milestone 2, never started.
- **Tiled image textures** (M). A greyscale height map (leather, wood grain, fabric, a scanned
  surface) tiled seamlessly over the unfolded sheet, at a size in mm.
- **SVG logos and outlines** (M). Exact vector shapes for Text, Picture and the pattern shapes, so a
  logo's edges stay sharp at any voxel size (Picture is raster now).
- **Curvature field** (M). Mean or Gaussian curvature of the CAD surface as a field: fade textures off
  tight fillets, or make patterns denser on curved areas.
- **Patterns along an edge or curve** (M). A row of shapes along a CAD edge or a fillet's middle (vent
  slots along a rim, studs along a seam).
- **Voronoi cells clipped at the border** (S-M). Cells cut at the selection border instead of dropped
  whole. Open in FINDINGS.
- **Preset thumbnails** (S). Rendered pictures on the library's preset tiles.

### Simulation

- **Deformed shape view** (S). Displacements exaggerated and animated on the body. Open in
  TOPOLOGY_PLAN phase 3.
- **Gravity and acceleration loads** (S). Self-weight from the material's density, plus a g-load in a
  direction (a part that is dropped or shaken).
- **One-click check** (S). Check a topology or frame result with Static Analysis from its card.
  Open in UI_AUDIT.
- **Analysis report** (M). One page (HTML, printable to PDF) with pictures, supports and loads per
  case, safety factor, deflection and mass. Something to send with a design.
- **Natural frequencies** (L). The first modes and their shapes, from the same voxel model and
  multigrid (a shift-invert Lanczos on top). For brackets and lamp arms that hum or wobble.
- **Overhang limit in topology** (L). Self-supporting designs for the build direction. TOPOLOGY_PLAN
  phase 3.

### Exploring designs

- <a id="design-study"></a>**Design study** (M-L). Pick one or two numbers and their ranges. The engine
  runs each combination headless and shows a grid of thumbnails with mass, safety factor, print-check
  results and triangle count. A click keeps one as a variant. Builds on variants and `run_graph.py`.
- **Render export** (S-M). A high-resolution PNG with material looks (white SLS, PLA, brass, oak) and
  soft studio light. For judging the look and for clients. (Turntable video: skipped, 2026-11-08.)
- **Compare two variants** (M). Side by side with linked cameras, or overlaid with the difference
  coloured (distance between the surfaces).

### Editor

- **Measure between two points, and angles** (S). Measure has wall thickness and sections today.
- From UI_AUDIT, not done yet: node search with CAD words (gyroid, stl, simplify, fuse, rotate,
  hollow); a wire dropped on a node goes to the right input; a Number dragged out of an input starts at
  its value; cards say what came out (body count, "x150", triangles and volume); Shell inside /
  centred / outside; shape textures scaled to the face; Undo / Redo buttons and a shortcut list; a
  warning when a wall is thinner than about 2 preview voxels; load cases and Keep / Avoid in the
  analysis wizard.

### Engine and speed

From FINDINGS, DUAL_CONTOURING_PLAN and MEMORY_AUDIT, not done yet:

- GPU marching cubes for blocks (snap Warp's float32 border vertices first).
- Wave noise as one fused kernel (50 of ~80 ms per Final block of the key cabinet).
- The CPU sign test as a kernel; warm workers after opening a recipe.
- Remove loose specks without fusing.
- Transforms of implicit bodies (a move is a shifted grid; a turn needs points off the grid).
- Dual contouring D4: Warp kernels, crease neighbourhoods first.
- Memory: views held two or three times (M4), worker caches outside the budgets (C3, C6, C8), GPU
  memory held when idle (C9).

## 4. Already planned elsewhere

| Doc | What is open there |
|---|---|
| TOPOLOGY_PLAN.md | Phase 3: deformed view, overhang limit, sliding supports, gravity |
| FRAME_PLAN.md | Tapered bars, plates, a Frame Analysis node, a profile system |
| PATTERN_PLAN.md | Surface Coordinates, shape icons and hole markers in the editor |
| VIEWPORT_MENU_PLAN.md | Selection-first mode, face-based Keep, pinned value labels |
| UI_AUDIT.md | The editor list above |
| MEMORY_AUDIT.md | The memory list above |
| DUAL_CONTOURING_PLAN.md | Views and details, Warp kernels |
| NODE_AUDIT.md | Remaining P2 / P3 items and suggested node improvements |

## 5. Questions

1. What should come next: the lamp, brackets and analysis for clients, or molds and cast parts?
2. Which printers and processes matter most (FDM, SLS, resin)? That decides overhang and drain-hole defaults.
3. Does a SolidWorks edit moving picked faces happen in practice, or are the parts mostly finished
   before they come here?

## 6. Progress

Work on the user's list started 2026-11-08. Nothing counts as done until it is checked in a test editor
and the user approves it.

Whole suite on the merged working tree (2026-11-09): 572 passed, 41 skipped (the GPU tests: 41 passed with `--gpu`),
2 failed that failed before today too: `test_recipes_are_valid` (the user's own `kakformsgrej.json` and `testlamp2.json`
have an unwired Body input) and `test_sdf::test_step_import_is_kept_on_the_ssd` (fails on the last commit as well).

| Item | State |
|---|---|
| GPU and CPU together: timeline trace (step 1) | Merged into the working tree (not committed), 2026-11-09: `engine/trace.py`, `tools/trace_run.py`, `tools/trace_report.py`, `start.bat --trace`. GPU traces taken (see "What the traces show" below) |
| Worker garbage collection: `gc.freeze()` after a worker's imports | Done in the working tree, 2026-11-09: lamp Final, worker collects 149 -> 8.5 ms each (10.9 -> 0.6 s of worker time). Wall time unchanged: the collects mostly ran while workers waited |
| Topology: live view and checkpoints in a thread beside the optimizer | Tried and dropped, 2026-11-09. In the bench editor the run was the same as before (medium level 75 iterations in 17 s, fine 1.7-2.3 s per iteration, identical designs), and at fine detail a live view costs 104 ms every 3 iterations and a checkpoint 48 ms every 5: under 3%, not worth a thread |
| Two-stage meshing | Merged into the working tree (not committed), 2026-11-09. Meshes bit-identical (unit tests on CPU and GPU; hex_studs' STL the same SHA-256). Headless with 4 GPU workers: lamp petals 15.4 -> 10.4 s (+0.7 GB); hex_studs export 18.6 -> 17.2 s. **In the editor** (8 GPU workers, 8 GB of the card free): first only 2 CPU workers fit the free RAM and became the bottleneck (lamp Final 21.9 s, GPU workers 80% idle); fixed by letting a GPU worker run a stage itself while the CPU queue is full (and capping new CPU workers by free RAM). Then the lamp's Final after an edit 12.1 s against 12.7 s with two-stage off, peak 10.6 against 9.8 GB: ~5% here, because one stage already keeps 8 processes busy on 16 cores; more when GPU memory limits the GPU workers (other programs on the card). The seam step is not split: not bit-identical (decision open) |
| Lamp in the editor | First Final after opening 19-46 s (cold workers and caches vary), an edit's Final 12-13 s, the edit's Draft ~6.4 s (now the largest single wait after an edit). The lamp's Final needs ~10 GB beside the user's programs (~13 GB free) |
| Viewer: timing line, views packed off the run's thread, adaptive refine pause, right-click tree in a worker | Merged into the working tree (not committed), 2026-11-09. Checked in the bench editor: production build bundles the worker; vent_holes edit -> Final 5.3-5.6 s (was 6.8-6.9); an edit during Final drops back to Draft in 0.9 s with no stale view after; Stop 0.12 s, the Medium view kept; while a value is held only Draft runs, Medium and Final follow on release; FEA colours, slices and the export queue work; `__viewTiming` records engine, pack (41-54 ms for a Final view), download and parse times. **Still to check with the browser pane shown** (a hidden pane draws nothing): normals and first-frame times, the right-click tree from the worker, face menus and picking |
| Offset Faces O1 | Merged into the working tree (not committed), 2026-11-09: node Offset Faces (Sides straight / round / blend; Edge and Foot sharp / chamfer / round / smooth G2; Apply to; a field Distance), Hole Pattern's "Faces moved by", `recipes/offset_faces_demo.json`. 29 tests; GPU = CPU within the part's band to 0.0006 mm (the GPU test was too strict: it compared the far field, where the body's own distances differ by up to 0.46 mm). In the bench editor: the demo runs (Final 7.8 s cold, 1.6-1.9 s after a change), the card shows Width only for blend and sizes only when not sharp, the note reads "1 face raised 1.5 mm / no foot along 22% of the border", sinking, round, blend and all other faces (26 faces, 14.3 s) work. **Still to see with the pane shown**: the look of each Edge / Foot kind, Pick on the card. Defaults: Foot is sharp (not "as Edge"); straight sides (extend is O2) |
| Viewer: normals from the field (V1 / D1) | Merged into the working tree (not committed), 2026-11-09. Views, pieces and details of implicit bodies carry int16 normals from the field's gradient; the browser no longer runs `computeVertexNormals` for them (0.69 s at 12M triangles). Smooth textures shade much better (jagged highlights gone on the wave and the lamp: pictures in the scratchpad `fn\png`); creases (hex studs) come out softer, and View ▸ Field normals switches back. Cost: +17% bytes, lamp petals view +1.3%, center in blocks +8% (6% more samples). Tests 140 + 43 CPU and 41 GPU pass; in the bench editor the wave's implicit view and both lamp views (144 petals, the center in blocks) arrive with unit normals, Mesh from Implicit Body views without. **Still to see with the pane shown**: the shading itself, block seams, the toggle |
| Shared CAD arrays (MEMORY_AUDIT #11 and #12; GPU and CPU idea 4) | Done in the working tree (not committed), 2026-11-09. `Shared` arrays are memory-mapped (one copy in RAM for every process, no commit); a mesh from 100k triangles gets its normals and cracks (CPU) and its 2 mm split, crack flags and normals (GPU) worked out once by the first process that needs them and mapped by the others (`shared.prepared`); SSD cache hits from 32 MB are mapped (smaller ones are read: mapped reads are up to 2x slower); the engine's memory meter sums private working sets, not RSS. Measured on a 1.31M-triangle mesh: 6 CPU workers 2.86 -> 1.89 GB, a worker's start 4.4 -> 1.8 s once prepared; 4 GPU workers' start-up peak 3.45 -> 1.33 GB, 3.5 -> 0.8 s. Key cabinet export (hex_studs, 109k triangles): the STL byte for byte the same, meshing 22.5 -> 20.9 s, peak 4.38 -> 4.22 GB; in the test editor the export took 19.6 s (meter 4.4 GB against an RSS sum of 6.1 GB), Surface Pattern Invert (a face mask) previews to Final. The lamp is unchanged (its parts are under 100k triangles: 25.6 / 24.8 s, 6.4 / 6.3 GB). Tests: 581 passed, GPU 42 passed |
| Lamp Draft after an edit | **Fixed in the working tree, 2026-11-09.** Headless the edit's Draft took 0.66 s after a Draft but 5.1 s after a Medium: the SSD cache kept no array under 1 MB ("cheaper to recompute"), and a petal's Draft distance field is ~0.4 MB that costs ~160 ms to compute against ~1 ms to read, so once Medium and Final pushed the Draft arrays out of the workers' RAM every edit computed all 144 again. `DiskCache.MIN_BYTES` 1 MB -> 64 KB: the edit's Draft 0.8-1.0 s after a Medium. **In the editor: lamp edit -> Draft on screen 1.6 s (was ~7 s), -> Final on screen 17-18 s (was 23.5-24.9 s)** |

### What the traces show (2026-11-09, GPU, headless through the editor's own Session)

- **lamp_noise, Final preview: 25.9 s.** The 144 petals 19.0 s: once started, every worker is busy without gaps,
  ~0.65 s a petal: decimation ~0.40 s (CPU), field ~0.18 s (GPU-labelled, the GPU only 23-26% used), marching
  cubes ~0.03 s (GPU). Workers started 4.6 s into the node in the headless run (cold processes; the editor keeps
  them warm). The center (21 blocks) 5.8 s, after the petals although it does not need them (idea 7).
- **hex_studs, export at Final: 26.6 s.** 944 blocks on 7 workers 18.5 s: field ~40 ms (GPU), marching cubes
  ~19 ms (CPU), reduction ~56 ms (CPU); workers ~60% busy. Then one core: seams 5.0 s (their reduction 4.0 s) and
  the probe 1.4 s, 24% of the run.
- **hex_studs, Final preview: 4.0 s,** one grid in the main process (no workers): its GPU field step takes 1.4 s with
  the GPU near idle, so that time is CPU-side work (the stud pattern's per-shape evaluation, array glue).
- So two-stage meshing (idea 1) targets the petals and every export; parallel seams (idea 6) the export's one-core
  tail; independent nodes (idea 7) the lamp's center.

### Baselines (before any change)

A test editor with its own empty cache (port 8768), the user's editor stopped, other programs holding
3.7 GB of the card, the agents' unit tests possibly running at the same time. "Edit" = one setting
changed, timed until each quality is on screen (the debounce, Draft, the refine pauses, Medium, Final).

| Case | Draft shown | Medium shown | Final shown | Notes |
|---|---|---|---|---|
| vent_holes, edit grille spacing | 0.65 s | 1.73-1.95 s | 6.84-6.92 s | Final run 4.1-4.2 s; pauses 2 x ~0.86 s |
| wave_offset, edit wave b | 0.63-0.65 s | 1.71-1.74 s | 4.77-4.92 s | Final run 2.2 s; the pauses are ~35% of the wait |
| fea_bracket, Static Analysis (medium, GPU) | | | | 1.8 s solve; the cold preview's Draft 1.9 s |
| topo_wall_bracket_bolted, Optimize at Final quality (cold) | | | | Medium level: 75 iterations in 17 s; fine level: 1.6-1.8 s per iteration (80-110), 2.3 s at 120; stopped at 120 |
| frame_bracket, Optimize (cold) | | | | 11.6 s (layout 5 s, one round of body check) |
| vent_holes, edit grille spacing, **after the viewer changes** | 0.64-0.75 s | 1.18-1.29 s | 5.28-5.55 s | Medium now starts ~0.3 s after the edit's Draft, Final ~0.4-0.5 s after Medium |
| wave_offset, edit wave b, **after the viewer changes** | 0.65 s | 1.08-1.09 s | 3.56-3.79 s | Was 4.77-4.92 s: ~1.2 s (25%) less |
| lamp_noise, cold | 49.2 s | 52.6 s | not measured | Cold STEP import. The cold Final grew the editor to 11.3 GB beside the agents' tests: the kill switch stopped it at 2.45 GB free. Lamp and hex_studs wait until the agents are done |
