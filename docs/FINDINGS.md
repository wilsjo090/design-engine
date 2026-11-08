# Findings

What was learned building this tool, with the measurements behind each decision. Numbers are
from the user's PC unless stated otherwise. Newest knowledge wins: update this file when a
finding changes.

## Machine and environment

- Windows 11 Pro, 16 cores / 32 threads, RTX 3080 Ti (12 GB), 31.9 GB RAM. Other programs
  typically use 14–17 GB of RAM, so only ~15–17 GB is free. C: has ~93 GB free.
- Conda env `synkad`, Python 3.12. Key versions: cadquery-ocp 8.0.1, libigl 2.6.3 (nanobind API),
  trimesh 5.1.1, numpy 2.5. Node 24 / npm 11; React 18.3, @xyflow/react 12, three 0.169,
  @react-three/fiber 8, drei 9, zustand 4.5, vite 5, TypeScript 5.6.
- OCP 8 API: `TopoDS.Edge(...)` / `TopoDS.Face(...)` (no `_s` suffix), but
  `BRep_Tool.Triangulation_s` still has it.
- libigl's `AABB`, `FastWindingNumberBVH`, `per_face_normals`, `per_vertex_normals`,
  `per_edge_normals` are exposed and can be built once and reused.

## The user and the goal

- Designs parts in SolidWorks and wants a simple personal tool for nice-looking
  parts (decorative textures for 3D printing). Single user, not for sale, so Python was chosen.
- Writes informally (Swedish, typos); number fields accept `0,35` as well as `0.35`.
- The original wave graph: `Wave = 0.35 · acos(cos(4 · Body_1 / Body_2))` → Offset Body (distance =
  Wave) → Smoothen Body (grid 0.1 mm, 3 iterations). `acos(cos(x))` is a triangle wave.
- Pattern the user wants (reference picture): contour lines of smooth noise. "The same wave but
  varying in 3D, the noise is the variation"; both ridge spacing and ridge height must vary.

## The lamp (main test case)

**Petals** — `lamskarmhalldelar.STEP` (75 MB, not in the repo):
- 150 solids, all the same part (243.3 mm³): hexagonal dish 15.3 × 16.5 × 4 mm, ~1 mm wall,
  round tube rim, flat bottom at radius 59 mm from the lamp axis (Y). Rows are 19.75 mm apart.
- 6 bodies are exact duplicates sitting on others (18–20 = 0–2, 39–41 = 34–36) → 144 unique;
  `Import STEP` drops them (`skip_duplicates`).
- Reading and tessellating the file takes ~48–55 s on every server start.

**Center** — `lamskarmhall-center.STEP`: one solid, a faceted cylinder about Y (radius 53–61.3,
y −151…12.7). Wall at radius 53–55 with a hexagonal hole under each petal; six thin fins per
petal rise to ~61 mm and touch the petal's **sloped underside** (the petals rest on the fins).

**Scaling petals while keeping them seated** (`cad.scale` node, `engine/transform.py`):
- The six contact faces' planes meet in one point (apex) on the petal axis at radius 56.632 mm.
  Scaling about that point maps each plane onto itself, so the seat is kept exactly. Found
  automatically from the contact points for all 150 petals.
- Fin contact error: 0.0000 mm up to 1.3×; at 1.5× the fin tips miss the sloped faces (0.16 mm).
- Gap between neighbours if every petal is scaled: 6.43 mm at 1.0×, 5.11 at 1.1, 3.79 at 1.2,
  2.46 at 1.3, 1.14 at 1.4. With texture on both sides, keep scale ≤ 1.3.

**Texture vs. fins**:
- Offsetting the whole petal also grows the underside, pushing it up to ~0.6 mm into the fins.
  `samples/lamp_wave.stl` and anything exported from the noise recipe before the fix have this.
- Fix in `lamp_noise.json`: texture × `smoothstep(center_distance, 0, 0.4)`, then subtract the
  center grown by `clearance`. At a 0.1 mm voxel an exact fit leaves ≤ 0.13 mm interference at the
  fin-tip corners (marching cubes cannot make a sharp concave notch); with clearance 0.15 mm the
  closest approach is 0.12–0.14 mm and nothing overlaps.
- For one printed piece `clearance` became `overlap` (petals reach into the fins); see "Fusing
  petals and center" below.
- Still open: should the round rim tube be textured or smooth?

**Current settings** (`recipes/lamp_noise.json`): pattern noise scale 55 mm (the user's change
from 45), 3 octaves, roughness 0.45, seed 3; density 140 (`acos(cos(140 · noise))` remapped to 0–1); height variation
noise scale 20 mm, seed 7, remapped to 0.6–1.4; amplitude 0.63 mm. Size pattern: noise scale
35 mm, seed 11, smoothstep 0.05–0.35, `size_amount` 0.3 → 41 of 144 petals enlarged, 18 at 1.3×.
Mesh voxel 0.1 mm, keep 0.3; `overlap` 0.3 mm; petals and center fused into one solid.
On a flat slice, scale 45 / density 140 matches the reference picture best (scale 25 / density
25 gave too few, too regular lines).

## Geometry and meshing

- **OpenCascade tessellation has cracks**: T-junctions at fillet corner blends (8 open edges on
  the test part, 6 per petal). The pseudonormal sign is wrong only for points whose closest
  feature lies on such an edge; those get a winding-number test.
- **Sign methods**: pseudonormal matches the exact winding number with 0 sign mismatches and
  distances within 5e-7 mm. libigl's `FAST_WINDING_NUMBER` *signed_distance* mode returns
  approximate distances (up to 0.02 mm off): never use it as a reference.
- **Watertight STL** needed three fixes, all in `engine/meshing.py`:
  1. Marching cubes output is closed on its own; trimesh `process=True` (vertex merging) pinches
     it, so meshes are built with `process=False`.
  2. Field values (almost) exactly zero put several vertices on one grid corner; STL stores no
     connectivity and readers re-merge by position → pinches. Values with |v| < 0.01·voxel are
     pushed off zero (surface moves ~0.001 mm).
  3. Decimation leaves vertices that differ in float64 but coincide in float32 (STL precision) →
     snap to float32, merge, drop collapsed triangles.
  The Phase 1 lamp export had 487,000 bad edges after reloading; now every edge is shared by
  exactly two triangles.
- **Decimation**: low aggression avoids specks (sliver triangles) on curved areas. Never
  decimate one huge merged mesh: 12.5M faces → target 1.5M only reached 10.6M, took 57 s and
  3 GB. Decimate per body instead.
- **The decimation threshold is in mm²** (quadric error, `alpha·(iteration+3)^aggression`), so
  the same settings behave differently at every voxel size. Whole bodies used fast-simplification
  unscaled while blocks scaled it (`alpha = 1e-9/voxel²`): a real lamp petal at 0.1 mm with keep
  0.3 stopped at 48% of its 196k triangles after 0.45 s (550 slivers < 2°). Now both use
  `meshing.simplify` (pyfqmr, scaled, aggression 3): exactly 30% in 0.21 s, 227 slivers, within
  0.0045 mm of the full mesh, watertight after float32 reload. The lamp export dropped from ~15M
  to 8.5M triangles.
- **Smoothen**: n passes of a 3×3×3 box filter ≈ Gaussian with σ = grid_size·√(2n/3). Done in mm
  so Draft (4× coarser voxels) looks like Final; volume differs from the box filter by 0.004%.
- **Noise**: Perlin "improved" noise, fractal sum. Smooth noise is sampled on a coarser grid
  (≥ 48 samples per finest feature) and interpolated; error < 2e-4, i.e. < 0.03 rad of phase even
  at density 140.

## Fusing petals and center (one printed piece)

- **Don't union the center inside the per-petal chain.** Import STEP's list makes every node
  run once per petal, so `union(petal, center)` gave 144 bodies "petal + whole center". Each mesh
  grid became the lamp's box (~130 × 170 × 130 mm, ~500× a petal's box): 106M voxels at 0.3 mm,
  2.9 billion at 0.1 mm (over the 300M cap). Too big for free VRAM → CPU, one at a time on one
  core, ~3 min per petal (~7 h for the lamp), 144 copies of the center, and `seated` then cut
  the center back out anyway.
- **Instead**: the center goes through `Mesh from CAD` (its own tessellation: 22,176 triangles,
  closed, no cracks, genus 145, 96,282 mm³, read + tessellate 4.6 s). The petals are meshed per
  body as before, then `Merge Meshes` with fuse = yes does a mesh Boolean union (manifold3d 3.5).
- **Touching is not fusing**: petals rest on the fins with coincident faces (zero overlap).
  `overlap` (0.3 mm): `contact_grow` = overlap × (1 − clip_fade) is added to the texture offset,
  so a petal grows into the fins only where it touches them (fading out within 0.4 mm, which also
  leaves a small weld fillet). `seated` subtracts the center shrunk by `overlap`, which caps the
  depth. A negative `overlap` gives a gap instead (clip-on petals).
- manifold3d fused 145 meshes (10.8M triangles at Final, keep 0.3) into one solid in 11.9 s, in
  the main process. Draft and Medium previews only concatenate (the server sets fuse = "no"):
  same look, and coarse draft meshes may not overlap enough to join.
- **Pitfall**: the union's output had 4 sliver triangles whose corners coincide in float32. The
  reloaded STL had 8 edges with 4 triangles each (not watertight, 4 zero-volume "bodies"). Fuse
  output now goes through `snap_float32`, like decimation. Result: watertight, 1 body, 0 bad
  edges, 163,565 mm³.
- Times: Final preview 56 s (petals 43.8 s on the GPU, fuse 11.9 s); export 68 s after a Draft
  preview (STEP files already read).

## Big bodies in blocks (`engine/tiles.py`)

- Why: texturing the center needs a grid over its whole box: 103M voxels at 0.3 mm (5–8 GB
  with all arrays), 2.7 billion at 0.1 mm (10 GB per array). SSD offloading (Dask, memmap) was
  rejected: ~10 arrays at once means ~100 GB of SSD traffic per pass, marching cubes needs its
  whole array anyway, and most of the box is empty (cylinder hole, corners). Center surface
  125,196 mm² (as much as all petals together); only 22% / 33% / 51% of the box is within reach
  of the surface at 3.2 / 6.4 / 12.8 mm block size.
- Blocks of 96 voxels (97 samples, outer ones shared with neighbours). Seams close only if
  shared samples are bit-identical in both blocks; three things were needed:
  1. positions from a global index (`Grid.block`, also inside the Warp kernel), not
     block origin + local index (differs in the last bits);
  2. block starts multiples of 12, so the coarse sampling lattices of noise (factor 2–4) and
     distance fields (2, 4) coincide. Factor 3 interpolation weights differ in the last float64
     bit between blocks but are identical after the cast to float32;
  3. a halo for smoothing (3σ + 1 samples, rounded up to 12).
  `tests/test_tiles.py` checks block values against the whole grid with `array_equal`, and the
  tiled mesh has exactly as many triangles as the whole-grid mesh.
- Welding by grid edge: a border vertex has two whole-number coordinates (in sample units) and
  one in between; (floor coords, axis) is a unique integer key, so tiny value differences between
  workers (e.g. Warp BVH tie-breaking) cannot open cracks. Only a sign disagreement could, which
  needs a value within ~1e-13 of zero.
- Simplification per block with pyfqmr `preserve_border=True` (border vertices stay bit-identical;
  checked), same algorithm as fast-simplification; run in voxel units with
  `alpha = 1e-9 / voxel²` so its error threshold matches the mm-based settings.
- Probe: field sampled every 12 voxels; a block is computed if the probe shows a sign change or
  |value| < 2.5 × half a probe cell's diagonal. Blocks whose surface crosses into an uncomputed
  neighbour pull it in (next round), so a probe miss cannot leave a hole.
- **pyfqmr pitfall**: it does not check that a collapse keeps the surface manifold. In ~1 of 200
  blocks it left an edge shared by 4 triangles (marching cubes alone: never). Such a block is now
  kept unsimplified (+0.6% triangles on the center).
- **Measured** (textured center from `lamp_noise2`, 0.1 mm, keep 0.3, GPU, 6 workers):
  2,650M voxels in the box, 1,754 of 3,528 blocks computed (50%), meshing 61 s, whole run 68 s,
  13.8M triangles, watertight after STL reload, 137,749 mm³. Peak RAM: whole process tree
  5.85 GB, main process 2.24 GB. In the editor (Final preview): 66 s, server + workers 5.9 GB.
  Before the fixes the main process peaked at 4.7 GB: trimesh's `.volume` and STL export build
  full-size triangle arrays (~1 GB each plus temporaries); both are now done in chunks.
- **Preview of a list ran out of memory**: the 3D preview of an implicit node used `cells`
  voxels along *each* body's longest side, so `petals` (144 bodies) at Final got 512 each:
  0.05 mm voxels (finer than the 0.1 mm export), ~9 billion voxels, 100M+ triangles kept in the
  main process → stopped at 12.1 GB. Now `cells` applies to the whole group (min 40 per body):
  petals at Final 2.3M triangles, 27 s with 3 workers, 3.6 GB in total.
- The texture pinches off tiny pieces in tight corners (fin against wall): 5 solid specks
  (0.07 mm³ in total, ≤ 0.8 mm) and 47 bubbles inside the solid (0.014 mm³). Fusing drops
  separate pieces under 1 mm³ (`meshing.SPECK_VOLUME`) and says so in the log.

## The runaway import (2026-10-11 evening)

A new file importing `cyt-give-step.step` (924 MB AP242 from Spatial InterOp, one body,
41.5 × 41.5 × 15 mm, 22,397 complex faces, a sparse 2,047 mm³ structure) took the editor to a
23 GB commit and the PC to 0.3 GB free. Causes, all fixed:
- **Angle, not tolerance, set the triangle count**: BRepMesh's angular deflection (was fixed at
  0.1 rad) gave 12.2M triangles at 0.05 mm and 16.2M at 0.01 mm; at 0.5 rad it is 1.7–1.9M (16 s
  instead of 90 s) with the same chord tolerance, 0 open edges, volume within 0.1%. Import STEP
  now has an `angle` input, default 0.5.
- **Import memory**: Python lists of tuples for 9.4M vertices, a redundant second
  `merge_vertices`, trimesh `.volume` (+3.9 GB) and its edge tables (+2.7 GB). Now per-face
  arrays, chunked `signed_volume`, integer-key edge count, cache cleared: 12.2M triangles peak
  4.4 GB instead of 7.8 (2 GB of which is OpenCascade holding the 924 MB file; reading takes
  64–77 s). Output is bit-identical to before (test), so cached distance fields stay valid.
- **Two runs at once**: `Session.start` waited for the old run but did not stop two requests
  from waiting together; a run that cannot stop quickly let them pile up, then all started.
  Now one start at a time, and only the newest waiting request runs.
- **Cancel could not interrupt one huge body**: `tessellate` now checks for cancel after every face.
- **Watchdog blind spot**: resident memory drops when Windows pages out. Committed memory is
  not usable instead (an idle engine process commits 1.3 GB it never touches, CUDA +0.4 GB), so
  the free-RAM checks are the backstop, plus a last resort: if memory stays >125% of the limit
  (or < 0.4 GB free) for 10 s after a stop, the process exits instead of freezing the PC.
- **Queued payloads**: the pool pickled and queued every item up front; feeder threads held
  them all in memory (7 stuck feeders seen). Now at most 2 items wait per worker. Worse, after
  workers were stopped the feeders blocked forever (this process holds the pipe's read end too,
  so it never breaks on Windows): the memory was never freed and a script could not exit.
  `StickyPool.terminate` now reads the queues empty.
- **Every worker builds the part's search structures**: for the 2.4M-triangle part, ~1.2 GB per
  worker on the CPU (libigl), 0.8 GB peak on the GPU (Warp), so 8 workers + main went past
  10 GB. The block tasks' memory estimate now includes 0.6 GB per worker plus 500 B per CAD
  triangle (GPU: 150 B extra per triangle), so big parts get fewer workers; the main process
  drops its copy after the probe. Result: import 72 s (reading), meshing at 0.1 mm 21 s with
  4 GPU workers, peak 7.9 GB, watertight, 2,041 mm³ (CAD 2,047).
- **`/api/memory`** stat-ed every SSD-cache file on each poll; on a thrashing disk the requests
  tied up the server's threads. The size is now measured at most every 30 s. The cache had
  grown to 25 GB of 20 (each worker trimmed only after writing 2 GB); now trimmed at startup
  and after every 0.5 GB.

## Lattices (2026-10-12)

- Face Mask *Reach: volume* first put its border halfway between the chosen and the other faces.
  On the 36 mm board a chosen side then only reached ~18 mm (to the bisector with the top and
  bottom), so with Fade 50 the mask peaked at 0.066 and no strut got solid. Now it is simply
  1 at the chosen faces fading to 0 at Fade mm from them (distance to the chosen triangles, igl
  AABB on a 4x coarser grid, interpolated).

- Strut lattices fold every point into the octant [0, 0.5]³ of a cell and measure to the struts
  clipped to it (cubic 3, bcc 1, fcc 3, octet 6, cubic + bcc 4 struts there). Checked against a
  brute-force distance to the 27 surrounding cells: max difference 1e-4 mm. First octet attempt
  had 27 struts: "neighbouring face centres" also matched opposite ones (L1 distance 1); use the
  Euclidean distance (0.71 vs 1.0).
- TPMS: solid modes fill 50% ± 2% (gyroid, schwarz_p, diamond); sheet volume = surface area ×
  thickness within 6% (thickness really is in mm). GPU vs CPU: 2e-6 mm near the walls; up to
  3e-4 mm 10 mm away from them (float32 trig, distance estimate amplifies it), irrelevant.
- Voronoi: "the 16 nearest strut midpoints" missed the true nearest strut by up to 0.34 mm near
  long struts at the foam's edge. Now struts are cut into pieces ≤ cell/3 and a point is re-checked
  with 4x more candidates until the best distance ≤ (k-th midpoint distance − half a piece):
  exact against brute force. Test part at Medium: ~6 s (gyroid and octet: ~1 s).
- Voronoi on the GPU and CPU (2026-10-13, `engine/segments.py`): a Warp BVH over the pieces'
  boxes (padded 1e-3 mm) and, per point, a search cube of half size r = cell/4 doubled until the
  best distance ≤ r (every piece outside the cube is farther: exact). 4.9M voxels, cell 20:
  KD-tree + numpy 28 s; Warp GPU 0.04 s; the same kernel on Warp's CPU device 3.7 s on one
  thread, 0.23 s with 16 (Warp's CPU launches release the GIL, so plain Python threads scale).
  GPU vs CPU within 1e-5 mm.
- TPMS with exact gradients instead of 6 extra evaluations: gyroid 0.47 → 0.2 s, lidinoid
  1.44 → 0.6 s on 16M voxels (differences up to 0.03 mm, all central-difference error).

## Usability round (2026-10-13)

- **Fades in mm look like steps.** A Voronoi foam with 5 mm cells is 42% solid at 2 mm struts,
  70% at 3, 89% at 4, 98% at 5 and solid from ~7 mm (2 x the largest distance to a strut, 3.65 mm).
  The user's Remap went to 20 mm, so with Fade 100 the board stayed 100% solid for 70 mm and
  changed to the lattice in the last 30 mm. Density units sample the lattice's distances (one cell
  for periodic lattices, up to 400k points of the foam) and use their quantiles: asking for 30%
  gives 30% (within 1-2% on test grids).
- **Face selections on different bodies.** Face numbers are per body; a plain number applied to
  every body (right for the identical petals, wrong for a STEP with different parts: picking face
  5 on a box also chose face 5 of a cylinder). Picking now stores `body:face` when the bodies are
  not copies. "Copies" = same face count, triangles per face and share of the area per face (two
  differently sized boxes have the same topology; scaled petals keep their area shares).
- **A list of face masks was drawn with body 0's mask everywhere** (slices and surface colours
  composite a list of per-body fields by their bounds, and a mask has none). Masks now carry an
  `extent` used only for display; bounds would have grown mesh grids by the band.
- **Surface colours on CAD views mislead**: a planar CAD face is two triangles, so a field sampled
  at its 4 corners (all on the selection's edge for a mask) shows 0 on the whole face. CAD views
  are not coloured; implicit previews have a vertex every voxel.
- **Variants froze the 3D view**: CameraFit passed the bodies' box to the floor-grid state and then
  grew the same object with the variants' boxes, so bounds -> variant offset -> bounds grew
  without end and the camera position became invalid (blank canvas, no errors). Copy boxes before
  handing them to state.
- **Export queue**: the last run's thread is still alive when it publishes the final queue state,
  so a job is marked done first (otherwise the toolbar kept showing "1 running").
- Print check wall thickness: ray from each vertex along the inward normal (Warp `mesh_query_ray`),
  hits from outside or right at the start stepped over. A sphere of radius 5 measures 10.000.

## The key cabinet (`testlamp2.json`, 2026-10-14)

`nyckellada.STEP` (1.5 MB): one body, 157 x 28 x 217 mm, 109k triangles, 90 CAD faces. Wave
texture (noise scale 100 x density 140, amplitude 2 x 0.6-1.4) masked to 53 faces = 89% of the
surface (Face Mask surface reach, fade 10, band 10), mesh voxel 0.2 (Draft 2.7M, Medium 19.6M,
Final 149M voxels in 216 blocks). The Voronoi branch is not connected to the mesh: not computed.

| GPU, cold | Draft | Medium | Final |
|---|---|---|---|
| Face mask at full resolution | 1.3 s | 14.6 s (2.2 s without memory pressure) | 175 ms per block |
| Face mask sampled coarser (6/3/2x) | 0.09 s | 0.35 s | 28 ms per block |
| Whole textured body | 1.4 s -> 0.16 s | 26 s -> 0.68 s | 16-18 s in all |

- **Memory pressure was the largest factor.** The user's idle editor held 10.6 GB of RAM and
  ~6 GB of the GPU's 12 GB (8 worker CUDA contexts and their cached search structures), leaving
  2.6 GB: the wave noise took 2.0 s at Medium instead of 0.33 s, the mask 14.6 s instead of 2.2 s
  (Windows moves GPU allocations into shared system memory instead of failing).
- **Wide-fade surface masks are smooth**, so they are sampled ~6 times across the fade on a grid
  2, 3 or 6 times coarser (factors dividing ALIGN together with nearest_on_grid's own pass of 2)
  and interpolated. Within the band: mean difference 0.0004, 99.9% under 0.035; worst 0.05
  outside / 0.10 inside the part where the closest face switches at an edge (the mask jumps
  there anyway). Blocks stay bit-identical. The border KD-tree query uses the worker's threads.
- **Final** (209 of 216 blocks have surface, the part is 2 blocks thick): probe 0.7-0.9 s,
  blocks 11-13 s, stitching 5.7M triangles 2.9 s, import 1.1 s. Per block on the GPU: distance
  4 ms, mask 28 ms, wave noise 50 ms, then marching cubes on the CPU. 8 GPU workers 16.4 s,
  4: 18.0 s, 2: 22.9 s (one GPU shared; fewer only saves memory: 1.5 GB VRAM and 1.4 GB RAM at 4).
  The old full-resolution mask was no slower end to end at Final (16.5 s): there the mask was a
  third of the GPU work per block and block start-up/stitching dominate.
- Recipe notes: the wave reaches 2.8 mm outward but Implicit Body from CAD has band 2 (beyond it
  distances are interpolated): band 3 is the honest setting. Face Mask band 10 only needs to cover
  the largest offset (3 would do). Ridges are 2.5-4.5 mm apart (5th percentile / median): voxel
  0.2 = 12-22 voxels per ridge, 0.3 would still give 8-15 with 3.4x fewer voxels.

## Face Mask "Apply to" (2026-10-14)

- On the key cabinet the texture had to go almost everywhere, so 53 of 90 faces (89% of the
  surface) were picked one by one. *Apply to: all other faces* uses the complement: pick the few
  faces that stay plain (shown red while picking); with none picked everything gets the effect.
  Fade and Position then refer to the faces that get the effect; volume reach measures from them.

## Reduce Mesh (2026-10-14)

- Key cabinet Final mesh, 5.7M triangles. pyfqmr `simplify_mesh_lossless(epsilon=tol²)`: 0.01 ->
  1.15M in 7 s (within 0.019 mm) but no border locking; `simplify_mesh(lossless=True,
  threshold_lossless=...)` kept 5,678,552 of 5,678,636 (it hardly reduces). `simplify_mesh` with
  target 4, 40 iterations, aggressiveness 3 and alpha = tol² / (39 + 3)^3 (threshold ends at tol²):
  1.10M in 8.5 s, within 0.0058 mm, and `preserve_border` works.
- Every run left a few edges used by 4 triangles (~1 per 100k-1M): mostly folds, two triangles on
  the same three vertices back to back (10 on this mesh); they cancel in pairs (an odd copy is
  surface and stays: removing all of three opened a hole). The rest (a pinched edge) is handled by
  keeping the original triangles within 0.5 mm and reducing the rest again with that patch's
  border locked: one retry, 103 triangles kept. 28 s in all; watertight after STL reload; 5
  bodies and 1 tiny cavity before and after (the texture pinches off 4 specks on this part).
- Flat +Y face (14,100 mm²): 705,882 -> 2,824 triangles; textured rest 4.97M -> 1.10M.
- Folded into meshing (Mesh from Implicit Body `tolerance`): each block is reduced in its worker
  with its open border kept, the seams after stitching (triangles touching inner block planes,
  tolerance/2). Key cabinet Final: 18.4 s in all vs 45.0 s (meshing 15.8 + Reduce Mesh 28.0); the
  workers' share added 1.5 s. 1.14M triangles (3% more than one global pass), watertight, same 5
  bodies and volume; surface within 0.0028 mm for 99.9% of it, 0.029 mm at worst (a block's
  quadrics miss the neighbour's planes near its border). A second global pass is no substitute:
  it starts a new error budget (1.10M -> 0.78M, the surface moving up to another tolerance).
- No GPU version: edge collapse is a long chain of dependent steps; no installed library does it
  on the GPU, and the block workers already run it on all cores next to the GPU's field work.
  At 0.05 mm: 464k triangles, but the surface moved up to 0.09 mm (quadric error underestimates on
  curved detail).

## Static analysis (FEA phase 1, 2026-10-14)

Plan and design: [TOPOLOGY_PLAN.md](TOPOLOGY_PLAN.md). Test parts: a 100 x 10 x 10 mm cantilever
(tests) and `samples/bracket.step`, an L bracket (70 x 40 x 6 base, 55 mm upright, 6 mm inner
fillet, two bolt holes held, 50 N +X on a pin hole near the top; `recipes/fea_bracket.json`).

- Accuracy, cantilever (1 mm elements, 10 through the height): tip deflection 8.8% short at
  first. Two causes: the grid started half an element outside the part, so the flat faces were
  half-filled elements (stiffness smeared further out), and plain hex8 locks in bending. Grid
  starting at the part's lowest corner + Wilson's incompatible modes (condensed out: on identical
  cubes still one constant 24x24 matrix, so free): 1.1% short (2 mm elements: 1.3%; plain hex8
  1.8% / 3.9%), bending stress at element centres exact; the rest is the fully clamped end.
  Supports fix only the corners within 0.8 h per axis of a face point (a face on element faces
  clamps one layer of nodes, not two).
- Multigrid smoothing: damped Jacobi (omega 0.6) diverged where stiff meets soft material:
  lmax(D^-1 K) is 3.1 on a uniform part, 4.2-4.9 with 1:1e6 contrast. Chebyshev (degree 3, aimed at
  lmax/10..lmax, lmax by 12 power iterations x 1.15, capped by Gershgorin): 56 iterations to 1e-6
  on 32^3 blobs at 1e-6 and 1e-9 contrast, 6-8 on uniform parts.
- Sparse parts: the bracket fills 17% of its box. Products over the whole grid (8 slices) cost 66 ms
  at draft; element by element over the filled ones only (index gather, bincount / scatter_add):
  16 ms, 0.86 -> 0.27 s per CPU iteration. Levels at least half full keep the slab path.
- Bug met: `xp.zeros_like` of a transposed view (prolong output) keeps that layout, its
  `reshape(-1, 3)` is then a copy, and scatter-adds into it were lost (the product came back 0).
  The result arrays are now created C-ordered and checked.
- GPU precision: all-float32 CG said "converged, residual 7.6e-6" while the bracket bent 3.60 mm
  instead of 4.21. For smooth, nearly rigid displacements the element terms of K u cancel and
  float32 gets the product ~10% wrong (products and smoothers alone agreed with the CPU to 1e-7).
  Iterative refinement (float64 residual, float32 corrections) was right but took 110 iterations
  instead of 22; reliable updates stalled at 6e-2. Mixed precision works: CG and the fine-level
  product in float64, the multigrid preconditioner in float32, flexible (Polak-Ribiere) beta:
  22 iterations, same answer as the CPU to every printed digit.
- Times, bracket, warm (CuPy compiles its kernels on the first analysis in a process: +26 s once):

  | detail | elements (box / with material) | GPU | CPU |
  |---|---|---|---|
  | draft | 246k / 42k, 0.89 mm | 0.9 s | 6 s |
  | medium | 1.5M / 255k, 0.49 mm | 3.3 s | 47 s |
  | fine | 6.6M / 985k, 0.31 mm | 11.6 s | (not run) |

  Memory per box element: CPU 480-620 bytes, GPU 540 (fine: 3.6 GB of GPU memory, the float64
  CG vectors over the whole box are most of it). The node frees CuPy's pool afterwards.
  Since 2026-10-19 (see "FEA solver speed-ups"): GPU medium 1.2 s, fine 4 s; CPU draft 2.8 s, medium 16 s.
- Artificial peaks at supports: the bracket's max stress was 12.8 / 14.0 / 21.8 MPa at draft /
  medium / fine, at the edge of a held bolt hole (perfectly rigid faces make a singular stress
  there). The report and the colour scale use the highest stress more than 2 elements from fixed
  nodes (11.6 MPa at medium, next to a bolt hole, where the section is smallest) and mention the
  peak separately.
- In the editor: a medium analysis of the bracket took 5.5 s on the GPU, colours with units and
  hover values, supports and loads drawn over the part (they sit inside holes), safety factor in
  reversed colours. The turbo map's near-black ends made safe areas look black: results use a
  blue-to-red "rainbow" (turbo without its ends).

## Topology optimization (FEA phase 2, 2026-10-14)

`engine/fea/topology.py`, node Topology Optimization. Test parts: a 60 x 20 x 10 mm cantilever box
(probe) and 30 x 10 x 10 (tests); `samples/wall_bracket_space.step`, a 60 x 30 x 50 mm block glued
on its back face with 200 N down on a pin hole near the far top edge
(`recipes/topo_wall_bracket.json`).

- Method: SIMP (penalty 3, empty = 1e-6 E), cone density filter (radius min thickness / 2, at
  least 1.5 elements), Heaviside projection (beta 1 -> 16, doubled when the compliance settles or
  after 25 iterations), optimality criteria with the projected volume held by bisection. Holds
  Keep % to 0.1%, ends nearly black and white (0.02% grey on the cantilever), the cantilever shows
  the textbook flanges and web.
- "No density changes more than 0.01" never happened: a few elements flip back and forth for ever
  (change stuck at the move limit while the compliance had stopped moving at iteration ~105 of
  125). Stopping and sharpening when the compliance changes < 0.1% over 5 iterations: 68
  iterations, compliance 242.6 instead of 242.1.
- Inner solves to 1e-2 (warm-started from the last displacement) gave the same design as 1e-5 and
  1e-3 (compliance 242.1 all three) at 1.1 instead of 2.8 s per CPU iteration.
- Coarsest multigrid level: with up to 4000 elements a small grid (30 x 10 x 10) had no multigrid at
  all and SuperLU refactorized 11k unknowns every iteration: 1.0 s, the tests took 118 s. Now at
  most 500 elements (375: 0.012 s) and minimum-degree ordering on A + A^T (3x faster than COLAMD on
  thin grids): tests 7.5 s. Static analysis results unchanged.
- Density to body: 0.5 surface on a 2x finer trilinear grid, 10 Taubin passes, pieces under 1% of
  the largest dropped, the exact distance field of that mesh, intersected with the design space.
  The FE grid starts at the part's lowest corner, so the densities are padded and extended 2
  elements outward first: otherwise the surface sat exactly on the clamped face and the crop left
  it 0.005 mm short instead of giving back the CAD face.
- Times (fresh cache, through the node):

  | case | design elements | GPU | CPU |
  |---|---|---|---|
  | cantilever probe, draft | 39k | 29 s (68 it.) | 91 s |
  | cantilever probe, medium | 240k | 107 s | - |
  | wall bracket, draft | 32k | 29 s (80 it.) | ~80 s in the editor |
  | wall bracket, medium | 241k | 146 s (79 it.) | - |

  An optimization is ~70 solves, so the node uses the GPU from 10k grid elements (one analysis
  from 60k). The GPU memory pool is released at the end. Since 2026-10-19 about 2x faster (see "FEA
  solver speed-ups").
- Wall bracket result: back plate, wide upper flange and a strut to the pin; 25% of the block kept,
  stiffness 26% of the full block (medium: 25%). At draft the flat top is a little wavy.
- In the editor: Optimize queues like an export (label "optimize <node>", the queue button now says
  Jobs), the shape is drawn every 3 iterations (raw 0.5 surface), Stop saves the state (every 5
  iterations anyway) and the stopped shape shows; Optimize continued at iteration 30 -> 80. States
  live in `cache\topology` (keyed by every input), so a finished result loads after a restart
  without running again. The hidden mesher's events used to overwrite a node's status (note and
  range tag) when its body was shown; they keep them now.

## Max thickness (2026-10-15)

The user wanted designs that look better rather than the stiffest: thick members split into thin
branches.

- Local volume constraint (Wu et al. 2018): every neighbourhood at most `LOCAL_SHARE` (0.6) full.
  Gaussian neighbourhoods (separable, cheap at any size), sigma = max thickness / 1.68 (a plate of
  that thickness fills 0.6 at its middle), aggregated as a p-mean (p = 16) scaled adaptively to
  the maximum; only free elements count. Two constraints need a two-multiplier update:
  OC x * sqrt(-dc / (l1 dv + l2 dg)), l1 for the exact projected volume, l2 for the linearized
  local constraint, found by nested Illinois root finding (~60 evaluations an iteration; nested
  bisection needed ~300). A linearized volume ended 3.5 points under Keep %.
- Wall bracket, medium, GPU: without, thickest member 10.0 mm (99%), 25% kept, stiffness 25%;
  max thickness 4 mm: 4.3 mm, fullest neighbourhood 0.61, but only 21.5% kept (stiffness 18%):
  25% does not fit in that block with no neighbourhood over 60% full. The max thickness wins and
  the node says so; the demo `recipes/topo_wall_bracket_branchy.json` keeps 20%.
- The look: the single truss web becomes a branchy web of thin struts, with openings in the side
  plates. At draft (1.4 mm elements) a max thickness under ~6 mm cannot be resolved (it must be
  1.5 x the min member of 3 elements): it is raised with a note.
- Smoothing: the result's flat faces showed terraces where a plate's thickness changed by one
  element. A Gaussian blur of the densities (sigma 1 element) before taking the 0.5 surface turns
  them into smooth waves (volume -1.8%); the live view stays raw.
- GPU products in batches of 256 MB instead of 64: a V-cycle on a full 96 x 48 x 80 grid 212 ->
  54 ms, the medium wall bracket 150 -> 54 s (with max thickness 404 -> 143 s). Static analysis
  of full parts gains the same.
- The topology fingerprint only includes the max thickness when it is set, so results from before
  stay valid.

## Bearing load, and the big flat member (2026-10-15)

The user asked why the wall bracket kept a big flat member (its top flange) and for a contact
load: a pin pulled down presses only on the bottom of its hole, not all round it.

- Bearing load (Load kind `bearing`, as SolidWorks'): pressure square to the face, cosine
  weighted (largest where the face looks straight against the force, 0 at the sides), none on the
  far side, scaled so the resultant is the force: p0 = F / sum(area w^2). On the 8 mm pin hole,
  200 N -Z: resultant within 1% of (0, 0, -200), nothing above the hole's middle. Needs an axis
  or custom direction; the markers show only the contact side.
- Medium, GPU, Keep 20%, PETG, bearing 200 N on the pin (`docs/topology_bracket_variants.png`):

  | design space | variant | iterations | time | kept | stiffness | thickest |
  |---|---|---|---|---|---|---|
  | block, back face glued | no max | 80 | 53 s | 20.0% | 19% | 9.8 mm |
  | | max 4 mm | 95 | 100 s | 19.1% | 16% | 4.3 mm |
  | | max 4 mm, struts only | 125 (cap) | 220 s | 17.3% | 12% | 4.3 mm |
  | block, 4 bolt holes | no max | 81 | 57 s | 20.0% | 25% | 7.1 mm |
  | | max 4 mm | 110 | 109 s | 17.7% | 20% | 4.0 mm |
  | | max 4 mm, struts only | 125 (cap) | 136 s | 15.2% | 18% | 3.2 mm |

- The flat flange came from the setup: with the whole back face glued and the pin through the
  full width, every cross-section is loaded alike (a nearly 2D problem) and the stiffest answer
  is extruded plates. A thickness limit cannot forbid a wide plate; "struts only" (Max thickness
  measured on round struts, so plates at most 0.62 of it, members at least 0.65 of it) still left
  a perforated top plate and did not settle in 125 iterations.
- Held by 4 M5 insert holes instead (`samples/wall_bracket_bolts.step`, demo
  `recipes/topo_wall_bracket_bolted.json`): four organic arms from the bolts to the pin, no
  flange, and stiffer at the same mass (25% vs 19%). With max thickness 4 mm the arms split into
  finger-like branches; struts only gives a web with holes but is softer and slower.
- Lesson for the user: hold the design space where the real part is held (bolt holes, clamp
  faces) and load it the way it is loaded (bearing for pins); whole glued faces and loads spread
  over whole holes make plate-like designs.
- The pin eye stayed a whole tube even with the bearing load: Skin kept 2 mm solid around every
  picked face. For a bearing load the skin now covers only the contact side (where the pressure
  is > 0); whether the top of the eye is needed is up to the optimizer (a Keep body closes it if
  wanted). Topology keys with a bearing load got a marker, so older results are not reused.
- Aiming a load: with a Load node selected the 3D view draws a yellow arrow at its faces (from the
  centre of its faces plus Offset, 30% of the scene long; a bigger invisible ball around the head
  is the grab target). Dragging the head moves it on a sphere (pointer ray against the sphere, or
  its closest point when the pointer is off it), snapping within 6° to the axes, 45° diagonals and
  cube diagonals; on release Direction becomes that axis, or custom with the Vector. Esc cancels;
  camera rotation is off while dragging.

## Torque and offset forces (2026-10-15)

- One mechanism for both: a moment M on the picked faces as a "turn" w x r about their area
  centre c, w = I^-1 M with I the faces' area inertia tensor about c (exact per flat triangle:
  int r r^T dA = A/12 (sum v v^T + s s^T)). It adds up to no force and exactly M; it is SolidWorks'
  rigid remote load / a distributing coupling. On a full hole about its axis it is the usual
  tangential torque, the same size all round; on bolts on a circle each gets a share by radius.
- Load kind `torque`: N·m about Direction (right-hand rule). "into the face" on a flat face = the
  axis into it (driving a screw in); on a round face it is refused (no single normal), the user
  picks an axis or drags the arrow. 2 N·m on the bracket's pin hole: no net force (1e-11 N),
  moment 2000 N·mm, tangential within the facets' 3°.
- Offset (force and bearing): the force acts at c + offset, so the faces carry the force plus
  offset x F. Keys of loads without an offset are unchanged (results stay valid).
- Checked against textbooks (100 x 10 x 10 PLA beam, 1 mm elements): 100 N with a 50 mm offset
  beyond the free end, tip deflection 1.0% under Timoshenko + end moment (the plain cantilever is
  1.1% under); 5 N·m torque, twist rate between 10 and 90 mm 0.9% under Saint-Venant (J = 0.1406
  a^4), 0.2% at 0.5 mm elements. Measured on the top face's middle line, where the square section
  does not warp.
- The reaction check divided by the net load, which is ~0 for a torque: it now divides by the
  larger of the net load and a tenth of the summed nodal force sizes (unchanged for plain forces).
- 3D view: a torque draws an arc with an arrowhead round its axis arrow; a force or bearing load
  has a ball at the arrow's tail that drags the point of action in the plane facing the camera
  (rounded to 0.5 / 1 / 5 mm by scene size), with a thin line from the faces' centre. The markers
  carry `centre` and `arrow` (the force's direction or the torque axis) for this.
- Bolted wall bracket, medium, GPU, PETG, Keep 20% (`docs/topology_bracket_torque.png`): bearing
  200 N -Z, max 4 mm: 106 it, 100 s, 17.4% kept, stiffness 22% (an open cradle under the pin);
  torque 5 N·m about the pin axis, max 4 mm: 123 it, 87 s, 19.0%, 38%; torque, no max: 82 it,
  39 s, 20.0%, 45%. Under torque the optimizer wraps the pin in a closed thin-walled tube, the
  textbook answer (a closed section is far stiffer in torsion than an open one), tied to the bolts
  by arms above and below. The size of the torque does not change the shape (one load case).

## Smoother surfaces (2026-10-15)

The user asked how to make the optimized parts look better. The visible flaw was rings (terraces)
on gently sloped faces: an optimized result is nearly black and white per element, so where the
surface lies inside an element is lost, and the 0.5 level follows the element steps.

- Tried on the bolted bracket (medium, bearing, max 4 mm; `docs/topology_smoothing.png`):
  60 instead of 10 Taubin passes: no visible change. Blur 1.5 instead of 1 at level 0.5: most
  terraces gone but 5% of the volume lost. Smoothing the signed distance while keeping it within
  half an element of where it was: worse (the terraces stay inside that band). Blur 2 with the
  level chosen to keep the volume: smooth, rounded arms, same mass (15.3 vs 15.2 cm³ cropped).
  Blur 2.5+: soft, small dimples appear.
- Node input Smoothing (elements, 0-5; default 2 at first, back to 1 since: see "Smoothing 2 was
  weaker"); `density_mesh` picks the level so the
  samples above it inside the design space match the densities' volume there (clamped to
  0.3-0.7). The count must be a share of the design space's samples: the upsampled grid has
  slightly more than up^3 samples per element (it asked for 27% instead of 30% on the test beam).
  The level must also be matched inside the design space only: the extension's outer edge loses
  volume when blurred, and matching the total moved it into the part (28% -> 43% on the beam).
- The extension past the design space grew by dilation, which also spread material sideways along
  the outside; with 3-4 elements (needed for the blur) it closed the test beam's open sides into a
  cavity. Now each outside element takes the density of the nearest inside one (EDT indices).
- Fine detail on the same setup: 0.45 mm elements, 111 iterations, 279 s on the GPU (medium:
  100 s), 17.1% kept, stiffness 21%, thinner branches and an extra opening in the side plate.
  Smoothing and fine detail add up: Smoothing for any result, fine for the final design.

## Gap to faces, and material leaking into Avoid (2026-10-15)

The user asked for an option that keeps the shape off the design space's box: the bracket's top
and side plates lay flat against it (the stiffest place for bending material), and the crop by the
design space then cut them flat.

- Node input Gap to faces (mm, 0 = off): elements whose centre is less than the gap inside the
  design space's faces (its signed distance) are kept empty, except within skin + gap of the faces
  of supports and loads (all their faces, also a bearing load's far side, so the optimizer may close
  a pin eye) and within the gap of Keep. Keep % stays a share of the whole design space (`total`
  counts the gap), and "stiffness % of the full design space" still compares with the whole design
  space filled (`full_phi`), so a gap's cost shows. Keep % above 97% of what is left is refused.
- First look (bolted bracket, medium, bearing, max 4 mm): gap 3 mm, 17.0% kept, stiffness 22%,
  "free rounded arms"; 6 mm: 16.3%, 20%. That look was false: see "Smoothing 2 was weaker" below.
  On the 10 mm test beam a 2 mm gap cost 2.5x the compliance (its outer 2 mm are where bending
  material helps most).
- Bug found on the way: `density_mesh` extended material past "phi > 0", but Avoid regions (and now
  the gap) have phi 0, so the extension filled them from the side, 2-4 elements deep, and the crop
  by the design space did not remove it. The state now saves `space` (design space and Keep, before
  Avoid and the gap) and the extension goes only outside it; the result is also cut by the Avoid
  body exactly. Older saved states have no `space` and fall back to phi > 0.

## Smoothing 2 was weaker; what the gap really does (2026-10-15)

The user saw short sharp stubs and found the new versions "look much weaker". They were: checked
with Static Analysis of each cropped surface (bolt holes held, bearing 200 N -Z, medium, PETG;
max deflection, `scratchpad/analyze_surfaces.py`):

  | design | plain 0.5 surface | Smoothing 1 | Smoothing 2 | filtered field x~ at 0.5 |
  |---|---|---|---|---|
  | no gap | 0.199 mm | 0.200 | 0.199 (10.7% of the solid moved) | 0.199 |
  | gap 3 mm, first version | 0.212 | 0.220 | 0.265 (25% weaker, 14% moved) | 0.204 |
  | gap 3 mm, fixed | 0.177 | 0.174 | 0.183 | |
  | gap 3 mm, struts only, fixed | 0.212 | 0.215 | 0.258 (21% weaker) | |

- Smoothing 2 blurs a 3-element member to a peak of 0.25, below any level: thin members vanish
  and leave stubs where they joined. It also erased the thin plates the gap made, which is what
  had looked like "free rounded arms". Default back to 1 (at most 1-4% here).
- The gap does not remove plates: the first layer past it was 38% solid (1-5% in the gap): the
  plate just moves in to the gap's edge, the outermost place left. There it was 1-2 elements thin,
  because the density filter is renormalized at domain edges. Fixed: the filter now averages over
  the gap as over empty material, so members along it are as thick as elsewhere (gap 3 mm then:
  112 it, 65 s, 18.4% kept, stiffness 24%, a closed shell 3 mm inside the box).
- Plates come from the amount of material: at Keep 20% a wide bracket's stiffest answer is a
  shell. Keep 10% (max 4 mm): 101 it, 92 s, stiffness 10% at half the mass, four clean arms from
  the bolts to the pin and no plates; with gap 3 mm (91 it, 63 s) the arms also keep off the side
  faces (`docs/topology_gap.png`). Struts only with a gap gave a perforated shell, not branches.
- Lesson: judge a surface change with an analysis, not a picture; a blur that "looks organic" can
  be deleting material.

## FEA solver speed-ups (2026-10-19)

A review of the simulations for speed, then the changes that keep the results. Cases: static
analysis of the bracket (`samples/bracket.step`, PETG, bolt holes held, 50 N +X on the pin hole) and
topology of the bolted wall bracket (`samples/wall_bracket_bolts.step`, PETG, Keep 20%, bearing
200 N -Z); warm runs, the user's editor not running.

| case | before | after | result |
|---|---|---|---|
| static, medium, GPU | 2.9 s (33 iterations) | 1.2 s (24) | max stress and deflection the same to every printed digit |
| static, fine, GPU | 9.3-10.9 s (35) | 3.9-4.2 s (29) | same |
| static, draft, CPU | 9.7 s (28) | 2.8 s (28) | same |
| static, medium, CPU | 64 s (33) | 16 s (24) | same |
| topology, draft, GPU | 20.9 s (92 iterations) | 11.2 s (93) | 0 elements flipped, compliance +0.003% |
| topology, medium, GPU | 49.6 s (84) | 20.2 s (80) | 79 of 50k solid elements flipped (surface), +0.06% |
| topology, draft, CPU | 105 s (92) | 46.5 s (93) | 0 flipped, +0.003% |

- **The GPU's float32 multigrid ran in float64.** lmax came from `np.sqrt`, a NumPy float64 scalar,
  and NumPy 2 / CuPy 14 turn a float32 array into a float64 one when such a scalar takes part: from
  the first smoothing step on, the "float32" preconditioner worked in float64 (consumer GPUs do
  float64 at 1/64 the rate). lmax as a Python float alone: the old code's GPU solve 1.5 -> 1.0 s at
  medium, 4.9 -> 2.0 s at fine, same iterations and results. Scalars in the multigrid are Python
  floats now, its updates in place (they keep the type), and a test checks the preconditioner's type.
- CG tested convergence only after the next V-cycle: one wasted per solve. Topology solves to 1e-2
  take ~3 iterations: 339 -> 247 V-cycles on the draft wall bracket, the same design.
- Topology: the smoothers' lmax estimates start from the last iteration's power vectors (3 steps
  instead of 12 from a random vector) from iteration 4 on: 67 -> 15 ms per iteration at draft,
  180 -> 28 at medium (GPU). They came out 0.995-1.27 x a fresh estimate on every level, never too low
  (the earlier review saw CG blow up when they started in iteration 1). A warm solve that needs more
  than 100 iterations would be repeated with fresh estimates (never happened). Unfinished states keep
  the vectors (a resumed run continues exactly; medium checkpoint 13 -> 18.5 MB, 22 ms). This is what
  moves the design a little (the table); TOPO_VERSION is not bumped, saved results stay valid.
- Static analysis: the coarsest multigrid level may have up to 3000 elements (`MAX_COARSE_STATIC`;
  topology keeps 500, it refactorizes every iteration): 33 -> 24 iterations at medium, 35 -> 29 at
  fine; the grid is padded less (160x88x128 -> 144x88x120 elements at medium).
- Products: the first coarse level's 8 child products as one wide matmul and an einsum (CPU: 30 ->
  11 ms for 34k elements); `take` instead of fancy indexing for the element gathers; an optimized
  einsum for the stored coarse levels (CPU: hierarchy 8.6 -> 1.9 s at medium; topology 3.7 -> 1.9 s
  per iteration at medium).
- CPU: static analysis uses the float32 multigrid too (medium 25 -> 16 s, same answers). np.bincount
  converts float32 values to float64, which cost more than the sums (34 of 64 ms per product): a 0/1
  CSR matrix per element batch scatters them (13 ms, 20 ms to make once). Float64 levels (CPU
  topology) keep bincount, bit-identical.
- Outside the solve (fine, GPU: model and post-processing were 3.4-5.3 of 9.3-10.9 s, now 2.0 of
  4): `apply_conditions` built (points x 8 x 3)
  arrays for 1.25M face samples; per-axis weights and flat node numbers: 1.2 -> 0.56 s, bit-identical
  (checked on 9 cases). `_neighbour_fill` and `nodal_average` without repeated products, one
  displacement norm instead of two: post-processing 2-2.9 -> 1.3 s.
- Displacements on the SSD are kept for E = 1 under part, conditions, Poisson's ratio and detail: a
  new density, strength or E reuses them (they scale with 1 / E; stresses do not depend on E).
- Max thickness: the root search for l2 kept the x of its result, not found again: 8.4% fewer
  evaluations, bit-identical over 70 iterations.
- Live view while optimizing: `topology.LiveSurface` (the extension's nearest elements found once
  per run, pieces by vertex labels instead of mesh splits): 135 -> 50 ms at medium, 33 -> 12 ms at
  draft, the same triangles (specks dropped as before).
- Where the time goes now: static fine on the GPU (4 s) = model 0.7, hierarchy 0.5, solve 1.45,
  post-processing 1.3 (host); topology medium on the GPU (0.25 s per iteration) = solve 176 ms
  (V-cycles 146), hierarchy 47 (lmax 28), OC 18.
- Not done, as they change results (the user's call): CG tolerance 1e-4 for static analysis,
  EIG_RATIO 30, a looser topology tolerance in the first iterations, OC brackets started from the
  last iteration's multipliers, another continuation schedule (BETA_EVERY, PLATEAU).
- Found on the way: `topology_key` includes density and strength, which do not change a design:
  editing them optimizes again (changing the key would orphan saved results).

### Sampling the faces of loads and supports

`sample_triangles` split every CAD triangle into m x m equal ones, m from its longest edge. A hole
is tessellated into long thin strips (the wall bracket's pin hole: 30 x ~2 mm), so a strip got m^2
points where its width needs a few: the pin hole 2.0M points at medium (4.0M, the cap, at fine)
where ~31k would cover it evenly, and `apply_conditions` took 1.3 s at medium, 2.6 s at fine.

- A remesh that keeps the faces (halving each triangle's longest edge until it is short) does not
  help much: the halves of a thin triangle are thin again (0.36M points; trimesh's
  `subdivide_to_size` 0.43M). What works is a grid stretched to each thin triangle: rows from the
  corner opposite its shortest edge, a spacing apart, and points across each row. It is used where it
  needs at most half the points of the m x m split; other triangles are sampled as before.
- With one point per spacing across, the stress next to the loaded pin hole came out up to 0.9% off
  a sampling 8x finer (the old split: 0.12%, being dense across the strips). Four per spacing across
  (`ACROSS`): within 0.12%; the pin hole then gets 112k points at medium, 279k at fine.
- Supports must keep the old split. Which nodes a support fixes (corners within 0.8 h of a face
  point) is a coverage test, and it only settles at a spacing of h/20-h/40: with stretched grids the
  bracket at fine lost 370 fixed nodes and its stress away from the supports dropped 7%. So supports,
  topology's skin and the gap use `stretch=False` and are bit-identical to before. Even the old sets
  are not settled: at medium the bracket's bolt holes miss 171 nodes (8%) that the rule names, the
  wall bracket's 234. An exact test (does the face cross the box of 1.6 h around a node) would make
  supports independent of the tessellation, but moves stresses near supports by several percent: not
  done.
- Result (GPU, supports unchanged): the bracket's results unchanged to every printed digit; the wall
  bracket's stress away from supports within 0.07%, deflections within 0.002%. Wall bracket
  `apply_conditions` 0.37 -> 0.06 s at draft, 1.27 -> 0.20 at medium, 2.63 -> 0.52 at fine; the whole
  analysis 0.85 -> 0.45 s, 2.3 -> 1.2 s, 5.4 -> 3.1 s. Topology (bearing load on the pin hole):
  prepare 5.1 -> 3.75 s at medium; the medium design identical (0 elements flipped), draft 4 of 8,279
  (compliance -0.025%). FEA_VERSION and TOPO_VERSION not bumped.

## Fused GPU kernels for the solver and the OC update (2026-10-23)

`engine/fea/gpu.py`, switched by `solver.GPU_KERNELS` (False: the CuPy code, kept as the reference and
for the comparisons in `tests/test_gpu.py`). Case: topology of the bolted wall bracket (Keep 20%, Max
thickness 4, bearing 200 N), GPU, the user's editor running idle (4.7 GB of GPU memory free).

- **Where the time went** (draft, 413 ms an iteration, 41k design elements): solve 270 ms, OC update
  125, hierarchy 47. A product took 1.2 ms (~190 an iteration) on a grid where the arithmetic takes
  ~20 us: ~50 CuPy kernels (8 gathers, matmul, 8 scatters, mask), each ~20 us to launch from Python.
  The OC root finding tried ~60 multipliers an iteration at ~20 kernels and a filter each.
- **Products**: a thread per node sums its 3 rows of the 8 elements around it (fine: per[e] KE,
  children: sum_c per[e, c] M_c, stored: per[e]; element-list levels through a cell -> row map).
  Deterministic (no atomics); the same FLOPs as the element-wise matmul. The smoother's vector updates
  and the residual are fused into the product kernel (`mode`), restriction and prolongation (with
  the masks and the correction added) are one kernel each: a V-cycle level is 9 launches.
  Against the CuPy code on the bracket's hierarchies (draft and medium, every level kind) and a full
  box: products, smoothing and V-cycles within 1.6e-7 (float32 rounding), the float64 fine product
  within 1e-16, solves to 1e-6 the same iteration counts, displacements within 3e-12.
- **OC update** (`gpu.Volume`): one kernel for the design at (l1, l2), one for the filter (only the
  cone's nonzero taps, about half), projection, solid regions and phi, summed per block of 256 in
  float64 by Warp tiles and added up on the host: deterministic. Elements outside the design space
  and solid ones skip the convolution. Big struts-only kernels keep the FFT, then the same kernel
  without convolving. Densities within 3e-6 of the CuPy update, volumes equal to 1e-6. Per value
  tried: draft 0.62 -> 0.20 ms, medium 0.58 -> 0.22 ms; per update 77-88 -> 19-22 ms (Max thickness),
  25 -> 7 ms (without).
- **Launch costs** (measured, draft-sized arrays): a CuPy op ~20 us, a Warp launch with arrays
  wrapped per call (`from_dlpack`) 128 us, `wp.array(ptr=)` 51 us, raw pointers built into arrays
  inside the kernel 40 us, a recorded `wp.Launch` with parameters set again 19 us (launch 12.6 us,
  `set_param_at_index` 1.5 us each: only changed ones are set). Warp on CuPy's stream (a `wp.Stream`
  around CuPy's current stream, also the default one, pointer 0) needs no synchronizing; the
  existing Warp code (`sdf_gpu`, `segments`) synchronizes the device around each launch instead.
- **Compile time**: Warp unrolls loops with constant counts. With the 2 x 2 x 2 element loop and the
  8 children unrolled around the 24-term sums, each product kernel took 60-82 s to compile (4.5 min
  the first time). The counts are now passed in (`eight`): ~1 s per precision, run time the same.
  Unrolling the 24-term loops too (`max_unroll` 24) compiled 3x slower and ran no faster.
- **Results**, iterations from the start (median ms per iteration, CuPy -> kernels): draft 412 -> 92
  (4.5x), medium 879 -> 154 (5.7x). 40 draft / 30 medium iterations: compliance within 8.5e-7 /
  3.1e-6, densities within 2e-6 / 1.8e-5, no element crosses 0.5. Draft per iteration after: solve
  52 ms (V-cycle 33.7 -> 3.7 ms), hierarchy 33, OC 20. Whole optimizations (to done, prepare
  not counted): draft 125 iterations 27.1 -> 7.8 s (3.5x; later iterations are cheaper on both),
  medium 89 iterations 62.7 -> 12.4 s (5.1x); no element crosses 0.5, final compliance within 3e-7 /
  8e-8, the largest density difference 2e-5 / 8e-4.
- **In the editor** (test editor on 8766, Optimize on `topo_wall_bracket_bolted.json`, medium, each run
  with its own empty `DESIGN_ENGINE_CACHE`; `DESIGN_ENGINE_FEA_KERNELS=0` for the CuPy run): "done in
  75 s" -> "done in 20 s", the whole job (with the check of the shown body) ~91 -> 30.7 s; 0.7-0.9 ->
  0.1-0.2 s per iteration in the log. Both: 89 iterations, the same progress lines, the same saved
  result key, design compliance 27.9412, the same card note (15.8% kept, members thickened to 2.1 mm,
  safety factor 6.96, max deflection 0.167 mm), no element across 0.5, the shapes alike on screen.
  Seen on the way (not from the kernels): after the server restart the tab reopened its last recipe,
  and after Optimize the preview stayed at Draft (run tag version 4, `graphVersion` 2, so no refine)
  until Update was pressed; opened with Open... it refined to Final by itself.
- **A reference cycle held GPU memory** (found by the user the same evening): a level kept its
  kernels' data (`Level._gpu`) and that kept the level, so every iteration's hierarchy waited for
  Python's cycle collector: at medium up to 47 levels alive, 383 MB in use (CuPy code: 21 MB), 196 MB
  still after the run. After a fine Optimize (Start coarse, gap 0.8 mm) the result's check, Static
  Analysis of the shown body at fine (5.5M box elements), wants 5.5M x 600 B + 3 x 256 MB batches +
  the 1.5 GB headroom = ~5.6 GB free on the card, about all there is next to other programs (6.1 GB
  idle): it went to the CPU, which said "needs about 4.1 GB ... but only 4.1 GB can be used" (the
  editor held 6.2 of 7.0 GB). `LevelKernels` refers to its level weakly now: 21 MB in use, 5 MB after
  the run (`test_gpu_solver_kernels_let_their_hierarchy_go`). A fine check stays close to the limit
  either way.
- **Leak check** (one process doing Optimize 3-8 times in a row through the evaluator, each with
  another Keep %, then Static Analysis 8 times): no GPU leak (pool in use ~0 after each run, GPU free
  only wanders with other programs). Another cycle, older: the coarsest level's direct solver (a
  closure) referred to its level, so every hierarchy's coarsest level and SuperLU factors waited for
  gc (one more live level per analysis); fixed. RAM grows ~35-40 MB per Optimize with the kernels and
  without (tracemalloc's own bookkeeping made it look like 120): the mesh caches bounded by count, not
  size, `sdf._QUERIES` (24 libigl trees) and `sdf_gpu._MESHES` (48 Warp meshes), get a new entry for
  every new result shape, ~25 MB each at medium (arrays alone, trees and BVHs on top).
- **GPU memory of a fine check** (Static Analysis, sampled NVML low point): bolted bracket space 1.33M
  elements 548-622 MB, key box 5.3M elements 2.1 GB, 413-491 bytes per box element in all, against
  `_array_module`'s 600 per element + 768 MB of batches (3.8 GB + 1.5 GB headroom for the key box: it
  goes to the CPU whenever less than 5.3 GB is free, though 3.6 GB would do). The kernels first used
  10% more than the CuPy code: the exact float64 level got float64 copies of `free` and `inv_diag`;
  the mask is float32 in both precisions now and apply does not take inv_diag: 2,092 MB, as CuPy.
  A fine topology result (bolted bracket, gap 0.8 mm, as the user's) analysed at fine: 6.3M box
  elements, 2.47 GB (412 B each). Topology itself: bolted bracket medium 0.37M elements 268 MB, fine
  1.33M 795 MB, key box medium struts only (FFT filter) 1.35M 922 MB: ~65 MB + 550-630 B per element.
  Now `nodes_fea.gpu_bytes`: 128 MB + 470 B per element (Static Analysis), 720 B (topology), the old
  768 MB of batches only with the kernels off.
- **The pool hid free memory too**: right after a fine Optimize the result's check saw 4.0 GB free
  while this process's CuPy pool held 615 MB of unused blocks (`_optimize` released the pool while
  the last level's arrays were still alive). `_array_module` releases the pool before it reads the
  free memory. End to end (the user's case: bolted bracket, fine, gap 0.8, max 4): the check of 5.9M
  elements, estimated 2.8 GB, ran on the GPU in 11 s (safety factor 7.28, 0.147 mm); before, it went to
  the CPU and failed there ("needs about 4.4 GB ... only 1.2 GB can be used").
- **Idle workers held the check's memory** (the user's editor, a medium Optimize after a Final preview):
  8 workers stayed alive for the next preview, 5 of them on the GPU at ~650 MB of RAM each plus a CUDA
  context (5.4 of 7.2 GB in the toolbar, the card 8.0 of 12 GB used); the optimized part's check (6.4M
  box elements at medium: the shape fills little of its box) needed 4.6 GB of GPU with the headroom,
  4.07 were free, and 4.8 GB of RAM on the CPU, 1.7 left. The watchdog stops idle workers only when
  memory is tight (85% of the limit, < 3 GB free): it was 75%. Now `_array_module` stops the run's idle
  workers when an analysis fits neither on the GPU nor in RAM as things are (`core.stop_idle_workers`,
  a line in the log) and decides again (waiting up to 2 s for the GPU memory). Measured: 5 idle GPU
  workers held 3.8 GB of RAM and 1.2 GB of GPU, 3 held 2.5 GB and 0.67 GB; with 3 workers and 2.5 GB
  of headroom the check of 5.9M elements stopped them and ran on the GPU (safety factor 7.28).
- **Mesh caches bounded by size**: `sdf._QUERIES` at most `MAX_QUERY_BYTES` (768 MB; `MeshQuery.nbytes`
  = arrays + `TREE_BYTES` 450 per triangle per search tree; measured 345-580 B per triangle in all,
  the winding-number tree 70-1100 more) and `sdf_gpu._MESHES` at most `MAX_MESH_BYTES` (512 MB of GPU;
  measured per triangle after splitting to 2 mm with Warp's memory pool off: 166 B, 610-650 B with the
  winding-number data of a mesh with cracks), the newest entry always kept, the count caps too. A CAD
  part with big flat faces splits into many more triangles (test part 10k -> 176k: 10 KB of GPU per
  original triangle, key box 717 B), where `tiles.worker_memory` counts 150 B: not changed.
- **Saved optimization states**: one file per topology key, overwritten at every checkpoint (5
  iterations), Stop, level hand-over and the end; the history (4 numbers per iteration) is all that
  is kept of earlier iterations. Fine states ~50 MB, mostly the float64 displacement (kept so a
  stopped run continues exactly; a finished one does not need it). 24 states, 821 MB on 2026-10-23
  (cap 2 GB, oldest first).
- Static analysis of the bracket (results the same to every printed digit, same iterations): draft
  0.57 -> 0.32 s, medium 1.29 -> 1.09 s, fine 4.49 -> 4.03 s: there the model and post-processing are
  most of the time, and big grids already ran in 256 MB CuPy batches.
- Left as it was: the hierarchy build (diagonals by the CuPy scatter path, coarse-level einsums,
  the coarsest level factorized on the CPU every iteration), sensitivities and filters outside the OC
  update, and the CPU path. Thresholds for the GPU (`GPU_MIN_ELEMENTS` 60k, `TOPOLOGY_GPU_MIN` 10k)
  were set before these kernels and may now be lower.

## Min thickness, skin and load arrows (2026-10-20)

The user's key box (`nyckellada.STEP`: a 157 x 28 x 217 mm tray with ~5-6 mm walls; Min thickness 3,
Max thickness 15, struts only, medium; four 4 mm boss faces carry the support and three forces, a
200 N bearing load in -Y sits on the whole front rim, 9,670 mm2): "Min thickness is not honoured,
some parts look too thin", and the arrows looked wrong.

- Measure on the body the node shows (the density mesh cut by the design space, meshed as the
  Final preview), never on the density mesh: that is extended past the design space, so every
  member touching a face looked thick there (0.6% under 3 mm) while the shown body had 66%. Voxel
  medial-axis measures were no better: every stair-step bump counts as a thin ridge.
- Why: a cone filter of radius r and a projection at 0.5 do not stop members thinner than 2r (a
  ridge of the filtered density just over 0.5 is a member). With 1.5 elements (Min thickness 3 at
  0.96 mm) the key box got members of 1-2 elements. Now the stiffness is that of the eroded design
  (projected at ERODED 0.8): a plate needs a filtered core of 0.8 to carry, which with the cone filter
  needs a plate at least the radius thick (struts 1.3 x; checked on the voxel kernel), so the radius
  is now the min member, not half of it. Members too thin for that cost volume and give nothing.
- The eroded design alone broke apart when Keep % left no room for members that thick (30 x 10 x
  10 box at 20%: compliance 167,351). What erosion removes still carries THIN_SHARE 0.15 of its
  density (0.3% of its stiffness): connected again (compliance 70), and thin members still do not pay.
- The stiffness shown is the design's own (one extra solve at the end, every 10 iterations for the
  log): the eroded design is much softer (key box struts only: 59% vs 76%).
- The eroded objective creeps for long: runs often reach the 125-iteration cap; 200 iterations gave
  the same design (compliance 34.64 vs 34.67), so the cap stays.
- Skin: the bearing load's 2 mm skin over the rim was nearly all that stayed thin. The skin is now
  never thinner than a Min thickness that is set.
- Max thickness counted a neighbourhood as a share of the design space, so a solid 6 mm wall read as
  100% full whatever the max thickness and every wall got holes. Outside the design space now counts
  as empty (as it is in the part).
- Struts only now really asks for members of 0.65 x Max thickness (at 15 mm: a 21^3 kernel). Direct
  convolution made an iteration 3x slower; filters of 1000 taps or more go through an FFT (14.6 -> 0.9 ms
  a filter on the GPU, ~100 filters an iteration in the root finding).
- Medium, GPU, shown body (`under` = share of its surface thinner than the min member, `away` = of
  that, away from the support / load skins):

  | case | method | iterations | time | stiffness | under | away |
  |---|---|---|---|---|---|---|
  | key box, user's settings | old | 125 | 127 s | 81% | 66% | ~32% |
  | | new | 110 | 80 s | 76% | 6.4% | 0.8% |
  | key box, Min 3 only | old | 54 | 30 s | 93.5% | 78% | 42% |
  | | new | 88 | 47 s | 86.8% | 26% | 15% |
  | wall bracket, 25% | old | 68 | 23 s | 27.3% | 13.7% | 1.1% |
  | | new | 116 | 30 s | 26.4% | 11.4% | 0% |
  | bolted bracket, 20% | old | 80 | 26 s | 26.4% | 16.9% | 1.1% |
  | | new | 123 | 33 s | 25.2% | 17.0% | 0% |

  What is left under 3 mm with Min 3 only is 2.2-3 mm (old: down to 1 mm): the blur of Smoothing,
  and feather edges where members meet the design space's faces at a slant. A radius of 1.2 x the
  min member halved it but cost 3.6 points of stiffness: not done. On the brackets the rest is the
  2 mm skins (Min thickness automatic, 2.1 mm).
- Load arrows: every arrow was as long as every other, so a bearing load's sides (a few % of its
  pressure) looked as loaded as its middle, and a 4 mm boss face got 24 arrows of 13 mm (a bundle;
  72 with three loads). Markers now carry `sizes` (0..1 of the load's most), arrows are 0.25-1 of the
  length by them, and points closer than 0.4 of an arrow are left out (the most central kept).
- A bearing load presses square to the surface, so on a rounded rim its arrows lean with the
  surface (sideways parts cancel; the total is the force). For "push the front in -Y" use a force.
- TOPO_VERSION 2: saved results are optimized again.

### Do the optimizers keep their promises on the shown body? (2026-10-21)

Each optimizer node run as Optimize does, its shown body meshed as the Final preview, then
measured and put through Static Analysis on the optimizer's own grid (medium, GPU):

- Topology stiffness: the wall bracket's note said 26%, Static Analysis of its body 14.5%. On the
  optimizer's grid and solver the body matched the densities (IoU 0.97, 25.9% vs 26.4%; solver
  tolerance made no difference). The cause was Static Analysis: the bearing load on the pin hole,
  whose unloaded side the optimizer had removed (27% of the faces), was put back by area: 276 N
  instead of 200. Now put back by what each point carries (`apply_conditions`, FEA_VERSION 2):
  27.5%. Bolted bracket 19.2% / 18% in the note, key box 78% / 76%.
- Keep % (25.0%, 30.2% of the design space), Gap 3 mm (no material within it away from supports and
  loads), Max thickness 4 mm (largest inscribed ball 99% <= 5.1 mm, joints) and Frame Min size (2.0
  mm) hold on the body. Ray thickness overstates plates and joints (rays from an edge cross the
  width), so maxima were measured with the distance transform.
- Frame Optimization's Safety factor 2 does not hold on the body: beam model utilisation 0.92 (2.17),
  Static Analysis of the body 1.57 at medium and 1.42 at fine, at the junctions of bars with the
  bosses at the pin and the bolt holes (the beam model cannot see them; the peak grows with detail).
  Deflection 0.53 mm against 0.88 in the beam model. Not fixed (NODE_AUDIT 19).
- The key box as set up (one 4 mm boss held, 500 N of loads) came out at safety factor 0.16 and 189 mm
  deflection, while its note said "stiffness 76% of the full design space": relative numbers only
  (NODE_AUDIT 20).
- Safety factor colours: a fixed scale from 1 (or the smallest value, below 1) to 10, so a part safe
  everywhere is no longer painted in the "bad" colour and the range is never backwards.
- The topology key takes only E and Poisson's ratio of the material.

### The optimizers check their own results; Min thickness after Smoothing (2026-10-21)

The user expected the optimizers to check what they make; they only judged it with their own models
(Topology: compliance of the densities, no stress; Frame: beams without joints or bosses).

- `nodes_fea.check_result`: Static Analysis of the body an optimizer shows, at its detail (Frame:
  medium). Topology's note gets "safety factor X, max deflection Y mm", kept with the shape
  (`_TOPO_RESULTS`, keyed by the material too, since strength is no longer in the topology key);
  warnings below safety factor 1 and above 2% of the part's size in deflection. The key box: 0.165-
  0.18 and 141-165 mm, warned. A load whose faces lost their material (see below) is a warning.
- Frame Optimization (`nodes_frame._size_for_body`, FRAME_VERSION 3): after the beam sizing the body
  is analysed at medium, and the bars within (their size + skin + 2 mm) of its most stressed spot are
  made thicker by (safety / reached)^(1/3) x 1.02, rounded up to the size step, until the body
  reaches the safety factor (6 rounds at most; stops when thicker bars stop helping). The state keeps
  the beam sizes, the body settings checked (skin, joints, fillet) and the check, so previews show it
  without analysing again; if skin, joints or fillet change, the preview analyses the body and asks
  for Optimize, which sizes the bars again without a new layout. Bracket: 1.57 -> 1.91 -> 2.38 in 2
  rounds (23 s instead of 10), 3.94 -> 5.38 g of bars, body 8.8 -> 11.0 g. At fine the same body
  had 1.90 (the fillet peaks grow with detail); sizing at fine reached 2.07 but one fine analysis of
  the sparse frame took 87 s on the CPU (its box did not fit the GPU next to the reserve), so it stays
  at medium, as Static Analysis's default. The body's key now includes the bar sizes (it was keyed by
  frame, profile, joints and fillet only).
- Min thickness on the final shape, in three parts:
  1. The optimizer: 40% of the key box's material was plates lying on a wall face under 3 mm, because
     the density filter only averages over the design space (a plate on a face looked twice as thick
     to the eroded design). The eroded design now uses `filter_empty` (the faces counted as empty, as
     in the part); the design itself keeps the normalized filter, so the CAD faces are still
     reproduced. Medium, shown body (A: faces empty, B: as before):

     | case | stiffness A / B | surface under the min member A / B |
     |---|---|---|
     | wall bracket 25% | 27.0% / 26.4% | 3.5% / 11.4% |
     | bolted bracket 20% | 25.6% / 25.2% | 17.5% / 17.0% (the 2 mm skins) |
     | key box, four bosses, Min 3 | 38.8% / 39.8% | 6.7% / 11.6% (0.5% / 4.0% away from skins) |

     The 30 x 10 x 10 mm test box suffers (R = 3 mm against a 10 mm box: its flanges lose a layer in the
     eroded design): tests moved from the eroded compliance to the design's own.
  2. Smoothing: see "Lumpy surfaces" below (the distance to the surface is blurred now).
  3. `thicken`: what is still thin moves its surface out smoothly: half thickness = the deepest point
     within a whole Min thickness (faces counted as empty), lack = Min/2 - that, blurred, added to the
     values along their gradient, half a step a pass (a full step overshot: 2 mm became 4).
  First tried and dropped (they made the part look bad, user: "not a very good looking part"): balls
  on the middle of thin members and filling in from the faces, on the samples (steps of a sample,
  knobs, ladders along the rim, ragged edges), then the same as smooth fields (rows of bubbles), and
  rebuilding erased members from their middle line (beaded rails). The middle line needed opposite
  nearest faces (> 120 degrees) or it took every corner (+36% material), a sample's distance runs half a
  sample past the surface (t/2 + voxel/2, or 2.5 mm struts passed for 3 mm) and even members need both
  middle layers.
- Key box results (the user's settings: Min 3, Max 15, struts only, one boss held): no material added,
  1.8% of the surface away from the skins under 3 mm before `thicken`, smooth surfaces; reported
  stiffness 66% (76% before the faces counted as empty), the body 73%. Held at all four bosses the same
  settings give a symmetric rim with four gussets: safety factor 4.75, 0.89 mm (one boss: 0.18, 197 mm).
  Fine detail looked the same (struts must be 0.65 x 15 mm). Renders: plain shading of the Final
  preview mesh (`scratchpad/render_topo.py`, matplotlib): thickness colours hid how rough it looked.

### Lumpy surfaces (2026-10-22)

"It still looks a bit lumpy." Close-ups at full resolution, flat against smooth shading: the
corners were smooth (the mottling in the overall pictures was the render: half the triangles, drawn
small), but where a band of material ends inside a wall its edge was wavy at the elements' scale
(1-2 mm), with a few bumps on the rim.

- The cause: Smoothing blurred the densities. A blur wide enough to iron out element-scale waves also
  thinned members and took thin ones away, so thin members were held to a light blur, which is where
  the waves stayed (on a thin-walled part, everywhere).
- Now `density_mesh` blurs the signed distance to the densities' 0.5 surface (`_surface_distance`:
  distance transforms, within a sample of the surface from the densities, so not stepped). Across a
  member the distance runs straight, so the blur hardly moves the surface. Members under half the
  blur's sigma still get only LIGHT_BLUR, blended up to the full blur at a sigma (a member goes once its
  half thickness is under ~0.8 sigma; with the light blur up to twice the sigma, as first, the edges
  stayed as wavy as at Smoothing 1). The surface is then shifted, within half an element, so the
  material in the design space stays what the densities give (the bolted bracket lost 10% at
  Smoothing 3 without it). `thicken` gets a field with slope 1, so its push is exact.
- Bolted bracket, 3 mm gap, body stiffness: Smoothing 1 27.2%, 3 27.0% (blurring the densities: 25%
  weaker at 2). Key box, four bosses, Min 3, plates and struts: 40.3 / 40.5 / 40.1% at 1 / 2 / 3, 1.1-1.4%
  of the surface away from the skins under 3 mm. The band edges are straight from Smoothing 2: the
  default is now 2.
- Pieces under SPECK of the largest were dropped by size: an Avoid pocket's inner surface (a cavity,
  negative volume) fell just under it and the pocket filled. Only pieces of positive volume can be
  specks now.

### Smoothness while optimizing, and Symmetry (2026-10-22)

The user wanted the shape smoothed by the optimizer itself. The commercial tools (OptiStruct /
Inspire, Ansys, SOLIDWORKS, Fusion) offer member size and symmetry during the optimization and
smooth afterwards (volume-preserving) with a re-check; none has a user-facing perimeter penalty. Both
are node inputs now, off by default (results and keys without them unchanged).

- Smoothness: the optimizer minimizes compliance x outline^Smoothness (`Optimizer.outline`: sum of
  |grad| of the projected design at every element corner, forward differences on a zero-padded grid,
  so the whole surface counts; gradient through the projection and the filter, checked against finite
  differences). Scale-free: relative changes of both are weighed, Smoothness 1 = 1% of surface is worth
  1% of compliance. Where the penalty outweighs the stiffness the OC update gets `OUTLINE_FLOOR` of the
  mean sensitivity, not 0: with 0 those densities only shrank and a cantilever ended at 9% instead of
  30%.
- The outline of the filtered densities (the first prototype, weighted by mean |dc| / mean |ds|)
  looked good by its own measure (12% less) but did not shrink the real surface: key box, struts only,
  261 -> 260 / 264 / 280 cm2 at 0.5 / 1 / 2. A thin member's filtered peak stays under 1, so its |grad|
  is small: it rewards thin, grey members. Always measure the mesh (`shown.area`), not the proxy.
- Key box, four bosses, Min 3, medium, Smoothing 2 (stiffness of the full design space, surface of the
  shown body, safety factor):

  | Smoothness | plates and struts | struts only, Max 15 |
  |---|---|---|
  | 0 | 38.8%, 309 cm2, >10 | 18.8%, 261 cm2, 5.5 |
  | 0.25 | 38.6%, 293 cm2 | |
  | 0.5 | 38.1%, 281 cm2 | 19.5%, 257 cm2, 6.5 |
  | 1 | 36.2%, 266 cm2 | 21.2%, 247 cm2, 9.0 |
  | 2 | | 20.0%, 243 cm2, 9.1 |

  Webs end in cleaner edges, ridges along the sides go, and with struts only the gussets wrap round
  the bosses instead of standing apart (a better optimum: stiffer and safer). Renders:
  `scratchpad/smoothness_compare.png`, `smoothness_struts_compare.png` (session of 2026-10-22).
- A 40 x 4 x 20 mm plate (only a little thicker than the 3 mm thinnest member, so its eroded design
  carries almost nothing and the stiffness signal is weak) broke its top chord at 1 (10x the
  compliance); at 0.5: 6 -> 3 openings, 5% less outline, same stiffness (the test).
- Symmetry (x, y, z or combinations): planes through the design space's bounding-box middle. The grid
  is laid symmetrically about each plane (n = ceil(L / h) elements centred on it, so the low faces are
  no longer exactly on element faces), and dc, dv and dg are averaged with their mirror images where
  both elements are free (`Problem.symmetric`): from the uniform start the free densities stay exact
  mirror images (bit for bit; averaging x after the update instead would shift the volume). A note
  gives each plane and warns when 5% or more of the design space or of the skin and Keep has no mirror
  image. Four-boss key box, x and z, plates and struts: 38.7%, 310 cm2 (as without: it is nearly
  symmetric anyway, now the four corners match); with Smoothness 0.5 38.0%, 282 cm2.
- Brackets, medium (kept, stiffness, surface, safety factor):

  | Smoothness | wall bracket, Keep 25% | bolted bracket, Keep 20%, Max thickness 4 |
  |---|---|---|
  | 0 | 25%, 27.0%, 97 cm2, 9.4 | 15.8%, 17.7%, 108 cm2, 7.0 |
  | 0.5 | 25%, 27.2%, 88 cm2, >10 | 14.3%, 15.9%, 94 cm2, 5.2 |
  | 1 | 25%, 26.8%, 85 cm2, >10 | 10.3%, 10.3%, 73 cm2, 4.9 |

  The wall bracket's diagonal and top arm become one smooth web, in fewer iterations (114 -> 68).
  Smoothness works against a tight Max thickness (it wants fewer, thicker members), so less of Keep %
  fits (the note says so); stiffness per material kept stays about the same at 0.5.
- Two faults the brackets showed, both fixed:
  1. Loose pieces (bolted bracket, 10 at Smoothness 1): Max thickness had squeezed the volume under
     Keep %, so `_update_two`'s volume multiplier sat at its lower bound 1e-9 and material was free;
     elements carrying nothing (1e-7 of the mean sensitivity) then grew wherever no neighbourhood was
     full, and pieces the penalty had cut off stayed and grew. A floor on the multiplier relative to the
     unlimited one did nothing (that one was at the bound too, the move limit keeps Keep % out of reach);
     `PRICE_FLOOR` = 1e-3 in the normalized units does: none, and without Smoothness the same result.
     A floor of the mean sensitivity for penalty-dominated elements (first version of `OUTLINE_FLOOR`)
     had the same effect from the other side: +73% surface, +28% thickening; now a share of their own.
  2. Flakes on the wall bracket's pin eye (0.3-1.8 mm3, 0.5 mm thin, by the side faces): bits of the
     smoothed surface that joined the part only outside the design space, so the crop cut them off.
     `density_mesh` empties pieces under SPECK that stand alone inside it (phi >= 0.5) before meshing.
     What is left are zero-volume slivers of the crop (also without Smoothness).
- Min 1 / Max 2 mm on the bolted bracket (Keep 20%): members need 3 elements and Max 1.5 x the min
  member, so they are raised (notes): draft 3.9 / 5.9, medium 2.13 / 3.19, fine 1.35 / 2.03 mm (the note
  said "using 2 mm" for 2.03: three digits now). Medium: 14.0% kept, stiffness 14.3%, safety 6.0, 126 s,
  a branchier version of Max 4. Fine (942k elements, 406 s): a bone-like web of many thin branches,
  11.9% kept, stiffness 9.9%, safety 1.9, 0.31 mm, no loose pieces. A true 1 mm needs 0.33 mm elements
  (~2.5M on this part), finer than the finest detail.

### Soft guides: "prefer material here" (prototype, 2026-10-22)

The user wants to nudge where the optimizer puts material and keep it organic (Keep forces an exact
solid, Avoid an exact hole). Prototype (`scratchpad/prefer.py`, not in the engine): a ball with a 5 mm
fade; the design's material in it is rewarded in the sensitivities (strength x the mean stiffness
sensitivity, through projection and filter), the optimizer's shape is kept as it is (no union). Wall
bracket, medium, ball r 5 mm in the middle of the open triangle (empty without it):

| strength | stiffness | safety | ball filled | look |
|---|---|---|---|---|
| 0 | 27.0% | 9.4 | 2% | triangle open |
| 0.5 | 27.2% | 9.3 | 99% | a web with holes across the triangle |
| 1 | 26.4% | 9.0 | 100% | a hub with branches to the arm, the diagonal and the plate |
| 2 | 25.1% | 8.5 | 100% | the hub bulges round the ball |

Organic throughout: members grow to the region and merge the optimizer's way.

Built in: Topology Optimization's Prefer (a body) and Prefer strength (default 0.5, -3..3; negative =
keep away softly), the fade 2 x the thinnest member; nodes Ball, Bar (a capsule between two points) and
Box (rounded edges), usable anywhere an implicit body is. The user: "It needs to be very easy to place
the bodies in the view": the viewport draws them itself (live while dragged, the engine no longer meshes
them for the view), Place on the card + a click on a shown body, the selected one's handles to drag.
In the test editor: Place first put the ball half way along the view ray, which on a slanted view ran
out through the bottom (z 6 where the click was ~21); now half way through the wall under the click
along the surface's normal (21, 15, 13 on the 30 mm thick space: y = 15). The ball inside the opaque
design space was hidden: guides are drawn a second time faintly through the bodies. Optimize with the
ball in the open triangle (medium): a web grew to it, 27%, safety 9.3.

### Coarse-to-fine: Start coarse (2026-10-22)

The user asked for preliminary results early in one run. Prototype first (`scratchpad/progressive.py`):
draft until the projection sharpness reaches 4, the design variables moved trilinearly to the medium grid
(element centres), on until sharpness 8, then fine to the end; the sharpness, iteration count and history
carry over, `full_compliance` is the new level's. Fine, GPU, editor stopped:

| | first shape | done | kept | stiffness | safety |
|---|---|---|---|---|---|
| wall bracket, direct | ~30 s (fine setup) | 110 s | 25% | 28.3% | 9.2 |
| wall bracket, coarse-to-fine | 9 s (draft done) | 58 s | 25% | 29.0% | >10 |
| bolted bracket (Max 4), direct | ~30 s | 180 s | 17.4% | 21.2% | 8.8 |
| bolted, from draft | 12 s | 120 s | 11.8% | 12.7% | |
| bolted, from medium | ~12-20 s | 118 s | 16.7% | 20.2% | 7.6 |

- The draft shape already has the final layout; the wall bracket's final is cleaner than the direct one
  (no thin diagonal strut, 88 against 108 cm2).
- Max thickness must hold at the first level: draft raised Max 4 to 5.9 mm, the draft design got a few
  thick members, and when 4 mm applied at medium they were thinned rather than split (19.8 -> 11.6%
  kept). Levels too coarse for the Max thickness are skipped (bolted: starts at medium).
- Setting up the fine level (element fill, conditions, the full design space's solve) takes ~30 s
  either way; the fine iterations drop from ~110 to 24-50.
- In the node (input Start coarse, default yes; `T.levels`, `Optimizer.run(hand_over=)`, `handover` /
  `take_over`, states carry `level`), Optimize as the editor runs it: wall bracket fine: first shape
  2.8 s, 66 s optimizing (direct 110), 29%, safety >10; bolted bracket fine: starts at medium, first
  shape 6.8 s, 139 s (direct 180), 20%, 7.6. Key box medium (draft -> medium): first shape 4.7 s, 52 s,
  19%, 5.57: the same as without (medium iterations are as quick as draft's on the GPU), so Start coarse
  applies to fine only and medium results keep their keys.
- First bug: with no saved state the run started at the last level (a resume rule took "no level" for
  the chosen detail); the node test stops on draft and continues there.
- In the editor (test editor on 8766, wall bracket at fine): the card shows Start coarse; Optimize
  showed a shape after ~4 s, the log "draft done at iteration 50 (10 s, stiffness 23%): moving on to
  medium elements", "medium done at iteration 71 (19 s, 27%)", done in 66 s, 29%, safety >10. Cancel at
  iteration 60 (on medium): card "stopped at iteration 60 on medium elements (Optimize continues)" with
  the shape so far; Optimize continued at 60 and finished (95 iterations, 28%). No console or server
  errors.
- The 30 x 10 x 10 mm test box handed over from 2 mm to 1 mm elements ends 1.25 x the compliance of a
  direct run: 5 elements across its 10 mm, the coarse layout is cruder. Parts 20+ elements across at
  draft (the brackets, the key box) came out as stiff.

### Loose flakes after Smoothing (2026-10-24)

The user saw islands on the bolted bracket (fine, Keep 20, Max 4 struts only, gap 1, three bearing loads)
after smoothing. The densities themselves were one piece (plus one lone element), and so were the
smoothed samples inside the design space once SPECK had emptied 13 bits of 0.07-2.4 mm3. The islands were
under a sample: on the unloaded side of the bolt holes (faces 6, 8) and the pin hole (face 10), where the
blur and the thickening bulge the surface out past the hole wall and it hangs over wall that is empty
inside. Marching cubes put that surface a fraction of a sample inside the face, so the crop to the design
space left paper-thin shells (0.1 mm x 4 mm; one of 29 mm along the pin hole). Shells of the cropped body
in the views (draft / medium / final, 0.47 / 0.24 / 0.12 mm): 2-4 / 1-4 / 7-16 for Smoothing 1-3,
Min thickness 0 or 1.5; at draft a 3 mm3 bit hung on by a neck thinner than 0.4 mm.

- `density_mesh(crop=)` (the node passes the design space less Avoid): after the speck removal, which now
  finds bits in the crop itself (the gap band included, sampled exactly), every sample outside it is made
  no more solid than the nearest sample inside it (`distance_transform_edt` indices), so the surface runs
  straight out through the faces and never hangs over them. Volume and stiffness unchanged (9733 mm3
  either way, safety 0.111 both).
- After: 1 shell at medium and final in all six cases, none in 0.06 mm close-ups of the three holes. At
  draft, Min 1.5 still shows 1-3 extra shells: a 0.4 mm sheet where a member meets the bolt hole wall at a
  shallow angle, joined to the part (at 0.05 mm it is one piece), too thin for 0.47 mm samples.
- Cost: density_mesh 17-20 s -> 21-25 s (the design space's distance on the 13.7M samples, the nearest
  indices), +0.35 GB peak (0.85 -> 1.2 GB).

### Thin tips at holes and folds (2026-11-04)

The user circled shreds in a bolt hole (tube bracket, fine, Min 1.5, gap 1): "why, with a min thickness?". They
were joined to the part: two points on the hole's wall, pointing at each other. Min 1.5 is 2.01 mm at fine, by the way:
members are at least 3 elements (`MIN_MEMBER`). The points' section 3 mm from the hole's centre line was ~1 mm2.

- Cause: a load on a flat face (a pick left from an older STEP) keeps a 2 mm skin on it, and no gap near it. The
  hole cuts across the fold where that face ends, so the face runs out in two 28 degree points, and so does the skin.
  The same skin wrapped round folds and inner corners leaves ribs of a quarter circle's section, up to 28 mm long.
- `thicken` can only add, and the crop takes it off again. It even makes them worse: asked for 1.5 mm the points
  ended 4.7 mm from the hole's centre line without it, 2.7 mm with it.
- `cut_tips` (after `thicken`, on the samples in the crop): solid farther than a little from every ball of 0.8 x Min
  that fits in it, reaching `TIP_OUT` radii past the balls, hanging by one patch, mean section (volume / length along
  the longest axis) under `TIP_SECTION` x Min^2. Cut along the distance from those balls, Gaussian-smoothed by a sample,
  so the stump is round. The clamp outside the crop then carries the cut through the faces. The rules against wrong
  cuts came from the bracket of the next section.
- Tried first, and why not:
  - The opening alone: 8% of a key box state went, plates of 105 x 32 mm. The distance on the samples moves in half
    samples, so skins near the threshold have a patchy core, and a skin can be thinner than Min (plates on a face only
    grow half inwards; Min can be 3 elements of a coarse grid). One patch does not save them: nothing passes through.
  - Limits on the widths across (second and third axes of the region): spared the very tip the user circled, since a
    lip curving round a hole looks wide. Mean section does not care how it curves.
  - `TIP_SECTION` 1/3 cut 9 of 10 and left the left point (0.34): the right one had joined a rib, which lowered its
    average. Tests passed and the note said "9 thin tips cut off"; only a render of the hole showed it. Compare
    both sides, and look. Tips reach 0.34, the first real plate is at 0.49, stubs of 0.40-0.45 in between, hence 0.45.
  - Not tried: a depth finer than the samples (from the nearest outside sample's value), which would sharpen the ball
    test itself. Judging whole regions was enough here; it is the next step if a plate is ever cut.
- Result on that bracket, node's analysis (fine, CPU): safety 8.87 both, deflection 0.0237 mm both, 89535 -> 89513
  mm3 (-0.02%), one piece, 4 tips (the two points circled and two loose bits by them). Cost at fine: ~12 s on the CPU,
  +0.4 GB peak (39M samples). The state is not touched, so Optimize is not needed: the shape changes with the next
  preview, its analysis runs once.

### Thin tips: tested on a part made for it (2026-11-05)

`cut_tips` had been tuned on the one bracket that showed the problem, and its tests passed. A bracket built to break it
(STEP made with OCP: four holes across folds, a 4 mm loaded pin, a 2.5 mm loaded tab, a 1.6 mm fin; real optimizations
at draft, medium and fine with the user's settings; every result rendered with the cut on and off) found three faults
in it, each visible only in renders or in the analysis, none in the tests or the card's note:

- Medium: 23 "tips", 0.57% of the material, and holes, a crack and a notch in solid skins. Min was 3.3 mm for lack of
  elements (3 x 1.1), the skin 2 mm, so every skin counted as thin. A patch closed in by thicker material hangs by one
  ring-shaped patch and is small: `_free_all_round` (empty on two sides that do not meet: taking it off makes a hole).
- Draft: Min 6 mm took the 4 mm pin and left its load nowhere: safety factor over 10 -> 0.81. `TIP_FACE_SHARE`: a tip of
  a skin takes a corner of a load's faces, a pin all of them.
- Draft: the fin went, deflection +7%. Both come from measuring against the grid's limit, which nobody asked for:
  thin is now against Min or Skin as set, at most the real Min (`tips_under`), and with neither set nothing is cut.
- Fine, after those: hole rims came out wavy. A plate meeting a hole at a slant has a sharp edge, which sticks out of
  every ball, so it was nibbled in patches. How far a piece reaches past the balls splits them cleanly on all three
  parts: points and loose bits 4.1-7.1 radii, rims and strips at most 2.44, hence `TIP_OUT` 3 (it was 1). A ratio of held
  to free surface (`TIP_HELD`, tried on the way) did nothing any more after that, and went. A straight edge sharper than
  ~29 degrees reaches `TIP_OUT` (radius / sin(half the angle)), and then only its section keeps it: not tried on a part.

Final, cut against not cut:

| Part | Tips | Safety factor | Deflection | Material |
|---|---|---|---|---|
| Own, fine | 6 | 6.61 / 6.61 | 0.0117 / 0.0117 mm | -0.03% |
| Own, medium | 2 | 7.97 / 7.97 | 0.0103 / 0.0103 mm | -0.01% |
| Own, draft, Gap 0 | 6 | >10 / >10 | 0.00988 / 0.00987 mm | -0.06% |
| Bolted bracket, fine | 4 | 8.87 / 8.87 | 0.0237 / 0.0237 mm | -0.02% |

Pin, tab and fin stay. Smoothing 0 (no tips) and 3.5 (3) give the same analysis on and off; no Min and no Skin set
cuts nothing.

Seen on the way, not from the cut:

- The safety factor can jump with nothing wrong: 0.15% less volume moved `tidy`'s element size from 0.45 to 0.47 mm and
  the factor from 6.61 to 5.76, at the sharp root of the pin. With the same element size it is the same with or without
  the cut (6.61 both at 0.45, 5.76 both at 0.47). Compare analyses at one element size.
- A load on something thinner than the settings allow is lost: Gap 1 mm leaves a 2.5 mm tab no room, so it is cut off
  and the analysis cannot run ("its faces are not on the part"); Min 5 on the 4 mm pin stops after 6 iterations at
  stiffness 0%. The note only says the result could not be checked.
- `body_from_mesh` once read a 224 mm3 chunk of solid as empty (Min 5, the broken design above, cut off): samples solid,
  mesh without a surface there, field +6.9 mm. Not looked into.

### Dead ends (2026-10-24)

After the flakes the user pointed at peninsulas: branches and fins that end in nothing. Coloured by the
final body's own Static Analysis (fine), regions under 0.87 MPa (a tenth of the median surface stress)
that meet the rest of the part through one patch: 22 of them, 303 mm3, 3.1% of the bolted bracket's
9730 mm3 (the biggest an 84 mm3 fin under the middle, a vertical finger, a knob at the top). They are in
the optimizer's own densities already; smoothing keeps them, it does not make them.

- Why the optimizer keeps them: Keep 20% does not fit under Max 4 mm struts only (11.2% does), so the
  volume's multiplier sits at `PRICE_FLOOR` (1e-3 of the mean sensitivity) and material is nearly free.
  A dearer floor (re-optimized at fine each time, Smoothing 2):

  | PRICE_FLOOR | kept | stiffness | deflection | surface under 0.87 MPa |
  |---|---|---|---|---|
  | 0.001 | 11.2% | 15% | 2.48 mm | 6.1% |
  | 0.03 | 11.1% | 14% | 2.41 mm | 5.4% |
  | 0.1 | 10.0% | 12% | 2.83 mm | 4.9% |
  | 0.3 | 6.5% | 7% | 9.60 mm | 4.4% |

  It takes working members as fast as dead ends: in the optimizer's own model they are not idle. Its
  stiffness is the eroded design's and its sensitivities are filtered over the member radius, so a thin
  fin strains with the member it hangs on (the 84 mm3 fin at 1.6 x the median strain energy; a trim by
  the optimizer's last displacements found 16 mm3 in all).
- `T.dead_ends` on the analysis of the final body instead, sampled at the element centres: regions of the
  design under `DEAD_SHARE` (0.1) of the median stress that reach the surface and meet the rest through
  one patch, or whose 90th percentile is under `DEAD_IDLE` (0.02) of it (the finger touched another member
  along its side: 2 patches, stress 0), away from supports and loads (skin + an element). Emptied, smoothed
  and analysed again; a second round takes the roots a cut leaves (`DEAD_PASSES` = 2).
- Bolted bracket: one round 32 dead ends (-1.5%), two rounds 67 (-2.0%); mass 12.38 -> 12.11 g, deflection
  2.479 -> 2.495 mm, safety factor 0.111 -> 0.354 (the highest stress away from the supports, 405 MPa,
  was at a stub's notch; now 127). Left under 0.87 MPa: the skins of the pin and bolt holes (kept on
  purpose) and roots of 1-8 mm3.
- Cost: the node's result 28 -> 90 s (two more smoothings and analyses), once per result: the cut is
  kept on the SSD, so after a restart it is one smoothing and an analysis again.
- The user then asked for the optimizer itself to penalize members without stress. Pruning while
  optimizing (`Optimizer._prune`): a solve with the design's own stiffness (projected densities, SIMP),
  sqrt of the strain energy density as the stress, `dead_ends` on the design; what it finds is kept at 0
  from then on (its growth capped, so the volume's root finding sees it). Re-optimized at fine:

  | | kept | stiffness | deflection | safety | after the trim |
  |---|---|---|---|---|---|
  | none | 11.2% | 15% | 2.48 mm | 0.111 | 2.50 mm, 0.354 |
  | every 10 from sharpness 8 | 11.7% | 17% | 2.45 mm | 0.199 | 2.46 mm, 0.172 |
  | every 5 from sharpness 4 | 12.3% | 18% | 1.90 mm | 0.451 | 1.90 mm, 0.438 |

  The material the stubs held goes to members that carry: 23% stiffer at 10% more mass, and a more even
  web. It keeps sprouting new stubs (5-15 per check at sharpness 4, 30-70 at 16), so the trim after
  smoothing stays (41 more, -0.5%). Each check 0.5-1.8 s (GPU), ~15 per run. A short wide stub on a beam
  in bending is no dead end (a flange: it carries); a finger 3 x its width long is.
- Min thickness (the user's point): emptied elements right against a member keep it from growing to the
  min thickness on that side. A design variable at exactly 0 never grows again in this OC update (x times
  a factor, lower bound max(x - move, 0)), so emptying a whole dead end locks all of it, whatever is
  marked. A free band of the thinnest member beside everything that stays (`Problem.radius`, 2.6 mm: struts
  only's floor of 0.65 x Max) left nearly every dead end alone: bit for bit the unpruned run. Then
  (`prune_lock`, since replaced: see "Max iterations") a band of Min thickness (`Problem.min_member`: the
  setting, at least 3 elements) was left only beside members still thinner than that (half thickness from
  the distance transform, as `thicken`); the rest of each dead end emptied and locked. The band beside members under 2.6 mm: 11.9% kept,
  2.37 mm. Under Min thickness: auto (1.35 mm) 11.8%, 1.96 mm (no band 1.90); Min 1.5 11.8%, 2.10 mm
  against 2.48 unpruned (the same unpruned design at both: the struts-only floor rules). The trim after
  smoothing is thickened to Min thickness again by `density_mesh` anyway.
- Results optimized without pruning are still shown with Dead ends: remove (the key has `prune`), trimmed,
  with "optimized without pruning dead ends: press Optimize for a cleaner, stiffer result".

### Max section (2026-11-07)

"Even when I have struts only it looks like plates anyway": the bolted bracket's saved fine states (Max thickness
10, struts only) were 40-62% plate by local shape (Westin's measures on the solid), half the solid exactly 6.7 mm
thick, which is the limit (0.62 x Max). A thickness limit cannot see width: a plate just thin enough passes. Node
input Max section (`max_section`, mm2) limits the area of a member's cross-section instead.

- **How it reads an area.** A Gaussian wider than the member integrates to 1 along it, so at a long member's axis
  the local density is A / (2 pi sigma^2) (round: 1 - exp of minus that) whatever the section's shape, up to ~2 sigma
  wide. Wider than that it reads as a plate filling erf(t / (2 sqrt(2) sigma)), so sigma is chosen so that the
  thickest plate that fits is `SECTION_PLATE` of the thinnest member. 0.7 left plates (31% at 60 mm2, 27% at 35),
  0.5 leaves 6-10%. Only `free` material counts: a skin cannot get thinner.
- **A joint is not a failure.** The log's "widest section ~44 (max 35)" looked like the limit not holding. It was one
  joint: 0.1% of the free elements over, at most 1.19 x, 18-25 mm from any skin. A joint of n members holds n / 2
  sections. The log now says "fullest spot"; on the finished fine body sections as round members were 47 / 59 / 66
  mm2 at the 50 / 90 / 99th percentile, for a limit of 60.
- **The flip.** At a sharp projection the linearized constraint overshoots: material near the limit leaves and comes
  back every other iteration, a load path going and returning, compliance swinging 5458 <-> 1915 even at 300
  iterations (6% stiffness, never settled), so more iterations do not help. `_track_moves` shrinks the move limit of
  a variable that turned back (MMA's asymptotes): the swing is gone, compliance tails are smooth, the fullest spot
  sits on the limit. Only with Max section, so other results stay the same.
- **What fits.** The neighbourhood may be `section_share` full, and that does not depend on the area: 15% with a Max
  thickness (the filter radius lands on the upper clamp, diameter / 1.5), 8% with Max section alone (`SECTION_MIN`).
  A larger area only trades many thin members for fewer thick ones, so Keep is out of reach above about 15-20%
  (skins included) at any area.

Bolted bracket, loads fixed, medium (1.1 mm) unless said, kept with the skins; "+" = Max 10, struts only, Smoothness
0.83, "alone" = neither:

| | Keep | kept | stiffness | safety | deflection | strut / plate |
|---|---|---|---|---|---|---|
| + no Max section | 30 | 26.9% | 28% | 4.49 | 0.0238 mm | 56 / 43 |
| + 60 mm2 | 20 | 16.5% | 12% | 2.52 | 0.0668 mm | 92 / 8 |
| + 60 mm2, fine (0.67 mm) | 20 | 16.2% | 12% | 2.14 | 0.0649 mm | 92 / 7 |
| + 35 mm2 | 30 | 13.6% | 9% | 1.95 | 0.0732 mm | 92 / 6 |
| + 35 mm2 | 15 | 12.2% | 8% | 2.14 | 0.108 mm | 92 / 7 |
| alone 60 mm2 | 15 | 14.6% | 13% | 2.75 | 0.0684 mm | 90 / 9 |
| alone 60 mm2 | 30 | 18.7% | 14% | 3.47 | 0.0521 mm | 90 / 10 |
| alone 120 mm2 | 15 | 15.0% | 10% | 2.70 | 0.0546 mm | 82 / 15 |

- The numbers do not show the shape: Keep 30 alone has the best safety factor and looks worst, a web of extra struts
  (Keep out of reach leaves material nearly free). Keep near what fits gives few bold struts. Look at the shape.
- Every run ended "stopped at Max iterations, still changing": a slow creep from pruning, not the flip.
- Struts cost real stiffness: 12% of the full design space against 28%, ~3 x the deflection.

**Fine ran 100 x slower once, and not because of the setting.** A fine test on the card took 250 s per iteration on
the CPU (96 minutes to iteration 99); the same run headless took ~3 s per iteration on the GPU (2.7-3.6; 142 s in
all), as fast as without Max section. `_array_module` wanted 2.89 GB + `GPU_HEADROOM` 1.5 = 4.39 GB and saw 4.31 GB,
short by 0.08. What had gone: ~0.5 GB in the idle workers of the other editor (a run can only stop its own) and
~0.3 GB of the run's own contexts, which no level gives back (at the hand-over 0 CuPy arrays and 34 MB of Warp arrays
were alive, and `gc.collect` freed nothing: no leak). Other programs held ~7 GB of the 12. The estimate is high here:
the fine level used 2.6 GB, against 720 B per box element, and `GPU_FIXED` is counted again on a later level. The one
sign on screen was the word CPU in a log line, so look for it before believing a timing.

### Edge cases of Topology Optimization (2026-11-07)

About 90 odd settings through the real node (box 30 x 10 x 10, 1 mm elements, a process per case with a memory limit
of 6 GB), then slender parts at real detail on the GPU.

**The GPU kernels assumed 8 children per element.** `gpu.rows` read `per[row * 8 + ch]` for a "children" level, which
is the level under the fine one. That has 8 children only if the first step halves all three axes: an axis of 1, 3 or 5
elements is not halved (`plan_coarsenings`), so a sheet has 4 and a bar 2. The kernels read past each element's row: a
plain solid 200 x 200 x 1 sheet gave nan or stopped at residual 1, in Static Analysis too, while the CPU solved it.
Float64 failed the same way, and that is what ruled out rounding; `DESIGN_ENGINE_FEA_KERNELS=0` solved it. `kids` is
now passed in. `test_gpu.py` only used (3, 2, 3), so it never saw this: grids with a 0 coarsening are in it now.

**Float32 rounding beside soft elements is a separate failure** that stayed after that: with the kernels right, solid
solves were fine but the optimization stopped once elements got soft (200 x 200 x 1 at iteration 1, 500 x 5 x 5 at 50,
600 x 2 x 2 at 1, 1000 x 2 x 2 at 8). Solid sheet, 1e-6: float32 238 iterations, float64 98. The whole hierarchy in float64
ran all of those to 125 iterations (8-146 s), so `Optimizer._solve` hands over to float64 when float32 gives up, for
the rest of the run (`wide` in the state), and `analysis.solve_cases` retries once the same way. Bulky parts never get
there and stay bit for bit what they were.

| GPU, draft, Keep 30 | Before | Now |
|---|---|---|
| 200 x 200 x 1 | nan on the first solve | 125 iterations, 147 s |
| 500 x 5 x 5 | stopped at iteration 50 | 125 iterations, 86 s |
| 600 x 2 x 2 | nan on the first solve | runs (float64 throughout: 125 iterations, 146 s) |
| 300 x 2 x 2 | nan on the first solve | draft stops at iteration 32 (below); medium runs |

300 x 2 x 2 at draft is 5 elements across, coarsened only along its length: float64 reaches 2.5e-2 after 1000 iterations
where 1e-2 is asked for, and later 1.0e-1. Up to 10 x the tolerance counts (`SOLVE_NEAR`: enough for a step, and the next
starts from it); past that the node says so and keeps what is done. Its error's "finer Detail" is measured: at medium
(1280 x 10 x 10) the same bar ran 60 iterations at 2-4 s each.

**Limits past the design space asked for gigabytes.** Filter radius and neighbourhoods are sized by the limit, not by the
part, so three numbers a designer could type as "no limit" went over 6 GB in 5-8 s: Max section 1e6 mm2, Max thickness
400 mm with struts only, and Min thickness 400 mm with no skin (kernel (2R + 1)^3). Max section 1e5 took 3.1 GB and 309 s.
Worse than the memory, Max section 1000 mm2 ran fine and gave a grey blob at 2% stiffness that reached no load: its
members were 18 mm on a 10 mm box. `topology._room` measures the design space: a limit that cannot bind is off, with a
note (largest section, thickest plate), so 1000, 1e5 and 1e6 mm2 now give exactly the no-limit result; forced members
stop at what fits; sigma stops at half the longest side; a Min thickness over the box's second side is refused (not its
thinnest: in a sheet members have room across).

Other things found: a truncated state file failed the node on every run (`zipfile.BadZipFile`, not a ValueError); a
load all on Fixed Faces gave compliance 0 and melted the design to 3.3% with a note blaming "Max thickness (0 mm)"; the
skin's error said "make Skin thinner" when Min thickness had made it thick, or Skin was already 0; "not connected to any
Fixed Faces" did not say Avoid or the gap had cut it; a Keep body outside the design space stayed in the result
silently; `_prune` could cut one side of a symmetric design only; Max iterations ran 1 more per level of a coarse start.

Not bugs: a load off a thin path at low Keep. Keep 15 on that box leaves 8 mm2 of free section, a member of 3 mm filter
radius needs 12 mm2 to carry, and no setting of the optimizer gets there: the warning says the load is not reached and
now what to change. Clean with no change: Keep 1-99.9, loads of 1e-9 and 1e9 N, parts of 0.03 mm and 30 m, a Stop
before the first iteration, two bodies with one left out, load cases with Max section stopped and continued (2e-8, the
state's float32).

### The result's check (2026-11-08)

The user's "analysing the shape" ran over 10 minutes after a fine Optimize (`topo_wall_bracket_bolted-frame2`:
struts only, Max section 10 mm², Min thickness 1.2, 125 iterations on 208 x 88 x 176 elements of 0.67 mm). A
snapshot of the editor (py-spy) showed it in the conjugate gradients of the result's Static Analysis, on numpy:
iteration 238, 2.3 s each. Three causes, measured on that shape (rebuilt from the saved state; scratch scripts):

- **The check's grid.** Sized from the shape's own volume (1M elements in the part at fine), a web of struts
  filling 2.3% of its box got the box cap: 288 x 128 x 240 elements of 0.49 mm, 8.85M, 2.7x the optimization's
  3.2M (200k with material against 86k). Analysed on the optimization's 0.67 mm instead (members 3-5 elements
  across) it came out far too stiff: max deflection 0.234 mm against 0.375, safety factor 0.374 against 0.538
  (peak 949 MPa elsewhere against 660), mass 140.4 g against 139.3. So the check keeps its own size; whether
  0.49 mm is converged for such thin members is not known (Static Analysis' advice: compare two details).
- **The GPU did not take it.** 8.85M elements need 128 MB + 470 B each + the 1.5 GB headroom = 5.8 GB free; the
  card had 5.6 (other programs ~6.5 GB). `_array_module` now also drops this process's
  Warp meshes and BVHs (`backend.release_gpu_caches`) before stopping idle workers, and says in the log when an
  analysis stays on the CPU.
- **The multigrid is weak on a strut web.** 154 iterations at 0.67 mm (~250 at 0.49), where bulky parts take
  24-35: a tenfold drop per ~25 iterations instead of 4-5. Not the smoother: lmax from 12 power steps (3.74 with
  the margin) is below 40 Lanczos steps' 4.09, but lmax at the Gershgorin bound (5.99), 1.05 x Lanczos and 60
  power steps all took 154-155; float64 was slower (0.16 at iteration 43); Chebyshev degree 6 took 122 at 2x the
  work (280 s against 295). Not hinges: the part is one piece through faces. It is the coarse levels: their
  elements (1.34, 2.68, 5.36, 10.7 mm) grow past the struts (~3 mm, 2 x the 90th percentile of the distance to the
  surface), and coarse elements holding separate pieces go 0, 0.6, 4.9, 20%. With the coarsest level factorized
  directly at level 3 / 2 / 1: 106 / 67 / 28 iterations (CPU, the same displacement, 0.2343 mm). The r.z <= 0 seen
  in the editor (|r.z| ~ 1e-12 near the end) was float32 rounding, which the flexible beta allows for: none in
  any run here.
- **SuperLU's pivoting.** With its default partial pivoting, level 2 (23k unknowns) and level 1 (86k) were not
  factorized after 10 minutes; without pivots (`diag_pivot_thresh=0`, SymmetricMode: the matrices are symmetric
  positive definite) 0.2 and 1.2 s. The fill depends on the shape, not the unknowns: the web's 86k kept 18M
  nonzeros, the bracket's (`fea_bracket.json`, medium) level 1 with 121k 200M (44 s, 1.6 GB). The fill ratio of the
  last level predicts the next: the web's grew 1.0-1.3x a level, the bracket's 2.75x. `Hierarchy(direct_fill=)`
  walks up from the coarsest (`_finest_direct`) while the predicted fill stays within the budget.

The budget (`analysis.DIRECT_FILL`): the GPU does an iteration in milliseconds and solves the coarsest level on the
host in every V-cycle, so a big factorization costs more than the iterations it saves there; on the CPU an
iteration takes 1-3 s. CPU: 25M or 5 per box element, whichever is more; GPU: 1 per box element. Topology keeps
the grid's levels (it factorizes every iteration).

| static solve (`solve_cases`) | grid's levels | budget 25M | final budget |
|---|---|---|---|
| strut web 0.67 mm (3.2M), CPU | 154 iterations, 172.8 s | 28, 42.1 s (hierarchy 5.4 s, level 1) | the same |
| bracket medium (1.5M), CPU | 24, 31.7 s | 18, 26.8 s (level 2) | the same |
| strut web 0.49 mm (8.85M), CPU | ~250 (the editor) | 127, 365 s with the model (level 2) | 95, 258 s for the solve (44M, level 1) |
| strut web 0.67 mm, GPU | 154, 2.7 s | 28, 4.1 s | 106, 1.7 s (level 3) |
| bracket medium, GPU | 24, 0.5 s | 18, 2.2 s | 24, 0.4 s |
| bracket fine (6.6M), GPU | 29, 1.6 s | 20, 1.8 s | 20, 1.4 s |
| strut web 0.49 mm, GPU, whole check | 361, 20.4 s | | 113, 14.2 s |

Displacements and the checks' numbers the same to the printed digit in every run (0.2343 / 0.3754 mm for the web,
4.3337 / 4.3510 for the bracket). CPU runs 2-4 at once, GPU timings with the user's editor idle (1.3 GB of RAM).
The check on the GPU used 3.31 GB at most (NVML low point): 402 B per box element, under `gpu_bytes`' 470, so it
asked for 5.8 GB free where 4.8 would have done; not changed (other parts measured up to 491). Left: on the CPU
half an iteration's time is arithmetic over the whole box (97% empty for the web): vectors over the nodes with
material only would cut that, and the GPU memory, a larger change. The card now follows the check's solver
(`_solver_progress`: "analysing the shape (solver iteration N), solved 87 / 100").

### Building the result: the hardware sat idle (2026-11-08)

The user saw neither the CPU nor the GPU work hard while "analysing the shape" and "removing dead ends". Each
stage of `_result` timed (frame2 recipe's saved state, Max section 10, the editor idle; its own cache folder):

| stage | time | cores busy | GPU |
|---|---|---|---|
| smoothing (`density_mesh`) | 95.6 s | 2.2 | none |
| check, 8.85M elements | 116.7 s | 2.0 | CPU: 2.0 GB free, the estimate asks 5.5 |
| finding dead ends | 0.7 s | 1.0 | |
| smoothing again | 89.5 s | 2.3 | none |
| check again, 7.74M | 102.7 s | 2.2 | CPU: 4.4 GB free, 5.0 asked |

407 s in all. "Removing dead ends" is a second smoothing and check; finding them takes under a second. The card's
35-39% load was the other programs' (the same before the run). The smoothing's own Python took 3.8 s: over half of
its 118 s (py-spy, per thread; cProfile in Python 3.12 counts every thread, the watchdog's sleeps too) was
scipy.ndimage's distance transforms, a dozen on the UPSAMPLE x finer grid of 35-40M samples, single-threaded C,
7.2 s each; then Gaussian and maximum filters. The check's CPU solver is numpy over the whole box, bound by memory:
4 threads made elementwise work 1.6x faster, scipy's sparse products (they let go of the GIL) 4x.

**Smoothing on the GPU** (`density_mesh` -> `_density_mesh(xp=)`; `thicken`, `_surface_distance`, `cut_tips` take
their device from their arrays, `_modules`). cupyx.scipy.ndimage (CuPy 14.2) has the distance transform, the filters
and labelling: one distance transform of 34M samples 7.2 s -> 0.02 s, the same to 1e-16. On the user's state (39.7M
samples): 87.2 s -> 8.6 s, the same 988,412 triangles, vertices within 5.6e-6 mm, the same volume, thickening, tips
and unreached loads. What it took to get there:
- cupyx's `ndimage.maximum` with labels grows with labels x samples (95 ms for 4M samples and 4k labels; for the
  real grid and its pieces seconds in one kernel: a kernel running that long makes Windows reset the display driver).
  `_label_max` does it in one `cupyx.scatter_max` pass (9 ms), and binary dilation repeated needs `brute_force=True`.
- Warp's marching cubes gives the same vertices but splits cells along other diagonals than scikit-image (9715 of
  12340 triangles): the 10 Taubin passes then moved vertices up to 0.28 mm apart. Marching cubes stays on the host, so
  the shape (and its check) does not depend on whether the GPU had room. The field before it: within 7.6e-6 (cupyx's
  float32 Gaussian).
- Memory: 1.6 GB in use at the peak (a distance transform inside `thicken`), but CuPy's pool held 3.3 GB (88 B a
  sample): freed blocks that did not fit the next steps' arrays. Given back between the steps, 2.1 GB (56 B; 59 B on
  the card, NVML). `SMOOTH_GPU_BYTES` 64, plus the 1.5 GB headroom: 4 GB free for this grid (3.8-5.7 GB seen that day).
  `thicken` and `_surface_distance` now work in place and a gradient axis at a time (the same numbers).
- On running out of memory anyway it does it all on the CPU (`backend.is_out_of_memory`).
- Where the card's 12 GB went (Windows' per-process GPU counters; Firefox's and dwm's are garbage there, 9.6 TB and
  12 GB, the adapter's total is right): with the editor closed 5.5 GB in use, Plex 1.9 GB, Firefox and the desktop
  compositor ~1.2 together, Snipping Tool 0.44, ten Electron apps 0.1-0.2 each. A running editor adds its main
  process's and each GPU worker's CUDA context and caches. So the node makes room before each smoothing as for an
  analysis (`nodes_fea._room_for_smoothing`: this process's Warp meshes, then idle workers) and logs it when the
  smoothing still has to run on the CPU.

Not done: Windows' shared GPU memory (system RAM the card may borrow over PCIe, ~30x slower than its own) would let
the check and the smoothing allocate more, but they read every array many times a step: the engine's GPU timings
came out ~10x slower when Windows had moved allocations there, about the CPU's speed, and it takes from the RAM
and commit that other programs need.

### Max iterations (2026-10-25)

The user asked for more iterations or a setting: every bracket run stopped at the fixed 125. Max iterations
is now a node input (default 125, not in the key): states record whether they `settled`, and one stopped at
the limit goes on when it is raised (lmax vectors kept, so exactly as an uninterrupted run;
`test_a_run_stopped_by_max_iterations_goes_on_when_it_is_raised`). The bolted bracket at Min 1.5, each row
continued from the one before (fine, GPU, ~2.6 s per iteration; deflection of the final body):

| | 125 | 200 | 300 |
|---|---|---|---|
| no pruning (reach rule below, 2 dead ends) | 2.49 mm | 2.46 mm | |
| pruning, whole dead ends locked, band beside thin members | 2.10 | 2.30 | 2.35 |
| pruning, far part locked, near part left at PRUNE_REGROW | 2.12 | 2.57 | |
| the same, nothing locked for good | 2.13 | 2.60 | |
| the same, pruning only until 25 iterations at beta 16 | 2.12 | 2.76 | |

- With Keep % out of reach (Max thickness) the design never settles: the kept share and the stiffness swing
  between checks (iteration 100: 14.2% / 23%, 120: 11.3% / 14%, 125: 11.8% / 16%), the optimizer grows ~70
  new stubs between checks, and after pruning stops it grows them back (the trim then cut 3.2%). More
  iterations do not help here; pruning to the last iteration is the least bad.
- Locked elements piled up against members: after 300 iterations 7123 locked (649 mm3), 3871 of them
  beside the design, 95 pieces of 1-2 elements: quiet patches of member surfaces, counted as dead ends.
  Requiring the trim's dead ends to stick out farther than Min thickness (`dead_ends(reach=)`) keeps it
  off member surfaces; in the optimizer that rule left almost nothing to prune (2 by 125, 2.49 mm), so
  there patches still count but are only lowered to `PRUNE_REGROW` (a variable at exactly 0 never grows
  again), not locked, within Min thickness of what stays.

### Load cases (2026-10-25)

The user asked for different load cases, easy to tell apart, grouped in the nodes and copy-pasteable. Until
then every support and load in a chain acted at once (the bolted bracket was optimized for the pin's 200 N
and the bolts' 300 + 600 N together).

- Model: a Load Case node tags the loads of its Loads chain (`Condition.case`); supports and loads in no
  case act in every case; two Load Case nodes with one name are one case; a chain's supports met twice (in
  Loads and in Add to) count once (by key and case). Without Load Case nodes nothing changes, keys included
  (a case is in a condition's key only when set; one-case states save `u` as before).
- Static Analysis: one multigrid hierarchy, a solve per case; each case's max stress, safety factor and
  deflection in the log, the rest of the report and the colours the worst of them per element (stress,
  displacement). Cantilever: each case equals its own analysis to 1e-6; the cases at once bend more than
  either (`test_load_cases_act_apart_and_the_worst_counts`).
- Topology: the sum of the cases' compliances (energies added in the sensitivities); a solve per case and
  iteration (bracket at draft, 3 cases: 55 s for the whole optimization in the test editor); pruning keeps
  material that carries in any case (the most energy of any).
- Editor: per case a colour (by name order, none the loads' red or the supports' green) on the Load Case
  node and its loads (card accent, name chip), their arrows in the 3D view, and a Load cases legend
  (hover: the other cases' arrows faint; click: its node). Paste of a Load Case with its loads: "Case 3"
  after "Case 1" / "Case 2", joined after the end of the chain of cases, the loads keeping their body (a
  pasted load without its body was useless; Ctrl+V dropped it). Checked in the test editor on the bolted
  bracket: two cases, copy-paste of the second into a third, Optimize at draft: "(the worst of 3 load cases:
  Case 1)". React Flow's multi-select key is read from keydown events: the browser pane's ctrl+click needs a
  dispatched Control keydown first.
- Frame Optimization refuses several cases (its LP takes one set of loads).

### Mesh relaxation instead of Smoothing? (2026-11-02)

Would relaxing the triangle mesh (Taubin, HC, Laplacian) look better than blurring the distance to the
densities' 0.5 surface? Prototype outside the engine (`scratchpad/relax.py`: a copy of `density_mesh` without
the blur, its last 10 Taubin steps replaced by the relaxation, extended 4 elements past the design space as at
Smoothing 2), from saved states: the bolted bracket as its recipe (fine, 0.45 mm elements, Max 4 struts only,
Min automatic 1.35 mm) and the plain wall bracket (`samples/wall_bracket_space.step`, Keep 25%, medium, 0.71 mm,
Min 2.13 mm; optimized once for this, 114 iterations). Every body thickened to Min thickness before meshing (as
the engine does), cropped as the node does, measured on the Final view mesh and analysed as the node's check.
Moved = the share of the unsmoothed body's solid that changes side. Surface under Min = surface samples (h/4
apart, away from supports and loads) whose local thickness is under 0.9 x Min:

| method | wall: moved, surface, deflection | bolted: moved, surface, deflection, under Min |
|---|---|---|
| none (0.5 surface, 10 Taubin steps) | 0, 101.6 cm2, 0.123 mm | 0, 118.0 cm2, 2.14 mm, 0.24% |
| Smoothing 2 (now) | 5.8%, 97.2, 0.125 | 9.1%, 114.1, 2.11, 0 |
| Taubin λ 0.5 / μ -0.53, 100 pairs | 0.8%, 101.3, 0.123 | 1.4%, 118.4, 2.12, 0.23% |
| Taubin, 400 pairs | 1.9%, 102.1, 0.123 | 5.1%, 121.0, 2.07, 0.08% |
| HC (α 0.1, β 0.5), 50 passes | 0.3%, 101.2, 0.123 | 0.3%, 117.7, 2.14, 0.22% |
| Laplacian 30 + volume put back | 2.2%, 99.3, 0.122 | 4.4%, 115.3, 2.14, 0.33% |
| Laplacian 100 + volume put back | 5.3%, 97.3, 0.121 | 10.8%, 112.8, 2.12, 0.65% |
| Smoothing 1 + Taubin 100 | 1.9%, 100.3, 0.123 | 4.2%, 116.3, 2.10, 0 |

Mass within 1% of the unsmoothed body throughout (Taubin 400 on the bolted bracket +4.4%), stiffness the same
within 2%. The safety factor swings with one notch's peak (bolted 0.13-0.27, wall 9.0-9.5), so the deflection is
the measure. Time is no argument either way: the filters take 1-2 s next to 6-20 s of distance transforms and
marching cubes (Taubin 400: +7 s).

- Taubin and HC leave the terraces: they keep low frequencies by design, and an element staircase on a gently
  sloped face has its steps a tread apart, many elements on a gentle slope: in their passband. Only the step
  corners round (the 2026-10-15 "60 instead of 10 Taubin passes: no visible change"). 400 pairs made blotchy
  bumps (more surface, +4% mass on the bolted bracket); HC pulls back to the original and hardly moves at all.
  Steps along the bolted bracket's bolt bosses (faces cut by the design space) stay too.
- Laplacian (umbrella) with the volume put back by an even offset along the normals: at 100 passes as smooth as
  Smoothing 2 on broad faces and the bosses, same mass and deflection. But a strut's cross-section curves as
  sharply as the terraces, so it shrinks thin members and the even offset gives the volume to thick ones (section
  x = 25: every thin strut inside the others' outlines; 0.65% of the surface under Min, and `thicken` cannot run
  after it: it acts on the field). A hollow knob in the densities (a thin shell at about (32, 8, 27)) collapsed
  into crumpled sheets sticking 3 mm out: broken geometry. Smoothing 2 drops that shell (thinner than the blur),
  Taubin and HC keep it lumpy. trimesh's own `filter_laplacian(volume_constraint=True)` scales the whole mesh
  about its centre of mass: useless for a cropped part.
- Smoothing 1 + Taubin 100: between Smoothing 1 and 2 to look at; no gain.
- Wall bracket section (y = 15): Taubin follows every element step of the 0.5 surface; Smoothing 2 and Laplacian
  100 both cut straight through them and both fill a sharp opening's tip a little (Smoothing 2 more: it nearly
  closes a 1 mm hole that the Laplacian keeps).
- Kept as it is: mesh relaxation either leaves the element steps (Taubin, HC) or acts as an even blur without
  the light blur for thin members, the level match and `thicken` (Laplacian), and on a mesh it can tear. Renders
  (session of 2026-11-02): `scratchpad/wall_closeup.png`, `wall_section.png`, `bolted_overview.png`,
  `bolted_closeup.png`, `bolted_section.png`.

### The filtered field, and joint fillets (2026-11-03)

Two more ways, on the same two states and measured the same way (`scratchpad/capture2.py`, `smooth2.py`):

- **Mesh the optimizer's filtered field** instead of the densities. The densities are nearly binary (2.5% of the
  wall bracket's design space between 0.05 and 0.95, 8% of the bolted one's), so their 0.5 surface follows the
  element faces. The filtered field xt = `Problem.filter(x)` crosses `ETA` exactly where the projected density
  crosses 0.5 (same elements solid: 0 mismatches on both states, rebuilt with `T.prepare(h=)` from the state's
  `x`), but it runs smoothly across the surface (a cone over the thinnest member), so it places the surface
  between the elements. Field: `where(solid, 1, xt) * domain`, the dead-end cut set to 0, then `density_mesh`
  unchanged.
- **Joint fillets: a closing** (grow by r, shrink by r: a rolling ball in every inside corner) after `thicken`.
  Three things were needed. (1) Exact distances: distance transforms measure only to the nearest sample, and
  fillets made from them were lumpy. Each sample instead takes the distance to the surface point of its nearest
  sample next to the surface (a Newton step), via the transform's feature map. (2) A closing fills every opening
  narrower than 2r: unguarded, r = 2 x Min filled the wall bracket's 5 mm window and its 1 mm hole (safety 9.4 ->
  11, +4.6% mass) and the bolted bracket gained 36% mass. Pieces of the added material that change the solid's
  Euler number (`skimage.measure.euler_number`) are dropped. (3) Between members closer than 2r it leaves webs
  ending in a knife edge, which the Euler number misses since the opening is not closed: the bolted bracket's
  safety fell 0.20 -> 0.08, stress away from supports 222 -> 578 MPa. What a ball of Min diameter cannot
  reach inside the closed body is taken off again (an opening of what was added).

| method | wall: mass, moved, surface, deflection | bolted: mass, moved, surface, deflection, safety |
|---|---|---|
| none (0.5 surface) | 28.12 g, 0, 101.6 cm2, 0.123 mm | 12.84 g, 0, 118.0 cm2, 2.14 mm, 0.25 |
| Smoothing 2 (now) | 28.48, 5.8%, 97.2, 0.125 | 12.92, 9.1%, 114.1, 2.11, 0.20 |
| filtered, Smoothing 0 | 28.43, 2.1%, 101.0, 0.122 | 13.05, 3.7%, 119.1, 2.12, 0.25 (0.31% under Min) |
| filtered, Smoothing 1 | 28.43, 3.0%, 99.8, 0.123 | 13.00, 5.6%, 115.6, 2.12, 0.20 |
| filtered, Smoothing 2 | 28.52, 5.9%, 97.2, 0.125 | 12.99, 10.1%, 114.1, 2.10, 0.27 |
| Smoothing 2 + fillets r = Min | 28.59, 6.1%, 96.9, 0.124 | 12.98, 9.4%, 114.3, 2.09, 0.22 |
| Smoothing 2 + fillets r = 2 x Min | 28.82, 6.9%, 97.2, 0.124 | 13.17, 10.6%, 115.0, 2.02, 0.21 |
| filtered, Smoothing 1 + fillets r = Min | 28.61, 3.6%, 99.4, 0.122 | 13.05, 5.9%, 115.8, 2.10, 0.21 |

Wall safety 9.3-9.4 throughout. Nothing under Min except where noted. Mesh time the same for the filtered field;
the fillets add 3-10 s (four exact distance passes on the 2x grid).

- The filtered field at Smoothing 1 looks as smooth as Smoothing 2 of the densities on both parts (zebra stripes
  alike, bosses smooth) and moves about half as much material. The unsmoothed surface is ±h/2 steps, so removing
  them alone moves ~2-4% (filtered at Smoothing 0). Smoothing 2 of the densities moves 2.5x that, rounding things
  off as well. It keeps what the blur loses: the wall bracket's 1 mm hole stays open, an opening's tip is not
  pulled back by 2 mm, and a strut leaving the y = 8 section of the bolted bracket keeps going where Smoothing 2
  ends it. Same mass and stiffness. At Smoothing 0 it is not enough: faint ripples on the wall bracket and, on
  the bolted one, the hollow knob stays as a lumpy blob with loose bits. The bolt bosses stay stepped there
  because the solid skin around supports is whole elements in the field too; a smooth version would be the skin's
  own distance (reach minus the distance to the faces).
- Fillets change little on topology results: their members already meet in smooth joints. r = Min hardly shows.
  r = 2 x Min rounds the inside corners of the wall bracket's big window (section y = 15) and stiffens the bolted
  bracket 4.5% for 1.9% mass, but still makes webs at least Min thick in acute openings (section x = 28.8). On
  the bolted bracket it half fills an oval opening (a ragged hole) and leaves lumpy patches. The safety factor
  does not move: these parts' weakest spots are not in a joint.
- Worth building: the filtered field (with Smoothing 1 as the default it gives today's look and a truer shape,
  at no cost). It needs the filter at result time: `T.prepare` on the state's grid rebuilds it (domain, the
  gap's norm, the solid skin), or the state saves xt. Fillets as they are: no.
  Renders: `scratchpad/bolted_new.png`, `bolted_fillets.png`, `bolted_section2.png`, `wall_filtered.png`,
  `wall_section4.png`.

### Stress-driven surface growth after topology (2026-11-03)

Where the parts' stress peaks (`scratchpad/peaks.py`, the node's check of the Smoothing 2 body, strength
45 MPa): the wall bracket's highest stress is 4.8 MPa, at its load face (median 1.3): nothing to even out. On the
bolted bracket one spot dominates: 222 MPa (element) at (2, 10, 33), in a notch between two members 6 mm from a
load. The next spots are 60-77 MPa, 2.6-4.7 mm from the bolt holes. 99% of the part is under 41 MPa, the median
is 8.5. So shape optimization can only help the bolted bracket.

Prototype (`scratchpad/cao.py`, CAO / "biological growth" at constant mass), on the bolted bracket's Smoothing 2
body as a distance field on h/2 = 0.225 mm. Each round:
- the node's Static Analysis (fine);
- the von Mises stress of the samples within an element inside the surface, carried out by nearest sample and
  blurred over an element;
- the surface moved by 0.25 mm x clip(stress / reference - 1, -0.5, 1), the reference found by bisection so the
  material in the design space stays;
- frozen within skin + an element (2.45 mm) of the support and load faces, fading in over 1.5 mm;
- redistanced (`exact_distance`), blurred 0.75 element, `thicken` to Min, specks dropped, and outside the design
  space no more solid than its nearest sample inside.

~25 s per round.

| body | mass | safety (fine / medium) | deflection | surface stress p99.9 | surface |
|---|---|---|---|---|---|
| Smoothing 2 (now) | 12.92 g | 0.20 / 0.24 | 2.11 mm | 66 MPa | 114.1 cm2 |
| growth, round 3 (best) | 13.00 | 0.54 / 0.53 | 1.80 | 33 | 116.5 |
| growth, round 8 | 12.86 | 0.43 / 0.48 | 1.73 | 21 | 117.1 |

- It works as a stress fix. The notch peak goes in 1-3 rounds, the stress evens out (the surface's 99th
  percentile 46 -> 26 MPa by round 8), and the part also gets 15-19% stiffer at the same mass. Taking material
  from what carries nothing helps compliance too. The same gain at medium (0.35 mm elements) as at fine (0.22),
  so it is not tuned to one analysis grid.
- The engine's safety factor is one element's stress, so it jumps between rounds as the worst spot moves (round 7
  of the run above: 465 MPa at one element, 0.10, between 0.48 and 0.43). Judge rounds by a percentile of the
  surface stress and report the element maximum. After round 3 the worst spot is next to a bolt hole, at
  fully fixed faces whose peak grows with detail: not the part's.
- What went wrong on the way:
  - Growth outside the design space was not held. The body ballooned there and left paper-thin sheets on the
    design space's faces inside it, and the mass control thinned the members to pay for them (safety 0.47 at
    +7%, a mess).
  - Shrinking at the full step everywhere thinned what topology made, and correcting the mass by the summed
    drift instead of the last round's lost 14% of it.
  - A hard edge to the frozen zone at supports made blocky, ribbed tubes round the bosses: growth piles up
    against it (their artificial peaks attract it).
  - With a 0.5-element blur, beads on struts and wavier zebra stripes than Smoothing 2.
  - The faded edge and a 0.75 blur fixed most of it. Small ribbed strips remain along two boss edges where they
    meet the design space's faces, and by round 8 low-stress lumps grow there.
- As an engine feature: a "refine for stress" stage after Topology Optimization's result (a few rounds, saved by
  key like the states, the best round by surface-stress percentile), ~25 s per round at fine on this part. To
  solve first: the bosses' edge strips, stopping when the percentile stops falling, and supports' artificial
  peaks attracting growth (start the fade further out from fixed faces). It cannot make a part strong enough
  that is too light for its loads: with its stress evened out at safety ~0.5, the bolted bracket would need
  roughly twice the material for 1 (an estimate, not run).
  Renders: `scratchpad/cao_stress.png` (stress colours), `cao_v4_try.png` (shapes).

### Polishing: a smooth field instead of the mesh, and an even-out flow (2026-11-03)

The user found the bolted bracket "a bit jagged, not polished". Close-ups (`scratchpad/stria.png`) show two scales:
- Stripes at the element spacing along every member, plus triangle facets on top. They are in the node's
  body, not in the smoothed field. `density_mesh` marching-cubes the field at h/2, the node takes the exact
  distance to that mesh (`body_from_mesh`), and the view marching-cubes that again at its own voxel. Each
  facet of the h/2 mesh shows in the shading.
- Uneven members: bulges and necks along struts, stubs, knobs, creased joints, small holes. The blur cannot
  touch these on thin members, which only get LIGHT_BLUR (a blur thins a strut at ~sigma^2 / 2r).

Tried (`scratchpad/fine.py`, `flow.py`, `polish.py`; bolted bracket, fine, h = 0.45 mm, Min 1.35 mm; every body
cropped, Final view, the node's check; loose = pieces of the view mesh apart from the part, `pieces.py`):

| body | mass | safety | deflection | surface | loose |
|---|---|---|---|---|---|
| now (Smoothing 2, distance to the h/2 mesh) | 12.92 g | 0.20 | 2.11 mm | 114.1 cm2 | 0 |
| smooth field: the same values, cubic B-spline, no mesh | 12.95 | 0.23 | 2.08 | 114.3 | 8 (0.01 mm3) |
| smooth field, Smoothing 3 / 4 | 12.95 / 12.96 | 0.28 / 0.28 | 2.07 / 2.05 | 114.6 / 115.4 | |
| smooth field + even-out flow 0.8 mm2 | 12.95 | 0.22 | 2.00 | 113.0 | 11 (0.005) |
| the same flow, not gated at ridges | 12.95 | 0.38 | 1.94 | 111.9 | 443 (0.19) |
| the gated flow at 1.5 mm2 | 12.94 | (solver did not converge) | | 112.0 | 161 |
| mesh normal filtering 30 / 100 / 300 passes | 12.92-12.97 | 0.27 / 0.21 / 0.24 | 2.11-2.13 | 112.0-113.9 | |

Wall bracket (medium, h = 0.71): now 28.48 g, safety 9.36, 0.125 mm, 97.2 cm2; smooth field 28.48, 9.37, 0.125,
97.3, 5 loose (0.15 mm3); with the flow 28.47, 9.32, 0.126, 96.6, 2 loose (0.02 mm3).

- **Smooth field** (`fine.cubic_body`): density_mesh's own final values (h/2 samples, after blur, `thicken`,
  speck removal and the crop clamp) as the body, interpolated by a cubic B-spline (`ndimage.spline_filter` once,
  then `map_coordinates(order=3, prefilter=False)`). The surface is C2, so the stripes and facets are gone on both
  brackets (`stria2.png`), at the same mass and stiffness. A pure function of position, so block meshing stays
  exact. Costs: the Final view took 16 s on the CPU against 7.3 s for the mesh's distance (GPU), so the engine
  version wants CuPy's `map_coordinates`. B-splines ring a little between samples: a few islands of 0.01-0.1 mm3
  next to near-solid samples. Push every piece that is within a third of a sample of solid, and apart from the
  part, out of it before interpolating; that took the wall bracket from 5 to 2.
- **Even-out flow** (`flow.kmin_flow`): the level set moves inwards by its smaller principal curvature, run after
  the blur and before `thicken`, and the material put back by the level. Along a straight strut k_min = 0, so
  struts and plates keep their thickness. Bulges and necks even out, joints get fillets, stubs pull back and small
  holes in plates shrink. Mean curvature flow does all that too, but also thins a strut at 1/r. It makes the part
  look finished (`polish_final.png`). The bolted bracket deflects 2.00 mm instead of 2.11 at the same mass, with fewer small holes,
  even struts, and a clean fillet where two members cross. What it took:
  - A soft min, (k1 + k2 + d - sqrt((k1 - k2)^2 + d^2)) / 2 with d = 0.3 /mm. The hard min wrinkled plates within
    40 steps: where k1 ~ k2 it diffuses in one random direction only.
  - Compact 3-sample second derivatives. np.gradient twice cannot see a sample-to-sample zigzag, so the flow never
    damped it: fine wrinkles everywhere.
  - No `exact_distance` before or after. Its Newton surface points differ a little between neighbouring samples,
    and the cubic field showed that as stripes again.
  - Motion only near the surface, faded out between 3 and 6 samples, and not where |grad| < 0.5 (fading in to 0.8).
    Half way between members the distance has ridges the samples cannot resolve. A fixed band let the flow pull
    them under zero: 443 loose flakes in the gaps and a pinched waist where two members cross. A hard band found
    again each step crumpled the surface where it crossed it and broke a lump off.
  - Explicit steps of 0.1 voxel^2; curvature capped at 1/voxel.
  - Cost: 159 steps, 200 s on the CPU for the bolted bracket at fine (14M samples); 25 s for the wall bracket at
    medium. Plain array maths, so CuPy would run it on the GPU as it is.
  - At 1.5 mm2 the gate was not enough (161 flakes, and the analysis did not converge). 0.8 mm2 (~4 h^2) is the
    setting that worked.
- **Smoothing 3-4** on the smooth field rounds the plates and holes further, but the thin struts stay beaded
  (light blur, then `thicken`). The flow is what evens them.
- **Mesh normal filtering** (`polish.py`: iterated 1-ring normal averaging, then vertices fitted to the normals,
  Sun et al. 2007, volume put back): 30 passes changed little. 100-300 passes folded the marching-cubes triangles
  (creases, specks) and still left the stripes. Not pursued.
- Engine plan: (1) the smooth field as the node's body, with the margin cleanup, the interpolation on the GPU
  where there is one: cheap, removes the jaggedness. (2) The gated flow in `density_mesh` after the blur, on the
  GPU, time ~4 h^2. Before it ships: a loose-piece check in a test (flakes break the analysis), and more parts than
  these two brackets.

### Thin fins and necks: Members "struts only" never settles (2026-11-03)

The user still saw "small thin parts connecting the struts and in the web" on the bolted bracket (fine, Keep 20,
Max 4, struts only). Wall thickness by rays (as the print check) on the body: 2.8% of the surface under 1.5 mm,
10% under 2 mm (Min thickness 1.35), in 33 spots away from supports and loads: flat fins and tabs between struts,
necks where struts meet, thin patches in the webs, 0.7-1.3 mm thick (`scratchpad/thin_close.png`).

- They are in the optimizer's design, not made by smoothing: the middle of every thin wall lies inside the
  unsmoothed 0.5 surface (99% of the spots' vertices). The densities there are undecided: 0.24-0.82 at the spots,
  their 5 x 5 x 5 neighbourhoods 54-78% grey (0.2-0.95) and almost no solid element. The 0.5 level runs through
  that grey as a thin sheet. `thicken` misses them: it measures a member by the deepest point within one Min
  thickness, and these hang right beside a thick strut.
- Why grey: the struts-only optimization never settles. From sharpness 8 on it flip-flops every iteration at the
  move limit (change 0.1 each iteration; the eroded compliance alternates 26k / 74k, volume 11.3 / 12.3%) until Max
  iterations stops it, so the result is one half-step of an oscillation: 25% of the material elements grey (0.2-0.8).
  The design variables there are intermediate too (82% between 0.05 and 0.95). Its local density is 0.25 at the
  spots, far under LOCAL_SHARE: no neighbourhood is held full there.
- Every saved bolted-bracket state with Max thickness (125 and 200 iterations, draft and fine) is unsettled with
  20-25% grey; every state without it settled with 4-8% (the one-element edge of every member).
- Medium tests of the bolted bracket (`scratchpad/osc_run.sh`, its own cache):

  | settings | settled | kept | grey | compliance over the last 10 iterations, max / min |
  |---|---|---|---|---|
  | as the recipe: Max 4, struts only, Keep 20 | no (125) | 11.1% | 25% | 1.60 |
  | Max thickness off | yes, 76 | 19.7% | 5% | 1.00 |
  | Max 4, struts only, Keep 12 | no (125) | 8.5% | 19% | 1.43 |
  | Max 4, struts only, Keep 7 | no (125) | 5.8% | 13% | 3.00 |
  | Max 4, plates and struts, Keep 20 | yes, 124 | 16.7% | 9% | 1.00 |

  A lower Keep % does not help: struts only never reaches its target (7% kept 5.8%), so the volume's price sits at
  `PRICE_FLOOR` whatever Keep is. Plates and struts with the same Max thickness settles. Bodies
  (`scratchpad/osc_look.png`): struts only 2.8% of the surface under 1.5 mm, plates and struts 1.4%, Max off 1.9%
  (the last two mostly at the pin's cut edges); the masses differ (9.8 / 14.8 / 17.4 cm3).
- Suspects for the oscillation (not tested): `_update_two` with the volume at its floor, so only the one aggregated
  p-mean (p = 16) holds the material, linearized; `local_scale` re-averaged every iteration; struts only's filter
  radius 2.6 mm with ERODED 0.8 at sharpness 16.

### Trying to make struts only settle (2026-11-03): no fix found

The user asked to fix the optimizer. Prototypes patched into `Optimizer._update_two` (`scratchpad/osc_fixes.py`,
`osc_driver.py`: the recipe's node, instrumented, own cache). No engine change.

- **Mechanism.** The max thickness is never met: the fullest neighbourhood stays at 64-68% against 60%, so `l2` is
  active every iteration, the volume's price sits at `PRICE_FLOOR` from iteration 20 on, and the linearized g + dg
  dx <= 0 overshoots: `l2` alternates 0.4 / 1.4, 13% of the elements move by the full move limit and 14% reverse
  direction every iteration. Why it cannot be met: in struts only a strut of Max thickness fills exactly
  LOCAL_SHARE at its axis, so every junction overfills its neighbourhood, and members must still be at least the
  filter radius (0.65 x Max = 2.6 mm) to carry in the eroded design. The allowed range, 2.6-4 mm, has no room for a
  junction.
- Medium (Keep 20, Max 4, struts only); thin spots = patches of the body under 1.5 mm, 4 mm away from supports and
  loads (Min thickness there is 2.13 mm):

  | variant | settled | kept | stiffness | thin spots |
  |---|---|---|---|---|
  | now | no | 11.1% | 14.1% | 32 |
  | per-element move limits (x 0.7 on a reversal, x 1.2 back, 0.01-move) | at 135 of 250 | 13.7% | 14.5% | 72 (at 125) |
  | max thickness evaluated exactly in the root finding, as the volume | no, swings 4.6x | 14.6% | 2.0% | |
  | move limits + PRICE_FLOOR 0.03 | at 109 | 11.8% | 14.8% | 56 |
  | move limits + PRICE_FLOOR 0.1 | no | 8.2% | 8.9% | 7 |
  | PRICE_FLOOR 0.1 | no | 10.1% | 12.8% | 20 |

  With the move limits the medium designs become real strut networks (no webs) but pinched where struts meet.
- Fine (the user's case, start coarse):

  | variant | kept | stiffness | thin spots | body |
  |---|---|---|---|---|
  | now | 11.8% | 16.2% | 33 | 12.96 g, safety 0.20, 2.11 mm |
  | move limits | 13.4% | 14% | 130 (thin plates) | 14.7 g, 0.25, 2.34 mm |
  | PRICE_FLOOR 0.03 | 11.7% | 15.7% | 37 | |
  | move limits + PRICE_FLOOR 0.03 | 10.9% | 12.8% | 76 (whole struts 1.5-2 mm) | |

  None settles in 125 iterations at fine; the move limits cut the swing (1.08 against 1.6) but leave more grey and
  thin material, not less (`scratchpad/osc_fine.png`).
- The thin material is not idle. 40% of the design's elements have no core that carries in the optimizer's model
  (no eroded element within the erosion width, 0.87 mm); dropping all of it left a loaded piece without a path to
  the supports. Only 1.1% of the design is both such and under 10% of the median stress in the body's Static
  Analysis; trimming that (two rounds) changed little (12.81 g, 0.208, 2.13 mm, thin surface 1.17% -> 1.07%). Half
  the struts are under 2.6 mm (median wall 2.65 mm): the eroded model says they carry nothing, the part says they
  do.
- `thicken` with a true local thickness (largest inscribed ball, as an opening per radius step, `thick2.py`) sees
  necks and fins beside thick members, but at medium most members are within a sample of Min thickness and it added
  80% material. Not pursued.
- What works today: Members plates and struts with the same Max thickness (settles, no thin spots), or no Max
  thickness. A struts-only that settles needs a formulation where junctions fit (a cap measured per member, not per
  neighbourhood) — not tried.
- Frame Optimization on the same bracket (PETG, the recipe's supports and three bearing loads, round bars;
  `scratchpad/frame_run.py`, `frame_cmp.png`): Safety factor 1: 26 bars, body 17.5 g, safety 1.11, deflection 1.00
  mm (52 s), clean struts with G2 fillets, lumpy only where bars meet the bolt bosses' skin. Safety factor 2: the
  body check stops at 1.56, limited by the 600 N pin pressing on face 8, but first thickens bars for 6 rounds
  (up to 30.5 mm, the design space nearly full: 44.8 g, 0.53 mm). The topology result for comparison: 12.9 g,
  safety 0.20, 2.11 mm.

### Frame Max size (2026-11-03)

A Sizes list capping the bars at 6 mm failed at once ("a bar needs more than the largest size in the list"): the
layout did not know the cap. Max size now bounds each candidate's tension and compression in the layout LPs (and
the joint moves' LPs) by what a bar of that size carries over its length (`sizing.force_limits`), times a share.

- The share is needed: the layout splits a load between parallel bars freely, the elastic beam check splits it
  by stiffness and adds bending at the clamped joints, so bars at the cap came out over it. Block 60 x 20 x 30 mm,
  200 N, PLA, safety 2 (unlimited: 37 bars, 2-4.5 mm, 10.5 g): Max 4 at 0.8 passes (37 bars, 9.6 g); Max 3 at
  0.8 had a bar at 1.38 of its strength, at 0.6 it passes (163 bars, 2-3 mm, 18.6 g, 38 s). `MAX_SIZE_SHARES` =
  0.8, 0.6, 0.45 are tried in turn (`MaxSizeShort`), with a note naming the share used.
- Bolted bracket, PETG, safety factor 1 (`scratchpad/frame_max6.png`): unlimited 26 bars of 2-10 mm, body 17.5 g,
  safety 1.11, 1.00 mm, 52 s; Max 6 mm: 42 bars of 2-6 mm, 20.0 g, safety 1.05, 0.98 mm, 181 s (0.8 and 0.6
  failed, 0.45 passed: three layouts). Where one thick bar was, the layout puts parallel bars between neighbouring
  lattice nodes; they run side by side and the fillets fuse them into wider, flatter members.
- The user asked why every bar meets in the middle of the long hole (the 200 N bearing load, 8 mm x 30 mm) and
  whether spreading them would be stronger. A hole is one node on its axis (a pin, F1 item 2); the tube round it
  (2 mm skin, 12 mm across) carries the load along to that node at ~5.5 MPa (12% of PETG's 45). Forced to
  `ground` "around" (6 nodes on the pressed wall; `scratchpad/frame_spread.png`): 65 bars, body 25.6 g, safety
  1.07, 0.74 mm, 78 s, against 17.5 g, 1.11, 1.00 mm: 46% heavier for the same safety (stiffer, though). The
  body check's weakest spots stayed at the 600 N hole and the wall, never at the long tube. With Max 6 mm the
  spread layout failed at every share (a 6 mm bar at 1.32 of its strength at (7, 7, 20)). Spreading pays only
  when the load may not act where it is modelled (a hook at one end of the tube): model it there (an Offset
  along the axis gives the face several nodes, `TURNING_NODES`).

## Frame optimization F1 (2026-10-15)

Plan: docs/FRAME_PLAN.md ("F1 as built" lists what differs from it). Engine only, no node yet.

- Checked against textbooks (`tests/test_frame.py`): sections of every profile against a pixel
  integration (0.5%); two- and three-bar trusses and Hemp's 45 degree two-bar truss off a long wall
  (V = 2 F L / sigma) exactly; member adding = the full LP; cantilever deflection and fixed-end
  moment to 1e-9 (with and without shear); a clamped column's buckling factor within 1% of Euler.
- First run on the bolted wall bracket: 323 bars, 16.6 g, 108 s, a tangle. Three causes:
  1. The plain LP spreads the load over hundreds of bars carrying a few N (1.7 g of plastic
     layout); at the 2 mm smallest size that is 17 g. Fixed by reweighted LPs that price weak
     bars at the smallest size's real cost: 170 bars, 9.4 g.
  2. Bolt and pin nodes were off the lattice and bars reached only 3 spacings, so a straight bolt
     to pin bar could only be imitated by zig-zag chains; and the pin's bearing load was split over
     6 nodes on the hole's wall, whose sideways pushes needed bars across. Fixed: a hole (or a
     load whose pushes cancel) is one node at its centre, and support / load nodes connect to every
     node: 24 bars at joint cost 0, 9 bars (3 joints, 3.8 g) at 0.02-0.05.
  3. All four bolt holes were one Fixed Faces whose normals cancel like one hole's: one support
     node in the middle, which cannot hold a moment (HiGHS: "status unknown, primal infeasible").
     Supports and loads are now split into connected pieces of faces first.
- Member adding stopped at its 60-round limit on 15,000 candidates (107 s); solving them all at
  once is faster here: HiGHS interior point 0.9 s, its simplex 2.5 s, same optimum. Whole runs
  now 11-16 s, mostly the 12 reweighted LPs.
- Sideways load: 80 N along the pin on top of the 200 N bearing load (a second Load on the pin;
  both land on one merged node) makes the truss lean over and thicker on one side: 9 bars, 4.4 g.
- Against SIMP (bolted, Keep 20%, max 4 mm): 19.3 g and 0.199 mm (voxel) vs the frame's 3.8 g and
  0.84 mm (beam model): about the same stiffness per gram; the frame is sized for strength.
- First editor version (node, round bars, plain joints, skin): the skin was first the size of the
  open support / load faces' signed distance field (`abs(body_from_mesh(...)) - skin`): away from
  the exact band it is interpolated with arbitrary signs, which showed as stair-stepped sheets and
  specks. Now the exact distance per connected piece of faces (libigl AABB, in the piece's box).
- Voxel Static Analysis of the demo's body agrees on deflection (1.05 mm vs 0.91 mm from the frame
  check) but shows 81-92 MPa (24 allowed) at the sharp corner where a bar meets a bolt boss, growing
  with finer elements. The frame check cannot see joints: fillets are needed (F2).
- The layout used only 3 of the 4 bolts (the fourth carried nothing in the optimum), but the skin
  put a boss around every hole: a loose ring in the part. Supports the layout leaves out are now
  joined to the frame by the shortest bars to a node that is not held (option Supports: connect
  all), or get no skin (only needed).
- Profiles and curving (FRAME_PLAN.md, "Profiles, joints and curving"): rounding only free two-bar
  corners changed nothing on the bracket (every bend there has a third bar); a curve through the
  joint (control point 2v - (A+B)/2) curves the knee. The I section's 12% flange and web were
  0.3 mm at 2.5 mm (mesh in 41 pieces): now at least 0.8 mm. A ball at the pin node bulged out of
  the eye: no balls at bolts and pins. Extra mesh surfaces were all cavities (negative volume):
  closed tube ends, and two I bars side by side enclosing pockets.
- The user found the curves unnatural and the ball joints clumsy. Curving now makes each path of
  bars one spline through all its nodes (no local bends), and the joints are fillets: one Warp
  kernel draws every profile and blends only *different* bars (the two nearest at each point),
  also into the bosses. The first spline version curved nothing on the bracket: the curve check
  failed next to the pin, whose node sits in its hole (now exempt within the node's free radius).
- Mid-bar joints, joining frames, frame view (FRAME_PLAN.md, last section; 2026-10-16):
  Nodes added along the used bars (a second LP) were used in none of three cases: a pin-jointed
  truss cannot land a branch on a straight bar (the sideways force needs a kink), and the lattice
  already has the landing points that pay; finer points (half / quarter spacing) took over 12 min
  instead of 2 s. Bars run on through joints only through curving (block: 54 bars instead of 75,
  through 21 of 28 joints, but +22% mass); with curving 0, mid-bar on and off give the same frame.
  Two design spaces that only touch: the union's field is exactly 0 along the shared face, so the
  inside check refused every bar across it and cropping cut the body in two. A frame connected to
  another failed with "Factor is exactly singular": a lone 0.1 N bar the LP left (its neighbours
  fell under the drop share) floated free; leftovers that reach no support are dropped now. The
  reweighted LPs cycle between a few used sets; stopping on a repeat made runs 2-4x faster.

### Supports on one line (2026-10-25)

The user's frame recipe (bolted bracket held by its two lower bolts only, faces 5 and 7; loads on the pin
and on the two upper bolt holes) failed on Optimize with "no frame inside the design space can carry the
loads to the supports", at any node spacing (the demo, all four bolts held, worked). A bolt hole is one
node on its axis, so two held holes are two pin joints on one line, and a pin-jointed frame turns about
it: no load with a lever about that line (the pin's 200 N alone already) can be carried. Now
(`ground._on_a_line`) such supports get two nodes per hole on its axis, a quarter of its length in from
each end (four points not on one line); held at points on the hole's wall instead, most bars from them
cut through the hole or skimmed the wall's face ("no bar reaches it"). Other recipes keep one node per
hole (bit for bit: the same random draws). The recipe at automatic spacing (5.6 mm): 30 bars, 41 g, beam
model 0.97 mm; its body reaches safety factor 1.54 of 2 (the 600 N on the upper bolt hole: thicker bars
there did not help). At the user's 2.68 mm spacing (139k candidate bars) the LP ran over 30 minutes.

With round bars the same recipe then failed "the frame did not pass its check after resizing": the
beam check's most stressed bar was a 0.5 mm piece of a curved bar at a held bolt node, mostly bending,
47 MPa falling 0.3-0.6 MPa per 0.5 mm thicker (stiffer, it draws more moment), still 26.5 of 24 MPa
allowed after the 12 rounds (square bars got through). `MAX_ROUNDS` 30: through at round 17 (34 bars,
21.4 g, beam model 1.02 mm); frames through within 12 rounds come out the same. The error names the spot
now. The body check then thickened three rounds (1.29, 1.39, 1.56) and a fourth made it 1.13 (another
spot took the stress), which it kept: `_size_for_body` keeps the safest round now (1.56, 31.6 g of bars
instead of 1.13 at 36.2 g).

The user asked whether it tried moving or splitting the bars at the weak spot: it did not (one layout,
then only thicker bars there). Now (`nodes_frame._spread_where_weak`, `FRAME_VERSION` 4): when the body
stays under the safety factor within the skin + `SPREAD_NEAR` of a support's or load's faces, those faces
are spread (`ground` mode "around": 3-6 nodes on the wall, a load's only where it presses, set a quarter
spacing into the material, the same total) and the frame laid out and sized again, the safer body kept, up
to `SPREAD_ROUNDS`. On the bracket the 600 N hole spread over three nodes gave 79 bars instead of 34 and
1.37 instead of 1.56: the first layout kept. The spot (1, 6, 39) is on that hole's own pressed wall at its
open edge (~31 MPa against 24 allowed): Skin 3.5 and 5 mm gave 1.55, so it is the bearing pressure itself.
The warning says so now when the weakest spot is on a load's faces (stronger material, lower safety
factor, more face: a longer hole or a bigger pin), instead of suggesting a larger Skin.

Holes on one axis (the user's bracket_mid.STEP: Fixed Faces on two lugs' holes, r 12.5 mm, both along
x at y 0, z -73.6, as an axle's): the axis nodes of both holes were on that same line, so "no frame inside
the design space can carry the loads" again. When the held nodes are still on one line after that, each
hole is held at nodes round its wall (`_piece_nodes` "around", set into the material), as Fixed Faces
clamp the hole all round in Static Analysis (`test_holes_on_one_axis_are_held_round_their_walls`). The
layout then took 4 s (93 bars, 115 g). The run took 21 minutes in all, but only because of its body
checks: each a Static Analysis of 192 x 80 x 176 elements (0.76 mm; the frame fills 6% of the design
space's box) that went to the CPU, 312 s (solve 306 s, 382 CG iterations), against 12 s on the GPU
(solve 6 s). It ran beside the user's idle editor, which held most of the card: 2.3-2.7 GB free, and
`_gpu_fits` wants the estimate (1.4 GB; measured 1.3 GB) plus `GPU_HEADROOM` 1.5 GB. In the editor
itself `_array_module` stops the run's idle workers (0.8 GB of GPU each) before it settles for the CPU.
The face numbers of the recipe's loads came partly from the earlier STEP file: 10 (a whole side face),
6 (a bottom face), 8 and 15 (the end face and a large face) next to the holes 2 + 28 and 4 + 26.

### Bars that carry nothing; Show (2026-10-26)

"Does the frame have parts that take no load?" Per bar: the beam model's utilisation and axial force and
the body check's von Mises stress at 0.7 of the radius round the middle of each piece clear of the bosses.
The bolted bracket (27 bars): none idle; the lightest three, all at the 2 mm Min size, 9-11 N (6-15% used,
6-12 MPa against 20-100 elsewhere). The layout keeps only bars with a force, so idle bars came from two
faults, both on bracket_mid:
- Four bars at 0 N joined support nodes to a lattice node held by support nodes only: Supports: connect
  all joined every unused support node, and a hole held round its wall (the axle case) has three, of which
  the layout uses one or two. Ground nodes now carry their `piece`; a hole is joined only when none of its
  nodes is used.
- About 20 bars were not in the part at all (0 MPa in the body check while the beam model loaded them): the
  600 N load's faces 8 and 15 (an end face and a large inner face, left from the earlier STEP file) pressed
  in ways that partly cancelled, so they acted like a pin, at one node in the middle of the part, with bars
  free to leave the design space within its reach, 96 mm. They ran through empty space and the crop removed
  them. Only a hole or pin (`_cylinder`) or a piece within 1.5 node spacings of its middle (`_compact`) acts
  at one node now; faces 8 and 15 then give "no bar reaches it inside the design space" (`FRAME_VERSION` 5).
  bracket_mid with the holes only (faces 2 + 28, 4 + 26): 47 bars, none idle, lightest 13 N, body safety
  1.55 of 1.43.

On the way, the bracket frame's body check said 7570 MPa (safety 0.047), in elements 3% full at a blind
bolt hole's bottom. Wrong: Static Analysis loaded displacements from the SSD kept for an earlier version of
the bosses, whose key had not changed (the same cause made a "fix", a cap closing the ring past a hole's end,
read 11000 MPa in full elements, with four corners of one element not moving at all). Solved fresh: 306 MPa
in a sliver element (safety 1.16; full elements at most 145 MPa). The frame body's and the bosses' keys now
carry `geometry.GEOMETRY` (rule in CLAUDE.md); the cap was taken out again.

Show on Frame Optimization (shape, stress, safety factor, displacement): the body check's analysis colours
the body, as on Topology Optimization; its displacements are on the SSD after Optimize, so a preview only
recomputes the fields. The 21-minute bracket_mid run was this analysis on the CPU: 2.7M box elements at
0.76 mm for a frame filling 6% of its box; the same check took 12 s on the GPU.

### Thin walls and the automatic spacing (2026-11-01)

The table bracket's frame (bracket_mid, frame_1) stopped with "600 N bearing on 8, 15: no bar reaches it
inside the design space". Faces 8 and 15 touch, so they are one piece of 6 k-means nodes; one landed at
(-62, 0, -20) on the inner face of the 8 mm bottom plate, in the 8.5 mm strip between the hole at x = -49 and
the plate's end, 0.4 mm from the hole. The automatic spacing (diagonal / 15 = 13 mm) put its lattice layers at
z = -28.5 and -15.5, both outside the plate (z -28..-20), so the plate had no lattice node except where the
webs cross it, and each of the 227 candidate bars from that node crossed the hole or the box's hollow (the
best 2.7 mm outside; Min size only sets the 0.5 mm margin). By spacing: 13 mm failed, 10 built, 8 failed (on
the Fixed Faces), 7 and finer built. Now an automatic spacing is tried `FINER` (0.8x) while a support or load
gets no bar (`ground.Unreached`), down to `AUTO_FLOOR` (a third), and the log says which spacing it used; a
typed spacing stays, and the error names the point and the spacing
(`test_automatic_spacing_gets_finer_when_a_load_gets_no_bar`: a 4 mm fin, 7.8 -> 5 mm). Frames whose
automatic spacing worked come out the same. The bracket at 10 mm: 114 bars, 140 g, beam model 0.79 mm (layout,
sizing and beam check 21 s; body check not run). Its faces 8 and 15 are still the earlier STEP file's (above).

### Light bars (2026-10-26)

The user saw a part of the wall bracket's bottom hardly working in the stress view: two struts from the
lower bolts (y 7 and 23) to the pin (y 15) bend up at x = 18.8, and a cross tie and diagonals land on the
bend, all at the 2 mm Min size, 5-30% used. Not a modelling gap: the layout itself gives them 4-25 N
(the struts 200-290 N), with curved or straight bars alike. Leaning towards the pin in the middle, each strut
pushes sideways at its bend (~35 N), which a pin joint must hold with other bars. Light bars: remove (Frame
Optimization, `optimize._drop_light`) takes such bars out one by one after sizing, lightest force first,
keeping a removal only when the frame, sized again by the beam check, passes and weighs less. On the wall
bracket: 6 of its 8 light bars, 27 -> 21 bars, 24.0 -> 21.0 g before the body check; through the node 25.4
-> 24 g of bars and the body's safety factor 1.14 -> 1.22 (of 1.43, both short of it). A brace that halves a
compressed bar's buckling length stays (without it the bar must be thicker: heavier); one hanging between
two bars in tension goes (`test_light_bars_go_where_the_frame_passes_without_them`).

### Move joints (2026-10-26)

"Can't it move the bars to where they take more load?" The layout picks its joints from the lattice (5.6 mm
apart on the wall bracket), so its lower struts bend at a lattice node and need bracing there. Move joints
(Frame Optimization, default on, `engine/frame/joints.py`) is geometry optimization after the layout (He &
Gilbert 2015): sequential LPs in the bars' forces and the free joints' moves together, each joint's balance
and the bars' lengths linearized in the moves, every coordinate within a box that grows on success and halves
on failure; a step stays when the layout's LP over the moved joints costs less and every bar is still inside
(joints whose bars would leave are held for that step). After each round joints that met merge, idle bars go
and joints a bar now runs straight through are taken out. Hand check: a bar bent through a free joint becomes
the symmetric two-bar truss exactly (`test_moved_joints_straighten_a_bent_bar_into_the_hand_solution`).

What did not work on the way:
- Steepest descent with the volume's gradient from the LP's duals (envelope theorem). It matched finite
  differences to 0.3% at most joints, but joints on short bars carrying 900-1700 N had gradients ~100x larger
  with kinks (the finite difference changed with the step), so a step scaled by the largest one left every
  other joint where it was. Projecting the moves into the plane of joints whose bars lie in one plane (out of
  it the forces needed jump) did not fix that. One LP per step in forces and moves together moves joints in
  concert, and planar joints are handled by their linearized balance.
- Judging rounds by the LP's cost, or by one reweighted LP: the smallest size enters by reweighting (a bar
  under what the smallest one carries costs as if it carried that much), which is not monotone in the real
  volume. On bracket_mid the kept state was the unmoved one, with 11 idle bars still in it and 6 N of slack.
  Every state is now judged after 3 reweighted LPs (`_settle`, the lightest counts), slack costs 1000x the
  dearest bar per newton and no state may need more than the layout did plus DROP of the largest force, and
  idle bars are cleared before the layout is handed on (`_carrying`).

Results (layout volume with the smallest size, no joint cost): wall bracket 34 -> 23 bars, 30% lighter, 7
joints moved up to 2.2 mm, 0.8 s; bracket_mid with the holes only 97 -> 74 bars, 11% lighter (10 joints taken
out of straight bars), 0.4 s. Through the node, wall bracket: 27 -> 19 bars, 24.0 -> 20.7 g of bars after
the beam sizing. Its body check first stopped at 1.24 of 1.43: thicker bars at the weak spot dipped (1.24,
1.19) before they passed (1.40, 1.60), and the check stopped after one round without gain. It now stops only
after two in a row (`test_the_body_check_takes_one_round_that_dips_and_keeps_the_safest`). Both layouts then
pass: joints on the lattice 41.6 g after 5 rounds (safety factor 1.49), moved 28.6 g after 3 rounds (1.60),
31% lighter. The thickening grows every bar whose axis is within its size + skin + 2 mm of the spot, so the
5 mm main strut, 3.6 mm off the second spot's surface, grew there too (5 -> 6.5 mm): a tighter choice of
bars may save more.

### Two editor tabs after a restart (2026-10-26)

After restarting the editor and pressing Optimize, the user's wall-bracket frame showed a 140 x 56 x 126 mm
part and failed with "joint cost, node spacing, skin and max sizes must be 0 or more", although the recipe
held the 60 x 30 x 50 box and valid settings. start.bat opens a new tab, and the tabs already open reconnect:
one held the bracket_mid recipe (its body 4 is that part, its frame node is also `frame_1`), another the wall
bracket with unsaved changes (Max sizes -33, an accidental drag on the label). The engine sends every event to
every tab and keeps views by node id, so each tab drew the others' results as its own. Now one tab uses the
engine at a time (CLAUDE.md "Server"); tabs loaded before this change still take every event until they are
reloaded.

### Watching the frame optimize (2026-10-26)

The 3D view shows the frame after every step of an Optimize (`live_frame` events, CLAUDE.md "Frames"). The
wall bracket's run (safety 1.55, 60 s), as the editor received it: the first LP at 2 s, then 11 min-size
rounds, one every ~2 s (60, 96, 34, 82, 39, 83, 46, ... bars: the reweighting cycles, the lightest is kept);
mid-bar joints 28-50 s, 10 rounds; moving joints 0.65 s; beam sizing 6 rounds in 0.9 s; body check 3
rounds 55-60 s. Joints moving and sizing pass too fast to watch, so the editor keeps every step of the run
and the bars panel replays them (200 ms a step). The live view first sent at most every 0.1 s and dropped
19 of 22 joint steps; every 0.02 s sends them all (a frame of ~25-100 bars is a few kB).

### Torn fillet seams (2026-10-25)

The user's frame (Steel S355, 27 bars of 2-7 mm) showed a jagged line along some fillets, as if the
surface were torn. The kernel blended only the two nearest bars, with k = 0.6 x the thinner of them:
where the second and third nearest bars were equally far but of different sizes (five bars of 2 and 3 mm
at the joint near (18, 7.5, 3)), k jumped between 1.2 and 1.8 mm across a line, and so did the field
(159 samples near the surface at 0.2 mm where neighbours differed by more than the voxel, in three
spots). Now the kernel keeps up to `SLOTS` (8) nearest different bars and takes the union of every
pair's blend: the same for two bars, symmetric, so continuous at every swap, and the fillet between two
bars no longer disappears where a third is nearer. No jumps left on the user's frame; GPU equals CPU.
Where three fillets meet, their union leaves a crease (as before with equal sizes): replaced the same day
by an ordered blend (next section). The bodies' keys carry `geometry.BLEND`, so cached values of an older
blend are not used.

### G2 and G3 fillets (2026-10-25)

The user asked for smoother fillets, G2 or G3. The polynomial smooth minimum min(a, b) - k B(h), h = 1 -
|a - b| / k, as a function of d = a - b: G1, B = h^2 / 4 (the old one), is a parabola inside the blend and
joins the bars with a curvature jump (a faint ring in reflections); G2, B = h^3 / 6, is curvature
continuous at the edge, but its third derivative jumps in the middle (d = 0: the term |d|^3); G3, B = h^4
(5 - 2h) / 20, worked out so the middle has no odd terms up to the third and the edge none up to the
third derivative: C3 through and through. All keep the field's slope in 0..1/2 per input (a valid
distance-like union), and only the field's continuity carries over to the surface where its gradient is
not 0. Tested in `test_fillet_profiles_have_the_continuity_they_are_named_for`.

Size: at the same k, G2 and G3 fillets are shallower in the middle (k/6 and 3k/20 against k/4) and twice
as curved there. On the user's 5-bar joint (6 x 6 mm around it, 0.04 mm): G1 added 25.4 mm3 over plain
joints with a 99th-percentile mean curvature of 5.6 /mm; G2 at the same k 13.1 mm3 and 6.1 /mm; G2 at
1.5 k (the same depth as G1) 27.6 mm3 and 5.0 /mm; G3 at 5/3 k 27.3 mm3 and 5.1 /mm; at 2 k 46.5 / 38.1
mm3. So G2 and G3 are made `CONTINUITY_SIZE` larger (1.5, 5/3): about the same material at a joint,
reaching farther along the bars, with a gentler peak. Frame Optimization and Join Frames take Fillet
continuity (G2 by default; bosses joined with `fields.smooth_union(order=)` alike); a state sized as G1
is sized again on Optimize.

That alone was not it: the user saw at once that "all of those fillets aren't G2 or G3". A G2 blend of two
bars does not make the body G2 when (1) the body is the union (min) of every pair's blend: where two
fillets of a joint cross, a crease (not even G1); (2) a curved bar is a polyline of capsules, creased at
every node (4-13 degrees between pieces on the user's frame); (3) a bar's half-ball end meets it with a
curvature jump, inside the fillets at joints. Now: (1) the kernel blends the nearby bars in a fixed order,
thickest first, each into those before it with its own k: a composition of C2 (C3) blends of C2 (C3)
fields is C2 (C3) whatever the number of bars; (2) each continuous bar is one quintic spline through its
nodes (C4; second and third derivatives 0 at its ends, where its curve goes on straight, so that join is
C3), the distance taken by Newton steps on the quintic piece nearest; (3) ends rounded as
sqrt(rho^2 + ramp(s)^2) with `ramp` a C2 / C3 smoothed max(s, 0), from the bar's nearest curve point.

What went wrong on the way, measured with a curvature test (mean curvature from the field at 0.05 and
0.025 mm; on a G2 surface its change between neighbours halves with the spacing, where it jumps it does
not, at a crease it grows): rounding each piece's end before taking the bar's nearest piece let a piece
farther from the end (less rounded) win, a crease (now: nearest piece by its curve point, then rounding);
the curve's straight run past an end was infinite, and on a curved bar it passed by the bar itself (now as
long as the rounding and two fillets reach, and past it measured from the end's line); and the ordered
blend carried far bars into the values later bars blend with: nine bars within 4 mm at one point, eight
kept, the ninth dropping out made a crease. Now a bar's fillet fades out (C3) between one and two of its
fillet sizes away (`fade`), the kernel keeps 12, and each bar's box and the outside value cover two fillet
sizes. The fade makes the field up to 1.4x steeper than a distance 2.4-2.9 mm off the surface (no jump:
the same at half the spacing), never on it.

On six spots of the user's frame (2.5 mm around the busiest joints and curved bars, 0.025 mm): the old
kernel 94-2272 non-settling samples per spot (peaks of 12-48 /mm: creases), now G1 0-198 (its curvature
jumps at fillet edges and half-ball ends, as named), G2 and G3 none. The float32 field cannot resolve G3
against G2 numerically; G3 rests on its parts, each tested: the spline's pieces agree to the fourth
derivative, `bulge`, `ramp`^2 and `fade` have their continuity, and the kernel equals those formulas for a
bar's end and for a 3-bar fold. Zebra stripes: torn and stepped before, kinked at fillet edges at G1,
smooth at G2 and G3. Cost: the whole frame at 0.15 mm (30M points) 3.4 s on the CPU and 0.18 s on the
GPU, from 0.56 / 0.03 s (bigger boxes, 12 slots, Newton on quintics). Still not smooth: the design
space's crop (holes, mounting faces: sharp by design) and rectangle profiles (mitred pieces, sharp
corners).

The bosses at supports and loads ("does it do that towards the support/loads as well?": no): the skin was
the distance to the faces' triangles, flat over each and round over each edge (stepped zebra stripes; on
two of the user's support bosses 10-13% of the surface samples did not settle), and capped 1 mm out
while the fillet into it reaches 2.7 mm (it creased where the cap began). Every support and load piece
of the user's bracket is an exact hole (fit residual 0, 354 degrees covered: OpenCascade puts mesh nodes
on the surface), so `_cylinder` fits it (normals square to one axis and turning round it, corners on a
circle; a hole, normals towards the axis, becomes a ring all round) and `skin_body` measures it exactly;
other pieces keep their triangles; the cap is the fillet's reach + 1 mm. Then two G1 places were left in
the cut body: the ring's ends where the design space goes on past a hole (now rounded by `ramp`, as the
bars' ends), and the ring's axis past its ends, where every point is as far from the whole rim (a ridge
the fillet of a bar crossing it took on; now the distance from the axis is rounded within half the radius
by an even C3 polynomial: only the hole's inside and the axis past its ends change, the boss's inner face
by ~0.01 mm). Left on the two bosses, away from the design space's faces: 70 and 5 flagged samples, with
no curvature jump on lines through them (second-derivative changes <= 0.06 at 0.01 and 0.02 mm steps, a G1
jump stays ~0.2-1): detector noise.

### Faster layout LPs, and the calculation shown (2026-11-08)

Where a fine Node spacing's time goes: at 2 mm the block has 3,666 nodes and 208,917 candidate bars; the ground takes
1.6 s, the LPs everything else. Each was HiGHS's interior point with crossover through `linprog`, built anew: 15 s for
member adding's first 40k bars, 22-26 s for each of the next ones, 5 rounds a layout and 3 layouts (the plain one, two
Min size rounds), 250 s in all. Measured on those LPs (11k rows, scipy's own HiGHS 1.12 bindings,
`scipy.optimize._highspy._core._Highs`):

| Solve | Time |
|---|---|
| interior point, no crossover, 40k bars / 82k bars | 9.2 s (47 iterations) / 9.1 s (34) |
| ... with crossover (as `linprog`) | 15 s / 22 s |
| ... optimality tolerance 1e-6 or 1e-5, or presolve off | 8.2-8.7 s (28-33 iterations) |
| every one of the 209k candidates at once, no crossover | 23 s |
| dual simplex from scratch, PDLP | over 120 s, not done |
| primal simplex from the last basis after adding columns | over 5 minutes, not done (degenerate) |

The interior point's time hardly grows with the columns but fast with the rows and with how far bars reach: the 2 mm
mid-bar ground (5,070 nodes, its new nodes' bars reaching twice as far) took 45-70 s an LP. The interior duals also
name far fewer bars to add (1,150 in round 2 where the vertex's named 7,380; 153, 22 and 9 after), so member adding
reaches the optimum in 4 LPs, the 4th lowering it by 4e-6 (`STALL_ROUNDS` 1 now). A layout's LPs are now one model kept
in HiGHS (`layout._Model`: member adding adds columns, the Min size rounds change costs), each solved without crossover;
Stop interrupts it from HiGHS's IPM callback, which runs on the node's thread (no process of its own, which cost 2 s to
start), and the same callback puts the iteration on the card. 2 mm: 250 s to 77 s (96 bars, 17.7 g, against 94 and
17.3: a degenerate LP's interior and vertex optima are different layouts, and the Min size rounds go on from them). The
automatic spacing's frame is the same bar for bar (44 bars, 305 pieces, 16.7 g before the body check). With mid-bar
joints 2 mm took 11 minutes instead of 18 (69 bars, 17.3 g; 15.1 before), 3 mm 77 s (110k candidates in the mid-bar
layout, 13.2 g, lighter than either at 2 mm), so mid-bar joints are left out over `optimize.MID_BAR_LP` = 150k
candidates (was 80k). The wall bracket at 3 mm (138k candidates, 140k with mid-bar joints): 116 s in the editor.

Tried and dropped: a Min size round solved once, its bars to add left for the next round (60 s instead of 77, but
19.2 g instead of 17.7: those 124 bars moved the next round elsewhere); the mid-bar layout starting from all the bars
the first layout's LP had (the same frame, 656 s instead of 686, from bigger LPs). FRAME_VERSION was not raised:
frames under 80k candidates are the same, and those laid out with member adding before are as valid.

Shown: every member adding round and Min size round is a live frame with its numbers (LP count, bars in the LP of the
candidates, bars to add next, volume at the bars' forces and with every bar at least Min size), every step has its
bars' mass as drawn, and the bars panel shows them (`CalcView`): the stages done, now and left out, what the stage
does, the LP's numbers and the iteration being solved, and the mass over the steps on a log scale (the first round of
short bars can weigh ten times the rest), where a click replays from that step.

### Edge cases of Frame Optimization (2026-11-08)

About 90 odd settings and parts through the real node (block 60 x 20 x 30 mm held at x = 0, 200 N -Z at x = 60,
PLA, safety factor 2; a process per case with a memory limit of 6 GB, the body check's Static Analysis on the GPU),
then the slow ones with their LPs timed. Frames that were fine stay the same bar for bar (baseline: 44 bars, 18.2 g,
body 2.1; every profile, joints, curving 0 and 1, mid-bar off, move off, light bars, torque, a hole held).

**A typed Node spacing far too small asked for 341 GB.** `ground._build` sampled the design space at a third of the
spacing before it counted the lattice: 0.01 mm failed with "Unable to allocate 341 GiB", and spacings between could
take gigabytes first. The lattice is counted first now (the error says what would do: "use 0.54 mm or more"), and
`_Inside` takes at most `MAX_INSIDE` samples (a 300 mm cube at 5 mm with 0.5 mm bars asked for 113M: 1.2 GB peak now).

**Bars thicker than the design space.** The beam model sizes bars without asking whether they fit, and the body is
cut by the design space: safety factor 100 made 113 mm bars in the 20 mm block ("4.85 kg of bars", the body the
block), Size step 50 or sizes "100" every bar 50 or 100 mm, 1e9 N bars of 1.5 km, and with no skin the body check
thickened bars at a support to 25.5 mm. `optimize._room_size` is the largest size whose section's thinner side fits
the thickest spot (`_Inside.thickest`: measured again round the deepest sample on grids of a tenth and a hundredth of
the step; bars are turned, so a 2 x 20 mm rectangle stands in the block's 30 mm). A Size step or list that cannot fit
is refused; bars that would need more fail as `MaxSizeShort` with `room`, and `optimize_frame` lays the frame out
again with that as Max size (force limits in the LP), or says the bars would have to be thicker than the design
space; the body check stops there too. Before all that, the first LP's cost bounds the material any truss of these
bars needs (`Layout.least`, times the shortest bar's L / (L + joint length)): over twice the design space's volume,
the frame is refused at once. 1e5 N had been laid out again at the room three times for 15 minutes; now 4 s: "the
loads need at least 448 cm3 ... the design space holds 36.2 cm3".

**The LPs' tolerances are absolute.** At 1e-9 N every force came back as noise, all bars were dropped and
`move_joints` crashed on an empty array. Loads outside `LP_LOADS` (0.01 to 1e5 N on a node) are scaled for every LP,
and Min size's area with them (same bars and volume; scaling the allowed stress instead left costs of 1e-10, and
member adding asked for 97,000 bars). 1e-9 N: a frame of 2 mm bars in 17 s.

**Member adding never stopped.** At 2 mm spacing (209k candidates) the block's volume was 2616.53 from round 3 on
while the duals of a degenerate optimum named ~1000 more bars a round, 30 s an LP, for up to 60 rounds, in each of the
12 min-size rounds, each starting from the shortest bars again: killed after 15 minutes in its first layout. Member
adding stops after `STALL_ROUNDS` rounds lowering the volume less than `ADD_STALL` (0.01 %), each round goes on from
the bars of the one before, and with member adding the min-size rounds stop after `PATIENCE` without a lighter layout
(smaller grounds are untouched): 2 mm took 18 minutes on a loaded machine, 48 bars, 15.1 g (16.7 g at the automatic
4.7 mm). Its mid-bar layout was 13 of them, and Max size = Min size (2 mm) spread the load over so many bars that the
nodes along them made 165k candidates, over an hour for its three shares. Mid-bar joints are left out, with a note,
when they would make more than `layout.FULL_LP` candidates (the block and brackets: ~22k): 2 mm then takes 4 minutes
(94 bars, 17.3 g), and Max size = Min size says in 29 s that 2 mm bars cannot carry the load (2.25 of their strength).

**A load on its own support** made a 1.3 g frame of short bars beside it: the support and the load were clustered from
different random states, so their nodes never merged. A piece chosen again is clustered as the first time
(`_face_nodes`' memo): all of a load on held faces is refused, two loads that cancel too (they had made a 4.2 g frame
carrying nothing, which Static Analysis then would not check), half a load on a support says so in a note, and a
0 N load says it is 0.

**Parts thin or small for their bars.** A 5 mm cube with 2 mm bars reached none of its faces: bars from a face node
could leave the design space within half a spacing (0.29 mm at the automatic one) where a bar starting on a face needs
1 mm to get its centre 1 mm in, and finer spacings made it worse. Now at least `FREE_SIZE` x Min size (0.75; the same
for every spacing over 1.5 x Min size). A 4 mm bar sampled every 2.5 mm interpolated its middle 0.96 mm deep, not 2:
a part under 4 steps thick is sampled again at a quarter of its thickness. With no sample inside at all, `_Inside`'s
seam correction had replaced every value (0.75 mm everywhere, a 1 mm plate measured 0). Min size over the thickness:
"no bar that thick fits in the design space, which is at most 0.99 mm thick"; nearly as thick (15 mm in 20, 1 mm in 1),
the LP's "no frame" says so (`LY.NoFrame`, `_why_none`) instead of "a load not connected to a support?".

**Nodes on a line.** A 300 x 4 x 4 mm bar's automatic 20 mm lattice is one row down its middle, and the LP found no
frame. `ground._across` raises `Flat` when every node lies on a line or in a plane (within a tenth of the spacing: the
face nodes sat 1 mm off the row) and a load acts across it; the automatic spacing gets finer as for an unreached face,
and the error suggests a spacing a bar apart across, or says the part is too thin (under 3 x Min size) or too slender
(`SLENDER`, 40 x as long as thick) for a frame. Taking the suggested spacing at once was tried and dropped: the 4 mm
bar at 1 mm made one LP of 4 minutes and 3.9 GB for 963 bars filling it (bars from the end faces to every node of a
300 mm row, nearly dense for HiGHS), at 1.5 mm with 1 mm bars 6 GB.

**Cut off.** Halves joined by a 1 mm neck, or two bodies: `_joined` names the load no chain of candidate bars joins to
a support (its nodes red in Show: candidate nodes). A Connect to body 500 mm away counted as within a spacing (the
design space's distance clamped at its grid's edge, now growing beyond it), so the note that nothing is near never
came. A frame held only by Connect to: the body check said "add Fixed Faces"; now that Static Analysis does not see
what it connects to.

Also: Size step 0.01 grew failing bars 0.01 mm a round and gave up after 30 (`optimize._grown`: at least 2 % a round,
which is one 0.5 mm step up to 25 mm bars, so frames stay the same); a spacing larger than the part with Joint cost 0
crashed (an empty lattice); the body check's "the load itself presses" is now for pins and holes only, and on flat faces
says a thicker Skin helps (held at both ends and loaded on top: 1.2 with a 2 mm skin, 1.32 with 4); Optimize on a
finished frame did nothing anyone could see: the card says Finished, as Topology Optimization's does.

Left as they are: the 60 m block (37.5 MPa where 25 is allowed after 30 rounds: stiffer bars draw more moment at
clamped joints), a 5 mm cube with 200 N (body 0.94, warned), and fine spacings, which are still minutes of LPs.

## Stop (2026-10-25)

"The stop button doesnt always work": a run stops at node boundaries, `report_progress` and
`check_cancel()`, and `/api/cancel` set the flag and then waited for the run to end (`Session.cancel`
joins its thread), so during a long step without a check the request hung and nothing on screen changed.
Steps without one: the frame's LP solves (HiGHS interior point, which Python cannot interrupt; over 30
minutes at the bracket's 2.68 mm spacing), its resizing rounds, body checks and spread layouts; the
topology node's fine `prepare` and its result building (smoothing, thickening, check analyses and
dead-end trims: ~90 s at fine). Now `/api/cancel` only sets the flag and returns (`Session.request_stop`;
the editor says "Stopping…" from the `stopping` event until run_done, then "Stopped"); frame LPs over
`layout.APART` (30k) columns run in a spawned process, polled every 0.2 s and terminated on Stop; checks
between LP solves and in every round; `density_mesh` / `thicken` and `static_analysis` take `cancel=`,
called between their steps (and each thickening pass); the Optimizer's first, full-design solve gets the
solver callback.

Measured with `Evaluator.cancel()` from a second thread (bolted bracket): the frame at 2.68 mm stopped
0.03-0.2 s after Stop inside its LPs (the LP process gone after), at automatic spacing 0.03-0.5 s in body
checks and 2.7 s in a smaller LP solved in-process; topology (fine, 10 iterations) 0.3-2.4 s anywhere,
11.7-20 s before. In the editor, Stop on a frame Optimize showed "Stopping…" and "Stopped" 95 ms later.
Still in one piece: an LP under `APART` columns (seconds), a multigrid hierarchy build (≤ 1 s), one
marching cubes, Merge Meshes' fuse (~12 s).

Found on the way: topology's fine `prepare` took 33 s, 26 of them one `cKDTree.query` (840k element
centres against 1.56M points on support and load faces, within the skin). scipy's `compact_nodes`
(default) shrinks every tree node's box to its points; on flat faces those boxes are flat, and queries with
`distance_upper_bound` beside such a face searched most of the tree (a 6.6 mm slice of the part: 20 s).
Built with `compact_nodes=False, balanced_tree=False` (`topology._face_tree`, also for the gap and
dead-end tests): 1.5 s, the same distances bit for bit. Fine prepare 33 -> 3.3 s; the 10-iteration fine run
135 -> 94 s, the same result. Other KD-trees on face samples may gain the same (facemask's border edges):
not measured.

## The PC freeze during a test run (2026-10-12, 21:15)

- The user's editor exported the lamp at 21:10 (762 MB STL) and was computing full-resolution
  petals again at 21:13. At 21:15 a full test run started; it reached test_gpu.py. The PC froze
  and had to be reset (Windows logged only Kernel-Power 41; its "21:02" is a stale heartbeat).
- Why a freeze and not an error: the page file is on F:, a mechanical Hitachi disk. Once RAM is
  full Windows pages to it and the machine stops responding long before memory is exhausted.
  GPU memory that overflows on Windows (WDDM) also spills into RAM invisibly to every process
  measurement.
- Fixes: GPU tests only with `--gpu`; watchdog thresholds raised (free caches below 3 GB free,
  stop the run below 1.5 GB, last resort below 0.75 GB) and checked twice a second; parallel
  slots keep 3 GB free; an idle editor releases its workers, RAM cache and cached meshes when the
  PC runs low (another program needs the memory). Test suite without GPU: 1.65 GB peak.
- 2026-10-24, idle editor with lamp_noise2 (petals and the textured center shown at Final): 6.4 GB.
  Main process 2.4 GB (the center's 4.8M-triangle view from 103M voxels, 144 petal views, the
  0.56 GB RAM cache, ~1 GB program and CUDA), 4 GPU workers ~0.65 GB each, and 4 workers that never
  got an item (free GPU memory allowed 4 at a time) at 0.29 GB each. Workers now start when first
  given work. Releasing idle workers after a pause would give back the rest, but a lamp Draft edit
  with cold workers takes 8-10 s instead of 0.5-0.7 s.
- Recommended to the user (system settings, theirs to change): page file on the NVMe SSD (C:),
  and NVIDIA "CUDA - Sysmem Fallback Policy: Prefer No Sysmem Fallback" for python.exe so a full
  graphics card gives an error (the engine then uses the CPU) instead of silently eating RAM.
- **2026-10-24: the page file is now only on C: (Samsung 970 EVO NVMe, system managed, 47 GB
  allocated); nothing on the hard disks.** Checked with a probe (20 ms timer at high priority, a 128 MB
  buffer touched every 2 s, `cmd /c exit` started every 3 s at normal priority; paging counters by PDH
  English names, Windows here is Swedish) and a kill switch:
  - Lamp, cold cache, 8 workers, GPU: textured center at 0.1 mm (43.8M triangles, 66 s, tree peak 9.4
    GB), then 144 petals at 0.1 mm + fuse (8.5M triangles, 64 s, peak 9.4 GB; Merge Meshes stopped the
    idle workers for its 5.1 GB). Free RAM never under 6 GB, nothing paged out, timer at most 1.4 ms
    late: the engine's own limits keep it out of the page file.
  - A test program taking all free RAM plus 8 GB of random data (31.5 GB on top of ~17 GB of other
    programs), then reading it all back twice: 13 min of thrashing, 58 GB written to the page file
    (up to 1.2 GB/s), page-ins up to 830 MB/s, the page file at 21 GB. The PC never stalled: timer at
    most 55 ms late, the 128 MB buffer at most 135 ms, a process start 33 ms typical and 0.54 s worst.
    The read-backs took 221 s and 517 s (31.5 GB each): running out of RAM is now very slow, not a
    freeze. Memory of ones instead of random data is not a test: Windows compresses it in RAM (~19 GB
    left the program while the page file grew 2 GB).
  - So the watchdog's free-RAM thresholds went from 3 / 1.5 / 0.75 GB to 1.5 / 0.5 / 0.25 GB (last resort
    after 30 s), and the memory limit is set at the start of each run instead of once at start-up. A
    cold lamp run with another program holding all but 5 GB was stopped by the limit (75% of the free
    5 GB) with 2-3 GB still free, before those thresholds were reached (docs/MEMORY_AUDIT.md, "Done").
- **2026-10-31: big jobs against the guards, page file on the NVMe.** A gyroid (5 mm cells, 30% solid, 1 mm
  skin) in the 60 x 30 x 50 mm wall bracket design space, run headless with 8 workers on the GPU, the guards
  changed only in that process, a monitor with a kill switch (as 2026-10-24, plus commit and per-process
  private bytes). The PC had ~10 GB free and ~53 GB committed by other programs (Plex 4.3 GB, the idle
  editor 3.4 GB, WSL, Firefox, SolidWorks), commit limit 77.8 GB (system-managed page file, 47 GB):
  - The editor's Final view (512 cells, 21.4M triangles before the display cut to 15M): 18-19 s, tree peak
    6.3-7.3 GB, nothing paged, same time with the guards relaxed. An STL at 0.1 mm (1.55 GB, ~31M triangles):
    31 s, peak 7.1 GB, nothing paged. These sizes are not a memory problem when ~10 GB is free.
  - At 0.05 mm (955M voxels, 1176 blocks with surface): today's guards stopped it at 7.4 GB (their limit,
    75% of the free RAM) with 4 GB still free. Relaxed (limit 38 GB, stop under 0.05 GB free) it meshed every
    block and the main process reached 14.8 GB; Windows paged other programs out at up to 583 MB/s (page file
    4.6 -> 6.6 GB), free RAM touched 0 for ~5 s in all, the PC stayed responsive (20 ms timer late > 50 ms in 2
    of 215 samples, at most 163 ms; process starts <= 48 ms). It stopped itself at 0 GB free. No warnings or
    crashes in the System or Application logs.
  - **Commit, not RAM, is the limit now**: 28 s in, commit reached 77.7 of 77.8 GB while the main process was
    only at 4 GB: each GPU worker commits ~2.5 GB of private memory but holds ~0.7 GB, so 7 of them commit
    17.5 GB for ~5 GB of RAM. The system-managed page file did not grow during the run. Out of commit, an
    allocation fails in whichever program asks (SolidWorks, Firefox), not just the engine: that is the
    remaining danger, and what the guards should watch instead of free RAM (they never measured commit).
  - **Where a worker's commit went** (private bytes step by step in a fresh process): numpy +0.73 GB and
    `scipy.linalg` +0.74 GB, both while resident memory grew by ~20 MB: each loads its own OpenBLAS, which
    commits ~23 MB of buffers per thread when it loads, one thread per logical core (32 here). The engine's
    other imports +0.14 GB; CuPy starting CUDA +0.26 GB, the CUDA context +0.31 GB, Warp +0.07 GB. With
    `OPENBLAS_NUM_THREADS` = 1 / 4 / 8 the imports commit 0.17 / 0.36 / 0.61 GB instead of 1.62 GB.
    `StickyPool` now starts each worker with its share of the cores (as libigl's threads), so a GPU worker
    commits 1.1-1.3 GB at its peak instead of 2.5 GB. The main process keeps all 32: the CPU FEA's
    `U @ KE` products use them.
  - **Guards on commit since then** (`cache.commit_room`, `COMMIT_TIGHT` / `COMMIT_STOP` /
    `COMMIT_LAST_RESORT` 6 / 3 / 1.5 GB): the limit is min(60% of RAM, held + commit room - 3 GB); low free
    RAM only frees caches and stops idle workers, a run stops when free RAM stays under 0.25 GB for 30 s
    (Windows cannot page out as fast as it is asked); the RAM caches still only take free RAM. Same jobs
    again: 0.1 mm 33 s, tree peak 8.4 GB, commit peak 66.3 GB. 0.05 mm **finished**: 126 s, a 6.2 GB STL
    (133M triangles), main process 13.0 GB resident / 16.8 GB private, limit 19.1 GB. Free RAM under 0.3 GB for
    at most 2.5 s, 1.4 GB paged out and 0.8 GB back in (peaks ~700 MB/s), commit at most 69.7 of 77.8 GB, the
    20 ms timer late > 50 ms in 3 of 249 samples (at most 158 ms). The watchdog stopped the idle workers after
    the blocks (free RAM under 1.5 GB) and freed caches; nothing else.

## Face masks (2026-10-11)

- All 150 petals in the lamp STEP have the same 150 faces in the same order (checked by face
  area), so one list of face numbers works for every petal.
- Measuring "distance to the other faces" fails on thin walls: the inside of a 1 mm petal wall is
  1 mm from the outside face. The mask instead measures from each point's closest surface point
  to the selection's border edges, and takes inside/outside from the closest triangle's face.
- Values: fade 2 mm, top face of a 20 mm box: *inside* 0.16 / 0.5 / 1.0 at 0.5 / 1 / 2 mm from
  the edge (sides 0); *center* crosses the edge symmetrically; *outside* spills 2 mm down the sides.
  Blocks compute it bit-identically; the GPU agrees with the CPU except right at face borders.
- Tangent groups (Shift+click) join faces whose median bend along the shared edge is under 30°
  (triangles near a tangent edge can lean by half the tessellation angle on each side). On the
  petal that joins 136 of 150 faces, so Ctrl+drag painting was added for adjoining faces.
- Picking on 144 petals (853k triangles) needs three-mesh-bvh for raycasting. The browser
  automation cannot hold Ctrl during a drag; painting was tested with synthetic pointer events.
- **Bug found by the user (2026-10-12)**: `computeBoundsTree()` re-sorts the geometry's index
  buffer in place (and the store's `indices` array with it), so a raycast's triangle number no
  longer matched the per-triangle CAD face list: clicking a straight side of the board
  (`tavlaunderlägg.stp`) highlighted the side plus scattered patches of the top. Fixed with
  `computeBoundsTree({ indirect: true })` (keeps the index order; 0.7.8 supports it but its types
  lack it). Checked on the board: the top selects exactly face 15, a side exactly its own face.
  That board's sides are two faces each (a flat lower part and a curved band where the wavy top
  turns down); its top is one big face with small flat faces where the peaks are cut flat.

## Face patterns: vent holes (2026-10-18)

- OpenCascade's binary triangulation dump holds the nodes' surface parameters as 2 float64 per
  node right after the nodes (flag byte `has_uv`); they equal `Poly_Triangulation.UVNode` bit for
  bit. Every BRepMesh triangulation has them. Stored per triangle corner as float32: evaluated back
  through `BRep_Tool.Surface(face).Value(u, v)` they land within 1e-4 mm of the corners (box,
  cylinder, B-spline, located body, hollow box with reversed faces).
- A cylinder face's seam vertices are merged by `process=True` but carry u = 0 on one side and
  u = 2π on the other: per-corner storage keeps both. Holes laid out on the seam are found twice
  (both ends) and the second is dropped.
- A hole's axis from its triangle's normal was ~0.9° off radial on a tessellated tube (R 15,
  tolerance 0.01); area-weighted vertex normals still 0.35° (lopsided quad splits). The face's own
  angle-weighted vertex normals, interpolated, give the exact radial direction.
- First design measured U/V per voxel from the closest point on the chosen faces (the plan). That
  cuts a hole in half at the border between two chosen faces and costs a mesh query per voxel;
  one straight prism per hole, computed only on the samples around it, has neither problem.
  Measured: placement 0.05-0.4 s (test part, 73-211 holes, with ray casts; Warp's first import
  ~1-2 s more), cutter 0.54 s for 211 holes on a 625 x 426 x 175 grid (0.1 mm), bit-identical in
  blocks. `recipes/vent_holes.json` (76 slots + 64 hex holes on a round case) at 0.25 mm: 2.7 s
  end to end, watertight after reload, genus 140 = the hole count.
- Emboss / deboss with the same node: Depth > 0 gives pockets (24 round 4 mm x 1 mm pockets in a
  plate: volume within 0.4%, section diameters within 0.08 mm at 0.2 mm voxels); Height > 0 gives
  raised shapes to unite with the part. On a curved wall a flat-bottomed boss leaves a gap under its
  rim unless it reaches as deep as the wall falls away under the outline (R 15, r 1.8: 0.108 mm),
  so it is sunk by that drop + 0.3 mm, measured by rays (chords make it up to 0.01 mm more).
  `docs/emboss_deboss.png`: 155 raised hexagons on the round case's wall.
- Size scale driven by a Face Mask: first read with `paint.field_at_points` (a 64-cell grid over the
  holes, interpolated). Fine for volume masks (≤ 0.018 off), but a surface mask lives within its
  band (2 mm) and the grid's cells are 5 mm on a 320 mm part: 0.45 off (x2: 0.05). Now each centre
  is read on its own one-sample grid: 0.1-0.8 ms per hole, exact.
- Surface Pattern: the relief uses the cutter's per-shape loop with a smooth step inside each
  outline. Round 4 mm studs, Softness 0.5, amplitude 0.8 on a plate at 0.15 mm: tops at 3.8 mm
  (±0.02), 3.5 mm across half way up as designed (±0.1), nothing raised between. Invert (Face Mask
  × (1 - relief)) gives one raised web with a cell per shape. `recipes/hex_studs.json` (204 hex
  studs on the round case) at 0.15 mm: 7.7 s headless (56M-voxel box, in blocks), watertight.
- Voronoi cells: 147 random cells on the round case placed in 0.9 s (Qhull neighbours, Python
  half-plane clipping per cell). A random seed can fall outside its own cell once the cell is
  moved in by half the web; centring each cell on its centroid fixed a web that read 0.84 mm.
  Cells are drawn in their tangent planes: on R 15 with 5 mm cells the web is 0.96-1.0 mm.
- A relief that is constant along the axis for Band (2 mm) on a 2.5 mm wall: 0.5 mm dimples put
  the field exactly at 0 at the band's end inside the wall (zero-volume slivers in the STL, 5
  "bodies"); deeper ones would dimple the inside. Stopping at half the wall (one ray per shape)
  fixed it.
- The ten pattern presets on the round case: 9-18 s each through the evaluator (4 workers, 0.15 mm,
  knurl 0.1 mm), all one watertight body.
- More standard patterns (2026-10-19): a chevron's box centre lies in its notch, outside it, and
  outlines are traced from the centre (star-shaped): centred on the middle of its apex instead.
  Polygons rounded by growing a smaller one by the radius are exact outside but short inside near
  valleys (a 6-point star's centre: 1.43 instead of 1.58): harmless, like the cross. Tilted holes
  first measured depth from the tilted cross-section, which put half the samples in the air above
  a flat plate ("wall varies" on every hole); measuring from where each axis line meets the face
  gives 4 / cos 30° + the outline's slant, and the holes come out leaning by exactly 3 tan 30° over
  3 mm. Rounded hole edges (exact fillet blend, radius 0.6): 4.16 mm across 0.3 mm below both faces
  of a 4 mm hole (chamfer: 4.6), 4 mm in the middle, within 0.06 mm at 0.05 mm voxels. Alternating triangles tile with one web: 2.196 mm between
  every pair of neighbours (spacing 6, size 4).
- Morphs as a blend of two shapes' distances: round to square checked exactly (outline at 45° at
  2 / ((1 - t) + t / sqrt 2), flats fixed at 2 mm); round into staggered Voronoi cells at t = 0.5
  is 4 mm across between 3 and 5. The blend of a circle and a stadium is pointed half way (lens
  shapes), a property of distance blending. Four morphs on the round case: 11 s each, watertight.
- Join Faces (2026-10-19): the key box (`nyckellada.STEP`, hex studs on the inside) has a plane
  floor and twelve B-spline faces rising 69° to the rim. Per face the pattern had a band without
  studs at every border (79 dropped as too close to a neighbour's, 116 near the border, 0.74 mm
  narrowest gap) and the B-spline faces' own parameters made the cells 0.82x-1.47x across one face.
  Unfolded as one sheet (LSCM start, then as-rigid-as-possible steps): 0.09 s for 6400 triangles,
  95% of the area within 2.3% stretch, at most 7.6%; 897 studs, none dropped at borders, narrowest
  gap 1.38 mm (5.5 spacing, 4 mm studs). A developable strip (plane, fillet, plane) unfolds exactly:
  rows 5 mm apart along the surface within 0.0025 mm, and the largest face's holes stay put within
  1e-7 mm (the sheet is laid onto its CAD frame). Box corners must be cut open at folds (bend > 30°)
  or the sheet can't flatten; a rounded box's top, fillets and sides then unfold within 1%. The
  halves of a cylinder form a tube (two border loops turning ~0 instead of +-2 pi): left per face.
- "The shape names don't match: voronoi gives hexagons" (2026-10-19) was no mapping bug: with
  Randomness 0 the cells of the staggered layout are the honeycomb's hexagons, the same as Shape
  hexagon (893 vs 897 studs on the key box); every Voronoi demo and preset had Randomness 0.7-0.9.
  The card and the log now say when cells are regular. Two more things hide the shapes: the body
  preview samples 128 / 256 / 512 cells along the part's longest side (Draft / Medium / Final), 1.7 /
  0.86 / 0.43 mm on the 217 mm key box, so its 4 mm studs only show their outline in Final (in Draft
  every shape is the same blob); and settings stay when the shape changes (a star, Length 8, at the
  hexagons' Spacing 5.5 overlaps its neighbours into one web).

## Pattern on Faces: copies of a CAD part on a surface (2026-11-03)

- Test vase: an arc profile (radius 24 at the foot, 31 at the belly, 14 at the neck, 92 mm tall)
  revolved about z, one face (surface of revolution, u = angle); a filleted 8 mm hexagonal stud drawn
  standing on y = 0. Spacing 10, staggered: same size 171 copies (9 to 19 a row), same count 170,
  scale 187 (0.60x to 1.13x), Voronoi Randomness 0.8 171 (0.29x to 1.29x); each in 0.1-0.5 s. 89
  Voronoi copies sunk 0.3 mm (Lift), meshed at 0.25 mm and fused with the vase: one watertight
  solid, 1.0M triangles, 13.5 s in all.
- Girth by triangle Jacobians was 6% off on a cone: its triangles run the whole height, and a
  triangle's dP/du is the chord of the edge it was taken along (the bottom or the top), not the
  girth at its middle. Measuring along lines of constant parameter across the triangles is exact
  (size / radius constant within 0.7% on the test cone).
- Girth same size with staggered rows: where a row has one copy fewer than the next, the rows
  cannot nest and some copies sit straight above each other (16 overlapping pairs on the vase with
  rows 8.7 mm apart and studs 9.2 mm deep). Inherent: the log suggests Layout grid (rows a whole
  Spacing apart) or a larger Spacing.
- Where the pattern met the ends of the face (2026-11-03, the user: "it doesn't look good"): rows sat
  where the grid happened to fall, a bare 7 mm band at the vase's foot and the top row tight at the
  rim; same size changed the count almost every row near the neck (staggered rows stopped nesting:
  16 overlapping pairs); same count overlapped at the neck (17 pairs). Now rows are spread evenly
  between the ends (end rows Edge margin in), a row keeps the count of the row before while 0.9-1.25
  Spacings apart and nests by its start angle, and copies closer than Gap shrink (Crowded): 0
  overlaps in all three modes, copies 0.18-0.25 mm inside the vase's end planes.
- Fitting end rows by the distance along the surface was not enough: a flat 6 mm footprint at the
  foot of a 24° cone reaches 0.06 mm nearer the end's circle (it bends towards the footprint by
  (w/2)^2 / 2r x the slope), and where the face widens away from an end (a flaring foot) the copies
  lean out over its plane by height x tan(lean): 0.9 mm on the vase. Both are added to the end rows'
  room. The copy on the seam (u = 0) took its row direction from a triangle Jacobian (0.7° off on
  long cone triangles) because the chord past the seam found no face: wrapped round the period now.
- Voronoi cells of a jittered honeycomb are irregular, so a hexagonal part fills only part of its
  cell: Turn to fit (48 angles) made the copies larger on average, 0.78x to 0.82x on the vase (a
  hexagon turns little before it meets its cell again; a 6 x 4 block gained over 5%). Cells at the edge of the
  pattern have no neighbour beyond it: capped at 0.6-1.0 x Spacing, else edge copies grew to 2x.
- Fit (2026-11-03, the user: "could the number of sides be dynamic, or the hexagon scaled non uniformly?").
  Plate 90 x 100, the filleted stud, Spacing 11, Gap 1.2, Randomness 0.85, the same 87 cells: scale covers
  42% of the plate, stretch (turned, up to 1.7:1, moved) 56%, fill (the outline is the cell) 80%. Cutting the
  cells at the border raised scale from 36% (75 copies) on its own. Fill on the vase: 171 copies, 768k
  triangles (4.5k a copy from 704), 15 s; on flat faces it skips the surface queries.
- Fill's warp keeps the distance from the outline in a band (the edge's width x 1.25): a point d in from the
  footprint's edge goes, along the same direction from the largest inscribed circle's centre, to the point
  d in from the cell's edge, so a 0.6 mm fillet stays 0.6 mm all round; farther in, a radial scale.
- Fill's triangle splitting measured an edge's bent midpoint against the straight line between its bent
  ends at first: sliding along a straight side or across a flat top counted, 26k triangles a copy. Against
  the planes of the edge's bent triangles: 4.8k. Bent midpoints of edges kept from the round before are
  reused (vase 20 -> 11 s for the copies).
- Filled copies on a curved face must be bent onto it (each point to the surface under it, standing out
  along the normal there): larger tiles with flat bottoms stood off the vase at their edges. Moving each copy to its cell's middle tilts its tangent plane, and the cell drawn in the old one
  left gaps of 0.94 mm for a Gap of 1.0: the cells are drawn again round the moved copies (0.99).
- A copy standing on its normal leans over the plane of a neighbouring face where the surface narrows
  towards it (the vase's foot): filled tiles went 0.7 mm below the foot of a 14° cone. Cells there stop
  as far in as the copy is tall times that lean (the place is worked out again with the first heights,
  from the same start; the border points within reach of that, not only the cell's own reach).

## Performance (16-core PC)

| Change | Before | After |
|---|---|---|
| Distance sign: pseudonormal, winding only near cracks | 92 s (test part, 0.1 mm) | 34 s |
| Work in processes instead of threads (GIL) | 16 petals: 34 s (1 thread), 28 s (8 threads) | 22 s (8 processes) |
| Ship raw arrays to workers, not trimesh objects | 7.5 MB per petal | 0.88 MB |
| Build AABB/normals once per mesh (`MeshQuery`) | ~0.1–0.15 s overhead per libigl call | once |
| Sticky worker assignment (warm caches per body) | lamp Draft edit 8–10 s | 0.5–0.7 s |
| Cancel skips queued work instead of killing workers | next edit after a cancel 9.7 s | 1.0 s |
| Coarse pass factor 2 on small grids | cold lamp Draft 16 s (100% exact queries) | 7.4 s (50%) |
| Faster Perlin hashing + coarse noise sampling | 12.4 s per final petal | 3.8 s |
| Lean VTK render (float32, no normal splitting) | 13 s for the full lamp | 4.4 s |
| SSD cache warm (second full export) | mesh step 328 s | 105 s |
| STEP import kept on the SSD (`cache.load_step_meshes`) | board 16.3 s on every start | 0.11 s |
| Triangulation read in bulk (`BinTools_ShapeSet`) instead of per node/triangle | ~3.5 µs per triangle (49 s for 12M) | ~0.2 µs |
| Marching cubes with Warp on the GPU | petal (2.9M voxels) 0.077 s | 0.010 s |
| Decimation threshold scaled to the voxel (pyfqmr) | petal 0.45 s, reached 48% of keep 0.3 | 0.21 s, 30% |

- A libigl AABB query costs ~11 µs per point on one thread; a cold Draft petal is ~0.9 CPU-s of
  real work, so the machine is close to saturated during cold runs.
- **Would a Rust rewrite be faster? (2026-10-23)** py-spy (in the env; without `--native`, which took
  10+ min for an 11 s run on Windows) over six recipes in one process (`Evaluator(workers=1)`, cold
  temp cache): busy time in compiled libraries 7-35% (VTK thumbnail, marching cubes, pyfqmr, HiGHS,
  BLAS; libigl's own query only 1-2%), startup imports 2-35% (gone in the long-running editor), and
  35-86% on our numpy / CuPy lines: unfused array maths (a pass over memory per operation on the CPU,
  a launch per operation on the GPU). That is not Python's interpreter, and Rust would need the same
  fused kernels. The CPU sign test (`sdf.MeshQuery._pseudonormal_sign`) as one Warp CPU kernel: 660
  -> 42 ms on 2M points on one thread, 0 differences (`_Cracks.near` is the same kind of code). The
  topology solver and OC update are done (see "Fused GPU kernels"); the CPU distance-field sign and
  crack test, block mesh post-processing (`snap_float32`, welding: numpy sorts) and the VTK thumbnail
  (~1.2 s, mostly creating the render window) are the next candidates.
- Reading a STEP file: OpenCascade's own parsing (7.3 s for the board) and meshing (5.9 s) are
  most of it; copying the triangles out cost another ~3.5 µs per triangle until it was read in one
  binary dump. Adding the faces themselves to the `BinTools_ShapeSet` registers their edges'
  polygon-on-triangulation, which pulls *neighbouring* faces' triangulations in first (10 of 27
  faces out of order); a bare face built around each triangulation keeps the order. Faces with a
  location (assembly instances) are moved with numpy, checked bit for bit against
  `gp_Pnt.Transformed` on 16 nodes.
- Big arrays in task payloads: a field that reads a 2.4M-triangle mesh pickled to ~90 MB per
  task. `shared.Shared` sends a fingerprint instead (arrays ≥ 1 MB; the lamp's center, 22k
  triangles at angle 0.5, is below that and still goes inline).
- Lamp export path (petals 0.1 mm keep 0.3 + fuse + STL, warm SSD cache, 8 workers, GPU) on
  2026-10-13: 44.6 s (petals 27.3 s, fuse 13.9 s, write 2.2 s); process tree peak 13.5 GB.
- Typical times on the CPU: lamp first Draft ~78 s (≈60 s of that is reading both STEP files,
  now ~1 s from the SSD after the first time),
  Draft edit ~1.5 s, full export ~3 min with a warm SSD cache (~7 min cold). With the GPU see the
  GPU section.
- **Where the cores sit idle (2026-11-01)**, 32 logical cores, the editor stopped; the process tree's CPU
  sampled every 0.25 s (main process vs workers), recipes run headless with 8 workers as the editor does
  (`server.py`: `min(8, cpus // 2)`):
  - Block / per-body meshing uses 5-8 cores, the GPU 25-35%: only 9-10 workers may hold a CUDA context (0.8 GB
    each, card otherwise empty; ~5 in normal use), and per item the CPU work (decimation) outweighs the
    GPU's. 16 workers instead of 8: lamp meshing 18.2 -> 18.1 s, hex studs no faster. The way on is to
    keep the GPU workers on fields while CPU-only processes (or a second thread per worker) decimate.
  - One-core tails after parallel work: hex studs' `tiles.stitch` (seam reduction, pyfqmr) ~5 s of a
    23.7 s mesh step (67% of the main process's busy time); Merge Meshes on the lamp: manifold3d's union
    uses 15-25 cores for ~10 s, then ~4 s on one core (not profiled yet); the last petals ~3 s on 1-2 workers.
    pyfqmr holds the GIL (8 threads: 1.5x), so seams would need processes, not threads.
  - Static Analysis on the CPU (`DESIGN_ENGINE_GPU=0`, fea_bracket at fine, 985k elements): 72 s at 2.2 cores
    on average; 88% in `solver.solve`, 58% the matrix-free product (gather 33%, scatter-add 20%, BLAS 16%).
    A product-like numpy loop gains 3.1x on 8 threads; the GPU's Warp kernels (`fea/gpu.py`) on Warp's CPU
    device in threads (as `segments.py` does) would gain more. This is where bracket_mid's frame body
    checks spent 312 s each when the card was full.
  - Frame Optimization (frame_bracket, 17 s): one core throughout; 49% the layout LPs (scipy `linprog`
    highs-ipm, single-threaded, each reweighted LP needs the last), 46% the body check (model 18%, solve 10%).
  - The editor's main thread for a 12M-triangle view (Node, same V8): `computeVertexNormals` 0.7 s, the
    right-click BVH (three-mesh-bvh, indirect) 3.7 s; 2M triangles 0.13 / 0.58 s. No Web Workers yet.
  - Already parallel: STEP meshing (`BRepMesh_IncrementalMesh` in parallel), manifold3d's union, BLAS, Warp CPU
    kernels in threads (print checks, frame body, Voronoi foam), the worker pool.
- **What a refine waits for (2026-11-02)**, the server driven in-process like the editor (`have`, Draft ->
  Medium -> Final after a changed number), the editor stopped: a run publishes its views only after its
  last node (`Session._run_job` loops over `views` after `evaluator.run`), and a list's view only when every
  item is back. Lamp (`mesh`: 144 petals, `from_implicit_1`: one body in 21 blocks at Final), warm caches:
  Draft 1.2-6 s, Medium 2.4-3.4 s, Final 13-24 s; the petals come back steadily (~0.1 s apart at Final,
  ~0.35 s at a cold Medium, which took 51 s after the first Draft), yet nothing changes on screen until all
  are done, and the petals' view waited 6-8.5 s more for `from_implicit_1`. Hex studs: 0.03 / 1.7 / 0.55 s,
  less than the editor's two `REFINE_DELAY` pauses (0.8 s each). Packing a view is 0.1-0.2 s; Final views
  are 145-150 MB (8.2M triangles) and 86 MB.
  Since then views go out when their node is done and list items as they are made (CLAUDE.md, Server). The lamp in
  the test editor, Final after Medium: the first sharp petal on screen 1.0 s into the run, all 144 by 18.1 s
  (64 store updates, batched every 150 ms), the petals' whole view at 19.0 s while `from_implicit_1` still meshed
  (run done at 24.7 s); before, nothing changed until 24.7 s. A change during the stream makes the petals turn
  into the new Draft one by one (coarser for a moment) until the Draft view replaces them. A part fetched after its
  whole view came answers 404 (one console line, ignored).

## Memory

- **What crashed the PC**: every intermediate array was cached (~20 per petal), and the cache
  budget (60% of *total* RAM) ignored the workers' working memory and other programs. Measured:
  +1 GB every 5 s during meshing, 14 GB after 45 s and still climbing.
- **Now**: per-step freeing in `Context`, cache only expensive fields, RAM + SSD cache tiers,
  limit = min(60% of RAM, 75% of free RAM at start) (~12–13 GB here), workers limited by a
  per-item estimate, watchdog (frees caches at 85%, stops the run at 100%).
- **Measured after the fix**: Draft editing ~5.5 GB and flat over 10 edits; full export peak
  9.6 GB; workers ~0.6 GB each (of which ~0.3 GB is libraries); the main process peaks ~5 GB at
  the end (merged results, export, render).
- trimesh meshes contain reference cycles, so they are freed only by Python's cycle collector →
  a 5.5 → 8 GB sawtooth. `gc.collect()` runs when a worker is idle or every 1.5 s, and in the
  server after each run. Collecting after every task cost ~0.1 s each (edits 1.5 s → 2.9 s).
- After one full export the SSD cache held ~7.7 GB (463 files).
- GPU memory is not useful as storage (transfers cost more than recomputing); the GPU helps by
  running the field math itself (see the GPU section).
- **Memory audit 2026-10-24** (fix list in docs/MEMORY_AUDIT.md; P1 1-3 done the same day): `mesh.volume` in
  `field_to_mesh` leaves trimesh's `triangles` + `triangles_cross` on the mesh (96 B/tri, 2.7x the mesh;
  pickled with it to the main process); nothing prunes the evaluator's results, the server's views or the
  workers of a closed recipe; `memory_limit()` is fixed at start-up (a test editor started with 8 GB
  free kept 6.2 GB and meshed blocks 2 at a time with 10.9 GB free); `np.load(mmap_mode='r')` of a cache
  file adds 0 private memory (np.load: a full private copy), the way to read the SSD tier on this PC.
  Fusing peaks inside manifold3d's union at ~640 B per input triangle (1M triangles in); freeing
  Python's copies before `snap_float32` does not lower it.

## GPU

- Stack: CuPy 14.2 (`cupy-cuda12x[ctk]`, CUDA libraries from pip) for field math, NVIDIA Warp
  1.18 for distance fields (BVH mesh queries), NVML (`nvidia-ml-py`) for GPU memory. Driver 591.86
  (CUDA 13.1). Other programs typically hold 5–6 GB of the 12 GB VRAM.
- **Distance field per petal**: Warp 41 ms vs CPU 2,142 ms at 0.1 mm (3.5M points), 3 ms vs
  157 ms at Draft. Results agree with the exact CPU reference within ~0.0005 mm, no sign errors.
- **Pitfalls found**:
  1. Warp's closest-point maths is float32: on long thin triangles (flat CAD faces tessellate
     into 50 mm slivers) distances were off by up to 0.027 mm. Splitting triangles to ≤ 2 mm
     (shape unchanged) → 0.00045 mm; ≤ 4 mm → 0.001 mm.
  2. Warp's `mesh_query_point_sign_normal` uses a single face normal: on ties (a point on the
     axis of a hole) it returned the wrong sign (−8 instead of +8). Our kernel does the
     angle-weighted pseudonormal test (face/edge/corner), like the CPU.
  3. Winding-number sign needs `support_winding_number=True` and is 6× slower (202 vs 33 ms per
     petal), so it runs only for points whose closest triangle touches a crack.
  4. Warp uses its own CUDA stream: CuPy and Warp work must be synchronized explicitly.
  5. On Windows (WDDM driver) CUDA's `memGetInfo` reports a per-process view: it showed 1.2 GB in
     use while the card had 10.7 GB in use. NVML gives the real numbers. WDDM also silently
     spills VRAM to system RAM instead of failing, so overcommitting does not crash, it slows.
- **Costs**: each worker process that touches the GPU keeps a CUDA context: ~0.35 GB more RAM
  and ~0.5–0.8 GB of GPU memory. Draft-size grids are no faster on the GPU (edits 1.6 s vs 1.5 s),
  so grids under 1M voxels stay on the CPU and Draft editing starts no CUDA at all.
- **Per full-resolution petal on the GPU**: fields 0.13 s, marching cubes 0.22 s (CPU),
  decimation 0.59 s (CPU). The rest of the pipeline is now CPU-bound, so several workers still
  help; GPU workers are limited to what fits in free VRAM (5 of 8 with 6.4 GB free).
- **Measured on the lamp** (cold SSD cache):

| | CPU | GPU |
|---|---|---|
| Meshing 144 petals at full resolution | 328 s | 54–63 s |
| Whole export | 412 s | 125–135 s |
| Final preview in the editor | — | 44 s |
| Draft edit | 1.5 s | 1.4 s (runs on the CPU) |

- Display copies for the 3D view: the careful decimation used for files reached only 10.6M of
  a 2.5M-triangle target on the 12.5M lamp and took 24 s. The browser draws 12M+ triangles fine,
  so the view gets full meshes up to 15M triangles (fast decimation above that).

## Refining the view and on-screen detail (2026-10-23)

- Refine (toolbar): a change previews at Draft, then Medium and Final follow while nothing changes;
  views whose content key is unchanged are kept from finer runs, so only edited bodies are meshed
  again. Test part (`wave_offset`, 0.1 mm mesh): an edit showed Draft after 0.4 s and Final after
  7.6 s; an edit during the Final step dropped back to Draft 0.2 s later (blocks are short).
- On-screen detail (zoomed in, camera still): the visible part of a shown body is meshed again in
  blocks near its surface and drawn instead of the view inside its box, both cut per pixel in the
  shader: the user found the edges seamless. Smoothen body of the test part, middle of the top at
  0.036 mm (asked 0.03; the budget of 40 blocks made it coarser): 1.6 s with warm workers, ~7 s
  cold (worker processes and their CUDA contexts starting). The whole part asked at 0.02 mm came
  back at 0.17 mm in 1.6 s (the budget), coarser than the Final view: the editor ignores such ones.
- Whole blocks reach up to 96 voxels past the box; trimming to the box (plus 2 voxels) before
  sending keeps the download to what is drawn.
- Testing in the hidden browser pane: the page reports visible, but no frames, ResizeObserver or
  requestAnimationFrame run there (the canvas stays 300x150 and draws nothing), so the look of a
  3D change can only be judged with the pane shown or in the user's editor.

## Dual contouring (2026-10-24)

`engine/dualcontour.py`, Mesh from Implicit Body's Method (one grid only); plan, method and the full
table in docs/DUAL_CONTOURING_PLAN.md. From the same samples as marching cubes:

- Vent case, CAD only, against its CAD mesh: sharp edges 0.0084 mm off on average at 0.45 mm voxels,
  against marching cubes' 0.050 at 0.15 mm (25x the samples) and 0.104 at 0.45. Holed vent case at
  0.3 mm: hole rims 0.012 against 0.072. Closed and manifold.
- Grid samples alone do not give sharp edges: a cell at a crease has every crossing within a voxel of it,
  so all its central-difference normals blend both faces (plain dual contouring: 0.20 voxel off against
  marching cubes' 0.34). The faces' planes have to come from straight crossings 2-3 cells away, and a
  vertex must be allowed half a voxel outside its cell (an edge that clips a cell without enclosing a
  sample gives that cell no vertex; clamping kept edges 0.3 voxel off).
- A ridge rounded finer than a voxel (a smoothed texture, a small CAD fillet) cannot be told from a
  sharp one by the samples; sharpening it overshoots (0.14 mm on the smoothed wave at 0.3 mm). The
  prototype sharpens only where the field is a distance field (slope 1), so textures get plain dual
  contouring; small CAD fillets still come out sharp. A piece that sees no straight crossing at all
  cannot show it is on a distance field: letting those pass (an empty weighted mean is 0) sharpened 99%
  of a texture's "sharp" vertices, 0.33 mm off at 0.6 mm.
- Speed in numpy: 2-4x marching cubes' extraction, 10-15x on parts with many holes (every crease piece
  builds a neighbourhood of ~60 planes). Textures were 15x until a cheap first look (sparse products
  with vectors) refused their pieces before the neighbourhoods were built.
- In blocks (plan's D2): blocks can join by vertex names instead of positions when each keeps only the
  vertices of its own cells and the quads of its own edges, computed with a halo that holds everything
  they depend on (12 samples); on one grid's own samples that gave that grid's mesh to 2e-12 mm. The
  lamp center at 0.3 mm: 28 s against marching cubes' 10.6, but +3.1 GB against +5.3.
- Check dual contouring by reloading the STL: fold repair put two neighbouring cells' vertices on their
  shared corner, distinct in memory, one vertex in the file (edges of four triangles); now kept 1% of
  a voxel inside. Dense textures finer than the voxel leave folded slivers (0.02% of the lamp center's
  triangles at 0.3 mm, 0.4% for a denser texture at 0.4); marching cubes has none.
- Memory: building all crease neighbourhoods at once took 150 bytes per voxel on the holed vent case
  (the node budgets 48); in chunks of 20k pieces, 74 at 6.6M voxels and 36 at 21M. Surface-heavy grids
  break the 48 bytes per voxel budget for marching cubes too: a gyroid sheet filling a box took 370
  (805 bytes per crossing; dual contouring 1,000).

## Tooling pitfalls met

- PowerShell 5.1 `Get-Content`/`Set-Content` reads BOM-less UTF-8 as Windows-1252: non-ASCII text
  came back garbled (`â—` for `●`, `â†` for `←`). Find damage with `grep "â|Ã|Â"`.
- `np.save(name)` appends `.npy` to names that don't end with it, which broke atomic
  write-to-temp-then-rename; write through a file object.
- `fast_winding_number`-based libigl calls also have per-call setup cost; reuse the BVH classes.
- Browser automation cannot produce a held Ctrl for React Flow's Ctrl+click (it tracks the key
  itself) or native copy/paste events; the editor handles shortcuts in `keydown` with an in-memory
  clipboard fallback so they still work.
- The browser pane reloads itself after a server restart. The user also edits there: before
  forcing a reload, check `__store.getState().dirty` and diff the editor graph against the file.
- `window.__store` (zustand store) is exposed for debugging from the browser console.

## Status and next steps

- Done: core engine, node graph and recipes, node editor with 3D view, field slices, copy/paste,
  undo/redo, noise and 3D wave nodes, petal scaling with seating, memory limits and SSD cache,
  launcher and desktop shortcut, GPU evaluation (distance fields and field math), one-piece
  export (Mesh from CAD + Merge Meshes with fuse), block-wise meshing of big bodies, face masks
  with face picking in the 3D view, help text for every node, lattices (TPMS, struts, Voronoi).
- Done 2026-10-13: STEP import cache, bulk triangulation read, exact TPMS gradients, Voronoi on
  the GPU / Warp CPU threads, shared arrays for workers, GPU marching cubes, scaled decimation.
- Done 2026-10-13 (usability): surface colours by a field, section view, lattice density units,
  groups, drag-to-change numbers, print checks, autosave/recovery, variants, export queue with
  notifications, named face selections and Clear, body:face selections, presets.
- Done 2026-10-14: key cabinet profiled (memory pressure; wide-fade masks sampled coarser),
  Face Mask *Apply to: all other faces*, Reduce Mesh node, tolerance reduction folded into
  Mesh from Implicit Body (per block in the workers + seam pass).
- Done 2026-10-14: FEA phase 1, static stress analysis (Material, Custom Material, Fixed Faces,
  Load, Static Analysis; see "Static analysis" above and TOPOLOGY_PLAN.md).
- Done 2026-10-14 (evening): FEA phase 2, topology optimization (node, Optimize / Stop, live shape, saved
  states; see "Topology optimization" above).
- Done 2026-10-18: surface parameters in the STEP import, 2D shape library, Hole Pattern node
  (vent holes, pockets, raised shapes; fields for size, depth, height, rotation) and Surface
  Pattern node (studs, dimples, raised web; PATTERN_PLAN.md milestones 1, 3, 4, 5), Voronoi
  cells, Randomness, sunflower spiral, ten pattern presets. Next there: Surface Coordinates, shape
  icons and hole markers in the editor, Voronoi cells clipped at the border (now dropped whole).
- Done 2026-10-19: FEA solver speed-ups (static analysis ~2.5x on the GPU and 3.5-4x on the CPU,
  topology 1.9-2.5x; see "FEA solver speed-ups"), including the float32 multigrid that ran in float64.
- Done 2026-10-19: Join Faces on Hole / Surface Pattern: faces joined smoothly share one unfolded
  pattern (`engine/unfold.py`; see "Face patterns" above).
- Done 2026-10-23: fused Warp kernels for the FEA solver and topology's OC update on the GPU
  (`engine/fea/gpu.py`; topology 3.5-5.7x faster, the same designs; see "Fused GPU kernels").
- Candidates: FEA phase 3 (TOPOLOGY_PLAN.md: symmetry, several load cases, overhang limit,
  stress-graded lattices); GPU marching cubes for blocks (the weld identifies border vertices by exact grid
  coordinates, < 1e-6 voxel; Warp's float32 positions would need snapping first); faster noise on
  the GPU (one fused kernel: 50 of ~80 ms per Final block of the key cabinet is the wave noise);
  remove loose specks without fusing (the key cabinet's texture pinches off 4); release idle
  workers' GPU memory when the card runs short; decide on the rim tube (a Face Mask can exclude it).
