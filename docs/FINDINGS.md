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

- Designs parts in SolidWorks, used nTopology, wants a simpler personal tool for nice-looking
  parts (decorative textures for 3D printing). Single user, not for sale, so Python was chosen.
- Writes informally (Swedish, typos); number fields accept `0,35` as well as `0.35`.
- The original nTop graph: `Wave = 0.35 · acos(cos(4 · Body_1 / Body_2))` → Offset Body (distance =
  Wave) → Smoothen Body (grid 0.1 mm, 3 iterations). `acos(cos(x))` is a triangle wave.
- Pattern the user wants (reference picture): contour lines of smooth noise. "The same wave but
  varying in 3D, the noise is the variation"; both ridge spacing and ridge height must vary.

## The lamp (main test case)

**Petals** — `C:\Users\willi\OneDrive - PJW Konsult\PJW\PJW\lamskarmhalldelar.STEP` (75 MB):
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
- Recommended to the user (system settings, theirs to change): page file on the NVMe SSD (C:),
  and NVIDIA "CUDA - Sysmem Fallback Policy: Prefer No Sysmem Fallback" for python.exe so a full
  graphics card gives an error (the engine then uses the CPU) instead of silently eating RAM.

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
- Candidates: GPU marching cubes for blocks (the weld identifies border vertices by exact grid
  coordinates, < 1e-6 voxel; Warp's float32 positions would need snapping first); faster noise on
  the GPU (one fused kernel: 50 of ~80 ms per Final block of the key cabinet is the wave noise);
  remove loose specks without fusing (the key cabinet's texture pinches off 4); release idle
  workers' GPU memory when the card runs short; decide on the rim tube (a Face Mask can exclude it).
