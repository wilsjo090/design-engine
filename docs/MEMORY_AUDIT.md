# Memory audit (2026-10-24)

How the engine, the server and the editor use RAM, GPU memory and the SSD, and what could be done
better. One coordinator (evaluator, caches, workers, server) and four agents (browser and view
transport; CAD meshes, distance fields and GPU; FEA, topology and frames; fields and the mesh
pipeline) read the code. Small measurements ran on the CPU (each under ~300 MB, private cache folder).
The test editor (port 8766) was used by hand under a memory kill switch: open a recipe, Optimize,
change Keep % and Detail, Optimize again, switch the STEP file, switch recipes. **Checked** = read in
the code and/or measured again by the coordinator after the agent reported it.

Not covered: GPU measurements (code read only, except the editor session), the lamp itself (numbers
for it are extrapolated per triangle), Final exports, Chrome's own GPU-process memory.

## The short version

The engine's defences against running out of memory (watchdog, slots per estimate, SSD cache) work:
in the editor session the watchdog stepped in once and the PC never got below 3.5 GB free. What costs
memory is mostly **things kept that nobody needs**, and a few **big steps nothing estimates**:

1. **Hidden copies.** trimesh caches derived arrays on the mesh objects: `mesh.volume` in
   `field_to_mesh` leaves 96 B per triangle on every Draft/Medium mesh (2.7x the mesh, measured), and
   they travel through the worker pipe too. CAD meshes get 72-216 B per triangle from `.triangles`,
   `.volume`, `.is_watertight`, `area_faces`, `face_normals` in FEA and face-picking code.
2. **Nothing is forgotten.** The evaluator keeps the outputs of nodes that left the graph (another
   recipe, a deleted node), the server keeps every view it ever packed, the browser keeps views on undo,
   workers keep a closed recipe's caches and search structures, and the low-memory release skips FEA
   results and search trees. Seen in the editor: a closed recipe's views were still served, and the
   start-up recipe's 4 workers held 2.6 GB for 7 minutes while a recipe without workers ran.
3. **Unestimated peaks in the main process:** Merge Meshes with fuse (~485 B per input triangle: 5-10
   GB for the lamp), topology's `density_mesh` (~500 B per FE element: 0.65-4 GB), `tiles.stitch`
   (~200 B per triangle above the pieces against a flat 1.5 GB estimate), FEA support sampling (up to 4M
   points x 300 B), `/api/check` (300 B per triangle), the Voronoi foam build (6.5 KB per cell, the guard
   allows 2M cells = ~12 GB).
4. **The SSD tier is read into private memory.** Every cache hit and every `Shared` array becomes
   private RAM in each process, which Windows must write to the page file on the hard disk when RAM runs
   short. Memory-mapped reads (measured: +0 MB private for a 64 MB array) are the "streaming from the
   SSD" that pays here: pages Windows can drop for free, one copy for all workers.
5. **The limit is fixed at start-up.** Started with 8 GB free, the test editor kept a 6.2 GB limit after
   other programs freed memory (10.9 GB free) and meshed blocks 2 at a time: Final preview 23.5 s.

## What to do, in order

P1: small changes, large effect, no change to results. ✅ = done (1-3 on 2026-10-24, see "Done" at the end).

| # | Change | Saves | Where |
|---|---|---|---|
| 1 ✅ | `signed_volume` instead of `mesh.volume` in `field_to_mesh`; `np.float32` eps in the off-zero `where` (also `tiles._mesh_block`, `dualcontour.off_zero`) | 96 B/tri kept (and piped), peak 17 -> 9 B/voxel, 498 -> 264 B/tri; meshes bit-identical (measured) | [F1](#f1) |
| 2 ✅ | Prune stale state: evaluator entries of nodes not in the graph, server `views` / `view_faces` / `view_sources` of nodes not shown, browser store on undo/redo | a closed recipe's whole result set | [M2](#m2), [M3](#m3), [V7](#v7) |
| 3 ✅ | Merge Meshes: an estimate (650 B per input triangle to fuse), stop idle workers first, else a readable error | 2.6-3 GB of idle workers, readable error instead of a freeze | [F2](#f2) |
| 4 | `density_mesh`: estimate and check before it (fall back to upsample 1), float32 distance transforms, free earlier | 0.65-4 GB peak made safe | [T1](#t1) |
| 5 | Stop trimesh caching on CAD meshes (`_chosen_faces`, `_part`, `same_part`, `_straightest`) | 72-216 B per CAD triangle (0.5 GB on the 2.4M part) | [C1](#c1) |
| 6 | Workers: stop the pool when another recipe is opened; after ~1 min idle drop their search structures and GPU memory (keep the process) | 0.6-0.7 GB per worker, up to 1.3 GB per worker of stale trees | [M8](#m8), [C3](#c3), [C9](#c9) |
| 7 ✅ | Memory limit per run instead of fixed at start-up; reset shrunk cache budgets at each run | speed (2 -> more blocks at once), safety when others grow | [M5](#m5), [M6](#m6) |
| 8 | Lower the Voronoi foam guard (2M cells -> ~250k) | a possible 12 GB freeze | [C10](#c10) |
| 9 | Solver: `del H, level` after the energy; no float64 copy of `free`; `del f`; one z vector | ~100 B/elem VRAM, 72 B/node; `GPU_BYTES_PER_ELEMENT` 470 -> ~400 | [T3](#t3), [T4](#t4) |
| 10 | `pack_meshes` into one preallocated buffer; send views in 1 MB slices | 3x -> 1x the blob while packing (measured), no send copy | [V1](#v1) |

P2: medium effort.

| # | Change | Saves | Where |
|---|---|---|---|
| 11 ✅ | SSD tier and `Shared` arrays memory-mapped (`np.load(mmap_mode='r')`); watchdog on private bytes or USS instead of RSS | private copies of every cache hit; 8 workers share one copy of a part's V/F | [S2](#s2), [C6](#c6) |
| 12 ✅ | Precompute a CAD mesh's normals and crack data once next to its `Shared` file, mapped by all workers | ~40% of each worker's per-triangle memory, the build peak in every worker | [C6](#c6), [C8](#c8) |
| 13 | `tiles.stitch` with int32 faces filled in place, one `remove_unreferenced_vertices`, no `merge_vertices` after plain marching cubes; estimate per triangle | stitch peak ~200 -> ~100 B/tri (1.6 GB on the 13.8M center) | [F3](#f3) |
| 14 | Browser: build the pick tree only for bodies the ray reaches; free CPU normals/colours after upload | ~1.5 GB transient for the lamp on the first right-click; 90-210 MB per big view | [V2](#v2), [V3](#v3) |
| 15 | FEA support/load sampling in chunks of ~200k points | up to 1.2 GB per support | [T2](#t2) |
| 16 | Topology states: read only the keys a preview needs; done states without `u` / power vectors | ~40 B/elem per preview, files 3-4x smaller | [T5](#t5) |
| 17 | GPU estimate per *split* triangle and cracks (180 / 680 B) | WDDM spill to RAM on cracked or flat-faced parts | [C2](#c2) |
| 18 | One `release_mesh_caches()` that clears every search-structure cache and the GPU pools, called by the low-memory release; FEA results and `sim.*` outputs released there too | 0.3-1.5 GB of the idle main process | [C4](#c4), [T6](#t6) |
| 19 | RAM cache written to the SSD behind its back (write-behind), so releasing it loses nothing | recompute after releases/restarts | [M7](#m7) |
| 20 | Views kept as files on the SSD, served with `FileResponse` | kept views cost disk, not RAM | [S1](#s1) |

P3: smaller or rarer: Picture/Text field in float32 on its band only ([F6](#f6)), Bar/Box in float32
([F7](#f7)), `/api/check` guarded and chunked ([F8](#f8)), `_merged` without a merged copy
([F9](#f9)), pattern nodes' whole-body `face_adjacency` ([F10](#f10)), STEP import peak
([C5](#c5)), `estimate_volume` on the GPU ([T7](#t7)), frame candidate bars ([T10](#t10)), the
browser's wire index, variants and highlight copies ([V4](#v4)-[V6](#v6)), a lower Final view cap
([V8](#v8), a product decision).

Not worth it now: a float32/int32 mesh class (0.2 GB for the lamp as shipped, 0.8 GB at 22M
triangles, ~10 call sites, CAD meshes must stay float64 for igl and fingerprints); worker results
through the SSD (block messages are 0.2-2 MB; only inside a streaming stitch); SSD-backed field arrays
(rejected in FINDINGS: marching cubes needs whole arrays).

## Streaming from the SSD: where it pays

- **S2 (P2 #11). The SSD tier and `Shared` arrays as memory-mapped files.** `DiskCache.load`
  (cache.py:139) and `shared._load` use plain `np.load`: a private copy per process. Measured on Windows:
  `np.load` +61 MB private for a 64 MB array; `np.load(mmap_mode='r')` and reading all of it: +0 MB
  private (the pages are in RSS but file-backed: Windows drops them without touching the page file).
  libigl (`AABB.init`, `squared_distance`), `per_face_normals` and `trimesh.Trimesh(process=False)` take
  read-only memmaps with identical results and no per-call copy (C-agent measured). Caveats: a mapped file
  cannot be deleted (`PermissionError`; trim already skips `OSError`); `Evaluator.memory_in_use` sums RSS,
  which counts mapped pages in every process: switch it to `memory_info().private` or USS first; first
  touch reads from disk, as `np.load` does now.
- **S1 (P2 #20). Views as files.** Pack each view straight into a file in the cache folder (one per
  version), serve it with `FileResponse` (streamed), read it through a read-only memmap in `check`,
  `_paint` and `face_region`. Kept views then cost disk, not RAM, and the packing and send copies go.
- **Big node results.** Meshes over ~50 MB in the evaluator cache (petal lists, fused lamp, center) could
  be written as float32/int32 `.npy` and memory-mapped when a node reads them (keyed by the node
  signature they could even survive a restart, if file inputs carry size and mtime as the STEP cache
  does). Do M2/M4 first: most of what sits there is not needed at all.
- **Streaming stitch.** Workers write block pieces as `.npy`, the main process fills preallocated final
  arrays from mapped pieces: stitch peak ~1.6 GB -> about the final mesh (250-500 MB) for the 13.8M
  center. Only worth it after F3.
- **Browser.** Read `/api/view` with `res.body.getReader()` into a buffer sized by `Content-Length`
  ([V9](#v9)).

## Measured in the test editor

Free RAM at start 8.1 GB (other programs ~23 GB), so the engine's limit was 6.2 GB. GPU free 4-5 GB.

| Step | Time | Main process | Workers | Notes |
|---|---|---|---|---|
| Start-up reopened the last test recipe (vent case, refine to Final in blocks) | | 0.8-1.3 GB | 4: 2.5-2.9 GB resident, 9.5-10.2 GB committed | |
| Open the bolted wall bracket (topology, detail medium); state already saved | 0.3 s | | the vent workers stay, idle | |
| Keep % 20 -> 25, Optimize | 40 s | 1.2 -> 1.8 GB | idle, 2.6 GB | GPU free 4.2 -> 3.8 GB |
| Detail medium -> fine, Optimize (medium 21 s, then fine) | 113 s | peak 3.1 GB resident, 6.3 GB committed (density mesh, check) | idle until stopped | tree peak 5.15 GB, free min 3.5 GB, GPU free min 3.35 GB; watchdog: "Memory 5.4 GB of 6.2 GB: stopped idle worker processes"; the next preview started 3 workers again within 2 s (2.4 GB) |
| Import STEP -> the key cabinet | 0 s import (SSD), 1.6 s Medium | 1.7 GB | 3: 2.0 GB | the closed vent recipe's 39 MB view still served |
| Open the vent recipe again | Draft 2.3 s, Medium 5.5 s, Final 23.5 s | 1.7 GB | 3: 2.0 GB, 7.8 GB committed | "Blocks: running 2 at a time to stay within the memory limit" with 10.9 GB free; the topology recipe's views (15.5 + 6.1 MB) still served |

Idle afterwards: main 1.73 GB resident (3.5 committed), workers 0.67 GB resident and 2.6 GB committed
each. System commit 47 of 78 GB (peak since boot 78.9). Browser tab JS heap 16-97 MB.

---

## Main process: evaluator, caches, server (coordinator)

### <a id="m1"></a>M1. Meshes keep trimesh's volume arrays (see F1). **Checked**
`vent_holes.json` at Medium: the view mesh (812,504 triangles) holds 27.9 MB of its own and 74.4 MB of
cached `triangles` (55.8) and `triangles_cross` (18.6); the packed view is 13.9 MB.

### <a id="m2"></a>M2. The evaluator never forgets nodes that left the graph. **Checked**
core.py:512, :736: `_cache` is only ever assigned. Node ids are names (`import`, `mesh`, `petals`), so
opening another recipe replaces the ids that repeat and keeps the rest for the life of the server.
`release_memory` (:542) looks types up in the *current* graph, so it never releases those either.
Fix: at the start of `_run` drop entries whose id is not in `self.graph.nodes`; store the node type in
the cache tuple. Risk low (an undo of a delete recomputes, mostly from the SSD cache).

### <a id="m3"></a>M3. The server keeps every view it made. **Checked (seen in the editor)**
server.py:332-335: `views`, `view_faces` (holds CAD mesh objects, can pin an old version of a body),
`view_sources` (a graph per view) have no `pop` anywhere except `view_faces` on a kind change. Fix:
after a preview drop entries (and `paint:` / `slice:` / `detail:`) of nodes not in the run's visible set.

### <a id="m4"></a>M4. Each shown body is held two or three times
The hidden `mesh.view` output stays in the evaluator (36 B/tri + 96 B/tri trimesh cache until F1) and
the packed blob in `Session.views` (~18 B/tri). After packing only a rerun with the same signature reads
the mesh. Fix: keep a stub (voxel, face count) for hidden view nodes; keep `mesh.from_implicit` outputs.
Lamp Final views (7.1M triangles): 0.25-0.9 GB.

### <a id="m5"></a>M5. The memory limit is fixed at start-up. **Checked (seen in the editor)**
cache.py:66-73 caches `_LIMIT`. Fix: per run, `min(0.6 * total, engine_in_use + 0.75 * available_now)`;
the FREE_* checks stay.

### <a id="m6"></a>M6. Caches shrunk under pressure never grow back. **Checked**
core.py:624-627 halves `ARRAY_CACHE.max_bytes` and the workers' cap every 0.5 s while tight
(workers.py:96, floor 16 MB); only a new pool resets them. Fix: reset to `ram_cache_budget(n)` at the
start of each run.

### <a id="m7"></a>M7. The RAM cache is not on the SSD, though two comments say it is. **Checked**
cache.py:341-353 writes to the SSD only on eviction; core.py:619 ("their cached results are on the SSD
anyway") and server.py:560 (`ARRAY_CACHE.clear()  # ... stay in the SSD cache`) assume otherwise.
Stopped workers, the idle release and restarts lose distance fields that were never evicted. Fix:
write-behind thread for `cache=True` arrays >= 1 MB.

### <a id="m8"></a>M8. Idle workers keep a closed recipe's memory. **Measured**
See the editor table. Fix: stop the pool when another recipe is opened; after ~1 min without work let
workers drop RAM caches, search structures and GPU memory (C3, C9) but keep the process (a cold start
costs 8-10 s on the lamp).

### <a id="m9"></a>M9. GPU workers commit ~2.6 GB each. **Measured, fixed 2026-10-31**
0.67 GB resident, 2.6 GB committed per worker; 8 GPU workers would commit ~21 GB. Commit is not RAM, but at
the commit limit Windows grows the page file or refuses allocations. M8 keeps the count down. Cause
(2026-10-31): mostly not CUDA (+0.6 GB) but numpy's and scipy's own OpenBLAS, 0.73 GB each at 32 threads;
workers now load it with their share of the cores, 1.2 GB per GPU worker ("Done").

## Fields and the mesh pipeline (agent; F1, F2 checked)

### <a id="f1"></a>F1. `field_to_mesh` calls `mesh.volume`. **Checked**
meshing.py:59. `.volume` peaks at 344 B/tri and leaves `triangles` + `triangles_cross` (96 B/tri) in
`mesh._cache`; pickle and cloudpickle include `_cache`, so a worker sends 132 B/tri instead of 36
(lamp Final petals: 300 MB through the pipe instead of 83, 220 MB of dead cache in the main process).
`np.where(..., -eps, eps)` with a Python float eps makes two float64 full-grid temporaries (meshing.py:45,
tiles.py:116, dualcontour.py:292). With `signed_volume` and `np.float32` eps on a 3.4M-voxel grid: peak
17.0 -> 9.0 B/voxel, 498 -> 264 B/tri, retained 132 -> 36 B/tri, vertices and faces `array_equal`. This
is most of FINDINGS' "805 bytes per crossing" for surface-dense grids. Also used by topology's
`density_mesh` / `LiveSurface`.

### <a id="f2"></a>F2. Merge Meshes (fuse) has no estimate. **Checked**
meshing.py:127-153, nodes.py:837 (not parallel, no estimate; `Evaluator` only uses estimates in
`_call_parallel`). Measured ~485 B per input triangle peak (manifold3d ~220 B per input triangle per
Manifold, the result while `solids` lives, then `result` / `parts` / `out` alive during
`snap_float32`). Lamp as shipped (~11.8M triangles in): ~5-6 GB in the main process; with the 0.1 mm
center (22M): ~10 GB, while idle workers hold 2.6-3 GB and the watchdog cannot interrupt C++. Fix: use
estimates for non-parallel nodes too (500 B x input triangles), `stop_idle_workers()` first, a readable
error if it still does not fit; `del` before `snap_float32`; optionally union pairwise in a tree.
**Done** (2026-10-24): measured again in a fresh process per version (three overlapping spheres, 983k
triangles in): the peak is inside manifold3d's union, 640 B per input triangle, so `FUSE_BYTES = 650`;
deleting `parts` / `result` / `out` before `snap_float32` did not lower it (634-644 B either way) and
was left out. A tree of pairwise unions remains the only way below that peak.

### <a id="f3"></a>F3. `tiles.stitch` peaks ~200 B/tri above the block pieces
tiles.py:203-246. Measured per triangle above live data: F as a list of int64 copies then concatenated
(85 incl. pieces), `remap[F]` (65), `merge_vertices` (114 peak). The 13.8M center: ~1.2 then ~1.6 GB,
matching FINDINGS' 2.24 GB main-process peak. `TILED_ESTIMATE` is a flat 1.5 GB (a 40M-triangle body
needs ~4.5). Fix: preallocated int32 F, chunked remap, one `remove_unreferenced_vertices`, skip
`merge_vertices` after unsimplified marching cubes (vertices distinct by construction); estimate ~150 B
x expected triangles. Welding by grid-edge key stays.

### <a id="f4"></a>F4. Stale cache entries (= M2/M3). <a id="f5"></a>F5. Views held twice (= M4).

### <a id="f6"></a>F6. Picture / Text evaluate float64 over the whole grid on the CPU
picture.py:156-172: 105 B/voxel measured (3.4 GB at 32M voxels against a 48 B/voxel estimate). Fix:
float32, only the slab `|off| <= band`, written into a zero grid (like the hole cutter).

### <a id="f7"></a>F7. Bar and Box are float64 and spread it downstream. **Checked**
fields.py:389, 403: `np.asarray(start, float)` makes `a[k]` a float64 scalar (not weak in NumPy 2):
32 B/voxel against Ball's 8, and every op after it stays float64. Fix: `float(a[k])`; values change at
1e-7, check the tests.

### <a id="f8"></a>F8. `/api/check` `mesh_report`: 296 B/tri, no guard
checks.py:21-51: 1.4 GB for the 4.8M view, 4.1 GB at 13.8M. `vertex_normals` 128 B/tri. Fix: chunk like
`signed_volume`, int32 edges, refuse above free memory.

### <a id="f9"></a>F9. `_merged` / `trimesh.util.concatenate`: 96 B/tri transient
nodes.py:808-810, 866-870 (Export, Merge with fuse off). Fix: preallocate, or write the STL per mesh.

### <a id="f10"></a>F10. Pattern nodes build whole-body adjacency in the main process
holes.py:692-704, facemask.py:281, :466, unfold.py: `face_adjacency` & co. at 408 B/tri peak (~1 GB on
the 2.4M part per pattern node), 300 B/tri until the cycle collector. Fix: selected triangles plus
neighbours only. Also: `voronoi_cells` allows 4M cells (cap ~200k); the hole cutter evaluates each hole
over its prism's AABB in float64 (~0.8 GB for a diagonal 40 mm slot at 0.1 mm on the whole-body path).

### <a id="f11"></a>F11. Smaller
Dual contouring block results 52.5 B/tri (marching cubes 25); `Context` evaluates the left child first
and cannot work in place (`writeable=False`): long chains peak at 3 arrays instead of 2; `Grid.coords()`
is dead code; `paint.field_at_points` puts throwaway paint grids into the ARRAY_CACHE and makes ~4 float64
copies per vertex (720 MB for a 15M view); 3MF/OBJ/PLY export are not chunked.

## CAD meshes, distance fields, GPU (agent; C1-C4, C10 checked)

### <a id="c1"></a>C1. trimesh caches refilled on CAD meshes. **Checked**
`cad_import.tessellate` clears the cache on purpose; then nodes_fea.py:93 (`np.asarray(body.triangles)[on]`:
the whole body's (n,3,3) float64 for a few faces), :235 (`.volume`, `.is_watertight`: 216 B/face measured),
facemask.py:143 (`same_part`: `area_faces`, 104 B/tri, run for every CAD view opened for picking), :270
(`face_normals`, 120 B/tri) fill it again, on the very mesh the evaluator and `view_faces` keep. 2.4M part:
+170-520 MB; a 12M import several GB. Fix: `body.vertices[body.faces[on]]`; `signed_volume` plus an
open-edge count stored by `tessellate`; areas and normals only for the triangles needed; `_cache.clear()`
as a safety net.

### <a id="c2"></a>C2. The GPU estimate counts original triangles. **Checked**
tiles.py:69 (`MESH_GPU_BYTES = 150` per original triangle) vs sdf_gpu.py:107, 141 (180, or 680 with
cracks, per *split* triangle; flat faces split up to 17x). A cracked 2.4M part: 1.63 GB per worker
against 0.36 GB estimated; 3 workers then ask more than the card has and WDDM spills into RAM. Fix:
`tessellate` stores open edges and a split estimate in `mesh.metadata`; the estimate uses them.

### <a id="c3"></a>C3. Worker search-structure caches are outside every budget
sdf.py:123-151 (`_QUERIES` 24 / 768 MB), sdf_gpu.py:100-155 (`_MESHES` 48 / 512 MB GPU), shared.py:21
(`_kept` 256 MB), per worker. A changed mesh (Scale slider, new topology result) is a new key; old ones
stay until the cap. Worst case 8 x (768 + 256 MB) RAM + 8 x 512 MB GPU, not counted by `worker_memory` or
`ram_cache_budget`, not shrunk by the watchdog. Fix: drop entries unused in the last 2 runs when the
worker is idle; caps in a shared `Value` the watchdog can lower.

### <a id="c4"></a>C4. `release_mesh_caches` is incomplete and has one caller. **Checked**
fields.py:458-467 misses `facemask._TREES`, `_FACE_TREES`, `unfold._FLAT` and the CuPy pools; only
tiles.py:82 calls it. `_face_tree` (facemask.py:148-164) always keeps the newest tree whatever its size:
~2.3 GB after one right-click on a 12M part (194 B/tri). `mesh_fingerprint` copies V and F with
`.tobytes()` on every call (430 MB at 12M; `memoryview` gives the same digest). Fix: complete it, call it
from `release_memory` and `watch_idle_memory`, drop the newest-kept exemption above ~3M triangles.

### <a id="c5"></a>C5. STEP import peak
cad_import.py: `_open_edges` ~170 B/tri at the end (~2 GB for 12M; a min/max key with in-place sort
needs ~55), fingerprint copies (C4), int32 <-> int64 face copies in the STEP cache, OpenCascade
triangulations alive until `import_step` returns, `scale_about` copies F needlessly. `cad_uv` (24 B/tri,
37% of a CAD mesh) is only read by the pattern nodes: could be a separate lazily mapped file.

### <a id="c6"></a>C6. Eight private copies of the same CAD arrays (see S2)
Each worker holds V/F (36 B/tri) and recomputes normals (96 B/tri) and cracks (~240 B/tri transient):
for the 2.4M part ~2.5 GB of byte-identical copies over 8 workers. Fix: map the `Shared` files; write
normals and crack data once next to them (temp file + `os.replace`, versioned) and map them too. The igl
tree (194 B/tri) stays per worker; Warp has no IPC on Windows.

### <a id="c7"></a>C7. CPU distance-field transients
sdf.py:106-120: `signed()` holds ~375 B per query point: ~750 MB per worker at `_CHUNK = 2_000_000`,
not in any estimate. `_CHUNK = 500_000` -> ~190 MB, no numeric change. `_Cracks` builds a throwaway
trimesh (a reference cycle: ~113 B/tri waits for the collector). The Warp CPU sign kernel (FINDINGS,
660 -> 42 ms) removes the transient.

### <a id="c8"></a>C8. `_GpuMesh` host work repeated in every worker
sdf_gpu.py:116-141: `subdivide_to_size` copies and sorts all edges even when nothing needs splitting
(265 B/tri, measured), then cracks and float64 normals only to cast them to float32; ~290 B/tri host peak
in up to 8 workers at once. Fix: skip the split when the longest edge <= 2 mm, skip crack detection when
the import found no open edges, take normals from C6's file; `parent` as int32. `tests/test_gpu.py` must
stay green.

### <a id="c9"></a>C9. GPU memory held when idle
`paint.field_at_points` (main process, GPU up to 12M voxels) never calls `release_gpu_memory()`
(**checked**); workers keep their CUDA context and `_MESHES` forever; `watch_idle_memory` only looks at
RAM. Fix: release after painting; idle workers drop `_MESHES` + pools after ~60 s; act on low GPU free
memory too.

### <a id="c10"></a>C10. Voronoi foam: 6.5 KB per cell in the server, guard at 2M cells. **Checked (guard)**
lattice.py:280. Measured at 30k cells (scipy Voronoi ~3.8 KB/cell, a Python set of edge tuples): 2M
cells would be ~12.5 GB (extrapolated), reached with a 1.2 mm cell on a 150 mm box. Fix: guard ~250k
cells with a message naming a cell size that fits; vectorised edge set; drop `vor` before splitting.

### <a id="c11"></a>C11. Smaller
Face Mask pickles its border segments into every block task and samples all of them per block (crop to
the block, `Shared`, cache the KD-tree); `nearest_on_grid_gpu` downloads 16 B per voxel; a mask built on
another mesh than the bodies is not counted by `worker_memory`; `body_from_mesh` keys F by V's
fingerprint, so a scale slider writes identical F files into `cache\shared`.

## FEA, topology, frames (agent; T1-T5 checked)

### <a id="t1"></a>T1. `density_mesh`: ~500 B per FE element, unestimated. **Checked (seen in the editor)**
topology.py:834 (`UPSAMPLE = 2`: 8 voxels per element), :884-894, :849-881: scipy distance transforms
(49 B/voxel, the first held while the second runs), gradient, maximum filter, label: ~63 B/voxel x 8.
Bolted bracket fine 0.65 GB, key box fine 2.7 GB, the 8M cap ~4 GB, in the main process. In the editor the
fine run's post-processing took the main process to 3.1 GB resident / 6.3 GB committed and tripped the
watchdog. Fix: estimate and check first (fall back to upsample 1 or a clear message); cast each EDT to
float32 and `del` early; later z-slabs with a ~40-voxel halo (the material level is global: two passes)
or CuPy's ndimage on the card.

### <a id="t2"></a>T2. Support and load sampling: up to 4M points x ~300 B
model.py:24 (`MAX_SAMPLES = 4_000_000`), :465-538: one 40 mm bolt hole at h = 0.5 mm wants 7.4M points;
`apply_conditions` peaked at 1.15 GB (measured). Supports keep the unstretched sampling, so ordinary holes
reach 1-4M points. `prepare` adds a KD-tree over ~3M skin points. Fix: chunks of ~200k points (supports
OR into `fixed`: bit-identical; loads `bincount` per chunk: 1e-16).

### <a id="t3"></a>T3. `Optimizer.step` keeps the multigrid hierarchy through the OC update. **Checked**
topology.py:694-703: `H` and `level` stay referenced through the filters and ~60 FFT convolutions; only
`H.power` is used after the energy. `del H, level` saves ~100 B/elem VRAM (177 B/elem RAM on the CPU) at
the OC-phase peak.

### <a id="t4"></a>T4. CG holds three float64 vectors it does not need. **Checked**
solver.py:579-581: a float64 copy of the 0/1 `free` mask, `f` kept to the end, two z vectors during the
preconditioner. Bit-identical fix saves 72 B/node: `GPU_BYTES_PER_ELEMENT` 470 -> ~400, fewer CPU
fallbacks on fine parts. Check `tests/test_gpu.py --gpu`.

### <a id="t5"></a>T5. Topology states read whole on every preview. **Checked**
cache.py:264-272 reads all keys: `x`, float64 `u`, power vectors (~42 of 51 B/elem) for a preview that
needs density, phi, space, grid and history. Done states never resume but keep `u` (~75% of the file).
Fix: load only the keys asked for; done states without `u` / `power_*`.

### <a id="t6"></a>T6. FEA results survive the low-memory release
`analysis._RESULTS` (4 x 12 B/node), `_TOPO_RESULTS` (4 x density mesh + rho), `sim.*` evaluator
outputs, main-process search trees and Warp meshes of result meshes: 0.3-1.5 GB. `Shared` keeps its
array after writing it. Fix: clear them in `release_memory` / `watch_idle_memory`; `Shared` drops `_array`
after `_store` and maps the file.

### <a id="t7"></a>T7. Result checks build a CPU libigl tree of the result mesh
model.py:121-127 (`estimate_volume` hard-codes numpy): ~0.6 GB per 1M triangles kept in `sdf._QUERIES`,
only to pick `h`, while the analysis then runs on the GPU. Fix: pass the analysis' `xp`, or the volume
from the state (`sum(phi * density) * h**3`).

### <a id="t8"></a>T8. Main-process Warp meshes count against the GPU check
`_array_module` (nodes_fea.py:269-288) compares free GPU memory while `sdf_gpu._MESHES` may hold 512 MB
of earlier meshes that `release_gpu_memory()` cannot free. Fix: `release_mesh_caches()` in its `again`
branch and after `element_fill`.

### <a id="t9"></a>T9. Smaller
`u` copied to the host four times around the displacement save (~400 MB at 6.6M nodes; do `* E` and the
norm on the device); `CPU_BYTES_PER_ELEMENT = 800` has no margin for topology (~790 measured incl.
optimizer arrays); the GPU path never checks host RAM (~150-200 B/elem); cuFFT plan cache not cleared by
`release_gpu_memory()` (unverified size); `estimate_lmax` draws 24 B/node random vectors on the host per
level; `_transfer` builds three float64 meshgrids on the fine grid; `Volume` converts two masks every
iteration.

### <a id="t10"></a>T10. Frames
ground.py:256: `query_pairs` as a Python set (172 B/pair) instead of `output_type="ndarray"` (33);
`_stays_inside` chunks by pairs, not samples (0.5-0.8 GB per chunk); candidates are not counted before
they are built (MAX_LATTICE allows ~1.7M); `_Inside` can reach 122M voxels with a tiny user-set spacing.
Small with automatic settings.

## Browser and view transport (agent; V1, V2, V7 checked)

A 15M-triangle view costs ~360 MB of JS ArrayBuffers (download 270 + normals 90) and ~360 MB of GPU;
replacing a view frees the old one (two generations coexist until garbage collection).

### <a id="v1"></a>V1. `pack_meshes` holds 3 copies; uvicorn copies once more. **Checked (code)**
server.py:215-237: `uint32` cast, `+ offset` (a copy even at 0), concatenate, three `tobytes`, two bytes
concatenations: 3.0x the blob at the peak (measured, 810 MB for a 15M view). Sending a `bytes` body:
uvicorn writes headers and body separately and the Proactor transport copies the body into a bytearray
(+64 MB for a 64 MB body, measured). Fix: fill one preallocated buffer via `np.frombuffer` (1.0x,
measured); `StreamingResponse` of 1 MB memoryview slices (+2 MB) or `FileResponse` (S1).

### <a id="v2"></a>V2. The first right-click builds a pick tree for every view. **Checked**
Viewport.tsx:268 calls `ensureBoundsTree` for all tagged meshes. three-mesh-bvh 0.7.8 allocates 96 B per
triangle plus ~360 B per node object while building: ~1.5 GB transient for the lamp's 7.1M triangles,
~3 GB at 15M; 14 B/tri kept. Fix: only for meshes whose bounding box the ray hits; bigger leaves for big
views; or pick on the server (it already answers `face_at` / `node_at`).

### <a id="v3"></a>V3. CPU normals and colours kept after upload
`computeVertexNormals` (90 MB at 15M) and Float32 colours (90 MB) stay in JS though nothing reads them
(except PickableView). Fix: `onUpload` frees them (handle context loss); or flat shading above ~4M
triangles; or the paint as a scalar attribute with a LUT shader (no 90 MB colour array, range changes are
uniforms).

### <a id="v4"></a>V4. Wire toggle
three builds the wire index in a plain JS array (90M numbers at 15M), keeps 360 MB CPU + 360 MB GPU until
the geometry is disposed, also after Wire is off. Fix: refuse or decimate above ~2M triangles, or put
`wireframe` in the geometry deps.

### <a id="v5"></a>V5. Variants
Each keeps its whole view (up to 4) and, when shown, a second geometry and normals of identical data.
Fix: one refcounted geometry per view object; cap variants by total triangles.

### <a id="v6"></a>V6. Right-click highlight re-uploads all positions and normals
Viewport.tsx:283-306: ~180 MB CPU + GPU at 15M while the menu is open. Fix: share the main geometry's
attributes, only the index is new.

### <a id="v7"></a>V7. Undo/redo keep views of nodes no longer shown. **Checked**
store.ts:1104-1131 swap `visible` and touch no view; deleting a node keeps `faceData`. Fix: one prune of
views, details, slices, paints, faceData, markers, frames to `visible`, called by undo, redo, node
changes, `applyEdit`, `run_done`.

### <a id="v8"></a>V8. `DISPLAY_MAX_FACES = 15M`
Every number above scales with it, while the screen has ~1.3M pixels and on-screen detail now handles
zooming. A cap of ~4-6M for Final views with Refine on would cut server packing, blob, browser memory and
pick trees by 60-70%; exports keep full meshes; check numbers would describe the shown view. Product
decision.

### <a id="v9"></a>V9. Downloads
`res.arrayBuffer()` per view, no abort: superseded views still download in full, two `view_ready` for one
node download in parallel. Fix: reader into a buffer sized by `Content-Length`, one AbortController per
node.

### <a id="v10"></a>V10. Smaller
`thicknessRange` sorts a boxed JS array on every Check panel render (~6 s at 7.5M vertices; a typed
`sort()` 1.3 s); stale details and result paints stay after their view changed; the painted -> plain
switch may keep the old colour buffer in three's VAO cache (unverified); `PickableView` builds a string
per triangle on every pointer move.

## Done

2026-10-24, P1 1-3 (CPU test suite: 383 passed, 29 skipped, 1 failed: `test_recipes_are_valid` on
`testlamp2.json`, which failed before these changes; GPU tests not run):

1. `meshing.field_to_mesh` decides the orientation with `signed_volume`; the off-zero `where` uses a
   float32 eps there, in `tiles._mesh_block` and in `dualcontour.off_zero`. Old and new gave identical
   vertices and faces at four voxel sizes with samples placed on and around ±eps; no trimesh cache is left
   on the mesh; peak 21 -> 13 B per voxel on a 614k-voxel sphere.
2. Previews call `Evaluator.forget` (results of nodes not in the recipe) and `Session._forget_views`
   (views, colours, slices, details, CAD meshes and graphs of nodes not shown); cache entries carry their
   node type, so `release_memory` no longer depends on the last run's graph. The editor's undo/redo keep
   only what is drawn for nodes still shown (`store.shownOnly`); deleting a node and `left_out` also drop
   its face data. Tests: `test_memory.py::test_forget_drops_the_results_of_nodes_left_out`,
   `test_server.py::test_views_and_results_of_nodes_no_longer_shown_are_forgotten`. In the test editor:
   delete the shown vent mesh, undo (the view comes back), redo: the view is gone at once and
   `/api/view/mesh` answers 404 after the next preview.
3. Merge Meshes has an estimate (`FUSE_BYTES` 650 per input triangle, `MERGE_BYTES` 100 when not fused);
   a node with an estimate that runs in the main process calls `Evaluator._make_room` first (stops idle
   workers if needed, else a GraphError saying how much is needed and free). Test:
   `test_merge.py::test_a_fuse_that_does_not_fit_stops_idle_workers_or_says_why`.

2026-10-24, P1 7 and lower thresholds (after the page file moved to the NVMe SSD; CPU suite 384 passed, 29
skipped, the same `testlamp2.json` failure):

7. `cache.refresh_memory_limit` sets the limit when each run starts (`Evaluator.run`): min(60% of RAM, what
   the engine holds + 75% of the RAM free then), fixed during the run; `Evaluator._reset_cache_budgets`
   gives the RAM caches their share of it again (the watchdog's halvings no longer last the session).
   Test: `test_memory.py::test_the_memory_limit_is_set_from_free_ram_when_each_run_starts`.
- `FREE_TIGHT` / `FREE_STOP` / `FREE_LAST_RESORT` 3 / 1.5 / 0.75 GB -> 1.5 / 0.5 / 0.25 GB, the last resort
  after 30 s instead of 10.
- Lamp, cold cache, 8 workers, GPU, 17.8 GB free: textured center at 0.1 mm 70 s (43.8M triangles), petals at
  0.1 mm + fuse 70 s, tree peak 11.3 GB, free RAM never under 7.7 GB; no "freeing caches" any more (the
  earlier run with a 10.9 GB limit had it). Run times as before (the center is limited by GPU memory).
- The same with another program holding all but 5 GB: the limit (75% of the free 5 GB: 3.8 GB, then 4.8 GB)
  stopped both stages with 2-3 GB still free, before the new free-RAM thresholds were reached; the PC stayed
  responsive (process start at most 49 ms). With little free RAM the 75% share, not the thresholds, decides.

2026-10-31, guards on commit instead of free RAM (FINDINGS, "The PC freeze", 2026-10-31):

- [M9](#m9): workers start with `OPENBLAS_NUM_THREADS` = their share of the cores (`StickyPool._start`):
  numpy's and scipy's OpenBLAS each committed ~23 MB per logical core (1.5 GB per worker, untouched); a GPU
  worker's commit 2.5 -> 1.2 GB. Test: `test_workers.py::test_workers_load_blas_with_their_share_of_the_cores`.
- Guards on commit: `cache.commit_room()` (GetPerformanceInfo: commit limit - commit) and `memory_room()`
  (less `COMMIT_TIGHT`) replace free RAM in the limit (min(60% of RAM, held + commit room - `COMMIT_STOP`)),
  the parallel slots, `_make_room`, `nodes_fea._array_module` and the whole-grid mesh check. The watchdog frees
  caches when free RAM < 1.5 GB or commit room < 6 GB, stops the run when commit room < 3 GB, use > limit, or
  free RAM stays under 0.25 GB for 30 s; the last resort is commit room < 1.5 GB (or use > 1.25 x limit) for
  30 s. The RAM caches stay sized from free RAM (`_CACHES`). `/api/memory` reports `commit_room` (the
  toolbar's tooltip). Tests: `test_memory.py::test_the_memory_limit_is_set_from_the_commit_room_when_each_run_starts`,
  `::test_the_watchdog_stops_runs_on_commit_not_on_a_full_ram`, the idle editor on low RAM or commit.
- The 0.05 mm gyroid export that the old limit stopped at 7.4 GB (4 GB free) now finishes: 126 s, 6.2 GB STL,
  engine 13 GB, commit at most 69.7 of 77.8 GB, free RAM under 0.3 GB for at most 2.5 s.

2026-11-09, P2 11 and 12 (CPU suite 581 passed, 42 skipped, the same `test_recipes_are_valid` failure on the
user's own recipes; GPU 42 passed):

11. `cache.read_npy` memory-maps `.npy` files (read-only, a plain ndarray view): `Shared` arrays from 1 MB
    (`shared._load`), `DiskCache` hits from `DiskCache.MAP_BYTES` (32 MB; mapped arrays read whole cost more the
    first time, page faults: 1 MB 0.98 against 0.43 ms, 32 MB 17 / 14 ms, 128 MB 74 / 60 ms with the file in
    Windows' cache, so small hits are read). `Evaluator.memory_in_use` sums `cache.resident_memory`, each
    process's private working set (`PROCESS_MEMORY_COUNTERS_EX2`), instead of RSS, which counts mapped files in
    every process: in the test editor's hex_studs export the meter peaked at 4.4 GB, the RSS sum at 6.1 GB.
    Tests: `test_shared.py`.
12. `shared.prepared(kind, key, build)`: arrays worked out from shared data, built by the first process that
    needs them (a lock file holding its pid; the others wait for its files, a dead holder's lock is taken
    over), written as `.npy` files plus a list written last, mapped by every process. Meshes from
    `sdf.PREPARE_TRIANGLES` (100k) with a key: `sdf.query_arrays` (normals, edge map, cracks; MeshQuery keeps
    only its igl tree of its own) and `sdf_gpu.gpu_arrays` (the 2 mm split, crack flags, float32 normals; the
    GPU worker uploads and keeps only `parent`). Results identical (`test_shared.py`, `test_gpu.py
    ::test_gpu_mesh_from_prepared_arrays_is_the_same`; hex_studs' exported STL the same SHA-256). Measured,
    1.31M-triangle sphere: 6 CPU workers private 0.48 -> 0.32 GB each (sum 2.86 -> 1.89 GB, commit 4.07 ->
    3.10 GB), a worker's build 4.4 -> 3.3 s (first) / 1.8 s (prepared); 4 GPU workers' start-up peak 3.45 ->
    1.33 GB, build 3.5 -> 2.7 / 0.8 s. `tiles.MESH_BYTES` 500 -> 370. hex_studs export (key cabinet, 109k
    triangles): meshing 22.5 -> 20.9 s, peak 4.38 -> 4.22 GB. The lamp is unchanged (its parts are under 100k
    triangles and its arrays under 1 MB).
