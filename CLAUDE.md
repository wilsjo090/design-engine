# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A personal, single-user nTopology-style tool for giving CAD parts decorative surfaces (wave and
noise textures). Pipeline: STEP → triangle mesh → signed distance field → node graph of field math
→ marching cubes → STL. Python engine + React node editor, Windows only in practice.
Detailed measurements, geometry facts and lessons learned: [docs/FINDINGS.md](docs/FINDINGS.md).
Read it before changing performance, memory, meshing or the lamp recipes.

## Commands

Python is the conda env `synkad` (3.12), not on PATH:
`C:\Users\willi\miniconda3\envs\synkad\python.exe` (below: `python`).

| Task | Command |
|---|---|
| Start the editor | `start.bat` or `python app.py` (opens http://127.0.0.1:8765). Flags: `--port`, `--no-browser`, `--no-build`, `--clear-cache` |
| All tests | `python -m pytest tests -q -p no:warnings` (~30 s) |
| One test | `python -m pytest tests/test_sdf.py::test_signed_distance_matches_reference -q` |
| GPU vs CPU equivalence | `python -m pytest tests/test_gpu.py -q --gpu` (skipped without `--gpu`; the rest of the suite is pinned to the CPU by `tests/conftest.py`) |
| Run a recipe headless | `python run_graph.py recipes/lamp_noise.json --set amplitude.value=0.5` (`--nodes` lists node types, `--save` writes the graph) |
| Build the UI | `cd frontend && npm run build` (`tsc -b` is the only type/lint check) |
| UI dev server | `cd frontend && npm run dev` (:5173, proxies `/api` and `/ws` to a running `app.py`) |
| Desktop shortcut | `python tools/create_shortcut.py` |

`app.py` rebuilds the UI when `frontend/src` is newer than `frontend/dist`, and only opens the
browser if a server already answers on the port. Python changes need a server restart; the browser
pane launch config is `.claude/launch.json` ("design-engine").

## Architecture

**Fields** (`engine/fields.py`): a `Field` is a lazy expression node with `fn(ctx)`, a content
fingerprint `key`, `children`, optional `bounds`, and a `cache` flag. Fields are closures and
are shipped to worker processes with cloudpickle. `Context(grid, cache)` evaluates a tree on one
`Grid`: it counts how often each sub-expression is requested and frees each array after its last
use; only `cache=True` fields (CAD distance fields, noise, 3D waves) are stored in the shared
`ArrayCache`. A field's `children` must list exactly what its `fn` evaluates (a mismatch is still
correct but keeps arrays alive longer).

**Distance fields** (`engine/sdf.py`): `MeshQuery` builds a libigl AABB tree and normals once
per mesh (cached per process by mesh fingerprint) and signs distances with the pseudonormal
test; points whose closest feature lies on an open edge (OpenCascade leaves T-junction cracks at
fillet corners) fall back to a winding-number tree. `mesh_to_sdf` computes exact distances only
in a narrow band, interpolating the rest from a coarse pass. Fields hold big arrays (CAD meshes,
the Voronoi foam) as `shared.Shared`: pickled into a task as a fingerprint (the array goes to
`cache\shared` once), read by each worker once; `sdf.arrays()` unwraps them, and search
structures only read them when they have to be built.

**STEP import** (`engine/cad_import.py`): `tessellate` reads all triangulations of a body in one
go through OpenCascade's binary writer (`BinTools_ShapeSet.WriteTriangulation`, each one added via
a bare face so the order is the faces' order); bit-identical to the old per-node loop, which
`tests/test_sdf.py` keeps as the reference (also for located bodies). `input.import_step` keeps
the meshes on the SSD (`cache.load_step_meshes`, `cache\step`, keyed by path, size, mtime,
tolerance and angle): bump `cache.STEP_VERSION` whenever `tessellate`'s output changes.

**GPU** (`engine/backend.py`, `engine/sdf_gpu.py`): field code is written against numpy and
runs unchanged on CuPy arrays (numpy ufuncs and array functions dispatch to CuPy); `Context.xp`
is the array module, caches always hold numpy arrays. Distance fields on the GPU use NVIDIA Warp
BVH queries with our own pseudonormal sign test (Warp's built-in normal sign is wrong on ties) and
winding numbers only for triangles touching cracks; meshes are split to ≤ 2 mm edges first
because Warp's float32 closest-point maths drifts on long thin triangles. Only grids ≥
`GPU_MIN_VOXELS` (1M) use the GPU, and only as many workers as the card's free memory allows
(read via NVML; CUDA's memGetInfo is wrong under Windows WDDM). Out-of-memory falls back to the
CPU per body. Marching cubes of a CuPy grid runs on the GPU too (`meshing._marching_cubes_gpu`,
Warp's `IsoSurfaceMarchingCubes`: same vertices and triangles as skimage); block meshing and
decimation stay on the CPU.

**Big bodies in blocks** (`engine/tiles.py`): grids above `TILE_MIN_VOXELS` (32M) are meshed
block by block: a coarse probe picks blocks near the surface, `core.parallel_map` sends them to
the worker pool (GPU), each block is marching-cubed and simplified with pyfqmr (borders locked),
and the pieces are welded by the grid edge each border vertex lies on. Blocks whose surface runs
into an uncomputed neighbour pull it into the next round. Seams are exact only because shared
samples are bit-identical in both blocks: `Grid.block` computes positions from a global index
(also in the Warp kernel), block starts are multiples of `ALIGN` so `Grid.coarser` lattices
(noise, distance-field coarse passes) coincide, and smoothing gets a halo (`fields.halo`).

**Face masks** (`engine/facemask.py`): `tessellate` stores the B-rep face number of every
triangle in `mesh.face_attributes["cad_face"]` (kept by `scale_about`). `field.face_mask` is 1 on
chosen faces: per voxel near the surface it finds the closest triangle (igl AABB, or the Warp
kernel `_nearest_grid`, which maps split triangles back via `_GpuMesh.parent`), takes the sign
from that triangle's face and the magnitude from the distance between the closest point and the
selection's border edges (KD-tree). A wide fade is sampled on a coarser grid and interpolated
(`facemask.sample_factor`: factors 6/3/2, chosen to keep block meshing exact). *Apply to: all
other faces* (`invert`) uses the complement of the picked faces, so the picked ones stay plain. Input type `faces` is a text list: `15` = face 15 on every
body (copies like the petals), `2:15` = face 15 of body 2 only (`mesh.metadata["cad_body"]`, set
by Import STEP, kept by `scale_about`), or names of saved selections (recipe `selections`,
resolved in `Graph.from_dict`). The editor picks them on a CAD view (`/api/faces/{node}`: face and
body per displayed triangle, whether the bodies are copies of one part (same faces, triangles and
area share per face), tangent groups). Masks have no bounds but an `extent` (`fields.display_extent`),
so slices and surface colours of a list of per-body masks show each on its own body.

**Lattices** (`engine/lattice.py`): TPMS (value / |grad|, exact gradients in `TPMS_GRADIENT`, so
thickness is in mm), periodic strut lattices (points folded into one octant of a cell,
`|q - rint(q)|`; valid because every lattice there is mirror-symmetric about cell faces and
mid-planes: a lattice without that symmetry, e.g. diamond struts, needs another method) and a
Voronoi foam (struts cut into pieces ≤ cell/3; exact nearest-piece distance in
`engine/segments.py`, one Warp kernel with a BVH and a doubling search cube, run on the GPU or on
Warp's CPU device in threads). All are pure functions of position, so block meshing works.
`units="density"` turns the thickness (TPMS solid: offset) input into the solid share: the
lattice's distance values are sampled once (one cell, or the foam's box) and their quantiles give
the thickness for each density (`lattice._density_curve`, `_lookup`).

**FEA** (`engine/fea/`, nodes in `engine/graph/nodes_fea.py`; plan in docs/TOPOLOGY_PLAN.md):
static stress analysis on a voxel grid. `model.py` lays the grid from the part's lowest corner
(element size from `detail` and the part's volume), fills elements from the part's signed distance
(`phi`, stiffness `phi * E`), and turns Fixed Faces / Load (CAD triangles sampled h/3 apart, via
`facemask` face numbers) into fixed DOFs and nodal forces; `Condition.traction(normals, points)`
gives force, pressure, bearing (contact side, cosine weighted) and torque, and a torque or a
force's Offset is a rigid-connection turn about the faces' centre (`_turning`: w = I^-1 M, adds
up to exactly M and no force). `element.py`: hex8 with Wilson's
incompatible modes (bending exact on cubes). `solver.py`: matrix-free products (slabs for full
levels, element lists for sparse ones), multigrid with Galerkin coarse levels and Chebyshev
smoothing, CG; float32 hierarchies (GPU) run CG and the fine product in float64 (all-float32 was
15% wrong while reporting convergence). `analysis.py`: element-centre von Mises stress, nodal
fields (`Field.unit`, `display_range` without the artificial peak at supports), report lines,
displacements kept on the SSD (`FEA_VERSION`). Port types `material`, `conditions`, `part`; node
options `paint` (Static Analysis colours its own body, `paint_ready` source "result") and `preview`
(Fixed Faces / Load run in every preview once wired, their `markers` go out with `node_done`).
`topology.py`: SIMP with density filter, Heaviside projection and optimality criteria on the same
grid and solver (`prepare` -> `Problem`, `Optimizer.step/run/state`); Max thickness adds a local
volume constraint (Gaussian neighbourhoods at most `LOCAL_SHARE` full, adaptively scaled p-mean)
handled by a two-multiplier OC update (`_update_two`, Illinois root finding); `density_mesh` turns the
densities into a smooth closed mesh (extended straight out past the design space, the state's `space`
mask, not into Avoid or the gap, from the nearest element inside, which the design space then crops;
Gap to faces keeps elements near the design space's faces empty except around supports, loads and
Keep, `_gap_cells`; blurred by the node's Smoothing, the level chosen
so the volume in the design space stays that of the densities; Smoothing is not part of the
topology key, so changing it never optimizes again). Node
Topology Optimization only computes when its hidden `compute` input is set: `build_run_graph`
sets it for the node of an "optimize" job (`/api/optimize`, queued like exports: `LONG_JOBS`) and
in exports, and `stamp` (`Session.stamp`, bumped after such jobs) makes previews read the saved
state again. States are in `cache\topology` (`load_topology` / `save_topology`, keyed by
`topology_key`; bump `TOPO_VERSION` when the method changes), checkpointed every 5 iterations and
on Stop; `core.report_view` shows the shape while it runs (`live_view` event); a node's first
output can carry a `note` shown on its card.

**Frames** (`engine/frame/`, plan and status in docs/FRAME_PLAN.md; node Frame Optimization in
`engine/graph/nodes_frame.py`, run on Optimize like Topology Optimization, results in `cache\frame`,
bump `FRAME_VERSION` when the method changes; the body, `frame/geometry.py`: one Warp kernel for
every profile (capsules, annulus, rectangles with holes, mitred pieces) that blends the two nearest
*different* bars with a smooth minimum (fillet joints; pieces of one bar never blend), plus the skin
at support / load faces joined with the same fillet; `frame/curves.py`: `smooth_bars` makes each
path of linked bars one Hermite spline through all its nodes, and the bars' orientation, shared
with the frame check):
`optimize.optimize_frame` builds a ground structure (`ground.py`: lattice nodes inside the design
space, one node per hole or pin and clustered nodes on flat support / load faces, candidate bars
that stay inside), solves the plastic minimum-volume truss LP (`layout.py`, HiGHS interior point;
member adding above 80k candidates; reweighted LPs so weak bars pay for the smallest size; joint
cost as extra length per bar), joins straight-through pieces into continuous bars, sizes them
(`sizing.py`: stress, Euler/Johnson buckling over the unbraced length, optimal grouping into *max
sizes*) and checks the frame with 12-dof beam elements (`beam.py`: clamped supports, von Mises at
the bar ends, linear buckling factor), bumping sizes until it passes.

**Meshes** (`engine/meshing.py`): `simplify` (pyfqmr quadric collapse, error threshold scaled by
voxel², so `keep` is reached at any voxel size) serves both `decimate` and the blocks. `reduce`
(node Reduce Mesh) works to a tolerance instead: the threshold schedule ends at tolerance²
(pyfqmr's own lossless mode cannot keep borders, `simplify_mesh(lossless=True)` hardly reduces),
folds (triangle pairs on the same vertices) are removed, and around remaining bad edges the
original triangles are kept and the rest reduced again with those borders locked. Mesh from
Implicit Body's `tolerance` runs it per block in the workers (open block borders kept, `snap=False`)
and `tiles.stitch` then reduces only the triangles touching inner block planes (half tolerance).
`mesh.merge` with fuse = yes is a manifold3d Boolean union (main process; ~12 s for the lamp);
`build_run_graph` switches fuse off below Final. Every mesh that ends up in a file after
decimation or fusing goes through `snap_float32`.
`core.report_message` puts a line in the editor log from inside a node.

**Value ranges**: `Field.range` is propagated by interval arithmetic (`fields._interval`,
`_unary_range`; constants, noise and 3D waves -1..1, masks and Smooth Step 0..1 set explicitly).
`node_done` events carry it to the editor (range tag on the card); Remap's `auto` uses it. A new
field type with a known range should pass `range=` to `Field`.

**Caching and memory** (`engine/cache.py`): `memory_limit()` = min(60% of RAM, 75% of RAM free
at startup). `ArrayCache` is a small RAM LRU in front of `DiskCache`, an SSD folder of `.npy`
files (`%LOCALAPPDATA%\DesignEngine\cache`) named by content fingerprint, so all processes share
it and it survives restarts. Bump `CACHE_VERSION` whenever SDF or noise numerics change. Two
subfolders: `step` (imported STEP meshes), `shared` (arrays for workers) and `topology`
(optimization states); `clear()` empties all.

**Graph** (`engine/graph/`): `core.py` holds ports, the `@node` registry, the JSON `Graph`
model and the `Evaluator`. `Graph.from_dict` expands groups (recipe `groups`; a node of type
`group:<name>` becomes copies of the group's nodes with ids `<node>__<inner id>`, `$in.<name>`
links are its inputs, Number nodes inside its settings) and resolves face selection names, so
the evaluator only sees plain nodes; `Graph.owners`/`aliases`/`groups` map back for events and
previews. The evaluator caches each node's outputs by a signature of its inputs
and upstream signatures (so an edit only recomputes downstream nodes), maps nodes over `Many`
lists (one STEP file with many bodies flows through the same graph), runs `parallel=True` nodes in
`workers.StickyPool`, and runs a memory watchdog during each run. `StickyPool` sends item *i* to
worker *i % active*, so a worker keeps the caches for "its" bodies between edits; cancelling only
skips queued tasks (killing workers would throw the warm caches away). `nodes.py` is the node
library; a new `@node` function appears in the editor automatically. Heavy per-body nodes take
`parallel=True` plus an `estimate` (bytes per call) that limits how many run at once.

**Server** (`engine/server.py`): the browser owns the graph and POSTs the whole recipe for every
run. `build_run_graph` applies preview quality (Draft = voxel ×4, Medium ×2, no decimation except
Final), adds hidden `mesh.view` nodes for visible implicit bodies and slice jobs for visible plain
fields (also group nodes). Progress streams over `/ws` (events of nodes inside a group are sent as
the group node; it reports done once, when its output is); meshes, slices and surface colours are
fetched as binary from `/api/view/{id}` (`slice:{id}`, `paint:{id}`: a field sampled on a grid and
interpolated at the view's vertices, `engine/paint.py`; CAD views are not coloured). Exports queue
(`Session.queue`; never cancelled by previews, which wait in `pending`); `/api/check` runs print
checks on a shown view (`engine/checks.py`: shells and cavities by connected components, wall
thickness by Warp ray casts); `/api/autosave` keeps unsaved recipes in
`%LOCALAPPDATA%\DesignEngine\autosave`; `/api/presets` stores node clips in `presets/`.

**Frontend** (`frontend/src`): one zustand store (`store.ts`) holds the graph, groups, face
selections, visibility, run status, views, slices, paints, variants, the export queue and undo
history (snapshots include groups and selections); `graphConvert.ts` converts recipe JSON ↔ React
Flow (and a group ↔ its edit view with `groupIn`/`groupOut` nodes). Group specs are built in the
browser (`groupSpec`). While a group is open, `nodes`/`edges` are its inner graph and the recipe's
own are in `editing.outer`; `recipe()` composes both. Edits bump `graphVersion`, which triggers a
debounced (450 ms) preview run. The viewport is three.js / react-three-fiber with Z up (section
plane via `clippingPlanes`, variants drawn to the left of the result, check panel).

**Recipes** (`recipes/*.json`): `{"version": 1, "nodes": {id: {"type", "inputs", "pos",
"visible"}}}`; an input is a literal or `{"link": "node.port"}`. Unknown keys are preserved, so
UI data lives in the same file. `lamp_noise.json` is the user's current main design.

## Rules that are easy to get wrong

- Edit files only with the Edit/Write tools or Python with `encoding="utf-8"`. PowerShell 5.1
  `Get-Content`/`Set-Content` round trips garble UTF-8 (`●` became `â—` and broke the UI).
- Worker processes re-import `__main__` (spawn): every entry script needs an
  `if __name__ == "__main__":` guard.
- Check an exported STL by reloading the file. In-memory checks miss problems caused by float32
  storage and vertex merging (`engine/meshing.py` handles both; keep `process=False` there).
- libigl's `FAST_WINDING_NUMBER` signed distance returns approximate distances (~0.02 mm off);
  use `WINDING_NUMBER` as the test reference.
- Measure memory with a kill switch (psutil, whole process tree) before claiming a memory fix;
  an uncapped run once crashed the user's PC. The machine has 32 GB, other programs use ~15-21 GB,
  and the page file is on a mechanical hard disk: running out of RAM freezes the whole PC.
- Before running tests or anything heavy, check that the user's own editor (port 8765) is not
  busy. A full test run (then including GPU tests) on top of an export froze the PC
  (2026-10-12); GPU tests now need `--gpu`. The engine's free-RAM thresholds are
  `core.FREE_TIGHT` / `FREE_STOP` (3 / 1.5 GB), and an idle editor releases workers and cached
  meshes when the PC runs low (`Session.watch_idle_memory`).
- Tests redirect the SSD cache to a temp folder and pin the CPU path (`tests/conftest.py`).
- Block meshing depends on fields being pure functions of position: a coarse sampling grid must
  come from `grid.coarser(factor)` with a factor dividing `tiles.ALIGN` (12), and a field that
  reads neighbouring samples must declare its reach (like `smoothen`'s `blur`) so `fields.halo`
  counts it. `tests/test_tiles.py` checks block values against the whole grid bit for bit.
- Memory per worker scales with the CAD meshes a field reads (each worker builds its own
  search structures): `tiles.worker_memory` counts them so big parts get fewer workers. Wrap big
  arrays a field captures in `shared.Shared` instead of putting them in task payloads, and don't
  queue all payloads at once (the pool feeds 2 per worker; terminate drains the queues).
- Never combine a single big body (the lamp center) with a per-body list in implicit nodes:
  every list item then meshes the big body's whole box (hours). Mesh them separately and join
  them with `Merge Meshes`. Bodies only fuse where they overlap; touching is not enough.
- Any GPU change must keep `tests/test_gpu.py` green: it compares GPU results with the CPU on a
  part with cracks and a hole (the cases that broke earlier GPU versions).
- Warp kernels run on Warp's own CUDA stream: synchronize between CuPy and Warp work
  (see `sdf_gpu._launch`).
- FEA on the GPU: keep the conjugate gradients and the fine-level product in float64 (only the
  multigrid in float32), and check a change against the CPU (`tests/test_gpu.py --gpu`): a
  float32 solver can report convergence and still be 15% off. Fully fixed faces make artificial
  stress peaks that grow with finer detail; don't read them as the part's strength.
- Write TypeScript with the Edit/Write tools, not Python string patches: escapes (`\n`, `\\`)
  were silently changed twice.
- Profile with the user's editor stopped, or say that it ran: an idle editor held 10.6 GB of RAM
  and ~6 GB of the 12 GB GPU (worker CUDA contexts, cached search structures), and with 2.6 GB of
  GPU memory left Windows moved allocations to shared memory: GPU timings came out ~10x slow.
  Ask before stopping it; check `/api/memory` (gpu_free) first.
- Testing the editor: use the "design-engine-test" launch config (port 8766), never restart the
  user's editor on 8765 (it may hold unsaved changes). The test editor shares the autosave folder
  and `presets/`: delete what a test created.
