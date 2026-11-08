# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A personal, single-user implicit-modeling tool for giving CAD parts decorative surfaces (wave and
noise textures). Pipeline: STEP → triangle mesh → signed distance field → node graph of field math
→ marching cubes → STL. Python engine + React node editor, Windows only in practice.
Detailed measurements, geometry facts and lessons learned: [docs/FINDINGS.md](docs/FINDINGS.md).
Read it before changing performance, memory, meshing or the lamp recipes.

## Commands

Python is the conda env `synkad` (3.12), not on PATH:
`%USERPROFILE%\miniconda3\envs\synkad\python.exe` (below: `python`).

| Task | Command |
|---|---|
| Start the editor | `start.bat` or `python app.py` (opens http://127.0.0.1:8765). Flags: `--port`, `--no-browser`, `--no-build`, `--clear-cache` |
| All tests | `python -m pytest tests -q -p no:warnings` (~7 min, peak ~2 GB) |
| One test | `python -m pytest tests/test_sdf.py::test_signed_distance_matches_reference -q` |
| GPU vs CPU equivalence | `python -m pytest tests/test_gpu.py -q --gpu` (skipped without `--gpu`; the rest of the suite is pinned to the CPU by `tests/conftest.py`) |
| Run a recipe headless | `python run_graph.py recipes/lamp_noise.json --set amplitude.value=0.5` (`--nodes` lists node types, `--save` writes the graph) |
| Build the UI | `cd frontend && npm run build` (`tsc -b` is the only type/lint check) |
| UI dev server | `cd frontend && npm run dev` (:5173, proxies `/api` and `/ws` to a running `app.py`) |
| Desktop shortcut | `python tools/create_shortcut.py` |

`app.py` rebuilds the UI when `frontend/src` is newer than `frontend/dist`, and only opens the
browser if a server already answers on the port (and then exits). Python changes need a server restart;
the browser pane's `.claude/launch.json` entry "design-engine" only attaches to the user's running editor
(started by start.bat; the pane must never own or restart it), "design-engine-test" starts the test editor.

## Architecture

**Fields** (`engine/fields.py`): a `Field` is a lazy expression node with `fn(ctx)`, a content
fingerprint `key`, `children`, optional `bounds`, and a `cache` flag. Fields are closures and
are shipped to worker processes with cloudpickle. `Context(grid, cache)` evaluates a tree on one
`Grid`: it counts how often each sub-expression is requested and frees each array after its last
use; only `cache=True` fields (CAD distance fields, noise, 3D waves) are stored in the shared
`ArrayCache`. A field's `children` must list exactly what its `fn` evaluates (a mismatch is still
correct but keeps arrays alive longer).

**Distance fields** (`engine/sdf.py`): `MeshQuery` builds a libigl AABB tree and normals once
per mesh (cached per process by mesh fingerprint, at most 24 and `MAX_QUERY_BYTES`; the GPU's
`sdf_gpu._MESHES` 48 and `MAX_MESH_BYTES`: every new shape is a new key) and signs distances with the pseudonormal
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
tolerance and angle): bump `cache.STEP_VERSION` whenever `tessellate`'s output changes. Besides
`cad_face` it keeps, for face patterns, the surface parameters at every triangle corner
(`face_attributes["cad_uv"]`, float32 (n, 3, 2): per corner because `process=True` merges the
nodes where faces meet, which carry a different (u, v) on each face; flipped with the corners when
the mesh is inverted) and per CAD face its surface type and u/v periods
(`metadata["cad_surfaces"]`). Both are kept by `scale_about` and stripped by `meshing.export`
for non-STL files. Each body's name in the file (SolidWorks writes its Solid Bodies name: "Fillet9",
"CirPattern1[3]"; `load_step(names=True)`, the STEP item each solid came from) is `metadata["cad_name"]`, and
`metadata["cad_solid"]` its solid's number in the file (doubles skipped count); names are kept beside the meshes
in a `.names.json` of their own (`cache.load_step_names`), so imports kept before names were stay valid: they
have no `cad_name` until the editor asks `/api/bodies` with names=true, which reads the file once
(`nodes.step_names`, up to half a minute for a big one) and names the cached bodies, copies too, by `cad_solid`.

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

**Transforms** (`engine/transform.py`, category Transform in `nodes.py`): Scale, Move, Rotate, Mirror and
Pattern Bodies act on CAD bodies, before Implicit Body from CAD Body (a field cannot be turned on the
axis-aligned voxel grid). `transformed` applies a 4 x 4 matrix and keeps face numbers, `cad_uv` and
metadata, so faces picked on a body stay picked on its copies; a mirror reverses every triangle's corners
and the per-corner `cad_uv` with them (the body stays outward). Move / Rotate / Mirror / Pattern take the
whole list (`many=True`: a list turns about the middle of all of it, as one); a single body gives a single
body back. Pattern lists the originals first, then each copy in turn; copies keep `cad_body` ("2:15" picks
that face on every copy of body 2). Mirror's plane can be a flat face (`face_plane`, its `cad` / `faces`
ports so the card's Pick works). The 3D view: *Transform ▸* on a CAD body, *Mirror about this face* on a
flat face (`viewActions.transformEdit`: after the CAD output, its users relinked, the result shown).
Pattern on Faces (`cad.pattern_faces` in `nodes_pattern.py`, `engine/copies.py`): copies of a CAD part (Part,
`many`: a list goes as one) standing on picked faces of another body (port `cad`, so Pick works), placed by the
face patterns' own `holes._frames` / `_face_holes` (layouts, Join Faces, Align, Randomness) and turned into 4 x 4
matrices for `transformed` (the copies keep the part's face numbers). The part stands as drawn: Up (auto = the
axis whose plane through the origin its box starts on, `copies.frame`) along the normal, its x along the rows, Base
bottom centre or origin; its footprint (convex hull seen along Up) is every copy's outline for the border and
crowding checks (`Holes.cells`, shape kind voronoi). Girth (`holes.GIRTH`, `_face_holes(girth=)` → `holes._around`,
only on faces that close around one way; holes pass girth None and keep the plain frame): rows run around, spread
evenly between the face's ends (the end rows Edge margin in with the footprint, `_reach_along`, plus the bend of a
flat footprint over a narrowing end's circle and the copies' lean over an end where the face widens away from it,
`height`, so nothing pokes past an end's plane); same size = each row's count by `_counts` (the widest row the
nearest, then a row keeps the count before it while 0.9-1.25 Spacings apart: `MIN_STEP` / `MAX_STEP`), staggered
rows of another count started where they nest best (`_nest`); same count = one count, arc-length rows; scale = the
along-parameter remapped conformally, everything grown by girth / mean girth (`grow`, the 8th result of
`_face_holes`, also `Holes.grow` so the Gap grows with it). Row directions there are chords along the row's line
(round the seam), not triangle Jacobians, and the girth and the distance along are measured on the face's own
triangles along lines of constant parameter (`_param_slices`): a cone's triangles run its whole height, and their
Jacobians smeared the girth by 6% and turned copies 0.7°. Crowded (`copies._crowd`): pairs closer than Gap (exact
footprint outlines, `_outline_gap`) shrink by bisection (`_largest`), or one is left out; At edges
(`_shrink_to_edges`): copies over the border shrink to fit, or (leave out) only those needing under 5% (`EDGE_TRIM`). Fit to cells voronoi: each copy's Voronoi cell among the
others in its own tangent plane (`copies._cells`, opposite normals no neighbours, capped at `REACH` x Spacing, cut
Edge margin in from the border where it faces the copy, `_border_lines`: along a neighbouring face's plane where it
meets the chosen ones at a fold, farther in by the copy's height x its lean over that plane, worked out again from the
same start with the first pass's heights, `_leaning`), the copy moved to the middle of a closed cell and back onto the
face (`_Surface`, igl), then by Fit (`FITS`): scale = the largest scale whose footprint fits (`_fits`: rays through the
hull's corners), Turn to fit tries `TURNS` angles; stretch = turned and stretched (area kept, up to `STRETCH`, along
the cell's long axis ± `STRETCH_TILT`), the `KEEP` best at the middle then moved by a pattern search (`_fit_shapes`,
`_homothets`), kept as `Holes.stretch` (turned frame, det 1) in the matrices; fill = the copy moved to its cell's
inscribed circle, the cells drawn again, the outline the cell itself with corners rounded as wide as the part's edge
(`_edge_depth`, `_fill_outlines`), and `Placed.bodies` bends each copy (`_Fill`, `_Warp`: distances from the outline
kept in a band, radial inside; `_refine`: red-green splits where a bent edge leaves its triangles' planes by
`FILL_TOL`; `_carried`: face numbers and interpolated `cad_uv`), wrapped onto curved faces. `place` returns a `Placed`
(unpacks as matrices, sizes, lines; for fill the matrices are only the copies' frames). Output a `Many`
whose own `note` is shown (`core._output_extras` reads a list's note first).

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

**Face patterns** (`engine/holes.py`, shapes and layouts in `engine/pattern2d.py`, nodes Hole
Pattern (`cutter`: holes, pockets, raised shapes for the Booleans) and Surface Pattern (`relief`:
0..1 for a Texture node, smooth step inside each outline, Invert via a surface Face Mask) in
`engine/graph/nodes_pattern.py`; plan and status in docs/PATTERN_PLAN.md): per chosen
face a 2D frame in mm from `cad_uv` (area-weighted Jacobian scales, turned by Align/Angle; faces
covering ≥ ~half a periodic surface get a whole number of cells around and only 90° turns), cell
centres located in the face's (u, v) triangles (centres in cut-outs find none), axes from the
face's own angle-weighted vertex normals. Join Faces `tangent` (the default): chosen faces joined
smoothly share one frame instead, an unfolded sheet (`engine/unfold.py`: cut open at folds over
`FOLD`, LSCM start then as-rigid-as-possible steps, cached per process) laid onto the CAD frame of
its largest face (`holes._unfolded`), so that face's pattern stays put; sheets that close around
(border loops that turn ~0 on the surface: a split cylinder) or stretch over `MAX_STRETCH` keep a
frame per face. Holes are dropped whole: outline closer than Edge margin
to the selection's border (edges to unchosen faces and sharp folds, outline tested against border
points, not a circle), found twice (seams), or too close to another face's holes. Depth from Warp
ray casts (`checks.ray_thickness` / `ray_first_hit`, centre plus 8 points around); Depth > 0 makes
pockets, Height > 0 raised shapes (for a union: they sink into the wall by the face's drop under
their outline + `JOIN`, at most 90% of the wall; a pocket as deep as the wall goes through, no
further). Size scale, Depth, Height and Rotation take numbers or fields; fields are read exactly at
each centre (`holes.read_at`, a one-sample grid per hole: a coarse grid misread surface Face Masks,
whose band is thinner than its cells), and 0 there means no shape; Min Size (Hole Pattern 1 mm,
Surface Pattern 0) leaves out shapes that a Size scale would shrink to pinholes. The cutter is
one straight prism per hole along its axis, evaluated hole by hole on the grid samples around it
(`FAR` elsewhere): a pure function of position, cheap (0.5 s for 211 holes on 46M voxels), the
same in blocks bit for bit. 2D shapes are exact distances (`Shape.sdf`; the cross is a union of two
boxes, exact outside only); Size is always the inner width (the largest pin that fits). Voronoi
cells give every shape its own convex outline (`Holes.cells`, `pattern2d.polygon_sdf`; every check
goes through `Holes.unit_sdf` / `radius` / `unit_outlines`, never `shape.sdf` directly).
Randomness is hashed per lattice cell with wrap-around on closed faces (`pattern2d.jitter`; a
field Randomness is read where each centre would be). Morph To blends a second shape per hole,
`unit_sdf = (1 - t) d_A + t d_B` (`Holes.shape_b`, `Holes.morph`; `_Shapes._unit` in the cutter). Surface Pattern's relief
stops at half the wall under each shape (`Holes.wall`), so offsets never reach a thin wall's far
side. Stars and chevrons are simple polygons (`simple_polygon_sdf`); `delaunay` cells are the
layout's Delaunay triangles (holes on the triangles' centroids); Alternate turns every other cell
(the triangle tiling needs odd polygons centred half way up, `Shape._lift`); Tilt leans hole axes
(depths from where each sample's axis meets the face). Boolean Subtract/Union take Edge round /
chamfer (`fields.subtract_edged`, `union_edged`); a cutter must be exact as far out as the radius,
so the Booleans call its `regrow`. Picture and Text (`engine/picture.py`) project an image onto a
face (flat, or around a cylinder's axis) as a 0..1 field. Standard patterns ship as presets in
`presets/` (25).

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
`facemask` face numbers; a load partly off the part keeps its total, put back by what each point
carries, not by area; `sample_triangles` puts loads' long thin triangles on stretched grids, while
supports and topology's skin keep `stretch=False`, since which nodes lie near a face depends on how
densely it is covered) into fixed DOFs and nodal forces; `Condition.traction(normals, points)`
gives force, pressure, bearing (contact side, cosine weighted; a direction along the faces, a hole's axis, is refused
by `_bearing_norm`, `MAX_BEARING_GROWTH`: scaling rounding noise up to the amount gave 1.2e18 N, and compliances of
1e31 that left the real loads hanging in the air) and torque, and a torque or a
force's Offset is a rigid-connection turn about the faces' centre (`_turning`: w = I^-1 M, adds
up to exactly M and no force). Load cases: node Load Case (`sim.load_case`, Loads / Name / Add to) sets
`Condition.case` on the loads of its chain that have none (in the key only when set);
`Conditions.cases()` gives each case with the supports and the loads in no case (those act in every
case), in natural order of the names; `apply_cases` -> fixed DOFs and forces (cases, X, Y, Z, 3);
`Model.forces` / `cases`, `analysis.solve_cases` (one hierarchy, a solve per case, displacements kept
per case on the SSD); Static Analysis reports each case (`report["cases"]`) and colours by the worst of
them everywhere; topology minimizes the sum of the cases' compliances (energies added; `Optimizer.u` a
list; pruning takes the most stress of any case); Frame Optimization takes one case. The editor
(`loadCases.ts`) colours a case's Load Case node and loads (card accent and a name chip), their arrows
in the 3D view and a legend (`CasePanel`: hover fades the other cases' arrows, click goes to the node);
pasting a Load Case gives it the next free name and joins it after the end of its original's chain of
cases (`pasteClip`; supports' and loads' body links are kept on any paste); the face menu's "add load"
names the case each chain end is in (`perTail`). `element.py`: hex8 with Wilson's
incompatible modes (bending exact on cubes). `solver.py`: matrix-free products (slabs for full
levels, element lists for sparse ones), multigrid with Galerkin coarse levels and Chebyshev
smoothing, CG; float32 hierarchies (GPU, and static analysis on the CPU) run CG and the fine
product in float64 (all-float32 was 15% wrong while reporting convergence); topology passes each
iteration's lmax power vectors to the next hierarchy (`power`). On the GPU the products (with the
Chebyshev and residual updates fused in), restriction / prolongation and topology's OC root finding
are fused Warp kernels (`fea/gpu.py`, `solver.GPU_KERNELS`, off with `DESIGN_ENGINE_FEA_KERNELS=0`;
the CuPy code stays as the reference):
a thread per node over its 8 elements (no atomics, block sums in fixed order: the same result every
run), launches recorded once and run on CuPy's current stream (no synchronizing; only changed
parameters are set). Loops with constant counts are unrolled by Warp: a 2x2x2 loop around the
element maths made each kernel compile ~60 s. Objects that hold GPU arrays must not sit in
reference cycles (a level and its kernels' data, a direct solver's closure and its level were): they
then wait for Python's cycle collector. GPU or CPU per analysis: `nodes_fea._array_module` releases
this process's CuPy pool, then compares `gpu_bytes` (measured: 128 MB + 470 B per box element,
topology 720 B) plus `GPU_HEADROOM` with the card's free memory; when it does not fit it drops this
process's GPU search structures (`backend.release_gpu_caches`: sdf_gpu's meshes, the foam's and frames'
BVHs), then stops the run's idle workers (`core.stop_idle_workers`, they hold RAM and a CUDA context
each), deciding again after each; an analysis left on the CPU says so in the log. `analysis.py`: element-centre von
Mises stress, nodal fields (`Field.unit`, `display_range` without the artificial peak at supports),
report lines, displacements kept on the SSD for E = 1 (`FEA_VERSION`; keyed without density,
strength and E). Static analysis' coarsest multigrid level, factorized directly, is the finest one whose
factorization is predicted to keep at most `direct_fill` nonzeros (`analysis.DIRECT_FILL`: on the CPU 25M or 5
per box element, on the GPU, where every millisecond iteration solves it on the host, 1 per box element;
`Hierarchy(direct_fill=)`,
`_finest_direct`: from the coarsest up, by the last level's fill ratio and how it grew; SuperLU without
pivots, `solver._factorize`, as the matrices are symmetric positive definite: with its partial pivoting a
23k-unknown level was not done in 10 minutes, 0.2 s without). Coarse levels whose elements are wider than
a part's members cannot bend them (a strut web: 154 iterations with the grid's levels, 28 with one), and the
fill depends on the shape, not the unknowns (a web's 86k 1.2 s, a solid block's 121k 44 s). Topology keeps the
grid's levels (it factorizes every iteration, `MAX_COARSE`). Port types `material`, `conditions`, `part`; node
options `paint` (Static Analysis colours its own body, `paint_ready` source "result") and `preview`
(Fixed Faces / Load run in every preview once wired, their `markers` go out with `node_done`).
`topology.py`: SIMP with density filter (cone, radius = the thinnest member; by FFT from
`FFT_TAPS`), Heaviside projection and optimality criteria on the same grid and solver (`prepare` ->
`Problem`, `Optimizer.step/run/state`); the stiffness is the eroded design's (`carrying`: projected at
`ERODED`, plus `THIN_SHARE` of what erosion removes, so the loads keep a path), so members thinner
than Min thickness cost mass and give nothing (the filter and a 0.5 projection alone made members of
1-2 elements), and reports use the design's own stiffness (`design_compliance`, saved as
`compliance`); Skin is never thinner than a set Min thickness; Max thickness adds a local
volume constraint (Gaussian neighbourhoods, outside the design space counting as empty, at most
`LOCAL_SHARE` full, adaptively scaled p-mean) handled by a two-multiplier OC update (`_update_two`, Illinois root finding;
the volume's multiplier never under `PRICE_FLOOR`: when Keep % does not fit, material must not be free, or
what carries nothing grows, e.g. loose pieces); Max section (`max_section`, mm2, in the key only when set) is the same
constraint on a neighbourhood wider than the members (`section_limit`, `Problem.section`, pooled with Max thickness's in
`_local_constraint`): along a long member the Gaussian integrates to 1, so at its axis the local density is its section
area / (2 pi sigma^2) whatever the section's shape, and sigma is chosen so the thickest plate that fits is `SECTION_PLATE`
of the thinnest member (0.5 x; filter radius between `SECTION_MIN` of a round member's diameter and diameter / 1.5). Only
`free` material counts (a skin cannot get thinner), a joint of n members holds n / 2 sections, the share does not depend on the area (15% with a Max
thickness, 8% alone), so Keep % above about 15-20% is not reached at any area. `_room` measures the design space (thickest
spot, second side, longest side, largest section): a Max thickness or Max section that cannot bind is off with a note, forced
members stop at what fits and sigma at half the longest side, and a Min thickness over the second side is refused (typed as
"no limit" these asked for gigabytes of filter in seconds, or a grey blob reaching no load). Max section's linearization overshoots at a sharp projection, a load path going
and coming back every other iteration, so with Max section a design variable that turned back gets a smaller move limit
(`_track_moves`, `TURN_SHRINK`, as MMA's asymptotes; `moves` and `last_step` are in the state; never without Max section,
so other results stay the same); `density_mesh` turns the
densities into a smooth closed mesh (its array work on the GPU when `SMOOTH_GPU_BYTES` per sample fit beside the
headroom, `_smoothing_module`; the node makes room first as for an analysis, `nodes_fea._room_for_smoothing`:
this process's GPU caches, then idle workers, and logs a smoothing left on the CPU; else or on running out of
memory the CPU: the same code on either, `_modules` by the
arrays' device; a fine result's 39.7M samples 87 -> 8.6 s; CuPy's pool given back between the steps; marching cubes
on the host, since Warp's splits cells along other diagonals and the Taubin smoothing then moved vertices half a
sample: the shape is the same to 1e-5 mm whichever ran it; extended straight out past the design space, the state's `space`
mask, not into Avoid or the gap, from the nearest element inside, which the design space then crops;
after smoothing, outside `crop` (the design space less Avoid) no sample is more solid than the nearest
one inside it, and loose bits are found in `crop` itself: a surface bulging over a hole wall left
paper-thin loose shells once cropped;
Gap to faces keeps elements near the design space's faces empty except around supports, loads and
Keep, `_gap_cells`; Smoothing blurs the signed distance to the densities' 0.5 surface
(`_surface_distance`; members keep their thickness, thin ones get at most `LIGHT_BLUR`), the level
chosen so the material in the design space stays that of the densities; Smoothing (default 2) is not
part of the topology key, so changing it never optimizes again; `thicken` moves what is still thinner
than Min thickness out smoothly, and `cut_tips` takes off what it cannot help, the points and ribs the crop leaves of
skins and plates at holes and folds (`TIP_SHARE`, `TIP_SECTION`; only what reaches `TIP_OUT` ball radii out, hangs by
one patch, has empty space all round it, `_free_all_round`, and under `TIP_FACE_SHARE` of a support's or load's faces on
it: a sharp edge, a strut as thin between two thicker places, a plate or flange as thin, a thin spot in a skin or a
loaded pin stays; thin is measured against Min or Skin as set, `tips_under`, never the grid's 3 elements, and with neither set
nothing is cut; the card's note counts them), and it says in `mesh.metadata["unreached"]` which loads the shape leaves
in the air (`_unreached`: over `REACH_MISS` of the force on no piece joined to a support, so a load on a loose piece or
one of its two holes counts; a warning on the card, from `_result`); the eroded design's filter counts the faces as empty,
`filter_empty`, so plates lying on them are not half as thick). Smoothness (off by default) minimizes
compliance x outline^Smoothness, the projected design's outline (`Optimizer.outline`; the filtered
densities' outline rewarded thin grey members and never shrank the real surface); Symmetry lays the
grid symmetrically about planes through the design space's middle and averages the sensitivities with
their mirror images (`Problem.mirror`, `symmetric`), so free densities stay exact mirror images; both
are in the key only when set. Start coarse (default yes, fine detail only, in the key then): `T.levels`
gives the coarser details whose elements can hold the Max thickness, `_optimize` runs each until the
projection sharpness reaches `HAND_OVER[level]` (`Optimizer.run(hand_over=)`), then `handover()` /
`take_over()` move the design variables (trilinear, `_transfer`), sharpness, iteration count and history
to the next grid; states carry their `level`, so a Stop continues on that level. Prefer (a guide body,
`Problem.prefer`: 0..1 per element, fading over `GUIDE_FADE` x the thinnest member) rewards the design's
material there in the sensitivities (Prefer strength x the mean stiffness sensitivity; negative keeps
material away softly); in the key only when set. The node analyses its
shown body with Static Analysis (`nodes_fea._check`, sized from the shape's own volume, finer than the
optimization's grid: on that grid a strut web came out 38% too stiff; the solver's progress on the card,
`_solver_progress`; kept in `_TOPO_RESULTS`): safety factor
and deflection in the note, warnings below 1 or for large deflections; Show colours the body by that
analysis (`paint="result"`, output `result`, None while Show is shape). Dead ends (default remove,
`T.dead_ends`: regions of the design under `DEAD_SHARE` of the median stress that reach the surface and
meet the rest through one patch, or carry nothing, `DEAD_IDLE`; with `reach`, only those sticking out
farther than that): while optimizing, every `PRUNE_EVERY` iterations from projection sharpness
`PRUNE_BETA` to the last, a solve with the design's own stiffness finds them (`Optimizer._prune`, quiet
patches of member surfaces included; its sensitivities come from the eroded design, where they look
loaded): what lies farther than Min thickness (`min_member`) from what stays is emptied and kept empty
(`pruned`, saved in the state; `_upper` caps growth at 0), nearer it is set to `PRUNE_REGROW` (a variable
at exactly 0 never grows again), so members keep room to grow; in the key as `prune` (results optimized
without are still shown, trimmed, with a note); after smoothing what sticks out farther than Min thickness
is emptied in the densities by the final body's analysis and the body smoothed and analysed again,
`DEAD_PASSES` rounds, the cut kept on the SSD (`topology-dead-ends`). Max iterations (input, default `MAX_ITERATIONS`) is not in the
key: states say whether they `settled` or stopped at it (`T.settled`, `T.finished`; older states: done
under 125 = settled), and a stopped one continues when it is raised (`Optimizer(max_iterations=)`, power
vectors kept unless settled, so it continues exactly). A state with nothing left to do (`T.finished`) is told to the
card as soon as the node has read it (`core.report_finished`, event `finished`; `out.finished` in node_done for the
cached node; `status.finished`): the result takes minutes to build, and a settled design's Optimize looked dead, so
`OptimizeBar` shows Finished ✓ and why instead of a button that does nothing. A click shows Starting… (`store.starting`)
until `/api/optimize` answers, which is when a running preview has stopped. Node
Topology Optimization only computes when its hidden `compute` input is set: `build_run_graph`
sets it for the node of an "optimize" job (`/api/optimize`, queued like exports: `LONG_JOBS`) and
in exports, and `stamp` (`Session.stamp`, bumped after such jobs) makes previews read the saved
state again. States are in `cache\topology` (`load_topology` / `save_topology`, keyed by
`topology_key`; bump `TOPO_VERSION` when the method changes), checkpointed every 5 iterations and
on Stop (unfinished ones with the power vectors, so a resumed run continues exactly);
`core.report_view` shows the shape while it runs (`live_view` event, `topology.LiveSurface`); a node's first
output can carry a `note` shown on its card.

**Frames** (`engine/frame/`, plan and status in docs/FRAME_PLAN.md; node Frame Optimization in
`engine/graph/nodes_frame.py`, run on Optimize like Topology Optimization, results in `cache\frame`,
bump `FRAME_VERSION` when the method changes; the body, `frame/geometry.py`: one Warp kernel for
every profile: round bars and tubes round one quintic spline per continuous bar (`geometry._curves`,
C4, straight on past its ends; the kernel's `_closest` Newton steps; per bar the piece whose curve point
is nearest decides, then its ends are rounded, `ramp`), rectangles with holes as mitred pieces; it keeps
the nearest *different* bars (`kernel.SLOTS`) and blends them thickest first, each into those before it
(k from that bar, the thinner of each pair), its fillet faded out between one and two fillet sizes away
(`fade`; the boxes and the outside value `cap` cover two); Fillet continuity G1 / G2 (default) / G3
picks the blend's polynomial (`fields.bulge`, the kernel's `_bulge`) and the ends' rounding (G2 and G3
are `CONTINUITY_SIZE` larger, so the fillet's middle is as deep as G1's; in the frame state's `shape`
only when not G1, since older states were G1). Two nearest bars only tore seams, a union of pair blends
creased, a polyline of capsules creased at every node: FINDINGS "G2 and G3 fillets". Plus the skin
at support / load faces joined with the same fillet (`skin_body`: a piece of faces that is a cylinder,
`_cylinder`, is measured exactly as one, a hole as a ring all round, its ends rounded by `ramp`, the
distance from its axis rounded within half the radius; other pieces by their triangles; the field capped
only past the fillet's reach); `frame/curves.py`: `smooth_bars` makes each
path of linked bars one Hermite spline through all its nodes, and the bars' orientation, shared
with the frame check):
`optimize.optimize_frame` builds a ground structure (`ground.py`: lattice nodes inside the design
space, an automatic spacing made `FINER` while a support or load gets no bar (`Unreached`: a lattice
coarser than a wall can miss it) or all nodes lie on one line or plane with a load across it (`_across`, `Flat`: a
slender bar's single row; its error suggests a spacing a bar apart across where the part is 3 x Min size thick, else
says no frame fits: jumping there made LPs of 6 GB); a typed
spacing is counted before the design space is sampled (`_Inside`: at most `MAX_INSIDE` samples, beyond its grid the
distance grows, `thickest()` measured again round the deepest sample, a part under 4 steps thick sampled again at a
quarter of it; Min size thicker than that is refused), one node per hole or pin and clustered nodes on flat support / load faces; supports whose
nodes lie on one line, `_on_a_line`, get two nodes per bolt hole along its axis instead, else the
pin-jointed frame turns about that line, and holes on one axis (an axle's lugs) nodes round their walls;
a piece of faces acts at one node in its middle, with bars free to leave the design space within its reach,
only if it is a hole or pin (`_cylinder`) or small (`_compact`), else it gets nodes on its faces; nodes
carry their `piece`, so Supports: connect all joins a hole only when none of its nodes is used; a piece chosen twice
(a load on a held face, two loads on one) is clustered alike (`_face_nodes`' memo), so their nodes merge and a load
all on supports or loads that cancel are refused; bars from a face node may leave the design space within
max(half a spacing, `FREE_SIZE` x Min size); a load no chain of candidate bars joins to a support (a neck, separate
bodies) is `Unreached` (`_joined`); candidate bars that stay inside), solves the plastic minimum-volume truss LP (`layout.py`;
one layout's LPs are one model kept in scipy's own HiGHS bindings, `layout._Model`: member adding adds columns, the
Min size rounds change costs, each solve interior point without crossover, the vertex crossover cost as much again and
its duals named 6x the bars to add; a simplex from the last basis, PDLP and the dual simplex took minutes; Stop and the
card's "LP 8, interior point iteration 28" come from HiGHS's IPM callback, `TICK`; without the bindings `linprog`;
member adding above 80k candidates, stopped after `STALL_ROUNDS` rounds that lower the volume by less than `ADD_STALL`
(a degenerate optimum's duals kept naming bars: 30 s LPs for 60 rounds) and going on from the round before's bars,
its reweighting stopped after `PATIENCE` rounds without a lighter layout; the first LP's cost bounds the volume any
truss needs (`Layout.least`): over twice the design space's, the frame is refused at once; `LY.NoFrame` says when Min
size is nearly the part's thickness (`_why_none`);
loads outside `optimize.LP_LOADS` are scaled for every LP together with Min size's area, as the tolerances are
absolute; reweighted LPs so weak bars pay for the smallest size; joint
cost as extra length per bar), moves the free joints off the lattice (Move joints, `joints.py`: sequential
LPs in forces and joint moves together, each joint's balance and the bars' lengths linearized in the moves,
moves within a shrinking box, kept when the layout's LP over the moved joints costs less and every bar stays
inside (`_stays_inside`); then joints that met merge, idle bars go, joints on a straight bar are taken out;
states are judged by their volume after `REWEIGHT` LPs, `_settle`, and the lightest is kept, without idle
bars; `LY.connect` keeps moved positions and joins left-out supports only through joints that did not
move; in the node key only when on), joins straight-through pieces into continuous bars, sizes them
(`sizing.py`: stress, Euler/Johnson buckling over the unbraced length, optimal grouping into *max
sizes*; Max size, `max_size`: the layout's LPs and the joint moves' LPs bound each candidate's tension and
compression by what a bar of that size carries over its own length, `sizing.force_limits`, times a share, so
a load goes over more bars; `MAX_SIZE_SHARES` are tried in turn while bars at Max size fail the beam check,
`MaxSizeShort`, since the elastic frame does not share a load as the layout did; sizing, the beam check and
the body check never go above it; in the node key only when set) and checks the frame with 12-dof beam elements (`beam.py`: clamped supports, von Mises at
the bar ends, linear buckling factor), bumping sizes until it passes (`optimize._checked`, at least `MIN_GROWTH` a
round; never thicker than fits the design space, `_room_size`: the section's thinner side against the thickest spot;
bars that need more fail as `MaxSizeShort` with `room`, and `optimize_frame` lays the frame out again with that as Max
size; the body check stops there too, the report's `room`); Light bars:
remove then takes out bars under `LIGHT_SHARE` of their strength one by one, lightest force first, each
only if the frame, checked again, passes and weighs less (`_drop_light`: dangling bars go along, loads and
supports keep a bar, nodes left without bars are held in the check; a brace that shortens a compressed
bar's buckling length stays, since that bar would have to be thicker); then the node analyses the
body (joints, fillets, bosses) with Static Analysis at medium and thickens the bars at its most
stressed spot until it reaches the safety factor (`nodes_frame._size_for_body`, keeping the safest
round; one round may dip, two in a row without gain stop it; the beam check before it thickens up to `MAX_ROUNDS`; still short
by a support's or load's faces after a fresh layout, `_spread_where_weak` spreads those faces over several
nodes, `ground` mode "around", lays out again and keeps the safer body; the state keeps
`beam_sizes`, the body settings checked as `shape`, and the `check`, so a changed skin, joints or
fillet sizes the bars again on Optimize without a new layout; the body's key includes the sizes). *Mid-bar joints*:
`ground.refine` adds nodes along the used bars and the LP runs again (not when that makes over `optimize.MID_BAR_LP`
candidates: a note says so; their bars reach twice as far, so those LPs grow dear fast: 2 mm on the block 374k, 11
minutes); a bar runs on (straight or as
one curve) through joints where others land (`layout.continuous_bars`). `layout._held_bars` drops
solver leftovers that reach no support (a lone 0.1 N bar made the check singular). Joining: a list
of design spaces is one ground structure (`ground._Inside` replaces the union's zero seam where
spaces only touch by a voxel distance; bodies are cropped by `geometry.seamless`); *Connect to*
(port type `frame`: a Frame output or a CAD body) adds held nodes on the other frame's nodes and
bars or the body's surface (`ground._attach_nodes`, `Ground.attach`; the log gives the force put
on it, which is not added to the other frame); Join Frames (`geometry.join_body`) draws all frames'
bars in one kernel so joints between frames get the same fillets. A frame laid out and sized for these settings (its state's `shape` the
same) reports `finished` (`core.report_finished`), as Topology Optimization does. Frame view: `node_done` carries
`frames` (`nodes_frame.frame_view`), drawn by `FrameBars` / `FramePanel` in Viewport.tsx. While it optimizes,
`optimize_frame(live=)` reports the frame after every step (the layout's LPs, mid-bar LPs, each joint move and
tidy, sizing rounds, light bars; `_size_for_body` each body-check round with its weakest `spot`) to
`nodes_frame._Live`, which sends `live_frame` events (`core.report_frame`, `live_view`: bars not sized yet as
thick as their force needs; the layout's every member adding round and Min size round with its `numbers`: LP count,
bars in the LP of the candidates, bars to add, volume and volume with Min size; every step's `mass` as drawn; at most
every `LIVE_EVERY` s, every new stage at once, which also moves the card's progress through `optimize.STAGES`; an LP's
iterations go to the card's step through `_Live.tick`, every `TICK_EVERY` s); the editor draws `store.liveFrames`
instead of the node's frames, its body see-through (`ghost`), until the node's result or run_done, and keeps the
run's steps (`liveHistory`) for Replay in the bars panel (`ReplayBar`: joints move and bars are sized in under a
second) and its calculation view (`CalcView`: the stages done, now and skipped, what the stage does, `STAGE_INFO`,
the LP's numbers, the iteration it solves, and the mass over the steps on a log scale; a click there replays from
that step). Show (stress, safety
factor, displacement) colours the body from the body check's analysis (`nodes_fea._check` at `CHECK_DETAIL`,
output `result`, `paint="result"`, as Topology Optimization). Show: candidate nodes (`NODES_SHOW`, also before
Optimize) builds the ground structure Optimize would (`_ground_view`; when a support or load gets no bar,
`ground.Unreached` carries what there was and which nodes are `lonely`), sent as node_done `ground`
(`store.grounds`): `GroundNodes` in Viewport.tsx draws the lattice grey, support / load / connection nodes in
their colours and unreached ones as large red balls over everything, the node's body see-through; with the
section on only the lattice layer one spacing behind the cut, so a wall the lattice misses shows no dots.

**Meshes** (`engine/meshing.py`): `simplify` (pyfqmr quadric collapse, error threshold scaled by
voxel², so `keep` is reached at any voxel size) serves both `decimate` and the blocks. `reduce`
(node Reduce Mesh) works to a tolerance instead: the threshold schedule ends at tolerance²
(pyfqmr's own lossless mode cannot keep borders, `simplify_mesh(lossless=True)` hardly reduces),
folds (triangle pairs on the same vertices) are removed, and around remaining bad edges the
original triangles are kept and the rest reduced again with those borders locked. Mesh from
Implicit Body's `tolerance` runs it per block in the workers (open block borders kept, `snap=False`)
and `tiles.stitch` then reduces only the triangles touching inner block planes (half tolerance).
`engine/dualcontour.py` is dual contouring (sharp CAD edges and Boolean rims at a coarser voxel; plan,
measurements and what did not work in docs/DUAL_CONTOURING_PLAN.md): Mesh from Implicit Body's
`method` (`nodes.MESHERS`); its views get no detail (`server.detail_floor`: details are marching cubes).
In blocks (`mesh_tiled(method=)`, `tiles.dual_blocks`): each block is sampled with `BLOCK_HALO` more,
keeps the vertices of its own cells and the quads of its own edges, names vertices by global cell and
piece, and `dualcontour.join_blocks` joins them by name (fold-repair moves of a neighbour's vertex are
sent as requests); Keep / Tolerance then reduce the whole mesh in the main process.
`mesh.merge` with fuse = yes is a manifold3d Boolean union (main process; ~12 s for the lamp);
`build_run_graph` switches fuse off below Final. Every mesh that ends up in a file after
decimation or fusing goes through `snap_float32`. Never call trimesh's `.volume`, `.triangles`,
`.area_faces`, `.face_normals` or `.is_watertight` on a mesh that is kept (a node result, a CAD body):
trimesh caches 72-216 B per triangle on it, pickled with it to and from workers; use
`meshing.signed_volume` (chunked) or compute on the triangles needed. `field_to_mesh` and the blocks replace NaN /
inf samples (`finite_values`: NaN and +inf one voxel outside, -inf inside, the same in every
block) and record `mesh.metadata` `nonfinite` and `clipped` (the body reaches the grid's outer
samples); `nodes._report_mesh` warns about both (not for a Domain or a lattice's own box,
`fills_box`).
`core.report_message(text, level)` puts a line in the editor log from inside a node (levels
info, warning, error); `core.report_progress(done, total, step)` drives the card's and status bar's
progress (`parallel_map`'s `title` is its step); runs slowed down to stay within memory say so in the
log (`Evaluator._notice`). Warnings and errors are kept with the node's cached result and sent with
every `node_done` (`messages`), so the card shows them on cached runs too (the hidden mesher's as
`viewWarnings`; a group's inner nodes' on the group). In a worker process the lines go back with
the item's result (`_current.sink`); the same line from many list items is reported once, "(xN)".

**Value ranges**: `Field.range` is propagated by interval arithmetic (`fields._interval`,
`_unary_range`; constants, noise and 3D waves -1..1, masks and Smooth Step 0..1 set explicitly).
`node_done` events carry it to the editor (range tag on the card); Remap's `auto` uses it. A new
field type with a known range should pass `range=` to `Field`. Offsets use it for bounds: body -
d (or + d) grows the bounds by d's reach, and `Field.grown` keeps (the box before, the reach), so
the mesher (`fields.mesh_box`) uses max(Padding, reach - voxel) around the original box: the grid,
and the distance fields cached on it, do not change while Padding covers the offset. A custom
field that keeps a body's bounds should copy its `grown` (otherwise it just gets a bigger grid).

**Caching and memory** (`engine/cache.py`): `memory_limit()` = min(60% of RAM, what the engine
holds + the PC's commit room (`commit_room`: commit limit = RAM + page file, less what all programs
committed) - `COMMIT_STOP`), set when each run starts (`refresh_memory_limit` in `Evaluator.run`,
which also resets the RAM caches' budgets: 20% of the limit, but only of what is held and free) and fixed
during it (recomputed on every check it would grow with the engine's own use). Planning (parallel slots,
`_make_room`, an analysis's CPU memory, a whole-grid mesh) uses `memory_room()` = commit room -
`COMMIT_TIGHT`, not free RAM: a full RAM pages other programs out to the NVMe, while running out of commit
fails any program's allocations. `ArrayCache` is a small RAM LRU in front of `DiskCache`, an SSD folder of `.npy`
files (`%LOCALAPPDATA%\DesignEngine\cache`) named by content fingerprint, so all processes share
it and it survives restarts. Bump `CACHE_VERSION` whenever SDF or noise numerics change. Two
subfolders: `step` (imported STEP meshes), `shared` (arrays for workers) and `topology`
(optimization states); `clear()` empties all.

**Graph** (`engine/graph/`): `core.py` holds ports, the `@node` registry, the JSON `Graph`
model and the `Evaluator`. `Graph.from_dict` expands groups (recipe `groups`; a node of type
`group:<name>` becomes copies of the group's nodes with ids `<node>__<inner id>`, `$in.<name>`
links are its inputs, Number nodes inside its settings) and resolves face selection names, so
the evaluator only sees plain nodes; `Graph.owners`/`aliases`/`groups` map back for events and
previews. A node with `"bypass": true` passes its first linked input of its first output's type on
(`bypass_input`): `from_dict` relinks what used it (chains followed, `Graph.bypassed`), and a shown
bypassed node shows what it passes on. The evaluator caches each node's outputs by a signature of its inputs
and upstream signatures (so an edit only recomputes downstream nodes), maps nodes over `Many`
lists (one STEP file with many bodies flows through the same graph), runs `parallel=True` nodes in
`workers.StickyPool`, and runs a memory watchdog during each run. `StickyPool` sends item *i* to
worker *i % active*, so a worker keeps the caches for "its" bodies between edits (each worker starts
when first given work: runs limited by free GPU memory left half of them idle at ~0.3 GB each); cancelling only
skips queued tasks (killing workers would throw the warm caches away). `nodes.py` is the node
library; a new `@node` function appears in the editor automatically. A `Port` can say when it applies
(`when={"shape": [...]}`, a list of such dicts = any of them; the editor's `portApplies` hides it otherwise,
on the card and in QuickEdit) and be `advanced` (under the card's "▾ N more settings" row while it has its
default and no link): keep cards short, e.g. Surface Pattern shows 12 of its 28 inputs for round shapes.
A number input also takes a `unit` ("mm", "°", "×"; or `{"kind": {...}}`, by another input's value, as the
Load's Amount), `zero` (what 0 means: "auto", "no limit", "sharp", "through"; the field shows that word and
emptying it sets 0) and `slider=(low, high)` (a bar under the field for the usual values; typing may go
beyond). The editor's fields are one set, `components/PortInputs.tsx`, used by cards, QuickEdit and the
analysis wizard: yes/no and on/off choices are switches, up to 4 choices of 20 characters in all are
buttons, Shape / Morph To / Layout / Profile are pictures (`choicePictures.tsx`, keyed by input name), a text
input whose default is three numbers (Centre, Size, Vector) is three X / Y / Z fields.
A `Port` can also give a list to choose from (`pick`, `components/PickLists.tsx`, on cards and in QuickEdit):
"body" (an int, Pick Body's Index: every item of the node's first cad input with its STEP name, size and faces
from `/api/bodies`, read again after each run; the hovered one's box is drawn in the 3D view, `store.bodyHint`,
`BodyHint.tsx`; a filter box, arrows and Enter; a list not made yet offers to show its source) or "font" (Text's
Font: `/api/fonts`, `picture.font_list`, every installed font shown in itself). `switch=True` shows a 1 / 0 int
as a switch (Skip duplicates); int inputs named seed get a dice; path fields show the end of the path (the file
name) and a folder button (`/api/reveal`: Explorer with the file selected, or its nearest folder). The lists are
drawn on `document.body` (readable at any zoom), so QuickEdit's outside-click check lets `.pick-list` clicks by.
Every wired cad input on a card (not Pick Body's own list) has a body chip (`PickLists.BodyChip`) after "← source",
so the body is chosen where it is used: with a Pick Body above it (`pickAbove`: the wired node, or up through
nodes with exactly one wired cad input, as Scale or Mirror) the chip shows that body and a choice changes the Pick
Body (every node after it follows); an input that gets a list of more than one body shows "all N bodies", and a
choice puts a Pick Body in front of it (one edit: an existing Pick Body of that list and body, else a new one).
Every chip asks `/api/bodies` after each run: `Session._body_infos` keeps each result's answer by its signature.
Math nodes are `numeric`: with plain numbers in they give a float64 number out (`nodes._number`), so
"spacing = 1.5 x size" can set number and int inputs. A scalar output may feed those when
`Graph.gives_number` (the editor: `graphConvert.outputType`, used to check wires) finds every scalar input
of the Math node, and of Math nodes before it, typed in or a number; otherwise `problems()` says the input
needs a plain number.
Texture (`implicit.texture`) is Offset Body for textures: pattern stretched from its `Field.range` to
0..1 (`stretch` auto), times Height, times Fade (or 1 - Fade), so a texture is Pattern → Texture instead
of Multiply, Multiply and Offset Body (`tests/test_fields.py` checks it equals that chain; the texture
presets use it). Heavy per-body nodes take
`parallel=True` plus an `estimate` (bytes per call) that limits how many run at once; a node with an
`estimate` that runs in the main process (Merge Meshes: 650 B per input triangle to fuse) makes room
first (`Evaluator._make_room`: stops the idle workers when it does not fit beside them, else a
readable GraphError). Previews drop the cached results of nodes no longer in the recipe
(`Evaluator.forget`; node ids are names recipes share) and the server's views of nodes not shown
(`Session._forget_views`); before, a closed recipe's results stayed for the life of the server.
Memory findings and the remaining fix list: docs/MEMORY_AUDIT.md.

**Server** (`engine/server.py`): the browser owns the graph and POSTs the whole recipe for every
run. `build_run_graph` applies preview quality (Draft = voxel ×4, Medium ×2: a typed voxel is scaled, a
linked one through Mesh from Implicit Body's hidden `coarser`; no decimation and Reduce Mesh's hidden `skip`
except at Final; `mesh.reduce` is one of the `QUALITY_TYPES`, so its view refines on to Final), makes no
views in exports (the editor keeps what it shows; they meshed every shown body again at 512 cells), adds
hidden `mesh.view` nodes for visible implicit bodies and slice jobs for visible plain
fields (also group nodes). Each view (`ViewSpec`) has a content key (`_content_key`: type and inputs
of every node it needs, before the quality changed them, no positions) and `exact` (nothing in it
changes with the quality: no `QUALITY_TYPES` node, not an implicit body, no own colours); a preview
gets `have` (the editor's views: key, level, version) and leaves out the views it shows with the same
key at that quality or finer (`kept`; their field colours are redone only for a changed field), so
refining only recomputes what changed. run_start echoes the editor's `tag` and lists `keep_status`
(cards that keep their state), `phase` events name the steps after the nodes (views, colours,
slices), run_done has `shown` (the coarsest quality on screen) and `sharper`; `/api/queue` reports
the `running` run, so an editor opened during an export shows it (only a run of its own tab). One editor tab
uses the engine at a time: every tab gets every event and views are kept by node id, so an old tab
reconnecting after a restart (start.bat opens a new one) showed its part in the new tab under the same node
name. A tab sends its `store.CLIENT` with every run, optimize and detail (`Session.claim`, `/api/claim`, an
`active` event to all, also on connecting) and in the run's `tag`; the store takes only its own runs' events,
and a tab another one claimed from runs nothing and shows `InactiveBanner` ("Use this tab" claims back and
makes every view anew). A freshly loaded page claims; a reconnecting one does not. Every connection first
gets `hello` with `Session.engine` (pid and start time): a tab that reconnects to another engine (a restart)
numbered runs and views from 1 again and had dropped every new view as older than its own, so it now resets
its run, sets its views' versions to 0 (keys dropped), forgets details, colours and face data, reads the
catalog again and previews anew (`api.engineRestarted`, `store.engineRestarted`). Details (`/api/detail`,
`Session.detail`, job mode "detail"): the part lo..hi of a shown body meshed finer than its view,
from the graph of the run that made the view (`view_sources`) plus a hidden `mesh.detail` node
(`tiles.mesh_tiled(close=False)`: open where it leaves the box, blocks near the surface only, the
voxel grown to `DETAIL_MAX_BLOCKS`, trimmed to the box); a body's view may go down to
`DETAIL_MIN_VOXEL`, a Mesh from Implicit Body only to its own voxel (`detail_floor`; view_ready
carries `voxel` and `floor`). A detail never stops another run (it answers busy) and any preview
stops it; its cached node_done events are not sent, so the cards keep their state. A preview keeps going past a failing node (`Evaluator.run(keep_going=True)`,
`Evaluator.failed`): only what needs it is left out, `run_done` lists those nodes (`left_out`) and the
editor drops their old views and markers (supports picked on the previous STEP file once kept a new part
from showing at all); exports and optimizations still stop at the first error. Stop (`/api/cancel`,
`Session.request_stop`) only sets the evaluator's flag and returns (a `stopping` event: the editor's
`run.stopping`, "Stopping…" until run_done); a run stops at node boundaries, `report_progress` and
`check_cancel()`, so a long step inside one node must call it (FEA solver callbacks, frame LPs and
resizing rounds, topology's result building; frame LPs are stopped from HiGHS's callback, `layout._Model`, or
without scipy's HiGHS bindings run over `layout.APART` columns in a spawned process that Stop terminates). The flag
is cleared by `Session._launch` before the
run's thread starts and when a run ends (`Evaluator.clear_stop`), never as a run begins: a stop in between, from
the next request in line (Optimize right behind a preview a changed setting started), was wiped out there, and that
request waited out the whole run. Progress streams over `/ws` (events of nodes inside a group are sent as
the group node; it reports done once, when its output is). Views go out as soon as their node is done
(`Evaluator.on_result`, `Session._publish_view`; coloured ones at the end, with their colours), and a list's items
while they are made (`Evaluator.on_item`: a node mapped over a list, or `parallel_map(parts=)` as `mesh.view` does;
`StickyPool.map(on_item=)`): `view_part` events, `/api/view/part:{i}:{node}`, dropped when the whole view comes, whose
view_ready lists each item's triangles (`parts`, from `display_arrays`); meshes, slices and surface colours are
fetched as binary from `/api/view/{id}` (`slice:{id}`, `paint:{id}`: a field sampled on a grid and
interpolated at the view's vertices, `engine/paint.py`; CAD views are not coloured). Exports queue
(`Session.queue`; never cancelled by previews, which wait in `pending`; `/api/run` `only` writes just
those Export nodes); `/api/check` runs print
checks on a shown view (`engine/checks.py`: shells and cavities by connected components, wall
thickness by Warp ray casts); `/api/autosave` keeps unsaved recipes in
`%LOCALAPPDATA%\DesignEngine\autosave` (one slot per recipe path, "untitled" for a new one). While the
recovery banner offers a slot's changes the editor does not write that slot (`Toolbar.mayAutosave`: the
first edit after opening used to overwrite it), Discard asks first, and New offers the untitled slot;
`/api/presets` stores node clips in `presets/`. Every Open / Save in the editor (`FileBrowser.useBrowser.open`)
is Windows' own dialog: `/api/file_dialog` runs `engine/native_dialog.py` as a process of its own (tkinter's
dialogs are the system's IFileDialog; shell extensions load into that process, not the engine), its hidden owner
topmost, since the engine cannot bring a window to the front; one at a time (`busy`). It starts where
`FileBrowser.startOf` says: the current file's folder, else the last folder used for those extensions, else where
the editor saves them (`recipes\`, else the project folder). When it is `unavailable` the editor shows its own
dialog (`/api/browse`), as it does with localStorage `fileDialog` = "editor".

**Frontend** (`frontend/src`): one zustand store (`store.ts`) holds the graph, groups, face
selections, visibility, run status, views, slices, paints, variants, the export queue and undo
history (snapshots include groups and selections; undo/redo keep only what is drawn for nodes still
shown, `shownOnly`); `graphConvert.ts` converts recipe JSON ↔ React
Flow (and a group ↔ its edit view with `groupIn`/`groupOut` nodes). Group specs are built in the
browser (`groupSpec`). While a group is open, `nodes`/`edges` are its inner graph and the recipe's
own are in `editing.outer`; `recipe()` composes both. Edits bump `graphVersion`, which triggers a
debounced (450 ms) preview run. Refine (`store.refine`; it and the quality are kept in localStorage):
a change previews at Draft, and when a preview is done and `graphVersion` is still its `tag.version`,
the store sets `run.next` and `App.useRefine` runs it after `REFINE_DELAY`, up to the chosen quality.
Then, while the camera rests and nothing runs, `Viewport.DetailProbe` finds each body's part on screen
(front-facing sampled triangles inside the frustum, `onScreen`) and the voxel that spans
`PX_PER_VOXEL` pixels there, and asks for a detail when that is `DETAIL_GAIN` finer than the view;
`store.details` are drawn instead of their view inside their box, both cut per pixel in the shader
(`useBoxCut`), so they meet exactly; kept while the view's key is the same and they are finer. A list's items
from a run still going on (`store.pieces`: `api.fetchPiece`, handed over every `PIECE_BATCH` ms) are drawn instead
of those items in the view shown (`base`; `hideItems` collapses their triangles in the geometry's own copy of the
index, by the view's `parts`) until the whole view comes (`setView` drops them); not over coloured views.
The toolbar's `RunStatus` shows the run kind and level, the running node (progress `step`, the hidden
view mesher, a group's inner node), the `phase`, view downloads (`loadingViews`), a change waiting
for an uninterruptible step to stop (`requested`) and, with Auto off, changes not shown yet. The
toolbar fits the window like the library's tabs (`useToolbarFit`, `data-fit` 0-4: brand and memory
figures, action button words, quality words, then the status on a row of its own) so the status
always keeps `STATUS_ROOM`; content changes refit it through a MutationObserver, not ResizeObserver. The node library (`NodeLibrary.tsx`) is a ribbon over the graph: All,
then the categories in build order (`ORDER`), groups and presets; each opens a panel of tiles (first
sentence of the doc, input → output type chips without plain settings); a click adds at the canvas
centre, a drag (`NODE_DRAG_TYPE`) where it is dropped (`NodeEditor` onDrop, `addFromLibrary`); "/"
focuses its search; tabs drop counts, then names, as the pane narrows (`data-fit`). The viewport is three.js / react-three-fiber with Z up (section
plane via `clippingPlanes`, variants drawn to the left of the result, check panel). Ball, Bar and Box
nodes (`fields.ball/bar/box`, exact distances) are drawn by the viewport itself (`Guides.tsx`; the server
runs them but adds no view, `server.DRAWN_BY_VIEWPORT`), translucent and faintly through the bodies:
Place (`store.placing`) puts a point in the middle of the wall under a click on a shown body (meshes
marked `userData.placeable`), the selected one's yellow handles drag in the view's plane.
CAD bodies (views of kind `cad`, unpainted; `components/CadBody.tsx`, also inside `PickableView`) are drawn as a CAD
program does: `useLook` loads the view's face data (`api.loadFaces`, for every shown CAD view) and `cadLook.requestLook`
has `cadLook.worker.ts` (a Web Worker, typed arrays, once per view and face data, cached in a WeakMap by view; 853k
triangles of the lamp's petals 0.36 s) split the vertices where the CAD face changes (smooth, angle-weighted normals
within a face, shared across tangent borders) and find the lines: sharp face borders, open edges, and tangent borders (two
faces whose triangle pairs along the border mostly bend under 30°, as `facemask.tangent_groups`; two flat faces only when
coplanar). Without face data (a decimated view, a piece or detail, an error) the edges are where the surface bends over
35°; views over `LOOK_MAX_TRIANGLES` (3M) are drawn plainly. Triangles keep their order (the geometry's
`userData.source` is its whole index: `hideRanges`, `faceSubset`), so picking and menus work unchanged. Lines are
`LineSegments` with a patched `LineBasicMaterial`: pulled `EDGE_PULL` of their distance towards the camera (no z-fighting,
over face highlights), clipped by the section plane, box-cut (`viewCut.ts`: `useBoxCut`, `hideRanges`, `CAP_COLOR`), and on
opaque bodies hidden where both triangles beside a line face away (else a section showed them on the cut face). Options
(`useCadLook`, localStorage `cadLook`): the toolbar's Edges (also E and the View menu) ▾ `CadLookPanel`: Shaded,
See-through, Lines (hidden lines removed), Wire; tangent edges shown / dimmed / hidden; colour by faces or bodies
(`tintColors`); colour swatches. Shaded bodies get a `RoomEnvironment` reflection (`useCadEnvironment`, CAD only).
Right-click menus in the 3D view (plan and status in docs/VIEWPORT_MENU_PLAN.md): a DOM `contextmenu`
on the canvas (only when the right button moved ≤ 5 px, since right-drag pans) raycasts the meshes
tagged `userData.menu = {view}` itself (`MenuProbe`; not R3F events, so a left-drag never raycasts
big bodies), building their search tree on the first right-click (`ensureBoundsTree`, always
`indirect: true`: the default build reorders triangles and breaks CAD face numbers), skipping hits
the section plane cuts away. `viewActions.ts` builds the entries (`faceMenu`, `bodyMenu`,
`pickingMenu`, `markerMenu`) from the store alone; every action is one `store.applyEdit(GraphEdit)`:
one undo step, nodes placed next to their anchor without overlapping (`freeSpot`), changed nodes
flash in the graph (`flash`), a toast lists the changes with Undo (`toast`, `ViewportMenu.tsx`).
Face data per CAD view is cached in `store.faceData` (`api.loadFaces`, by view version) for picking
and menus; `api.loadFaceInfo` (`/api/faceinfo`, `facemask.face_info`) gives a face's surface type
and size. New supports and loads link to the source the body's face nodes already use (a Pick Body
of that body before Import STEP; a new Pick Body in the same edit when other bodies of the list have
one) and are appended to its supports-and-loads chain (`add_to` from the tail, the tail's consumers
relinked to the new node); deleting one from its marker splices the chain back. Support and load
arrows carry invisible hit balls (`userData.menu.marker`; markers list the CAD face and body of
each point, `faces` / `bodies`). After adding a load, `store.quickEdit` opens `QuickEdit.tsx` at the
cursor (inputs per node type in `QUICK_INPUTS`; Shift on the menu item skips it). On an implicit
body or mesh the face menu acts on the CAD face under the click: `/api/face_at` walks the last run's
graph (`Session.view_sources`, `cad_sources`) up to the first `cad` outputs and finds the nearest
face (`facemask.nearest_face`, bodies ranked by box distance, trees in an LRU of ~2M triangles).
The menu's face is highlighted while it is open: on a CAD view from `faceData`, on other views by
`/api/face_region` (`facemask.face_region`: the shown triangles, and a detail's, that lie straightest
off that face, `_straightest`, so an offset's rounded edges split down the middle).
Slices (`userData.menu.slice`) and frame bars (`userData.menu.bar`, instance = bar piece) have their
own menus. A still left-click (≤ 4 px, no modifier, not picking or placing; objects with their own
R3F click handlers, like the load arrow's handles and guide shapes, keep their clicks) selects,
flashes and pans to the node that made what is under the cursor: a marker's, slice's or bar's node,
and for a body `/api/node_at` (`engine/creator.py`: walks up from what the view shows, following
`implicit` inputs whose surface also passes through the point, within max(`tol`, 1.5 × the node's
own value there); where none does, that node made the surface: an Offset Body's bump, a Hole
Pattern's hole wall, the CAD node for a plain face). `tol` is a quarter of the clicked triangle's
longest edge: the whole edge (a Draft edge is as long as a 1 mm bump is high) sent every stud click
to the CAD node. The axis gizmo's corner (`GIZMO_MARGIN`, its own scene) never selects. A double-click
opens `QuickEdit` for that node (any node: `QUICK_INPUTS`, else all plain inputs; several nodes' inputs
with `fields`; a linked input links to its source node). Changes to a shown body (*Texture ▸* on a face:
Noise / Wave → Texture with a Face Mask as Fade, or a Surface Pattern style (`SHAPE_TEXTURES`, the
presets' settings) → Texture, in one edit; *Holes ▸*: a Hole Pattern (`HOLES`, the vent presets' settings)
into Boolean Subtract, or Union for raised shapes; *Text or picture ▸*: Text / Picture → Texture, a picture
asking for its file first; all three through `faceMenu.changeItem` and a `BuildChange`; *Modify ▸*: Offset,
Shell, Smoothen, lattice fill, cut / add a Ball, Box or Bar at the click) go in through `viewActions.bodySlot` / `modifyEdit`:
after an implicit view's output (its users relinked, the result shown instead), before a Mesh from
Implicit Body, or after a new Implicit Body from CAD Body of a CAD view. Ctrl+click chooses faces
(`store.faceSel`, highlighted; `/api/similar_faces` → `facemask.similar_faces` for *Similar faces*); a
right-click on a chosen face acts on all of them (`FaceHit.targets`). *Again: …* repeats the last face
action on another face; typing in any `ViewportMenu` searches all its items. Measure (M,
`/api/thickness`: a Warp ray straight in; `/api/section`, `checks.member_section`: the member's cross-section, the
smallest closed cut through the point halfway in, cut loops linked by mesh edge, `SectionMarks`), standard views (Space, 1-7; `store.camera` → `CameraTurn`, Z
stays up), *Section here*, the section plane's drag ball (`SectionGizmo`), Sliders for Number nodes and
group settings (`SettingsPanel.tsx`), variant menus (`store.restoreVariant`). Plan and status:
docs/VIEWPORT_MENU_PLAN.md (M4). Node editor: right-click a node (`nodeMenu`: bypass B, collapse H,
rename, duplicate, reset, select up/downstream, tidy, group, delete and reconnect = `store.deleteNodes`)
or a wire (insert a node; also double-click), Ctrl+F finds, Tidy lays out (`tidyPositions`); `bypass` and
`collapsed` are recipe node keys (`graphConvert`). The graph's View menu (`wires.ts` options, kept in localStorage):
zoomed out below `FAR_ZOOM`, simple cards (NodeEditor sets `data-far` and `--zoom` on `.editor` from React Flow's
store subscription, so zooming re-renders nothing; CSS covers each card with its `FarFace`, name sized by `--zoom` and
cq units, the card keeping its size so nothing moves; only `.wired` rows keep their dots; frame labels grow above
their box). Wires: the selected nodes' wires stand out and the rest fade (lightly while only hovering, after
`HOVER_WAIT`); wires of an output with `SHARED_WIRES` or more and those longer than `LONG_WIRE` are drawn only
while an end is selected or hovered (dashed), else as labels on the inputs they feed (`wireLabels`, memoized on
the nodes and edges arrays; a click goes to the source). React Flow draws `drawnEdges` (copies: the store's edges stay as
they are). A wire's input end moves: edges are `reconnectable: 'target'` (`makeEdge`); pressing a wired input's dot
starts React Flow's reconnect from the wire's end anchor (`wires.pickUpWire`: a mousedown dispatched on
`.react-flow__edgeupdater-target`, the dot's own pointerdown stopped; a labelled wire is drawn first), so it can
be picked up from the dot or the wire's end. `NodeEditor.onReconnect` moves it (`store.reconnectEdge`, one undo
step); `onReconnectEnd` (React Flow's `toHandle` is the nearest port, valid or not): an output there feeds the
input instead, empty canvas removes the wire, anything else (a card, an unfitting input, outside the graph) keeps it. Frames in the graph (`graphFrames.ts`, `GraphFrameNode.tsx`; recipe key `frames`, store `graphFrames`,
in undo snapshots; not the bar frames of Frame Optimization, store `frames`): labelled boxes made round the
selection (F, node menu, selection bar). Membership is by position (a node's or smaller frame's middle inside), so
there are no member lists to keep up to date except a folded frame's `folded` (what it hid when folded; `hiddenBy`
maps hidden ids to the outermost folded bar, `foldedEdges` ends their wires on it). They are React Flow nodes only
in `NodeEditor` (ids `frame:N`, zIndex below the wires, `measured` given: without it React Flow drops their handle
bounds), never in `s.nodes`; `onFlowNodesChange` sends their changes to `store.frameChanges`, which moves what is
in a dragged frame along (except what is selected: React Flow moves that). Only the title bar and resize border take
the pointer. Tidy with nothing selected lays out frames as blocks (`tidyFrames`, each frame's contents first, fitted);
a frame's menu tidies its own. A folded frame takes no Delete key (React Flow would delete the wires on its bar).
Not drawn while a group is open. Analysis wizard (`AnalysisWizard.tsx`, the 3D view's
Analyse button or a body's *Analyse ▸ Wizard ▸*): a tool panel for a goal (`GOALS`: Topology
Optimization or Frame Optimization, run with Optimize; Static Analysis, run by showing it) with steps
Part (a CAD output; a body of a list gets a Pick Body), Held, Loads, Settings (the goal's few inputs),
Optimize / Analyse (progress, Stop, the node's note and warnings, Colour by). Its first step adds Fixed
Faces, Material and the goal's node in one edit (one Undo); the steps pick faces into those real nodes
with the normal picking (`store.picking`, its banner hidden while the wizard says what to pick), loads
are a chain after the supports; its state (`useWizard`) is in memory only and resets on another recipe
or when its nodes are gone. Static Analysis's card note has its safety factor, deflection, stress and mass
(`nodes_fea.judge`, shared with the optimizers' checks). Texture wizard (`TextureWizard.tsx`, the 3D view's
Texture button or a face's *Texture ▸ Wizard…*, which starts on those faces at the Texture step): Faces (a
Face Mask whose faces are a new saved selection `texture_faces_N`, so a Surface Pattern shares them and
picking updates both), Texture (the face menu's kinds, `viewActions.TEXTURES` / `SHAPE_TEXTURES`; the first
Next adds pattern → Texture into the shown body as *Texture ▸* does, `bodySlot` / `modifyEdit`, or after a new
Implicit Body when only the CAD body is shown; another kind replaces the pattern node), Masks (the mask's
edge fade and Apply to; Multiply nodes into the Texture's Fade: a Face Mask "all other faces" to leave faces
out, a Smooth Step of a Coordinate to fade along an axis), More (Smoothen / Shell after the texture, another
texture on top of the result, Mesh and export). It keeps only the mask, the Texture and which kind each
pattern node it made is; the pattern, the Fade chain and the steps after are read from the graph
(`patternOf`, `factorsOf`, `afterOf`), so Undo and edits in the graph keep it right. After picking, the CAD
body goes out of sight again while the textured one is shown (`settle`). Lattice wizard (`LatticeWizard.tsx`, the
Lattice button or a body's *Modify ▸ Fill with lattice ▸ Wizard…*): Part, Lattice (`LATTICES`: TPMS, struts,
Voronoi with the CAD body as its domain; always Units density, so Thickness is the solid share; Fill Body with
Lattice with a 1 mm skin; the section plane is turned on, `store.requestViewTool`; `Amount` gives the volume of
the shown view, `meshVolume`, against the part's before), Solid, Grading, Check (the print check panel, the same
request). Density changes are a chain into the lattice's Thickness: per change a Max (A: what came before, the
first holds the base share) of a Remap 0..1 → 0..To High of a factor: a Face Mask with Reach volume (To High 2:
solid for the first half of its Fade), a Smooth Step of a Coordinate, or of a Static Analysis' safety factor
(densest under 2; not its stress, whose range has the support peak); open faces are a Face Mask "all other
faces" times the skin in the Fill's Skin. All read back from the graph (`modsOf`, `openOf`). The wizards share
`wizardParts.tsx` (finding the part and the body a change goes into, `Setting` rows) and `wizards.ts` (opening
one closes the others). The toast of an edit that also shows a node keeps its Undo
(`applyEdit` re-stamps its version after `toggleVisible`).

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
- Raise `frame.geometry.GEOMETRY` with every change to the frame kernel, `_curves`, `skin_body` or
  `_cylinder_distance` (and `FRAME_VERSION` with every change to the layout): field values and Static
  Analysis's displacements are kept on the SSD by key, and a change that kept the key loaded another shape's
  displacements (11000 MPa in a boss; the editor showed safety factor 0.047 for a frame at 1.16).
- A `cKDTree` of points sampled on flat faces, queried with `distance_upper_bound`: build it with
  `compact_nodes=False` (`topology._face_tree`). The default's flat boxes made one query 25 s instead
  of 1.5 s (same distances).
- A step inside a node that can run for more than a few seconds must call `check_cancel()` (or take
  a `cancel` callable) between its parts: Stop only takes effect there.
- Measure memory with a kill switch (psutil, whole process tree) before claiming a memory fix;
  an uncapped run once crashed the user's PC. The machine has 32 GB, other programs use ~15-21 GB.
  Since 2026-10-24 the page file is on the NVMe SSD: running out of RAM no longer froze the PC in a
  13-minute thrashing test, but a program that does not fit crawls (31.5 GB read back in 4-9 min;
  FINDINGS, "The PC freeze"). Since 2026-10-31 the guards watch commit, not free RAM: other programs
  commit ~53 GB of the 77.8 GB, and out of commit allocations fail in any program (SolidWorks too).
  Every process that loads numpy and scipy commits ~23 MB per BLAS thread for each (1.5 GB at 32
  threads, never touched): worker processes start with `OPENBLAS_NUM_THREADS` = their share of the
  cores (`StickyPool._start`); a new kind of child process should too.
- Before running tests or anything heavy, check that the user's own editor (port 8765) is not
  busy. A full test run (then including GPU tests) on top of an export froze the PC
  (2026-10-12); GPU tests now need `--gpu`. The engine's thresholds: commit room `cache.COMMIT_TIGHT` /
  `COMMIT_STOP` / `COMMIT_LAST_RESORT` 6 / 3 / 1.5 GB (free caches and stop idle workers / stop the run /
  end the process 30 s after a stop); free RAM `core.FREE_TIGHT` 1.5 GB only frees caches, and a run stops
  when it stays under `FREE_STOP` 0.25 GB for `STARVED_SECONDS` (30). An idle editor releases workers and
  cached meshes when the PC runs low on either (`Session.watch_idle_memory`).
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
- No single GPU kernel may run for seconds: Windows then resets the display driver (TDR), and every process's CUDA
  context with it. cupyx's labelled measurements grow with labels x samples (`ndimage.maximum`: 95 ms for 4M
  samples and 4k labels); use one scatter pass (`topology._label_max`, `cupyx.scatter_max`). Time a new CuPy or
  Warp step on a grid an eighth of the real size first.
- Two threads loading one Warp module on the GPU at once fail with CUDA error 700, which spoils the process's CUDA
  context for good (every later allocation fails). The server answers requests in threads: Measure's wall and section
  requests did this, so `checks._cast` runs one ray cast at a time (`_CAST_LOCK`). New Warp code reached from request
  threads needs the same.
- A "children" level has 8, 4 or 2 children per element (`gpu.rows`'s `kids`): a sheet or bar is not halved along every
  axis. Test GPU kernels on such grids too, not only on bulky ones.
- On slender parts and sheets a float32 multigrid gives up beside soft elements: `Optimizer._solve` hands over to float64
  for the rest of the run (`wide` in the state) and takes float64 within `SOLVE_NEAR` x the tolerance; `solve_cases` retries.
- FEA on the GPU: keep the conjugate gradients and the fine-level product in float64 (only the
  multigrid in float32), and check a change against the CPU (`tests/test_gpu.py --gpu`): a
  float32 solver can report convergence and still be 15% off. Scalars in the multigrid must be
  Python floats: a NumPy float64 scalar (lmax was one) silently made the float32 multigrid float64
  (NumPy 2, CuPy 14), 2.5x slower on the GPU; `tests/test_fea.py` checks the preconditioner's type. Fully fixed faces make artificial
  stress peaks that grow with finer detail; don't read them as the part's strength.
- Write TypeScript with the Edit/Write tools, not Python string patches: escapes (`\n`, `\\`)
  were silently changed twice.
- Profile with the user's editor stopped, or say that it ran: an idle editor held 10.6 GB of RAM
  and ~6 GB of the 12 GB GPU (worker CUDA contexts, cached search structures), and with 2.6 GB of
  GPU memory left Windows moved allocations to shared memory: GPU timings came out ~10x slow.
  Ask before stopping it; check `/api/memory` (gpu_free) first.
- Testing the editor: use the "design-engine-test" launch config (port 8766), never restart the
  user's editor on 8765 (it may hold unsaved changes). The test editor shares the autosave folder
  and `presets/`: delete what a test created. Autosaves are keyed by recipe path, so editing a
  recipe in the test editor overwrites the user's autosave of it: copy that file first and put it
  back after (never click Discard on the restore banner). The test tab reopens its own
  `lastRecipe` at start-up (once the user's hex_studs, whose autosave was then lost): back up the
  whole autosave folder before starting it, and open a test recipe by typing its full path in
  Open… and pressing Enter (the test tab has localStorage `fileDialog` = "editor", so Open… shows the
  editor's own dialog: without it Windows' dialog opens on the user's desktop, where only a script finding
  the "#32770" window by its title can answer it; native confirm dialogs are auto-cancelled in the browser pane: with
  unsaved changes Open… does nothing until `window.confirm = () => true` in the test tab; the path
  field is pre-filled with the last folder, so select all before typing). Copying the backup back
  changes the autosaves' file dates: restore those too.
