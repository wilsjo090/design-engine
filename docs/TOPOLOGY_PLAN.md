# FEA and topology optimization: plan

Status 2026-10-14: phases 1 (static analysis) and 2 (topology optimization, with the Optimize
button: the user's choice) built and tested; measurements and what changed from this plan in
FINDINGS.md, "Static analysis" and "Topology optimization". Phase 3 is open. Decisions still
marked **(open)** are the user's.

Changes from the plan in phase 2: the stopping test is the compliance settling (no density
changing by < 0.01 never happened); inner solves to 1e-2; the result body is the density's
smoothed 0.5 surface as an exact distance field (no separate connectivity filter beyond dropping
pieces under 1% of the largest); `compute` / `stamp` hidden inputs as planned; the live shape is
the raw 0.5 surface every 3 iterations.

Changes from the plan in phase 1: "Show stress" is the node's own eye plus a *Show* input
(stress / safety factor / displacement); supports and loads are drawn as cones and arrows at a
few points of their faces (over the part), not as hatching; the GPU runs CG in float64 with a
float32 multigrid (all-float32 was wrong); the report and colours leave out the artificial stress
peak at fully fixed faces.

Goal: find where a part is weak (static analysis) and let the engine remove material where it
does no work (topology optimization), as easily as the texture tools: pick faces, type a force,
press a button, watch the shape grow. Everything is computed on the voxel grids the engine
already uses, so the results are ordinary fields and bodies: they can be smoothed, textured,
filled with lattices, meshed and exported like anything else.

## What the user does in the editor

1. Model the **design space** in SolidWorks: the largest envelope the part may fill, with its
   bolt holes and mounting faces. Import it as usual (Import STEP, Pick Body).
2. **Material** node: choose PLA, PETG, ... from a list.
3. **Fixed Faces** node: click Pick, click the faces that are bolted or glued.
4. **Load** node: Pick the faces the force acts on, type the force in N, choose the direction.
   Arrows on those faces in the 3D view show what was set; fixed faces are drawn hatched.
5. Either
   - **Static Analysis**: the part is coloured by stress (or safety factor, or how far it
     moves); hovering shows the value; the log gives max stress, min safety factor, max
     deflection and mass. Runs by itself like the other nodes (seconds).
   - **Topology Optimization**: set *Keep %* (share of the mass to keep) and *Min thickness*,
     press **Optimize**. The shape grows live in the 3D view with a progress bar and a log line
     every few iterations ("iteration 40/110 · 30% mass · 81% of the full block's stiffness").
     **Stop** keeps the shape so far; Optimize again continues from there. The result is a body:
     Smoothen Body, a texture, Mesh from Implicit Body, Export Mesh as usual.
6. Check the result: connect the optimized body to a Static Analysis node.

Two presets (`presets/`) give the whole chain in one drag: "Static analysis" and "Topology".

```
Import STEP > Pick Body ──┬──> Fixed Faces (bolt holes) > Load (500 N, -Z, top face) ──┐
                          └──> design space ───────────────────────────────────────────┤
Material (PETG) ───────────────────────────────────────────────────────────────────────┴─> Topology Optimization
     > Smoothen Body > Mesh from Implicit Body > Export Mesh
```

## Nodes

Units everywhere: mm, N, MPa (N/mm²), g/cm³. Category "Simulation" in the palette.

| Node | Inputs (default) | Outputs |
|---|---|---|
| **Material** | preset: PLA / PETG / ABS / Nylon PA12 / Aluminium 6061 / Steel S355 | material |
| **Custom Material** | stiffness E (MPa), Poisson's ratio, density (g/cm³), strength (MPa) | material |
| **Fixed Faces** | Body (cad), faces (picked), add to (conditions, optional) | conditions |
| **Load** | Body (cad), faces, kind: force / pressure, amount (N or MPa), direction: into the face / out of the face / +X / -X / +Y / -Y / +Z / -Z / custom, vector ("0, 0, -1", custom only), add to (optional) | conditions |
| **Static Analysis** | part (CAD body or implicit body), material, conditions, detail: draft / medium / fine | stress (MPa), safety factor, displacement (mm), body (the part, to show it coloured) |
| **Topology Optimization** | design space (CAD or implicit), material, conditions, keep % (30), min thickness (mm, 0 = auto: 3 elements), skin (2 mm kept solid at fixed and loaded faces), keep (body, optional), avoid (body, optional), detail | body, density (0..1) |

- Conditions chain: Fixed Faces > Load > Load ... (each adds itself to what comes in on *add to*),
  so one link carries all of them; a node input takes one link.
- *add to* chaining is used instead of a list on purpose: a `Many` list would make the evaluator
  run the analysis once per item.
- One body per analysis in the first version (a `Many` list gives the error "pick one body with
  Pick Body"). Assemblies and contact between bodies are out of scope.
- Faces work like Face Mask's: numbers, `body:face`, or saved selection names, picked on the
  body connected to *Body*.
- Material values (typical datasheet values, to check against the user's filaments **(open)**):

  | preset | E (MPa) | Poisson | density | strength (MPa) |
  |---|---|---|---|---|
  | PLA | 3500 | 0.36 | 1.24 | 50 |
  | PETG | 2000 | 0.38 | 1.27 | 45 |
  | ABS | 2200 | 0.35 | 1.04 | 40 |
  | Nylon PA12 | 1700 | 0.39 | 1.01 | 48 |
  | Aluminium 6061-T6 | 69000 | 0.33 | 2.70 | 276 |
  | Steel S355 | 210000 | 0.30 | 7.85 | 355 |

  Printed parts are weaker between layers; the safety factor's help text says so.

## Phases

### Phase 1: static analysis (the FE core)

1. Solver core in `engine/fea/` (see Technical design) with its tests (see Tests).
2. Nodes Material, Custom Material, Fixed Faces, Load, Static Analysis; result fields; log report.
3. Editor: new port types and colours, load arrows and support markers in the 3D view, unit (MPa,
   mm) on the colour legend, a "Show stress" button on Static Analysis (makes its body and stress
   visible in Surface colour mode), preset "Static analysis".
4. Measure time and memory for the three detail levels on a real part, editor stopped (or say
   it ran); write them into FINDINGS.md; update CLAUDE.md's architecture section.

Done when: the cantilever test matches the textbook to a few %, a test recipe
(`recipes/fea_cantilever.json`) shows the expected colours, and a real bracket runs at Medium in
seconds.

### Phase 2: topology optimization

1. SIMP loop with density filter, projection, OC update; keep / avoid / skin regions;
   checkpoints and resume; result store on the SSD.
2. Density to body: iso-surface, smoothing, loose bits removed, exact distance field, trimmed to
   the design space.
3. Server and editor: Optimize / Stop on the node, live shape, progress, stiffness report;
   exports run the optimization when it is missing; preset "Topology".
4. Tests (see Tests), then tune the defaults (move limit, projection schedule, iterations, void
   stiffness) on 2-3 real parts.

Done when: a bracket optimizes at Medium in a few minutes with the shape growing live, the
exported STL reloads clean (CLAUDE.md rule), and Static Analysis of the result looks sensible.

### Phase 3: extras (by need)

Symmetry planes; several load cases (stiff for each); sliding supports, bearing loads, torque,
gravity; a stress-graded lattice preset (Static Analysis stress > Remap > lattice thickness: a
"simulation-driven lattice"); deformed-shape view (exaggerated); overhang limit for printing; a
guided side panel (1 Design space, 2 Fixed, 3 Loads, 4 Goal, Run) over the same nodes **(open)**;
a Warp kernel for the stiffness product if CuPy is the bottleneck.

## Technical design

### The FE grid

- Hexahedral elements on a regular grid, element size `h` chosen from *detail* and the design
  space's volume (targets: draft ~40k, medium ~250k, fine ~1M elements inside the part), rounded
  to a tidy value. The grid covers the part's bounds plus one element, padded to a multiple of
  2^L elements per axis for multigrid.
- Element fill `phi` (0..1) from the part's signed distance, sampled 2x2x2 per element (the
  normal field evaluation, so CAD bodies use the GPU distance fields):
  `phi = mean(clamp(0.5 - d / (h/2), 0, 1))`. Elements with `phi < 0.01` are inactive; nodes
  touching no active element are taken out of the system (treated like fixed nodes).
- A partly filled element has stiffness `phi * E`, which keeps curved surfaces from being stair
  steps in stiffness.
- *detail* is not scaled by the preview quality (Draft/Medium/Final): what is set is what is
  computed, and results are cached.

### Element and units

Trilinear 8-node hexahedron, 2x2x2 Gauss integration, unit-E stiffness `KE` (24x24) computed in
code for any Poisson's ratio; element stiffness `E_e * h * KE` (in 3D a cube's stiffness scales
with its size). Fully integrated hex8 is too stiff in bending with fewer than ~3 elements
through a member: *min thickness* (auto = 3 elements) prevents that in optimized shapes, and the
analysis log warns when the part has walls thinner than 3 elements at this detail.

### Supports and loads from CAD faces

- Points are sampled on the chosen CAD triangles about `h/3` apart, each carrying its share of
  the area (the triangles' `cad_face` attribute, as in `facemask`).
- Fixed: every active node of each element containing a fixed-face sample is fixed in x, y and z
  (clamps a one-element layer). The solver masks each direction separately, so sliding supports
  (phase 3) need no solver change.
- Load: each sample's force (area x traction; force: traction = F / total area x direction;
  pressure: p x normal) is spread to the 8 corners of its element with trilinear weights over the
  active nodes. The total equals the force typed in (tested).
- Direction "into/out of the face" uses each triangle's normal; custom vectors are normalised.
- For a modified part (textured, lattice-filled) the samples still come from the original CAD
  faces: elements within ~1 element of them count. Large offsets can miss; the node reports
  "no part material found at the fixed faces" instead of solving a free-floating part.
- Checks before solving: at least one support; at least one load; a part that is not held at all
  gives a clear error (the solver would not converge).

### Solver

- Matrix-free: `K u = sum_e E_e (h KE) u_e`. On a regular grid gathering an element's 24 values is
  8 array slices (no index arrays); the product is one `(n, 24) @ (24, 24)` and a scatter of 8
  slice additions. Done in slabs along x to cap temporary memory (~2 x n x 24 floats per slab).
- Written against `xp` (numpy or CuPy) like the field code. GPU: float32 (the 3080 Ti does
  float64 at 1/64 speed), dot products summed in float64. CPU: float64.
- Preconditioned conjugate gradients with one geometric multigrid V-cycle as preconditioner
  (MGCG, as in Amir et al. 2014): levels by factor 2, trilinear prolongation `P`, Galerkin coarse
  operators. A coarse element's matrix is exactly `sum over its 8 children of E_child * M_c` with
  8 constant 24x24 matrices `M_c = P_c^T KE P_c`, so level 1 stays matrix-free at the cost of one
  fine product; levels 2 and deeper store their element matrices (fine: ~100 MB at level 2); the
  coarsest level (a few thousand unknowns) is assembled and factorized with scipy on the CPU once
  per design iteration. Damped Jacobi smoothing (diagonals scattered the same way), 2 sweeps before
  and after.
- Fixed and inactive nodes are masked on the fine level. On coarse levels a node is fixed if any
  fine node in its prolongation stencil is fixed (fixing only coinciding nodes can leave a coarse
  level with no support, i.e. singular). The V-cycle only has to be symmetric positive definite:
  the answer is always that of the exact fine operator.
- Stopping: relative residual 1e-6 for analysis, 1e-4 inside the optimization (inexact solves
  are fine there), warm-started from the previous displacement.
- `core.check_cancel()` every few CG iterations; `core.report_progress` for the node's bar.

### Results as fields

- Stress per element at its centre, `sigma = D(E, nu) B(0) u_e` with the material's full E (the
  stress in the material, also in partly filled elements); von Mises; averaged to nodes over the
  active elements weighted by `phi`. Displacement is nodal. Element centres lie half an element
  inside the surface, so surface stress comes out low by about that much (10% on the cantilever
  at 1 mm): if the test shows it matters, extrapolate to surface nodes (patch recovery).
- A result field interpolates the nodal array trilinearly at the evaluation points: a pure
  function of position, so block meshing, slices and surface colours work unchanged (values
  outside the FE grid: 0). `range` set from the computed min/max; `extent` = the part's box.
- Safety factor = strength / von Mises, clamped to 0..10 for display.
- Log report: max von Mises (and where), min safety factor, max displacement, mass, and the sum
  of reaction forces against the applied load (an equilibrium check, should match to 1e-6).
- Voxel stresses near curved or sharp edges are approximate (peaks at fillets come out low at
  coarse detail). The help text says so and suggests comparing Medium with Fine.
- Results cached in the SSD cache (`cache\fea`, keyed by fingerprints of the part field, the
  conditions, the material and the detail; `FEA_VERSION` to bump when the numerics change).

### Topology optimization

- Design variables `x_e` for active design-space elements, upper bound `phi_e` (partly filled
  boundary elements), start at keep %. Passive solid: elements within *skin* mm of fixed and
  loaded faces, and inside *keep*. Passive void: inside *avoid*.
- SIMP: `E_e = Emin + xbar_e^3 (E - Emin)`, `Emin` = 1e-6 E to start (1e-9 is the textbook value;
  choose by the multigrid convergence tests).
- Density filter for *min thickness*: normalised convolution with a cone kernel of radius
  `r = min thickness / 2` (`cupyx.scipy.ndimage` / `scipy.ndimage`), divided by the convolution of
  the design mask so the boundary is not thinned. Sensitivities go through the transpose (same
  convolution of `g / N`).
- Heaviside projection (tanh, threshold 0.5), beta 1 > 2 > 4 > 8 > 16 every ~20 iterations or on
  convergence, so the result is nearly 0/1 and the walls are definite.
- Optimality-criteria update with move limit 0.2 (0.1 once beta >= 4), bisection on the
  Lagrange multiplier for the volume (keep %) constraint on the projected densities.
- Stops at max change < 0.01 at the last beta, or ~110 iterations.
- Reported: mass (g), compliance, and stiffness relative to the full design space (one extra
  solve at the start: "81% of the full block's stiffness at 30% of its mass").
- Checkpoint (x, beta, iteration, displacement) every 10 iterations and on Stop, in the result
  store; Optimize with the same inputs resumes from it. The result is the same as an
  uninterrupted run (tested to float tolerance), so caching stays honest.

### Density to body

The projected density on the FE grid > trilinear field > marching cubes at 0.5 on a grid 2x finer
> a few Taubin smoothing passes > remove pieces not connected to the skin / keep regions (loose
bits) > `fields.body_from_mesh` (exact distance field, SSD-cached, GPU, block meshing all work as
for CAD bodies) > intersect with the design space (outer faces stay exactly the CAD faces) > union
with *keep*. The `density` output is the filtered density field itself (for lattice grading).

### Running it

- Static Analysis runs inside normal preview runs (seconds; cancelled by edits like any node).
- Topology Optimization only computes on **Optimize** (or in an export when missing); in previews
  it looks up the result store. Not stored for these inputs: it outputs the design space and the
  node says "not optimized for these settings: press Optimize". Recommended over automatic
  re-runs on every edit, which would restart a minutes-long GPU job each time **(open)**.
- Mechanics: `build_run_graph` injects two hidden inputs (Port `hidden=True`, not drawn by the
  editor): `compute` (true for Optimize and export runs) and `stamp` (bumped by the server when an
  optimization of that node finishes or stops, so the node's signature changes and the evaluator
  re-reads the store and recomputes everything downstream).
- `POST /api/optimize {recipe, node}` queues a run like an export: never cancelled by previews,
  previews wait in `pending`. **Stop** cancels it; the loop saves its checkpoint before the
  `Cancelled` goes up, and the following preview shows the shape so far.
- Live shape: a new `core.report_view(mesh)` emits a `live_view` event; `Session._on_event` packs
  it into `views[node]` and publishes `view_ready` (through the group-owner mapping), every ~5
  iterations (marching cubes on the FE grid, ~50 ms).
- Runs in the server process (one body, nothing to parallelise), under the evaluator's memory
  watchdog.

### Memory and GPU

- Before solving: estimate GPU bytes (vectors ~12 x 3 x nodes x 4 B, slab temporaries, level-2
  matrices, element arrays) and compare with `backend.gpu_memory()` free minus `GPU_HEADROOM`.
  Not enough: say so, suggest a lower detail, or run on the CPU (draft and medium only). Out of
  memory during the run: fall back to the CPU, as distance fields do.
- An idle editor's workers hold ~6 GB of the 12 GB card (FINDINGS): releasing idle workers' GPU
  memory before an optimization (already a FINDINGS candidate) may be needed.
- RAM on the CPU path is about twice the GPU figure (float64); the watchdog covers it.

Targets, to be measured in phase 1/2 (estimates, editor stopped):

| detail | elements in part | analysis | optimization (~110 it.) | GPU memory |
|---|---|---|---|---|
| draft | ~40k | ~1 s | ~20-40 s | < 0.3 GB |
| medium | ~250k | ~3-5 s | ~1-3 min | < 0.6 GB |
| fine | ~1M | ~10-20 s | ~5-10 min | ~1-1.5 GB |

### Editor changes

- `types.ts`: colours for `material`, `conditions`, `part`; skip ports with `hidden`.
- Node card: Optimize / Stop button and the iteration line on Topology Optimization (like
  FacesInput's Pick, by type id); "Show stress" on Static Analysis.
- `Viewport.tsx`: arrows for Load nodes (at each chosen face's area centre, along the direction,
  length fixed on screen) and hatched fixed faces, drawn while the node is selected or its body
  is visible; uses the per-triangle faces the picker already fetches (`/api/faces`).
- Colour legend: unit from the field (`Field.unit`, sent with `paint_ready` / `slice_ready`).

### Files

| file | what |
|---|---|
| `engine/fea/materials.py` | presets |
| `engine/fea/element.py` | hex8 `KE`, `B(0)`, `D` |
| `engine/fea/model.py` | FE grid, `phi`, active masks, supports and loads from CAD faces |
| `engine/fea/solver.py` | matrix-free operator, multigrid, PCG (numpy / CuPy) |
| `engine/fea/analysis.py` | static analysis, stresses, result fields, report |
| `engine/fea/topology.py` | SIMP loop, filter, projection, OC, checkpoints, density to body |
| `engine/graph/nodes_fea.py` | the six nodes (imported by `nodes.py`) |
| `engine/graph/core.py` | port types `material`, `conditions`, `part`; `Port.hidden`; `report_view` |
| `engine/server.py` | `/api/optimize`, hidden inputs, live views, stamps |
| `engine/cache.py` | `fea` folder, `FEA_VERSION` |
| `frontend/src/...` | see Editor changes |
| `tests/test_fea.py`, `tests/test_topology.py`, `tests/test_gpu.py` | see Tests |
| `recipes/fea_cantilever.json`, `presets/` | test recipe, presets |

## Tests

- **Operator**: the matrix-free product equals an assembled scipy sparse matrix on a tiny grid
  with random densities and a mask (1e-12 on CPU).
- **Patch test**: a block in uniform tension, held by sliding supports set directly in the solver
  (per-direction masks; a clamped end would restrain the Poisson contraction), gives uniform
  stress F/A and stretch FL/(EA), exact for hex8 at any element size.
- **Cantilever**: 100 x 10 x 10 mm, fixed end, tip load, 1 mm elements: tip deflection within
  ~3-5% of beam theory (FL³/3EI plus the shear term); element-centre bending stress at mid-length
  within ~5% of M·y/I at the same height.
  The geometry is written as a STEP with OCP, as `tests/test_facemask.py` does.
- **Equilibrium**: reaction forces sum to the applied load.
- **Multigrid**: converges to 1e-6 in a bounded number of iterations on a 64³ grid with a 1:1e6
  stiffness contrast; the coarse levels are never singular (support only on odd nodes).
- **Loads**: total applied force equals the input for force and pressure, also on curved faces.
- **Topology**: volume within 0.5% of keep %; same inputs give the same result; stop + resume
  equals an uninterrupted run; keep / avoid / skin respected; the classic 3D cantilever gives
  the expected two-flange truss shape (compliance against a stored value).
- **Blocks**: the result fields and the topology body give bit-identical values block by block
  (`tests/test_tiles.py` style).
- **GPU** (`--gpu` only): GPU results equal the CPU within float32 tolerance.
- Run the suite only with the user's editor idle (CLAUDE.md).

## Risks

- Voxel stresses at fillets and thin walls are approximate: shown as such, never as a
  certification. Comparing two detail levels shows whether a peak is real.
- float32 CG on the GPU may stall before 1e-6: float64 sums, compare with the CPU, relax the
  analysis tolerance if needed.
- The GPU can be short of memory while the editor's workers hold it: estimate first, release
  idle workers, or fall back.
- A minutes-long Optimize blocks previews (like an export): Stop is always visible and the live
  shape shows progress.
- Thresholding can leave thin or disconnected bits: projection, the connectivity filter and min
  thickness handle most; Smoothen Body is still available.

Added after phase 2 (2026-10-15): **Max thickness** on Topology Optimization (local volume
constraint), for branchier, more organic designs; see FINDINGS.md, "Max thickness".

Added 2026-10-15: a **bearing** Load kind (a pin presses only on the side of its hole the force
points to) and Members **struts only**; the answer to "a big flat member" turned out to be holding
the part where it is really held (bolt holes, not a whole glued face). See FINDINGS.md, "Bearing
load, and the big flat member".

Added 2026-10-15: a **torque** Load kind and an **Offset** for forces (a force acting away from
its faces), both as a rigid-connection moment on the faces; dragged in the 3D view. See
FINDINGS.md, "Torque and offset forces".

## Open questions for the user

1. Optimize button (recommended) or automatic re-run after every change?
2. A first real part: a design space modelled in SolidWorks (e.g. a bracket with bolt holes),
   with where it is held and what force it takes.
3. Which materials and filaments to list, and their values if known.
4. A guided side panel later, or are the nodes and presets enough?
