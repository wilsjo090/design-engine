# Dual contouring: plan

Status 2026-10-24: D1 (one grid, measured against marching cubes) built and measured; D3 built: Mesh
from Implicit Body's Method (marching cubes / dual contouring); D2 built: big bodies in blocks.

## Why

Marching cubes puts every vertex on a grid edge, so a sharp edge that runs between grid edges is
cut off: CAD edges, the rims of Hole Pattern holes and every Boolean's crease come out bevelled about
a voxel wide. Today the cure is a smaller voxel, which costs time and triangles that decimation then
removes again. Dual contouring puts one vertex inside each cell the surface passes through, where the
planes through the surface's crossings meet best, so it lands on the edge or corner when the cell holds
one. If that keeps edges crisp at 2-4x the voxel, Final previews and exports get 4-60x fewer samples.

## How (D1, built)

`engine/dualcontour.py`, `dual_contour(values, grid)`: the same input and output as
`meshing.field_to_mesh` (values on a `Grid`, inside negative; a closed mesh with the same `metadata`),
CPU / numpy. The module's docstring has the details; in short:

- Crossings: every grid edge whose two samples differ in sign; the point by linear interpolation, the
  normal from central differences at both samples, interpolated. Each crossing is *trusted* by how
  straight the field is around its samples (forward and backward differences agree).
- What did not work, and why the rest is there: with only a cell's own crossings, edges came out barely
  sharper than marching cubes (0.20 voxel off against 0.34): every crossing in a cell at a crease lies
  within a voxel of it, so all their normals blend both faces. A piece with an untrusted crossing takes
  the planes of the crossings up to three steps away along the surface, weighted by trust, distance,
  and how well its own crossings lie on each plane (a piece on one face beside an edge must not take the
  other face's plane). The fit has a weak pull towards the crossings' mean (a hard eigenvalue cutoff let
  one weak plane decide a direction completely), and a vertex may lie half a voxel outside its cell: a
  corner that clips a cell without enclosing a sample gives that cell no vertex, so a neighbour's must
  reach it (clamping to the cell kept edges 0.3 voxel off).
- Manifold: a vertex per piece of surface in a cell (Manifold Dual Contouring); on a face with four
  crossings the inside corners join or not as marching cubes decides (the asymptotic decider), the same
  in both cells; where a piece crosses a face twice beside one that does too, each arc gets a vertex of
  its own (else four quads share one edge).
- Quads are split along the diagonal whose triangles face best along the field's normal; a sliver that
  still faces against it has its vertices put back into their cells, then onto their crossings' mean.

Checked by tests (`tests/test_dualcontour.py`): a turned box's corners and edges (against marching
cubes), a sphere's radius, an L's concave crease, two plates closer than a voxel (closed, manifold, the
same topology as marching cubes), the cell tables.

## D3 (built): Method on Mesh from Implicit Body

`nodes.MESHERS` maps the node's `method` to `field_to_mesh` or `dual_contour`; `_mesh` / `_mesh_on`
pass it down. Default marching cubes, so recipes without the input are unchanged. A body over
`tiles.TILE_MIN_VOXELS` (32M voxels) is meshed in blocks (D2). Its views get no on-screen detail
(`server.detail_floor` returns None): details are marching cubes in blocks and would bevel the edges
the view keeps. Body views (`mesh.view`) stay marching cubes. Tests:
`tests/test_dualcontour.py::test_mesh_from_implicit_meshes_by_the_chosen_method`,
`tests/test_server.py::test_a_mesh_made_by_dual_contouring_gets_no_marching_cubes_detail`.

## D2 (built): in blocks

`tiles.mesh_tiled(method="dual contouring")` runs `tiles.dual_blocks` in the workers and
`dualcontour.join_blocks` in the main process. `dual_contour` is split into `_vertices` (crossings,
pieces, fits) and `_faces` (quads to triangles, fold repair, arc vertices), so a block runs the same code:
- Each block is sampled with BLOCK_HALO (12, a multiple of `tiles.ALIGN` so the coarse noise and
  distance-field lattices still coincide) more samples on every side; a vertex depends on samples up
  to ~RINGS + 3 cells away, and the block's quads use the vertices of the cells just outside it. Outside
  the box the samples are set outside, like a single grid's pad.
- A block keeps the vertices of its own cells and the quads of its own grid edges (by the edge's lower
  sample; the box's outer layer of cells and edges to the outermost blocks), names vertices by global
  cell x 4 + piece, and lists the blocks its quads and cells reach into (`needs`: the next round, like
  marching cubes' face crossings). The pieces join by name, nothing is matched by position.
- Fold repair runs per block on its own quads; when it moves a neighbour's vertex it sends the request
  (how far: into the cell, to the mean), and join_blocks moves each vertex as far as any block asked
  (from the owner's fitted position and mean). Arc vertices are named by their two vertices and inside
  sample, made from all the block's quads (so both blocks make the same one) and kept once.
- Keep and Tolerance reduce the joined mesh in the main process (marching cubes reduces per block in the
  workers, borders locked: the blocks' borders here are not where a vertex's owner is).
- Checked: on one grid's own samples, blocks give that grid's mesh (to 2e-12 mm);
  `tests/test_tiles.py` compares blocks with one grid over the same box (CAD part, texture, a hole),
  a probe that missed blocks, a body cut off by the box, and Keep.
- On the GPU the distance field differs from the CPU's (the test part: 3,745 near-surface sign flips,
  9% more vertices, marching cubes alike), so blocks evaluated on the GPU are compared with a GPU grid.
- The user's lamp center (lamp_noise2, a dense triangle-wave texture, 0.3 mm: 103M voxels, 126 of 150
  blocks, 4 workers with 3 on the GPU): marching cubes 10.6 s, +5.3 GB; dual contouring 28.1 s, +3.1 GB,
  4.78M triangles, 864 folds left, closed after reloading the STL. Per block 0.38 s against marching
  cubes' 0.11 (numpy), peak up to 226 MB (`tiles.DUAL_BYTES` budgets 200 bytes per sample, 354 MB).
  On the CPU the halo doubles the sampling (121^3 against 97^3 samples per block): 187 s against 107 at 0.4.
- Found on the way, in all dual contouring: fold repair clipped vertices onto their cell's border, so two
  neighbours' vertices could sit on their shared corner: one vertex in a float32 file, edges of four
  triangles (the lamp center at 0.6 mm: 18 such edges after reloading; meshes were checked in memory
  only). Clipped INSET inside now, and plain (not sharp) vertices are kept inside their cells from the
  start (PLAIN_MARGIN): the lamp center's folds at 0.6 mm 1,276 -> 898, accuracy unchanged (holes'
  edges 0.0116 -> 0.0118 mm, the wave's surface 0.0054 -> 0.0061 mean at 0.3 mm, its edges' 95% better).
  Dense textures finer than the voxel still fold (a lamp-like texture on the test part at 0.4 mm:
  0.4% of triangles); the card warns above FOLDS_TO_WARN.

## Later

- The views (`mesh.view`) and details (blocks with `close=False`: quads whose cells leave the box dropped).
- D4: Warp kernels (GPU), like `meshing._marching_cubes_gpu`.

## Results (D1, 2026-10-24)

Both from the same samples (CPU extraction; `scratchpad` benchmark against the exact CAD mesh, or against
marching cubes at a third of the voxel where there is none). "Edges": the true sharp edges' distance to
the result (the bevel); "surface": the result's vertices' distance to the true surface.

| Part | Voxel | Samples | MC edges mean / 95% | DC edges mean / 95% | MC / DC surface max | MC / DC time |
|---|---|---|---|---|---|---|
| vent case, CAD only | 0.15 | 50M | 0.050 / 0.074 | 0.0024 / 0.004 | 0.036 / 0.005 | 2.6 / 6.2 s |
| | 0.45 | 2M | 0.104 / 0.180 | 0.0084 / 0.018 | 0.100 / 0.013 | 0.2 / 0.6 s |
| vent case, slots + hex holes | 0.3 | 6.6M | 0.072 / 0.134 | 0.012 / 0.041 | 0.100 / 0.095 | 0.6 / 5.3 s |
| | 0.45 | 2M | 0.112 / 0.207 | 0.021 / 0.071 | 0.148 / 0.180 | 0.2 / 3.2 s |
| textured test part (smoothed wave) | 0.3 | 1.9M | 0.019 / 0.043 | 0.014 / 0.032 | 0.054 / 0.058 | 0.3 / 0.9 s |
| | 0.6 | 0.3M | 0.044 / 0.088 | 0.032 / 0.062 | 0.169 / 0.179 | 0.1 / 0.2 s |
| key box, CAD only (filleted) | 0.8 | 2.4M | - | - | 0.065 / 0.153 | 0.2 / 0.6 s |

- **CAD edges and holes: the point of it.** At 3x the voxel (0.45 against 0.15 mm: 25x fewer samples, 9x
  fewer triangles) the vent case's edges are still 6x sharper than marching cubes at 0.15 mm. Its holes'
  rims 6x sharper at the same voxel. Closed and manifold everywhere; 10-14 sliver folds (0.003% of the
  area, on the surface) left on the holed case.
- **Smooth textures: no gain, no loss.** Crease handling is off there (the field is not a distance
  field), so plain dual contouring: ridges a bit closer than marching cubes, the surface about the same.
  With crease handling on it overshot ridges rounded finer than a voxel (0.14 mm at 0.3 mm), which the
  samples cannot tell from sharp. (Before the speed fix, crease pieces that saw no straight crossing
  passed the distance-field test by default: 99% of the wave's sharp vertices, and its outliers, 0.33 mm
  at 0.6 mm; now refused.)
- **CAD fillets under a voxel come out sharp** (key box at 0.8 mm: 88 of 135k vertices, up to 0.15 mm;
  marching cubes bevels them, 0.065).
- **Slower**: 2-4x marching cubes' extraction on CAD parts, textures and lattices (a gyroid filling a
  box, 8M voxels: 23 s against 6.3), 10-15x with many holes (the vent case: 5.6 s against 0.6 at 0.3 mm),
  where every crease piece builds its neighbourhood (`_on_plane` and the fit are 70% of it). Still small
  next to what the coarser voxel saves in sampling (the vent case: 50M samples to 2M).
- **Speed fix (2026-10-24)**: a first look at the slope around each crease piece (sparse products with
  vectors over the same RINGS steps) refuses those that are no distance field, or see no straight crossing,
  before their neighbourhoods are built: the textured part 4.7 s to 0.9 s at 0.3 mm; the CAD parts'
  meshes unchanged bit for bit (the vent case, its holes, the key box).
- **Memory**: neighbourhoods are built CREASE_CHUNK pieces at a time; all at once, the holed vent case
  peaked at 0.94 GB for 6.6M voxels (150 bytes per voxel, the node budgets 48), now 0.45 (marching cubes
  0.21); at 0.2 mm (21M voxels) 0.71 GB against 1.58. Per surface crossing it needs ~25% more than
  marching cubes (gyroid: 1,000 against 800 bytes).

## Next

- D2 (blocks): big bodies at Final, the body views and details.
- D4 (Warp): the crease neighbourhoods first.
