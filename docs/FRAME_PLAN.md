# Frame (truss) optimization: plan

Status 2026-10-16: **F1 built** (engine core: `engine/frame/`, `tests/test_frame.py`) plus a **first
editor version** of F2: the Frame Optimization node (round bars, plain joints, skin at supports and
loads, Optimize button, results in `cache\frame`), demo `recipes/frame_bracket.json`, then **all
profiles in the body, fillet / ball joints and curving**, then **mid-bar joints, joining frames
(several design spaces, Connect to, Join Frames) and the frame view** (bars coloured by size in the
3D view). Not started: drain holes, F3. See "F1 as built", "First editor version", "Profiles,
joints and curving" and "Mid-bar joints, joining frames and the frame view" at the end. Decisions
marked **(open)** are the user's.
Builds on FEA phases 1 and 2 ([TOPOLOGY_PLAN.md](TOPOLOGY_PLAN.md), Topology Optimization in
`engine/fea/topology.py`).

## What the user decided (2026-10-15)

- Instead of the organic SIMP shape, the part is built from **bars**: members with one profile
  shape (round, square, tube, ...) that stays **the same along the whole bar**, joined at nodes.
  The result should look made by hand.
- Bars can be **straight or curved**.
- **Different sizes are fine.** No specific beam or profile system for now (generic profiles).
- Parts are printed with **SLS**, so no overhang limits and no cut list.
- Joints are **merged with fillets**, and the joint style should be a choice.
- **No plates** in this MVP.
- The look is a **truss** (Swedish *fackverk*): bars meeting at shared nodes, mostly in triangles.
  That is what the layout step produces by nature (a minimum-volume truss), so it is the core of
  the plan, not an extra.
- A bar may **meet another bar anywhere along it**, not only at its ends (asked 2026-10-15): the
  meeting point is a node, and the bar it lands on runs straight (or curved) through it as one
  continuous bar.
- Separate trusses must also be able to **join each other** (and other bodies) in the design,
  with the same filleted joints (see "Joining frames").

Reading of "the section is the same the whole way" used below: each bar keeps one cross-section
from end to end (no tapering), and a frame uses one profile family; bars can still differ in size.
Tell me if you meant something else **(open)**.

## The idea in one paragraph

Continuum topology optimization (SIMP, Topology Optimization)
answers "which voxels carry load", which is why it looks organic; even its *Max thickness* with
*Members: struts only* gives branching struts of varying thickness, not bars. The frame version asks "which bars
between which points carry load, and how thick must each be". That is classic **ground-structure
layout optimization**: scatter candidate nodes in the design space, connect them with thousands of
candidate bars, let a linear program pick the lightest set of bars that holds the loads, smooth
chains of bars into curves where wanted, size each bar, check the frame with beam elements (stress,
deflection, buckling), and turn the bars into an implicit body with their real profile and
filleted joints. It reuses what is in main from phase 1 (Material, Fixed Faces, Load, supports and
loads from CAD faces, Static Analysis to verify the result) and does not need the SIMP loop.

## What the user does in the editor

```
Import STEP > Pick Body ──┬─> Fixed Faces > Load ──────────────────────────┐
                          └─> design space ────────────────────────────────┤
Material (Nylon PA12) ─────────────────────────────────────────────────────┴─> Frame Optimization
     > body > Mesh from Implicit Body > Export Mesh
     > body > Static Analysis (voxel check of the real shape)
```

1. Same design space, supports and loads as for Topology Optimization.
2. **Frame Optimization** node: profile (round, square tube, ...), sizes, how curved, node spacing,
   joint style and fillet radius, safety factor. Press **Optimize**; bars appear in the 3D view
   coloured by size, the log prints mass, utilisation and sizes used ("3 sizes, 14 bars, 86 g").
3. Changing only the joint style, fillet or curving is instant (no new optimization).
4. Check the result with Static Analysis like any other body.

## Node

One node for the MVP; a separate Section node only if profiles get many settings later.

**Frame Optimization** (category Simulation)

| input | default | meaning |
|---|---|---|
| design space | | CAD or implicit body the bars must stay inside |
| material | | from Material / Custom Material (PA12 for SLS) |
| conditions | | from Fixed Faces / Load |
| profile | round | round / round tube / square / square tube / rectangle / rect tube / I / angle |
| proportions | | rectangles: height / width; tubes: wall (mm or share of the width); I: flange and web share |
| sizes | any | `any` (each bar its own size, rounded to *size step*) or a list, e.g. `4, 6, 8, 12` (mm, the profile's width) |
| size step | 0.5 mm | rounding for `any` |
| max sizes | 0 | 0 = no limit; otherwise the bars are grouped into this many sizes (a calmer look) |
| min size | 2 mm | smallest bar (SLS: walls of tubes at least ~0.8-1 mm) |
| curving | 0.5 | 0 = straight bars only, 1 = smooth curves wherever a chain of bars bends gently |
| node spacing | 0 = auto | spacing of the candidate nodes (finer = more possible bars, slower) |
| reach | 3 | how far a candidate bar may span, in node spacings |
| mid-bar joints | on | bars may land anywhere along another bar (off: only where bars end) |
| joint cost | 0.1 | penalty per node: higher gives fewer, longer bars (the main "how hand-made" knob) |
| joints | fillet | sharp (plain union) / fillet / ball |
| fillet | 0 = auto | fillet radius in mm (auto: 0.5 x the thinner bar) |
| orientation | follow load | non-round profiles: strong axis in the bending plane, or aligned to world Z **(open)** |
| safety factor | 2 | on the material strength |
| connect to | (optional) | other frames or bodies this frame joins (see Joining frames) |

Outputs: **body** (implicit), **frame** (the bars, for the 3D view and a later Frame Analysis node).
The log reports mass, max utilisation, max deflection, buckling factor and sizes used.

It behaves like Topology Optimization: computes on **Optimize** (or in an export when missing) and
otherwise reads its result store, with the same hidden `compute` / `stamp` inputs (set by
`server._set_compute` for any node that has them, so no server change is needed for that). Joints, fillet,
orientation and the curving of an existing layout are applied in normal previews, since they only
change the geometry step. The layout itself takes seconds, so it could re-run on every change
instead **(open)**.

## Technical design

### 1. Ground structure (`engine/frame/ground.py`)

- Candidate nodes: a regular lattice inside the design space at *node spacing* (auto: design space
  diagonal / ~15), kept where the design space's distance field says "inside, at least half a min
  size deep". Plus nodes at supports and loads: each Fixed Faces / Load condition's triangles are
  sampled with `model.sample_triangles` and clustered (k-means) to a few nodes per face (1 for a small
  face, up to ~6 for a large one, so a bolted face can take a moment).
- Candidate bars: every pair within *reach*; dropped when the segment leaves the design space
  (distance field sampled along it) or passes through another node (the LP chains the short pieces).
  Thousands to a few hundred thousand candidates.
- Loads: each load condition's total force split over its nodes by area share (sum exact, as in
  `apply_conditions`). Supports: support nodes fixed in x, y, z.

### 2. Layout optimization (`engine/frame/layout.py`)

- Plastic minimum-volume truss LP (Dorn, Gomory, Greenberg 1964): tension and compression force per
  bar, minimise sum of length x force / allowable stress, subject to nodal equilibrium.
  Allowable = strength / safety factor.
- Solved with `scipy.optimize.linprog(method="highs")`; for big ground structures **member adding**
  (Gilbert & Tyas 2003): start with nearest-neighbour bars, add only bars whose dual "virtual
  strain" exceeds 1, repeat. Keeps the LP small; the optimum is still global.
- *Joint cost*: a second pass that penalises nodes (reweighted LP: each bar's cost gets a term for
  the nodes it would add, repeated a few times) so the result has fewer, longer bars.
- Mid-bar joints: in the LP a joint along a bar is just a node that the bar passes through, so the
  method allows it already; the question is only where such nodes can be. Ground nodes give a
  fixed set of positions, so after the first solve nodes are added along every bar that is used
  (every ~node spacing / 4) and the LP is solved again with member adding: a branch can then land
  at almost any point along a bar, and the optimizer chooses where.
- Clean up: drop bars with tiny force, snap nearly coincident nodes, and join the pieces of a bar
  that passes straight through nodes into one **continuous bar** (also where other bars meet it
  along the way). A continuous bar keeps one section from end to end: its size is that of its most
  loaded piece, so the look stays "one bar with others welded onto it".
- A setting *mid-bar joints* (on / off, default on) for when only end-to-end joints are wanted.

### 3. Curving (`engine/frame/curves.py`)

Fine ground structures already approximate curved load paths (Michell's arcs) as chains of short
straight bars through nodes that only have two bars. *Curving* turns such chains into smooth bars:

- Find chains: runs of bars through 2-bar nodes whose direction changes by less than a limit
  (scaled by *curving*; 0 = never).
- Fit a smooth curve through the chain's nodes (centripetal Catmull-Rom or a smoothing cubic
  spline), ends fixed at the real joints, tangents at the ends kept along the original end bars so
  joints stay clean. Curves must stay inside the design space (checked with its distance field;
  less smoothing where they would leave it).
- One bar = one curve with one section: its size is the largest any piece of the chain needed.
- Internally a curved bar is a polyline of short pieces (chord error < 1% of its width), used by
  the beam check and the geometry.

### 4. Sizing (`engine/frame/sizing.py`)

- Profile properties (area, second moments, section modulus, radius of gyration) computed from the
  profile and its size, so any size works without a table.
- For each bar: the smallest size with enough area for its force, and for bars in compression an
  Euler/Johnson buckling check over its length (effective length factor 0.7 for printed rigid
  joints, 1.0 as the safe choice for the first version **(open)**). Rounded up to *size step* or the
  next size in the list, never below *min size*.
- *Max sizes*: group the required sizes into that many groups (1D optimal grouping by dynamic
  programming over the sorted demands; each group gets the size of its largest bar).

### 5. Frame check (`engine/frame/beam.py`)

- 3D frame elements (12 dof, Euler-Bernoulli; Timoshenko for short thick bars), rigid joints (a
  printed part is one piece), curved bars as chains of straight elements. Sparse assembly,
  `scipy.sparse.linalg.spsolve`: a few thousand dof, milliseconds.
- Axial + bending stress per bar, deflection at the loads, linear buckling factor (`eigsh` on the
  geometric stiffness). Bars that fail get the next size up and the frame is checked again, a few
  rounds at most (fully stressed design).
- Report: mass, max utilisation, max deflection, buckling factor, sizes used. Printed parts are
  weaker between layers (less so for SLS than FDM): the help text says so, as for Static Analysis.

### 6. Bars to a body (`engine/frame/geometry.py`)

- Every bar piece is a short straight prism: the 2D profile's signed distance (circle, rectangle,
  hollow versions, I, angle) in the piece's local frame. Frames along a curved bar come from a
  rotation-minimising frame, so a square tube does not twist. Pieces of one bar are cut at the
  bisecting planes between them (mitred), so a curved bar is one smooth constant section with no
  bumps at the piece joints.
- Joint styles, all applied only near nodes so the bars keep their exact section elsewhere:
  - **sharp**: plain union; bar ends reach a bit past the node so they overlap cleanly.
  - **fillet**: smooth union (polynomial smooth min with radius *fillet*) of the two nearest bars at
    each point. Away from a node the second-nearest bar is far, so nothing changes there.
  - **ball**: a sphere at each node, sized from the thickest bar meeting there, then filleted.
- Hollow profiles: tubes are closed where they meet a joint. Closed hollow bars trap unsintered
  powder in SLS, so hollow profiles get small drain holes near each end (diameter set, default
  2 mm), or are filled solid inside the joints **(open)**.
- One Warp kernel, like `engine/segments.py`: a BVH over the piece boxes and a growing search cube,
  returning the nearest and second-nearest bar distances (for the fillet); GPU or Warp's CPU device;
  the profile kind is an integer switch. Pure function of position, so block meshing, slices and
  surface colours work.
- Union with the passive regions (skin at fixed and loaded faces, *keep* body, the same inputs and
  code as Topology Optimization's *skin* and *Keep*) so bolt bosses and mounting faces stay as in
  the CAD part.

### 7. Joining frames

Two cases, both in the MVP:

- **Optimized together** (the frames share the load): Frame Optimization's *design space* takes a
  list of bodies (several design spaces, e.g. a base frame and an arm). One ground structure is
  built over all of them, with candidate bars allowed between them only where the design spaces
  touch or overlap, so the LP itself decides where and how the frames connect. The result is one
  frame, split back per design space only for colouring. This is the stronger answer: joints land
  where the forces need them.
- **Joined to an existing frame or body** (*connect to*): a frame optimized earlier, or any body
  (the lamp centre, a CAD mount). Its surface becomes support for the new frame: candidate nodes
  are placed on it (as for Fixed Faces), so the new bars end on it, and its reactions are reported
  so the other frame can be checked with them as a load. For another frame, nodes are placed at its
  nodes and along its bars, so new bars can land anywhere on it.
- Geometry: a **Join Frames** node unions frames (and bodies) with the same joint styles: where a
  bar of one meets a bar or node of another, the fillet / ball joint is built exactly as inside one
  frame (the kernel takes the bars of all joined frames, so the nearest and second-nearest bar can
  come from different frames). Joining an ordinary body uses a fillet against its distance field.
- Joined frames can use different profiles and sizes; only the joint settings must agree (the Join
  node's own settings win).
- CLAUDE.md's rule still holds: don't combine one big body with a per-body list in implicit nodes;
  Join Frames takes the frames as separate inputs and meshes them as one body only where they meet.

### 8. Server and editor

- The Optimize button, the `optimize` long job, `compute` / `stamp` and the result store on the SSD
  already exist for Topology Optimization and are reused as they are; the editor's Optimize / Stop
  button is shown by node type, so Frame Optimization is added to that list.
- 3D view: bars as coloured tubes per size (a light `frame` view, cheaper than meshing the body),
  hover shows size, length, force, utilisation.
- New port type `frame`; colour in `types.ts`.

### Files

| file | what |
|---|---|
| `engine/frame/profiles.py` | profile shapes, properties, 2D distance |
| `engine/frame/ground.py` | candidate nodes and bars from design space and conditions |
| `engine/frame/layout.py` | LP with member adding, joint cost, clean-up |
| `engine/frame/curves.py` | chains to smooth curves |
| `engine/frame/sizing.py` | sizes, grouping, buckling |
| `engine/frame/beam.py` | 3D frame FE, utilisation, buckling factor |
| `engine/frame/geometry.py` | bar distance kernel (Warp), joints, drain holes, body |
| `engine/frame/join.py` | several design spaces in one ground structure, attachment to frames and bodies |
| `engine/graph/nodes_frame.py` | Frame Optimization, Join Frames (Frame Analysis later) |
| `engine/server.py`, `engine/cache.py` | optimize route, result store (`cache\frame`, `FRAME_VERSION`) |
| `frontend/src/...` | port type, frame view |
| `tests/test_frame.py` | see Tests |
| `recipes/frame_bracket.json`, `presets/` | test recipe and a "Frame" preset |

## Phases

### F1: engine core, no editor (headless with `run_graph.py`)
Profiles, ground structure, LP with member adding, sizing, frame check, straight bars only. Done
when the textbook cases below pass and the wall bracket (`recipes/topo_wall_bracket.json`, same design
space, supports and load) gives a sensible truss in seconds, with mass and stiffness compared to
the SIMP result for the same part.

### F2: body, curves and the node
Bar distance kernel with the three joint styles, curving, the node, joining frames (several
design spaces, *connect to*, Join Frames), result store, report in the log. Done when the frame exports as a clean STL (reload check per CLAUDE.md), curved bars show no
bumps, and Static Analysis of the body agrees with the frame check within ~10-20% on max stress away
from joints.

### F3: editor
Frame view coloured by size with hover, preset "Frame", tuning of joint cost and curving defaults
on 2-3 real parts.

### Later (by need)
Several load cases, symmetry planes, a separate Section node or a specific profile system,
tapered bars, plates, Frame Analysis node.

## Tests

- **Two-bar and three-bar truss**: LP volume and forces equal the hand solution.
- **Michell cantilever**: LP volume approaches the analytic Michell volume as node spacing shrinks
  (within a few % at fine spacing); with curving on, the chains become arcs close to Michell's.
- **Member adding** gives the same optimum as the full LP on a small ground structure.
- **Sizing**: every bar passes its stress and buckling check; at most *max sizes* sizes; nothing
  rounded down.
- **Beam**: cantilever tip deflection and fixed-end moment equal beam theory to 1e-9; Euler
  buckling of a column equals pi²EI/(KL)² within 1%; a curved bar of many pieces converges to the
  ring/arch solution.
- **Geometry**: kernel equals a numpy reference for every profile and joint style; a curved bar's
  section is constant along it (distance from the centre curve); blocks bit-identical to the whole
  grid (`tests/test_tiles.py` style); GPU equals CPU (`--gpu` only).
- **Equilibrium**: reactions sum to the applied loads in the frame check.
- **Mid-bar joints**: a branch landing on a bar leaves that bar continuous and of one size; with
  the setting off, no bar passes through a node where another bar meets it.
- **Joining**: two design spaces optimized together give one connected frame; a frame attached
  with *connect to* ends only on the other frame's bars or nodes; Join Frames fillets a joint
  between two frames the same as one inside a frame (same distance values).

## Risks

- Large ground structures: member adding keeps the LP to tens of thousands of active bars; the
  node spacing default should keep it to seconds.
- The plastic LP ignores buckling and bending; slender compression bars can come out too thin until
  sizing and the frame check fix them, which can move the optimum somewhat. Fine for this purpose.
- LP optima can contain many near-equal alternatives (zig-zag bars): clean-up, joint cost, curving
  and max sizes steer toward simpler frames.
- Smoothing a chain into a curve changes how it carries load (a curve under axial load bends): the
  frame check sizes it with the real curved shape, so it is safe, but heavy curving costs mass.
- Voxel Static Analysis of thin tube walls needs ~3 elements through the wall, so the frame check is
  the main verification and the voxel check is for joints and mounts.

## Open questions for the user

1. Is the reading right: one profile per frame and no tapering along a bar, but sizes can differ?
2. Non-round profiles: turned to carry load best, or all lined up with the world (looks tidier)?
3. Hollow profiles in SLS: drain holes near the bar ends, or solid inside the joints?
4. Optimize button like SIMP, or re-run the layout automatically since it takes seconds?
5. Joining frames: mostly frames designed together (one optimization), or new frames added onto an
   existing one?

## F1 as built (2026-10-15)

`engine/frame/`: `profiles.py` (all eight profiles, properties from rectangles or exact round
formulas, principal axes for the angle), `ground.py`, `layout.py`, `sizing.py`, `beam.py`,
`optimize.py` (`optimize_frame(space, conditions, material, ...)` returns a `Frame`). No node yet.

What differs from the plan, and why (details and numbers in FINDINGS.md, "Frame optimization F1"):

- **Holes are one node.** A support or load is split into connected pieces of faces; a round piece
  (a bolt hole, a pin hole: its normals cancel) or a load whose pushes cancel (a bearing load) gets
  one node at its centre, as a bolt or pin acts. Bars may leave the design space within that
  node's reach of the faces. Flat faces are clustered as planned. Nodes of different loads at one
  point (two loads on one pin) are merged.
- **Support and load nodes connect to every node**, not only within *reach*: a straight bar from
  a bolt to a pin was otherwise impossible and got imitated by zig-zag chains.
- **The smallest size is part of the layout.** The plain LP spreads the load over hundreds of bars
  of a few N, all of which are then printed at the smallest size. The LP is solved again with the
  cost per N of weak bars raised to what the smallest size really costs (iteratively reweighted,
  up to 12 rounds), keeping the lightest real layout. This, not the joint cost, is what makes the
  frame simple.
- **Joint cost** is Parkes et al.'s joint length: every bar costs as if it were *joint cost* x the
  design space's diagonal longer. Default 0.05 (0 and 0.02 left more bars on the glued bracket).
  With a joint cost, bars may pass over lattice nodes (overlap).
- **LP solver:** all candidates at once up to 80,000 (HiGHS interior point, ~1 s for 20,000);
  member adding (Gilbert & Tyas) above that, in larger batches.
- **Frame check supports are clamped** (bolted or glued faces). Buckling uses K = 1.0 per bar
  over its unbraced length (the effective length in Sizing is still open), plus the frame's linear
  buckling factor, which must reach the safety factor.
- **Orientation** of non-round profiles: the strong axis is turned toward world Z for now (open
  question 2).
- **Not yet:** mid-bar joints by inserting nodes along used bars (bars running straight through a
  node are already one continuous bar), curving, geometry, the node and joining frames: F2.

Wall bracket, PETG, safety factor 2, round bars, min size 2 mm, auto spacing (~5.6 mm), joint cost
0.05 (`docs/frame_f1_bracket.png`):

| case | bars | joints | sizes, mm | mass | max deflection | time |
|---|---|---|---|---|---|---|
| bolted, bearing 200 N down | 9 | 3 | 2, 2.5, 4.5, 5 | 3.8 g | 0.84 mm | 11 s |
| bolted, + 80 N along the pin | 9 | 4 | 2-5.5 (6 sizes) | 4.4 g | 0.74 mm | 14 s |
| glued, + 80 N along the pin | 9 | 4 | 2, 4, 4.5, 5 | 4.2 g | 0.93 mm | 16 s |

Against SIMP on the bolted bracket (Keep 20%, max thickness 4 mm: 19.3 g, 0.199 mm in a voxel
analysis): the frame is 5x lighter and 4x more flexible, about the same stiffness per gram, because
it is sized for strength (safety factor 2) rather than given a mass. The frame's deflection is from
the beam model; the voxel check of its real body comes with F2's geometry.

## First editor version (2026-10-15)

- Node **Frame Optimization** (`engine/graph/nodes_frame.py`, category Simulation): design space,
  material, supports & loads, min size, sizes (any or a list), size step, max sizes, safety factor,
  joint cost, node spacing, skin; hidden compute / stamp. Round bars only for now (the engine sizes
  every profile, the body draws round ones). Optimize runs the layout (~10-20 s) and stores nodes,
  bars, sizes and the summary in `cache\frame` (`load_frame` / `save_frame`, keyed by every input
  but skin; `FRAME_VERSION`); previews read it; the summary is the node's note.
- Body (`engine/frame/geometry.py`): union of round bars (capsules) per size with the Warp segment
  kernel (exact for mixed sizes), plus the skin: exact distance to each connected piece of support
  and load faces (libigl AABB, in that piece's box, capped), all cut by the design space so holes
  stay open. Bars meet with plain joints.
- The editor's Optimize / Stop button is shown per node type (`OPTIMIZERS` in EngineNode.tsx).
- **Supports: connect all** (default; asked 2026-10-15, one bolt boss of the demo was left loose):
  the layout leaves out supports it does not need (the demo used 3 of 4 bolts). Each is joined by
  the shortest chain of candidate bars (Dijkstra over the ground structure, `layout.connect`) to a
  frame node that is not held, so it shares the load; the frame check sizes those bars (demo: 10
  bars, 3.9 g, all four bolts carry load). *only needed* keeps the layout as it is and gives unused
  supports no skin, so no loose boss either way.
- Voxel check of the demo (bolted bracket, PA12, 200 N bearing + 80 N along the pin; 9 bars,
  3.7 g): max deflection 1.05 mm (medium) / 1.06 mm (fine) against the frame check's 0.91 mm, so
  the beam model is 15% stiffer (target 10-20%). Peak stress 81 / 92 MPa against 24 MPa allowed:
  at the sharp corner where the most loaded bar (3.5 mm, 142 N, ~15 MPa in the bar) meets a bolt
  boss next to the rigid support, growing with finer elements (a corner, not the bar). Filleted
  joints (F2) are the fix; the frame check sizes bars, not joints.

## Profiles, joints and curving (2026-10-15)

- **All eight profiles in the body**, drawn by **one Warp kernel** (`geometry._frame_grid`, GPU or
  Warp's CPU threads, a BVH over the pieces' boxes): round bars as capsules, round tubes as an
  annulus, the others as their rectangles (holes negative) in the piece's local frame (strong axis
  toward world Z, carried along each bar as a rotation-minimising frame, `curves.orientation`,
  also in the frame check), between end planes; pieces of one bar are mitred. Exact against
  `Profile.distance` and the capsule union (tests); GPU equals CPU (`test_gpu.py --gpu`); a block
  equals the whole grid bit for bit.
- **Joints:** *fillet* (default): the kernel keeps the two nearest *different* bars and blends them
  with a polynomial smooth minimum, k = 0.6 x the thinner bar (or the node's Fillet, mm); pieces of
  one bar never blend (a curve would get rings). The bars are joined to the bolt bosses and the
  pin's eye with the same fillet (`F.smooth_union`). *plain*: a hard union. *ball*: a filleted ball
  (0.6 x the largest bar) at each free joint, none at bolts and pins. The first ball joints (stuck-on
  spheres) looked clumsy, and plain joints left the sharp corners that showed 81-92 MPa.
- **Curving** (`curves.smooth_bars`, node default 0.5): at every node nothing holds or loads, the
  two bars that go on most nearly straight (turn <= 70 degrees) are linked; each path of linked bars
  becomes one curved bar: a cubic Hermite spline through all its nodes (joints where other bars land
  stay on it), tangent direction at an inner node = the bisector of its two chords, at the ends the
  direction that makes the end span a parabola; *curving* scales the tangents (1: a full even arc,
  smaller: flatter, never a kink). Pieces turn at most 4 degrees. Curves stay inside the design
  space except within a hole node's free radius (the bars to a pin's axis cross its hole; checking
  there dropped every curve to the pin); else flatter, else straight.
  The first version rounded only a short stretch at each bend (straight, tight bend, straight):
  it looked unnatural.
- I sections keep flange and web at least `min_wall` (0.8 mm): 12% shares were 0.3 mm at 2.5 mm.
- Wall bracket, PA12, 200 N + 80 N along the pin, fillet joints (`docs/frame_profiles.png`), 16-17 s:

  | profile, curving | bars (curved) | sizes, mm | mass | deflection |
  |---|---|---|---|---|
  | round, 1.0 | 8 (2) | 2-6.5 | 4.8 g | 0.83 mm |
  | round, 0.5 | 8 (2) | 2-6 | 3.9 g | 0.88 mm |
  | square tube, 1.0 | 8 (2) | 2-7 | 3.9 g | 0.81 mm |
  | I, 1.0 | 8 (2) | 2-10.5 | 4.7 g | 0.71 mm |

  A curved bar also bends under its axial force, so more curving costs mass (round: +23% at 1.0);
  on a dense 75-bar truss in a block, curving 1 cost +44% (11.2 -> 16.1 g).
- Closed spaces: hollow tubes are closed where they meet a fillet or boss (cavities: trapped powder
  in SLS, drain holes still open); the print Check finds them. Draft previews 2-5 mm bars coarsely;
  Medium shows them clean.

## Mid-bar joints, joining frames and the frame view (2026-10-16)

- **Mid-bar joints** (node setting, default on): after the first layout, `ground.refine` adds
  nodes along every bar it uses (one per node spacing, none on top of a node already there) and
  candidate bars from them to each other, to the used bars' ends and to the support / load nodes
  within twice the reach (not along their own bar; to every node within reach cost three times the
  time for the same frames); the used bars also come as their pieces between the new nodes. A
  second LP (the reweighting started from the first layout) chooses. A bar runs on through every
  joint where others land, straight or as one curve through all its nodes. *off*: bars end at every
  joint (`continuous_bars(through_joints=False)`), and only free two-bar corners curve.
  **Measured:** the second LP used **none** of the added nodes in all three cases (bracket, the
  60 x 20 x 30 mm block, frame B of the join demo: 193 added); the first layout stayed. A
  pin-jointed minimum-volume truss cannot land a branch on a perfectly straight bar (the branch's
  sideways force needs a kink there), so every landing is a kink, and the lattice already offers
  the landing points that pay. Bars run on through joints only where curving links the two pieces
  that go on most nearly straight (turn <= 70 degrees) into one curve:

  | case | mid-bar off | mid-bar on |
  |---|---|---|
  | bracket, curving 0.5 | 10 straight bars, 3.91 g, 5 s | 8 bars (2 curved), 3.94 g, 9 s |
  | block, curving 0.5 | 75 bars, 11.2 g | 54 bars, a bar runs through 21 of 28 joints, 13.7 g |
  | block, curving 0 | 75 bars, 11.2 g | the same 75 bars (nothing is straight through a joint) |

  The block's extra mass is the curves (a curved bar also bends) and one size per continuous bar.
  Points every half or quarter spacing (the plan said spacing / 4): the block ran over 12 minutes
  instead of 2 s (the larger candidate set goes through member adding in every reweighting round)
  and was stopped. The user decided (2026-10-19) to keep the second LP as it is, although it costs
  time and has not changed a layout yet.
- **Several design spaces** (Design space takes a list, as planned): one ground structure over their
  union. Where two spaces only touch, the union's field is 0 along the shared face, so every bar
  across it was refused and the body was cut there: `ground._Inside` replaces the field by the
  voxels' own distance where the two disagree by more than two voxels (only at such seams), and
  the body is cropped by the union grown 0.02 mm (`geometry.seamless`).
- **Connect to** (port type `frame`, a list): another frame's Frame output or a CAD body. Its nodes
  and points every half spacing along its bars (a body: up to 12 clustered points on its surface),
  if within a node spacing of this design space, become held nodes (clamped in the check, free
  radius = the bar's radius + half a spacing, no skin). They are reached by bars within the reach
  only (support faces connect to every node). Fixed Faces are then optional. The log says what this
  frame puts on each: "connect to #1: joined at 6 points, this frame puts (0, 0, -120) N on it".
  That force is **not** added to the other frame: add it as a Load there (or optimize both as one
  frame with two design spaces, the stronger answer when they are designed together).
- **Join Frames** (Frame A-D, joints, fillet): the bars of all frames in one kernel call (bar ids
  offset per frame), so a joint between frames gets the same fillet as one inside a frame; each
  frame's skin; cropped by the union of their design spaces (grown as above); CAD bodies joined
  with the same fillet. A frame that is not optimized yet stops it with "Frame B: that frame is not
  optimized yet: press Optimize on it first".
- **Frame view**: Frame Optimization's `node_done` carries its bars (`frames`: nodes, pieces, bar
  per piece, size / utilisation / length per bar, axial force per piece). The 3D view's *Bars*
  button draws them as tubes coloured by size, with a legend per frame; hovering a bar shows e.g.
  "bar 63 of frame_b: 3.5 mm round, 53.3 mm long, 22 N compression here, utilisation 0.84". Hide
  a frame's body to see its bars.
- **Reweighting stops on a repeat**: the reweighted LPs cycle (the bracket: 12, 9, 9, 12, ... bars);
  stopping at the first used set seen before, keeping the lightest, made runs 2-4x faster.
- **Solver leftovers**: frame B of the demo kept a lone 0.1 N bar (the bars around it fell below
  the 0.1% drop share), not connected to anything: the frame check was singular. Bars ending at a
  node nothing else meets, and pieces that reach no support, are now dropped (`layout._held_bars`).
- Demo `recipes/frame_join.json` (`samples/two_blocks.step`: A 40 x 20 x 24 mm and B 50 x 20 x 24
  mm touching at x = 40; PA12): frame A held at its wall face with 120 N at x = 40 (52 bars), frame
  B connected to frame A with 120 N at its end (82 bars, 23 curved, 7.8 g, utilisation 0.91, 26 s,
  joined at 6 points), Join Frames: one part (`docs/frame_join.jpg`). Loads on whole end faces
  make dense trusses (clustered load nodes each need bars), like the 75-bar block earlier.
