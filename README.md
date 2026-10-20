# Design engine

Implicit-modeling engine for giving CAD parts decorative surfaces, lattices and patterns.
Pipeline: **STEP → triangle mesh → signed distance field → field math → marching cubes → STL**.

- Core engine: STEP import, distance fields, offsets, Booleans, smoothing, meshing
- Node editor with a live 3D preview; recipes are JSON, with caching, groups and presets
- Textures from noise and waves, face masks (texture only on chosen faces), lattices
- Face patterns: holes, pockets, raised shapes, pictures and text on CAD faces
  ([docs/PATTERN_PLAN.md](docs/PATTERN_PLAN.md))
- Static stress analysis, topology optimization and frame optimization
  ([docs/TOPOLOGY_PLAN.md](docs/TOPOLOGY_PLAN.md), [docs/FRAME_PLAN.md](docs/FRAME_PLAN.md))
- GPU: distance fields and field math on an NVIDIA card (full lamp export ~2 min instead of ~7)

## Start the editor

Double-click the **Design Engine** shortcut on the desktop (or `start.bat` in this folder). It opens
http://127.0.0.1:8765 in your browser. Close the console window to stop it.

- Already running? It just opens the editor again.
- The editor UI is rebuilt automatically when `frontend/src` changed (needs Node.js).
- Port taken by another program: `start.bat --port 8766`. Other options: `--no-browser`, `--no-build`.
- Shortcut deleted? Recreate it with `python tools/create_shortcut.py`.

### Using it

| Action | How |
|---|---|
| Add a node | Right-click the canvas, press **Tab**, or **+ Node**. Dragging a wire into empty space offers only nodes that fit. |
| Connect | Drag from an output dot (right) to an input dot (left). Colours show the type; incompatible links are refused. |
| Edit a value | Type in the node. `0,35` and `0.35` both work. **↑/↓** nudge the last digit (Shift = ×10). Or **drag the input's name** left/right (dotted underline): the value changes about 1% per few pixels, Shift = 10× faster, Alt = finer; the Draft preview follows when you pause. |
| Show in 3D | Click **○** on a node (bodies, meshes, CAD). Several can be shown at once. A list of bodies (e.g. all petals) is previewed with the detail of one body the size of the whole group. |
| Value ranges | After a run, a node's output shows the values it can take (e.g. `-1 … 1` for Noise, `0 … 1` for a Face Mask) when that follows from the formulas. **Remap** with *Auto: yes* uses it for From Low/High. Distances and coordinates depend on the part: no range shown. |
| Look at a field | Click **○** on a field node (Noise, 3D Wave, Face Mask, math…). It appears as a coloured slice plane; the panel bottom-left picks the plane (X/Y/Z), its position and the colours (Bands / Heat / ± Signed). **On bodies** colours the visible bodies with the field instead (e.g. where a Face Mask is 1 and where it fades). Hover to read values. One field at a time. |
| Look inside | **Section** (top right of the 3D view) cuts the view open with a plane: X/Y/Z, position, Flip. The cut face shows orange. |
| Compare versions | **Keep variant** keeps a copy of what is shown; it stays to the left while you change settings. Its tooltip lists what changed since. Up to 4; rename, hide or remove them in the list top-left. |
| Print check | **Check**: size, volume, weight (set the material density), separate bodies, closed cavities, open edges and the wall thickness, which is also coloured on the part (red = thin). Checks what is shown: use Final for exact numbers. |
| Group nodes | Select nodes, then **Group** (bar at the top) or **Ctrl+G**. They become one node; Number nodes inside become its settings. **Edit** (or double-click) opens it, changes apply to every copy; **Ungroup** (Ctrl+Shift+G) puts the nodes back. Groups are saved in the recipe and show up under *Groups* when adding a node. |
| Presets | Select nodes, **Save as preset**. Presets appear under *Presets* when adding a node, in every recipe (files in `presets/`). |
| Delete | Select and press **Delete**. |
| Select several | **Ctrl+click** nodes, or **Shift+drag** a box around them. |
| Copy / cut / paste | **Ctrl+C / Ctrl+X / Ctrl+V**. Pastes at the mouse, with the links between the copied nodes. Works between recipes and browser tabs. |
| Paste keeping inputs | **Ctrl+Shift+V**: the copies stay connected to the nodes that fed the originals. |
| Duplicate | **Ctrl+D**: copy next to the original, keeping its inputs. |
| Undo / redo | **Ctrl+Z** / **Ctrl+Y** |
| Save | **Ctrl+S** (recipes are JSON, positions and visibility included). Unsaved changes are also kept by the engine every few seconds: after a crash, opening the recipe offers to restore them. |
| Fit the 3D view | **Fit** or **F** |

**Quality:** with *Auto* on, every change re-runs a preview. *Draft* uses 4× coarser voxels than
the recipe (fast), *Medium* 2×, *Final* the real size. **Export** always runs at full resolution
and writes every Export Mesh / Preview Image file. Exports queue up: you can export, keep editing
and export again; previews wait until the exports are done. The **Exports** button shows the queue
(× removes one), and a desktop notification says when an export is finished (allow it once).

**Speed:** only the nodes downstream of a change re-run. On multi-body files each body is always
processed by the same worker process, so after the first preview, tweaks reuse each body's
distance field. On the lamp: first Draft ~8 s (plus ~50 s to read the STEP once), then ~0.5 s
per edit.

## Memory

The engine sets itself a memory limit when it starts: 60% of all RAM, or 75% of what is free at
that moment, whichever is smaller (so other programs keep their share). The toolbar shows the
current use against that limit.

- Intermediate results are freed as soon as nothing needs them; only expensive ones (CAD distance
  fields, noise) are kept for reuse, in a small RAM cache with an **SSD cache** behind it
  (`%LOCALAPPDATA%\DesignEngine\cache`, up to 20 GB). The SSD cache survives restarts, so a
  repeated export is ~3x faster. Empty it with `start.bat --clear-cache`.
- Big jobs run fewer bodies in parallel, and a job that cannot fit stops before it starts.
- **Big parts are meshed in blocks.** A body whose grid would be over 32 million voxels (e.g. the
  lamp center: 2.7 billion at 0.1 mm) is split into cubes of 96 voxels. Only the cubes its surface
  passes through are computed, each on the GPU in a worker process, and the pieces are stitched
  into one closed mesh. Memory stays low no matter how big the part is; the log says how many
  blocks had surface. The textured lamp center at 0.1 mm: ~1 min, ~6 GB for everything.
- A watchdog frees caches when memory gets tight and, as a last resort, stops the run with a
  message instead of letting the PC run out of memory.

Typical use on the lamp (144 petals): editing at Draft ~5.5 GB, full-resolution export peak
~10–11 GB.

## GPU

An open editor keeps its worker processes, and each holds GPU memory (about 0.5-0.8 GB) even
while idle. If the GPU runs short (the memory meter's tooltip shows GPU memory in use), GPU work
slows down a lot instead of failing; closing the editor's console window gives it all back.

With an NVIDIA card the engine computes big jobs (Medium, Final, Export) on the GPU: distance
fields via NVIDIA Warp, noise and field math via CuPy. The **GPU** button in the toolbar switches
between GPU and CPU. Small Draft grids stay on the CPU (the GPU is no faster there and would cost
memory in every worker). Only as many workers use the GPU as fit in its free memory; if it runs
out anyway, that body is computed on the CPU. On the lamp: export ~2 min (CPU ~7), Final preview
~45 s.

## Command line

`python` here is the Python of the engine's conda environment (`synkad`, Python 3.12), e.g.
`%USERPROFILE%\miniconda3\envs\synkad\python.exe`.

```
python run_graph.py recipes\lamp_petals.json
python run_graph.py recipes\wave_offset.json --set import.path=C:/parts/bracket.step --set export.path=samples/bracket_wave.stl
python run_graph.py --nodes
```

- `--set NODE.INPUT=VALUE` changes any input for one run, e.g. `--set amplitude.value=0.15`.
- `--save my_recipe.json` stores the graph including your `--set` changes.
- `--workers 8` sets how many processes handle multi-body files.

SolidWorks parts: *File → Save As → STEP AP214*.

**Import STEP settings:** *Tolerance* (mm) is the accuracy of the triangles. *Angle* (radians,
default 0.5) only limits how sharply neighbouring triangles may bend; lower values add triangles
without adding accuracy. On an intricate part, angle 0.1 gave 12 million triangles and 0.5 gave
1.9 million at the same tolerance. The log warns when a body has more than 3 million triangles.
Very large STEP files take a while to read the first time (a 924 MB file: ~70 s). The result is
kept on the SSD, so opening the same file again with the same settings takes about a second; a
file saved again from the CAD program (new date) or another Tolerance/Angle is read afresh.

The lamp and key-box recipes read their STEP files from a `parts/` folder next to `recipes/`
(not in the repo: put the files there, or link the folder where they live, e.g.
`mklink /J parts "D:\path\to\your\step\files"`).

| Recipe | What it does |
|---|---|
| `wave_offset.json` | The basic texture: Wave → Offset Body → Smoothen Body |
| `lamp_petals.json` | Wave rings on every lamp petal; mounting face (radius 59 mm) kept flat |
| `lamp_noise.json` | Organic contour ridges: the same wave run on 3D noise, so spacing and height vary and every petal is unique. Petals and center are fused into one printable solid. |

### Noise patterns

`lamp_noise.json` uses the wave on a **Noise** field instead of a coordinate:
`acos(cos(density × noise))`. Ridges follow the noise's contour lines: tight where the noise
changes fast, wide bands and closed islands where it is flat.

| Knob (node.input) | Effect |
|---|---|
| `pattern_noise.scale` | Size of the islands/swirls (mm). Smaller = more variation per petal. |
| `density.value` | Number of ridges. Raise it with a larger scale. |
| `pattern_noise.seed` | A completely different pattern with the same character. |
| `pattern_noise.octaves` / `roughness` | Smooth (1–2) vs. wrinkly (4+) contours. |
| `height_factor.to_low` / `to_high` | How much the ridge height varies (fed by the `variation` noise). |
| `amplitude.value` | Overall ridge height (mm). |

### Petal sizes and the center

`lamp_noise.json` also loads the center (`import_step_1`) and uses it twice:

- **Scale Bodies** resizes petals while keeping them seated. Each petal rests on the center's fins
  with six sloped faces; those faces' planes meet in one point, and scaling about that point keeps
  every face on its fin exactly. Useful range ~1.0–1.3×: at 1.4× neighbours come within ~1 mm,
  and beyond ~1.4× the fins start to miss the sloped faces.
- The texture fades out right at the center, so ridges never push into a fin. Where a petal
  touches a fin it instead grows `overlap` mm into it, and anything deeper is cut off.

| Knob | Effect |
|---|---|
| `size_amount.value` | Extra size for the largest petals (0.3 = up to 1.3×). |
| `size_noise.scale` / `seed` | Size and placement of the groups of larger petals. |
| `size_mask.edge0` / `edge1` | How many petals grow (lower edge0 = more) and how sharp the groups are. |
| `overlap.value` | How far each petal reaches into the fins (mm), so petals and center fuse into one piece (0.3). Negative = a gap instead, for separate clip-on petals (e.g. −0.15). |

### One printed piece

The petals are meshed one by one (fast, on the GPU), the center goes through **Mesh from CAD**
(its own triangles: exact and instant), and **Merge Meshes** joins them at the end:

- *fuse = yes*: a Boolean union into one closed solid (~12 s for the lamp). Parts only join
  where they overlap; touching is not enough, which is what `overlap` is for. The log says
  "fused 145 meshes into one solid", or how many separate pieces there are. Tiny specks and
  bubbles a texture can pinch off (under 1 mm³) are removed.
- *fuse = no*: just puts them in one file.
- Draft and Medium previews never fuse (it looks the same); Final and Export do.

Don't connect the center into the petal chain with Boolean Union: every node after Import STEP
runs once per petal, so each petal would carry the whole center and mesh the whole lamp's box,
which takes hours.

Any field can drive the size, e.g. `Coordinate y` → `Cos` → `Remap` for rings of large petals.

### Texture only on chosen faces (Face Mask)

**Face Mask** gives 1 on the CAD faces you choose and 0 elsewhere, with a smooth fade at the
edge of the selection. Multiply the texture's distance by it (e.g. `uneven_wave × face_mask`) to
put the texture only on those faces.

1. Connect its **Body** to the CAD bodies the texture is on (for the lamp: `scaled.body`).
2. Click **Pick**. The bodies appear in the 3D view (other previews turn see-through).
3. Choose faces. You always pick whole CAD faces from the STEP file, not triangles:
   - **click** a face to add it, click again to remove it;
   - **Ctrl+drag** to paint over several adjoining faces (starting on a chosen face removes instead);
   - **Shift+click** adds a face and every face joined to it tangentially. On parts with fillets
     everywhere (like the petals) that can be most of the part.
4. **Done** or **Esc**. The face numbers are stored in the node and can also be typed.
   With **Apply to: all other faces** the faces you pick are the ones left out (shown red while
   picking): easier when only a few faces should stay plain. With none picked, everything gets it.
   **✕** clears them (also **Clear** in the picking bar).
5. **☰** saves the faces under a name (e.g. `top`) and lists saved names: other Face Masks can
   use the name, and picking while a name is chosen updates the saved selection everywhere.

| Input | Effect |
|---|---|
| Fade | Width of the fade, mm. 0 = hard edge. |
| Position | *inside*: the fade is inside the chosen faces, so the texture never reaches the others. *center*: centred on the edge. *outside*: the chosen faces get the full texture and it fades out over the neighbours. |
| Band | Surface reach: how far from the surface the mask is computed; at least the largest offset. |
| Reach | *surface* (textures; also right on thin walls) or *volume* (lattices): 1 at the chosen faces, fading to 0 at *Fade* mm from them, straight through the part. Position and Band only apply to *surface*. |

With a list of bodies, a plain face number (`15`) means that face on every body. The lamp's 150
petals are identical (150 faces each, same order), so picking a face on one petal selects it on
all. When the bodies are different parts, picking stores `body:face` (`2:15` = face 15 of body 2)
so only the face you clicked is chosen. The mask measures the fade along the surface to the edge
of the selection, so thin walls behave: the inside of a petal wall is not "near" its outside face.

### Smaller files: Tolerance and Reduce Mesh

**Mesh from Implicit Body → Tolerance** (e.g. 0.01 mm; 0 = off) does the smart reduction while
meshing, in the worker processes (each block or body as it is meshed). On the key cabinet at Final:
18 s in all instead of 45 s for meshing + a Reduce Mesh node, 1.14M triangles, surface within
0.003 mm almost everywhere (0.03 mm at worst, near block seams). Used at Final and for exports.

**Reduce Mesh** (Mesh category; for meshes from elsewhere, e.g. merged or CAD meshes) removes
triangles only where the shape does not need them: *Tolerance* is how far the surface may move
(default 0.01 mm). Flat faces become a few big triangles, textures and curves keep theirs. On the
key cabinet: 5.7M -> 1.1M triangles (19%), the flat centre face 706k -> 2.8k, surface within
0.006 mm, watertight; about 28 s. Larger tolerances remove more (0.05: 8%, but on curved detail
the surface can move up to ~2x the value). *Keep* in Mesh from Implicit Body instead removes a
share everywhere, flat or not.

### Lattices

The **Lattice** category fills parts with lattices. Every thickness input takes a field as well
as a number, so a lattice can get thicker or thinner across the part (graded).

| Node | What it makes |
|---|---|
| **TPMS Lattice** | Smooth minimal-surface lattices: gyroid, schwarz_p, diamond, neovius, lidinoid. *sheet* = a wall of the given thickness following the surface; *solid* = the material on one side (*offset* adds or removes material). |
| **Strut Lattice** | Beam lattices: cubic, bcc, fcc, octet, cubic + bcc, with strut thickness and a *blend* that rounds the nodes. |
| **Voronoi Lattice** | An organic foam of struts along random cells. *randomness* 0 = regular box-like cells, 1 = fully irregular; *seed* = another foam. Needs a body (*domain*) to fill. |
| **Fill Body with Lattice** | Puts a lattice into a part (an implicit body), optionally with a solid *skin* of the given thickness along the surface. |

**Units: mm or density.** With *Units: density* a lattice's Thickness (TPMS solid modes: Offset)
is the share of solid material instead of mm: 1 = completely solid, 0.3 = 30% solid, worked out
from the lattice's own shape. In mm the solid share grows very unevenly (a Voronoi foam with 5 mm
cells is 42% solid at 2 mm struts, 98% at 5 mm, solid from ~7 mm), so fades look like a step.

**Solid at chosen faces, fading into the lattice:** Face Mask (*Reach: volume*, pick the faces)
→ Remap (*Auto: yes*, To Low = how open the far end is, e.g. 0.4, To High = 1) → the lattice's
*Thickness* with *Units: density*. The Face Mask's *Fade* sets how long the transition is, and the
lattice now opens up evenly over it. (In mm: To High about 1.2 × the cell size; anything thicker
looks the same.)

Typical chain: Import STEP → Implicit Body from CAD Body → **Fill Body with Lattice** (lattice from
one of the lattice nodes) → Mesh from Implicit Body. A TPMS or strut lattice fills all of space,
so to preview it on its own connect a *domain* body. All lattices run on the GPU when the grid is
big enough (a Voronoi foam of 5 million voxels: 0.04 s; on the CPU 0.2 s).

The **3D Wave** node gives regular patterns instead: `ridges` (the triangle wave along an axis)
or `gyroid` / `schwarz` / `diamond`, each optionally bent in 3D by noise with `warp`.

## Recipes (graph JSON)

Each node has a type and inputs. An input is a value or a link to another node's output:

```json
"offset": {
  "type": "implicit.offset",
  "inputs": {"body": {"link": "petals.out"}, "distance": {"link": "wave.out"}},
  "pos": [1600, 0]
}
```

- **Lists:** `Import STEP` outputs every body in the file. Nodes downstream run once per body
  automatically; `Export Mesh` merges them into one file. Use `Pick Body` to work on one, and
  `Merge Meshes` to combine meshes from different chains (optionally fused into one solid).
- **Bounds:** `band` (Implicit Body from CAD Body) should be at least the largest offset:
  distances are exact only within it. Offsets by a number, or by a field whose range is known
  (shown on its card), get room in the mesh box automatically; for others (e.g. a field driven
  by a coordinate) raise Mesh from Implicit Body's `padding`. A node warns on its card when the
  surface is cut off at the edge of its box.
- **Smoothen Body** takes iterations and a grid size, and looks the same at any preview quality.

## Development

```
python -m pytest tests                       # engine + server tests (CPU; not while the editor is busy)
python -m pytest tests/test_gpu.py --gpu     # GPU vs CPU tests, only on request
cd frontend && npm install && npm run build  # rebuild the UI after changing frontend/src
cd frontend && npm run dev                   # live-reloading UI on :5173 (keep app.py running)
```

New node types are plain functions registered with `@node(...)` in `engine/graph/nodes.py`;
they appear in the editor automatically.

Measurements, lamp geometry facts and lessons learned are collected in
[docs/FINDINGS.md](docs/FINDINGS.md).

## Layout

- `engine/cad_import.py` — STEP reading and tessellation (OpenCascade)
- `engine/sdf.py` — mesh → signed distance (narrow band, cached search trees)
- `engine/fields.py` — field expressions with fingerprints for caching
- `engine/cache.py` — shared array cache (RAM + SSD), imported STEP meshes
- `engine/shared.py` — big arrays sent to worker processes once
- `engine/lattice.py`, `engine/segments.py` — lattices; distance to the Voronoi struts (Warp)
- `engine/meshing.py` — marching cubes, decimation, fusing (Boolean union), export
- `engine/tiles.py` — meshing big bodies block by block
- `engine/graph/core.py` — ports, registry, graph model, evaluator
- `engine/graph/nodes.py` — the node library
- `engine/graph/workers.py` — sticky worker processes for multi-body files
- `engine/server.py` — web server behind the editor
- `frontend/` — the editor (React, React Flow, Three.js)
- `app.py`, `start.bat` — start the editor; `run_graph.py` — command-line runner
