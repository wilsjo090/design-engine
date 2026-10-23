# Right-click menus in the 3D view: plan

Status 2026-10-23: M1, M2 and M3 built (see Status at the end). Decisions marked **(open)** are the user's.
Code read for this plan: `frontend/src/components/Viewport.tsx` (views, face picking, markers,
the Load arrow, frame bars), `frontend/src/store.ts` (`addNode`, `connect`, `setInput`,
`pasteClip`, `picking`, undo), `NodeEditor.tsx` (palette and auto-wiring), `engine/server.py`
(`/api/faces`, `pack_faces`, `view_faces`), `engine/graph/nodes_fea.py`, `nodes_pattern.py`,
`nodes_frame.py`, the FEA recipes and the Topology / Static analysis presets.

## Why

Today every action starts in the node graph: add a Load, wire its Body, click Pick, find the face,
click Done, wire Add to, wire the analysis. Seven steps across two panes for "this face carries
200 N". The 3D view already knows almost everything needed to do that in one step: which node a
view belongs to, which CAD face and body every triangle is (`/api/faces`), which faces are
tangent-joined, and which nodes hold supports and loads (`markers`). What is missing is a way to
ask the view for an action.

Goal: right-click on what you see (a face, a body, a load arrow, a bar, empty space) and get the
actions that make sense *there*, which then create or change nodes in the graph exactly as if you
had done it by hand: same nodes, same wiring, one undo step, nothing hidden.

## What the user does in the editor

1. Show the part (eye on Import STEP or Pick Body), right-click its mounting face:
   **Fix this face**. A Fixed Faces node appears in the graph, wired to the body, face filled in,
   green cones on the face.
2. Right-click the hook hole: **Add load here ▸ Bearing (pin in hole)**. A Load node appears,
   `add_to` chained to the Fixed Faces, selected, so the yellow direction arrow is already on the
   hole: drag it to -Z, type 200 N in the small box next to the cursor.
3. Right-click empty space or the body: **Optimize this body ▸ Topology optimization**. Material
   and Topology Optimization appear, wired to the body and to the end of the supports-and-loads
   chain. Press Optimize.
4. Later: right-click a load arrow: **Edit 'hang'**, **Flip direction**, **Remove this face**,
   **Delete load**. Right-click a face: the menu says it is used by `wall` (Fixed Faces) and
   `mask1` (Face Mask) and offers to remove it from either.

The graph stays the source of truth: every menu action is a small, visible graph edit, and the
graph pane flashes the nodes it touched.

## Which things get a menu

One right-click handler on the canvas finds the thing under the cursor by priority (the topmost
drawn, not the nearest: arrows and the gizmo are drawn over the part with `depthTest: false`):

| Priority | Target | Found by | Menu |
|---|---|---|---|
| 1 | Load arrow gizmo | its hit spheres | (none: it is already interactive; right-click falls through) |
| 2 | Support cone / load arrow | new hit proxies tagged with the node id | Marker menu |
| 3 | Frame bar | `FrameBars` instance id (exists) | Bar menu |
| 4 | CAD face on a CAD view | triangle → `FaceData` (exists for picking) | Face menu + Body menu |
| 5 | Implicit body / mesh / variant | the view under the cursor | Body menu (+ Face menu later, see phase 3) |
| 6 | Field slice | `SliceObject` | Slice menu |
| 7 | Nothing | R3F `onPointerMissed` with `type === 'contextmenu'` | View menu |

**Edges are not in this plan.** The engine has no CAD edges (only faces per triangle), and
nothing in it takes an edge: supports and loads act on faces, and edge rounds live on the
Booleans. A face-border could be found from triangle adjacency, but there is no action to hang on
it yet. If you want "fix this edge" or "round this edge" later, that is engine work first
**(open)**.

### Face menu (CAD views)

Header line (not clickable): `Face 15 · body 2 · cylinder Ø 8.0 mm · 3.2 cm²` (surface type and
size from the new face info, phase 2; phase 1 shows `Face 15 · body 2`).

- **Fix this face** → Fixed Faces.
- **Add load here ▸** Force · Pressure · Bearing (pin in hole) · Torque → Load with that `kind`.
  The order adapts to the face: a cylinder that is a hole (normal pointing to the axis) lists
  Bearing and Torque first, a plane lists Force and Pressure first. Defaults from the face: a plane
  gets `direction = into the face`, a hole bearing gets the axis-free default `-Z` **(open: or ask
  every time)**.
- **Add to ▸** one line per existing Fixed Faces / Load / Face Mask / Hole Pattern / Surface
  Pattern / Picture / Text node on the same body ("Add to wall (Fixed Faces)"), so a second bolt
  hole does not make a second node. If the node's faces are a saved selection, the selection is
  updated (same rule as picking today, `writeFaces`).
- **Used by ▸** the nodes whose faces contain this face, each with *Select node* and *Remove this
  face*. Hidden when nothing uses it.
- **Decorate ▸** Face Mask here · Hole pattern here · Surface pattern here · Text here · Picture
  here. Creates the node with Body and Faces filled in, placed next to the body's node, selected;
  its output is left for you to wire (where a texture goes is a design choice), and the log says
  which input to connect next. If exactly one Offset Body / Boolean on this body has a free
  matching input, offer "…and connect to *offset*" as a second line.
- **Select ▸** Tangent faces (the Shift+click group) · Same face on every copy (only when the
  bodies are copies, `faces.same`) · Save as selection…
- **Pick more faces…** at the end of every action that created a node: the same action, then
  picking mode on the new node, so you can click the other bolt holes right away.

With copies of one part (the petals, `faces.same`), every action that writes faces asks once
through a two-line choice at the top of its submenu: *on this body only* (`2:15`) or *on every
copy* (`15`). Default: this body only for supports and loads, every copy for decorations
**(open)**.

### Body menu (any shown body)

Header: node id and what it is (`space · Pick Body · 12.4k tris`).

- **Select node in graph** (and pan the graph to it). **Hide**. **Show only this**.
- **Print check this body** (opens Check with this body chosen; meshes and implicit bodies only).
- **Keep variant**.
- **Analyse ▸** Static analysis · Topology optimization · Frame optimization (CAD bodies, and
  implicit ones for Static). Creates Material (PLA, or the Material already in the graph) and the
  analysis node, wired to this body and to the tail of an existing supports-and-loads chain on the
  same body. Uses the existing `presets/Static analysis.json` / `Topology.json` as templates.
- **Use as ▸** Design space of *topology* · Keep region of *topology* · Avoid region of *topology*
  · Connect to of *frame*. Keep and Avoid take an implicit body, so a CAD body goes through an
  inserted *Implicit Body from CAD Body*; if Keep is already wired, a submenu offers *replace* or
  *add* (inserts a Boolean Union of the old and the new). This is how keep / exclude regions get a
  viewport path: they are bodies in this engine, not faces.
- **Mesh and export…** (Mesh from Implicit Body + Export Mesh, implicit bodies only).
- For a STEP with several bodies shown through Import STEP: **Pick this body** (adds Pick Body with
  the right index; `cad_body` is the index in the file, so the clicked triangle's body number is
  the index).

### Marker menu (support cones, load arrows)

- **Edit *hang*** (selects the node: the yellow arrow appears for a Load).
- **Change to ▸** Force · Pressure · Bearing · Torque (Load), **Flip direction** (into ↔ out of,
  +Z ↔ -Z, custom vector negated).
- **Remove this face** (the face whose marker point was clicked) / **Pick faces…**.
- **Delete** (removes the node and splices the Add to chain: the next node's `add_to` and the
  consumer's `conditions` go to what this node's `add_to` was, so the chain never breaks).

### Bar menu (frame bars)

Bar info as the header (size, length, force, utilisation, as in `FramePanel`), **Select frame
node**, **Hide bars**.

### Slice menu

The value under the cursor as the header, **Move the slice here** (sets `position` to the hit's
coordinate along the slice axis), **Colour the bodies with this field** (switches to surface mode),
**Hide**.

### View menu (empty space)

**Fit view** (F), **Wireframe**, **Section ▸** X · Y · Z (on at the centre), **Show loads**,
**Show bars**, **Add node…** (opens the node palette over the 3D view; a new node lands in the
graph's visible centre), and **Import STEP…** when the graph has no Import STEP.

While picking, the face menu is replaced by the picking menu: **Add / Remove this face**,
**Add tangent faces**, **Clear**, **Done**. While a group is open for editing, only view items and
*Select node* show (picking is off there today too).

## How actions edit the graph

### One atomic edit

All actions go through a new store action, so each is one undo step and one preview run:

```ts
interface GraphEdit {
  add?: { key: string; type: string; inputs?: Record<string, unknown>; near?: string }[]  // key: local name
  inputs?: { node: string; port: string; value: unknown }[]     // node: existing id or an add key
  links?: { from: string; port: string; to: string; input: string }[]  // replaces what was linked
  remove?: string[]                                              // node ids
  select?: string[]; pick?: string; flash?: boolean
}
applyEdit(edit: GraphEdit, label: string): Record<string, string>  // add key -> new node id
```

It reuses `newNodeId` / `freeId`, `makeEdge` (type-checked against `specs` and
`catalog.compatibility`, like `connect`), and `remember(s, 'viewport:<label>')`. `pasteClip`
cannot do this because it never changes existing nodes' links (needed to append to a chain).
Every action is a pure function `(target, state) → GraphEdit | null`, in a new
`frontend/src/viewActions.ts`, so the menu can show an action greyed out with the reason when it
returns null ("Connect a body first", "This view has too many triangles to pick faces").

### Which body a new node links to

The clicked view belongs to a node (`visible` id); its first output of type `cad` is the source.
Rules, in order:

1. If the face's body already has supports, loads or face nodes, link to the **same source they
   use** (so faces stay consistent with the analysis). Example: Import STEP is shown but the chain
   uses `space` (Pick Body index 0): a click on body 0 links to `space` with plain face numbers;
   a click on body 1 offers *Pick this body* first.
2. Otherwise link to the clicked view's node. With a list of different bodies the face key is
   `body:face`, as picking writes it today (`faceKey`).

### Where the supports-and-loads chain goes

Fixed Faces and Load chain through `add_to`; the last node of the chain feeds `conditions` of
Static Analysis / Topology / Frame Optimization.

- New support or load, a chain exists on that body: **append to its tail**: `new.add_to = tail`,
  and every consumer whose `conditions` was linked to the tail is relinked to `new`. One consumer
  per chain is the normal case; with two chains on one body (two analyses), a submenu asks which.
- No chain: the node stands alone; the toast offers **Analyse ▸** right away.
- Delete from the marker menu splices the chain (above).

### Where new nodes go in the graph

Next to the node they link to: one column (320 px) to its right, below the lowest node already in
that column, never on top of another node. Chain nodes go in the chain's row, after its tail (the
existing recipes lay them out that way). The graph pane does not jump; new nodes get a short
highlight and a toast in the viewport: "Added *load2* (Load) ← *space* · Undo".

After the edit the new node is selected, which already does the useful thing for loads: the
yellow direction arrow appears on the faces (`aimNode` in Viewport.tsx).

### Quick edit popover

After *Add load here*, a small popover opens at the cursor with the 2-3 inputs that matter
(Load: Amount with its unit for the kind, Direction; Fixed Faces: nothing, so no popover;
Hole Pattern: Shape, Size, Spacing). It writes with `setInput`, closes on Enter / Esc / click away
and is skipped with Shift **(open: popover, or rely on the node card)**.

## Technical design

### 1. Right-click without breaking right-drag pan (`Viewport.tsx`)

OrbitControls pans with the right button, and on Windows `contextmenu` fires on button *release*,
also after a pan. So: record the pointer position on right `pointerdown` on the canvas; on
`contextmenu` open the menu only if it moved under 5 px (the same idea as `e.delta > 4` on
clicks). R3F gives `onContextMenu` on objects with `delta` and calls `onPointerMissed` for misses,
so one handler on a root `<group>` plus the Canvas' `onPointerMissed` covers everything.
OrbitControls already calls `preventDefault` on the browser's own menu.

The handler takes `e.intersections`, drops hits that the section plane cuts away (raycasts ignore
`clippingPlanes`: test the point against the plane) and ghost views, then picks the target by the
priority table above.

### 2. Raycasting big bodies

`ViewObject` has no search tree (only `PickableView` builds one), and a million-triangle body
raycasts slowly without it. Build the tree lazily on the first right-click on that view and keep
it with the geometry. It must be `computeBoundsTree({ indirect: true })`, as in `PickableView`:
the default build reorders the index and triangle numbers no longer match CAD faces (the
"scattered patches" bug noted in the code). Move that call into one helper used by both.

### 3. Face data for any CAD view

Move `fetchFaces` out of Viewport.tsx into `api.ts` and cache the result in the store per view id
and version (`faceData[id]`), shared by picking and the menu. Fetched on the first right-click on a
CAD view (the server keeps it cached per view version already). A 409 (decimated view) leaves only
the body menu, with the reason shown greyed out.

### 4. Face info (server, phase 2)

New `GET /api/faceinfo/{node}?body=&face=` → `{type, area, centre, normal | axis, radius, hole}`
from the body's triangles of that face and `metadata["cad_surfaces"]` (surface type per face).
`hole` = a cylinder whose normals point to its axis. Used for the header line, the order of load
kinds and the default direction. Small, cached with the view's `view_faces` entry.

### 5. Marker hit proxies

`MarkerArrows` draws `ArrowHelper`s with no node id. Add `userData = { node, face }` to each, and
one invisible sphere per marker point (radius ~ the arrow's head) for hits; the cone and arrow
lines are hard to hit otherwise. `markers` needs the face per point for *Remove this face*: add it
to `Marker` in `fea.model.markers` (one int per point).

### 6. Menu component

`frontend/src/components/ViewportMenu.tsx`: an HTML overlay at the cursor (clamped to the
viewport), sections with separators, submenus on hover and →, keyboard (↑ ↓ → ← Enter Esc), greyed
items with their reason as a tooltip. Same look as `NodePalette` (`styles.css`). While it is open
the hovered face stays highlighted (the `PICK_HOVER` overlay), so you see what the action is for.

### Files

| File | Change |
|---|---|
| `frontend/src/components/Viewport.tsx` | right-click dispatch, lazy BVH helper, marker proxies, menu and popover mount |
| `frontend/src/components/ViewportMenu.tsx` | new: menu and quick-edit popover |
| `frontend/src/viewActions.ts` | new: target → menu items → `GraphEdit` (pure) |
| `frontend/src/store.ts` | `applyEdit`, `faceData` cache, `flash` ids, `paletteRequest` from the view |
| `frontend/src/api.ts` | `fetchFaces` (moved), `faceInfo` |
| `frontend/src/components/NodeEditor.tsx` | highlight flashed nodes, pan to *Select node* |
| `frontend/src/styles.css` | menu, popover, flash |
| `engine/server.py` | `/api/faceinfo` (phase 2), `/api/face_at` (phase 3) |
| `engine/fea/model.py` (`markers`) | face number per marker point |
| `HelpPanel.tsx`, CLAUDE.md (Frontend) | document the menus |

## Phases

### M1: the core loop (supports and loads)
Right-click detection, lazy BVH, face data cache, menu component, `applyEdit`. Face menu with
**Fix this face**, **Add load here** (four kinds, no smart order yet), **Add to** existing,
**Used by** / remove, **Pick more faces…**; chain append; placement and toast; View menu with Fit,
Wireframe, Section. That alone turns the seven steps into two right-clicks.

### M2: context and the rest of the FEA workflow
Face info endpoint (header, hole detection, ordered kinds, default directions), quick-edit
popover, Body menu (**Analyse**, **Use as** design space / keep / avoid, Pick this body, Select
node, Hide, Show only), marker menu with chain splice on delete, copies choice (`2:15` or `15`).

### M3: decoration and implicit views
**Decorate ▸** (Face Mask, Hole / Surface Pattern, Text, Picture). Face actions on *implicit*
bodies (the textured lamp, a topology result): new `GET /api/face_at/{node}?x=&y=&z=` walks the
graph upstream from the shown node to its CAD sources (`implicit.from_cad`, Booleans, Offset) and
returns the nearest CAD face to the clicked point (the igl AABB that `facemask` already builds per
body). Then "Apply texture to this face" works where you actually look at the texture. Slice and
bar menus.

### Later (by need)
A selection-first mode (plain click selects faces without a node, then right-click acts on all
selected); edges if the engine gets edge actions; face-based Keep ("keep solid around this face",
needs a Topology input that takes faces); right-click on a result paint to pin a value label.

## How to check it

There are no frontend tests (`tsc -b` is the only check), so:

- `viewActions.ts` stays pure, so each action can be checked by hand in the browser console on a
  loaded recipe: `topo_wall_bracket_bolted.json` (Pick Body + chain + topology), `fea_holes.json`
  (bearing on holes), `lamp_petals.json` (copies: `15` vs `2:15`), a two-body STEP through
  Import STEP (body keys).
- Rebuild the FEA recipes from scratch with menus only and compare the saved JSON with the
  originals (same nodes, links and face numbers; positions may differ).
- Right-drag pan never opens the menu; the section plane's cut-away part is never hit; a 1M
  triangle body opens its first menu in under ~0.5 s and later ones instantly.
- Undo after every action restores the graph exactly (one step).
- Use the "design-engine-test" launch config (port 8766) and back up the autosave folder first, as
  CLAUDE.md says.

## Risks

- **Triangle order.** Any BVH built without `indirect: true` silently breaks face numbers. One
  shared helper, and a check that `faceData.ids.length === view.faces` before any face action
  (as picking does).
- **Wrong body source.** Linking a Load to Import STEP when the analysis uses Pick Body would mix
  face keys. Rule 1 above, and the toast always names the source.
- **Surprise edits.** Actions that rewire existing nodes (chain append, Keep union) are the ones
  that could confuse. The toast lists every changed link, and undo is one step.
- **Clutter.** Menus that list everything are slow to read. Cap the top level at ~8 items, move
  the rest into submenus, hide (not grey out) what can never apply to the target.

## Open questions for the user

1. **Popover or not**: after *Add load here*, a small Amount / Direction box at the cursor, or just
   select the node and edit it on its card?
2. **Copies**: on the petals, should supports and loads default to *this body only* and
   decorations to *every copy*?
3. **Bearing default direction**: -Z, or always ask?
4. **Selection-first**: should plain left-click on a face (outside picking) start selecting faces
   for a later right-click action, or stay as it is (nothing)? It is the bigger change of habit, so
   it is in *Later*.
5. **Edges**: is there an edge action you actually want (fix an edge, round it)? Otherwise edges
   stay out.

## Status (2026-10-23)

**Built (M1, checked in the test editor):** right-click on the canvas only when the right button
moved ≤ 5 px (right-drag pan opens nothing); own raycast of meshes tagged `userData.menu`
(`MenuProbe`), search tree built on the first right-click (`ensureBoundsTree`, shared with picking),
section-cut hits skipped; `store.faceData` cache (`api.loadFaces`); `ViewportMenu` (submenus,
keyboard, greyed items with reasons), toast with the change list, Undo and *Pick more faces* (the
"Pick more faces…" item lives in the toast, so the menu does not list every action twice); graph
flash and pan (`focusNode`); `store.applyEdit` (one undo step, placement without overlaps: chain
nodes in the tail's row, under the tail when that row is taken). Face menu: Fix this face, Add load
here ▸ (four kinds; two chains on the body ask *After …* / *On its own*), Add to ▸, Used by ▸
(Select node, Remove this face, Pick faces…); source rule 1 (a Pick Body of the clicked body that is
already used wins over Import STEP), `body:face` keys on different bodies. Picking menu. View menu
(Fit, Wireframe, Section ▸, Show loads, Show bars), also as *View ▸* under a body's menu.

**Built (M2, checked in the test editor):** `/api/faceinfo` (header `Face 10 · cylinder Ø 8.0 mm
(hole) · 7.5 cm²`; holes list Bearing and Torque first). Quick-edit box (`QuickEdit.tsx`,
`QUICK_INPUTS`, `store.quickEdit`) after *Add load here* and on a load marker's *Edit*: Amount with
the kind's unit, Direction, Vector for custom; Enter commits and closes, Esc drops unfinished typing;
Shift on the menu item skips it (`MenuItem.run({ shift })`). Body menu: Select node in graph, Hide,
Show only this, **Analyse ▸** Static / Topology / Frame (the graph's material or a new PLA one, the
body's chain tail into Supports & loads, *After …* when the body has two chains; toast *Show result*
/ *Optimize*), **Use as ▸** Design space / Keep region / Avoid region / Guide (Prefer) / Connect to of
each Topology or Frame node (a CAD body through Implicit Body from CAD Body; a filled Keep asks
*Replace* or *Add to … (Boolean Union)*; nodes the body comes from are left out), **Pick this body**
(a Pick Body with the clicked body's index), *Mesh and export…* and *Print check* (implicit bodies),
Keep variant. Marker menu (invisible balls over each drawn arrow, `MarkerArrows`; they win over the
part, which they are drawn over): Edit (selects, opens the box for loads), Change to ▸ kind (a
bearing gets −Z when it was into / out of the face), Flip direction (opposite axis, into ↔ out of,
custom vector negated; not for pressure), Remove face N (the face under that arrow, from the
server's `faces` / `bodies` per marker point), Pick faces…, Delete with chain splice. Source rule
completed: a Pick Body of the clicked body is used even before anything is linked to it, and when
other bodies of a list already have their own Pick Body, face actions add one for this body in the
same edit ("new nodes use a new Pick Body (body 0)"). Copies (open question 2, the plan's default):
Fix / each load kind ask *On this body only* (`2:15`) or *On every copy* (`15`); Add to follows the
node's own style. A new bearing load gets Direction −Z (open question 3: the plan's suggestion).

**Built (M3, checked in the test editor):** `GET /api/face_at/{node}?x=&y=&z=` (`Session.face_at`,
`cad_sources`: breadth-first up the last run's graph from what the view shows, stopping at the first
`cad` output on each path, skipping group inner nodes; `facemask.nearest_face`: bodies ranked by
bounding-box distance, an igl AABB per body in an LRU capped at ~2M triangles; answers source, port,
face, body, distance, the copies test and `face_info`). Right-clicking an implicit body or mesh opens
the face menu for the CAD face under the click ("the CAD face of import under offset", with the
distance when over 0.5 mm): supports, loads, Add to / Used by and Decorate all link to that CAD
source; the body part of the menu still treats the shown body as itself (Analyse the textured body,
not its CAD body). **Decorate ▸** Face Mask / Hole pattern / Surface pattern / Text / Picture here:
the node with Body and Faces filled in, next to the body's node, selected, with the quick-edit box
(shape, size, spacing, depth, height; text and height; picture and width); the toast and the log say
what to connect next. When exactly one input on this body is free for it, a second line wires it
straight on: "…and cut out with cut" (Boolean Subtract's B; Union: "add with"), "…and connect to
offset's Distance" (Offset Body). On copies decorations ask *On every copy* first. Slice menu (the
value under the cursor, *Turn it ▸ Across X/Y/Z* through the clicked point, colour the bodies,
Hide); *Move the slice here* on any body while a field slice is shown. Bar menu (bar, size, length,
force, utilisation; Select frame node; Hide bars). View menu: *Add node…* (opens the graph's
palette), *Import STEP…* when the graph has none.

**Left-click (the user's answer to open question 4, 2026-10-23):** a still left-click selects the
node that made what is under the cursor, not faces: on a body the node that made the surface at
that spot (`/api/node_at`: an Offset Body for a texture's bump, the Hole Pattern for a hole's wall,
the CAD body's node for a plain face, Topology for its result), on a marker / slice / bar their
node. The graph pans to it and flashes it; a toast names it (the walk as its tooltip).

**Not done:** the rest of the *Later* list (edges, face-based Keep, pinned value labels).
**Not checked:** the first right-click on a million-triangle body (a 122k one was instant),
`/api/face_at` on the lamp (145 bodies; the box ranking should keep it to a few trees), the
two-chains submenu, the copies choice (lamp_petals is too heavy to open for a menu test; no small
STEP with copies in samples/).
