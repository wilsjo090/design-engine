import type { Edge, Node } from '@xyflow/react'

export interface PortSpec {
  name: string
  type: string
  label: string
  default?: unknown
  doc?: string
  choices?: string[]
  many?: boolean
  optional?: boolean
  /** set by the engine (e.g. Topology Optimization's compute flag), not shown on the node */
  hidden?: boolean
  /** shown only while it applies: {other input: its values} (every one must hold), or a list of such
   *  (any one may hold), e.g. a pattern's Sides only for polygons and stars */
  when?: Record<string, string[]> | Record<string, string[]>[]
  /** tucked under the card's "more" row while it has its default value and no link */
  advanced?: boolean
  /** shown in the number field: "mm", or {other input: {its value: unit}} (a Load's Amount by its Kind) */
  unit?: string | Record<string, Record<string, string>>
  /** what 0 means ("auto", "no limit", "sharp"), shown in the field in place of 0 */
  zero?: string
  /** a list to choose the value from (components/PickLists.tsx): "body" (which item of the node's first cad
   *  input), "font" (an installed font) */
  pick?: 'body' | 'font'
  /** an int that is 1 or 0, shown as an on/off switch */
  switch?: boolean
  /** [low, high]: the usual values, a slider under the field (typing may go beyond) */
  slider?: [number, number]
}

/** The unit of a number input: its own, or the one for the value of the input it depends on
 *  (undefined while that input is linked). */
export function unitOf(port: PortSpec, value: (name: string) => unknown): string | undefined {
  const u = port.unit
  if (!u || typeof u === 'string') return u
  const [name, units] = Object.entries(u)[0]
  const v = value(name)
  return v === undefined ? undefined : units[String(v)]
}

/** Whether an input applies with the node's other values (`PortSpec.when`). `value` reads another
 *  input: undefined when it is linked (it may be anything, so the input applies). */
export function portApplies(port: PortSpec, value: (name: string) => unknown): boolean {
  if (!port.when) return true
  const conditions = Array.isArray(port.when) ? port.when : [port.when]
  return conditions.some((c) => Object.entries(c).every(([name, allowed]) => {
    const v = value(name)
    return v === undefined || allowed.map(String).includes(String(v))
  }))
}

/** An input set to something other than its default (shown even when it is `advanced`). */
export const changedFromDefault = (port: PortSpec, value: unknown) =>
  value !== undefined && JSON.stringify(value) !== JSON.stringify(port.default)

export interface NodeSpec {
  type: string
  title: string
  category: string
  doc: string
  inputs: PortSpec[]
  outputs: PortSpec[]
  /** Math: the scalar output is a plain number while every scalar input is one (see `outputType`) */
  numeric?: boolean
}

export interface Catalog {
  nodes: NodeSpec[]
  compatibility: Record<string, string[]>
  qualities: string[]
  root: string
}

export type Link = { link: string }

export interface RecipeNode {
  type: string
  inputs?: Record<string, unknown>
  pos?: [number, number]
  visible?: boolean
  /** passes its first input of its output's type straight on, as if it were not there (engine: bypass_input) */
  bypass?: boolean
  /** drawn as its header only in the node editor */
  collapsed?: boolean
}

/** A group: own node made from other nodes. Inner links "$in.<name>" are the group's inputs;
 *  Number nodes inside are its settings (one input per Number node, named by its id). */
export interface GroupDef {
  nodes: Record<string, RecipeNode>
  inputs: { name: string; type: string }[]
  outputs: { name: string; type: string; link: string }[]
  doc?: string
}

export interface Recipe {
  version: number
  nodes: Record<string, RecipeNode>
  groups?: Record<string, GroupDef>
  /** named face selections: name -> face list ("10, 11" or "2:15") */
  selections?: Record<string, string>
  /** boxes drawn behind nodes in the graph (graphFrames.ts), by id ("frame:1"); the engine ignores them */
  frames?: Record<string, GraphFrame>
}

/** A frame in the node graph: a labelled box behind nodes. Its nodes are the ones whose middle lies inside
 *  it (and the frames inside it), so dragging it moves them; folded, it is a title bar that hides them. */
export interface GraphFrame {
  label: string
  /** "#rrggbb", one of FRAME_COLORS */
  color: string
  pos: [number, number]
  /** unfolded size (a folded frame is drawn as its title bar) */
  size: [number, number]
  /** folded: the nodes and frames it hides (the ones inside it when it was folded) */
  folded?: string[]
  /** selected in the graph (not saved) */
  selected?: boolean
}

export const GROUP_PREFIX = 'group:'
export const GROUP_IN = '$in'
export const GROUP_OUT = '$out'
export const SETTING_TYPE = 'input.number'

export interface EngineNodeData extends Record<string, unknown> {
  typeId: string
  /** literal values typed into the node; kept even while a link overrides them */
  inputs: Record<string, unknown>
  /** see RecipeNode */
  bypass?: boolean
  collapsed?: boolean
}

/** The input a bypassed node passes on: its first linked input of the same type as its first output
 *  (as the engine's bypass_input), or null. */
export function bypassInput(spec: NodeSpec | undefined, linked: (port: string) => boolean): string | null {
  const out = spec?.outputs[0]
  return spec?.inputs.find((p) => out && p.type === out.type && linked(p.name))?.name ?? null
}

export type EngineNode = Node<EngineNodeData, 'engine' | 'groupIn' | 'groupOut'>
export type EngineEdge = Edge

export type Quality = 'draft' | 'medium' | 'final'
/** coarse to fine */
export const QUALITIES: Quality[] = ['draft', 'medium', 'final']
export const QUALITY_LABEL: Record<Quality, string> = { draft: 'Draft', medium: 'Medium', final: 'Final' }

export interface NodeStatus {
  state: 'running' | 'done' | 'cached' | 'error'
  time?: number
  message?: string
  progress?: [number, number]
  /** while running: what the progress counts ("Blocks", "iteration") */
  step?: string | null
  /** while running: the hidden mesher is meshing the node's body for the 3D view */
  view?: boolean
  /** while running: the step inside a group (the group node shows its inner nodes' events) */
  inner?: string | null
  /** the values the node's output can take, when the engine can tell (e.g. 0 ... 1 for a mask) */
  range?: [number, number] | null
  /** a line from the node about its result (e.g. "not optimized for these settings yet") */
  note?: string | null
  /** an optimizer's saved result has nothing left to do with these settings: why (there is no Optimize to press) */
  finished?: string | null
  /** warnings and errors the node reported while computing its result (kept while it is cached) */
  warnings?: Warning[] | null
  /** the same from meshing the node's body for the 3D view (e.g. "cut flat at the edge of its box") */
  viewWarnings?: Warning[] | null
}

/** [text, level]: level 'warning' or 'error' */
export type Warning = [string, string]

export interface ViewMesh {
  version: number
  kind: string
  faces: number
  positions: Float32Array
  indices: Uint32Array
  /** the field's normals at the vertices, int16 x 3 (a normalized attribute: x / 32767), when the engine sent them
   *  (views, pieces and details of implicit bodies); else the 3D view works them out from the triangles */
  normals?: Int16Array
  bounds: [[number, number, number], [number, number, number]]
  /** what the shown output is made of (the engine's content key) and the quality it was made at: a
   *  later preview at this quality or coarser keeps it instead of computing it again */
  key?: string
  level?: Quality
  /** looks the same at every quality (a CAD body) */
  exact?: boolean
  /** mm: the voxel it was meshed with, and the finest a detail of it may use (null: no detail) */
  voxel?: number | null
  floor?: number | null
  /** a list's view: the triangles of each item, one after the other (a later run's items are drawn over it
   *  as they come, see Pieces) */
  parts?: number[] | null
}

/** A list's items from a run still going on, drawn as each is made (the lamp's 144 petals, 13-24 s at Final),
 *  each instead of that item in the view shown (`base`), until the run's whole view comes. */
export interface Pieces {
  /** the version of the view they are drawn over (0: none shown yet) */
  base: number
  /** total items in the list */
  total: number
  meshes: Record<number, ViewMesh>
}

/** Part of a shown body meshed finer than its view (the part on screen, when zoomed in): drawn
 *  instead of the view inside the box lo..hi. */
export interface Detail {
  /** the engine's version of this detail, and of the view it was made from */
  version: number
  viewVersion: number
  /** what the view it refines is made of: it stays while the view shows the same */
  key: string
  lo: [number, number, number]
  hi: [number, number, number]
  voxel: number
  mesh: ViewMesh
}

export type Axis = 'x' | 'y' | 'z'
export type ColorMap = 'bands' | 'heat' | 'signed' | 'rainbow'

export interface SliceView {
  version: number
  axis: Axis
  position: number
  /** u0, v0, u1, v1 in mm (u, v = the plane's two axes in x, y, z order) */
  rect: [number, number, number, number]
  min: number
  max: number
  width: number
  height: number
  values: Float32Array
}

export interface SliceSettings {
  axis: Axis
  /** mm along the axis; null = middle of the scene */
  position: number | null
  cmap: ColorMap
  /** plane = a coloured slice; surface = the field coloured onto the visible bodies */
  mode: 'plane' | 'surface'
}

/** A value per vertex of a view, shown as colours on it (a field, the wall thickness, or a node's
 *  own result such as Static Analysis's stress). */
export interface Paint {
  version: number
  /** the view version these values belong to (vertex count must match) */
  viewVersion: number
  values: Float32Array
  min: number
  max: number
  label: string
  source: 'field' | 'thickness' | 'result'
  unit?: string
  /** low values are the bad ones (a safety factor): red at the low end */
  reverse?: boolean
  /** some values lie beyond min..max (a stress peak at a support), shown in the end colour */
  clipped?: boolean
  /** a field's colours: what the field is made of and the quality it was sampled at (see ViewMesh) */
  key?: string | null
  level?: Quality
}

/** A frame's bars (Frame Optimization) for the 3D view's frame view: pieces between nodes, each
 *  piece's bar, and per bar its size (mm), utilisation and length; axial force (N) per piece. */
export interface FrameView {
  profile: string
  nodes: [number, number, number][]
  bars: [number, number][]
  chains: number[]
  sizes: number[]
  utilisation: number[]
  axial: number[]
  lengths: number[]
  /** a frame being made (a `live_frame` event while Frame Optimization runs) */
  live?: boolean
  /** live: the bars have their real sizes (else they are as thick as their force needs) */
  sized?: boolean
  /** live: the step of the run (optimize.STAGES) and what it just did */
  stage?: string
  what?: string
  /** live, body check: its most stressed spot */
  spot?: [number, number, number] | null
  /** live: what the step's LP did (`layout.optimize`): LPs solved so far, bars in the LP of all candidates,
   *  bars member adding takes in next, volume of the bars at their forces and with every bar at least Min size (mm3) */
  numbers?: { lp?: number; in_lp?: number; candidates?: number; adding?: number; volume?: number; real?: number }
  /** live: the bars' mass as drawn, g */
  mass?: number
}

/** A frame's candidate nodes (Frame Optimization, Show: candidate nodes): the points bars may meet at,
 *  as Optimize lays them out (`nodes_frame._ground_view`). */
export interface GroundView {
  /** mm between the lattice's nodes */
  spacing: number
  /** the Node spacing was automatic */
  auto: boolean
  /** the automatic spacing that left a support or load without bars (this one is finer) */
  coarser: number | null
  /** lattice nodes inside the design space (that bars reach) */
  lattice: [number, number, number][]
  /** nodes at supports, loads and connections, each with its kind and its node's label */
  points: [number, number, number][]
  kinds: ('fixed' | 'load' | 'attach')[]
  labels: string[]
  /** points (indices) no bar reaches: Optimize stops with `error` */
  lonely: number[]
  /** candidate bars */
  bars: number
  error: string | null
}

/** Supports and loads drawn in the 3D view (from Fixed Faces and Load nodes). */
export interface Marker {
  kind: 'fixed' | 'load'
  label: string
  points: [number, number, number][]
  dirs: [number, number, number][]
  /** the arrow pushes into the surface (its tip touches it); else it pulls away from it */
  push: boolean[]
  /** how hard the load presses at each point, 0..1 of its most (arrow lengths; absent = all alike) */
  sizes?: number[]
  /** loads: their kind, the centre of their faces and the direction of the force (a torque's axis) */
  load?: string
  centre?: [number, number, number]
  arrow?: [number, number, number] | null
  /** the CAD face and body number under each point */
  faces?: number[]
  bodies?: number[]
}

/** What the editor sends with a run and gets back with its run_start. */
export interface RunTag {
  /** graphVersion when the run was asked for: its result is out of date once this changes */
  version: number
  /** a step of refining the view (not the first look after a change) */
  refining?: boolean
  /** the editor tab that asked for the run (store.CLIENT): the others take none of its events */
  client?: string
}

export interface QueueState {
  current: { id: number; label: string } | null
  exports: { id: number; label: string; queued: number }[]
  preview_waiting: boolean
  /** the run going on now, of any kind (an editor opened during it adopts it) */
  running?: { run: number; mode: string; quality: Quality; label: string; tag: RunTag | null; elapsed: number } | null
}

export type Bounds = [[number, number, number], [number, number, number]]

export interface RunState {
  id: number
  mode: string
  /** "optimize <node>" for an optimization */
  label?: string
  running: boolean
  startedAt: number
  /** Date.now() when its run_done came */
  endedAt?: number
  message?: string
  quality?: Quality
  /** undefined from an engine that does not send it back (one not restarted since an update) */
  tag?: RunTag | null
  /** what the engine does after the nodes (preparing views, colouring, slicing) */
  phase?: string
  /** refining: the quality the next preview runs at, once this one is done and nothing changed */
  next?: Quality
  /** after the run: the coarsest quality of what the view shows (null: nothing that changes with it) */
  shown?: Quality | null
  /** a detail run: the voxel it was meshed with, mm */
  voxel?: number
  /** Stop was pressed (the engine said so): the run ends at its next stopping point */
  stopping?: boolean
}

export interface LogEntry {
  time: number
  text: string
  level: 'info' | 'warning' | 'error' | 'success'
}

export const isLink = (v: unknown): v is Link =>
  typeof v === 'object' && v !== null && 'link' in v && Object.keys(v).length === 1

export const DISPLAYABLE = ['mesh', 'cad', 'implicit']

export const CATEGORY_COLORS: Record<string, string> = {
  Inputs: '#c9a227',
  Fields: '#9c6ade',
  Math: '#5fb83f',
  Implicit: '#26a69a',
  Mesh: '#3d8fe0',
  Output: '#e8833a',
  Transform: '#d05f8c',
  Lattice: '#e5534b',
  Patterns: '#c08a5b',
  Simulation: '#4cc9e0',
  Groups: '#d4a5ff',
  Presets: '#9ec7ff',
}

export const TYPE_COLORS: Record<string, string> = {
  scalar: '#8bc34a',
  implicit: '#2ec4b0',
  number: '#b0b6bf',
  int: '#b0b6bf',
  cad: '#d9a925',
  mesh: '#4ea3f5',
  path: '#b48ce8',
  text: '#b48ce8',
  faces: '#d9a925',
  part: '#2ec4b0',
  material: '#c9a27e',
  conditions: '#ef6f6c',
  frame: '#8fb8de',
}

/** For face picking: the CAD face and body of every triangle of a CAD view, and the smooth-join
 *  group of every face (one list if the bodies are copies of one part, else one per body). */
export interface FaceData {
  version: number
  ids: Uint32Array
  bodies: Uint32Array
  /** the bodies are copies of one part: a plain face number means that face on all of them */
  same: boolean
  groups: Uint32Array[]
}

/** An entry of a right-click menu in the 3D view (ViewportMenu). */
export type MenuEntry = { kind: 'header'; text: string } | { kind: 'separator' } | MenuItem

export interface MenuItem {
  kind: 'item'
  label: string
  /** small grey text on the right (a shortcut, a node type) */
  hint?: string
  /** greyed out: why it cannot be used (its tooltip) */
  disabled?: string
  /** tooltip of an enabled item */
  title?: string
  /** a check mark in front (toggles such as Wireframe) */
  checked?: boolean
  /** a submenu (opens on hover or →) */
  items?: MenuEntry[]
  /** shift: Shift was held (e.g. skips the quick-edit popover) */
  run?: (o: { shift: boolean }) => void
}

/** A small graph change made from the 3D view: one undo step, one preview run. Nodes are named by
 *  id, or by the `key` of a node the same edit adds. */
export interface GraphEdit {
  /** new nodes, placed next to `near`: one column to its right ('right', the default) or in its row
   *  after it ('after': the next node of a supports-and-loads chain), never on top of another node */
  add?: { key: string; type: string; inputs?: Record<string, unknown>; near?: string; place?: 'right' | 'after' }[]
  /** typed-in values */
  inputs?: { node: string; port: string; value: unknown }[]
  /** links (from.port -> to.input); each replaces whatever was linked into to.input */
  links?: { from: string; port: string; to: string; input: string }[]
  remove?: string[]
  /** saved face selections to set */
  selections?: Record<string, string>
  /** select these nodes (and no others) */
  select?: string[]
  /** start picking faces on this node afterwards */
  pick?: string
  /** show these nodes in the 3D view, hide these */
  show?: string[]
  hide?: string[]
  /** turn a node's bypass on or off */
  bypass?: { node: string; on: boolean }[]
}

export interface ToastAction { label: string; run: () => void }

/** One input in the quick-edit box: of `node`, with its own label (else the port's). */
export interface QuickField { node: string; port: string; label?: string }

/** The quick-edit box in the 3D view at client x, y: a node's main inputs (QUICK_INPUTS, else all its
 *  plain ones), or `fields` of several nodes under `title` (e.g. a texture's amplitude and scale). */
export interface QuickEditState { node: string; x: number; y: number; fields?: QuickField[]; title?: string }

/** A CAD face: its number and its body's number, as /api/faces reports them. */
export interface FaceRef { face: number; body: number }

/** Faces chosen in the 3D view with Ctrl+click or Select similar, for one right-click action on all of
 *  them: the clicked view, the CAD output they are faces of (`source`.`port`), whether its bodies are
 *  copies of one part and how many there are; `marks`: the triangles of shown views that highlight each
 *  face (by "body:face"). */
export interface FaceSelection {
  view: string
  source: string
  port: string
  same: boolean
  bodyCount: number
  faces: FaceRef[]
  marks: Record<string, { view: string; version: number; triangles: Uint32Array }>
}

/** A small text box in the 3D view asking for a name (Save as selection…). */
export interface PromptState { x: number; y: number; label: string; value: string; done: (text: string) => void }

/** Turn the 3D view's camera: look along -`dir` (the camera sits on the dir side of the target) at
 *  `target` (else at what it looks at now). */
export interface CameraRequest { dir: [number, number, number]; target?: [number, number, number]; at: number }

/** A short note in the 3D view after an action there: what changed, with Undo. */
export interface Toast {
  text: string
  /** every change, one line each (its tooltip) */
  detail: string[]
  actions: ToastAction[]
  /** Undo is offered while graphVersion is still this (nothing changed since) */
  version: number
  at: number
}

/** Face keys in a faces text: "3" (face 3 on every body) or "2:15" (face 15 of body 2), sorted. */
export function parseFaceKeys(v: unknown): string[] {
  const keys = new Set<string>()
  for (const m of String(v ?? '').matchAll(/(\d+)\s*:\s*(\d+)|(\d+)/g)) keys.add(m[3] ?? `${Number(m[1])}:${Number(m[2])}`)
  const num = (k: string) => k.split(':').map(Number)
  return [...keys].map((k) => (k.includes(':') ? k : String(Number(k)))).sort((a, b) => {
    const [a1, a2 = -1] = num(a), [b1, b2 = -1] = num(b)
    return a.includes(':') === b.includes(':') ? a1 - b1 || a2 - b2 : a.includes(':') ? 1 : -1
  })
}

/** Names of saved selections in a faces text resolved to their face keys. */
export function resolveFaces(v: unknown, selections: Record<string, string>): string {
  return String(v ?? '').replace(/[A-Za-z_][\w-]*/g, (name) => selections[name] ?? name)
}

/** "3, 7 12" -> [3, 7, 12] (sorted, unique; "2:15" counts as 15) */
export const parseFaces = (v: unknown): number[] =>
  [...new Set(parseFaceKeys(v).map((k) => Number(k.split(':').pop())))].sort((a, b) => a - b)
