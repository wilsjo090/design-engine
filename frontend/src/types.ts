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
}

export interface NodeSpec {
  type: string
  title: string
  category: string
  doc: string
  inputs: PortSpec[]
  outputs: PortSpec[]
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
}

export const GROUP_PREFIX = 'group:'
export const GROUP_IN = '$in'
export const GROUP_OUT = '$out'
export const SETTING_TYPE = 'input.number'

export interface EngineNodeData extends Record<string, unknown> {
  typeId: string
  /** literal values typed into the node; kept even while a link overrides them */
  inputs: Record<string, unknown>
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
  running: boolean
  startedAt: number
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
}

export interface ToastAction { label: string; run: () => void }

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
