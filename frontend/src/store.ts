import { create } from 'zustand'
import { applyEdgeChanges, applyNodeChanges, type Connection, type EdgeChange, type NodeChange } from '@xyflow/react'
import {
  GROUP_IN_ID, GROUP_OUT_ID, flowToGroup, flowToRecipe, groupSpec, groupToFlow, makeEdge, newNodeId, recipeToFlow,
} from './graphConvert'
import type {
  Bounds, Catalog, Detail, EngineEdge, EngineNode, FaceData, FrameView, GraphEdit, GroupDef, LogEntry, Marker, NodeSpec, NodeStatus,
  Paint, Quality, QueueState, Recipe, RunState, RunTag, SliceSettings, SliceView, Toast, ToastAction, ViewMesh, Warning,
} from './types'
import { DISPLAYABLE, GROUP_IN, GROUP_PREFIX, QUALITIES, QUALITY_LABEL, SETTING_TYPE, isLink } from './types'

/* eslint-disable @typescript-eslint/no-explicit-any */
type EngineEvent = Record<string, any>

/** A kept copy of what the 3D view showed, displayed next to the current result (variants). */
export interface Variant {
  id: number
  label: string
  views: ViewMesh[]
  recipe: Recipe
  shown: boolean
}

interface State {
  catalog: Catalog | null
  /** node specs: the catalog's plus one per group */
  specs: Record<string, NodeSpec>
  nodes: EngineNode[]
  edges: EngineEdge[]
  visible: string[]
  groups: Record<string, GroupDef>
  selections: Record<string, string>
  /** a group open for editing: the canvas shows its nodes; the recipe's own are kept aside */
  editing: { name: string; outer: { nodes: EngineNode[]; edges: EngineEdge[]; past: Snapshot[]; future: Snapshot[] } } | null
  recipePath: string | null
  dirty: boolean
  /** bumps whenever something changes that affects results (not node positions) */
  graphVersion: number
  status: Record<string, NodeStatus>
  problems: Record<string, string[]>
  /** the preview quality; with `refine`, the finest one the view is refined to */
  quality: Quality
  /** after a change, show a Draft first and then refine the view step by step up to `quality` */
  refine: boolean
  autoRun: boolean
  run: RunState | null
  /** a preview asked for that has not started yet (the engine is still stopping the previous one) */
  requested: { version: number; at: number } | null
  /** views, slices and colours being downloaded from the engine */
  loadingViews: number
  views: Record<string, ViewMesh>
  /** the part of a view on screen meshed finer (Refine, when zoomed in), per view */
  details: Record<string, Detail>
  /** colours on views: a field painted on the surfaces, the wall thickness, or a node's result */
  paints: Record<string, Paint>
  /** supports and loads to draw in the 3D view, per Fixed Faces / Load node */
  markers: Record<string, Marker[]>
  /** frames' bars (Frame Optimization) for the 3D view's frame view */
  frames: Record<string, FrameView[]>
  variants: Variant[]
  queue: QueueState | null
  log: LogEntry[]
  fitRequest: number
  /** bumps when a recipe is loaded (the node editor re-frames itself) */
  loadCount: number
  connected: boolean
  past: Snapshot[]
  future: Snapshot[]
  /** coloured slice planes of visible field nodes */
  slices: Record<string, SliceView>
  slice: SliceSettings
  /** box around everything shown so far; slices cover it */
  sceneBounds: Bounds
  /** face picking in the 3D view: the Face Mask node being edited and the node whose CAD body is picked on */
  picking: { node: string; source: string } | null
  /** placing a guide shape (Ball, Bar, Box) by clicking on a shown body: its node and the point input
   *  the next click sets (a Bar: start, then end) */
  placing: { node: string; port: string } | null
  /** which CAD face and body each triangle of a shown CAD view is (face picking, right-click menus) */
  faceData: Record<string, FaceData>
  /** nodes an action in the 3D view just added or changed: the graph highlights them briefly */
  flash: { ids: string[]; at: number }
  /** a short note in the 3D view after an action there */
  toast: Toast | null
  /** pan the node graph to this node */
  focus: { id: string; at: number } | null
  /** the small box in the 3D view with a new node's main inputs (at client x, y) */
  quickEdit: { node: string; x: number; y: number } | null
  /** bumps when the 3D view asks the node editor to open its palette */
  paletteRequest: number

  setCatalog: (c: Catalog) => void
  loadRecipe: (recipe: Recipe, path: string | null, dirty?: boolean) => void
  recipe: () => Recipe
  onNodesChange: (changes: NodeChange<EngineNode>[]) => void
  onEdgesChange: (changes: EdgeChange<EngineEdge>[]) => void
  connect: (c: Connection) => void
  setInput: (id: string, port: string, value: unknown) => void
  addNode: (typeId: string, position: { x: number; y: number }) => string
  toggleVisible: (id: string) => void
  setQuality: (q: Quality) => void
  setRefine: (b: boolean) => void
  setAutoRun: (b: boolean) => void
  setRequested: (r: { version: number; at: number } | null) => void
  /** +1 when a download of a view starts, -1 when it ends */
  countLoading: (delta: number) => void
  handleEvent: (e: EngineEvent) => void
  setView: (id: string, view: ViewMesh) => void
  setDetail: (id: string, detail: Detail) => void
  setPaint: (id: string, paint: Paint) => void
  clearPaints: (source?: Paint['source']) => void
  addLog: (text: string, level?: LogEntry['level']) => void
  markSaved: (path: string) => void
  requestFit: () => void
  setConnected: (b: boolean) => void
  undo: () => void
  redo: () => void
  setSlice: (id: string, slice: SliceView) => void
  setSliceSettings: (s: Partial<SliceSettings>) => void
  /** the selected nodes as clipboard data (null if nothing is selected) */
  copySelection: () => Clip | null
  /** add clipboard nodes; `at` = where the top-left node goes (flow coords). Returns how many. */
  pasteClip: (clip: Clip, at: { x: number; y: number } | null, keepExternal: boolean) => number
  /** start picking faces for a Face Mask node (shows its CAD body), or stop (null) */
  setPicking: (p: { node: string; source: string } | null) => void
  setPlacing: (p: { node: string; port: string } | null) => void
  /** make a group from the selected nodes; returns an error message or null */
  createGroup: (name: string) => string | null
  ungroup: (id: string) => void
  editGroup: (name: string) => void
  finishGroupEdit: () => void
  addGroupPort: (side: 'in' | 'out', name: string, type: string) => string | null
  saveSelection: (name: string, faces: string) => void
  deleteSelection: (name: string) => void
  addVariant: () => void
  updateVariant: (id: number, patch: Partial<Variant>) => void
  removeVariant: (id: number) => void
  setQueue: (q: QueueState) => void
  setFaceData: (id: string, data: FaceData) => void
  /** apply a graph change from the 3D view as one undo step, with a toast listing what changed.
   *  Returns the new nodes' ids by their keys, or null if the edit could not be made. */
  applyEdit: (edit: GraphEdit, label: string, toast?: {
    text?: string | ((ids: Record<string, string>) => string)
    actions?: ToastAction[] | ((ids: Record<string, string>) => ToastAction[])
  }) => Record<string, string> | null
  hideToast: () => void
  /** a toast that changes nothing (no Undo) */
  showToast: (text: string, detail?: string[], actions?: ToastAction[]) => void
  /** select a node, pan the graph to it and flash it */
  focusNode: (id: string) => void
  /** highlight nodes in the graph for a moment */
  flashNodes: (ids: string[]) => void
  setQuickEdit: (q: { node: string; x: number; y: number } | null) => void
  requestPalette: () => void
}

/** Clipboard format for nodes (plain JSON, so it survives the system clipboard). Also the
 *  format of presets. */
export interface Clip {
  kind: 'design-engine/nodes'
  version: 1
  nodes: {
    id: string
    type: string
    inputs: Record<string, unknown>   // typed-in values
    links: Record<string, string>     // port -> "source.port"
    pos: [number, number]
  }[]
  /** groups the nodes use (and the groups those use) */
  groups?: Record<string, GroupDef>
  doc?: string
}

export const isClip = (x: unknown): x is Clip =>
  typeof x === 'object' && x !== null && (x as Clip).kind === 'design-engine/nodes' && Array.isArray((x as Clip).nodes)

/** "offset_1" -> "offset_2", "size_noise" -> "size_noise_2": the first free id. */
export function freeId(id: string, taken: Set<string>) {
  if (!taken.has(id)) return id
  const m = id.match(/^(.*?)_(\d+)$/)
  const base = m ? m[1] : id
  for (let n = m ? Number(m[2]) + 1 : 2; ; n++) {
    const candidate = `${base}_${n}`
    if (!taken.has(candidate)) return candidate
  }
}

interface Snapshot {
  nodes: EngineNode[]
  edges: EngineEdge[]
  visible: string[]
  groups: Record<string, GroupDef>
  selections: Record<string, string>
  at: number
  kind: string
}

const displayable = (spec?: NodeSpec) => !!spec?.outputs.some((o) => DISPLAYABLE.includes(o.type))
/** a plain field (no body): shown as a coloured slice plane */
export const isFieldNode = (spec?: NodeSpec) =>
  !!spec && !displayable(spec) && spec.outputs.some((o) => o.type === 'scalar')

function unionBounds(views: Record<string, ViewMesh>): Bounds | null {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  for (const v of Object.values(views)) {
    if (!v.faces) continue
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], v.bounds[0][k])
      hi[k] = Math.max(hi[k], v.bounds[1][k])
    }
  }
  return lo[0] === Infinity ? null : [lo as Bounds[0], hi as Bounds[1]]
}

function specsWith(catalog: Catalog | null, groups: Record<string, GroupDef>) {
  const specs: Record<string, NodeSpec> = Object.fromEntries((catalog?.nodes ?? []).map((n) => [n.type, n]))
  for (const [name, def] of Object.entries(groups)) specs[GROUP_PREFIX + name] = groupSpec(name, def)
  return specs
}

/** The groups `types` use, and the groups those use. */
function groupsUsed(types: string[], groups: Record<string, GroupDef>) {
  const out: Record<string, GroupDef> = {}
  const stack = types.filter((t) => t.startsWith(GROUP_PREFIX)).map((t) => t.slice(GROUP_PREFIX.length))
  while (stack.length) {
    const name = stack.pop()!
    if (out[name] || !groups[name]) continue
    out[name] = groups[name]
    Object.values(groups[name].nodes).forEach((n) => n.type.startsWith(GROUP_PREFIX) && stack.push(n.type.slice(GROUP_PREFIX.length)))
  }
  return out
}

/** History entry for the state *before* a change. Rapid edits of the same kind (typing a number,
 *  nudging with arrow keys, dragging a value) collapse into one undo step. */
function remember(s: State, kind: string): Pick<State, 'past' | 'future'> {
  const last = s.past[s.past.length - 1]
  const now = Date.now()
  // One gesture can fire several changes (deleting a node also removes its wires): one step.
  if (last && now - last.at < 50) return { past: s.past, future: [] }
  if (last && last.kind === kind && kind.startsWith('input:') && now - last.at < 800) {
    return { past: [...s.past.slice(0, -1), { ...last, at: now }], future: [] }
  }
  const snap: Snapshot = {
    nodes: s.nodes, edges: s.edges, visible: s.visible, groups: s.groups, selections: s.selections, at: now, kind,
  }
  return { past: [...s.past.slice(-99), snap], future: [] }
}

// --- where nodes added from the 3D view go ------------------------------------------------------
const COLUMN = 320            // one column of the graph (the recipes' layout)
const NEW_W = 260, NEW_H = 240  // a new node's size before React Flow has measured it
const GAP = 24

interface Box { x: number; y: number; w: number; h: number }
const nodeBox = (n: EngineNode): Box =>
  ({ x: n.position.x, y: n.position.y, w: n.measured?.width ?? NEW_W, h: n.measured?.height ?? NEW_H })

/** The first spot for a new node next to `anchor` that overlaps no other node: one column to its
 *  right ('right'), moving down until it is free; or in its row after it ('after'), and when that is
 *  taken (usually by what uses the chain) under the anchor. */
function freeSpot(anchor: EngineNode | undefined, place: 'right' | 'after', boxes: Box[]) {
  const hits = (x: number, y: number) => boxes.some((b) =>
    x < b.x + b.w + GAP && x + NEW_W + GAP > b.x && y < b.y + b.h + GAP && y + NEW_H + GAP > b.y)
  if (!anchor) return { x: 0, y: boxes.length ? Math.max(...boxes.map((b) => b.y + b.h)) + GAP * 2 : 0 }
  let x = anchor.position.x + COLUMN
  let y = anchor.position.y
  if (place === 'after' && hits(x, y)) {
    x = anchor.position.x
    y += nodeBox(anchor).h + GAP
  }
  for (let i = 0; i < 400 && hits(x, y); i++) y += 20
  return { x, y }
}

let variantIds = 0

/** The queue's state, and the run going on in the engine if this editor has not seen it start (the
 *  page was opened or reloaded during an export or an optimization). */
function withQueue(s: State, queue: QueueState): Partial<State> {
  const r = queue.running
  if (!r || (s.run && r.run <= s.run.id)) return { queue }
  return { queue, run: { id: r.run, mode: r.mode, quality: r.quality, tag: r.tag, running: true,
                         startedAt: Date.now() - r.elapsed * 1000 } }
}

/** A preview setting kept between sessions (in this browser). */
function stored<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return allowed.includes(v as T) ? (v as T) : fallback
  } catch { return fallback }
}
const saveSetting = (key: string, value: string) => { try { localStorage.setItem(key, value) } catch { /* ignore */ } }

export const useStore = create<State>((set, get) => ({
  catalog: null,
  specs: {},
  nodes: [],
  edges: [],
  visible: [],
  groups: {},
  selections: {},
  editing: null,
  recipePath: null,
  dirty: false,
  graphVersion: 0,
  status: {},
  problems: {},
  quality: stored('quality', QUALITIES, 'final'),
  refine: stored('refine', ['on', 'off'], 'on') === 'on',
  autoRun: true,
  run: null,
  requested: null,
  loadingViews: 0,
  views: {},
  details: {},
  paints: {},
  markers: {},
  frames: {},
  variants: [],
  queue: null,
  log: [],
  fitRequest: 0,
  loadCount: 0,
  connected: false,
  past: [],
  future: [],
  slices: {},
  slice: { axis: 'z', position: null, cmap: 'bands', mode: 'plane' },
  sceneBounds: [[-50, -50, -50], [50, 50, 50]],
  picking: null,
  placing: null,
  faceData: {},
  flash: { ids: [], at: 0 },
  toast: null,
  focus: null,
  quickEdit: null,
  paletteRequest: 0,

  setPicking: (picking) => {
    set({ picking, placing: picking ? null : get().placing })
    if (picking && !get().visible.includes(picking.source)) get().toggleVisible(picking.source)
  },
  setPlacing: (placing) => set({ placing, picking: placing ? null : get().picking }),

  setCatalog: (catalog) => set((s) => ({ catalog, specs: specsWith(catalog, s.groups) })),

  loadRecipe: (recipe, path, dirty = false) => {
    const groups = recipe.groups ?? {}
    const specs = specsWith(get().catalog, groups)
    const { nodes, edges, visible } = recipeToFlow(recipe, specs)
    // Without saved visibility, show the final meshes.
    const shown = visible.length
      ? visible
      : nodes.filter((n) => n.data.typeId === 'mesh.from_implicit').map((n) => n.id)
    set((s) => ({
      nodes, edges, visible: shown, groups, specs, selections: recipe.selections ?? {}, editing: null,
      recipePath: path, dirty,
      variants: path === s.recipePath ? s.variants : [],  // variants compare versions of one recipe
      status: {}, problems: {}, views: {}, details: {}, slices: {}, paints: {}, markers: {}, frames: {},
      graphVersion: s.graphVersion + 1,
      fitRequest: s.fitRequest + 1, loadCount: s.loadCount + 1, past: [], future: [], picking: null, placing: null,
      faceData: {}, toast: null, quickEdit: null,
    }))
  },

  recipe: () => {
    const { nodes, edges, visible, groups, selections, editing } = get()
    let all = groups
    let r: Recipe
    if (editing) {
      all = { ...groups, [editing.name]: flowToGroup(nodes, edges, groups[editing.name]) }
      r = flowToRecipe(editing.outer.nodes, editing.outer.edges, visible)
    } else {
      r = flowToRecipe(nodes, edges, visible)
    }
    if (Object.keys(all).length) r.groups = all
    if (Object.keys(selections).length) r.selections = selections
    return r
  },

  onNodesChange: (changes) =>
    set((s) => {
      // the input and output nodes of an open group cannot be deleted
      changes = changes.filter((c) => !(c.type === 'remove' && (c.id === GROUP_IN_ID || c.id === GROUP_OUT_ID)))
      const semantic = changes.some((c) => c.type === 'remove' || c.type === 'add' || c.type === 'replace')
      const moved = changes.some((c) => c.type === 'position' && c.dragging === false)
      const removed = new Set(changes.filter((c) => c.type === 'remove').map((c) => (c as { id: string }).id))
      const views = { ...s.views }
      const details = { ...s.details }
      const slices = { ...s.slices }
      const paints = { ...s.paints }
      const markers = { ...s.markers }
      const frames = { ...s.frames }
      if (!s.editing) removed.forEach((id) => {
        delete views[id]; delete details[id]; delete slices[id]; delete paints[id]; delete markers[id]; delete frames[id]
      })
      return {
        ...(semantic ? remember(s, 'nodes') : {}),
        nodes: applyNodeChanges(changes, s.nodes),
        visible: s.editing ? s.visible : s.visible.filter((id) => !removed.has(id)),
        views,
        details: removed.size ? details : s.details,
        slices,
        paints,
        markers,
        frames,
        dirty: s.dirty || semantic || moved,
        graphVersion: semantic ? s.graphVersion + 1 : s.graphVersion,
      }
    }),

  onEdgesChange: (changes) =>
    set((s) => {
      const semantic = changes.some((c) => c.type === 'remove' || c.type === 'add' || c.type === 'replace')
      return {
        ...(semantic ? remember(s, 'edges') : {}),
        edges: applyEdgeChanges(changes, s.edges),
        dirty: s.dirty || semantic,
        graphVersion: semantic ? s.graphVersion + 1 : s.graphVersion,
      }
    }),

  connect: ({ source, sourceHandle, target, targetHandle }) =>
    set((s) => {
      if (!source || !target || !sourceHandle || !targetHandle) return {}
      const srcType = source === GROUP_IN_ID && s.editing
        ? s.groups[s.editing.name]?.inputs.find((i) => i.name === sourceHandle)?.type
        : s.specs[s.nodes.find((n) => n.id === source)?.data.typeId ?? '']?.outputs.find((o) => o.name === sourceHandle)?.type
      // An input takes one link: replace whatever was connected before.
      const edges = s.edges.filter((e) => !(e.target === target && e.targetHandle === targetHandle))
      edges.push(makeEdge(source, sourceHandle, target, targetHandle, srcType))
      return { ...remember(s, 'connect'), edges, dirty: true, graphVersion: s.graphVersion + 1 }
    }),

  setInput: (id, port, value) =>
    set((s) => ({
      ...remember(s, `input:${id}.${port}`),
      nodes: s.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, inputs: { ...n.data.inputs, [port]: value } } } : n,
      ),
      dirty: true,
      graphVersion: s.graphVersion + 1,
    })),

  addNode: (typeId, position) => {
    const s = get()
    const id = newNodeId(typeId.startsWith(GROUP_PREFIX) ? `group.${typeId.slice(GROUP_PREFIX.length)}` : typeId,
      new Set(s.nodes.map((n) => n.id)))
    const node: EngineNode = { id, type: 'engine', position, data: { typeId, inputs: {} }, selected: true }
    set({
      ...remember(s, 'add'),
      nodes: [...s.nodes.map((n) => ({ ...n, selected: false })), node],
      dirty: true,
      graphVersion: s.graphVersion + 1,
    })
    return id
  },

  toggleVisible: (id) =>
    set((s) => {
      if (s.editing) return {} // visibility belongs to the recipe's own nodes
      if (s.visible.includes(id)) {
        const views = { ...s.views }
        const details = { ...s.details }
        const slices = { ...s.slices }
        const paints = { ...s.paints }
        const faceData = { ...s.faceData }
        delete views[id]
        delete details[id]
        delete slices[id]
        delete paints[id]
        delete faceData[id]
        const spec = s.specs[s.nodes.find((n) => n.id === id)?.data.typeId ?? '']
        // hiding the painted field takes its colours off the bodies
        const cleared = isFieldNode(spec) ? Object.fromEntries(Object.entries(paints).filter(([, p]) => p.source !== 'field')) : paints
        return { visible: s.visible.filter((v) => v !== id), views, details, slices, paints: cleared, faceData, dirty: true }
      }
      const specOf = (nid: string) => s.specs[s.nodes.find((n) => n.id === nid)?.data.typeId ?? '']
      const spec = specOf(id)
      if (isFieldNode(spec)) {
        // one field at a time: slices of several fields would sit on top of each other
        const others = s.visible.filter((v) => isFieldNode(specOf(v)))
        return {
          visible: [...s.visible.filter((v) => !others.includes(v)), id],
          slices: {}, dirty: true, graphVersion: s.graphVersion + 1,
        }
      }
      if (!displayable(spec)) return {}
      return { visible: [...s.visible, id], dirty: true, graphVersion: s.graphVersion + 1 }
    }),

  setQuality: (quality) => {
    saveSetting('quality', quality)
    set((s) => ({ quality, graphVersion: s.graphVersion + 1 }))
  },
  setRefine: (refine) => {
    saveSetting('refine', refine ? 'on' : 'off')
    // a new preview: on, from Draft up (views already as fine are kept); off, at the chosen quality
    set((s) => ({ refine, graphVersion: s.graphVersion + 1 }))
  },
  setAutoRun: (autoRun) => set({ autoRun }),
  setRequested: (requested) => set({ requested }),
  countLoading: (delta) => set((s) => ({ loadingViews: Math.max(0, s.loadingViews + delta) })),

  handleEvent: (e) =>
    set((s) => {
      if (e.type === 'queue') return withQueue(s, e as unknown as QueueState)
      // an optimization finished or stopped: update the preview so the result (and what depends on it) shows
      if (e.type === 'optimized') return { graphVersion: s.graphVersion + 1 }
      if (e.run !== undefined && s.run && e.run < s.run.id) return {} // stale event from an older run
      switch (e.type) {
        case 'run_start': {
          // nodes behind views kept from a finer run keep their cards; the run reports all others again
          const keepStatus = new Set((e.keep_status ?? []) as string[])
          const tag = e.tag as RunTag | null | undefined  // undefined: an engine that cannot refine (not restarted)
          return {
            run: { id: e.run, mode: e.mode, quality: e.quality, tag, running: true, startedAt: Date.now(),
                   // what the whole view shows stays the same during a detail
                   ...(e.mode === 'detail' ? { shown: s.run?.shown } : {}) },
            // (a detail only meshes part of a body again: every card keeps what it shows)
            status: e.mode === 'detail' ? s.status : Object.fromEntries(Object.entries(s.status).filter(([id]) => keepStatus.has(id))),
            // (an engine without tags, not restarted since an update: any run is the one asked for)
            requested: s.requested && (!tag || tag.version >= s.requested.version) ? null : s.requested,
          }
        }
        case 'phase':
          return s.run?.running ? { run: { ...s.run, phase: e.text } } : {}
        case 'problems': {
          const problems: Record<string, string[]> = {}
          for (const [node, msg] of e.problems as [string | null, string][]) {
            if (node) (problems[node] ??= []).push(msg)
          }
          return { problems }
        }
        // e.view: events of the hidden mesher drawing a node's body; they keep the node's own range, note and
        // warnings (the mesher's warnings are kept apart, as viewWarnings)
        case 'node_start': {
          const prev = s.status[e.node]
          const keep = e.view ? { range: prev?.range, note: prev?.note, warnings: prev?.warnings } : {}
          return { status: { ...s.status, [e.node]: { state: 'running', view: !!e.view, inner: e.inner ?? null, ...keep } } }
        }
        case 'progress': {
          const prev = s.status[e.node]
          const keep = e.view ? { range: prev?.range, note: prev?.note, warnings: prev?.warnings } : {}
          return { status: { ...s.status, [e.node]: { state: 'running', progress: [e.done, e.total], step: e.step ?? null,
            view: !!e.view, inner: e.inner ?? null, ...keep } } }
        }
        case 'node_done': {
          const prev = s.status[e.node]
          const viewWarnings = (e.messages as Warning[] | undefined) ?? null
          if (e.view && prev && prev.state !== 'running') {
            // keep the node's own result
            return viewWarnings?.length || prev.viewWarnings
              ? { status: { ...s.status, [e.node]: { ...prev, viewWarnings } } }
              : {}
          }
          const status: Record<string, NodeStatus> = {
            ...s.status,
            [e.node]: e.view
              ? { state: e.cached ? 'cached' : 'done', time: e.time, range: prev?.range, note: prev?.note,
                  warnings: prev?.warnings, viewWarnings }
              : { state: e.cached ? 'cached' : 'done', time: e.time, range: e.range, note: e.note,
                  warnings: (e.messages as Warning[] | undefined) ?? null },
          }
          const extra: Partial<State> = {}
          if (e.markers && !e.inner) extra.markers = { ...s.markers, [e.node]: e.markers as Marker[] }
          if (!e.view && !e.inner && (e.frames || s.frames[e.node])) {
            // a frame's bars (Frame Optimization); gone when it is not optimized for its settings
            const frames = { ...s.frames }
            if (e.frames) frames[e.node] = e.frames as FrameView[]
            else delete frames[e.node]
            extra.frames = frames
          }
          return { status, ...extra }
        }
        case 'node_error':
          return { status: { ...s.status, [e.node]: { state: 'error', message: e.inner ? `${e.inner}: ${e.message}` : e.message } } }
        case 'message':
          return { log: [...s.log.slice(-199), { time: Date.now(), text: e.text, level: e.level ?? 'info' }] }
        case 'run_done': {
          const run: RunState | null = s.run ? { ...s.run, running: false, phase: undefined } : null
          const secs = `${(e.time ?? 0).toFixed(1)}s`
          const what = s.run?.mode === 'export' ? 'Export' : s.run?.mode === 'optimize' ? 'Optimization'
            : s.run?.mode === 'detail' ? 'Detail' : s.refine && s.run?.quality ? `${QUALITY_LABEL[s.run.quality]} preview` : 'Update'
          let log = s.log
          const add = (text: string, level: LogEntry['level']) => {
            log = [...log.slice(-199), { time: Date.now(), text, level }]
          }
          if (e.cancelled) {
            // Usually superseded by a newer auto-update, so only worth a status, not a log line.
            if (run) run.message = 'Cancelled'
          } else if (!e.ok) {
            if (run) run.message = 'Failed'
            add(e.error ?? 'Run failed', 'error')
          } else {
            if (run) run.message = `Done in ${secs}`
            const files = Object.values(e.files ?? {}) as string[]
            if (files.length) files.forEach((f) => add(`Exported ${f}`, 'success'))
            else if (s.run?.mode === 'export') add('Nothing to export: add an Export Mesh node', 'error')
            if (e.time > 1) add(`${what} finished in ${secs}`, 'info')
          }
          if (run && !e.cancelled && run.mode !== 'detail') run.shown = e.shown ?? null
          if (run && run.mode === 'detail' && e.voxel) run.voxel = e.voxel
          // refining: the next quality, if this preview showed what works (also around a failing node),
          // a finer one would show more and nothing has changed since it was asked for. Views kept from
          // a finer run count as what they are (no run that would change nothing)
          if (run && run.mode === 'preview' && s.refine && !e.cancelled && (e.ok || e.left_out) && e.sharper &&
              run.tag?.version === s.graphVersion && run.quality) {
            const at = Math.max(QUALITIES.indexOf(run.quality), QUALITIES.indexOf(e.shown ?? run.quality))
            if (at < QUALITIES.indexOf(s.quality)) run.next = QUALITIES[at + 1]
          }
          // nodes still marked running were stopped (cancelled, or the run failed elsewhere)
          const status = Object.fromEntries(Object.entries(s.status).filter(([, st]) => st.state !== 'running'))
          const problems = { ...s.problems }
          for (const [node, msg] of (e.problems ?? []) as [string | null, string][]) {
            if (node && !problems[node]?.includes(msg)) (problems[node] ??= []).push(msg)
          }
          const requested = s.requested && run?.tag && run.tag.version >= s.requested.version ? null : s.requested
          // nodes the preview left out (they failed, or need one that did): their old results go, so an
          // old part's shape and loads do not sit on a new STEP file
          const gone = new Set((e.left_out ?? []) as string[])
          if (!gone.size) return { run, log, problems, status, requested }
          const keep = <T,>(m: Record<string, T>) => Object.fromEntries(Object.entries(m).filter(([id]) => !gone.has(id)))
          return { run, log, problems, status, requested, views: keep(s.views), details: keep(s.details), slices: keep(s.slices),
                   paints: keep(s.paints), markers: keep(s.markers), frames: keep(s.frames) }
        }
        default:
          return {}
      }
    }),

  setView: (id, view) =>
    set((s) => {
      if (!s.visible.includes(id)) return {}
      const prev = s.views[id]
      // (the same version again: only to learn what it is made of, if that was not known)
      if (prev && (prev.version > view.version || (prev.version === view.version && (prev.key || !view.key)))) return {}
      const views = { ...s.views, [id]: view }
      return { views, sceneBounds: unionBounds(views) ?? s.sceneBounds }
    }),

  setDetail: (id, detail) =>
    set((s) => {
      if (!s.visible.includes(id) || s.views[id]?.key !== detail.key) return {}  // the view has changed meanwhile
      if (s.details[id] && s.details[id].version >= detail.version) return {}
      // the engine made it coarser to stay within its budget, down to no finer than the view: keep the old one
      if (detail.voxel >= (s.views[id].voxel ?? Infinity)) return {}
      return { details: { ...s.details, [id]: detail } }
    }),

  setPaint: (id, paint) =>
    set((s) => {
      const prev = s.paints[id]
      if (prev && prev.source === paint.source && prev.version >= paint.version) return {}
      return { paints: { ...s.paints, [id]: paint } }
    }),

  clearPaints: (source) =>
    set((s) => ({ paints: source ? Object.fromEntries(Object.entries(s.paints).filter(([, p]) => p.source !== source)) : {} })),

  setSlice: (id, slice) =>
    set((s) => {
      if (!s.visible.includes(id)) return {}
      if (s.slices[id] && s.slices[id].version >= slice.version) return {}
      return { slices: { [id]: slice } }
    }),

  copySelection: () => {
    const s = get()
    const picked = s.nodes.filter((n) => n.selected && n.type === 'engine')
    if (!picked.length) return null
    const groups = groupsUsed(picked.map((n) => n.data.typeId), s.groups)
    return {
      kind: 'design-engine/nodes',
      version: 1,
      nodes: picked.map((n) => ({
        id: n.id,
        type: n.data.typeId,
        inputs: structuredClone(n.data.inputs),
        links: Object.fromEntries(s.edges.filter((e) => e.target === n.id && e.targetHandle && e.source !== GROUP_IN_ID)
          .map((e) => [e.targetHandle!, `${e.source}.${e.sourceHandle}`])),
        pos: [n.position.x, n.position.y],
      })),
      ...(Object.keys(groups).length ? { groups } : {}),
    }
  },

  pasteClip: (clip, at, keepExternal) => {
    const s = get()
    // groups that come with the nodes: added, or renamed if this recipe has a different one by that name
    const groups = { ...s.groups }
    const renamed: Record<string, string> = {}
    for (const [name, def] of Object.entries(clip.groups ?? {})) {
      if (groups[name] && JSON.stringify(groups[name]) === JSON.stringify(def)) continue
      const target = freeId(name, new Set(Object.keys(groups)))
      renamed[name] = target
      groups[target] = def
    }
    const typeOf = (t: string) =>
      t.startsWith(GROUP_PREFIX) && renamed[t.slice(GROUP_PREFIX.length)] ? GROUP_PREFIX + renamed[t.slice(GROUP_PREFIX.length)] : t
    for (const name of Object.values(renamed)) {
      const def = groups[name]
      groups[name] = { ...def, nodes: Object.fromEntries(Object.entries(def.nodes).map(([id, n]) => [id, { ...n, type: typeOf(n.type) }])) }
    }
    const specs = specsWith(s.catalog, groups)
    const known = clip.nodes.map((n) => ({ ...n, type: typeOf(n.type) })).filter((n) => specs[n.type])
    if (!known.length) return 0
    const taken = new Set(s.nodes.map((n) => n.id))
    const idMap: Record<string, string> = {}
    for (const n of known) {
      idMap[n.id] = freeId(n.id, taken)
      taken.add(idMap[n.id])
    }
    const minX = Math.min(...known.map((n) => n.pos[0]))
    const minY = Math.min(...known.map((n) => n.pos[1]))
    const dx = at ? at.x - minX : 40
    const dy = at ? at.y - minY : 40
    const existing = new Set(s.nodes.map((n) => n.id))
    const nodes: EngineNode[] = known.map((n) => ({
      id: idMap[n.id], type: 'engine', selected: true,
      position: { x: n.pos[0] + dx, y: n.pos[1] + dy },
      data: { typeId: n.type, inputs: structuredClone(n.inputs) },
    }))
    const edges: EngineEdge[] = []
    for (const n of known) {
      const ports = new Set(specs[n.type].inputs.map((p) => p.name))
      for (const [port, ref] of Object.entries(n.links)) {
        const [src, ...rest] = ref.split('.')
        const srcPort = rest.join('.')
        // links inside the copied group follow the copies; others only with keepExternal
        const source = idMap[src] ?? (keepExternal && existing.has(src) ? src : null)
        if (!source || !ports.has(port)) continue
        const srcType = idMap[src]
          ? specs[known.find((k) => k.id === src)!.type]?.outputs.find((o) => o.name === srcPort)?.type
          : specs[s.nodes.find((k) => k.id === src)!.data.typeId]?.outputs.find((o) => o.name === srcPort)?.type
        if (!srcType) continue
        edges.push(makeEdge(source, srcPort, idMap[n.id], port, srcType))
      }
    }
    set({
      ...remember(s, 'paste'),
      groups, specs,
      nodes: [...s.nodes.map((n) => (n.selected ? { ...n, selected: false } : n)), ...nodes],
      edges: [...s.edges.map((e) => (e.selected ? { ...e, selected: false } : e)), ...edges],
      dirty: true,
      graphVersion: s.graphVersion + 1,
    })
    return nodes.length
  },

  setSliceSettings: (patch) =>
    set((s) => {
      const slice = { ...s.slice, ...patch }
      if (patch.axis && patch.axis !== s.slice.axis && patch.position === undefined) slice.position = null
      // colours are applied in the browser; plane, position and mode need new values from the engine
      const rerun = patch.axis !== undefined || patch.position !== undefined || (patch.mode !== undefined && patch.mode !== s.slice.mode)
      const paints = patch.mode === 'plane' ? Object.fromEntries(Object.entries(s.paints).filter(([, p]) => p.source !== 'field')) : s.paints
      return { slice, paints, slices: patch.mode === 'surface' ? {} : s.slices, graphVersion: rerun ? s.graphVersion + 1 : s.graphVersion }
    }),

  // --- groups -----------------------------------------------------------------------------
  createGroup: (name) => {
    const s = get()
    if (s.editing) return 'Finish editing the open group first'
    if (!/^[A-Za-z_][\w-]*$/.test(name)) return 'A group name starts with a letter and has no spaces (e.g. noise_wave)'
    if (s.groups[name]) return `There is already a group called ${name}`
    const picked = s.nodes.filter((n) => n.selected && n.type === 'engine')
    if (!picked.length) return 'Select the nodes to group first'
    const ids = new Set(picked.map((n) => n.id))
    const outType = (node: string, port: string) =>
      s.specs[s.nodes.find((n) => n.id === node)?.data.typeId ?? '']?.outputs.find((o) => o.name === port)?.type ?? 'scalar'
    const inner = s.edges.filter((e) => ids.has(e.source) && ids.has(e.target))
    const incoming = s.edges.filter((e) => !ids.has(e.source) && ids.has(e.target))
    const outgoing = s.edges.filter((e) => ids.has(e.source) && !ids.has(e.target))
    const names = new Set(picked.filter((n) => n.data.typeId === SETTING_TYPE).map((n) => n.id)) // settings
    const unique = (base: string) => { const n = freeId(base, names); names.add(n); return n }

    const def: GroupDef = { nodes: {}, inputs: [], outputs: [] }
    const inputOf = new Map<string, string>()   // "source.port" outside -> group input name
    const innerEdges = [...inner]
    for (const e of incoming) {
      const key = `${e.source}.${e.sourceHandle}`
      if (!inputOf.has(key)) {
        const inName = unique(e.targetHandle ?? 'in')
        inputOf.set(key, inName)
        def.inputs.push({ name: inName, type: outType(e.source, e.sourceHandle!) })
      }
      innerEdges.push({ ...e, source: GROUP_IN, sourceHandle: inputOf.get(key)! })
    }
    def.nodes = flowToRecipe(picked, innerEdges, []).nodes
    const outputOf = new Map<string, string>()  // "inner.port" -> group output name
    const outNames = new Set<string>()
    const addOutput = (node: string, port: string) => {
      const key = `${node}.${port}`
      if (outputOf.has(key)) return outputOf.get(key)!
      const outName = freeId(port, outNames)
      outNames.add(outName)
      outputOf.set(key, outName)
      def.outputs.push({ name: outName, type: outType(node, port), link: key })
      return outName
    }
    outgoing.forEach((e) => addOutput(e.source, e.sourceHandle!))
    if (!def.outputs.length) {  // nothing used outside yet: offer the end nodes' first outputs
      for (const n of picked) {
        const out = s.specs[n.data.typeId]?.outputs[0]
        if (out && !inner.some((e) => e.source === n.id)) addOutput(n.id, out.name)
      }
    }

    const instanceId = freeId(name, new Set(s.nodes.map((n) => n.id)))
    const xs = picked.map((n) => n.position.x), ys = picked.map((n) => n.position.y)
    const instance: EngineNode = {
      id: instanceId, type: 'engine', selected: true,
      position: { x: xs.reduce((a, b) => a + b, 0) / xs.length, y: ys.reduce((a, b) => a + b, 0) / ys.length },
      data: {
        typeId: GROUP_PREFIX + name,
        inputs: Object.fromEntries(picked.filter((n) => n.data.typeId === SETTING_TYPE)
          .map((n) => [n.id, (n.data.inputs.value as number | undefined) ?? 0])),
      },
    }
    const edges = s.edges.filter((e) => !ids.has(e.source) && !ids.has(e.target))
    for (const [key, inName] of inputOf) {
      const [src, ...rest] = key.split('.')
      edges.push(makeEdge(src, rest.join('.'), instanceId, inName, outType(src, rest.join('.'))))
    }
    for (const e of outgoing) {
      const outName = outputOf.get(`${e.source}.${e.sourceHandle}`)!
      edges.push(makeEdge(instanceId, outName, e.target, e.targetHandle!, outType(e.source, e.sourceHandle!)))
    }
    const groups = { ...s.groups, [name]: def }
    const wasShown = s.visible.some((v) => ids.has(v))
    set({
      ...remember(s, 'group'),
      groups, specs: specsWith(s.catalog, groups),
      nodes: [...s.nodes.filter((n) => !ids.has(n.id)), instance],
      edges,
      visible: [...s.visible.filter((v) => !ids.has(v)), ...(wasShown ? [instanceId] : [])],
      dirty: true,
      graphVersion: s.graphVersion + 1,
    })
    return null
  },

  ungroup: (id) => {
    const s = get()
    const inst = s.nodes.find((n) => n.id === id)
    const name = inst?.data.typeId.startsWith(GROUP_PREFIX) ? inst.data.typeId.slice(GROUP_PREFIX.length) : null
    const def = name ? s.groups[name] : undefined
    if (!inst || !def || s.editing) return
    const taken = new Set(s.nodes.filter((n) => n.id !== id).map((n) => n.id))
    const idMap: Record<string, string> = {}
    for (const innerId of Object.keys(def.nodes)) {
      idMap[innerId] = freeId(innerId, taken)
      taken.add(idMap[innerId])
    }
    const linkInto = (port: string) => s.edges.find((e) => e.target === id && e.targetHandle === port)
    const pos = Object.values(def.nodes).map((n) => n.pos ?? [0, 0])
    const cx = pos.reduce((a, p) => a + p[0], 0) / Math.max(pos.length, 1)
    const cy = pos.reduce((a, p) => a + p[1], 0) / Math.max(pos.length, 1)
    const specs = s.specs
    const typeOfOut = (node: string, port: string, nodesList: EngineNode[]) =>
      specs[nodesList.find((n) => n.id === node)?.data.typeId ?? '']?.outputs.find((o) => o.name === port)?.type
    const nodes: EngineNode[] = []
    const edges: EngineEdge[] = s.edges.filter((e) => e.target !== id && e.source !== id)
    const replacedSettings: Record<string, EngineEdge> = {}  // Number nodes whose value came over a link
    for (const [innerId, spec] of Object.entries(def.nodes)) {
      const inputs: Record<string, unknown> = {}
      for (const [port, value] of Object.entries(spec.inputs ?? {})) if (!isLink(value)) inputs[port] = value
      if (spec.type === SETTING_TYPE) {
        const linked = linkInto(innerId)
        if (linked) { replacedSettings[innerId] = linked; continue }
        if (inst.data.inputs[innerId] !== undefined) inputs.value = inst.data.inputs[innerId]
      }
      const [x, y] = spec.pos ?? [cx, cy]
      nodes.push({ id: idMap[innerId], type: 'engine', selected: true,
        position: { x: inst.position.x + x - cx, y: inst.position.y + y - cy }, data: { typeId: spec.type, inputs } })
    }
    const allNodes = [...s.nodes, ...nodes]
    for (const [innerId, spec] of Object.entries(def.nodes)) {
      if (replacedSettings[innerId]) continue
      for (const [port, value] of Object.entries(spec.inputs ?? {})) {
        if (!isLink(value)) continue
        const [src, ...rest] = value.link.split('.')
        const srcPort = rest.join('.')
        if (src === GROUP_IN) {
          const outside = linkInto(srcPort)
          if (outside) edges.push(makeEdge(outside.source, outside.sourceHandle!, idMap[innerId], port,
            typeOfOut(outside.source, outside.sourceHandle!, allNodes)))
          continue
        }
        const via = replacedSettings[src]
        if (via) edges.push(makeEdge(via.source, via.sourceHandle!, idMap[innerId], port, typeOfOut(via.source, via.sourceHandle!, allNodes)))
        else if (idMap[src]) edges.push(makeEdge(idMap[src], srcPort, idMap[innerId], port, typeOfOut(idMap[src], srcPort, allNodes)))
      }
    }
    let shownSource: string | null = null
    for (const e of s.edges.filter((x) => x.source === id)) {
      const out = def.outputs.find((o) => o.name === e.sourceHandle)
      if (!out) continue
      const [src, ...rest] = out.link.split('.')
      if (idMap[src]) edges.push(makeEdge(idMap[src], rest.join('.'), e.target, e.targetHandle!, typeOfOut(idMap[src], rest.join('.'), allNodes)))
    }
    if (s.visible.includes(id) && def.outputs[0]) shownSource = idMap[def.outputs[0].link.split('.')[0]] ?? null
    const stillUsed = s.nodes.some((n) => n.id !== id && n.data.typeId === inst.data.typeId) ||
      Object.entries(s.groups).some(([g, d]) => g !== name && Object.values(d.nodes).some((n) => n.type === inst.data.typeId))
    const groups = { ...s.groups }
    if (!stillUsed) delete groups[name!]
    set({
      ...remember(s, 'ungroup'),
      groups, specs: specsWith(s.catalog, groups),
      nodes: [...s.nodes.filter((n) => n.id !== id).map((n) => ({ ...n, selected: false })), ...nodes],
      edges,
      visible: [...s.visible.filter((v) => v !== id), ...(shownSource ? [shownSource] : [])],
      dirty: true,
      graphVersion: s.graphVersion + 1,
    })
  },

  editGroup: (name) => {
    const s = get()
    const def = s.groups[name]
    if (!def || s.editing) return
    const { nodes, edges } = groupToFlow(def, s.specs)
    set({
      editing: { name, outer: { nodes: s.nodes, edges: s.edges, past: [...s.past, { nodes: s.nodes, edges: s.edges,
        visible: s.visible, groups: s.groups, selections: s.selections, at: Date.now(), kind: 'edit group' }], future: [] } },
      nodes, edges, past: [], future: [], picking: null, placing: null, loadCount: s.loadCount + 1,
    })
  },

  finishGroupEdit: () => {
    const s = get()
    if (!s.editing) return
    const name = s.editing.name
    const def = flowToGroup(s.nodes, s.edges, s.groups[name])
    const groups = { ...s.groups, [name]: def }
    const specs = specsWith(s.catalog, groups)
    // links to settings or outputs that are gone are dropped
    const edges = s.editing.outer.edges.filter((e) => {
      const src = s.editing!.outer.nodes.find((n) => n.id === e.source)
      const dst = s.editing!.outer.nodes.find((n) => n.id === e.target)
      const okOut = !src || specs[src.data.typeId]?.outputs.some((o) => o.name === e.sourceHandle)
      const okIn = !dst || specs[dst.data.typeId]?.inputs.some((p) => p.name === e.targetHandle)
      return okOut && okIn
    })
    set({
      groups, specs, nodes: s.editing.outer.nodes, edges, editing: null,
      past: s.editing.outer.past, future: [], dirty: true, graphVersion: s.graphVersion + 1, loadCount: s.loadCount + 1,
    })
  },

  addGroupPort: (side, name, type) => {
    const s = get()
    if (!s.editing) return 'No group is open'
    if (!/^[A-Za-z_][\w-]*$/.test(name)) return 'A name starts with a letter and has no spaces'
    const def = s.groups[s.editing.name]
    const list = side === 'in' ? def.inputs : def.outputs
    if (list.some((p) => p.name === name) || (side === 'in' && def.nodes[name])) return `${name} is already used`
    const next: GroupDef = side === 'in'
      ? { ...def, inputs: [...def.inputs, { name, type }] }
      : { ...def, outputs: [...def.outputs, { name, type, link: '' }] }
    const groups = { ...s.groups, [s.editing.name]: next }
    set({ groups, specs: specsWith(s.catalog, groups), dirty: true })
    return null
  },

  // --- named face selections ------------------------------------------------------------------
  saveSelection: (name, faces) =>
    set((s) => ({ ...remember(s, 'selection'), selections: { ...s.selections, [name]: faces }, dirty: true,
      graphVersion: s.graphVersion + 1 })),
  deleteSelection: (name) =>
    set((s) => {
      const selections = { ...s.selections }
      delete selections[name]
      return { ...remember(s, 'selection'), selections, dirty: true, graphVersion: s.graphVersion + 1 }
    }),

  // --- variants: kept copies of the 3D view, shown next to the current result ------------
  addVariant: () =>
    set((s) => {
      const views = s.visible.map((id) => s.views[id]).filter((v): v is ViewMesh => !!v && v.faces > 0)
      if (!views.length) return {}
      const id = ++variantIds
      const variant: Variant = { id, label: `Variant ${String.fromCharCode(64 + ((id - 1) % 26) + 1)}`, views,
        recipe: get().recipe(), shown: true }
      return { variants: [...s.variants.slice(-3), variant], fitRequest: s.fitRequest + 1 }
    }),
  updateVariant: (id, patch) => set((s) => ({ variants: s.variants.map((v) => (v.id === id ? { ...v, ...patch } : v)) })),
  removeVariant: (id) => set((s) => ({ variants: s.variants.filter((v) => v.id !== id) })),

  setQueue: (queue) => set((s) => withQueue(s, queue)),

  setFaceData: (id, data) => set((s) => (s.visible.includes(id) ? { faceData: { ...s.faceData, [id]: data } } : {})),

  // --- graph changes from the 3D view (right-click menus) --------------------------------------
  applyEdit: (edit, label, toast) => {
    const s = get()
    if (s.editing) return null
    const specOf = (nodes: EngineNode[], id: string) => s.specs[nodes.find((n) => n.id === id)?.data.typeId ?? '']
    const taken = new Set(s.nodes.map((n) => n.id))
    const ids: Record<string, string> = {}
    for (const a of edit.add ?? []) {
      ids[a.key] = newNodeId(a.type, taken)
      taken.add(ids[a.key])
    }
    const id = (ref: string) => ids[ref] ?? ref
    const removed = new Set(edit.remove ?? [])
    let nodes = s.nodes.filter((n) => !removed.has(n.id))
    let edges = s.edges.filter((e) => !removed.has(e.source) && !removed.has(e.target))
    const detail: string[] = []
    const touched = new Set<string>()

    const boxes = nodes.map(nodeBox)
    for (const a of edit.add ?? []) {
      const spec = s.specs[a.type]
      if (!spec) { s.addLog(`${label}: unknown node type ${a.type}`, 'error'); return null }
      const anchor = a.near ? nodes.find((n) => n.id === id(a.near!)) : undefined
      const position = freeSpot(anchor, a.place ?? 'right', boxes)
      boxes.push({ x: position.x, y: position.y, w: NEW_W, h: NEW_H })
      nodes.push({ id: ids[a.key], type: 'engine', position, data: { typeId: a.type, inputs: { ...(a.inputs ?? {}) } } })
      touched.add(ids[a.key])
      detail.push(`+ ${ids[a.key]} (${spec.title})`)
    }
    for (const { node, port, value } of edit.inputs ?? []) {
      const target = id(node)
      nodes = nodes.map((n) => (n.id === target ? { ...n, data: { ...n.data, inputs: { ...n.data.inputs, [port]: value } } } : n))
      touched.add(target)
      if (!Object.values(ids).includes(target)) detail.push(`${target}.${port} = ${String(value) || '(none)'}`)
    }
    const compat = s.catalog?.compatibility ?? {}
    for (const l of edit.links ?? []) {
      const from = id(l.from), to = id(l.to)
      const out = specOf(nodes, from)?.outputs.find((o) => o.name === l.port)?.type
      const inp = specOf(nodes, to)?.inputs.find((p) => p.name === l.input)?.type
      if (!out || !inp || !(compat[inp] ?? [inp]).includes(out)) {
        s.addLog(`${label}: cannot connect ${from}.${l.port} to ${to}.${l.input}`, 'error')
        return null
      }
      edges = edges.filter((e) => !(e.target === to && e.targetHandle === l.input))
      edges.push(makeEdge(from, l.port, to, l.input, out))
      touched.add(to)
      detail.push(`${to}.${l.input} ← ${from}`)
    }
    for (const r of removed) detail.push(`− ${r}`)
    const selections = edit.selections ? { ...s.selections, ...edit.selections } : s.selections
    for (const [name, faces] of Object.entries(edit.selections ?? {})) detail.push(`selection ${name} = ${faces || '(none)'}`)
    if (edit.select) {
      const chosen = new Set(edit.select.map(id))
      nodes = nodes.map((n) => (n.selected !== chosen.has(n.id) ? { ...n, selected: chosen.has(n.id) } : n))
      edges = edges.map((e) => (e.selected ? { ...e, selected: false } : e))
    }
    const drop = <T,>(m: Record<string, T>) => Object.fromEntries(Object.entries(m).filter(([k]) => !removed.has(k)))
    const graphVersion = s.graphVersion + 1
    set({
      ...remember(s, `viewport:${label}`),
      nodes, edges, selections,
      visible: s.visible.filter((v) => !removed.has(v)),
      views: drop(s.views), details: drop(s.details), slices: drop(s.slices), paints: drop(s.paints), markers: drop(s.markers),
      frames: drop(s.frames),
      faceData: drop(s.faceData),
      dirty: true, graphVersion,
      flash: { ids: [...touched].filter((t) => !removed.has(t)), at: Date.now() },
      toast: {
        text: typeof toast?.text === 'function' ? toast.text(ids) : toast?.text ?? label,
        detail, version: graphVersion, at: Date.now(),
        actions: [...(typeof toast?.actions === 'function' ? toast.actions(ids) : toast?.actions ?? []),
          { label: 'Undo', run: () => get().undo() }],
      },
    })
    if (edit.pick) {
      const node = id(edit.pick)
      const source = edges.find((e) => e.target === node && e.targetHandle === 'cad')?.source
      if (source) get().setPicking({ node, source })
    }
    return ids
  },

  hideToast: () => set({ toast: null }),
  setQuickEdit: (quickEdit) => set({ quickEdit }),
  requestPalette: () => set((s) => ({ paletteRequest: s.paletteRequest + 1 })),

  flashNodes: (ids) => set({ flash: { ids, at: Date.now() } }),

  showToast: (text, detail = [], actions = []) =>
    set((s) => ({ toast: { text, detail, actions, version: s.graphVersion, at: Date.now() } })),

  focusNode: (id) =>
    set((s) => (s.editing && !s.nodes.some((n) => n.id === id) ? {} : {
      nodes: s.nodes.map((n) => (n.selected !== (n.id === id) ? { ...n, selected: n.id === id } : n)),
      edges: s.edges.map((e) => (e.selected ? { ...e, selected: false } : e)),
      focus: { id, at: Date.now() },
      flash: { ids: [id], at: Date.now() },
    })),

  addLog: (text, level = 'info') =>
    set((s) => ({ log: [...s.log.slice(-199), { time: Date.now(), text, level }] })),

  markSaved: (path) => set({ recipePath: path, dirty: false }),
  requestFit: () => set((s) => ({ fitRequest: s.fitRequest + 1 })),
  setConnected: (connected) => set({ connected }),

  undo: () =>
    set((s) => {
      const prev = s.past[s.past.length - 1]
      if (!prev) return {}
      const current: Snapshot = { nodes: s.nodes, edges: s.edges, visible: s.visible, groups: s.groups,
        selections: s.selections, at: Date.now(), kind: 'undo' }
      return {
        nodes: prev.nodes, edges: prev.edges, visible: prev.visible, groups: prev.groups, selections: prev.selections,
        specs: specsWith(s.catalog, prev.groups),
        past: s.past.slice(0, -1), future: [...s.future, current],
        dirty: true, graphVersion: s.graphVersion + 1,
      }
    }),

  redo: () =>
    set((s) => {
      const next = s.future[s.future.length - 1]
      if (!next) return {}
      const current: Snapshot = { nodes: s.nodes, edges: s.edges, visible: s.visible, groups: s.groups,
        selections: s.selections, at: Date.now(), kind: 'redo' }
      return {
        nodes: next.nodes, edges: next.edges, visible: next.visible, groups: next.groups, selections: next.selections,
        specs: specsWith(s.catalog, next.groups),
        future: s.future.slice(0, -1), past: [...s.past, current],
        dirty: true, graphVersion: s.graphVersion + 1,
      }
    }),
}))
