import { create } from 'zustand'
import { applyEdgeChanges, applyNodeChanges, type Connection, type EdgeChange, type NodeChange } from '@xyflow/react'
import {
  GROUP_IN_ID, GROUP_OUT_ID, flowToGroup, flowToRecipe, groupSpec, groupToFlow, makeEdge, newNodeId, recipeToFlow,
} from './graphConvert'
import type {
  Bounds, Catalog, EngineEdge, EngineNode, GroupDef, LogEntry, Marker, NodeSpec, NodeStatus, Paint, Quality,
  QueueState, Recipe, RunState, SliceSettings, SliceView, ViewMesh,
} from './types'
import { DISPLAYABLE, GROUP_IN, GROUP_PREFIX, SETTING_TYPE, isLink } from './types'

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
  quality: Quality
  autoRun: boolean
  run: RunState | null
  views: Record<string, ViewMesh>
  /** colours on views: a field painted on the surfaces, the wall thickness, or a node's result */
  paints: Record<string, Paint>
  /** supports and loads to draw in the 3D view, per Fixed Faces / Load node */
  markers: Record<string, Marker[]>
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
  setAutoRun: (b: boolean) => void
  handleEvent: (e: EngineEvent) => void
  setView: (id: string, view: ViewMesh) => void
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

let variantIds = 0

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
  quality: 'draft',
  autoRun: true,
  run: null,
  views: {},
  paints: {},
  markers: {},
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

  setPicking: (picking) => {
    set({ picking })
    if (picking && !get().visible.includes(picking.source)) get().toggleVisible(picking.source)
  },

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
      status: {}, problems: {}, views: {}, slices: {}, paints: {}, markers: {}, graphVersion: s.graphVersion + 1,
      fitRequest: s.fitRequest + 1, loadCount: s.loadCount + 1, past: [], future: [], picking: null,
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
      const slices = { ...s.slices }
      const paints = { ...s.paints }
      const markers = { ...s.markers }
      if (!s.editing) removed.forEach((id) => { delete views[id]; delete slices[id]; delete paints[id]; delete markers[id] })
      return {
        ...(semantic ? remember(s, 'nodes') : {}),
        nodes: applyNodeChanges(changes, s.nodes),
        visible: s.editing ? s.visible : s.visible.filter((id) => !removed.has(id)),
        views,
        slices,
        paints,
        markers,
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
        const slices = { ...s.slices }
        const paints = { ...s.paints }
        delete views[id]
        delete slices[id]
        delete paints[id]
        const spec = s.specs[s.nodes.find((n) => n.id === id)?.data.typeId ?? '']
        // hiding the painted field takes its colours off the bodies
        const cleared = isFieldNode(spec) ? Object.fromEntries(Object.entries(paints).filter(([, p]) => p.source !== 'field')) : paints
        return { visible: s.visible.filter((v) => v !== id), views, slices, paints: cleared, dirty: true }
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

  setQuality: (quality) => set((s) => ({ quality, graphVersion: s.graphVersion + 1 })),
  setAutoRun: (autoRun) => set({ autoRun }),

  handleEvent: (e) =>
    set((s) => {
      if (e.type === 'queue') return { queue: e as unknown as QueueState }
      // an optimization finished or stopped: update the preview so the result (and what depends on it) shows
      if (e.type === 'optimized') return { graphVersion: s.graphVersion + 1 }
      if (e.run !== undefined && s.run && e.run < s.run.id) return {} // stale event from an older run
      switch (e.type) {
        case 'run_start':
          return {
            run: { id: e.run, mode: e.mode, running: true, startedAt: Date.now() },
            status: {},
          }
        case 'problems': {
          const problems: Record<string, string[]> = {}
          for (const [node, msg] of e.problems as [string | null, string][]) {
            if (node) (problems[node] ??= []).push(msg)
          }
          return { problems }
        }
        // e.view: events of the hidden mesher drawing a node's body; they keep the node's own range and note
        case 'node_start': {
          const prev = s.status[e.node]
          const keep = e.view ? { range: prev?.range, note: prev?.note } : {}
          return { status: { ...s.status, [e.node]: { state: 'running', ...keep } } }
        }
        case 'progress': {
          const prev = s.status[e.node]
          const keep = e.view ? { range: prev?.range, note: prev?.note } : {}
          return { status: { ...s.status, [e.node]: { state: 'running', progress: [e.done, e.total], ...keep } } }
        }
        case 'node_done': {
          const prev = s.status[e.node]
          if (e.view && prev && prev.state !== 'running') return {} // keep the node's own result
          const status: Record<string, NodeStatus> = {
            ...s.status,
            [e.node]: e.view
              ? { state: e.cached ? 'cached' : 'done', time: e.time, range: prev?.range, note: prev?.note }
              : { state: e.cached ? 'cached' : 'done', time: e.time, range: e.range, note: e.note },
          }
          if (!e.markers || e.inner) return { status }
          return { status, markers: { ...s.markers, [e.node]: e.markers as Marker[] } }
        }
        case 'node_error':
          return { status: { ...s.status, [e.node]: { state: 'error', message: e.inner ? `${e.inner}: ${e.message}` : e.message } } }
        case 'message':
          return { log: [...s.log.slice(-199), { time: Date.now(), text: e.text, level: e.level ?? 'info' }] }
        case 'run_done': {
          const run = s.run ? { ...s.run, running: false } : null
          const secs = `${(e.time ?? 0).toFixed(1)}s`
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
            if (e.time > 1) add(`${s.run?.mode === 'export' ? 'Export' : 'Update'} finished in ${secs}`, 'info')
          }
          const problems = { ...s.problems }
          for (const [node, msg] of (e.problems ?? []) as [string | null, string][]) {
            if (node && !problems[node]?.includes(msg)) (problems[node] ??= []).push(msg)
          }
          return { run, log, problems }
        }
        default:
          return {}
      }
    }),

  setView: (id, view) =>
    set((s) => {
      if (!s.visible.includes(id)) return {}
      if (s.views[id] && s.views[id].version >= view.version) return {}
      const views = { ...s.views, [id]: view }
      return { views, sceneBounds: unionBounds(views) ?? s.sceneBounds }
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
      nodes, edges, past: [], future: [], picking: null, loadCount: s.loadCount + 1,
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

  setQueue: (queue) => set({ queue }),

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
