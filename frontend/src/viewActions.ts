// What the 3D view's right-click menus offer, and the graph edits they make. Each edit is built from
// the current state alone (no side effects) and applied as one undo step by the store's applyEdit.
import type { FaceInfo } from './api'
import { startOptimize } from './api'
import { hasQuickEdit } from './components/QuickEdit'
import { fmt } from './components/SliceView'
import { useStore } from './store'
import type { EngineNode, FaceData, GraphEdit, Marker, MenuEntry, MenuItem, ToastAction } from './types'
import { parseFaceKeys, resolveFaces } from './types'

type State = ReturnType<typeof useStore.getState>

/** Nodes that take faces of the CAD body on their `cad` input. */
const FACE_NODES = ['sim.fixed', 'sim.load', 'field.face_mask', 'pattern.holes', 'pattern.surface', 'pattern.picture', 'pattern.text']
/** Supports and loads: chained through `add_to`; the last of a chain feeds an analysis' `conditions`. */
const CHAIN_NODES = ['sim.fixed', 'sim.load']
const MATERIALS = ['sim.material', 'sim.custom_material']
/** Analyses a body can get, and the input the body goes into. */
const ANALYSES: { type: string; port: string; label: string }[] = [
  { type: 'sim.static', port: 'part', label: 'Static analysis' },
  { type: 'sim.topology', port: 'space', label: 'Topology optimization' },
  { type: 'sim.frame', port: 'space', label: 'Frame optimization' },
]
const LOAD_KINDS: [string, string][] = [
  ['force', 'Force'], ['pressure', 'Pressure'], ['bearing', 'Bearing (pin in hole)'], ['torque', 'Torque'],
]
/** A load's direction when it is added (or turned into this kind): the node's own default, except a
 *  bearing load, which needs an axis direction (pressing on the bottom of the hole). */
const LOAD_DIRECTION: Record<string, string> = { bearing: '-Z' }
/** Decorations a face can get, and what to connect their output to next. */
const DECORATIONS: { type: string; label: string; next: string }[] = [
  { type: 'field.face_mask', label: 'Face Mask here', next: 'multiply a texture by its Out, so the texture stays on these faces' },
  { type: 'pattern.holes', label: 'Hole pattern here',
    next: 'connect its Out to the B of a Boolean Subtract (holes, pockets) or Union (raised shapes)' },
  { type: 'pattern.surface', label: 'Surface pattern here', next: "connect its Out (0..1) to an Offset Body's Distance" },
  { type: 'pattern.text', label: 'Text here', next: "connect its Out (0..1) to an Offset Body's Distance" },
  { type: 'pattern.picture', label: 'Picture here', next: "connect its Out (0..1) to an Offset Body's Distance" },
]
const OPPOSITE: Record<string, string> = {
  'into the face': 'out of the face', 'out of the face': 'into the face',
  '+X': '-X', '-X': '+X', '+Y': '-Y', '-Y': '+Y', '+Z': '-Z', '-Z': '+Z',
}

/** Face keys as picking writes them: on copies of one part a plain face number ("15": that face on
 *  every copy, like the petals), on different parts "body:face" ("2:15"). */
export function faceKey(faces: FaceData, t: number) {
  return faces.same ? String(faces.ids[t]) : `${faces.bodies[t]}:${faces.ids[t]}`
}

/** A right-click on a CAD face: the node whose CAD output it is on (`source`, output `port`, else its
 *  first CAD output), the face and body numbers, whether that output's bodies are copies of one part,
 *  and the view that was clicked (an implicit body offers the faces of the CAD body it is built from;
 *  `distance`: mm from the click to that face). */
export interface FaceHit {
  view: string
  source: string
  port?: string
  face: number
  body: number
  same: boolean
  bodyCount: number
  distance?: number
}

/** The face under triangle `t` of the CAD view of node `view`. */
export function faceHitOf(view: string, faces: FaceData, t: number): FaceHit {
  let bodyCount = 0
  const seen = new Set<number>()
  for (const b of faces.bodies) if (!seen.has(b)) { seen.add(b); bodyCount++ }
  return { view, source: view, face: faces.ids[t], body: faces.bodies[t], same: faces.same, bodyCount }
}
/** A right-click on a support or load marker: point `point` of marker `index` of node `node`. */
export interface MarkerHit { node: string; index: number; point: number }
/** A right-click on a field slice: its node, the field's value under the cursor (null outside the
 *  field) and the point. */
export interface SliceHit { node: string; value: number | null; point: [number, number, number] }
/** A right-click on a frame's bar: piece `piece` of frame `frame` of node `node` (store.frames). */
export interface BarHit { node: string; frame: number; piece: number }
/** What the menus need from the 3D view: where the menu opened (the quick-edit box opens there), the
 *  point clicked in the scene, and the view's own print check and bar view. */
export interface MenuContext {
  at: { x: number; y: number }
  point?: [number, number, number]
  check?: (node: string) => void
  hideBars?: () => void
}

/** A node output carrying the clicked body, and the clicked face's key on it. `pick`: the output is a
 *  Pick Body the edit adds first (key `pick`), taking body `index` from the view's list. */
interface Source { node: string; port: string; key: string; pick?: { from: string; port: string; index: number } }

interface FaceContext {
  st: State
  /** the node whose CAD output the face is on */
  cad: string
  face: number
  body: number
  bodyCount: number
  /** the bodies are copies of one part (supports and loads ask: this body only, or every copy) */
  copies: boolean
  /** the clicked view's node, then Pick Body nodes taking this body from it */
  sources: Source[]
  /** where new nodes link: the source this body's face nodes already use */
  main: Source
  /** face nodes on this body whose faces contain the clicked face, and the others */
  users: { node: EngineNode; source: Source }[]
  others: { node: EngineNode; source: Source }[]
  /** the last nodes of the supports-and-loads chains on this body */
  tails: string[]
}

const nodeOf = (st: State, id: string) => st.nodes.find((n) => n.id === id)
const linkInto = (st: State, node: string, input: string) =>
  st.edges.find((e) => e.target === node && e.targetHandle === input)
const title = (st: State, node: EngineNode) => st.specs[node.data.typeId]?.title ?? node.data.typeId
const named = (st: State, node: EngineNode) => `${node.id} (${title(st, node)})`
const outputOf = (st: State, id: string, types: string[]) =>
  st.specs[nodeOf(st, id)?.data.typeId ?? '']?.outputs.find((o) => types.includes(o.type))
const hasFace = (keys: string[], face: number, body: number) => keys.includes(String(face)) || keys.includes(`${body}:${face}`)
const facesOf = (st: State, node: EngineNode) => parseFaceKeys(resolveFaces(node.data.inputs.faces, st.selections))
/** The node a source really is (a Pick Body still to add is taken from the view's node). */
const real = (s: Source) => s.pick?.from ?? s.node

/** Every node that uses `id`'s outputs, and what uses those, and so on. */
function downstream(st: State, id: string) {
  const out = new Set<string>()
  const stack = [id]
  while (stack.length) {
    const n = stack.pop()!
    for (const e of st.edges) if (e.source === n && !out.has(e.target)) { out.add(e.target); stack.push(e.target) }
  }
  return out
}

/** Every node `id` is built from: what feeds its inputs, and what feeds those, and so on. */
function upstream(st: State, id: string) {
  const out = new Set<string>()
  const stack = [id]
  while (stack.length) {
    const n = stack.pop()!
    for (const e of st.edges) if (e.target === n && !out.has(e.source)) { out.add(e.source); stack.push(e.source) }
  }
  return out
}

/** The ends of the supports-and-loads chains that the nodes `from` are in. */
function chainTails(st: State, from: string[]) {
  const next = (id: string) =>
    st.edges.find((e) => e.source === id && e.sourceHandle === 'conditions' && e.targetHandle === 'add_to')?.target
  const tails = new Set<string>()
  for (const id of from) {
    let cur = id
    const seen = new Set<string>()
    for (let n = next(cur); n && !seen.has(n); n = next(cur)) { seen.add(cur); cur = n }
    tails.add(cur)
  }
  return [...tails]
}

function faceContext(st: State, hit: FaceHit): FaceContext {
  const { face, body, bodyCount } = hit
  const cad = hit.source
  const port = hit.port ?? outputOf(st, cad, ['cad'])?.name ?? 'body'
  const sources: Source[] = [{ node: cad, port, key: hit.same ? String(face) : `${body}:${face}` }]
  let picked = 0  // Pick Body nodes taking any body from the source
  for (const n of st.nodes) {
    if (n.data.typeId !== 'input.pick_body') continue
    const e = linkInto(st, n.id, 'bodies')
    if (!e || e.source !== cad || e.sourceHandle !== port) continue
    picked++
    let index = Number(n.data.inputs.index ?? 0)
    if (index < 0) index += bodyCount
    // Pick Body's output is that one body: plain face numbers
    if (index === body) sources.push({ node: n.id, port: 'body', key: String(face) })
  }

  const users: FaceContext['users'] = [], others: FaceContext['others'] = []
  for (const n of st.nodes) {
    if (!FACE_NODES.includes(n.data.typeId) || linkInto(st, n.id, 'faces')) continue  // linked faces: not ours to edit
    const e = linkInto(st, n.id, 'cad')
    const source = e && sources.find((s) => s.node === e.source && s.port === e.sourceHandle)
    if (!source) continue
    ;(hasFace(facesOf(st, n), face, body) ? users : others).push({ node: n, source })
  }
  const onBody = [...users, ...others]
  const used = (s: Source) => onBody.some((u) => u.source === s)
  // The body's own nodes decide (so face numbers stay consistent with the analysis); else a Pick Body
  // of this body (the one something uses, if any); else, when other bodies of the list are worked on
  // through their own Pick Body, a new one for this body; else the view's node.
  const main = sources.slice(1).find(used) ?? (used(sources[0]) ? sources[0] : undefined)
    ?? sources.slice(1).find((s) => st.edges.some((e) => e.source === s.node)) ?? sources[1]
    ?? (picked && bodyCount > 1
      ? { node: 'pick', port: 'body', key: String(face), pick: { from: cad, port, index: body } } : sources[0])
  const tails = chainTails(st, onBody.filter((u) => CHAIN_NODES.includes(u.node.data.typeId)).map((u) => u.node.id))
  return { st, cad, face, body, bodyCount, copies: hit.same && bodyCount > 1, sources, main, users, others, tails }
}

// --- edits ----------------------------------------------------------------------------------------
/** The edit with the source's Pick Body (when it is a new one) added first. */
function withSource(s: Source | undefined, edit: GraphEdit): GraphEdit {
  if (!s?.pick) return edit
  return {
    ...edit,
    add: [{ key: 'pick', type: 'input.pick_body', inputs: { index: s.pick.index }, near: s.pick.from }, ...(edit.add ?? [])],
    links: [{ from: s.pick.from, port: s.pick.port, to: 'pick', input: 'bodies' }, ...(edit.links ?? [])],
  }
}

/** The clicked face's key for a new support or load: on copies of one part this body only ("2:15")
 *  or every copy ("15"); a single body (Pick Body) always plain. */
const keyFor = (c: FaceContext, everyCopy: boolean) =>
  c.copies && c.main.node === c.cad && !everyCopy ? `${c.body}:${c.face}` : c.main.key

/** A new support or load on the clicked face, appended to the chain ending in `tail`: it adds to the
 *  tail, and what used the tail's supports and loads now uses the new node's. */
function chainEdit(c: FaceContext, type: string, inputs: Record<string, unknown>, tail: string | null, key: string): GraphEdit {
  const edit: GraphEdit = {
    add: [{ key: 'new', type, inputs: { faces: key, ...inputs }, near: tail ?? c.main.node, place: tail ? 'after' : 'right' }],
    links: [{ from: c.main.node, port: c.main.port, to: 'new', input: 'cad' }],
    select: ['new'],
  }
  if (tail) {
    edit.links!.push({ from: tail, port: 'conditions', to: 'new', input: 'add_to' })
    for (const e of c.st.edges) {
      if (e.source === tail && e.sourceHandle === 'conditions' && e.targetHandle) {
        edit.links!.push({ from: 'new', port: 'conditions', to: e.target, input: e.targetHandle })
      }
    }
  }
  return withSource(c.main, edit)
}

/** Writes a node's faces: into the saved selection when the node uses one by name (every node using
 *  it follows, as picking does), else into the node. */
function facesEdit(st: State, node: EngineNode, keys: string[]): GraphEdit {
  const raw = String(node.data.inputs.faces ?? '').trim()
  const text = parseFaceKeys(keys.join(', ')).join(', ')
  if (st.selections[raw] !== undefined) return { selections: { [raw]: text }, select: [node.id] }
  return { inputs: [{ node: node.id, port: 'faces', value: text }], select: [node.id] }
}

/** Adds the clicked face to a node; on copies in the node's own style (it lists "body:face" keys:
 *  this body only, else every copy). */
function addFaceEdit(c: FaceContext, node: EngineNode, source: Source) {
  const keys = facesOf(c.st, node)
  const key = c.copies && source.node === c.cad && keys.some((k) => k.includes(':')) ? `${c.body}:${c.face}` : source.key
  return facesEdit(c.st, node, [...keys, key])
}
const removeFaceEdit = (st: State, node: EngineNode, face: number, body: number) =>
  facesEdit(st, node, facesOf(st, node).filter((k) => k !== String(face) && k !== `${body}:${face}`))

/** An analysis of the body at `src`, with the graph's material (or a new PLA one) and the supports
 *  and loads ending in `tail`. */
function analyseEdit(st: State, src: Source, a: (typeof ANALYSES)[number], tail: string | null): GraphEdit {
  const material = st.nodes.find((n) => MATERIALS.includes(n.data.typeId))
  const edit: GraphEdit = {
    add: [{ key: 'new', type: a.type, near: tail ?? src.node, place: tail ? 'after' : 'right' }],
    links: [{ from: src.node, port: src.port, to: 'new', input: a.port }],
    select: ['new'],
  }
  if (material) edit.links!.push({ from: material.id, port: 'material', to: 'new', input: 'material' })
  else {
    edit.add!.push({ key: 'material', type: 'sim.material', inputs: { preset: 'PLA' }, near: src.node })
    edit.links!.push({ from: 'material', port: 'material', to: 'new', input: 'material' })
  }
  if (tail) edit.links!.push({ from: tail, port: 'conditions', to: 'new', input: 'conditions' })
  return withSource(src, edit)
}

/** The body at `src` as an implicit body into `port` of `target` (Keep, Avoid, Prefer): a CAD body
 *  goes through Implicit Body from CAD Body; `union` keeps what was there, joined by a Boolean Union. */
function regionEdit(st: State, src: Source, cad: boolean, target: string, port: string, union: boolean): GraphEdit {
  const edit: GraphEdit = { add: [], links: [], select: [target] }
  let from = src.node, out = src.port
  if (cad) {
    edit.add!.push({ key: 'implicit', type: 'implicit.from_cad', near: src.node })
    edit.links!.push({ from: src.node, port: src.port, to: 'implicit', input: 'cad' })
    from = 'implicit'
    out = 'out'
  }
  const old = linkInto(st, target, port)
  if (union && old) {
    edit.add!.push({ key: 'union', type: 'implicit.union', near: cad ? 'implicit' : src.node })
    edit.links!.push({ from: old.source, port: old.sourceHandle!, to: 'union', input: 'a' },
      { from, port: out, to: 'union', input: 'b' })
    from = 'union'
    out = 'out'
  }
  edit.links!.push({ from, port: out, to: target, input: port })
  return withSource(src, edit)
}

/** An input a decoration's output can go into straight away. */
interface FreeInput { node: string; input: string; label: string }

/** Inputs nothing is linked to yet on a body built from the clicked CAD body: Offset Body's Distance
 *  for a field (Surface Pattern, Text, Picture), Boolean Subtract's or Union's B for a Hole Pattern's
 *  cutter. A Face Mask has none (it multiplies a texture: a design choice). */
function freeInputs(c: FaceContext, type: string): FreeInput[] {
  const st = c.st
  const roots = new Set([c.cad, ...c.sources.map((s) => s.node)])
  const fromBody = (node: string, input: string) => {
    const e = linkInto(st, node, input)
    return !!e && (roots.has(e.source) || [...upstream(st, e.source)].some((u) => roots.has(u)))
  }
  if (type === 'pattern.holes') {
    return st.nodes.filter((n) => (n.data.typeId === 'implicit.subtract' || n.data.typeId === 'implicit.union') &&
      !linkInto(st, n.id, 'b') && fromBody(n.id, 'a'))
      .map((n) => ({ node: n.id, input: 'b', label: n.data.typeId === 'implicit.subtract' ? `cut out with ${n.id}` : `add with ${n.id}` }))
  }
  if (type === 'field.face_mask') return []
  return st.nodes.filter((n) => n.data.typeId === 'implicit.offset' && !linkInto(st, n.id, 'distance') && fromBody(n.id, 'body'))
    .map((n) => ({ node: n.id, input: 'distance', label: `connect to ${n.id}'s Distance` }))
}

/** A decoration node on the clicked face (Face Mask, Hole / Surface Pattern, Text, Picture), next to
 *  the body's node; its output goes to `connect` when given, else it is left for the user to wire. */
function decorateEdit(c: FaceContext, type: string, key: string, connect?: FreeInput): GraphEdit {
  const out = c.st.specs[type]?.outputs[0]?.name ?? 'out'
  const edit: GraphEdit = {
    add: [{ key: 'new', type, inputs: { faces: key }, near: c.main.node }],
    links: [{ from: c.main.node, port: c.main.port, to: 'new', input: 'cad' }],
    select: ['new'],
  }
  if (connect) edit.links!.push({ from: 'new', port: out, to: connect.node, input: connect.input })
  return withSource(c.main, edit)
}

/** Deletes a support or load and keeps its chain whole: what used its supports and loads now uses
 *  what came in on its Add to. */
function deleteChainEdit(st: State, node: string): GraphEdit {
  const prev = linkInto(st, node, 'add_to')
  const edit: GraphEdit = { remove: [node], links: [] }
  if (prev) {
    for (const e of st.edges) {
      if (e.source === node && e.sourceHandle === 'conditions' && e.targetHandle) {
        edit.links!.push({ from: prev.source, port: prev.sourceHandle!, to: e.target, input: e.targetHandle })
      }
    }
  }
  return edit
}

// --- menus ----------------------------------------------------------------------------------------
const apply = (edit: GraphEdit, label: string, toast?: Parameters<State['applyEdit']>[2]) =>
  useStore.getState().applyEdit(edit, label, toast)
const item = (label: string, run?: MenuItem['run'], extra: Partial<MenuItem> = {}): MenuItem => ({ kind: 'item', label, run, ...extra })
const header = (text: string): MenuEntry => ({ kind: 'header', text })
const sep: MenuEntry = { kind: 'separator' }
const editingReason = (st: State) => (st.editing ? 'Close the open group first' : undefined)

/** "cylinder Ø 8.0 mm (hole) · 1.6 cm²" */
function describe(info: FaceInfo) {
  const area = info.area < 100 ? `${info.area.toFixed(1)} mm²` : `${(info.area / 100).toFixed(1)} cm²`
  const size = info.radius ? ` Ø ${(info.radius * 2).toFixed(1)} mm${info.hole ? ' (hole)' : ''}` : ''
  return `${info.type}${size} · ${area}`
}

/** One item per chain end when there are several (two analyses on one body), else `run` directly. */
function perTail(st: State, label: string, tails: string[], run: (tail: string | null) => MenuItem['run'], alone: string): MenuItem {
  if (tails.length < 2) return item(label, run(tails[0] ?? null))
  const users = (tail: string) => st.edges.filter((e) => e.source === tail && e.sourceHandle === 'conditions').map((e) => e.target)
  return item(label, undefined, {
    items: [
      ...tails.map((tail) => item(`After ${tail}`, run(tail), { hint: users(tail).length ? `→ ${users(tail).join(', ')}` : undefined })),
      item(alone, run(null)),
    ],
  })
}

/** The face menu: supports and loads on it, adding it to nodes on the same body, the nodes using it.
 *  `info` (the face's surface type and size, when the engine tells) orders the load kinds: a hole
 *  lists Bearing and Torque first. */
export function faceMenu(hit: FaceHit, info: FaceInfo | null, ctx: MenuContext): MenuEntry[] {
  const c = faceContext(useStore.getState(), hit)
  const { st, main } = c
  const source = main.pick ? `a new Pick Body (body ${main.pick.index})` : main.node
  const pickMore = (ids: Record<string, string>): ToastAction[] =>
    [{ label: 'Pick more faces', run: () => useStore.getState().setPicking({ node: ids.new, source: ids.pick ?? main.node }) }]

  /** an item adding a support or load; asks which chain (two analyses on this body) and, on copies,
   *  whether on this body only or on every copy */
  const chainItem = (label: string, type: string, inputs: Record<string, unknown>): MenuItem => {
    const make = (everyCopy: boolean) => (tail: string | null): MenuItem['run'] => (o) => {
      const ids = apply(chainEdit(c, type, inputs, tail, keyFor(c, everyCopy)), label, {
        text: (ids) => `Added ${ids.new} (${st.specs[type]?.title ?? type}) ← ${ids.pick ?? main.node}${tail ? `, after ${tail}` : ''}`,
        actions: pickMore,
      })
      if (ids && type === 'sim.load' && !o.shift) useStore.getState().setQuickEdit({ node: ids.new, ...ctx.at })
    }
    if (!c.copies || main.node !== c.cad) return perTail(st, label, c.tails, make(false), 'On its own')
    return item(label, undefined, {
      items: [
        perTail(st, 'On this body only', c.tails, make(false), 'On its own'),
        { ...perTail(st, 'On every copy', c.tails, make(true), 'On its own'), title: `Face ${c.face} of all ${c.bodyCount} copies` },
      ],
    })
  }

  /** a decoration on this face (copies: every copy first), optionally connected straight on */
  const decorateItem = (d: (typeof DECORATIONS)[number], connect?: FreeInput): MenuItem => {
    const title = st.specs[d.type]?.title ?? d.type
    const make = (everyCopy: boolean): MenuItem['run'] => (o) => {
      const ids = apply(decorateEdit(c, d.type, keyFor(c, everyCopy), connect), title, {
        text: (ids) => `Added ${ids.new} (${title}) ← ${ids.pick ?? main.node}` +
          (connect ? ` → ${connect.node}.${connect.input}` : `: ${d.next}`),
        actions: pickMore,
      })
      if (!ids) return
      if (!connect) useStore.getState().addLog(`${ids.new}: ${d.next}`)
      if (!o.shift && hasQuickEdit(d.type)) useStore.getState().setQuickEdit({ node: ids.new, ...ctx.at })
    }
    const label = connect ? `…and ${connect.label}` : d.label
    if (!c.copies || main.node !== c.cad) return item(label, make(true))
    return item(label, undefined, { items: [item('On every copy', make(true)), item('On this body only', make(false))] })
  }

  const kinds = info?.hole ? [2, 3, 0, 1].map((i) => LOAD_KINDS[i]) : LOAD_KINDS
  const far = hit.distance !== undefined && hit.distance > 0.5 ? `, ${hit.distance.toFixed(1)} mm from the click` : ''
  const entries: MenuEntry[] = [
    header(`Face ${c.face}${c.bodyCount > 1 ? ` · body ${c.body}` : ''}${info ? ` · ${describe(info)}` : ''}`),
    ...(hit.view !== hit.source ? [header(`the CAD face of ${hit.source} under ${hit.view}${far}`)] : []),
    ...(main.node !== hit.source ? [header(`new nodes use ${source}`)] : []),
    chainItem('Fix this face', 'sim.fixed', {}),
    item('Add load here', undefined, {
      items: kinds.map(([kind, label]) =>
        chainItem(label, 'sim.load', { kind, ...(LOAD_DIRECTION[kind] ? { direction: LOAD_DIRECTION[kind] } : {}) })),
    }),
    item('Decorate', undefined, {
      items: DECORATIONS.flatMap((d) => {
        const free = freeInputs(c, d.type)
        return [decorateItem(d), ...(free.length === 1 ? [decorateItem(d, free[0])] : [])]
      }),
    }),
  ]
  if (c.others.length) {
    entries.push(item('Add to', undefined, {
      items: c.others.map(({ node, source }) => item(named(st, node),
        () => apply(addFaceEdit(c, node, source), `Add face ${c.face} to ${node.id}`, { text: `Face ${c.face} added to ${node.id}` }))),
    }))
  }
  if (c.users.length) {
    entries.push(item('Used by', undefined, {
      hint: String(c.users.length),
      items: c.users.map(({ node }) => item(named(st, node), undefined, {
        items: [
          item('Select node', () => useStore.getState().focusNode(node.id)),
          item('Remove this face', () => apply(removeFaceEdit(st, node, c.face, c.body), `Remove face ${c.face} from ${node.id}`,
            { text: `Face ${c.face} removed from ${node.id}` })),
          item('Pick faces…', () => startPicking(node.id)),
        ],
      })),
    }))
  }
  return entries
}

function startPicking(node: string) {
  const e = linkInto(useStore.getState(), node, 'cad')
  if (e) useStore.getState().setPicking({ node, source: e.source })
}

/** A shown body: its node, hiding it, analysing it, using it in an optimization, picking it out of
 *  its STEP file, meshing it. `hit`: the face under the cursor on a CAD view (decides the source
 *  as for the face menu). */
export function bodyMenu(view: string, hit: FaceHit | null, ctx: MenuContext): MenuEntry[] {
  const st = useStore.getState()
  const node = nodeOf(st, view)
  const v = st.views[view]
  const tris = v ? `${(v.faces / 1000).toFixed(v.faces < 10000 ? 1 : 0)}k tris` : ''
  const others = st.visible.filter((id) => id !== view)
  const off = editingReason(st)
  const entries: MenuEntry[] = [
    header([view, node ? title(st, node) : '', tris].filter(Boolean).join(' · ')),
    item('Select node in graph', () => useStore.getState().focusNode(view), { disabled: off }),
    item('Hide', () => useStore.getState().toggleVisible(view), { disabled: off }),
    item('Show only this', () => others.forEach((id) => useStore.getState().toggleVisible(id)),
      { disabled: off ?? (!others.length ? 'Nothing else is shown' : undefined) }),
  ]
  if (st.editing || !node) return entries

  // the body as a source: as the face menu would link it on a CAD view, else the node's own output
  // (a textured body is analysed or used as itself, not as the CAD body under it)
  const c = hit && hit.source === view ? faceContext(st, hit) : null
  const out = outputOf(st, view, v?.kind === 'cad' ? ['cad'] : ['implicit', 'scalar'])
  const src: Source | null = c ? c.main : out ? { node: view, port: out.name, key: '' } : null
  const cad = v?.kind === 'cad'
  if (src) {
    const tails = c ? c.tails : chainTails(st, st.nodes.filter((n) => CHAIN_NODES.includes(n.data.typeId)).map((n) => n.id))
    const via = src.pick ? ` (through a new Pick Body of body ${src.pick.index})` : src.node !== view ? ` (through ${src.node})` : ''
    entries.push(sep, item('Analyse', undefined, {
      title: `Material and the analysis, connected to this body${via} and to its supports and loads`,
      items: ANALYSES.map((a) => perTail(st, a.label, tails, (tail) => () => apply(analyseEdit(st, src, a, tail), a.label, {
        text: (ids) => `Added ${ids.new} (${st.specs[a.type]?.title}) ← ${ids.pick ?? src.node}` +
          (tail ? `, supports & loads from ${tail}` : ': right-click faces to add supports and loads'),
        actions: (ids) => a.type === 'sim.static'
          ? [{ label: 'Show result', run: () => useStore.getState().toggleVisible(ids.new) }]
          : a.type === 'sim.topology' && tail ? [{ label: 'Optimize', run: () => { startOptimize(ids.new) } }] : [],
      }), 'Without supports and loads')),
    }))
    const useAs = useAsEntries(st, src, cad)
    if (useAs.length) entries.push(item('Use as', undefined, { items: useAs }))
  }
  if (c && c.bodyCount > 1 && c.sources.length === 1) {
    entries.push(item('Pick this body', () => apply({
      add: [{ key: 'new', type: 'input.pick_body', inputs: { index: c.body }, near: view }],
      links: [{ from: view, port: c.sources[0].port, to: 'new', input: 'bodies' }],
      select: ['new'],
    }, 'Pick this body', {
      text: (ids) => `Added ${ids.new} (Pick Body ${c.body}) ← ${view}`,
      actions: (ids) => [{ label: 'Show only it', run: () => {
        const s = useStore.getState()
        s.toggleVisible(ids.new)
        if (s.visible.includes(view)) s.toggleVisible(view)
      } }],
    }), { title: `A Pick Body with index ${c.body}: this body alone, for its own supports, loads and analysis` }))
  }
  if (!cad && v) {
    if (v.kind === 'implicit' && out) {
      entries.push(item('Mesh and export…', () => apply({
        add: [{ key: 'mesh', type: 'mesh.from_implicit', near: view }, { key: 'new', type: 'output.export', inputs: { path: `${view}.stl` }, near: 'mesh' }],
        links: [{ from: view, port: out.name, to: 'mesh', input: 'body' }, { from: 'mesh', port: 'mesh', to: 'new', input: 'mesh' }],
        select: ['new'],
      }, 'Mesh and export', { text: (ids) => `Added ${ids.mesh} and ${ids.new}: set the file, then Export` })))
    }
    if (ctx.check) entries.push(item('Print check', () => ctx.check!(view)))
  }
  if (ctx.point && Object.keys(st.slices).length && st.slice.mode === 'plane') {
    const k = 'xyz'.indexOf(st.slice.axis)
    entries.push(item('Move the slice here', () => useStore.getState().setSliceSettings({ position: ctx.point![k] }),
      { hint: `${st.slice.axis} = ${ctx.point[k].toFixed(1)}`, title: 'The field slice goes through the clicked point' }))
  }
  entries.push(item('Keep variant', () => useStore.getState().addVariant(),
    { title: 'Keep a copy of what is shown, to compare with your next changes (shown to the left)' }))
  return entries
}

/** A field slice: its value here, turning it through this point, colouring the bodies instead. */
export function sliceMenu(hit: SliceHit): MenuEntry[] {
  const st = useStore.getState()
  const node = nodeOf(st, hit.node)
  const value = hit.value === null ? '' : ` · ${fmt(hit.value)} here`
  return [
    header(`${hit.node}${node ? ` · ${title(st, node)}` : ''} · field slice${value}`),
    item('Select node in graph', () => useStore.getState().focusNode(hit.node), { disabled: editingReason(st) }),
    item('Turn it', undefined, {
      title: 'The slice across another axis, through the clicked point',
      items: (['x', 'y', 'z'] as const).map((ax) => item(`Across ${ax.toUpperCase()}`,
        () => useStore.getState().setSliceSettings({ axis: ax, position: hit.point['xyz'.indexOf(ax)] }),
        { checked: ax === st.slice.axis, hint: `${ax} = ${hit.point['xyz'.indexOf(ax)].toFixed(1)}` })),
    }),
    item('Colour the bodies with this field', () => useStore.getState().setSliceSettings({ mode: 'surface' }),
      { disabled: Object.keys(st.views).length ? undefined : 'Show a body to colour first' }),
    item('Hide', () => useStore.getState().toggleVisible(hit.node), { disabled: editingReason(st) }),
  ]
}

/** A frame's bar: its size, length, force and utilisation, the frame's node, hiding the bars. */
export function barMenu(hit: BarHit, ctx: MenuContext): MenuEntry[] {
  const st = useStore.getState()
  const f = st.frames[hit.node]?.[hit.frame]
  if (!f) return []
  const c = f.chains[hit.piece], force = f.axial[hit.piece]
  return [
    header(`bar ${c + 1} of ${hit.node}: ${f.sizes[c]} mm ${f.profile}, ${f.lengths[c].toFixed(1)} mm long`),
    header(`${Math.abs(force).toFixed(0)} N ${force < 0 ? 'compression' : 'tension'} here · utilisation ${f.utilisation[c].toFixed(2)}`),
    item('Select frame node', () => useStore.getState().focusNode(hit.node), { disabled: editingReason(st) }),
    ...(ctx.hideBars ? [item('Hide bars', ctx.hideBars)] : []),
  ]
}

/** Import STEP… for a graph that has none (null otherwise). */
export function importItem(): MenuItem | null {
  const st = useStore.getState()
  if (st.editing || st.nodes.some((n) => n.data.typeId === 'input.import_step')) return null
  return item('Import STEP…', () => apply({ add: [{ key: 'new', type: 'input.import_step' }], select: ['new'] }, 'Import STEP', {
    text: (ids) => `Added ${ids.new}: choose the file with … on the node, then click its eye`,
    actions: (ids) => [{ label: 'Show node', run: () => useStore.getState().focusNode(ids.new) }],
  }))
}

/** Using the body in an optimization: its design space, a Keep / Avoid / guide body, or (a CAD body)
 *  what a frame connects to. Nodes the body comes from are left out (that would be a loop). */
function useAsEntries(st: State, src: Source, cad: boolean): MenuEntry[] {
  const targets = st.nodes.filter((n) => (n.data.typeId === 'sim.topology' || n.data.typeId === 'sim.frame') &&
    n.id !== real(src) && !downstream(st, n.id).has(real(src)))
  const linked = (target: string, port: string) => {
    const e = linkInto(st, target, port)
    return e && e.source === src.node && e.sourceHandle === src.port
  }
  const now = (target: string, port: string) => linkInto(st, target, port)?.source
  const forTarget = (t: EngineNode): MenuItem[] => {
    const items: MenuItem[] = []
    const link = (port: string, label: string) => () => apply(withSource(src, {
      links: [{ from: src.node, port: src.port, to: t.id, input: port }], select: [t.id],
    }), `${label} of ${t.id}`, { text: (ids) => `${t.id}: ${label.toLowerCase()} ← ${ids.pick ?? src.node}` })
    if (!linked(t.id, 'space')) {
      items.push(item('Design space', link('space', 'Design space'), { hint: now(t.id, 'space') ? `now ${now(t.id, 'space')}` : undefined }))
    }
    if (t.data.typeId === 'sim.topology') {
      for (const [port, label] of [['keep_body', 'Keep region'], ['avoid_body', 'Avoid region'], ['prefer_body', 'Guide (Prefer)']]) {
        const region = (union: boolean) => () => apply(regionEdit(st, src, cad, t.id, port, union), `${label} of ${t.id}`,
          { text: `${t.id}: ${label.toLowerCase()} ${union ? 'joined with' : '←'} ${src.pick ? 'the new Pick Body' : src.node}` })
        const old = now(t.id, port)
        items.push(old
          ? item(label, undefined, { hint: `now ${old}`, items: [item(`Replace ${old}`, region(false)), item(`Add to ${old} (Boolean Union)`, region(true))] })
          : item(label, region(false)))
      }
    } else if (cad && !linked(t.id, 'connect_to')) {
      items.push(item('Connect to', link('connect_to', 'Connect to'),
        { hint: now(t.id, 'connect_to') ? `now ${now(t.id, 'connect_to')}` : undefined, title: 'This body holds the frame like a support' }))
    }
    return items
  }
  if (targets.length === 1) return forTarget(targets[0]).map((i) => ({ ...i, label: `${i.label} of ${targets[0].id}` }))
  return targets.map((t) => item(named(st, t), undefined, { items: forTarget(t) })).filter((i) => i.items!.length)
}

/** A support or load marker: editing its node, changing a load, removing the face it sits on,
 *  deleting it without breaking its chain. */
export function markerMenu(hit: MarkerHit, ctx: MenuContext): MenuEntry[] {
  const st = useStore.getState()
  const node = nodeOf(st, hit.node)
  const marker: Marker | undefined = st.markers[hit.node]?.[hit.index]
  if (!node || !marker) return []
  const off = editingReason(st)
  const entries: MenuEntry[] = [header(`${node.id} · ${title(st, node)} · ${marker.label}`)]
  if (!CHAIN_NODES.includes(node.data.typeId)) {  // a group's markers: its nodes are inside it
    return [...entries, item('Select node in graph', () => useStore.getState().focusNode(node.id), { disabled: off })]
  }
  const load = node.data.typeId === 'sim.load'
  const inputs = node.data.inputs
  const spec = st.specs[node.data.typeId]
  const value = (port: string) => inputs[port] ?? spec?.inputs.find((p) => p.name === port)?.default
  const kind = String(value('kind') ?? 'force')
  entries.push(item(`Edit ${node.id}`, () => {
    useStore.getState().focusNode(node.id)
    if (load) useStore.getState().setQuickEdit({ node: node.id, ...ctx.at })
  }, { disabled: off }))
  if (load) {
    const set = (changes: Record<string, unknown>, text: string) => () =>
      apply({ inputs: Object.entries(changes).map(([port, v]) => ({ node: node.id, port, value: v })), select: [node.id] }, text, { text })
    const direction = String(value('direction') ?? 'into the face')
    entries.push(item('Change to', undefined, {
      items: LOAD_KINDS.map(([k, label]) => item(label, set({
        kind: k, ...(LOAD_DIRECTION[k] && (direction === 'into the face' || direction === 'out of the face') ? { direction: LOAD_DIRECTION[k] } : {}),
      }, `${node.id} is now a ${label.toLowerCase()} load`), { checked: k === kind, disabled: linkInto(st, node.id, 'kind') ? 'Kind is linked' : undefined })),
    }))
    const linked = linkInto(st, node.id, 'direction') || (direction === 'custom' && linkInto(st, node.id, 'vector'))
    const flipped = direction === 'custom'
      ? { vector: String(value('vector') ?? '0, 0, -1').split(',').map((x) => -Number(x) || 0).join(', ') }
      : OPPOSITE[direction] ? { direction: OPPOSITE[direction] } : null
    entries.push(item('Flip direction', flipped ? set(flipped, `${node.id}: direction flipped`) : undefined, {
      disabled: kind === 'pressure' ? 'A pressure always pushes square to the faces: type a negative amount to pull'
        : linked ? 'Direction is linked' : !flipped ? 'Unknown direction' : undefined,
      hint: flipped && 'direction' in flipped ? `→ ${flipped.direction}` : undefined,
    }))
  }
  entries.push(sep)
  const face = marker.faces?.[hit.point], body = marker.bodies?.[hit.point]
  if (face !== undefined) {
    entries.push(item(`Remove face ${face}`, () => apply(removeFaceEdit(st, node, face, body ?? -1), `Remove face ${face} from ${node.id}`,
      { text: `Face ${face} removed from ${node.id}` }), { disabled: off ?? (linkInto(st, node.id, 'faces') ? 'Faces are linked' : undefined) }))
  }
  entries.push(item('Pick faces…', () => startPicking(node.id), { disabled: off }))
  const prev = linkInto(st, node.id, 'add_to')?.source
  const users = st.edges.filter((e) => e.source === node.id && e.sourceHandle === 'conditions').map((e) => e.target)
  entries.push(item(`Delete ${node.id}`, () => apply(deleteChainEdit(st, node.id), `Delete ${node.id}`, {
    text: `Deleted ${node.id}${prev && users.length ? `: ${users.join(', ')} now ${users.length === 1 ? 'uses' : 'use'} ${prev}` : ''}`,
  }), { disabled: off, title: prev && users.length ? `${users.join(', ')} will take the supports and loads of ${prev}` : undefined }))
  return entries
}

/** While picking faces for a node: the picking actions on the clicked face. */
export function pickingMenu(o: { node: string; face: string; chosen: boolean; count: number
  toggle: (withGroup: boolean) => void; clear: () => void; done: () => void }): MenuEntry[] {
  return [
    header(`Picking faces for ${o.node} · face ${o.face}`),
    item(o.chosen ? 'Remove this face' : 'Add this face', () => o.toggle(false)),
    item(o.chosen ? 'Remove tangent faces' : 'Add tangent faces', () => o.toggle(true),
      { hint: 'Shift+click', title: 'This face and every face joined to it smoothly' }),
    sep,
    item('Clear', o.clear, { disabled: o.count ? undefined : 'No faces chosen' }),
    item('Done', o.done, { hint: 'Esc' }),
  ]
}
