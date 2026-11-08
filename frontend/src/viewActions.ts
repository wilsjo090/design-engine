// What the 3D view's right-click menus offer, and the graph edits they make. Each edit is built from
// the current state alone (no side effects) and applied as one undo step by the store's applyEdit.
import type { FaceInfo } from './api'
import { loadSimilarFaces, startOptimize, startRun, stopRun } from './api'
import { useBrowser } from './components/FileBrowser'
import { hasQuickEdit } from './components/QuickEdit'
import { type Goal, openWizard } from './components/AnalysisWizard'
import { openLatticeWizard } from './components/LatticeWizard'
import { openTextureWizard } from './components/TextureWizard'
import { fmt } from './components/SliceView'
import { LOAD_CASE, loadCases } from './loadCases'
import { useStore } from './store'
import type {
  Axis, EngineNode, FaceData, FaceRef, GraphEdit, Marker, MenuEntry, MenuItem, QuickField, Recipe, ToastAction,
} from './types'
import { bypassInput, isLink, parseFaceKeys, resolveFaces } from './types'

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
/** The analysis wizard's goal for each analysis. */
const WIZARD_GOALS: Record<string, Goal> = { 'sim.static': 'static', 'sim.topology': 'topology', 'sim.frame': 'frame' }
const LOAD_KINDS: [string, string][] = [
  ['force', 'Force'], ['pressure', 'Pressure'], ['bearing', 'Bearing (pin in hole)'], ['torque', 'Torque'],
]
/** A load's direction when it is added (or turned into this kind): the node's own default, except a
 *  bearing load, which needs an axis direction (pressing on the bottom of the hole). */
const LOAD_DIRECTION: Record<string, string> = { bearing: '-Z' }
/** Decorations a face can get, and what to connect their output to next. */
const DECORATIONS: { type: string; label: string; next: string }[] = [
  { type: 'field.face_mask', label: 'Face Mask here', next: "connect its Out to a Texture's Fade, so the texture stays on these faces" },
  { type: 'pattern.holes', label: 'Hole pattern here',
    next: 'connect its Out to the B of a Boolean Subtract (holes, pockets) or Union (raised shapes)' },
  { type: 'pattern.surface', label: 'Surface pattern here', next: "connect its Out (0..1) to a Texture's Pattern" },
  { type: 'pattern.text', label: 'Text here', next: "connect its Out (0..1) to a Texture's Pattern" },
  { type: 'pattern.picture', label: 'Picture here', next: "connect its Out (0..1) to a Texture's Pattern" },
]
/** Textures a face can get in one step: a pattern into a Texture node on the shown body. Fields fade in
 *  on the faces through a Face Mask; a Surface Pattern lies on them already. `shape`: the Surface
 *  Pattern's settings and the height (from the presets of the same name). */
export interface TextureKind {
  kind: 'noise' | 'waves' | 'gyroid' | 'shapes'
  label: string
  title: string
  shape?: { height: number; inputs: Record<string, unknown> }
}
export const TEXTURES: TextureKind[] = [
  { kind: 'noise', label: 'Noise bumps', title: 'Noise → Texture, faded in on these faces by a Face Mask' },
  { kind: 'waves', label: 'Waves', title: '3D Wave (ridges) → Texture, faded in on these faces by a Face Mask' },
  { kind: 'gyroid', label: 'Gyroid cells', title: '3D Wave (gyroid) → Texture, faded in on these faces by a Face Mask' },
]
const SHAPE_TITLE = 'Surface Pattern on these faces → Texture'
export const SHAPE_TEXTURES: TextureKind[] = ([
  ['Round studs', 0.5, {}],
  ['Hex studs', 0.8, { shape: 'hexagon', size: 4, corner_radius: 0.4, spacing: 5.5, softness: 0.6, margin: 1 }],
  ['Dimples', -0.5, { shape: 'round', size: 3, spacing: 4.5, softness: 1.2, margin: 1.5 }],
  ['Tiles', 0.5, { shape: 'rectangle', size: 5, length: 5, corner_radius: 0.6, layout: 'grid', spacing: 6, softness: 0.4, margin: 1 }],
  ['Bricks', 0.6, { shape: 'rectangle', size: 3.5, length: 9, spacing: 4.5, softness: 0.3, margin: 1 }],
  ['Knurl', 0.5, { shape: 'diamond', size: 1.6, length: 1.6, layout: 'grid', spacing: 1.7, softness: 0.8, margin: 1 }],
  ['Voronoi web', 0.6, { shape: 'voronoi', web: 0.8, corner_radius: 0.4, spacing: 5, randomness: 0.9, invert: 'yes', softness: 0.3, margin: 1 }],
  ['Scattered dots', 0.5, { shape: 'round', size: 1.5, spacing: 4, randomness: 1, softness: 0.4, margin: 1 }],
  ['Low-poly facets', 0.6, { shape: 'delaunay', web: 0.8, corner_radius: 0.2, spacing: 6, randomness: 0.7, softness: 0.5, margin: 1 }],
] as [string, number, Record<string, unknown>][]).map(([label, height, inputs]) =>
  ({ kind: 'shapes', label, title: SHAPE_TITLE, shape: { height, inputs } }))
/** A texture's height when it is added, mm. */
export const AMPLITUDE = 0.5
const OPPOSITE: Record<string, string> = {
  'into the face': 'out of the face', 'out of the face': 'into the face',
  '+X': '-X', '-X': '+X', '+Y': '-Y', '-Y': '+Y', '+Z': '-Z', '-Z': '+Z',
}
const MESH_EXTS = ['.stl', '.3mf', '.obj', '.ply']

/** Face keys as picking writes them: on copies of one part a plain face number ("15": that face on
 *  every copy, like the petals), on different parts "body:face" ("2:15"). */
export function faceKey(faces: FaceData, t: number) {
  return faces.same ? String(faces.ids[t]) : `${faces.bodies[t]}:${faces.ids[t]}`
}

/** 0.37 -> 0.5, 13 -> 10, 26 -> 20: a round number near x (1, 2 or 5 times a power of ten). */
export function roundNice(x: number) {
  if (!(x > 0) || !Number.isFinite(x)) return 1
  const p = Math.pow(10, Math.floor(Math.log10(x)))
  return [1, 2, 5, 10].map((m) => m * p).reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a))
}

/** A right-click on a CAD face: the node whose CAD output it is on (`source`, output `port`, else its
 *  first CAD output), the face and body numbers, whether that output's bodies are copies of one part,
 *  and the view that was clicked (an implicit body offers the faces of the CAD body it is built from;
 *  `distance`: mm from the click to that face). `targets`: the faces to act on when several are
 *  chosen (the clicked one is among them). */
export interface FaceHit {
  view: string
  source: string
  port?: string
  face: number
  body: number
  same: boolean
  bodyCount: number
  distance?: number
  targets?: FaceRef[]
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
 *  point clicked in the scene and the surface's outward normal there, and the view's own tools. */
export interface MenuContext {
  at: { x: number; y: number }
  point?: [number, number, number]
  normal?: [number, number, number]
  check?: (node: string) => void
  hideBars?: () => void
  /** the section plane through a point, across an axis */
  section?: (axis: Axis, position: number) => void
  /** chosen faces (Ctrl+click): add these, make them the choice, or drop them */
  select?: (faces: FaceRef[], mode: 'add' | 'replace' | 'remove') => void
  /** the faces smoothly joined to the clicked one (CAD views) */
  tangent?: FaceRef[]
  /** the clicked face is among the chosen ones; how many are chosen */
  selected?: boolean
  selectionSize?: number
  /** the size of what is shown (new shapes are made a fraction of it), mm */
  sceneSize?: number
}

/** A node output carrying the clicked body, and the clicked face's key on it. `pick`: the output is a
 *  Pick Body the edit adds first (key `pick`), taking body `index` from the view's list. */
export interface Source { node: string; port: string; key: string; pick?: { from: string; port: string; index: number } }

interface FaceContext {
  st: State
  /** the node whose CAD output the face is on */
  cad: string
  face: number
  body: number
  bodyCount: number
  /** the faces to act on (the clicked one, or all the chosen ones) */
  targets: FaceRef[]
  /** the bodies are copies of one part (one face number names that face on all of them) */
  same: boolean
  /** ...and there are several (supports and loads ask: this body only, or every copy) */
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
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

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
export function upstream(st: State, id: string) {
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
  const targets = hit.targets?.length ? hit.targets : [{ face, body }]
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
  // through their own Pick Body, a new one for this body; else the view's node. Faces of several
  // bodies at once can only be named on the view's node.
  const severalBodies = new Set(targets.map((t) => t.body)).size > 1
  const main = severalBodies ? sources[0]
    : sources.slice(1).find(used) ?? (used(sources[0]) ? sources[0] : undefined)
      ?? sources.slice(1).find((s) => st.edges.some((e) => e.source === s.node)) ?? sources[1]
      ?? (picked && bodyCount > 1
        ? { node: 'pick', port: 'body', key: String(face), pick: { from: cad, port, index: body } } : sources[0])
  const tails = chainTails(st, onBody.filter((u) => CHAIN_NODES.includes(u.node.data.typeId)).map((u) => u.node.id))
  return { st, cad, face, body, bodyCount, targets, same: hit.same, copies: hit.same && bodyCount > 1, sources, main, users,
    others, tails }
}

// --- edits ----------------------------------------------------------------------------------------
/** The edit with the source's Pick Body (when it is a new one) added first. */
export function withSource(s: Source | null | undefined, edit: GraphEdit): GraphEdit {
  if (!s?.pick || edit.add?.some((a) => a.key === 'pick')) return edit
  return {
    ...edit,
    add: [{ key: 'pick', type: 'input.pick_body', inputs: { index: s.pick.index }, near: s.pick.from }, ...(edit.add ?? [])],
    links: [{ from: s.pick.from, port: s.pick.port, to: 'pick', input: 'bodies' }, ...(edit.links ?? [])],
  }
}

/** The faces to act on as a faces text for a new node: on copies of one part this body only ("2:15")
 *  or every copy ("15"); different bodies "body:face"; a single body (Pick Body) plain numbers. */
function keysFor(c: FaceContext, everyCopy: boolean) {
  return c.targets.map((t) => (c.main.node !== c.cad || c.main.pick ? String(t.face)
    : c.same && (everyCopy || !c.copies) ? String(t.face) : `${t.body}:${t.face}`)).join(', ')
}

/** "face 15", "4 faces" */
const facesLabel = (c: FaceContext) => (c.targets.length === 1 ? `face ${c.targets[0].face}` : plural(c.targets.length, 'face'))

/** A new support or load on the faces, appended to the chain ending in `tail`: it adds to the tail, and
 *  what used the tail's supports and loads now uses the new node's. */
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

/** Adds the faces to a node; on copies in the node's own style (it lists "body:face" keys: this body
 *  only, else every copy). */
function addFacesEdit(c: FaceContext, node: EngineNode, source: Source) {
  const keys = facesOf(c.st, node)
  const bodyStyle = source.node === c.cad && (!c.same || (c.copies && keys.some((k) => k.includes(':'))))
  return facesEdit(c.st, node, [...keys, ...c.targets.map((t) => (bodyStyle ? `${t.body}:${t.face}` : String(t.face)))])
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

/** Inputs nothing is linked to yet on a body built from the clicked CAD body: a Texture's Pattern or an
 *  Offset Body's Distance for a field (Surface Pattern, Text, Picture), Boolean Subtract's or Union's B for a Hole Pattern's
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
  const free = (typeId: string, input: string, label: string) => st.nodes
    .filter((n) => n.data.typeId === typeId && !linkInto(st, n.id, input) && fromBody(n.id, 'body'))
    .map((n) => ({ node: n.id, input, label: `connect to ${n.id}'s ${label}` }))
  return [...free('implicit.texture', 'pattern', 'Pattern'), ...free('implicit.offset', 'distance', 'Distance')]
}

/** A decoration node on the faces (Face Mask, Hole / Surface Pattern, Text, Picture), next to the
 *  body's node; its output goes to `connect` when given, else it is left for the user to wire. */
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

/** Where a change to the shown body of `view` goes in (Texture, Modify): after the output carrying it
 *  (`from`.`port`), or for a CAD body after a new Implicit Body from CAD Body of `cad`; `consumers` take
 *  the changed body instead; `swap`: the changed body is shown in place of the view (a mesh made from
 *  the body shows the change itself). A string says why there is none. */
export interface BodySlot { from: string; port: string; cad: Source | null; consumers: { node: string; input: string }[]; view: string; swap: boolean }

export function bodySlot(st: State, view: string, cad: Source | null): BodySlot | string {
  const v = st.views[view]
  const node = nodeOf(st, view)
  if (!v || !node) return `${view} is not shown`
  if (v.kind === 'cad') {
    const out = outputOf(st, view, ['cad'])
    const src = cad ?? (out ? { node: view, port: out.name, key: '' } : null)
    return src ? { from: '', port: '', cad: src, consumers: [], view, swap: true } : 'No CAD body here'
  }
  const implicit = outputOf(st, view, ['implicit'])
  if (implicit) {
    const consumers = st.edges.filter((e) => e.source === view && e.sourceHandle === implicit.name)
      .map((e) => ({ node: e.target, input: e.targetHandle! }))
    return { from: view, port: implicit.name, cad: null, consumers, view, swap: true }
  }
  // a mesh made from an implicit body: the change goes in before it
  const input = st.specs[node.data.typeId]?.inputs.find((p) => p.type === 'implicit' && linkInto(st, view, p.name))
  const e = input && linkInto(st, view, input.name)
  if (input && e) return { from: e.source, port: e.sourceHandle!, cad: null, consumers: [{ node: view, input: input.name }], view, swap: false }
  return 'Only bodies can be changed here: a CAD body, an implicit body, or a mesh made from one'
}

/** `chain` (new nodes and their own links) put into the body at `slot`: the body goes into `into`, and
 *  what used the body uses `out` instead. */
export function modifyEdit(slot: BodySlot, chain: GraphEdit, into: { key: string; input: string }, out: { key: string; port: string }): GraphEdit {
  const edit: GraphEdit = { ...chain, add: [...(chain.add ?? [])], links: [...(chain.links ?? [])] }
  let from = slot.from, port = slot.port
  if (slot.cad) {
    edit.add!.unshift({ key: 'implicit', type: 'implicit.from_cad', near: slot.cad.node })
    edit.links!.push({ from: slot.cad.node, port: slot.cad.port, to: 'implicit', input: 'cad' })
    from = 'implicit'
    port = 'out'
  }
  edit.links!.push({ from, port, to: into.key, input: into.input })
  for (const c of slot.consumers) edit.links!.push({ from: out.key, port: out.port, to: c.node, input: c.input })
  if (slot.swap) { edit.show = [out.key]; edit.hide = [slot.view] }
  return withSource(slot.cad, edit)
}
/** The node new nodes of a change to `slot` are placed next to. */
export const slotAnchor = (slot: BodySlot) => (slot.cad ? 'implicit' : slot.from)

/** A texture's pattern node (key `src`, next to `near`): Noise or a 3D Wave sized to the faces' `size`
 *  (√area, mm), or a Surface Pattern on `faces` (its `cad` input is left to link). */
export function patternNode(t: TextureKind, size: number | null, near: string, faces: string) {
  if (t.kind === 'shapes') return { key: 'src', type: 'pattern.surface', inputs: { faces, ...t.shape?.inputs }, near }
  if (t.kind === 'noise') return { key: 'src', type: 'field.noise', inputs: { scale: size ? Math.min(50, Math.max(1, roundNice(size / 4))) : 10 }, near }
  return { key: 'src', type: 'field.wave3d', near,
    inputs: { pattern: t.kind === 'waves' ? 'ridges' : 'gyroid', period: size ? Math.min(20, Math.max(1, roundNice(size / 8))) : 5 } }
}

/** A Face Mask's edge fade for a texture on faces of `size`: none under shapes (a Surface Pattern keeps
 *  them inside the faces itself). */
export const maskFade = (t: TextureKind, size: number | null) =>
  t.kind === 'shapes' ? 0 : size ? Math.min(2, roundNice(size / 10)) : 2

/** A texture's main settings: of its pattern node `src` and its Texture `texture` (and the mask's fade). */
export function textureFields(t: TextureKind, src: string, texture: string, mask?: string): QuickField[] {
  const height: QuickField = { node: texture, port: 'height', label: 'Height (mm)' }
  if (t.kind === 'shapes') {
    return [{ node: src, port: 'shape' }, { node: src, port: 'size' }, { node: src, port: 'web' }, { node: src, port: 'spacing' }, height]
  }
  return [
    height,
    ...(t.kind === 'noise'
      ? [{ node: src, port: 'scale', label: 'Scale (mm)' }, { node: src, port: 'octaves' }, { node: src, port: 'seed' }]
      : [{ node: src, port: 'period', label: 'Period (mm)' }, ...(t.kind === 'waves' ? [{ node: src, port: 'axis' }] : []),
          { node: src, port: 'warp', label: 'Warp (mm)' }]),
    ...(mask ? [{ node: mask, port: 'fade', label: 'Edge fade (mm)' }] : []),
  ]
}

/** A texture on the faces in one step (see TEXTURES), into the shown body at `slot`; `size`: the face's
 *  size (√area, mm) to scale the pattern to. Returns the edit and the quick-edit box's inputs. */
function textureEdit(c: FaceContext, slot: BodySlot, t: TextureKind, key: string, size: number | null): BodyChange {
  const near = slotAnchor(slot)
  const chain: GraphEdit = { add: [], links: [], select: ['new'] }
  let fields: (ids: Record<string, string>) => QuickField[]
  if (t.kind === 'shapes') {
    chain.add!.push(patternNode(t, size, c.main.node, key),
      { key: 'new', type: 'implicit.texture', inputs: { height: t.shape?.height ?? AMPLITUDE }, near })
    chain.links!.push({ from: c.main.node, port: c.main.port, to: 'src', input: 'cad' },
      { from: 'src', port: 'out', to: 'new', input: 'pattern' })
    fields = (ids) => textureFields(t, ids.src, ids.new)
  } else {
    const mask = { key: 'mask', type: 'field.face_mask', near: c.main.node, inputs: { faces: key, fade: maskFade(t, size) } }
    // Texture stretches the field's -1..1 to 0..Height, faded in on the faces by the mask
    chain.add!.push(patternNode(t, size, near, key), mask, { key: 'new', type: 'implicit.texture', inputs: { height: AMPLITUDE }, near: 'src' })
    chain.links!.push({ from: c.main.node, port: c.main.port, to: 'mask', input: 'cad' },
      { from: 'src', port: 'out', to: 'new', input: 'pattern' }, { from: 'mask', port: 'out', to: 'new', input: 'fade' })
    fields = (ids) => textureFields(t, ids.src, ids.new, ids.mask)
  }
  return { edit: withSource(c.main, modifyEdit(slot, chain, { key: 'new', input: 'body' }, { key: 'new', port: 'out' })), fields }
}

/** A change to the shown body on chosen faces: the edit (its result is node `new`, the node on the faces
 *  `src`) and the quick-edit box's inputs. */
interface BodyChange { edit: GraphEdit; fields: (ids: Record<string, string>) => QuickField[] }
type BuildChange = (c: FaceContext, slot: BodySlot, key: string, size: number | null, path?: string) => BodyChange

/** Holes a face can get in one step: a Hole Pattern cut out of the shown body (Boolean Subtract) or, for
 *  raised shapes, joined to it (Union); settings from the presets of the same name. */
interface HoleKind { label: string; title: string; through: boolean; op: string; inputs: Record<string, unknown>; edge?: Record<string, unknown> }
const HOLES: HoleKind[] = [
  { label: 'Round holes', title: 'Round through holes, staggered', through: true, op: 'implicit.subtract',
    inputs: { shape: 'round', size: 4, layout: 'staggered', spacing: 6, margin: 2 } },
  { label: 'Honeycomb vents', title: 'Hexagonal through holes packed as a honeycomb', through: true, op: 'implicit.subtract',
    inputs: { shape: 'hexagon', size: 4, corner_radius: 0.4, layout: 'staggered', spacing: 5.2, margin: 2 } },
  { label: 'Slot vents', title: 'Rows of upright slots with round ends', through: true, op: 'implicit.subtract',
    inputs: { shape: 'slot', size: 2.5, length: 12, rotation: 90, layout: 'grid', spacing: 5, margin: 3 } },
  { label: 'Round grille', title: 'Round holes in a sunflower spiral, for round faces (speaker, fan)', through: true, op: 'implicit.subtract',
    inputs: { shape: 'round', size: 2.5, layout: 'spiral', spacing: 4, margin: 2 } },
  { label: 'Louvre vents', title: 'Slots drilled at 30° (the outside end lower) on an upright wall: rain and eyes stay out',
    through: true, op: 'implicit.subtract',
    inputs: { shape: 'slot', size: 2, length: 14, rotation: 90, align: 'z', layout: 'grid', spacing: 5, tilt: -30, margin: 3 } },
  { label: 'Voronoi vents', title: 'Organic openings with an even web, their edges rounded on both sides', through: true,
    op: 'implicit.subtract', edge: { edge: 'round', radius: 0.5 },
    inputs: { shape: 'voronoi', web: 1.4, corner_radius: 0.6, layout: 'staggered', spacing: 7, randomness: 0.8, margin: 3 } },
  { label: 'Pockets', title: 'Round pockets 1 mm deep', through: false, op: 'implicit.subtract',
    inputs: { shape: 'round', size: 4, layout: 'staggered', spacing: 6, depth: 1, margin: 2 } },
  { label: 'Raised hexagons', title: 'Flat-topped hexagons standing 1 mm off the face', through: false, op: 'implicit.union',
    inputs: { shape: 'hexagon', size: 4, corner_radius: 0.3, layout: 'staggered', spacing: 6, height: 1, margin: 2 } },
]

function holeEdit(c: FaceContext, slot: BodySlot, h: HoleKind, key: string): BodyChange {
  const chain: GraphEdit = {
    add: [{ key: 'src', type: 'pattern.holes', inputs: { faces: key, ...h.inputs }, near: c.main.node },
      { key: 'new', type: h.op, inputs: h.edge ?? {}, near: slotAnchor(slot) }],
    links: [{ from: c.main.node, port: c.main.port, to: 'src', input: 'cad' }, { from: 'src', port: 'out', to: 'new', input: 'b' }],
    select: ['new'],
  }
  return {
    edit: withSource(c.main, modifyEdit(slot, chain, { key: 'new', input: 'a' }, { key: 'new', port: 'out' })),
    fields: (ids) => [
      { node: ids.src, port: 'shape' }, { node: ids.src, port: 'size' }, { node: ids.src, port: 'web' },
      { node: ids.src, port: 'length' }, { node: ids.src, port: 'spacing' },
      ...(h.op === 'implicit.union' ? [{ node: ids.src, port: 'height' }] : h.through ? [] : [{ node: ids.src, port: 'depth' }]),
      { node: ids.src, port: 'margin' }, { node: ids.new, port: 'edge', label: 'Edges' }, { node: ids.new, port: 'radius', label: 'Edge size' },
    ],
  }
}

/** Lettering or a picture on a face in one step: Text or Picture → Texture (a negative height sinks it). */
interface LetterKind { label: string; title: string; type: 'pattern.text' | 'pattern.picture'; height: number }
const LETTERING: LetterKind[] = [
  { label: 'Embossed text', title: 'Raised lettering, 0.6 mm', type: 'pattern.text', height: 0.6 },
  { label: 'Engraved text', title: 'Lettering sunk 0.5 mm into the face', type: 'pattern.text', height: -0.5 },
  { label: 'Embossed picture…', title: 'A logo raised 0.8 mm off the face (dark parts up): choose a PNG, JPG or BMP',
    type: 'pattern.picture', height: 0.8 },
  { label: 'Engraved picture…', title: 'A logo sunk 0.5 mm into the face (dark parts down): choose a PNG, JPG or BMP',
    type: 'pattern.picture', height: -0.5 },
]
const PICTURE_EXTS = ['.png', '.jpg', '.jpeg', '.bmp']

function letterEdit(c: FaceContext, slot: BodySlot, l: LetterKind, key: string, size: number | null, path?: string): BodyChange {
  const text = l.type === 'pattern.text'
  // sized to the face: letters about a fifth of it, a picture most of it
  const inputs = text ? { faces: key, text: 'Text', height: size ? Math.min(8, Math.max(2, roundNice(size / 5))) : 8 }
    : { faces: key, path: path ?? '', width: size ? Math.min(40, roundNice(size * 0.6)) : 25 }
  const chain: GraphEdit = {
    add: [{ key: 'src', type: l.type, inputs, near: c.main.node },
      { key: 'new', type: 'implicit.texture', inputs: { height: l.height }, near: slotAnchor(slot) }],
    links: [{ from: c.main.node, port: c.main.port, to: 'src', input: 'cad' }, { from: 'src', port: 'out', to: 'new', input: 'pattern' }],
    select: ['new'],
  }
  return {
    edit: withSource(c.main, modifyEdit(slot, chain, { key: 'new', input: 'body' }, { key: 'new', port: 'out' })),
    fields: (ids) => [
      ...(text ? [{ node: ids.src, port: 'text' }, { node: ids.src, port: 'height', label: 'Letter height' }, { node: ids.src, port: 'font' }]
        : [{ node: ids.src, port: 'width' }, { node: ids.src, port: 'invert' }]),
      { node: ids.src, port: 'angle' }, { node: ids.new, port: 'height', label: l.height < 0 ? 'Depth (negative)' : 'Raised by' },
    ],
  }
}

/** Deletes a node and keeps what runs through it whole: what used its first output now uses what came
 *  into its input of that type (a support or load's Add to, an Offset Body's body). */
function spliceEdit(st: State, node: string): GraphEdit {
  const n = nodeOf(st, node)
  const spec = st.specs[n?.data.typeId ?? '']
  const input = bypassInput(spec, (p) => !!linkInto(st, node, p))
  const prev = input ? linkInto(st, node, input) : undefined
  const edit: GraphEdit = { remove: [node], links: [] }
  if (prev && spec) {
    for (const e of st.edges) {
      if (e.source === node && e.sourceHandle === spec.outputs[0].name && e.targetHandle) {
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

/** One item per chain end when there are several (two analyses on one body, or load cases: a load after
 *  a case's last load joins that case, after the last Load Case it acts in every case), else `run`. */
function perTail(st: State, label: string, tails: string[], run: (tail: string | null) => MenuItem['run'], alone: string): MenuItem {
  if (tails.length < 2) return item(label, run(tails[0] ?? null))
  const users = (tail: string) => st.edges.filter((e) => e.source === tail && e.sourceHandle === 'conditions').map((e) => e.target)
  const cases = loadCases(st.nodes, st.edges)
  const where = (tail: string) => {
    if (!cases.list.length) return `After ${tail}`
    const c = cases.of[tail]
    const isCase = st.nodes.find((n) => n.id === tail)?.data.typeId === LOAD_CASE
    return c && !isCase ? `In ${c.name} (after ${tail})` : `In every case (after ${tail})`
  }
  return item(label, undefined, {
    items: [
      ...tails.map((tail) => item(where(tail), run(tail), { hint: users(tail).length ? `→ ${users(tail).join(', ')}` : undefined })),
      item(alone, run(null)),
    ],
  })
}

/** The last action that put something on a face, to do again on another face ("Again: …"). */
let again: { label: string; run: (hit: FaceHit, info: FaceInfo | null, ctx: MenuContext) => void } | null = null
// another recipe has other faces and nodes: nothing to do again there
useStore.subscribe((s, prev) => { if (s.loadCount !== prev.loadCount) again = null })

/** The face menu: supports, loads, textures and decorations on it (or on all the chosen faces), adding
 *  it to nodes on the same body, the nodes using it, choosing faces. `info` (the face's surface type
 *  and size, when the engine tells) orders the load kinds (a hole lists Bearing and Torque first) and
 *  sizes new textures. */
export function faceMenu(hit: FaceHit, info: FaceInfo | null, ctx: MenuContext): MenuEntry[] {
  const c = faceContext(useStore.getState(), hit)
  const { st, main } = c
  const many = c.targets.length > 1
  const source = main.pick ? `a new Pick Body (body ${main.pick.index})` : main.node
  const pickMore = (node: string) => (ids: Record<string, string>): ToastAction[] =>
    [{ label: 'Pick more faces', run: () => useStore.getState().setPicking({ node: ids[node], source: ids.pick ?? main.node }) }]
  /** an item whose action can be done again on another face: `act` builds it for any face */
  const repeatable = (label: string, act: (h: FaceHit, i: FaceInfo | null, x: MenuContext, shift: boolean) => void): MenuItem['run'] =>
    (o) => {
      again = { label, run: (h, i, x) => act(h, i, x, false) }
      act(hit, info, ctx, o.shift)
    }
  /** the same chain end on another face, if it has it, else its only one */
  const sameTail = (c2: FaceContext, tail: string | null) => (tail && c2.tails.includes(tail) ? tail : c2.tails.length === 1 ? c2.tails[0] : null)

  /** an item adding a support or load; asks which chain (two analyses on this body) and, on copies,
   *  whether on this body only or on every copy */
  const chainItem = (label: string, type: string, inputs: Record<string, unknown>): MenuItem => {
    const make = (everyCopy: boolean) => (tail: string | null): MenuItem['run'] => repeatable(label, (h, _i, x, shift) => {
      const c2 = faceContext(useStore.getState(), h)
      const t2 = h === hit ? tail : sameTail(c2, tail)
      const ids = apply(chainEdit(c2, type, inputs, t2, keysFor(c2, everyCopy)), label, {
        text: (ids) => `Added ${ids.new} (${st.specs[type]?.title ?? type}) on ${facesLabel(c2)} ← ${ids.pick ?? c2.main.node}` +
          (t2 ? `, after ${t2}` : ''),
        actions: pickMore('new'),
      })
      if (ids && type === 'sim.load' && !shift) useStore.getState().setQuickEdit({ node: ids.new, ...x.at })
    })
    if (!c.copies || main.node !== c.cad) return perTail(st, label, c.tails, make(false), 'On its own')
    return item(label, undefined, {
      items: [
        perTail(st, 'On this body only', c.tails, make(false), 'On its own'),
        { ...perTail(st, 'On every copy', c.tails, make(true), 'On its own'), title: `Face ${c.face} of all ${c.bodyCount} copies` },
      ],
    })
  }

  /** a decoration on the faces (copies: every copy first), optionally connected straight on */
  const decorateItem = (d: (typeof DECORATIONS)[number], connect?: FreeInput): MenuItem => {
    const title = st.specs[d.type]?.title ?? d.type
    const make = (everyCopy: boolean): MenuItem['run'] => repeatable(title, (h, _i, x, shift) => {
      const c2 = faceContext(useStore.getState(), h)
      const to = h === hit ? connect : undefined
      const ids = apply(decorateEdit(c2, d.type, keysFor(c2, everyCopy), to), title, {
        text: (ids) => `Added ${ids.new} (${title}) on ${facesLabel(c2)} ← ${ids.pick ?? c2.main.node}` +
          (to ? ` → ${to.node}.${to.input}` : `: ${d.next}`),
        actions: pickMore('new'),
      })
      if (!ids) return
      if (!to) useStore.getState().addLog(`${ids.new}: ${d.next}`)
      if (!shift && hasQuickEdit(d.type)) useStore.getState().setQuickEdit({ node: ids.new, ...x.at })
    })
    const label = connect ? `…and ${connect.label}` : d.label
    if (!c.copies || main.node !== c.cad) return item(label, make(true))
    return item(label, undefined, { items: [item('On every copy', make(true)), item('On this body only', make(false))] })
  }

  /** a change on the faces in one step, into the shown body (copies: every copy first): a texture, holes,
   *  lettering; `ask` first asks for what it needs (a picture file) */
  const slot = bodySlot(st, hit.view, hit.view === hit.source ? main : null)
  const changeItem = (label: string, title: string, build: BuildChange, ask?: () => Promise<string | null>): MenuItem => {
    const make = (everyCopy: boolean): MenuItem['run'] => repeatable(label, async (h, i, x, shift) => {
      const path = ask ? await ask() : undefined
      if (path === null) return
      const s2 = useStore.getState()
      const c2 = faceContext(s2, h)
      const slot2 = bodySlot(s2, h.view, h.view === h.source ? c2.main : null)
      if (typeof slot2 === 'string') { s2.showToast(slot2); return }
      const size = !h.targets || h.targets.length < 2 ? (i ? Math.sqrt(i.area) : null) : null
      const { edit, fields } = build(c2, slot2, keysFor(c2, everyCopy), size, path)
      const ids = apply(edit, label, {
        text: (ids) => `${label} on ${facesLabel(c2)}: ${ids.src} → ${ids.new}` +
          (slot2.consumers.length ? ` → ${slot2.consumers.map((k) => k.node).join(', ')}` : slot2.swap ? ' (shown)' : ''),
        actions: (ids) => (ids.mask ? pickMore('mask')(ids) : pickMore('src')(ids)),
      })
      if (ids && !shift) s2.setQuickEdit({ node: ids.new, ...x.at, fields: fields(ids), title: `${label} on ${facesLabel(c2)}` })
    })
    const off = typeof slot === 'string' ? slot : editingReason(st)
    if (!c.copies || main.node !== c.cad) return item(label, make(true), { title, disabled: off })
    return item(label, undefined, { title, disabled: off,
      items: [item('On every copy', make(true)), item('On this body only', make(false))] })
  }
  const textureItem = (t: TextureKind) =>
    changeItem(t.label, t.title, (c2, slot2, key, size) => textureEdit(c2, slot2, t, key, size))
  const holeItem = (h: HoleKind) => changeItem(h.label, h.title, (c2, slot2, key) => holeEdit(c2, slot2, h, key))
  const letterItem = (l: LetterKind) => changeItem(l.label, l.title, (c2, slot2, key, size, path) => letterEdit(c2, slot2, l, key, size, path),
    l.type === 'pattern.picture' ? () => useBrowser.getState().open({ mode: 'open', exts: PICTURE_EXTS, title: 'Choose the picture', initial: '' }) : undefined)

  const kinds = info?.hole ? [2, 3, 0, 1].map((i) => LOAD_KINDS[i]) : LOAD_KINDS
  const far = hit.distance !== undefined && hit.distance > 0.5 ? `, ${hit.distance.toFixed(1)} mm from the click` : ''
  const entries: MenuEntry[] = [
    header(many ? `${plural(c.targets.length, 'face')} chosen${new Set(c.targets.map((t) => t.body)).size > 1 ? ' on several bodies' : ''}`
      : `Face ${c.face}${c.bodyCount > 1 ? ` · body ${c.body}${info?.body_name ? ` ${info.body_name}` : ''}` : ''}` +
        `${info ? ` · ${describe(info)}` : ''}`),
    ...(hit.view !== hit.source ? [header(`the CAD face${many ? 's' : ''} of ${hit.source} under ${hit.view}${far}`)] : []),
    ...(main.node !== hit.source ? [header(`new nodes use ${source}`)] : []),
    ...(again && !st.editing ? [item(`Again: ${again.label}`, ((a) => () => a.run(hit, info, ctx))(again),
      { title: `Do the last face action again, on ${facesLabel(c)}` })] : []),
    chainItem(many ? `Fix these ${c.targets.length} faces` : 'Fix this face', 'sim.fixed', {}),
    item(many ? 'Add load on these faces' : 'Add load here', undefined, {
      items: kinds.map(([kind, label]) =>
        chainItem(label, 'sim.load', { kind, ...(LOAD_DIRECTION[kind] ? { direction: LOAD_DIRECTION[kind] } : {}) })),
    }),
    item('Texture', undefined, {
      title: 'A texture on these faces in one step, added to the shown body',
      items: [
        item('Wizard…', () => openTextureWizard({ cad: main, faces: keysFor(c, true), view: hit.view,
          size: c.targets.length === 1 && info ? Math.sqrt(info.area) : null }), {
          title: 'Step by step: these faces, the texture and its settings, where it fades, then smoothing or a shell',
          disabled: typeof slot === 'string' ? slot : editingReason(st),
        }),
        sep,
        ...TEXTURES.map(textureItem), header('Shapes'), ...SHAPE_TEXTURES.map(textureItem),
      ],
    }),
    item('Holes', undefined, {
      title: 'Holes, pockets or raised shapes on these faces in one step, cut into (or added to) the shown body',
      items: [...HOLES.filter((h) => h.through).map(holeItem), header('Not through'), ...HOLES.filter((h) => !h.through).map(holeItem)],
    }),
    item('Text or picture', undefined, {
      title: 'Lettering or a logo on this face in one step, raised or sunk into the shown body',
      items: LETTERING.map(letterItem),
    }),
    item('Decorate', undefined, {
      items: DECORATIONS.flatMap((d) => {
        const free = freeInputs(c, d.type)
        return [decorateItem(d), ...(free.length === 1 ? [decorateItem(d, free[0])] : [])]
      }),
    }),
  ]
  const cadOut = outputOf(st, c.cad, ['cad'])
  if (info?.type === 'plane' && hit.view === hit.source && !many && cadOut) {
    entries.push(item('Mirror about this face', () => {
      const key = keysFor(c, false)
      apply(transformEdit(st, c.cad, cadOut.name, 'cad.mirror', 'cad', { plane: 'face', faces: key }), 'Mirror bodies',
        { text: (ids) => `${ids.new} mirrors ${c.cad}'s bodies about face ${key} (Keep original: yes)` })
    }, { disabled: editingReason(st), title: 'A mirrored copy of the bodies, about the plane of this face: half a part made whole' }))
  }
  if (c.others.length) {
    entries.push(item('Add to', undefined, {
      items: c.others.map(({ node, source }) => item(named(st, node),
        () => apply(addFacesEdit(c, node, source), `Add ${facesLabel(c)} to ${node.id}`, { text: `${facesLabel(c)} added to ${node.id}` }))),
    }))
  }
  if (c.users.length && !many) {
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
  entries.push(...selectEntries(c, hit, info, ctx))
  return entries
}

/** Choosing faces: this one (Ctrl+click), similar ones, tangent ones; saving them as a named selection. */
function selectEntries(c: FaceContext, hit: FaceHit, info: FaceInfo | null, ctx: MenuContext): MenuEntry[] {
  const out: MenuEntry[] = []
  const many = c.targets.length > 1
  if (ctx.select) {
    const select = ctx.select
    const items: MenuEntry[] = []
    if (!many) {
      items.push(item(ctx.selected ? 'Remove this face' : 'Add this face', () => select([{ face: c.face, body: c.body }],
        ctx.selected ? 'remove' : 'add'), { hint: 'Ctrl+click' }))
      const similar = (mode: 'size' | 'plane' | 'type', label: string) => item(label, async () => {
        const faces = await loadSimilarFaces(hit.view, c.body, c.face, mode,
          hit.view === hit.source ? undefined : { node: hit.source, port: hit.port ?? 'body' })
        const st = useStore.getState()
        if (typeof faces === 'string') { st.showToast(`Similar faces: ${faces}`); return }
        select(faces.map((face) => ({ face, body: c.body })), 'replace')
        st.showToast(`${plural(faces.length, 'face')} chosen: right-click one of them to act on all`)
      })
      const d = info?.radius ? `Ø ${(info.radius * 2).toFixed(info.radius < 5 ? 2 : 1)} mm` : ''
      items.push(item('Similar faces', undefined, {
        items: [
          similar('size', info?.radius ? `Same size: every ${info.hole ? 'hole' : 'shaft'} ${d}` : 'Same kind and area'),
          ...(info?.type === 'plane' ? [similar('plane', 'In the same plane')] : []),
          similar('type', info ? `Every ${info.type}${info.type === 'cylinder' ? (info.hole ? ' hole' : ' shaft') : ''}` : 'Same kind of surface'),
        ],
      }))
      if (ctx.tangent && ctx.tangent.length > 1) {
        items.push(item('Tangent faces', () => select(ctx.tangent!, 'add'),
          { hint: String(ctx.tangent.length), title: 'This face and every face joined to it smoothly' }))
      }
    }
    if (ctx.selectionSize) items.push(item('Clear the choice', () => useStore.getState().setFaceSel(null), { hint: 'Esc' }))
    out.push(item(many ? 'Chosen faces' : 'Choose faces', undefined, { items, hint: ctx.selectionSize ? String(ctx.selectionSize) : undefined }))
  }
  out.push(item('Save as selection…', () => {
    const st = useStore.getState()
    const keys = keysFor(c, true)
    st.setPrompt({ ...ctx.at, label: `Name for ${facesLabel(c)} (e.g. top, rim)`, value: '', done: (name) => {
      if (!/^[A-Za-z_][\w-]*$/.test(name)) { useStore.getState().showToast('A name starts with a letter and has no spaces'); return }
      apply({ selections: { [name]: keys } }, `Selection ${name}`,
        { text: `Saved ${facesLabel(c)} as "${name}": type the name into a Faces input (or its ☰ list)` })
    } })
  }, { disabled: editingReason(c.st), title: 'A named face selection: nodes can use the name, and picking with it updates every node that does' }))
  return out
}

function startPicking(node: string) {
  const e = linkInto(useStore.getState(), node, 'cad')
  if (e) useStore.getState().setPicking({ node, source: e.source })
}

/** A nice size for a new shape, about `share` of what is shown. */
const shapeSize = (ctx: MenuContext, share: number) => roundNice((ctx.sceneSize ?? 50) * share)
const vec = (p: [number, number, number]) => p.map((x) => Number(x.toFixed(2))).join(', ')

/** Changes to the shown body: grow or shrink it, hollow it, round it, fill it with a lattice, cut a
 *  shape out or add one at the click. */
function modifyEntries(st: State, view: string, cad: Source | null, ctx: MenuContext): MenuEntry[] {
  const slot = bodySlot(st, view, cad)
  if (typeof slot === 'string') return [item('Modify', undefined, { disabled: slot })]
  const near = slotAnchor(slot)
  /** one new node `type` taking the body on `input` */
  const simple = (label: string, type: string, input: string, inputs: Record<string, unknown>, note: string) => item(label, () => {
    const ids = apply(modifyEdit(slot, { add: [{ key: 'new', type, inputs, near }], select: ['new'] }, { key: 'new', input },
      { key: 'new', port: 'out' }), label, { text: (ids) => `${label}: ${ids.new}${note}` })
    if (ids && hasQuickEdit(type)) useStore.getState().setQuickEdit({ node: ids.new, ...ctx.at })
  })
  const lattice = (label: string, type: string, inputs: Record<string, unknown>, needsDomain: boolean) => item(label, () => {
    const chain: GraphEdit = { add: [{ key: 'lattice', type, inputs, near }, { key: 'new', type: 'lattice.fill', near: 'lattice' }],
      links: [{ from: 'lattice', port: 'out', to: 'new', input: 'lattice' }], select: ['new'] }
    if (needsDomain && cad) chain.links!.push({ from: cad.node, port: cad.port, to: 'lattice', input: 'domain' })
    const ids = apply(withSource(needsDomain ? cad : null, modifyEdit(slot, chain, { key: 'new', input: 'body' }, { key: 'new', port: 'out' })),
      label, { text: (ids) => `${label}: ${ids.lattice} in ${ids.new}` })
    if (ids) {
      useStore.getState().setQuickEdit({ node: ids.new, ...ctx.at, title: label, fields: [
        { node: ids.lattice, port: 'cell', label: 'Cell (mm)' }, { node: ids.lattice, port: 'thickness', label: 'Thickness (mm)' },
        ...(type === 'lattice.voronoi' ? [] : [{ node: ids.lattice, port: 'pattern' }]), { node: ids.new, port: 'skin', label: 'Skin (mm)' }] })
    }
  }, { disabled: needsDomain && !cad ? 'Needs the CAD body: right-click a face of it' : undefined })
  /** a Ball, Box or Bar at the click, cut out (Subtract) or added (Union) */
  const shape = (op: 'implicit.subtract' | 'implicit.union', type: string, label: string) => item(label, () => {
    const p = ctx.point ?? [0, 0, 0]
    const s = shapeSize(ctx, 0.12)
    const inputs: Record<string, unknown> = type === 'implicit.ball' ? { centre: vec(p), radius: s / 2 }
      : type === 'implicit.box' ? { centre: vec(p), size: `${s}, ${s}, ${s}` }
      : { start: vec([p[0] - s, p[1], p[2]]), end: vec([p[0] + s, p[1], p[2]]), radius: s / 4 }
    const verb = op === 'implicit.subtract' ? 'Cut out' : 'Added'
    const chain: GraphEdit = { add: [{ key: 'shape', type, inputs, near }, { key: 'new', type: op, near: 'shape' }],
      links: [{ from: 'shape', port: 'out', to: 'new', input: 'b' }], select: ['shape'] }
    const ids = apply(modifyEdit(slot, chain, { key: 'new', input: 'a' }, { key: 'new', port: 'out' }), `${verb} ${label}`,
      { text: (ids) => `${verb} ${ids.shape} (${label.toLowerCase()}) with ${ids.new}: drag its yellow handles to move it` })
    if (ids) useStore.getState().setQuickEdit({ node: ids.shape, ...ctx.at, title: `${label} (${ids.shape})`, fields: [
      ...(type === 'implicit.box' ? [{ node: ids.shape, port: 'size' }, { node: ids.shape, port: 'rounding' }] : [{ node: ids.shape, port: 'radius' }]),
      { node: ids.new, port: 'edge', label: 'Edge' }, { node: ids.new, port: 'radius', label: 'Edge radius' }] })
  }, { disabled: ctx.point ? undefined : 'Right-click on the body' })
  const shapes = (op: 'implicit.subtract' | 'implicit.union') =>
    [['implicit.ball', 'Ball'], ['implicit.box', 'Box'], ['implicit.bar', 'Bar']].map(([t, l]) => shape(op, t, l))
  const off = editingReason(st)
  return [item('Modify', undefined, {
    disabled: off,
    title: slot.swap ? 'Changes this body (the result is shown instead)' : `Changes the body ${view} is made from`,
    items: [
      simple('Offset (grow / shrink)', 'implicit.offset', 'body', { distance: 1 }, ': positive grows, negative shrinks'),
      simple('Shell (hollow wall)', 'implicit.shell', 'body', { thickness: 1 }, ': a wall centred on the surface'),
      simple('Smoothen', 'implicit.smoothen', 'body', {}, ': rounds off creases'),
      item('Fill with lattice', undefined, { items: [
        item('Wizard…', () => openLatticeWizard({ cad, view }),
          { title: 'Step by step: the lattice (seen in a section, with what it weighs), solid near bolt holes, open faces, grading, check' }),
        sep,
        lattice('Gyroid (TPMS)', 'lattice.tpms', { pattern: 'gyroid' }, false),
        lattice('Strut lattice', 'lattice.struts', {}, false),
        lattice('Voronoi foam', 'lattice.voronoi', {}, true),
      ] }),
      item('Cut out', undefined, { items: shapes('implicit.subtract'), title: 'A shape at the click, cut out of the body' }),
      item('Add shape', undefined, { items: shapes('implicit.union'), title: 'A shape at the click, joined to the body' }),
    ],
  })]
}

/** A Transform node (Move, Rotate, Mirror or Pattern Bodies; the bodies go into its `input`) right after the
 *  CAD output `out` of `node`: what used that output uses the transformed bodies instead, shown in its place. */
function transformEdit(st: State, node: string, out: string, type: string, input: string, inputs: Record<string, unknown>): GraphEdit {
  const users = st.edges.filter((e) => e.source === node && e.sourceHandle === out)
  return {
    add: [{ key: 'new', type, inputs, near: node }],
    links: [{ from: node, port: out, to: 'new', input },
      ...users.map((e) => ({ from: 'new', port: 'body', to: e.target, input: e.targetHandle! }))],
    select: ['new'], show: ['new'], hide: [node],
  }
}

/** Move, turn, mirror or repeat the CAD bodies of `view` (all it shows), with the new node's settings box
 *  open. A mirror plane beside the bodies (or through 0 where they end there: half a part). */
function transformEntries(st: State, view: string, ctx: MenuContext): MenuEntry[] {
  const out = outputOf(st, view, ['cad'])
  if (!out || st.views[view]?.kind !== 'cad') return []
  const b = st.views[view].bounds
  const edge = Math.abs(b[0][0]) < 1e-3 || Math.abs(b[1][0]) < 1e-3 ? 0 : Number(b[1][0].toFixed(2))
  const step = shapeSize(ctx, 1.2)
  const one = (label: string, type: string, input: string, inputs: Record<string, unknown>) => item(`${label}…`, () => {
    const ids = apply(transformEdit(st, view, out.name, type, input, inputs), `${label} bodies`,
      { text: (ids) => `${label}: ${ids.new} after ${view}${st.edges.some((e) => e.source === view && e.sourceHandle === out.name)
        ? ', what used its bodies uses these' : ''}` })
    if (ids) useStore.getState().setQuickEdit({ node: ids.new, ...ctx.at })
  })
  return [item('Transform', undefined, {
    disabled: editingReason(st),
    title: `Move, turn, mirror or repeat the bodies of ${view} (all of them); faces picked on them stay picked`,
    items: [
      one('Move', 'cad.move', 'body', {}),
      one('Rotate', 'cad.rotate', 'body', {}),
      one('Mirror', 'cad.mirror', 'cad', { position: edge }),
      one('Pattern', 'cad.pattern', 'body', { spacing: `${step}, 0, 0` }),
    ],
  })]
}

/** Export the mesh `node`.`port` to a file now: its Export Mesh node (or a new one), the file asked for,
 *  then an export of just that node. */
async function exportNow(node: string, port: string) {
  const st = useStore.getState()
  const existing = st.nodes.find((n) => n.data.typeId === 'output.export' && linkInto(st, n.id, 'mesh')?.source === node)
  const path = await useBrowser.getState().open({ mode: 'save', exts: MESH_EXTS, title: 'Export as',
    initial: String(existing?.data.inputs.path ?? `${node}.stl`) })
  if (!path) return
  let target = existing?.id
  if (existing) {
    apply({ inputs: [{ node: existing.id, port: 'path', value: path }], select: [existing.id] }, 'Export', { text: `Exporting ${path}…` })
  } else {
    const ids = apply({ add: [{ key: 'new', type: 'output.export', inputs: { path }, near: node }],
      links: [{ from: node, port, to: 'new', input: 'mesh' }], select: ['new'] }, 'Export', { text: (ids) => `Added ${ids.new}: exporting ${path}…` })
    target = ids?.new
  }
  if (target) await startRun('export', undefined, [target])
}

/** The Mesh from Implicit Body made from `node`'s body, if there is one. */
function meshOf(st: State, node: string) {
  const body = outputOf(st, node, ['implicit'])
  return body && st.edges.find((e) => e.source === node && e.sourceHandle === body.name &&
    nodeOf(st, e.target)?.data.typeId === 'mesh.from_implicit')?.target
}

/** Export the implicit body of `node` to a file now: through its Mesh from Implicit Body (or a new one). */
export async function meshAndExport(node: string) {
  const st = useStore.getState()
  const body = outputOf(st, node, ['implicit'])
  if (!body) return
  let m = meshOf(st, node)
  if (!m) {
    const ids = apply({ add: [{ key: 'new', type: 'mesh.from_implicit', near: node }],
      links: [{ from: node, port: body.name, to: 'new', input: 'body' }], select: ['new'] }, 'Mesh', { text: (ids) => `Added ${ids.new}` })
    m = ids?.new
  }
  if (m) await exportNow(m, 'mesh')
}

/** What only this node does: optimize, colour by a result, choose its file, export. */
function nodeEntries(st: State, view: string, ctx: MenuContext): MenuEntry[] {
  const node = nodeOf(st, view)
  if (!node) return []
  const type = node.data.typeId
  const spec = st.specs[type]
  const out: MenuEntry[] = []
  const off = editingReason(st)
  if (type === 'sim.topology' || type === 'sim.frame') {
    const running = st.run?.running && st.run.mode === 'optimize' && st.status[view]?.state === 'running'
    out.push(running ? item('Stop optimizing', () => { stopRun() }, { title: 'Keeps the result so far; Optimize continues from it' })
      : item('Optimize', () => { startOptimize(view) }, { disabled: off }))
  }
  const show = spec?.inputs.find((p) => p.name === 'show' && p.choices)
  if (show && (type === 'sim.static' || type === 'sim.topology')) {
    const now = String(node.data.inputs.show ?? show.default)
    out.push(item('Colour by', undefined, {
      items: show.choices!.map((ch) => item(ch, () => apply({ inputs: [{ node: view, port: 'show', value: ch }] }, `Show ${ch}`,
        { text: `${view} shows ${ch}` }), { checked: ch === now })),
    }))
  }
  if (type === 'input.import_step') {
    out.push(item('Choose STEP file…', async () => {
      const path = await useBrowser.getState().open({ mode: 'open', exts: ['.step', '.stp'], title: 'Open STEP file',
        initial: String(node.data.inputs.path ?? '') })
      if (path) apply({ inputs: [{ node: view, port: 'path', value: path }] }, 'STEP file', { text: `${view} reads ${path}` })
    }, { disabled: off }))
  }
  const v = st.views[view]
  const mesh = v?.kind === 'mesh' ? outputOf(st, view, ['mesh']) : undefined
  if (mesh) out.push(item('Export…', () => { exportNow(view, mesh.name) }, { disabled: off, title: 'Choose the file, then export it at Final quality' }))
  if (v?.kind === 'implicit') {
    const meshed = meshOf(st, view)
    if (outputOf(st, view, ['implicit'])) {
      out.push(item('Mesh and export…', () => { meshAndExport(view) },
        { disabled: off, title: meshed ? `Exports ${meshed}'s mesh` : 'Adds Mesh from Implicit Body, then exports it at Final quality' }))
    }
  }
  if (v && v.kind !== 'cad' && ctx.check) out.push(item('Print check', () => ctx.check!(view)))
  return out
}

/** A shown body: its node, editing and changing it, analysing it, using it in an optimization, picking
 *  it out of its STEP file, exporting it, the view around it. `hit`: the face under the cursor on a CAD
 *  view (decides the source as for the face menu); `behind`: the CAD face under an implicit body. */
export function bodyMenu(view: string, hit: FaceHit | null, ctx: MenuContext, behind?: FaceHit | null): MenuEntry[] {
  const st = useStore.getState()
  const node = nodeOf(st, view)
  const v = st.views[view]
  const tris = v ? `${(v.faces / 1000).toFixed(v.faces < 10000 ? 1 : 0)}k tris` : ''
  const others = st.visible.filter((id) => id !== view)
  const off = editingReason(st)
  const entries: MenuEntry[] = [
    header([view, node ? title(st, node) : '', tris, node?.data.bypass ? 'bypassed' : ''].filter(Boolean).join(' · ')),
    item('Edit settings…', () => useStore.getState().setQuickEdit({ node: view, ...ctx.at }),
      { disabled: off ?? (!node ? 'No node' : undefined), hint: 'double-click' }),
    item('Select node in graph', () => useStore.getState().focusNode(view), { disabled: off }),
  ]
  if (st.editing || !node) return [...entries, ...viewSpotEntries(ctx)]
  entries.push(...nodeEntries(st, view, ctx))

  // the body as a source: as the face menu would link it on a CAD view, else the node's own output
  // (a textured body is analysed or used as itself, not as the CAD body under it)
  const c = hit && hit.source === view ? faceContext(st, hit) : null
  const out = outputOf(st, view, v?.kind === 'cad' ? ['cad'] : ['implicit', 'scalar'])
  const src: Source | null = c ? c.main : out ? { node: view, port: out.name, key: '' } : null
  const cad = v?.kind === 'cad'
  entries.push(sep)
  // the CAD body behind (for a Voronoi foam's box): the clicked one, or the one under an implicit body
  const cadSource = c ? c.main : behind ? faceContext(st, behind).main : null
  entries.push(...modifyEntries(st, view, cadSource, ctx))
  if (cad) entries.push(...transformEntries(st, view, ctx))
  if (src) {
    const tails = c ? c.tails : chainTails(st, st.nodes.filter((n) => CHAIN_NODES.includes(n.data.typeId)).map((n) => n.id))
    const via = src.pick ? ` (through a new Pick Body of body ${src.pick.index})` : src.node !== view ? ` (through ${src.node})` : ''
    entries.push(item('Analyse', undefined, {
      title: `Material and the analysis, connected to this body${via} and to its supports and loads`,
      items: [
        ...(cad ? [item('Wizard', undefined, {
          title: 'Step by step: click the held and the loaded faces, set a few numbers, run',
          items: ANALYSES.map((a) => item(`${a.label}…`,
            () => openWizard(src.pick?.from ?? src.node, src.pick?.index ?? c?.body ?? 0, WIZARD_GOALS[a.type]))),
        }), sep] : []),
        ...ANALYSES.map((a) => perTail(st, a.label, tails, (tail) => () => apply(analyseEdit(st, src, a, tail), a.label, {
          text: (ids) => `Added ${ids.new} (${st.specs[a.type]?.title}) ← ${ids.pick ?? src.node}` +
            (tail ? `, supports & loads from ${tail}` : ': right-click faces to add supports and loads'),
          actions: (ids) => a.type === 'sim.static'
            ? [{ label: 'Show result', run: () => useStore.getState().toggleVisible(ids.new) }]
            : a.type === 'sim.topology' && tail ? [{ label: 'Optimize', run: () => { startOptimize(ids.new) } }] : [],
        }), 'Without supports and loads')),
      ],
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
  entries.push(item('Keep variant', () => useStore.getState().addVariant(),
    { title: 'Keep a copy of what is shown, to compare with your next changes (shown to the left)' }))
  entries.push(sep,
    item('Hide', () => useStore.getState().toggleVisible(view)),
    item('Show only this', () => others.forEach((id) => useStore.getState().toggleVisible(id)),
      { disabled: !others.length ? 'Nothing else is shown' : undefined }),
    ...viewSpotEntries(ctx))
  if (ctx.point && Object.keys(st.slices).length && st.slice.mode === 'plane') {
    const k = 'xyz'.indexOf(st.slice.axis)
    entries.push(item('Move the slice here', () => useStore.getState().setSliceSettings({ position: ctx.point![k] }),
      { hint: `${st.slice.axis} = ${ctx.point[k].toFixed(1)}`, title: 'The field slice goes through the clicked point' }))
  }
  entries.push(...nodeEditEntries(st, view))
  return entries
}

/** Looking at the clicked spot: the camera square to the surface, the section plane through it. */
function viewSpotEntries(ctx: MenuContext): MenuEntry[] {
  const out: MenuEntry[] = []
  const { point, normal } = ctx
  if (point && normal) {
    out.push(item('Look at this spot', () => useStore.getState().requestCamera({ dir: normal, target: point }),
      { title: 'Turns the view to look square onto the surface here' }))
  }
  if (point && ctx.section) {
    const section = ctx.section
    // the axis most across the surface first
    const order = (['x', 'y', 'z'] as Axis[]).sort((a, b) =>
      Math.abs(normal?.['xyz'.indexOf(b)] ?? 0) - Math.abs(normal?.['xyz'.indexOf(a)] ?? 0))
    out.push(item('Section here', undefined, {
      items: order.map((ax) => item(`Across ${ax.toUpperCase()}`, () => section(ax, point['xyz'.indexOf(ax)]),
        { hint: `${ax} = ${point['xyz'.indexOf(ax)].toFixed(1)}` })),
    }))
  }
  return out
}

/** Bypassing and deleting a node (keeping what runs through it whole). */
function nodeEditEntries(st: State, node: string): MenuEntry[] {
  const n = nodeOf(st, node)
  if (!n || st.editing) return []
  const spec = st.specs[n.data.typeId]
  const through = bypassInput(spec, (p) => !!linkInto(st, node, p))
  const on = !!n.data.bypass
  const splice = spliceEdit(st, node)
  return [
    sep,
    item('Bypass', () => apply({ bypass: [{ node, on: !on }] }, on ? 'Turn on' : 'Bypass',
      { text: on ? `${node} is on again` : `${node} is bypassed: ${through} passes straight through` }),
      { checked: on, disabled: !through && !on ? `Nothing to pass on: ${node} has no linked input of its output's type` : undefined,
        title: 'Leave the node out as if it were not there (its input passes straight on), to compare' }),
    item('Delete', undefined, { items: [
      item(`Delete ${node}`, () => apply({ remove: [node] }, `Delete ${node}`, { text: `Deleted ${node}` })),
      item('Delete and reconnect', () => apply(splice, `Delete ${node}`, {
        text: `Deleted ${node}${splice.links!.length ? `: ${splice.links!.map((l) => l.to).join(', ')} now ${splice.links!.length === 1 ? 'uses' : 'use'} ${splice.links![0].from}` : ''}`,
      }), { disabled: splice.links!.length ? undefined : 'Nothing runs through it', title: 'What used it takes what came into it' }),
    ] }),
  ]
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

/** What differs between a variant's recipe and the current one. */
export function recipeDiff(before: Recipe, after: Recipe): string[] {
  const out: string[] = []
  for (const [id, node] of Object.entries(after.nodes)) {
    const old = before.nodes[id]
    if (!old) { out.push(`+ ${id}`); continue }
    for (const port of new Set([...Object.keys(old.inputs ?? {}), ...Object.keys(node.inputs ?? {})])) {
      const a = old.inputs?.[port], b = node.inputs?.[port]
      if (JSON.stringify(a) === JSON.stringify(b)) continue
      const show = (v: unknown) => (v === undefined ? 'default' : isLink(v) ? `← ${v.link}` : String(v))
      out.push(`${id}.${port}: ${show(a)} → ${show(b)}`)
    }
    if (!!old.bypass !== !!node.bypass) out.push(`${id} ${node.bypass ? 'bypassed' : 'on again'}`)
  }
  for (const id of Object.keys(before.nodes)) if (!after.nodes[id]) out.push(`− ${id}`)
  if (JSON.stringify(before.groups ?? {}) !== JSON.stringify(after.groups ?? {})) out.push('groups changed')
  return out
}

/** A kept variant (drawn left of the result): what differs from now, putting it back, hiding it. */
export function variantMenu(id: number): MenuEntry[] {
  const st = useStore.getState()
  const v = st.variants.find((x) => x.id === id)
  if (!v) return []
  const diff = recipeDiff(v.recipe, st.recipe())
  return [
    header(`${v.label}: ${diff.length ? `${plural(diff.length, 'change')} since` : 'same settings as now'}`),
    ...diff.slice(0, 6).map((d) => header(`  ${d}`)),
    ...(diff.length > 6 ? [header(`  … ${diff.length - 6} more`)] : []),
    item('Restore this variant', () => {
      useStore.getState().restoreVariant(id)
      useStore.getState().showToast(`Restored ${v.label}`, diff, [{ label: 'Undo', run: () => useStore.getState().undo() }])
    }, { disabled: editingReason(st) ?? (diff.length ? undefined : 'Nothing differs'), title: 'Puts its settings back (Undo takes them away again)' }),
    item('Hide', () => useStore.getState().updateVariant(id, { shown: false })),
    item('Remove', () => useStore.getState().removeVariant(id)),
  ]
}

/** Import STEP… for a graph that has none (null otherwise). */
export function importItem(): MenuItem | null {
  const st = useStore.getState()
  if (st.editing || st.nodes.some((n) => n.data.typeId === 'input.import_step')) return null
  return item('Import STEP…', async () => {
    const path = await useBrowser.getState().open({ mode: 'open', exts: ['.step', '.stp'], title: 'Open STEP file', initial: '' })
    if (!path) return
    apply({ add: [{ key: 'new', type: 'input.import_step', inputs: { path } }], select: ['new'], show: ['new'] }, 'Import STEP', {
      text: (ids) => `Added ${ids.new} reading ${path}`,
      actions: (ids) => [{ label: 'Show node', run: () => useStore.getState().focusNode(ids.new) }],
    })
  })
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
    useStore.getState().setQuickEdit({ node: node.id, ...ctx.at })
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
  const splice = spliceEdit(st, node.id)
  const prev = splice.links![0]?.from
  const users = splice.links!.map((l) => l.to)
  entries.push(item(`Delete ${node.id}`, () => apply(splice, `Delete ${node.id}`, {
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

/** The standard views (Space in the 3D view, or their number key): where the camera looks from, Z up. */
export const STANDARD_VIEWS: { label: string; key: string; dir: [number, number, number] }[] = [
  { label: 'Front', key: '1', dir: [0, -1, 0] },
  { label: 'Back', key: '2', dir: [0, 1, 0] },
  { label: 'Left', key: '3', dir: [-1, 0, 0] },
  { label: 'Right', key: '4', dir: [1, 0, 0] },
  { label: 'Top', key: '5', dir: [0, 0, 1] },
  { label: 'Bottom', key: '6', dir: [0, 0, -1] },
  { label: 'Isometric', key: '7', dir: [1, -1, 1] },
]

/** The standard views as menu items. */
export function viewDirectionItems(): MenuEntry[] {
  return STANDARD_VIEWS.map((v) => item(v.label, () => useStore.getState().requestCamera({ dir: v.dir }), { hint: v.key }))
}
