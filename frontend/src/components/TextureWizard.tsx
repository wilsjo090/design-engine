import { useEffect, useRef } from 'react'
import { create } from 'zustand'
import { loadFaceInfo } from '../api'
import { useStore } from '../store'
import type { EngineNode, GraphEdit, QuickField } from '../types'
import { parseFaceKeys, resolveFaces } from '../types'
import {
  AMPLITUDE, SHAPE_TEXTURES, type Source, TEXTURES, type TextureKind, maskFade, meshAndExport,
  modifyEdit, patternNode, slotAnchor, textureFields, withSource,
} from '../viewActions'
import { closeOtherWizards, registerWizard } from '../wizards'
import { PickFaces, cadPort, faceCount, nodeOf, specOf, useSources, valueOf } from './AnalysisWizard'
import { useBrowser } from './FileBrowser'
import {
  AxisButtons, Setting, defaultSource, extentOf, findBase, linkInto, round1, slotFor, typeOf,
} from './wizardParts'

/** The steps, in order. */
const FACES = 0, TEXTURE = 1, MASKS = 2, MORE = 3
const STEP_LABELS = ['Faces', 'Texture', 'Masks', 'More']

/** What the wizard made: a Face Mask (`mask`) on faces of the CAD output `cad`.`port`, the faces kept in the
 *  saved selection `faces` (a Surface Pattern uses it too, and picking updates both); the shown body the
 *  texture goes into (`view`; null: found when it is added) and the faces' size (√area, mm: the pattern's
 *  scale); then its Texture node, and which texture each pattern node it made is (`kinds`). The pattern,
 *  the masks multiplied into the Texture's Fade and the steps after it are read from the graph, so Undo
 *  and edits in the graph keep the panel right. */
interface Made {
  cad: string
  port: string
  faces: string
  mask: string
  view: string | null
  size: number | null
  texture?: string
  kinds: Record<string, string>
}

interface TextureWizardState {
  open: boolean
  step: number
  /** the CAD node to pick faces on, before the mask is made (null: a shown body's) */
  source: string | null
  /** the shown body the next texture goes into (Another texture: the one just made) */
  base: string | null
  made: Made | null
}

/** The texture wizard: the faces, the texture and its settings (shown on the part), where it fades, then
 *  smoothing or a shell; it builds the nodes the face menu's Texture items do (Face Mask, Noise / 3D Wave /
 *  Surface Pattern, Texture), plus Multiply chains into the Fade and Smoothen / Shell after it. */
export const useTextureWizard = create<TextureWizardState>(() => ({ open: false, step: FACES, source: null, base: null, made: null }))

// another recipe: its nodes are not the ones the wizard made (the panel stays open)
useStore.subscribe((s, prev) => {
  if (s.loadCount !== prev.loadCount && !s.editing && !prev.editing) {
    useTextureWizard.setState({ step: FACES, source: null, base: null, made: null })
  }
})

type State = ReturnType<typeof useStore.getState>
const kinds = () => [...TEXTURES, ...SHAPE_TEXTURES]
const apply = (edit: GraphEdit, label: string, text: string | ((ids: Record<string, string>) => string)) =>
  useStore.getState().applyEdit(edit, label, { text })
const fresh = () => useTextureWizard.getState().made

const KIND_HINTS: Record<TextureKind['kind'], string> = {
  noise: 'Smooth random bumps; Scale is the size of the largest.',
  waves: 'Parallel ridges across an axis; Warp bends them.',
  gyroid: 'A 3D cell pattern of soft ridges.',
  shapes: 'Shapes laid out on the faces: raised, or sunk in with a negative Height.',
}

// --- reading the graph ----------------------------------------------------------------------------
/** A Surface Pattern's texture: the one whose settings it shares most (its shape counting twice). */
function guessKind(node: EngineNode | undefined): TextureKind | undefined {
  const t = node?.data.typeId, inputs = node?.data.inputs ?? {}
  if (t === 'field.noise') return TEXTURES.find((k) => k.kind === 'noise')
  if (t === 'field.wave3d') {
    const pattern = String(inputs.pattern ?? 'ridges')
    return TEXTURES.find((k) => k.kind === (pattern === 'ridges' ? 'waves' : pattern))
  }
  if (t !== 'pattern.surface') return undefined
  const score = (k: TextureKind) => Object.entries(k.shape?.inputs ?? {})
    .reduce((n, [p, v]) => n + (String(inputs[p] ?? (p === 'shape' ? 'round' : '')) === String(v) ? (p === 'shape' ? 2 : 1) : 0), 0)
  return SHAPE_TEXTURES.reduce((a, b) => (score(b) > score(a) ? b : a))
}

/** The pattern node feeding the Texture, and which texture it is. */
function patternOf(st: State, made: Made) {
  const e = made.texture ? linkInto(st, made.texture, 'pattern') : undefined
  if (!e) return null
  const label = made.kinds[e.source]
  return { src: e.source, kind: kinds().find((k) => k.label === label) ?? guessKind(nodeOf(st, e.source)) }
}

/** What is multiplied into the Texture's Fade after the mask, in order: per factor a Multiply (`mul`, the
 *  factor on its B) and the factor (`node`): a Face Mask leaving faces out, or a Smooth Step of a
 *  Coordinate (`coord`) fading along an axis. */
interface Factor { mul: string; node: string; kind: 'leave' | 'along' | 'other'; coord?: string }

function factorsOf(st: State, made: Made): Factor[] {
  const out: Factor[] = []
  let e = made.texture ? linkInto(st, made.texture, 'fade') : undefined
  while (e && e.source !== made.mask && typeOf(st, e.source) === 'math.multiply' && !out.some((f) => f.mul === e!.source)) {
    const mul = e.source
    const node = linkInto(st, mul, 'b')?.source ?? ''
    const t = typeOf(st, node)
    const coord = t === 'math.smoothstep' ? linkInto(st, node, 'x')?.source : undefined
    const along = !!coord && typeOf(st, coord) === 'field.coordinate'
    out.unshift({ mul, node, kind: t === 'field.face_mask' ? 'leave' : along ? 'along' : 'other', coord: along ? coord : undefined })
    e = linkInto(st, mul, 'a')
  }
  return out
}

const AFTER = ['implicit.smoothen', 'implicit.shell']
/** The steps after the Texture (Smoothen, Shell), each taking the one before on its Body. */
function afterOf(st: State, made: Made): string[] {
  const out: string[] = []
  for (let last = made.texture; last;) {
    const next = st.edges.filter((e) => e.source === last && e.sourceHandle === 'out')
    if (next.length !== 1 || next[0].targetHandle !== 'body' || !AFTER.includes(typeOf(st, next[0].target) ?? '') ||
      out.includes(next[0].target)) break
    last = next[0].target
    out.push(last)
  }
  return out
}
const resultOf = (st: State, made: Made) => {
  const after = afterOf(st, made)
  return after.length ? after[after.length - 1] : made.texture
}
/** The last of a list's Multiply nodes, else the mask: what the next factor multiplies. */
const lastMul = (made: Made, factors: Factor[]) => (factors.length ? factors[factors.length - 1].mul : made.mask)

/** The view the textured part is seen in: the last step's own, or the mesh made from it. */
function shownResult(st: State, made: Made) {
  const r = resultOf(st, made)
  if (r && st.visible.includes(r)) return r
  return made.view && made.view !== made.cad && st.visible.includes(made.view) ? made.view : null
}

/** The nodes whose faces the wizard picks: the mask, and Face Masks leaving faces out. */
const pickNodes = (st: State, made: Made) =>
  [made.mask, ...factorsOf(st, made).filter((f) => f.kind === 'leave').map((f) => f.node)]

/** The 3D view's pick banner gives way to the panel while it picks faces. */
export function useTextureWizardPicking() {
  const made = useTextureWizard((w) => (w.open ? w.made : null))
  return useStore((s) => !!made && !!s.picking && pickNodes(s, made).includes(s.picking.node))
}

/** √ of the chosen faces' area (up to 12 of them), from the shown CAD body; null when it is not shown. */
async function facesSize(made: Made) {
  const st = useStore.getState()
  const v = st.views[made.cad]
  if (!v || v.kind !== 'cad') return null
  const keys = parseFaceKeys(resolveFaces(made.faces, st.selections)).slice(0, 12)
  const infos = await Promise.all(keys.map((k) => {
    const [body, face] = k.includes(':') ? k.split(':').map(Number) : [0, Number(k)]
    return loadFaceInfo(made.cad, v.version, body, face)
  }))
  const area = infos.reduce((a, i) => a + (i?.area ?? 0), 0)
  return area > 0 ? Math.sqrt(area) : null
}

/** mm from the lowest to the highest point of the part along `axis` (from what is shown of it). */
const extent = (st: State, made: Made, axis: string) => extentOf(st, [shownResult(st, made), made.view, made.cad], axis)

// --- edits ------------------------------------------------------------------------------------------
/** After picking: stop picking the wizard's faces, and put the CAD body out of sight again while the
 *  textured part is shown (picking shows it). */
function settle() {
  const made = fresh()
  const st = useStore.getState()
  if (!made) return
  if (st.picking && pickNodes(st, made).includes(st.picking.node)) st.setPicking(null)
  const shown = made.texture ? shownResult(st, made) : null
  if (shown && shown !== made.cad && useStore.getState().visible.includes(made.cad)) useStore.getState().toggleVisible(made.cad)
}

/** Goes to a step: the first picks the faces, the others stop that. */
function go(step: number) {
  const made = fresh()
  useTextureWizard.setState({ step })
  if (!made) return
  if (step === FACES) useStore.getState().setPicking({ node: made.mask, source: made.cad })
  else settle()
}

/** The Face Mask on the faces (in a new saved selection), picking them next when there are none yet. */
function createMask(src: Source, faces: string, view: string | null, size: number | null): Made | null {
  const st = useStore.getState()
  let name = ''
  for (let i = 1; !name || st.selections[name] !== undefined; i++) name = `texture_faces_${i}`
  const ids = apply(withSource(src, {
    add: [{ key: 'mask', type: 'field.face_mask', inputs: { faces: name }, near: src.node }],
    links: [{ from: src.node, port: src.port, to: 'mask', input: 'cad' }],
    selections: { [name]: faces },
    select: ['mask'],
    ...(faces ? {} : { pick: 'mask' }),
  }), 'Texture wizard', (ids) => `Texture wizard: ${ids.mask} on the faces of ${ids.pick ?? src.node} (the selection "${name}")`)
  if (!ids) return null
  return { cad: ids.pick ?? src.node, port: src.pick ? 'body' : src.port, faces: name, mask: ids.mask, view, size, kinds: {} }
}

/** The texture `kind` on the faces, into the shown body (the CAD body: through a new Implicit Body),
 *  faded in by the mask; shown instead of that body. */
function addTexture(made: Made, kind: TextureKind) {
  const st = useStore.getState()
  const view = made.view && (st.visible.includes(made.view) || made.view === made.cad) ? made.view : findBase(st, made.cad) ?? made.cad
  const slot = slotFor(st, { node: made.cad, port: made.port }, view)
  if (typeof slot === 'string') { st.showToast(slot); return }
  const shapes = kind.kind === 'shapes'
  const chain: GraphEdit = {
    add: [patternNode(kind, made.size, shapes ? made.mask : slotAnchor(slot), made.faces),
      { key: 'new', type: 'implicit.texture', inputs: { height: kind.shape?.height ?? AMPLITUDE }, near: 'src' }],
    links: [
      ...(shapes ? [{ from: made.cad, port: made.port, to: 'src', input: 'cad' }] : []),
      { from: 'src', port: 'out', to: 'new', input: 'pattern' },
      { from: made.mask, port: 'out', to: 'new', input: 'fade' },
    ],
    inputs: [{ node: made.mask, port: 'fade', value: maskFade(kind, made.size) }],
    select: ['new'],
  }
  const ids = apply(modifyEdit(slot, chain, { key: 'new', input: 'body' }, { key: 'new', port: 'out' }), `Texture ${kind.label}`,
    (ids) => `${kind.label}: ${ids.src} → ${ids.new}, on the faces of ${made.mask}` +
      (slot.consumers.length ? ` → ${slot.consumers.map((c) => c.node).join(', ')}` : ''))
  if (!ids) return
  useTextureWizard.setState({ made: { ...made, view, texture: ids.new, kinds: { ...made.kinds, [ids.src]: kind.label } }, step: TEXTURE })
  settle()
}

let adding = false
/** From the faces to the texture: made with the faces' size the first time. */
async function toTexture() {
  const made = fresh()
  if (!made || adding) return
  if (made.texture) { go(TEXTURE); return }
  adding = true
  try {
    const size = made.size ?? await facesSize(made)
    const st = useStore.getState()
    if (st.picking?.node === made.mask) st.setPicking(null)
    const now = fresh()
    if (now && now.mask === made.mask && !now.texture) addTexture({ ...now, size }, TEXTURES[0])
  } finally {
    adding = false
  }
}

/** Another pattern for the Texture: the pattern node replaced (its height reset between fields and
 *  shapes, the shapes' own height for each shape; the mask's edge fade off under shapes). */
function switchKind(kind: TextureKind) {
  const made = fresh()
  const st = useStore.getState()
  const p = made && patternOf(st, made)
  if (!made?.texture || !p || p.kind?.label === kind.label) return
  const shapes = kind.kind === 'shapes', wasShapes = typeOf(st, p.src) === 'pattern.surface'
  const shared = st.edges.some((e) => e.source === p.src && !(e.target === made.texture && e.targetHandle === 'pattern'))
  const ids = apply({
    remove: shared ? [] : [p.src],
    add: [patternNode(kind, made.size, made.mask, made.faces)],
    links: [...(shapes ? [{ from: made.cad, port: made.port, to: 'src', input: 'cad' }] : []),
      { from: 'src', port: 'out', to: made.texture, input: 'pattern' }],
    inputs: [
      ...(shapes || wasShapes ? [{ node: made.texture, port: 'height', value: kind.shape?.height ?? AMPLITUDE }] : []),
      ...(shapes !== wasShapes ? [{ node: made.mask, port: 'fade', value: maskFade(kind, made.size) }] : []),
    ],
    select: [made.texture],
  }, `Texture ${kind.label}`, (ids) => `${made.texture}: ${kind.label.toLowerCase()} (${ids.src})`)
  if (ids) useTextureWizard.setState({ made: { ...made, kinds: { ...made.kinds, [ids.src]: kind.label } } })
}

/** Links the factors' Multiply nodes after the mask in order, the last into the Texture's Fade. */
function chainLinks(made: Made, factors: Factor[]) {
  const links: NonNullable<GraphEdit['links']> = factors.map((f, i) =>
    ({ from: i ? factors[i - 1].mul : made.mask, port: 'out', to: f.mul, input: 'a' }))
  links.push({ from: lastMul(made, factors), port: 'out', to: made.texture!, input: 'fade' })
  return links
}

/** Multiplies a factor into the Texture's Fade: a Face Mask leaving faces out (picked next), or a fade
 *  along Z over the lower half of the part. */
function addFactor(kind: 'leave' | 'along') {
  const made = fresh()
  const st = useStore.getState()
  if (!made?.texture) return
  const prev = lastMul(made, factorsOf(st, made))
  const edit: GraphEdit = { add: [], links: [], select: ['factor'] }
  const [lo, hi] = extent(st, made, 'z')
  if (kind === 'leave') {
    const fade = valueOf(st, made.mask, 'fade')
    edit.add!.push({ key: 'factor', type: 'field.face_mask', near: made.mask,
      inputs: { faces: '', apply: 'all other faces', fade: typeof fade === 'number' ? fade : 2 } })
    edit.links!.push({ from: made.cad, port: made.port, to: 'factor', input: 'cad' })
    edit.pick = 'factor'
  } else {
    edit.add!.push({ key: 'coord', type: 'field.coordinate', inputs: { axis: 'z' }, near: made.mask },
      { key: 'factor', type: 'math.smoothstep', inputs: { edge0: lo, edge1: round1((lo + hi) / 2) }, near: 'coord' })
    edit.links!.push({ from: 'coord', port: 'out', to: 'factor', input: 'x' })
  }
  edit.add!.push({ key: 'mul', type: 'math.multiply', near: 'factor' })
  edit.links!.push({ from: prev, port: 'out', to: 'mul', input: 'a' }, { from: 'factor', port: 'out', to: 'mul', input: 'b' },
    { from: 'mul', port: 'out', to: made.texture, input: 'fade' })
  apply(edit, kind === 'leave' ? 'Leave out faces' : 'Fade along an axis', (ids) => kind === 'leave'
    ? `${ids.factor} × ${made.texture}'s fade: click the faces to leave out`
    : `${ids.coord} → ${ids.factor} × ${made.texture}'s fade: none at ${lo}, all from ${round1((lo + hi) / 2)} mm along Z`)
}

function removeFactor(f: Factor) {
  const made = fresh()
  const st = useStore.getState()
  if (!made?.texture) return
  const rest = factorsOf(st, made).filter((g) => g.mul !== f.mul)
  apply({ remove: [f.mul, f.node, ...(f.coord ? [f.coord] : [])], links: chainLinks(made, rest) },
    `Remove ${f.node}`, `Removed ${f.node}: ${made.texture}'s fade no longer has it`)
}

/** The fade along another axis, over the same share of the part (flipped stays flipped). */
function setAxis(f: Factor, axis: string) {
  const made = fresh()
  const st = useStore.getState()
  if (!made || !f.coord) return
  const e0 = valueOf(st, f.node, 'edge0'), e1 = valueOf(st, f.node, 'edge1')
  const [lo, hi] = extent(st, made, axis)
  const mid = round1((lo + hi) / 2)
  const flipped = typeof e0 === 'number' && typeof e1 === 'number' && e0 > e1
  apply({ inputs: [{ node: f.coord, port: 'axis', value: axis },
    { node: f.node, port: 'edge0', value: flipped ? hi : lo }, { node: f.node, port: 'edge1', value: mid }] },
  `Fade along ${axis.toUpperCase()}`, `${f.node} fades along ${axis.toUpperCase()}`)
}

function flip(f: Factor) {
  const st = useStore.getState()
  const e0 = valueOf(st, f.node, 'edge0'), e1 = valueOf(st, f.node, 'edge1')
  if (typeof e0 !== 'number' || typeof e1 !== 'number') return
  apply({ inputs: [{ node: f.node, port: 'edge0', value: e1 }, { node: f.node, port: 'edge1', value: e0 }] },
    'Flip fade', `${f.node}: the texture now fades out the other way`)
}

/** Smoothen or Shell after the texture (after the last such step): what used it uses this, shown instead. */
function addAfter(type: string, inputs: Record<string, unknown>, label: string) {
  const made = fresh()
  const st = useStore.getState()
  const last = made && resultOf(st, made)
  if (!last) return
  const users = st.edges.filter((e) => e.source === last && e.sourceHandle === 'out' && e.targetHandle)
  const shown = st.visible.includes(last)
  apply({
    add: [{ key: 'new', type, inputs, near: last }],
    links: [{ from: last, port: 'out', to: 'new', input: 'body' },
      ...users.map((e) => ({ from: 'new', port: 'out', to: e.target, input: e.targetHandle! }))],
    select: ['new'],
    ...(shown ? { show: ['new'], hide: [last] } : {}),
  }, label, (ids) => `${label}: ${ids.new} after ${last}`)
}

function removeAfter(id: string) {
  const made = fresh()
  const st = useStore.getState()
  if (!made?.texture) return
  const after = afterOf(st, made)
  const k = after.indexOf(id)
  const prev = k > 0 ? after[k - 1] : made.texture
  const users = st.edges.filter((e) => e.source === id && e.sourceHandle === 'out' && e.targetHandle)
  apply({ remove: [id], links: users.map((e) => ({ from: prev, port: 'out', to: e.target, input: e.targetHandle! })),
    ...(st.visible.includes(id) ? { show: [prev] } : {}) }, `Remove ${id}`, `Removed ${id}`)
}

/** Closes the panel (the nodes stay; reopening goes on where it was). */
export function closeTextureWizard() {
  if (!useTextureWizard.getState().open) return
  settle()
  useTextureWizard.setState({ open: false })
}
registerWizard('texture', closeTextureWizard)

/** Opens the wizard; with `start` (the face menu) a new texture on those faces goes on at once. */
export function openTextureWizard(start?: { cad: Source; faces: string; view: string; size: number | null }) {
  closeOtherWizards('texture')
  useTextureWizard.setState({ open: true })
  if (!start || useStore.getState().editing) return
  settle()
  const made = createMask(start.cad, start.faces, start.view, start.size)
  if (made) {
    useTextureWizard.setState({ made, step: FACES, source: null, base: null })
    addTexture(made, TEXTURES[0])
  }
}

// --- the panel --------------------------------------------------------------------------------------
function PartStep() {
  const { source, base } = useTextureWizard()
  const sources = useSources()
  const fallback = useStore((s) => defaultSource(s, sources))
  const chosen = source && sources.includes(source) ? source : fallback
  const title = useStore((s) => (chosen ? specOf(s, chosen)?.title : ''))

  const openStep = async () => {
    const path = await useBrowser.getState().open({ mode: 'open', exts: ['.step', '.stp'], title: 'Open the part (STEP)', initial: '' })
    if (!path) return
    const ids = useStore.getState().applyEdit({ add: [{ key: 'new', type: 'input.import_step', inputs: { path } }], show: ['new'] },
      'Import STEP', { text: (ids) => `Added ${ids.new} reading ${path}` })
    if (ids) useTextureWizard.setState({ source: ids.new, base: null })
  }
  const start = () => {
    if (!chosen) return
    const port = cadPort(useStore.getState(), chosen)
    if (!port) return
    const made = createMask({ node: chosen, port, key: '' }, '', base, null)
    if (made) useTextureWizard.setState({ made, step: FACES, base: null })
  }

  if (!sources.length) {
    return (
      <>
        <div>Open the <b>part</b> to put a texture on.</div>
        <div className="wiz-buttons"><button className="primary" onClick={openStep}>Open STEP file…</button></div>
      </>
    )
  }
  return (
    <>
      <div>Which part, then click the <b>faces</b> that get the texture.</div>
      <label className="wiz-row">
        <span>Part</span>
        <select value={chosen ?? ''} onChange={(e) => useTextureWizard.setState({ source: e.target.value, base: null })}>
          {sources.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>
      <div className="muted small">{title}{base ? ` · on top of ${base}` : ''}</div>
      <div className="wiz-buttons">
        <button className="small-btn" onClick={openStep}>Other STEP file…</button>
        <button className="primary" disabled={!chosen} onClick={start}>Pick faces ›</button>
      </div>
    </>
  )
}

function FacesStep({ made }: { made: Made }) {
  const count = useStore((s) => faceCount(s, made.mask))
  const others = useStore((s) => valueOf(s, made.mask, 'apply') === 'all other faces')
  return (
    <>
      <div>Click the faces that {others ? <><b>stay plain</b> (the texture goes on all the others)</> : <>get the <b>texture</b></>}.</div>
      <div className="muted small">Click a face again to drop it · Shift+click: every face joined smoothly to it ·
        Ctrl+drag: paint over faces</div>
      <PickFaces node={made.mask} space={made.cad} what={others ? 'kept plain' : 'chosen'} />
      <div className="muted small">on {made.cad}; kept as the selection "{made.faces}"</div>
      <div className="wiz-buttons">
        <button className="primary" disabled={!count && !others} onClick={() => { toTexture() }}>Next ›</button>
      </div>
    </>
  )
}

function TextureStep({ made }: { made: Made }) {
  // (read from the graph as it is: re-rendered when it changes)
  useStore((s) => s.nodes)
  useStore((s) => s.edges)
  const p = patternOf(useStore.getState(), made)
  const texture = made.texture!
  const fields: QuickField[] = p?.kind ? textureFields(p.kind, p.src, texture) : [{ node: texture, port: 'height', label: 'Height (mm)' }]
  const button = (k: TextureKind) => (
    <button key={k.label} className={`small-btn ${p?.kind?.label === k.label ? 'on' : ''}`} title={KIND_HINTS[k.kind]}
      onClick={() => switchKind(k)}>{k.label}</button>
  )
  return (
    <>
      <div>Choose the texture: it is shown on the part. Change the numbers until it looks right.</div>
      <div className="tex-kinds">{TEXTURES.map(button)}</div>
      <div className="muted small">Shapes on the faces</div>
      <div className="tex-kinds">{SHAPE_TEXTURES.map(button)}</div>
      {p?.kind ? <div className="muted small">{KIND_HINTS[p.kind.kind]}</div>
        : <div className="muted small">{p ? `The pattern is ${p.src}: its settings are on its node.` : 'No pattern is connected.'}</div>}
      {fields.map((f) => <Setting key={`${f.node}.${f.port}`} node={f.node} port={f.port} label={f.label} />)}
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(FACES)}>‹ Faces</button>
        <button className="small-btn" onClick={() => useStore.getState().focusNode(texture)}>Show node</button>
        <button className="primary" onClick={() => go(MASKS)}>Next ›</button>
      </div>
    </>
  )
}

function LeaveCard({ made, f }: { made: Made; f: Factor }) {
  return (
    <div className="wiz-load">
      <div className="wiz-row">
        <b>Leave out faces</b> <span className="muted">{f.node}</span>
        <button className="small-btn" title={`Remove ${f.node}`} onClick={() => removeFactor(f)}>×</button>
      </div>
      <PickFaces node={f.node} space={made.cad} what="left out" />
      <Setting node={f.node} port="fade" label="Edge fade" />
    </div>
  )
}

function AlongCard({ f }: { f: Factor }) {
  const axis = useStore((s) => String(valueOf(s, f.coord!, 'axis') ?? 'z'))
  return (
    <div className="wiz-load">
      <div className="wiz-row">
        <b>Fade along</b>
        <AxisButtons axis={axis} onPick={(a) => setAxis(f, a)} />
        <button className="small-btn" title="Fade out the other way" onClick={() => flip(f)}>Flip</button>
        <button className="small-btn" title={`Remove ${f.node} and ${f.coord}`} onClick={() => removeFactor(f)}>×</button>
      </div>
      <Setting node={f.node} port="edge0" label="None at (mm)" />
      <Setting node={f.node} port="edge1" label="Full at (mm)" />
      <div className="muted small">Along {axis.toUpperCase()}: no texture at "None at", all of it from "Full at" on.</div>
    </div>
  )
}

function MasksStep({ made }: { made: Made }) {
  useStore((s) => s.nodes)
  useStore((s) => s.edges)
  const factors = factorsOf(useStore.getState(), made)
  const fade = useStore((s) => valueOf(s, made.mask, 'fade'))
  const others = useStore((s) => valueOf(s, made.mask, 'apply') === 'all other faces')
  const shapes = useStore((s) => {
    const src = made.texture ? linkInto(s, made.texture, 'pattern')?.source : undefined
    return !!src && typeOf(s, src) === 'pattern.surface'
  })
  return (
    <>
      <div>Where the texture goes, and where it fades out.</div>
      <Setting node={made.mask} port="fade" label="Edge fade" />
      {typeof fade === 'number' && fade > 0 && <Setting node={made.mask} port="position" label="Fade" />}
      {!shapes && <Setting node={made.mask} port="apply" label="Apply to" />}
      <div className="muted small">
        {shapes ? 'The shapes stay inside the faces by themselves (their Edge Margin is on the pattern node).'
          : others ? 'The texture goes on every face except those picked in the first step.'
          : 'Edge fade: how wide the texture runs out at the border of the faces (0: a hard edge).'}
      </div>
      {factors.map((f) => f.kind === 'leave' ? <LeaveCard key={f.mul} made={made} f={f} />
        : f.kind === 'along' ? <AlongCard key={f.mul} f={f} />
        : (
          <div className="wiz-load" key={f.mul}>
            <div className="wiz-row">
              <span>× <b>{f.node || '(nothing)'}</b> <span className="muted">in {f.mul}</span></span>
              <button className="small-btn" title={`Remove ${f.mul}`} onClick={() => removeFactor(f)}>×</button>
            </div>
          </div>
        ))}
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => addFactor('leave')} title="Faces on which the texture stays off, picked next">+ Leave out faces</button>
        <button className="small-btn" onClick={() => addFactor('along')} title="The texture runs out along an axis, e.g. toward the bottom">+ Fade along an axis</button>
      </div>
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(TEXTURE)}>‹ Texture</button>
        <button className="primary" onClick={() => go(MORE)}>Next ›</button>
      </div>
    </>
  )
}

/** A step after the texture, on or off, with its settings while on. */
function AfterRow({ made, type, label, hint, inputs, ports }: {
  made: Made; type: string; label: string; hint: string; inputs: Record<string, unknown>; ports: [string, string][]
}) {
  const id = useStore((s) => afterOf(s, made).find((n) => typeOf(s, n) === type))
  return (
    <div className={`wiz-load ${id ? 'on' : ''}`}>
      <div className="wiz-row">
        <button type="button" role="switch" aria-checked={!!id} className={`toggle ${id ? 'on' : ''}`}
          onClick={() => (id ? removeAfter(id) : addAfter(type, inputs, label))}>
          <span className="toggle-knob" />
        </button>
        <b>{label}</b>{id && <span className="muted">{id}</span>}
      </div>
      <div className="muted small">{hint}</div>
      {id && ports.map(([port, text]) => <Setting key={port} node={id} port={port} label={text} />)}
    </div>
  )
}

function MoreStep({ made }: { made: Made }) {
  const another = () => {
    const now = fresh()
    if (!now?.texture) return
    settle()
    const st = useStore.getState()
    useTextureWizard.setState({ made: null, step: FACES, source: now.cad, base: shownResult(st, now) ?? resultOf(st, now) ?? null })
  }
  return (
    <>
      <div>Anything else for the textured part:</div>
      <AfterRow made={made} type="implicit.smoothen" label="Smoothen" inputs={{}}
        hint="Rounds off the texture's sharp edges and creases." ports={[['iterations', 'Passes'], ['grid_size', 'Grid (mm)']]} />
      <AfterRow made={made} type="implicit.shell" label="Hollow (shell)" inputs={{ thickness: 1 }}
        hint="A wall of even thickness, centred on the textured surface (a lamp shade)." ports={[['thickness', 'Thickness']]} />
      <div className="wiz-buttons">
        <button className="small-btn" onClick={another} title="Another texture on top of this one: on other faces, or the same">+ Another texture</button>
        <button className="small-btn" onClick={() => { const r = resultOf(useStore.getState(), made); if (r) meshAndExport(r) }}
          title="Mesh the textured part and save it (at Final quality)">Mesh and export…</button>
      </div>
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(MASKS)}>‹ Masks</button>
        <button className="primary" onClick={closeTextureWizard}>Done</button>
      </div>
    </>
  )
}

/** The wizard's panel beside the 3D view. */
export function TextureWizard() {
  const { open, step, made } = useTextureWizard()
  const editing = useStore((s) => !!s.editing)
  const maskGone = useStore((s) => !!made && !nodeOf(s, made.mask))
  const textureGone = useStore((s) => !!made?.texture && !nodeOf(s, made.texture))
  const picking = useStore((s) => s.picking?.node ?? null)
  const picked = useRef(picking)

  // the nodes were deleted or undone away: start over, or from the faces
  useEffect(() => { if (maskGone) useTextureWizard.setState({ made: null, step: FACES }) }, [maskGone])
  useEffect(() => {
    const now = fresh()
    if (textureGone && now) useTextureWizard.setState({ made: { ...now, texture: undefined }, step: FACES })
  }, [textureGone])
  // picking ended in the 3D view (Done, Esc): the CAD body goes out of sight again
  useEffect(() => {
    const before = picked.current
    picked.current = picking
    const now = fresh()
    if (before && !picking && now?.texture && pickNodes(useStore.getState(), now).includes(before)) settle()
  }, [picking])

  if (!open) return null
  const reached = made?.texture ? MORE : FACES
  return (
    <div className="tool-panel wizard">
      <div className="slice-row">
        <span className="slice-title"><b>Texture</b>{made?.texture && <span className="muted"> · {made.texture}</span>}</span>
        {made && <button className="small-btn" title="Start another texture (the nodes made so far stay in the graph)"
          onClick={() => { settle(); useTextureWizard.setState({ made: null, step: FACES, source: made.cad, base: null }) }}>New</button>}
        <button className="small-btn" onClick={closeTextureWizard} title="Close (the nodes stay in the graph; reopen to go on)">×</button>
      </div>
      <div className="wiz-steps">
        {STEP_LABELS.map((label, k) => (
          <button key={label} className={`wiz-step ${k === step ? 'on' : ''} ${k < step ? 'done' : ''}`}
            disabled={!made || k > reached || k === step} onClick={() => go(k)}>
            <span className="wiz-num">{k < step ? '✓' : k + 1}</span>{label}
          </button>
        ))}
      </div>
      {editing ? <div className="muted">Close the open group first.</div>
        : !made ? <PartStep />
        : step === FACES || !made.texture ? <FacesStep made={made} />
        : step === TEXTURE ? <TextureStep made={made} />
        : step === MASKS ? <MasksStep made={made} />
        : <MoreStep made={made} />}
    </div>
  )
}
