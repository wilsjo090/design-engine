import { useEffect, useMemo, useRef } from 'react'
import { create } from 'zustand'
import { useStore } from '../store'
import type { EngineNode, GraphEdit } from '../types'
import { type Source, meshAndExport, modifyEdit, roundNice, slotAnchor, upstream, withSource } from '../viewActions'
import { closeOtherWizards, registerWizard } from '../wizards'
import { PickFaces, cadPort, nodeOf, specOf, useSources, valueOf } from './AnalysisWizard'
import { useBrowser } from './FileBrowser'
import {
  AxisButtons, Setting, defaultSource, extentOf, findBase, linkInto, meshVolume, nearestCad, slotFor, typeOf,
} from './wizardParts'

/** The steps, in order. */
const PART = 0, LATTICE = 1, SOLID = 2, GRADING = 3, CHECK = 4
const STEP_LABELS = ['Part', 'Lattice', 'Solid', 'Grading', 'Check']

/** The lattices to choose from: a node type and its pattern. */
interface LatticeKind { label: string; type: 'lattice.tpms' | 'lattice.struts' | 'lattice.voronoi'; pattern?: string; hint: string }
const LATTICES: LatticeKind[] = [
  { label: 'Gyroid', type: 'lattice.tpms', pattern: 'gyroid',
    hint: 'Smooth twisting walls without sharp nodes: about as strong in every direction, and easy to print.' },
  { label: 'Schwarz P', type: 'lattice.tpms', pattern: 'schwarz_p', hint: 'Walls round tubes running along X, Y and Z.' },
  { label: 'Diamond', type: 'lattice.tpms', pattern: 'diamond', hint: 'Stiff four-way walls.' },
  { label: 'BCC struts', type: 'lattice.struts', pattern: 'bcc', hint: 'Struts along the cells\' diagonals.' },
  { label: 'Octet', type: 'lattice.struts', pattern: 'octet', hint: 'A very stiff strut lattice.' },
  { label: 'Voronoi foam', type: 'lattice.voronoi', hint: 'An organic foam of struts round random cells.' },
]
/** The solid share a new lattice starts with (Units: density) and the density 1 = solid. */
const SHARE = 0.3
/** g/cm³, for the mass estimate. */
const PLA_DENSITY = 1.24

/** What the wizard made: Fill Body with Lattice (`fill`) put into the shown body `view`, the CAD output
 *  `cad`.`port` its face masks (and a Voronoi foam's box) read, the part's volume before (mm³, from its
 *  view; null when it was not shown) and which lattice each lattice node it made is (`kinds`). The
 *  lattice, the grading feeding its Thickness and the open faces feeding Skin are read from the graph. */
interface Made {
  cad: string
  port: string
  view: string
  fill: string
  solid: number | null
  kinds: Record<string, string>
}

interface LatticeWizardState {
  open: boolean
  step: number
  /** the CAD node of the part, before the lattice is made */
  source: string | null
  /** the shown body the lattice goes into (null: found from the part) */
  base: string | null
  made: Made | null
}

/** The lattice wizard: the part, the lattice (seen in a section, with what it weighs), solid regions and
 *  open faces, grading, then the print check and export; it builds Fill Body with Lattice and a TPMS,
 *  Strut or Voronoi Lattice in density units, Max / Remap chains into the lattice's Thickness and a
 *  Multiply of a Face Mask into the Fill's Skin. */
export const useLatticeWizard = create<LatticeWizardState>(() => ({ open: false, step: PART, source: null, base: null, made: null }))

// another recipe: its nodes are not the ones the wizard made (the panel stays open)
useStore.subscribe((s, prev) => {
  if (s.loadCount !== prev.loadCount && !s.editing && !prev.editing) {
    useLatticeWizard.setState({ step: PART, source: null, base: null, made: null })
  }
})

type State = ReturnType<typeof useStore.getState>
const apply = (edit: GraphEdit, label: string, text: string | ((ids: Record<string, string>) => string)) =>
  useStore.getState().applyEdit(edit, label, { text })
const fresh = () => useLatticeWizard.getState().made
const num = (v: unknown, fallback: number) => (typeof v === 'number' ? v : fallback)

// --- reading the graph ----------------------------------------------------------------------------
function guessKind(node: EngineNode | undefined) {
  const t = node?.data.typeId
  const pattern = node?.data.inputs.pattern ?? (t === 'lattice.tpms' ? 'gyroid' : t === 'lattice.struts' ? 'bcc' : undefined)
  return LATTICES.find((k) => k.type === t && (k.pattern === undefined || k.pattern === pattern))
}

/** The lattice node in the Fill, and which lattice it is. */
function latticeOf(st: State, made: Made) {
  const id = linkInto(st, made.fill, 'lattice')?.source
  if (!id) return null
  return { id, kind: LATTICES.find((k) => k.label === made.kinds[id]) ?? guessKind(nodeOf(st, id)) }
}

/** One change to the lattice's density: a Max (`max`) of what comes before (its A) and a Remap (`remap`,
 *  0..1 to 0..its To High) of a 0..1 `factor`: a Face Mask (solid near faces, To High ≥ 1; denser near faces),
 *  a Smooth Step of a Coordinate (`coord`: along an axis) or of a Static Analysis' safety factor. */
interface Mod { max: string; remap: string; factor: string; kind: 'solid' | 'near' | 'axis' | 'stress' | 'other'; coord?: string }

/** The changes feeding the lattice's Thickness, the first (holding the base share on its A) first. */
function modsOf(st: State, lattice: string): Mod[] {
  const out: Mod[] = []
  let e = linkInto(st, lattice, 'thickness')
  while (e && typeOf(st, e.source) === 'math.max' && !out.some((m) => m.max === e!.source)) {
    const max = e.source
    const remap = linkInto(st, max, 'b')?.source ?? ''
    const factor = (typeOf(st, remap) === 'math.remap' ? linkInto(st, remap, 'x')?.source : undefined) ?? ''
    const t = typeOf(st, factor)
    const x = t === 'math.smoothstep' ? linkInto(st, factor, 'x')?.source ?? '' : ''
    const kind: Mod['kind'] = t === 'field.face_mask' ? (num(valueOf(st, remap, 'to_high'), 0) >= 1 ? 'solid' : 'near')
      : typeOf(st, x) === 'field.coordinate' ? 'axis' : typeOf(st, x) === 'sim.static' ? 'stress' : 'other'
    out.unshift({ max, remap, factor, kind, coord: kind === 'axis' ? x : undefined })
    e = linkInto(st, max, 'a')
  }
  return out
}

/** Where the base solid share is typed: the first Max's A, else the lattice's Thickness. */
const baseInput = (lattice: string, mods: Mod[]) => (mods.length ? { node: mods[0].max, port: 'a' } : { node: lattice, port: 'thickness' })

/** Open faces: the Fill's Skin times a Face Mask that is 0 on them (`mask`, in Multiply `mul`, whose B is
 *  the skin's thickness). */
function openOf(st: State, made: Made) {
  const mul = linkInto(st, made.fill, 'skin')?.source
  if (!mul || typeOf(st, mul) !== 'math.multiply') return null
  const mask = linkInto(st, mul, 'a')?.source
  return mask && typeOf(st, mask) === 'field.face_mask' ? { mul, mask } : null
}

/** The view the filled part is seen in: the Fill's own, or the mesh made from it. */
function shownResult(st: State, made: Made) {
  if (st.visible.includes(made.fill)) return made.fill
  return made.view !== made.cad && st.visible.includes(made.view) && st.views[made.view]?.kind !== 'cad' ? made.view : null
}

/** The nodes whose faces the wizard picks: the solid / denser near faces masks and the open faces' mask. */
function pickNodes(st: State, made: Made) {
  const lattice = latticeOf(st, made)
  const masks = lattice ? modsOf(st, lattice.id).filter((m) => m.kind === 'solid' || m.kind === 'near').map((m) => m.factor) : []
  const open = openOf(st, made)
  return open ? [...masks, open.mask] : masks
}

/** The 3D view's pick banner gives way to the panel while it picks faces. */
export function useLatticeWizardPicking() {
  const made = useLatticeWizard((w) => (w.open ? w.made : null))
  return useStore((s) => !!made && !!s.picking && pickNodes(s, made).includes(s.picking.node))
}

/** Static Analysis nodes of this part (its supports and loads give the stress to grade by). */
const analysesOf = (st: State, made: Made) =>
  st.nodes.filter((n) => n.data.typeId === 'sim.static' && (upstream(st, n.id).has(made.cad))).map((n) => n.id)

// --- edits ------------------------------------------------------------------------------------------
/** After picking: stop picking the wizard's faces, and put the CAD body out of sight again while the
 *  filled part is shown (picking shows it). */
function settle() {
  const made = fresh()
  const st = useStore.getState()
  if (!made) return
  if (st.picking && pickNodes(st, made).includes(st.picking.node)) st.setPicking(null)
  const shown = shownResult(st, made)
  if (shown && shown !== made.cad && useStore.getState().visible.includes(made.cad)) useStore.getState().toggleVisible(made.cad)
}

function go(step: number) {
  useLatticeWizard.setState({ step })
  settle()
}

/** The new lattice node for `kind` (key `lattice`): cell, units and (typed) thickness carried over. */
function latticeNode(kind: LatticeKind, inputs: Record<string, unknown>, near: string) {
  return { key: 'lattice', type: kind.type, near, inputs: { ...(kind.pattern ? { pattern: kind.pattern } : {}), ...inputs } }
}

/** Fills the part with a gyroid at 30% solid and a 1 mm skin, into the shown body `view` (a CAD view:
 *  through a new Implicit Body), shown cut open by the section plane. */
function createFill(src: Source, view: string): Made | null {
  const st = useStore.getState()
  const slot = slotFor(st, src, view)
  if (typeof slot === 'string') { st.showToast(slot); return null }
  // the part's volume before, and a cell about a sixth of its smallest size
  const shown = st.views[view] ?? st.views[src.node]
  const solid = shown ? meshVolume(shown) : null
  const size = shown ? Math.min(...[0, 1, 2].map((k) => shown.bounds[1][k] - shown.bounds[0][k])) : 30
  const cell = Math.min(20, Math.max(2, roundNice(size / 6)))
  const kind = LATTICES[0]
  const chain: GraphEdit = {
    add: [latticeNode(kind, { cell, units: 'density', thickness: SHARE }, slotAnchor(slot)),
      { key: 'fill', type: 'lattice.fill', inputs: { skin: 1 }, near: 'lattice' }],
    links: [{ from: 'lattice', port: 'out', to: 'fill', input: 'lattice' }],
    select: ['fill'],
  }
  const ids = apply(withSource(src, modifyEdit(slot, chain, { key: 'fill', input: 'body' }, { key: 'fill', port: 'out' })),
    'Lattice wizard', (ids) => `Lattice wizard: ${ids.lattice} (gyroid, ${SHARE * 100}% solid) in ${ids.fill}, 1 mm skin` +
      (slot.consumers.length ? ` → ${slot.consumers.map((c) => c.node).join(', ')}` : ''))
  if (!ids) return null
  st.requestViewTool({ kind: 'section' })
  return { cad: ids.pick ?? src.node, port: src.pick ? 'body' : src.port, view, fill: ids.fill, solid, kinds: { [ids.lattice]: kind.label } }
}

/** Another lattice in the Fill: the lattice node replaced, its cell, units and solid share (or grading)
 *  kept. */
function switchLattice(kind: LatticeKind) {
  const made = fresh()
  const st = useStore.getState()
  const now = made && latticeOf(st, made)
  if (!made || !now || now.kind?.label === kind.label) return
  const old = nodeOf(st, now.id)!
  const graded = linkInto(st, now.id, 'thickness')
  const carried: Record<string, unknown> = { cell: valueOf(st, now.id, 'cell'), units: valueOf(st, now.id, 'units') }
  if (!graded && old.data.inputs.thickness !== undefined) carried.thickness = old.data.inputs.thickness
  const shared = st.edges.some((e) => e.source === now.id && !(e.target === made.fill && e.targetHandle === 'lattice'))
  const ids = apply({
    remove: shared ? [] : [now.id],
    add: [latticeNode(kind, carried, made.fill)],
    links: [{ from: 'lattice', port: 'out', to: made.fill, input: 'lattice' },
      ...(graded ? [{ from: graded.source, port: graded.sourceHandle!, to: 'lattice', input: 'thickness' }] : []),
      ...(kind.type === 'lattice.voronoi' ? [{ from: made.cad, port: made.port, to: 'lattice', input: 'domain' }] : [])],
    select: [made.fill],
  }, `Lattice ${kind.label}`, (ids) => `${made.fill}: ${kind.label.toLowerCase()} (${ids.lattice})`)
  if (ids) useLatticeWizard.setState({ made: { ...made, kinds: { ...made.kinds, [ids.lattice]: kind.label } } })
}

const MOD_LABELS: Record<Mod['kind'], string> = {
  solid: 'Solid near faces', near: 'Denser near faces', axis: 'Denser along an axis', stress: 'Denser where stressed', other: 'Grading',
}

/** One more change to the density, after those there are: solid or denser near faces (picked next), along
 *  Z (denser at the top), or where the Static Analysis `analysis` finds the safety factor low. */
function addMod(kind: Exclude<Mod['kind'], 'other'>, analysis?: string) {
  const made = fresh()
  const st = useStore.getState()
  const lattice = made && latticeOf(st, made)
  if (!made || !lattice) return
  const before = linkInto(st, lattice.id, 'thickness')
  const top = kind === 'solid' ? 2 : kind === 'stress' ? 0.7 : 0.6  // a Remap to 2: solid for the first half of the reach
  const edit: GraphEdit = { add: [], links: [], select: [kind === 'axis' || kind === 'stress' ? 'factor' : 'remap'] }
  let note = ''
  if (kind === 'solid' || kind === 'near') {
    const fade = kind === 'solid' ? 4 : 10
    edit.add!.push({ key: 'factor', type: 'field.face_mask', inputs: { faces: '', reach: 'volume', fade }, near: made.fill })
    edit.links!.push({ from: made.cad, port: made.port, to: 'factor', input: 'cad' })
    edit.pick = 'factor'
    note = ': click the faces'
  } else if (kind === 'axis') {
    const [lo, hi] = extentOf(st, [shownResult(st, made), made.view, made.cad], 'z')
    edit.add!.push({ key: 'coord', type: 'field.coordinate', inputs: { axis: 'z' }, near: made.fill },
      { key: 'factor', type: 'math.smoothstep', inputs: { edge0: lo, edge1: hi }, near: 'coord' })
    edit.links!.push({ from: 'coord', port: 'out', to: 'factor', input: 'x' })
    note = `: ${SHARE * 100}% at Z ${lo}, ${top * 100}% at ${hi}`
  } else {
    edit.add!.push({ key: 'factor', type: 'math.smoothstep', inputs: { edge0: 6, edge1: 2 }, near: made.fill })
    edit.links!.push({ from: analysis!, port: 'safety', to: 'factor', input: 'x' })
    note = `: densest where ${analysis}'s safety factor is under 2`
  }
  // every factor is 0..1 (not Auto: a mask with no faces picked yet is constant, which Auto refuses)
  edit.add!.push({ key: 'remap', type: 'math.remap', inputs: { from_low: 0, from_high: 1, to_low: 0, to_high: top }, near: 'factor' },
    { key: 'max', type: 'math.max', inputs: before ? {} : { a: num(valueOf(st, lattice.id, 'thickness'), SHARE) }, near: 'remap' })
  edit.links!.push({ from: 'factor', port: 'out', to: 'remap', input: 'x' }, { from: 'remap', port: 'out', to: 'max', input: 'b' },
    ...(before ? [{ from: before.source, port: before.sourceHandle!, to: 'max', input: 'a' }] : []),
    { from: 'max', port: 'out', to: lattice.id, input: 'thickness' })
  apply(edit, MOD_LABELS[kind], (ids) => `${MOD_LABELS[kind]}: ${ids.factor} → ${ids.remap} → ${ids.max} → ${lattice.id}'s thickness${note}`)
}

/** Takes a change out: the next one takes what came before it (the first's base share moves on). */
function removeMod(m: Mod) {
  const made = fresh()
  const st = useStore.getState()
  const lattice = made && latticeOf(st, made)
  if (!lattice) return
  const mods = modsOf(st, lattice.id)
  const k = mods.findIndex((x) => x.max === m.max)
  const rest = mods.filter((x) => x.max !== m.max)
  const base = num(valueOf(st, mods[0].max, 'a'), SHARE)
  const edit: GraphEdit = { remove: [m.max, m.remap, m.factor, ...(m.coord ? [m.coord] : [])], links: [], inputs: [] }
  if (!rest.length) edit.inputs!.push({ node: lattice.id, port: 'thickness', value: base })
  else {
    if (k === 0) edit.inputs!.push({ node: rest[0].max, port: 'a', value: base })
    else if (k < mods.length - 1) edit.links!.push({ from: mods[k - 1].max, port: 'out', to: mods[k + 1].max, input: 'a' })
    else edit.links!.push({ from: mods[k - 1].max, port: 'out', to: lattice.id, input: 'thickness' })
  }
  apply(edit, `Remove ${MOD_LABELS[m.kind].toLowerCase()}`, `Removed ${m.factor}, ${m.remap} and ${m.max}`)
}

/** The fade along another axis, over the whole part (flipped stays flipped). */
function setAxis(m: Mod, axis: string) {
  const made = fresh()
  const st = useStore.getState()
  if (!made || !m.coord) return
  const flipped = num(valueOf(st, m.factor, 'edge0'), 0) > num(valueOf(st, m.factor, 'edge1'), 1)
  const [lo, hi] = extentOf(st, [shownResult(st, made), made.view, made.cad], axis)
  apply({ inputs: [{ node: m.coord, port: 'axis', value: axis },
    { node: m.factor, port: 'edge0', value: flipped ? hi : lo }, { node: m.factor, port: 'edge1', value: flipped ? lo : hi }] },
  `Grade along ${axis.toUpperCase()}`, `${m.factor} grades along ${axis.toUpperCase()}`)
}

function flip(m: Mod) {
  const st = useStore.getState()
  const e0 = valueOf(st, m.factor, 'edge0'), e1 = valueOf(st, m.factor, 'edge1')
  if (typeof e0 !== 'number' || typeof e1 !== 'number') return
  apply({ inputs: [{ node: m.factor, port: 'edge0', value: e1 }, { node: m.factor, port: 'edge1', value: e0 }] },
    'Flip grading', `${m.factor}: denser at the other end now`)
}

/** Faces with no skin (the lattice shows, powder or resin can get out): the Skin times a Face Mask that is
 *  0 on them, picked next. */
function addOpen() {
  const made = fresh()
  const st = useStore.getState()
  if (!made || openOf(st, made)) return
  const skin = num(valueOf(st, made.fill, 'skin'), 1)
  apply({
    add: [{ key: 'mask', type: 'field.face_mask', inputs: { faces: '', apply: 'all other faces', fade: 0.5 }, near: made.fill },
      { key: 'mul', type: 'math.multiply', inputs: { b: skin }, near: 'mask' }],
    links: [{ from: made.cad, port: made.port, to: 'mask', input: 'cad' }, { from: 'mask', port: 'out', to: 'mul', input: 'a' },
      { from: 'mul', port: 'out', to: made.fill, input: 'skin' }],
    select: ['mask'],
    pick: 'mask',
  }, 'Open faces', (ids) => `${ids.mask} × skin → ${made.fill}: click the faces to leave open`)
}

function removeOpen() {
  const made = fresh()
  const st = useStore.getState()
  const open = made && openOf(st, made)
  if (!made || !open) return
  apply({ remove: [open.mask, open.mul], inputs: [{ node: made.fill, port: 'skin', value: num(valueOf(st, open.mul, 'b'), 1) }] },
    'Close all faces', `Removed ${open.mask} and ${open.mul}: the skin goes all round again`)
}

/** Closes the panel (the nodes stay; reopening goes on where it was). */
export function closeLatticeWizard() {
  if (!useLatticeWizard.getState().open) return
  settle()
  useLatticeWizard.setState({ open: false })
}
registerWizard('lattice', closeLatticeWizard)

/** Opens the wizard; with `start` (a body's menu) on that body: with its CAD body known the lattice goes
 *  in at once, else the part is chosen first. */
export function openLatticeWizard(start?: { cad: Source | null; view: string }) {
  closeOtherWizards('lattice')
  useLatticeWizard.setState({ open: true })
  const st = useStore.getState()
  if (!start || st.editing) return
  settle()
  if (!start.cad) {
    useLatticeWizard.setState({ made: null, step: PART, source: nearestCad(st, start.view)[0] ?? null, base: start.view })
    return
  }
  const made = createFill(start.cad, start.view)
  if (made) useLatticeWizard.setState({ made, step: LATTICE, source: null, base: null })
}

// --- the panel --------------------------------------------------------------------------------------
/** What the filled part weighs, from its view: its volume, its share of the solid part, in PLA. */
function Amount({ made }: { made: Made }) {
  const shown = useStore((s) => shownResult(s, made))
  const view = useStore((s) => (shown ? s.views[shown] : undefined))
  const running = useStore((s) => !!s.run?.running)
  const volume = useMemo(() => (view ? meshVolume(view) : null), [view])
  if (!volume) return <div className="muted small">{running ? 'Computing the filled part…' : 'Show the filled part to see what it weighs.'}</div>
  const cm3 = volume / 1000
  const share = made.solid ? Math.round((100 * volume) / made.solid) : null
  return (
    <div className="lat-amount" title={`Measured on the ${view!.level ?? ''} view: it gets exact as the preview refines`}>
      <b>{cm3.toFixed(cm3 < 10 ? 1 : 0)} cm³</b>{share !== null && <> · <b>{share}%</b> of the solid part</>}
      {' '}· {(cm3 * PLA_DENSITY).toFixed(0)} g in PLA
      {view!.level && view!.level !== 'final' && <span className="muted"> ({view!.level})</span>}
    </div>
  )
}

function PartStep() {
  const { source, base } = useLatticeWizard()
  const sources = useSources()
  const fallback = useStore((s) => defaultSource(s, sources))
  const chosen = source && sources.includes(source) ? source : fallback
  const title = useStore((s) => (chosen ? specOf(s, chosen)?.title : ''))
  const into = useStore((s) => (chosen ? base ?? findBase(s, chosen) : null))

  const openStep = async () => {
    const path = await useBrowser.getState().open({ mode: 'open', exts: ['.step', '.stp'], title: 'Open the part (STEP)', initial: '' })
    if (!path) return
    const ids = useStore.getState().applyEdit({ add: [{ key: 'new', type: 'input.import_step', inputs: { path } }], show: ['new'] },
      'Import STEP', { text: (ids) => `Added ${ids.new} reading ${path}` })
    if (ids) useLatticeWizard.setState({ source: ids.new, base: null })
  }
  const start = () => {
    if (!chosen) return
    const st = useStore.getState()
    const port = cadPort(st, chosen)
    if (!port) return
    const made = createFill({ node: chosen, port, key: '' }, into ?? chosen)
    if (made) useLatticeWizard.setState({ made, step: LATTICE, base: null })
  }

  if (!sources.length) {
    return (
      <>
        <div>Open the <b>part</b> to fill with a lattice.</div>
        <div className="wiz-buttons"><button className="primary" onClick={openStep}>Open STEP file…</button></div>
      </>
    )
  }
  return (
    <>
      <div>Which part to fill with a <b>lattice</b>: lighter, and it saves material and print time.</div>
      <label className="wiz-row">
        <span>Part</span>
        <select value={chosen ?? ''} onChange={(e) => useLatticeWizard.setState({ source: e.target.value, base: null })}>
          {sources.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>
      <div className="muted small">{title}{into ? ` · goes into ${into}, as it is shown` : ''}</div>
      <div className="wiz-buttons">
        <button className="small-btn" onClick={openStep}>Other STEP file…</button>
        <button className="primary" disabled={!chosen} onClick={start}>Fill with lattice ›</button>
      </div>
    </>
  )
}

function LatticeStep({ made }: { made: Made }) {
  // (read from the graph as it is: re-rendered when it changes)
  useStore((s) => s.nodes)
  useStore((s) => s.edges)
  const st = useStore.getState()
  const lattice = latticeOf(st, made)
  const mods = lattice ? modsOf(st, lattice.id) : []
  const open = openOf(st, made)
  const base = lattice && baseInput(lattice.id, mods)
  return (
    <>
      <div>Choose the lattice. The view is cut open so you see it inside the part (drag the orange ball to move the cut).</div>
      <div className="tex-kinds">
        {LATTICES.map((k) => (
          <button key={k.label} className={`small-btn ${lattice?.kind?.label === k.label ? 'on' : ''}`} title={k.hint}
            onClick={() => switchLattice(k)}>{k.label}</button>
        ))}
      </div>
      {lattice?.kind && <div className="muted small">{lattice.kind.hint}</div>}
      {lattice && (
        <>
          <Setting node={lattice.id} port="cell" label="Cell" />
          {base && <Setting node={base.node} port={base.port} label={mods.length ? 'Solid share (base)' : 'Solid share'} />}
          <Setting node={lattice.id} port="blend" label="Rounded nodes" />
          <Setting node={lattice.id} port="randomness" label="Randomness" />
        </>
      )}
      {open ? <Setting node={open.mul} port="b" label="Skin" /> : <Setting node={made.fill} port="skin" label="Skin" />}
      <div className="muted small">Solid share: 0.3 = 30% of the lattice is material. Skin: a closed wall round the part (0: the
        lattice shows at the surface).</div>
      <Amount made={made} />
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => useStore.getState().focusNode(made.fill)}>Show node</button>
        <button className="primary" onClick={() => go(SOLID)}>Next ›</button>
      </div>
    </>
  )
}

/** A change to the density near picked faces: the faces, how far it reaches, and (denser) the share there. */
function NearCard({ made, m }: { made: Made; m: Mod }) {
  return (
    <div className="wiz-load">
      <div className="wiz-row">
        <b>{MOD_LABELS[m.kind]}</b> <span className="muted">{m.factor}</span>
        <button className="small-btn" title={`Remove ${m.factor}, ${m.remap} and ${m.max}`} onClick={() => removeMod(m)}>×</button>
      </div>
      <PickFaces node={m.factor} space={made.cad} what={m.kind === 'solid' ? 'made solid' : 'denser'} />
      <Setting node={m.factor} port="fade" label="Reach" />
      {m.kind === 'near' && <Setting node={m.remap} port="to_high" label="Share there" />}
      <div className="muted small">{m.kind === 'solid' ? 'Solid for the first half of the reach (bolt holes, mounting faces), then opening into the lattice.'
        : 'Densest at the faces, back to the base share at the reach.'}</div>
    </div>
  )
}

function AxisCard({ m }: { m: Mod }) {
  const axis = useStore((s) => String(valueOf(s, m.coord!, 'axis') ?? 'z'))
  return (
    <div className="wiz-load">
      <div className="wiz-row">
        <b>Denser along</b>
        <AxisButtons axis={axis} onPick={(a) => setAxis(m, a)} />
        <button className="small-btn" title="Denser at the other end" onClick={() => flip(m)}>Flip</button>
        <button className="small-btn" title={`Remove ${m.coord}, ${m.factor}, ${m.remap} and ${m.max}`} onClick={() => removeMod(m)}>×</button>
      </div>
      <Setting node={m.factor} port="edge0" label="Base at (mm)" />
      <Setting node={m.factor} port="edge1" label="Densest at (mm)" />
      <Setting node={m.remap} port="to_high" label="Share there" />
    </div>
  )
}

function StressCard({ m }: { m: Mod }) {
  const analysis = useStore((s) => linkInto(s, m.factor, 'x')?.source)
  return (
    <div className="wiz-load">
      <div className="wiz-row">
        <b>Denser where stressed</b> <span className="muted">{analysis}</span>
        <button className="small-btn" title={`Remove ${m.factor}, ${m.remap} and ${m.max}`} onClick={() => removeMod(m)}>×</button>
      </div>
      <Setting node={m.factor} port="edge1" label="Densest below safety" />
      <Setting node={m.factor} port="edge0" label="Base above safety" />
      <Setting node={m.remap} port="to_high" label="Share there" />
      <div className="muted small">From the safety factor of {analysis} (the solid part under its loads).</div>
    </div>
  )
}

function ModCard({ made, m }: { made: Made; m: Mod }) {
  if (m.kind === 'solid' || m.kind === 'near') return <NearCard made={made} m={m} />
  if (m.kind === 'axis') return <AxisCard m={m} />
  if (m.kind === 'stress') return <StressCard m={m} />
  return (
    <div className="wiz-load">
      <div className="wiz-row">
        <span>Max with <b>{m.remap || '(nothing)'}</b> <span className="muted">in {m.max}</span></span>
        <button className="small-btn" title={`Remove ${m.max}`} onClick={() => removeMod(m)}>×</button>
      </div>
    </div>
  )
}

function SolidStep({ made }: { made: Made }) {
  useStore((s) => s.nodes)
  useStore((s) => s.edges)
  const st = useStore.getState()
  const lattice = latticeOf(st, made)
  const mods = (lattice ? modsOf(st, lattice.id) : []).filter((m) => m.kind === 'solid')
  const open = openOf(st, made)
  return (
    <>
      <div>Where the part stays <b>solid</b>, and which faces stay <b>open</b>.</div>
      {mods.map((m) => <ModCard key={m.max} made={made} m={m} />)}
      {open && (
        <div className="wiz-load">
          <div className="wiz-row">
            <b>Open faces</b> <span className="muted">{open.mask}</span>
            <button className="small-btn" title={`Remove ${open.mask} and ${open.mul}`} onClick={removeOpen}>×</button>
          </div>
          <PickFaces node={open.mask} space={made.cad} what="open" />
          <div className="muted small">No skin there: the lattice shows, and powder or resin can get out.</div>
        </div>
      )}
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => addMod('solid')} title="Solid round bolt holes and mounting faces, picked next">+ Solid near faces</button>
        {!open && <button className="small-btn" onClick={addOpen} title="Faces without skin, picked next">+ Open faces</button>}
      </div>
      <Amount made={made} />
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(LATTICE)}>‹ Lattice</button>
        <button className="primary" onClick={() => go(GRADING)}>Next ›</button>
      </div>
    </>
  )
}

function GradingStep({ made }: { made: Made }) {
  useStore((s) => s.nodes)
  useStore((s) => s.edges)
  const st = useStore.getState()
  const lattice = latticeOf(st, made)
  const mods = (lattice ? modsOf(st, lattice.id) : []).filter((m) => m.kind !== 'solid')
  const analyses = analysesOf(st, made)
  return (
    <>
      <div>Make the lattice <b>denser</b> where the part needs it, lighter elsewhere.</div>
      {mods.map((m) => <ModCard key={m.max} made={made} m={m} />)}
      <div className="wiz-buttons wrap">
        <button className="small-btn" onClick={() => addMod('near')} title="Denser toward faces you pick, fading back over Reach">+ Near faces</button>
        <button className="small-btn" onClick={() => addMod('axis')} title="Denser toward one end of the part">+ Along an axis</button>
        <button className="small-btn" disabled={!analyses.length} onClick={() => addMod('stress', analyses[0])}
          title={analyses.length ? `Denser where ${analyses[0]} finds the part working hard`
            : 'Needs a Static Analysis of this part: right-click the part ▸ Analyse ▸ Wizard ▸ Static analysis'}>+ Where stressed</button>
      </div>
      <Amount made={made} />
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(SOLID)}>‹ Solid</button>
        <button className="primary" onClick={() => go(CHECK)}>Next ›</button>
      </div>
    </>
  )
}

function CheckStep({ made }: { made: Made }) {
  const shown = useStore((s) => shownResult(s, made))
  const closed = useStore((s) => !openOf(s, made) && num(valueOf(s, made.fill, 'skin'), 1) !== 0)
  return (
    <>
      <Amount made={made} />
      <div className="muted small">Print check: the thinnest walls and struts, loose bits, closed cavities.</div>
      {closed && <div className="wiz-warning small">With a closed skin the inside is a cavity: printed in powder or resin it stays
        full. Make some faces open (‹ Solid), or add drain holes (right-click a face ▸ Holes).</div>}
      <div className="wiz-buttons">
        <button className="small-btn" disabled={!shown} onClick={() => useStore.getState().requestViewTool({ kind: 'check', node: shown! })}>Print check</button>
        <button className="small-btn" onClick={() => { meshAndExport(made.fill) }} title="Mesh the filled part and save it (at Final quality)">Mesh and export…</button>
      </div>
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(GRADING)}>‹ Grading</button>
        <button className="primary" onClick={closeLatticeWizard}>Done</button>
      </div>
    </>
  )
}

/** The wizard's panel beside the 3D view. */
export function LatticeWizard() {
  const { open, step, made } = useLatticeWizard()
  const editing = useStore((s) => !!s.editing)
  const gone = useStore((s) => !!made && !nodeOf(s, made.fill))
  const picking = useStore((s) => s.picking?.node ?? null)
  const picked = useRef(picking)

  // the Fill was deleted or undone away: start over
  useEffect(() => { if (gone) useLatticeWizard.setState({ made: null, step: PART }) }, [gone])
  // picking ended in the 3D view (Done, Esc): the CAD body goes out of sight again
  useEffect(() => {
    const before = picked.current
    picked.current = picking
    const now = fresh()
    if (before && !picking && now && pickNodes(useStore.getState(), now).includes(before)) settle()
  }, [picking])

  if (!open) return null
  const reached = made ? CHECK : PART
  return (
    <div className="tool-panel wizard">
      <div className="slice-row">
        <span className="slice-title"><b>Lattice</b>{made && <span className="muted"> · {made.fill}</span>}</span>
        {made && <button className="small-btn" title="Fill another part (the nodes made so far stay in the graph)"
          onClick={() => { settle(); useLatticeWizard.setState({ made: null, step: PART, source: null, base: null }) }}>New</button>}
        <button className="small-btn" onClick={closeLatticeWizard} title="Close (the nodes stay in the graph; reopen to go on)">×</button>
      </div>
      <div className="wiz-steps">
        {STEP_LABELS.map((label, k) => (
          <button key={label} className={`wiz-step ${k === step ? 'on' : ''} ${k < step ? 'done' : ''}`}
            disabled={!made || k === PART || k > reached || k === step} onClick={() => go(k)}>
            <span className="wiz-num">{k < step ? '✓' : k + 1}</span>{label}
          </button>
        ))}
      </div>
      {editing ? <div className="muted">Close the open group first.</div>
        : !made || step === PART ? <PartStep />
        : step === LATTICE ? <LatticeStep made={made} />
        : step === SOLID ? <SolidStep made={made} />
        : step === GRADING ? <GradingStep made={made} />
        : <CheckStep made={made} />}
    </div>
  )
}
