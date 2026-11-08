import { useEffect } from 'react'
import { create } from 'zustand'
import { useShallow } from 'zustand/react/shallow'
import { loadFaces, startOptimize, stopRun } from '../api'
import { useStore } from '../store'
import type { GraphEdit, NodeSpec } from '../types'
import { isLink, parseFaceKeys, resolveFaces, unitOf } from '../types'
import { meshAndExport } from '../viewActions'
import { closeOtherWizards, registerWizard } from '../wizards'
import { useBrowser } from './FileBrowser'
import { ChoiceInput, NumberInput } from './PortInputs'

/** What the wizard sets up: the node it ends in, the input the part goes into, that node's own first
 *  settings, and what its last step does (Optimize: a queued job; Analyse: shown, so a preview runs it). */
export type Goal = 'topology' | 'frame' | 'static'
const GOALS: Record<Goal, { label: string; type: string; port: string; inputs: Record<string, unknown>; run: 'Optimize' | 'Analyse' }> = {
  topology: { label: 'Topology optimization', type: 'sim.topology', port: 'space', inputs: { keep: 30, detail: 'draft' }, run: 'Optimize' },
  frame: { label: 'Frame optimization', type: 'sim.frame', port: 'space', inputs: {}, run: 'Optimize' },
  static: { label: 'Static analysis', type: 'sim.static', port: 'part', inputs: { detail: 'medium', show: 'safety factor' }, run: 'Analyse' },
}
const GOAL_ORDER: Goal[] = ['topology', 'frame', 'static']
const GOAL_HINTS: Record<Goal, string> = {
  topology: 'the stiffest shape for a given mass, grown inside the design space',
  frame: 'the lightest frame of bars that carries the loads, inside the design space',
  static: 'how the part bends and how hard the material works: its safety factor',
}

/** The steps, in order (the last is named by the goal). */
const PART = 0, HELD = 1, LOADS = 2, SETTINGS = 3, RUN = 4
const stepLabels = (goal: Goal) => ['Part', 'Held', 'Loads', 'Settings', GOALS[goal].run]

/** The nodes the wizard made: the part's node (`space`: a Pick Body it added, or the chosen source),
 *  the supports, the loads (a chain after the supports, in order), the material and the goal's node. */
interface Made { goal: Goal; space: string; fixed: string; loads: string[]; material: string; result: string }

interface WizardState {
  open: boolean
  step: number
  goal: Goal
  /** the node whose CAD output is the part, and for a list of bodies which one (null: all of them,
   *  as one part) */
  source: string | null
  body: number | null
  made: Made | null
}

/** The analysis wizard: picking the part, the held and loaded faces, a few settings, then Optimize or
 *  Analyse; it builds the same nodes as the node library would (Fixed Faces, Load, Material, and
 *  Topology Optimization, Frame Optimization or Static Analysis). */
export const useWizard = create<WizardState>(() => ({ open: false, step: PART, goal: 'topology', source: null, body: 0, made: null }))

// another recipe: its nodes are not the ones the wizard made or chose (the panel stays open)
useStore.subscribe((s, prev) => {
  if (s.loadCount !== prev.loadCount && !s.editing && !prev.editing) {
    useWizard.setState({ step: PART, source: null, body: 0, made: null })
  }
})

/** Closes the panel, and stops picking faces for it (the nodes stay; reopening goes on). */
export function closeAnalysisWizard() {
  const { open, made } = useWizard.getState()
  if (!open) return
  const st = useStore.getState()
  if (made && st.picking && [made.fixed, ...made.loads].includes(st.picking.node)) st.setPicking(null)
  useWizard.setState({ open: false })
}
registerWizard('analysis', closeAnalysisWizard)

/** Opens the wizard; on a body (the body menu) a new setup for `goal` starts on it, unless one is
 *  under way. */
export function openWizard(source?: string, body = 0, goal?: Goal) {
  closeOtherWizards('analysis')
  const w = useWizard.getState()
  if (source && (!w.made || w.step === PART)) {
    useWizard.setState({ source, body, made: null, step: PART, ...(goal ? { goal } : {}) })
  } else if (goal && !w.made) useWizard.setState({ goal })
  useWizard.setState({ open: true })
}

const LOAD_HINTS: Record<string, string> = {
  force: 'a total force spread over the faces (100 N ≈ 10 kg)',
  pressure: 'pushing square to the faces everywhere',
  bearing: 'a pin or bolt pressing on one side of a hole',
  torque: 'twisting about the direction (right-hand rule)',
}
const DETAIL_HINTS: Record<Goal, Record<string, string>> = {
  topology: {
    draft: '~40,000 elements: about half a minute',
    medium: '~250,000 elements: a few minutes',
    fine: '~1 million elements: thinner members, longer (needs the GPU)',
  },
  frame: {},
  static: {
    draft: '~40,000 elements: about a second',
    medium: '~250,000 elements: seconds',
    fine: '~1 million elements: finest stresses (needs the GPU to be quick)',
  },
}

// (shared with the texture wizard)
type State = ReturnType<typeof useStore.getState>
export const nodeOf = (st: State, id: string) => st.nodes.find((n) => n.id === id)
export const specOf = (st: State, id: string): NodeSpec | undefined => st.specs[nodeOf(st, id)?.data.typeId ?? '']
export const cadPort = (st: State, id: string) => specOf(st, id)?.outputs.find((o) => o.type === 'cad')?.name
/** An input of a node: what is typed in, else the port's default. */
export const valueOf = (st: State, id: string, port: string) =>
  nodeOf(st, id)?.data.inputs[port] ?? specOf(st, id)?.inputs.find((p) => p.name === port)?.default
export const faceCount = (st: State, id: string) => {
  const v = nodeOf(st, id)?.data.inputs.faces
  return isLink(v) ? 0 : parseFaceKeys(resolveFaces(v, st.selections)).length
}
const set = (node: string, port: string, value: unknown) => useStore.getState().setInput(node, port, value)
export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

/** One input of a node as on its card: a number with its unit, what 0 means and its slider, or its
 *  choices (a switch, buttons, pictures or a list). Keys typed in it stay away from the 3D view. */
export function Field({ node, port }: { node: string; port: string }) {
  const spec = useStore((s) => specOf(s, node)?.inputs.find((p) => p.name === port))
  const value = useStore((s) => valueOf(s, node, port))
  const unit = useStore((s) => (spec ? unitOf(spec, (q) => valueOf(s, node, q)) : undefined))
  if (!spec) return null
  return (
    <span className="wiz-field" onKeyDown={(e) => e.stopPropagation()}>
      {spec.choices
        ? <ChoiceInput port={spec} value={value} onCommit={(v) => set(node, port, v)} />
        : <NumberInput value={value} integer={spec.type === 'int'} unit={unit} zero={spec.zero} slider={spec.slider}
            onCommit={(v) => set(node, port, v)} />}
    </span>
  )
}

/** A labelled input of a node (a div: in a label, a click on the text would press a choice's first button). */
export function Row({ node, port, label, title }: { node: string; port: string; label: string; title?: string }) {
  return (
    <div className="wiz-row" title={title}>
      <span className="wiz-label">{label}</span><Field node={node} port={port} />
    </div>
  )
}

/** Picking faces for `node` in the 3D view: a button that starts and ends it, and how many are chosen. */
export function PickFaces({ node, space, what }: { node: string; space: string; what: string }) {
  const picking = useStore((s) => s.picking?.node === node)
  const count = useStore((s) => faceCount(s, node))
  return (
    <div className="wiz-row">
      <span className={count ? '' : 'muted'}>{count ? plural(count, 'face') : 'no faces yet'} {what}</span>
      <button className={`small-btn ${picking ? 'on' : ''}`}
        onClick={() => useStore.getState().setPicking(picking ? null : { node, source: space })}>
        {picking ? 'Done picking' : count ? 'Change faces' : 'Pick faces'}
      </button>
    </div>
  )
}

/** The CAD bodies the part can come from: nodes with a CAD output (Import STEP, Pick Body...). */
export function useSources() {
  return useStore(useShallow((s) => s.nodes.filter((n) => n.type === 'engine' && !!cadPort(s, n.id)).map((n) => n.id)))
}

/** How many bodies the CAD output of `node` holds (from its view's faces; null until known). */
function useBodyCount(node: string | null) {
  const shown = useStore((s) => !!node && !!s.views[node])
  const data = useStore((s) => (node ? s.faceData[node] : undefined))
  useEffect(() => { if (node && shown && !data) loadFaces(node) }, [node, shown, data])
  if (!node || !data) return null
  let most = 0
  for (let i = 0; i < data.bodies.length; i++) most = Math.max(most, data.bodies[i])
  return data.bodies.length ? most + 1 : 0
}

function PartStep() {
  const { source, body, goal } = useWizard()
  const sources = useSources()
  const chosen = source && sources.includes(source) ? source : sources[0] ?? null
  const count = useBodyCount(chosen)
  const editing = useStore((s) => !!s.editing)
  const title = useStore((s) => (chosen ? specOf(s, chosen)?.title : ''))
  const shown = useStore((s) => !!chosen && s.visible.includes(chosen))
  const g = GOALS[goal]
  const space = goal !== 'static'

  // the part is shown, so its faces can be picked (and its bodies counted)
  useEffect(() => { if (chosen && !shown) useStore.getState().toggleVisible(chosen) }, [chosen, shown])

  const openStep = async () => {
    const path = await useBrowser.getState().open({ mode: 'open', exts: ['.step', '.stp'], title: 'Open the part (STEP)', initial: '' })
    if (!path) return
    const ids = useStore.getState().applyEdit({ add: [{ key: 'new', type: 'input.import_step', inputs: { path } }], show: ['new'] },
      'Import STEP', { text: (ids) => `Added ${ids.new} reading ${path}` })
    if (ids) useWizard.setState({ source: ids.new, body: 0 })
  }

  const next = () => {
    if (!chosen) return
    const st = useStore.getState()
    const port = cadPort(st, chosen)!
    const pick = count !== null && count > 1 && body !== null
    const part = pick ? 'pick' : chosen
    const partPort = pick ? 'body' : port
    const edit: GraphEdit = {
      add: [
        ...(pick ? [{ key: 'pick', type: 'input.pick_body', inputs: { index: body }, near: chosen }] : []),
        { key: 'fixed', type: 'sim.fixed', inputs: { faces: '' }, near: part },
        { key: 'material', type: 'sim.material', inputs: { preset: 'PLA' }, near: 'fixed' },
        { key: 'result', type: g.type, inputs: g.inputs, near: 'fixed', place: 'after' },
      ],
      links: [
        ...(pick ? [{ from: chosen, port, to: 'pick', input: 'bodies' }] : []),
        { from: part, port: partPort, to: 'fixed', input: 'cad' },
        { from: part, port: partPort, to: 'result', input: g.port },
        { from: 'material', port: 'material', to: 'result', input: 'material' },
        { from: 'fixed', port: 'conditions', to: 'result', input: 'conditions' },
      ],
      show: pick ? ['pick'] : [],
      hide: pick ? [chosen] : [],
      select: ['fixed'],
      pick: 'fixed',
    }
    const ids = st.applyEdit(edit, `${g.label} wizard`, {
      text: (ids) => `${g.label} wizard: added ${[ids.pick, ids.fixed, ids.material, ids.result].filter(Boolean).join(', ')}`,
    })
    if (!ids) return
    useWizard.setState({ step: HELD, made: { goal, space: ids.pick ?? chosen, fixed: ids.fixed, loads: [], material: ids.material, result: ids.result } })
  }

  if (editing) return <div className="muted">Close the open group first.</div>
  const goalRow = (
    <label className="wiz-row">
      <span>Find</span>
      <select value={goal} onChange={(e) => useWizard.setState({ goal: e.target.value as Goal })}>
        {GOAL_ORDER.map((k) => <option key={k} value={k}>{GOALS[k].label}</option>)}
      </select>
    </label>
  )
  if (!sources.length) {
    return (
      <>
        {goalRow}
        <div className="muted small">{GOAL_HINTS[goal]}</div>
        <div>Open the {space ? <><b>design space</b>: the largest room the part may fill, modelled with its bolt holes and the
          faces it is mounted and loaded on.</> : <><b>part</b> to analyse.</>}</div>
        <div className="wiz-buttons"><button className="primary" onClick={openStep}>Open STEP file…</button></div>
      </>
    )
  }
  return (
    <>
      {goalRow}
      <div className="muted small">{GOAL_HINTS[goal]}</div>
      {space && <div className="muted">The <b>design space</b>: the largest room the part may fill, with its bolt holes and the
        faces it is held and loaded on.</div>}
      <label className="wiz-row">
        <span>Part</span>
        <select value={chosen ?? ''} onChange={(e) => useWizard.setState({ source: e.target.value, body: 0 })}>
          {sources.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>
      {count !== null && count > 1 && (
        <label className="wiz-row">
          <span>Body</span>
          <select value={body === null ? 'all' : String(body)}
            onChange={(e) => useWizard.setState({ body: e.target.value === 'all' ? null : Number(e.target.value) })}>
            {Array.from({ length: count }, (_, k) => <option key={k} value={k}>body {k}</option>)}
            <option value="all">all {count} bodies as one part</option>
          </select>
        </label>
      )}
      <div className="muted small">{title}{count === null ? ' · loading its faces…' : count > 1 ? ` · ${count} bodies` : ''}</div>
      <div className="wiz-buttons">
        <button className="small-btn" onClick={openStep}>Other STEP file…</button>
        <button className="primary" disabled={!chosen || count === null} onClick={next}>Next ›</button>
      </div>
    </>
  )
}

function HeldStep({ made }: { made: Made }) {
  const count = useStore((s) => faceCount(s, made.fixed))
  return (
    <>
      <div>Click the faces that are <b>bolted, glued or clamped</b>: they are held still. The green cones show them.</div>
      <div className="muted small">Click a face again to drop it · Shift+click: every face joined smoothly to it ·
        Ctrl+drag: paint over faces</div>
      <PickFaces node={made.fixed} space={made.space} what="held" />
      <div className="wiz-buttons">
        <button className="primary" disabled={!count} onClick={() => go(LOADS)}>Next ›</button>
      </div>
    </>
  )
}

/** Adds a load after the last of the chain (the supports when there is none), picking its faces next. */
function addLoad(made: Made) {
  const st = useStore.getState()
  const tail = made.loads[made.loads.length - 1] ?? made.fixed
  const ids = st.applyEdit({
    add: [{ key: 'new', type: 'sim.load', inputs: { faces: '', kind: 'force', amount: 100 }, near: tail, place: 'after' }],
    links: [
      { from: made.space, port: cadPort(st, made.space) ?? 'body', to: 'new', input: 'cad' },
      { from: tail, port: 'conditions', to: 'new', input: 'add_to' },
      { from: 'new', port: 'conditions', to: made.result, input: 'conditions' },
    ],
    select: ['new'],
    pick: 'new',
  }, 'Add load', { text: (ids) => `Added ${ids.new}: click the faces it acts on` })
  if (ids) useWizard.setState({ made: { ...made, loads: [...made.loads, ids.new] } })
}

/** Takes a load out of the chain: what used it takes what came into it. */
function removeLoad(made: Made, id: string) {
  const st = useStore.getState()
  const k = made.loads.indexOf(id)
  const before = k > 0 ? made.loads[k - 1] : made.fixed
  const links = st.edges.filter((e) => e.source === id && e.sourceHandle === 'conditions' && e.targetHandle)
    .map((e) => ({ from: before, port: 'conditions', to: e.target, input: e.targetHandle! }))
  if (st.applyEdit({ remove: [id], links }, `Remove ${id}`, { text: `Removed ${id}` })) {
    useWizard.setState({ made: { ...made, loads: made.loads.filter((l) => l !== id) } })
  }
}

function LoadCard({ made, id, index }: { made: Made; id: string; index: number }) {
  const kind = useStore((s) => String(valueOf(s, id, 'kind') ?? 'force'))
  const selected = useStore((s) => !!nodeOf(s, id)?.selected)
  return (
    <div className={`wiz-load ${selected ? 'on' : ''}`}>
      <div className="wiz-row">
        <button className="case-row" title="Select it: drag the yellow arrow in the 3D view to aim it"
          onClick={() => useStore.getState().applyEdit({ select: [id] }, `Select ${id}`, { text: `${id} selected: drag its yellow arrow to aim it` })}>
          <b>Load {index + 1}</b> <span className="muted">{id}</span>
        </button>
        {made.loads.length > 1 && <button className="small-btn" title={`Remove ${id}`} onClick={() => removeLoad(made, id)}>×</button>}
      </div>
      <PickFaces node={id} space={made.space} what="loaded" />
      <div className="wiz-row">
        <Field node={id} port="kind" />
        <Field node={id} port="amount" />
      </div>
      {kind !== 'pressure' && <Row node={id} port="direction" label={kind === 'torque' ? 'Axis' : 'Direction'} />}
      <div className="muted small">{LOAD_HINTS[kind]}</div>
    </div>
  )
}

function LoadsStep({ made }: { made: Made }) {
  // the first load comes by itself
  useEffect(() => { if (!made.loads.length) addLoad(made) }, [made])
  const ready = useStore((s) => made.loads.length > 0 && made.loads.every((l) => faceCount(s, l) > 0))
  return (
    <>
      <div>Click the faces the <b>load</b> acts on, then say how much. The red arrows show it; select a load to aim
        its yellow arrow by dragging.</div>
      {made.loads.map((id, k) => <LoadCard key={id} made={made} id={id} index={k} />)}
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(HELD)}>‹ Back</button>
        <button className="small-btn" onClick={() => addLoad(made)} title="Another load acting at the same time">+ Load</button>
        <button className="primary" disabled={!ready} onClick={() => go(SETTINGS)}>Next ›</button>
      </div>
    </>
  )
}

/** The goal's few settings; the rest are on its node ("more settings"). */
function SettingsStep({ made }: { made: Made }) {
  const r = made.result
  const detail = useStore((s) => String(valueOf(s, r, 'detail') ?? ''))
  const run = GOALS[made.goal].run
  return (
    <>
      <Row node={made.material} port="preset" label="Material" />
      {made.goal === 'topology' && (
        <>
          <Row node={r} port="keep" label="Keep" title="How much of the design space's mass stays: less is lighter but bendier" />
          <Row node={r} port="min_thickness" label="Min thickness" title="The thinnest member, for printing: empty = automatic (3 elements)" />
        </>
      )}
      {made.goal === 'frame' && (
        <>
          <Row node={r} port="profile" label="Profile" />
          <Row node={r} port="min_size" label="Min size" title="The thinnest bar (its width; round bars: the diameter)" />
          <Row node={r} port="max_size" label="Max size" title="The thickest bar: empty = no limit; where one bar would need more, the load is shared by more bars" />
          <Row node={r} port="safety" label="Safety factor" title="On the material's strength, for stress and buckling; the body is checked too" />
          <Row node={r} port="joints" label="Joints" />
          <Row node={r} port="curving" label="Curving" title="Bars that run on nearly straight through a joint become one smooth curve: 0 = straight, 1 = arcs" />
        </>
      )}
      {made.goal !== 'frame' && <Row node={r} port="detail" label="Detail" />}
      <div className="muted small">
        {made.goal === 'frame' ? 'Takes seconds to a minute; the bars grow in the 3D view. '
          : `${DETAIL_HINTS[made.goal][detail] ?? ''}. ${made.goal === 'topology' ? 'Start with draft. ' : ''}`}
        More settings are on the node.
      </div>
      <div className="wiz-buttons">
        <button className="small-btn" onClick={() => go(LOADS)}>‹ Back</button>
        <button className="small-btn" onClick={() => useStore.getState().focusNode(r)}>Show node</button>
        <button className="primary" onClick={() => { go(RUN); start(made) }}>{run} ›</button>
      </div>
    </>
  )
}

/** Hides the part and runs the goal: an optimization (its result shown while it grows), or the
 *  analysis (shown, so the preview computes it). */
function start(made: Made) {
  const st = useStore.getState()
  st.setPicking(null)
  if (st.visible.includes(made.space)) st.toggleVisible(made.space)
  st.focusNode(made.result)   // (a selected load would show its aiming arrow over the result)
  if (GOALS[made.goal].run === 'Optimize') startOptimize(made.result)
  else if (!useStore.getState().visible.includes(made.result)) useStore.getState().toggleVisible(made.result)
}

function RunStep({ made }: { made: Made }) {
  const status = useStore((s) => s.status[made.result])
  const optimizing = useStore((s) => !!s.run?.running && s.run.mode === 'optimize' && s.status[made.result]?.state === 'running')
  const running = optimizing || status?.state === 'running'
  const stopping = useStore((s) => !!s.run?.stopping)
  const done = !!status?.note && !status.note.startsWith('not optimized')
  const progress = status?.progress
  const { run } = GOALS[made.goal]
  const verb = made.goal === 'static' ? 'Analysing' : 'Optimizing'
  return (
    <>
      {running ? (
        <>
          <div>{verb}{made.goal === 'topology' ? ': the shape grows in the 3D view.' : made.goal === 'frame' ? ': the bars appear in the 3D view.' : '…'}</div>
          {progress && (
            <>
              <div className="progress"><div style={{ width: `${(100 * progress[0]) / progress[1]}%` }} /></div>
              <div className="muted small">{status?.step ?? ''} {made.goal === 'static' ? `${progress[0]}%` : `${progress[0]} / ${progress[1]}`}</div>
            </>
          )}
          {optimizing && (
            <>
              <div className="wiz-buttons">
                <button className="danger" disabled={stopping} onClick={() => stopRun()}>{stopping ? 'Stopping…' : 'Stop'}</button>
              </div>
              {made.goal === 'topology' && <div className="muted small">Stop keeps the shape so far; Optimize goes on from it.</div>}
            </>
          )}
        </>
      ) : status?.state === 'error' ? (
        <>
          <div className="warn">{status.message}</div>
          <div className="wiz-buttons">
            <button className="small-btn" onClick={() => go(SETTINGS)}>‹ Settings</button>
            <button className="primary" onClick={() => start(made)}>{run} again</button>
          </div>
        </>
      ) : done ? (
        <>
          <div className="wiz-note">{status!.note}</div>
          {status?.finished && <div className="muted small">{status.finished}</div>}
          {(status?.warnings ?? []).map(([text, level], i) => <div className={`small ${level === 'error' ? 'warn' : 'wiz-warning'}`} key={i}>{text}</div>)}
          <Row node={made.result} port="show" label="Colour by" />
          <div className="wiz-buttons">
            <button className="small-btn" onClick={() => go(SETTINGS)}>‹ Settings</button>
            {run === 'Optimize' && <>
              <button className="small-btn" disabled={!!status?.finished} onClick={() => start(made)}
                title={status?.finished ?? 'Go on from the result (after raising Max iterations), or run changed settings'}>Optimize</button>
              <button className="primary" onClick={() => { meshAndExport(made.result) }}>Mesh and export…</button>
            </>}
          </div>
        </>
      ) : (
        <>
          <div className="muted">Starting…</div>
          <div className="wiz-buttons"><button className="primary" onClick={() => start(made)}>{run}</button></div>
        </>
      )}
    </>
  )
}

/** Goes to a step: picking follows it (the held faces, the newest load), and stops elsewhere. */
function go(step: number) {
  const { made } = useWizard.getState()
  const st = useStore.getState()
  useWizard.setState({ step })
  if (!made) return
  const target = step === HELD ? made.fixed : step === LOADS ? made.loads[made.loads.length - 1] : null
  if (target) st.setPicking({ node: target, source: made.space })
  else if (st.picking && [made.fixed, ...made.loads].includes(st.picking.node)) st.setPicking(null)
}

/** The wizard's panel beside the 3D view. */
export function AnalysisWizard() {
  const { open, step, made, goal } = useWizard()
  const exists = useStore((s) => !made || (!!nodeOf(s, made.result) && !!nodeOf(s, made.fixed)))
  const loadsGone = useStore((s) => !!made && made.loads.some((l) => !nodeOf(s, l)))

  // the nodes were deleted or undone away: start over; loads deleted in the graph leave the list
  useEffect(() => { if (!exists) useWizard.setState({ made: null, step: PART }) }, [exists])
  useEffect(() => {
    if (!made || !loadsGone) return
    const st = useStore.getState()
    useWizard.setState({ made: { ...made, loads: made.loads.filter((l) => !!nodeOf(st, l)) } })
  }, [made, loadsGone])

  if (!open) return null
  const close = closeAnalysisWizard
  const shownGoal = made?.goal ?? goal
  const reached = made ? step : PART
  return (
    <div className="tool-panel wizard">
      <div className="slice-row">
        <span className="slice-title"><b>{GOALS[shownGoal].label}</b></span>
        {made && <button className="small-btn" title="Start a new setup (the nodes made so far stay in the graph)"
          onClick={() => useWizard.setState({ made: null, step: PART, source: null, body: 0 })}>New</button>}
        <button className="small-btn" onClick={close} title="Close (the nodes stay in the graph; reopen to go on)">×</button>
      </div>
      <div className="wiz-steps">
        {stepLabels(shownGoal).map((label, k) => (
          <button key={label} className={`wiz-step ${k === step ? 'on' : ''} ${k < reached ? 'done' : ''}`}
            disabled={!made || k === PART || k > reached} onClick={() => go(k)}>
            <span className="wiz-num">{k < reached ? '✓' : k + 1}</span>{label}
          </button>
        ))}
      </div>
      {step === PART || !made ? <PartStep />
        : step === HELD ? <HeldStep made={made} />
        : step === LOADS ? <LoadsStep made={made} />
        : step === SETTINGS ? <SettingsStep made={made} />
        : <RunStep made={made} />}
    </div>
  )
}
