// What the texture and lattice wizards share: finding the part and the shown body a change goes
// into, a labelled input row, the part's size and volume.
import { useStore } from '../store'
import type { ViewMesh } from '../types'
import { portApplies } from '../types'
import { type BodySlot, type Source, bodySlot, upstream } from '../viewActions'
import { Row, cadPort, specOf, valueOf } from './AnalysisWizard'

type State = ReturnType<typeof useStore.getState>
export const linkInto = (st: State, node: string, input: string) =>
  st.edges.find((e) => e.target === node && e.targetHandle === input)
export const typeOf = (st: State, id: string) => st.nodes.find((n) => n.id === id)?.data.typeId
export const round1 = (x: number) => Number(x.toFixed(1))

const isCad = (st: State, id: string) => !!cadPort(st, id)
/** The CAD nodes a body is made from, nearest first (a Pick Body, not the Import STEP before it). */
export function nearestCad(st: State, id: string) {
  const cads = [...upstream(st, id)].filter((n) => isCad(st, n))
  return cads.filter((c) => !cads.some((d) => d !== c && upstream(st, d).has(c)))
}

/** The shown body made from the CAD output of `cad` that a change goes on: the last of them (the
 *  finished body, e.g. after an earlier texture), or null when none is shown. */
export function findBase(st: State, cad: string) {
  const bodies = st.visible.filter((v) => v !== cad && st.views[v] && st.views[v].kind !== 'cad' &&
    typeof bodySlot(st, v, null) !== 'string' && nearestCad(st, v).includes(cad))
  return bodies.find((v) => !bodies.some((w) => w !== v && upstream(st, w).has(v))) ?? null
}

/** The part to start on: what a shown body is made from, else a shown CAD body, else the first. */
export function defaultSource(st: State, sources: string[]) {
  const shown = st.visible.filter((v) => st.views[v])
  for (const v of shown) {
    if (st.views[v].kind === 'cad') continue
    const near = nearestCad(st, v).find((c) => sources.includes(c))
    if (near) return near
  }
  return shown.find((v) => sources.includes(v)) ?? sources[0] ?? null
}

/** Where a change goes in: after the shown body `view` (bodySlot), or on a CAD view a new Implicit
 *  Body from `cad`'s output (with its Pick Body, when that is still to be added). */
export function slotFor(st: State, cad: Omit<Source, 'key'>, view: string): BodySlot | string {
  if (view === cad.node || st.views[view]?.kind === 'cad' || (!st.views[view] && isCad(st, view))) {
    return { from: '', port: '', cad: { ...cad, key: '' }, consumers: [], view, swap: true }
  }
  return bodySlot(st, view, null)
}

/** mm from the lowest to the highest point along `axis` of the first of `views` that is shown. */
export function extentOf(st: State, views: (string | null | undefined)[], axis: string): [number, number] {
  const k = 'xyz'.indexOf(axis)
  const id = views.find((v) => v && st.views[v])
  const b = id ? st.views[id].bounds : null
  return b ? [round1(b[0][k]), round1(b[1][k])] : [0, 50]
}

/** The volume a shown mesh encloses, mm³ (it is closed: the sum of its triangles' signed tetrahedra). */
export function meshVolume(view: ViewMesh) {
  const p = view.positions, t = view.indices
  let v = 0
  for (let i = 0; i < t.length; i += 3) {
    const a = 3 * t[i], b = 3 * t[i + 1], c = 3 * t[i + 2]
    v += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c])
      + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c])
  }
  return Math.abs(v) / 6
}

/** An input of a node as a labelled row (hidden while it does not apply; a linked one names its source). */
export function Setting({ node, port, label }: { node: string; port: string; label?: string }) {
  const spec = useStore((s) => specOf(s, node)?.inputs.find((p) => p.name === port))
  const from = useStore((s) => linkInto(s, node, port)?.source)
  const applies = useStore((s) => {
    const p = specOf(s, node)?.inputs.find((q) => q.name === port)
    return !!p && portApplies(p, (q) => (linkInto(s, node, q) ? undefined : valueOf(s, node, q)))
  })
  if (!spec || !applies) return null
  const text = label ?? spec.label ?? port
  if (from) {
    return (
      <div className="wiz-row">
        <span className="wiz-label">{text}</span>
        <button className="case-row muted" onClick={() => useStore.getState().focusNode(from)} title="Set by this node: select it">← {from}</button>
      </div>
    )
  }
  return <Row node={node} port={port} label={text} title={spec.doc?.split(/(?<=\.)\s/)[0]} />
}

/** X / Y / Z buttons: the axis a fade or grading runs along. */
export function AxisButtons({ axis, onPick }: { axis: string; onPick: (axis: string) => void }) {
  return (
    <div className="seg" role="radiogroup">
      {['x', 'y', 'z'].map((a) => (
        <button key={a} type="button" role="radio" aria-checked={a === axis} className={a === axis ? 'on' : ''}
          onClick={() => { if (a !== axis) onPick(a) }}>{a.toUpperCase()}</button>
      ))}
    </div>
  )
}
