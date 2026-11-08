import type { PointerEvent as ReactPointerEvent } from 'react'
import { create } from 'zustand'
import { GROUP_IN_ID, GROUP_OUT_ID } from './graphConvert'
import type { EngineEdge, EngineNode } from './types'

/** How the graph draws its wires and cards (the View menu over the graph; kept in localStorage). */
export interface WireOptions {
  /** zoomed out past FAR_ZOOM, cards show only their name (big enough to read) and their wired ports */
  simple: boolean
  /** the wires of the selected or hovered nodes stand out, the others fade */
  focus: boolean
  /** an output with SHARED_WIRES or more wires shows them as labels on the inputs they feed */
  shared: boolean
  /** a wire longer than LONG_WIRE shows as a label too */
  long: boolean
}

/** what the pointer rests on: a node (all its wires) or one wire (or a label of one) */
export interface WireHover { node?: string; edge?: string }

interface WireState extends WireOptions {
  hover: WireHover | null
  set: (patch: Partial<WireOptions>) => void
}

export const SHARED_WIRES = 3
/** px from the source card's right side to the target card's corner (about five Tidy columns) */
export const LONG_WIRE = 1600
/** ms the pointer rests before the wires follow it (moving across the graph does not flicker them) */
const HOVER_WAIT = 120

/** below this zoom the cards' text is too small to read: with `simple`, they show their name instead */
export const FAR_ZOOM = 0.45

const DEFAULTS: WireOptions = { simple: true, focus: true, shared: true, long: true }
const KEY = 'wires'

function loadOptions(): WireOptions {
  try { return { ...DEFAULTS, ...(JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<WireOptions>) } } catch { return DEFAULTS }
}

export const useWires = create<WireState>((set, get) => ({
  ...loadOptions(),
  hover: null,
  set: (patch) => {
    set(patch)
    const { simple, focus, shared, long } = get()
    try { localStorage.setItem(KEY, JSON.stringify({ simple, focus, shared, long })) } catch { /* storage unavailable */ }
  },
}))

let hoverTimer: number | undefined
/** The pointer rests on `hover` (null: on nothing) after `wait` ms; a newer call replaces a waiting one. */
export function hoverWires(hover: WireHover | null, wait = HOVER_WAIT) {
  window.clearTimeout(hoverTimer)
  hoverTimer = window.setTimeout(() => {
    const now = useWires.getState().hover
    if (now?.node !== hover?.node || now?.edge !== hover?.edge) useWires.setState({ hover })
  }, wait)
}

/** Pressing a wired input's dot picks up its wire, as dragging the wire's end does: React Flow's reconnect,
 *  started by a mousedown on the end's anchor (React Flow starts a new wire from a pressed dot instead,
 *  which `stopPropagation` keeps from happening). A wire shown as a label is drawn first; the press is
 *  dropped if the button is up before it is. NodeEditor's onReconnect / onReconnectEnd say where it goes. */
export function pickUpWire(e: ReactPointerEvent, edge: string) {
  if (e.button !== 0 || e.ctrlKey || e.shiftKey || e.altKey || e.metaKey) return
  e.stopPropagation()
  const { clientX, clientY } = e
  let released = false
  window.addEventListener('pointerup', () => { released = true }, { once: true, capture: true })
  hoverWires({ edge }, 0)
  const start = (tries: number) => {
    if (released) return
    const anchor = document.querySelector(`.react-flow__edge[data-id="${CSS.escape(edge)}"] .react-flow__edgeupdater-target`)
    if (anchor) anchor.dispatchEvent(new MouseEvent('mousedown', { clientX, clientY, button: 0, buttons: 1, bubbles: true, cancelable: true, view: window }))
    else if (tries) requestAnimationFrame(() => start(tries - 1))
  }
  start(10)
}

const isGroupIO = (id: string) => id === GROUP_IN_ID || id === GROUP_OUT_ID

interface Labels {
  /** the wires drawn as labels */
  ids: Set<string>
  /** per node, as a string (cards re-render only when theirs changes): {"in": {input: wire id}, "out": {output: labels}} */
  of: Record<string, string>
}

let memo: { nodes: EngineNode[]; edges: EngineEdge[]; shared: boolean; long: boolean; labels: Labels } | null = null

/** The wires shown as labels on the inputs they feed instead of as lines (while neither end is in focus):
 *  those of outputs with SHARED_WIRES or more wires, and those longer than LONG_WIRE. Not the wires of an
 *  open group's input and output nodes. */
export function wireLabels(nodes: EngineNode[], edges: EngineEdge[], shared: boolean, long: boolean): Labels {
  if (memo && memo.nodes === nodes && memo.edges === edges && memo.shared === shared && memo.long === long) return memo.labels
  const ids = new Set<string>()
  if (shared || long) {
    const count = new Map<string, number>()
    for (const e of edges) count.set(`${e.source}.${e.sourceHandle}`, (count.get(`${e.source}.${e.sourceHandle}`) ?? 0) + 1)
    const at = new Map(nodes.map((n) => [n.id, n]))
    for (const e of edges) {
      if (isGroupIO(e.source) || isGroupIO(e.target)) continue
      if (shared && count.get(`${e.source}.${e.sourceHandle}`)! >= SHARED_WIRES) { ids.add(e.id); continue }
      const a = at.get(e.source), b = at.get(e.target)
      if (long && a && b && Math.hypot(b.position.x - a.position.x - (a.measured?.width ?? 260), b.position.y - a.position.y) > LONG_WIRE) {
        ids.add(e.id)
      }
    }
  }
  const per: Record<string, { in: Record<string, string>; out: Record<string, number> }> = {}
  const entry = (id: string) => (per[id] ??= { in: {}, out: {} })
  for (const e of edges) {
    if (!ids.has(e.id)) continue
    entry(e.target).in[e.targetHandle ?? ''] = e.id
    const out = entry(e.source).out
    out[e.sourceHandle ?? ''] = (out[e.sourceHandle ?? ''] ?? 0) + 1
  }
  const labels = { ids, of: Object.fromEntries(Object.entries(per).map(([id, p]) => [id, JSON.stringify(p)])) }
  memo = { nodes, edges, shared, long, labels }
  return labels
}

/** What the wires focus on: the selected nodes (and what is in selected frames), the hovered ones (a
 *  hovered frame: what is in it) and the hovered wire. */
export interface WireFocus { picked: Set<string>; hovered: Set<string>; edge?: string }

/** The wires as drawn: labelled ones only while an end is in focus (dashed then), and with focus on,
 *  those in focus bright and the rest faded: strongly with a selection, lightly while only hovering. */
export function drawnEdges(nodes: EngineNode[], edges: EngineEdge[], o: Omit<WireOptions, 'simple'>, focus: WireFocus) {
  const labelled = wireLabels(nodes, edges, o.shared, o.long).ids
  const { picked, hovered } = focus
  const chosen = picked.size > 0 || edges.some((e) => e.selected)
  const inFocus = (e: EngineEdge) => picked.has(e.source) || picked.has(e.target) || !!e.selected ||
    e.id === focus.edge || hovered.has(e.source) || hovered.has(e.target)
  const fade = o.focus && (chosen || hovered.size > 0 || !!focus.edge) ? (chosen ? 'wire-dim' : 'wire-dim soft') : ''
  if (!labelled.size && !fade) return edges
  return edges.map((e) => {
    const lit = inFocus(e)
    const hidden = labelled.has(e.id) && !lit
    const className = [labelled.has(e.id) ? 'wire-tag' : '', lit ? '' : fade].filter(Boolean).join(' ')
    return hidden || className ? { ...e, hidden, className } : e
  })
}
