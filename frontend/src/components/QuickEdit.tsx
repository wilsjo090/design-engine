import { useEffect, useLayoutEffect, useRef, useState, type MutableRefObject, type ReactNode, type SyntheticEvent } from 'react'
import { createPortal } from 'react-dom'
import { useStore } from '../store'
import type { PortSpec } from '../types'

/** The inputs each node type shows in the quick-edit box (add a type by listing its input names). */
export const QUICK_INPUTS: Record<string, string[]> = {
  'sim.load': ['amount', 'direction', 'vector'],
  'pattern.holes': ['shape', 'size', 'spacing', 'depth', 'height'],
  'pattern.surface': ['shape', 'size', 'spacing'],
  'pattern.text': ['text', 'height'],
  'pattern.picture': ['path', 'width'],
}

/** Port types typed in as numbers (an int is rounded). */
const NUMERIC = ['number', 'scalar', 'int']

/** Whether nodes of this type have a quick-edit box. */
export function hasQuickEdit(typeId: string): boolean {
  return Object.prototype.hasOwnProperty.call(QUICK_INPUTS, typeId)
}

/** Reads one input of the node: what is typed in, else the port's default. */
type Get = (port: string) => unknown

/** What depends on the node's other values: whether an input applies right now, and its unit. */
interface Rules {
  shows?: (name: string, get: Get) => boolean
  unit?: (name: string, get: Get) => string | undefined
}

/** The unit of the Load's Amount, by kind. */
const LOAD_UNITS: Record<string, string> = { force: 'N', bearing: 'N', pressure: 'MPa', torque: 'N·m' }

const RULES: Record<string, Rules> = {
  'sim.load': {
    // a pressure is always square to the faces; Vector is only read for a custom direction
    shows: (name, get) => {
      if (name === 'direction') return get('kind') !== 'pressure'
      if (name === 'vector') return get('kind') !== 'pressure' && get('direction') === 'custom'
      return true
    },
    unit: (name, get) => (name === 'amount' ? LOAD_UNITS[String(get('kind'))] : undefined),
  },
}

/** Gap kept between the box and the window's edge. */
const MARGIN = 4
/** How far the box sits from the cursor, right and down. */
const OFFSET = 12

const show = (v: unknown) => (v === undefined || v === null ? '' : String(v))
/** "0,35" and "0.35" are both 0.35; anything that is not a number is undefined. */
const parseNumber = (raw: string) => {
  const v = Number(raw.replace(',', '.'))
  return raw.trim() !== '' && Number.isFinite(v) ? v : undefined
}
const stop = (e: SyntheticEvent) => e.stopPropagation()

/** What a field needs from the box. */
interface FieldProps {
  value: unknown
  commit: (v: unknown) => void
  /** Enter was pressed: the box closes */
  done: () => void
  /** set while the box closes without committing (Esc, a click outside): a blur then keeps its typing to itself */
  closing: MutableRefObject<boolean>
}

/** A text box that commits on Enter or blur; `parse` returns the value to store, or undefined to ignore the text. */
function TextField({ value, commit, done, closing, parse }: FieldProps & { parse: (raw: string) => unknown }) {
  const [text, setText] = useState(show(value))
  useEffect(() => {
    // Don't overwrite while mid-typing (e.g. "0,") a text that already means this value.
    if (parse(text) !== value) setText(show(value))
  }, [value]) // eslint-disable-line react-hooks/exhaustive-deps

  const finish = () => {
    if (text === show(value)) return
    const v = parse(text)
    if (v === undefined) setText(show(value))  // not a number: back to what it was
    else if (v !== value) commit(v)
  }
  return (
    <input className="vp-quick-field" type="text" spellCheck={false} value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => { if (!closing.current) finish() }}
      onKeyDown={(e) => { if (e.key === 'Enter') { finish(); done() } }} />
  )
}

/** A choice list; picking commits at once. */
function ChoiceField({ value, commit, done, choices }: FieldProps & { choices: string[] }) {
  const current = show(value)
  const options = choices.includes(current) ? choices : [current, ...choices]
  return (
    <select className="vp-quick-field" value={current}
      onChange={(e) => commit(e.target.value)}
      onKeyDown={(e) => { if (e.key === 'Enter') done() }}>
      {options.map((c) => <option key={c} value={c}>{c}</option>)}
    </select>
  )
}

/** One line of the box: the label on the left, the field (and a unit) on the right. */
function Row({ label, unit, children }: { label: string; unit?: string; children: ReactNode }) {
  return (
    <label className="vp-quick-row">
      <span className="vp-quick-label" title={label}>{label}</span>
      {children}
      {unit && <span className="vp-quick-unit">{unit}</span>}
    </label>
  )
}

/** A small box at the cursor with the few inputs of a node that matter (see QUICK_INPUTS), so a
 *  value can be typed in right where the node was added. Enter commits the field and closes, Esc or a
 *  click outside closes without the field's unfinished typing; values already committed stay. */
export function QuickEdit({ node, x, y, onClose }: { node: string; x: number; y: number; onClose: () => void }) {
  const n = useStore((s) => s.nodes.find((m) => m.id === node))
  const spec = useStore((s) => (n ? s.specs[n.data.typeId] : undefined))
  const edges = useStore((s) => s.edges)
  const ref = useRef<HTMLDivElement>(null)
  const closing = useRef(false)
  const names = n ? QUICK_INPUTS[n.data.typeId] : undefined
  const gone = !n || !spec || !names

  /** close without committing what is typed in a field */
  const cancel = () => {
    closing.current = true
    onClose()
  }

  // the node was deleted (or undone away) while the box was open
  useEffect(() => {
    if (gone) onClose()
  }, [gone, onClose])

  // Placed by hand before the first paint, and again whenever rows come or go (measured at the corner).
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.left = '0px'
    el.style.top = '0px'
    const w = el.offsetWidth, h = el.offsetHeight
    const clamp = (v: number, size: number, room: number) => Math.max(MARGIN, Math.min(v, room - size - MARGIN))
    el.style.left = `${clamp(x + OFFSET, w, window.innerWidth)}px`
    el.style.top = `${clamp(y + OFFSET, h, window.innerHeight)}px`
  })

  // The first field that can be edited gets the focus, its text selected.
  useEffect(() => {
    const first = ref.current?.querySelector<HTMLInputElement | HTMLSelectElement>('input, select')
    first?.focus({ preventScroll: true })
    if (first instanceof HTMLInputElement) first.select()
  }, [])

  // A click anywhere else (the 3D view still gets it) or a resize closes the box.
  useEffect(() => {
    const outside = (e: Event) => {
      if (!ref.current?.contains(e.target as Node | null)) cancel()
    }
    window.addEventListener('pointerdown', outside, true)
    window.addEventListener('resize', cancel)
    return () => {
      window.removeEventListener('pointerdown', outside, true)
      window.removeEventListener('resize', cancel)
    }
  }, [onClose]) // eslint-disable-line react-hooks/exhaustive-deps

  if (gone) return null
  const get: Get = (port) => n.data.inputs[port] ?? spec.inputs.find((p) => p.name === port)?.default
  const rules = RULES[n.data.typeId]
  const ports = names
    .map((name) => spec.inputs.find((p) => p.name === name))
    .filter((p): p is PortSpec => !!p && (rules?.shows?.(p.name, get) ?? true))

  const done = () => {
    closing.current = true
    onClose()
  }
  return createPortal(
    <div ref={ref} className="vp-quick" role="dialog" aria-label={`Quick edit ${node}`}
      // events of a portal bubble through the React tree: keep them away from the 3D view's handlers
      onKeyDown={(e) => { stop(e); if (e.key === 'Escape') { e.preventDefault(); cancel() } }}
      onKeyUp={stop} onPointerDown={stop} onMouseDown={stop} onClick={stop} onDoubleClick={stop}
      onContextMenu={stop} onWheel={stop}>
      <div className="vp-quick-head"><b>{node}</b><span className="vp-quick-title"> · {spec.title}</span></div>
      {ports.map((p) => {
        const label = p.label || p.name
        if (edges.some((e) => e.target === node && e.targetHandle === p.name)) {
          return (
            <div className="vp-quick-row" key={p.name}>
              <span className="vp-quick-label" title={label}>{label}</span>
              <span className="vp-quick-linked">linked</span>
            </div>
          )
        }
        const props: FieldProps = {
          value: get(p.name),
          commit: (v) => useStore.getState().setInput(node, p.name, v),
          done, closing,
        }
        return (
          <Row key={p.name} label={label} unit={rules?.unit?.(p.name, get)}>
            {p.type === 'text' && p.choices?.length ? <ChoiceField {...props} choices={p.choices} />
              : <TextField {...props} parse={!NUMERIC.includes(p.type) ? (raw) => raw
                : p.type === 'int' ? (raw) => { const v = parseNumber(raw); return v === undefined ? v : Math.round(v) }
                : parseNumber} />}
          </Row>
        )
      })}
      <div className="vp-quick-foot">Enter ✓ · Esc</div>
    </div>,
    document.body,
  )
}
