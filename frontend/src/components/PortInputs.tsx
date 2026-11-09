import { useEffect, useRef, useState, type KeyboardEvent, type MutableRefObject, type PointerEvent, type ReactNode } from 'react'
import { startValueDrag } from '../store'
import type { PortSpec } from '../types'
import { ChoicePicture, PICTURES } from './choicePictures'

/** The fields of a node's settings, shared by its card, the quick-edit box and the analysis wizard:
 *  numbers with their unit, what 0 means and a slider over the usual values; a switch for yes / no;
 *  a row of buttons for a few short choices, pictures for shapes, layouts and profiles; x, y, z as
 *  three numbers. */

export const fmt = (v: unknown) => (v === undefined || v === null ? '' : String(v))
/** "0,35" and "0.35" are both 0.35; anything that is not a number is undefined. */
export const parseNumber = (raw: string) => {
  const v = Number(raw.replace(',', '.'))
  return raw.trim() !== '' && Number.isFinite(v) ? v : undefined
}
/** A choice as it reads: schwarz_p -> "schwarz p". */
const pretty = (c: string) => c.replace(/_/g, ' ')

/** How a field commits: while typing (the cards, the wizard), or on Enter and blur (the quick-edit box,
 *  which closes on Enter and drops what is half typed when it closes with Esc: `closing` is then set). */
export interface CommitMode {
  live?: boolean
  onDone?: () => void
  closing?: MutableRefObject<boolean>
}

/** Step per few pixels when dragging a value: about 1% of its size (140 -> 1, 0.63 -> 0.001). */
function scrubStep(v: number, integer?: boolean) {
  if (integer) return 1
  return Math.pow(10, Math.floor(Math.log10(Math.abs(v) || 1)) - 2)
}
const decimalsOf = (step: number) => Math.max(0, -Math.floor(Math.log10(step) + 1e-9))
const clamp = (v: number, range?: [number, number]) => (range ? Math.min(range[1], Math.max(range[0], v)) : v)

/** A port label you can drag left/right to change its number (Shift = 10x faster, Alt = finer); with a
 *  slider it stays within the slider's values. */
export function ScrubLabel({ label, value, integer, range, onChange }: {
  label: string; value: unknown; integer?: boolean; range?: [number, number]; onChange: (v: number) => void
}) {
  const start = useRef<{ x: number; v: number; step: number } | null>(null)
  const down = (e: PointerEvent<HTMLSpanElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    const v = Number(value) || 0
    start.current = { x: e.clientX, v, step: range ? sliderStep(range, integer) : scrubStep(v, integer) }
    document.body.classList.add('scrubbing')
    startValueDrag()
  }
  const move = (e: PointerEvent<HTMLSpanElement>) => {
    const s = start.current
    if (!s) return
    const factor = e.shiftKey ? 10 : e.altKey ? 0.1 : 1
    const steps = Math.round((e.clientX - s.x) / (integer ? 8 : 3))
    const step = s.step * factor
    const next = clamp(Number((s.v + steps * step).toFixed(decimalsOf(step))), range)
    if (next !== Number(value)) onChange(integer ? Math.round(next) : next)
  }
  const up = (e: PointerEvent<HTMLSpanElement>) => {
    if (!start.current) return
    start.current = null
    e.currentTarget.releasePointerCapture(e.pointerId)
    document.body.classList.remove('scrubbing')
  }
  return (
    <span className="port-label scrub nodrag" title="Drag left/right to change (Shift = faster, Alt = finer)"
      onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
      {label}
    </span>
  )
}

/** A slider's step: about a hundredth of its values, rounded down to 1, 0.1, 0.01... */
const sliderStep = ([lo, hi]: [number, number], integer?: boolean) =>
  integer ? 1 : Math.pow(10, Math.floor(Math.log10((hi - lo) / 100) + 1e-9))

/** The bar under a number field: its fill shows where the value is among the usual ones; click or drag
 *  along it to set the value. */
function Slider({ value, range, integer, onChange }: {
  value: number; range: [number, number]; integer?: boolean; onChange: (v: number) => void
}) {
  const dragging = useRef(false)
  const [lo, hi] = range
  const set = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
    const step = sliderStep(range, integer)
    const v = Number((Math.round((lo + t * (hi - lo)) / step) * step).toFixed(decimalsOf(step)))
    if (v !== value) onChange(v)
  }
  const pct = Math.min(100, Math.max(0, (100 * (value - lo)) / (hi - lo)))
  return (
    <div className="num-slider nodrag" title={`Drag to set it: ${lo} … ${hi} (typing may go beyond)`}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        e.preventDefault()
        e.currentTarget.setPointerCapture(e.pointerId)
        dragging.current = true
        startValueDrag()
        set(e)
      }}
      onPointerMove={(e) => { if (dragging.current) set(e) }}
      onPointerUp={(e) => { dragging.current = false; e.currentTarget.releasePointerCapture(e.pointerId) }}
      onPointerCancel={() => { dragging.current = false }}>
      <div className="num-slider-fill" style={{ width: `${pct}%` }} />
      <div className="num-slider-thumb" style={{ left: `${pct}%` }} />
    </div>
  )
}

/** A number field: accepts "0,35" as well as "0.35", arrow keys nudge (Shift = x10). Its unit shows at
 *  its right; where 0 means something (`zero`: "auto", "no limit") a 0 shows as that, and emptying the
 *  field sets 0; `slider` adds a bar under it for the usual values. */
export function NumberInput({ value, integer, unit, zero, slider, axis, onCommit, live = true, onDone, closing,
  inputClass = 'nodrag nowheel value-input' }: {
  value: unknown; integer?: boolean; unit?: string; zero?: string; slider?: [number, number]
  /** x, y or z: a coloured letter in front (one of a point's three fields) */
  axis?: string
  onCommit: (v: number) => void; inputClass?: string
} & CommitMode) {
  const show = (v: unknown) => (zero && Number(v) === 0 ? '' : fmt(v))
  /** the number a text means (an empty field is 0 where 0 means something) */
  const read = (raw: string) => (zero && raw.trim() === '' ? 0 : parseNumber(raw))
  const [text, setText] = useState(show(value))
  useEffect(() => {
    // Don't overwrite while mid-typing (e.g. "0." or "0,3") a text that already means this value.
    if (read(text) !== value) setText(show(value))
  }, [value]) // eslint-disable-line react-hooks/exhaustive-deps
  const commit = (raw: string) => {
    const v = read(raw)
    if (v === undefined) return false
    const next = integer ? Math.round(v) : v
    if (next !== value) onCommit(next)
    return true
  }
  const finish = () => {
    if (live || text === show(value)) return
    if (!commit(text)) setText(show(value))  // not a number: back to what it was
  }
  const key = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      finish()
      onDone?.()
      return
    }
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
    e.preventDefault()
    const current = read(text) ?? 0
    const decimals = (text.split(/[.,]/)[1] ?? '').length
    let step = integer ? 1 : Math.pow(10, -Math.max(decimals, 1))
    if (e.shiftKey) step *= 10
    const next = +(current + (e.key === 'ArrowUp' ? step : -step)).toFixed(integer ? 0 : Math.max(decimals, 1))
    setText(show(next))
    onCommit(next)
  }
  const set = (v: number) => {
    setText(show(v))
    onCommit(v)
  }
  return (
    <div className="num-field">
      <div className="num-box">
        {axis && <span className={`num-axis axis-${axis}`}>{axis.toUpperCase()}</span>}
        <input className={inputClass} value={text} inputMode="decimal" placeholder={zero}
          style={{ paddingLeft: axis ? 17 : undefined, paddingRight: unit && text !== '' ? 10 + 6.2 * unit.length : undefined }}
          onChange={(e) => { setText(e.target.value); if (live) commit(e.target.value) }}
          onBlur={() => { if (!closing?.current) finish() }} onKeyDown={key} />
        {unit && text !== '' && <span className="num-unit">{unit}</span>}
      </div>
      {slider && <Slider value={Number(value) || 0} range={slider} integer={integer} onChange={set} />}
    </div>
  )
}

/** x, y, z typed as "1, 0, -1": undefined when the text is not three numbers. */
export function parseVec3(v: unknown): number[] | undefined {
  const parts = fmt(v).split(',').map(parseNumber)
  return parts.length === 3 && parts.every((p) => p !== undefined) ? (parts as number[]) : undefined
}
/** A text input that holds a point or a direction (its default is three numbers). */
export const isVec3 = (port: PortSpec) => port.type === 'text' && !port.choices && !!parseVec3(port.default)

/** A point or a direction as three number fields (call only when parseVec3(value) is defined). */
export function Vec3Input({ value, onCommit, ...mode }: { value: unknown; onCommit: (v: string) => void } & CommitMode
  & { inputClass?: string }) {
  const xyz = parseVec3(value)!
  return (
    <div className="vec3">
      {['x', 'y', 'z'].map((axis, k) => (
        <NumberInput key={axis} axis={axis} value={xyz[k]} {...mode}
          onCommit={(v) => onCommit(xyz.map((c, j) => (j === k ? v : c)).join(', '))} />
      ))}
    </div>
  )
}

const TOGGLES = [['no', 'yes'], ['off', 'on']]
/** yes / no and on / off: [off, on], or undefined for other choices. */
const toggleOf = (choices: string[]) =>
  choices.length === 2 ? TOGGLES.find((t) => t.every((c) => choices.includes(c))) : undefined

/** At most this many characters in all: the choices fit side by side as buttons. */
const SEGMENT_CHARS = 20

/** The tiles of a choice with pictures: the current one on a button, every one in a grid under the row
 *  while it is open (the row wraps to take it). */
function PicturePicker({ choices, value, pictures, onCommit }: {
  choices: string[]; value: string; pictures: Record<string, ReactNode>; onCommit: (v: string) => void
}) {
  const [open, setOpen] = useState(false)
  const button = useRef<HTMLButtonElement>(null)
  const grid = useRef<HTMLDivElement>(null)
  // a click anywhere else or Esc closes it
  useEffect(() => {
    if (!open) return
    const outside = (e: Event) => {
      const t = e.target as Node | null
      if (!button.current?.contains(t) && !grid.current?.contains(t)) setOpen(false)
    }
    const esc = (e: globalThis.KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); setOpen(false) } }
    window.addEventListener('pointerdown', outside, true)
    window.addEventListener('keydown', esc, true)
    return () => {
      window.removeEventListener('pointerdown', outside, true)
      window.removeEventListener('keydown', esc, true)
    }
  }, [open])
  return (
    <>
      <button ref={button} type="button" className={`pick-current nodrag ${open ? 'on' : ''}`}
        title={open ? 'Close' : 'Choose…'} onClick={() => setOpen((o) => !o)}>
        {pictures[value] && <ChoicePicture of={pictures[value]} />}
        <span className="pick-name">{pretty(value)}</span>
        <span className="pick-caret">{open ? '▴' : '▾'}</span>
      </button>
      {open && (
        <div ref={grid} className="pick-grid nodrag nowheel" role="listbox">
          {choices.map((c) => (
            <button key={c} type="button" role="option" aria-selected={c === value} title={pretty(c)}
              className={`pick-tile ${c === value ? 'on' : ''}`}
              onClick={() => { setOpen(false); if (c !== value) onCommit(c) }}>
              <ChoicePicture of={pictures[c]} />
              <span>{pretty(c)}</span>
            </button>
          ))}
        </div>
      )}
    </>
  )
}

/** One of an input's choices: a switch for yes / no and on / off, pictures for shapes, layouts and
 *  profiles, buttons side by side for a few short ones, else a list. */
export function ChoiceInput({ port, value, onCommit, onDone, selectClass = 'nodrag value-input' }: {
  port: PortSpec; value: unknown; onCommit: (v: string) => void; onDone?: () => void; selectClass?: string
}) {
  const choices = port.choices ?? []
  const current = fmt(value ?? port.default)
  const toggle = toggleOf(choices)
  if (toggle) {
    const on = current === toggle[1]
    return (
      <span className="toggle-field">
        <button type="button" role="switch" aria-checked={on} className={`toggle nodrag ${on ? 'on' : ''}`}
          title={`${on ? toggle[1] : toggle[0]}: click for ${on ? toggle[0] : toggle[1]}`}
          onClick={() => onCommit(on ? toggle[0] : toggle[1])}
          onKeyDown={(e) => { if (e.key === 'Enter') onDone?.() }}>
          <span className="toggle-knob" />
        </button>
      </span>
    )
  }
  const pictures = PICTURES[port.name]
  if (pictures && choices.includes(current) && choices.every((c) => pictures[c])) {
    return <PicturePicker choices={choices} value={current} pictures={pictures} onCommit={onCommit} />
  }
  if (choices.length <= 4 && choices.join('').length <= SEGMENT_CHARS && choices.includes(current)) {
    return (
      <div className="seg nodrag" role="radiogroup">
        {choices.map((c) => (
          <button key={c} type="button" role="radio" aria-checked={c === current} className={c === current ? 'on' : ''}
            onClick={() => { if (c !== current) onCommit(c) }}>{pretty(c)}</button>
        ))}
      </div>
    )
  }
  const options = choices.includes(current) ? choices : [current, ...choices]
  return (
    <select className={selectClass} value={current} onChange={(e) => onCommit(e.target.value)}
      onKeyDown={(e) => { if (e.key === 'Enter') onDone?.() }}>
      {options.map((c) => <option key={c} value={c}>{pretty(c)}</option>)}
    </select>
  )
}
