// Lists to choose an input's value from on a card, instead of typing what has to be looked up elsewhere:
// Pick Body's body (Port pick "body"), Text's font (pick "font"); plus the dice of seed inputs and the switch
// of 1 / 0 ints (Port switch).
import { type CSSProperties, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, type BodyInfo } from '../api'
import { useStore } from '../store'
import type { PortSpec } from '../types'
import { NumberInput, fmt } from './PortInputs'
import './PickLists.css'

interface Row {
  key: string
  label: string
  detail?: string
  title?: string
  style?: CSSProperties
  /** lower case text the filter matches */
  search: string
}

const WIDTH = 360

/** A list under (or over) `anchor`, on top of the page so it stays readable however far the graph is zoomed
 *  out: a filter box (Enter takes the row marked, or the only one left), arrows move, Esc or a click
 *  elsewhere closes. `typed`: what Enter does with a filter no row matches (e.g. a number typed). */
function PickList({ anchor, rows, current, note, placeholder, onPick, onHover, onClose, typed, children }: {
  anchor: HTMLElement; rows: Row[]; current?: string; note?: string; placeholder: string
  onPick: (key: string) => void; onHover?: (key: string | null) => void; onClose: () => void
  typed?: (text: string) => boolean; children?: React.ReactNode
}) {
  const [filter, setFilter] = useState('')
  const [at, setAt] = useState(-1)
  const box = useRef<HTMLDivElement>(null)
  const field = useRef<HTMLInputElement>(null)
  // placed at once (a hidden box cannot take the keyboard: typing went to the card's field)
  const place = useMemo<CSSProperties>(() => {
    const r = anchor.getBoundingClientRect()
    const below = window.innerHeight - r.bottom - 12
    const above = r.top - 12
    const left = Math.max(8, Math.min(r.left, window.innerWidth - WIDTH - 8))
    return below >= 260 || below >= above
      ? { left, top: r.bottom + 3, maxHeight: Math.max(120, below) }
      : { left, bottom: window.innerHeight - r.top + 3, maxHeight: Math.max(120, above) }
  }, [anchor])
  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase()
    return f ? rows.filter((r) => f.split(/\s+/).every((w) => r.search.includes(w))) : rows
  }, [rows, filter])

  useLayoutEffect(() => { field.current?.focus() }, [])
  useEffect(() => {
    const down = (e: MouseEvent) => {
      const t = e.target as Node
      if (!box.current?.contains(t) && !anchor.contains(t)) onClose()
    }
    window.addEventListener('mousedown', down, true)
    return () => window.removeEventListener('mousedown', down, true)
  }, [anchor, onClose])
  useEffect(() => { setAt(-1) }, [filter])
  useEffect(() => { onHover?.(at >= 0 ? shown[at]?.key ?? null : null) }, [at, shown, onHover])
  useEffect(() => { // the current value in sight when the list opens
    box.current?.querySelector('.pick-row.current')?.scrollIntoView({ block: 'center' })
  }, [rows.length])
  useEffect(() => { box.current?.querySelector('.pick-row.at')?.scrollIntoView({ block: 'nearest' }) }, [at])

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setAt((a) => Math.min(shown.length - 1, a + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setAt((a) => Math.max(0, a - 1)) }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onClose() }
    else if (e.key === 'Enter') {
      e.preventDefault()
      e.stopPropagation() // (the list is drawn inside a card or a quick edit box: they must not take it too)
      const row = shown[at] ?? (shown.length === 1 ? shown[0] : undefined)
      if (row) onPick(row.key)
      else if (filter.trim() && typed?.(filter.trim())) onClose()
    }
  }
  return createPortal(
    <div className="pick-list" ref={box} style={{ ...place, width: WIDTH }} onMouseLeave={() => setAt(-1)}>
      <input ref={field} className="pick-filter" value={filter} placeholder={placeholder} spellCheck={false}
        onChange={(e) => setFilter(e.target.value)} onKeyDown={onKey} />
      {note && <div className="pick-note">{note}</div>}
      {children}
      <div className="pick-rows">
        {shown.map((r, i) => (
          <div key={r.key} className={`pick-row ${r.key === current ? 'current' : ''} ${i === at ? 'at' : ''}`}
            title={r.title} onMouseEnter={() => setAt(i)} onMouseDown={(e) => e.preventDefault()}
            onClick={() => onPick(r.key)}>
            <span className="pick-label" style={r.style}>{r.label}</span>
            {r.detail && <span className="pick-detail">{r.detail}</span>}
          </div>
        ))}
        {rows.length > 0 && !shown.length && <div className="pick-empty">Nothing matches “{filter}”</div>}
      </div>
    </div>,
    document.body,
  )
}

const mm = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2))
  .replace(/\.?0+$/, '')
const bodyName = (b: BodyInfo) => b.name || `Body ${b.index + 1}`
const bodySize = (b: BodyInfo) => b.hi.map((h, k) => mm(h - b.lo[k])).join(' × ') + ' mm'

/** The bodies on a node's list input (/api/bodies), read again after every run while `live`; names are read
 *  from the STEP file (seconds, once per file) only when `withNames`. */
function useBodies(nodeId: string, input: string, live: boolean, withNames: boolean) {
  const link = useStore((s) => s.edges.find((e) => e.target === nodeId && e.targetHandle === input))
  const runEnd = useStore((s) => (s.run && !s.run.running ? s.run.id : null))
  const [list, setList] = useState<BodyInfo[] | null | undefined>(undefined) // undefined: asking
  const [naming, setNaming] = useState(false)
  const source = link?.source ?? ''
  const port = link?.sourceHandle ?? ''
  useEffect(() => {
    if (!live) return
    if (!source) { setList(null); return }
    let current = true
    ;(async () => {
      try {
        const r = await api.bodies(nodeId, input, source, port)
        if (!current) return
        setList(r.bodies)
        if (r.bodies && r.names_unread && withNames) {
          setNaming(true)
          const named = await api.bodies(nodeId, input, source, port, true)
          if (current) setList(named.bodies)
        }
      } catch {
        if (current) setList(null)
      } finally {
        if (current) setNaming(false)
      }
    })()
    return () => { current = false }
  }, [nodeId, input, source, port, runEnd, live, withNames])
  return { list, naming, source }
}

/** A row per body: number and name, size and faces, its box in the tooltip. */
function bodyRows(list: BodyInfo[] | null | undefined): Row[] {
  return (list ?? []).map((b) => ({
    key: String(b.index), label: `${b.index}  ${bodyName(b)}`,
    detail: `${bodySize(b)}${b.faces ? ` · ${b.faces} faces` : ''}`,
    title: `x ${mm(b.lo[0])} … ${mm(b.hi[0])}, y ${mm(b.lo[1])} … ${mm(b.hi[1])}, z ${mm(b.lo[2])} … ${mm(b.hi[2])} mm`,
    search: `${b.index} ${bodyName(b)} ${bodySize(b)}`.toLowerCase(),
  }))
}

/** Draws the hovered body's box in the 3D view (store.bodyHint); cleared when the field goes away. */
function useHint(list: BodyInfo[] | null | undefined) {
  useEffect(() => () => { useStore.setState({ bodyHint: null }) }, [])
  return useCallback((key: string | null) => {
    const b = key === null ? undefined : list?.[Number(key)]
    useStore.setState({ bodyHint: b ? { lo: b.lo, hi: b.hi, label: `${b.index} · ${bodyName(b)}` } : null })
  }, [list])
}

type State = ReturnType<typeof useStore.getState>

/** The Pick Body the body on `node.input` comes from: the node wired into it, or further up through nodes with
 *  one wired body input (Scale, Move, Mirror...); "" when there is none (the input gets the whole list). */
function pickAbove(s: State, node: string, input: string): string {
  let edge = s.edges.find((e) => e.target === node && e.targetHandle === input)
  for (let depth = 0; edge && depth < 16; depth++) {
    const src = s.nodes.find((n) => n.id === edge!.source)
    if (!src) return ''
    if (src.data.typeId === 'input.pick_body') return src.id
    const wired = (s.specs[src.data.typeId]?.inputs ?? []).filter((p) => p.type === 'cad')
      .map((p) => s.edges.find((e) => e.target === src.id && e.targetHandle === p.name)).filter((e) => !!e)
    if (wired.length !== 1) return ''
    edge = wired[0]
  }
  return ''
}

/** Which body a later node works on, chosen on its own card (a wired body input), so there is no going back
 *  to the Pick Body: with a Pick Body above it, the chip shows that body and a choice changes the Pick Body
 *  (every node after it follows); an input that gets a whole list shows "all N bodies", and a choice puts a
 *  Pick Body in front of it (one that already picks that body from the same list, else a new one). */
export function BodyChip({ nodeId, input }: { nodeId: string; input: string }) {
  const pick = useStore((s) => pickAbove(s, nodeId, input))
  const pickInput = useStore((s) => {
    const n = s.nodes.find((m) => m.id === pick)
    return n ? s.specs[n.data.typeId]?.inputs.find((p) => p.type === 'cad')?.name ?? '' : ''
  })
  const raw = useStore((s) => Number(s.nodes.find((n) => n.id === pick)?.data.inputs.index ?? 0))
  const [open, setOpen] = useState(false)
  const btn = useRef<HTMLButtonElement>(null)
  const { list, naming } = useBodies(pick || nodeId, pick ? pickInput : input, true, open)
  const hint = useHint(list)
  const close = useCallback(() => { setOpen(false); useStore.setState({ bodyHint: null }) }, [])
  const rows = useMemo(() => bodyRows(list), [list])
  if (!pick && !(list && list.length > 1)) return null

  const index = list && raw < 0 ? list.length + raw : raw
  const body = pick ? list?.[index] : undefined
  const text = pick ? (body ? `${index} · ${bodyName(body)}` : `body ${raw}`) : `all ${list!.length} bodies`
  const choose = (key: string) => {
    const n = Number(key)
    const st = useStore.getState()
    close()
    if (pick) {
      st.setInput(pick, 'index', n)
      st.flashNodes([pick])
      return
    }
    const link = st.edges.find((e) => e.target === nodeId && e.targetHandle === input)
    if (!link) return
    const name = list?.[n] ? ` ${bodyName(list[n])}` : ''
    const same = st.nodes.find((m) => m.data.typeId === 'input.pick_body' && Number(m.data.inputs.index ?? 0) === n &&
      st.edges.some((e) => e.target === m.id && e.targetHandle === 'bodies' && e.source === link.source &&
        e.sourceHandle === link.sourceHandle))
    if (same) {
      st.applyEdit({ links: [{ from: same.id, port: 'body', to: nodeId, input }] }, 'Pick body',
        { text: `${nodeId} takes body ${n}${name} from ${same.id}` })
      return
    }
    st.applyEdit({
      add: [{ key: 'pick', type: 'input.pick_body', inputs: { index: n }, near: link.source }],
      links: [{ from: link.source, port: link.sourceHandle ?? '', to: 'pick', input: 'bodies' },
        { from: 'pick', port: 'body', to: nodeId, input }],
    }, 'Pick body', { text: (ids) => `Added ${ids.pick} (body ${n}${name}) between ${link.source} and ${nodeId}` })
  }
  return (
    <>
      <button ref={btn} className={`nodrag body-chip ${open ? 'on' : ''}`}
        title={pick ? `The body ${pick} picks${body ? `: ${bodyName(body)}, ${bodySize(body)}` : ''}. Click to choose ` +
          `another: ${pick} changes, and every node after it follows`
          : `This input gets all ${list!.length} bodies. Click to choose one: a Pick Body goes in front of it`}
        onClick={() => (open ? close() : setOpen(true))}>
        <span className="pick-field-text">{text}</span>
        <span className="pick-field-arrow">▾</span>
      </button>
      {open && btn.current && (
        <PickList anchor={btn.current} rows={rows} current={body ? String(index) : undefined}
          placeholder={list ? `Filter ${list.length} bodies: name or number` : 'Type a body number, Enter'}
          note={list === undefined ? 'Asking the engine…'
            : list === null ? 'The bodies are not made yet: show the part (or anything after it) to list them.'
            : naming ? 'Reading the body names from the STEP file (once per file)…'
            : pick ? `Changes ${pick}: every node after it follows.` : 'Puts a Pick Body in front of this input.'}
          onPick={choose} onHover={hint} onClose={close}
          typed={(t) => { const n = Number(t); if (!Number.isInteger(n) || !pick) return false; choose(String(n)); return true }} />
      )}
    </>
  )
}

/** Which body of a list (Pick Body's Index): the field shows the body's number and name; a click lists every
 *  body with its STEP name, size and faces, and the hovered one's box is drawn in the 3D view, so it is
 *  found without looking for it there. */
export function BodyPick({ nodeId, input, port, value, onCommit }: {
  nodeId: string; input: string; port: PortSpec; value: unknown; onCommit: (v: number) => void
}) {
  const [open, setOpen] = useState(false)
  const btn = useRef<HTMLButtonElement>(null)
  const { list, naming, source } = useBodies(nodeId, input, true, open)
  const raw = Number(value ?? port.default ?? 0)
  const index = list && raw < 0 ? list.length + raw : raw
  const body = list?.[index]
  const hint = useHint(list)
  const close = useCallback(() => { setOpen(false); useStore.setState({ bodyHint: null }) }, [])
  const rows = useMemo(() => bodyRows(list), [list])
  const shown = useStore((s) => s.visible.includes(source))
  const pick = (key: string) => { onCommit(Number(key)); close() }

  return (
    <>
      <button ref={btn} className={`nodrag value-input pick-field ${open ? 'on' : ''}`}
        title={body ? `${bodyName(body)}: ${bodySize(body)}. Click for the list of bodies` : 'Click for the list of bodies'}
        onClick={() => (open ? close() : setOpen(true))}>
        <span className="pick-field-text">{body ? `${index} · ${bodyName(body)}` : fmt(raw)}</span>
        <span className="pick-field-arrow">▾</span>
      </button>
      {open && btn.current && (
        <PickList anchor={btn.current} rows={rows} current={body ? String(index) : undefined}
          placeholder={list ? `Filter ${list.length} bodies: name or number` : 'Type a body number, Enter'}
          note={list === undefined ? 'Asking the engine…'
            : list === null ? (source ? `${source} has not made its bodies yet: show it (or anything after it) to list them.`
              : `Connect a list of bodies to ${input} first.`)
            : naming ? 'Reading the body names from the STEP file (once per file)…' : undefined}
          onPick={pick} onHover={hint} onClose={close}
          typed={(t) => { const n = Number(t); if (!Number.isInteger(n)) return false; onCommit(n); return true }}>
          {list === null && source && !shown && (
            <button className="pick-action" onClick={() => useStore.getState().toggleVisible(source)}>Show {source}</button>
          )}
        </PickList>
      )}
    </>
  )
}

type Font = Awaited<ReturnType<typeof api.fonts>>['fonts'][number]
let fonts: Promise<Font[]> | null = null // read once per page (the engine opens every font file once: ~3 s)

/** The CSS that shows a font's own letters in the list (the browser finds installed fonts by family). */
function fontStyle(f: Font): CSSProperties {
  const s = f.style.toLowerCase()
  const weight = /black|heavy/.test(s) ? 900 : /extra ?bold|ultra ?bold/.test(s) ? 800 : /semi ?bold|demi ?bold/.test(s) ? 600
    : /bold/.test(s) ? 700 : /medium/.test(s) ? 500 : /extra ?light|ultra ?light/.test(s) ? 200 : /light/.test(s) ? 300
      : /thin|hairline/.test(s) ? 100 : 400
  return { fontFamily: `"${f.family}", var(--font)`, fontWeight: weight, fontStyle: /italic|oblique/.test(s) ? 'italic' : 'normal',
    fontStretch: /narrow|condensed/.test(s) ? 'condensed' : 'normal' }
}

/** A list button next to a font field (Text's Font): every installed font by name, shown in itself. */
export function FontPick({ value, onCommit }: { value: unknown; onCommit: (v: string) => void }) {
  const [open, setOpen] = useState(false)
  const [list, setList] = useState<Font[] | null>(null)
  const [failed, setFailed] = useState('')
  const btn = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!open || list) return
    fonts ??= api.fonts().then((r) => r.fonts)
    fonts.then(setList).catch((e: Error) => { fonts = null; setFailed(e.message) })
  }, [open, list])
  const rows = useMemo<Row[]>(() => (list ?? []).map((f) => ({
    key: f.value, label: f.style === 'Regular' ? f.family : `${f.family} ${f.style}`, detail: f.value.split(/[\\/]/).pop(),
    style: fontStyle(f), title: f.value, search: `${f.family} ${f.style} ${f.value}`.toLowerCase(),
  })), [list])
  const close = useCallback(() => setOpen(false), [])
  return (
    <>
      <button ref={btn} className={`nodrag browse-btn ${open ? 'on' : ''}`} title="Choose an installed font"
        onClick={() => setOpen((o) => !o)}>▾</button>
      {open && btn.current && (
        <PickList anchor={btn.current} rows={rows} current={fmt(value)} placeholder="Filter fonts: name or style"
          note={failed ? `Cannot list the fonts: ${failed}` : list ? undefined : 'Reading the installed fonts…'}
          onPick={(v) => { onCommit(v); close() }} onClose={close}
          typed={(t) => { onCommit(t); return true }} />
      )}
    </>
  )
}

/** A seed (any whole number): the field and a dice for another random pattern in one click. */
export function SeedInput({ value, onCommit }: { value: unknown; onCommit: (v: number) => void }) {
  return (
    <div className="path-input">
      <NumberInput value={value} integer onCommit={(v) => onCommit(Number(v))} />
      <button className="nodrag browse-btn" title="Another random pattern (a new seed)"
        onClick={() => {
          let next = Number(value ?? 0)
          while (next === Number(value ?? 0)) next = 1 + Math.floor(Math.random() * 9999)
          onCommit(next)
        }}>🎲</button>
    </div>
  )
}

/** An int that is 1 or 0 (Port switch), as an on/off switch. */
export function IntSwitch({ value, onCommit }: { value: unknown; onCommit: (v: number) => void }) {
  const on = Number(value ?? 0) !== 0
  return (
    <span className="toggle-field">
      <button type="button" role="switch" aria-checked={on} className={`toggle nodrag ${on ? 'on' : ''}`}
        title={`${on ? 'On' : 'Off'}: click for ${on ? 'off' : 'on'}`} onClick={() => onCommit(on ? 0 : 1)}>
        <span className="toggle-knob" />
      </button>
    </span>
  )
}
