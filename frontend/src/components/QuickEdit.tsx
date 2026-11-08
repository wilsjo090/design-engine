import { useEffect, useLayoutEffect, useRef, useState, type MutableRefObject, type ReactNode, type SyntheticEvent } from 'react'
import { createPortal } from 'react-dom'
import { useStore } from '../store'
import type { NodeSpec, PortSpec, PromptState, QuickEditState, QuickField } from '../types'
import { changedFromDefault, portApplies, unitOf } from '../types'
import { ChoiceInput, NumberInput, Vec3Input, isVec3, parseVec3 } from './PortInputs'
import { BodyPick, IntSwitch, SeedInput } from './PickLists'

/** The inputs each node type shows in the quick-edit box (add a type by listing its input names);
 *  other types show all their plain inputs. */
export const QUICK_INPUTS: Record<string, string[]> = {
  'sim.load': ['amount', 'direction', 'vector'],
  'pattern.holes': ['shape', 'size', 'spacing', 'depth', 'height'],
  'pattern.surface': ['shape', 'size', 'spacing'],
  'pattern.text': ['text', 'height'],
  'pattern.picture': ['path', 'width'],
  'implicit.offset': ['distance'],
  'implicit.shell': ['thickness'],
  'implicit.smoothen': ['iterations', 'grid_size'],
  'implicit.ball': ['centre', 'radius'],
  'implicit.box': ['centre', 'size', 'rounding'],
  'implicit.bar': ['start', 'end', 'radius'],
  'input.number': ['value'],
}

/** Port types typed in as numbers (an int is rounded). */
const NUMERIC = ['number', 'scalar', 'int']
/** Port types the box can edit (others are only ever linked). */
const EDITABLE = [...NUMERIC, 'text', 'path', 'faces']

/** Whether nodes of this type have a curated quick-edit box (offered right after they are added). */
export function hasQuickEdit(typeId: string): boolean {
  return Object.prototype.hasOwnProperty.call(QUICK_INPUTS, typeId)
}

/** Reads one input of the node: what is typed in, else the port's default. */
type Get = (port: string) => unknown

/** Gap kept between the box and the window's edge. */
const MARGIN = 4
/** How far the box sits from the cursor, right and down. */
const OFFSET = 12

const show = (v: unknown) => (v === undefined || v === null ? '' : String(v))
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

/** One line of the box: the label on the left, the field on the right (a div: in a label, a click on
 *  the text would press a choice's first button). */
function Row({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <div className="vp-quick-row" title={title}>
      <span className="vp-quick-label" title={label}>{label}</span>
      {children}
    </div>
  )
}

/** The inputs a node shows in the box: its curated list, else every input the box can edit or that is
 *  linked (a link can be followed to the node it comes from), but not the advanced ones left as they are. */
function portsOf(spec: NodeSpec, typeId: string, linked: (port: string) => boolean, value: (port: string) => unknown): PortSpec[] {
  const names = QUICK_INPUTS[typeId]
  if (names) return names.map((n) => spec.inputs.find((p) => p.name === n)).filter((p): p is PortSpec => !!p)
  return spec.inputs.filter((p) => !p.hidden && (linked(p.name) ||
    (EDITABLE.includes(p.type) && !(p.advanced && !changedFromDefault(p, value(p.name))))))
}

/** A small box at the cursor with the inputs of a node that matter (see QUICK_INPUTS; other nodes show
 *  all their plain inputs), or with chosen inputs of several nodes (`fields`, e.g. a texture's
 *  amplitude and scale). A linked input shows where it comes from: click it to edit that node instead
 *  (‹ goes back). Enter commits the field and closes, Esc or a click outside closes without the
 *  field's unfinished typing; values already committed stay. */
export function QuickEdit({ state, onClose }: { state: QuickEditState; onClose: () => void }) {
  const { x, y } = state
  /** the boxes followed into (the first is what was asked for) */
  const [trail, setTrail] = useState<{ node: string; fields?: QuickField[]; title?: string }[]>(
    () => [{ node: state.node, fields: state.fields, title: state.title }])
  const current = trail[trail.length - 1]
  const nodes = useStore((s) => s.nodes)
  const specs = useStore((s) => s.specs)
  const edges = useStore((s) => s.edges)
  const ref = useRef<HTMLDivElement>(null)
  const closing = useRef(false)
  const main = nodes.find((m) => m.id === current.node)
  const gone = !main || !specs[main.data.typeId]

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

  // The first field that can be edited gets the focus, its text selected (again after following a link).
  useEffect(() => {
    const first = ref.current?.querySelector<HTMLInputElement | HTMLSelectElement>('input, select')
    first?.focus({ preventScroll: true })
    if (first instanceof HTMLInputElement) first.select()
  }, [trail.length])

  // A click anywhere else (the 3D view still gets it) or a resize closes the box.
  useEffect(() => {
    const outside = (e: Event) => {
      const t = e.target as Node | null
      // (a field's list, PickLists, is drawn on top of the page: a click in it is still in the box)
      if (!ref.current?.contains(t) && !(t instanceof Element && t.closest('.pick-list'))) cancel()
    }
    window.addEventListener('pointerdown', outside, true)
    window.addEventListener('resize', cancel)
    return () => {
      window.removeEventListener('pointerdown', outside, true)
      window.removeEventListener('resize', cancel)
    }
  }, [onClose]) // eslint-disable-line react-hooks/exhaustive-deps

  if (gone) return null
  const done = () => {
    closing.current = true
    onClose()
  }
  const linkOf = (node: string, port: string) => edges.find((e) => e.target === node && e.targetHandle === port)
  /** the node's list of bodies, for a body list (Port pick "body"): its first cad input */
  const listInput = (node: string) => {
    const n = nodes.find((m) => m.id === node)
    return n ? specs[n.data.typeId]?.inputs.find((q) => q.type === 'cad')?.name : undefined
  }
  const rows: { node: string; port: PortSpec; label: string }[] = current.fields
    ? current.fields.flatMap((f) => {
      const n = nodes.find((m) => m.id === f.node)
      const p = n && specs[n.data.typeId]?.inputs.find((q) => q.name === f.port)
      return p ? [{ node: f.node, port: p, label: f.label ?? p.label ?? p.name }] : []
    })
    : portsOf(specs[main.data.typeId], main.data.typeId, (p) => !!linkOf(main.id, p), (p) => main.data.inputs[p])
      .map((p) => ({ node: main.id, port: p, label: p.label || p.name }))

  return createPortal(
    <div ref={ref} className="vp-quick" role="dialog" aria-label={`Quick edit ${current.node}`}
      // events of a portal bubble through the React tree: keep them away from the 3D view's handlers
      onKeyDown={(e) => { stop(e); if (e.key === 'Escape') { e.preventDefault(); cancel() } }}
      onKeyUp={stop} onPointerDown={stop} onMouseDown={stop} onClick={stop} onDoubleClick={stop}
      onContextMenu={stop} onWheel={stop}>
      <div className="vp-quick-head">
        {trail.length > 1 && (
          <button className="vp-quick-back" title={`Back to ${trail[trail.length - 2].node}`}
            onClick={() => setTrail((t) => t.slice(0, -1))}>‹</button>
        )}
        <b>{current.title ?? current.node}</b>
        {!current.title && <span className="vp-quick-title"> · {specs[main.data.typeId].title}</span>}
        <button className="vp-quick-graph" title="Show the node in the graph"
          onClick={() => useStore.getState().focusNode(current.node)}>⌖</button>
      </div>
      <div className="vp-quick-body">
        {!rows.length && <div className="vp-quick-row muted">Nothing to set here</div>}
        {rows.map(({ node, port: p, label }) => {
          const n = nodes.find((m) => m.id === node)!
          const spec = specs[n.data.typeId]
          const get: Get = (port) => n.data.inputs[port] ?? spec.inputs.find((q) => q.name === port)?.default
          // the node's other values (a linked one may be anything)
          const other = (q: string) => (linkOf(node, q) ? undefined : get(q))
          // inputs that do not apply with them
          if (!linkOf(node, p.name) && !portApplies(p, other)) return null
          const key = `${node}.${p.name}`
          const link = linkOf(node, p.name)
          if (link) {
            const from = nodes.find((m) => m.id === link.source)
            return (
              <div className="vp-quick-row" key={key}>
                <span className="vp-quick-label" title={label}>{label}</span>
                {from && from.type === 'engine'
                  ? <button className="vp-quick-link" title={`Edit ${link.source}, where ${label} comes from`}
                      onClick={() => setTrail((t) => [...t, { node: link.source }])}>← {link.source} ▸</button>
                  : <span className="vp-quick-linked">linked</span>}
              </div>
            )
          }
          const props: FieldProps = {
            value: get(p.name),
            commit: (v) => useStore.getState().setInput(node, p.name, v),
            done, closing,
          }
          const mode = { live: false, onDone: done, closing, inputClass: 'vp-quick-field' }
          return (
            <Row key={key} label={label} title={p.doc}>
              {p.pick === 'body' && listInput(node)
                ? <BodyPick nodeId={node} input={listInput(node)!} port={p} value={props.value} onCommit={props.commit} />
                : p.switch ? <IntSwitch value={props.value ?? p.default} onCommit={props.commit} />
                : p.type === 'int' && p.name === 'seed' ? <SeedInput value={props.value ?? p.default} onCommit={props.commit} />
                : p.choices?.length
                ? <ChoiceInput port={p} value={props.value} onCommit={props.commit} onDone={done} selectClass="vp-quick-field" />
                : NUMERIC.includes(p.type)
                  ? <NumberInput value={props.value} integer={p.type === 'int'} unit={unitOf(p, other)} zero={p.zero}
                      slider={p.slider} onCommit={props.commit} {...mode} />
                  : isVec3(p) && parseVec3(props.value)
                    ? <Vec3Input value={props.value} onCommit={props.commit} {...mode} />
                    : <TextField {...props} parse={(raw) => raw} />}
            </Row>
          )
        })}
      </div>
      <div className="vp-quick-foot">Enter ✓ · Esc</div>
    </div>,
    document.body,
  )
}

/** A small box at the cursor asking for a name (Save as selection…): Enter takes it, Esc or a click
 *  outside drops it. */
export function NamePrompt({ prompt, onClose }: { prompt: PromptState; onClose: () => void }) {
  const [text, setText] = useState(prompt.value)
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const clamp = (v: number, size: number, room: number) => Math.max(MARGIN, Math.min(v, room - size - MARGIN))
    el.style.left = `${clamp(prompt.x + OFFSET, el.offsetWidth, window.innerWidth)}px`
    el.style.top = `${clamp(prompt.y + OFFSET, el.offsetHeight, window.innerHeight)}px`
  }, [prompt])
  useEffect(() => {
    const outside = (e: Event) => { if (!ref.current?.contains(e.target as Node | null)) onClose() }
    window.addEventListener('pointerdown', outside, true)
    return () => window.removeEventListener('pointerdown', outside, true)
  }, [onClose])
  return createPortal(
    <div ref={ref} className="vp-quick" role="dialog" aria-label={prompt.label}
      onKeyDown={(e) => {
        stop(e)
        if (e.key === 'Escape') { e.preventDefault(); onClose() }
        if (e.key === 'Enter' && text.trim()) { onClose(); prompt.done(text.trim()) }
      }}
      onKeyUp={stop} onPointerDown={stop} onMouseDown={stop} onClick={stop} onContextMenu={stop} onWheel={stop}>
      <div className="vp-quick-head"><b>{prompt.label}</b></div>
      <input className="vp-quick-field vp-quick-name" autoFocus spellCheck={false} value={text}
        onChange={(e) => setText(e.target.value)} />
      <div className="vp-quick-foot">Enter ✓ · Esc</div>
    </div>,
    document.body,
  )
}
