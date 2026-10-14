import { memo, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useShallow } from 'zustand/react/shallow'
import { isFieldNode, useStore } from '../store'
import { useBrowser } from './FileBrowser'
import type { EngineNode, PortSpec } from '../types'
import { CATEGORY_COLORS, DISPLAYABLE, GROUP_PREFIX, TYPE_COLORS, parseFaceKeys, resolveFaces } from '../types'

const fmt = (v: unknown) => (v === undefined || v === null ? '' : String(v))

/** 0.6000000238 -> "0.6", 3.14159 -> "3.14" */
const fmtNum = (v: number) => String(Number(v.toPrecision(3)))
const fmtRange = ([lo, hi]: [number, number]) => (lo === hi ? fmtNum(lo) : `${fmtNum(lo)} … ${fmtNum(hi)}`)

/** Remap with Auto on ignores its typed From Low / From High */
const ignoredByAuto = (data: EngineNode['data'], port: string) =>
  data.typeId === 'math.remap' && data.inputs.auto === 'yes' && (port === 'from_low' || port === 'from_high')

const NUMERIC = ['number', 'scalar', 'int']

/** Number field: accepts "0,35" as well as "0.35"; arrow keys nudge (Shift = x10). */
function NumberInput({ value, integer, onCommit }: { value: unknown; integer?: boolean; onCommit: (v: number) => void }) {
  const [text, setText] = useState(fmt(value))
  useEffect(() => {
    // Don't overwrite while mid-typing (e.g. "0." or "0,3") a text that already means this value.
    if (Number(text.replace(',', '.')) !== value) setText(fmt(value))
  }, [value]) // eslint-disable-line react-hooks/exhaustive-deps
  const commit = (raw: string) => {
    const v = Number(raw.replace(',', '.'))
    if (raw.trim() !== '' && Number.isFinite(v)) onCommit(integer ? Math.round(v) : v)
  }
  const nudge = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
    e.preventDefault()
    const current = Number(text.replace(',', '.')) || 0
    const decimals = (text.split(/[.,]/)[1] ?? '').length
    let step = integer ? 1 : Math.pow(10, -Math.max(decimals, 1))
    if (e.shiftKey) step *= 10
    const next = +(current + (e.key === 'ArrowUp' ? step : -step)).toFixed(integer ? 0 : Math.max(decimals, 1))
    setText(String(next))
    onCommit(next)
  }
  return (
    <input className="nodrag nowheel value-input" value={text} inputMode="decimal"
      onChange={(e) => { setText(e.target.value); commit(e.target.value) }} onKeyDown={nudge} />
  )
}

/** Step per few pixels when dragging a value: about 1% of its size (140 -> 1, 0.63 -> 0.001). */
function scrubStep(v: number, integer?: boolean) {
  if (integer) return 1
  return Math.pow(10, Math.floor(Math.log10(Math.abs(v) || 1)) - 2)
}

/** A port label you can drag left/right to change its number (Shift = 10x faster, Alt = finer). */
function ScrubLabel({ label, value, integer, onChange }: {
  label: string; value: unknown; integer?: boolean; onChange: (v: number) => void
}) {
  const start = useRef<{ x: number; v: number; step: number } | null>(null)
  const down = (e: PointerEvent<HTMLSpanElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    const v = Number(value) || 0
    start.current = { x: e.clientX, v, step: scrubStep(v, integer) }
    document.body.classList.add('scrubbing')
  }
  const move = (e: PointerEvent<HTMLSpanElement>) => {
    const s = start.current
    if (!s) return
    const factor = e.shiftKey ? 10 : e.altKey ? 0.1 : 1
    const steps = Math.round((e.clientX - s.x) / (integer ? 8 : 3))
    const step = s.step * factor
    const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9))
    const next = Number((s.v + steps * step).toFixed(decimals))
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

function TextInput({ value, onCommit, placeholder }: { value: unknown; onCommit: (v: string) => void; placeholder?: string }) {
  const [text, setText] = useState(fmt(value))
  useEffect(() => setText(fmt(value)), [value])
  return (
    <input className="nodrag nowheel value-input" value={text} placeholder={placeholder}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => text !== fmt(value) && onCommit(text)}
      onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
  )
}

/** Face numbers (editable as text, or the name of a saved selection), a button that starts picking
 *  them in the 3D view on the body connected to the node's `cad` input, Clear, and saved selections. */
function FacesInput({ nodeId, value, onCommit }: { nodeId: string; value: unknown; onCommit: (v: string) => void }) {
  const source = useStore((s) => s.edges.find((e) => e.target === nodeId && e.targetHandle === 'cad')?.source)
  const picking = useStore((s) => s.picking?.node === nodeId)
  const setPicking = useStore((s) => s.setPicking)
  const selections = useStore((s) => s.selections)
  const saveSelection = useStore((s) => s.saveSelection)
  const deleteSelection = useStore((s) => s.deleteSelection)
  const editing = useStore((s) => !!s.editing)
  const leaveOut = useStore((s) => s.nodes.find((n) => n.id === nodeId)?.data.inputs.apply === 'all other faces')
  const [menu, setMenu] = useState(false)
  const keys = parseFaceKeys(resolveFaces(value, selections))
  const saveAs = () => {
    const name = prompt('Name for these faces (e.g. top, rim). Other Face Masks can then use the name, and picking ' +
      'with the name chosen updates it everywhere.')?.trim()
    if (!name) return
    if (!/^[A-Za-z_][\w-]*$/.test(name)) { alert('A name starts with a letter and has no spaces.'); return }
    saveSelection(name, keys.join(', '))
    onCommit(name)
    setMenu(false)
  }
  return (
    <div className="path-input faces-input">
      <TextInput value={value} onCommit={onCommit} placeholder={leaveOut ? 'none left out' : 'no faces'} />
      <button className={`nodrag browse-btn pick-btn ${picking ? 'on' : ''}`} disabled={!source || editing}
        title={!source ? 'Connect a body first' : editing ? 'Pick faces outside the group'
          : picking ? 'Stop picking (Esc)'
          : leaveOut ? `Pick the faces to leave out in the 3D view (${keys.length} left out)`
          : `Pick faces in the 3D view (${keys.length} chosen)`}
        onClick={() => setPicking(picking ? null : { node: nodeId, source: source! })}>
        {picking ? 'Done' : 'Pick'}
      </button>
      <button className="nodrag browse-btn" disabled={!fmt(value)} title="Clear: no faces chosen"
        onClick={() => onCommit('')}>✕</button>
      <button className={`nodrag browse-btn ${menu ? 'on' : ''}`} title="Saved selections" onClick={() => setMenu((m) => !m)}>☰</button>
      {menu && (
        <div className="faces-menu nodrag nowheel" onMouseLeave={() => setMenu(false)}>
          {Object.entries(selections).map(([name, faces]) => (
            <div className="faces-menu-row" key={name}>
              <button className="faces-menu-name" title={`Use "${name}": ${faces}`}
                onClick={() => { onCommit(name); setMenu(false) }}>{name}</button>
              <span className="faces-menu-count">{parseFaceKeys(faces).length} faces</span>
              <button className="faces-menu-del" title={`Forget "${name}"`} onClick={() => deleteSelection(name)}>×</button>
            </div>
          ))}
          {!Object.keys(selections).length && <div className="faces-menu-empty">No saved selections yet</div>}
          <button className="faces-menu-save" disabled={!keys.length} onClick={saveAs}>
            Save these {keys.length} faces as…
          </button>
        </div>
      )}
    </div>
  )
}

const PATH_EXTS: Record<string, { exts: string[]; mode: 'open' | 'save' }> = {
  'input.import_step': { exts: ['.step', '.stp'], mode: 'open' },
  'output.export': { exts: ['.stl', '.3mf', '.obj', '.ply'], mode: 'save' },
  'output.preview': { exts: ['.png'], mode: 'save' },
}

function PortEditor({ nodeId, typeId, port, value }: { nodeId: string; typeId: string; port: PortSpec; value: unknown }) {
  const setInput = useStore((s) => s.setInput)
  const openBrowser = useBrowser((s) => s.open)
  const commit = (v: unknown) => setInput(nodeId, port.name, v)
  const shown = value ?? port.default

  if (port.choices) {
    return (
      <select className="nodrag value-input" value={fmt(shown)} onChange={(e) => commit(e.target.value)}>
        {port.choices.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
    )
  }
  switch (port.type) {
    case 'number':
    case 'scalar':
      return <NumberInput value={shown} onCommit={commit} />
    case 'int':
      return <NumberInput value={shown} integer onCommit={commit} />
    case 'text':
      return <TextInput value={shown} onCommit={commit} />
    case 'faces':
      return <FacesInput nodeId={nodeId} value={shown} onCommit={commit} />
    case 'path': {
      const cfg = PATH_EXTS[typeId] ?? { exts: [], mode: 'open' as const }
      return (
        <div className="path-input">
          <TextInput value={shown} onCommit={commit} placeholder={cfg.mode === 'open' ? 'choose a file…' : 'output file'} />
          <button className="nodrag browse-btn" title="Browse"
            onClick={async () => {
              const p = await openBrowser({ mode: cfg.mode, exts: cfg.exts, initial: fmt(shown),
                title: cfg.mode === 'open' ? 'Open file' : 'Save as' })
              if (p) commit(p)
            }}>…</button>
        </div>
      )
    }
    default:
      return <span className="placeholder">{port.optional ? 'Optional' : 'connect…'}</span>
  }
}

function EyeIcon({ open }: { open: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" />
      {open ? <circle cx="8" cy="8" r="2.1" fill="currentColor" /> : <path d="M3 13L13 3" />}
    </svg>
  )
}

function EngineNodeView({ id, data, selected }: NodeProps<EngineNode>) {
  const spec = useStore((s) => s.specs[data.typeId])
  const links = useStore(useShallow((s) =>
    s.edges.filter((e) => e.target === id).map((e) => `${e.targetHandle}|${e.source}`)))
  const status = useStore((s) => s.status[id])
  const problems = useStore((s) => s.problems[id])
  const visible = useStore((s) => s.visible.includes(id))
  const toggleVisible = useStore((s) => s.toggleVisible)
  const setInput = useStore((s) => s.setInput)
  const editing = useStore((s) => !!s.editing)

  if (!spec) return <div className="enode enode-error">Unknown node type “{data.typeId}”</div>

  const linked = Object.fromEntries(links.map((l) => l.split('|')))
  const color = CATEGORY_COLORS[spec.category] ?? '#888'
  const isField = isFieldNode(spec)
  const canShow = !editing && (isField || spec.outputs.some((o) => DISPLAYABLE.includes(o.type)))
  const error = status?.state === 'error' ? status.message : undefined
  const messages = [...(error ? [error] : []), ...(problems ?? []).filter((p) => p !== error)]
  const group = data.typeId.startsWith(GROUP_PREFIX) ? data.typeId.slice(GROUP_PREFIX.length) : null

  return (
    <div className={`enode ${selected ? 'selected' : ''} ${messages.length ? 'has-error' : ''} ${group ? 'group' : ''}`}
      style={{ ['--cat' as string]: color }} title={spec.doc}
      onDoubleClick={group && !editing ? () => useStore.getState().editGroup(group) : undefined}>
      <div className="enode-header">
        <span className={`status-dot ${status?.state ?? 'idle'}`} />
        <span className="enode-title">{spec.title}</span>
        <span className="enode-id">{id}</span>
        {group && !editing && (
          <>
            <button className="nodrag mini-btn" title="Open the group to change its nodes (every copy follows; also double-click)"
              onClick={() => useStore.getState().editGroup(group)}>Edit</button>
            <button className="nodrag mini-btn" title="Put the group's nodes back in place of this node"
              onClick={() => useStore.getState().ungroup(id)}>Ungroup</button>
          </>
        )}
        {canShow && (
          <button className={`eye nodrag ${visible ? 'shown' : ''}`}
            title={isField ? (visible ? 'Hide the field' : 'Show this field in the 3D view (as a slice, or as colours on the bodies)')
              : (visible ? 'Hide in 3D view' : 'Show in 3D view')}
            onClick={() => toggleVisible(id)}>
            <EyeIcon open={visible} />
          </button>
        )}
      </div>

      {spec.outputs.map((o, i) => (
        <div className="enode-row output" key={o.name} title={o.doc}>
          {i === 0 && status?.range && (
            <span className="range-tag" title="The values this output can take (Remap with Auto uses this)">
              {fmtRange(status.range)}
            </span>
          )}
          <span className="port-label out">{o.label}</span>
          <span className="type-tag" style={{ color: TYPE_COLORS[o.type] }}>{o.type}</span>
          <Handle type="source" position={Position.Right} id={o.name} className="handle"
            style={{ background: TYPE_COLORS[o.type] }} />
        </div>
      ))}

      {spec.inputs.map((p) => {
        const value = data.inputs[p.name] ?? p.default
        const scrubbable = !linked[p.name] && NUMERIC.includes(p.type) && !p.choices && !ignoredByAuto(data, p.name)
        return (
          <div className={`enode-row ${ignoredByAuto(data, p.name) ? 'ignored' : ''}`} key={p.name}
            title={ignoredByAuto(data, p.name) ? 'Ignored: Auto takes this from X\'s range' : p.doc}>
            <Handle type="target" position={Position.Left} id={p.name} className="handle"
              style={{ background: TYPE_COLORS[p.type] }} />
            {scrubbable
              ? <ScrubLabel label={p.label} value={value} integer={p.type === 'int'}
                  onChange={(v) => setInput(id, p.name, v)} />
              : <span className="port-label">{p.label}</span>}
            {linked[p.name]
              ? <span className="linked">← {linked[p.name]}</span>
              : <PortEditor nodeId={id} typeId={data.typeId} port={p} value={data.inputs[p.name]} />}
          </div>
        )
      })}

      {status?.state === 'running' && status.progress && (
        <div className="progress"><div style={{ width: `${(100 * status.progress[0]) / status.progress[1]}%` }} /></div>
      )}
      {(status?.state === 'running' || (status?.time ?? 0) >= 0.05) && (
        <div className="enode-footer">
          {status?.state === 'running'
            ? status.progress ? `${status.progress[0]} / ${status.progress[1]}` : 'working…'
            : `${status!.time!.toFixed(1)} s`}
        </div>
      )}
      {messages.map((m, i) => <div className="enode-message" key={i}>{m}</div>)}
    </div>
  )
}

export const EngineNodeComponent = memo(EngineNodeView)

// --- the input and output nodes shown while a group is open ---------------------------------
const PORT_TYPES = ['scalar', 'implicit', 'cad', 'mesh', 'number']

function AddPort({ side }: { side: 'in' | 'out' }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [type, setType] = useState(side === 'in' ? 'implicit' : 'implicit')
  const add = () => {
    const error = useStore.getState().addGroupPort(side, name.trim(), type)
    if (error) { alert(error); return }
    setName('')
    setOpen(false)
  }
  if (!open) {
    return <button className="nodrag add-port" onClick={() => setOpen(true)}>+ {side === 'in' ? 'input' : 'output'}</button>
  }
  return (
    <div className="add-port-form nodrag">
      <input className="value-input nowheel" value={name} placeholder="name" autoFocus
        onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && add()} />
      <select className="value-input" value={type} onChange={(e) => setType(e.target.value)}>
        {PORT_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
      <button onClick={add}>Add</button>
    </div>
  )
}

function GroupIONodeView({ type }: NodeProps<EngineNode>) {
  const side = type === 'groupIn' ? 'in' : 'out'
  const def = useStore((s) => (s.editing ? s.groups[s.editing.name] : null))
  if (!def) return null
  const ports = side === 'in' ? def.inputs : def.outputs
  return (
    <div className="enode group-io" style={{ ['--cat' as string]: CATEGORY_COLORS.Groups }}>
      <div className="enode-header">
        <span className="enode-title">{side === 'in' ? 'Group inputs' : 'Group outputs'}</span>
      </div>
      {ports.map((p) => (
        <div className={`enode-row ${side === 'in' ? 'output' : ''}`} key={p.name}>
          {side === 'out' && <Handle type="target" position={Position.Left} id={p.name} className="handle"
            style={{ background: TYPE_COLORS[p.type] }} />}
          <span className={`port-label ${side === 'in' ? 'out' : ''}`}>{p.name}</span>
          <span className="type-tag" style={{ color: TYPE_COLORS[p.type] }}>{p.type}</span>
          {side === 'in' && <Handle type="source" position={Position.Right} id={p.name} className="handle"
            style={{ background: TYPE_COLORS[p.type] }} />}
        </div>
      ))}
      {side === 'in' && <div className="group-io-note">Number nodes inside are the group's settings</div>}
      <div className="enode-row"><AddPort side={side} /></div>
    </div>
  )
}

export const GroupIONode = memo(GroupIONodeView)
