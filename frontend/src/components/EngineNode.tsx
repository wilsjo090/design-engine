import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useShallow } from 'zustand/react/shallow'
import { api, startOptimize, stopRun } from '../api'
import { LOAD_CASE, loadCases } from '../loadCases'
import { isFieldNode, useStore } from '../store'
import { useBrowser } from './FileBrowser'
import { PLACE_PORTS } from './Guides'
import { BodyChip, BodyPick, FontPick, IntSwitch, SeedInput } from './PickLists'
import { ChoiceInput, NumberInput, ScrubLabel, Vec3Input, fmt, isVec3, parseVec3 } from './PortInputs'
import type { EngineNode, PortSpec } from '../types'
import {
  CATEGORY_COLORS, DISPLAYABLE, GROUP_PREFIX, SETTING_TYPE, TYPE_COLORS, changedFromDefault, parseFaceKeys, portApplies,
  resolveFaces, unitOf,
} from '../types'
import { hoverWires, pickUpWire, useWires, wireLabels } from '../wires'

/** 0.6000000238 -> "0.6", 3.14159 -> "3.14" */
const fmtNum = (v: number) => String(Number(v.toPrecision(3)))
const fmtRange = ([lo, hi]: [number, number]) => (lo === hi ? fmtNum(lo) : `${fmtNum(lo)} … ${fmtNum(hi)}`)

/** Remap with Auto on ignores its typed From Low / From High */
const ignoredByAuto = (data: EngineNode['data'], port: string) =>
  data.typeId === 'math.remap' && data.inputs.auto === 'yes' && (port === 'from_low' || port === 'from_high')

const NUMERIC = ['number', 'scalar', 'int']

/** How long a node's card keeps its flash class (the animation itself is 1.4 s). */
const FLASH_MS = 1500

/** `end`: show the end of a long text while not typing in it (a path's file name, not its C:\Users\…). */
function TextInput({ value, onCommit, placeholder, end }: {
  value: unknown; onCommit: (v: string) => void; placeholder?: string; end?: boolean
}) {
  const [text, setText] = useState(fmt(value))
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => setText(fmt(value)), [value])
  useEffect(() => {
    const el = ref.current
    if (end && el && document.activeElement !== el) el.scrollLeft = el.scrollWidth
  }, [end, text])
  return (
    <input ref={ref} className="nodrag nowheel value-input" value={text} placeholder={placeholder}
      title={end && text ? text : undefined}
      onChange={(e) => setText(e.target.value)}
      onBlur={(e) => {
        if (text !== fmt(value)) onCommit(text)
        const el = e.currentTarget
        if (end) requestAnimationFrame(() => { el.scrollLeft = el.scrollWidth })
      }}
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
  'pattern.picture': { exts: ['.png', '.jpg', '.jpeg', '.bmp'], mode: 'open' },
}

function PortEditor({ nodeId, typeId, port, value, unit }: {
  nodeId: string; typeId: string; port: PortSpec; value: unknown; unit?: string
}) {
  const setInput = useStore((s) => s.setInput)
  const openBrowser = useBrowser((s) => s.open)
  const listInput = useStore((s) => s.specs[typeId]?.inputs.find((p) => p.type === 'cad')?.name)
  const commit = (v: unknown) => setInput(nodeId, port.name, v)
  const shown = value ?? port.default

  if (port.choices) return <ChoiceInput port={port} value={shown} onCommit={commit} />
  if (port.pick === 'body' && listInput) {
    return <BodyPick nodeId={nodeId} input={listInput} port={port} value={shown} onCommit={commit} />
  }
  if (port.switch) return <IntSwitch value={shown} onCommit={commit} />
  if (port.type === 'int' && port.name === 'seed') return <SeedInput value={shown} onCommit={commit} />
  switch (port.type) {
    case 'number':
    case 'scalar':
    case 'int':
      return <NumberInput value={shown} integer={port.type === 'int'} unit={unit} zero={port.zero} slider={port.slider}
        onCommit={commit} />
    case 'text':
      if (isVec3(port) && parseVec3(shown)) return <Vec3Input value={shown} onCommit={commit} />
      if (port.pick === 'font') {
        return (
          <div className="path-input">
            <TextInput value={shown} onCommit={commit} />
            <FontPick value={shown} onCommit={commit} />
          </div>
        )
      }
      return <TextInput value={shown} onCommit={commit} />
    case 'faces':
      return <FacesInput nodeId={nodeId} value={shown} onCommit={commit} />
    case 'path': {
      const cfg = PATH_EXTS[typeId] ?? { exts: [], mode: 'open' as const }
      return (
        <div className="path-input">
          <TextInput value={shown} onCommit={commit} end placeholder={cfg.mode === 'open' ? 'choose a file…' : 'output file'} />
          <button className="nodrag browse-btn" title="Browse"
            onClick={async () => {
              const p = await openBrowser({ mode: cfg.mode, exts: cfg.exts, initial: fmt(shown),
                title: cfg.mode === 'open' ? 'Open file' : 'Save as' })
              if (p) commit(p)
            }}>…</button>
          <button className="nodrag browse-btn" disabled={!fmt(shown)}
            title={cfg.mode === 'save' ? 'Show the file in Explorer (its folder if it is not written yet)' : 'Show the file in Explorer'}
            onClick={() => api.reveal(fmt(shown)).catch((e: Error) =>
              useStore.getState().addLog(`Cannot show ${fmt(shown)}: ${e.message}`, 'warning'))}>📂</button>
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

/** Long jobs run on request: Optimize (queued like an export) and Stop. Per node type: what the
 *  buttons say they do. */
const OPTIMIZERS: Record<string, { run: string; stop: string }> = {
  'sim.topology': {
    run: 'Find the stiffest shape (takes a while: the shape grows in the 3D view)',
    stop: 'Stop: keeps the shape so far; Optimize continues from it',
  },
  'sim.frame': {
    run: 'Find the lightest frame of bars that carries the loads (takes some seconds)',
    stop: 'Stop the frame optimization',
  },
}

function OptimizeBar({ nodeId, typeId }: { nodeId: string; typeId: string }) {
  // (from its run_start: the nodes before it are read first, and the card must not offer Optimize meanwhile)
  const running = useStore((s) => s.run?.running && s.run.mode === 'optimize' &&
    (s.run.label === `optimize ${nodeId}` || s.status[nodeId]?.state === 'running'))
  const stopping = useStore((s) => !!s.run?.stopping)
  const editing = useStore((s) => !!s.editing)
  const starting = useStore((s) => s.starting === nodeId)
  const finished = useStore((s) => s.status[nodeId]?.finished)
  const text = OPTIMIZERS[typeId]
  return (
    <>
      <div className="enode-row optimize-row">
        {running
          ? <button className="nodrag optimize-btn stop" title={stopping ? 'Stops at the next stopping point' : text.stop}
              disabled={stopping} onClick={() => stopRun()}>{stopping ? 'Stopping…' : 'Stop'}</button>
          : <button className="nodrag optimize-btn" disabled={editing || starting || !!finished}
              title={editing ? 'Optimize outside the group' : starting ? 'Waiting for the update running now to stop'
                : finished ?? text.run}
              onClick={() => startOptimize(nodeId)}>{starting ? 'Starting…' : finished ? 'Finished ✓' : 'Optimize'}</button>}
      </div>
      {finished && <div className="enode-note">{finished}</div>}
    </>
  )
}

/** Ball, Bar and Box: click on a shown body to put the shape there (a Bar: its start, then its end). */
function PlaceBar({ nodeId, typeId }: { nodeId: string; typeId: string }) {
  const placing = useStore((s) => s.placing?.node === nodeId)
  const setPlacing = useStore((s) => s.setPlacing)
  const editing = useStore((s) => !!s.editing)
  const bar = typeId === 'implicit.bar'
  return (
    <div className="enode-row optimize-row">
      <button className={`nodrag optimize-btn ${placing ? 'stop' : ''}`} disabled={editing}
        title={editing ? 'Place shapes outside the group'
          : placing ? 'Stop placing (Esc)'
          : `Click on a shown body in the 3D view to put the ${bar ? "bar's start, then its end" : 'shape'} there (in the ` +
            'middle of the wall under the click). Select the node to drag it by its yellow handles.'}
        onClick={() => {
          if (placing) { setPlacing(null); return }
          const s = useStore.getState()
          if (!s.visible.includes(nodeId)) s.toggleVisible(nodeId)
          setPlacing({ node: nodeId, port: PLACE_PORTS[typeId][0] })
        }}>
        {placing ? 'Done' : bar ? 'Place (2 clicks)' : 'Place'}
      </button>
    </div>
  )
}

/** A wire shown as a label (the graph's Wires menu): hovering it shows the wire, a click goes to its source. */
function WireLabel({ nodeId, edge, source, color }: { nodeId: string; edge: string; source: string; color: string }) {
  return (
    <button className="nodrag wire-label" style={{ ['--wire' as string]: color }}
      title={`Linked from ${source}. The wire shows while either node is hovered or selected; click to go to ${source}.`}
      onMouseEnter={() => hoverWires({ edge }, 0)} onMouseLeave={() => hoverWires({ node: nodeId }, 0)}
      onClick={(e) => { e.stopPropagation(); useStore.getState().focusNode(source) }}>
      ← {source}
    </button>
  )
}

/** The node's id; double-click to rename it (links and the 3D view follow). */
function NodeName({ id }: { id: string }) {
  const [text, setText] = useState<string | null>(null)
  const editing = useStore((s) => !!s.editing)
  const finish = (commit: boolean) => {
    if (commit && text !== null) {
      const error = useStore.getState().renameNode(id, text)
      if (error) useStore.getState().addLog(error, 'error')
    }
    setText(null)
  }
  if (text === null) {
    return (
      <span className="enode-id" title={editing ? id : `${id} (double-click to rename)`}
        onDoubleClick={(e) => { e.stopPropagation(); setText(id) }}>{id}</span>
    )
  }
  return (
    <input className="nodrag value-input rename-input" autoFocus value={text} spellCheck={false}
      onChange={(e) => setText(e.target.value)} onBlur={() => finish(true)}
      onDoubleClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Enter') finish(true); if (e.key === 'Escape') finish(false) }} />
  )
}

function EngineNodeView({ id, data, selected }: NodeProps<EngineNode>) {
  const spec = useStore((s) => s.specs[data.typeId])
  const links = useStore(useShallow((s) =>
    s.edges.filter((e) => e.target === id).map((e) => `${e.targetHandle}|${e.source}`)))
  // outputs with wires (zoomed out, only wired ports keep their dots)
  const usedOutputs = useStore((s) => [...new Set(s.edges.filter((e) => e.source === id).map((e) => e.sourceHandle))].sort().join(' '))
  // its wires shown as labels: per input the wire's id, per output how many
  const [shared, long] = useWires(useShallow((w) => [w.shared, w.long]))
  const labelText = useStore((s) => wireLabels(s.nodes, s.edges, shared, long).of[id] ?? '')
  const labels = useMemo(() => (labelText ? JSON.parse(labelText) as { in: Record<string, string>; out: Record<string, number> }
    : { in: {}, out: {} }), [labelText])
  const status = useStore((s) => s.status[id])
  const problems = useStore((s) => s.problems[id])
  const visible = useStore((s) => s.visible.includes(id))
  const toggleVisible = useStore((s) => s.toggleVisible)
  const setInput = useStore((s) => s.setInput)
  const editing = useStore((s) => !!s.editing)
  // a Load Case node and its loads: "<colour> <name>" (a string, so cards re-render only when it changes)
  const inCase = useStore((s) => {
    const c = loadCases(s.nodes, s.edges).of[id]
    return c ? `${c.color} ${c.name}` : ''
  })

  // A change made from the 3D view flashes the node's card for a moment (a new flash restarts it).
  const flashAt = useStore((s) => (s.flash.ids.includes(id) ? s.flash.at : 0))
  const [flash, setFlash] = useState({ n: 0, on: false })
  const flashTimer = useRef<number>()
  useEffect(() => () => window.clearTimeout(flashTimer.current), [])
  useEffect(() => {
    const left = flashAt + FLASH_MS - Date.now()
    if (!flashAt || left <= 0) return
    setFlash((f) => ({ n: f.n + 1, on: true }))
    window.clearTimeout(flashTimer.current)
    flashTimer.current = window.setTimeout(() => setFlash((f) => ({ ...f, on: false })), left)
  }, [flashAt])

  /** the advanced inputs are shown ("more" row) */
  const [more, setMore] = useState(false)

  if (!spec) return <div className="enode enode-error">Unknown node type “{data.typeId}”</div>

  const linked = Object.fromEntries(links.map((l) => l.split('|')))
  // inputs that do not apply with the others' values are left out (a pattern's Sides for round holes);
  // advanced ones with their default value wait under the "more" row
  const valueOf = (name: string) => (linked[name] ? undefined
    : data.inputs[name] ?? spec.inputs.find((q) => q.name === name)?.default)
  const applying = spec.inputs.filter((p) => !p.hidden && (linked[p.name] || portApplies(p, valueOf)))
  const isTucked = (p: PortSpec) => !!p.advanced && !linked[p.name] && !changedFromDefault(p, data.inputs[p.name])
  const tucked = applying.filter(isTucked).length
  const inputs = more ? applying : applying.filter((p) => !isTucked(p))
  const caseColor = inCase ? inCase.slice(0, inCase.indexOf(' ')) : null
  const caseLabel = inCase ? inCase.slice(inCase.indexOf(' ') + 1) : null
  const color = caseColor ?? CATEGORY_COLORS[spec.category] ?? '#888'
  const isField = isFieldNode(spec)
  const canShow = !editing && (isField || spec.outputs.some((o) => DISPLAYABLE.includes(o.type)))
  const error = status?.state === 'error' ? status.message : undefined
  const messages = [...(error ? [error] : []), ...(problems ?? []).filter((p) => p !== error)]
  // what the node (and the meshing of its body for the 3D view) warned about; stays while it is cached
  const warnings = error ? [] : [...(status?.warnings ?? []), ...(status?.viewWarnings ?? [])]
  const group = data.typeId.startsWith(GROUP_PREFIX) ? data.typeId.slice(GROUP_PREFIX.length) : null

  return (
    <div className={`enode ${selected ? 'selected' : ''} ${messages.length ? 'has-error' : ''} ${group ? 'group' : ''}` +
      `${data.collapsed ? ' collapsed' : ''}${data.bypass ? ' bypassed' : ''}` +
      (flash.on ? ` node-flash node-flash-${flash.n % 2}` : '')}
      style={{ ['--cat' as string]: color }} title={spec.doc}
      onDoubleClick={group && !editing ? () => useStore.getState().editGroup(group) : undefined}>
      <div className="enode-header">
        <button className="nodrag collapse-btn" title={data.collapsed ? 'Show the inputs (H)' : 'Collapse to the header (H)'}
          onClick={() => useStore.getState().setNodeFlags([id], { collapsed: !data.collapsed })}>{data.collapsed ? '▸' : '▾'}</button>
        <span className={`status-dot ${status?.state ?? 'idle'}`} />
        <span className="enode-title">{spec.title}</span>
        <NodeName id={id} />
        {caseLabel && data.typeId !== LOAD_CASE && (
          <span className="case-chip" title={`In load case ${caseLabel}: it acts with that case's other loads only`}>
            {caseLabel}
          </span>
        )}
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
        <div className={`enode-row output ${usedOutputs.split(' ').includes(o.name) ? 'wired' : ''}`} key={o.name} title={o.doc}>
          {i === 0 && status?.range && (
            <span className="range-tag" title="The values this output can take (Remap with Auto uses this)">
              {fmtRange(status.range)}
            </span>
          )}
          <span className="port-label out">{o.label}</span>
          <span className="type-tag" style={{ color: TYPE_COLORS[o.type] }}>{o.type}</span>
          {labels.out[o.name] > 0 && (
            <span className="wire-count" title={`${labels.out[o.name]} of its wires show as labels on the inputs they feed ` +
              '(hover or select this node to see them)'}>{labels.out[o.name]} →</span>
          )}
          <Handle type="source" position={Position.Right} id={o.name} className={`handle ${labels.out[o.name] ? 'labelled' : ''}`}
            style={{ background: TYPE_COLORS[o.type] }} />
        </div>
      ))}

      {inputs.map((p) => {
        const value = data.inputs[p.name] ?? p.default
        const scrubbable = !linked[p.name] && NUMERIC.includes(p.type) && !p.choices && !ignoredByAuto(data, p.name)
        return (
          <div className={`enode-row ${ignoredByAuto(data, p.name) ? 'ignored' : ''} ${linked[p.name] ? 'wired' : ''}`} key={p.name}
            title={ignoredByAuto(data, p.name) ? 'Ignored: Auto takes this from X\'s range' : p.doc}>
            <Handle type="target" position={Position.Left} id={p.name} className={`handle ${labels.in[p.name] ? 'labelled' : ''}`}
              style={{ background: TYPE_COLORS[p.type] }}
              title={linked[p.name] ? 'Drag to move this wire to another input (an output: that output feeds this input; empty space: removes it)' : undefined}
              onPointerDownCapture={linked[p.name] ? (e) => {
                const wire = useStore.getState().edges.find((w) => w.target === id && w.targetHandle === p.name)
                if (wire) pickUpWire(e, wire.id)
              } : undefined} />
            {scrubbable
              ? <ScrubLabel label={p.label} value={value} integer={p.type === 'int'} range={p.slider}
                  onChange={(v) => setInput(id, p.name, v)} />
              : <span className="port-label">{p.label}</span>}
            {linked[p.name]
              ? <>
                  {labels.in[p.name]
                    ? <WireLabel nodeId={id} edge={labels.in[p.name]} source={linked[p.name]} color={TYPE_COLORS[p.type] ?? '#888'} />
                    : <span className="linked">← {linked[p.name]}</span>}
                  {p.type === 'cad' && data.typeId !== 'input.pick_body' && <BodyChip nodeId={id} input={p.name} />}
                </>
              : <PortEditor nodeId={id} typeId={data.typeId} port={p} value={data.inputs[p.name]}
                  unit={unitOf(p, valueOf)} />}
          </div>
        )
      })}
      {tucked > 0 && (
        <div className="enode-row more-row">
          <button className="nodrag more-btn" onClick={() => setMore((m) => !m)}
            title={more ? 'Hide the settings most designs leave as they are' : 'Settings most designs leave as they are'}>
            {more ? '▴ fewer settings' : `▾ ${tucked} more setting${tucked === 1 ? '' : 's'}`}
          </button>
        </div>
      )}

      {OPTIMIZERS[data.typeId] && <OptimizeBar nodeId={id} typeId={data.typeId} />}
      {PLACE_PORTS[data.typeId] && <PlaceBar nodeId={id} typeId={data.typeId} />}
      {status?.note && status.state !== 'running' && <div className="enode-note">{status.note}</div>}
      {warnings.map(([text, level], i) => (
        <div className={`enode-warning ${level}`} key={`w${i}`} title={text}>{text}</div>
      ))}
      {status?.state === 'running' && status.progress && (
        <div className="progress"><div style={{ width: `${(100 * status.progress[0]) / status.progress[1]}%` }} /></div>
      )}
      {(status?.state === 'running' || (status?.time ?? 0) >= 0.05) && (
        <div className="enode-footer">
          {status?.state === 'running'
            ? status.progress ? `${status.step ? `${status.step} ` : ''}${status.progress[0]} / ${status.progress[1]}`
              : status.view ? 'meshing for the view…' : 'working…'
            : `${status!.time!.toFixed(1)} s`}
        </div>
      )}
      {messages.map((m, i) => <div className="enode-message" key={i}>{m}</div>)}
      <FarFace title={data.typeId === SETTING_TYPE ? id : spec.title}
        sub={data.typeId === SETTING_TYPE ? `= ${fmt(data.inputs.value ?? spec.inputs.find((p) => p.name === 'value')?.default)}` : id}
        state={status?.state ?? 'idle'} />
    </div>
  )
}

/** What a card shows zoomed out (NodeEditor's `data-far`): its name, sized to read, over the card, which
 *  keeps its size; a Number node shows its name and value. Only wired ports keep their dots (CSS). */
function FarFace({ title, sub, state }: { title: string; sub: string; state: string }) {
  return (
    <div className="enode-far" aria-hidden>
      <div className="enode-far-title">{title}</div>
      <div className="enode-far-sub"><span className={`status-dot ${state}`} /><span>{sub}</span></div>
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
