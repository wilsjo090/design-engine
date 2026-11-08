import { useState } from 'react'
import { useStore } from '../store'
import type { PortSpec } from '../types'
import { CATEGORY_COLORS, TYPE_COLORS } from '../types'

const readOpen = () => {
  try { return localStorage.getItem('helpPanel') !== 'closed' } catch { return true }
}

const shownDefault = (v: unknown) =>
  v === undefined || v === null || v === '' ? null : typeof v === 'number' ? String(+v.toFixed(6)) : String(v)

function PortHelp({ port }: { port: PortSpec }) {
  const def = shownDefault(port.default)
  const unit = typeof port.unit === 'string' ? ` ${port.unit}` : ''
  return (
    <div className="help-port">
      <div className="help-port-head">
        <span className="help-port-name">{port.label}</span>
        <span className="type-tag" style={{ color: TYPE_COLORS[port.type] }}>{port.type}</span>
        {port.optional && <span className="help-muted">optional</span>}
        {def !== null && <span className="help-muted">default {port.zero && port.default === 0 ? port.zero : `${def}${unit}`}</span>}
        {port.zero && port.default !== 0 && <span className="help-muted">0 = {port.zero}</span>}
      </div>
      {port.doc && <div className="help-port-doc">{port.doc}</div>}
    </div>
  )
}

/** The mouse and keys of the 3D view and the graph. */
const KEYS: [string, [string, string][]][] = [
  ['3D view', [
    ['Right-click', 'actions for what is under the cursor; type to search the menu'],
    ['Click / double-click', 'select / edit the node that made that spot'],
    ['Ctrl+click', 'choose faces; right-click one of them to act on all'],
    ['F · Space · 1–7', 'fit · views menu · front, back, left, right, top, bottom, iso'],
    ['M', 'measure (distance, face, wall thickness, member section)'],
    ['Esc', 'clear chosen faces, stop measuring or picking'],
  ]],
  ['Graph', [
    ['Tab / right-click canvas', 'add a node'],
    ['Right-click node / wire', 'node menu / insert a node on the wire (also double-click it)'],
    ['B · H', 'bypass · collapse the selected nodes'],
    ['Ctrl+F · Ctrl+D · Ctrl+G', 'find · duplicate · group'],
    ['Double-click a name', 'rename the node'],
  ]],
]

function KeysHelp() {
  const [open, setOpen] = useState(false)
  if (!open) return <button className="help-reopen" title="Mouse and keys" onClick={() => setOpen(true)}>?</button>
  return (
    <div className="help-panel">
      <div className="help-head">
        <span className="help-title">Mouse and keys</span>
        <button className="help-close" title="Close" onClick={() => setOpen(false)}>✕</button>
      </div>
      <div className="help-body">
        {KEYS.map(([section, rows]) => (
          <div key={section}>
            <div className="help-section">{section}</div>
            {rows.map(([k, v]) => (
              <div className="help-port" key={k}>
                <div className="help-port-head"><span className="help-port-name">{k}</span></div>
                <div className="help-port-doc">{v}</div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/** What the selected node does, and every input and output explained; with nothing selected, the mouse
 *  and keys. */
export function HelpPanel() {
  const selected = useStore((s) => s.nodes.filter((n) => n.selected))
  const spec = useStore((s) => (selected.length === 1 ? s.specs[selected[0].data.typeId] : undefined))
  const [open, setOpen] = useState(readOpen)
  const toggle = (next: boolean) => {
    setOpen(next)
    try { localStorage.setItem('helpPanel', next ? 'open' : 'closed') } catch { /* storage unavailable */ }
  }
  if (!spec) return selected.length ? null : <KeysHelp />
  if (!open) {
    return <button className="help-reopen" title="Show help for the selected node" onClick={() => toggle(true)}>?</button>
  }
  return (
    <div className="help-panel" style={{ ['--cat' as string]: CATEGORY_COLORS[spec.category] ?? '#888' }}>
      <div className="help-head">
        <span className="help-title">{spec.title}</span>
        <span className="help-muted">{spec.category}</span>
        <button className="help-close" title="Hide help (? brings it back)" onClick={() => toggle(false)}>✕</button>
      </div>
      <div className="help-body">
        {spec.doc && <p className="help-doc">{spec.doc}</p>}
        {spec.inputs.length > 0 && <div className="help-section">Inputs</div>}
        {spec.inputs.filter((p) => !p.hidden).map((p) => <PortHelp key={p.name} port={p} />)}
        {spec.outputs.length > 0 && <div className="help-section">Outputs</div>}
        {spec.outputs.map((p) => <PortHelp key={p.name} port={p} />)}
      </div>
    </div>
  )
}
