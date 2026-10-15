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
  return (
    <div className="help-port">
      <div className="help-port-head">
        <span className="help-port-name">{port.label}</span>
        <span className="type-tag" style={{ color: TYPE_COLORS[port.type] }}>{port.type}</span>
        {port.optional && <span className="help-muted">optional</span>}
        {def !== null && <span className="help-muted">default {def}</span>}
      </div>
      {port.doc && <div className="help-port-doc">{port.doc}</div>}
    </div>
  )
}

/** What the selected node does, and every input and output explained. */
export function HelpPanel() {
  const selected = useStore((s) => s.nodes.filter((n) => n.selected))
  const spec = useStore((s) => (selected.length === 1 ? s.specs[selected[0].data.typeId] : undefined))
  const [open, setOpen] = useState(readOpen)
  const toggle = (next: boolean) => {
    setOpen(next)
    try { localStorage.setItem('helpPanel', next ? 'open' : 'closed') } catch { /* storage unavailable */ }
  }
  if (!spec) return null
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
