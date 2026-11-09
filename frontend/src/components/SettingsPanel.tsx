import { useEffect, useMemo, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { startValueDrag, useStore } from '../store'
import { GROUP_PREFIX, SETTING_TYPE } from '../types'

/** One setting: a Number node's value, or a group node's setting (a Number node inside it). */
interface Setting { node: string; port: string; label: string; value: number }

/** The slider's span for a value: 0 to twice it (a negative value: twice it to 0; 0: 0 to 1). */
function spanFor(v: number): [number, number] {
  if (v > 0) return [0, v * 2]
  if (v < 0) return [v * 2, 0]
  return [0, 1]
}

/** About 200 steps across the span, a round size (0.01, 0.05, 0.1...). */
function stepFor([lo, hi]: [number, number]) {
  const raw = (hi - lo) / 200
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)))
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? p * 10
}

const fmt = (v: number) => String(Number(v.toPrecision(4)))

function SettingRow({ s }: { s: Setting }) {
  const [span, setSpan] = useState<[number, number]>(() => spanFor(s.value))
  const [text, setText] = useState(fmt(s.value))
  // a value typed outside the slider's span widens it; one typed elsewhere shows here
  useEffect(() => {
    if (s.value < span[0] || s.value > span[1]) setSpan(spanFor(s.value))
    if (Number(text.replace(',', '.')) !== s.value) setText(fmt(s.value))
  }, [s.value]) // eslint-disable-line react-hooks/exhaustive-deps
  const set = (v: number) => useStore.getState().setInput(s.node, s.port, v)
  return (
    <div className="slice-row setting-row">
      <button className="setting-name" title={`Show ${s.node} in the graph`} onClick={() => useStore.getState().focusNode(s.node)}>
        {s.label}
      </button>
      <input type="range" min={span[0]} max={span[1]} step={stepFor(span)} value={s.value}
        onPointerDown={(e) => { if (e.button === 0) startValueDrag() }}
        onChange={(e) => { setText(fmt(Number(e.target.value))); set(Number(e.target.value)) }}
        onDoubleClick={() => setSpan(spanFor(s.value))} title="Drag to change (double-click: span around the value)" />
      <input className="value-input setting-value" value={text} inputMode="decimal"
        onChange={(e) => {
          setText(e.target.value)
          const v = Number(e.target.value.replace(',', '.'))
          if (e.target.value.trim() !== '' && Number.isFinite(v)) set(v)
        }}
        onKeyDown={(e) => e.stopPropagation()} />
    </div>
  )
}

/** The recipe's settings as sliders: every Number node and every group's settings, so a design can be
 *  tuned while looking at it (each change previews at Draft, then refines). */
export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const { nodes, edges, specs } = useStore(useShallow((s) => ({ nodes: s.nodes, edges: s.edges, specs: s.specs })))
  const settings = useMemo(() => {
    const out: Setting[] = []
    const linked = (node: string, port: string) => edges.some((e) => e.target === node && e.targetHandle === port)
    for (const n of nodes) {
      if (n.type !== 'engine') continue
      if (n.data.typeId === SETTING_TYPE && !linked(n.id, 'value')) {
        out.push({ node: n.id, port: 'value', label: n.id, value: Number(n.data.inputs.value ?? 0) })
      } else if (n.data.typeId.startsWith(GROUP_PREFIX)) {
        for (const p of specs[n.data.typeId]?.inputs ?? []) {
          if (p.type === 'number' && p.doc?.startsWith('Setting:') && !linked(n.id, p.name)) {
            out.push({ node: n.id, port: p.name, label: `${n.id} · ${p.name}`, value: Number(n.data.inputs[p.name] ?? p.default ?? 0) })
          }
        }
      }
    }
    return out
  }, [nodes, edges, specs])
  return (
    <div className="tool-panel settings-panel">
      <div className="slice-row">
        <span className="slice-title">Settings</span>
        <button className="small-btn" onClick={onClose} title="Close">×</button>
      </div>
      {!settings.length && <div className="muted small">No Number nodes in this recipe. A Number node's value shows here as a slider.</div>}
      {settings.map((s) => <SettingRow key={`${s.node}.${s.port}`} s={s} />)}
    </div>
  )
}

/** Whether the recipe has settings for the panel. */
export function useHasSettings() {
  return useStore((s) => s.nodes.some((n) => n.data.typeId === SETTING_TYPE ||
    (n.data.typeId.startsWith(GROUP_PREFIX) && !!s.specs[n.data.typeId]?.inputs.some((p) => p.doc?.startsWith('Setting:')))))
}
