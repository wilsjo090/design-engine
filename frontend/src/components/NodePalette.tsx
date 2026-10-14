import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import type { NodeSpec } from '../types'
import { CATEGORY_COLORS, GROUP_PREFIX } from '../types'

export interface PaletteFilter {
  /** dragging from an output of this type: only nodes with a matching input */
  fromOutput?: string
  /** dragging from an input of this type: only nodes with a matching output */
  fromInput?: string
}

export function compatibleInput(spec: NodeSpec, outType: string, compat: Record<string, string[]>) {
  return spec.inputs.find((p) => (compat[p.type] ?? [p.type]).includes(outType))
}

export function compatibleOutput(spec: NodeSpec, inType: string, compat: Record<string, string[]>) {
  return spec.outputs.find((o) => (compat[inType] ?? [inType]).includes(o.type))
}

interface Item {
  key: string
  title: string
  category: string
  doc: string
  type?: string    // a node type
  preset?: string  // a preset name
}

export function NodePalette({ x, y, filter, onPick, onPickPreset, onClose }: {
  x: number; y: number; filter?: PaletteFilter; onPick: (typeId: string) => void; onPickPreset: (name: string) => void
  onClose: () => void
}) {
  const catalog = useStore((s) => s.catalog)
  const specs = useStore((s) => s.specs)
  const editingGroup = useStore((s) => s.editing?.name)
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const [presets, setPresets] = useState<{ name: string; nodes: number; doc: string }[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  useEffect(() => inputRef.current?.focus(), [])
  useEffect(() => { if (!filter) api.presets().then(setPresets).catch(() => {}) }, [filter])

  const items = useMemo(() => {
    if (!catalog) return []
    const q = query.trim().toLowerCase()
    const match = (text: string) => !q || text.toLowerCase().includes(q)
    // catalog order (by category), then your groups, then presets
    const nodeSpecs = [...catalog.nodes, ...Object.values(specs).filter((s) => s.type.startsWith(GROUP_PREFIX))]
    const out: Item[] = nodeSpecs.filter((n) => {
      if (editingGroup && n.type === GROUP_PREFIX + editingGroup) return false  // a group cannot contain itself
      if (filter?.fromOutput && !compatibleInput(n, filter.fromOutput, catalog.compatibility)) return false
      if (filter?.fromInput && !compatibleOutput(n, filter.fromInput, catalog.compatibility)) return false
      return match(`${n.title} ${n.category} ${n.type} ${n.doc}`)
    }).map((n) => ({ key: n.type, title: n.title, category: n.category, doc: n.doc, type: n.type }))
    if (!filter) {
      for (const p of presets) {
        if (match(`${p.name} preset ${p.doc}`)) {
          out.push({ key: `preset:${p.name}`, title: p.name, category: 'Presets', preset: p.name,
            doc: p.doc || `${p.nodes} node${p.nodes === 1 ? '' : 's'}` })
        }
      }
    }
    return out
  }, [catalog, specs, query, filter, presets, editingGroup])

  useEffect(() => setIndex(0), [query])

  // keep the popup on screen
  const left = Math.min(x, window.innerWidth - 300)
  const top = Math.min(y, window.innerHeight - 420)
  let lastCategory = ''
  const pick = (item: Item) => (item.preset ? onPickPreset(item.preset) : onPick(item.type!))

  return (
    <div className="palette-backdrop" onMouseDown={onClose} onContextMenu={(e) => { e.preventDefault(); onClose() }}>
      <div className="palette" style={{ left, top }} onMouseDown={(e) => e.stopPropagation()}>
        <input ref={inputRef} value={query} placeholder="Search nodes and presets…" onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') onClose()
            else if (e.key === 'ArrowDown') { e.preventDefault(); setIndex((i) => Math.min(i + 1, items.length - 1)) }
            else if (e.key === 'ArrowUp') { e.preventDefault(); setIndex((i) => Math.max(i - 1, 0)) }
            else if (e.key === 'Enter' && items[index]) pick(items[index])
          }} />
        <div className="palette-list">
          {items.map((n, i) => {
            const header = n.category !== lastCategory
            lastCategory = n.category
            return (
              <div key={n.key}>
                {header && <div className="palette-cat" style={{ color: CATEGORY_COLORS[n.category] }}>{n.category}</div>}
                <div className={`palette-item ${i === index ? 'active' : ''}`} title={n.doc}
                  onMouseEnter={() => setIndex(i)} onClick={() => pick(n)}>
                  <span className="palette-title">
                    {n.title}
                    {n.preset && (
                      <button className="palette-del" title={`Delete the preset "${n.preset}"`}
                        onClick={async (e) => {
                          e.stopPropagation()
                          if (!confirm(`Delete the preset "${n.preset}"?`)) return
                          await api.deletePreset(n.preset!)
                          setPresets((p) => p.filter((x) => x.name !== n.preset))
                        }}>×</button>
                    )}
                  </span>
                  <span className="palette-doc">{n.doc}</span>
                </div>
              </div>
            )
          })}
          {items.length === 0 && <div className="palette-empty">No matching nodes</div>}
        </div>
      </div>
    </div>
  )
}
