import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useReactFlow } from '@xyflow/react'
import { api } from '../api'
import { useStore } from '../store'
import type { NodeSpec, PortSpec } from '../types'
import { CATEGORY_COLORS, GROUP_PREFIX, TYPE_COLORS } from '../types'

/** Drag-and-drop type of a library tile: a node type id, or "preset:<name>". */
export const NODE_DRAG_TYPE = 'application/x-design-engine-node'
const PRESET_PREFIX = 'preset:'
/** The panel shows search results, or every category (else it shows the tiles of one category tab). */
const SEARCH = '?search'
const ALL = 'All'
/** Categories in the order a design is built (others after them, in catalog order). */
const ORDER = ['Inputs', 'Transform', 'Fields', 'Patterns', 'Math', 'Implicit', 'Lattice', 'Mesh', 'Simulation', 'Output']
const rank = (category: string) => (ORDER.includes(category) ? ORDER.indexOf(category) : ORDER.length)
/** Half the width and some of the height of a node card: puts a new node's middle, not its corner, at the spot. */
const CENTRE_SHIFT = { x: 125, y: 40 }

/** Tab glyphs (U+FE0E asks for the plain text form of the gear, not the emoji). */
const GLYPHS: Record<string, string> = {
  All: '☰', Inputs: '⇲', Fields: '∿', Math: '∑', Implicit: '◐', Transform: '⤧', Lattice: '▦', Patterns: '⁘',
  Mesh: '△', Simulation: '⚙︎', Output: '⇪', Groups: '▣', Presets: '★',
}
const glyph = (category: string) => GLYPHS[category] ?? '•'
const categoryColor = (category: string) => (category === ALL ? '#c9ced6' : CATEGORY_COLORS[category] ?? '#8a9099')

interface PresetInfo { name: string; nodes: number; doc: string }

/** One tile: a node type, a group or a preset. */
interface Entry {
  key: string
  title: string
  category: string
  /** the full doc: the tile's tooltip */
  doc: string
  /** the first sentence of the doc */
  summary: string
  /** what a click or a drop adds: a node type id, or "preset:<name>" */
  payload: string
  /** distinct port types of the visible inputs and of the outputs (the chips) */
  ins: string[]
  outs: string[]
  /** a line instead of the chips (a preset's size) */
  meta?: string
  /** lower-case text the search looks in */
  haystack: string
}

/** The first sentence of a doc: up to a full stop before a capital letter or a bracket, else all of it. */
function firstSentence(doc: string | undefined) {
  const text = (doc ?? '').replace(/\s+/g, ' ').trim()
  const m = /^.*?[.!?](?=\s+[A-Z(]|$)/.exec(text)
  return m ? m[0] : text
}

const typesOf = (ports: PortSpec[]) => [...new Set(ports.map((p) => p.type))]
/** Port types that are settings typed into the node rather than what it connects to. */
const SETTINGS = ['number', 'int', 'text']
/** The chips of the inputs: what connects (bodies, faces, fields…); the settings only for a node with nothing else. */
function inputTypes(spec: NodeSpec) {
  const all = typesOf(spec.inputs.filter((p) => !p.hidden))
  const data = all.filter((t) => !SETTINGS.includes(t))
  return data.length ? data : all
}

function nodeEntry(spec: NodeSpec): Entry {
  return {
    key: spec.type, title: spec.title, category: spec.category, doc: spec.doc, summary: firstSentence(spec.doc),
    payload: spec.type, ins: inputTypes(spec), outs: typesOf(spec.outputs),
    haystack: `${spec.title} ${spec.category} ${spec.type} ${spec.doc}`.toLowerCase(),
  }
}

function presetEntry(p: PresetInfo): Entry {
  const doc = p.doc ?? ''
  const meta = `${p.nodes} node${p.nodes === 1 ? '' : 's'}`
  return {
    key: PRESET_PREFIX + p.name, title: p.name, category: 'Presets', doc: doc || meta, summary: firstSentence(doc),
    payload: PRESET_PREFIX + p.name, ins: [], outs: [], meta, haystack: `${p.name} preset ${doc}`.toLowerCase(),
  }
}

/** A position for a new node: `at`, moved down and right while another node already sits there. */
function freeSpot(at: { x: number; y: number }) {
  const nodes = useStore.getState().nodes
  const spot = { ...at }
  for (let i = 0; i < 40 && nodes.some((n) => Math.hypot(n.position.x - spot.x, n.position.y - spot.y) < 30); i++) {
    spot.x += 40
    spot.y += 40
  }
  return spot
}

/** Add what a library tile stands for (a node type id, or "preset:<name>") with its corner at `at`
 *  (graph coordinates), and flash it. Used by clicks on a tile and by drops on the canvas. */
export async function addFromLibrary(payload: string, at: { x: number; y: number }) {
  const s = useStore.getState()
  const spot = freeSpot(at)
  if (!payload.startsWith(PRESET_PREFIX)) {
    if (!s.specs[payload]) return
    s.flashNodes([s.addNode(payload, spot)])
    return
  }
  const name = payload.slice(PRESET_PREFIX.length)
  try {
    const clip = await api.preset(name)
    const n = clip ? useStore.getState().pasteClip(clip, spot, false) : 0
    const now = useStore.getState()
    if (n) now.flashNodes(now.nodes.filter((x) => x.selected).map((x) => x.id))
    now.addLog(n ? `Added preset "${name}"` : `Preset "${name}" has no nodes this editor knows`, n ? 'info' : 'error')
  } catch (err) {
    useStore.getState().addLog(`Could not load preset "${name}": ${(err as Error).message}`, 'error')
  }
}

/** The port types of a tile: inputs, an arrow, outputs. */
function TypeChips({ ins, outs }: { ins: string[]; outs: string[] }) {
  if (!ins.length && !outs.length) return null
  const chip = (t: string) => (
    <span key={t} className="nlib-chip" style={{ ['--t' as string]: TYPE_COLORS[t] ?? '#9aa0a8' }}>{t}</span>
  )
  const MAX = 4
  return (
    <div className="nlib-tile-types">
      {ins.slice(0, MAX).map(chip)}
      {ins.length > MAX && <span className="nlib-more" title={ins.join(', ')}>+{ins.length - MAX}</span>}
      <span className="nlib-arrow">→</span>
      {outs.map(chip)}
    </div>
  )
}

function Tile({ entry, highlighted, onAdd, onHover, onDrag }: {
  entry: Entry; highlighted: boolean; onAdd: () => void; onHover: () => void; onDrag: () => void
}) {
  return (
    <div className={`nlib-tile${highlighted ? ' active' : ''}`} role="button" title={entry.doc}
      style={{ ['--cat' as string]: categoryColor(entry.category) }}
      draggable onClick={onAdd} onMouseEnter={onHover}
      onDragStart={(e) => {
        e.dataTransfer.setData(NODE_DRAG_TYPE, entry.payload)
        e.dataTransfer.effectAllowed = 'copy'
        onDrag()
      }}>
      <div className="nlib-tile-title">{entry.title}</div>
      {entry.summary && <div className="nlib-tile-doc">{entry.summary}</div>}
      {entry.meta ? <div className="nlib-tile-meta">{entry.meta}</div> : <TypeChips ins={entry.ins} outs={entry.outs} />}
    </div>
  )
}

/** The ribbon above the graph: a tab per node category (and Groups, Presets) that opens a panel of
 *  node tiles over the canvas, and a search box. Click a tile to add it in the middle of the canvas, or
 *  drag it where it should go (the drop is handled in NodeEditor). */
export function NodeLibrary() {
  const catalog = useStore((s) => s.catalog)
  const specs = useStore((s) => s.specs)
  const editingGroup = useStore((s) => s.editing?.name)
  const rf = useReactFlow()
  const [panel, setPanel] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const [presets, setPresets] = useState<PresetInfo[] | null>(null)
  const root = useRef<HTMLDivElement>(null)
  const strip = useRef<HTMLDivElement>(null)
  const panelEl = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const keyNav = useRef(false)

  // the categories in build order, then the user's groups, then presets
  const { categories, byCategory } = useMemo(() => {
    const categories: string[] = []
    const byCategory: Record<string, Entry[]> = {}
    const put = (e: Entry) => {
      if (!byCategory[e.category]) byCategory[e.category] = []
      byCategory[e.category].push(e)
    }
    for (const n of catalog?.nodes ?? []) {
      if (!categories.includes(n.category)) categories.push(n.category)
      put(nodeEntry(n))
    }
    for (const g of Object.values(specs)) {
      if (!g.type.startsWith(GROUP_PREFIX) || (editingGroup && g.type === GROUP_PREFIX + editingGroup)) continue  // a group cannot contain itself
      put(nodeEntry(g))
    }
    for (const p of presets ?? []) put(presetEntry(p))
    categories.sort((a, b) => rank(a) - rank(b))  // stable: unknown categories keep the catalog's order
    return { categories, byCategory }
  }, [catalog, specs, editingGroup, presets])

  const sections = useMemo(() => [...categories, ...(byCategory.Groups ? ['Groups'] : []), 'Presets'], [categories, byCategory])
  const tabs = useMemo(() => [ALL, ...sections], [sections])
  const nodeCount = categories.reduce((n, c) => n + (byCategory[c]?.length ?? 0), 0)
  const count = (tab: string) => (tab === ALL ? String(nodeCount)
    : tab === 'Presets' && presets === null ? '' : String(byCategory[tab]?.length ?? 0))

  const shown = useMemo(() => {
    if (panel === SEARCH) {
      const words = query.toLowerCase().split(/\s+/).filter(Boolean)
      const first = (e: Entry) => (e.title.toLowerCase().includes(words[0] ?? '') ? 0 : 1)  // title matches first
      return sections
        .map((category) => ({
          category,
          items: (byCategory[category] ?? []).filter((e) => words.every((w) => e.haystack.includes(w))).sort((a, b) => first(a) - first(b)),
        }))
        .filter((s) => s.items.length)
    }
    if (panel === ALL) return sections.filter((c) => byCategory[c]?.length).map((category) => ({ category, items: byCategory[category] }))
    return panel && byCategory[panel]?.length ? [{ category: panel, items: byCategory[panel] }] : []
  }, [panel, query, sections, byCategory])
  const flat = useMemo(() => shown.flatMap((s) => s.items), [shown])
  const active = panel === SEARCH ? Math.min(index, flat.length - 1) : -1

  // presets: loaded when the Presets tab opens (every time) or a search first needs them
  const loadPresets = useCallback(() => {
    api.presets().then(setPresets).catch(() => setPresets((p) => p ?? []))
  }, [])
  useEffect(() => { if (panel === 'Presets') loadPresets() }, [panel, loadPresets])
  const needPresets = (panel === SEARCH || panel === ALL) && presets === null
  useEffect(() => { if (needPresets) loadPresets() }, [needPresets, loadPresets])

  useEffect(() => setIndex(0), [query])
  useEffect(() => {  // keep the highlighted result in view (only after the arrow keys, not when the mouse moves it)
    if (!keyNav.current) return
    keyNav.current = false
    panelEl.current?.querySelector('.nlib-tile.active')?.scrollIntoView({ block: 'nearest' })
  }, [index])

  // Fit the tabs to the pane: the counts go first, then the names (the counts come back for the
  // glyphs), then the counts again; the row never wraps or scrolls.
  const tabSig = tabs.map((t) => `${t}:${count(t)}`).join('|')
  useLayoutEffect(() => {
    const el = strip.current
    if (!el) return
    const fit = () => {
      for (const level of ['0', '1', '2', '3']) {
        el.dataset.fit = level
        if (el.scrollWidth <= el.clientWidth + 1) break
      }
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    void document.fonts?.ready.then(fit)
    return () => ro.disconnect()
  }, [tabSig])

  // Esc closes (and clears the search), a press outside the ribbon and panel closes too.
  useEffect(() => {
    if (!panel && !query) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || (!panel && document.activeElement !== input.current)) return
      e.stopPropagation()  // this Esc is for the library only (not the group edit, picking...)
      setPanel(null)
      setQuery('')
      input.current?.blur()
    }
    const onDown = (e: PointerEvent) => {
      if (root.current?.contains(e.target as Node)) return
      setPanel(null)
      setQuery('')
    }
    document.addEventListener('keydown', onKey, true)
    document.addEventListener('pointerdown', onDown, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      document.removeEventListener('pointerdown', onDown, true)
    }
  }, [panel, query])

  // "/" focuses the search (unless the user is typing somewhere)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return
      const el = e.target as HTMLElement | null
      if (el && (['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName) || el.isContentEditable)) return
      e.preventDefault()
      input.current?.focus()
      input.current?.select()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  /** Add at the middle of the visible canvas, then close. */
  const add = (entry: Entry) => {
    const canvas = root.current?.closest('.editor-pane')?.querySelector('.react-flow') ?? document.querySelector('.react-flow')
    const r = canvas?.getBoundingClientRect()
    const at = rf.screenToFlowPosition(r ? { x: r.left + r.width / 2, y: r.top + r.height / 2 } : { x: 400, y: 300 })
    void addFromLibrary(entry.payload, { x: at.x - CENTRE_SHIFT.x, y: at.y - CENTRE_SHIFT.y })
    setPanel(null)
    setQuery('')
    input.current?.blur()
  }

  // a drag starts later: removing the dragged tile during dragstart would cancel the drag
  const closeSoon = () => window.setTimeout(() => { setPanel(null); setQuery('') }, 0)

  if (!catalog) return null
  const sectionNote = (category: string, items: number) =>
    panel === SEARCH || (panel === ALL && category !== shown[0]?.category) ? String(items)
      : `${items} ${category === 'Presets' ? 'preset' : 'node'}${items === 1 ? '' : 's'} · click to add, or drag onto the canvas`

  return (
    <div className="nlib" ref={root}>
      <div className="nlib-bar">
        <div className="nlib-tabs" ref={strip}>
          {tabs.map((tab) => (
            <button key={tab} className={`nlib-tab${panel === tab ? ' active' : ''}${tab === ALL ? ' nlib-all' : ''}`}
              aria-expanded={panel === tab}
              style={{ ['--cat' as string]: categoryColor(tab) }}
              title={count(tab) ? `${tab} (${count(tab)})` : tab}
              onClick={() => { setQuery(''); setPanel((p) => (p === tab ? null : tab)) }}
              onMouseEnter={() => setPanel((p) => (p && p !== SEARCH ? tab : p))}>
              <span className="nlib-glyph">{glyph(tab)}</span>
              <span className="nlib-name">{tab}</span>
              <span className="nlib-count">{count(tab)}</span>
            </button>
          ))}
        </div>
        <div className="nlib-search">
          <input ref={input} value={query} placeholder="Find a node…" aria-label="Find a node" spellCheck={false}
            onChange={(e) => { setQuery(e.target.value); setPanel(e.target.value.trim() ? SEARCH : null) }}
            onFocus={() => { if (query.trim()) setPanel(SEARCH) }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault()
                if (panel !== SEARCH) { if (query.trim()) setPanel(SEARCH); return }
                keyNav.current = true
                setIndex(Math.max(0, Math.min(active + (e.key === 'ArrowDown' ? 1 : -1), flat.length - 1)))
              } else if (e.key === 'Enter' && panel === SEARCH && flat[active]) {
                add(flat[active])
              }
            }} />
        </div>
      </div>
      {panel && (
        <div className={`nlib-panel${panel === ALL ? ' all' : ''}`} ref={panelEl}>
          {shown.map((section) => (
            <section key={section.category} className="nlib-section" style={{ ['--cat' as string]: categoryColor(section.category) }}>
              <div className="nlib-head">
                <span className="nlib-head-name">{section.category}</span>
                <span className="nlib-head-note">{sectionNote(section.category, section.items.length)}</span>
              </div>
              <div className="nlib-grid">
                {section.items.map((e) => {
                  const i = flat.indexOf(e)
                  return <Tile key={e.key} entry={e} highlighted={i === active} onAdd={() => add(e)} onHover={() => setIndex(i)} onDrag={closeSoon} />
                })}
              </div>
            </section>
          ))}
          {flat.length === 0 && (
            <div className="nlib-empty">
              {panel === SEARCH ? 'No matching nodes'
                : panel === 'Presets' ? (presets === null ? 'Loading…' : 'No presets yet: select nodes and use Save as preset')
                  : 'Nothing here'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
