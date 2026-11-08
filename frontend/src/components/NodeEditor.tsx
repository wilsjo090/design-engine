import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background, BackgroundVariant, Controls, MiniMap, ReactFlow, useReactFlow, useStoreApi,
  type Connection, type Edge, type FinalConnectionState, type IsValidConnection, type Node, type NodeChange,
} from '@xyflow/react'
import { useShallow } from 'zustand/react/shallow'
import { api } from '../api'
import { isClip, isFieldNode, unselectFrames, useStore, type Clip } from '../store'
import { GROUP_IN_ID, GROUP_OUT_ID, outputType } from '../graphConvert'
import {
  FRAME_COLORS, contents, foldedEdges, frameNodes, hiddenBy, isFrameId, type Frames,
} from '../graphFrames'
import { EngineNodeComponent, GroupIONode } from './EngineNode'
import { GraphFrameNode } from './GraphFrameNode'
import { HelpPanel } from './HelpPanel'
import { NODE_DRAG_TYPE, addFromLibrary } from './NodeLibrary'
import { NodePalette, compatibleInput, compatibleOutput, type PaletteFilter } from './NodePalette'
import { NamePrompt } from './QuickEdit'
import { ViewportMenu } from './ViewportMenu'
import type { EngineNode, MenuEntry, MenuItem, PromptState } from '../types'
import { CATEGORY_COLORS, DISPLAYABLE, GROUP_PREFIX, bypassInput } from '../types'
import { FAR_ZOOM, LONG_WIRE, SHARED_WIRES, drawnEdges, hoverWires, useWires } from '../wires'

const nodeTypes = { engine: EngineNodeComponent, groupIn: GroupIONode, groupOut: GroupIONode, graphFrame: GraphFrameNode }
const NO_FRAMES: Frames = {}

interface PaletteState {
  x: number
  y: number
  filter?: PaletteFilter
  /** connect the new node to this handle */
  from?: { nodeId: string; handle: string; side: 'source' | 'target'; type: string }
  /** put the new node on this wire (source.sourceHandle -> target.targetHandle, with their types) */
  insert?: { source: string; sourceHandle: string; target: string; targetHandle: string; outType: string; inType: string }
}

const menuItem = (label: string, run?: MenuItem['run'], extra: Partial<MenuItem> = {}): MenuItem => ({ kind: 'item', label, run, ...extra })
const firstSentence = (doc: string) => doc.split(/(?<=\.)\s/)[0]

/** Selects exactly these nodes (and no wires). */
function selectOnly(ids: string[]) {
  const chosen = new Set(ids)
  useStore.setState((s) => ({
    nodes: s.nodes.map((n) => (!!n.selected !== chosen.has(n.id) ? { ...n, selected: chosen.has(n.id) } : n)),
    edges: s.edges.map((e) => (e.selected ? { ...e, selected: false } : e)),
    graphFrames: unselectFrames(s.graphFrames),
  }))
}

/** Every node `id` feeds (`down`) or is fed by, through any number of links. */
function reach(id: string, down: boolean) {
  const { edges } = useStore.getState()
  const out = new Set<string>([id])
  const stack = [id]
  while (stack.length) {
    const n = stack.pop()!
    for (const e of edges) {
      const next = down ? (e.source === n ? e.target : null) : (e.target === n ? e.source : null)
      if (next && !out.has(next)) { out.add(next); stack.push(next) }
    }
  }
  return [...out]
}

const sep: MenuEntry = { kind: 'separator' }

/** The selected nodes and frames (a frame around them takes both). */
function pickedForFrame() {
  const s = useStore.getState()
  return [...s.nodes.filter((n) => n.selected && n.type === 'engine').map((n) => n.id),
    ...Object.entries(s.graphFrames).filter(([, f]) => f.selected).map(([id]) => id)]
}

/** The right-click menu of a frame. */
function frameMenu(id: string, rename: () => void): MenuEntry[] {
  const s = useStore.getState()
  const f = s.graphFrames[id]
  if (!f) return []
  const nodesIn = () => contents(id, s.graphFrames, s.nodes).filter((x) => s.nodes.some((n) => n.id === x))
  const count = nodesIn().length
  return [
    { kind: 'header', text: `${f.label} · ${count} node${count === 1 ? '' : 's'}` },
    menuItem('Rename…', rename, { hint: 'double-click the name' }),
    menuItem('Colour', undefined, {
      items: Object.entries(FRAME_COLORS).map(([name, color]) =>
        menuItem(name, () => useStore.getState().updateFrame(id, { color }), { checked: f.color === color })),
    }),
    menuItem(f.folded ? 'Unfold' : 'Fold', () => useStore.getState().foldFrame(id, !f.folded),
      { title: f.folded ? 'Show its nodes again' : 'Only its title bar: its nodes are hidden and their wires end on the bar' }),
    sep,
    menuItem('Select its nodes', () => {
      if (f.folded) useStore.getState().foldFrame(id, false)
      selectOnly(nodesIn())
    }),
    menuItem('Tidy its nodes', () => useStore.getState().tidyFrame(id), {
      disabled: f.folded ? 'Unfold it first' : undefined, title: 'In columns along the links, and the frame fitted round them',
    }),
    sep,
    menuItem('Remove frame', () => useStore.getState().removeFrame(id, false),
      { hint: f.folded ? undefined : 'Del', title: 'Its nodes stay where they are' }),
    menuItem('Delete frame and its nodes', () => useStore.getState().removeFrame(id, true)),
  ]
}

/** The right-click menu of a node: it acts on the selection when the node is part of it, else on the node. */
function nodeMenu(id: string, rename: () => void, frame: (ids: string[]) => void): MenuEntry[] {
  const s = useStore.getState()
  const node = s.nodes.find((n) => n.id === id)
  if (!node || node.type !== 'engine') return []
  const picked = node.selected ? s.nodes.filter((n) => n.selected && n.type === 'engine').map((n) => n.id) : [id]
  const many = picked.length > 1
  const spec = s.specs[node.data.typeId]
  const byId = (nid: string) => s.nodes.find((n) => n.id === nid)
  const linked = (nid: string) => (p: string) => s.edges.some((e) => e.target === nid && e.targetHandle === p)
  const canPass = picked.some((nid) => !!bypassInput(s.specs[byId(nid)?.data.typeId ?? ''], linked(nid)))
  const bypassed = picked.every((nid) => byId(nid)?.data.bypass)
  const collapsed = picked.every((nid) => byId(nid)?.data.collapsed)
  const canShow = !s.editing && (isFieldNode(spec) || !!spec?.outputs.some((o) => DISPLAYABLE.includes(o.type)))
  const group = node.data.typeId.startsWith(GROUP_PREFIX)
  /** select what the action works on first (copying, grouping and presets use the selection) */
  const onPicked = (run: () => void) => () => { selectOnly(picked); run() }
  return [
    { kind: 'header', text: many ? `${picked.length} nodes` : `${id} · ${spec?.title ?? node.data.typeId}` },
    ...(!many && spec?.doc ? [{ kind: 'header', text: firstSentence(spec.doc) } as MenuEntry] : []),
    ...(canShow && !many ? [menuItem(s.visible.includes(id) ? 'Hide in 3D view' : 'Show in 3D view', () => useStore.getState().toggleVisible(id))] : []),
    ...(!many ? [menuItem('Rename…', rename, { hint: 'double-click the name' })] : []),
    menuItem('Bypass', () => useStore.getState().setNodeFlags(picked, { bypass: !bypassed }), {
      hint: 'B', checked: bypassed,
      disabled: !bypassed && !canPass ? "Nothing to pass on: no linked input of the output's type" : undefined,
      title: 'Leave the node out as if it were not there (its input passes straight on), to compare',
    }),
    menuItem(collapsed ? 'Expand' : 'Collapse', () => useStore.getState().setNodeFlags(picked, { collapsed: !collapsed }), { hint: 'H' }),
    sep,
    menuItem('Duplicate', onPicked(() => {
      const clip = useStore.getState().copySelection()
      if (clip) useStore.getState().pasteClip(clip, null, true)
    }), { hint: 'Ctrl+D' }),
    menuItem('Reset to defaults', () => {
      for (const nid of picked) for (const port of Object.keys(byId(nid)?.data.inputs ?? {})) useStore.getState().setInput(nid, port, undefined)
    }, { title: 'Every typed-in value back to the default (links stay)' }),
    menuItem('Select upstream', () => selectOnly([...new Set(picked.flatMap((nid) => reach(nid, false)))]), { title: 'Everything this is made from' }),
    menuItem('Select downstream', () => selectOnly([...new Set(picked.flatMap((nid) => reach(nid, true)))]), { title: 'Everything made from this' }),
    ...(many ? [menuItem('Tidy up', () => useStore.getState().tidy(picked), { title: 'In columns along the links' })] : []),
    sep,
    menuItem('Put in a frame…', () => frame(node.selected ? pickedForFrame() : [id]), {
      hint: 'F', disabled: s.editing ? 'Frames are drawn outside groups' : undefined,
      title: 'A labelled box behind them: dragging it moves them, folding it hides them behind its title bar',
    }),
    menuItem('Group…', onPicked(groupSelection), { hint: 'Ctrl+G', disabled: s.editing ? 'Finish editing the open group first' : undefined }),
    ...(group && !many ? [menuItem('Ungroup', () => useStore.getState().ungroup(id), { hint: 'Ctrl+Shift+G' })] : []),
    menuItem('Save as preset…', onPicked(saveSelectionAsPreset)),
    sep,
    menuItem('Delete', () => useStore.getState().deleteNodes(picked, false), { hint: 'Del' }),
    menuItem('Delete and reconnect', () => useStore.getState().deleteNodes(picked, true),
      { title: 'What used them takes what came into them (as if bypassed), so a chain stays whole' }),
  ]
}

/** The View menu: how the graph draws its cards and wires (useWires). */
function viewMenu(): MenuEntry[] {
  const w = useWires.getState()
  return [
    { kind: 'header', text: 'Cards' },
    menuItem('Simple cards when zoomed out', () => w.set({ simple: !w.simple }), {
      checked: w.simple,
      title: 'Zoomed out so far that their text is too small to read, cards show only their name, large, and the ports ' +
        'that are wired (each card keeps its size, so nothing moves)',
    }),
    { kind: 'header', text: 'Wires' },
    menuItem("Highlight the selected node's wires", () => w.set({ focus: !w.focus }), {
      checked: w.focus, title: 'The wires of the selected nodes stand out and the others fade (lightly while you only hover a node)',
    }),
    menuItem('Shared outputs as labels', () => w.set({ shared: !w.shared }), {
      checked: w.shared,
      title: `An output with ${SHARED_WIRES} or more wires (a part, a material) shows as a label on each input it feeds. ` +
        'The wires show while either end is hovered or selected; a click on a label goes to its node.',
    }),
    menuItem('Long wires as labels', () => w.set({ long: !w.long }), {
      checked: w.long, title: `A wire longer than about ${Math.round(LONG_WIRE / 320)} columns shows as a label too`,
    }),
  ]
}

/** A box listing the graph's nodes by name or type: Enter (or a click) pans to one and selects it. */
function GraphFind({ onClose }: { onClose: () => void }) {
  const nodes = useStore((s) => s.nodes)
  const specs = useStore((s) => s.specs)
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const q = query.trim().toLowerCase()
  const found = nodes.filter((n) => n.type === 'engine' &&
    `${n.id} ${specs[n.data.typeId]?.title ?? ''} ${n.data.typeId}`.toLowerCase().includes(q)).slice(0, 12)
  const go = (id: string) => { useStore.getState().focusNode(id); onClose() }
  return (
    <div className="graph-find" onKeyDown={(e) => e.stopPropagation()}>
      <input className="value-input" autoFocus placeholder="Find a node (name or type)…" value={query}
        onChange={(e) => { setQuery(e.target.value); setIndex(0) }}
        onBlur={() => setTimeout(onClose, 150)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') onClose()
          else if (e.key === 'ArrowDown') { e.preventDefault(); setIndex((i) => Math.min(i + 1, found.length - 1)) }
          else if (e.key === 'ArrowUp') { e.preventDefault(); setIndex((i) => Math.max(i - 1, 0)) }
          else if (e.key === 'Enter' && found[index]) go(found[index].id)
        }} />
      {found.map((n, i) => (
        <div key={n.id} className={`graph-find-row ${i === index ? 'active' : ''}`}
          onMouseEnter={() => setIndex(i)} onMouseDown={(e) => { e.preventDefault(); go(n.id) }}>
          <span>{n.id}</span><span>{specs[n.data.typeId]?.title ?? n.data.typeId}</span>
        </div>
      ))}
      {!found.length && <div className="graph-find-row">No node matches</div>}
    </div>
  )
}

/** True if adding source -> target would create a loop. */
function createsCycle(edges: Edge[], source: string, target: string) {
  const stack = [target]
  const seen = new Set<string>()
  while (stack.length) {
    const id = stack.pop()!
    if (id === source) return true
    if (seen.has(id)) continue
    seen.add(id)
    edges.filter((e) => e.source === id).forEach((e) => stack.push(e.target))
  }
  return false
}

/** Type of a port, also on the input and output nodes of an open group. */
function portType(nodeId: string, handle: string | null | undefined, side: 'source' | 'target') {
  const s = useStore.getState()
  if (s.editing && (nodeId === GROUP_IN_ID || nodeId === GROUP_OUT_ID)) {
    const def = s.groups[s.editing.name]
    return (nodeId === GROUP_IN_ID ? def?.inputs : def?.outputs)?.find((p) => p.name === handle)?.type
  }
  if (side === 'source') return outputType(s.nodes, s.edges, s.specs, nodeId, handle)
  const spec = s.specs[s.nodes.find((n) => n.id === nodeId)?.data.typeId ?? '']
  return spec?.inputs.find((p) => p.name === handle)?.type
}

function groupSelection() {
  const s = useStore.getState()
  if (!s.nodes.some((n) => n.selected && n.type === 'engine')) return
  const name = prompt('Name for the new group (e.g. noise_wave). Its Number nodes become its settings.')?.trim()
  if (!name) return
  const error = s.createGroup(name)
  if (error) alert(error)
  else s.addLog(`Grouped into "${name}": Edit opens it, Ungroup puts the nodes back`, 'success')
}

async function saveSelectionAsPreset() {
  const s = useStore.getState()
  const clip = s.copySelection()
  if (!clip) return
  const name = prompt('Name for the preset (it appears under Presets when you add a node, in every recipe):')?.trim()
  if (!name) return
  const doc = prompt('A short description (optional):')?.trim()
  try {
    await api.savePreset(name, { ...clip, ...(doc ? { doc } : {}) })
    s.addLog(`Saved preset "${name}" (${clip.nodes.length} node${clip.nodes.length === 1 ? '' : 's'})`, 'success')
  } catch (err) {
    s.addLog(`Could not save the preset: ${(err as Error).message}`, 'error')
  }
}

function SelectionBar({ onFrame }: { onFrame: () => void }) {
  const count = useStore((s) => s.nodes.filter((n) => n.selected && n.type === 'engine').length)
  const groupNode = useStore((s) => {
    const picked = s.nodes.filter((n) => n.selected && n.type === 'engine')
    return picked.length === 1 && picked[0].data.typeId.startsWith(GROUP_PREFIX) ? picked[0].id : null
  })
  const editing = useStore((s) => s.editing)
  if (editing) {
    return (
      <div className="group-banner">
        <span>Editing group <b>{editing.name}</b>: changes apply to every copy. Number nodes are its settings.</span>
        <button className="primary" onClick={() => useStore.getState().finishGroupEdit()} title="Close the group (Esc)">Done</button>
      </div>
    )
  }
  if (!count) return null
  return (
    <div className="selection-bar">
      <span>{count} selected</span>
      <button onClick={onFrame} title="Put them in a frame: a labelled box behind them that moves them and can fold them away (F)">Frame</button>
      <button onClick={groupSelection} title="Make one node of the selected ones (Ctrl+G)">Group</button>
      {groupNode && <button onClick={() => useStore.getState().ungroup(groupNode)} title="Ctrl+Shift+G">Ungroup</button>}
      <button onClick={saveSelectionAsPreset} title="Keep these nodes as a preset you can add in any recipe">Save as preset</button>
    </div>
  )
}

export function NodeEditor({ paletteSignal }: { paletteSignal: number }) {
  const { nodes, edges, onNodesChange, onEdgesChange, connect } = useStore(useShallow((s) => ({
    nodes: s.nodes, edges: s.edges, onNodesChange: s.onNodesChange, onEdgesChange: s.onEdgesChange, connect: s.connect,
  })))
  const rf = useReactFlow()
  const [palette, setPalette] = useState<PaletteState | null>(null)
  const [menu, setMenu] = useState<{ x: number; y: number; entries: MenuEntry[] } | null>(null)
  const [prompt, setPrompt] = useState<PromptState | null>(null)
  const [finding, setFinding] = useState(false)
  const mouse = useRef({ x: 300, y: 300 })
  const container = useRef<HTMLDivElement>(null)

  // Frames (not inside an open group) go behind the nodes; what folded frames hide is not drawn and its
  // wires end on their bars. The wires focus on the selected or hovered nodes, a frame's standing for
  // what is in it.
  const graphFrames = useStore((s) => s.graphFrames)
  const editing = useStore((s) => !!s.editing)
  const frames = editing ? NO_FRAMES : graphFrames
  const hidden = useMemo(() => hiddenBy(frames), [frames])
  const flowNodes = useMemo<Node[]>(() => (Object.keys(frames).length
    ? [...frameNodes(frames, nodes, hidden), ...nodes.map((n) => (hidden.has(n.id) ? { ...n, hidden: true } : n))]
    : nodes), [frames, nodes, hidden])
  const wires = useWires(useShallow((w) => ({ focus: w.focus, shared: w.shared, long: w.long, hover: w.hover })))
  const shownEdges = useMemo(() => {
    const standsFor = (id: string) => (isFrameId(id) ? contents(id, frames, nodes) : [id])
    const picked = new Set([...nodes.filter((n) => n.selected).map((n) => n.id),
      ...Object.entries(frames).filter(([, f]) => f.selected).flatMap(([id]) => standsFor(id))])
    const hovered = new Set(wires.hover?.node ? standsFor(wires.hover.node) : [])
    return foldedEdges(drawnEdges(nodes, edges, wires, { picked, hovered, edge: wires.hover?.edge }), hidden)
  }, [nodes, edges, wires, frames, hidden])

  /** frame changes go to the frames, the rest to the nodes */
  const onFlowNodesChange = useCallback((changes: NodeChange[]) => {
    const mine = changes.filter((c) => 'id' in c && isFrameId(c.id))
    if (mine.length) useStore.getState().frameChanges(mine)
    if (mine.length < changes.length) onNodesChange(changes.filter((c) => !('id' in c && isFrameId(c.id))) as NodeChange<EngineNode>[])
  }, [onNodesChange])

  /** ask for a name, then put a frame around `ids` */
  const frameIds = useCallback((ids: string[]) => {
    if (!ids.length || useStore.getState().editing) return
    const { x, y } = mouse.current
    setPrompt({ x, y, label: 'Name of the frame', value: 'Frame', done: (name) => { useStore.getState().addFrame(ids, name) } })
  }, [])
  const renameFrame = (id: string, x: number, y: number) => {
    const f = useStore.getState().graphFrames[id]
    if (f) setPrompt({ x, y, label: 'New name for the frame', value: f.label, done: (name) => useStore.getState().updateFrame(id, { label: name }) })
  }

  // the 3D view's "find" (and Ctrl+F over the graph) opens the find box
  const findRequest = useStore((s) => s.findRequest)
  useEffect(() => { if (findRequest) setFinding(true) }, [findRequest])

  const openNodeMenu = (x: number, y: number, id: string) => setMenu({ x, y, entries: isFrameId(id)
    ? frameMenu(id, () => renameFrame(id, x, y))
    : nodeMenu(id, () => setPrompt({
      x, y, label: `New name for ${id}`, value: id, done: (name) => {
        const error = useStore.getState().renameNode(id, name)
        if (error) useStore.getState().addLog(error, 'error')
      },
    }), frameIds) })
  /** a wire: put a node on it (the palette lists the nodes that fit between its ends), or delete it */
  const insertOn = (shown: Edge, x: number, y: number) => {
    const s = useStore.getState()
    const edge = s.edges.find((e) => e.id === shown.id) ?? shown   // not the end a folded frame's bar stands in for
    const outType = portType(edge.source, edge.sourceHandle, 'source')
    const inType = portType(edge.target, edge.targetHandle, 'target')
    if (!outType || !inType || !edge.sourceHandle || !edge.targetHandle || s.editing && (edge.source === GROUP_IN_ID || edge.target === GROUP_OUT_ID)) return
    setPalette({ x, y, filter: { fromOutput: outType, fromInput: inType },
      insert: { source: edge.source, sourceHandle: edge.sourceHandle, target: edge.target, targetHandle: edge.targetHandle, outType, inType } })
  }
  const openEdgeMenu = (x: number, y: number, shown: Edge) => {
    const edge = useStore.getState().edges.find((e) => e.id === shown.id) ?? shown
    setMenu({ x, y, entries: [
      { kind: 'header', text: `${edge.source}.${edge.sourceHandle} → ${edge.target}.${edge.targetHandle}` },
      menuItem('Insert node…', () => insertOn(edge, x, y), { hint: 'double-click', title: 'A node that fits between the two ends, wired in' }),
      menuItem('Delete wire', () => onEdgesChange([{ type: 'remove', id: edge.id }]), { hint: 'Del' }),
    ] })
  }

  // Zoomed out past FAR_ZOOM the cards draw their simple face (CSS on `data-far`), and `--zoom` keeps names
  // readable; set on the element as the view moves, so zooming re-renders nothing.
  const flow = useStoreApi()
  const simple = useWires((w) => w.simple)
  useEffect(() => {
    const el = container.current
    if (!el) return
    let last = -1
    const apply = (zoom: number) => {
      if (zoom === last) return
      last = zoom
      el.style.setProperty('--zoom', String(zoom))
      el.toggleAttribute('data-far', simple && zoom < FAR_ZOOM)
    }
    apply(flow.getState().transform[2])
    return flow.subscribe((s) => apply(s.transform[2]))
  }, [flow, simple])

  // Frame the whole graph after a recipe loads (once React Flow has measured the nodes).
  const loadCount = useStore((s) => s.loadCount)
  useEffect(() => {
    if (!loadCount) return
    const t = setTimeout(() => rf.fitView({ padding: 0.08, maxZoom: 1, duration: 200 }), 80)
    return () => clearTimeout(t)
  }, [loadCount, rf])

  // Pan to a node the 3D view points at (keeps the zoom).
  const focus = useStore((s) => s.focus)
  useEffect(() => {
    if (!focus) return
    const node = useStore.getState().nodes.find((n) => n.id === focus.id)
    if (!node) return
    const w = node.measured?.width ?? 260, h = node.measured?.height ?? 200
    rf.setCenter(node.position.x + w / 2, node.position.y + h / 2, { zoom: rf.getZoom(), duration: 300 })
  }, [focus, rf])

  // Copy / cut / paste / duplicate nodes. Native clipboard events, so text in input fields still
  // copies normally, and nodes can go between recipes and browser tabs.
  const mouseInside = useRef(false)
  const keepExternalNext = useRef(false)
  useEffect(() => {
    const typing = () => {
      const el = document.activeElement as HTMLElement | null
      return !!el && (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName) || el.isContentEditable)
    }
    const pasteAt = () => (mouseInside.current ? rf.screenToFlowPosition(mouse.current) : null)
    const report = (text: string) => useStore.getState().addLog(text)
    const plural = (n: number) => `${n} node${n === 1 ? '' : 's'}`

    // Shortcuts act on keydown and keep their own copy of the clipboard; the system clipboard is
    // written too (and read on paste when the browser delivers it) so nodes travel between tabs.
    let lastClip: Clip | null = null
    let copiedNow = false  // this Ctrl+C/X copied nodes (else leave the clipboard alone)
    let pasteEventSeen = false

    const copy = (cut: boolean) => {
      const clip = useStore.getState().copySelection()
      if (!clip) return null
      lastClip = clip
      navigator.clipboard?.writeText(JSON.stringify(clip)).catch(() => {})
      if (cut) rf.deleteElements({ nodes: clip.nodes.map((n) => ({ id: n.id })) })
      report(`${cut ? 'Cut' : 'Copied'} ${plural(clip.nodes.length)}`)
      return clip
    }
    const paste = (clip: Clip, keep: boolean) => {
      const n = useStore.getState().pasteClip(clip, pasteAt(), keep)
      report(n ? `Pasted ${plural(n)}${keep ? ' (inputs kept)' : ''}` : 'Nothing to paste: unknown node types')
    }

    const onCopyEvent = (e: ClipboardEvent) => {  // put the same data on the event's clipboard
      if (!copiedNow || !lastClip || !e.clipboardData) return
      copiedNow = false
      e.clipboardData.setData('text/plain', JSON.stringify(lastClip))
      e.preventDefault()
    }
    const onPasteEvent = (e: ClipboardEvent) => {
      if (typing()) return
      pasteEventSeen = true
      let clip: unknown
      try { clip = JSON.parse(e.clipboardData?.getData('text/plain') ?? '') } catch { return }
      if (!isClip(clip)) return
      e.preventDefault()
      paste(clip, keepExternalNext.current)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && useStore.getState().editing && !typing() && !useStore.getState().picking) {
        useStore.getState().finishGroupEdit()
        return
      }
      // over the graph: B bypasses the selected nodes, H collapses them, F puts them in a frame
      if (mouseInside.current && !typing() && !e.ctrlKey && !e.metaKey && !e.altKey && ['b', 'h', 'f'].includes(e.key)) {
        const s = useStore.getState()
        if (e.key === 'f') { frameIds(pickedForFrame()); return }
        const picked = s.nodes.filter((n) => n.selected && n.type === 'engine')
        if (!picked.length) return
        if (e.key === 'b') s.setNodeFlags(picked.map((n) => n.id), { bypass: !picked.every((n) => n.data.bypass) })
        else s.setNodeFlags(picked.map((n) => n.id), { collapsed: !picked.every((n) => n.data.collapsed) })
        return
      }
      if (!(e.ctrlKey || e.metaKey) || e.altKey || typing()) return
      const key = e.key.toLowerCase()
      if (key === 'f' && mouseInside.current) {
        e.preventDefault()
        setFinding(true)
      } else if (key === 'c' || key === 'x') {
        copiedNow = copy(key === 'x') !== null
      } else if (key === 'v') {
        keepExternalNext.current = e.shiftKey
        pasteEventSeen = false
        // the paste event (system clipboard) normally follows; if it never comes, use our copy
        setTimeout(() => { if (!pasteEventSeen && lastClip) paste(lastClip, keepExternalNext.current) }, 80)
      } else if (key === 'd') {  // duplicate next to the original, still connected to its inputs
        e.preventDefault()
        const clip = useStore.getState().copySelection()
        if (clip) report(`Duplicated ${plural(useStore.getState().pasteClip(clip, null, true))}`)
      } else if (key === 'g') {
        e.preventDefault()
        const s = useStore.getState()
        if (e.shiftKey) {
          const g = s.nodes.find((n) => n.selected && n.data.typeId.startsWith(GROUP_PREFIX))
          if (g) s.ungroup(g.id)
        } else {
          groupSelection()
        }
      }
    }
    window.addEventListener('copy', onCopyEvent)
    window.addEventListener('cut', onCopyEvent)
    window.addEventListener('paste', onPasteEvent)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('copy', onCopyEvent)
      window.removeEventListener('cut', onCopyEvent)
      window.removeEventListener('paste', onPasteEvent)
      window.removeEventListener('keydown', onKey)
    }
  }, [rf])

  // Toolbar "+ Node" (or the 3D view's menu): open the palette in the middle of the canvas.
  const paletteRequest = useStore((s) => s.paletteRequest)
  useEffect(() => {
    if (!(paletteSignal || paletteRequest) || !container.current) return
    const r = container.current.getBoundingClientRect()
    setPalette({ x: r.left + r.width / 2 - 140, y: r.top + r.height / 3 })
  }, [paletteSignal, paletteRequest])

  const isValidConnection: IsValidConnection = useCallback((c) => {
    const s = useStore.getState()
    if (!s.catalog || c.source === c.target) return false
    const out = portType(c.source, c.sourceHandle, 'source')
    const inp = portType(c.target, c.targetHandle, 'target')
    if (!out || !inp) return false
    if (!(s.catalog.compatibility[inp] ?? [inp]).includes(out)) return false
    return !createsCycle(s.edges, c.source, c.target)
  }, [])

  // A wire picked up by its input end (wires.pickUpWire, or its end dragged): dropped on an input it moves
  // there (onReconnect); on an output, that output feeds the input instead; on empty canvas it is removed;
  // anywhere else (a card, an input it does not fit, the minimap, outside the graph) it stays as it was.
  const reconnecting = useRef<{ id: string; moved: boolean } | null>(null)
  const realEdge = (id: string) => useStore.getState().edges.find((e) => e.id === id)
  const onReconnect = useCallback((shown: Edge, c: Connection) => {
    if (reconnecting.current) reconnecting.current.moved = true
    const old = realEdge(shown.id)
    if (!old) return
    // the end it was picked up by moves; the other stays (also when a folded frame's bar stands in for it)
    useStore.getState().reconnectEdge(old.id, { ...c, source: old.source, sourceHandle: old.sourceHandle ?? null })
  }, [])
  const onReconnectEnd = useCallback((event: MouseEvent | TouchEvent, shown: Edge, _end: unknown, state: FinalConnectionState) => {
    const r = reconnecting.current
    reconnecting.current = null
    const old = realEdge(shown.id)
    if (r?.moved || !old) return
    // the port it was dropped on or near (React Flow snaps within its connection radius, valid or not)
    const to = state.toHandle
    if (to?.type === 'source' && to.id) {
      const c = { source: to.nodeId, sourceHandle: to.id, target: old.target, targetHandle: old.targetHandle ?? null }
      if (isValidConnection(c)) useStore.getState().reconnectEdge(old.id, c)
      return
    }
    const pt = 'changedTouches' in event ? event.changedTouches[0] : event
    const below = document.elementFromPoint(pt.clientX, pt.clientY)
    if (to || !below?.closest('.react-flow') || below.closest('.react-flow__node, .react-flow__panel, .react-flow__handle')) return
    onEdgesChange([{ type: 'remove', id: old.id }])
    useStore.getState().addLog(`Disconnected ${old.target}.${old.targetHandle} from ${old.source} (Ctrl+Z puts it back)`)
  }, [onEdgesChange])

  const onConnectEnd = useCallback((event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
    if (reconnecting.current) return   // a picked-up wire (onReconnectEnd)
    if (state.isValid || !state.fromHandle || state.toHandle) return
    const pt = 'changedTouches' in event ? event.changedTouches[0] : event
    const side = state.fromHandle.type
    const type = portType(state.fromHandle.nodeId, state.fromHandle.id, side)
    if (!type || !state.fromHandle.id) return
    setPalette({
      x: pt.clientX, y: pt.clientY,
      filter: side === 'source' ? { fromOutput: type } : { fromInput: type },
      from: { nodeId: state.fromHandle.nodeId, handle: state.fromHandle.id, side, type },
    })
  }, [])

  const addFromPalette = (typeId: string) => {
    if (!palette) return
    const s = useStore.getState()
    const position = rf.screenToFlowPosition({ x: palette.x, y: palette.y })
    const id = s.addNode(typeId, palette.from?.side === 'target' || palette.insert ? { x: position.x - 130, y: position.y - 20 } : position)
    const spec = useStore.getState().specs[typeId]
    const compat = s.catalog?.compatibility ?? {}
    if (palette.insert && spec) {
      // on a wire: its source into the new node, the new node into its target (replacing the wire)
      const w = palette.insert
      const input = compatibleInput(spec, w.outType, compat)
      const output = compatibleOutput(spec, w.inType, compat)
      if (input) connect({ source: w.source, sourceHandle: w.sourceHandle, target: id, targetHandle: input.name } as Connection)
      if (output) connect({ source: id, sourceHandle: output.name, target: w.target, targetHandle: w.targetHandle } as Connection)
    } else if (palette.from && spec) {
      const f = palette.from
      if (f.side === 'source') {
        const port = compatibleInput(spec, f.type, compat)
        if (port) connect({ source: f.nodeId, sourceHandle: f.handle, target: id, targetHandle: port.name } as Connection)
      } else {
        const port = compatibleOutput(spec, f.type, compat)
        if (port) connect({ source: id, sourceHandle: port.name, target: f.nodeId, targetHandle: f.handle } as Connection)
      }
    }
    setPalette(null)
  }

  const addPreset = async (name: string) => {
    if (!palette) return
    const at = rf.screenToFlowPosition({ x: palette.x, y: palette.y })
    setPalette(null)
    try {
      const clip = await api.preset(name)
      const n = clip ? useStore.getState().pasteClip(clip, at, false) : 0
      useStore.getState().addLog(n ? `Added preset "${name}"` : `Preset "${name}" has no nodes this editor knows`, n ? 'info' : 'error')
    } catch (err) {
      useStore.getState().addLog(`Could not load preset "${name}": ${(err as Error).message}`, 'error')
    }
  }

  return (
    <div className="editor" ref={container}
      onMouseMove={(e) => { mouse.current = { x: e.clientX, y: e.clientY }; mouseInside.current = true }}
      onMouseLeave={() => { mouseInside.current = false }}
      onDragOver={(e) => {  // a tile of the node library: drop it where it should go
        if (!e.dataTransfer.types.includes(NODE_DRAG_TYPE)) return
        e.preventDefault()
        e.dataTransfer.dropEffect = 'copy'
      }}
      onDrop={(e) => {
        const payload = e.dataTransfer.getData(NODE_DRAG_TYPE)
        if (!payload) return
        e.preventDefault()
        void addFromLibrary(payload, rf.screenToFlowPosition({ x: e.clientX, y: e.clientY }))
      }}
      onKeyDown={(e) => {
        const target = e.target as HTMLElement
        if (e.key === 'Tab' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) {
          e.preventDefault()
          setPalette({ ...mouse.current })
        }
      }}
      tabIndex={-1}>
      <ReactFlow
        nodes={flowNodes} edges={shownEdges} nodeTypes={nodeTypes}
        onNodeMouseEnter={(_, node) => hoverWires({ node: node.id })} onNodeMouseLeave={() => hoverWires(null)}
        onEdgeMouseEnter={(_, edge) => hoverWires({ edge: edge.id })} onEdgeMouseLeave={() => hoverWires(null)}
        onNodesChange={onFlowNodesChange} onEdgesChange={onEdgesChange} onConnect={connect}
        isValidConnection={isValidConnection} onConnectEnd={onConnectEnd}
        edgesReconnectable={false} onReconnect={onReconnect} onReconnectEnd={onReconnectEnd}
        onReconnectStart={(_, edge) => { reconnecting.current = { id: edge.id, moved: false } }}
        onPaneContextMenu={(e) => { e.preventDefault(); setPalette({ x: e.clientX, y: e.clientY }) }}
        onNodeContextMenu={(e, node) => {
          e.preventDefault()
          if (node.type === 'engine' || node.type === 'graphFrame') openNodeMenu(e.clientX, e.clientY, node.id)
        }}
        onSelectionContextMenu={(e, picked) => {
          e.preventDefault()
          const first = picked.find((n) => n.type === 'engine') ?? picked[0]
          if (first) openNodeMenu(e.clientX, e.clientY, first.id)
        }}
        onEdgeContextMenu={(e, edge) => { e.preventDefault(); openEdgeMenu(e.clientX, e.clientY, edge) }}
        onEdgeDoubleClick={(e, edge) => insertOn(edge, e.clientX, e.clientY)}
        deleteKeyCode={['Delete', 'Backspace']}
        colorMode="dark" fitView fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
        minZoom={0.15} maxZoom={2} proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{ type: 'default' }}
        zoomOnDoubleClick={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="#3a3d44" />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable nodeColor={(n) => {
          const s = useStore.getState()
          if (n.type === 'graphFrame') return `${s.graphFrames[n.id]?.color ?? '#8a8f98'}40`
          return CATEGORY_COLORS[s.specs[(n.data as { typeId: string }).typeId]?.category] ?? CATEGORY_COLORS.Groups
        }} maskColor="rgba(20,21,24,0.7)" style={{ background: '#1f2125' }} />
      </ReactFlow>
      {nodes.length === 0 && (
        <div className="editor-empty">Right-click or press Tab to add a node</div>
      )}
      <SelectionBar onFrame={() => frameIds(pickedForFrame())} />
      <HelpPanel />
      {finding ? <GraphFind onClose={() => setFinding(false)} /> : nodes.length > 0 && (
        <div className="graph-tools">
          <button onClick={() => setFinding(true)} title="Find a node by name or type (Ctrl+F over the graph)">Find</button>
          <button onClick={() => {
            const picked = useStore.getState().nodes.filter((n) => n.selected && n.type === 'engine').map((n) => n.id)
            useStore.getState().tidy(picked.length > 1 ? picked : [])
            if (picked.length < 2) setTimeout(() => rf.fitView({ padding: 0.08, maxZoom: 1, duration: 200 }), 50)
          }} title="Lay the selected nodes (or all) out in columns along the links (Undo puts them back)">Tidy</button>
          <button onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect()
            setMenu({ x: r.left, y: r.bottom + 4, entries: viewMenu() })
          }} title="How cards and wires are drawn: simple cards when zoomed out, highlight the selected node's wires, shared and long wires as labels">View</button>
        </div>
      )}
      {palette && (
        <NodePalette x={palette.x} y={palette.y} filter={palette.filter}
          onPick={addFromPalette} onPickPreset={addPreset} onClose={() => setPalette(null)} />
      )}
      {menu && <ViewportMenu x={menu.x} y={menu.y} entries={menu.entries} onClose={() => setMenu(null)} />}
      {prompt && <NamePrompt prompt={prompt} onClose={() => setPrompt(null)} />}
    </div>
  )
}
