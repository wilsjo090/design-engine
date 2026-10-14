import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Background, BackgroundVariant, Controls, MiniMap, ReactFlow, useReactFlow,
  type Connection, type Edge, type FinalConnectionState, type IsValidConnection,
} from '@xyflow/react'
import { useShallow } from 'zustand/react/shallow'
import { api } from '../api'
import { isClip, useStore, type Clip } from '../store'
import { GROUP_IN_ID, GROUP_OUT_ID } from '../graphConvert'
import { EngineNodeComponent, GroupIONode } from './EngineNode'
import { HelpPanel } from './HelpPanel'
import { NodePalette, compatibleInput, compatibleOutput, type PaletteFilter } from './NodePalette'
import { CATEGORY_COLORS, GROUP_PREFIX } from '../types'

const nodeTypes = { engine: EngineNodeComponent, groupIn: GroupIONode, groupOut: GroupIONode }

interface PaletteState {
  x: number
  y: number
  filter?: PaletteFilter
  /** connect the new node to this handle */
  from?: { nodeId: string; handle: string; side: 'source' | 'target'; type: string }
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
  const spec = s.specs[s.nodes.find((n) => n.id === nodeId)?.data.typeId ?? '']
  return (side === 'source' ? spec?.outputs : spec?.inputs)?.find((p) => p.name === handle)?.type
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

function SelectionBar() {
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
  const mouse = useRef({ x: 300, y: 300 })
  const container = useRef<HTMLDivElement>(null)

  // Frame the whole graph after a recipe loads (once React Flow has measured the nodes).
  const loadCount = useStore((s) => s.loadCount)
  useEffect(() => {
    if (!loadCount) return
    const t = setTimeout(() => rf.fitView({ padding: 0.08, maxZoom: 1, duration: 200 }), 80)
    return () => clearTimeout(t)
  }, [loadCount, rf])

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
      if (!(e.ctrlKey || e.metaKey) || e.altKey || typing()) return
      const key = e.key.toLowerCase()
      if (key === 'c' || key === 'x') {
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

  // Toolbar "+ Node": open the palette in the middle of the canvas.
  useEffect(() => {
    if (!paletteSignal || !container.current) return
    const r = container.current.getBoundingClientRect()
    setPalette({ x: r.left + r.width / 2 - 140, y: r.top + r.height / 3 })
  }, [paletteSignal])

  const isValidConnection: IsValidConnection = useCallback((c) => {
    const s = useStore.getState()
    if (!s.catalog || c.source === c.target) return false
    const out = portType(c.source, c.sourceHandle, 'source')
    const inp = portType(c.target, c.targetHandle, 'target')
    if (!out || !inp) return false
    if (!(s.catalog.compatibility[inp] ?? [inp]).includes(out)) return false
    return !createsCycle(s.edges, c.source, c.target)
  }, [])

  const onConnectEnd = useCallback((event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
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
    const id = s.addNode(typeId, palette.from?.side === 'target' ? { x: position.x - 260, y: position.y - 20 } : position)
    const spec = useStore.getState().specs[typeId]
    const compat = s.catalog?.compatibility ?? {}
    if (palette.from && spec) {
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
      onKeyDown={(e) => {
        const target = e.target as HTMLElement
        if (e.key === 'Tab' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) {
          e.preventDefault()
          setPalette({ ...mouse.current })
        }
      }}
      tabIndex={-1}>
      <ReactFlow
        nodes={nodes} edges={edges} nodeTypes={nodeTypes}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={connect}
        isValidConnection={isValidConnection} onConnectEnd={onConnectEnd}
        onPaneContextMenu={(e) => { e.preventDefault(); setPalette({ x: e.clientX, y: e.clientY }) }}
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
          return CATEGORY_COLORS[s.specs[(n.data as { typeId: string }).typeId]?.category] ?? CATEGORY_COLORS.Groups
        }} maskColor="rgba(20,21,24,0.7)" style={{ background: '#1f2125' }} />
      </ReactFlow>
      {nodes.length === 0 && (
        <div className="editor-empty">Right-click or press Tab to add a node</div>
      )}
      <SelectionBar />
      <HelpPanel />
      {palette && (
        <NodePalette x={palette.x} y={palette.y} filter={palette.filter}
          onPick={addFromPalette} onPickPreset={addPreset} onClose={() => setPalette(null)} />
      )}
    </div>
  )
}
