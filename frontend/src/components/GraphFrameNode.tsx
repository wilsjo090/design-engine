import { memo, useEffect, useState } from 'react'
import { Handle, NodeResizer, Position, useUpdateNodeInternals, type Node, type NodeProps } from '@xyflow/react'
import { useStore } from '../store'
import { FRAME_HEADER } from '../graphFrames'
import type { GraphFrame } from '../types'

export type FrameNode = Node<{ frame: GraphFrame; count: number }, 'graphFrame'>

/** A frame in the graph (graphFrames.ts): its title bar drags it with what is in it; selected, its
 *  border resizes it. Folded, it is only the bar, and the wires of the nodes it hides end on it. */
function GraphFrameView({ id, data, selected }: NodeProps<FrameNode>) {
  const { frame, count } = data
  const folded = !!frame.folded
  const [text, setText] = useState<string | null>(null)
  // folding adds or removes the bar's handles: React Flow has to find them again
  const updateNodeInternals = useUpdateNodeInternals()
  useEffect(() => updateNodeInternals(id), [folded, id, updateNodeInternals])
  const finish = (commit: boolean) => {
    const label = text?.trim()
    if (commit && label && label !== frame.label) useStore.getState().updateFrame(id, { label })
    setText(null)
  }
  return (
    <div className={`gframe ${folded ? 'folded' : ''} ${selected ? 'selected' : ''}`} style={{ ['--frame' as string]: frame.color }}>
      <NodeResizer isVisible={selected && !folded} minWidth={160} minHeight={FRAME_HEADER + 40} color={frame.color}
        handleClassName="gframe-handle" lineClassName="gframe-line" />
      <div className="gframe-header" title={folded
        ? `${frame.label}: ${count} node${count === 1 ? '' : 's'} folded away. Drag to move, ▸ (or double-click here) unfolds, right-click for more.`
        : `${frame.label}: drag the bar to move the frame with its nodes; right-click for more`}
        onDoubleClick={folded ? () => useStore.getState().foldFrame(id, false) : undefined}>
        <button className="nodrag collapse-btn" title={folded ? 'Unfold: show its nodes' : 'Fold to this bar (its nodes are hidden, their wires end here)'}
          onClick={() => useStore.getState().foldFrame(id, !folded)}>{folded ? '▸' : '▾'}</button>
        {text === null
          ? <span className="gframe-label" title="Double-click to rename"
              onDoubleClick={(e) => { e.stopPropagation(); setText(frame.label) }}>{frame.label}</span>
          : <input className="nodrag value-input gframe-rename" autoFocus value={text} spellCheck={false}
              onChange={(e) => setText(e.target.value)} onBlur={() => finish(true)}
              onDoubleClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Enter') finish(true); if (e.key === 'Escape') finish(false) }} />}
        {folded && <span className="gframe-count">{count} node{count === 1 ? '' : 's'}</span>}
      </div>
      {folded && (
        <>
          <Handle type="target" position={Position.Left} id="in" isConnectable={false} className="gframe-port" />
          <Handle type="source" position={Position.Right} id="out" isConnectable={false} className="gframe-port" />
        </>
      )}
    </div>
  )
}

export const GraphFrameNode = memo(GraphFrameView)
