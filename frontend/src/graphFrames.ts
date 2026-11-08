import type { EngineEdge, EngineNode, GraphFrame } from './types'

/** Frames in the node graph: labelled boxes behind nodes (recipe `frames`; not the bar frames of Frame
 *  Optimization). Membership is by position: a node belongs to the smallest frame its middle lies in, a
 *  frame to the smallest frame it lies in, so dragging a frame moves what is in it and a node dragged into
 *  a frame joins it. A folded frame is its title bar: it keeps the list of what it hides (`folded`), and
 *  the wires of those nodes end on the bar. */

export type Frames = Record<string, GraphFrame>

export const FRAME_PREFIX = 'frame:'
export const isFrameId = (id: string) => id.startsWith(FRAME_PREFIX)
/** height of a frame's title bar (a folded frame is just that) */
export const FRAME_HEADER = 34
const FOLDED_W = 300
/** room between a frame's border and its nodes */
export const FRAME_PAD = 24
/** a new frame takes the first one no other frame has */
export const FRAME_COLORS: Record<string, string> = {
  Blue: '#5b8def', Green: '#6cb26a', Orange: '#d9894a', Purple: '#9a7ad6',
  Teal: '#45b3a9', Yellow: '#d6b94c', Red: '#d0605e', Grey: '#8a8f98',
}
const NODE_W = 260, NODE_H = 200   // a node's size before React Flow has measured it
const COL_GAP = 60, ROW_GAP = 28   // between columns and between nodes of a column (as tidyPositions)

interface Box { x: number; y: number; w: number; h: number }
const nodeBox = (n: EngineNode): Box =>
  ({ x: n.position.x, y: n.position.y, w: n.measured?.width ?? NODE_W, h: n.measured?.height ?? NODE_H })
/** the box as drawn: a folded frame is its title bar, at most a wide card's width */
export const frameBox = (f: GraphFrame): Box =>
  ({ x: f.pos[0], y: f.pos[1], w: f.folded ? Math.min(f.size[0], FOLDED_W) : f.size[0], h: f.folded ? FRAME_HEADER : f.size[1] })
const holdsMiddle = (outer: Box, b: Box) =>
  b.x + b.w / 2 > outer.x && b.x + b.w / 2 < outer.x + outer.w && b.y + b.h / 2 > outer.y && b.y + b.h / 2 < outer.y + outer.h

export function nextFrameId(frames: Frames) {
  let n = 1
  while (`${FRAME_PREFIX}${n}` in frames) n++
  return `${FRAME_PREFIX}${n}`
}

/** What folded frames hide: node or frame id -> the folded frame whose bar is shown for it (the outermost). */
export function hiddenBy(frames: Frames) {
  const by = new Map<string, string>()
  const hide = (id: string, bar: string) => {
    for (const x of frames[id]?.folded ?? []) {
      if (by.has(x) || x === bar) continue
      by.set(x, bar)
      hide(x, bar)
    }
  }
  const inside = new Set(Object.values(frames).flatMap((f) => f.folded ?? []))
  for (const [id, f] of Object.entries(frames)) if (f.folded && !inside.has(id)) hide(id, id)
  return by
}

/** The shown nodes and frames directly or indirectly inside the unfolded frame `id` (by position): those
 *  whose middle lies in it, frames only if smaller (a frame made round some of a frame's nodes sits flush
 *  with its border, or just over it). */
export function inFrame(id: string, frames: Frames, nodes: EngineNode[], hidden = hiddenBy(frames)) {
  const box = frameBox(frames[id])
  const smaller = (b: Box) => b.w * b.h < box.w * box.h
  return [
    ...nodes.filter((n) => !hidden.has(n.id) && holdsMiddle(box, nodeBox(n))).map((n) => n.id),
    ...Object.entries(frames).filter(([g, f]) => g !== id && !hidden.has(g) && holdsMiddle(box, frameBox(f)) && smaller(frameBox(f)))
      .map(([g]) => g),
  ]
}

/** Everything that moves with frame `id`: what lies in it, or what it hides when folded, and what the
 *  folded frames among those hide. */
export function contents(id: string, frames: Frames, nodes: EngineNode[]) {
  const f = frames[id]
  if (!f) return []
  const out = new Set<string>()
  const add = (x: string) => {
    if (out.has(x) || x === id) return
    out.add(x)
    for (const y of frames[x]?.folded ?? []) add(y)
  }
  ;(f.folded ?? inFrame(id, frames, nodes)).forEach(add)
  return [...out]
}

/** A frame around these nodes and frames (null if none of them is there). */
export function frameAround(ids: string[], frames: Frames, nodes: EngineNode[]): Pick<GraphFrame, 'pos' | 'size'> | null {
  const chosen = new Set(ids)
  const boxes = [...nodes.filter((n) => chosen.has(n.id)).map(nodeBox),
    ...Object.entries(frames).filter(([id]) => chosen.has(id)).map(([, f]) => frameBox(f))]
  if (!boxes.length) return null
  const x0 = Math.min(...boxes.map((b) => b.x)), y0 = Math.min(...boxes.map((b) => b.y))
  const x1 = Math.max(...boxes.map((b) => b.x + b.w)), y1 = Math.max(...boxes.map((b) => b.y + b.h))
  return { pos: [x0 - FRAME_PAD, y0 - FRAME_PAD - FRAME_HEADER], size: [x1 - x0 + 2 * FRAME_PAD, y1 - y0 + 2 * FRAME_PAD + FRAME_HEADER] }
}

/** Frames as React Flow nodes, drawn behind everything: bigger frames further back, so a frame inside
 *  another stays on top of it. A folded one does not take the Delete key: React Flow would delete the
 *  wires drawn to its bar along with it, and those are the hidden nodes' own (the menu removes it).
 *  `measured` is their size: without it React Flow forgets where a node's handles are whenever the node
 *  object is new, and the wires to a folded bar were not drawn. */
export function frameNodes(frames: Frames, nodes: EngineNode[], hidden: Map<string, string>) {
  const byArea = Object.entries(frames).sort(([, a], [, b]) => b.size[0] * b.size[1] - a.size[0] * a.size[1])
  const exists = new Set(nodes.map((n) => n.id))
  return byArea.map(([id, f], i) => {
    const box = frameBox(f)
    const count = f.folded ? contents(id, frames, nodes).filter((x) => exists.has(x)).length : 0
    return {
      id, type: 'graphFrame' as const, position: { x: box.x, y: box.y }, width: box.w, height: box.h,
      measured: { width: box.w, height: box.h },
      zIndex: -2000 + i, selected: !!f.selected, hidden: hidden.has(id), connectable: false, deletable: !f.folded,
      data: { typeId: '', inputs: {}, frame: f, count },
    }
  })
}

/** Wires as drawn with folded frames: a wire to a hidden node ends on its frame's bar; one inside a
 *  folded frame is not drawn. */
export function foldedEdges(edges: EngineEdge[], hidden: Map<string, string>) {
  if (!hidden.size) return edges
  return edges.map((e) => {
    const s = hidden.get(e.source), t = hidden.get(e.target)
    if (!s && !t) return e
    if (s && s === t) return { ...e, hidden: true }
    return { ...e, ...(s ? { source: s, sourceHandle: 'out' } : {}), ...(t ? { target: t, targetHandle: 'in' } : {}) }
  })
}

// --- tidying with frames ----------------------------------------------------------------------

interface Unit { id: string; w: number; h: number; y: number; nodes: Set<string> }

/** Columns along the links among `units` (a link between nodes of two units links the units), each
 *  column as wide as its widest unit, ordered by where their inputs sit; positions from (0, 0). */
function columns(units: Unit[], edges: EngineEdge[]) {
  const unitOf = new Map<string, string>()
  for (const u of units) for (const n of u.nodes) unitOf.set(n, u.id)
  const links: [string, string][] = []
  for (const e of edges) {
    const a = unitOf.get(e.source), b = unitOf.get(e.target)
    if (a && b && a !== b) links.push([a, b])
  }
  const depth: Record<string, number> = {}
  const visit = (id: string, stack: Set<string>): number => {
    if (depth[id] !== undefined) return depth[id]
    if (stack.has(id)) return 0
    stack.add(id)
    let d = 0
    for (const [a, b] of links) if (b === id) d = Math.max(d, visit(a, stack) + 1)
    stack.delete(id)
    return (depth[id] = d)
  }
  units.forEach((u) => visit(u.id, new Set()))
  const cols: Unit[][] = []
  for (const u of units) (cols[depth[u.id]] ??= []).push(u)
  const placed: Record<string, { x: number; y: number }> = {}
  let x = 0
  for (const col of cols) {
    if (!col) continue
    const key = (u: Unit) => {
      const ys = links.filter(([a, b]) => b === u.id && placed[a]).map(([a]) => placed[a].y)
      return ys.length ? ys.reduce((s, v) => s + v, 0) / ys.length : u.y
    }
    let y = 0
    for (const u of [...col].sort((a, b) => key(a) - key(b))) {
      placed[u.id] = { x, y }
      y += u.h + ROW_GAP
    }
    x += Math.max(...col.map((u) => u.w)) + COL_GAP
  }
  return placed
}

/** Tidy positions for what is in frame `root` (null: the whole graph): each unfolded frame's contents
 *  laid out in columns first and the frame fitted round them, then the frames and nodes around them laid
 *  out in columns as blocks. `root` keeps its corner (the whole graph its top-left); what folded frames
 *  hide moves with them. */
export function tidyFrames(nodes: EngineNode[], edges: EngineEdge[], frames: Frames, root: string | null) {
  const hidden = hiddenBy(frames)
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const shownFrames = Object.keys(frames).filter((id) => !hidden.has(id))
  // each shown node and frame belongs to the smallest unfolded frame it is in
  const area = (id: string) => frames[id].size[0] * frames[id].size[1]
  const open = shownFrames.filter((id) => !frames[id].folded).sort((a, b) => area(a) - area(b))
  const parent = new Map<string, string | null>()
  for (const item of [...nodes.filter((n) => !hidden.has(n.id)).map((n) => n.id), ...shownFrames]) {
    parent.set(item, open.find((f) => f !== item && inFrame(f, frames, nodes, hidden).includes(item)) ?? null)
  }
  const children = (id: string | null) => [...parent].filter(([, p]) => p === id).map(([c]) => c)
  /** the nodes a unit stands for (links into them link the unit) */
  const nodesOf = (id: string) => new Set(frames[id] ? contents(id, frames, nodes).filter((x) => byId.has(x)) : [id])

  const rel: Record<string, { x: number; y: number }> = {}   // relative to the parent's content corner
  const size: Record<string, [number, number]> = {}
  const lay = (container: string | null) => {
    const units: Unit[] = children(container).map((id) => {
      if (frames[id] && !frames[id].folded) lay(id)
      const f = frames[id]
      const [w, h] = f ? (f.folded ? [frameBox(f).w, FRAME_HEADER] : size[id]) : [nodeBox(byId.get(id)!).w, nodeBox(byId.get(id)!).h]
      return { id, w, h, y: f ? f.pos[1] : byId.get(id)!.position.y, nodes: nodesOf(id) }
    })
    Object.assign(rel, columns(units, edges))
    if (container) {
      const w = Math.max(0, ...units.map((u) => rel[u.id].x + u.w)), h = Math.max(0, ...units.map((u) => rel[u.id].y + u.h))
      size[container] = units.length
        ? [Math.max(w + 2 * FRAME_PAD, 200), h + 2 * FRAME_PAD + FRAME_HEADER]
        : frames[container].size
    }
  }
  lay(root)

  // absolute positions: from the root's corner down through the frames
  const nodePos: Record<string, { x: number; y: number }> = {}
  const framePatch: Record<string, Pick<GraphFrame, 'pos' | 'size'>> = {}
  const tops = children(root)
  const corner = root
    ? { x: frames[root].pos[0] + FRAME_PAD, y: frames[root].pos[1] + FRAME_HEADER + FRAME_PAD }
    : { x: Math.min(...tops.map((id) => frames[id]?.pos[0] ?? byId.get(id)!.position.x)),
        y: Math.min(...tops.map((id) => frames[id]?.pos[1] ?? byId.get(id)!.position.y)) }
  if (root) framePatch[root] = { pos: frames[root].pos, size: size[root] }
  const put = (id: string, x: number, y: number) => {
    const f = frames[id]
    if (!f) { nodePos[id] = { x, y }; return }
    framePatch[id] = { pos: [x, y], size: f.folded ? f.size : size[id] }
    if (f.folded) {   // what it hides moves with it
      const dx = x - f.pos[0], dy = y - f.pos[1]
      for (const c of contents(id, frames, nodes)) {
        const n = byId.get(c)
        if (n) nodePos[c] = { x: n.position.x + dx, y: n.position.y + dy }
        else if (frames[c]) framePatch[c] = { pos: [frames[c].pos[0] + dx, frames[c].pos[1] + dy], size: frames[c].size }
      }
    } else {
      for (const c of children(id)) put(c, x + FRAME_PAD + rel[c].x, y + FRAME_HEADER + FRAME_PAD + rel[c].y)
    }
  }
  for (const id of tops) put(id, corner.x + rel[id].x, corner.y + rel[id].y)
  return { nodes: nodePos, frames: framePatch }
}
