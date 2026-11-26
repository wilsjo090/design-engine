import { useLayoutEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useStore as useFlow } from '@xyflow/react'
import { useShallow } from 'zustand/react/shallow'
import { useStore } from '../store'
import { useWires } from '../wires'

/** px between a peek and the graph's edge */
const PAD = 14
/** an end is "off screen" when less than this share of its card is in view */
const SHOWN = 0.6
/** a peek is never taller than this share of the graph (scaled down to fit) */
const MOST_HEIGHT = 0.8
const ARROWS = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']

interface End { id: string; port: string | null | undefined; side: 'source' | 'target' }

/** While a wire (or the label standing for one) is hovered, each of its ends that lies off screen is shown
 *  as a copy of its card, pinned to the side of the graph the node lies on, its wired port ringed: so
 *  what a long or labelled wire comes from or goes to can be read without panning there. The copy is a
 *  picture (no pointer events): the wire keeps the hover, and moving off it takes the peek away. */
export function WirePeek() {
  const edgeId = useWires((w) => w.hover?.edge)
  const edge = useStore((s) => (edgeId ? s.edges.find((e) => e.id === edgeId) : undefined))
  if (!edge) return null
  const ends: End[] = [
    { id: edge.source, port: edge.sourceHandle, side: 'source' },
    { id: edge.target, port: edge.targetHandle, side: 'target' },
  ]
  return <>{ends.map((end) => <Peek key={`${end.side}:${end.id}`} end={end} other={end.side === 'source' ? edge.target : edge.source} />)}</>
}

function Peek({ end, other }: { end: End; other: string }) {
  const node = useStore((s) => s.nodes.find((n) => n.id === end.id))
  const otherNode = useStore((s) => s.nodes.find((n) => n.id === other))
  const [tx, ty, zoom] = useFlow((s) => s.transform)
  const { width, height, dom } = useFlow(useShallow((s) => ({ width: s.width, height: s.height, dom: s.domNode })))
  const holder = useRef<HTMLDivElement>(null)

  // where the node is on screen, and whether enough of it is in view
  const w = node?.measured?.width ?? 260, h = node?.measured?.height ?? 160
  const x = node ? node.position.x * zoom + tx : 0, y = node ? node.position.y * zoom + ty : 0
  const seenW = Math.max(0, Math.min(x + w * zoom, width) - Math.max(x, 0))
  const seenH = Math.max(0, Math.min(y + h * zoom, height) - Math.max(y, 0))
  const off = !!node && !node.hidden && seenW * seenH < SHOWN * w * zoom * h * zoom

  // the copy of the card, made again when the node changes (values, size, results)
  useLayoutEffect(() => {
    const box = holder.current
    if (!box || !off || !dom) return
    const card = dom.querySelector(`.react-flow__node[data-id="${CSS.escape(end.id)}"] > *`)
    if (!(card instanceof HTMLElement)) return
    const copy = card.cloneNode(true) as HTMLElement
    // typed values live in the fields' properties, not their attributes
    const from = card.querySelectorAll('input, select, textarea'), to = copy.querySelectorAll('input, select, textarea')
    from.forEach((f, i) => {
      const t = to[i]
      if (f instanceof HTMLSelectElement && t instanceof HTMLSelectElement) t.selectedIndex = f.selectedIndex
      else if ((f instanceof HTMLInputElement && t instanceof HTMLInputElement) || (f instanceof HTMLTextAreaElement && t instanceof HTMLTextAreaElement)) {
        t.value = f.value
        if (f instanceof HTMLInputElement && t instanceof HTMLInputElement) t.checked = f.checked
      }
    })
    copy.querySelectorAll('.react-flow__handle').forEach((el) => {
      const handle = el as HTMLElement
      if (handle.dataset.handleid === end.port && handle.classList.contains(end.side)) {
        handle.classList.add('peek-port')
        handle.closest('.enode-row')?.classList.add('peek-row')
      }
      // React Flow finds handles by these: the copy's must not stand in for the card's own
      handle.removeAttribute('data-id')
      handle.removeAttribute('data-nodeid')
    })
    copy.classList.remove('selected')
    copy.querySelectorAll('[id]').forEach((el) => el.removeAttribute('id'))
    box.replaceChildren(copy)
  }, [off, dom, end.id, end.port, end.side, node])

  if (!off || !node || !dom) return null

  // pinned to the graph's side towards the node: along the line from the wire's other end (or the
  // middle of the view) to the node, clamped inside the view
  const rect = dom.getBoundingClientRect()
  const scale = Math.min(1, (height * MOST_HEIGHT) / h, (width - 2 * PAD) / w)
  const pw = w * scale, ph = h * scale
  const cx = x + (w * zoom) / 2, cy = y + (h * zoom) / 2
  let fx = width / 2, fy = height / 2
  if (otherNode) {
    const ox = otherNode.position.x * zoom + tx + ((otherNode.measured?.width ?? 260) * zoom) / 2
    const oy = otherNode.position.y * zoom + ty + ((otherNode.measured?.height ?? 160) * zoom) / 2
    if (ox > 0 && ox < width && oy > 0 && oy < height) { fx = ox; fy = oy }
  }
  const left = Math.min(Math.max(cx - pw / 2, PAD), width - pw - PAD)
  const top = Math.min(Math.max(cy - ph / 2, PAD + 28), height - ph - PAD)
  const angle = Math.atan2(cy - fy, cx - fx)
  const arrow = ARROWS[(Math.round(angle / (Math.PI / 4)) + 8) % 8]
  const away = Math.round(Math.hypot(cx - fx, cy - fy) / zoom)

  return createPortal(
    <div className="wire-peek" style={{ left: rect.left + left, top: rect.top + top - 24, width: pw }}>
      <div className="wire-peek-caption">
        <span className="wire-peek-arrow">{arrow}</span>
        {end.side === 'source' ? 'from' : 'to'} <b>{end.id}</b>, off screen
        <span className="wire-peek-away">{away > 2000 ? `${(away / 1000).toFixed(1)}k` : away} px</span>
      </div>
      <div className="wire-peek-frame" style={{ width: pw, height: ph }}>
        <div className="wire-peek-card" style={{ width: w, height: h, transform: `scale(${scale})` }} ref={holder} />
      </div>
    </div>,
    document.body,
  )
}
