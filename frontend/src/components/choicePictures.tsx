import type { ReactNode } from 'react'

/** Small pictures (24 x 24) of choices that are easier to see than to read: a pattern's shapes and
 *  layouts and a frame's bar profiles. Keyed by input name, then by choice; a picker is shown only
 *  when every choice has one. */

/** A regular polygon's points: `n` corners on radius `r` around (cx, cy), the first at `turn` degrees
 *  from straight up. */
function ring(n: number, r: number, turn = 0, cx = 12, cy = 12) {
  return Array.from({ length: n }, (_, k) => {
    const a = ((turn + (360 * k) / n) * Math.PI) / 180
    return `${(cx + r * Math.sin(a)).toFixed(2)},${(cy - r * Math.cos(a)).toFixed(2)}`
  }).join(' ')
}

const star = Array.from({ length: 10 }, (_, k) => {
  const a = (k * Math.PI) / 5, r = k % 2 ? 4 : 9.5
  return `${(12 + r * Math.sin(a)).toFixed(2)},${(12.8 - r * Math.cos(a)).toFixed(2)}`
}).join(' ')

const dots = (points: [number, number][]) => points.map(([x, y], k) => <circle key={k} cx={x} cy={y} r={1.9} />)

const spiral: [number, number][] = Array.from({ length: 9 }, (_, k) => {
  const t = 0.9 + k * 0.75
  return [12 + 1.05 * t * Math.cos(t * 1.1), 12 + 1.05 * t * Math.sin(t * 1.1)]
})

const SHAPES: Record<string, ReactNode> = {
  none: <><circle cx={12} cy={12} r={7.5} className="pic-line" strokeDasharray="2.5 2.5" /></>,
  round: <circle cx={12} cy={12} r={8} />,
  square: <rect x={4.5} y={4.5} width={15} height={15} />,
  hexagon: <polygon points={ring(6, 8.8, 30)} />,
  triangle: <polygon points={ring(3, 9.5, 0, 12, 13.5)} />,
  octagon: <polygon points={ring(8, 8.8, 22.5)} />,
  polygon: <polygon points={ring(5, 9, 0, 12, 12.6)} />,
  slot: <rect x={2.5} y={8} width={19} height={8} rx={4} />,
  diamond: <polygon points="2,12 12,6.5 22,12 12,17.5" />,
  cross: <path d="M9.5 3h5v6.5H21v5h-6.5V21h-5v-6.5H3v-5h6.5z" />,
  rectangle: <rect x={2.5} y={7} width={19} height={10} />,
  star: <polygon points={star} />,
  chevron: <polygon points="3,7 12,13 21,7 21,12.5 12,18.5 3,12.5" />,
  voronoi: <path className="pic-line" d="M7 8l6-2 3 5-4 5-6-2zM7 8L3 4M13 6l1-4M16 11l6-2M12 16l1 6M6 14l-4 3" />,
  delaunay: <path className="pic-line" d="M3 17L8 4l9 1 4 10-9 6zM11 11L3 17M11 11L8 4M11 11l6-6M11 11l10 4M11 11l1 10" />,
}

const LAYOUTS: Record<string, ReactNode> = {
  grid: dots([[5, 5], [12, 5], [19, 5], [5, 12], [12, 12], [19, 12], [5, 19], [12, 19], [19, 19]]),
  staggered: dots([[4, 5], [12, 5], [20, 5], [8, 12], [16, 12], [4, 19], [12, 19], [20, 19]]),
  rings: dots([[12, 12], ...Array.from({ length: 8 }, (_, k): [number, number] =>
    [12 + 8.5 * Math.sin((k * Math.PI) / 4), 12 - 8.5 * Math.cos((k * Math.PI) / 4)])]),
  spiral: dots(spiral),
}

/** Cross-sections, as cut through the bar. */
const PROFILES: Record<string, ReactNode> = {
  round: <circle cx={12} cy={12} r={8.5} />,
  'round tube': <path fillRule="evenodd" d="M12 3.5a8.5 8.5 0 1 1 0 17 8.5 8.5 0 1 1 0-17zm0 4a4.5 4.5 0 1 0 0 9 4.5 4.5 0 1 0 0-9z" />,
  square: <rect x={4} y={4} width={16} height={16} />,
  'square tube': <path fillRule="evenodd" d="M4 4h16v16H4zm4 4v8h8V8z" />,
  rectangle: <rect x={6.5} y={2.5} width={11} height={19} />,
  'rect tube': <path fillRule="evenodd" d="M6.5 2.5h11v19h-11zm3.5 3.5v12h4V6z" />,
  I: <path d="M5 3h14v3.5h-5v11h5V21H5v-3.5h5v-11H5z" />,
  angle: <path d="M5 3h4v14h10v4H5z" />,
}

export const PICTURES: Record<string, Record<string, ReactNode>> = {
  shape: SHAPES,
  morph_to: SHAPES,
  layout: LAYOUTS,
  profile: PROFILES,
}

/** One of the pictures above, drawn at the text's size. */
export function ChoicePicture({ of }: { of: ReactNode }) {
  return (
    <svg className="pic" width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">{of}</svg>
  )
}
