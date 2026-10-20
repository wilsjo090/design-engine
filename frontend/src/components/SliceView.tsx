import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import { isFieldNode, useStore } from '../store'
import type { Axis, ColorMap, Paint, SliceView } from '../types'

const STOPS: Record<ColorMap, string[]> = {
  // black valleys, cyan crests
  bands: ['#000000', '#00585a', '#00c8c8', '#00ffff'],
  // "turbo": easy to read differences
  heat: ['#30123b', '#4662d7', '#36aaf9', '#1ae4b6', '#72fe5e', '#c8ef34', '#faba39', '#f66b19', '#ca2a04', '#7a0403'],
  // blue below zero, red above: for distances (inside / outside)
  signed: ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b'],
  // blue to red as in FEA programs: turbo without its near-black ends (a safe part read as black)
  rainbow: ['#3b6fe0', '#36aaf9', '#1ae4b6', '#72fe5e', '#c8ef34', '#faba39', '#f66b19', '#d92b04'],
}

export const COLOR_MAP_LABELS: Record<ColorMap, string> = { bands: 'Bands', heat: 'Heat', signed: '± Signed', rainbow: 'Rainbow' }

export function lut(cmap: ColorMap) {
  const stops = STOPS[cmap].map((h) => new THREE.Color(h))
  const out = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1)
    const k = Math.min(Math.floor(t), stops.length - 2)
    const c = stops[k].clone().lerp(stops[k + 1], t - k)
    out[i * 3] = Math.round(c.r * 255)
    out[i * 3 + 1] = Math.round(c.g * 255)
    out[i * 3 + 2] = Math.round(c.b * 255)
  }
  return out
}

export const gradientCss = (cmap: ColorMap) => `linear-gradient(90deg, ${STOPS[cmap].join(', ')})`

/** Field value -> 0..1 colour position. Signed maps zero to the middle colour. */
export function normalise(cmap: ColorMap, min: number, max: number) {
  if (cmap === 'signed' && min < 0 && max > 0) {
    return (v: number) => (v < 0 ? 0.5 - 0.5 * (v / min) : 0.5 + 0.5 * (v / max))
  }
  const span = max - min || 1
  return (v: number) => (v - min) / span
}

function colorize(slice: SliceView, cmap: ColorMap) {
  const table = lut(cmap)
  const norm = normalise(cmap, slice.min, slice.max)
  const rgba = new Uint8Array(slice.width * slice.height * 4)
  for (let i = 0; i < slice.values.length; i++) {
    const v = slice.values[i]
    if (Number.isNaN(v)) continue // transparent where the field does not reach
    const k = Math.max(0, Math.min(255, Math.round(norm(v) * 255))) * 3
    rgba[i * 4] = table[k]
    rgba[i * 4 + 1] = table[k + 1]
    rgba[i * 4 + 2] = table[k + 2]
    rgba[i * 4 + 3] = 255
  }
  return rgba
}

const UNIT = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)]

export function planeAxes(axis: Axis) {
  const a = 'xyz'.indexOf(axis)
  const [u, v] = [0, 1, 2].filter((i) => i !== a)
  return { a, u, v }
}

/** World-space box of a slice plane (for framing the camera). */
export function sliceBox(slice: SliceView) {
  const { a, u, v } = planeAxes(slice.axis)
  const lo = [0, 0, 0], hi = [0, 0, 0]
  lo[u] = slice.rect[0]; hi[u] = slice.rect[2]
  lo[v] = slice.rect[1]; hi[v] = slice.rect[3]
  lo[a] = hi[a] = slice.position
  return new THREE.Box3(new THREE.Vector3(...lo), new THREE.Vector3(...hi))
}

export interface HoverValue {
  value: number
  point: [number, number, number]
  /** the view hovered (when it is a node's body) */
  node?: string
}

export function SliceObject({ slice, cmap, onHover }: {
  slice: SliceView; cmap: ColorMap; onHover: (h: HoverValue | null) => void
}) {
  const texture = useMemo(() => {
    const t = new THREE.DataTexture(colorize(slice, cmap), slice.width, slice.height, THREE.RGBAFormat)
    t.magFilter = THREE.LinearFilter
    t.minFilter = THREE.LinearFilter
    t.colorSpace = THREE.SRGBColorSpace
    t.needsUpdate = true
    return t
  }, [slice, cmap])
  useEffect(() => () => texture.dispose(), [texture])

  const { a, u, v } = planeAxes(slice.axis)
  const [u0, v0, u1, v1] = slice.rect
  const position: [number, number, number] = [0, 0, 0]
  position[u] = (u0 + u1) / 2
  position[v] = (v0 + v1) / 2
  position[a] = slice.position
  const quaternion = useMemo(() => {
    const U = UNIT[u], V = UNIT[v]
    const N = new THREE.Vector3().crossVectors(U, V)
    return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(U, V, N))
  }, [u, v])

  const hover = (e: ThreeEvent<PointerEvent>) => {
    if (!e.uv) return
    e.stopPropagation()
    const col = Math.round(e.uv.x * (slice.width - 1))
    const row = Math.round(e.uv.y * (slice.height - 1))
    const value = slice.values[row * slice.width + col]
    if (Number.isNaN(value)) return onHover(null)
    onHover({ value, point: e.point.toArray() as [number, number, number] })
  }

  return (
    <mesh position={position} quaternion={quaternion} onPointerMove={hover} onPointerOut={() => onHover(null)}>
      <planeGeometry args={[Math.max(u1 - u0, 1e-3), Math.max(v1 - v0, 1e-3)]} />
      <meshBasicMaterial map={texture} side={THREE.DoubleSide} transparent toneMapped={false} />
    </mesh>
  )
}

export const fmt = (x: number) => (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toPrecision(3))

/** Range of the field colours on the bodies (all painted views together). */
export function fieldPaintRange(paints: Record<string, Paint>): [number, number] | null {
  const list = Object.values(paints).filter((p) => p.source === 'field')
  if (!list.length) return null
  return [Math.min(...list.map((p) => p.min)), Math.max(...list.map((p) => p.max))]
}

export function SlicePanel({ hover }: { hover: HoverValue | null }) {
  const slices = useStore((s) => s.slices)
  const settings = useStore((s) => s.slice)
  const bounds = useStore((s) => s.sceneBounds)
  const setSettings = useStore((s) => s.setSliceSettings)
  const paints = useStore((s) => s.paints)
  const fieldNode = useStore((s) => s.visible.find((id) => isFieldNode(s.specs[s.nodes.find((n) => n.id === id)?.data.typeId ?? ''])))
  const anyBody = useStore((s) => s.visible.some((id) => s.views[id]))
  const surface = settings.mode === 'surface'
  const entry = Object.entries(slices)[0]
  if (!fieldNode || (!surface && !entry)) return null
  const id = surface ? fieldNode : entry[0]
  const range = surface ? fieldPaintRange(paints) : [entry[1].min, entry[1].max]
  const { a } = planeAxes(settings.axis)
  const lo = bounds[0][a], hi = bounds[1][a]
  const position = settings.position ?? (lo + hi) / 2

  return (
    <div className="slice-panel">
      <div className="slice-row">
        <span className="slice-title">Field <b>{id}</b></span>
        <div className="segmented small" title="Show the field on a plane, or as colours on the visible bodies">
          <button className={!surface ? 'on' : ''} onClick={() => setSettings({ mode: 'plane' })}>Plane</button>
          <button className={surface ? 'on' : ''} onClick={() => setSettings({ mode: 'surface' })}>On bodies</button>
        </div>
        {!surface && (
          <div className="segmented small">
            {(['x', 'y', 'z'] as Axis[]).map((ax) => (
              <button key={ax} className={settings.axis === ax ? 'on' : ''} title={`Slice across ${ax.toUpperCase()}`}
                onClick={() => setSettings({ axis: ax })}>{ax.toUpperCase()}</button>
            ))}
          </div>
        )}
        <select value={settings.cmap} onChange={(e) => setSettings({ cmap: e.target.value as ColorMap })}>
          {(Object.keys(STOPS) as ColorMap[]).map((c) => <option key={c} value={c}>{COLOR_MAP_LABELS[c]}</option>)}
        </select>
      </div>
      {!surface && (
        <div className="slice-row">
          <span className="slice-label">{settings.axis} =</span>
          <input type="range" min={lo} max={hi} step={(hi - lo) / 400 || 0.1} value={position}
            onChange={(e) => setSettings({ position: Number(e.target.value) })} />
          <span className="slice-value">{position.toFixed(1)} mm</span>
        </div>
      )}
      {surface && !anyBody && <div className="slice-row muted">Show a body (its eye) to colour it with this field</div>}
      <div className="slice-bar" style={{ background: gradientCss(settings.cmap) }} />
      <div className="slice-row slice-range">
        <span>{range ? fmt(range[0]) : '…'}</span>
        <span className="slice-hover">
          {hover ? `${fmt(hover.value)}  at (${hover.point.map((p) => p.toFixed(1)).join(', ')})`
            : surface ? 'hover a body for values' : 'hover the slice for values'}
        </span>
        <span>{range ? fmt(range[1]) : '…'}</span>
      </div>
    </div>
  )
}
