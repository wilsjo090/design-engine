import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Grid, Html, OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh'
import { api, fetchPaint, type CheckReport } from '../api'
import { useStore, type Variant } from '../store'
import type { Axis, ColorMap, FaceData, FrameView, Marker, Paint, Recipe, ViewMesh } from '../types'
import { isLink, parseFaceKeys, resolveFaces } from '../types'
import {
  SliceObject, SlicePanel, fieldPaintRange, fmt, gradientCss, lut, normalise, sliceBox, type HoverValue,
} from './SliceView'

// CAD convention: Z is up.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1)
// Picking faces on bodies with a million triangles needs a search tree (built only for picking).
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree
THREE.Mesh.prototype.raycast = acceleratedRaycast

const PICK_SELECTED = '#4ea3f5'
const PICK_LEFT_OUT = '#e5534b'   // Face Mask "Apply to: all other faces": the picked faces stay plain
const PICK_HOVER = '#ffd166'
const CAP_COLOR = '#c8643f'   // inside of a part cut by the section plane

const KIND_STYLE: Record<string, { color: string; opacity: number }> = {
  mesh: { color: '#c9ced6', opacity: 1 },
  implicit: { color: '#86d6c9', opacity: 1 },
  cad: { color: '#e2b64a', opacity: 0.35 },
  variant: { color: '#b4bfd6', opacity: 1 },
}

const read = <T,>(key: string, fallback: T): T => {
  try { const v = localStorage.getItem(key); return v === null ? fallback : (JSON.parse(v) as T) } catch { return fallback }
}
const write = (key: string, value: unknown) => { try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* ignore */ } }

/** Colours of a paint (value per vertex), or null when it does not belong to this view. */
function paintColors(view: ViewMesh, paint: Paint | undefined, cmap: ColorMap, range: [number, number] | null) {
  if (!paint || !range || paint.viewVersion !== view.version || paint.values.length * 3 !== view.positions.length) return null
  const map = paintMap(paint, cmap)
  const table = lut(map)
  const norm = normalise(map, range[0], range[1])
  // thin walls and low safety factors red, the good end blue
  const reverse = paint.source === 'thickness' || !!paint.reverse
  const colors = new Float32Array(paint.values.length * 3)
  for (let i = 0; i < paint.values.length; i++) {
    const v = paint.values[i]
    if (Number.isNaN(v)) { colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 0.35; continue }
    let t = Math.max(0, Math.min(1, norm(v)))
    if (reverse) t = 1 - t
    const k = Math.round(t * 255) * 3
    // the table is in sRGB; three.js works in linear colour
    for (let c = 0; c < 3; c++) colors[i * 3 + c] = Math.pow(table[k + c] / 255, 2.2)
  }
  return colors
}

/** Thickness and results have their own colours (red = the bad end); fields use the chosen map. */
const paintMap = (paint: Paint, cmap: ColorMap): ColorMap =>
  (paint.source === 'field' ? cmap : paint.source === 'result' ? 'rainbow' : 'heat')

/** Colour range of a paint that is not a field (fields share one range across all bodies). */
function ownRange(paint: Paint): [number, number] {
  return paint.source === 'thickness' ? thicknessRange(paint) : [paint.min, paint.max]
}

/** A thickness paint's colour range: from 0 to a high-but-typical value (outliers would wash it out). */
export function thicknessRange(paint: Paint): [number, number] {
  const finite = Array.from(paint.values.filter((v) => Number.isFinite(v)))
  if (!finite.length) return [0, 1]
  finite.sort((a, b) => a - b)
  return [0, finite[Math.floor(finite.length * 0.95)] || paint.max]
}

function ViewObject({ view, wireframe, ghost, clip, paint, cmap, range, onHover, kind, node }: {
  view: ViewMesh; wireframe: boolean; ghost?: boolean; clip: THREE.Plane[]; paint?: Paint; cmap: ColorMap
  range: [number, number] | null; onHover?: (h: HoverValue | null) => void; kind?: string; node?: string
}) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(view.positions, 3))
    g.setIndex(new THREE.BufferAttribute(view.indices, 1))
    g.computeVertexNormals()
    g.computeBoundingSphere()
    return g
  }, [view])
  useEffect(() => () => geometry.dispose(), [geometry])
  const colors = useMemo(() => paintColors(view, paint, cmap, range), [view, paint, cmap, range])
  useEffect(() => {
    if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    else geometry.deleteAttribute('color')
  }, [geometry, colors])
  const style = KIND_STYLE[kind ?? view.kind] ?? KIND_STYLE.mesh
  const opacity = ghost ? Math.min(style.opacity, 0.25) : style.opacity // see-through while picking faces
  const transparent = opacity < 1
  const cut = clip.length > 0 && !transparent
  const hover = (e: ThreeEvent<PointerEvent>) => {
    if (!onHover || !colors || !paint || !e.face) return
    e.stopPropagation()
    const { a, b, c } = e.face
    const p = (i: number) => new THREE.Vector3(view.positions[i * 3], view.positions[i * 3 + 1], view.positions[i * 3 + 2])
    const w = new THREE.Triangle(p(a), p(b), p(c)).getBarycoord(e.point, new THREE.Vector3())
    if (!w) return
    const value = paint.values[a] * w.x + paint.values[b] * w.y + paint.values[c] * w.z
    onHover(Number.isFinite(value) ? { value, point: e.point.toArray() as [number, number, number], node } : null)
  }
  return (
    <group>
      <mesh geometry={geometry} renderOrder={transparent ? 1 : 0} raycast={ghost ? () => null : undefined}
        onPointerMove={colors ? hover : undefined} onPointerOut={colors && onHover ? () => onHover(null) : undefined}>
        <meshStandardMaterial key={colors ? 'painted' : 'plain'} color={colors ? '#ffffff' : style.color}
          vertexColors={!!colors} metalness={colors ? 0.05 : 0.3} roughness={colors ? 0.7 : 0.42}
          wireframe={wireframe} transparent={transparent} opacity={opacity} depthWrite={!transparent}
          side={cut ? THREE.FrontSide : THREE.DoubleSide} clippingPlanes={clip} />
      </mesh>
      {cut && !wireframe && (
        // the inside seen through the cut: back faces in one flat colour read as a solid cut face
        <mesh geometry={geometry} raycast={() => null}>
          <meshBasicMaterial color={CAP_COLOR} side={THREE.BackSide} clippingPlanes={clip} />
        </mesh>
      )}
    </group>
  )
}

/** The triangles passing `keep`, as their own small mesh (positions and smooth normals copied from
 *  the full body), or null if there are none. */
function faceSubset(view: ViewMesh, normals: THREE.BufferAttribute, count: number, keep: (triangle: number) => boolean) {
  let n = 0
  for (let t = 0; t < count; t++) if (keep(t)) n++
  if (!n) return null
  const pos = new Float32Array(n * 9)
  const nor = new Float32Array(n * 9)
  const src = normals.array as Float32Array
  let k = 0
  for (let t = 0; t < count; t++) {
    if (!keep(t)) continue
    for (let c = 0; c < 3; c++) {
      const v = view.indices[t * 3 + c] * 3
      for (let a = 0; a < 3; a++) {
        pos[k] = view.positions[v + a]
        nor[k++] = src[v + a]
      }
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3))
  return g
}

function Overlay({ geometry, color, order, clip }: { geometry: THREE.BufferGeometry | null; color: string; order: number; clip: THREE.Plane[] }) {
  useEffect(() => () => geometry?.dispose(), [geometry])
  if (!geometry) return null
  return (
    <mesh geometry={geometry} renderOrder={order} raycast={() => null}>
      <meshStandardMaterial color={color} metalness={0.1} roughness={0.5} side={THREE.DoubleSide}
        polygonOffset polygonOffsetFactor={-order} polygonOffsetUnits={-order} clippingPlanes={clip} />
    </mesh>
  )
}

/** Face keys: on copies of one part a plain face number ("5": face 5 on every copy, like the
 *  petals), on different parts "body:face" ("2:5"). A plain number always means all bodies. */
function faceKey(faces: FaceData, t: number) {
  return faces.same ? String(faces.ids[t]) : `${faces.bodies[t]}:${faces.ids[t]}`
}
const isChosen = (faces: FaceData, chosen: Set<string>, t: number) =>
  chosen.has(String(faces.ids[t])) || chosen.has(`${faces.bodies[t]}:${faces.ids[t]}`)

/** A CAD body you can pick faces on: click toggles a face, Shift+click the face and every face
 *  smoothly joined to it, Ctrl+drag paints. */
function PickableView({ view, faces, chosen, onPick, onPaint, wireframe, clip, chosenColor }: {
  view: ViewMesh; faces: FaceData | null; chosen: Set<string>; wireframe: boolean; clip: THREE.Plane[]; chosenColor: string
  onPick: (triangle: number, withGroup: boolean) => void
  onPaint: (triangle: number, add: boolean) => void
}) {
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  // Ctrl+drag paints: every face under the cursor is added (or removed, if the stroke started on
  // a chosen face) and the view does not rotate meanwhile.
  const painting = useRef<{ add: boolean } | null>(null)
  useEffect(() => {
    const stop = () => {
      if (!painting.current) return
      painting.current = null
      if (controls) controls.enabled = true
    }
    window.addEventListener('pointerup', stop)
    return () => window.removeEventListener('pointerup', stop)
  }, [controls])
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(view.positions, 3))
    g.setIndex(new THREE.BufferAttribute(view.indices, 1))
    g.computeVertexNormals()
    g.computeBoundingSphere()
    // indirect: the default build re-sorts the index buffer in place, so triangle numbers (and the
    // CAD face of each) no longer matched: clicking a side face selected scattered patches.
    // (the option exists in three-mesh-bvh 0.7.8 but is missing from its type definitions)
    g.computeBoundsTree({ indirect: true } as unknown as Parameters<THREE.BufferGeometry['computeBoundsTree']>[0])
    return g
  }, [view])
  useEffect(() => () => { geometry.disposeBoundsTree(); geometry.dispose() }, [geometry])
  const [hover, setHover] = useState<{ triangle: number; group: boolean } | null>(null)
  const data = faces && faces.ids.length === view.faces ? faces : null
  const normals = geometry.getAttribute('normal') as THREE.BufferAttribute
  const chosenGeo = useMemo(() => (data ? faceSubset(view, normals, view.faces, (t) => isChosen(data, chosen, t)) : null),
    [view, normals, data, chosen])
  const hoverGeo = useMemo(() => {
    if (!data || !hover) return null
    const members = new Set(faceGroup(data, hover.triangle, hover.group))
    return faceSubset(view, normals, view.faces, (t) => members.has(faceKey(data, t)))
  }, [view, normals, data, hover])

  const triangleAt = (e: ThreeEvent<PointerEvent | MouseEvent>) =>
    data && e.faceIndex !== undefined && e.faceIndex !== null ? e.faceIndex : null
  return (
    <group>
      <mesh geometry={geometry}
        onPointerDown={(e) => {
          const t = triangleAt(e)
          if (t === null || !(e.ctrlKey || e.metaKey) || e.button !== 0) return
          e.stopPropagation()
          if (controls) controls.enabled = false
          painting.current = { add: !isChosen(data!, chosen, t) }
          onPaint(t, painting.current.add)
        }}
        onPointerMove={(e) => {
          e.stopPropagation()
          const t = triangleAt(e)
          setHover(t === null ? null : { triangle: t, group: e.shiftKey })
          if (painting.current && t !== null) onPaint(t, painting.current.add)
        }}
        onPointerOut={() => setHover(null)}
        onClick={(e) => {
          const t = triangleAt(e)
          // a drag that rotated the view is not a click; Ctrl was handled as painting
          if (t === null || e.delta > 4 || e.ctrlKey || e.metaKey) return
          e.stopPropagation()
          onPick(t, e.shiftKey)
        }}>
        <meshStandardMaterial color="#d9d4c7" metalness={0.15} roughness={0.6} side={THREE.DoubleSide}
          wireframe={wireframe} clippingPlanes={clip} />
      </mesh>
      <Overlay geometry={chosenGeo} color={chosenColor} order={2} clip={clip} />
      <Overlay geometry={hoverGeo} color={PICK_HOVER} order={3} clip={clip} />
    </group>
  )
}

/** Keys of the face under triangle t, or of every face smoothly joined to it. */
function faceGroup(faces: FaceData, t: number, withGroup: boolean): string[] {
  const face = faces.ids[t], body = faces.bodies[t]
  if (!withGroup) return [faceKey(faces, t)]
  let list = 0
  if (!faces.same) {  // group lists come one per body, in the order the bodies appear
    const order: number[] = []
    for (let i = 0; i < faces.bodies.length; i++) if (!order.includes(faces.bodies[i])) order.push(faces.bodies[i])
    list = order.indexOf(body)
  }
  const groups = faces.groups[list] ?? faces.groups[0]
  const members: string[] = []
  groups.forEach((g, f) => { if (g === groups[face]) members.push(faces.same ? String(f) : `${body}:${f}`) })
  return members
}

async function fetchFaces(node: string): Promise<FaceData | string> {
  const res = await fetch(`/api/faces/${encodeURIComponent(node)}`)
  if (!res.ok) {
    try { return (await res.json()).detail ?? res.statusText } catch { return res.statusText }
  }
  const buf = new Uint32Array(await res.arrayBuffer())
  const n = buf[0]
  const ids = buf.subarray(1, 1 + n)
  const bodies = buf.subarray(1 + n, 1 + 2 * n)
  const same = buf[1 + 2 * n] === 1
  const lists = buf[2 + 2 * n]
  const groups: Uint32Array[] = []
  let pos = 3 + 2 * n
  for (let i = 0; i < lists; i++) {
    groups.push(buf.subarray(pos + 1, pos + 1 + buf[pos]))
    pos += 1 + buf[pos]
  }
  return { version: Number(res.headers.get('X-View-Version') ?? 0), ids, bodies, same, groups }
}

function unionBox(views: ViewMesh[]) {
  const box = new THREE.Box3()
  for (const v of views) {
    if (v.faces === 0) continue
    box.expandByPoint(new THREE.Vector3(...v.bounds[0]))
    box.expandByPoint(new THREE.Vector3(...v.bounds[1]))
  }
  return box
}

/** Frames the camera on the visible meshes (and field slices) when asked, or when content first appears. */
function CameraFit({ views, extra, onBounds }: {
  views: ViewMesh[]; extra: THREE.Box3[]; onBounds: (b: THREE.Box3) => void
}) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const fitRequest = useStore((s) => s.fitRequest)
  const lastFit = useRef(-1)
  const hadContent = useRef(false)

  useEffect(() => {
    const box = unionBox(views)
    if (!box.isEmpty()) onBounds(box.clone()) // the floor grid follows the bodies only (box grows below)
    extra.forEach((b) => box.union(b))
    const hasContent = !box.isEmpty()
    const wanted = fitRequest !== lastFit.current || (hasContent && !hadContent.current)
    hadContent.current = hasContent
    if (!wanted || !hasContent || !controls) return
    lastFit.current = fitRequest
    const center = box.getCenter(new THREE.Vector3())
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1)
    const dist = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.05
    const dir = new THREE.Vector3(1, -1.25, 0.85).normalize()
    camera.position.copy(center).addScaledVector(dir, dist)
    camera.near = radius / 200
    camera.far = radius * 200
    camera.updateProjectionMatrix()
    controls.target.copy(center)
    controls.update()
  }, [views, extra, fitRequest, camera, controls, onBounds])
  return null
}

function ClippingOn() {
  const gl = useThree((s) => s.gl)
  useEffect(() => { gl.localClippingEnabled = true }, [gl])
  return null
}

// --- section view ---------------------------------------------------------------------------
interface Section { on: boolean; axis: Axis; flip: boolean; position: number | null }

function SectionPanel({ section, setSection, lo, hi }: {
  section: Section; setSection: (s: Section) => void; lo: number; hi: number
}) {
  const position = section.position ?? (lo + hi) / 2
  return (
    <div className="tool-panel section-panel">
      <div className="slice-row">
        <span className="slice-title">Section</span>
        <div className="segmented small">
          {(['x', 'y', 'z'] as Axis[]).map((ax) => (
            <button key={ax} className={section.axis === ax ? 'on' : ''} title={`Cut across ${ax.toUpperCase()}`}
              onClick={() => setSection({ ...section, axis: ax, position: null })}>{ax.toUpperCase()}</button>
          ))}
        </div>
        <button className="small-btn" title="Show the other side" onClick={() => setSection({ ...section, flip: !section.flip })}>⇄ Flip</button>
      </div>
      <div className="slice-row">
        <span className="slice-label">{section.axis} =</span>
        <input type="range" min={lo} max={hi} step={(hi - lo) / 500 || 0.1} value={position}
          onChange={(e) => setSection({ ...section, position: Number(e.target.value) })} />
        <span className="slice-value">{position.toFixed(1)} mm</span>
      </div>
    </div>
  )
}

// --- print check ----------------------------------------------------------------------------
function CheckPanel({ candidates, onClose }: { candidates: string[]; onClose: () => void }) {
  const quality = useStore((s) => s.quality)
  const views = useStore((s) => s.views)
  const paints = useStore((s) => s.paints)
  const [node, setNode] = useState(candidates[0] ?? '')
  const [density, setDensity] = useState(() => read('check.density', 1.24))
  const [minWall, setMinWall] = useState(() => read('check.minWall', 0.8))
  const [busy, setBusy] = useState(false)
  const [report, setReport] = useState<CheckReport | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { if (!candidates.includes(node)) setNode(candidates[0] ?? '') }, [candidates, node])
  const stale = report && views[node] && report.view_version !== views[node].version
  const paint = paints[node]?.source === 'thickness' ? paints[node] : undefined

  const run = async () => {
    if (!node) return
    setBusy(true)
    setError('')
    write('check.density', density)
    write('check.minWall', minWall)
    try {
      const r = await api.check(node, density, true, minWall)
      setReport(r)
      if (r.paint) await fetchPaint(node, { ...r.paint, label: 'wall thickness (mm)' }, 'thickness')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const num = (label: string, value: number, set: (v: number) => void, unit: string, title: string) => (
    <label className="check-input" title={title}>
      {label}
      <input className="value-input" type="number" step="0.01" value={value}
        onChange={(e) => Number.isFinite(Number(e.target.value)) && set(Number(e.target.value))} />
      <span className="muted">{unit}</span>
    </label>
  )
  const t = report?.thickness
  return (
    <div className="tool-panel check-panel">
      <div className="slice-row">
        <span className="slice-title">Print check</span>
        <button className="small-btn" onClick={onClose} title="Close (removes the thickness colours)">×</button>
      </div>
      {!candidates.length ? <div className="muted">Show a body or mesh first.</div> : (
        <>
          <div className="slice-row">
            <select value={node} onChange={(e) => setNode(e.target.value)}>
              {candidates.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <button className="primary" disabled={busy} onClick={run}>{busy ? 'Checking…' : report ? 'Check again' : 'Check'}</button>
          </div>
          <div className="slice-row">
            {num('Density', density, setDensity, 'g/cm³', 'Material density: PLA 1.24, PETG 1.27, resin ~1.1, nylon PA12 1.01')}
            {num('Min wall', minWall, setMinWall, 'mm', 'Thinnest wall your printer makes reliably')}
          </div>
          {quality !== 'final' && <div className="muted small">Checks the {quality} preview: use Final for exact numbers.</div>}
          {error && <div className="warn">{error}</div>}
          {report && (
            <div className={`check-report ${stale ? 'stale' : ''}`}>
              {stale && <div className="muted small">The view has changed since: check again.</div>}
              <Row k="Size" v={report.size ? report.size.map((x) => x.toFixed(1)).join(' × ') + ' mm' : '–'} />
              <Row k="Volume" v={report.volume !== undefined ? `${(report.volume / 1000).toFixed(2)} cm³` : '–'} />
              <Row k="Weight" v={report.weight !== undefined ? `${report.weight.toFixed(1)} g` : '–'} />
              <Row k="Surface" v={report.area !== undefined ? `${(report.area / 100).toFixed(1)} cm²` : '–'} />
              <Row k="Bodies" v={String(report.bodies ?? '–')} warn={(report.bodies ?? 1) > 1}
                title="Separate solid pieces. More than one: loose parts that print separately (or not at all)." />
              <Row k="Closed cavities" v={report.cavities ? `${report.cavities} (${(report.cavity_volume! / 1000).toFixed(2)} cm³)` : 'none'}
                warn={!!report.cavities} title="Empty space sealed inside the part: traps resin or powder, and slicers may fill it." />
              <Row k="Open edges" v={String(report.open_edges ?? '–')} warn={(report.open_edges ?? 0) > 0}
                title="Edges not shared by exactly two triangles: the mesh has holes and may not slice." />
              <Row k="Triangles" v={report.triangles.toLocaleString()} />
              {t && (
                <>
                  <Row k="Thinnest wall" v={`${t.thinnest.toFixed(2)} mm (1%: ${t.p1.toFixed(2)})`} warn={t.p1 < t.min_wall}
                    title="Measured straight into the part from every point of the surface. The 1% value ignores single spots." />
                  <Row k={`Below ${t.min_wall} mm`} v={`${(t.below * 100).toFixed(1)}% of the surface`} warn={t.below > 0.001} />
                  <Row k="Typical wall" v={`${t.median.toFixed(2)} mm`} />
                </>
              )}
              {paint && (
                <>
                  <div className="slice-bar" style={{ background: gradientCss('heat'), transform: 'scaleX(-1)' }} />
                  <div className="slice-row slice-range"><span>0 mm (thin)</span><span>{fmt(thicknessRange(paint)[1])} mm</span></div>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Row({ k, v, warn, title }: { k: string; v: string; warn?: boolean; title?: string }) {
  return <div className={`check-row ${warn ? 'warn' : ''}`} title={title}><span>{k}</span><span>{v}</span></div>
}

// --- variants: kept copies shown next to the current result ----------------------------------
/** What differs between a variant's recipe and the current one (for its tooltip). */
function recipeDiff(before: Recipe, after: Recipe): string[] {
  const out: string[] = []
  for (const [id, node] of Object.entries(after.nodes)) {
    const old = before.nodes[id]
    if (!old) { out.push(`+ ${id}`); continue }
    for (const port of new Set([...Object.keys(old.inputs ?? {}), ...Object.keys(node.inputs ?? {})])) {
      const a = old.inputs?.[port], b = node.inputs?.[port]
      if (JSON.stringify(a) === JSON.stringify(b)) continue
      const show = (v: unknown) => (v === undefined ? 'default' : isLink(v) ? `← ${v.link}` : String(v))
      out.push(`${id}.${port}: ${show(a)} → ${show(b)}`)
    }
  }
  for (const id of Object.keys(before.nodes)) if (!after.nodes[id]) out.push(`− ${id}`)
  if (JSON.stringify(before.groups ?? {}) !== JSON.stringify(after.groups ?? {})) out.push('groups changed')
  return out
}

function VariantsList({ variants }: { variants: Variant[] }) {
  const update = useStore((s) => s.updateVariant)
  const remove = useStore((s) => s.removeVariant)
  const recipe = useStore((s) => s.recipe)
  if (!variants.length) return null
  return (
    <div className="variants">
      {variants.map((v, i) => {
        const diff = recipeDiff(v.recipe, recipe())
        return (
          <div className="variant-row" key={v.id}
            title={diff.length ? `Now compared with this variant:\n${diff.slice(0, 12).join('\n')}${diff.length > 12 ? '\n…' : ''}`
              : 'Same settings as now'}>
            <span className="variant-pos">{i === 0 ? '◀' : '◀'.repeat(Math.min(i + 1, 3))}</span>
            <input className="variant-name" value={v.label} onChange={(e) => update(v.id, { label: e.target.value })} />
            <span className="muted">{diff.length ? `${diff.length} change${diff.length === 1 ? '' : 's'}` : 'same'}</span>
            <button className="small-btn" title={v.shown ? 'Hide' : 'Show'} onClick={() => update(v.id, { shown: !v.shown })}>
              {v.shown ? '◉' : '○'}
            </button>
            <button className="small-btn" title="Remove" onClick={() => remove(v.id)}>×</button>
          </div>
        )
      })}
    </div>
  )
}

// --- simulation: supports, loads and results -----------------------------------------------------
const LOAD_COLOR = 0xff4d6d
const FIXED_COLOR = 0x3fcf6e

/** Arrows for the loads (tip on the face when pushing, tail when pulling) and small cones pointing
 *  into the held faces, sized to the scene. */
function MarkerArrows({ markers, size }: { markers: Record<string, Marker[]>; size: number }) {
  const group = useMemo(() => {
    const g = new THREE.Group()
    const len = size * 0.06
    for (const list of Object.values(markers)) {
      for (const m of list) {
        m.points.forEach((p, i) => {
          const dir = new THREE.Vector3(...m.dirs[i]).normalize()
          const at = new THREE.Vector3(...p)
          if (m.kind === 'fixed') {
            const h = len * 0.4
            g.add(new THREE.ArrowHelper(dir, at.clone().addScaledVector(dir, -h), h, FIXED_COLOR, h, h * 0.6))
          } else {
            const start = m.push[i] ? at.clone().addScaledVector(dir, -len) : at
            g.add(new THREE.ArrowHelper(dir, start, len, LOAD_COLOR, len * 0.3, len * 0.16))
          }
        })
      }
    }
    // drawn over the part: supports and loads often sit inside holes (bolt holes, a pin hole)
    g.traverse((o) => {
      const mat = (o as THREE.Mesh).material as THREE.Material | undefined
      if (!mat) return
      mat.depthTest = false
      mat.transparent = true
      mat.opacity = 0.9
      o.renderOrder = 10
    })
    return g
  }, [markers, size])
  useEffect(() => () => group.children.forEach((c) => (c as THREE.ArrowHelper).dispose()), [group])
  return <primitive object={group} />
}

// --- aiming a load: a big arrow you drag by its head ---------------------------------------------
const GIZMO_COLOR = '#ffc93c'
const GIZMO_HOVER = '#ffe9a8'
const AXIS_DIRECTIONS: Record<string, [number, number, number]> = {
  '+X': [1, 0, 0], '-X': [-1, 0, 0], '+Y': [0, 1, 0], '-Y': [0, -1, 0], '+Z': [0, 0, 1], '-Z': [0, 0, -1],
}
// the arrow snaps to the axes, the 45° diagonals between two axes and the cube diagonals within 6°
const SNAP_DIRECTIONS = [-1, 0, 1].flatMap((x) => [-1, 0, 1].flatMap((y) => [-1, 0, 1].map((z) => [x, y, z])))
  .filter(([x, y, z]) => x || y || z).map(([x, y, z]) => new THREE.Vector3(x, y, z).normalize())
const SNAP_ANGLE = (6 * Math.PI) / 180

function snapDirection(v: THREE.Vector3) {
  let best: THREE.Vector3 | null = null
  let angle = SNAP_ANGLE
  for (const s of SNAP_DIRECTIONS) {
    const a = v.angleTo(s)
    if (a < angle) { angle = a; best = s }
  }
  return (best ?? v).clone()
}

const fmtComponent = (x: number) => String(Number(x.toFixed(3)))
const vectorText = (v: THREE.Vector3) => `${fmtComponent(v.x)}, ${fmtComponent(v.y)}, ${fmtComponent(v.z)}`
const parseVec = (value: unknown) => {
  const v = String(value ?? '').split(/[\s,;]+/).filter(Boolean).map(Number)
  return v.length === 3 && v.every(Number.isFinite) ? new THREE.Vector3(...v) : null
}
/** Offsets are dragged in whole steps: 0.5 mm in small scenes, 1 mm up to 300 mm, else 5 mm. */
const offsetStep = (sceneSize: number) => (sceneSize <= 50 ? 0.5 : sceneSize <= 300 ? 1 : 5)
const Y_UP = new THREE.Vector3(0, 1, 0)
const Z_UP = new THREE.Vector3(0, 0, 1)

/** The arrow of the selected Load node: drag its head to point the force (a torque: its axis); on
 *  release the node's Direction becomes the axis it snapped to, or custom with that Vector. With
 *  `onMove`, drag the ball at its tail to move where the force acts (the node's Offset from the
 *  faces' centre, in steps of `step` mm). Esc cancels a drag. */
function DirectionGizmo({ anchor, dir, length, onCommit, torque = false, centre, step = 1, onMove }: {
  anchor: THREE.Vector3; dir: THREE.Vector3; length: number; onCommit: (v: THREE.Vector3) => void
  torque?: boolean; centre?: THREE.Vector3; step?: number; onMove?: (offset: THREE.Vector3) => void
}) {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const [aiming, setAiming] = useState<THREE.Vector3 | null>(null)
  const [moving, setMoving] = useState<THREE.Vector3 | null>(null)
  const [hover, setHover] = useState<'head' | 'tail' | null>(null)
  const latest = useRef<THREE.Vector3 | null>(null)
  const grab = useRef<{ plane: THREE.Plane; from: THREE.Vector3; offset: THREE.Vector3 | null } | null>(null)
  const active = aiming !== null
  const shifting = moving !== null

  useEffect(() => {
    if (!shifting || !centre || !onMove) return
    const raycaster = new THREE.Raycaster()
    const move = (ev: PointerEvent) => {
      const g = grab.current
      if (!g) return
      const r = gl.domElement.getBoundingClientRect()
      raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1,
        -((ev.clientY - r.top) / r.height) * 2 + 1), camera)
      // the ball moves in the plane facing the camera (turn the view to move it in depth)
      const hit = raycaster.ray.intersectPlane(g.plane, new THREE.Vector3())
      if (!hit) return
      const off = hit.sub(g.from).add(anchor).sub(centre)
      off.set(Math.round(off.x / step) * step, Math.round(off.y / step) * step, Math.round(off.z / step) * step)
      g.offset = off
      setMoving(centre.clone().add(off))
    }
    const finish = (commit: boolean) => {
      if (controls) controls.enabled = true
      if (commit && grab.current?.offset) onMove(grab.current.offset)
      grab.current = null
      setMoving(null)
    }
    const up = () => finish(true)
    const key = (ev: KeyboardEvent) => { if (ev.key === 'Escape') finish(false) }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('keydown', key)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('keydown', key)
    }
  }, [shifting, anchor, centre, step, camera, gl, controls, onMove])

  useEffect(() => {
    if (!active) return
    const raycaster = new THREE.Raycaster()
    const aim = (ev: PointerEvent) => {
      const r = gl.domElement.getBoundingClientRect()
      raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1,
        -((ev.clientY - r.top) / r.height) * 2 + 1), camera)
      // where the pointer meets the sphere the arrow's head moves on (or comes closest to it)
      const ray = raycaster.ray
      const v = ray.origin.clone().sub(anchor)
      const b = v.dot(ray.direction)
      const disc = b * b - (v.lengthSq() - length * length)
      const t = disc >= 0 ? -b - Math.sqrt(disc) : -b
      const hit = ray.origin.clone().addScaledVector(ray.direction, Math.max(t, 0)).sub(anchor)
      if (hit.lengthSq() < 1e-12) return
      latest.current = snapDirection(hit.normalize())
      setAiming(latest.current)
    }
    const finish = (commit: boolean) => {
      if (controls) controls.enabled = true
      if (commit && latest.current) onCommit(latest.current)
      latest.current = null
      setAiming(null)
    }
    const up = () => finish(true)
    const key = (ev: KeyboardEvent) => { if (ev.key === 'Escape') finish(false) }
    window.addEventListener('pointermove', aim)
    window.addEventListener('pointerup', up)
    window.addEventListener('keydown', key)
    return () => {
      window.removeEventListener('pointermove', aim)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('keydown', key)
    }
  }, [active, anchor, length, camera, gl, controls, onCommit])

  const at = moving ?? anchor
  const d = (aiming ?? dir).clone().normalize()
  const quat = new THREE.Quaternion().setFromUnitVectors(Y_UP, d)
  const head = at.clone().addScaledVector(d, length * 0.86)
  const headColor = hover === 'head' || active ? GIZMO_HOVER : GIZMO_COLOR
  const tailColor = hover === 'tail' || shifting ? GIZMO_HOVER : GIZMO_COLOR
  const material = (color: string, opacity = 0.95) => (
    <meshBasicMaterial color={color} depthTest={false} transparent opacity={opacity} />
  )
  const hoverOn = (part: 'head' | 'tail') => (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    setHover(part)
    gl.domElement.style.cursor = 'grab'
  }
  const hoverOff = () => { setHover(null); gl.domElement.style.cursor = '' }
  // a line from the faces' centre to where the force acts, when that is elsewhere
  const lever = centre && at.distanceTo(centre) > 1e-6 ? at.clone().sub(centre) : null
  const ring = length * 0.2
  return (
    <group>
      <mesh position={at.clone().addScaledVector(d, length * 0.36)} quaternion={quat} renderOrder={20} raycast={() => null}>
        <cylinderGeometry args={[length * 0.03, length * 0.03, length * 0.72, 12]} />
        {material(headColor)}
      </mesh>
      <mesh position={head} quaternion={quat} renderOrder={21} raycast={() => null}>
        <coneGeometry args={[length * 0.12, length * 0.28, 24]} />
        {material(headColor)}
      </mesh>
      {torque && (  // the turn about the axis (right-hand rule): an arc with an arrowhead
        <group position={at.clone().addScaledVector(d, length * 0.42)}
          quaternion={new THREE.Quaternion().setFromUnitVectors(Z_UP, d)}>
          <mesh renderOrder={20} raycast={() => null}>
            <torusGeometry args={[ring, length * 0.022, 8, 48, Math.PI * 1.5]} />
            {material(headColor)}
          </mesh>
          <mesh position={[length * 0.07, -ring, 0]} rotation={[0, 0, -Math.PI / 2]} renderOrder={21} raycast={() => null}>
            <coneGeometry args={[length * 0.06, length * 0.14, 16]} />
            {material(headColor)}
          </mesh>
        </group>
      )}
      {/* an invisible ball around the head: a bigger target to grab */}
      <mesh position={head} renderOrder={22} onPointerOver={hoverOn('head')} onPointerOut={hoverOff}
        onPointerDown={(e) => {
          if (e.button !== 0) return
          e.stopPropagation()
          if (controls) controls.enabled = false
          latest.current = d.clone()
          setAiming(d.clone())
        }}>
        <sphereGeometry args={[length * 0.2, 16, 12]} />
        <meshBasicMaterial transparent opacity={0} depthTest={false} depthWrite={false} />
      </mesh>
      {onMove && centre && (
        <>
          <mesh position={at} renderOrder={21} raycast={() => null}>
            <sphereGeometry args={[length * 0.055, 16, 12]} />
            {material(tailColor)}
          </mesh>
          <mesh position={at} renderOrder={22} onPointerOver={hoverOn('tail')} onPointerOut={hoverOff}
            onPointerDown={(e) => {
              if (e.button !== 0) return
              e.stopPropagation()
              if (controls) controls.enabled = false
              const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), anchor)
              grab.current = { plane, from: e.ray.intersectPlane(plane, new THREE.Vector3()) ?? anchor.clone(), offset: null }
              setMoving(anchor.clone())
            }}>
            <sphereGeometry args={[length * 0.12, 16, 12]} />
            <meshBasicMaterial transparent opacity={0} depthTest={false} depthWrite={false} />
          </mesh>
        </>
      )}
      {lever && centre && (
        <>
          <mesh position={centre.clone().addScaledVector(lever, 0.5)}
            quaternion={new THREE.Quaternion().setFromUnitVectors(Y_UP, lever.clone().normalize())}
            renderOrder={19} raycast={() => null}>
            <cylinderGeometry args={[length * 0.012, length * 0.012, lever.length(), 8]} />
            {material(GIZMO_COLOR, 0.6)}
          </mesh>
          <mesh position={centre} renderOrder={19} raycast={() => null}>
            <sphereGeometry args={[length * 0.025, 12, 8]} />
            {material(GIZMO_COLOR, 0.6)}
          </mesh>
        </>
      )}
      {active && (
        <>
          <mesh position={at} raycast={() => null} renderOrder={19}>
            <sphereGeometry args={[length, 32, 16]} />
            <meshBasicMaterial color={GIZMO_COLOR} wireframe transparent opacity={0.08} depthTest={false} />
          </mesh>
          <Html position={head} center style={{ pointerEvents: 'none' }}>
            <div className="gizmo-label">{torque ? 'axis ' : ''}{vectorText(d)}</div>
          </Html>
        </>
      )}
      {shifting && centre && (
        <Html position={at} center style={{ pointerEvents: 'none' }}>
          <div className="gizmo-label gizmo-label-below">offset {vectorText(at.clone().sub(centre))} mm</div>
        </Html>
      )}
    </group>
  )
}

// --- frames: the bars of Frame Optimization nodes, coloured by size ------------------------------
const SIZE_COLOURS = ['#4ea3f5', '#3fcfb4', '#8fd45a', '#e6d84d', '#ffa64d', '#ff6b6b', '#d36bff']

/** Each size of a frame (sorted) and its colour: smallest blue to largest violet. */
function sizeColours(sizes: number[]) {
  const unique = [...new Set(sizes)].sort((a, b) => a - b)
  const last = Math.max(unique.length - 1, 1)
  return new Map(unique.map((s, i) => [s, SIZE_COLOURS[Math.round((i * (SIZE_COLOURS.length - 1)) / last)]]))
}

interface BarHover { node: string; bar: number; size: number; length: number; force: number; utilisation: number; profile: string }

/** A frame's bars as tubes at their size (drawn round whatever the profile), with a ball at each
 *  node; hovering one reports its bar. Much cheaper than meshing the frame's body. */
function FrameBars({ id, frame, onHover }: { id: string; frame: FrameView; onHover: (h: BarHover | null) => void }) {
  const group = useMemo(() => {
    const colours = sizeColours(frame.sizes)
    const up = new THREE.Vector3(0, 1, 0)
    const tubes = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 14),
      new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.05 }), frame.bars.length)
    const m = new THREE.Matrix4()
    const q = new THREE.Quaternion()
    const radius = new Array(frame.nodes.length).fill(0)
    frame.bars.forEach(([a, b], i) => {
      const pa = new THREE.Vector3(...frame.nodes[a])
      const pb = new THREE.Vector3(...frame.nodes[b])
      const d = pb.clone().sub(pa)
      const r = frame.sizes[frame.chains[i]] / 2
      q.setFromUnitVectors(up, d.clone().normalize())
      m.compose(pa.clone().add(pb).multiplyScalar(0.5), q, new THREE.Vector3(r, d.length(), r))
      tubes.setMatrixAt(i, m)
      tubes.setColorAt(i, new THREE.Color(colours.get(frame.sizes[frame.chains[i]]) ?? '#cccccc'))
      radius[a] = Math.max(radius[a], r)
      radius[b] = Math.max(radius[b], r)
    })
    const balls = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 14, 10),
      new THREE.MeshStandardMaterial({ color: '#c8ccd4', roughness: 0.5 }), frame.nodes.length)
    frame.nodes.forEach((p, i) => {
      m.compose(new THREE.Vector3(...p), new THREE.Quaternion(), new THREE.Vector3(radius[i], radius[i], radius[i]))
      balls.setMatrixAt(i, m)
    })
    balls.raycast = () => null
    const g = new THREE.Group()
    g.add(tubes, balls)
    return { g, tubes, balls }
  }, [frame])
  useEffect(() => () => {
    for (const o of [group.tubes, group.balls]) {
      o.geometry.dispose()
      ;(o.material as THREE.Material).dispose()
    }
  }, [group])
  return (
    <primitive object={group.g}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        if (e.object !== group.tubes || e.instanceId === undefined) return
        e.stopPropagation()
        const i = e.instanceId
        const c = frame.chains[i]
        onHover({ node: id, bar: c + 1, size: frame.sizes[c], length: frame.lengths[c], force: frame.axial[i],
          utilisation: frame.utilisation[c], profile: frame.profile })
      }}
      onPointerOut={() => onHover(null)} />
  )
}

/** The sizes of the shown frames with their colours, and the bar under the mouse. */
function FramePanel({ frames, hover }: { frames: [string, FrameView][]; hover: BarHover | null }) {
  return (
    <div className="tool-panel">
      {frames.map(([id, f]) => {
        const colours = sizeColours(f.sizes)
        const length = new Map<number, number>()
        f.sizes.forEach((s, c) => length.set(s, (length.get(s) ?? 0) + f.lengths[c]))
        return (
          <div key={id}>
            <div className="slice-row"><span className="slice-title"><b>{id}</b>: {f.sizes.length} bars, {f.profile}
              {f.profile === 'round' ? '' : ' (drawn round here)'}</span></div>
            <div className="frame-sizes">
              {[...colours.entries()].map(([s, col]) => (
                <span key={s} className="frame-size" title={`${(length.get(s) ?? 0).toFixed(0)} mm of bar this size`}>
                  <span className="legend-swatch" style={{ background: col }} />{s} mm
                </span>
              ))}
            </div>
          </div>
        )
      })}
      <div className="muted small">
        {hover
          ? <>bar {hover.bar} of <b>{hover.node}</b>: {hover.size} mm {hover.profile}, {hover.length.toFixed(1)} mm long,
            {' '}{Math.abs(hover.force).toFixed(0)} N {hover.force < 0 ? 'compression' : 'tension'} here,
            utilisation {hover.utilisation.toFixed(2)}</>
          : 'hover a bar: its size, length, force and how much of its strength it uses'}
      </div>
    </div>
  )
}

/** Colour scale of a node's result (Static Analysis) with its unit, and the value under the mouse. */
function ResultPanel({ node, paint, hover }: { node: string; paint: Paint; hover: HoverValue | null }) {
  const unit = paint.unit ? ` ${paint.unit}` : ''
  return (
    <div className="tool-panel">
      <div className="slice-row">
        <span className="slice-title"><b>{node}</b>: {paint.label}{paint.unit ? ` (${paint.unit})` : ''}</span>
      </div>
      <div className="slice-bar" style={{ background: gradientCss('rainbow'), transform: paint.reverse ? 'scaleX(-1)' : undefined }} />
      <div className="slice-row slice-range">
        <span>{fmt(paint.min)}{unit}</span>
        <span className="slice-hover">
          {hover ? `${fmt(hover.value)}${unit} at (${hover.point.map((p) => p.toFixed(1)).join(', ')})` : 'hover the part for values'}
        </span>
        <span>{fmt(paint.max)}{unit}</span>
      </div>
      {paint.reverse && <div className="muted small">Below 1 the material yields or breaks (shown up to 10).</div>}
      {paint.clipped && (
        <div className="muted small">
          Right at the fixed faces it goes past this scale: holding them perfectly rigid makes an artificial peak
          there (see the log).
        </div>
      )}
    </div>
  )
}

export function Viewport() {
  const viewsMap = useStore((s) => s.views)
  const visible = useStore((s) => s.visible)
  const status = useStore((s) => s.status)
  const requestFit = useStore((s) => s.requestFit)
  const slices = useStore((s) => s.slices)
  const cmap = useStore((s) => s.slice.cmap)
  const paints = useStore((s) => s.paints)
  const variants = useStore((s) => s.variants)
  const addVariant = useStore((s) => s.addVariant)
  const clearPaints = useStore((s) => s.clearPaints)
  const selections = useStore((s) => s.selections)
  const [wireframe, setWireframe] = useState(false)
  const [bounds, setBoundsState] = useState<THREE.Box3 | null>(null)
  // only a real change re-renders (a new but equal box each time looped with the variants' boxes)
  const setBounds = useMemo(() => (b: THREE.Box3) => {
    const copy = b.clone()
    setBoundsState((prev) => (prev && prev.equals(copy) ? prev : copy))
  }, [])
  const [hover, setHover] = useState<HoverValue | null>(null)
  const [section, setSectionState] = useState<Section>(() => ({ ...read('section', { on: false, axis: 'z', flip: false, position: null }), on: false }))
  const setSection = (s: Section) => { setSectionState(s); write('section', { ...s, position: null }) }
  const [checking, setChecking] = useState(false)
  const views = useMemo(() => visible.filter((id) => viewsMap[id]).map((id) => viewsMap[id]), [visible, viewsMap])
  const sliceBoxes = useMemo(() => Object.values(slices).map(sliceBox), [slices])
  const fieldRange = useMemo(() => fieldPaintRange(paints), [paints])
  const ownRanges = useMemo(() => Object.fromEntries(Object.entries(paints)
    .filter(([, p]) => p.source !== 'field').map(([id, p]) => [id, ownRange(p)])), [paints])
  const markers = useStore((s) => s.markers)
  const [showMarkers, setShowMarkersState] = useState(() => read('markers.shown', true))
  const setShowMarkers = (on: boolean) => { setShowMarkersState(on); write('markers.shown', on) }
  const anyMarkers = Object.values(markers).some((list) => list.some((m) => m.points.length))
  const results = visible.filter((id) => paints[id]?.source === 'result')
  const frameMap = useStore((s) => s.frames)
  const frames = useMemo(() => Object.entries(frameMap).flatMap(([id, list]) => list.map((f) => [id, f] as [string, FrameView])),
    [frameMap])
  const [showFrames, setShowFramesState] = useState(() => read('frames.shown', true))
  const setShowFrames = (on: boolean) => { setShowFramesState(on); write('frames.shown', on) }
  const [barHover, setBarHover] = useState<BarHover | null>(null)

  // --- aiming the selected Load node's force (a draggable arrow at its faces) ---
  const aimNode = useStore((s) => (s.editing ? undefined
    : s.nodes.find((n) => n.selected && n.data.typeId === 'sim.load')))
  const aim = useMemo(() => {
    if (!aimNode) return null
    const inputs = aimNode.data.inputs
    const kind = String(inputs.kind ?? 'force')
    const direction = String(inputs.direction ?? 'into the face')
    const load = markers[aimNode.id]?.find((m) => m.kind === 'load' && m.points.length)
    if (kind === 'pressure' || !load || isLink(inputs.direction) || isLink(inputs.vector)) return null
    const torque = kind === 'torque'
    // the force acts at the faces' centre, moved by Offset (a torque has no point it acts at)
    const centre = load.centre ? new THREE.Vector3(...load.centre)
      : load.points.reduce((a, p) => a.add(new THREE.Vector3(...p)), new THREE.Vector3()).divideScalar(load.points.length)
    const movable = !torque && !isLink(inputs.offset)
    const offset = movable ? parseVec(inputs.offset ?? '0, 0, 0') ?? new THREE.Vector3() : new THREE.Vector3()
    let dir: THREE.Vector3
    if (AXIS_DIRECTIONS[direction]) dir = new THREE.Vector3(...AXIS_DIRECTIONS[direction])
    else if (direction === 'custom') {
      const v = parseVec(inputs.vector ?? '0, 0, -1')
      dir = v && v.lengthSq() > 0 ? v : new THREE.Vector3(0, 0, -1)
    } else if (load.arrow) dir = new THREE.Vector3(...load.arrow)  // into / out of the face, from the engine
    else {  // the average of the arrows drawn there
      dir = load.dirs.reduce((a, p) => a.add(new THREE.Vector3(...p)), new THREE.Vector3())
      if (dir.lengthSq() < 1e-12) dir = new THREE.Vector3(0, 0, -1)
    }
    return { id: aimNode.id, torque, movable, centre, anchor: centre.clone().add(offset), dir: dir.normalize() }
  }, [aimNode, markers])
  const setAimInput = useStore((s) => s.setInput)
  const commitAim = useMemo(() => (v: THREE.Vector3) => {
    if (!aim) return
    const axis = Object.entries(AXIS_DIRECTIONS).find(([, a]) => v.angleTo(new THREE.Vector3(...a)) < 1e-3)?.[0]
    if (axis) setAimInput(aim.id, 'direction', axis)
    else {
      setAimInput(aim.id, 'vector', vectorText(v))
      setAimInput(aim.id, 'direction', 'custom')
    }
  }, [aim, setAimInput])
  const commitOffset = useMemo(() => (v: THREE.Vector3) => {
    if (aim) setAimInput(aim.id, 'offset', vectorText(v))
  }, [aim, setAimInput])

  // --- face picking (a Face Mask node's Pick button) ---
  const picking = useStore((s) => s.picking)
  const setPicking = useStore((s) => s.setPicking)
  const setInput = useStore((s) => s.setInput)
  const saveSelection = useStore((s) => s.saveSelection)
  const facesValue = useStore((s) => (s.picking ? s.nodes.find((n) => n.id === s.picking!.node)?.data.inputs.faces : undefined))
  const leaveOut = useStore((s) => !!s.picking &&
    s.nodes.find((n) => n.id === s.picking!.node)?.data.inputs.apply === 'all other faces')
  const pickNodeExists = useStore((s) => !s.picking || s.nodes.some((n) => n.id === s.picking!.node))
  const chosen = useMemo(() => new Set(parseFaceKeys(resolveFaces(facesValue, selections))), [facesValue, selections])
  const pickView = picking ? viewsMap[picking.source] : undefined
  const [faceData, setFaceData] = useState<FaceData | null>(null)
  const [faceError, setFaceError] = useState<string | null>(null)

  useEffect(() => { if (!pickNodeExists) setPicking(null) }, [pickNodeExists, setPicking])
  useEffect(() => {
    setFaceData(null)
    setFaceError(null)
    if (!picking || !pickView) return
    let alive = true
    fetchFaces(picking.source).then((r) => {
      if (!alive) return
      if (typeof r === 'string') setFaceError(r)
      else setFaceData(r)
    }).catch((err) => alive && setFaceError(String(err)))
    return () => { alive = false }
  }, [picking, pickView])
  useEffect(() => {
    if (!picking) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPicking(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [picking, setPicking])

  // Read the faces from the store at event time: a render may not have happened between two
  // quick clicks, or between the moves of a paint stroke.
  const currentRaw = () => String(useStore.getState().nodes.find((n) => n.id === picking?.node)?.data.inputs.faces ?? '')
  const currentKeys = () => new Set(parseFaceKeys(resolveFaces(currentRaw(), useStore.getState().selections)))
  const writeFaces = (keys: Set<string>) => {
    if (!picking) return
    const text = parseFaceKeys([...keys].join(', ')).join(', ')
    const raw = currentRaw().trim()
    // picking with a saved selection chosen updates that selection (every mask using it follows)
    if (useStore.getState().selections[raw] !== undefined) saveSelection(raw, text)
    else setInput(picking.node, 'faces', text)
  }
  const onPick = (t: number, withGroup: boolean) => {
    if (!faceData) return
    const next = currentKeys()
    const members = faceGroup(faceData, t, withGroup)
    const remove = isChosen(faceData, next, t)
    for (const key of members) {
      if (remove) { next.delete(key); next.delete(key.split(':').pop()!) } else next.add(key)
    }
    writeFaces(next)
  }
  const onPaint = (t: number, add: boolean) => {
    if (!faceData) return
    const next = currentKeys()
    if (isChosen(faceData, next, t) === add) return
    const key = faceKey(faceData, t)
    if (add) next.add(key)
    else { next.delete(key); next.delete(String(faceData.ids[t])) }
    writeFaces(next)
  }

  // Grid scaled to the part: ~10 cells across, snapped to 1/2/5 x 10^n mm.
  const size = bounds ? bounds.getSize(new THREE.Vector3()) : new THREE.Vector3(100, 100, 100)
  const span = Math.max(size.x, size.y, 1)
  const step = Math.pow(10, Math.floor(Math.log10(span / 10)))
  const cell = [1, 2, 5, 10].map((m) => m * step).find((c) => span / c <= 15) ?? step * 10
  const floorZ = bounds ? bounds.min.z - span * 0.02 : 0
  const center = bounds ? bounds.getCenter(new THREE.Vector3()) : new THREE.Vector3()

  // section plane: keeps the side below the position (flip: above)
  const axisIndex = 'xyz'.indexOf(section.axis)
  const lo = bounds ? bounds.min.getComponent(axisIndex) : -50
  const hi = bounds ? bounds.max.getComponent(axisIndex) : 50
  const cutAt = section.position ?? (lo + hi) / 2
  const clip = useMemo(() => {
    if (!section.on) return []
    const n = new THREE.Vector3().setComponent(axisIndex, section.flip ? 1 : -1)
    return [new THREE.Plane(n, section.flip ? -cutAt : cutAt)]
  }, [section.on, section.flip, axisIndex, cutAt])

  // variants sit to the left of the current result, one part width (plus a gap) apart
  const shownVariants = useMemo(() => variants.filter((v) => v.shown), [variants])
  const variantStep = (bounds ? size.x : 100) * 1.15 + 5
  const variantBoxes = useMemo(() => shownVariants.map((v, i) => {
    const box = unionBox(v.views)
    return box.translate(new THREE.Vector3(-(i + 1) * variantStep, 0, 0))
  }), [shownVariants, variantStep])
  const extraBoxes = useMemo(() => [...sliceBoxes, ...variantBoxes], [sliceBoxes, variantBoxes])
  const checkCandidates = visible.filter((id) => viewsMap[id] && viewsMap[id].kind !== 'cad')

  return (
    <div className="viewport"
      onKeyDown={(e) => { if (e.key === 'f' || e.key === 'F') requestFit() }} tabIndex={0}>
      <Canvas camera={{ position: [150, -180, 130], fov: 35, near: 0.1, far: 20000 }} dpr={[1, 2]}
        gl={{ antialias: true }}>
        <ClippingOn />
        <color attach="background" args={['#24262b']} />
        <hemisphereLight args={['#f4f6fa', '#3a3d44', 1.2]} />
        <directionalLight position={[2, -3, 4]} intensity={1.8} />
        <directionalLight position={[-3, 2, -1]} intensity={0.6} />
        {visible.map((id) => viewsMap[id] && (picking?.source === id
          ? <PickableView key={id} view={viewsMap[id]} faces={faceData} chosen={chosen} onPick={onPick} onPaint={onPaint}
              wireframe={wireframe} clip={clip} chosenColor={leaveOut ? PICK_LEFT_OUT : PICK_SELECTED} />
          : <ViewObject key={id} view={viewsMap[id]} wireframe={wireframe} ghost={!!picking} clip={clip}
              paint={paints[id]} cmap={cmap} range={paints[id] && paints[id].source !== 'field' ? ownRanges[id] : fieldRange}
              onHover={setHover} node={id} />))}
        {showMarkers && anyMarkers && <MarkerArrows markers={markers} size={Math.max(size.x, size.y, size.z, 1)} />}
        {showFrames && !picking && frames.map(([id, f], k) => <FrameBars key={`${id}-${k}`} id={id} frame={f} onHover={setBarHover} />)}
        {aim && !picking && (
          <DirectionGizmo anchor={aim.anchor} dir={aim.dir} length={Math.max(size.x, size.y, size.z, 1) * 0.3}
            onCommit={commitAim} torque={aim.torque} centre={aim.centre}
            step={offsetStep(Math.max(size.x, size.y, size.z, 1))} onMove={aim.movable ? commitOffset : undefined} />
        )}
        {shownVariants.map((v, i) => (
          <group key={v.id} position={[-(i + 1) * variantStep, 0, 0]}>
            {v.views.map((view, k) => <ViewObject key={k} view={view} wireframe={wireframe} clip={[]} cmap={cmap}
              range={null} kind="variant" />)}
          </group>
        ))}
        {Object.entries(slices).map(([id, s]) => <SliceObject key={id} slice={s} cmap={cmap} onHover={setHover} />)}
        <Grid position={[center.x, center.y, floorZ]} rotation={[Math.PI / 2, 0, 0]}
          cellSize={cell / 5} sectionSize={cell} cellColor="#34373d" sectionColor="#454951"
          cellThickness={0.6} sectionThickness={1} fadeDistance={span * 6} fadeStrength={1.5} infiniteGrid />
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
        <CameraFit views={views} extra={extraBoxes} onBounds={setBounds} />
        <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
          <GizmoViewport axisColors={['#e5534b', '#57ab5a', '#539bf5']} labelColor="#111" />
        </GizmoHelper>
      </Canvas>

      <div className="viewport-overlay">
        {visible.length === 0 && <div className="viewport-hint">Click ○ on a node to show it here</div>}
        {visible.map((id) => (
          <div className="legend-row" key={id}>
            <span className="legend-swatch" style={{
              background: paints[id] ? gradientCss(paintMap(paints[id], cmap)) : slices[id] ? gradientCss(cmap)
                : KIND_STYLE[viewsMap[id]?.kind ?? 'mesh']?.color }} />
            <span className="legend-name">{id}</span>
            <span className="legend-info">
              {status[id]?.state === 'running' ? 'updating…'
                : viewsMap[id] ? `${(viewsMap[id].faces / 1000).toFixed(viewsMap[id].faces < 10000 ? 1 : 0)}k tris` +
                  (paints[id] ? ` · coloured by ${paints[id].label}${paints[id].unit ? ` (${paints[id].unit})` : ''}` : '')
                : slices[id] ? `field slice, ${slices[id].axis} = ${slices[id].position.toFixed(1)}`
                : status[id]?.state === 'error' ? 'error' : '…'}
            </span>
          </div>
        ))}
        <VariantsList variants={variants} />
      </div>
      {aim && !picking && (
        <div className="gizmo-hint">
          {aim.torque
            ? <>Drag the yellow arrowhead to set the axis <b>{aim.id}</b> turns about (right-hand rule: thumb along
              the arrow, fingers show the turn)</>
            : <>Drag the yellow arrowhead to aim the force of <b>{aim.id}</b>, the ball at its tail to move where it
              acts (in the view's plane)</>}
          {' '}· snaps to the axes and 45° · Esc cancels
        </div>
      )}
      {picking && (
        <div className="pick-banner">
          <span>{leaveOut ? 'Faces to leave out' : 'Picking faces'} for <b className={leaveOut ? 'left-out' : ''}>{picking.node}</b></span>
          {faceError ? <span className="warn">{faceError}</span>
            : !pickView || !faceData || faceData.ids.length !== pickView.faces
              ? <span className="muted">loading {picking.source}…</span>
            : <span className="muted">
              {leaveOut ? 'the effect goes on every face except the red ones · ' : ''}
              click a CAD face to add/remove it · Ctrl+drag: paint over faces ·
              Shift+click: every face joined to it tangentially · {chosen.size} {chosen.size === 1 ? 'face' : 'faces'} {leaveOut ? 'left out' : 'chosen'}
              {faceData && !faceData.same ? ' · different bodies: faces are saved as body:face' : ''}</span>}
          <button disabled={!chosen.size} onClick={() => writeFaces(new Set())} title="Remove every chosen face">Clear</button>
          <button onClick={() => setPicking(null)} title="Esc">Done</button>
        </div>
      )}
      <div className="viewport-tools">
        <button onClick={requestFit} title="Fit view (F)">Fit</button>
        <button className={wireframe ? 'on' : ''} onClick={() => setWireframe((w) => !w)} title="Wireframe">Wire</button>
        <button className={section.on ? 'on' : ''} onClick={() => setSection({ ...section, on: !section.on })}
          title="Cut the view open with a plane to look inside">Section</button>
        <button onClick={addVariant} disabled={!views.length}
          title="Keep a copy of what is shown, to compare with your next changes (shown to the left)">Keep variant</button>
        <button className={checking ? 'on' : ''} title="Print check: size, weight, loose parts, cavities, wall thickness"
          onClick={() => { if (checking) clearPaints('thickness'); setChecking((c) => !c) }}>Check</button>
        {anyMarkers && (
          <button className={showMarkers ? 'on' : ''} onClick={() => setShowMarkers(!showMarkers)}
            title="Show the supports (green) and loads (red arrows) of the Fixed Faces and Load nodes">Loads</button>
        )}
        {frames.length > 0 && (
          <button className={showFrames ? 'on' : ''} onClick={() => setShowFrames(!showFrames)}
            title="Show the bars of the Frame Optimization nodes, coloured by size (hover one for its force); hide a frame's body to see them">Bars</button>
        )}
      </div>
      <div className="tool-panels">
        {!picking && results.map((id) => (
          <ResultPanel key={id} node={id} paint={paints[id]} hover={hover?.node === id ? hover : null} />
        ))}
        {showFrames && frames.length > 0 && !picking && <FramePanel frames={frames} hover={barHover} />}
        {section.on && <SectionPanel section={section} setSection={setSection} lo={lo} hi={hi} />}
        {checking && <CheckPanel candidates={checkCandidates} onClose={() => { clearPaints('thickness'); setChecking(false) }} />}
      </div>
      <SlicePanel hover={hover} />
    </div>
  )
}
