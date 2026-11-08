import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Grid, Html, OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh'
import {
  api, fetchPaint, loadFaceAt, loadFaceInfo, loadFaceRegion, loadFaces, loadNodeAt, loadSection, loadThickness, requestDetail,
  type CheckReport, type MemberSection,
} from '../api'
import { casesKey, parseCases, type LoadCase } from '../loadCases'
import { useStore, type Variant } from '../store'
import type { Axis, ColorMap, FaceData, FaceRef, FaceSelection, FrameView, GroundView, Marker, MenuEntry, Paint, ViewMesh } from '../types'
import { QUALITY_LABEL, isLink, parseFaceKeys, resolveFaces } from '../types'
import {
  STANDARD_VIEWS, barMenu, bodyMenu, faceHitOf, faceKey, faceMenu, importItem, markerMenu, pickingMenu, recipeDiff, sliceMenu,
  variantMenu, viewDirectionItems, type FaceHit, type MarkerHit, type MenuContext,
} from '../viewActions'
import { NamePrompt, QuickEdit } from './QuickEdit'
import { SettingsPanel, useHasSettings } from './SettingsPanel'
import { AnalysisWizard, useWizard } from './AnalysisWizard'
import { TextureWizard, closeTextureWizard, openTextureWizard, useTextureWizard, useTextureWizardPicking } from './TextureWizard'
import { LatticeWizard, closeLatticeWizard, openLatticeWizard, useLatticeWizard, useLatticeWizardPicking } from './LatticeWizard'
import { closeOtherWizards } from '../wizards'
import { ViewportMenu, ViewportToast } from './ViewportMenu'
import {
  SliceObject, SlicePanel, fieldPaintRange, fmt, gradientCss, lut, normalise, sliceBox, type HoverValue,
} from './SliceView'
import { GUIDE_TYPES, Guides, offsetStep, parseVec, vectorText } from './Guides'
import { BodyHint } from './BodyHint'
import { CAD_ENV, CadBody, CadEdges, CadLookPanel, toggleCadEdges, useCadEnvironment, useLook, useLookGeometry } from './CadBody'
import { CAP_COLOR, hideRanges, useBoxCut, type CutBox } from './viewCut'
import { edgesShown, useCadLook } from '../cadLook'

// CAD convention: Z is up.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1)
// Picking faces on bodies with a million triangles needs a search tree (built for picking, and on
// the first right-click on a body).
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree
THREE.Mesh.prototype.raycast = acceleratedRaycast

const PICK_SELECTED = '#4ea3f5'
const PICK_LEFT_OUT = '#e5534b'   // Face Mask "Apply to: all other faces": the picked faces stay plain
const PICK_HOVER = '#ffd166'
const GIZMO_MARGIN = 64       // the axis gizmo's centre from the view's bottom-right corner, px

const KIND_STYLE: Record<string, { color: string; opacity: number }> = {
  mesh: { color: '#c9ced6', opacity: 1 },
  implicit: { color: '#86d6c9', opacity: 1 },
  cad: { color: '#e2b64a', opacity: 0.35 },
  variant: { color: '#b4bfd6', opacity: 1 },
}

/** Builds the geometry's search tree for raycasts once. indirect: the default build re-sorts the
 *  index buffer in place, so triangle numbers (and the CAD face of each) no longer matched: clicking a
 *  side face selected scattered patches. (The option exists in three-mesh-bvh 0.7.8 but is missing
 *  from its type definitions.) */
function ensureBoundsTree(g: THREE.BufferGeometry) {
  if (!g.boundsTree) g.computeBoundsTree({ indirect: true } as unknown as Parameters<THREE.BufferGeometry['computeBoundsTree']>[0])
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

function ViewObject({ view, wireframe, ghost, clip, paint, cmap, range, onHover, kind, node, box, patch, variant, hide }: {
  view: ViewMesh; wireframe: boolean; ghost?: boolean; clip: THREE.Plane[]; paint?: Paint; cmap: ColorMap
  range: [number, number] | null; onHover?: (h: HoverValue | null) => void; kind?: string; node?: string
  /** cut at this box (a detail in it, or this is the detail) */
  box?: CutBox | null
  /** a detail or a list item's piece: not picked, placed on or right-clicked (its view is) */
  patch?: boolean
  /** a kept variant's view: right-clicks open the variant's menu */
  variant?: number
  /** list items drawn by their pieces instead (store.pieces): not drawn here */
  hide?: number[]
}) {
  const cut = useBoxCut(box)
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(view.positions, 3))
    g.setIndex(new THREE.BufferAttribute(view.indices, 1))
    g.computeVertexNormals()
    g.computeBoundingSphere()
    return g
  }, [view])
  useEffect(() => () => { geometry.disposeBoundsTree(); geometry.dispose() }, [geometry])
  useEffect(() => hideRanges(geometry, view.indices, view.parts, 3, hide), [geometry, view, hide])
  const colors = useMemo(() => paintColors(view, paint, cmap, range), [view, paint, cmap, range])
  useEffect(() => {
    if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    else geometry.deleteAttribute('color')
  }, [geometry, colors])
  const style = KIND_STYLE[kind ?? view.kind] ?? KIND_STYLE.mesh
  const opacity = ghost ? Math.min(style.opacity, 0.25) : style.opacity // see-through while picking faces
  const transparent = opacity < 1
  const sectioned = clip.length > 0 && !transparent
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
      <mesh geometry={geometry} renderOrder={transparent ? 1 : 0} raycast={ghost || patch ? () => null : undefined}
        // a guide shape's Place clicks on it (Guides.tsx); right-clicks open its node's menu
        userData={patch ? {} : { placeable: kind !== 'variant',
          menu: node && !ghost ? { view: node } : variant !== undefined ? { variant } : undefined }}
        onPointerMove={colors ? hover : undefined} onPointerOut={colors && onHover ? () => onHover(null) : undefined}>
        <meshStandardMaterial key={colors ? 'painted' : 'plain'} color={colors ? '#ffffff' : style.color}
          vertexColors={!!colors} metalness={colors ? 0.05 : 0.3} roughness={colors ? 0.7 : 0.42}
          wireframe={wireframe} transparent={transparent} opacity={opacity} depthWrite={!transparent}
          side={sectioned ? THREE.FrontSide : THREE.DoubleSide} clippingPlanes={clip} {...cut} />
      </mesh>
      {sectioned && !wireframe && (
        // the inside seen through the cut: back faces in one flat colour read as a solid cut face
        <mesh geometry={geometry} raycast={() => null}>
          <meshBasicMaterial color={CAP_COLOR} side={THREE.BackSide} clippingPlanes={clip} {...cut} />
        </mesh>
      )}
    </group>
  )
}

/** The triangles passing `keep`, as their own small mesh (positions and normals copied from the body's
 *  geometry, whose whole index is `userData.source`; flat normals without them), or null if there are none. */
function faceSubset(geometry: THREE.BufferGeometry, count: number, keep: (triangle: number) => boolean) {
  let n = 0
  for (let t = 0; t < count; t++) if (keep(t)) n++
  if (!n) return null
  const pos = new Float32Array(n * 9)
  const nor = new Float32Array(n * 9)
  const index = (geometry.userData.source ?? geometry.index!.array) as Uint32Array
  const positions = geometry.getAttribute('position').array as Float32Array
  const src = (geometry.getAttribute('normal') as THREE.BufferAttribute | undefined)?.array as Float32Array | undefined
  let k = 0
  for (let t = 0; t < count; t++) {
    if (!keep(t)) continue
    for (let c = 0; c < 3; c++) {
      const v = index[t * 3 + c] * 3
      for (let a = 0; a < 3; a++) {
        pos[k] = positions[v + a]
        nor[k++] = src ? src[v + a] : 0
      }
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  if (src) g.setAttribute('normal', new THREE.BufferAttribute(nor, 3))
  else g.computeVertexNormals()
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

// --- right-click menus ---------------------------------------------------------------------------
/** What objects carry in `userData.menu`: a shown node's view (`picking`: the body faces are being
 *  picked on), a support or load marker, a field slice's node, or a frame's bars. */
interface MenuTag {
  view?: string; picking?: boolean; marker?: MarkerHit; slice?: string; bar?: { node: string; frame: number }; variant?: number
}
/** What a right-click hit: its tag, the point, the surface's normal there (towards the camera), and the
 *  triangle, texture position and instance there. */
interface MenuHit {
  tag: MenuTag; point: [number, number, number]; normal: [number, number, number] | null; triangle: number | null
  uv: [number, number] | null; instance: number | null
}
/** The tagged object under a point of the canvas (`click`: none when something there takes clicks itself). */
type Probe = (x: number, y: number, click?: boolean) => MenuHit | null

/** The longest edge of triangle `t` of a view (the view's resolution there). */
function longestEdge(view: ViewMesh, t: number) {
  const p = (c: number) => new THREE.Vector3().fromArray(view.positions, view.indices[t * 3 + c] * 3)
  const a = p(0), b = p(1), c = p(2)
  return Math.max(a.distanceTo(b), b.distanceTo(c), c.distanceTo(a))
}
/** The face a menu is open for: its triangles in a shown view (numbered in the view's `version`) and in
 *  the detail drawn in the view's box. */
interface MenuFace {
  view: string; version: number; triangles: Uint32Array
  detail?: { version: number; triangles: Uint32Array }
}

/** Lets the right-click handler (outside the canvas) raycast the scene: the nearest object tagged
 *  `userData.menu` under a point, not cut away by the section plane. Raycasting ignores R3F's
 *  pointer events, so a left-drag that turns the view never raycasts the bodies. */
function MenuProbe({ probe, clip }: { probe: { current: Probe | null }; clip: THREE.Plane[] }) {
  const camera = useThree((s) => s.camera)
  const scene = useThree((s) => s.scene)
  const gl = useThree((s) => s.gl)
  useEffect(() => {
    const raycaster = new THREE.Raycaster()
    probe.current = (x, y, click = false) => {
      const r = gl.domElement.getBoundingClientRect()
      raycaster.setFromCamera(new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1), camera)
      if (click) {
        // something that takes clicks itself (the load arrow's handles, a guide shape) keeps them
        const own: THREE.Object3D[] = []
        scene.traverse((o) => {
          const h = (o as { __r3f?: { handlers?: Record<string, unknown> } }).__r3f?.handlers
          if (h && (h.onClick || h.onPointerDown)) own.push(o)
        })
        if (own.length && raycaster.intersectObjects(own, false).length) return null
      }
      const targets: THREE.Object3D[] = []
      scene.traverse((o) => { if (o.userData.menu) targets.push(o) })
      // a body's first right-click builds its search tree (~0.3 s for a million triangles)
      for (const o of targets) {
        if ((o as THREE.Mesh).isMesh && (o.userData.menu.view || o.userData.menu.variant !== undefined)) ensureBoundsTree((o as THREE.Mesh).geometry)
      }
      const hits = raycaster.intersectObjects(targets, false).filter((h) => clip.every((p) => p.distanceToPoint(h.point) >= 0))
      // markers are drawn over the part (supports and loads sit in holes): they win wherever they are
      const hit = hits.find((h) => h.object.userData.menu.marker) ?? hits[0]
      if (!hit) return null
      // the side of the surface the camera sees (bodies are drawn two-sided)
      const normal = hit.face ? hit.face.normal.clone().transformDirection(hit.object.matrixWorld) : null
      if (normal && normal.dot(raycaster.ray.direction) > 0) normal.negate()
      return { tag: hit.object.userData.menu as MenuTag, point: hit.point.toArray() as [number, number, number],
        normal: normal ? normal.toArray() as [number, number, number] : null,
        triangle: hit.faceIndex ?? null, uv: hit.uv ? [hit.uv.x, hit.uv.y] : null, instance: hit.instanceId ?? null }
    }
    return () => { probe.current = null }
  }, [camera, scene, gl, probe, clip])
  return null
}

/** The face a menu is open for (or a chosen face, in `color`), highlighted on its body: the triangles
 *  of `view` it lies on (smooth normals, sharing the view's points), cut at the box like the body under it. */
function MenuHighlight({ view, triangles, clip, box, color = PICK_HOVER }: {
  view: ViewMesh; triangles: Uint32Array; clip: THREE.Plane[]; box?: CutBox | null; color?: string
}) {
  const cut = useBoxCut(box)
  const geometry = useMemo(() => {
    const index = new Uint32Array(triangles.length * 3)
    for (let k = 0; k < triangles.length; k++) {
      for (let c = 0; c < 3; c++) index[k * 3 + c] = view.indices[triangles[k] * 3 + c]
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(view.positions, 3))
    g.setIndex(new THREE.BufferAttribute(index, 1))
    g.computeVertexNormals()
    return g
  }, [view, triangles])
  useEffect(() => () => geometry.dispose(), [geometry])
  if (!triangles.length) return null
  return (
    <mesh geometry={geometry} renderOrder={4} raycast={() => null}>
      <meshStandardMaterial color={color} metalness={0.1} roughness={0.5} side={THREE.DoubleSide}
        polygonOffset polygonOffsetFactor={-4} polygonOffsetUnits={-4} clippingPlanes={clip} {...cut} />
    </mesh>
  )
}

const isChosen = (faces: FaceData, chosen: Set<string>, t: number) =>
  chosen.has(String(faces.ids[t])) || chosen.has(`${faces.bodies[t]}:${faces.ids[t]}`)

/** A CAD body you can pick faces on: click toggles a face, Shift+click the face and every face
 *  smoothly joined to it, Ctrl+drag paints. */
function PickableView({ node, view, faces, chosen, onPick, onPaint, wireframe, clip, chosenColor }: {
  node: string; view: ViewMesh; faces: FaceData | null; chosen: Set<string>; wireframe: boolean; clip: THREE.Plane[]
  chosenColor: string
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
  // drawn as CAD bodies are (CadBody.tsx: crisp face borders, edges), always shaded while picking
  const look = useLook(view, node)
  const geometry = useLookGeometry(view, look)
  ensureBoundsTree(geometry)
  const options = useCadLook()
  const env = useCadEnvironment()
  const [hover, setHover] = useState<{ triangle: number; group: boolean } | null>(null)
  const data = faces && faces.ids.length === view.faces ? faces : null
  const chosenGeo = useMemo(() => (data ? faceSubset(geometry, view.faces, (t) => isChosen(data, chosen, t)) : null),
    [geometry, view, data, chosen])
  const hoverGeo = useMemo(() => {
    if (!data || !hover) return null
    const members = new Set(faceGroup(data, hover.triangle, hover.group))
    return faceSubset(geometry, view.faces, (t) => members.has(faceKey(data, t)))
  }, [geometry, view, data, hover])

  const triangleAt = (e: ThreeEvent<PointerEvent | MouseEvent>) =>
    data && e.faceIndex !== undefined && e.faceIndex !== null ? e.faceIndex : null
  return (
    <group>
      <mesh geometry={geometry} userData={{ menu: { view: node, picking: true } }}
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
        <meshStandardMaterial color={options.color} metalness={0.08} roughness={0.5} envMap={env} envMapIntensity={CAD_ENV}
          side={THREE.DoubleSide} wireframe={wireframe} clippingPlanes={clip} />
      </mesh>
      {look && !wireframe && <CadEdges look={look} options={{ ...options, style: 'shaded' }} clip={clip} />}
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
    // the narrower of the two view angles (a tall, narrow view cut the part off at the sides)
    const half = Math.min(THREE.MathUtils.degToRad(camera.fov / 2), Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect))
    const dist = (radius / Math.sin(half)) * 1.05
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

/** Turns the camera when asked (standard views, Look at this spot): it looks along -dir at the target
 *  (else at what it looks at now) from the same distance. Z stays up (the orbit controls turn about it),
 *  so a view straight down or up is tilted a hair towards the front. */
function CameraTurn() {
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const request = useStore((s) => s.camera)
  useEffect(() => {
    if (!request || !controls) return
    const target = request.target ? new THREE.Vector3(...request.target) : controls.target.clone()
    const dir = new THREE.Vector3(...request.dir).normalize()
    if (Math.abs(dir.z) > 0.9999) dir.set(0, -1e-3, Math.sign(dir.z)).normalize()
    const distance = camera.position.distanceTo(controls.target)
    camera.position.copy(target).addScaledVector(dir, distance)
    controls.target.copy(target)
    controls.update()
  }, [request, camera, controls])
  return null
}

const SECTION_COLOR = '#c8643f'

/** The section plane drawn as an outline at the cut, with a handle in its middle that drags it along
 *  its axis (within lo..hi). */
function SectionGizmo({ axis, position, box, onMove }: {
  axis: Axis; position: number; box: THREE.Box3; onMove: (p: number) => void
}) {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const [drag, setDrag] = useState(false)
  const [hover, setHover] = useState(false)
  const k = 'xyz'.indexOf(axis)
  const lo = box.min.getComponent(k), hi = box.max.getComponent(k)
  const size = box.getSize(new THREE.Vector3())
  const centre = box.getCenter(new THREE.Vector3()).setComponent(k, position)
  const a = new THREE.Vector3().setComponent(k, 1)
  useEffect(() => {
    if (!drag) return
    const raycaster = new THREE.Raycaster()
    const move = (ev: PointerEvent) => {
      const r = gl.domElement.getBoundingClientRect()
      raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), camera)
      // the point of the axis line through the handle nearest to the pointer's ray
      const { origin, direction: d } = raycaster.ray
      const w = centre.clone().sub(origin)
      const b = a.dot(d), denom = 1 - b * b
      if (denom < 1e-6) return
      const t = (b * d.dot(w) - a.dot(w)) / denom
      onMove(Math.min(hi, Math.max(lo, position + t)))
    }
    const up = () => { setDrag(false); if (controls) controls.enabled = true }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
  }, [drag]) // eslint-disable-line react-hooks/exhaustive-deps
  // the outline spans the other two axes
  const [u, v] = [0, 1, 2].filter((i) => i !== k)
  const su = size.getComponent(u) * 1.1 + 2, sv = size.getComponent(v) * 1.1 + 2
  const corner = (cu: number, cv: number) => centre.clone().setComponent(u, centre.getComponent(u) + cu * su / 2)
    .setComponent(v, centre.getComponent(v) + cv * sv / 2)
  const outline = useMemo(() => new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1), corner(-1, -1)]),
    new THREE.LineBasicMaterial({ color: SECTION_COLOR, depthTest: false, transparent: true, opacity: 0.8 })),
  [axis, position, su, sv, centre.x, centre.y, centre.z]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { outline.geometry.dispose(); (outline.material as THREE.Material).dispose() }, [outline])
  const r = Math.max(size.length() * 0.02, 0.5)
  return (
    <group>
      <primitive object={outline} />
      <mesh position={centre} renderOrder={22}
        onPointerOver={(e) => { e.stopPropagation(); setHover(true); gl.domElement.style.cursor = 'grab' }}
        onPointerOut={() => { setHover(false); gl.domElement.style.cursor = '' }}
        onPointerDown={(e) => { if (e.button !== 0) return; e.stopPropagation(); if (controls) controls.enabled = false; setDrag(true) }}>
        <sphereGeometry args={[r, 16, 12]} />
        <meshBasicMaterial color={hover || drag ? '#ffd2bf' : SECTION_COLOR} depthTest={false} transparent opacity={0.95} />
      </mesh>
      {drag && (
        <Html position={centre} center style={{ pointerEvents: 'none' }}>
          <div className="gizmo-label gizmo-label-below">{axis} = {position.toFixed(1)} mm</div>
        </Html>
      )}
    </group>
  )
}

/** "cylinder Ø 8.00 mm (hole)", "plane 12.5 cm²" */
function faceText(info: { type: string; area: number; radius: number | null; hole: boolean }) {
  if (info.radius) return `${info.type} Ø ${(info.radius * 2).toFixed(2)} mm${info.hole ? ' (hole)' : ''}`
  return `${info.type} ${info.area < 100 ? `${info.area.toFixed(1)} mm²` : `${(info.area / 100).toFixed(2)} cm²`}`
}

/** A measured point: where, the surface's normal there, on which view, and what was found there. */
interface MeasurePoint {
  id: number
  point: [number, number, number]
  normal: [number, number, number] | null
  view: string | null
  /** the CAD face under it ("cylinder Ø 8.0 mm (hole)") and the wall thickness under it, once known */
  face?: string | null
  thickness?: number | null
  /** the cross-section of the member under it, once known (null: none, no wall behind) */
  section?: MemberSection | null
}

const MEMBER_CUT_COLOR = '#4cc9f0'

/** "12.4 mm²", "1.25 cm²" */
function areaText(area: number) {
  return area < 1000 ? `${area.toFixed(area < 10 ? 2 : 1)} mm²` : `${(area / 100).toFixed(2)} cm²`
}

/** A member section's cut as a flat shape (holes left open) in its own plane. */
function sectionFill(s: Extract<MemberSection, { area: number }>) {
  const n = new THREE.Vector3(...s.normal).normalize()
  const e1 = new THREE.Vector3().crossVectors(n, Math.abs(n.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)).normalize()
  const e2 = new THREE.Vector3().crossVectors(n, e1)
  const c = new THREE.Vector3(...s.centre)
  const flat = (loop: [number, number, number][]) => loop.map((q) => {
    const v = new THREE.Vector3(...q).sub(c)
    return new THREE.Vector2(v.dot(e1), v.dot(e2))
  })
  const shape = new THREE.Shape(flat(s.outline))
  shape.holes = s.holes.filter((h) => h.length > 2).map((h) => new THREE.Path(flat(h)))
  const g = new THREE.ShapeGeometry(shape)
  g.applyMatrix4(new THREE.Matrix4().makeBasis(e1, e2, n).setPosition(c))
  return g
}

/** The member sections found at the measured points: outline and a see-through fill, over everything, with the area. */
function SectionMarks({ points }: { points: MeasurePoint[] }) {
  const sections = points.map((p) => p.section).filter((s): s is Extract<MemberSection, { area: number }> => !!s && !s.open)
  const key = sections.map((s) => s.centre.join(',')).join(';')
  const objects = useMemo(() => {
    const line = new THREE.LineBasicMaterial({ color: MEMBER_CUT_COLOR, depthTest: false })
    const fill = new THREE.MeshBasicMaterial({ color: MEMBER_CUT_COLOR, transparent: true, opacity: 0.35, depthTest: false,
      depthWrite: false, side: THREE.DoubleSide })
    const out: THREE.Object3D[] = []
    for (const s of sections) {
      for (const loop of [s.outline, ...s.holes]) {
        if (loop.length > 2) out.push(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(loop.map((q) => new THREE.Vector3(...q))), line))
      }
      try { out.push(new THREE.Mesh(sectionFill(s), fill)) } catch { /* (an outline earcut cannot fill: the lines stay) */ }
    }
    return { out, materials: [line, fill] }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => {
    objects.out.forEach((o) => (o as THREE.Mesh).geometry?.dispose())
    objects.materials.forEach((m) => m.dispose())
  }, [objects])
  return (
    <group>
      {objects.out.map((o, i) => <primitive key={i} object={o} renderOrder={24} raycast={() => null} />)}
      {sections.map((s, i) => (
        <Html key={`s${i}`} position={s.centre} style={{ pointerEvents: 'none' }}>
          <div className="gizmo-label measure-section-label">{areaText(s.area)}</div>
        </Html>
      ))}
    </group>
  )
}

/** The measured points (A, B) and the line between them, drawn over everything. */
function MeasureMarks({ points, size }: { points: MeasurePoint[]; size: number }) {
  const r = Math.max(size * 0.006, 0.15)
  const line = useMemo(() => (points.length === 2
    ? new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p.point))),
      new THREE.LineBasicMaterial({ color: '#ffd166', depthTest: false })) : null), [points])
  useEffect(() => () => { line?.geometry.dispose(); (line?.material as THREE.Material | undefined)?.dispose() }, [line])
  const mid = points.length === 2 ? new THREE.Vector3(...points[0].point).add(new THREE.Vector3(...points[1].point)).multiplyScalar(0.5) : null
  return (
    <group>
      {points.map((p, i) => (
        <mesh key={i} position={p.point} renderOrder={23} raycast={() => null}>
          <sphereGeometry args={[r, 12, 8]} />
          <meshBasicMaterial color="#ffd166" depthTest={false} />
        </mesh>
      ))}
      {line && <primitive object={line} renderOrder={23} />}
      <SectionMarks points={points} />
      {points.map((p, i) => (
        <Html key={`l${i}`} position={p.point} style={{ pointerEvents: 'none' }}>
          <div className="gizmo-label measure-label">{'AB'[i]}</div>
        </Html>
      ))}
      {mid && (
        <Html position={mid} center style={{ pointerEvents: 'none' }}>
          <div className="gizmo-label">{new THREE.Vector3(...points[0].point).distanceTo(new THREE.Vector3(...points[1].point)).toFixed(2)} mm</div>
        </Html>
      )}
    </group>
  )
}

const DETAIL_DELAY = 600       // ms the view must rest (camera still, nothing running) before a detail is asked for
const DETAIL_SAMPLES = 40000   // triangles per view looked at to find what is on screen
const PX_PER_VOXEL = 1.25      // a detail's voxel spans about this many screen pixels where it is nearest
const DETAIL_GAIN = 1.5        // ...and is at least this much finer than the view's, or it is not worth it

/** The part of each shown body on screen and how fine it needs to be there: its front-facing triangles
 *  inside the view, a box around them and the voxel that spans PX_PER_VOXEL pixels at the nearest
 *  fifth of them. Null when the body is too small on screen to need more than its view. */
function onScreen(view: ViewMesh, camera: THREE.PerspectiveCamera, height: number) {
  const step = Math.max(1, Math.ceil(view.faces / DETAIL_SAMPLES))
  const P = view.positions, I = view.indices
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3()
  const centre = new THREE.Vector3(), seen = new THREE.Vector3(), eye = camera.position
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  const depths: number[] = []
  const toView = camera.matrixWorldInverse
  for (let t = 0; t < view.faces; t += step) {
    a.fromArray(P, I[t * 3] * 3); b.fromArray(P, I[t * 3 + 1] * 3); c.fromArray(P, I[t * 3 + 2] * 3)
    centre.copy(a).add(b).add(c).divideScalar(3)
    n.subVectors(b, a).cross(c.sub(a))
    if (n.dot(seen.subVectors(centre, eye)) >= 0) continue  // facing away
    seen.copy(centre).applyMatrix4(toView)
    const depth = -seen.z
    if (depth <= camera.near) continue
    seen.applyMatrix4(camera.projectionMatrix)
    if (Math.abs(seen.x) > 1 || Math.abs(seen.y) > 1) continue
    depths.push(depth)
    for (let k = 0; k < 3; k++) {
      const v = centre.getComponent(k)
      if (v < lo[k]) lo[k] = v
      if (v > hi[k]) hi[k] = v
    }
  }
  if (depths.length < 30) return null
  depths.sort((x, y) => x - y)
  const pixel = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) / height  // mm per pixel at depth 1
  const voxel = Math.max(PX_PER_VOXEL * pixel * depths[Math.floor(depths.length * 0.2)], view.floor ?? Infinity)
  // a margin of a few of the view's voxels: what lies between the samples, and the box's own edge
  const pad = 3 * (view.voxel ?? voxel)
  return { lo: lo.map((v) => v - pad) as [number, number, number], hi: hi.map((v) => v + pad) as [number, number, number], voxel }
}

/** "0.05 mm" with two significant digits */
export const fmtVoxel = (v: number) => `${Number(v.toPrecision(2))} mm`

/** What a view's voxel means, on its legend line: below Final, thin things are not what the file will have
 *  (a 0.6 mm shell had a quarter of its volume in Draft). */
function voxelNote(view: ViewMesh | undefined) {
  if (!view?.voxel || view.exact) return undefined
  const coarse = view.level && view.level !== 'final'
  return `Meshed with ${fmtVoxel(view.voxel)} voxels. Walls, bumps and gaps thinner than about two voxels ` +
    `(${fmtVoxel(2 * view.voxel)}) look thinner or vanish` + (coarse
    ? `: ${QUALITY_LABEL[view.level!]} is ${view.level === 'draft' ? 4 : 2}x coarser than the recipe. Final (or zooming in, with Refine) shows them.`
    : '.')
}

const inside = (lo: number[], hi: number[], outerLo: number[], outerHi: number[]) =>
  lo.every((v, k) => v >= outerLo[k]) && hi.every((v, k) => v <= outerHi[k])

/** Refine (zoomed in): when the view has rested for a moment and nothing runs, asks the engine for the
 *  part of the shown body on screen meshed finer than its view, one body at a time, the one that gains
 *  most first. Bodies coloured by a field or result, CAD bodies and picking get none. */
function DetailProbe() {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const height = useThree((s) => s.size.height)
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const asked = useRef<Record<string, { key: string; lo: number[]; hi: number[]; voxel: number }>>({})
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined
    const check = async () => {
      const s = useStore.getState()
      const run = s.run
      const busy = !s.refine || !s.connected || s.picking || s.placing || s.editing || s.requested ||
        (run && (run.running || run.next || (run.tag && run.tag.version !== s.graphVersion)))
      if (busy) return
      camera.updateMatrixWorld()
      let best: { id: string; view: ViewMesh; lo: number[]; hi: number[]; voxel: number; gain: number } | null = null
      for (const id of s.visible) {
        const view = s.views[id]
        if (!view?.key || view.exact || !view.voxel || !view.floor || s.paints[id]) continue
        const want = onScreen(view, camera, height)
        if (!want || want.voxel * DETAIL_GAIN > view.voxel) continue
        // already shown that fine there, or asked for (the engine may have had to make it coarser)
        const d = s.details[id]
        if (d && d.key === view.key && d.voxel <= want.voxel * 1.3 && inside(want.lo, want.hi, d.lo, d.hi)) continue
        const a = asked.current[id]
        if (a && a.key === view.key && a.voxel <= want.voxel * 1.3 && inside(want.lo, want.hi, a.lo, a.hi)) continue
        const gain = view.voxel / want.voxel
        if (!best || gain > best.gain) best = { id, view, ...want, gain }
      }
      if (!best) return
      asked.current[best.id] = { key: best.view.key!, lo: best.lo, hi: best.hi, voxel: best.voxel }
      if (!(await requestDetail(best.id, best.view, best.lo, best.hi, best.voxel))) delete asked.current[best.id]
    }
    const schedule = () => { clearTimeout(timer); timer = setTimeout(check, DETAIL_DELAY) }
    controls?.addEventListener('change', schedule)
    const unsubscribe = useStore.subscribe((s, prev) => {
      if (s.run !== prev.run || s.views !== prev.views || s.details !== prev.details || s.refine !== prev.refine ||
          s.paints !== prev.paints || s.picking !== prev.picking) schedule()
    })
    schedule()
    return () => { clearTimeout(timer); controls?.removeEventListener('change', schedule); unsubscribe() }
  }, [camera, controls, height])
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
function CheckPanel({ candidates, initial, onClose }: { candidates: string[]; initial?: string; onClose: () => void }) {
  const quality = useStore((s) => s.quality)
  const views = useStore((s) => s.views)
  const paints = useStore((s) => s.paints)
  const [node, setNode] = useState(initial ?? candidates[0] ?? '')
  useEffect(() => { if (initial) setNode(initial) }, [initial])  // a body's menu chose it
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

// --- measuring --------------------------------------------------------------------------------
function MeasurePanel({ points, onClose }: { points: MeasurePoint[]; onClose: () => void }) {
  const xyz = (p: [number, number, number]) => p.map((v) => v.toFixed(2)).join(', ')
  const d = points.length === 2 ? new THREE.Vector3(...points[1].point).sub(new THREE.Vector3(...points[0].point)) : null
  return (
    <div className="tool-panel measure-panel">
      <div className="slice-row">
        <span className="slice-title">Measure</span>
        <button className="small-btn" onClick={onClose} title="Stop measuring (M, Esc)">×</button>
      </div>
      {!points.length && (
        <div className="muted small">
          Click on a body: point A, then B (a third click starts again). Each point shows the wall and the member's
          cross-section under it.
        </div>
      )}
      {points.map((p, i) => (
        <div className="check-report" key={p.id}>
          <Row k={'AB'[i]} v={`${xyz(p.point)} mm`} title={p.view ? `on ${p.view}` : undefined} />
          {p.face !== undefined && p.face !== null && <Row k="Face" v={p.face} />}
          {p.normal && p.thickness !== undefined && (
            <Row k="Wall" v={p.thickness === null ? 'no wall behind' : `${p.thickness.toFixed(2)} mm`}
              title="Straight into the part from this point, through the shown mesh" />
          )}
          {p.normal && p.view && <SectionRows section={p.section} />}
        </div>
      ))}
      {d && (
        <div className="check-report">
          <Row k="Distance" v={`${d.length().toFixed(2)} mm`} />
          <Row k="ΔX ΔY ΔZ" v={`${d.x.toFixed(2)}, ${d.y.toFixed(2)}, ${d.z.toFixed(2)}`} />
        </div>
      )}
    </div>
  )
}

const SECTION_TITLE = 'The cross-section of the member under this point: the smallest cut through its middle (halfway ' +
  'straight in), through the shown mesh, holes taken off. Compare it with Topology Optimization\'s Max section (mm²).'

/** A measured point's member section: area (and the round member as large), how wide across, or why there is none. */
function SectionRows({ section }: { section: MemberSection | null | undefined }) {
  if (section === undefined) return <Row k="Section" v="finding…" title={SECTION_TITLE} />
  if (section === null) return null
  if (section.open) {
    return <Row k="Section" v="a wall or plate, not a member"
      title={`No closed cut within ${Math.round(24 * section.thickness)} mm (24 wall thicknesses) of the middle. ${SECTION_TITLE}`} />
  }
  return (
    <>
      <Row k="Section" v={`${areaText(section.area)} (round Ø ${section.diameter.toFixed(2)} mm)`} title={SECTION_TITLE} />
      <Row k="Across" v={`${section.narrowest.toFixed(2)} – ${section.widest.toFixed(2)} mm`}
        title="The cut's narrowest and widest extent, measured across its outline" />
    </>
  )
}

function Row({ k, v, warn, title }: { k: string; v: string; warn?: boolean; title?: string }) {
  return <div className={`check-row ${warn ? 'warn' : ''}`} title={title}><span>{k}</span><span>{v}</span></div>
}

// --- variants: kept copies shown next to the current result ----------------------------------
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
            <button className="small-btn" title="Restore: put this variant's settings back (Undo takes them away again)"
              disabled={!diff.length} onClick={() => useStore.getState().restoreVariant(v.id)}>↺</button>
            <button className="small-btn" title="Remove" onClick={() => remove(v.id)}>×</button>
          </div>
        )
      })}
    </div>
  )
}

// --- simulation: supports, loads and results -----------------------------------------------------
const LOAD_COLOR = 0xff4d6d
const LOAD_CSS = '#ff4d6d'
const FIXED_COLOR = 0x3fcf6e

/** The points of a marker far enough apart to draw (`gap` scene units), the most central first: a
 *  small face (a 4 mm boss) gets one arrow instead of a bundle of 24 on top of each other. */
function spreadOut(points: [number, number, number][], gap: number) {
  const centre = points.reduce((a, p) => a.add(new THREE.Vector3(...p)), new THREE.Vector3())
    .divideScalar(Math.max(points.length, 1))
  const order = points.map((p, i) => [new THREE.Vector3(...p).distanceToSquared(centre), i] as [number, number])
    .sort((a, b) => a[0] - b[0])
  const kept: number[] = []
  for (const [, i] of order) {
    const p = new THREE.Vector3(...points[i])
    if (kept.every((k) => p.distanceTo(new THREE.Vector3(...points[k])) >= gap)) kept.push(i)
  }
  return kept
}

const HIT_GEOMETRY = new THREE.SphereGeometry(1, 8, 6)
const HIT_MATERIAL = new THREE.MeshBasicMaterial({ visible: false })  // not drawn, but raycast

/** Arrows for the loads (tip on the face when pushing, tail when pulling; shorter where the load
 *  presses less, as on a bearing load's sides) and small cones pointing into the held faces, sized
 *  to the scene. Each drawn point has an invisible ball around it for right-clicks (the arrows' thin
 *  lines are hard to hit), tagged with its node, marker and point. */
function MarkerArrows({ markers, size, colors, faded }: {
  markers: Record<string, Marker[]>; size: number
  /** loads in a load case: the case's colour, by node */
  colors: Record<string, string>
  /** nodes whose arrows are drawn faint (hovering a case in the legend) */
  faded: Set<string>
}) {
  const group = useMemo(() => {
    const g = new THREE.Group()
    const hits: THREE.Mesh[] = []
    const faint: THREE.ArrowHelper[] = []
    const len = size * 0.06
    for (const [node, list] of Object.entries(markers)) {
      for (const [index, m] of list.entries()) {
        for (const i of spreadOut(m.points, len * 0.4)) {
          const dir = new THREE.Vector3(...m.dirs[i]).normalize()
          const at = new THREE.Vector3(...m.points[i])
          let start: THREE.Vector3, l: number, arrow: THREE.ArrowHelper
          if (m.kind === 'fixed') {
            l = len * 0.4
            start = at.clone().addScaledVector(dir, -l)
            arrow = new THREE.ArrowHelper(dir, start, l, FIXED_COLOR, l, l * 0.6)
          } else {
            l = len * (0.25 + 0.75 * (m.sizes?.[i] ?? 1))
            start = m.push[i] ? at.clone().addScaledVector(dir, -l) : at
            arrow = new THREE.ArrowHelper(dir, start, l, colors[node] ?? LOAD_COLOR, l * 0.3, l * 0.16)
          }
          g.add(arrow)
          if (faded.has(node)) faint.push(arrow)
          // the ball covers the arrow (half the gap between points, at least)
          const hit = new THREE.Mesh(HIT_GEOMETRY, HIT_MATERIAL)
          hit.position.copy(start).addScaledVector(dir, l / 2)
          hit.scale.setScalar(Math.max(l / 2, len * 0.2))
          hit.userData.menu = { marker: { node, index, point: i } }
          hits.push(hit)
        }
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
    for (const a of faint) for (const part of [a.line, a.cone]) (part.material as THREE.Material).opacity = 0.12
    if (hits.length) g.add(...hits)
    return g
  }, [markers, size, colors, faded])
  useEffect(() => () => group.children.forEach((c) => { if (c instanceof THREE.ArrowHelper) c.dispose() }), [group])
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
/** a frame being optimized, before its bars are sized: pulled and pushed bars, and the body check's weakest spot */
const TENSION_COLOUR = '#ff6b6b'
const COMPRESSION_COLOUR = '#4ea3f5'
const SPOT_COLOUR = '#ffd84d'
/** Replay: ms per step (the wall bracket's ~100 steps in ~20 s) */
const REPLAY_STEP_MS = 200

/** Each size of a frame (sorted) and its colour: smallest blue to largest violet. */
function sizeColours(sizes: number[]) {
  const unique = [...new Set(sizes)].sort((a, b) => a - b)
  const last = Math.max(unique.length - 1, 1)
  return new Map(unique.map((s, i) => [s, SIZE_COLOURS[Math.round((i * (SIZE_COLOURS.length - 1)) / last)]]))
}

interface BarHover {
  node: string; bar: number; size: number; length: number; force: number; utilisation: number; profile: string
  /** a live frame's bar not sized yet: `size` is what its force needs */
  unsized?: boolean
}

/** A frame's bars as tubes at their size (drawn round whatever the profile), with a ball at each
 *  node; hovering one reports its bar, right-clicking one opens its menu (frame `index` of node
 *  `id`). Much cheaper than meshing the frame's body. A live frame (being optimized) before sizing:
 *  tubes as thick as their force needs, red pulled, blue pushed; in the body check its weakest spot. */
function FrameBars({ id, index, frame, onHover }: {
  id: string; index: number; frame: FrameView; onHover: (h: BarHover | null) => void
}) {
  const unsized = !!frame.live && !frame.sized
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
      tubes.setColorAt(i, new THREE.Color(unsized ? (frame.axial[i] < 0 ? COMPRESSION_COLOUR : TENSION_COLOUR)
        : colours.get(frame.sizes[frame.chains[i]]) ?? '#cccccc'))
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
    // (live frames are not the node's result yet: no bar menu)
    if (!frame.live) tubes.userData.menu = { bar: { node: id, frame: index } }
    const g = new THREE.Group()
    g.add(tubes, balls)
    const parts: THREE.Mesh[] = [tubes, balls]
    if (frame.spot) {
      const r = Math.max(1.5, 0.35 * Math.max(...frame.sizes))   // (a mark, not hiding the joint)
      const spot = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 8),
        new THREE.MeshBasicMaterial({ color: SPOT_COLOUR, wireframe: true, depthTest: false, transparent: true, opacity: 0.9 }))
      spot.position.set(...frame.spot)
      spot.renderOrder = 3
      spot.raycast = () => null
      g.add(spot)
      parts.push(spot)
    }
    return { g, tubes, parts }
  }, [frame, id, index, unsized])
  useEffect(() => () => {
    for (const o of group.parts) {
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
          utilisation: frame.utilisation[c], profile: frame.profile, unsized })
      }}
      onPointerOut={() => onHover(null)} />
  )
}

/** The sizes of the shown frames with their colours, and the bar under the mouse. */
/** The load cases (Load Case nodes) in their colours: hover one to see its loads' arrows alone (with the
 *  supports and the loads in no case, which act in every case), click it to go to its node. */
function CasePanel({ cases, others, onHover }: { cases: LoadCase[]; others: boolean; onHover: (name: string | null) => void }) {
  return (
    <div className="tool-panel">
      <div className="slice-row"><span className="slice-title"><b>Load cases</b>: each acts on its own, the part must
        stand every one</span></div>
      <div className="frame-sizes">
        {cases.map((c) => (
          <button key={c.name} className="case-row" onMouseEnter={() => onHover(c.name)} onMouseLeave={() => onHover(null)}
            onClick={() => useStore.getState().focusNode(c.nodes[0])}
            title={`${c.loads.length ? c.loads.join(', ') : 'no Load nodes yet'} · click: its Load Case node`}>
            <span className="legend-swatch" style={{ background: c.color }} />{c.name}
          </button>
        ))}
        {others && (
          <span className="frame-size" title="Loads in no Load Case act in every case (as the supports do)">
            <span className="legend-swatch" style={{ background: LOAD_CSS }} />in every case
          </span>
        )}
      </div>
    </div>
  )
}

/** Frame Optimization's steps (optimize.STAGES), their colours in the calculation chart and what each does. */
const STAGES = ['layout', 'mid-bar joints', 'moving joints', 'sizing', 'body check']
const STAGE_COLOUR: Record<string, string> = {
  layout: '#4ea3f5', 'mid-bar joints': '#9b7bf0', 'moving joints': '#f0a04e', sizing: '#7fcf4f', 'body check': '#e5534b',
}
const STAGE_INFO: Record<string, string> = {
  layout: 'A linear program picks, of every candidate bar between the nodes, the lightest set that carries the loads to ' +
    'the supports at the allowed stress. Member adding solves it on the short bars first and takes in the bars whose ' +
    'strain says they would help; then rounds with every bar at least Min size make bars that carry little dearer, so ' +
    'the load moves onto fewer, fuller bars.',
  'mid-bar joints': 'Nodes added along the bars the layout uses, and the layout solved again: branches may now meet part ' +
    'way along a bar.',
  'moving joints': 'The joints slide off the lattice: each step is an LP in the forces and the joints\' moves together, ' +
    'kept where the frame gets lighter. Joints that meet merge, idle bars go.',
  sizing: 'The beam model checks every bar with its real section (stress, bending at the joints, buckling) and makes ' +
    'the bars that fail a size thicker until all pass.',
  'body check': 'Static Analysis of the printed body with its joints and fillets: the bars at the most stressed spot ' +
    'get thicker until the body reaches the safety factor.',
}

const cm3 = (mm3: number) => `${(mm3 / 1000).toPrecision(3)} cm³`

/** How a frame is made, at step `at` of a run (`steps`: its live frames): the stages done, now and to come, what
 *  this one does, its LP's numbers, `tick` (what the LP is doing now, while it runs) and the bars' mass at every
 *  step so far as a chart, coloured by stage; a click on it goes to that step (`onAt`). */
function CalcView({ steps, at, tick, done, onAt }: {
  steps: FrameView[]; at: number; tick: string | null; done?: boolean; onAt?: (i: number) => void
}) {
  const f = steps[at]
  const now = done ? STAGES.length : STAGES.indexOf(f.stage ?? '')
  const seen = new Set(steps.slice(0, at + 1).map((s) => s.stage))
  const n = done ? undefined : f.numbers
  const W = 306
  const H = 64
  const pad = 4
  const masses = steps.map((s) => s.mass ?? 0)
  const lo = Math.min(...masses)
  const hi = Math.max(...masses)
  const x = (i: number) => pad + (W - 2 * pad) * (steps.length > 1 ? i / (steps.length - 1) : 0.5)
  // (a log scale: the first member adding round's layout of short bars can weigh ten times the rest)
  const ln = (m: number) => Math.log(Math.max(m, 1e-6))
  const y = (m: number) => H - pad - (H - 2 * pad - 10) * (hi > lo ? (ln(m) - ln(lo)) / (ln(hi) - ln(lo)) : 0.5)
  const hit = Math.max(2, (W - 2 * pad) / Math.max(1, steps.length))
  return (
    <div className="calc">
      <div className="calc-stages">
        {STAGES.map((s, i) => {
          const state = i < now ? (seen.has(s) ? 'done' : 'skipped') : i === now ? 'now' : 'next'
          return (
            <span key={s} className={`calc-stage ${state}`} title={state === 'skipped' ? `${s}: off for this frame` : STAGE_INFO[s]}>
              <span className="legend-swatch" style={{ background: STAGE_COLOUR[s] }} />
              {state === 'done' ? '✓ ' : ''}{s}
            </span>
          )
        })}
      </div>
      {!done && <div className="muted small calc-info">{STAGE_INFO[f.stage ?? ''] ?? ''}</div>}
      {n?.lp !== undefined && (
        <div className="calc-numbers">
          LP {n.lp}: {n.in_lp?.toLocaleString()} of {n.candidates?.toLocaleString()} candidate bars in it
          {n.adding ? <> · {n.adding.toLocaleString()} more next</> : null}
          {n.volume !== undefined && <><br />{cm3(n.volume)} of bars at their forces
            {n.real !== undefined && <>, {cm3(n.real)} with every bar at least Min size</>}</>}
        </div>
      )}
      {tick && <div className="calc-tick">{tick}</div>}
      {steps.length > 1 && (
        <svg className="calc-chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
          <text x={pad} y={9} className="calc-axis">{hi.toPrecision(3)} g</text>
          <text x={W - pad} y={9} className="calc-axis" textAnchor="end">
            {done ? `mass of the bars over ${steps.length} steps · click: replay from there`
              : `mass of the bars, step ${at + 1} of ${steps.length}`}</text>
          {steps.slice(1).map((s, k) => (
            <line key={k} x1={x(k)} y1={y(masses[k])} x2={x(k + 1)} y2={y(masses[k + 1])}
              stroke={STAGE_COLOUR[s.stage ?? ''] ?? '#888'} strokeWidth={1.5} />
          ))}
          {!done && <circle cx={x(at)} cy={y(masses[at])} r={3.5} fill={STAGE_COLOUR[f.stage ?? ''] ?? '#888'}
            stroke="#fff" strokeWidth={1} />}
          <text x={pad} y={H - 1} className="calc-axis">{lo.toPrecision(3)} g</text>
          {onAt && steps.map((s, i) => (
            <rect key={i} x={x(i) - hit / 2} y={0} width={hit} height={H} fill="transparent" style={{ cursor: 'pointer' }}
              onClick={() => onAt(i)}>
              <title>{`step ${i + 1}: ${s.stage}: ${s.what} · ${(s.mass ?? 0).toPrecision(3)} g`}</title>
            </rect>
          ))}
        </svg>
      )}
      {!done && f.mass !== undefined && <div className="muted small">{f.mass.toPrecision(3)} g of bars as drawn
        {f.sized ? '' : ' (each as thick as its force needs, at least Min size)'}</div>}
    </div>
  )
}

/** Replay of a node's last optimization (its live frames, `store.liveHistory`): `at` null = not replaying. */
function ReplayBar({ steps, at, onAt }: { steps: FrameView[]; at: number | null; onAt: (at: number | null) => void }) {
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (!playing || at === null) return
    if (at >= steps.length - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => onAt(at + 1), REPLAY_STEP_MS)
    return () => clearTimeout(t)
  }, [playing, at, steps.length, onAt])
  if (at === null) {
    return (
      <div className="slice-row">
        <button className="small-btn" onClick={() => { onAt(0); setPlaying(true) }}
          title="Watch the optimization again step by step: the layout's rounds, joints moving, sizing and the body check">
          ▶ Replay the optimization</button>
        <span className="muted small">{steps.length} steps</span>
      </div>
    )
  }
  const go = (i: number) => { setPlaying(false); onAt(Math.max(0, Math.min(steps.length - 1, i))) }
  return (
    <div className="slice-row">
      <button className="small-btn" title={playing ? 'Pause' : 'Play'}
        onClick={() => { if (at >= steps.length - 1) onAt(0); setPlaying(!playing) }}>{playing ? '❚❚' : '▶'}</button>
      <button className="small-btn" title="One step back" onClick={() => go(at - 1)}>‹</button>
      <input type="range" min={0} max={steps.length - 1} step={1} value={at} onChange={(e) => go(Number(e.target.value))} />
      <button className="small-btn" title="One step on" onClick={() => go(at + 1)}>›</button>
      <span className="slice-value">{at + 1} / {steps.length}</span>
      <button className="small-btn" title="Back to the result" onClick={() => { setPlaying(false); onAt(null) }}>✕</button>
    </div>
  )
}

function FramePanel({ frames, hover, history, replay, onReplay }: {
  frames: [string, FrameView, number][]; hover: BarHover | null
  history: Record<string, { steps: FrameView[] }>; replay: { node: string; at: number } | null
  onReplay: (node: string, at: number | null) => void
}) {
  const status = useStore((s) => s.status)
  /** what the node's LP is doing now (its progress step "layout (LP 8, interior point iteration 28)") */
  const ticking = (id: string, stage?: string) => {
    const step = status[id]?.state === 'running' ? status[id]?.step : null
    return step && stage && step.startsWith(`${stage} (`) && step.endsWith(')')
      ? `solving ${step.slice(stage.length + 2, -1)}` : null
  }
  return (
    <div className="tool-panel">
      {frames.map(([id, f]) => {
        const colours = sizeColours(f.sizes)
        const length = new Map<number, number>()
        f.sizes.forEach((s, c) => length.set(s, (length.get(s) ?? 0) + f.lengths[c]))
        const unsized = f.live && !f.sized
        const replaying = replay?.node === id
        const steps = history[id]?.steps
        return (
          <div key={id}>
            {f.live && (
              <div className="slice-row"><span className="slice-title">
                <b>{id}</b> {replaying ? 'replay' : 'optimizing'} · {f.stage}: {f.what}</span></div>
            )}
            {steps && steps.length > 1 && (!f.live || replaying) && (
              <ReplayBar steps={steps} at={replaying ? replay.at : null} onAt={(at) => onReplay(id, at)} />
            )}
            {steps && steps.length > 0 && (f.live || steps.length > 1) && (
              <CalcView steps={steps} at={replaying ? replay.at : steps.length - 1} done={!f.live}
                tick={f.live && !replaying ? ticking(id, f.stage) : null}
                onAt={f.live && !replaying ? undefined : (at) => onReplay(id, at)} />
            )}
            <div className="slice-row"><span className="slice-title">{f.live ? '' : <><b>{id}</b>: </>}{f.sizes.length} bars
              {unsized ? ', as thick as their force needs' : `, ${f.profile}${f.profile === 'round' ? '' : ' (drawn round here)'}`}
            </span></div>
            <div className="frame-sizes">
              {unsized ? (
                <>
                  <span className="frame-size"><span className="legend-swatch" style={{ background: TENSION_COLOUR }} />tension</span>
                  <span className="frame-size"><span className="legend-swatch" style={{ background: COMPRESSION_COLOUR }} />compression</span>
                </>
              ) : [...colours.entries()].map(([s, col]) => (
                <span key={s} className="frame-size" title={`${(length.get(s) ?? 0).toFixed(0)} mm of bar this size`}>
                  <span className="legend-swatch" style={{ background: col }} />{s} mm
                </span>
              ))}
              {f.spot && (
                <span className="frame-size" title="where the body check found the most stress: the bars there get thicker">
                  <span className="legend-swatch" style={{ background: SPOT_COLOUR }} />weakest spot
                </span>
              )}
            </div>
          </div>
        )
      })}
      <div className="muted small">
        {hover
          ? hover.unsized
            ? <>bar {hover.bar} of <b>{hover.node}</b>: {Math.abs(hover.force).toFixed(0)} N
              {' '}{hover.force < 0 ? 'compression' : 'tension'}, needs {hover.size.toFixed(1)} mm, {hover.length.toFixed(1)} mm long</>
            : <>bar {hover.bar} of <b>{hover.node}</b>: {hover.size} mm {hover.profile}, {hover.length.toFixed(1)} mm long,
              {' '}{Math.abs(hover.force).toFixed(0)} N {hover.force < 0 ? 'compression' : 'tension'} here,
              utilisation {hover.utilisation.toFixed(2)}</>
          : 'hover a bar: its size, length, force and how much of its strength it uses'}
      </div>
    </div>
  )
}

// --- candidate nodes: where a frame's bars may meet (Frame Optimization, Show: candidate nodes) ----------
const GRID_COLOUR = '#aab3c2'
const ATTACH_COLOUR = '#4ea3f5'
const LONELY_COLOUR = '#ff2d2d'
const POINT_COLOUR = { fixed: '#3fcf6e', load: LOAD_CSS, attach: ATTACH_COLOUR }
const POINT_KIND = { fixed: 'support', load: 'load', attach: 'connection' }

interface GroundHover { node: string; label: string; kind: 'fixed' | 'load' | 'attach'; lonely: boolean; point: [number, number, number] }

/** A frame's candidate nodes as dots: the lattice small and grey, the nodes at supports, loads and
 *  connections larger in their colours, and a node no bar reaches as a large red ball with a ring, drawn
 *  over everything. With the section on, only the lattice layer just behind the cut (one spacing deep), so
 *  a cut through a wall shows whether any dots lie in it; supports and loads are cut as the part is. */
function GroundNodes({ id, ground, clip, onHover }: {
  id: string; ground: GroundView; clip: THREE.Plane[]; onHover: (h: GroundHover | null) => void
}) {
  const group = useMemo(() => {
    const s = ground.spacing
    const plane = clip[0]
    const m = new THREE.Matrix4()
    const q = new THREE.Quaternion()
    const balls = (points: [number, number, number][], r: number, material: THREE.Material, detail = 10) => {
      const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, detail, Math.max(4, detail - 2)), material,
        Math.max(points.length, 1))
      mesh.count = points.length
      points.forEach((p, i) => { m.compose(new THREE.Vector3(...p), q, new THREE.Vector3(r, r, r)); mesh.setMatrixAt(i, m) })
      return mesh
    }
    const layer = plane ? ground.lattice.filter((p) => {
      const d = plane.distanceToPoint(new THREE.Vector3(...p))
      return d >= 0 && d < s
    }) : ground.lattice
    const grid = balls(layer, 0.09 * s, new THREE.MeshStandardMaterial({ color: GRID_COLOUR, roughness: 0.6 }), 8)
    grid.raycast = () => null
    const points = balls(ground.points, 0.18 * s,
      new THREE.MeshStandardMaterial({ roughness: 0.5, clippingPlanes: clip }))
    ground.kinds.forEach((k, i) => points.setColorAt(i, new THREE.Color(POINT_COLOUR[k] ?? GRID_COLOUR)))
    const where = ground.lonely.map((i) => ground.points[i])
    const lonely = balls(where, 0.4 * s, new THREE.MeshBasicMaterial({ color: LONELY_COLOUR, depthTest: false, transparent: true,
      opacity: 0.95 }), 16)
    const rings = balls(where, 0.8 * s, new THREE.MeshBasicMaterial({ color: LONELY_COLOUR, wireframe: true, depthTest: false,
      transparent: true, opacity: 0.6 }), 12)
    rings.raycast = () => null
    lonely.renderOrder = rings.renderOrder = 10
    const g = new THREE.Group()
    g.add(grid, points, lonely, rings)
    return { g, points, lonely, parts: [grid, points, lonely, rings] }
  }, [ground, clip])
  useEffect(() => () => {
    for (const o of group.parts) {
      o.geometry.dispose()
      ;(o.material as THREE.Material).dispose()
    }
  }, [group])
  return (
    <primitive object={group.g}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        if (e.instanceId === undefined || (e.object !== group.points && e.object !== group.lonely)) return
        e.stopPropagation()
        const i = e.object === group.lonely ? ground.lonely[e.instanceId] : e.instanceId
        onHover({ node: id, label: ground.labels[i], kind: ground.kinds[i], lonely: ground.lonely.includes(i),
          point: ground.points[i] })
      }}
      onPointerOut={() => onHover(null)} />
  )
}

function GroundPanel({ grounds, hover, sectioned }: {
  grounds: [string, GroundView][]; hover: GroundHover | null; sectioned: boolean
}) {
  return (
    <div className="tool-panel">
      {grounds.map(([id, g]) => {
        const kinds = new Set(g.kinds)
        return (
          <div key={id}>
            <div className="slice-row"><span className="slice-title">
              <b>{id}</b>: candidate nodes {g.spacing} mm apart
              {g.auto ? (g.coarser ? ` (automatic: ${g.coarser} mm left a support or load without bars)` : ' (automatic)') : ''}
              {' · '}{(g.lattice.length + g.points.length).toLocaleString()} nodes, {g.bars.toLocaleString()} possible bars
            </span></div>
            <div className="frame-sizes">
              <span className="frame-size" title="Points inside the design space where bars may meet: a wall without dots is thinner than the Node spacing">
                <span className="legend-swatch" style={{ background: GRID_COLOUR }} />grid</span>
              {(['fixed', 'load', 'attach'] as const).filter((k) => kinds.has(k)).map((k) => (
                <span key={k} className="frame-size"><span className="legend-swatch" style={{ background: POINT_COLOUR[k] }} />{POINT_KIND[k]}</span>
              ))}
              {g.lonely.length > 0 && (
                <span className="frame-size" title="Every bar from this point would leave the design space: Optimize stops here">
                  <span className="legend-swatch" style={{ background: LONELY_COLOUR, boxShadow: `0 0 0 2px ${LONELY_COLOUR}66`,
                    borderRadius: '50%' }} />no bar reaches it</span>
              )}
            </div>
            {g.error && <div className="warn small">{g.error}</div>}
          </div>
        )
      })}
      <div className="muted small">
        {hover
          ? <>{hover.lonely ? 'no bar reaches this point of ' : `a ${POINT_KIND[hover.kind]} point of `}<b>{hover.label}</b>
            {' '}({hover.point.map((v) => v.toFixed(0)).join(', ')} mm)</>
          : sectioned ? 'the layer of grid nodes just behind the cut · hover a coloured dot: its support or load'
          : 'hover a coloured dot: its support or load · Section shows one layer of the grid'}
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
  // how CAD bodies are drawn (CadBody.tsx): Edges toggles their edges, ▾ opens the options
  const cadLook = useCadLook()
  const [lookOpen, setLookOpen] = useState(false)
  const [bounds, setBoundsState] = useState<THREE.Box3 | null>(null)
  // only a real change re-renders (a new but equal box each time looped with the variants' boxes)
  const setBounds = useMemo(() => (b: THREE.Box3) => {
    const copy = b.clone()
    setBoundsState((prev) => (prev && prev.equals(copy) ? prev : copy))
  }, [])
  const [hover, setHover] = useState<HoverValue | null>(null)
  const [section, setSectionState] = useState<Section>(() => ({ ...read('section', { on: false, axis: 'z', flip: false, position: null }), on: false }))
  const setSection = (s: Section) => { setSectionState(s); write('section', { ...s, position: null }) }
  /** the print check is open (with this node chosen, when a body's menu opened it) */
  const [checking, setChecking] = useState<boolean | string>(false)
  // a wizard turns on the section plane (to look into a lattice) or the print check of a node
  const viewTool = useStore((s) => s.viewTool)
  const toolDone = useRef(viewTool?.at)
  useEffect(() => {
    if (!viewTool || viewTool.at === toolDone.current) return
    toolDone.current = viewTool.at
    if (viewTool.kind === 'section') setSectionState((s) => ({ ...s, on: true }))
    else setChecking(viewTool.node ?? true)
  }, [viewTool])
  const views = useMemo(() => visible.filter((id) => viewsMap[id]).map((id) => viewsMap[id]), [visible, viewsMap])
  const cadShown = visible.filter((id) => viewsMap[id]?.kind === 'cad')
  // finer parts of views (Refine, zoomed in): shown while their view shows the same, uncoloured and coarser
  const details = useStore((s) => s.details)
  const pickingNow = useStore((s) => !!s.picking)
  const shownDetails = useMemo(() => Object.fromEntries(visible.flatMap((id) => {
    const d = details[id], v = viewsMap[id]
    return d && v && !pickingNow && !paints[id] && d.key === v.key && d.voxel < (v.voxel ?? Infinity) ? [[id, d]] : []
  })), [visible, details, viewsMap, paints, pickingNow])
  const detailCuts = useMemo(() => Object.fromEntries(Object.entries(shownDetails).map(([id, d]) =>
    [id, { outside: { lo: d.lo, hi: d.hi, keep: 'outside' } as CutBox, inside: { lo: d.lo, hi: d.hi, keep: 'inside' } as CutBox }])),
  [shownDetails])
  // a list's items as a run makes them, each instead of that item in the view shown (or with none shown yet)
  const pieceMap = useStore((s) => s.pieces)
  const shownPieces = useMemo(() => visible.flatMap((id) => {
    const p = pieceMap[id]
    return p && p.base === (viewsMap[id]?.version ?? 0) && !paints[id]
      ? Object.entries(p.meshes).map(([k, mesh]) => ({ id, item: Number(k), mesh })) : []
  }), [visible, pieceMap, viewsMap, paints])
  const hiddenItems = useMemo(() => {
    const out: Record<string, number[]> = {}
    for (const { id, item } of shownPieces) if (viewsMap[id]) (out[id] ??= []).push(item)
    return out
  }, [shownPieces, viewsMap])
  const sliceBoxes = useMemo(() => Object.values(slices).map(sliceBox), [slices])
  const fieldRange = useMemo(() => fieldPaintRange(paints), [paints])
  const ownRanges = useMemo(() => Object.fromEntries(Object.entries(paints)
    .filter(([, p]) => p.source !== 'field').map(([id, p]) => [id, ownRange(p)])), [paints])
  const markers = useStore((s) => s.markers)
  const [showMarkers, setShowMarkersState] = useState(() => read('markers.shown', true))
  const setShowMarkers = (on: boolean) => { setShowMarkersState(on); write('markers.shown', on) }
  // load cases: their loads' arrows in the case colour; hovering a case in the legend fades the others'
  const casesJson = useStore((s) => casesKey(s.nodes, s.edges))
  const cases = useMemo(() => parseCases(casesJson), [casesJson])
  const caseColors = useMemo(() => Object.fromEntries(cases.list.flatMap((c) => c.loads.map((id) => [id, c.color]))), [cases])
  const [hoverCase, setHoverCase] = useState<string | null>(null)
  const fadedLoads = useMemo(() => new Set(hoverCase === null ? [] : cases.list.filter((c) => c.name !== hoverCase)
    .flatMap((c) => c.loads)), [cases, hoverCase])
  const loadsInEveryCase = Object.entries(markers).some(([id, list]) =>
    !caseColors[id] && list.some((m) => m.kind === 'load' && m.points.length))
  const anyMarkers = Object.values(markers).some((list) => list.some((m) => m.points.length))
  const results = visible.filter((id) => paints[id]?.source === 'result')
  const frameMap = useStore((s) => s.frames)
  const runningFrames = useStore((s) => s.liveFrames)
  const liveHistory = useStore((s) => s.liveHistory)
  const [replayState, setReplay] = useState<{ node: string; at: number; run: number } | null>(null)
  // (a replay ends when that node optimizes again)
  const replay = replayState && liveHistory[replayState.node]?.run === replayState.run && !runningFrames[replayState.node]
    ? replayState : null
  // a frame being optimized, or a step of the last optimization replayed, is drawn instead of its node's
  // frames, with the node's body see-through
  const liveFrames = useMemo(() => {
    const step = replay ? liveHistory[replay.node].steps[replay.at] : undefined
    return replay && step ? { ...runningFrames, [replay.node]: step } : runningFrames
  }, [runningFrames, liveHistory, replay])
  const frames = useMemo(() => Object.entries({ ...frameMap, ...Object.fromEntries(
    Object.entries(liveFrames).map(([id, f]) => [id, [f]])) })
    .flatMap(([id, list]) => list.map((f, k) => [id, f, k] as [string, FrameView, number])),
    [frameMap, liveFrames])
  const [showFrames, setShowFramesState] = useState(() => read('frames.shown', true))
  const setShowFrames = (on: boolean) => { setShowFramesState(on); write('frames.shown', on) }
  const [barHover, setBarHover] = useState<BarHover | null>(null)
  // a frame's candidate nodes (Show: candidate nodes), while its node is shown: its body see-through
  const groundMap = useStore((s) => s.grounds)
  const grounds = useMemo(() => Object.entries(groundMap).filter(([id]) => visible.includes(id)), [groundMap, visible])
  const [groundHover, setGroundHover] = useState<GroundHover | null>(null)
  /** measuring (null: not): the points clicked so far */
  const [measure, setMeasure] = useState<MeasurePoint[] | null>(null)
  const [showSettings, setShowSettingsState] = useState(() => read('settings.shown', false))
  const setShowSettings = (on: boolean) => { setShowSettingsState(on); write('settings.shown', on) }
  const hasSettings = useHasSettings()
  const wizardOpen = useWizard((w) => w.open)
  const wizardMade = useWizard((w) => (w.open ? w.made : null))
  const textureOpen = useTextureWizard((w) => w.open)
  const texturePicking = useTextureWizardPicking()
  const latticeOpen = useLatticeWizard((w) => w.open)
  const latticePicking = useLatticeWizardPicking()
  const faceSel = useStore((s) => s.faceSel)
  const prompt = useStore((s) => s.prompt)
  const setPrompt = useStore((s) => s.setPrompt)
  /** the pointer's last position over the view (Space opens the views menu there) */
  const mouse = useRef({ x: 0, y: 0 })

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
  // the wizard's panel says what to pick (the banner would sit under it)
  const wizardPicking = texturePicking || latticePicking || (!!picking && !!wizardMade && [wizardMade.fixed, ...wizardMade.loads].includes(picking.node))
  const faceData = useStore((s) => {
    const data = s.picking ? s.faceData[s.picking.source] : undefined
    return data && data.version === s.views[s.picking!.source]?.version ? data : null
  })
  const [faceError, setFaceError] = useState<string | null>(null)

  useEffect(() => { if (!pickNodeExists) setPicking(null) }, [pickNodeExists, setPicking])
  useEffect(() => {
    setFaceError(null)
    if (!picking || !pickView) return
    let alive = true
    loadFaces(picking.source).then((r) => { if (alive && typeof r === 'string') setFaceError(r) })
      .catch((err) => alive && setFaceError(String(err)))
    return () => { alive = false }
  }, [picking, pickView])
  const menuOpen = useRef(false)
  useEffect(() => {
    if (!picking) return
    // Esc closes an open right-click menu first
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !menuOpen.current) setPicking(null) }
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
  const onPick = (t: number, withGroup: boolean) => { if (faceData) onPickWith(faceData, t, withGroup) }
  const onPickWith = (data: FaceData, t: number, withGroup: boolean) => {
    const next = currentKeys()
    const members = faceGroup(data, t, withGroup)
    const remove = isChosen(data, next, t)
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

  // --- placing a guide shape (a Ball, Bar or Box node's Place button) ---
  const placing = useStore((s) => s.placing)
  const setPlacing = useStore((s) => s.setPlacing)
  const placeNodeExists = useStore((s) => !s.placing || s.nodes.some((n) => n.id === s.placing!.node))
  const placingBar = useStore((s) => !!s.placing && s.nodes.find((n) => n.id === s.placing!.node)?.data.typeId === 'implicit.bar')
  const guideIds = useStore((s) => s.nodes.filter((n) => GUIDE_TYPES.includes(n.data.typeId)).map((n) => n.id).join(' '))
  useEffect(() => { if (!placeNodeExists) setPlacing(null) }, [placeNodeExists, setPlacing])
  useEffect(() => {
    if (!placing) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPlacing(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [placing, setPlacing])

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

  // --- right-click menus ---
  const [menu, setMenu] = useState<{ x: number; y: number; entries: MenuEntry[]; highlight?: MenuFace } | null>(null)
  menuOpen.current = menu !== null
  const probe = useRef<Probe | null>(null)
  const leftDown = useRef<{ x: number; y: number; menu: boolean } | null>(null)
  const quickEdit = useStore((s) => s.quickEdit)
  const setQuickEdit = useStore((s) => s.setQuickEdit)
  const rightDown = useRef<{ x: number; y: number } | null>(null)
  const menuRequest = useRef(0)

  const toggle = (label: string, checked: boolean, run: () => void): MenuEntry => ({ kind: 'item', label, checked, run })
  const viewEntries = (): MenuEntry[] => [
    { kind: 'item', label: 'Fit view', hint: 'F', run: requestFit },
    { kind: 'item', label: 'Views', hint: 'Space', items: viewDirectionItems() },
    toggle('Wireframe', wireframe, () => setWireframe((w) => !w)),
    ...(cadShown.length ? [{ kind: 'item', label: 'Edges on CAD bodies', hint: 'E', checked: edgesShown(cadLook), run: toggleCadEdges } as MenuEntry] : []),
    { kind: 'item', label: 'Section', checked: section.on, items: [
      ...(['x', 'y', 'z'] as Axis[]).map((ax) => toggle(`Across ${ax.toUpperCase()}`, section.on && section.axis === ax,
        () => setSection({ on: true, axis: ax, flip: false, position: null }))),
      ...(section.on ? [toggle('Off', false, () => setSection({ ...section, on: false }))] : []),
    ] },
    ...(anyMarkers ? [toggle('Show loads', showMarkers, () => setShowMarkers(!showMarkers))] : []),
    ...(frames.length ? [toggle('Show bars', showFrames, () => setShowFrames(!showFrames))] : []),
    { kind: 'item', label: 'Measure', hint: 'M', checked: measure !== null, run: () => setMeasure((m) => (m ? null : [])) },
    toggle('Settings sliders', showSettings, () => setShowSettings(!showSettings)),
    { kind: 'separator' },
    { kind: 'item', label: 'Add node…', hint: 'Tab', run: () => useStore.getState().requestPalette(),
      disabled: picking ? 'Finish picking first' : undefined },
    ...[importItem()].filter((i): i is NonNullable<typeof i> => i !== null),
  ]

  /** The menu for what is under (x, y): a marker, a face and its body, a body, a slice, a bar, or the
   *  view itself. */
  const openMenu = async (x: number, y: number) => {
    const request = ++menuRequest.current
    const hit = probe.current?.(x, y) ?? null
    const st = useStore.getState()
    const ctx: MenuContext = { at: { x, y }, point: hit?.point, normal: hit?.normal ?? undefined, check: (node) => setChecking(node),
      hideBars: () => setShowFrames(false), sceneSize: Math.max(size.x, size.y, size.z, 1),
      section: (axis, position) => setSection({ on: true, axis, flip: false, position }) }
    const show = (entries: MenuEntry[], highlight?: MenuFace) =>
      setMenu({ x, y, entries: entries.length ? [...entries, { kind: 'separator' }, { kind: 'item', label: 'View', items: viewEntries() }]
        : viewEntries(), highlight })
    const tag = hit?.tag
    if (tag?.variant !== undefined) return show(variantMenu(tag.variant))
    // while a group is open its nodes are not the ones shown; while picking, only the picked body acts
    if (!hit || !tag || st.editing || (picking && !tag.picking)) return show([])
    if (tag.marker) return show(markerMenu(tag.marker, ctx))
    if (tag.bar && hit.instance !== null) return show(barMenu({ ...tag.bar, piece: hit.instance }, ctx))
    if (tag.slice) {
      const s = st.slices[tag.slice]
      const value = s && hit.uv ? s.values[Math.round(hit.uv[1] * (s.height - 1)) * s.width + Math.round(hit.uv[0] * (s.width - 1))] : NaN
      return show(sliceMenu({ node: tag.slice, value: Number.isFinite(value) ? value : null, point: hit.point }))
    }
    const view = tag.view ? st.views[tag.view] : undefined
    if (!tag.view || !view) return show([])
    let entries: MenuEntry[] = []
    let highlight: MenuFace | undefined
    let faceHit: FaceHit | null = null
    let behind: FaceHit | null = null  // the CAD face under an implicit body or mesh
    /** the face menu acts on all chosen faces when the clicked one is among them */
    const withChoice = (h: FaceHit): FaceHit => {
      const sel = useStore.getState().faceSel
      const base = { view: h.view, source: h.source, port: h.port ?? cadPort(h.source), same: h.same, bodyCount: h.bodyCount }
      const chosen = !!sel && sel.source === base.source && sel.port === base.port &&
        sel.faces.some((f) => f.face === h.face && f.body === h.body)
      ctx.select = (faces, mode) => choose(base, faces, mode)
      ctx.selected = chosen
      ctx.selectionSize = sel?.faces.length
      return chosen && sel!.faces.length > 1 ? { ...h, targets: sel!.faces } : h
    }
    if (view.kind === 'cad' && hit.triangle !== null) {
      const faces = await loadFaces(tag.view)
      if (request !== menuRequest.current) return  // another right-click came meanwhile
      const t = hit.triangle
      if (typeof faces === 'string') {
        entries = [{ kind: 'item', label: 'Face actions', disabled: faces }]
      } else if (faces.ids.length !== view.faces || t >= faces.ids.length) {
        entries = [{ kind: 'item', label: 'Face actions', disabled: 'The view is updating: try again in a moment' }]
      } else {
        const on: number[] = []
        for (let k = 0; k < faces.ids.length; k++) if (faces.ids[k] === faces.ids[t] && faces.bodies[k] === faces.bodies[t]) on.push(k)
        highlight = { view: tag.view, version: view.version, triangles: Uint32Array.from(on) }
        if (tag.picking && picking) {
          const keys = currentKeys()
          entries = pickingMenu({ node: picking.node, face: faceKey(faces, t), chosen: isChosen(faces, keys, t),
            count: keys.size, toggle: (group) => onPickWith(faces, t, group), clear: () => writeFaces(new Set()),
            done: () => setPicking(null) })
        } else {
          const info = await loadFaceInfo(tag.view, view.version, faces.bodies[t], faces.ids[t])
          if (request !== menuRequest.current) return
          faceHit = faceHitOf(tag.view, faces, t)
          ctx.tangent = faceGroup(faces, t, true).map((k) => {
            const [a, b] = k.split(':').map(Number)
            return b === undefined ? { face: a, body: faces.bodies[t] } : { face: b, body: a }
          })
          entries = faceMenu(withChoice(faceHit), info, ctx)
        }
      }
    } else if (view.kind !== 'cad') {
      // an implicit body or mesh: the faces of the CAD body it is built from, nearest to the click
      const at = await loadFaceAt(tag.view, hit.point)
      if (request !== menuRequest.current) return
      if (at) {
        behind = { view: tag.view, source: at.source, port: at.port, face: at.face, body: at.body, same: at.same,
          bodyCount: at.bodies, distance: at.distance }
        entries = faceMenu(withChoice(behind), at.info, ctx)
      }
    }
    if (!tag.picking) entries.push(...(entries.length ? [{ kind: 'separator' } as MenuEntry] : []), ...bodyMenu(tag.view, faceHit, ctx, behind))
    show(entries, highlight)
    if (behind) highlightFaceAt(request, tag.view, view, hit)
  }

  // --- chosen faces (Ctrl+click, Select similar): one right-click action on all of them ---
  /** the name of a node's CAD output */
  const cadPort = (node: string) => {
    const st = useStore.getState()
    return st.specs[st.nodes.find((n) => n.id === node)?.data.typeId ?? '']?.outputs.find((o) => o.type === 'cad')?.name ?? 'body'
  }
  const refKey = (f: FaceRef) => `${f.body}:${f.face}`
  /** the triangles of a shown CAD view on these faces (to highlight them) */
  const cadMarks = (view: string, faces: FaceRef[]) => {
    const st = useStore.getState()
    const v = st.views[view], data = st.faceData[view]
    const out: FaceSelection['marks'] = {}
    if (!v || !data || data.version !== v.version || data.ids.length !== v.faces) return out
    const wanted = new Map(faces.map((f) => [refKey(f), [] as number[]]))
    for (let t = 0; t < data.ids.length; t++) wanted.get(`${data.bodies[t]}:${data.ids[t]}`)?.push(t)
    for (const [k, list] of wanted) if (list.length) out[k] = { view, version: v.version, triangles: Uint32Array.from(list) }
    return out
  }
  /** Adds faces to the choice, makes them the choice, or drops them. A face of another CAD output than
   *  the chosen ones starts a new choice. `marks`: their triangles on an implicit body's view. */
  const choose = (base: Omit<FaceSelection, 'faces' | 'marks'>, faces: FaceRef[], mode: 'add' | 'replace' | 'remove',
    marks: FaceSelection['marks'] = {}) => {
    const st = useStore.getState()
    const prev = st.faceSel && st.faceSel.source === base.source && st.faceSel.port === base.port && mode !== 'replace' ? st.faceSel : null
    const before = prev?.faces ?? []
    const drop = new Set(mode === 'remove' ? faces.map(refKey) : [])
    const list = mode === 'remove' ? before.filter((f) => !drop.has(refKey(f)))
      : [...before, ...faces.filter((f) => !before.some((b) => refKey(b) === refKey(f)))]
    if (!list.length) { st.setFaceSel(null); return }
    const all = { ...(prev?.marks ?? {}), ...(base.view === base.source ? cadMarks(base.view, faces) : {}), ...marks }
    for (const k of drop) delete all[k]
    st.setFaceSel({ ...base, view: prev?.view ?? base.view, faces: list, marks: all })
  }
  /** Ctrl+click: the face under (x, y) in or out of the choice. */
  const toggleFaceAt = async (x: number, y: number) => {
    const request = ++menuRequest.current
    const hit = probe.current?.(x, y) ?? null
    const st = useStore.getState()
    const id = hit?.tag.view
    const view = id ? st.views[id] : undefined
    if (!hit || !id || !view || st.editing) return
    let base: Omit<FaceSelection, 'faces' | 'marks'>
    let ref: FaceRef
    let marks: FaceSelection['marks'] = {}
    if (view.kind === 'cad' && hit.triangle !== null) {
      const faces = await loadFaces(id)
      if (request !== menuRequest.current) return
      if (typeof faces === 'string' || faces.ids.length !== view.faces) { st.showToast(typeof faces === 'string' ? faces : 'The view is updating'); return }
      const h = faceHitOf(id, faces, hit.triangle)
      base = { view: id, source: id, port: cadPort(id), same: h.same, bodyCount: h.bodyCount }
      ref = { face: h.face, body: h.body }
    } else if (view.kind !== 'cad') {
      const at = await loadFaceAt(id, hit.point)
      if (request !== menuRequest.current) return
      if (!at) { st.showToast(`No CAD face behind ${id}`); return }
      base = { view: id, source: at.source, port: at.port, same: at.same, bodyCount: at.bodies }
      ref = { face: at.face, body: at.body }
      const edge = hit.triangle !== null && hit.triangle < view.faces ? longestEdge(view, hit.triangle) : 0
      const region = await loadFaceRegion(id, hit.point, edge)
      if (region && region.version === view.version) marks = { [refKey(ref)]: { view: id, version: region.version, triangles: region.triangles } }
    } else return
    const sel = useStore.getState().faceSel
    const chosen = !!sel && sel.source === base.source && sel.port === base.port && sel.faces.some((f) => refKey(f) === refKey(ref))
    choose(base, [ref], chosen ? 'remove' : 'add', marks)
  }
  useEffect(() => {
    if (!faceSel) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !menuOpen.current && !useStore.getState().picking) useStore.getState().setFaceSel(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [faceSel])

  // --- measuring: click points on the bodies ---
  const measureIds = useRef(0)
  const measureAt = async (x: number, y: number) => {
    const hit = probe.current?.(x, y, true) ?? null
    if (!hit) return
    const id = ++measureIds.current
    const p: MeasurePoint = { id, point: hit.point, normal: hit.normal, view: hit.tag.view ?? null }
    setMeasure((m) => (m && m.length < 2 ? [...m, p] : [p]))
    const patch = (more: Partial<MeasurePoint>) => setMeasure((m) => m?.map((q) => (q.id === id ? { ...q, ...more } : q)) ?? m)
    const st = useStore.getState()
    const view = p.view ? st.views[p.view] : undefined
    if (!p.view || !view) return
    if (p.normal) {
      loadThickness(p.view, p.point, p.normal).then((thickness) => patch({ thickness }))
      loadSection(p.view, p.point, p.normal).then((section) => patch({ section }))
    }
    if (view.kind === 'cad' && hit.triangle !== null) {
      const faces = await loadFaces(p.view)
      if (typeof faces === 'string' || hit.triangle >= faces.ids.length) return
      const info = await loadFaceInfo(p.view, view.version, faces.bodies[hit.triangle], faces.ids[hit.triangle])
      patch({ face: info ? `face ${faces.ids[hit.triangle]}: ${faceText(info)}` : null })
    } else {
      const at = await loadFaceAt(p.view, p.point)
      patch({ face: at?.info ? `CAD face ${at.face} of ${at.source}: ${faceText(at.info)}` : null })
    }
  }
  useEffect(() => {
    if (measure === null) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !menuOpen.current) setMeasure(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [measure === null]) // eslint-disable-line react-hooks/exhaustive-deps

  /** A double-click: the edit box of the node that made what is under (x, y). */
  const editAt = async (x: number, y: number) => {
    const request = ++menuRequest.current
    const hit = probe.current?.(x, y, true) ?? null
    const st = useStore.getState()
    const tag = hit?.tag
    if (!hit || !tag || st.editing) return
    const node = tag.marker?.node ?? tag.bar?.node ?? tag.slice
    if (node) return st.setQuickEdit({ node, x, y })
    const view = tag.view ? st.views[tag.view] : undefined
    if (!tag.view || !view) return
    const tol = hit.triangle !== null && hit.triangle < view.faces ? Math.max(longestEdge(view, hit.triangle) / 4, 0.02) : 0.05
    const at = await loadNodeAt(tag.view, hit.point, tol)
    if (request !== menuRequest.current) return
    st.setQuickEdit({ node: at?.node ?? tag.view, x, y })
  }

  /** Highlights, once the engine has found them, the triangles of an implicit body or mesh (and of
   *  the detail shown in its box) on the CAD face under the click, while that click's menu is open. */
  const highlightFaceAt = async (request: number, node: string, view: ViewMesh, hit: MenuHit) => {
    const detail = shownDetails[node]
    const edge = hit.triangle !== null && hit.triangle < view.faces ? longestEdge(view, hit.triangle) : 0
    const [main, fine] = await Promise.all([loadFaceRegion(node, hit.point, edge),
      detail ? loadFaceRegion(node, hit.point, edge, true) : null])
    if (request !== menuRequest.current || !main || main.version !== view.version) return
    const highlight: MenuFace = { view: node, ...main,
      detail: fine && detail && fine.version === detail.version ? fine : undefined }
    setMenu((m) => (m ? { ...m, highlight } : m))
  }

  /** A left click: selects (and flashes, and pans the graph to) the node that made what is under the
   *  cursor: for a body, the node that made its surface at that spot (an Offset Body's bump, a Hole
   *  Pattern's hole, the CAD body's face); a support or load, a slice, a frame's bars: their node. */
  const selectAt = async (x: number, y: number) => {
    const request = ++menuRequest.current
    const hit = probe.current?.(x, y, true) ?? null
    const st = useStore.getState()
    const tag = hit?.tag
    if (!hit || !tag || st.editing) return
    const title = (id: string) => st.specs[st.nodes.find((n) => n.id === id)?.data.typeId ?? '']?.title ?? ''
    const select = (node: string, detail: string[] = [], under = '') => {
      st.focusNode(node)
      st.showToast(`${node} · ${title(node)}${under}`, detail,
        [{ label: 'Edit', run: () => useStore.getState().setQuickEdit({ node, x, y }) }])
    }
    if (tag.marker) return select(tag.marker.node)
    if (tag.bar) return select(tag.bar.node)
    if (tag.slice) return select(tag.slice)
    const view = tag.view ? st.views[tag.view] : undefined
    if (!tag.view || !view) return
    // A marching-cubes surface lies much closer to the true one than its edges are long (a Draft edge
    // is as long as a 1 mm bump is high): a quarter of it, and the engine adds how far the shown
    // body's own surface is from the point (larger at the corners the mesh cuts).
    const tol = hit.triangle !== null && hit.triangle < view.faces ? Math.max(longestEdge(view, hit.triangle) / 4, 0.02) : 0.05
    const at = await loadNodeAt(tag.view, hit.point, tol)
    if (request !== menuRequest.current) return
    if (!at) return select(tag.view)
    select(at.node, at.path.map((p) => `${p.node}${p.value === null ? '' : `: ${p.value.toFixed(2)} mm here`}`),
      at.node !== tag.view ? `: made this surface of ${tag.view}` : '')
  }

  return (
    <div className="viewport"
      onKeyDown={(e) => {
        if (e.ctrlKey || e.metaKey || e.altKey || (e.target as HTMLElement).tagName === 'INPUT') return
        const view = STANDARD_VIEWS.find((v) => v.key === e.key)
        if (e.key === 'f' || e.key === 'F') requestFit()
        else if (e.key === 'm' || e.key === 'M') setMeasure((m) => (m ? null : []))
        else if (e.key === 'e' || e.key === 'E') toggleCadEdges()
        else if (view) useStore.getState().requestCamera({ dir: view.dir })
        else if (e.key === ' ') {
          e.preventDefault()
          setMenu({ ...mouse.current, entries: [{ kind: 'header', text: 'Views' }, ...viewDirectionItems()] })
        }
      }} tabIndex={0}
      onPointerMove={(e) => { mouse.current = { x: e.clientX, y: e.clientY } }}
      onPointerDown={(e) => {
        if (e.button === 2) rightDown.current = { x: e.clientX, y: e.clientY }
        // a click that closes the menu only closes it
        if (e.button === 0) leftDown.current = { x: e.clientX, y: e.clientY, menu: menuOpen.current }
      }}
      onClick={(e) => {
        const down = leftDown.current
        leftDown.current = null
        const ctrl = (e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey
        if ((e.target as HTMLElement).tagName !== 'CANVAS' || e.button !== 0 || ((e.ctrlKey || e.metaKey) && !ctrl) || e.shiftKey || e.altKey) return
        // a drag turned the view; picking faces and placing a guide use clicks themselves
        if (!down || down.menu || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4 || picking || placing) return
        // the axis gizmo in the corner (its own scene) turns the view
        const r = (e.target as HTMLElement).getBoundingClientRect()
        if (Math.hypot(e.clientX - (r.right - GIZMO_MARGIN), e.clientY - (r.bottom - GIZMO_MARGIN)) < GIZMO_MARGIN) return
        if (measure !== null) measureAt(e.clientX, e.clientY)
        else if (ctrl) toggleFaceAt(e.clientX, e.clientY)
        else selectAt(e.clientX, e.clientY)
      }}
      onDoubleClick={(e) => {
        if ((e.target as HTMLElement).tagName !== 'CANVAS' || e.button !== 0 || e.ctrlKey || e.metaKey || picking || placing || measure !== null) return
        editAt(e.clientX, e.clientY)
      }}
      onContextMenu={(e) => {
        if ((e.target as HTMLElement).tagName !== 'CANVAS') return
        e.preventDefault()
        // right-drag pans the view (and Windows fires contextmenu on release): only a still click opens the menu
        const down = rightDown.current
        rightDown.current = null
        if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) <= 5) openMenu(e.clientX, e.clientY)
      }}>
      <Canvas camera={{ position: [150, -180, 130], fov: 35, near: 0.1, far: 20000 }} dpr={[1, 2]}
        gl={{ antialias: true }}>
        <ClippingOn />
        <color attach="background" args={['#24262b']} />
        <hemisphereLight args={['#f4f6fa', '#3a3d44', 1.2]} />
        <directionalLight position={[2, -3, 4]} intensity={1.8} />
        <directionalLight position={[-3, 2, -1]} intensity={0.6} />
        {visible.map((id) => viewsMap[id] && (picking?.source === id
          ? <PickableView key={id} node={id} view={viewsMap[id]} faces={faceData} chosen={chosen} onPick={onPick} onPaint={onPaint}
              wireframe={wireframe} clip={clip} chosenColor={leaveOut ? PICK_LEFT_OUT : PICK_SELECTED} />
          : viewsMap[id].kind === 'cad' && !paints[id]
            // CAD bodies: crisp face borders and edges, in the toolbar's Edges ▾ style (CadBody.tsx)
            ? <CadBody key={`cad-${id}`} view={viewsMap[id]} node={id} wireframe={wireframe}
                ghost={!!picking || !!liveFrames[id] || !!groundMap[id]} clip={clip} box={detailCuts[id]?.outside} hide={hiddenItems[id]} />
            : <ViewObject key={id} view={viewsMap[id]} wireframe={wireframe} ghost={!!picking || !!liveFrames[id] || !!groundMap[id]} clip={clip}
                paint={paints[id]} cmap={cmap} range={paints[id] && paints[id].source !== 'field' ? ownRanges[id] : fieldRange}
                onHover={setHover} node={id} box={detailCuts[id]?.outside} hide={hiddenItems[id]} />))}
        {shownPieces.map(({ id, item, mesh }) => picking?.source !== id && ((viewsMap[id]?.kind ?? mesh.kind) === 'cad'
          ? <CadBody key={`piece-${id}-${item}-${mesh.version}`} view={mesh} wireframe={wireframe}
              ghost={!!picking || !!liveFrames[id] || !!groundMap[id]} clip={clip} box={detailCuts[id]?.outside} patch />
          : <ViewObject key={`piece-${id}-${item}-${mesh.version}`} view={mesh} kind={viewsMap[id]?.kind ?? mesh.kind}
              wireframe={wireframe} ghost={!!picking || !!liveFrames[id] || !!groundMap[id]} clip={clip} cmap={cmap}
              range={null} box={detailCuts[id]?.outside} patch />
        ))}
        {Object.entries(shownDetails).map(([id, d]) => (viewsMap[id].kind === 'cad'
          ? <CadBody key={`detail-${id}`} view={d.mesh} wireframe={wireframe} clip={clip} box={detailCuts[id].inside} patch />
          : <ViewObject key={`detail-${id}`} view={d.mesh} kind={viewsMap[id].kind} wireframe={wireframe} clip={clip}
              cmap={cmap} range={null} box={detailCuts[id].inside} patch />
        ))}
        <DetailProbe />
        {showMarkers && anyMarkers && <MarkerArrows markers={markers} size={Math.max(size.x, size.y, size.z, 1)}
          colors={caseColors} faded={fadedLoads} />}
        {showFrames && !picking && frames.map(([id, f, k]) => <FrameBars key={`${id}-${k}`} id={id} index={k} frame={f} onHover={setBarHover} />)}
        {!picking && grounds.map(([id, g]) => <GroundNodes key={id} id={id} ground={g} clip={clip} onHover={setGroundHover} />)}
        {!picking && <Guides sceneSize={Math.max(size.x, size.y, size.z, 1)} />}
        <BodyHint />
        {aim && !picking && (
          <DirectionGizmo anchor={aim.anchor} dir={aim.dir} length={Math.max(size.x, size.y, size.z, 1) * 0.3}
            onCommit={commitAim} torque={aim.torque} centre={aim.centre}
            step={offsetStep(Math.max(size.x, size.y, size.z, 1))} onMove={aim.movable ? commitOffset : undefined} />
        )}
        {shownVariants.map((v, i) => (
          <group key={v.id} position={[-(i + 1) * variantStep, 0, 0]}>
            {v.views.map((view, k) => <ViewObject key={k} view={view} wireframe={wireframe} clip={[]} cmap={cmap}
              range={null} kind="variant" variant={v.id} />)}
          </group>
        ))}
        {faceSel && !picking && Object.entries(faceSel.marks).map(([k, m]) => viewsMap[m.view]?.version === m.version && (
          <MenuHighlight key={k} view={viewsMap[m.view]} triangles={m.triangles} clip={clip} box={detailCuts[m.view]?.outside}
            color={PICK_SELECTED} />
        ))}
        {measure && <MeasureMarks points={measure} size={Math.max(size.x, size.y, size.z, 1)} />}
        {section.on && bounds && (
          <SectionGizmo axis={section.axis} position={cutAt} box={bounds} onMove={(p) => setSection({ ...section, position: p })} />
        )}
        <CameraTurn />
        {Object.entries(slices).map(([id, s]) => <SliceObject key={id} node={id} slice={s} cmap={cmap} onHover={setHover} />)}
        <MenuProbe probe={probe} clip={clip} />
        {menu?.highlight && viewsMap[menu.highlight.view]?.version === menu.highlight.version && (
          <MenuHighlight view={viewsMap[menu.highlight.view]} triangles={menu.highlight.triangles} clip={clip}
            box={detailCuts[menu.highlight.view]?.outside} />
        )}
        {menu?.highlight?.detail && shownDetails[menu.highlight.view]?.version === menu.highlight.detail.version && (
          <MenuHighlight view={shownDetails[menu.highlight.view].mesh} triangles={menu.highlight.detail.triangles} clip={clip}
            box={detailCuts[menu.highlight.view].inside} />
        )}
        <Grid position={[center.x, center.y, floorZ]} rotation={[Math.PI / 2, 0, 0]}
          cellSize={cell / 5} sectionSize={cell} cellColor="#34373d" sectionColor="#454951"
          cellThickness={0.6} sectionThickness={1} fadeDistance={span * 6} fadeStrength={1.5} infiniteGrid />
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
        <CameraFit views={views} extra={extraBoxes} onBounds={setBounds} />
        <GizmoHelper alignment="bottom-right" margin={[GIZMO_MARGIN, GIZMO_MARGIN]}>
          <GizmoViewport axisColors={['#e5534b', '#57ab5a', '#539bf5']} labelColor="#111" />
        </GizmoHelper>
      </Canvas>

      <div className="viewport-overlay">
        {visible.length === 0 && <div className="viewport-hint">Click ○ on a node to show it here</div>}
        {visible.map((id) => (
          <div className="legend-row" key={id}>
            <span className="legend-swatch" style={{
              background: paints[id] ? gradientCss(paintMap(paints[id], cmap)) : slices[id] ? gradientCss(cmap)
                : viewsMap[id]?.kind === 'cad' ? cadLook.color : KIND_STYLE[viewsMap[id]?.kind ?? 'mesh']?.color }} />
            <span className="legend-name">{id}</span>
            <span className="legend-info" title={voxelNote(viewsMap[id])}>
              {status[id]?.state === 'running' ? 'updating…'
                : viewsMap[id] ? (viewsMap[id].level && !viewsMap[id].exact ? `${QUALITY_LABEL[viewsMap[id].level!]} · ` : '') +
                  (viewsMap[id].voxel && !viewsMap[id].exact ? `voxel ${fmtVoxel(viewsMap[id].voxel!)} · ` : '') +
                  `${(viewsMap[id].faces / 1000).toFixed(viewsMap[id].faces < 10000 ? 1 : 0)}k tris` +
                  (shownDetails[id] ? ` · on screen ${fmtVoxel(shownDetails[id].voxel)}` : '') +
                  (paints[id] ? ` · coloured by ${paints[id].label}${paints[id].unit ? ` (${paints[id].unit})` : ''}` : '')
                : slices[id] ? `field slice, ${slices[id].axis} = ${slices[id].position.toFixed(1)}`
                : status[id]?.state === 'error' ? 'error'
                : guideIds.split(' ').includes(id) ? 'shape (select its node to drag it)' : '…'}
            </span>
          </div>
        ))}
        <VariantsList variants={variants} />
      </div>
      <div className="viewport-bottom">
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
        <ViewportToast />
      </div>
      {picking && !wizardPicking && (
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
      {faceSel && !picking && !placing && (
        <div className="pick-banner">
          <span><b>{faceSel.faces.length}</b> {faceSel.faces.length === 1 ? 'face' : 'faces'} chosen on <b>{faceSel.source}</b></span>
          <span className="muted">right-click one of them for actions on all · Ctrl+click adds or removes a face · Esc clears</span>
          <button onClick={() => useStore.getState().setFaceSel(null)} title="Esc">Clear</button>
        </div>
      )}
      {placing && (
        <div className="pick-banner">
          <span>Placing <b>{placing.node}</b>{placingBar ? (placing.port === 'start' ? ': its start' : ': its end') : ''}</span>
          <span className="muted">
            {views.length
              ? 'click on a shown body: the point goes in the middle of the wall under the click · drag to turn the view'
              : 'show a body to click on (its ○ eye on a node)'}
          </span>
          <button onClick={() => setPlacing(null)} title="Esc">Done</button>
        </div>
      )}
      <div className="viewport-side">
        <div className="viewport-tools">
          <button onClick={requestFit} title="Fit view (F)">Fit</button>
          <button className={wireframe ? 'on' : ''} onClick={() => setWireframe((w) => !w)} title="Wireframe">Wire</button>
          {cadShown.length > 0 && (
            <span className="split-btn">
              <button className={edgesShown(cadLook) ? 'on' : ''} onClick={toggleCadEdges}
                title="Edges of the CAD faces on CAD bodies (E)">Edges</button>
              <button className={lookOpen ? 'on' : ''} onClick={() => setLookOpen((o) => !o)}
                title="How CAD bodies are drawn: shaded, see-through, lines or wire; tangent edges; colours">▾</button>
            </span>
          )}
          <button className={section.on ? 'on' : ''} onClick={() => setSection({ ...section, on: !section.on })}
            title="Cut the view open with a plane to look inside (drag the orange ball to move the cut)">Section</button>
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
          <button className={measure ? 'on' : ''} onClick={() => setMeasure((m) => (m ? null : []))}
            title="Measure (M): click two points for their distance; each point shows the CAD face, the wall thickness and the member's cross-section under it">Measure</button>
          {(hasSettings || showSettings) && (
            <button className={showSettings ? 'on' : ''} onClick={() => setShowSettings(!showSettings)}
              title="The recipe's Number nodes (and group settings) as sliders">Sliders</button>
          )}
          <button className={textureOpen ? 'on' : ''}
            onClick={() => (textureOpen ? closeTextureWizard() : openTextureWizard())}
            title={'Texture wizard: click the faces, choose the texture and its settings (shown on the part), where it fades, ' +
              'then smoothing or a shell'}>Texture</button>
          <button className={latticeOpen ? 'on' : ''}
            onClick={() => (latticeOpen ? closeLatticeWizard() : openLatticeWizard())}
            title={'Lattice wizard: fill a part with a lattice (seen in a section, with what it weighs), keep it solid where it ' +
              'is bolted, grade it, check and export'}>Lattice</button>
          <button className={wizardOpen ? 'on' : ''} onClick={() => {
            if (!wizardOpen) closeOtherWizards('analysis')
            useWizard.setState({ open: !wizardOpen })
          }}
            title={'Wizard for topology optimization, frame optimization or static analysis: pick the part, click the held and ' +
              'the loaded faces, set a few numbers, run'}>Analyse</button>
        </div>
        <div className="tool-panels">
          <TextureWizard />
          <LatticeWizard />
          <AnalysisWizard />
          {!picking && results.map((id) => (
            <ResultPanel key={id} node={id} paint={paints[id]} hover={hover?.node === id ? hover : null} />
          ))}
          {showFrames && frames.length > 0 && !picking && (
            <FramePanel frames={frames} hover={barHover} history={liveHistory} replay={replay}
              onReplay={(node, at) => setReplay(at === null ? null : { node, at, run: liveHistory[node].run })} />
          )}
          {grounds.length > 0 && !picking && <GroundPanel grounds={grounds} hover={groundHover} sectioned={section.on} />}
          {showMarkers && anyMarkers && cases.list.length > 0 && !picking && (
            <CasePanel cases={cases.list} others={loadsInEveryCase} onHover={setHoverCase} />
          )}
          {lookOpen && cadShown.length > 0 && <CadLookPanel shown={cadShown} onClose={() => setLookOpen(false)} />}
          {section.on && <SectionPanel section={section} setSection={setSection} lo={lo} hi={hi} />}
          {measure && <MeasurePanel points={measure} onClose={() => setMeasure(null)} />}
          {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}
          {checking && <CheckPanel candidates={checkCandidates} initial={typeof checking === 'string' ? checking : undefined}
            onClose={() => { clearPaints('thickness'); setChecking(false) }} />}
        </div>
      </div>
      <SlicePanel hover={hover} />
      {menu && <ViewportMenu x={menu.x} y={menu.y} entries={menu.entries} onClose={() => setMenu(null)} />}
      {quickEdit && <QuickEdit key={`${quickEdit.node}@${quickEdit.x},${quickEdit.y}`} state={quickEdit}
        onClose={() => setQuickEdit(null)} />}
      {prompt && <NamePrompt prompt={prompt} onClose={() => setPrompt(null)} />}
    </div>
  )
}
