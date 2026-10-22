import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { useThree, type ThreeEvent } from '@react-three/fiber'
import { Html, RoundedBox } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useShallow } from 'zustand/react/shallow'
import { useStore } from '../store'
import { isLink, type EngineNode, type NodeSpec } from '../types'

/** Simple shapes the 3D view draws itself (the engine does not mesh them for the view: server.py
 *  DRAWN_BY_VIEWPORT), so they follow a drag at once. Used as Topology Optimization's guides. */
export const GUIDE_TYPES = ['implicit.ball', 'implicit.bar', 'implicit.box']
/** the point inputs Place sets, one click each */
export const PLACE_PORTS: Record<string, string[]> = {
  'implicit.ball': ['centre'], 'implicit.bar': ['start', 'end'], 'implicit.box': ['centre'],
}
const GUIDE_COLOR = '#ff8c1a'
const HANDLE_COLOR = '#ffc93c'
const HANDLE_HOVER = '#ffe9a8'

const fmtComponent = (x: number) => String(Number(x.toFixed(3)))
export const vectorText = (v: THREE.Vector3) => `${fmtComponent(v.x)}, ${fmtComponent(v.y)}, ${fmtComponent(v.z)}`
export const parseVec = (value: unknown) => {
  const v = String(value ?? '').split(/[\s,;]+/).filter(Boolean).map(Number)
  return v.length === 3 && v.every(Number.isFinite) ? new THREE.Vector3(...v) : null
}
/** Dragged points and offsets move in whole steps: 0.5 mm in small scenes, 1 mm up to 300 mm, else 5 mm. */
export const offsetStep = (sceneSize: number) => (sceneSize <= 50 ? 0.5 : sceneSize <= 300 ? 1 : 5)

interface Shape {
  id: string
  type: string
  selected: boolean
  points: Record<string, THREE.Vector3>   // centre, or start and end
  radius: number
  size: THREE.Vector3
  rounding: number
}

/** A guide node's shape from its inputs (defaults from its spec), or null when an input is linked or
 *  not a valid value (the engine reports that on the card). */
function shapeOf(node: EngineNode, spec: NodeSpec | undefined, selected: boolean): Shape | null {
  const value = (name: string) => node.data.inputs[name] ?? spec?.inputs.find((p) => p.name === name)?.default
  const points: Record<string, THREE.Vector3> = {}
  for (const port of PLACE_PORTS[node.data.typeId]) {
    const v = value(port)
    const p = isLink(v) ? null : parseVec(v)
    if (!p) return null
    points[port] = p
  }
  const number = (name: string) => { const v = value(name); return isLink(v) ? NaN : Number(v) }
  const size = node.data.typeId === 'implicit.box' ? parseVec(value('size')) : new THREE.Vector3()
  const shape = { id: node.id, type: node.data.typeId, selected, points, radius: number('radius'),
    size: size ?? new THREE.Vector3(), rounding: Math.max(number('rounding') || 0, 0) }
  if (shape.type !== 'implicit.box' && !(shape.radius > 0)) return null
  if (shape.type === 'implicit.box' && !(size && size.x > 0 && size.y > 0 && size.z > 0)) return null
  return shape
}

const Y_UP = new THREE.Vector3(0, 1, 0)

/** A guide shape, drawn twice: as it is where nothing is in front, and faintly through the bodies
 *  (guides sit inside the part, where it would hide them). */
function ShapeMesh({ shape, onSelect }: { shape: Shape; onSelect?: () => void }) {
  const click = onSelect ? (e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); onSelect() } : undefined
  const pass = (xray: boolean) => {
    const material = (
      <meshStandardMaterial color={GUIDE_COLOR} transparent depthWrite={false} depthTest={!xray}
        opacity={xray ? (shape.selected ? 0.3 : 0.18) : (shape.selected ? 0.5 : 0.32)}
        roughness={0.6} metalness={0} side={THREE.DoubleSide} />
    )
    const order = xray ? 6 : 5
    const events = xray ? { raycast: () => null } : { onClick: click }
    if (shape.type === 'implicit.ball') {
      return (
        <mesh position={shape.points.centre} renderOrder={order} {...events}>
          <sphereGeometry args={[shape.radius, 40, 20]} />
          {material}
        </mesh>
      )
    }
    if (shape.type === 'implicit.bar') {
      const { start, end } = shape.points
      const d = end.clone().sub(start)
      const length = d.length()
      const quat = length > 1e-9 ? new THREE.Quaternion().setFromUnitVectors(Y_UP, d.clone().normalize()) : new THREE.Quaternion()
      return (
        <mesh position={start.clone().add(end).multiplyScalar(0.5)} quaternion={quat} renderOrder={order} {...events}>
          <capsuleGeometry args={[shape.radius, length, 8, 24]} />
          {material}
        </mesh>
      )
    }
    return (
      <RoundedBox args={[shape.size.x, shape.size.y, shape.size.z]}
        radius={Math.min(shape.rounding, shape.size.x / 2, shape.size.y / 2, shape.size.z / 2) || 0.0001}
        smoothness={4} position={shape.points.centre} renderOrder={order} {...events}>
        {material}
      </RoundedBox>
    )
  }
  return <group>{pass(false)}{pass(true)}</group>
}

/** The yellow balls of the selected shape: drag one to move that point (a Bar's middle one moves the
 *  whole bar) in the plane facing the camera, in whole steps; Esc cancels. */
function Handles({ shape, size, step, onDrag, onCommit }: {
  shape: Shape; size: number; step: number
  onDrag: (id: string, points: Record<string, THREE.Vector3> | null) => void
  onCommit: (id: string, points: Record<string, THREE.Vector3>) => void
}) {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const [hover, setHover] = useState<string | null>(null)
  const [dragging, setDragging] = useState<{ handle: string; at: THREE.Vector3 } | null>(null)
  const grab = useRef<{ plane: THREE.Plane; from: THREE.Vector3; points: Record<string, THREE.Vector3>; handle: string;
    result: Record<string, THREE.Vector3> | null } | null>(null)
  const handles = useMemo(() => {
    const p = shape.points
    const list: [string, THREE.Vector3][] = Object.entries(p)
    if (shape.type === 'implicit.bar') list.push(['middle', p.start.clone().add(p.end).multiplyScalar(0.5)])
    return list
  }, [shape])

  useEffect(() => {
    if (!dragging) return
    const raycaster = new THREE.Raycaster()
    const move = (ev: PointerEvent) => {
      const g = grab.current
      if (!g) return
      const r = gl.domElement.getBoundingClientRect()
      raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1,
        -((ev.clientY - r.top) / r.height) * 2 + 1), camera)
      const hit = raycaster.ray.intersectPlane(g.plane, new THREE.Vector3())
      if (!hit) return
      const shift = hit.sub(g.from)
      const snap = (v: THREE.Vector3) => v.set(Math.round(v.x / step) * step, Math.round(v.y / step) * step,
        Math.round(v.z / step) * step)
      const moved = Object.fromEntries(Object.entries(g.points).map(([k, v]) =>
        [k, g.handle === 'middle' || g.handle === k ? v.clone().add(shift) : v.clone()]))
      if (g.handle === 'middle') {   // keep the bar's length and direction: snap its start, move the end along
        const before = moved.start.clone()
        snap(moved.start)
        moved.end.add(moved.start.clone().sub(before))
      } else snap(moved[g.handle])
      g.result = moved
      onDrag(shape.id, moved)
      setDragging({ handle: g.handle, at: g.handle === 'middle' ? moved.start.clone().add(moved.end).multiplyScalar(0.5)
        : moved[g.handle].clone() })
    }
    const finish = (commit: boolean) => {
      if (controls) controls.enabled = true
      if (commit && grab.current?.result) onCommit(shape.id, grab.current.result)
      grab.current = null
      onDrag(shape.id, null)
      setDragging(null)
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
  }, [dragging !== null, shape.id, step, camera, gl, controls, onDrag, onCommit]) // eslint-disable-line react-hooks/exhaustive-deps

  const r = size * 0.018
  return (
    <group>
      {handles.map(([name, at]) => (
        <group key={name}>
          <mesh position={at} renderOrder={21} raycast={() => null}>
            <sphereGeometry args={[name === 'middle' ? r * 0.8 : r, 16, 12]} />
            <meshBasicMaterial color={hover === name || dragging?.handle === name ? HANDLE_HOVER : HANDLE_COLOR}
              depthTest={false} transparent opacity={0.95} />
          </mesh>
          {/* a bigger, invisible target to grab */}
          <mesh position={at} renderOrder={22}
            onPointerOver={(e) => { e.stopPropagation(); setHover(name); gl.domElement.style.cursor = 'grab' }}
            onPointerOut={() => { setHover(null); gl.domElement.style.cursor = '' }}
            onPointerDown={(e) => {
              if (e.button !== 0) return
              e.stopPropagation()
              if (controls) controls.enabled = false
              const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), at)
              grab.current = { plane, from: e.ray.intersectPlane(plane, new THREE.Vector3()) ?? at.clone(),
                points: shape.points, handle: name, result: null }
              setDragging({ handle: name, at: at.clone() })
            }}>
            <sphereGeometry args={[r * 2.4, 16, 12]} />
            <meshBasicMaterial transparent opacity={0} depthTest={false} depthWrite={false} />
          </mesh>
        </group>
      ))}
      {dragging && (
        <Html position={dragging.at} center style={{ pointerEvents: 'none' }}>
          <div className="gizmo-label gizmo-label-below">{vectorText(dragging.at)} mm</div>
        </Html>
      )}
    </group>
  )
}

/** Where a click puts a guide's point: in the middle of the wall under the clicked spot (half way
 *  from where the ray meets the first shown body to where a line straight in from there, against the
 *  surface's normal, leaves it), else on the surface. (Half way along the view ray, a slanted view put
 *  the point near another face.) */
function middleOfMaterial(raycaster: THREE.Raycaster, scene: THREE.Scene): THREE.Vector3 | null {
  const meshes: THREE.Mesh[] = []
  scene.traverse((o) => { if ((o as THREE.Mesh).isMesh && o.userData.placeable && o.visible) meshes.push(o as THREE.Mesh) })
  const sides = meshes.map((m) => (m.material as THREE.Material).side)
  meshes.forEach((m) => { (m.material as THREE.Material).side = THREE.DoubleSide })   // to find where it leaves too
  try {
    const first = raycaster.intersectObjects(meshes, false)[0]
    if (!first) return null
    if (!first.face) return first.point.clone()
    const normal = first.face.normal.clone().transformDirection(first.object.matrixWorld)
    if (normal.dot(raycaster.ray.direction) > 0) normal.negate()   // (a back face, seen through a cut)
    const inward = new THREE.Raycaster(first.point.clone().addScaledVector(normal, -1e-4), normal.clone().negate())
    const out = inward.intersectObjects(meshes, false).find((h) => h.distance > 1e-3)
    return out ? first.point.clone().add(out.point).multiplyScalar(0.5) : first.point.clone()
  } finally {
    meshes.forEach((m, i) => { (m.material as THREE.Material).side = sides[i] })
  }
}

/** While placing, a click (not a drag, which turns the view) on a shown body calls `onPlace`. */
function PlaceOnClick({ onPlace }: { onPlace: (p: THREE.Vector3) => void }) {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  useEffect(() => {
    const raycaster = new THREE.Raycaster()
    let down: { x: number; y: number } | null = null
    const press = (ev: PointerEvent) => { if (ev.button === 0) down = { x: ev.clientX, y: ev.clientY } }
    const release = (ev: PointerEvent) => {
      if (!down || ev.button !== 0) return
      const moved = Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 4
      down = null
      if (moved) return
      const r = gl.domElement.getBoundingClientRect()
      raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1,
        -((ev.clientY - r.top) / r.height) * 2 + 1), camera)
      const p = middleOfMaterial(raycaster, scene)
      if (p) onPlace(p)
    }
    const canvas = gl.domElement
    canvas.addEventListener('pointerdown', press)
    canvas.addEventListener('pointerup', release)
    canvas.style.cursor = 'crosshair'
    return () => {
      canvas.removeEventListener('pointerdown', press)
      canvas.removeEventListener('pointerup', release)
      canvas.style.cursor = ''
    }
  }, [camera, gl, scene, onPlace])
  return null
}

/** The guide shapes shown (eye on), selected or being placed, the selected one's handles, and Place. */
export function Guides({ sceneSize }: { sceneSize: number }) {
  const shapes = useStore(useShallow((s) => s.nodes.filter((n) => GUIDE_TYPES.includes(n.data.typeId) &&
    (s.visible.includes(n.id) || n.selected || s.placing?.node === n.id))))
  const specs = useStore((s) => s.specs)
  const placing = useStore((s) => s.placing)
  const setPlacing = useStore((s) => s.setPlacing)
  const setInput = useStore((s) => s.setInput)
  const [preview, setPreview] = useState<{ id: string; points: Record<string, THREE.Vector3> } | null>(null)
  const step = offsetStep(sceneSize)

  const list = useMemo(() => shapes.map((n) => shapeOf(n, specs[n.data.typeId], !!n.selected))
    .filter((s): s is Shape => s !== null), [shapes, specs])
  const select = (id: string) => {
    const s = useStore.getState()
    s.onNodesChange(s.nodes.map((n) => ({ id: n.id, type: 'select' as const, selected: n.id === id })))
  }
  const onPlace = useMemo(() => (p: THREE.Vector3) => {
    const s = useStore.getState()
    if (!s.placing) return
    const { node, port } = s.placing
    const type = s.nodes.find((n) => n.id === node)?.data.typeId ?? ''
    const snapped = p.clone().set(Math.round(p.x / step) * step, Math.round(p.y / step) * step, Math.round(p.z / step) * step)
    setInput(node, port, vectorText(snapped))
    const ports = PLACE_PORTS[type] ?? []
    const next = ports[ports.indexOf(port) + 1]
    setPlacing(next ? { node, port: next } : null)
  }, [step, setInput, setPlacing])
  const onDrag = useMemo(() => (id: string, points: Record<string, THREE.Vector3> | null) =>
    setPreview(points ? { id, points } : null), [])
  const onCommit = useMemo(() => (id: string, points: Record<string, THREE.Vector3>) => {
    for (const [port, v] of Object.entries(points)) setInput(id, port, vectorText(v))
  }, [setInput])

  return (
    <group>
      {list.map((shape) => {
        const shown = preview?.id === shape.id ? { ...shape, points: preview.points } : shape
        return (
          <group key={shape.id}>
            <ShapeMesh shape={shown} onSelect={placing ? undefined : () => select(shape.id)} />
            {shape.selected && !placing && (
              <Handles shape={shape} size={sceneSize} step={step} onDrag={onDrag} onCommit={onCommit} />
            )}
          </group>
        )
      })}
      {placing && <PlaceOnClick onPlace={onPlace} />}
    </group>
  )
}
