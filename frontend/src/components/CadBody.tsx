// CAD bodies in the 3D view, drawn the way a CAD program shows a part: smooth within each CAD face and
// crisp at its borders, the faces' edges as lines (tangent ones dimmed), styles and colours from the
// view options (cadLook.ts, the toolbar's Edges ▾).
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { useThree } from '@react-three/fiber'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { loadFaces } from '../api'
import { cancelBoundsTree, scheduleBoundsTree } from '../bvhBuild'
import { useStore } from '../store'
import { viewGeometry, viewNormals, watchFirstFrame } from '../viewTiming'
import type { FaceData, ViewMesh } from '../types'
import {
  CAD_COLORS, LOOK_MAX_TRIANGLES, edgesShown, requestLook, tintColors, useCadLook, type CadLookOptions, type EdgeSet,
  type LookResult,
} from '../cadLook'
import { CAP_COLOR, hideRanges, patchBoxCut, useBoxCut, useBoxCutUniforms, type CutBox } from './viewCut'

const EDGE_DARK = '#16181c'
const EDGE_LIGHT = '#e6e9ee'
/** faces of the Lines style (hidden lines removed) */
const LINES_FACE = '#eef0f3'
/** lines are drawn this share of their distance nearer the camera, so they win over the faces they lie on */
const EDGE_PULL = 8e-4
/** strength of the studio reflections on CAD bodies (on top of the scene's lights) */
export const CAD_ENV = 0.35
/** tangent edges when dimmed, as a share of the sharp ones */
const TANGENT_DIM = 0.3

/** How a body without CAD face data, or too big, is drawn (the options panel says so). */
export type LookState = 'faces' | 'angle' | 'loading' | 'big'
/** What the shown CAD views' looks are worked out from, by node (for the options panel's notes). */
const lookStates = { current: {} as Record<string, LookState>, listeners: new Set<() => void>() }

function setLookState(node: string, state: LookState | null) {
  const s = lookStates
  if ((s.current[node] ?? null) === state) return
  const next = { ...s.current }
  if (state === null) delete next[node]
  else next[node] = state
  s.current = next
  s.listeners.forEach((l) => l())
}

export function useLookStateMap() {
  const [, bump] = useState(0)
  useEffect(() => {
    const l = () => bump((n) => n + 1)
    lookStates.listeners.add(l)
    return () => { lookStates.listeners.delete(l) }
  }, [])
  return lookStates.current
}

/** The look of a view: with its CAD faces once they are loaded (for a shown node's own view), else
 *  worked out from the angles between triangles; null while it is worked out, or for a view too big. */
export function useLook(view: ViewMesh, node?: string): LookResult | null {
  const faces = useStore((s) => (node ? s.faceData[node] : undefined))
  const valid = faces && faces.version === view.version && faces.ids.length === view.faces ? faces : null
  const [failed, setFailed] = useState<ViewMesh | null>(null)
  const big = view.faces > LOOK_MAX_TRIANGLES
  useEffect(() => {
    if (!node || big || !view.faces) return
    let alive = true
    const fail = () => { if (alive) setFailed(view) }
    loadFaces(node).then((r) => { if (typeof r === 'string') fail() }, fail)
    return () => { alive = false }
  }, [node, view, big])
  const want: FaceData | 'plain' | null = big || !view.faces ? null : valid ?? (!node || failed === view ? 'plain' : null)
  const [look, setLook] = useState<{ view: ViewMesh; look: LookResult } | null>(null)
  useEffect(() => {
    if (!want) return
    let alive = true
    requestLook(view, want === 'plain' ? null : want).then((r) => { if (alive) setLook({ view, look: r }) }, () => {})
    return () => { alive = false }
  }, [view, want])
  const shown = look && look.view === view ? look.look : null
  useEffect(() => {
    if (!node) return
    setLookState(node, big ? 'big' : !shown ? 'loading' : shown.byFaces ? 'faces' : 'angle')
  }, [node, big, shown])
  useEffect(() => () => { if (node) setLookState(node, null) }, [node])
  return shown
}

/** The geometry a view is drawn with: the look's once it is there (same triangles, same order), else
 *  the view's own with smooth normals. `userData.source` is its whole index. */
export function useLookGeometry(view: ViewMesh, look: LookResult | null) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    if (look) {
      g.setAttribute('position', new THREE.BufferAttribute(look.positions, 3))
      g.setAttribute('normal', new THREE.BufferAttribute(look.normals, 3))
      g.setIndex(new THREE.BufferAttribute(look.index, 1))
      g.userData.source = look.index
    } else {
      g.setAttribute('position', new THREE.BufferAttribute(view.positions, 3))
      g.setIndex(new THREE.BufferAttribute(view.indices, 1))
      const t = performance.now()
      g.computeVertexNormals()
      viewNormals(view, performance.now() - t)
      g.userData.source = view.indices
    }
    g.computeBoundingSphere()
    viewGeometry(view)
    return g
  }, [view, look])
  useEffect(() => () => { cancelBoundsTree(geometry); geometry.disposeBoundsTree(); geometry.dispose() }, [geometry])
  return geometry
}

const environments = new WeakMap<THREE.WebGLRenderer, THREE.Texture>()

/** A soft studio light for CAD bodies' reflections (made once per renderer, offline: three's room). */
export function useCadEnvironment() {
  const gl = useThree((s) => s.gl)
  return useMemo(() => {
    let texture = environments.get(gl)
    if (!texture) {
      const pmrem = new THREE.PMREMGenerator(gl)
      const room = new RoomEnvironment()
      texture = pmrem.fromScene(room, 0.04).texture
      room.dispose()
      pmrem.dispose()
      environments.set(gl, texture)
    }
    return texture
  }, [gl])
}

/** One kind of lines (sharp or tangent edges). Lines whose two sides both face away from the camera are
 *  hidden when `cull` (an opaque body: such lines only show through a section cut, on the cut face). */
function EdgeLines({ set, color, opacity, cull, clip, box, hide, order }: {
  set: EdgeSet; color: string; opacity: number; cull: boolean; clip: THREE.Plane[]; box?: CutBox | null
  hide?: number[]; order: number
}) {
  const geometry = useMemo(() => {
    const n = set.positions.length / 3
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(set.positions, 3))
    g.setAttribute('edgeA', new THREE.BufferAttribute(set.a, 3))
    g.setAttribute('edgeB', new THREE.BufferAttribute(set.b, 3))
    if (set.parts.length > 1) {  // a list: an index to hide items with
      const index = new Uint32Array(n)
      for (let i = 0; i < n; i++) index[i] = i
      g.setIndex(new THREE.BufferAttribute(index, 1))
      g.userData.source = index
    }
    g.computeBoundingSphere()
    return g
  }, [set])
  useEffect(() => () => geometry.dispose(), [geometry])
  useEffect(() => {
    if (geometry.userData.source) hideRanges(geometry, geometry.userData.source, set.parts, 2, hide)
  }, [geometry, set, hide])
  const boxUniforms = useBoxCutUniforms(box)
  const keep = box?.keep ?? 'outside'
  const uniforms = useMemo(() => ({ uCull: { value: 0 }, uPull: { value: EDGE_PULL } }), [])
  useEffect(() => { uniforms.uCull.value = cull ? 1 : 0 }, [cull, uniforms])
  const patch = useMemo(() => ({
    onBeforeCompile: (shader: THREE.WebGLProgramParametersWithUniforms) => {
      patchBoxCut(shader, boxUniforms, keep)
      shader.uniforms.uCull = uniforms.uCull
      shader.uniforms.uPull = uniforms.uPull
      shader.vertexShader = 'attribute vec3 edgeA;\nattribute vec3 edgeB;\nuniform float uPull;\nvarying float vFacing;\n' +
        shader.vertexShader.replace('#include <project_vertex>', [
          '#include <project_vertex>',
          'vFacing = max(dot(normalMatrix * edgeA, -mvPosition.xyz), dot(normalMatrix * edgeB, -mvPosition.xyz));',
          'mvPosition.xyz *= 1.0 - uPull;',
          'gl_Position = projectionMatrix * mvPosition;',
        ].join('\n'))
      shader.fragmentShader = 'uniform float uCull;\nvarying float vFacing;\n' + shader.fragmentShader.replace(
        '#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\nif (uCull > 0.5 && vFacing < 0.0) discard;')
    },
    customProgramCacheKey: () => `cad-edges-${keep}`,
  }), [boxUniforms, uniforms, keep])
  if (!set.positions.length) return null
  const transparent = opacity < 1
  return (
    <lineSegments geometry={geometry} raycast={() => null} renderOrder={order}>
      <lineBasicMaterial key={transparent ? 't' : 'o'} color={color} transparent={transparent} opacity={opacity}
        depthWrite={!transparent} clippingPlanes={clip} {...patch} />
    </lineSegments>
  )
}

/** The sharp and tangent edges of a look, coloured for the style. */
export function CadEdges({ look, options, ghost, clip, box, hide }: {
  look: LookResult; options: CadLookOptions; ghost?: boolean; clip: THREE.Plane[]; box?: CutBox | null; hide?: number[]
}) {
  if (!edgesShown(options)) return null
  const light = options.style === 'see-through' || options.style === 'wire'
  const color = light ? EDGE_LIGHT : EDGE_DARK
  const strength = (options.style === 'see-through' ? 0.75 : options.style === 'shaded' ? 0.9 : 1) * (ghost ? 0.3 : 1)
  const cull = !ghost && (options.style === 'shaded' || options.style === 'lines')
  const tangent = options.tangent === 'shown' ? strength : options.tangent === 'dimmed' ? strength * TANGENT_DIM : 0
  return (
    <>
      <EdgeLines set={look.sharp} color={color} opacity={strength} cull={cull} clip={clip} box={box} hide={hide} order={2} />
      {tangent > 0 && <EdgeLines set={look.tangent} color={color} opacity={tangent} cull={cull} clip={clip} box={box}
        hide={hide} order={2} />}
    </>
  )
}

/** A CAD body (a view of kind `cad`), drawn in the chosen style. `node`: the shown node, whose CAD faces
 *  make the edges (a list item's piece or a detail has none: edges by angle). */
export function CadBody({ view, node, wireframe, ghost, clip, box, patch, hide }: {
  view: ViewMesh; node?: string; wireframe: boolean; ghost?: boolean; clip: THREE.Plane[]
  /** cut at this box (a detail in it, or this is the detail) */
  box?: CutBox | null
  /** a detail or a list item's piece: not picked, placed on or right-clicked (its view is) */
  patch?: boolean
  /** list items drawn by their pieces instead (store.pieces): not drawn here */
  hide?: number[]
}) {
  const options = useCadLook()
  const look = useLook(view, patch ? undefined : node)
  const geometry = useLookGeometry(view, look)
  useEffect(() => hideRanges(geometry, geometry.userData.source, view.parts, 3, hide), [geometry, view, hide])
  // right-clicks and clicks raycast it: its search tree is built in a worker meanwhile (bvhBuild.ts), on the
  // look's geometry (the one drawn from then on; a body too big for a look keeps the plain one)
  const menuTarget = !patch && !!node && !ghost && (!!look || view.faces > LOOK_MAX_TRIANGLES)
  useEffect(() => { if (menuTarget) scheduleBoundsTree(geometry, geometry.userData.source) }, [geometry, menuTarget])
  const meshRef = useRef<THREE.Mesh>(null)
  useLayoutEffect(() => watchFirstFrame(meshRef.current, view), [geometry, view])  // (before a frame can draw it)
  const style = wireframe ? 'shaded' : options.style
  const tinted = style === 'shaded' || style === 'see-through'
  const colors = useMemo(() => (look && tinted ? tintColors(look, options.tint, options.color) : null),
    [look, tinted, options.tint, options.color])
  useEffect(() => {
    if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    else geometry.deleteAttribute('color')
  }, [geometry, colors])
  const cut = useBoxCut(box)
  const env = useCadEnvironment()
  const seeThrough = style === 'see-through'
  const opacity = ghost ? (seeThrough ? 0.15 : 0.25) : seeThrough ? 0.3 : 1
  const transparent = opacity < 1
  const surface = style !== 'wire'
  const sectioned = clip.length > 0 && !transparent && surface && !wireframe
  const side = sectioned ? THREE.FrontSide : THREE.DoubleSide
  const key = `${style}-${!!colors}-${transparent}-${wireframe}`
  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} renderOrder={transparent ? 1 : 0} raycast={ghost || patch ? () => null : undefined}
        // a guide shape's Place clicks on it (Guides.tsx); right-clicks open its node's menu
        userData={patch ? {} : { placeable: true, menu: node && !ghost ? { view: node } : undefined }}>
        {style === 'lines'
          ? <meshBasicMaterial key={key} color={LINES_FACE} transparent={transparent} opacity={opacity}
              depthWrite={!transparent} side={side} clippingPlanes={clip} {...cut} />
          : <meshStandardMaterial key={key} color={colors ? '#ffffff' : options.color} vertexColors={!!colors}
              metalness={0.08} roughness={0.5} envMap={env} envMapIntensity={CAD_ENV} wireframe={wireframe}
              visible={surface} transparent={transparent} opacity={opacity} depthWrite={!transparent} side={side}
              clippingPlanes={clip} {...cut} />}
      </mesh>
      {sectioned && (
        // the inside seen through the cut: back faces in one flat colour read as a solid cut face
        <mesh geometry={geometry} raycast={() => null}>
          <meshBasicMaterial color={CAP_COLOR} side={THREE.BackSide} clippingPlanes={clip} {...cut} />
        </mesh>
      )}
      {look && !wireframe && <CadEdges look={look} options={options} ghost={ghost} clip={clip} box={box} hide={hide} />}
    </group>
  )
}

// --- the options panel ---------------------------------------------------------------------------
function Seg<T extends string>({ value, options, onPick, disabled }: {
  value: T; options: [T, string, string][]; onPick: (v: T) => void; disabled?: boolean
}) {
  return (
    <div className="seg">
      {options.map(([v, label, title]) => (
        <button key={v} className={value === v ? 'on' : ''} title={title} disabled={disabled} onClick={() => onPick(v)}>{label}</button>
      ))}
    </div>
  )
}

/** Edges on or off from the toolbar or E: Lines and Wire are all edges, so turning them off goes back to shaded. */
export function toggleCadEdges() {
  const o = useCadLook.getState()
  if (o.style === 'lines' || o.style === 'wire') o.set({ style: 'shaded', edges: false })
  else o.set({ edges: !o.edges })
}

/** How CAD bodies are drawn (the toolbar's Edges ▾). `shown`: the CAD views on screen, by node. */
export function CadLookPanel({ shown, onClose }: { shown: string[]; onClose: () => void }) {
  const o = useCadLook()
  const states = useLookStateMap()
  const allEdges = o.style === 'lines' || o.style === 'wire'
  const angle = shown.filter((id) => states[id] === 'angle')
  const big = shown.filter((id) => states[id] === 'big')
  const loading = shown.some((id) => states[id] === 'loading')
  return (
    <div className="tool-panel cad-look">
      <div className="slice-row">
        <span className="slice-title"><b>CAD bodies</b>{loading ? ' · finding edges…' : ''}</span>
        <button className="small-btn" title="Close" onClick={onClose}>✕</button>
      </div>
      <div className="look-row">
        <span>Style</span>
        <Seg value={o.style} onPick={(style) => o.set({ style })} options={[
          ['shaded', 'Shaded', 'Shaded faces'],
          ['see-through', 'See-through', 'Faces you can see through, with the edges behind them'],
          ['lines', 'Lines', 'Hidden lines removed: light faces with dark edges, like a drawing'],
          ['wire', 'Wire', 'Only the edges, hidden ones too'],
        ]} />
      </div>
      <div className="look-row">
        <span>Edges</span>
        <span className="toggle-field">
          <button type="button" role="switch" aria-checked={edgesShown(o)} className={`toggle ${edgesShown(o) ? 'on' : ''}`}
            disabled={allEdges} title={allEdges ? 'Lines and Wire always show the edges' : 'The borders of the CAD faces (E)'}
            onClick={() => o.set({ edges: !o.edges })}>
            <span className="toggle-knob" />
          </button>
        </span>
      </div>
      <div className="look-row">
        <span title="Edges where two faces meet smoothly, e.g. where a fillet starts">Tangent edges</span>
        <Seg value={o.tangent} onPick={(tangent) => o.set({ tangent })} disabled={!edgesShown(o)} options={[
          ['shown', 'Shown', 'As strong as the other edges'],
          ['dimmed', 'Dimmed', 'Fainter than the sharp edges'],
          ['hidden', 'Hidden', 'Only sharp edges'],
        ]} />
      </div>
      <div className="look-row">
        <span>Colour by</span>
        <Seg value={o.tint} onPick={(tint) => o.set({ tint })} disabled={o.style === 'lines' || o.style === 'wire'} options={[
          ['none', 'One colour', 'Every face in the colour below'],
          ['faces', 'Faces', 'Every CAD face in a slightly different shade, to tell faces apart'],
          ['bodies', 'Bodies', 'Every body in its own colour'],
        ]} />
      </div>
      <div className="look-row">
        <span>Colour</span>
        <div className="look-swatches">
          {CAD_COLORS.map((c) => (
            <button key={c.color} className={`look-swatch ${o.color === c.color ? 'on' : ''}`} style={{ background: c.color }}
              title={c.name} aria-label={c.name} onClick={() => o.set({ color: c.color })} />
          ))}
        </div>
      </div>
      {angle.length > 0 && (
        <div className="look-note">
          {angle.join(', ')}: no CAD faces for this view (decimated for display, or a mesh), so edges are where the
          surface bends sharply{o.tint === 'faces' ? ' and faces are coloured by body' : ''}
        </div>
      )}
      {big.length > 0 && (
        <div className="look-note">{big.join(', ')}: over {LOOK_MAX_TRIANGLES / 1e6}M triangles, drawn without edges</div>
      )}
    </div>
  )
}
