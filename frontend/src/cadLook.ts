// How CAD bodies (views of kind `cad`: Import STEP, Pick Body, the transforms) are drawn: the options
// (kept in localStorage) and the look of each view, worked out once per view and face data in
// cadLook.worker.ts.
import { create } from 'zustand'
import * as THREE from 'three'
import type { FaceData, ViewMesh } from './types'

export type CadStyle = 'shaded' | 'see-through' | 'lines' | 'wire'
export type TangentEdges = 'shown' | 'dimmed' | 'hidden'
export type CadTint = 'none' | 'faces' | 'bodies'

export interface CadLookOptions {
  /** shaded; see-through; lines = hidden lines removed (light flat faces, dark edges); wire = the edges only */
  style: CadStyle
  /** the CAD faces' borders drawn on shaded and see-through bodies (lines and wire always have them) */
  edges: boolean
  /** borders where two faces meet smoothly (a fillet's ends) */
  tangent: TangentEdges
  /** every face, or every body, in a slightly different shade of the colour */
  tint: CadTint
  color: string
}

export const CAD_COLORS: { name: string; color: string }[] = [
  { name: 'Sand', color: '#d3c3a1' },
  { name: 'Steel', color: '#b4bfcc' },
  { name: 'Light grey', color: '#d6d7d9' },
  { name: 'Brass', color: '#d4ac55' },
  { name: 'Blue', color: '#7fa6d6' },
  { name: 'Graphite', color: '#7d828b' },
]

const DEFAULTS: CadLookOptions = { style: 'shaded', edges: true, tangent: 'dimmed', tint: 'none', color: CAD_COLORS[0].color }
const STORAGE = 'cadLook'

function stored(): CadLookOptions {
  try {
    const v = localStorage.getItem(STORAGE)
    return v ? { ...DEFAULTS, ...(JSON.parse(v) as Partial<CadLookOptions>) } : DEFAULTS
  } catch { return DEFAULTS }
}

export const useCadLook = create<CadLookOptions & { set: (change: Partial<CadLookOptions>) => void }>((set, get) => ({
  ...stored(),
  set: (change) => {
    set(change)
    const s = get()
    const options: CadLookOptions = { style: s.style, edges: s.edges, tangent: s.tangent, tint: s.tint, color: s.color }
    try { localStorage.setItem(STORAGE, JSON.stringify(options)) } catch { /* ignore */ }
  },
}))

/** Edges are drawn in this style (lines and wire always have them). */
export const edgesShown = (o: CadLookOptions) => o.edges || o.style === 'lines' || o.style === 'wire'

// --- the look of one view ------------------------------------------------------------------------
/** Lines of one kind: two points per line, with the normals of the two triangles beside it. */
export interface EdgeSet {
  positions: Float32Array
  a: Float32Array
  b: Float32Array
  /** lines per list item (they come in the items' order) */
  parts: Uint32Array
}

export interface LookRequest {
  id: number
  positions: Float32Array
  indices: Uint32Array
  /** CAD face and body of every triangle (null: no face data, edges by angle) */
  faces: Uint32Array | null
  bodies: Uint32Array | null
  parts: number[] | null
}

export interface LookResult {
  id: number
  /** vertices split where the CAD face changes or the surface bends sharply */
  positions: Float32Array
  normals: Float32Array
  /** the view's triangles in the same order, on these vertices */
  index: Uint32Array
  /** face of every vertex (dense numbers), and per face its CAD face number and body */
  vface: Uint32Array
  faceIds: Uint32Array
  faceBodies: Uint32Array
  sharp: EdgeSet
  tangent: EdgeSet
  /** worked out from CAD faces (else by the angle between triangles) */
  byFaces: boolean
  ms: number
}

/** Bigger views are drawn plainly, without edges: the worker holds ~190 B per triangle while it works
 *  (0.6 GB at this size), the result ~40 B more per triangle beside the view. */
export const LOOK_MAX_TRIANGLES = 3_000_000

let worker: Worker | null = null
let nextId = 1
const waiting = new Map<number, { resolve: (r: LookResult) => void; reject: (e: unknown) => void }>()

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./cadLook.worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (e: MessageEvent<LookResult>) => {
      const w = waiting.get(e.data.id)
      waiting.delete(e.data.id)
      w?.resolve(e.data)
    }
    worker.onerror = (e) => {
      // a failed look leaves the bodies drawn plainly; the next request starts a new worker
      for (const w of waiting.values()) w.reject(e)
      waiting.clear()
      worker?.terminate()
      worker = null
    }
  }
  return worker
}

/** Looks per view (dropped with the view) and face data (`plain`: none). */
const looks = new WeakMap<ViewMesh, Map<FaceData | 'plain', Promise<LookResult>>>()

/** The look of a view, with its CAD faces when `faces` is given: worked out once, shared by every
 *  drawing of the view (the plain one is dropped once the one with faces is there). */
export function requestLook(view: ViewMesh, faces: FaceData | null): Promise<LookResult> {
  let byView = looks.get(view)
  if (!byView) looks.set(view, (byView = new Map()))
  const key = faces ?? 'plain'
  const known = byView.get(key)
  if (known) return known
  const id = nextId++
  const request: LookRequest = {
    id, positions: view.positions.slice(), indices: view.indices.slice(),
    faces: faces ? faces.ids.slice() : null, bodies: faces ? faces.bodies.slice() : null, parts: view.parts ?? null,
  }
  const promise = new Promise<LookResult>((resolve, reject) => {
    waiting.set(id, { resolve, reject })
    getWorker().postMessage(request, [request.positions.buffer, request.indices.buffer,
      ...(request.faces ? [request.faces.buffer] : []), ...(request.bodies ? [request.bodies.buffer] : [])])
  })
  byView.set(key, promise)
  promise.then((r) => {
    if (faces) byView!.delete('plain')
    console.debug(`CAD look: ${view.faces} triangles, ${r.sharp.parts.reduce((a, b) => a + b, 0)} sharp and ` +
      `${r.tangent.parts.reduce((a, b) => a + b, 0)} tangent edges${r.byFaces ? '' : ' (by angle)'} in ${r.ms.toFixed(0)} ms`)
  }, () => byView!.delete(key))
  return promise
}

// --- colours ------------------------------------------------------------------------------------
/** A number's spread over 0..1 (golden ratio), so neighbouring face numbers get different shades. */
const spread = (n: number) => (n * 0.6180339887) % 1

/** Vertex colours for face or body tints (linear, as three.js works), or null for one colour. */
export function tintColors(look: LookResult, tint: CadTint, base: string): Float32Array | null {
  if (tint === 'none') return null
  const hsl = { h: 0, s: 0, l: 0 }
  new THREE.Color(base).getHSL(hsl)
  const F = look.faceIds.length
  const table = new Float32Array(F * 3)
  const c = new THREE.Color()
  for (let f = 0; f < F; f++) {
    if (tint === 'faces') {
      // the same hue, a little lighter or darker, a hint warmer or cooler
      const k = spread(look.faceIds[f] + 7 * look.faceBodies[f])
      c.setHSL((hsl.h + (k - 0.5) * 0.1 + 1) % 1, Math.min(1, hsl.s * (0.7 + 0.6 * spread(look.faceIds[f] * 3 + 1))),
        Math.min(0.92, Math.max(0.2, hsl.l + (k - 0.5) * 0.28)))
    } else {
      // bodies: distinct hues, as light and soft as the colour
      const k = spread(look.faceBodies[f] + 1)
      c.setHSL(k, Math.max(0.28, Math.min(0.5, hsl.s + 0.15)), Math.min(0.8, Math.max(0.55, hsl.l)))
    }
    table[f * 3] = c.r; table[f * 3 + 1] = c.g; table[f * 3 + 2] = c.b
  }
  const out = new Float32Array(look.vface.length * 3)
  for (let v = 0; v < look.vface.length; v++) {
    const f = look.vface[v] * 3
    out[v * 3] = table[f]; out[v * 3 + 1] = table[f + 1]; out[v * 3 + 2] = table[f + 2]
  }
  return out
}
