// The timing line of every view the editor receives (view_ready): what the engine says (the node's seconds,
// how long its view waited and took to pack), the download, parsing, normals and the first frame drawn with
// it, all in ms by performance.now(). Kept in window.__viewTiming (the last MAX_RECORDS, always: cheap) for
// reading from the console; shown in a corner of the 3D view while on (the 3D view's View ▸ View timing, or
// localStorage viewTiming = "1").
import { useEffect, useState } from 'react'
import { create } from 'zustand'
import type * as THREE from 'three'
import type { ViewMesh } from './types'

/** What the engine adds to view_ready (`timing`; missing from an engine older than this). */
export interface EngineTiming {
  /** seconds the node that made the view took (0 when cached), whether it was cached */
  node_s?: number | null
  cached?: boolean | null
  /** seconds into the run when that node was done */
  run_s?: number | null
  /** ms from the node being done to its view's packing starting (other views ahead of it), and packing */
  wait_ms?: number | null
  pack_ms?: number
  bytes?: number
}

export interface ViewTiming {
  node: string
  /** the version announced, and the one served if a newer one came instead */
  version: number
  served?: number
  run?: number
  kind: string
  level?: string
  faces?: number
  /** performance.now() when view_ready came */
  at: number
  engine: EngineTiming | null
  download_ms?: number
  bytes?: number
  /** the blob into positions and indices, and the view's box */
  parse_ms?: number
  /** computeVertexNormals on the main thread (CAD bodies: only while their look is being worked out), or setting the
   *  engine's normals from the field (`normals` "engine": views of implicit bodies) */
  normals_ms?: number
  normals?: 'engine' | 'browser'
  /** from the geometry being made to the first frame that drew it (upload to the graphics card included) */
  frame_ms?: number
  /** from view_ready to that first frame */
  total_ms?: number
  /** not drawn: the store kept a newer view, or the download failed */
  skipped?: boolean
}

declare global {
  interface Window { __viewTiming?: ViewTiming[] }
}

const MAX_RECORDS = 200
const STORAGE = 'viewTiming'

function storedOn() {
  try { return localStorage.getItem(STORAGE) === '1' } catch { return false }
}

/** Whether the overlay is on, and a count bumped when a record changes while it is (to redraw it). */
export const useViewTiming = create<{ on: boolean; tick: number }>(() => ({ on: storedOn(), tick: 0 }))

export function setViewTiming(on: boolean) {
  try { localStorage.setItem(STORAGE, on ? '1' : '0') } catch { /* ignore */ }
  useViewTiming.setState({ on })
}

const FIELD_NORMALS = 'fieldNormals'

/** Whether views are drawn with the engine's normals from the field when it sent them (true surface normals: smooth
 *  textures, a crease softened over about a voxel) or with the triangles' own (computeVertexNormals): the 3D view's
 *  View ▸ Field normals, for comparing; localStorage `fieldNormals` "0" = off. */
export const useFieldNormals = create<{ on: boolean }>(() => {
  try { return { on: localStorage.getItem(FIELD_NORMALS) !== '0' } } catch { return { on: true } }
})

export function setFieldNormals(on: boolean) {
  try { localStorage.setItem(FIELD_NORMALS, on ? '1' : '0') } catch { /* ignore */ }
  useFieldNormals.setState({ on })
}

function changed() {
  if (useViewTiming.getState().on) useViewTiming.setState((s) => ({ tick: s.tick + 1 }))
}

function records() {
  if (!window.__viewTiming) window.__viewTiming = []
  return window.__viewTiming
}

/** Each view's record, by the view object the store keeps (drawings find theirs by it). */
const byView = new WeakMap<ViewMesh, ViewTiming>()
/** When each view's geometry was made, for frame_ms. */
const made = new WeakMap<ViewMesh, number>()

/** A view_ready came: its record starts (the download follows at once). */
export function viewArrived(e: { node: string; version: number; kind: string; level?: string; faces?: number; run?: number;
  timing?: EngineTiming | null }): ViewTiming {
  const record: ViewTiming = { node: e.node, version: e.version, run: e.run, kind: e.kind, level: e.level, faces: e.faces,
    at: performance.now(), engine: e.timing ?? null }
  const list = records()
  list.push(record)
  if (list.length > MAX_RECORDS) list.splice(0, list.length - MAX_RECORDS)
  changed()
  return record
}

/** Downloaded and parsed into `view` (the object handed to the store), or not (`view` null). */
export function viewFetched(record: ViewTiming, view: ViewMesh | null, times: { download_ms: number; bytes?: number;
  parse_ms?: number }) {
  Object.assign(record, times)
  if (view) {
    if (view.version !== record.version) record.served = view.version
    byView.set(view, record)
  } else record.skipped = true
  changed()
}

/** The normals of a view's geometry took `ms` (worked out here, or the engine's set); the geometry is made now. */
export function viewNormals(view: ViewMesh, ms: number, source: 'engine' | 'browser' = 'browser') {
  const record = byView.get(view)
  if (record && record.normals_ms === undefined) {
    record.normals_ms = ms
    record.normals = source
  }
}

/** A view's geometry is made now (the first frame that draws it is timed from here). */
export function viewGeometry(view: ViewMesh) {
  if (byView.has(view) && !made.has(view)) made.set(view, performance.now())
}

/** Times the first frame that draws `mesh` (its view's first geometry), once per view: three.js calls
 *  onAfterRender right after the mesh's draw call, its buffers uploaded. Returns the cleanup. */
export function watchFirstFrame(mesh: THREE.Mesh | null, view: ViewMesh) {
  const record = byView.get(view)
  if (!mesh || !record || record.frame_ms !== undefined) return undefined
  const own = mesh.onAfterRender
  mesh.onAfterRender = (...args) => {
    mesh.onAfterRender = own
    own.apply(mesh, args)
    const now = performance.now()
    if (record.frame_ms !== undefined) return
    record.frame_ms = now - (made.get(view) ?? now)
    record.total_ms = now - record.at
    changed()
  }
  return () => { mesh.onAfterRender = own }
}

/** The records still waiting for their first frame when a newer view of their node is drawn never get one. */
function skippedOf(list: ViewTiming[], r: ViewTiming, i: number) {
  return r.skipped || (r.frame_ms === undefined && list.slice(i + 1).some((o) => o.node === r.node && o.frame_ms !== undefined))
}

const ms = (v: number | undefined | null) => (v === undefined || v === null ? '–' : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)}`)
const mb = (b: number | undefined) => (b === undefined ? '–' : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`)

/** One record as a line: node, quality and size; engine node and packing; download, parse, normals, frame. */
export function timingLine(r: ViewTiming, skipped = false) {
  const e = r.engine
  const engine = e ? `node ${e.cached ? 'cached' : e.node_s === undefined || e.node_s === null ? '–' : `${e.node_s.toFixed(2)} s`}` +
    ` · pack ${ms(e.pack_ms)}${e.wait_ms ? ` (+${ms(e.wait_ms)} wait)` : ''}` : 'engine –'
  const tris = r.faces === undefined ? '' : ` ${r.faces >= 1e6 ? `${(r.faces / 1e6).toFixed(1)}M` : `${(r.faces / 1e3).toFixed(0)}k`}`
  return `${r.node} ${r.level ?? r.kind}${tris} | ${engine} | dl ${ms(r.download_ms)} ${mb(r.bytes)} · parse ${ms(r.parse_ms)}` +
    ` · normals ${ms(r.normals_ms)}${r.normals === 'engine' ? ' (engine)' : ''} · frame ${ms(r.frame_ms)}` +
    ` | total ${ms(r.total_ms)}${skipped ? ' (not drawn)' : ''}`
}

const SHOWN = 6

/** The last few records, in a corner of the 3D view, while on. */
export function ViewTimingOverlay() {
  const on = useViewTiming((s) => s.on)
  useViewTiming((s) => s.tick)
  // frames come after the last change of a record's other times: look again shortly after one
  const [, bump] = useState(0)
  useEffect(() => {
    if (!on) return
    const t = setInterval(() => bump((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [on])
  if (!on) return null
  const list = records()
  const from = Math.max(0, list.length - SHOWN)
  return (
    <div className="view-timing" title="View timing (ms unless marked): the engine's node and packing, download, parsing, normals, first frame; total from view_ready to that frame. All records: window.__viewTiming">
      {list.length === 0 && <div className="muted">View timing: no views received yet</div>}
      {list.slice(from).map((r, k) => (
        <div key={from + k} className={skippedOf(list, r, from + k) ? 'muted' : undefined}>{timingLine(r, skippedOf(list, r, from + k))}</div>
      ))}
    </div>
  )
}
