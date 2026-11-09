import { CLIENT, isClip, useStore, type Clip } from './store'
import type { Axis, Catalog, FaceData, Paint, Quality, QueueState, Recipe, RunTag, ViewMesh } from './types'
import { QUALITIES } from './types'
import { viewArrived, viewFetched, type EngineTiming } from './viewTiming'

async function check(res: Response) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch { /* not JSON */ }
    throw new Error(detail)
  }
  return res
}

export const getJSON = async <T,>(url: string): Promise<T> => (await check(await fetch(url))).json()

export const postJSON = async <T,>(url: string, body: unknown): Promise<T> =>
  (await check(await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }))).json()

/** A body of a list, as /api/bodies describes it: its STEP name (null: not known), box (mm), faces, triangles. */
export interface BodyInfo { index: number; name: string | null; lo: number[]; hi: number[]; faces: number | null; triangles: number }

export const api = {
  catalog: () => getJSON<Catalog>('/api/catalog'),
  recipes: () => getJSON<{ name: string; path: string }[]>('/api/recipes'),
  readRecipe: (path: string) => getJSON<Recipe>(`/api/file?path=${encodeURIComponent(path)}`),
  saveRecipe: (path: string, recipe: Recipe) =>
    postJSON<{ path: string }>(`/api/file?path=${encodeURIComponent(path)}`, recipe),
  browse: (path: string, exts: string[]) =>
    getJSON<{ path: string; parent: string | null; entries: { name: string; path: string; dir: boolean }[] }>(
      `/api/browse?path=${encodeURIComponent(path)}&exts=${encodeURIComponent(exts.join(','))}`,
    ),
  fileDialog: (req: { mode: 'open' | 'save'; title: string; exts: string[]; dir: string; name: string }) =>
    postJSON<{ path?: string | null; busy?: boolean; unavailable?: string }>('/api/file_dialog', req),
  bodies: (node: string, input: string, source: string, port: string, names = false) =>
    getJSON<{ bodies: BodyInfo[] | null; source?: string; names_unread?: boolean }>(
      `/api/bodies/${encodeURIComponent(node)}?input=${encodeURIComponent(input)}&source=${encodeURIComponent(source)}` +
      `&port=${encodeURIComponent(port)}&names=${names}`),
  fonts: () => getJSON<{ fonts: { value: string; family: string; style: string }[] }>('/api/fonts'),
  reveal: (path: string) => postJSON<{ shown: string }>('/api/reveal', { path }),
  cancel: () => postJSON('/api/cancel', {}),
  autosave: (path: string | null) =>
    getJSON<{ path?: string | null; time?: number; recipe?: Recipe }>(`/api/autosave?path=${encodeURIComponent(path ?? '')}`),
  putAutosave: (path: string | null, recipe: Recipe) => postJSON('/api/autosave', { path, recipe }),
  dropAutosave: (path: string | null) =>
    fetch(`/api/autosave?path=${encodeURIComponent(path ?? '')}`, { method: 'DELETE' }).catch(() => {}),
  presets: () => getJSON<{ name: string; nodes: number; doc: string }[]>('/api/presets'),
  preset: async (name: string): Promise<Clip | null> => {
    const clip = await getJSON<unknown>(`/api/presets/${encodeURIComponent(name)}`)
    return isClip(clip) ? clip : null
  },
  savePreset: (name: string, clip: Clip) => postJSON(`/api/presets/${encodeURIComponent(name)}`, clip),
  deletePreset: (name: string) => fetch(`/api/presets/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  check: (node: string, density: number, thickness: boolean, minWall: number) =>
    postJSON<CheckReport>('/api/check', { node, density, thickness, min_wall: minWall }),
  queue: () => getJSON<QueueState>('/api/queue'),
  removeQueued: (id: number) => postJSON<QueueState>('/api/queue/remove', { id }),
}

export interface CheckReport {
  triangles: number
  vertices: number
  size?: [number, number, number]
  area?: number
  volume?: number
  weight?: number
  bodies?: number
  cavities?: number
  cavity_volume?: number
  open_edges?: number
  view_version: number
  thickness?: { thinnest: number; p1: number; median: number; below: number; min_wall: number }
  paint?: { version: number; view_version: number; min: number; max: number }
}

/** Counts a download from the engine for the status bar ("loading the view"). */
async function loading<T>(work: Promise<T>): Promise<T> {
  useStore.getState().countLoading(1)
  try { return await work } finally { useStore.getState().countLoading(-1) }
}

/** Colours for a view (a value per vertex) from the engine. */
export async function fetchPaint(node: string, meta: { version: number; view_version: number; min: number; max: number;
  label: string; unit?: string; reverse?: boolean; clipped?: boolean; key?: string | null; level?: Quality },
  source: Paint['source']) {
  await loading((async () => {
    const res = await fetch(`/api/view/${encodeURIComponent(`paint:${node}`)}`)
    if (!res.ok) return
    const values = new Float32Array(await res.arrayBuffer())
    const version = Number(res.headers.get('X-View-Version') ?? meta.version)
    useStore.getState().setPaint(node, { version, viewVersion: meta.view_version, values, min: meta.min, max: meta.max,
      label: meta.label, source, unit: meta.unit, reverse: meta.reverse, clipped: meta.clipped, key: meta.key,
      level: meta.level })
  })())
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

const facesLoading = new Map<string, Promise<FaceData | string>>()

/** Which CAD face and body each triangle of a shown CAD view is: from the store while it matches
 *  the view's version, else fetched once (and kept in the store). A string says why there is none. */
export function loadFaces(node: string): Promise<FaceData | string> {
  const s = useStore.getState()
  const view = s.views[node]
  if (!view) return Promise.resolve(`${node} is not shown`)
  const cached = s.faceData[node]
  if (cached && cached.version === view.version) return Promise.resolve(cached)
  const key = `${node}@${view.version}`
  let p = facesLoading.get(key)
  if (!p) {
    p = fetchFaces(node).then((r) => {
      if (typeof r !== 'string') useStore.getState().setFaceData(node, r)
      return r
    }).finally(() => facesLoading.delete(key))
    facesLoading.set(key, p)
  }
  return p
}

/** One CAD face of a shown CAD view: its surface type and size (right-click menus). */
export interface FaceInfo {
  type: string
  area: number
  centre: [number, number, number]
  normal: [number, number, number] | null
  axis: [number, number, number] | null
  radius: number | null
  /** a cylinder whose normals point to its axis: a hole or bore */
  hole: boolean
  triangles: number
  /** the body's name in the STEP file ("Fillet9"), null when it has none or it is not known */
  body_name?: string | null
}

const faceInfos = new Map<string, Promise<FaceInfo | null>>()

/** The surface type and size of one face of a shown CAD view (null if the engine cannot tell). */
export function loadFaceInfo(node: string, version: number, body: number, face: number): Promise<FaceInfo | null> {
  const key = `${node}@${version}:${body}:${face}`
  let p = faceInfos.get(key)
  if (!p) {
    p = getJSON<FaceInfo>(`/api/faceinfo/${encodeURIComponent(node)}?body=${body}&face=${face}`).catch(() => null)
    if (faceInfos.size > 200) faceInfos.clear()
    faceInfos.set(key, p)
  }
  return p
}

/** The CAD face under a point of an implicit body's view: the CAD output it is built from (`source`.
 *  `port`), the face and body, how far the point is from it, and that body list's copies test. */
export interface FaceAt {
  source: string
  port: string
  face: number
  body: number
  distance: number
  same: boolean
  bodies: number
  info: FaceInfo | null
}

/** The CAD face nearest to `point` among the CAD bodies the view of `node` is built from (null if
 *  the engine finds none). */
export function loadFaceAt(node: string, point: [number, number, number]): Promise<FaceAt | null> {
  const [x, y, z] = point.map((v) => v.toFixed(4))
  return getJSON<FaceAt>(`/api/face_at/${encodeURIComponent(node)}?x=${x}&y=${y}&z=${z}`).catch(() => null)
}

/** Which triangles of the view of `node` (`detail`: of its detail) lie on the CAD face loadFaceAt finds
 *  at `point`, and the version of the mesh they are numbered in; `edge` is the view's triangle size
 *  there (mm). Null if the engine finds none. */
export async function loadFaceRegion(node: string, point: [number, number, number], edge: number, detail = false):
  Promise<{ version: number; triangles: Uint32Array } | null> {
  const [x, y, z] = point.map((v) => v.toFixed(4))
  try {
    const res = await fetch(`/api/face_region/${encodeURIComponent(node)}?x=${x}&y=${y}&z=${z}&edge=${edge.toFixed(4)}` +
      (detail ? '&detail=true' : ''))
    if (!res.ok) return null
    return { version: Number(res.headers.get('X-View-Version') ?? 0), triangles: new Uint32Array(await res.arrayBuffer()) }
  } catch {
    return null
  }
}

/** The node that made the surface of a shown body at a point (`node`: its recipe-level id, a group
 *  node for nodes inside a group), and the nodes passed on the way up from the shown one. */
export interface NodeAt {
  node: string
  inner: string | null
  type: string
  path: { node: string; value: number | null }[]
}

/** Which node made the surface of the view of `node` at `point`; `tol` is the preview's resolution
 *  there (mm). Null if the engine cannot tell. */
export function loadNodeAt(node: string, point: [number, number, number], tol: number): Promise<NodeAt | null> {
  const [x, y, z] = point.map((v) => v.toFixed(4))
  return getJSON<NodeAt>(`/api/node_at/${encodeURIComponent(node)}?x=${x}&y=${y}&z=${z}&tol=${tol.toFixed(4)}`).catch(() => null)
}

/** Select similar: the faces of body `body` like its face `face` (mode size, plane or type), on the CAD
 *  view of `node`, or on the CAD output `source`.`port` behind the view of `node`. A string says why not. */
export async function loadSimilarFaces(node: string, body: number, face: number, mode: 'size' | 'plane' | 'type',
  source?: { node: string; port: string }): Promise<number[] | string> {
  const q = `body=${body}&face=${face}&mode=${mode}` +
    (source ? `&source=${encodeURIComponent(source.node)}&port=${encodeURIComponent(source.port)}` : '')
  try {
    return (await getJSON<{ faces: number[] }>(`/api/similar_faces/${encodeURIComponent(node)}?${q}`)).faces
  } catch (err) {
    return (err as Error).message
  }
}

/** The wall thickness (mm) of the shown mesh of `node` under `point`, straight in along -`normal`
 *  (null: no opposite wall, or the engine cannot tell). */
export async function loadThickness(node: string, point: [number, number, number], normal: [number, number, number]) {
  const [x, y, z] = point.map((v) => v.toFixed(4))
  const [nx, ny, nz] = normal.map((v) => v.toFixed(5))
  try {
    return (await getJSON<{ thickness: number | null }>(
      `/api/thickness/${encodeURIComponent(node)}?x=${x}&y=${y}&z=${z}&nx=${nx}&ny=${ny}&nz=${nz}`)).thickness
  } catch {
    return null
  }
}

/** The cross-section of a member (see `loadSection`): area in mm², the diameter of a round member of that area,
 *  the narrowest and widest across, the wall thickness straight in, the cut's centre and normal and its outline
 *  (the outer loop, then holes); or `open` when the cut runs out wide (a wall or a plate). */
export type MemberSection =
  | { open: true; thickness: number; centre: [number, number, number] }
  | {
      open?: undefined; area: number; diameter: number; narrowest: number; widest: number; thickness: number
      centre: [number, number, number]; normal: [number, number, number]
      outline: [number, number, number][]; holes: [number, number, number][][]
    }

/** The cross-section of the member of the shown mesh of `node` under `point` (null: no wall behind it, or the
 *  engine cannot tell): the smallest closed cut through the member's middle, halfway straight in. */
export async function loadSection(node: string, point: [number, number, number], normal: [number, number, number]) {
  const [x, y, z] = point.map((v) => v.toFixed(4))
  const [nx, ny, nz] = normal.map((v) => v.toFixed(5))
  try {
    return (await getJSON<{ section: MemberSection | null }>(
      `/api/section/${encodeURIComponent(node)}?x=${x}&y=${y}&z=${z}&nx=${nx}&ny=${ny}&nz=${nz}`)).section
  } catch {
    return null
  }
}

/** A short name for an export in the queue: the recipe and the time it was asked for. */
function exportLabel() {
  const path = useStore.getState().recipePath
  const name = path ? path.split(/[\\/]/).pop()!.replace(/\.json$/i, '') : 'untitled'
  return `${name} ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`
}

/** Optimize a Topology Optimization node: a long job, queued like an export (previews wait for it).
 *  The node is shown so its shape can be watched growing. */
export async function startOptimize(node: string) {
  const s = useStore.getState()
  if (s.inactive) return
  if (!s.visible.includes(node)) s.toggleVisible(node)
  s.setRequested({ version: s.graphVersion, at: Date.now() })  // (it waits while a preview stops: the status bar says so)
  s.setStarting(node)   // (and the button: the answer comes once that preview has stopped, up to half a minute)
  try {
    await postJSON('/api/optimize', { graph: s.recipe(), node, quality: s.quality, visible: useStore.getState().visible,
      tag: { version: s.graphVersion, client: CLIENT } satisfies RunTag, client: CLIENT })
  } catch (err) {
    s.setRequested(null)
    s.addLog(`Could not start: ${(err as Error).message}`, 'error')
  } finally {
    s.setStarting(null)
  }
}

/** Stop what runs now. The engine answers at once and the run ends at its next stopping point; until
 *  then the status bar says it is stopping (Stop seemed to do nothing during a long step). */
export async function stopRun() {
  const s = useStore.getState()
  if (s.run?.running) useStore.setState({ run: { ...s.run, stopping: true } })
  try {
    await api.cancel()
  } catch (err) {
    s.addLog(`Could not stop: ${(err as Error).message}`, 'error')
  }
}

/** The next preview makes every view again (this tab took the engine back from another, whose views it
 *  holds under the same node names). */
let freshViews = false

/** This tab uses the engine from now on (Use this tab, or a freshly loaded page): the other tabs pause.
 *  `rerun`: preview again with every view made anew. */
export async function claimEngine(rerun: boolean) {
  try {
    await postJSON('/api/claim', { client: CLIENT })
  } catch {
    return   // (the engine is not there: the next connection claims)
  }
  useStore.setState({ inactive: false })
  if (rerun) {
    freshViews = true
    startRun('preview')
  }
}

/** What the 3D view shows now, for a preview: the engine keeps a view it would make the same at the
 *  run's quality or coarser (refining after a change then only recomputes what changed). */
function shownViews() {
  const s = useStore.getState()
  const have: Record<string, { key: string; level: Quality; version: number; paint?: { key: string; level: Quality } }> = {}
  if (freshViews) {
    freshViews = false
    return have
  }
  for (const id of s.visible) {
    const v = s.views[id]
    if (!v?.key || !v.level) continue
    const p = s.paints[id]
    have[id] = { key: v.key, level: v.level, version: v.version,
      ...(p?.source === 'field' && p.key && p.level && p.viewVersion === v.version ? { paint: { key: p.key, level: p.level } } : {}) }
  }
  return have
}

/** Start a run with the current graph. Export always runs at final quality, writing every Export Mesh
 *  node or, with `only`, just those. A preview runs at `level`, or, refining, at Draft (the next
 *  qualities follow while nothing changes), else at the chosen quality. */
export async function startRun(mode: 'preview' | 'export', level?: Quality, only?: string[]) {
  const s = useStore.getState()
  if (s.inactive) return   // (another tab uses the engine: Use this tab first)
  if (mode === 'export' && 'Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission().catch(() => {}) // to say when a long export is done
  }
  // slices cover the scene (plus a margin) on the chosen plane
  const [lo, hi] = s.sceneBounds
  const pad = lo.map((l, k) => (hi[k] - l) * 0.05 + 1)
  const a = 'xyz'.indexOf(s.slice.axis)
  const finest = QUALITIES.indexOf(s.quality)
  // an engine that sends no tag back cannot refine (it was not restarted since an update): the chosen quality
  const refine = s.refine && !(s.run && s.run.tag === undefined)
  const quality = mode === 'export' ? 'final'
    : QUALITIES[Math.min(finest, QUALITIES.indexOf(level ?? (refine ? 'draft' : s.quality)))]
  const tag: RunTag = { version: s.graphVersion, client: CLIENT, ...(level ? { refining: true } : {}) }
  if (mode === 'preview') s.setRequested({ version: s.graphVersion, at: Date.now() })
  try {
    await postJSON('/api/run', {
      graph: s.recipe(),
      mode,
      quality,
      visible: s.visible,
      label: mode === 'export' ? exportLabel() : 'preview',
      tag,
      client: CLIENT,
      ...(mode === 'export' && only ? { only } : {}),
      have: mode === 'preview' ? shownViews() : {},
      slice: {
        mode: s.slice.mode,
        axis: s.slice.axis,
        position: s.slice.position ?? (lo[a] + hi[a]) / 2,
        lo: lo.map((l, k) => l - pad[k]),
        hi: hi.map((h, k) => h + pad[k]),
      },
    })
  } catch (err) {
    s.setRequested(null)
    s.addLog(`Could not start: ${(err as Error).message}`, 'error')
  }
}

/** Ask the engine for the part lo..hi of a shown body at `voxel` mm (Refine, zoomed in). False when
 *  it is busy with something else (it is asked again when that is done). */
export async function requestDetail(node: string, view: ViewMesh, lo: number[], hi: number[], voxel: number) {
  const s = useStore.getState()
  if (s.inactive) return false
  try {
    const r = await postJSON<{ run?: number | null; busy?: boolean }>('/api/detail', {
      node, version: view.version, lo, hi, voxel, tag: { version: s.graphVersion, client: CLIENT } satisfies RunTag,
      client: CLIENT })
    return !r.busy
  } catch {
    return false  // the view changed meanwhile, or it gets no detail: nothing to tell
  }
}

/** A packed mesh (server.pack_meshes): uint32 vertex and triangle counts, float32 positions, uint32 indices, then,
 *  when the engine made them (views of implicit bodies: view_ready's `normals` is "i16x3"), the field's normals as
 *  int16 x 3, 6 bytes a vertex. Told by the length, so a view served newer than announced is read right; anything
 *  else after the indices is left alone (no normals: they are worked out from the triangles). */
export function parseMesh(buf: ArrayBuffer) {
  const [nv, nf] = new Uint32Array(buf, 0, 2)
  const positions = new Float32Array(buf, 8, nv * 3)
  const indices = new Uint32Array(buf, 8 + nv * 12, nf * 3)
  const at = 8 + nv * 12 + nf * 12
  const normals = nv > 0 && buf.byteLength === at + nv * 6 ? new Int16Array(buf, at, nv * 3) : undefined
  return { nf, positions, indices, normals }
}

async function fetchDetail(e: { node: string; version: number; view_version: number; lo: number[]; hi: number[];
  voxel: number }) {
  const s = useStore.getState()
  const view = s.views[e.node]
  if (!view?.key || view.version !== e.view_version) return
  const res = await fetch(`/api/view/${encodeURIComponent(`detail:${e.node}`)}`)
  if (!res.ok) return
  const { nf, positions, indices, normals } = parseMesh(await res.arrayBuffer())
  const lo = e.lo as [number, number, number], hi = e.hi as [number, number, number]
  useStore.getState().setDetail(e.node, { version: Number(res.headers.get('X-View-Version') ?? e.version),
    viewVersion: e.view_version, key: view.key, lo, hi, voxel: e.voxel,
    mesh: { version: e.version, kind: view.kind, faces: nf, positions, indices, normals, bounds: [lo, hi] } })
}

function boundsOf(positions: Float32Array): ViewMesh['bounds'] {
  const lo: [number, number, number] = [Infinity, Infinity, Infinity]
  const hi: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k]
      if (v < lo[k]) lo[k] = v
      if (v > hi[k]) hi[k] = v
    }
  }
  return [lo, hi]
}

async function fetchView(e: { node: string; version: number; kind: string; key?: string; level?: Quality; exact?: boolean;
  voxel?: number | null; floor?: number | null; parts?: number[] | null; faces?: number; run?: number;
  timing?: EngineTiming | null }) {
  const timing = viewArrived(e)   // (window.__viewTiming; the 3D view's View ▸ View timing shows it)
  const t0 = performance.now()
  const res = await fetch(`/api/view/${encodeURIComponent(e.node)}`)
  if (!res.ok) {
    viewFetched(timing, null, { download_ms: performance.now() - t0 })
    return
  }
  const buf = await res.arrayBuffer()
  const t1 = performance.now()
  const { nf, positions, indices, normals } = parseMesh(buf)
  const served = Number(res.headers.get('X-View-Version') ?? e.version)
  // what it is made of belongs to the version announced; a newer one served instead is announced itself
  const own = served === e.version
  const view: ViewMesh = { version: served, kind: e.kind, faces: nf, positions, indices, normals,
    bounds: boundsOf(positions),
    ...(own ? { key: e.key, level: e.level, exact: e.exact, voxel: e.voxel, floor: e.floor, parts: e.parts } : {}) }
  viewFetched(timing, view, { download_ms: t1 - t0, bytes: buf.byteLength, parse_ms: performance.now() - t1 })
  useStore.getState().setView(e.node, view)
  if (useStore.getState().views[e.node] !== view) timing.skipped = true   // (a newer one is shown)
}

/** Items of lists being made (view_part), fetched and waiting to go to the store together: one update and
 *  redraw every PIECE_BATCH ms, not one per item (a Draft makes the lamp's 144 petals in ~1 s). */
const PIECE_BATCH = 150
let pieceBatch: Parameters<ReturnType<typeof useStore.getState>['addPieces']>[0] = []
let pieceTimer: ReturnType<typeof setTimeout> | null = null

async function fetchPiece(e: { node: string; index: number; total: number; version: number; kind: string }) {
  const res = await fetch(`/api/view/${encodeURIComponent(`part:${e.index}:${e.node}`)}`)
  if (!res.ok) return  // the whole view is there already
  const { nf, positions, indices, normals } = parseMesh(await res.arrayBuffer())
  pieceBatch.push({ node: e.node, index: e.index, total: e.total,
    mesh: { version: e.version, kind: e.kind, faces: nf, positions, indices, normals, bounds: boundsOf(positions) } })
  pieceTimer ??= setTimeout(() => {
    const batch = pieceBatch
    pieceBatch = []
    pieceTimer = null
    useStore.getState().addPieces(batch)
  }, PIECE_BATCH)
}

async function fetchSlice(e: { node: string; version: number; axis: Axis; position: number;
  rect: [number, number, number, number]; min: number; max: number }) {
  const res = await fetch(`/api/view/${encodeURIComponent(`slice:${e.node}`)}`)
  if (!res.ok) return
  const buf = await res.arrayBuffer()
  const [width, height] = new Uint32Array(buf, 0, 2)
  const values = new Float32Array(buf, 8, width * height)
  const version = Number(res.headers.get('X-View-Version') ?? e.version)
  useStore.getState().setSlice(e.node, {
    version, axis: e.axis, position: e.position, rect: e.rect, min: e.min, max: e.max, width, height, values,
  })
}

/** A desktop notification when an export or an optimization finishes (they can take minutes). */
function notifyExport(e: { ok: boolean; label: string; files?: Record<string, string>; error?: string }) {
  const optimizing = e.label.startsWith('optimize ')
  const files = Object.values(e.files ?? {}).filter(Boolean)
  const body = e.ok ? (optimizing ? 'The result is shown in the 3D view' : files.length ? files.join('\n') : 'Nothing was written')
    : (e.error ?? 'Failed')
  const what = optimizing ? 'Optimization' : 'Export'
  document.title = `${e.ok ? '✓' : '✗'} Design Engine`
  setTimeout(() => { document.title = 'Design Engine' }, 15000)
  if ('Notification' in window && Notification.permission === 'granted') {
    try { new Notification(`${what} ${e.ok ? 'finished' : 'failed'}: ${e.label}`, { body }) } catch { /* ignore */ }
  }
}

let connecting = false
/** The engine process this page talks to (its `hello`): another one after a reconnect is a restarted engine. */
let engineId: string | null = null

/** The engine was restarted while this page stayed open: it numbers runs and views from 1 again, knows
 *  none of this page's views and may have new nodes. Everything is made again, the node library read again. */
async function engineRestarted() {
  freshViews = true
  useStore.getState().engineRestarted()
  try {
    useStore.getState().setCatalog(await api.catalog())
  } catch { /* the next restart tries again */ }
}

/** Keep a WebSocket open to the engine (reconnecting if the server restarts). */
export function connectEvents() {
  if (connecting) return // React dev mode runs startup effects twice
  connecting = true
  let retry = 500
  let loaded = true   // (a freshly loaded page is the tab in use; one reconnecting after a restart is not)
  const open = () => {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
    ws.onopen = () => {
      retry = 500
      useStore.getState().setConnected(true)
      api.queue().then((q) => useStore.getState().setQueue(q)).catch(() => {})
      if (loaded) claimEngine(false)
      loaded = false
    }
    ws.onmessage = (msg) => {
      const e = JSON.parse(msg.data)
      if (e.type === 'hello') {
        if (engineId && e.engine !== engineId) engineRestarted()
        engineId = e.engine
        return
      }
      useStore.getState().handleEvent(e)
      const st = useStore.getState()
      // (another tab's run, or an older one: its views are not this tab's)
      if (st.inactive || (e.run !== undefined && e.run !== st.run?.id)) return
      if (e.type === 'view_ready' && useStore.getState().visible.includes(e.node)) {
        loading(fetchView(e))
      }
      if (e.type === 'view_part' && useStore.getState().visible.includes(e.node)) {
        fetchPiece(e)
      }
      if (e.type === 'slice_ready' && useStore.getState().visible.includes(e.node)) {
        loading(fetchSlice(e))
      }
      if (e.type === 'detail_ready' && useStore.getState().visible.includes(e.node)) {
        loading(fetchDetail(e))
      }
      if (e.type === 'paint_ready' && e.source === 'result') {
        if (useStore.getState().visible.includes(e.node)) fetchPaint(e.node, e, 'result')
      } else if (e.type === 'paint_ready' && useStore.getState().slice.mode === 'surface') {
        fetchPaint(e.node, e, 'field')
      }
      if (e.type === 'run_done' && e.label && e.label !== 'preview' && e.label !== 'detail' && !e.cancelled) notifyExport(e)
    }
    ws.onclose = () => {
      useStore.getState().setConnected(false)
      setTimeout(open, retry)
      retry = Math.min(retry * 2, 5000)
    }
  }
  open()
}
