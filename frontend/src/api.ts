import { isClip, useStore, type Clip } from './store'
import type { Axis, Catalog, FaceData, Paint, Quality, QueueState, Recipe, RunTag, ViewMesh } from './types'
import { QUALITIES } from './types'

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
  if (!s.visible.includes(node)) s.toggleVisible(node)
  try {
    await postJSON('/api/optimize', { graph: s.recipe(), node, quality: s.quality, visible: useStore.getState().visible,
      tag: { version: s.graphVersion } satisfies RunTag })
  } catch (err) {
    s.addLog(`Could not start: ${(err as Error).message}`, 'error')
  }
}

/** What the 3D view shows now, for a preview: the engine keeps a view it would make the same at the
 *  run's quality or coarser (refining after a change then only recomputes what changed). */
function shownViews() {
  const s = useStore.getState()
  const have: Record<string, { key: string; level: Quality; version: number; paint?: { key: string; level: Quality } }> = {}
  for (const id of s.visible) {
    const v = s.views[id]
    if (!v?.key || !v.level) continue
    const p = s.paints[id]
    have[id] = { key: v.key, level: v.level, version: v.version,
      ...(p?.source === 'field' && p.key && p.level && p.viewVersion === v.version ? { paint: { key: p.key, level: p.level } } : {}) }
  }
  return have
}

/** Start a run with the current graph. Export always runs at final quality. A preview runs at
 *  `level`, or, refining, at Draft (the next qualities follow while nothing changes), else at the
 *  chosen quality. */
export async function startRun(mode: 'preview' | 'export', level?: Quality) {
  const s = useStore.getState()
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
  const tag: RunTag = { version: s.graphVersion, ...(level ? { refining: true } : {}) }
  if (mode === 'preview') s.setRequested({ version: s.graphVersion, at: Date.now() })
  try {
    await postJSON('/api/run', {
      graph: s.recipe(),
      mode,
      quality,
      visible: s.visible,
      label: mode === 'export' ? exportLabel() : 'preview',
      tag,
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
  try {
    const r = await postJSON<{ run?: number | null; busy?: boolean }>('/api/detail', {
      node, version: view.version, lo, hi, voxel, tag: { version: s.graphVersion } satisfies RunTag })
    return !r.busy
  } catch {
    return false  // the view changed meanwhile, or it gets no detail: nothing to tell
  }
}

async function fetchDetail(e: { node: string; version: number; view_version: number; lo: number[]; hi: number[];
  voxel: number }) {
  const s = useStore.getState()
  const view = s.views[e.node]
  if (!view?.key || view.version !== e.view_version) return
  const res = await fetch(`/api/view/${encodeURIComponent(`detail:${e.node}`)}`)
  if (!res.ok) return
  const buf = await res.arrayBuffer()
  const [nv, nf] = new Uint32Array(buf, 0, 2)
  const positions = new Float32Array(buf, 8, nv * 3)
  const indices = new Uint32Array(buf, 8 + nv * 12, nf * 3)
  const lo = e.lo as [number, number, number], hi = e.hi as [number, number, number]
  useStore.getState().setDetail(e.node, { version: Number(res.headers.get('X-View-Version') ?? e.version),
    viewVersion: e.view_version, key: view.key, lo, hi, voxel: e.voxel,
    mesh: { version: e.version, kind: view.kind, faces: nf, positions, indices, bounds: [lo, hi] } })
}

async function fetchView(e: { node: string; version: number; kind: string; key?: string; level?: Quality; exact?: boolean;
  voxel?: number | null; floor?: number | null }) {
  const res = await fetch(`/api/view/${encodeURIComponent(e.node)}`)
  if (!res.ok) return
  const buf = await res.arrayBuffer()
  const [nv, nf] = new Uint32Array(buf, 0, 2)
  const positions = new Float32Array(buf, 8, nv * 3)
  const indices = new Uint32Array(buf, 8 + nv * 12, nf * 3)
  const served = Number(res.headers.get('X-View-Version') ?? e.version)
  const lo: [number, number, number] = [Infinity, Infinity, Infinity]
  const hi: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k]
      if (v < lo[k]) lo[k] = v
      if (v > hi[k]) hi[k] = v
    }
  }
  // what it is made of belongs to the version announced; a newer one served instead is announced itself
  const own = served === e.version
  useStore.getState().setView(e.node, { version: served, kind: e.kind, faces: nf, positions, indices, bounds: [lo, hi],
    ...(own ? { key: e.key, level: e.level, exact: e.exact, voxel: e.voxel, floor: e.floor } : {}) })
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

/** Keep a WebSocket open to the engine (reconnecting if the server restarts). */
export function connectEvents() {
  if (connecting) return // React dev mode runs startup effects twice
  connecting = true
  let retry = 500
  const open = () => {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
    ws.onopen = () => {
      retry = 500
      useStore.getState().setConnected(true)
      api.queue().then((q) => useStore.getState().setQueue(q)).catch(() => {})
    }
    ws.onmessage = (msg) => {
      const e = JSON.parse(msg.data)
      useStore.getState().handleEvent(e)
      if (e.type === 'view_ready' && useStore.getState().visible.includes(e.node)) {
        loading(fetchView(e))
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
