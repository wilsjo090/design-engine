import { isClip, useStore, type Clip } from './store'
import type { Axis, Catalog, Paint, QueueState, Recipe } from './types'

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

/** Colours for a view (a value per vertex) from the engine. */
export async function fetchPaint(node: string, meta: { version: number; view_version: number; min: number; max: number;
  label: string; unit?: string; reverse?: boolean; clipped?: boolean }, source: Paint['source']) {
  const res = await fetch(`/api/view/${encodeURIComponent(`paint:${node}`)}`)
  if (!res.ok) return
  const values = new Float32Array(await res.arrayBuffer())
  const version = Number(res.headers.get('X-View-Version') ?? meta.version)
  useStore.getState().setPaint(node, { version, viewVersion: meta.view_version, values, min: meta.min, max: meta.max,
    label: meta.label, source, unit: meta.unit, reverse: meta.reverse, clipped: meta.clipped })
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
    await postJSON('/api/optimize', { graph: s.recipe(), node, quality: s.quality, visible: useStore.getState().visible })
  } catch (err) {
    s.addLog(`Could not start: ${(err as Error).message}`, 'error')
  }
}

/** Start a run with the current graph. Export always runs at final quality. */
export async function startRun(mode: 'preview' | 'export') {
  const s = useStore.getState()
  if (mode === 'export' && 'Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission().catch(() => {}) // to say when a long export is done
  }
  // slices cover the scene (plus a margin) on the chosen plane
  const [lo, hi] = s.sceneBounds
  const pad = lo.map((l, k) => (hi[k] - l) * 0.05 + 1)
  const a = 'xyz'.indexOf(s.slice.axis)
  try {
    await postJSON('/api/run', {
      graph: s.recipe(),
      mode,
      quality: mode === 'export' ? 'final' : s.quality,
      visible: s.visible,
      label: mode === 'export' ? exportLabel() : 'preview',
      slice: {
        mode: s.slice.mode,
        axis: s.slice.axis,
        position: s.slice.position ?? (lo[a] + hi[a]) / 2,
        lo: lo.map((l, k) => l - pad[k]),
        hi: hi.map((h, k) => h + pad[k]),
      },
    })
  } catch (err) {
    s.addLog(`Could not start: ${(err as Error).message}`, 'error')
  }
}

async function fetchView(node: string, version: number, kind: string) {
  const res = await fetch(`/api/view/${encodeURIComponent(node)}`)
  if (!res.ok) return
  const buf = await res.arrayBuffer()
  const [nv, nf] = new Uint32Array(buf, 0, 2)
  const positions = new Float32Array(buf, 8, nv * 3)
  const indices = new Uint32Array(buf, 8 + nv * 12, nf * 3)
  const served = Number(res.headers.get('X-View-Version') ?? version)
  const lo: [number, number, number] = [Infinity, Infinity, Infinity]
  const hi: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k]
      if (v < lo[k]) lo[k] = v
      if (v > hi[k]) hi[k] = v
    }
  }
  useStore.getState().setView(node, { version: served, kind, faces: nf, positions, indices, bounds: [lo, hi] })
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
        fetchView(e.node, e.version, e.kind)
      }
      if (e.type === 'slice_ready' && useStore.getState().visible.includes(e.node)) {
        fetchSlice(e)
      }
      if (e.type === 'paint_ready' && e.source === 'result') {
        if (useStore.getState().visible.includes(e.node)) fetchPaint(e.node, e, 'result')
      } else if (e.type === 'paint_ready' && useStore.getState().slice.mode === 'surface') {
        fetchPaint(e.node, e, 'field')
      }
      if (e.type === 'run_done' && e.label && e.label !== 'preview' && !e.cancelled) notifyExport(e)
    }
    ws.onclose = () => {
      useStore.getState().setConnected(false)
      setTimeout(open, retry)
      retry = Math.min(retry * 2, 5000)
    }
  }
  open()
}
