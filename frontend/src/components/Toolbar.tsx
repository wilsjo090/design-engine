import { useEffect, useState } from 'react'
import { create } from 'zustand'
import { api, getJSON, postJSON, startRun } from '../api'
import { useStore } from '../store'
import { useBrowser } from './FileBrowser'
import type { Quality, Recipe } from '../types'

const STARTER: Recipe = {
  version: 1,
  nodes: {
    import: { type: 'input.import_step', inputs: { path: '' }, pos: [0, 0] },
    part: { type: 'implicit.from_cad', inputs: { cad: { link: 'import.bodies' } }, pos: [320, 0] },
    mesh: { type: 'mesh.from_implicit', inputs: { body: { link: 'part.out' } }, pos: [640, 0], visible: true },
    export: { type: 'output.export', inputs: { mesh: { link: 'mesh.mesh' }, path: 'samples/out.stl' }, pos: [960, 0] },
  },
}

/** Unsaved changes found from an earlier session (the editor or the PC stopped before saving). */
export const useRecovery = create<{ offer: { path: string | null; time: number; recipe: Recipe } | null }>(() => ({ offer: null }))

export async function openRecipe(path: string) {
  const s = useStore.getState()
  try {
    s.loadRecipe(await api.readRecipe(path), path)
    try { localStorage.setItem('lastRecipe', path) } catch { /* ignore */ }
  } catch (err) {
    s.addLog(`Could not open ${path}: ${(err as Error).message}`, 'error')
    return
  }
  // changes kept by the autosave that never made it into the file?
  try {
    const saved = await api.autosave(path)
    const now = useStore.getState().recipe()
    if (saved.recipe && saved.time && JSON.stringify(saved.recipe) !== JSON.stringify(now)) {
      useRecovery.setState({ offer: { path, time: saved.time, recipe: saved.recipe } })
    } else {
      useRecovery.setState({ offer: null })
    }
  } catch { /* no autosave */ }
}

export async function saveRecipe(saveAs = false) {
  const s = useStore.getState()
  if (s.editing) s.finishGroupEdit()
  let path = s.recipePath
  const before = path
  if (saveAs || !path) {
    path = await useBrowser.getState().open({
      mode: 'save', title: 'Save recipe', exts: ['.json'],
      initial: path ?? `${s.catalog?.root ?? ''}\\recipes\\new_recipe.json`,
    })
    if (!path) return
  }
  try {
    const saved = await api.saveRecipe(path, useStore.getState().recipe())
    s.markSaved(saved.path)
    api.dropAutosave(saved.path)
    if (before !== saved.path) api.dropAutosave(before)
    try { localStorage.setItem('lastRecipe', saved.path) } catch { /* ignore */ }
    s.addLog(`Saved ${saved.path}`, 'success')
  } catch (err) {
    s.addLog(`Save failed: ${(err as Error).message}`, 'error')
  }
}

const confirmDiscard = () => {
  const s = useStore.getState()
  if (!s.dirty) return true
  if (!confirm('Discard unsaved changes?')) return false
  api.dropAutosave(s.recipePath)
  return true
}

/** Keeps unsaved changes on the server every few seconds, so a crash does not lose them. */
export function useAutosave() {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null
    const unsubscribe = useStore.subscribe((s, prev) => {
      const changed = s.nodes !== prev.nodes || s.edges !== prev.edges || s.groups !== prev.groups ||
        s.selections !== prev.selections || s.visible !== prev.visible
      if (!changed || !s.dirty) return
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => {
        const now = useStore.getState()
        if (now.dirty) api.putAutosave(now.recipePath, now.recipe()).catch(() => {})
      }, 2000)
    })
    return () => { unsubscribe(); if (timer) clearTimeout(timer) }
  }, [])
}

export function RecoveryBanner() {
  const offer = useRecovery((s) => s.offer)
  if (!offer) return null
  const name = offer.path ? offer.path.split(/[\\/]/).pop()!.replace(/\.json$/i, '') : 'a new recipe'
  const when = new Date(offer.time * 1000)
  const sameDay = when.toDateString() === new Date().toDateString()
  return (
    <div className="recovery-banner">
      <span>Unsaved changes to <b>{name}</b> from {sameDay ? '' : `${when.toLocaleDateString()} `}
        {when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} were kept (the editor closed before saving).</span>
      <button className="primary" onClick={() => {
        useStore.getState().loadRecipe(offer.recipe, offer.path, true)
        useStore.getState().addLog('Restored the unsaved changes: save to keep them', 'success')
        useRecovery.setState({ offer: null })
      }}>Restore</button>
      <button onClick={() => { api.dropAutosave(offer.path); useRecovery.setState({ offer: null }) }}>Discard</button>
    </div>
  )
}

function QueueButton() {
  const queue = useStore((s) => s.queue)
  const [open, setOpen] = useState(false)
  const waiting = queue?.exports.length ?? 0
  if (!queue || (!waiting && !queue.current)) return null
  return (
    <span className="queue">
      <button className={open ? 'on' : ''} onClick={() => setOpen((o) => !o)}
        title="Exports run one after another; previews wait until they are done">
        Exports: {queue.current ? 1 : 0} running{waiting ? `, ${waiting} waiting` : ''}
      </button>
      {open && (
        <div className="queue-menu" onMouseLeave={() => setOpen(false)}>
          {queue.current && <div className="queue-row"><span className="spinner" /> {queue.current.label}</div>}
          {queue.exports.map((j) => (
            <div className="queue-row" key={j.id}>
              <span>{j.label}</span>
              <button className="small-btn" title="Remove from the queue"
                onClick={() => api.removeQueued(j.id).then((q) => useStore.getState().setQueue(q))}>×</button>
            </div>
          ))}
          {queue.preview_waiting && <div className="queue-row muted">The preview updates after the exports</div>}
        </div>
      )}
    </span>
  )
}

function RunStatus() {
  const run = useStore((s) => s.run)
  const status = useStore((s) => s.status)
  const specs = useStore((s) => s.specs)
  const nodes = useStore((s) => s.nodes)
  const connected = useStore((s) => s.connected)
  const [, tick] = useState(0)
  useEffect(() => {
    if (!run?.running) return
    const t = setInterval(() => tick((n) => n + 1), 250)
    return () => clearInterval(t)
  }, [run?.running])

  if (!connected) return <span className="run-status error">● Engine offline: is python app.py running?</span>
  if (!run) return <span className="run-status">Ready</span>
  if (!run.running) return <span className={`run-status ${run.message === 'Failed' ? 'error' : ''}`}>{run.message}</span>
  const active = Object.entries(status).find(([, st]) => st.state === 'running')
  const title = active ? specs[nodes.find((n) => n.id === active[0])?.data.typeId ?? '']?.title ?? active[0] : ''
  const progress = active?.[1].progress ? ` ${active[1].progress[0]}/${active[1].progress[1]}` : ''
  const secs = ((Date.now() - run.startedAt) / 1000).toFixed(1)
  return (
    <span className="run-status running">
      <span className="spinner" />
      {run.mode === 'export' ? 'Exporting' : 'Updating'}{title ? `: ${title}${progress}` : '…'} · {secs}s
    </span>
  )
}

/** GPU on/off. Only shown when the engine found a CUDA GPU. */
function GpuToggle() {
  const connected = useStore((s) => s.connected)
  const [settings, setSettings] = useState<{ gpu: boolean; gpu_available: boolean } | null>(null)
  useEffect(() => {
    if (connected) getJSON<typeof settings>('/api/settings').then(setSettings).catch(() => {})
  }, [connected])
  if (!settings?.gpu_available) return null
  return (
    <button className={settings.gpu ? 'on' : ''}
      title={settings.gpu ? 'Computing on the graphics card (click for CPU)' : 'Computing on the CPU (click to use the graphics card)'}
      onClick={async () => {
        const next = await postJSON<typeof settings>('/api/settings', { gpu: !settings.gpu })
        setSettings(next)
        useStore.getState().addLog(next!.gpu ? 'Computing on the GPU' : 'Computing on the CPU')
        startRun('preview')
      }}>
      GPU
    </button>
  )
}

/** Engine memory (all its processes) against its limit, refreshed every few seconds. */
function MemoryMeter() {
  const connected = useStore((s) => s.connected)
  const [mem, setMem] = useState<{
    used: number; limit: number; free: number; ssd_cache: number; gpu_free: number; gpu_total: number
  } | null>(null)
  useEffect(() => {
    if (!connected) return
    let alive = true
    const poll = () => getJSON<typeof mem>('/api/memory').then((m) => alive && setMem(m)).catch(() => {})
    poll()
    const t = setInterval(poll, 3000)
    return () => { alive = false; clearInterval(t) }
  }, [connected])
  if (!mem || !connected) return null
  const gb = (b: number) => (b / 2 ** 30).toFixed(1)
  const share = mem.used / mem.limit
  return (
    <span className={`memory-meter ${share > 0.85 ? 'high' : share > 0.6 ? 'mid' : ''}`}
      title={`Engine memory ${gb(mem.used)} GB of its ${gb(mem.limit)} GB limit · ${gb(mem.free)} GB free on the PC · ` +
        `SSD cache ${gb(mem.ssd_cache)} GB` +
        (mem.gpu_total ? ` · GPU memory ${gb(mem.gpu_total - mem.gpu_free)} of ${gb(mem.gpu_total)} GB in use (all programs)` : '')}>
      <span className="memory-bar"><span style={{ width: `${Math.min(100, share * 100)}%` }} /></span>
      {gb(mem.used)} / {gb(mem.limit)} GB
    </span>
  )
}

export function Toolbar({ onAddNode }: { onAddNode: () => void }) {
  const recipePath = useStore((s) => s.recipePath)
  const dirty = useStore((s) => s.dirty)
  const quality = useStore((s) => s.quality)
  const setQuality = useStore((s) => s.setQuality)
  const autoRun = useStore((s) => s.autoRun)
  const setAutoRun = useStore((s) => s.setAutoRun)
  const running = useStore((s) => s.run?.running ?? false)
  const [recipes, setRecipes] = useState<{ name: string; path: string }[]>([])

  useEffect(() => { api.recipes().then(setRecipes).catch(() => {}) }, [recipePath])

  const name = recipePath ? recipePath.split(/[\\/]/).pop()!.replace(/\.json$/i, '') : 'untitled'

  return (
    <div className="toolbar">
      <span className="brand">Design Engine</span>
      <div className="tb-group">
        <select className="recipe-select" value={recipes.find((r) => r.path === recipePath)?.path ?? ''}
          onChange={(e) => e.target.value && confirmDiscard() && openRecipe(e.target.value)}>
          <option value="">{recipes.some((r) => r.path === recipePath) ? '' : name}</option>
          {recipes.map((r) => <option key={r.path} value={r.path}>{r.name}</option>)}
        </select>
        <span className={`dirty ${dirty ? 'on' : ''}`} title={dirty ? 'Unsaved changes (kept by the autosave until you save)' : 'Saved'}>●</span>
        <button title="New recipe" onClick={() => {
          if (!confirmDiscard()) return
          useStore.getState().loadRecipe(STARTER, null)
        }}>New</button>
        <button title="Open a recipe file" onClick={async () => {
          if (!confirmDiscard()) return
          const p = await useBrowser.getState().open({ mode: 'open', title: 'Open recipe', exts: ['.json'],
            initial: recipePath ?? undefined })
          if (p) openRecipe(p)
        }}>Open…</button>
        <button title="Save (Ctrl+S)" onClick={() => saveRecipe(false)}>Save</button>
        <button title="Save as a new file" onClick={() => saveRecipe(true)}>Save as…</button>
      </div>
      <div className="tb-group">
        <button title="Add a node (Tab or right-click on the canvas)" onClick={onAddNode}>+ Node</button>
      </div>
      <div className="tb-group">
        <div className="segmented" title="Preview resolution. Draft is 4x coarser than the recipe, Medium 2x.">
          {(['draft', 'medium', 'final'] as Quality[]).map((q) => (
            <button key={q} className={quality === q ? 'on' : ''} onClick={() => setQuality(q)}>
              {q[0].toUpperCase() + q.slice(1)}
            </button>
          ))}
        </div>
        <GpuToggle />
        <label className="auto" title="Update the preview automatically after every change">
          <input type="checkbox" checked={autoRun} onChange={(e) => setAutoRun(e.target.checked)} /> Auto
        </label>
        {running
          ? <button className="danger" onClick={() => api.cancel()} title="Stop what is running now (queued exports still follow)">■ Cancel</button>
          : <button title="Update the preview (Ctrl+Enter)" onClick={() => startRun('preview')}>▶ Update</button>}
        <button className="primary"
          title="Run at full resolution and write all export files. While something else runs it waits in the queue, and you can keep working."
          onClick={() => startRun('export')}>⤓ Export</button>
        <QueueButton />
      </div>
      <span className="spacer" />
      <RunStatus />
      <MemoryMeter />
    </div>
  )
}
