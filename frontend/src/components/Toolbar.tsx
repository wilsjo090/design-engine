import { type RefObject, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { create } from 'zustand'
import { api, getJSON, postJSON, startRun, stopRun } from '../api'
import { useStore } from '../store'
import { useBrowser } from './FileBrowser'
import type { Quality, Recipe } from '../types'
import { QUALITIES, QUALITY_LABEL } from '../types'

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
  await offerRecovery(path)
}

/** Offer the changes the autosave kept for `path` (null: a new recipe) that never made it into a file, when
 *  they differ from what is open now. */
async function offerRecovery(path: string | null) {
  try {
    const saved = await api.autosave(path)
    const now = useStore.getState().recipe()
    useRecovery.setState({ offer: saved.recipe && saved.time && JSON.stringify(saved.recipe) !== JSON.stringify(now)
      ? { path, time: saved.time, recipe: saved.recipe } : null })
  } catch {
    useRecovery.setState({ offer: null })
  }
}

/** Whether the autosave may write `path`'s slot: not while the banner offers what it holds back (the first
 *  edit after opening overwrote it, and the banner's copy in this page was all that was left). */
const mayAutosave = (path: string | null) => useRecovery.getState().offer?.path !== path

function autosaveNow() {
  const now = useStore.getState()
  if (now.dirty && mayAutosave(now.recipePath)) api.putAutosave(now.recipePath, now.recipe()).catch(() => {})
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
  if (mayAutosave(s.recipePath)) api.dropAutosave(s.recipePath)   // (else it holds the changes still offered)
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
      timer = setTimeout(autosaveNow, 2000)
    })
    return () => { unsubscribe(); if (timer) clearTimeout(timer) }
  }, [])
}

export function RecoveryBanner() {
  const offer = useRecovery((s) => s.offer)
  const edited = useStore((s) => s.dirty)
  if (!offer) return null
  const name = offer.path ? offer.path.split(/[\\/]/).pop()!.replace(/\.json$/i, '') : 'a new recipe'
  const when = new Date(offer.time * 1000)
  const sameDay = when.toDateString() === new Date().toDateString()
  const time = `${sameDay ? '' : `${when.toLocaleDateString()} `}${when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
  return (
    <div className="recovery-banner">
      <span>Unsaved changes to <b>{name}</b> from {time} were kept (the editor closed before saving).
        {edited && ' Your edits since are kept once you choose.'}</span>
      <button className="primary" onClick={() => {
        useStore.getState().loadRecipe(offer.recipe, offer.path, true)
        useStore.getState().addLog('Restored the unsaved changes: save to keep them', 'success')
        useRecovery.setState({ offer: null })
      }}>Restore</button>
      <button onClick={() => {
        if (!confirm(`Throw away the changes to ${name} from ${time}? They cannot be brought back.`)) return
        api.dropAutosave(offer.path)
        useRecovery.setState({ offer: null })
        autosaveNow()   // (edits made while it was offered)
      }}>Discard</button>
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
        title="Exports and optimizations run one after another; previews wait until they are done">
        Jobs: {queue.current ? 1 : 0} running{waiting ? `, ${waiting} waiting` : ''}
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
          {queue.preview_waiting && <div className="queue-row muted">The preview updates after these jobs</div>}
        </div>
      )}
    </span>
  )
}

/** What the engine and the 3D view are doing: the run (preview, refining step, export, optimization)
 *  with its node or later step, what follows it, and changes not shown yet. */
function RunStatus() {
  const run = useStore((s) => s.run)
  const status = useStore((s) => s.status)
  const specs = useStore((s) => s.specs)
  const nodes = useStore((s) => s.nodes)
  const connected = useStore((s) => s.connected)
  const queue = useStore((s) => s.queue)
  const requested = useStore((s) => s.requested)
  const loadingViews = useStore((s) => s.loadingViews)
  const refine = useStore((s) => s.refine)
  const target = useStore((s) => s.quality)
  const stale = useStore((s) => !s.autoRun && !!s.run?.tag && s.run.tag.version !== s.graphVersion)
  const [, tick] = useState(0)
  useEffect(() => {
    if (!run?.running) return
    const t = setInterval(() => tick((n) => n + 1), 250)
    return () => clearInterval(t)
  }, [run?.running])

  // one line, cut short with … when the window is narrow (the whole text in its tooltip)
  const line = (cls: string, text: string, spin = false, title = text) =>
    <span className={`run-status ${cls}`} title={title}>{spin && <span className="spinner" />}<span className="run-text">{text}</span></span>
  if (!connected) return line('error', '● Engine offline: is python app.py running?')
  const level = run?.quality ? QUALITY_LABEL[run.quality] : ''
  if (!run?.running) {
    if (loadingViews) return line('running', 'Loading the view…', true)
    if (stale) return line('stale', 'Changes not shown yet: ▶ Update', false, 'Auto is off')
    if (!run) return line('', 'Ready')
    // what the view shows (bodies kept from a finer run count as that); null: nothing depends on the quality
    const shown = run.shown === undefined ? level : run.shown ? QUALITY_LABEL[run.shown] : ''
    if (run.next) return line('', `${shown} shown · refining to ${QUALITY_LABEL[target]}…`)
    // refined as far as the quality chosen in the toolbar (it is remembered): say so when a finer one exists
    const ceiling = refine && run.mode === 'preview' && !!run.shown && run.shown !== 'final' &&
      QUALITIES.indexOf(run.shown) >= QUALITIES.indexOf(target)
    const text = run.mode === 'detail' && run.voxel
      ? `${shown ? `${shown} · ` : ''}on screen ${Number(run.voxel.toPrecision(2))} mm · ${run.message ?? ''}`
      : `${run.mode === 'preview' && shown ? `${shown} · ` : ''}${run.message ?? ''}` +
        (ceiling ? ` · stops at ${QUALITY_LABEL[target]}: pick Final to refine further` : '')
    return line(run.message === 'Failed' ? 'error' : ceiling ? 'stale' : '', text, false,
      ceiling ? `${text}\nRefine goes up to the quality chosen in the toolbar (${QUALITY_LABEL[target]})` : text)
  }
  const active = Object.entries(status).find(([, st]) => st.state === 'running')
  let detail = run.phase ?? ''
  if (active) {
    const [id, st] = active
    const title = specs[nodes.find((n) => n.id === id)?.data.typeId ?? '']?.title ?? id
    const named = st.inner ? `${title} › ${st.inner}` : title
    const progress = st.progress ? ` ${st.step ? `${st.step} ` : ''}${st.progress[0]}/${st.progress[1]}` : ''
    detail = `${st.view && run.mode !== 'detail' ? `meshing ${named} for the view` : named}${progress}`
  }
  const verb = run.mode === 'export' ? 'Exporting' : run.mode === 'optimize' ? 'Optimizing'
    : run.mode === 'detail' ? 'Sharpening what is on screen'
    : run.tag?.refining ? `Refining to ${level}` : refine && level ? `Updating (${level})` : 'Updating'
  // a change waits while the engine stops the run before it (a step that cannot be interrupted)
  const stopping = (run.mode === 'preview' || run.mode === 'detail') && requested && Date.now() - requested.at > 300
  const after = run.mode !== 'preview' && queue?.preview_waiting ? ' · the preview follows' : ''
  const secs = ((Date.now() - run.startedAt) / 1000).toFixed(1)
  const text = (run.stopping ? `Stopping${detail ? ` (${detail})` : ''}…`
    : stopping ? `Stopping the last update to show your change${detail ? ` (${detail})` : ''}…`
    : `${verb}${detail ? `: ${detail}` : '…'}`) + ` · ${secs}s${after}`
  return line('running', text, true, run.stopping ? `${text}\nIt ends at the next point where the step can stop`
    : stopping ? `${text}\nThe step running now cannot be interrupted` : text)
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
    used: number; limit: number; free: number; commit_room?: number; ssd_cache: number; gpu_free: number; gpu_total: number
  } | null>(null)
  useEffect(() => {
    if (!connected) return
    let alive = true
    const poll = () => getJSON<typeof mem>('/api/memory').then((m) => alive && setMem(m)).catch(() => {})
    poll()
    const t = setInterval(poll, 3000)
    return () => { alive = false; clearInterval(t) }
  }, [connected])
  if (!mem || !connected) return <span className="memory-meter" />  // (kept, so the toolbar's fit sees it fill)
  const gb = (b: number) => (b / 2 ** 30).toFixed(1)
  const share = mem.used / mem.limit
  return (
    <span className={`memory-meter ${share > 0.85 ? 'high' : share > 0.6 ? 'mid' : ''}`}
      title={`Engine memory ${gb(mem.used)} GB of its ${gb(mem.limit)} GB limit · ${gb(mem.free)} GB of RAM free on the PC` +
        (mem.commit_room != null ? ` · the PC can give out ${gb(mem.commit_room)} GB more (RAM and page file)` : '') +
        ` · SSD cache ${gb(mem.ssd_cache)} GB` +
        (mem.gpu_total ? ` · GPU memory ${gb(mem.gpu_total - mem.gpu_free)} of ${gb(mem.gpu_total)} GB in use (all programs)` : '')}>
      <span className="memory-bar"><span style={{ width: `${Math.min(100, share * 100)}%` }} /></span>
      <span className="memory-text">{gb(mem.used)} / {gb(mem.limit)} GB</span>
    </span>
  )
}

export function Toolbar({ onAddNode }: { onAddNode: () => void }) {
  const recipePath = useStore((s) => s.recipePath)
  const dirty = useStore((s) => s.dirty)
  const quality = useStore((s) => s.quality)
  const setQuality = useStore((s) => s.setQuality)
  const refine = useStore((s) => s.refine)
  const setRefine = useStore((s) => s.setRefine)
  const autoRun = useStore((s) => s.autoRun)
  const setAutoRun = useStore((s) => s.setAutoRun)
  const running = useStore((s) => s.run?.running ?? false)
  const stopping = useStore((s) => !!s.run?.stopping)
  const [recipes, setRecipes] = useState<{ name: string; path: string }[]>([])
  const bar = useRef<HTMLDivElement>(null)
  useToolbarFit(bar)

  useEffect(() => { api.recipes().then(setRecipes).catch(() => {}) }, [recipePath])

  const name = recipePath ? recipePath.split(/[\\/]/).pop()!.replace(/\.json$/i, '') : 'untitled'

  return (
    <div className="toolbar" ref={bar}>
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
          offerRecovery(null)   // (a new recipe that was never saved)
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
        <button title="Add a node (Tab or right-click on the canvas)" onClick={onAddNode}>+<span className="tb-label"> Node</span></button>
      </div>
      <div className="tb-group">
        <div className="segmented" title={'Preview resolution. Draft is 4x coarser than the recipe, Medium 2x; Final is ' +
          'what you export.' + (refine ? ' With Refine: the finest quality the view is refined to.' : '')}>
          {QUALITIES.map((q: Quality) => (
            <button key={q} className={quality === q ? 'on' : ''} onClick={() => setQuality(q)}>
              <span className="tb-long">{QUALITY_LABEL[q]}</span><span className="tb-short">{QUALITY_LABEL[q][0]}</span>
            </button>
          ))}
        </div>
        <GpuToggle />
        <label className="auto" title="Update the preview automatically after every change">
          <input type="checkbox" checked={autoRun} onChange={(e) => setAutoRun(e.target.checked)} /> Auto
        </label>
        <label className="auto" title={'After a change, show a quick Draft first, then refine the view step by step up to ' +
          'the chosen quality while nothing changes. Bodies that did not change keep their finer view.'}>
          <input type="checkbox" checked={refine} onChange={(e) => setRefine(e.target.checked)} /> Refine
        </label>
        {running
          ? <button className="danger" onClick={() => stopRun()} disabled={stopping}
              title={stopping ? 'Stopping: the run ends at its next stopping point' : 'Stop what is running now (queued exports still follow)'}>
              ■<span className="tb-label">{stopping ? ' Stopping…' : ' Cancel'}</span></button>
          : <button title="Update the preview (Ctrl+Enter)" onClick={() => startRun('preview')}>▶<span className="tb-label"> Update</span></button>}
        <button className="primary"
          title="Run at full resolution and write all export files. While something else runs it waits in the queue, and you can keep working."
          onClick={() => startRun('export')}>⤓<span className="tb-label"> Export</span></button>
        <QueueButton />
      </div>
      <span className="spacer" />
      <div className="tb-status">
        <RunStatus />
        <MemoryMeter />
      </div>
    </div>
  )
}

/** Room the status bar needs before the toolbar makes itself smaller (px). */
const STATUS_ROOM = 240

/** Fit the toolbar to the window, like the node library's tabs: the brand and the memory figures go
 *  first, then the action buttons' words (their icons stay), then the quality words (their first
 *  letters stay); when the status bar still has too little room, it gets a row of its own. It never
 *  disappears. */
function useToolbarFit(bar: RefObject<HTMLDivElement>) {
  useLayoutEffect(() => {
    const el = bar.current
    if (!el) return
    // the status is all that shrinks (its text is cut short), so its room is its width plus the
    // spacer's, whatever it says now
    const room = () => (el.querySelector<HTMLElement>('.run-status')?.offsetWidth ?? 0) +
      (el.querySelector<HTMLElement>('.spacer')?.offsetWidth ?? 0)
    const fit = () => {
      for (const level of ['0', '1', '2', '3', '4']) {
        el.dataset.fit = level
        if (el.scrollWidth <= el.clientWidth + 1 && room() >= STATUS_ROOM) break
      }
    }
    fit()
    // the window: a frame later, so a fit never resizes the toolbar while its size is being reported
    let frame = 0
    const ro = new ResizeObserver(() => { cancelAnimationFrame(frame); frame = requestAnimationFrame(fit) })
    ro.observe(el)
    // what is in it (GPU found, a job queued, Cancel for Update, the memory reading); not the status
    // text, which only gets cut shorter
    const mo = new MutationObserver((changes) => {
      if (changes.some((c) => !(c.target instanceof Element ? c.target : c.target.parentElement)?.closest('.run-status'))) fit()
    })
    mo.observe(el, { childList: true, subtree: true, characterData: true })
    void document.fonts?.ready.then(fit)
    return () => { ro.disconnect(); mo.disconnect(); cancelAnimationFrame(frame) }
  }, [bar])
}
