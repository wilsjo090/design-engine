import { useEffect, useRef, useState } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import { api, claimEngine, connectEvents, startRun } from './api'
import { useStore } from './store'
import { FileBrowser } from './components/FileBrowser'
import { LogPanel } from './components/LogPanel'
import { NodeEditor } from './components/NodeEditor'
import { NodeLibrary } from './components/NodeLibrary'
import { RecoveryBanner, Toolbar, openRecipe, saveRecipe, useAutosave } from './components/Toolbar'
import { Viewport } from './components/Viewport'

/** Re-run the preview shortly after the last change, when Auto is on. */
function useAutoRun() {
  const graphVersion = useStore((s) => s.graphVersion)
  const autoRun = useStore((s) => s.autoRun)
  const ready = useStore((s) => s.catalog !== null && s.connected)
  useEffect(() => {
    if (!autoRun || !ready || graphVersion === 0) return
    const t = setTimeout(() => startRun('preview'), 450)
    return () => clearTimeout(t)
  }, [graphVersion, autoRun, ready])
}

/** Another tab uses the engine (an old tab after a restart, which opens a new one): this one runs
 *  nothing and shows no results until it is used here again. */
function InactiveBanner() {
  const inactive = useStore((s) => s.inactive)
  if (!inactive) return null
  return (
    <div className="recovery-banner engine-elsewhere">
      <span><b>The editor is open in another tab</b>, which uses the engine now: this tab shows no results and
        runs nothing.</span>
      <button className="primary" onClick={() => claimEngine(true)}>Use this tab</button>
    </div>
  )
}

/** A pause before each finer preview: a change made meanwhile starts from Draft again instead of
 *  waiting for a heavy run to stop. */
const REFINE_DELAY = 800
/** A preview done in this long (ms, from its run_start to its run_done as the editor saw them) was quick: the
 *  next finer one starts as soon as the graph has been left alone for REFINE_DELAY (counted from the last
 *  change, not from the end of the preview), but at least SHORT_PAUSE after it. Hex studs: Draft 0.03 s, Medium
 *  1.7 s, Final 0.55 s, and the two full pauses added 1.6 s (so not 1.5 s here: its Medium would miss it).
 *  After a slower preview the pause stays REFINE_DELAY: the next quality is heavier still, and a change made
 *  meanwhile would have to wait for it to stop. */
const QUICK_RUN = 2500
const SHORT_PAUSE = 150

/** Refine the view (Refine on): when a preview is done and nothing has changed since it was asked
 *  for, the next finer quality runs, up to the chosen one (the store sets `run.next`). Never while a number is
 *  being dragged (`valueDrag`: the pointer may rest on a slider for a moment): only once it is let go. */
function useRefine() {
  const run = useStore((s) => s.run)
  const dragging = useStore((s) => s.valueDrag)
  const changedAt = useRef(0)   // performance.now() of the last graph change
  useEffect(() => useStore.subscribe((s, prev) => {
    if (s.graphVersion !== prev.graphVersion) changedAt.current = performance.now()
  }), [])
  useEffect(() => {
    if (!run || run.running || !run.next || dragging) return
    const took = (run.endedAt ?? Date.now()) - run.startedAt
    const quiet = performance.now() - changedAt.current
    const delay = took <= QUICK_RUN ? Math.max(SHORT_PAUSE, REFINE_DELAY - quiet) : REFINE_DELAY
    const t = setTimeout(() => {
      const s = useStore.getState()
      if (s.run === run && s.refine && s.connected && !s.valueDrag && run.tag?.version === s.graphVersion) {
        startRun('preview', run.next)
      }
    }, delay)
    return () => clearTimeout(t)
  }, [run, dragging])
}

function useSplit() {
  const [split, setSplit] = useState(() => {
    try { return Number(localStorage.getItem('split')) || 0.55 } catch { return 0.55 }
  })
  const dragging = useRef(false)
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragging.current) return
      setSplit(Math.min(0.8, Math.max(0.25, e.clientX / window.innerWidth)))
    }
    const up = () => {
      if (!dragging.current) return
      dragging.current = false
      document.body.classList.remove('resizing')
      try { localStorage.setItem('split', String(split)) } catch { /* ignore */ }
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [split])
  const startDrag = () => { dragging.current = true; document.body.classList.add('resizing') }
  return { split, startDrag }
}

export default function App() {
  const catalog = useStore((s) => s.catalog)
  const [error, setError] = useState('')
  const [paletteSignal, setPaletteSignal] = useState(0)
  const { split, startDrag } = useSplit()
  useAutoRun()
  useRefine()
  useAutosave()

  useEffect(() => {
    connectEvents()
    ;(async () => {
      try {
        const cat = await api.catalog()
        useStore.getState().setCatalog(cat)
        const recipes = await api.recipes()
        let last = ''
        try { last = localStorage.getItem('lastRecipe') ?? '' } catch { /* ignore */ }
        const start = last || recipes.find((r) => r.name === 'wave_offset')?.path || recipes[0]?.path
        if (start) await openRecipe(start)
      } catch (err) {
        setError(`Cannot reach the engine: ${(err as Error).message}. Start it with: python app.py`)
      }
    })()
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey
      const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)
      if (mod && e.key === 's') { e.preventDefault(); saveRecipe(false) }
      if (mod && e.key === 'Enter') { e.preventDefault(); startRun('preview') }
      if (mod && !typing && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) useStore.getState().redo()
        else useStore.getState().undo()
      }
      if (mod && !typing && e.key.toLowerCase() === 'y') { e.preventDefault(); useStore.getState().redo() }
    }
    const onUnload = (e: BeforeUnloadEvent) => {
      if (useStore.getState().dirty) e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('beforeunload', onUnload)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('beforeunload', onUnload) }
  }, [])

  if (error) return <div className="fatal">{error}</div>
  if (!catalog) return <div className="fatal">Connecting to the engine…</div>

  return (
    <ReactFlowProvider>
      <div className="app">
        <Toolbar onAddNode={() => setPaletteSignal((n) => n + 1)} />
        <div className="main">
          <div className="editor-pane" style={{ flexBasis: `${split * 100}%` }}>
            <NodeLibrary />
            <div className="editor-body">
              <NodeEditor paletteSignal={paletteSignal} />
              <LogPanel />
            </div>
          </div>
          <div className="divider" onMouseDown={startDrag} />
          <div className="viewport-pane"><Viewport /></div>
        </div>
        <RecoveryBanner />
        <InactiveBanner />
        <FileBrowser />
      </div>
    </ReactFlowProvider>
  )
}
