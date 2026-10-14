import { useEffect, useRef, useState } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import { api, connectEvents, startRun } from './api'
import { useStore } from './store'
import { FileBrowser } from './components/FileBrowser'
import { LogPanel } from './components/LogPanel'
import { NodeEditor } from './components/NodeEditor'
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
            <NodeEditor paletteSignal={paletteSignal} />
            <LogPanel />
          </div>
          <div className="divider" onMouseDown={startDrag} />
          <div className="viewport-pane"><Viewport /></div>
        </div>
        <RecoveryBanner />
        <FileBrowser />
      </div>
    </ReactFlowProvider>
  )
}
