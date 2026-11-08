import { useEffect, useState } from 'react'
import { create } from 'zustand'
import { api } from '../api'
import { useStore } from '../store'

interface Request {
  mode: 'open' | 'save'
  title: string
  exts: string[]
  initial?: string
  resolve: (path: string | null) => void
}

interface BrowserState {
  req: Request | null
  open: (r: Omit<Request, 'resolve'>) => Promise<string | null>
  close: (path: string | null) => void
}

const isAbsolute = (p: string) => /^[a-zA-Z]:[\\/]/.test(p) || p.startsWith('\\\\') || p.startsWith('/')
const dirname = (p: string) => p.replace(/[\\/][^\\/]*$/, '')
const basename = (p: string) => p.split(/[\\/]/).pop() ?? ''
const join = (dir: string, name: string) => (/[\\/]$/.test(dir) ? dir + name : `${dir}\\${name}`)

function lastDir(key: string) {
  try { return localStorage.getItem(`browse:${key}`) ?? '' } catch { return '' }
}
function rememberDir(key: string, dir: string) {
  try { localStorage.setItem(`browse:${key}`, dir) } catch { /* storage unavailable */ }
}

/** Where a dialog starts: the folder and name of `initial` (a relative path is in the project folder), else
 *  the folder last used for these file types, else where the editor saves them (a recipe in recipes\). */
function startOf(r: Omit<Request, 'resolve'>) {
  const root = useStore.getState().catalog?.root ?? ''
  let start = r.initial ?? ''
  if (start && !isAbsolute(start)) start = join(root, start.replace(/\//g, '\\'))
  const saves = r.exts.includes('.json') && root ? join(root, 'recipes') : root
  return { path: start, dir: start ? dirname(start) : lastDir(r.exts.join(',')) || saves, name: start ? basename(start) : '' }
}

/** A path saved without one of the wanted extensions gets the first. */
function withExt(path: string, r: Omit<Request, 'resolve'>) {
  return r.mode === 'save' && r.exts.length && !r.exts.some((e) => path.toLowerCase().endsWith(e)) ? path + r.exts[0] : path
}

let systemOff = false // the engine said it cannot show Windows' dialog: the editor's own from now on

/** Windows' own Open / Save dialog, shown by the engine on this PC (engine/native_dialog.py): the path, null
 *  when cancelled, undefined to show the editor's own. localStorage fileDialog = "editor" keeps the editor's
 *  own (the browser pane's test tab: Windows' dialog would open on the desktop). */
async function systemDialog(r: Omit<Request, 'resolve'>): Promise<string | null | undefined> {
  let editorOnly = false
  try { editorOnly = localStorage.getItem('fileDialog') === 'editor' } catch { /* storage unavailable */ }
  if (systemOff || editorOnly) return undefined
  const { dir, name } = startOf(r)
  let answer: Awaited<ReturnType<typeof api.fileDialog>>
  try {
    answer = await api.fileDialog({ mode: r.mode, title: r.title, exts: r.exts, dir, name })
  } catch {
    return undefined // an engine from before /api/file_dialog, or none
  }
  if (answer.busy) {
    useStore.getState().addLog('A file dialog is already open: finish that one first.', 'warning')
    return null
  }
  if (answer.unavailable !== undefined) {
    systemOff = true
    useStore.getState().addLog(`Windows' file dialog is not available, using the editor's own: ${answer.unavailable}`, 'warning')
    return undefined
  }
  if (!answer.path) return null
  const path = withExt(answer.path, r)
  rememberDir(r.exts.join(','), dirname(path))
  return path
}

/** Promise-based file dialog: Windows' own, or the editor's own backed by the server's file system (paths
 *  stay real paths either way). */
export const useBrowser = create<BrowserState>((set, get) => ({
  req: null,
  open: async (r) => {
    const chosen = await systemDialog(r)
    if (chosen !== undefined) return chosen
    get().req?.resolve(null)
    return new Promise((resolve) => set({ req: { ...r, resolve } }))
  },
  close: (path) => {
    get().req?.resolve(path)
    set({ req: null })
  },
}))

/** The editor's own dialog, when Windows' is not available. */
export function FileBrowser() {
  const req = useBrowser((s) => s.req)
  const close = useBrowser((s) => s.close)
  const [dir, setDir] = useState('')
  const [pathText, setPathText] = useState('')
  const [listing, setListing] = useState<Awaited<ReturnType<typeof api.browse>> | null>(null)
  const [selected, setSelected] = useState<string>('')
  const [filename, setFilename] = useState('')
  const [error, setError] = useState('')
  const key = req?.exts.join(',') ?? ''

  useEffect(() => {
    if (!req) return
    const start = startOf(req)
    setDir(start.dir)
    setSelected(start.path && req.mode === 'open' ? start.path : '')
    setFilename(start.name)
  }, [req])

  useEffect(() => {
    if (!req) return
    let current = true // a folder that answers late must not replace the one opened after it
    setError('')
    api.browse(dir, req.exts)
      .then((l) => { if (current) { setListing(l); setPathText(l.path) } })
      .catch((e: Error) => { if (current) setError(e.message) })
    return () => { current = false }
  }, [dir, req])

  useEffect(() => {
    if (!req) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [req, close])

  if (!req) return null

  const go = (folder: string) => { setDir(folder); setSelected('') }
  const choose = (path: string) => {
    const p = withExt(path, req)
    rememberDir(key, dirname(p))
    close(p)
  }
  const finish = () => {
    const entry = listing?.entries.find((e) => e.path === selected)
    if (entry?.dir && req.mode === 'open') return go(entry.path)
    const path = req.mode === 'save' ? (filename ? join(listing?.path ?? dir, filename) : '') : selected
    if (path) choose(path)
  }

  return (
    <div className="modal-backdrop">
      <div className="modal browser">
        <div className="modal-title">{req.title}</div>
        <div className="browser-path">
          <button onClick={() => listing?.parent !== null && go(listing?.parent ?? '')} title="Up one folder"
            disabled={listing?.parent === null}>↑</button>
          <input value={pathText} onChange={(e) => setPathText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key !== 'Enter') return
              // A pasted file path selects that file; a folder path opens it.
              if (req.exts.some((x) => pathText.toLowerCase().endsWith(x))) {
                if (req.mode === 'open') choose(pathText)
                else { go(dirname(pathText)); setFilename(basename(pathText)) }
              } else go(pathText)
            }}
            placeholder="Folder or full file path (Enter)" />
        </div>
        <div className="browser-list">
          {error && <div className="browser-error">{error}</div>}
          {listing?.entries.map((e) => (
            <div key={e.path}
              className={`browser-entry ${e.dir ? 'dir' : 'file'} ${selected === e.path ? 'selected' : ''}`}
              onClick={() => {
                setSelected(e.path)
                if (!e.dir) setFilename(e.name)
              }}
              onDoubleClick={() => {
                if (e.dir) return go(e.path)
                choose(req.mode === 'save' ? join(listing.path, e.name) : e.path)
              }}>
              <span className="icon">{e.dir ? '📁' : '📄'}</span>{e.name}
            </div>
          ))}
          {listing && listing.entries.length === 0 && <div className="browser-empty">No matching files</div>}
        </div>
        <div className="modal-footer">
          {req.mode === 'save' && (
            <input className="browser-filename" value={filename} placeholder={`name${req.exts[0] ?? ''}`}
              onChange={(e) => setFilename(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && finish()} autoFocus />
          )}
          <span className="spacer" />
          <span className="hint">Double-click a folder to open it · {req.exts.join(' ')}</span>
          <button onClick={() => close(null)}>Cancel</button>
          <button className="primary" onClick={finish} disabled={req.mode === 'save' ? !filename : !selected}>
            {req.mode === 'save' ? 'Save' : 'Open'}
          </button>
        </div>
      </div>
    </div>
  )
}
