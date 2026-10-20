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

/** Promise-based file dialog backed by the server's file system (paths stay real paths). */
export const useBrowser = create<BrowserState>((set, get) => ({
  req: null,
  open: (r) => new Promise((resolve) => set({ req: { ...r, resolve } })),
  close: (path) => {
    get().req?.resolve(path)
    set({ req: null })
  },
}))

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

export function FileBrowser() {
  const req = useBrowser((s) => s.req)
  const close = useBrowser((s) => s.close)
  const root = useStore((s) => s.catalog?.root ?? '')
  const [dir, setDir] = useState('')
  const [pathText, setPathText] = useState('')
  const [listing, setListing] = useState<Awaited<ReturnType<typeof api.browse>> | null>(null)
  const [selected, setSelected] = useState<string>('')
  const [filename, setFilename] = useState('')
  const [error, setError] = useState('')
  const key = req?.exts.join(',') ?? ''

  useEffect(() => {
    if (!req) return
    let start = req.initial ?? ''
    if (start && !isAbsolute(start)) start = join(root, start.replace(/\//g, '\\'))
    const startDir = start ? dirname(start) : lastDir(key) || root
    setDir(startDir)
    setSelected(start && req.mode === 'open' ? start : '')
    setFilename(start ? basename(start) : '')
  }, [req, root, key])

  useEffect(() => {
    if (!req) return
    setError('')
    api.browse(dir, req.exts)
      .then((l) => { setListing(l); setPathText(l.path) })
      .catch((e: Error) => setError(e.message))
  }, [dir, req])

  if (!req) return null

  const finish = () => {
    let path = req.mode === 'save' ? (filename ? join(listing?.path ?? dir, filename) : '') : selected
    if (!path) return
    if (req.mode === 'save' && req.exts.length && !req.exts.some((e) => path.toLowerCase().endsWith(e))) {
      path += req.exts[0]
    }
    rememberDir(key, dirname(path))
    close(path)
  }

  return (
    <div className="modal-backdrop" onMouseDown={() => close(null)}>
      <div className="modal browser" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-title">{req.title}</div>
        <div className="browser-path">
          <button onClick={() => listing?.parent !== null && setDir(listing?.parent ?? '')} title="Up one folder"
            disabled={listing?.parent === null}>↑</button>
          <input value={pathText} onChange={(e) => setPathText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key !== 'Enter') return
              // A pasted file path selects that file; a folder path opens it.
              if (req.exts.some((x) => pathText.toLowerCase().endsWith(x))) {
                if (req.mode === 'open') { rememberDir(key, dirname(pathText)); close(pathText) }
                else { setDir(dirname(pathText)); setFilename(basename(pathText)) }
              } else setDir(pathText)
            }}
            placeholder="Folder or full file path (Enter)" />
        </div>
        <div className="browser-list">
          {error && <div className="browser-error">{error}</div>}
          {listing?.entries.map((e) => (
            <div key={e.path}
              className={`browser-entry ${e.dir ? 'dir' : 'file'} ${selected === e.path ? 'selected' : ''}`}
              onClick={() => {
                if (e.dir) return setDir(e.path)
                setSelected(e.path)
                setFilename(e.name)
              }}
              onDoubleClick={() => {
                if (e.dir) return
                rememberDir(key, dirname(e.path))
                close(req.mode === 'save' ? join(listing.path, e.name) : e.path)
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
          <span className="hint">{req.exts.join(' ')}</span>
          <button onClick={() => close(null)}>Cancel</button>
          <button className="primary" onClick={finish} disabled={req.mode === 'save' ? !filename : !selected}>
            {req.mode === 'save' ? 'Save' : 'Open'}
          </button>
        </div>
      </div>
    </div>
  )
}
