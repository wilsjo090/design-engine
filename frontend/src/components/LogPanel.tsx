import { useState } from 'react'
import { useStore } from '../store'

export function LogPanel() {
  const log = useStore((s) => s.log)
  const [open, setOpen] = useState(false)
  if (log.length === 0) return null
  const shown = open ? log.slice(-60) : log.slice(-1)
  return (
    <div className={`log-panel ${open ? 'open' : ''}`} onClick={() => setOpen((o) => !o)}
      title={open ? 'Click to collapse' : 'Click to show the full log'}>
      {shown.map((e, i) => (
        <div key={i} className={`log-line ${e.level}`}>
          <span className="log-time">{new Date(e.time).toLocaleTimeString()}</span>
          <span className="log-text">{e.text}</span>
        </div>
      ))}
    </div>
  )
}
