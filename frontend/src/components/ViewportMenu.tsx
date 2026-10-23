import { useEffect, useLayoutEffect, useRef, useState, type MutableRefObject } from 'react'
import { createPortal } from 'react-dom'
import { useStore } from '../store'
import type { MenuEntry, MenuItem, Toast } from '../types'

/** How long the mouse rests on a row before its submenu opens (or the open one closes). */
const HOVER_MS = 150
/** Gap kept between a menu and the window's edge. */
const MARGIN = 4
/** A toast stays this long after it was made. */
const TOAST_MS = 10000

type Side = 'left' | 'right'
/** One mark per menu level: the highlighted row, or none. */
type Marks = (number | undefined)[]
/** Where a panel goes: at the cursor, or beside a row of the menu it opens from. */
type Anchor = { kind: 'point'; x: number; y: number } | { kind: 'row'; key: string }

const rowKey = (level: number, index: number) => `${level}:${index}`
const isItem = (e: MenuEntry | undefined): e is MenuItem => e?.kind === 'item'
/** A row that can be clicked (a disabled one has its reason as `disabled`). */
const usable = (e: MenuEntry | undefined): e is MenuItem => isItem(e) && e.disabled === undefined
const hasSub = (e: MenuEntry | undefined): e is MenuItem => isItem(e) && !!e.items?.length

/** The entries of each open level: the top menu, then the submenu opened in each. */
function levelsOf(entries: MenuEntry[], open: number[]): MenuEntry[][] {
  const out = [entries]
  for (const i of open) {
    const e = out[out.length - 1][i]
    if (!hasSub(e)) break
    out.push(e.items!)
  }
  return out
}

/** The marks of the levels above `level`, then these values. */
function put(marks: Marks, level: number, ...values: Marks): Marks {
  const out = marks.slice(0, level)
  while (out.length < level) out.push(undefined)
  return [...out, ...values]
}

/** What the panels need from the menu that owns them. */
interface Control {
  rows: MutableRefObject<Map<string, HTMLElement>>
  /** the side each placed panel opened to (a submenu goes the same way, if it fits) */
  sides: MutableRefObject<Side[]>
  hover: (level: number, index: number) => void
  /** the mouse is inside a panel: leave a pending submenu change alone */
  hold: () => void
  /** `shift`: Shift was held (passed on to the item's `run`) */
  activate: (level: number, index: number, first?: boolean, shift?: boolean) => void
  scrolled: (level: number) => void
}

/** One menu level, a fixed box placed so that it stays inside the window. */
function MenuPanel({ level, entries, anchor, marked, ctl }: {
  level: number; entries: MenuEntry[]; anchor: Anchor; marked: number | undefined; ctl: Control
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { rows, sides } = ctl
  const ax = anchor.kind === 'point' ? anchor.x : 0
  const ay = anchor.kind === 'point' ? anchor.y : 0
  const key = anchor.kind === 'row' ? anchor.key : ''

  // Placed by hand before the first paint (measured at the corner, where nothing squeezes the rows).
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.left = '0px'
    el.style.top = '0px'
    const w = el.offsetWidth, h = el.offsetHeight
    const W = window.innerWidth, H = window.innerHeight
    const clamp = (v: number, size: number, room: number) => Math.max(MARGIN, Math.min(v, room - size - MARGIN))
    let left: number, top: number, side: Side
    const rect = key ? rows.current.get(key)?.getBoundingClientRect() : undefined
    if (rect) {
      const fitsRight = rect.right + w <= W - MARGIN
      const fitsLeft = rect.left - w >= MARGIN
      side = sides.current[level - 1] ?? 'right'
      if (side === 'right' && !fitsRight && fitsLeft) side = 'left'
      else if (side === 'left' && !fitsLeft && fitsRight) side = 'right'
      left = side === 'right' ? rect.right - 2 : rect.left - w + 2
      top = rect.top - 5
    } else {
      side = ax + w > W - MARGIN ? 'left' : 'right'
      left = side === 'right' ? ax : ax - w
      top = ay + h > H - MARGIN && ay - h >= MARGIN ? ay - h : ay
    }
    sides.current[level] = side
    el.style.left = `${clamp(left, w, W)}px`
    el.style.top = `${clamp(top, h, H)}px`
  }, [ax, ay, key, entries, level, rows, sides])

  return (
    <div ref={ref} className="vp-menu" role="menu"
      onMouseEnter={ctl.hold}
      onScroll={() => ctl.scrolled(level)}
      onMouseDown={(e) => e.preventDefault()}
      onContextMenu={(e) => { e.preventDefault(); e.stopPropagation() }}
      // events of a portal bubble through the React tree: keep them away from the 3D view's handlers
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}>
      {entries.map((e, i) => {
        if (e.kind === 'header') return <div className="vp-menu-header" key={i}>{e.text}</div>
        if (e.kind === 'separator') return <div className="vp-menu-sep" role="separator" key={i} />
        const off = e.disabled !== undefined
        return (
          <div key={i} role="menuitem" aria-disabled={off} aria-haspopup={hasSub(e) || undefined}
            ref={(el) => { if (el) ctl.rows.current.set(rowKey(level, i), el); else ctl.rows.current.delete(rowKey(level, i)) }}
            className={`vp-menu-item${marked === i ? ' active' : ''}${off ? ' disabled' : ''}`}
            title={off ? e.disabled || undefined : e.title}
            onMouseEnter={() => ctl.hover(level, i)}
            onClick={(ev) => ctl.activate(level, i, false, ev.shiftKey)}>
            <span className="vp-menu-check">{e.checked ? '✓' : ''}</span>
            <span className="vp-menu-label">{e.label}</span>
            {e.hint && <span className="vp-menu-hint">{e.hint}</span>}
            {hasSub(e) && <span className="vp-menu-arrow">▸</span>}
          </div>
        )
      })}
    </div>
  )
}

/** A right-click menu at the cursor in the 3D view, with submenus (hover, → or Enter), check marks
 *  and greyed-out rows that say why. Keys: ↑ ↓ move, → opens, ← or Esc closes a level, Enter runs. */
export function ViewportMenu({ x, y, entries, onClose }: {
  x: number; y: number; entries: MenuEntry[]; onClose: () => void
}) {
  /** the row whose submenu is open, per level */
  const [open, setOpen] = useState<number[]>([])
  const [marks, setMarks] = useState<Marks>([])
  const rows = useRef(new Map<string, HTMLElement>())
  const sides = useRef<Side[]>([])
  const timer = useRef<number>()
  const levels = levelsOf(entries, open)
  const depth = levels.length - 1

  useEffect(() => () => window.clearTimeout(timer.current), [])

  /** open the submenu of a row; from the keyboard its first usable row is highlighted */
  const openSub = (level: number, index: number, first: boolean) => {
    const item = levels[level][index]
    if (!hasSub(item)) return
    const f = first ? item.items!.findIndex(usable) : -1
    setOpen([...open.slice(0, level), index])
    setMarks(put(marks, level, index, ...(f >= 0 ? [f] : [])))
  }
  const closeDeepest = () => {
    setOpen(open.slice(0, depth - 1))
    setMarks(marks.slice(0, depth))
  }
  const activate = (level: number, index: number, first = false, shift = false) => {
    const item = levels[level][index]
    if (!usable(item)) return
    window.clearTimeout(timer.current)
    if (item.run) {
      try { item.run({ shift }) } finally { onClose() }
    } else {
      openSub(level, index, first)
    }
  }
  const hover = (level: number, index: number) => {
    window.clearTimeout(timer.current)
    const item = levels[level][index]
    const wanted = usable(item) && hasSub(item) ? index : undefined
    setMarks((m) => {
      if (open[level] !== index) return put(m, level, usable(item) ? index : undefined)
      const out = m.slice()  // its submenu stays open, with its own highlight
      out[level] = index
      return out
    })
    if (wanted === open[level]) return
    timer.current = window.setTimeout(() => {
      setOpen((o) => (wanted === undefined ? o.slice(0, level) : [...o.slice(0, level), wanted]))
      setMarks((m) => m.slice(0, level + 1))
    }, HOVER_MS)
  }

  // Keys go to the deepest open level; handled ones never reach the 3D view's own shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const list = levels[depth]
      const current = marks[depth]
      switch (e.key) {
        case 'ArrowDown':
        case 'ArrowUp': {
          const step = e.key === 'ArrowDown' ? 1 : -1
          let i = current ?? (step > 0 ? -1 : list.length)
          for (let k = 0; k < list.length; k++) {
            i = (i + step + list.length) % list.length
            if (usable(list[i])) { setMarks(put(marks, depth, i)); break }
          }
          break
        }
        case 'ArrowRight':
          if (current !== undefined && usable(list[current]) && hasSub(list[current])) openSub(depth, current, true)
          break
        case 'ArrowLeft':
          if (depth > 0) closeDeepest()
          break
        case 'Enter':
          if (current !== undefined) activate(depth, current, true, e.shiftKey)
          break
        case 'Escape':
          if (depth > 0) closeDeepest()
          else onClose()
          break
        default:
          return
      }
      window.clearTimeout(timer.current)
      e.preventDefault()
      e.stopImmediatePropagation()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  // A click, right-click or scroll outside, or leaving the window, closes the menu.
  useEffect(() => {
    const outside = (e: Event) => {
      if (!(e.target as Element | null)?.closest?.('.vp-menu')) onClose()
    }
    window.addEventListener('pointerdown', outside, true)
    window.addEventListener('wheel', outside, { capture: true, passive: true })
    window.addEventListener('blur', onClose)
    window.addEventListener('resize', onClose)
    return () => {
      window.removeEventListener('pointerdown', outside, true)
      window.removeEventListener('wheel', outside, true)
      window.removeEventListener('blur', onClose)
      window.removeEventListener('resize', onClose)
    }
  }, [onClose])

  const ctl: Control = {
    rows, sides, hover, activate,
    hold: () => window.clearTimeout(timer.current),
    scrolled: (level) => setOpen((o) => (o.length > level ? o.slice(0, level) : o)),
  }
  return createPortal(
    <>
      {levels.map((list, level) => (
        <MenuPanel key={level === 0 ? 'top' : `${level}:${open[level - 1]}`} level={level} entries={list}
          anchor={level === 0 ? { kind: 'point', x, y } : { kind: 'row', key: rowKey(level - 1, open[level - 1]) }}
          marked={marks[level]} ctl={ctl} />
      ))}
    </>,
    document.body,
  )
}

/** The pill itself; one per toast (keyed by the toast's time), so its timer and hover start afresh. */
function ToastPill({ toast }: { toast: Toast }) {
  const version = useStore((s) => s.graphVersion)
  const hideToast = useStore((s) => s.hideToast)
  const [over, setOver] = useState(false)
  const deadline = useRef(toast.at + TOAST_MS)
  const left = useRef(TOAST_MS)

  // gone TOAST_MS after it was made; while the mouse is over it the clock stands still
  useEffect(() => {
    if (over) return
    const t = window.setTimeout(hideToast, Math.max(0, deadline.current - Date.now()))
    return () => window.clearTimeout(t)
  }, [over, hideToast])

  // Undo only while nothing else has changed: later it would undo something else
  const actions = toast.actions.filter((a) => a.label !== 'Undo' || version === toast.version)
  const hide = () => { if (useStore.getState().toast === toast) hideToast() }

  return (
    <div className="vp-toast" role="status"
      onMouseEnter={() => { left.current = deadline.current - Date.now(); setOver(true) }}
      onMouseLeave={() => { deadline.current = Date.now() + Math.max(left.current, 1500); setOver(false) }}
      onContextMenu={(e) => e.stopPropagation()}>
      <span className="vp-toast-text" title={toast.detail.length ? toast.detail.join('\n') : undefined}>{toast.text}</span>
      {actions.map((a, i) => (
        <button key={i} className="vp-toast-btn" onClick={() => { a.run(); hide() }}>{a.label}</button>
      ))}
      <button className="vp-toast-btn vp-toast-close" title="Dismiss" aria-label="Dismiss" onClick={hide}>×</button>
    </div>
  )
}

/** The note at the bottom of the 3D view after an action there: what changed, with Undo. */
export function ViewportToast() {
  const toast = useStore((s) => s.toast)
  return toast ? <ToastPill key={toast.at} toast={toast} /> : null
}
