import { Search } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { compact } from '../lib/format.js'
import { Poster } from './ui.jsx'

export default function MovieSearch({ onPick, disabled }) {
  const [q, setQ] = useState('')
  const [items, setItems] = useState([])
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [busy, setBusy] = useState(false)
  const box = useRef(null)

  useEffect(() => {
    if (q.trim().length < 2) { setItems([]); return }
    let alive = true
    setBusy(true)
    const t = setTimeout(() => {
      api.search(q).then((r) => { if (alive) { setItems(r); setActive(0); setOpen(true) } })
        .catch(() => alive && setItems([])).finally(() => alive && setBusy(false))
    }, 180)
    return () => { alive = false; clearTimeout(t) }
  }, [q])

  useEffect(() => {
    const close = (e) => { if (box.current && !box.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const pick = (m) => { onPick(m); setQ(''); setItems([]); setOpen(false) }
  const onKey = (e) => {
    if (!open || !items.length) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, items.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)) }
    else if (e.key === 'Enter') { e.preventDefault(); pick(items[active]) }
    else if (e.key === 'Escape') setOpen(false)
  }

  return (
    <div className="search-box" ref={box}>
      <Search size={17} className="lead" />
      <input className="input" placeholder="Search a movie you have seen…" value={q} disabled={disabled}
        onChange={(e) => setQ(e.target.value)} onFocus={() => items.length && setOpen(true)} onKeyDown={onKey}
        aria-label="Search movies" role="combobox" aria-expanded={open} />
      {open && (
        <div className="suggest" role="listbox">
          {items.map((m, i) => (
            <button key={m.movieId} className={i === active ? 'active' : ''} onMouseEnter={() => setActive(i)} onClick={() => pick(m)} role="option" aria-selected={i === active}>
              <Poster title={m.title} genres={m.genres} />
              <span>
                <div className="t">{m.title} <span className="faint">{m.year ?? ''}</span></div>
                <div className="m">{m.genres.slice(0, 3).join(' · ') || 'No genre'} · {compact(m.nRatings)} ratings</div>
              </span>
            </button>
          ))}
          {!items.length && !busy && q.trim().length >= 2 && <div className="muted" style={{ padding: 12 }}>No matching movies.</div>}
        </div>
      )}
    </div>
  )
}
