import { Info, Layers, ListChecks, Loader2, Sparkles, Trash2, Users, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { api } from '../api.js'
import MovieSearch from '../components/MovieSearch.jsx'
import StarRating from '../components/StarRating.jsx'
import { Card, Empty, Notice, Poster, Skeleton } from '../components/ui.jsx'
import { compact } from '../lib/format.js'
import { MODEL_KEYS } from '../lib/methods.js'
import { useFetch } from '../lib/useFetch.js'

const ALL_MODELS = Object.keys(MODEL_KEYS)
const STORE = 'cinematch.ratings.v1'

function loadSaved() {
  try { return JSON.parse(localStorage.getItem(STORE)) ?? [] } catch { return [] }
}

export default function Recommend() {
  const [rated, setRated] = useState(loadSaved)          // [{ ...card, rating }]
  const [models, setModels] = useState(ALL_MODELS)
  const [k, setK] = useState(10)
  const [genre, setGenre] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [view, setView] = useState('compare')
  const [single, setSingle] = useState('one_class')
  const [pickGenre, setPickGenre] = useState('')
  const seq = useRef(0)
  const location = useLocation()
  const navigate = useNavigate()

  const status = useFetch(() => api.status(), [])
  const ready = status.data?.ready
  const genres = useFetch(() => (ready ? api.genres() : Promise.resolve([])), [ready])
  const presets = useFetch(() => (ready ? api.presets() : Promise.resolve([])), [ready])
  const popular = useFetch(() => (ready ? api.popular(pickGenre || undefined, 18) : Promise.resolve([])), [ready, pickGenre])

  // Arrived from the dashboard with a sample profile or a tapped movie.
  useEffect(() => {
    const profile = new URLSearchParams(location.search).get('profile')   // deep link: /recommend?profile=scifi
    const st = location.state ?? (profile ? { preset: profile } : null)
    if (!st) return
    if (st.add) setRated((r) => (r.some((x) => x.movieId === st.add.movieId) ? r : [{ ...st.add, rating: 4 }, ...r]))
    else if (st.preset && presets.data?.length) {
      const p = presets.data.find((x) => x.id === st.preset)
      if (p) setRated(p.ratings.map((r) => ({ ...r })))
    } else return
    navigate('/recommend', { replace: true, state: null })
  }, [location.state, location.search, presets.data, navigate])

  useEffect(() => { try { localStorage.setItem(STORE, JSON.stringify(rated)) } catch { /* ignore */ } }, [rated])

  const ids = useMemo(() => new Set(rated.map((r) => r.movieId)), [rated])
  const add = (m, rating = 4) => setRated((r) => (r.some((x) => x.movieId === m.movieId) ? r : [{ ...m, rating }, ...r]))
  const setRating = (id, rating) => setRated((r) => r.map((x) => (x.movieId === id ? { ...x, rating } : x)))
  const remove = (id) => setRated((r) => r.filter((x) => x.movieId !== id))

  const run = useCallback(() => {
    if (!rated.length || !models.length) { setResult(null); return }
    const mine = ++seq.current
    setBusy(true); setError(null)
    api.recommend({ ratings: rated.map((r) => ({ movieId: r.movieId, rating: r.rating })), models, k, genre: genre || null })
      .then((res) => { if (mine === seq.current) setResult(res) })
      .catch((e) => { if (mine === seq.current) setError(e) })
      .finally(() => { if (mine === seq.current) setBusy(false) })
  }, [rated, models, k, genre])

  // Recompute shortly after any change (debounced) once the models are ready.
  useEffect(() => {
    if (!ready) return
    const t = setTimeout(run, 350)
    return () => clearTimeout(t)
  }, [run, ready])

  const toggleModel = (m) => setModels((cur) => (cur.includes(m) ? (cur.length > 1 ? cur.filter((x) => x !== m) : cur) : ALL_MODELS.filter((x) => cur.includes(x) || x === m)))
  const shown = view === 'single' ? [single] : models.filter((m) => result?.models[m])
  const consensus = useMemo(() => buildConsensus(result, models), [result, models])

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Get recommendations</h1>
          <p>Rate a few movies you know. Five different models score every other movie for you instantly, without retraining, so you can see how their taste differs.</p>
        </div>
      </div>

      {status.error && <Notice kind="bad">The API is not reachable. Start it with <code>uvicorn src.api:app --port 8000</code>.</Notice>}
      {status.data && !ready && <Notice kind="info">{status.data.error ? `Models failed to load: ${status.data.error}` : 'Models are still loading. This page will work in a moment.'}
        {' '}<button className="btn sm" onClick={status.data.error ? undefined : () => window.location.reload()}>Refresh</button></Notice>}

      <div className="rec-layout" style={{ marginTop: 16 }}>
        {/* ---------------- left: your ratings ---------------- */}
        <aside className="rec-side">
          <Card title="Your ratings" subtitle={rated.length ? `${rated.length} rated · ${rated.filter((r) => r.rating >= 4).length} liked (≥ 4 stars)` : 'Add at least one movie to begin.'}
            actions={rated.length > 0 && <button className="btn ghost sm" onClick={() => setRated([])}><Trash2 size={14} /> Clear</button>}>
            <div className="stack" style={{ gap: 12 }}>
              <MovieSearch onPick={(m) => add(m)} disabled={!ready} />
              {rated.length === 0 ? (
                <div className="muted" style={{ fontSize: 13.5 }}>Search above, pick from the popular movies, or start from a sample profile.</div>
              ) : (
                <div className="rated-list">
                  {rated.map((m) => (
                    <div className="rated-item" key={m.movieId}>
                      <div className="row" style={{ gap: 10, flexWrap: 'nowrap' }}>
                        <Poster title={m.title} genres={m.genres} />
                        <div style={{ minWidth: 0 }}>
                          <div className="t">{m.title} <span className="y">{m.year}</span></div>
                          <StarRating value={m.rating} onChange={(v) => setRating(m.movieId, v)} size={16} />
                        </div>
                      </div>
                      <button className="icon-btn" style={{ width: 30, height: 30 }} onClick={() => remove(m.movieId)} aria-label={`Remove ${m.title}`}><X size={15} /></button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>

          <Card title="Sample profiles" subtitle="One click to load a ready-made taste profile.">
            {presets.loading ? <Skeleton h={90} /> : (
              <div className="preset-grid">
                {(presets.data ?? []).map((p) => (
                  <button className="preset" key={p.id} onClick={() => setRated(p.ratings.map((r) => ({ ...r })))}>
                    <b>{p.name}</b><span>{p.ratings.length} movies · {p.blurb}</span>
                  </button>
                ))}
              </div>
            )}
          </Card>

          <Card title="Popular movies" subtitle="Tap to add (starts at 4 stars)."
            actions={<select className="select" value={pickGenre} onChange={(e) => setPickGenre(e.target.value)} aria-label="Genre">
              <option value="">All genres</option>{(genres.data ?? []).map((g) => <option key={g}>{g}</option>)}</select>}>
            <div className="pick-grid">
              {(popular.data ?? []).map((m) => (
                <button className="pick" key={m.movieId} disabled={ids.has(m.movieId)} onClick={() => add(m)}>
                  <Poster title={m.title} genres={m.genres} />
                  <span className="t">{m.title}</span>
                </button>
              ))}
            </div>
          </Card>
        </aside>

        {/* ---------------- right: recommendations ---------------- */}
        <div className="stack" style={{ minWidth: 0 }}>
          <Card>
            <div className="row between">
              <div className="row">
                {ALL_MODELS.map((m) => (
                  <button key={m} className="chip" aria-pressed={models.includes(m)} onClick={() => toggleModel(m)} title={MODEL_KEYS[m].about}>
                    <i className="swatch" style={{ background: MODEL_KEYS[m].color }} />{MODEL_KEYS[m].label}
                  </button>
                ))}
              </div>
              <div className="row">
                <select className="select" value={genre} onChange={(e) => setGenre(e.target.value)} aria-label="Restrict to genre">
                  <option value="">Any genre</option>{(genres.data ?? []).map((g) => <option key={g}>{g}</option>)}
                </select>
                <select className="select" value={k} onChange={(e) => setK(+e.target.value)} aria-label="List length">
                  {[5, 10, 15, 20].map((n) => <option key={n} value={n}>Top {n}</option>)}
                </select>
                <div className="segmented" role="group" aria-label="Layout">
                  <button aria-pressed={view === 'compare'} onClick={() => setView('compare')}><Layers size={14} style={{ verticalAlign: '-2px' }} /> Compare</button>
                  <button aria-pressed={view === 'single'} onClick={() => setView('single')}><ListChecks size={14} style={{ verticalAlign: '-2px' }} /> One model</button>
                </div>
              </div>
            </div>
            {view === 'single' && (
              <div className="row" style={{ marginTop: 12 }}>
                <span className="muted" style={{ fontSize: 13 }}>Show:</span>
                {models.map((m) => <button key={m} className="chip" aria-pressed={single === m} onClick={() => setSingle(m)}>{MODEL_KEYS[m].label}</button>)}
              </div>
            )}
          </Card>

          {!rated.length ? (
            <Card><Empty title="Nothing to recommend yet">Rate a few movies on the left, or load a sample profile, and the recommendations appear here.</Empty></Card>
          ) : (
            <>
              {error && <Notice kind="bad">{error.message}</Notice>}
              {consensus.length > 0 && view === 'compare' && (
                <Card title="Consensus picks" subtitle="Movies recommended by several models at once, the safest suggestions.">
                  <div className="consensus-row">
                    {consensus.map((c) => (
                      <div className="consensus-card" key={c.movieId}>
                        <Poster title={c.title} genres={c.genres} />
                        <div>
                          <div className="t">{c.title} <span className="faint">{c.year}</span></div>
                          <div className="vote-dots" title={`${c.votes.length} of ${models.length} models`}>
                            {c.votes.map((m) => <i key={m} style={{ background: MODEL_KEYS[m].color }} />)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
              <div className="model-cols" style={{ opacity: busy ? 0.6 : 1, transition: 'opacity 0.15s' }}>
                {!result && busy && models.map((m) => <Skeleton key={m} h={420} />)}
                {result && shown.map((m) => <ModelColumn key={m} id={m} data={result.models[m]} nRated={result.nRated} nLiked={result.nLiked} />)}
              </div>
              {busy && <div className="muted row" style={{ justifyContent: 'center' }}><Loader2 size={16} className="spin" /> Updating…</div>}
            </>
          )}
        </div>
      </div>
    </>
  )
}

function buildConsensus(result, models) {
  if (!result) return []
  const m = new Map()
  models.forEach((id) => (result.models[id]?.items ?? []).forEach((it) => {
    const e = m.get(it.movieId) ?? { ...it, votes: [], best: 99 }
    e.votes.push(id); e.best = Math.min(e.best, it.rank)
    m.set(it.movieId, e)
  }))
  const need = Math.min(3, models.length)
  return [...m.values()].filter((e) => e.votes.length >= need && models.length > 1)
    .sort((a, b) => b.votes.length - a.votes.length || a.best - b.best).slice(0, 8)
}

function ModelColumn({ id, data, nRated, nLiked }) {
  const meta = MODEL_KEYS[id]
  return (
    <section className="model-col">
      <div className="model-col-head" style={{ '--c': meta.color }}>
        <div className="row between"><b>{meta.label}</b><span className="faint num" style={{ fontSize: 12 }}>{data.ms} ms</span></div>
        <div className="faint" style={{ fontSize: 12, marginTop: 2 }}>{meta.about}</div>
        {data.fallback && (
          <div className="notice" style={{ marginTop: 8, padding: '8px 10px', fontSize: 12.5 }}>
            <Info size={14} /><div>{id === 'one_class' && nLiked === 0
              ? 'No movie rated 4★ or higher, so there is nothing to learn from. Showing popular movies.'
              : 'Not enough signal in these ratings for this model. Showing popular movies instead.'}</div>
          </div>
        )}
      </div>
      {data.items.map((it) => (
        <div className="reco" key={it.movieId}>
          <span className="rank num">{it.rank}</span>
          <div>
            <div className="t">{it.title} <span className="y">{it.year}</span></div>
            <div className="meta">
              {it.genres.slice(0, 3).map((g) => <span className="genre-tag" key={g}>{g}</span>)}
              <span title="Number of ratings in the training data">{compact(it.nRatings)} ratings</span>
              <span>★ {it.avgRating}</span>
            </div>
            {id !== 'popularity' && !data.fallback && it.because && <div className="because" title="Item-item similarity to a movie you rated 4 stars or more">Similar to {it.because.title}, which you liked</div>}
          </div>
        </div>
      ))}
    </section>
  )
}
