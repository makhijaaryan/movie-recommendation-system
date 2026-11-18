import { ArrowRight, Film, Layers, Sparkles, Star, Users, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api.js'
import { Poster, Skeleton } from '../components/ui.jsx'
import { compact } from '../lib/format.js'
import { MODEL_KEYS } from '../lib/methods.js'

/** Retries while the backend is still loading its models (503). */
function useLive(fn, deps) {
  const [state, setState] = useState({ data: null, loading: true, offline: false })
  useEffect(() => {
    let alive = true, timer
    const go = () => fn().then((data) => alive && setState({ data, loading: false, offline: false }))
      .catch((e) => { if (!alive) return; setState({ data: null, loading: e.status === 503, offline: e.status !== 503 }); timer = setTimeout(go, 3000) })
    setState((s) => ({ ...s, loading: true }))
    go()
    return () => { alive = false; clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}

const ENGINES = [
  { id: 'popularity', tag: 'Crowd favourites', text: 'What most people watch. A solid starting point that is the same for everyone.' },
  { id: 'item_cf', tag: 'More like this', text: 'Finds movies that people tend to rate the same way as the ones you loved.' },
  { id: 'user_cf', tag: 'Taste twins', text: 'Looks for viewers whose ratings match yours and suggests what they enjoyed.' },
  { id: 'mf', tag: 'Hidden patterns', text: 'Learns hidden taste dimensions (like “dark and cerebral”) behind everyone’s ratings.' },
  { id: 'one_class', tag: 'Best all-rounder', text: 'Focuses on what you liked and learns to rank the movies you are most likely to love next.' },
]

const STEPS = [
  { icon: Star, title: 'Rate a few movies', text: 'Search for films you know, or start from a ready-made taste profile. Even two or three ratings is enough to begin.' },
  { icon: Layers, title: 'Compare five engines', text: 'Each recommendation engine reads your taste differently. See their top picks side by side and where they agree.' },
  { icon: Sparkles, title: 'Discover your next watch', text: 'Filter by genre, change the list length and refine your ratings. Suggestions update instantly.' },
]

export default function Dashboard() {
  const nav = useNavigate()
  const [genre, setGenre] = useState('')
  const genres = useLive(() => api.genres(), [])
  const popular = useLive(() => api.popular(genre || undefined, 14), [genre])
  const presets = useLive(() => api.presets(), [])
  const ds = useLive(() => api.datasets(), [])
  const ml = ds.data?.find((d) => d.key === 'ml_25m')

  const quick = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Animation', 'Thriller', 'Romance', 'Horror']
  const chips = (genres.data ?? []).filter((g) => quick.includes(g))
  const start = (state) => nav('/recommend', { state })

  return (
    <div className="stack" style={{ gap: 36 }}>
      {/* ---------------- hero ---------------- */}
      <section className="hero">
        <div className="hero-copy">
          <span className="badge primary"><Zap size={12} /> Instant, personalised picks</span>
          <h1>Find your next favourite movie</h1>
          <p>Tell us a few films you love and CineMatch suggests what to watch next, using five different recommendation engines you can compare side by side.</p>
          <div className="row" style={{ marginTop: 6 }}>
            <button className="btn primary lg" onClick={() => nav('/recommend')}>Get recommendations <ArrowRight size={17} /></button>
            <button className="btn lg" onClick={() => document.getElementById('how')?.scrollIntoView({ behavior: 'smooth' })}>How it works</button>
          </div>
          <div className="hero-stats">
            <div><b className="num">{ml ? compact(ml.items) : '13K'}</b><span>movies</span></div>
            <div><b className="num">{ml ? compact(ml.ratings) : '24M'}</b><span>ratings analysed</span></div>
            <div><b>5</b><span>recommendation engines</span></div>
            <div><b>&lt;10 ms</b><span>per recommendation</span></div>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          {(popular.data ?? []).slice(0, 8).map((m, i) => (
            <div key={m.movieId} className="hero-card" style={{ '--i': i }}>
              <Poster title={m.title} genres={m.genres} large />
              <div><b>{m.title}</b><span>★ {m.avgRating} · {m.year}</span></div>
            </div>
          ))}
          {!popular.data && <Skeleton h={300} />}
        </div>
      </section>

      {/* ---------------- sample profiles ---------------- */}
      <section>
        <div className="section-head"><div><h2>Start with a taste profile</h2><p>Not sure where to begin? Pick someone who looks like you.</p></div></div>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))' }}>
          {presets.loading && !presets.data ? [0, 1, 2, 3].map((i) => <Skeleton key={i} h={130} />) :
            (presets.data ?? []).map((p) => (
              <button key={p.id} className="profile-card" onClick={() => start({ preset: p.id })}>
                <div className="stack-posters">
                  {p.ratings.slice(0, 4).map((r) => <Poster key={r.movieId} title={r.title} genres={r.genres} />)}
                </div>
                <b>{p.name}</b><span>{p.blurb}</span>
                <em>Try this profile <ArrowRight size={14} /></em>
              </button>
            ))}
        </div>
        {presets.offline && <p className="muted" style={{ marginTop: 10 }}>Connecting to the recommendation service…</p>}
      </section>

      {/* ---------------- popular ---------------- */}
      <section>
        <div className="section-head">
          <div><h2>Popular movies</h2><p>Tap a movie to rate it and start building your profile.</p></div>
          <div className="row" style={{ gap: 6 }}>
            <button className="chip" aria-pressed={genre === ''} onClick={() => setGenre('')}>All</button>
            {chips.map((g) => <button key={g} className="chip" aria-pressed={genre === g} onClick={() => setGenre(g)}>{g}</button>)}
          </div>
        </div>
        <div className="rail">
          {popular.data ? popular.data.map((m) => (
            <button key={m.movieId} className="rail-card" onClick={() => start({ add: m })}>
              <Poster title={m.title} genres={m.genres} large />
              <span className="t">{m.title}</span>
              <span className="s">{m.year} · ★ {m.avgRating}</span>
              <span className="s faint">{compact(m.nRatings)} ratings</span>
            </button>
          )) : [0, 1, 2, 3, 4, 5].map((i) => <Skeleton key={i} h={160} />)}
        </div>
      </section>

      {/* ---------------- how it works ---------------- */}
      <section id="how">
        <div className="section-head"><div><h2>How it works</h2><p>Three steps, no account needed.</p></div></div>
        <div className="grid three">
          {STEPS.map(({ icon: Icon, title, text }, i) => (
            <div className="card step" key={title}>
              <span className="step-no">{i + 1}</span>
              <span className="insight-icon"><Icon size={18} /></span>
              <h4>{title}</h4><p className="muted">{text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---------------- engines ---------------- */}
      <section>
        <div className="section-head"><div><h2>Meet the engines</h2><p>Every engine has its own idea of “what you will like”. Compare them to see which one fits you.</p></div></div>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))' }}>
          {ENGINES.map((e) => (
            <div className="card engine" key={e.id} style={{ '--c': MODEL_KEYS[e.id].color }}>
              <span className="engine-bar" />
              <span className="badge">{e.tag}</span>
              <h4>{MODEL_KEYS[e.id].label}</h4>
              <p className="muted">{e.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---------------- call to action ---------------- */}
      <section className="cta">
        <div>
          <h2>Curious how well they really work?</h2>
          <p>See how the engines score on millions of real ratings, and explore what happens for new viewers with only a handful of ratings.</p>
        </div>
        <div className="row">
          <button className="btn" onClick={() => nav('/overview')}>See the results</button>
          <button className="btn primary" onClick={() => nav('/explore')}>Explore the analyses <ArrowRight size={16} /></button>
        </div>
      </section>
    </div>
  )
}
