import { Activity, BarChart3, Image as ImageIcon, Layers, Snowflake, Users } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api.js'
import { Card, ChartTip, Insight, Notice, Segmented, Skeleton, Tabs } from '../components/ui.jsx'
import { fmt } from '../lib/format.js'
import { METHOD_ORDER, METRICS, color, metricLabel, short } from '../lib/methods.js'
import { useFetch } from '../lib/useFetch.js'

const DATASETS = [['ml_25m', 'MovieLens 25M'], ['netflix', 'Netflix Prize']]
const TABS = [
  { id: 'history', label: 'History size', icon: Users },
  { id: 'cold', label: 'Cold start', icon: Snowflake },
  { id: 'lists', label: 'List quality', icon: Layers },
  { id: 'training', label: 'Training', icon: Activity },
  { id: 'figures', label: 'Report figures', icon: ImageIcon },
]
const DEFAULT_SHOWN = ['Popularity (most rated)', 'Item-based CF', 'User-based CF', 'MF (biased)', 'One-class dual-regularized']

export default function Explore() {
  const q = new URLSearchParams(window.location.search)                   // deep links: ?tab=cold&dataset=netflix
  const [tab, setTab] = useState(TABS.some((t) => t.id === q.get('tab')) ? q.get('tab') : 'history')
  const [dataset, setDataset] = useState(q.get('dataset') === 'netflix' ? 'netflix' : 'ml_25m')
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Explore the analyses</h1>
          <p>How do the methods behave for users with little history, what kind of movies do they recommend, and how do the models train?</p>
        </div>
        {tab !== 'figures' && <Segmented label="Dataset" options={DATASETS} value={dataset} onChange={setDataset} />}
      </div>
      <Tabs tabs={TABS} value={tab} onChange={setTab} />
      {tab === 'history' && <History dataset={dataset} />}
      {tab === 'cold' && <ColdStart dataset={dataset} />}
      {tab === 'lists' && <ListQuality dataset={dataset} />}
      {tab === 'training' && <Training dataset={dataset} />}
      {tab === 'figures' && <Figures />}
    </>
  )
}

function MethodPicker({ methods, shown, setShown }) {
  const toggle = (m) => setShown((s) => (s.includes(m) ? (s.length > 1 ? s.filter((x) => x !== m) : s) : [...s, m]))
  return (
    <div className="row" style={{ gap: 6 }}>
      {methods.map((m) => (
        <button key={m} className="chip" aria-pressed={shown.includes(m)} onClick={() => toggle(m)}>
          <i className="swatch" style={{ background: color(m) }} />{short(m)}
        </button>
      ))}
    </div>
  )
}

function Lines({ data, xKey, series, height = 340, yFormat = (v) => fmt(v, 2), xLabel, yLabel, refY }) {
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 4, bottom: 24 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey={xKey} label={{ value: xLabel, position: 'insideBottom', offset: -14, fill: 'var(--muted)', fontSize: 12 }} />
          <YAxis tickFormatter={yFormat} domain={['auto', 'auto']} label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
          <Tooltip content={<ChartTip format={(v) => fmt(v, 4)} />} />
          {refY != null && <ReferenceLine y={refY} stroke="var(--faint)" strokeDasharray="5 4" />}
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={2.4} dot={data.length > 15 ? false : { r: 3.5, fill: s.color }} activeDot={{ r: 6 }}
              type="monotone" connectNulls isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

const Loading = () => <div className="stack"><Skeleton h={90} /><Skeleton h={380} /></div>

/* ------------------------------------------------------------------ history */
const BUCKETS = ['<30', '30-59', '60-119', '120-299', '300+']

function History({ dataset }) {
  const [metric, setMetric] = useState('ndcg@10')
  const [relative, setRelative] = useState(false)
  const [shown, setShown] = useState(DEFAULT_SHOWN)
  const { data, loading, error } = useFetch(() => api.analysis('history', dataset), [dataset])
  const methods = useMemo(() => (data ? METHOD_ORDER.filter((m) => data.some((r) => r.method === m)) : []), [data])
  if (error) return <Notice kind="bad">{error.message}</Notice>
  if (loading || !data) return <Loading />
  const val = (m, b, met) => data.find((r) => r.method === m && r.bucket === b && r.metric === met)?.value
  const rel = relative && metric !== 'rmse'
  const rows = BUCKETS.map((b) => {
    const row = { bucket: b }
    shown.forEach((m) => {
      const v = val(m, b, metric)
      const p = val('Popularity (most rated)', b, metric)
      row[m] = v == null ? null : rel ? v / p : v
    })
    return row
  })
  const series = shown.filter((m) => rows.some((r) => r[m] != null)).map((m) => ({ key: m, name: short(m), color: color(m) }))
  return (
    <div className="stack">
      <div className="grid three">
        <Insight icon={Users} title="More history, better predictions">RMSE falls steadily for every personalised model as users rate more movies. The first bucket (20–29 ratings) is the hardest.</Insight>
        <Insight icon={BarChart3} title="One-class leads in every bucket">Relative to popularity, one-class CF is about 2–3.5× better at every history size, and its edge is largest for light users.</Insight>
        <Insight icon={Snowflake} title="MF is weakest for light users">Rating-trained MF ranks below popularity everywhere and collapses further for short histories.</Insight>
      </div>
      <Card title="Metric by number of ratings in the user’s history"
        subtitle="Users are grouped by how many ratings they have in the training data (everyone has at least 20; true cold start is on the next tab)."
        actions={<div className="row">
          <Segmented label="Metric" options={['rmse', 'ndcg@10', 'hit@10', 'recall@10'].map((m) => [m, metricLabel(m)])} value={metric} onChange={setMetric} />
          {metric !== 'rmse' && <Segmented label="Scale" options={[[false, 'Absolute'], [true, '× popularity']]} value={relative} onChange={setRelative} />}
        </div>}>
        <MethodPicker methods={methods.filter((m) => !['Global mean'].includes(m))} shown={shown} setShown={setShown} />
        <div style={{ height: 12 }} />
        <Lines data={rows} xKey="bucket" series={series} xLabel="Ratings in training history" yLabel={rel ? 'NDCG relative to popularity' : metricLabel(metric)}
          yFormat={(v) => fmt(v, rel ? 1 : 2)} refY={rel ? 1 : undefined} />
        {metric !== 'rmse' && !rel && <p className="faint" style={{ fontSize: 12.5 }}>Absolute hit rate and NDCG rise with history for every method partly because heavy users have more held-out liked movies; use “× popularity” to compare fairly.</p>}
      </Card>
    </div>
  )
}

/* --------------------------------------------------------------- cold start */
const NS = ['1', '3', '5', '10', '20', '50', 'all']

function ColdStart({ dataset }) {
  const [metric, setMetric] = useState('ndcg@10')
  const [shown, setShown] = useState(DEFAULT_SHOWN)
  const { data, loading, error } = useFetch(() => api.analysis('coldstart', dataset), [dataset])
  const methods = useMemo(() => (data ? METHOD_ORDER.filter((m) => data.some((r) => r.method === m)) : []), [data])
  if (error) return <Notice kind="bad">{error.message}</Notice>
  if (loading || !data) return <Loading />
  const get = (m, n) => data.find((r) => r.method === m && String(r.N) === n)
  const rows = NS.map((n) => {
    const row = { n: n === 'all' ? 'all' : n }
    shown.forEach((m) => { row[m] = get(m, n)?.[metric] ?? null })
    return row
  })
  const series = shown.filter((m) => rows.some((r) => r[m] != null)).map((m) => ({ key: m, name: short(m), color: color(m) }))
  const popRef = metric === 'ndcg@10' || metric === 'hit@10' ? get('Popularity (most rated)', 'all')?.[metric] : undefined
  const fb = (m, n) => get(m, n)?.frac_fallback
  return (
    <div className="stack">
      <div className="grid three">
        <Insight icon={Snowflake} title="One rating: nothing beats popularity">With a single rating the CF models have no signal and fall back to the popularity list; one-class CF is slightly below it.</Insight>
        <Insight icon={Users} title="User CF helps first">From about 3 ratings user-based CF passes popularity. One-class CF overtakes it at roughly 5–20 ratings and keeps improving.</Insight>
        <Insight icon={Activity} title="Item CF needs more">Item-based CF is worse than popularity until about 20 ratings. Rating-trained MF never beats popularity for ranking.</Insight>
      </div>
      <Card title="Simulated cold start" subtitle="Each user reveals only their first N random ratings; the models are not retrained, new-user vectors are computed in closed form. Dashed line = popularity."
        actions={<Segmented label="Metric" options={['ndcg@10', 'hit@10', 'rmse'].map((m) => [m, metricLabel(m)])} value={metric} onChange={setMetric} />}>
        <MethodPicker methods={methods.filter((m) => !['Global mean', 'Popularity (most rated)'].includes(m))} shown={shown} setShown={setShown} />
        <div style={{ height: 12 }} />
        <Lines data={rows} xKey="n" series={series} xLabel="Ratings revealed (N)" yLabel={metricLabel(metric)} refY={popRef} />
        <p className="faint" style={{ fontSize: 12.5 }}>
          Share of users that fall back to the popularity list at N = 1: {['Item-based CF', 'User-based CF', 'One-class dual-regularized']
            .filter((m) => fb(m, '1') != null).map((m) => `${short(m)} ${Math.round(fb(m, '1') * 100)}%`).join(' · ')}.
        </p>
      </Card>
    </div>
  )
}

/* -------------------------------------------------------------- list quality */
const DIAG = {
  coverage: { label: 'Catalogue coverage', help: 'Share of all movies that appear in at least one user’s top 10. Higher = more varied, but not necessarily better.', fmt: (v) => `${(v * 100).toFixed(1)}%` },
  head_share: { label: 'Slots from the top 1% most-rated', help: 'Share of recommendation slots taken by the 1% most-rated movies. 100% = only blockbusters.', fmt: (v) => `${(v * 100).toFixed(0)}%` },
  tail_share: { label: 'Slots from the less-rated half', help: 'Share of slots filled with movies from the less-rated half of the catalogue.', fmt: (v) => `${(v * 100).toFixed(1)}%` },
  median_ratings: { label: 'Median ratings of recommended movies', help: 'How well-known the recommended movies are (median number of ratings).', fmt: (v) => Math.round(v).toLocaleString('en-US') },
  item_mean: { label: 'Average crowd rating', help: 'Mean star rating of the recommended movies.', fmt: (v) => v.toFixed(2) },
  genres: { label: 'Genres per list', help: 'Number of distinct genres in a user’s top 10 (MovieLens only).', fmt: (v) => v.toFixed(1) },
}

function ListQuality({ dataset }) {
  const [metric, setMetric] = useState('median_ratings')
  const { data, loading, error } = useFetch(() => api.analysis('diagnostics', dataset), [dataset])
  if (error) return <Notice kind="bad">{error.message}</Notice>
  if (loading || !data) return <Loading />
  const cfg = DIAG[metric]
  const rows = METHOD_ORDER.map((m) => data.find((r) => r.method === m)).filter((r) => r && r[metric] != null)
    .map((r) => ({ method: r.method, name: short(r.method), value: r[metric], ndcg: r['ndcg@10_sample'] }))
  return (
    <div className="stack">
      <div className="grid three">
        <Insight icon={Layers} title="Why MF ranks badly">Its top-10 lists are full of obscure movies (median ~1–2k ratings vs 10–40k for the others) because rarely-rated movies with a few high ratings score highest.</Insight>
        <Insight icon={BarChart3} title="One-class: mainstream yet personal">Its lists are as well-known as CF’s but far more varied than popularity’s single list.</Insight>
        <Insight icon={Activity} title="Everyone leans popular">Even the best ranker takes about half or more of its slots from the 1% most-rated movies. Observed ratings reward popularity.</Insight>
      </div>
      <Card title="What the top-10 lists look like" subtitle={cfg.help}
        actions={<select className="select" value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="Diagnostic">
          {Object.entries(DIAG).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}</select>}>
        <div className="chart">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 60, top: 4, bottom: 4 }}>
              <CartesianGrid horizontal={false} />
              <XAxis type="number" tickFormatter={cfg.fmt} />
              <YAxis type="category" dataKey="name" width={112} tick={{ fill: 'var(--text)', fontSize: 12.5 }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTip format={(v) => cfg.fmt(v)} title={(l) => l} />} />
              <Bar dataKey="value" name={cfg.label} radius={[0, 6, 6, 0]} barSize={20} isAnimationActive={false}
                label={{ position: 'right', formatter: cfg.fmt, fill: 'var(--text)', fontSize: 12, fontWeight: 600 }}>
                {rows.map((r) => <Cell key={r.method} fill={color(r.method)} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="faint" style={{ fontSize: 12.5 }}>Computed on a fixed random sample of 10,000 test users, using each method’s ranking configuration.</p>
      </Card>
    </div>
  )
}

/* ----------------------------------------------------------------- training */
function Training({ dataset }) {
  const { data, loading, error } = useFetch(() => api.analysis('curves', dataset), [dataset])
  if (error) return <Notice kind="bad">{error.message}</Notice>
  if (loading || !data) return <Loading />
  const mf = data.filter((r) => r.model === 'MF (biased)').map((r) => ({ step: r.step, train: r.train_rmse, val: r.val_rmse }))
  const oc = data.filter((r) => r.model !== 'MF (biased)').map((r) => ({ step: r.step, ndcg: r.val_ndcg10 }))
  return (
    <div className="grid two">
      <Card title="Matrix factorization: RMSE per epoch" subtitle="Validation error flattens around epoch 30 and does not rise, so regularisation keeps the model from overfitting even as training error keeps falling.">
        <Lines data={mf} xKey="step" xLabel="Epoch" yLabel="RMSE" yFormat={(v) => fmt(v, 2)}
          series={[{ key: 'train', name: 'Train (first 2M ratings)', color: 'var(--m-pop)' }, { key: 'val', name: 'Validation', color: 'var(--m-mf)' }]} />
      </Card>
      <Card title="One-class CF: validation NDCG@10 per ALS iteration" subtitle="Ranking quality converges within about 10 iterations; the final model keeps the best one.">
        <Lines data={oc} xKey="step" xLabel="ALS iteration" yLabel="NDCG@10" yFormat={(v) => fmt(v, 2)}
          series={[{ key: 'ndcg', name: 'Validation NDCG@10', color: 'var(--m-dual)' }]} />
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ figures */
function Figures() {
  const { data, loading, error } = useFetch(() => api.figures(), [])
  const [open, setOpen] = useState(null)
  if (error) return <Notice kind="bad">{error.message}</Notice>
  if (loading || !data) return <Loading />
  return (
    <>
      <p className="muted" style={{ marginBottom: 16 }}>The static figures prepared for the report. Click one to enlarge.</p>
      <div className="gallery">
        {data.map((f) => (
          <button key={f.file} className="figure" onClick={() => setOpen(f)}>
            <img src={f.url} alt={f.caption} loading="lazy" />
            <figcaption>{f.caption}</figcaption>
          </button>
        ))}
      </div>
      {open && (
        <div className="lightbox" onClick={() => setOpen(null)} role="dialog" aria-label={open.caption}>
          <img src={open.url} alt={open.caption} />
        </div>
      )}
    </>
  )
}
