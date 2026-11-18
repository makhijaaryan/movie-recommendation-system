import { useMemo, useState } from 'react'
import { Award, Gauge, Target, TrendingUp, Users, Scale, FlaskConical } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { api } from '../api.js'
import { Card, ChartTip, Insight, Kpi, Legend, MethodDot, Notice, Segmented, Skeleton } from '../components/ui.jsx'
import { fmt, fmtInt, pct } from '../lib/format.js'
import { METHOD_ORDER, METRICS, color, metricLabel, short } from '../lib/methods.js'
import { useFetch } from '../lib/useFetch.js'

const DATASETS = [['ml_25m', 'MovieLens 25M'], ['netflix', 'Netflix Prize']]
const RANK_METRICS = [['ndcg', 'NDCG'], ['hit', 'Hit rate'], ['precision', 'Precision'], ['recall', 'Recall']]

function useIndex(overview) {
  return useMemo(() => {
    if (!overview) return null
    const main = Object.fromEntries(overview.main.map((r) => [r.method, r]))
    const ci = {}
    overview.ci.forEach((r) => { ci[`${r.method}|${r.metric}`] = r })
    const seeds = Object.fromEntries(overview.seeds.map((r) => [r.method, r]))
    return { main, ci, seeds }
  }, [overview])
}

function barData(ix, metric, methods) {
  return methods
    .map((m) => ({ m, r: ix.ci[`${m}|${metric}`] }))
    .filter((x) => x.r && x.r.estimate != null)
    .map(({ m, r }) => ({ method: m, name: short(m), value: r.estimate, err: [r.estimate - r.ci_low, r.ci_high - r.estimate],
      lo: r.ci_low, hi: r.ci_high, label: fmt(r.estimate) }))
}

function MetricBars({ data, metric, height }) {
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" domain={[0, 'auto']} tickFormatter={(v) => fmt(v, 2)} />
          <YAxis type="category" dataKey="name" width={112} tick={{ fill: 'var(--text)', fontSize: 12.5 }} axisLine={false} tickLine={false} />
          <Tooltip cursor={{ fill: 'var(--surface-2)' }}
            content={<ChartTip title={(l) => l} format={(v) => fmt(v, 4)} />} />
          <Bar dataKey="value" name={metricLabel(metric)} radius={[0, 6, 6, 0]} barSize={20} isAnimationActive={false}>
            {data.map((d) => <Cell key={d.method} fill={color(d.method)} />)}
            <ErrorBar dataKey="err" direction="x" stroke="var(--text)" strokeWidth={1.3} width={5} />
            <LabelList dataKey="label" position="right" offset={12} style={{ fill: 'var(--text)', fontSize: 12, fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Overview() {
  const [dataset, setDataset] = useState(new URLSearchParams(window.location.search).get('dataset') === 'netflix' ? 'netflix' : 'ml_25m')
  const [rankMetric, setRankMetric] = useState('ndcg')
  const [k, setK] = useState(10)
  const [errMetric, setErrMetric] = useState('rmse')
  const [seedView, setSeedView] = useState(false)
  const { data, error, loading } = useFetch(() => api.overview(dataset), [dataset])
  const { data: dsInfo } = useFetch(() => api.datasets(), [])
  const ix = useIndex(data)

  if (error) return <Notice kind="bad">Could not load results: {error.message}. Is the API running on port 8000?</Notice>

  const ds = dsInfo?.find((d) => d.key === dataset)
  const rankKey = `${rankMetric}@${k}`
  const body = ix && (() => {
    const rating = METHOD_ORDER.filter((m) => ix.main[m]?.rmse != null)
    const ranking = METHOD_ORDER.filter((m) => ix.main[m]?.['ndcg@10'] != null)
    const bestRmse = rating.reduce((a, b) => (ix.main[a].rmse <= ix.main[b].rmse ? a : b))
    const bestNdcg = ranking.reduce((a, b) => (ix.main[a]['ndcg@10'] >= ix.main[b]['ndcg@10'] ? a : b))
    const pop = ix.main['Popularity (most rated)']
    const base = ix.main['Mean rating (min 50 ratings)']
    const mf = ix.main[bestRmse]
    return { rating, ranking, bestRmse, bestNdcg, pop, base, mf }
  })()

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Which recommender works best?</h1>
          <p>Nine methods compared on the same held-out ratings, with 95% bootstrap confidence intervals. Lower is better for error, higher for ranking quality.</p>
        </div>
        <Segmented label="Dataset" options={DATASETS} value={dataset} onChange={setDataset} />
      </div>

      {loading || !ix ? <div className="stack"><Skeleton h={110} /><Skeleton h={340} /><Skeleton h={340} /></div> : (
        <div className="stack">
          <div className="grid kpis">
            <Kpi icon={Gauge} label="Lowest rating error (RMSE)" value={fmt(body.mf.rmse)}
              sub={`${short(body.bestRmse)} · ${pct(1 - body.mf.rmse / body.base.rmse, 0)} below the item-mean baseline`} />
            <Kpi icon={Award} label="Best top-10 quality (NDCG@10)" value={fmt(body.bestNdcg && ix.main[body.bestNdcg]['ndcg@10'])}
              sub={`${short(body.bestNdcg)} · ${(ix.main[body.bestNdcg]['ndcg@10'] / body.pop['ndcg@10']).toFixed(1)}× popularity`} />
            <Kpi icon={Target} label="Users with a hit in top 10" value={pct(ix.main[body.bestNdcg]['hit@10'], 0)}
              sub={`${short(body.bestNdcg)} · popularity reaches ${pct(body.pop['hit@10'], 0)}`} />
            <Kpi icon={Users} label="Users evaluated" value={fmtInt(body.pop.n_users)}
              sub={ds ? `${fmtInt(ds.ratings)} ratings · ${fmtInt(ds.items)} movies` : ' '} />
          </div>

          <div className="grid three">
            <Insight icon={Scale} title="Best RMSE ≠ best recommender">
              {short(body.bestRmse)} predicts ratings best but ranks {(body.pop['ndcg@10'] / ix.main[body.bestRmse]['ndcg@10']).toFixed(1)}× <i>worse</i> than
              plain popularity: it promotes obscure movies whose few ratings are high.
            </Insight>
            <Insight icon={TrendingUp} title="Training for ranking wins">
              One-class CF learns from “liked / unknown” only and reaches NDCG@10 {fmt(ix.main[body.bestNdcg]['ndcg@10'], 2)},
              about {(ix.main[body.bestNdcg]['ndcg@10'] / ix.main['Item-based CF']['ndcg@10']).toFixed(1)}× item-based CF.
            </Insight>
            <Insight icon={FlaskConical} title="Robust, not luck">
              Rankings are identical across three random splits; intervals are narrow (±0.001–0.005). With this many users nearly any gap is detectable, so look at effect size.
            </Insight>
          </div>

          <div className="grid two">
            <Card title="Ranking quality" subtitle="How well the top-K list matches movies the user later liked (held-out rating ≥ 4). Whiskers = 95% CI."
              actions={<div className="row"><Segmented label="Metric" options={RANK_METRICS} value={rankMetric} onChange={setRankMetric} />
                <Segmented label="K" options={[[5, '@5'], [10, '@10'], [20, '@20']]} value={k} onChange={setK} /></div>}>
              <MetricBars data={barData(ix, rankKey, body.ranking)} metric={rankKey} height={350} />
            </Card>
            <Card title="Rating accuracy" subtitle="How close predicted star ratings are to the real ones (lower is better). Ranking-only models have no rating prediction."
              actions={<Segmented label="Error metric" options={[['rmse', 'RMSE'], ['mae', 'MAE']]} value={errMetric} onChange={setErrMetric} />}>
              <MetricBars data={barData(ix, errMetric, body.rating)} metric={errMetric} height={350} />
            </Card>
          </div>

          <Card title="Accuracy vs. recommendation quality"
            subtitle="Each dot is a method (whiskers = 95% CI). Moving right lowers rating error; moving up improves top-10 ranking. The dashed levels mark ranking-only methods, which do not predict ratings.">
            <TradeOff ix={ix} body={body} />
          </Card>

          <Card title="Full results" subtitle={`Test split, K = 10. Best value per column in green.${seedView ? ' Showing mean ± std over 3 random splits.' : ''}`}
            actions={<Segmented label="View" options={[[false, 'Seed 42'], [true, '3-seed mean ± std']]} value={seedView} onChange={setSeedView} />}>
            <ResultsTable ix={ix} seedView={seedView} />
          </Card>

          <Card title="Head-to-head comparisons" subtitle="Paired bootstrap on the same users: difference between two methods with its 95% confidence interval.">
            <Paired rows={data.paired} />
          </Card>
        </div>
      )}
    </>
  )
}

function TradeOff({ ix, body }) {
  const pts = body.rating.filter((m) => ix.main[m]['ndcg@10'] != null)
    .map((m) => ({ method: m, name: short(m), x: ix.main[m].rmse, y: ix.main[m]['ndcg@10'] }))
  const lines = ['One-class WRMF', 'One-class dual-regularized', 'Popularity (most rated)']
    .filter((m) => ix.main[m]?.['ndcg@10'] != null)
  const xs = pts.map((p) => p.x)
  const [x0, x1] = [Math.min(...xs) - 0.03, Math.max(...xs) + 0.03]
  return (
    <>
      <div className="chart">
        <ResponsiveContainer width="100%" height={340}>
          <ScatterChart margin={{ top: 10, right: 24, left: 4, bottom: 28 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="x" reversed domain={[x0, x1]} tickFormatter={(v) => v.toFixed(2)}
              label={{ value: 'RMSE  (better →)', position: 'insideBottom', offset: -14, fill: 'var(--muted)', fontSize: 12 }} />
            <YAxis type="number" dataKey="y" domain={[0, 'auto']} tickFormatter={(v) => v.toFixed(2)}
              label={{ value: 'NDCG@10  (better ↑)', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
            <ZAxis range={[130, 130]} />
            <Tooltip cursor={{ strokeDasharray: '4 4' }} content={({ active, payload }) => active && payload?.length ? (
              <div className="tip"><b>{payload[0].payload.name}</b>
                <div><span>RMSE</span><span className="num">{fmt(payload[0].payload.x)}</span></div>
                <div><span>NDCG@10</span><span className="num">{fmt(payload[0].payload.y)}</span></div></div>) : null} />
            {lines.map((m) => (
              <Scatter key={m} data={[{ x: x0, y: ix.main[m]['ndcg@10'] }, { x: x1, y: ix.main[m]['ndcg@10'] }]} line={{ stroke: color(m), strokeDasharray: '6 4', strokeWidth: 2 }}
                shape={() => null} legendType="none" isAnimationActive={false} />
            ))}
            <Scatter data={pts} isAnimationActive={false} shape={(p) => <circle cx={p.cx} cy={p.cy} r={8} fill={color(p.payload.method)} stroke="var(--surface)" strokeWidth={2} />} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="legend" style={{ marginTop: 6 }}>
        {pts.map((p) => <span key={p.method}><i className="swatch" style={{ background: color(p.method) }} />{p.name}</span>)}
        {lines.map((m) => <span key={m}><i className="swatch" style={{ background: color(m), height: 3, borderRadius: 0 }} />{short(m)} (ranking only)</span>)}
      </div>
    </>
  )
}

const COLS = [
  ['rmse', 'RMSE', 'lower'], ['mae', 'MAE', 'lower'], ['precision@10', 'Precision@10', 'higher'],
  ['recall@10', 'Recall@10', 'higher'], ['hit@10', 'Hit@10', 'higher'], ['ndcg@10', 'NDCG@10', 'higher'],
]

function ResultsTable({ ix, seedView }) {
  const methods = METHOD_ORDER.filter((m) => ix.main[m])
  const val = (m, c) => (seedView ? ix.seeds[m]?.[`mean_${c}`] : ix.main[m][c])
  const std = (m, c) => (seedView ? ix.seeds[m]?.[`std_${c}`] : null)
  const best = {}
  COLS.forEach(([c, , dir]) => {
    const vs = methods.map((m) => val(m, c)).filter((v) => v != null)
    best[c] = dir === 'lower' ? Math.min(...vs) : Math.max(...vs)
  })
  return (
    <div className="table-wrap">
      <table className="t">
        <thead><tr><th>Method</th>{COLS.map(([c, l]) => <th key={c} title={METRICS[c]?.help}>{l}</th>)}</tr></thead>
        <tbody>
          {methods.map((m) => (
            <tr key={m}>
              <td><MethodDot method={m} /></td>
              {COLS.map(([c]) => {
                const v = val(m, c)
                return <td key={c} className={v != null && v === best[c] ? 'best' : ''}>
                  {fmt(v)}{seedView && v != null && std(m, c) != null && <span className="faint"> ±{fmt(std(m, c), 4)}</span>}
                </td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Paired({ rows }) {
  const [metric, setMetric] = useState('ndcg@10')
  const metrics = [...new Set(rows.map((r) => r.metric))]
  const list = rows.filter((r) => r.metric === metric)
  return (
    <>
      <div style={{ marginBottom: 12 }}>
        <Segmented label="Metric" options={metrics.map((m) => [m, metricLabel(m)])} value={metric} onChange={setMetric} />
      </div>
      <div className="table-wrap">
        <table className="t">
          <thead><tr><th>A vs B</th><th>A</th><th>B</th><th>A − B</th><th>95% CI</th><th>Verdict</th></tr></thead>
          <tbody>
            {list.map((r) => {
              const lowerBetter = METRICS[r.metric]?.better === 'lower'
              const aWins = lowerBetter ? r.diff < 0 : r.diff > 0
              return (
                <tr key={`${r.a}|${r.b}`}>
                  <td><MethodDot method={r.a} /> <span className="faint">vs</span> <MethodDot method={r.b} /></td>
                  <td>{fmt(r.a_value)}</td><td>{fmt(r.b_value)}</td>
                  <td>{r.diff > 0 ? '+' : ''}{fmt(r.diff, 4)}</td>
                  <td className="faint">[{fmt(r.ci_low, 4)}, {fmt(r.ci_high, 4)}]</td>
                  <td>{r.excludes_zero ? <span className={`badge ${aWins ? 'good' : 'bad'}`}>{short(aWins ? r.a : r.b)} better</span> : <span className="badge">no clear difference</span>}</td>
                </tr>
              )
            })}
            {!list.length && <tr><td colSpan={6} className="muted">No comparisons for this metric.</td></tr>}
          </tbody>
        </table>
      </div>
    </>
  )
}
