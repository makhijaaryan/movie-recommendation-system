import { AlertTriangle, Info, SearchX } from 'lucide-react'
import { color, short } from '../lib/methods.js'

export const Card = ({ title, subtitle, actions, children, className = '' }) => (
  <section className={`card ${className}`}>
    {(title || actions) && (
      <div className="card-head">
        <div>
          {title && <h3>{title}</h3>}
          {subtitle && <p>{subtitle}</p>}
        </div>
        {actions}
      </div>
    )}
    {children}
  </section>
)

export const Kpi = ({ icon: Icon, label, value, sub }) => (
  <div className="card kpi">
    {Icon && <span className="kpi-icon"><Icon size={18} /></span>}
    <span className="kpi-label">{label}</span>
    <span className="kpi-value num">{value}</span>
    <span className="kpi-sub">{sub}</span>
  </div>
)

export const Insight = ({ icon: Icon, title, children }) => (
  <div className="card insight">
    <span className="insight-icon"><Icon size={18} /></span>
    <div><h4>{title}</h4><p>{children}</p></div>
  </div>
)

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => {
        const [v, text] = Array.isArray(o) ? o : [o, o]
        return <button key={v} aria-pressed={value === v} onClick={() => onChange(v)}>{text}</button>
      })}
    </div>
  )
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map(({ id, label, icon: Icon }) => (
        <button key={id} role="tab" aria-selected={value === id} onClick={() => onChange(id)}>
          {Icon && <Icon size={16} />}{label}
        </button>
      ))}
    </div>
  )
}

export const Skeleton = ({ h = 220 }) => <div className="skeleton" style={{ height: h }} />

export const Notice = ({ kind = 'warn', children }) => (
  <div className={`notice ${kind === 'warn' ? '' : kind}`}>
    {kind === 'info' ? <Info size={16} /> : <AlertTriangle size={16} />}
    <div>{children}</div>
  </div>
)

export const Empty = ({ title, children }) => (
  <div className="empty"><SearchX size={28} /><div><b>{title}</b></div><div>{children}</div></div>
)

export const MethodDot = ({ method }) => (
  <span className="method-cell"><i className="swatch" style={{ background: color(method) }} />{short(method)}</span>
)

export const Legend = ({ methods }) => (
  <div className="legend">
    {methods.map((m) => <span key={m}><i className="swatch" style={{ background: color(m) }} />{short(m)}</span>)}
  </div>
)

/** Recharts tooltip with consistent styling. `format(value, name)` customises numbers. */
export function ChartTip({ active, payload, label, format = (v) => v, title }) {
  if (!active || !payload?.length) return null
  return (
    <div className="tip">
      <b>{title ? title(label, payload) : label}</b>
      {payload.filter((p) => p.value != null).map((p) => (
        <div key={p.dataKey + p.name}>
          <span style={{ color: p.color || p.stroke || p.fill }}>● {p.name}</span>
          <span className="num">{format(p.value, p.name)}</span>
        </div>
      ))}
    </div>
  )
}

const GENRE_HUE = { Action: 8, Adventure: 28, Animation: 330, Children: 340, Comedy: 45, Crime: 275, Documentary: 175,
  Drama: 225, Fantasy: 290, Horror: 355, Musical: 310, Mystery: 255, Romance: 340, 'Sci-Fi': 200, Thriller: 15,
  War: 100, Western: 35, IMAX: 210, 'Film-Noir': 240 }

export function Poster({ title, genres = [], large }) {
  const hue = GENRE_HUE[genres[0]] ?? 230
  const hue2 = (hue + 40) % 360
  return (
    <span className={`poster ${large ? 'lg' : ''}`} aria-hidden="true"
      style={{ background: `linear-gradient(150deg, hsl(${hue} 65% 52%), hsl(${hue2} 70% 38%))` }}>
      {title?.replace(/^(The|A|An) /, '')[0]?.toUpperCase()}
    </span>
  )
}
