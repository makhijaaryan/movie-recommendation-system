export const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d))
export const fmtInt = (v) => (v == null ? '–' : Number(v).toLocaleString('en-US'))
export const pct = (v, d = 1) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`)
export const compact = (v) =>
  v == null ? '–' : Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(v)
