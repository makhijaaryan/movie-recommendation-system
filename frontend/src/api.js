// Thin client for the FastAPI backend (proxied by Vite in development).
async function request(path, options) {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail || detail } catch { /* not JSON */ }
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    err.status = res.status
    throw err
  }
  return res.json()
}

const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v != null && v !== '')).toString()

export const api = {
  status: () => request('/api/status'),
  datasets: () => request('/api/datasets'),
  overview: (dataset) => request(`/api/overview?${qs({ dataset })}`),
  analysis: (kind, dataset) => request(`/api/analysis/${kind}?${qs({ dataset })}`),
  figures: () => request('/api/figures'),
  search: (q) => request(`/api/movies/search?${qs({ q })}`),
  popular: (genre, limit = 24) => request(`/api/movies/popular?${qs({ genre, limit })}`),
  genres: () => request('/api/genres'),
  presets: () => request('/api/presets'),
  recommend: (body) =>
    request('/api/recommend', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
}
