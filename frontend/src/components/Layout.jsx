import { BarChart3, Clapperboard, Compass, LayoutDashboard, Moon, Sparkles, Sun } from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { useTheme } from '../theme.jsx'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, title: 'Dashboard' },
  { to: '/overview', label: 'Overview', icon: BarChart3, title: 'Overview' },
  { to: '/recommend', label: 'Get Recommendations', icon: Sparkles, title: 'Get recommendations' },
  { to: '/explore', label: 'Explore analyses', icon: Compass, title: 'Explore analyses' },
]

function BackendStatus() {
  const [s, setS] = useState({ kind: 'wait', text: 'Connecting…' })
  useEffect(() => {
    let alive = true, timer
    const tick = async () => {
      try {
        const st = await api.status()
        if (!alive) return
        if (st.error) setS({ kind: 'err', text: 'Models failed to load' })
        else if (st.ready) setS({ kind: 'ok', text: 'Models ready' })
        else { setS({ kind: 'wait', text: 'Loading models…' }); timer = setTimeout(tick, 2500) }
      } catch {
        if (!alive) return
        setS({ kind: 'err', text: 'API offline' })
        timer = setTimeout(tick, 5000)
      }
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [])
  return <span className="status-pill" role="status"><i className={`dot ${s.kind}`} />{s.text}</span>
}

export default function Layout({ children }) {
  const { theme, toggle } = useTheme()
  const { pathname } = useLocation()
  const page = NAV.find((n) => (n.to === '/' ? pathname === '/' : pathname.startsWith(n.to)))
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark"><Clapperboard size={20} /></span>
          <div><div className="brand-name">CineMatch</div><div className="brand-sub">Find your next watch</div></div>
        </div>
        <div className="nav-label">Navigate</div>
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <Icon size={18} />{label}
          </NavLink>
        ))}
        <div className="sidebar-foot">
          <b>Tip</b><br />
          Rate a few movies you love on <i>Get Recommendations</i> and compare what five engines suggest.
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <span className="topbar-title">{page?.title}</span>
          <div className="topbar-actions">
            <BackendStatus />
            <button className="icon-btn" onClick={toggle} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}>
              {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
            </button>
          </div>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  )
}
