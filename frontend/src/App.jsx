import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Overview from './pages/Overview.jsx'
import Recommend from './pages/Recommend.jsx'
import Explore from './pages/Explore.jsx'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/overview" element={<Overview />} />
        <Route path="/recommend" element={<Recommend />} />
        <Route path="/explore" element={<Explore />} />
        <Route path="*" element={<Dashboard />} />
      </Routes>
    </Layout>
  )
}
