import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Status from './pages/Status'
import Queen from './pages/Queen'
import Hives from './pages/Hives'
import Workers from './pages/Workers'
import Models from './pages/Models'
import Inference from './pages/Inference'

function App() {
  return (
    <BrowserRouter>
      <nav>
        <ul>
          <li><NavLink to="/">Status</NavLink></li>
          <li><NavLink to="/queen">Queen</NavLink></li>
          <li><NavLink to="/hives">Hives</NavLink></li>
          <li><NavLink to="/workers">Workers</NavLink></li>
          <li><NavLink to="/models">Models</NavLink></li>
          <li><NavLink to="/inference">Inference</NavLink></li>
        </ul>
      </nav>

      <div className="container">
        <Routes>
          <Route path="/" element={<Status />} />
          <Route path="/queen" element={<Queen />} />
          <Route path="/hives" element={<Hives />} />
          <Route path="/workers" element={<Workers />} />
          <Route path="/models" element={<Models />} />
          <Route path="/inference" element={<Inference />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

export default App
