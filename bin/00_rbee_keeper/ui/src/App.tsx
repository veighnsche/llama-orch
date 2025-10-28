// TEAM-295: rbee-keeper GUI - Main application component with routing
// TEAM-334: Uses Shell component for layout (titlebar + sidebar + content)

import { SidebarProvider } from '@rbee/ui/atoms'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Shell } from './components/Shell'
import HelpPage from './pages/HelpPage'
import KeeperPage from './pages/ServicesPage'
import SettingsPage from './pages/SettingsPage'

function App() {
  return (
    <BrowserRouter>
      <SidebarProvider>
        <Shell>
          <Routes>
            <Route path="/" element={<KeeperPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/help" element={<HelpPage />} />
          </Routes>
        </Shell>
      </SidebarProvider>
    </BrowserRouter>
  )
}

export default App
