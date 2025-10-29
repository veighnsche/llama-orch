// TEAM-295: rbee-keeper GUI - Main application component with routing
// TEAM-334: Uses Shell component for layout (titlebar + sidebar + content)
// TEAM-340: Added Queen page with iframe
// TEAM-342: Added Hive page with dynamic iframe
// TEAM-XXX: Added narration listener for Queen iframe events
// TEAM-350: Log build mode on startup

import { useEffect } from 'react'
import { SidebarProvider } from '@rbee/ui/atoms'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Shell } from './components/Shell'
import HelpPage from './pages/HelpPage'
import HivePage from './pages/HivePage'
import QueenPage from './pages/QueenPage'
import KeeperPage from './pages/ServicesPage'
import SettingsPage from './pages/SettingsPage'
import { setupNarrationListener } from './utils/narrationListener'

// TEAM-350: Log build mode on startup
const isDev = import.meta.env.DEV
if (isDev) {
  console.log('🔧 [KEEPER UI] Running in DEVELOPMENT mode')
  console.log('   - Vite dev server active (hot reload enabled)')
  console.log('   - Running on: http://localhost:5173')
} else {
  console.log('🚀 [KEEPER UI] Running in PRODUCTION mode')
  console.log('   - Tauri app (embedded)')
}

function App() {
  // TEAM-XXX: Setup listener for narration events from Queen iframe
  useEffect(() => {
    const cleanup = setupNarrationListener()
    return cleanup
  }, [])

  return (
    <BrowserRouter>
      <SidebarProvider>
        <Shell>
          <Routes>
            <Route path="/" element={<KeeperPage />} />
            <Route path="/queen" element={<QueenPage />} />
            <Route path="/hive/:hiveId" element={<HivePage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/help" element={<HelpPage />} />
          </Routes>
        </Shell>
      </SidebarProvider>
    </BrowserRouter>
  )
}

export default App
