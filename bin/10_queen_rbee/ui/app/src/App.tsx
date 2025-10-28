// TEAM-292: Main App component with React Router
// Ported from web-ui.old Next.js layout to React/Vite

import { SidebarInset, SidebarProvider } from '@rbee/ui/atoms'
import { ThemeProvider } from 'next-themes'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppSidebar } from './components/AppSidebar'
import DashboardPage from './pages/DashboardPage'
import HelpPage from './pages/HelpPage'
import KeeperPage from './pages/KeeperPage'
import SettingsPage from './pages/SettingsPage'

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <BrowserRouter>
        <SidebarProvider defaultOpen={true}>
          <AppSidebar />
          <SidebarInset>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/keeper" element={<KeeperPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/help" element={<HelpPage />} />
            </Routes>
          </SidebarInset>
        </SidebarProvider>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
