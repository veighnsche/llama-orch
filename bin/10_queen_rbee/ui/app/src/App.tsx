// Queen UI - Minimal interface for heartbeat monitoring and RHAI scheduling
// Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor

import { ThemeProvider } from 'next-themes'
import DashboardPage from './pages/DashboardPage'

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background">
        <DashboardPage />
      </div>
    </ThemeProvider>
  )
}

export default App
