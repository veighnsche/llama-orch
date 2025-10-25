// TEAM-292: Main App component with React Router
// Ported from web-ui.old Next.js layout to React/Vite

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from 'next-themes';
import { SidebarProvider, SidebarInset } from '@rbee/ui/atoms';
import { AppSidebar } from './components/AppSidebar';
import DashboardPage from './pages/DashboardPage';
import KeeperPage from './pages/KeeperPage';
import SettingsPage from './pages/SettingsPage';
import HelpPage from './pages/HelpPage';

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
  );
}

export default App;
