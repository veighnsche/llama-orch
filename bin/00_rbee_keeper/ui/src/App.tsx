// TEAM-295: rbee-keeper GUI - Main application component with routing
// TEAM-334: Splash screen removed - didn't help with Niri compatibility issue
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { KeeperSidebar } from "./components/KeeperSidebar";
import { SidebarProvider } from "@rbee/ui/atoms";
import KeeperPage from "./pages/ServicesPage";
import SettingsPage from "./pages/SettingsPage";
import HelpPage from "./pages/HelpPage";

function App() {
  return (
    <BrowserRouter>
      <SidebarProvider>
        <div className="h-screen w-screen overflow-hidden bg-background text-foreground flex">
          <KeeperSidebar />
          <Routes>
            <Route path="/" element={<KeeperPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/help" element={<HelpPage />} />
          </Routes>
        </div>
      </SidebarProvider>
    </BrowserRouter>
  );
}

export default App;
