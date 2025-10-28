// TEAM-295: rbee-keeper GUI - Main application component with routing
// TEAM-334: Uses Shell component for layout (titlebar + sidebar + content)
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Shell } from "./components/Shell";
import { SidebarProvider } from "@rbee/ui/atoms";
import KeeperPage from "./pages/ServicesPage";
import SettingsPage from "./pages/SettingsPage";
import HelpPage from "./pages/HelpPage";

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
  );
}

export default App;
