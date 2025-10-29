// Queen UI - Minimal interface for heartbeat monitoring and RHAI scheduling
// Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor

import { ThemeProvider } from "next-themes";
import DashboardPage from "./pages/DashboardPage";

// TEAM-350: Log build mode on startup
const isDev = import.meta.env.DEV;
if (isDev) {
  console.log("🔧 [QUEEN UI] Running in DEVELOPMENT mode");
  console.log("   - Vite dev server active (hot reload enabled)");
  console.log(
    "   - Loaded via: http://localhost:7833/dev (proxied from :7834)",
  );
} else {
  console.log("🚀 [QUEEN UI] Running in PRODUCTION mode");
  console.log("   - Serving embedded static files");
  console.log("   - Loaded via: http://localhost:7833/");
}

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background font-sans">
        <DashboardPage />
      </div>
    </ThemeProvider>
  );
}

export default App;
