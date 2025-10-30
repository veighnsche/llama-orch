// TEAM-352: Migrated to use @rbee/dev-utils for startup logging
// Old implementation: ~13 LOC of manual environment logging
// New implementation: 1 LOC using shared utility
// Reduction: 12 LOC

// Queen UI - Minimal interface for heartbeat monitoring and RHAI scheduling
// Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor

import { ThemeProvider } from "next-themes";
import { logStartupMode } from "@rbee/dev-utils";
import DashboardPage from "./pages/DashboardPage";

// TEAM-352: Use shared startup logging
logStartupMode("QUEEN UI", import.meta.env.DEV, 7834);

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
