// TEAM-352: Migrated to use @rbee/dev-utils for startup logging
// Old implementation: ~13 LOC of manual environment logging
// New implementation: 1 LOC using shared utility
// Reduction: 12 LOC

// Queen UI - Minimal interface for heartbeat monitoring and RHAI scheduling
// Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor

// ============================================================
// BUG FIX: TEAM-374 | Missing QueryClientProvider
// ============================================================
// SUSPICION:
// - RhaiIDE component uses React Query hooks
// - Error: "No QueryClient set, use QueryClientProvider to set one"
//
// INVESTIGATION:
// - Checked Hive UI App.tsx - has QueryClientProvider
// - Checked Queen UI App.tsx - missing QueryClientProvider
// - RhaiIDE in DashboardPage uses React Query
//
// ROOT CAUSE:
// - Queen UI was created without React Query setup
// - DashboardPage/RhaiIDE components use useQuery but no provider exists
//
// FIX:
// - Added QueryClient instantiation (same config as Hive UI)
// - Wrapped app in QueryClientProvider
// - Retry logic: 3 retries with exponential backoff
//
// TESTING:
// - Verified Queen UI loads without React Query errors
// - Confirmed RhaiIDE component renders
// - Checked browser console - no QueryClient errors
// ============================================================

import { QueryClient, QueryClientProvider } from '@rbee/queen-rbee-react'
import { logStartupMode } from "@rbee/dev-utils";
import DashboardPage from "./pages/DashboardPage";

// TEAM-352: Use shared startup logging
logStartupMode("QUEEN UI", import.meta.env.DEV, 7834);

// TEAM-374: Create QueryClient (imported from react package, not directly from @tanstack)
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background font-sans">
        <DashboardPage />
      </div>
    </QueryClientProvider>
  );
}

export default App;
