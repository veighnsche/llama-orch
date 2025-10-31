// Queen Dashboard - Heartbeat Monitor + RHAI IDE
// Minimal UI for Queen operations

import { useHeartbeat } from "@rbee/queen-rbee-react";
import { HeartbeatMonitor } from "../components/HeartbeatMonitor";
import { RhaiIDE } from "../components/RhaiIDE";
import { ConnectionStatus } from "../components/ConnectionStatus";

export default function DashboardPage() {
  // TEAM-352: Use default URL from hook (no hardcoded URL)
  const { data, connected, loading, error } = useHeartbeat();
  
  // ============================================================
  // BUG FIX: TEAM-377 | Hive count always showing 0
  // ============================================================
  // SUSPICION:
  // - Thought backend wasn't sending hive data
  // - Suspected useHeartbeat hook was broken
  //
  // INVESTIGATION:
  // - Checked useHeartbeat hook - correctly aggregates hives_online from backend ✓
  // - Checked backend heartbeat stream - sending correct hive counts ✓
  // - Found line 12: `const hives: any[] = []` - hardcoded empty array!
  // - Found line 14: `const hivesOnline = hives.length` - always 0 because hives is empty
  //
  // ROOT CAUSE:
  // - Line 12 had TODO comment: "TODO: Parse hives from heartbeat data"
  // - This was never implemented, left as empty array in production
  // - Line 14 calculated count from empty array instead of using data from hook
  // - Hook was receiving correct data, but UI was ignoring it
  //
  // FIX:
  // - Use data?.hives from hook instead of empty array
  // - Use data?.hives_online from hook instead of calculating from array length
  // - Backend already aggregates hive count correctly, just use it
  //
  // TESTING:
  // - Verified useHeartbeat returns data.hives_online correctly
  // - Verified useHeartbeat returns data.hives array correctly
  // - Will test in browser: Active Hives count should match running hives
  // ============================================================
  const hives = data?.hives || [];
  const workersOnline = data?.workers_online || 0;
  const hivesOnline = data?.hives_online || 0; // TEAM-377: Use backend count, not array length

  console.log({ hives, workersOnline, hivesOnline })

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-muted-foreground">Loading Queen UI...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4 text-red-500">⚠️ Error</h1>
          <p className="text-muted-foreground mb-4">Failed to load Queen UI</p>
          <p className="text-sm text-red-500">{error.message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start gap-4">
        <div>
          <h1 className="text-3xl font-bold">Queen Dashboard</h1>
          <p className="text-muted-foreground">
            Heartbeat Monitor & RHAI Scheduler
          </p>
        </div>
        <ConnectionStatus connected={connected} />
      </div>

      {/* Components Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <HeartbeatMonitor 
          workersOnline={workersOnline} 
          hivesOnline={hivesOnline}
          hives={hives} 
        />
        <RhaiIDE />
      </div>
    </div>
  );
}
