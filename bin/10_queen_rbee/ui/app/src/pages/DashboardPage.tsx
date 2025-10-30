// Queen Dashboard - Heartbeat Monitor + RHAI IDE
// Minimal UI for Queen operations

import { useHeartbeat } from "@rbee/queen-rbee-react";
import { HeartbeatMonitor } from "../components/HeartbeatMonitor";
import { RhaiIDE } from "../components/RhaiIDE";
import { ConnectionStatus } from "../components/ConnectionStatus";

export default function DashboardPage() {
  // TEAM-352: Use default URL from hook (no hardcoded URL)
  const { data, connected, loading, error } = useHeartbeat();
  const hives: any[] = []; // TODO: Parse hives from heartbeat data
  const workersOnline = data?.workers_online || 0;

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
        <HeartbeatMonitor workersOnline={workersOnline} hives={hives} />
        <RhaiIDE />
      </div>
    </div>
  );
}
