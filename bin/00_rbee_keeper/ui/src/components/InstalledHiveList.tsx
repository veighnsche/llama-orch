// TEAM-338: Installed hives list - shows all installed hives with lifecycle controls
// Fully self-contained component connected to hiveStore
// Displays installed hives with start/stop/uninstall actions

import { useEffect } from "react";
import { Loader2, AlertCircle } from "lucide-react";
import { useSshHivesStore } from "../store/hiveStore";
import { HiveCard } from "./HiveCard";

export function InstalledHiveList() {
  const { hives, installedHives, isLoading, error, fetchHives } =
    useSshHivesStore();

  // Fetch hives on mount
  useEffect(() => {
    fetchHives();
  }, [fetchHives]);

  // Get installed hive details
  const installedHiveDetails = hives.filter((hive) =>
    installedHives.includes(hive.host),
  );

  // Add localhost if installed but not in SSH config
  const hasLocalhost = installedHives.includes("localhost");
  const localhostInConfig = hives.some((h) => h.host === "localhost");
  const showLocalhost = hasLocalhost && !localhostInConfig;

  // Loading state
  if (isLoading && installedHives.length === 0) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center gap-2 text-destructive p-4 rounded-lg border border-destructive/50 bg-destructive/10">
        <AlertCircle className="h-4 w-4" />
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  // Empty state - return null, no cards needed
  if (installedHives.length === 0) {
    return null;
  }

  return (
    <>
      {/* Localhost hive (if installed) */}
      {showLocalhost && (
        <HiveCard
          hiveId="localhost"
          title="localhost"
          description="This machine"
        />
      )}

      {/* SSH hives - each gets its own Card */}
      {installedHiveDetails.map((hive) => (
        <HiveCard
          key={hive.host}
          hiveId={hive.host}
          title={hive.host}
          description={
            hive.host_subtitle || `${hive.user}@${hive.hostname}:${hive.port}`
          }
        />
      ))}
    </>
  );
}
