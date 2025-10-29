// TEAM-338: Installed hives list - shows all installed hives with lifecycle controls
// TEAM-353: Rewritten to use query hooks (deleted DaemonContainer pattern)

import { useSshHives, useInstalledHives } from "../store/hiveQueries";
import { HiveCard } from "./cards/HiveCard";
import { Card, CardHeader } from "@rbee/ui/atoms";
import { Loader2 } from "lucide-react";
import type { SshHive } from "../store/hiveQueries";

// TEAM-368: Get actual install status from backend (no more Zustand!)
export function InstalledHiveList() {
  const { data: hives = [], isLoading: hivesLoading } = useSshHives();
  const { data: installedHives = [], isLoading: installedLoading } = useInstalledHives();

  // TEAM-368: Show loading state while fetching
  if (hivesLoading || installedLoading) {
    return (
      <Card className="w-80 h-80 max-w-sm flex flex-col">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm text-muted-foreground">Loading installed hives...</span>
          </div>
        </CardHeader>
      </Card>
    );
  }

  // TEAM-368: Filter hives by actual install status from backend
  const installedSshHives = hives.filter(
    (hive: SshHive) => installedHives.includes(hive.host)
  );


  // Empty state - return null, no cards needed
  if (installedSshHives.length === 0) {
    return null;
  }

  return (
    <>
      {/* SSH hives - each HiveCard handles its own data fetching */}
      {installedSshHives.map((hive: SshHive) => (
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
