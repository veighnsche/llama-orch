// TEAM-338: Installed hives list - shows all installed hives with lifecycle controls
// TEAM-353: Rewritten to use query hooks (deleted DaemonContainer pattern)

import { useSshHives, useSshHivesStore } from "../store/hiveStore";
import { HiveCard } from "./cards/HiveCard";
import type { SshHive } from "../store/hiveStore";

// TEAM-352: Rewritten to use query hooks
// TEAM-350: Removed localhost logic - localhost now handled by separate LocalhostHive component
export function InstalledHiveList() {
  const { hives, isLoading } = useSshHives();
  const installedHivesStore = useSshHivesStore();
  const installedHives = installedHivesStore.installedHives;

  // TEAM-350: Filter OUT localhost - shown separately on Services page
  const installedSshHives = hives.filter(
    (hive: SshHive) => installedHives.includes(hive.host) && hive.host !== 'localhost'
  );

  // Empty state - return null, no cards needed
  if (installedSshHives.length === 0) {
    return null;
  }

  // Show loading or empty state
  if (isLoading && installedSshHives.length === 0) {
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
