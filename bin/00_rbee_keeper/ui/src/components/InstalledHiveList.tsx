// TEAM-338: Installed hives list - shows all installed hives with lifecycle controls
// TEAM-339: Uses DaemonContainer with React 19 use() hook (no useEffect)
// TEAM-340: Simplified - HiveCard is now self-contained with its own DaemonContainer
// Displays installed hives with start/stop/uninstall actions

import { DaemonContainer } from '../containers/DaemonContainer'
import { useSshHivesStore } from '../store/hiveStore'
import { HiveCard } from './HiveCard'

// TEAM-340: Inner component that renders after hives list is loaded
function InstalledHiveCards() {
  const { hives, installedHives } = useSshHivesStore()

  // Get installed hive details
  const installedHiveDetails = hives.filter((hive) => installedHives.includes(hive.host))

  // Add localhost if installed but not in SSH config
  const hasLocalhost = installedHives.includes('localhost')
  const localhostInConfig = hives.some((h) => h.host === 'localhost')
  const showLocalhost = hasLocalhost && !localhostInConfig

  // Empty state - return null, no cards needed
  if (installedHives.length === 0) {
    return null
  }

  return (
    <>
      {/* Localhost hive (if installed) - HiveCard handles its own data fetching */}
      {showLocalhost && <HiveCard hiveId="localhost" title="localhost" description="This machine" />}

      {/* SSH hives - each HiveCard handles its own data fetching */}
      {installedHiveDetails.map((hive) => (
        <HiveCard
          key={hive.host}
          hiveId={hive.host}
          title={hive.host}
          description={hive.host_subtitle || `${hive.user}@${hive.hostname}:${hive.port}`}
        />
      ))}
    </>
  )
}

export function InstalledHiveList() {
  // TEAM-340: Only fetch the hives list here, individual HiveCards fetch their own status
  return (
    <DaemonContainer
      cacheKey="hives-list"
      metadata={{
        name: 'Hives',
        description: 'SSH hive targets',
      }}
      fetchFn={() => useSshHivesStore.getState().fetchHives()}
    >
      <InstalledHiveCards />
    </DaemonContainer>
  )
}
