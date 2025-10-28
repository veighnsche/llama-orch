// TEAM-338: Installed hives list - shows all installed hives with lifecycle controls
// TEAM-339: Uses DaemonContainer with React 19 use() hook (no useEffect)
// Fully self-contained component connected to hiveStore
// Displays installed hives with start/stop/uninstall actions

import { DaemonContainer } from '../containers/DaemonContainer'
import { useSshHivesStore } from '../store/hiveStore'
import { HiveCard } from './HiveCard'

// Inner component that renders after data is loaded
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
      {/* Localhost hive (if installed) */}
      {showLocalhost && <HiveCard hiveId="localhost" title="localhost" description="This machine" />}

      {/* SSH hives - each gets its own Card */}
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
  // TEAM-339: DaemonContainer handles loading/error states via Suspense
  // Component only renders after successful fetch
  return (
    <DaemonContainer
      cacheKey="hives-list"
      metadata={{
        name: 'Hives',
        description: 'SSH hive targets',
      }}
      fetchFn={async () => {
        // TEAM-339: Fetch hives list, then fetch individual status for each
        await useSshHivesStore.getState().fetchHives()
        const { installedHives, fetchHiveStatus } = useSshHivesStore.getState()
        await Promise.all(installedHives.map((hiveId) => fetchHiveStatus(hiveId)))
      }}
    >
      <InstalledHiveCards />
    </DaemonContainer>
  )
}
