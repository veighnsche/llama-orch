// TEAM-297: SSH Hives data layer and container
// Handles data fetching with React 19 use() hook and Suspense
// Uses auto-generated Tauri bindings from tauri-specta v2.0.0-rc.21

import { use, useState, Suspense, useCallback } from "react";
import { commands } from "@/generated/bindings";
import type { SshTarget } from "@/generated/bindings";
import { SshHivesTable, LoadingHives, type SshHive } from "./SshHivesTable";

// TEAM-297: Convert tauri-specta SshTarget to SshHive for table component
function convertToSshHive(target: SshTarget): SshHive {
  return {
    host: target.host,
    host_subtitle: target.host_subtitle ?? undefined,
    hostname: target.hostname,
    user: target.user,
    port: target.port,
    status: target.status,
  };
}

// TEAM-297: Fetch function that returns a cached promise
// React docs: "Promises created in Client Components are recreated on every render"
// Solution: Use a cache to ensure the same promise is returned for the same key
const promiseCache = new Map<string, Promise<SshHive[]>>();

function fetchSshHives(key: string): Promise<SshHive[]> {
  // Check if we already have a promise for this key
  if (!promiseCache.has(key)) {
    // Create and cache the promise
    const promise = commands.hiveList().then((result) => {
      if (result.status === "ok") {
        return result.data.map(convertToSshHive);
      }
      throw new Error(result.error || "Failed to load SSH hives");
    });
    promiseCache.set(key, promise);
  }

  return promiseCache.get(key)!;
}

// TEAM-297: Container component with Suspense boundary
export function SshHivesContainer() {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = useCallback(() => {
    // Generate new key to force refetch
    const newKey = refreshKey + 1;
    setRefreshKey(newKey);
    // Clear the old promise from cache
    promiseCache.delete(`hives-${refreshKey}`);
  }, [refreshKey]);

  return (
    <Suspense fallback={<LoadingHives />}>
      <SshHivesContentWrapper
        promiseKey={`hives-${refreshKey}`}
        onRefresh={handleRefresh}
      />
    </Suspense>
  );
}

// TEAM-297: Wrapper to pass refresh handler to table
function SshHivesContentWrapper({
  promiseKey,
  onRefresh,
}: {
  promiseKey: string;
  onRefresh: () => void;
}) {
  const hives = use(fetchSshHives(promiseKey));
  return <SshHivesTable hives={hives} onRefresh={onRefresh} />;
}
