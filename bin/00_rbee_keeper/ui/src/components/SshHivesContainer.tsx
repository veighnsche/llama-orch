// TEAM-297: SSH Hives data layer and container
// Handles data fetching with React 19 use() hook and Suspense
// Uses auto-generated Tauri bindings from tauri-specta v2.0.0-rc.21

import { use, useState, Suspense } from "react";
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

// TEAM-297: Async function to fetch SSH hives using tauri-specta bindings
async function fetchSshHives(): Promise<SshHive[]> {
  const result = await commands.hiveList();

  if (result.status === "ok") {
    return result.data.map(convertToSshHive);
  }

  throw new Error(result.error || "Failed to load SSH hives");
}

// TEAM-296: Container component with Suspense boundary and data fetching
export function SshHivesContainer() {
  const [hivesPromise, setHivesPromise] = useState(() => fetchSshHives());

  const handleRefresh = () => {
    setHivesPromise(fetchSshHives());
  };

  return (
    <Suspense fallback={<LoadingHives />}>
      <SshHivesTable hives={use(hivesPromise)} onRefresh={handleRefresh} />
    </Suspense>
  );
}
