// TEAM-297: SSH Hives data layer and container
// Handles data fetching with React 19 use() hook and Suspense
// Uses auto-generated Tauri bindings from tauri-specta v2.0.0-rc.21

import {
  use,
  useState,
  Suspense,
  useCallback,
  Component,
  type ReactNode,
} from "react";
import { commands } from "@/generated/bindings";
import type { SshTarget } from "@/generated/bindings";
import { SshHivesTable, LoadingHives, type SshHive } from "./SshHivesTable";
import { AlertCircle } from "lucide-react";
import { Button } from "@rbee/ui/atoms";

// Re-export SshHive type for consumers
export type { SshHive };

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
    // TEAM-333: Updated to use ssh_list command with error handling
    const promise = commands
      .sshList()
      .then((result) => {
        if (result.status === "ok") {
          return result.data.map(convertToSshHive);
        }
        throw new Error(result.error || "Failed to load SSH hives");
      })
      .catch((error) => {
        // TEAM-333: Throw error to trigger ErrorBoundary
        console.error("Failed to fetch SSH targets:", error);
        throw error;
      });
    promiseCache.set(key, promise);
  }

  return promiseCache.get(key)!;
}

// TEAM-333: Error boundary for SSH hives loading
class SshHivesErrorBoundary extends Component<
  { children: ReactNode; onReset: () => void },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode; onReset: () => void }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 space-y-4">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold">
              Failed to load SSH targets
            </h3>
            <p className="text-sm text-muted-foreground max-w-md">
              {this.state.error?.message || "Unknown error occurred"}
            </p>
          </div>
          <Button
            onClick={() => {
              this.setState({ hasError: false, error: null });
              this.props.onReset();
            }}
          >
            Try Again
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}

// TEAM-297: Generic data provider component with render prop pattern
// Can be used by any component that needs SSH targets data
export function SshHivesDataProvider({
  children,
  fallback = <LoadingHives />,
}: {
  children: (hives: SshHive[], refresh: () => void) => ReactNode;
  fallback?: ReactNode;
}) {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = useCallback(() => {
    // Generate new key to force refetch
    const newKey = refreshKey + 1;
    setRefreshKey(newKey);
    // Clear the old promise from cache
    promiseCache.delete(`hives-${refreshKey}`);
  }, [refreshKey]);

  return (
    <SshHivesErrorBoundary onReset={handleRefresh}>
      <Suspense fallback={fallback}>
        <SshHivesContentWrapper promiseKey={`hives-${refreshKey}`}>
          {(hives) => children(hives, handleRefresh)}
        </SshHivesContentWrapper>
      </Suspense>
    </SshHivesErrorBoundary>
  );
}

// TEAM-297: Wrapper to use() the promise and pass data to render prop
function SshHivesContentWrapper({
  promiseKey,
  children,
}: {
  promiseKey: string;
  children: (hives: SshHive[]) => ReactNode;
}) {
  const hives = use(fetchSshHives(promiseKey));
  return <>{children(hives)}</>;
}

// TEAM-297: Backward compatible container for table view
// TEAM-333: Added ErrorBoundary for proper error handling
export function SshHivesContainer() {
  return (
    <SshHivesDataProvider>
      {(hives, onRefresh) => (
        <SshHivesTable hives={hives} onRefresh={onRefresh} />
      )}
    </SshHivesDataProvider>
  );
}
