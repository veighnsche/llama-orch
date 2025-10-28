// TEAM-338: SSH Hives data provider using React 19 use() hook
// Fetches data into Zustand store, handles loading/error states
// Children get data from store - NO data passing via props
// No useEffect needed - pure Suspense pattern

import {
  use,
  useState,
  Suspense,
  useCallback,
  Component,
  type ReactNode,
} from "react";
import { AlertCircle } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@rbee/ui/atoms";
import { Button } from "@rbee/ui/atoms";
import { useSshHivesStore } from "../store/hiveStore";

// Promise cache - CRITICAL: Promises must be cached, not created in render
const promiseCache = new Map<string, Promise<void>>();

function fetchSshHives(key: string): Promise<void> {
  if (!promiseCache.has(key)) {
    // Fetch into store - promise resolves when store is updated
    const promise = useSshHivesStore.getState().fetchHives();
    promiseCache.set(key, promise);
  }
  return promiseCache.get(key)!;
}

// Error boundary for SSH hives loading - React 19 idiomatic pattern
class SshHivesErrorBoundary extends Component<
  { children: ReactNode; onReset: () => void },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // React 19: Single consolidated error log
    console.error("SSH Hives Error:", error, errorInfo);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex items-center justify-center p-4">
          <div className="w-full max-w-md space-y-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Failed to load SSH targets</AlertTitle>
              <AlertDescription>
                {this.state.error.message || "An unexpected error occurred"}
              </AlertDescription>
            </Alert>
            <Button
              onClick={() => {
                this.setState({ error: null });
                this.props.onReset();
              }}
              className="w-full"
            >
              Try Again
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Fetcher component - triggers fetch via use(), then renders children
// Children get data from useSshHivesStore() themselves
function SshHivesFetcher({
  promiseKey,
  children,
}: {
  promiseKey: string;
  children: ReactNode;
}) {
  // use() hook - React will Suspend until promise resolves
  // This populates the store, then children can read from it
  use(fetchSshHives(promiseKey));
  return <>{children}</>;
}

// Data provider - fetches into store, children read from store
export function SshHivesDataProvider({
  children,
  fallback,
}: {
  children: ReactNode;
  fallback?: ReactNode;
}) {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = useCallback(() => {
    const newKey = refreshKey + 1;
    setRefreshKey(newKey);
    // Clear the old promise from cache
    promiseCache.delete(`hives-${refreshKey}`);
  }, [refreshKey]);

  return (
    <SshHivesErrorBoundary onReset={handleRefresh}>
      <Suspense fallback={fallback}>
        <SshHivesFetcher promiseKey={`hives-${refreshKey}`}>
          {children}
        </SshHivesFetcher>
      </Suspense>
    </SshHivesErrorBoundary>
  );
}

// Re-export type for consumers
export type { SshHive } from "../store/hiveStore";
