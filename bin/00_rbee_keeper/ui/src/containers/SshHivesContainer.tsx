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
import { useSshHivesStore } from "../store/sshHivesStore";

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

// Error boundary for SSH hives loading
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

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("SSH Hives Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 space-y-4">
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold text-destructive">
              Failed to load SSH targets
            </h3>
            <p className="text-sm text-muted-foreground max-w-md">
              {this.state.error?.message || "Unknown error occurred"}
            </p>
          </div>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null });
              this.props.onReset();
            }}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Try Again
          </button>
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
export type { SshHive } from "../store/sshHivesStore";
