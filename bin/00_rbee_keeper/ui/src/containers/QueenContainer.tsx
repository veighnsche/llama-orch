// TEAM-338: Queen service data layer and container
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
import { AlertCircle } from "lucide-react";
import { Button } from "@rbee/ui/atoms";

// TEAM-338: Re-export QueenStatus type from presentation component
export type { QueenStatus } from "../components/QueenCard";

// TEAM-338: Fetch function that returns a cached promise
// React docs: "Promises created in Client Components are recreated on every render"
// Solution: Use a cache to ensure the same promise is returned for the same key
const promiseCache = new Map<string, Promise<any>>();

async function fetchQueenStatus(key: string): Promise<any> {
  // Check if we already have a promise for this key
  if (!promiseCache.has(key)) {
    // Create and cache the promise
    // TEAM-338: For now, return a mock status until we have a status command
    // TODO: Replace with actual queen_status command when available
    const promise = Promise.resolve({
      isRunning: false,
      isInstalled: false,
    });
    promiseCache.set(key, promise);
  }

  return promiseCache.get(key)!;
}

// TEAM-338: Error boundary for Queen status loading
class QueenErrorBoundary extends Component<
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
              Failed to load Queen status
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

// TEAM-338: Generic data provider component with render prop pattern
// COMPONENT AGNOSTIC - consumers provide their own presentation
export function QueenDataProvider({
  children,
  fallback,
}: {
  children: (status: any, refresh: () => void) => ReactNode;
  fallback?: ReactNode;
}) {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = useCallback(() => {
    // Generate new key to force refetch
    const newKey = refreshKey + 1;
    setRefreshKey(newKey);
    // Clear the old promise from cache
    promiseCache.delete(`queen-${refreshKey}`);
  }, [refreshKey]);

  return (
    <QueenErrorBoundary onReset={handleRefresh}>
      <Suspense fallback={fallback}>
        <QueenContentWrapper promiseKey={`queen-${refreshKey}`}>
          {(status) => children(status, handleRefresh)}
        </QueenContentWrapper>
      </Suspense>
    </QueenErrorBoundary>
  );
}

// TEAM-338: Wrapper to use() the promise and pass data to render prop
function QueenContentWrapper({
  promiseKey,
  children,
}: {
  promiseKey: string;
  children: (status: any) => ReactNode;
}) {
  const status = use(fetchQueenStatus(promiseKey));
  return <>{children(status)}</>;
}
