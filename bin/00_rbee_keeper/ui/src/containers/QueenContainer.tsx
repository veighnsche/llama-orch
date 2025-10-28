// TEAM-338: Queen service data provider using React 19 use() hook
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
import { AlertCircle, Loader2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@rbee/ui/atoms";
import { Alert, AlertTitle, AlertDescription } from "@rbee/ui/atoms";
import { Button } from "@rbee/ui/atoms";
import { useQueenStore } from "../store/queenStore";

// Promise cache - CRITICAL: Promises must be cached, not created in render
const promiseCache = new Map<string, Promise<void>>();

function fetchQueenStatus(key: string): Promise<void> {
  if (!promiseCache.has(key)) {
    // Fetch into store - promise resolves when store is updated
    const promise = useQueenStore.getState().fetchStatus();
    promiseCache.set(key, promise);
  }
  return promiseCache.get(key)!;
}

// Error boundary for Queen loading - React 19 idiomatic pattern
class QueenErrorBoundary extends Component<
  { children: ReactNode; onReset: () => void },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // React 19: Single consolidated error log
    console.error("Queen Error:", error, errorInfo);
  }

  render() {
    if (this.state.error) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>Queen</CardTitle>
            <CardDescription>Smart API server</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Failed to load Queen status</AlertTitle>
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
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}

// Fetcher component - triggers fetch via use(), then renders children
// Children get data from useQueenStore() themselves
function QueenFetcher({
  promiseKey,
  children,
}: {
  promiseKey: string;
  children: ReactNode;
}) {
  // use() hook - React will Suspend until promise resolves
  // This populates the store, then children can read from it
  use(fetchQueenStatus(promiseKey));
  return <>{children}</>;
}

// Loading fallback component
function QueenLoadingFallback() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Queen</CardTitle>
        <CardDescription>Smart API server</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Job router that dispatches inference requests to workers in the
            correct hive
          </p>
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Data provider - fetches into store, children read from store
export function QueenDataProvider({
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
    promiseCache.delete(`queen-${refreshKey}`);
  }, [refreshKey]);

  return (
    <QueenErrorBoundary onReset={handleRefresh}>
      <Suspense fallback={fallback ?? <QueenLoadingFallback />}>
        <QueenFetcher promiseKey={`queen-${refreshKey}`}>
          {children}
        </QueenFetcher>
      </Suspense>
    </QueenErrorBoundary>
  );
}

// Re-export type for consumers
export type { QueenStatus } from "../store/queenStore";
