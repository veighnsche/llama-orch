// TEAM-339: Generic daemon service data provider using React 19 use() hook
// White-labeled version of QueenContainer for reuse across Queen/Hive cards
// Fetches data into any Zustand store, handles loading/error states
// Children get data from store - NO data passing via props
// No useEffect needed - pure Suspense pattern

import {
  Alert,
  AlertDescription,
  AlertTitle,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@rbee/ui/atoms'
import { AlertCircle, Loader2 } from 'lucide-react'
import { Component, type ReactNode, Suspense, use, useCallback, useState } from 'react'

// Promise cache - CRITICAL: Promises must be cached, not created in render
const promiseCache = new Map<string, Promise<void>>()

interface DaemonMetadata {
  name: string
  description: string
}

interface DaemonContainerProps {
  /** Unique cache key for this daemon (e.g., "queen", "hive-localhost") */
  cacheKey: string
  /** Daemon display metadata */
  metadata: DaemonMetadata
  /** Function that fetches daemon status into store */
  fetchFn: () => Promise<void>
  /** Children to render after successful fetch */
  children: ReactNode
  /** Optional custom loading fallback */
  fallback?: ReactNode
}

function fetchDaemonStatus(key: string, fetchFn: () => Promise<void>): Promise<void> {
  if (!promiseCache.has(key)) {
    const promise = fetchFn()
    promiseCache.set(key, promise)
  }
  return promiseCache.get(key)!
}

// Error boundary for daemon loading - React 19 idiomatic pattern
class DaemonErrorBoundary extends Component<
  { children: ReactNode; metadata: DaemonMetadata; onReset: () => void },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // React 19: Single consolidated error log
    console.error(`${this.props.metadata.name} Error:`, error, errorInfo)
  }

  render() {
    if (this.state.error) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>{this.props.metadata.name}</CardTitle>
            <CardDescription>{this.props.metadata.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Failed to load {this.props.metadata.name} status</AlertTitle>
                <AlertDescription>{this.state.error.message || 'An unexpected error occurred'}</AlertDescription>
              </Alert>
              <Button
                onClick={() => {
                  this.setState({ error: null })
                  this.props.onReset()
                }}
                className="w-full"
              >
                Try Again
              </Button>
            </div>
          </CardContent>
        </Card>
      )
    }

    return this.props.children
  }
}

// Fetcher component - triggers fetch via use(), then renders children
// Children get data from their respective store
function DaemonFetcher({
  promiseKey,
  fetchFn,
  children,
}: {
  promiseKey: string
  fetchFn: () => Promise<void>
  children: ReactNode
}) {
  // use() hook - React will Suspend until promise resolves
  // This populates the store, then children can read from it
  use(fetchDaemonStatus(promiseKey, fetchFn))
  return <>{children}</>
}

// Loading fallback component
function DaemonLoadingFallback({ metadata }: { metadata: DaemonMetadata }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{metadata.name}</CardTitle>
        <CardDescription>{metadata.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      </CardContent>
    </Card>
  )
}

// Generic data provider - fetches into store, children read from store
export function DaemonContainer({ cacheKey, metadata, fetchFn, children, fallback }: DaemonContainerProps) {
  const [refreshKey, setRefreshKey] = useState(0)

  const handleRefresh = useCallback(() => {
    const newKey = refreshKey + 1
    setRefreshKey(newKey)
    // Clear the old promise from cache
    promiseCache.delete(`${cacheKey}-${refreshKey}`)
  }, [cacheKey, refreshKey])

  return (
    <DaemonErrorBoundary metadata={metadata} onReset={handleRefresh}>
      <Suspense fallback={fallback ?? <DaemonLoadingFallback metadata={metadata} />}>
        <DaemonFetcher promiseKey={`${cacheKey}-${refreshKey}`} fetchFn={fetchFn}>
          {children}
        </DaemonFetcher>
      </Suspense>
    </DaemonErrorBoundary>
  )
}
