// TEAM-352: Generic query container - dumb UI for loading/error/data states
// Replaces DaemonContainer with simpler, type-safe pattern
// No promise caching, no Suspense - just conditional rendering

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
import type { ReactNode } from 'react'

interface QueryContainerProps<T> {
  isLoading: boolean
  error: string | null
  data: T | null
  children: (data: T) => ReactNode
  onRetry?: () => void
  metadata?: {
    name: string
    description?: string
  }
}

// TEAM-352: Generic container - type-safe, simple, dumb UI
// TEAM-354: Added stale-while-revalidate indicator per CORRECT_ARCHITECTURE.md
export function QueryContainer<T>({
  isLoading,
  error,
  data,
  children,
  onRetry,
  metadata,
}: QueryContainerProps<T>) {
  // Loading state (but show stale data if available)
  if (isLoading && !data) {
    return (
      <Card>
        {metadata && (
          <CardHeader>
            <CardTitle>{metadata.name}</CardTitle>
            {metadata.description && (
              <CardDescription>{metadata.description}</CardDescription>
            )}
          </CardHeader>
        )}
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    )
  }

  // Error state (but show stale data if available)
  if (error && !data) {
    return (
      <Card>
        {metadata && (
          <CardHeader>
            <CardTitle>{metadata.name}</CardTitle>
            {metadata.description && (
              <CardDescription>{metadata.description}</CardDescription>
            )}
          </CardHeader>
        )}
        <CardContent>
          <div className="space-y-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Failed to load data</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
            {onRetry && (
              <Button onClick={onRetry} className="w-full">
                Try Again
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  // No data state
  if (!data) {
    return null
  }

  // Success state - render children with type-safe data
  // TEAM-354: Show stale-while-revalidate indicator when refreshing
  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute top-2 right-2 z-10">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        </div>
      )}
      {children(data)}
    </div>
  )
}
