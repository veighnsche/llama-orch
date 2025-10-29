// TEAM-350: Localhost hive card - no installation workflow (localhost is always available)
// TEAM-356: Uses HiveActionButton for consistent UI with HiveCard

import { useHive, useHiveActions } from '@/store/hiveStore'
import {
  Alert,
  AlertDescription,
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@rbee/ui/atoms'
import { AlertCircle, Loader2, RefreshCw } from 'lucide-react'
import { StatusBadge } from '../StatusBadge'
import { ServiceActionButton } from './ServiceActionButton'
import { useCommandStore } from '@/store/commandStore'

// TEAM-350: Localhost component
// TEAM-357: Uses unified ServiceActionButton with rebuild and uninstall
export function LocalhostHive() {
  const { hive, isLoading, error, refetch } = useHive('localhost')
  const { start, stop, uninstall, rebuild } = useHiveActions()
  const { isExecuting } = useCommandStore()

  // Loading state
  if (isLoading && !hive) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Localhost Hive</CardTitle>
          <CardDescription>This machine</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    )
  }

  // Error state
  if (error && !hive) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Localhost Hive</CardTitle>
          <CardDescription>This machine</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
            <Button onClick={refetch} className="w-full">
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  const isInstalled = true // Localhost is always "installed"
  const isRunning = hive?.status === 'online'
  const badgeStatus = isRunning ? ('running' as const) : ('stopped' as const)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Localhost Hive</CardTitle>
        <CardDescription>This machine</CardDescription>
        <CardAction>
          <StatusBadge status={badgeStatus} onClick={refetch} />
        </CardAction>
      </CardHeader>
      <div className="flex-1" />
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Manage workers and models on this computer.
          </p>
          <ServiceActionButton
            serviceId="localhost"
            isInstalled={isInstalled}
            isRunning={isRunning}
            isExecuting={isExecuting}
            actions={{
              start: (id) => start(id!),
              stop: (id) => stop(id!),
              rebuild: (id) => rebuild(id!),
              uninstall: (id) => uninstall(id!),
            }}
          />
        </div>
      </CardContent>
    </Card>
  )
}
