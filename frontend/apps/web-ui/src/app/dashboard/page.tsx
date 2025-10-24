// TEAM-288: Live heartbeat monitoring dashboard
// TEAM-291: Updated to use zustand store

'use client';

import { useHeartbeat } from '@/src/hooks/useHeartbeat';
import { useRbeeStore } from '@/src/stores/rbeeStore';
import { Button } from '@rbee/ui/atoms';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms';
import { ThemeToggle } from '@rbee/ui/molecules';

export default function DashboardPage() {
  const { connected, loading, error } = useHeartbeat();
  
  // TEAM-291: Get state from zustand store
  const { 
    queen, 
    hives, 
    hivesOnline, 
    hivesAvailable,
    workersOnline,
    workersAvailable,
    workerIds,
    lastHeartbeat,
  } = useRbeeStore();

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-muted-foreground">Loading rbee SDK...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4 text-red-500">⚠️ Error</h1>
          <p className="text-muted-foreground mb-4">Failed to load rbee SDK</p>
          <p className="text-sm text-red-500">{error.message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor your queen, hives, workers, and models
          </p>
        </div>
        <ThemeToggle />
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Queen Status Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Queen Status
              <span className={connected ? 'text-green-500' : 'text-red-500'}>
                {connected ? '🟢' : '⚫'}
              </span>
            </CardTitle>
            <CardDescription>Central orchestrator</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <div
                className={`h-3 w-3 rounded-full ${
                  connected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="capitalize">
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            {queen.lastUpdate && (
              <p className="text-xs text-muted-foreground mt-2">
                Last update: {new Date(queen.lastUpdate).toLocaleTimeString()}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Hives Card */}
        <Card>
          <CardHeader>
            <CardTitle>Hives</CardTitle>
            <CardDescription>Pool managers</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {hivesOnline}
            </p>
            <p className="text-sm text-muted-foreground">
              {hivesAvailable} available
            </p>
            {hives.length > 0 && (
              <ul className="mt-2 space-y-1">
                {hives.map((hive) => (
                  <li key={hive.id} className="text-xs text-muted-foreground">
                    {hive.id}
                  </li>
                ))}
              </ul>
            )}
            <Button className="mt-4" variant="outline" size="sm">
              Add Hive
            </Button>
          </CardContent>
        </Card>

        {/* Workers Card */}
        <Card>
          <CardHeader>
            <CardTitle>Workers</CardTitle>
            <CardDescription>Active executors</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {workersOnline}
            </p>
            <p className="text-sm text-muted-foreground">
              {workersAvailable} available
            </p>
            {workerIds.length > 0 && (
              <ul className="mt-2 space-y-1">
                {workerIds.slice(0, 5).map((id) => (
                  <li key={id} className="text-xs text-muted-foreground">
                    {id}
                  </li>
                ))}
                {workerIds.length > 5 && (
                  <li className="text-xs text-muted-foreground">
                    +{workerIds.length - 5} more
                  </li>
                )}
              </ul>
            )}
            <Button className="mt-4" variant="outline" size="sm">
              Spawn Worker
            </Button>
          </CardContent>
        </Card>

        {/* Models Card */}
        <Card>
          <CardHeader>
            <CardTitle>Models</CardTitle>
            <CardDescription>Downloaded models</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">0</p>
            <p className="text-sm text-muted-foreground">No models available</p>
            <Button className="mt-4" variant="outline" size="sm">
              Download Model
            </Button>
          </CardContent>
        </Card>

        {/* Inference Card */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Quick Inference</CardTitle>
            <CardDescription>Test model with a prompt</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              {connected
                ? 'Ready for inference (feature coming soon)'
                : 'Connect to queen to start inference'}
            </p>
            <Button variant="default" disabled={!connected}>
              Run Inference
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
