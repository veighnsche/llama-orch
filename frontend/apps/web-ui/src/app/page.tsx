// TEAM-288: Live heartbeat monitoring dashboard

'use client';

import { useHeartbeat } from '@/src/hooks/useHeartbeat';
import { Button } from '@rbee/ui/atoms';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms';
import { ThemeToggle } from '@rbee/ui/molecules';

export default function HomePage() {
  const { heartbeat, connected, loading, error } = useHeartbeat();

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-background p-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">🐝 rbee Web UI</h1>
          <p className="text-muted-foreground">Loading rbee SDK...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-background p-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4 text-red-500">⚠️ Error</h1>
          <p className="text-muted-foreground mb-4">Failed to load rbee SDK</p>
          <p className="text-sm text-red-500">{error.message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <header className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2">🐝 rbee Web UI</h1>
          <p className="text-muted-foreground">
            Dashboard for managing queen, hives, workers, and models
          </p>
        </div>
        <ThemeToggle />
      </header>

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
            {heartbeat && (
              <p className="text-xs text-muted-foreground mt-2">
                Last update: {new Date(heartbeat.timestamp).toLocaleTimeString()}
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
              {heartbeat?.hives_online ?? 0}
            </p>
            <p className="text-sm text-muted-foreground">
              {heartbeat?.hives_available ?? 0} available
            </p>
            {heartbeat?.hive_ids && heartbeat.hive_ids.length > 0 && (
              <ul className="mt-2 space-y-1">
                {heartbeat.hive_ids.map((id) => (
                  <li key={id} className="text-xs text-muted-foreground">
                    {id}
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
              {heartbeat?.workers_online ?? 0}
            </p>
            <p className="text-sm text-muted-foreground">
              {heartbeat?.workers_available ?? 0} available
            </p>
            {heartbeat?.worker_ids && heartbeat.worker_ids.length > 0 && (
              <ul className="mt-2 space-y-1">
                {heartbeat.worker_ids.slice(0, 5).map((id) => (
                  <li key={id} className="text-xs text-muted-foreground">
                    {id}
                  </li>
                ))}
                {heartbeat.worker_ids.length > 5 && (
                  <li className="text-xs text-muted-foreground">
                    +{heartbeat.worker_ids.length - 5} more
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

      <footer className="mt-8 text-center text-sm text-muted-foreground">
        <p>rbee Web UI v0.1.0 - TEAM-288</p>
        <p className="mt-1">
          Status: Live heartbeat monitoring active
          {heartbeat && ` • Updates every 5 seconds`}
        </p>
      </footer>
    </div>
  );
}
