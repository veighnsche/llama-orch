'use client';

import { useState, useEffect } from 'react';
import { Button } from '@rbee/ui/components/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/components/card';

export default function HomePage() {
  const [queenStatus, setQueenStatus] = useState<'connected' | 'disconnected'>('disconnected');

  // TODO: Connect to rbee SDK
  useEffect(() => {
    // Placeholder - will connect to rbee SDK
    console.log('rbee Web UI loaded');
  }, []);

  return (
    <div className="min-h-screen bg-background p-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold mb-2">🐝 rbee Web UI</h1>
        <p className="text-muted-foreground">
          Dashboard for managing queen, hives, workers, and models
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Queen Status Card */}
        <Card>
          <CardHeader>
            <CardTitle>Queen Status</CardTitle>
            <CardDescription>Central orchestrator</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <div
                className={`h-3 w-3 rounded-full ${
                  queenStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="capitalize">{queenStatus}</span>
            </div>
            <Button className="mt-4" variant="outline" size="sm">
              Connect to Queen
            </Button>
          </CardContent>
        </Card>

        {/* Hives Card */}
        <Card>
          <CardHeader>
            <CardTitle>Hives</CardTitle>
            <CardDescription>Pool managers</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">0</p>
            <p className="text-sm text-muted-foreground">No hives configured</p>
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
            <p className="text-2xl font-bold">0</p>
            <p className="text-sm text-muted-foreground">No workers running</p>
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
              Connect to queen to start inference
            </p>
            <Button variant="default" disabled>
              Run Inference
            </Button>
          </CardContent>
        </Card>
      </div>

      <footer className="mt-8 text-center text-sm text-muted-foreground">
        <p>rbee Web UI v0.1.0 (Stub)</p>
        <p className="mt-1">
          Status: Design phase - SDK integration pending
        </p>
      </footer>
    </div>
  );
}
