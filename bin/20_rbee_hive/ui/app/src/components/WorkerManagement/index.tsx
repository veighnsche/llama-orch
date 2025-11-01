// TEAM-382: Worker Management - Main component with clean composition

import { useState } from 'react'
import { Server, Plus, Activity } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from '@rbee/ui/atoms'
import { useWorkers, useHiveOperations } from '@rbee/rbee-hive-react'
import { useModels } from '@rbee/rbee-hive-react'
import { ActiveWorkersView } from './ActiveWorkersView'
import { SpawnWorkerView } from './SpawnWorkerView'
import type { ViewMode, SpawnFormState } from './types'

export function WorkerManagement() {
  const { workers, loading, error } = useWorkers()
  const { models } = useModels()
  const { spawnWorker, isPending } = useHiveOperations()
  const [viewMode, setViewMode] = useState<ViewMode>('active')

  // Separate idle and active workers
  const idleWorkers = workers.filter((w: any) => w.gpu_util_pct === 0.0)
  const activeWorkers = workers.filter((w: any) => w.gpu_util_pct > 0.0)

  const handleSpawnWorker = (params: SpawnFormState) => {
    spawnWorker({
      modelId: params.modelId,
      workerType: params.workerType,
      deviceId: params.deviceId,
    })
  }

  return (
    <Card className="col-span-2">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Worker Management
            </CardTitle>
            <CardDescription>
              Spawn workers, monitor performance, and manage processes
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Badge variant="secondary">{idleWorkers.length} Idle</Badge>
            <Badge variant="default">{activeWorkers.length} Active</Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* View Mode Tabs */}
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="active">
              <Activity className="h-4 w-4 mr-2" />
              Active Workers ({workers.length})
            </TabsTrigger>
            <TabsTrigger value="spawn">
              <Plus className="h-4 w-4 mr-2" />
              Spawn Worker
            </TabsTrigger>
          </TabsList>

          {/* Active Workers Tab */}
          <TabsContent value="active" className="space-y-4">
            <ActiveWorkersView
              workers={workers}
              loading={loading}
              error={error}
            />
          </TabsContent>

          {/* Spawn Worker Tab */}
          <TabsContent value="spawn" className="space-y-4">
            <SpawnWorkerView
              models={models}
              onSpawn={handleSpawnWorker}
              isPending={isPending}
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

// Re-export types for convenience
export type { ViewMode, SpawnFormState } from './types'
