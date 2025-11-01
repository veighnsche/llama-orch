// TEAM-382: Worker Management - Main component with clean composition
// Updated to focus on worker installation lifecycle first

import { useState } from 'react'
import { Server, Plus, Activity, Package } from 'lucide-react'
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
import { useWorkers, useWorkerOperations } from '@rbee/rbee-hive-react'
import { useModels } from '@rbee/rbee-hive-react'
import { ActiveWorkersView } from './ActiveWorkersView'
import { SpawnWorkerView } from './SpawnWorkerView'
import { WorkerCatalogView } from './WorkerCatalogView'
import type { ViewMode, SpawnFormState } from './types'

export function WorkerManagement() {
  const { workers, loading, error } = useWorkers()
  const { models } = useModels()
  const { spawnWorker, installWorker, isPending, installProgress } = useWorkerOperations()
  const [viewMode, setViewMode] = useState<ViewMode>('catalog') // Start with catalog - install workers first!

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
  
  const handleInstallWorker = async (workerId: string) => {
    // TEAM-378: Call hive backend via SDK → JobClient → Job Server
    console.log('[WorkerManagement] 📥 Received install request for:', workerId)
    console.log('[WorkerManagement] 🚀 Calling useWorkerOperations.installWorker()...')
    installWorker(workerId)
    console.log('[WorkerManagement] ✓ installWorker() called (async operation started)')
  }
  
  const handleRemoveWorker = async (workerId: string) => {
    // TODO: Call hive backend to remove worker
    console.log('Removing worker:', workerId)
    // DELETE /v1/workers/{workerId}
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
              Install workers, monitor performance, and manage processes
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
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="catalog">
              <Package className="h-4 w-4 mr-2" />
              Worker Catalog
            </TabsTrigger>
            <TabsTrigger value="active">
              <Activity className="h-4 w-4 mr-2" />
              Active Workers ({workers.length})
            </TabsTrigger>
            <TabsTrigger value="spawn">
              <Plus className="h-4 w-4 mr-2" />
              Spawn Worker
            </TabsTrigger>
          </TabsList>

          {/* Worker Catalog Tab - FIRST PRIORITY */}
          <TabsContent value="catalog" className="space-y-4">
            <WorkerCatalogView
              onInstall={handleInstallWorker}
              onRemove={handleRemoveWorker}
              installProgress={installProgress}
            />
          </TabsContent>

          {/* Active Workers Tab */}
          <TabsContent value="active" className="space-y-4">
            <ActiveWorkersView
              workers={workers}
              loading={loading}
              error={error}
            />
          </TabsContent>

          {/* Spawn Worker Tab - REQUIRES INSTALLED WORKERS + MODELS */}
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
