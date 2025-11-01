// TEAM-382: Spawn worker form view
// Enhanced with worker catalog integration for build instructions

import { useState } from 'react'
import { Rocket, Cpu, AlertCircle, Info } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Input,
  Badge,
} from '@rbee/ui/atoms'
import { WORKER_TYPE_OPTIONS } from '@rbee/rbee-hive-react'
import { useWorkerCatalog, getWorkerByType, isWorkerSupported, getCurrentPlatform } from '../../hooks/useWorkerCatalog'
import type { ModelInfo } from '../ModelManagement/types'
import type { SpawnFormState } from './types'

interface SpawnWorkerViewProps {
  models: ModelInfo[]
  onSpawn: (params: SpawnFormState) => void
  isPending: boolean
}

export function SpawnWorkerView({ models, onSpawn, isPending }: SpawnWorkerViewProps) {
  // Fetch worker catalog for build instructions and metadata
  const { data: catalog, isLoading: catalogLoading, error: catalogError } = useWorkerCatalog()
  const currentPlatform = getCurrentPlatform()
  
  const [formState, setFormState] = useState<SpawnFormState>({
    modelId: '',
    workerType: 'cuda',
    deviceId: 0,
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (formState.modelId) {
      onSpawn(formState)
    }
  }

  // Only show downloaded models
  const availableModels = models.filter((m) => !m.loaded)

  return (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Rocket className="h-5 w-5" />
          Spawn New Worker
        </CardTitle>
        <CardDescription>
          Select a model and device to spawn a new worker process
        </CardDescription>
      </CardHeader>

      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Model Selection */}
          <div className="space-y-2">
            <Label htmlFor="model">Model</Label>
            <Select
              value={formState.modelId}
              onValueChange={(value) =>
                setFormState((prev) => ({ ...prev, modelId: value }))
              }
            >
              <SelectTrigger id="model">
                <SelectValue placeholder="Select a model..." />
              </SelectTrigger>
              <SelectContent>
                {availableModels.length === 0 ? (
                  <div className="px-2 py-6 text-center text-sm text-muted-foreground">
                    No downloaded models available
                  </div>
                ) : (
                  availableModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name || model.id}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>

          {/* Worker Type Selection */}
          <div className="space-y-2">
            <Label htmlFor="worker-type">Worker Type</Label>
            <Select
              value={formState.workerType}
              onValueChange={(value: 'cpu' | 'cuda' | 'metal') =>
                setFormState((prev) => ({ ...prev, workerType: value }))
              }
            >
              <SelectTrigger id="worker-type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {WORKER_TYPE_OPTIONS.map((option) => {
                  const workerEntry = getWorkerByType(catalog, option.value)
                  const supported = workerEntry ? isWorkerSupported(workerEntry, currentPlatform) : true
                  
                  return (
                    <SelectItem 
                      key={option.value} 
                      value={option.value}
                      disabled={!supported}
                    >
                      <div className="flex items-center justify-between gap-2 w-full">
                        <div className="flex flex-col">
                          <span>{option.label}</span>
                          <span className="text-xs text-muted-foreground">
                            {option.description}
                          </span>
                          {workerEntry && (
                            <span className="text-xs text-muted-foreground mt-1">
                              v{workerEntry.version} • {workerEntry.supported_formats.join(', ')}
                            </span>
                          )}
                        </div>
                        {!supported && (
                          <Badge variant="outline" className="text-xs">
                            <AlertCircle className="h-3 w-3 mr-1" />
                            Not supported on {currentPlatform}
                          </Badge>
                        )}
                      </div>
                    </SelectItem>
                  )
                })}
              </SelectContent>
            </Select>
            
            {/* Catalog Loading/Error State */}
            {catalogLoading && (
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <Info className="h-3 w-3" />
                Loading worker catalog...
              </p>
            )}
            {catalogError && (
              <p className="text-xs text-destructive flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                Failed to load catalog: {catalogError.message}
              </p>
            )}
            
            {/* Worker Details */}
            {formState.workerType && catalog && (
              <div className="mt-2 p-3 bg-muted rounded-md text-xs space-y-1">
                {(() => {
                  const worker = getWorkerByType(catalog, formState.workerType)
                  if (!worker) return null
                  
                  return (
                    <>
                      <div className="flex items-center gap-2">
                        <Info className="h-3 w-3" />
                        <span className="font-medium">{worker.name}</span>
                        <Badge variant="secondary" className="text-xs">
                          {worker.implementation}
                        </Badge>
                      </div>
                      <p className="text-muted-foreground">{worker.description}</p>
                      <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t">
                        <div>
                          <span className="text-muted-foreground">Build:</span> {worker.build_system}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Streaming:</span> {worker.supports_streaming ? '✓' : '✗'}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Max Context:</span> {worker.max_context_length?.toLocaleString() || 'N/A'}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Platforms:</span> {worker.platforms.join(', ')}
                        </div>
                      </div>
                    </>
                  )
                })()}
              </div>
            )}
          </div>

          {/* Device ID (for CUDA/Metal) */}
          {formState.workerType !== 'cpu' && (
            <div className="space-y-2">
              <Label htmlFor="device-id">Device ID</Label>
              <Input
                id="device-id"
                type="number"
                min="0"
                value={formState.deviceId}
                onChange={(e) =>
                  setFormState((prev) => ({
                    ...prev,
                    deviceId: parseInt(e.target.value) || 0,
                  }))
                }
              />
              <p className="text-xs text-muted-foreground">
                GPU device index (0 for first GPU, 1 for second, etc.)
              </p>
            </div>
          )}

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full"
            disabled={!formState.modelId || isPending}
          >
            {isPending ? (
              <>
                <Cpu className="h-4 w-4 mr-2 animate-pulse" />
                Spawning Worker...
              </>
            ) : (
              <>
                <Rocket className="h-4 w-4 mr-2" />
                Spawn Worker
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
