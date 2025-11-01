// Worker Catalog View - Browse, Install, and Remove Workers
// Shows available workers from catalog and installed workers

import { useState } from 'react'
import { useWorkerCatalog, type WorkerCatalogEntry, getCurrentPlatform } from '../../hooks/useWorkerCatalog'
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle,
  CardDescription,
  CardFooter 
} from '@rbee/ui/atoms/Card'
import { Button } from '@rbee/ui/atoms/Button'
import { Badge } from '@rbee/ui/atoms/Badge'
import { 
  Download, 
  Trash2, 
  CheckCircle, 
  AlertCircle, 
  Info,
  Cpu,
  Zap,
  Apple,
  Loader2
} from 'lucide-react'

interface WorkerCatalogViewProps {
  onInstall?: (workerId: string) => void
  onRemove?: (workerId: string) => void
}

export function WorkerCatalogView({ onInstall, onRemove }: WorkerCatalogViewProps) {
  const { data: catalog, isLoading, error } = useWorkerCatalog()
  const currentPlatform = getCurrentPlatform()
  const [installingWorker, setInstallingWorker] = useState<string | null>(null)
  const [removingWorker, setRemovingWorker] = useState<string | null>(null)
  
  // TODO: Track which workers are actually installed
  // For now, mock it - in real implementation, query hive backend
  const [installedWorkers] = useState<Set<string>>(new Set())
  
  const handleInstall = async (workerId: string) => {
    setInstallingWorker(workerId)
    try {
      await onInstall?.(workerId)
      // TODO: Add to installedWorkers set
    } finally {
      setInstallingWorker(null)
    }
  }
  
  const handleRemove = async (workerId: string) => {
    setRemovingWorker(workerId)
    try {
      await onRemove?.(workerId)
      // TODO: Remove from installedWorkers set
    } finally {
      setRemovingWorker(null)
    }
  }
  
  const getWorkerIcon = (workerType: string) => {
    switch (workerType) {
      case 'cpu':
        return <Cpu className="h-5 w-5" />
      case 'cuda':
        return <Zap className="h-5 w-5" />
      case 'metal':
        return <Apple className="h-5 w-5" />
      default:
        return <Cpu className="h-5 w-5" />
    }
  }
  
  const isWorkerSupported = (worker: WorkerCatalogEntry): boolean => {
    return worker.platforms.includes(currentPlatform)
  }
  
  const isWorkerInstalled = (workerId: string): boolean => {
    return installedWorkers.has(workerId)
  }
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-3 text-muted-foreground">Loading worker catalog...</span>
      </div>
    )
  }
  
  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-5 w-5" />
            Failed to Load Worker Catalog
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{error.message}</p>
          <p className="text-sm text-muted-foreground mt-2">
            Make sure the worker catalog service is running on port 8787.
          </p>
        </CardContent>
      </Card>
    )
  }
  
  if (!catalog || catalog.workers.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No Workers Available</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No workers found in the catalog.
          </p>
        </CardContent>
      </Card>
    )
  }
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Worker Catalog</h2>
        <p className="text-muted-foreground mt-1">
          Download, build, and install worker binaries on your system
        </p>
      </div>
      
      {/* Platform Info */}
      <Card className="bg-muted/50">
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 text-sm">
            <Info className="h-4 w-4" />
            <span>
              Detected platform: <strong>{currentPlatform}</strong>
            </span>
            <span className="text-muted-foreground">
              • Only compatible workers are shown
            </span>
          </div>
        </CardContent>
      </Card>
      
      {/* Worker Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {catalog.workers
          .filter(isWorkerSupported)
          .map((worker) => {
            const installed = isWorkerInstalled(worker.id)
            const installing = installingWorker === worker.id
            const removing = removingWorker === worker.id
            const busy = installing || removing
            
            return (
              <Card key={worker.id} className={installed ? 'border-primary' : ''}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      {getWorkerIcon(worker.worker_type)}
                      <div>
                        <CardTitle className="text-lg">{worker.name}</CardTitle>
                        <CardDescription className="text-xs mt-1">
                          v{worker.version}
                        </CardDescription>
                      </div>
                    </div>
                    {installed && (
                      <Badge variant="default" className="gap-1">
                        <CheckCircle className="h-3 w-3" />
                        Installed
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  {/* Description */}
                  <p className="text-sm text-muted-foreground">
                    {worker.description}
                  </p>
                  
                  {/* Metadata Grid */}
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="text-muted-foreground">Type:</span>
                      <div className="font-medium mt-1">
                        {worker.worker_type.toUpperCase()}
                      </div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Build:</span>
                      <div className="font-medium mt-1">
                        {worker.build_system}
                      </div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Formats:</span>
                      <div className="font-medium mt-1">
                        {worker.supported_formats.join(', ')}
                      </div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Streaming:</span>
                      <div className="font-medium mt-1">
                        {worker.supports_streaming ? '✓ Yes' : '✗ No'}
                      </div>
                    </div>
                  </div>
                  
                  {/* Dependencies */}
                  {worker.depends.length > 0 && (
                    <div className="pt-3 border-t">
                      <div className="text-xs text-muted-foreground mb-2">
                        Runtime Dependencies:
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {worker.depends.map((dep) => (
                          <Badge key={dep} variant="outline" className="text-xs">
                            {dep}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Build Dependencies */}
                  {worker.makedepends.length > 0 && (
                    <div className="pt-3 border-t">
                      <div className="text-xs text-muted-foreground mb-2">
                        Build Dependencies:
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {worker.makedepends.map((dep) => (
                          <Badge key={dep} variant="secondary" className="text-xs">
                            {dep}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
                
                <CardFooter className="flex gap-2">
                  {!installed ? (
                    <Button
                      className="flex-1"
                      onClick={() => handleInstall(worker.id)}
                      disabled={busy}
                    >
                      {installing ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Installing...
                        </>
                      ) : (
                        <>
                          <Download className="h-4 w-4 mr-2" />
                          Install Worker
                        </>
                      )}
                    </Button>
                  ) : (
                    <>
                      <Button
                        variant="outline"
                        className="flex-1"
                        disabled
                      >
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Installed
                      </Button>
                      <Button
                        variant="destructive"
                        size="icon"
                        onClick={() => handleRemove(worker.id)}
                        disabled={busy}
                      >
                        {removing ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                    </>
                  )}
                </CardFooter>
              </Card>
            )
          })}
      </div>
      
      {/* No Compatible Workers */}
      {catalog.workers.filter(isWorkerSupported).length === 0 && (
        <Card className="border-warning">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-warning">
              <AlertCircle className="h-5 w-5" />
              No Compatible Workers
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              No workers in the catalog are compatible with your platform ({currentPlatform}).
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              Available workers: {catalog.workers.map(w => `${w.name} (${w.platforms.join(', ')})`).join(', ')}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
