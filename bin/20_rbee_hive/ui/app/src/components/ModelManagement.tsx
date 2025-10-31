// Model Management - Comprehensive model lifecycle management
// Features: Download, Load to RAM, Search HuggingFace, Deploy to Workers

import { useState } from 'react'
import { HardDrive, Search, Download, Play, Trash2, Info } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Badge } from '@rbee/ui/atoms'
import { useModels, useModelOperations } from '@rbee/rbee-hive-react'

type ViewMode = 'downloaded' | 'loaded' | 'search'

interface Model {
  id: string
  name: string
  size: number
  status: string
  loaded?: boolean
  vram_mb?: number
}

export function ModelManagement() {
  const { models, loading, error } = useModels()
  const [viewMode, setViewMode] = useState<ViewMode>('downloaded')
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  
  // Use proper model operations hook from rbee-hive-react
  const { loadModel, unloadModel, deleteModel, isPending } = useModelOperations()

  // Filter models by state
  const downloadedModels = models.filter((m: Model) => !m.loaded)
  const loadedModels = models.filter((m: Model) => m.loaded)

  // Model operations - delegate to hook
  const handleLoadModel = (modelId: string, device: string = 'cuda:0') => {
    loadModel({ modelId, device })
  }

  const handleUnloadModel = (modelId: string) => {
    unloadModel({ modelId })
  }

  const handleDeleteModel = (modelId: string) => {
    deleteModel({ modelId })
  }

  return (
    <Card className="col-span-2">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Model Management
          </CardTitle>
          <div className="flex gap-2">
            <Badge variant="secondary">{downloadedModels.length} Downloaded</Badge>
            <Badge variant="default">{loadedModels.length} Loaded</Badge>
          </div>
        </div>
        <CardDescription>
          Download models, load to RAM, and deploy to workers
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* View Mode Tabs */}
        <div className="flex gap-2 border-b">
          <button
            onClick={() => setViewMode('downloaded')}
            className={`px-4 py-2 border-b-2 transition-colors ${
              viewMode === 'downloaded'
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            <HardDrive className="h-4 w-4 inline mr-2" />
            Downloaded ({downloadedModels.length})
          </button>
          <button
            onClick={() => setViewMode('loaded')}
            className={`px-4 py-2 border-b-2 transition-colors ${
              viewMode === 'loaded'
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            <Play className="h-4 w-4 inline mr-2" />
            Loaded in RAM ({loadedModels.length})
          </button>
          <button
            onClick={() => setViewMode('search')}
            className={`px-4 py-2 border-b-2 transition-colors ${
              viewMode === 'search'
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            <Search className="h-4 w-4 inline mr-2" />
            Search HuggingFace
          </button>
        </div>

        {/* Search Bar (shown in search mode) */}
        {viewMode === 'search' && (
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search HuggingFace models..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-md bg-background"
            />
          </div>
        )}

        {/* Main Content Area */}
        <div className="grid grid-cols-3 gap-4">
          {/* Model List (2/3 width) */}
          <div className="col-span-2 space-y-2">
            {loading && (
              <div className="text-center py-12 text-muted-foreground">
                Loading models...
              </div>
            )}

            {error && (
              <div className="text-center py-12 text-destructive">
                Error: {error.message}
              </div>
            )}

            {!loading && !error && (
              <>
                {viewMode === 'downloaded' && (
                  <DownloadedModelsTable
                    models={downloadedModels}
                    selectedModel={selectedModel}
                    onSelect={setSelectedModel}
                    onLoad={handleLoadModel}
                    onDelete={handleDeleteModel}
                  />
                )}

                {viewMode === 'loaded' && (
                  <LoadedModelsTable
                    models={loadedModels}
                    selectedModel={selectedModel}
                    onSelect={setSelectedModel}
                    onUnload={handleUnloadModel}
                  />
                )}

                {viewMode === 'search' && (
                  <SearchResultsTable
                    query={searchQuery}
                    onSelect={setSelectedModel}
                  />
                )}
              </>
            )}
          </div>

          {/* Model Details Panel (1/3 width) */}
          <div className="col-span-1">
            <ModelDetailsPanel model={selectedModel} />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// Downloaded Models Table
function DownloadedModelsTable({
  models,
  selectedModel,
  onSelect,
  onLoad,
  onDelete,
}: {
  models: Model[]
  selectedModel: Model | null
  onSelect: (model: Model) => void
  onLoad: (modelId: string) => void
  onDelete: (modelId: string) => void
}) {
  if (models.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No models downloaded yet. Search HuggingFace to download models.
      </div>
    )
  }

  return (
    <div className="border rounded-md">
      <table className="w-full">
        <thead className="bg-muted/50">
          <tr className="text-left text-sm">
            <th className="p-3 font-medium">Model</th>
            <th className="p-3 font-medium">Size</th>
            <th className="p-3 font-medium">Status</th>
            <th className="p-3 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr
              key={model.id}
              onClick={() => onSelect(model)}
              className={`border-t cursor-pointer hover:bg-accent/50 transition-colors ${
                selectedModel?.id === model.id ? 'bg-accent' : ''
              }`}
            >
              <td className="p-3">
                <div className="font-medium">{model.name || model.id}</div>
                <div className="text-xs text-muted-foreground">{model.id}</div>
              </td>
              <td className="p-3 text-sm">
                {(model.size / 1_000_000_000).toFixed(2)} GB
              </td>
              <td className="p-3">
                <Badge variant={model.status === 'available' ? 'default' : 'secondary'}>
                  {model.status}
                </Badge>
              </td>
              <td className="p-3">
                <div className="flex gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onLoad(model.id)
                    }}
                    className="p-1 hover:bg-primary/10 rounded"
                    title="Load to RAM"
                  >
                    <Play className="h-4 w-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(model.id)
                    }}
                    className="p-1 hover:bg-destructive/10 rounded text-destructive"
                    title="Delete"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Loaded Models Table (models in RAM)
function LoadedModelsTable({
  models,
  selectedModel,
  onSelect,
  onUnload,
}: {
  models: Model[]
  selectedModel: Model | null
  onSelect: (model: Model) => void
  onUnload: (modelId: string) => void
}) {
  if (models.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No models loaded in RAM. Load a downloaded model to start inference.
      </div>
    )
  }

  return (
    <div className="border rounded-md">
      <table className="w-full">
        <thead className="bg-muted/50">
          <tr className="text-left text-sm">
            <th className="p-3 font-medium">Model</th>
            <th className="p-3 font-medium">VRAM</th>
            <th className="p-3 font-medium">Worker</th>
            <th className="p-3 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr
              key={model.id}
              onClick={() => onSelect(model)}
              className={`border-t cursor-pointer hover:bg-accent/50 transition-colors ${
                selectedModel?.id === model.id ? 'bg-accent' : ''
              }`}
            >
              <td className="p-3">
                <div className="font-medium">{model.name || model.id}</div>
                <div className="text-xs text-muted-foreground">Ready for inference</div>
              </td>
              <td className="p-3 text-sm">
                {model.vram_mb ? `${(model.vram_mb / 1024).toFixed(1)} GB` : 'N/A'}
              </td>
              <td className="p-3">
                <Badge variant="outline">worker-gpu-0</Badge>
              </td>
              <td className="p-3">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onUnload(model.id)
                  }}
                  className="p-1 hover:bg-destructive/10 rounded text-destructive"
                  title="Unload from RAM"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Search Results Table (HuggingFace search)
function SearchResultsTable({
  query,
  onSelect,
}: {
  query: string
  onSelect: (model: any) => void
}) {
  // TODO: Implement HuggingFace search using @huggingface/hub
  // const { models, loading } = useHuggingFaceSearch(query)

  if (!query || query.length < 2) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        Enter a search query to find models on HuggingFace
      </div>
    )
  }

  return (
    <div className="text-center py-12 text-muted-foreground">
      Search results for "{query}" will appear here
      <div className="text-xs mt-2">
        TODO: Integrate @huggingface/hub package
      </div>
    </div>
  )
}

// Model Details Panel
function ModelDetailsPanel({ model }: { model: Model | null }) {
  if (!model) {
    return (
      <div className="border rounded-md p-6 text-center text-muted-foreground">
        <Info className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">Select a model to view details</p>
      </div>
    )
  }

  return (
    <div className="border rounded-md p-4 space-y-4">
      <div>
        <h3 className="font-semibold mb-2">Model Details</h3>
        <div className="space-y-2 text-sm">
          <div>
            <span className="text-muted-foreground">ID:</span>
            <div className="font-mono text-xs break-all">{model.id}</div>
          </div>
          <div>
            <span className="text-muted-foreground">Name:</span>
            <div>{model.name}</div>
          </div>
          <div>
            <span className="text-muted-foreground">Size:</span>
            <div>{(model.size / 1_000_000_000).toFixed(2)} GB</div>
          </div>
          <div>
            <span className="text-muted-foreground">Status:</span>
            <div>
              <Badge variant={model.status === 'available' ? 'default' : 'secondary'}>
                {model.status}
              </Badge>
            </div>
          </div>
        </div>
      </div>

      <div className="pt-4 border-t space-y-2">
        <h4 className="font-semibold text-sm">Operations</h4>
        
        {!model.loaded && (
          <button className="w-full px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors text-sm">
            <Play className="h-4 w-4 inline mr-2" />
            Load to RAM
          </button>
        )}

        {model.loaded && (
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">
              Deploy to Worker:
            </div>
            <select className="w-full px-3 py-2 border rounded-md text-sm">
              <option>GPU-0 (Available)</option>
              <option>GPU-1 (Available)</option>
              <option>CPU-0 (Available)</option>
            </select>
            <button className="w-full px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors text-sm">
              Start Inference
            </button>
          </div>
        )}

        <button className="w-full px-3 py-2 border border-destructive text-destructive rounded-md hover:bg-destructive/10 transition-colors text-sm">
          <Trash2 className="h-4 w-4 inline mr-2" />
          {model.loaded ? 'Unload from RAM' : 'Delete Model'}
        </button>
      </div>
    </div>
  )
}
