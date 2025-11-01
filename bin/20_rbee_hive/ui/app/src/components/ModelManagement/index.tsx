// TEAM-381: Model Management - Main component with clean composition

import { useState } from 'react'
import { HardDrive, Search, Play, Filter } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  Input,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from '@rbee/ui/atoms'
import { useModels, useModelOperations } from '@rbee/rbee-hive-react'
import { DownloadedModelsView } from './DownloadedModelsView'
import { LoadedModelsView } from './LoadedModelsView'
import { SearchResultsView } from './SearchResultsView'
import { FilterPanel } from './FilterPanel'
import { ModelDetailsPanel } from './ModelDetailsPanel'
import type { ViewMode, ModelInfo, FilterState } from './types'

export function ModelManagement() {
  const { models, loading, error } = useModels()
  const [viewMode, setViewMode] = useState<ViewMode>('downloaded')
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  
  // TEAM-381: Default filters for MVP (GGUF, LLaMA/Mistral/Phi)
  const [filters, setFilters] = useState<FilterState>({
    formats: ['gguf'],
    architectures: ['llama', 'mistral', 'phi'],
    maxSize: '15gb',
    openSourceOnly: true,
    sortBy: 'downloads',
  })
  
  const { loadModel, unloadModel, deleteModel, isPending } = useModelOperations()

  // Filter models by state
  const downloadedModels = models.filter((m: ModelInfo) => !m.loaded)
  const loadedModels = models.filter((m: ModelInfo) => m.loaded)

  // Model operations
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
          <div>
            <CardTitle className="flex items-center gap-2">
              <HardDrive className="h-5 w-5" />
              Model Management
            </CardTitle>
            <CardDescription>
              Download models, load to RAM, and deploy to workers
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Badge variant="secondary">{downloadedModels.length} Downloaded</Badge>
            <Badge variant="default">{loadedModels.length} Loaded</Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* View Mode Tabs */}
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="downloaded">
              <HardDrive className="h-4 w-4 mr-2" />
              Downloaded ({downloadedModels.length})
            </TabsTrigger>
            <TabsTrigger value="loaded">
              <Play className="h-4 w-4 mr-2" />
              Loaded in RAM ({loadedModels.length})
            </TabsTrigger>
            <TabsTrigger value="search">
              <Search className="h-4 w-4 mr-2" />
              Search HuggingFace
            </TabsTrigger>
          </TabsList>

          {/* Downloaded Models Tab */}
          <TabsContent value="downloaded" className="space-y-4">
            <DownloadedModelsView
              models={downloadedModels}
              loading={loading}
              error={error}
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
              onLoad={handleLoadModel}
              onDelete={handleDeleteModel}
            />
          </TabsContent>

          {/* Loaded Models Tab */}
          <TabsContent value="loaded" className="space-y-4">
            <LoadedModelsView
              models={loadedModels}
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
              onUnload={handleUnloadModel}
            />
          </TabsContent>

          {/* Search HuggingFace Tab */}
          <TabsContent value="search" className="space-y-4">
            <div className="flex gap-2">
              <div className="flex-1">
                <Input
                  type="text"
                  placeholder="Search HuggingFace models (e.g., 'llama', 'mistral', 'phi')..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full"
                />
              </div>
              <Button
                variant={showFilters ? 'default' : 'outline'}
                size="icon"
                onClick={() => setShowFilters(!showFilters)}
              >
                <Filter className="h-4 w-4" />
              </Button>
            </div>

            <div className="grid grid-cols-12 gap-4">
              {/* Filters Sidebar */}
              {showFilters && (
                <div className="col-span-3">
                  <FilterPanel filters={filters} onFiltersChange={setFilters} />
                </div>
              )}

              {/* Search Results */}
              <div className={showFilters ? 'col-span-9' : 'col-span-12'}>
                <SearchResultsView
                  query={searchQuery}
                  filters={filters}
                  onSelect={setSelectedModel}
                />
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Model Details Panel (always visible when model selected) */}
        {selectedModel && (
          <ModelDetailsPanel
            model={selectedModel}
            onLoad={handleLoadModel}
            onUnload={handleUnloadModel}
            onDelete={handleDeleteModel}
            isPending={isPending}
          />
        )}
      </CardContent>
    </Card>
  )
}

// Re-export types for convenience
export type { ViewMode, ModelInfo, HFModel, FilterState } from './types'
