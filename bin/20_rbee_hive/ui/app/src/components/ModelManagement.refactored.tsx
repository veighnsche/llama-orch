// TEAM-381: Model Management - Refactored with reusable UI components
// Features: Download, Load to RAM, Search HuggingFace with smart filtering

import { useState, useEffect } from 'react'
import { HardDrive, Search, Download, Play, Trash2, Info, Loader2, Filter, X } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  Input,
  Checkbox,
  RadioGroup,
  RadioGroupItem,
  Label,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  Skeleton,
  Empty,
  Spinner,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@rbee/ui/atoms'
import { FilterButton, MetricCard, ProgressBar } from '@rbee/ui/molecules'
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

// TEAM-381: HuggingFace model interface
interface HFModel {
  id: string
  modelId: string
  author: string
  downloads: number
  likes: number
  tags: string[]
  private: boolean
  gated: boolean | string
}

// TEAM-381: Filter state
interface FilterState {
  formats: string[]
  architectures: string[]
  maxSize: string
  openSourceOnly: boolean
  sortBy: 'downloads' | 'likes' | 'recent'
}

export function ModelManagement() {
  const { models, loading, error } = useModels()
  const [viewMode, setViewMode] = useState<ViewMode>('downloaded')
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  
  // TEAM-381: Filter state for HuggingFace search
  const [filters, setFilters] = useState<FilterState>({
    formats: ['gguf'], // Default to GGUF for MVP
    architectures: ['llama', 'mistral', 'phi'],
    maxSize: '15gb',
    openSourceOnly: true,
    sortBy: 'downloads',
  })
  
  const { loadModel, unloadModel, deleteModel, isPending } = useModelOperations()

  // Filter models by state
  const downloadedModels = models.filter((m: Model) => !m.loaded)
  const loadedModels = models.filter((m: Model) => m.loaded)

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

        {/* Model Details Panel (always visible) */}
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

// TEAM-381: Downloaded Models View
function DownloadedModelsView({
  models,
  loading,
  error,
  selectedModel,
  onSelect,
  onLoad,
  onDelete,
}: {
  models: Model[]
  loading: boolean
  error: Error | null
  selectedModel: Model | null
  onSelect: (model: Model) => void
  onLoad: (modelId: string) => void
  onDelete: (modelId: string) => void
}) {
  if (loading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    )
  }

  if (error) {
    return (
      <Empty
        icon={<Info className="h-12 w-12" />}
        title="Error loading models"
        description={error.message}
      />
    )
  }

  if (models.length === 0) {
    return (
      <Empty
        icon={<HardDrive className="h-12 w-12" />}
        title="No models downloaded"
        description="Search HuggingFace to download models"
      />
    )
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Size</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.map((model) => (
          <TableRow
            key={model.id}
            onClick={() => onSelect(model)}
            className={selectedModel?.id === model.id ? 'bg-accent' : 'cursor-pointer'}
          >
            <TableCell>
              <div>
                <div className="font-medium">{model.name || model.id}</div>
                <div className="text-xs text-muted-foreground">{model.id}</div>
              </div>
            </TableCell>
            <TableCell>{(model.size / 1_000_000_000).toFixed(2)} GB</TableCell>
            <TableCell>
              <Badge variant={model.status === 'available' ? 'default' : 'secondary'}>
                {model.status}
              </Badge>
            </TableCell>
            <TableCell className="text-right">
              <div className="flex gap-2 justify-end">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation()
                    onLoad(model.id)
                  }}
                  title="Load to RAM"
                >
                  <Play className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDelete(model.id)
                  }}
                  title="Delete"
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

// TEAM-381: Loaded Models View
function LoadedModelsView({
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
      <Empty
        icon={<Play className="h-12 w-12" />}
        title="No models loaded in RAM"
        description="Load a downloaded model to start inference"
      />
    )
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>VRAM</TableHead>
          <TableHead>Worker</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.map((model) => (
          <TableRow
            key={model.id}
            onClick={() => onSelect(model)}
            className={selectedModel?.id === model.id ? 'bg-accent' : 'cursor-pointer'}
          >
            <TableCell>
              <div>
                <div className="font-medium">{model.name || model.id}</div>
                <div className="text-xs text-muted-foreground">Ready for inference</div>
              </div>
            </TableCell>
            <TableCell>
              {model.vram_mb ? `${(model.vram_mb / 1024).toFixed(1)} GB` : 'N/A'}
            </TableCell>
            <TableCell>
              <Badge variant="outline">worker-gpu-0</Badge>
            </TableCell>
            <TableCell className="text-right">
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation()
                  onUnload(model.id)
                }}
                title="Unload from RAM"
              >
                <Trash2 className="h-4 w-4 text-destructive" />
              </Button>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

// TEAM-381: Filter Panel
function FilterPanel({
  filters,
  onFiltersChange,
}: {
  filters: FilterState
  onFiltersChange: (filters: FilterState) => void
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Filters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Format Filter */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Format</Label>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="format-gguf"
                checked={filters.formats.includes('gguf')}
                onCheckedChange={(checked) => {
                  const newFormats = checked
                    ? [...filters.formats, 'gguf']
                    : filters.formats.filter((f) => f !== 'gguf')
                  onFiltersChange({ ...filters, formats: newFormats })
                }}
              />
              <Label htmlFor="format-gguf" className="text-sm font-normal">
                GGUF (Quantized)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="format-safetensors"
                checked={filters.formats.includes('safetensors')}
                onCheckedChange={(checked) => {
                  const newFormats = checked
                    ? [...filters.formats, 'safetensors']
                    : filters.formats.filter((f) => f !== 'safetensors')
                  onFiltersChange({ ...filters, formats: newFormats })
                }}
              />
              <Label htmlFor="format-safetensors" className="text-sm font-normal">
                SafeTensors (Full)
              </Label>
            </div>
          </div>
        </div>

        {/* Architecture Filter */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Architecture</Label>
          <div className="space-y-2">
            {['llama', 'mistral', 'phi', 'gemma', 'qwen'].map((arch) => (
              <div key={arch} className="flex items-center space-x-2">
                <Checkbox
                  id={`arch-${arch}`}
                  checked={filters.architectures.includes(arch)}
                  onCheckedChange={(checked) => {
                    const newArchs = checked
                      ? [...filters.architectures, arch]
                      : filters.architectures.filter((a) => a !== arch)
                    onFiltersChange({ ...filters, architectures: newArchs })
                  }}
                />
                <Label htmlFor={`arch-${arch}`} className="text-sm font-normal capitalize">
                  {arch}
                </Label>
              </div>
            ))}
          </div>
        </div>

        {/* Size Filter */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Max Size</Label>
          <RadioGroup
            value={filters.maxSize}
            onValueChange={(value) => onFiltersChange({ ...filters, maxSize: value })}
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="5gb" id="size-5gb" />
              <Label htmlFor="size-5gb" className="text-sm font-normal">
                &lt; 5GB (Small)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="15gb" id="size-15gb" />
              <Label htmlFor="size-15gb" className="text-sm font-normal">
                &lt; 15GB (Medium)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="30gb" id="size-30gb" />
              <Label htmlFor="size-30gb" className="text-sm font-normal">
                &lt; 30GB (Large)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="all" id="size-all" />
              <Label htmlFor="size-all" className="text-sm font-normal">
                All
              </Label>
            </div>
          </RadioGroup>
        </div>

        {/* License Filter */}
        <div className="flex items-center space-x-2">
          <Checkbox
            id="open-source"
            checked={filters.openSourceOnly}
            onCheckedChange={(checked) =>
              onFiltersChange({ ...filters, openSourceOnly: !!checked })
            }
          />
          <Label htmlFor="open-source" className="text-sm font-normal">
            Open Source Only
          </Label>
        </div>

        {/* Sort By */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Sort By</Label>
          <Select
            value={filters.sortBy}
            onValueChange={(value) =>
              onFiltersChange({ ...filters, sortBy: value as FilterState['sortBy'] })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="downloads">Downloads</SelectItem>
              <SelectItem value="likes">Likes</SelectItem>
              <SelectItem value="recent">Recent</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  )
}

// TEAM-381: Search Results View with filtering
function SearchResultsView({
  query,
  filters,
  onSelect,
}: {
  query: string
  filters: FilterState
  onSelect: (model: any) => void
}) {
  const [results, setResults] = useState<HFModel[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // TEAM-381: Detect architecture from model ID and tags
  const detectArchitecture = (modelId: string, tags: string[]): string[] => {
    const architectures: string[] = []
    const lowerModelId = modelId.toLowerCase()
    
    if (/llama|alpaca|vicuna/i.test(lowerModelId)) architectures.push('llama')
    if (/mistral|mixtral/i.test(lowerModelId)) architectures.push('mistral')
    if (/phi/i.test(lowerModelId)) architectures.push('phi')
    if (/gemma/i.test(lowerModelId)) architectures.push('gemma')
    if (/qwen/i.test(lowerModelId)) architectures.push('qwen')
    
    tags.forEach((tag) => {
      const lowerTag = tag.toLowerCase()
      if (lowerTag.includes('llama')) architectures.push('llama')
      if (lowerTag.includes('mistral')) architectures.push('mistral')
      if (lowerTag.includes('phi')) architectures.push('phi')
      if (lowerTag.includes('gemma')) architectures.push('gemma')
      if (lowerTag.includes('qwen')) architectures.push('qwen')
    })
    
    return [...new Set(architectures)]
  }

  // TEAM-381: Detect format from model ID and tags
  const detectFormat = (modelId: string, tags: string[]): string[] => {
    const formats: string[] = []
    
    if (tags.includes('gguf') || /\.gguf/i.test(modelId)) {
      formats.push('gguf')
    }
    
    if (tags.includes('safetensors') || tags.includes('pytorch')) {
      formats.push('safetensors')
    }
    
    return formats
  }

  // TEAM-381: Filter results client-side
  const filterResults = (models: HFModel[]): HFModel[] => {
    return models.filter((model) => {
      // Architecture filter
      const modelArchs = detectArchitecture(model.modelId, model.tags)
      const hasMatchingArch =
        filters.architectures.length === 0 ||
        modelArchs.some((arch) => filters.architectures.includes(arch))
      
      if (!hasMatchingArch) return false
      
      // Format filter
      const modelFormats = detectFormat(model.modelId, model.tags)
      const hasMatchingFormat =
        filters.formats.length === 0 ||
        modelFormats.some((fmt) => filters.formats.includes(fmt))
      
      if (!hasMatchingFormat) return false
      
      // License filter (basic check)
      if (filters.openSourceOnly && model.gated) return false
      
      return true
    })
  }

  // Fetch from HuggingFace
  useEffect(() => {
    if (!query || query.length < 2) {
      setResults([])
      return
    }

    const searchHF = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const response = await fetch(
          `https://huggingface.co/api/models?search=${encodeURIComponent(query)}&limit=50&filter=text-generation`,
          {
            headers: {
              'Accept': 'application/json',
            },
          }
        )

        if (!response.ok) {
          throw new Error(`HuggingFace API error: ${response.status}`)
        }

        const data = await response.json()
        const filtered = filterResults(data)
        
        // Sort results
        filtered.sort((a, b) => {
          if (filters.sortBy === 'downloads') return b.downloads - a.downloads
          if (filters.sortBy === 'likes') return b.likes - a.likes
          return 0 // 'recent' would need lastModified field
        })
        
        setResults(filtered)
      } catch (err) {
        console.error('HuggingFace search error:', err)
        setError(err instanceof Error ? err.message : 'Failed to search HuggingFace')
      } finally {
        setLoading(false)
      }
    }

    const timeoutId = setTimeout(searchHF, 500)
    return () => clearTimeout(timeoutId)
  }, [query, filters])

  if (!query || query.length < 2) {
    return (
      <Empty
        icon={<Search className="h-12 w-12" />}
        title="Search HuggingFace"
        description="Enter a search query to find models (e.g., 'llama', 'mistral', 'phi')"
      />
    )
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Spinner className="h-8 w-8 mb-4" />
        <p className="text-muted-foreground">Searching HuggingFace for "{query}"...</p>
      </div>
    )
  }

  if (error) {
    return (
      <Empty
        icon={<Info className="h-12 w-12" />}
        title="Search Error"
        description={error}
      />
    )
  }

  if (results.length === 0) {
    return (
      <Empty
        icon={<Search className="h-12 w-12" />}
        title="No models found"
        description={`No models matching "${query}" with current filters`}
      />
    )
  }

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">
        Showing {results.length} results from HuggingFace
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead>Downloads</TableHead>
            <TableHead>Likes</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {results.map((model) => {
            const architectures = detectArchitecture(model.modelId, model.tags)
            const formats = detectFormat(model.modelId, model.tags)
            
            return (
              <TableRow
                key={model.id}
                onClick={() => onSelect(model)}
                className="cursor-pointer"
              >
                <TableCell>
                  <div>
                    <div className="font-medium">{model.modelId}</div>
                    <div className="flex gap-1 flex-wrap mt-1">
                      {formats.map((fmt) => (
                        <Badge key={fmt} variant="outline" className="text-xs">
                          {fmt}
                        </Badge>
                      ))}
                      {architectures.map((arch) => (
                        <Badge key={arch} variant="secondary" className="text-xs">
                          {arch}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </TableCell>
                <TableCell>{model.downloads.toLocaleString()}</TableCell>
                <TableCell>{model.likes.toLocaleString()}</TableCell>
                <TableCell className="text-right">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation()
                      console.log('Download:', model.modelId)
                      // TODO: Trigger download via backend
                    }}
                    title="Download model"
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}

// TEAM-381: Model Details Panel
function ModelDetailsPanel({
  model,
  onLoad,
  onUnload,
  onDelete,
  isPending,
}: {
  model: Model | null
  onLoad: (modelId: string) => void
  onUnload: (modelId: string) => void
  onDelete: (modelId: string) => void
  isPending: boolean
}) {
  if (!model) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Model Details</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
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

        <div className="space-y-2">
          {!model.loaded && (
            <Button
              className="w-full"
              onClick={() => onLoad(model.id)}
              disabled={isPending}
            >
              <Play className="h-4 w-4 mr-2" />
              Load to RAM
            </Button>
          )}

          {model.loaded && (
            <Button
              className="w-full"
              variant="outline"
              onClick={() => onUnload(model.id)}
              disabled={isPending}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Unload from RAM
            </Button>
          )}

          <Button
            className="w-full"
            variant="destructive"
            onClick={() => onDelete(model.id)}
            disabled={isPending}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete Model
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
