// TEAM-405: Marketplace Rbee Workers page - Using reusable components
// TEAM-421: Implemented with WorkerListTemplate and marketplace_list_workers command
// TEAM-423: Updated to match Next.js version with CategoryFilterBar and proper filtering
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: CategoryFilterBar + WorkerCard grid (matching Next.js)

import { UniversalFilterBar, WorkerCard } from '@rbee/ui/marketplace'
import { PageContainer } from '@rbee/ui/molecules'
import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

// Worker filter state (matching Next.js filters.ts)
interface WorkerFilters {
  category: 'all' | 'llm' | 'image'
  backend: 'all' | 'cpu' | 'cuda' | 'metal' | 'rocm'
  platform: 'all' | 'linux' | 'macos' | 'windows'
}

// Filter group definitions (matching Next.js)
const WORKER_FILTER_GROUPS = [
  {
    id: 'category',
    label: 'Worker Category',
    options: [
      { label: 'All Workers', value: 'all' },
      { label: 'LLM Workers', value: 'llm' },
      { label: 'Image Workers', value: 'image' },
    ],
  },
  {
    id: 'backend',
    label: 'Backend Type',
    options: [
      { label: 'All Backends', value: 'all' },
      { label: 'CPU', value: 'cpu' },
      { label: 'CUDA (NVIDIA)', value: 'cuda' },
      { label: 'Metal (Apple)', value: 'metal' },
      { label: 'ROCm (AMD)', value: 'rocm' },
    ],
  },
  {
    id: 'platform',
    label: 'Platform',
    options: [
      { label: 'All Platforms', value: 'all' },
      { label: 'Linux', value: 'linux' },
      { label: 'macOS', value: 'macos' },
      { label: 'Windows', value: 'windows' },
    ],
  },
]

export function MarketplaceRbeeWorkers() {
  const navigate = useNavigate()
  const [filters, setFilters] = useState<WorkerFilters>({
    category: 'all',
    backend: 'all',
    platform: 'all',
  })

  // DATA LAYER: Fetch workers from Tauri
  const {
    data: rawWorkers = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ['marketplace', 'rbee-workers'],
    queryFn: async () => {
      const result = await invoke<any[]>('marketplace_list_workers')
      return result
    },
    staleTime: 5 * 60 * 1000,
  })

  // TEAM-423: Filter workers based on current filter state (matching Next.js logic)
  const filteredWorkers = useMemo(() => {
    return rawWorkers.filter((worker) => {
      // Category filter (based on worker ID prefix)
      if (filters.category !== 'all') {
        const isLLM = worker.id.startsWith('llm-')
        const isImage = worker.id.startsWith('sd-')

        if (filters.category === 'llm' && !isLLM) return false
        if (filters.category === 'image' && !isImage) return false
      }

      // Backend filter (workerType)
      if (filters.backend !== 'all' && worker.workerType !== filters.backend) {
        return false
      }

      // Platform filter
      if (filters.platform !== 'all' && !worker.platforms.includes(filters.platform as 'linux' | 'macos' | 'windows')) {
        return false
      }

      return true
    })
  }, [rawWorkers, filters])

  // Build filter description
  const filterDescription = useMemo(() => {
    const parts: string[] = []

    if (filters.category !== 'all') {
      parts.push(filters.category === 'llm' ? 'LLM' : 'Image')
    }
    if (filters.backend !== 'all') {
      parts.push(filters.backend.toUpperCase())
    }
    if (filters.platform !== 'all') {
      parts.push(filters.platform.charAt(0).toUpperCase() + filters.platform.slice(1))
    }

    return parts.length > 0 ? parts.join(' · ') : 'All Workers'
  }, [filters])

  // Transform to WorkerCard format
  const workers = filteredWorkers.map((worker) => ({
    id: worker.id,
    name: worker.name,
    description: worker.description,
    version: worker.version,
    platform: worker.platforms.map((p: string) => p.toLowerCase()),
    architecture: worker.architectures.map((a: string) => a.toLowerCase()),
    workerType: worker.workerType.toLowerCase() as 'cpu' | 'cuda' | 'metal',
  }))

  // PRESENTATION LAYER: Render with CategoryFilterBar + WorkerCard grid (matching Next.js)
  return (
    <PageContainer
      title="Workers"
      description={`${filterDescription} · Browse and install rbee workers for LLM inference and image generation`}
      padding="default"
    >
      {isLoading && <div className="text-center py-12">Loading workers...</div>}
      {error && <div className="text-center py-12 text-destructive">Error: {String(error)}</div>}
      {!isLoading && !error && (
        <div className="space-y-6">
          {/* Stats */}
          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="size-2 rounded-full bg-primary" />
              <span>{workers.length} workers available</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="size-2 rounded-full bg-blue-500" />
              <span>CPU, CUDA, and Metal support</span>
            </div>
          </div>

          {/* Filter Bar */}
          <UniversalFilterBar
            groups={WORKER_FILTER_GROUPS}
            currentFilters={filters}
            onFiltersChange={(newFilters) => {
              // TEAM-423: UniversalFilterBar handles both SSG and GUI
              setFilters({ ...filters, ...newFilters })
            }}
          />

          {/* Workers Grid */}
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {workers.map((worker) => (
              <WorkerCard
                key={worker.id}
                worker={worker}
                onClick={() => navigate(`/marketplace/rbee-workers/${encodeURIComponent(worker.id)}`)}
              />
            ))}
          </div>

          {workers.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              <p className="text-lg font-medium">No workers found</p>
              <p className="text-sm">Try adjusting your filters</p>
            </div>
          )}
        </div>
      )}
    </PageContainer>
  )
}
