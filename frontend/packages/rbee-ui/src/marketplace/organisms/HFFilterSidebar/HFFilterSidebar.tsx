// TEAM-502: HuggingFace Filter Sidebar Component
// Design: .docs/TEAM_502_FILTER_SIDEBAR_DESIGN.md

import { Button } from '@rbee/ui/atoms/Button'
import { Input } from '@rbee/ui/atoms/Input'
import { ChevronDown, ChevronUp, RotateCcw, Search } from 'lucide-react'
import type React from 'react'
import { useState } from 'react'
import { FormatFilter } from './FormatFilter'
import { LanguageFilter } from './LanguageFilter'
import { LicenseFilter } from './LicenseFilter'
import { ParameterFilter } from './ParameterFilter'
import { SortFilter } from './SortFilter'
import { TaskFilter } from './TaskFilter'
import { WorkerFilter } from './WorkerFilter'

export interface HFFilterWorker {
  id: string
  name: string
  description: string
  marketplaceCompatibility?: {
    huggingface?: {
      tasks: string[]
      libraries: string[]
      formats: string[]
      languages?: string[]
      licenses?: string[]
      minParameters?: number
      maxParameters?: number
    }
  }
}

export interface HFFilterState {
  // Worker selection
  workers: string[] // Worker IDs

  // HuggingFace API filters
  tasks: string[]
  libraries: string[]
  formats: string[]

  // Client-side filters
  languages?: string[] | undefined
  licenses?: string[] | undefined
  minParameters?: number | undefined
  maxParameters?: number | undefined

  // Sorting
  sort: 'trending' | 'downloads' | 'likes' | 'updated' | 'created'
  direction: 1 | -1
}

export interface HFFilterOptions {
  // Available options (from GWC workers)
  availableWorkers: HFFilterWorker[]
  availableTasks: string[]
  availableLibraries: string[]
  availableFormats: string[]
  availableLanguages: string[]
  availableLicenses: string[]
}

interface HFFilterSidebarProps {
  // Current filter state
  filters: HFFilterState
  // Available filter options
  options: HFFilterOptions
  // Search query
  searchQuery: string
  // Callback when filters change
  onFiltersChange: (filters: HFFilterState) => void
  // Callback when search changes
  onSearchChange: (query: string) => void
  // Is sidebar collapsed (for mobile)
  collapsed?: boolean
  // On toggle collapse
  onToggleCollapse?: () => void
}

/**
 * Collapsible filter section
 */
interface FilterSectionProps {
  title: string
  children: React.ReactNode
  defaultExpanded?: boolean
  onReset?: () => void
  showReset?: boolean
  isActive?: boolean
}

const FilterSection: React.FC<FilterSectionProps> = ({
  title,
  children,
  defaultExpanded = true,
  onReset,
  showReset = false,
  isActive = false,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded)

  return (
    <div className={`border-b border-sidebar-border pb-4 ${isActive ? 'bg-sidebar-accent/10 -mx-4 px-4' : ''}`}>
      <div className="flex items-center justify-between mb-3">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-sm font-medium text-sidebar-foreground hover:text-sidebar-accent-foreground transition-colors"
        >
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          {title}
        </button>
        {showReset && onReset && (
          <button
            type="button"
            onClick={onReset}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-sidebar-accent-foreground transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>
        )}
      </div>
      {expanded && <div className="space-y-2">{children}</div>}
    </div>
  )
}

/**
 * Main HuggingFace Filter Sidebar
 */
export const HFFilterSidebar: React.FC<HFFilterSidebarProps> = ({
  filters,
  options,
  searchQuery,
  onFiltersChange,
  onSearchChange,
  collapsed = false,
  onToggleCollapse,
}) => {
  // Helper to update filters
  const updateFilters = (updates: Partial<HFFilterState>) => {
    onFiltersChange({ ...filters, ...updates })
  }

  // Helper to reset all filters
  const resetAllFilters = () => {
    onFiltersChange({
      workers: [],
      tasks: [],
      libraries: [],
      formats: [],
      languages: [],
      licenses: [],
      minParameters: undefined,
      maxParameters: undefined,
      sort: 'downloads',
      direction: -1,
    })
  }

  // Check if section is active (has filters selected)
  const isSectionActive = (section: keyof Omit<HFFilterState, 'sort' | 'direction'>) => {
    const value = filters[section]
    if (Array.isArray(value)) {
      return value.length > 0
    }
    return value !== undefined && value !== null
  }

  // Get combined tasks from selected workers
  const getAvailableTasks = () => {
    if (filters.workers.length === 0) {
      return options.availableTasks
    }

    const selectedWorkers = options.availableWorkers.filter((worker) => filters.workers.includes(worker.id))

    const tasks = new Set<string>()
    selectedWorkers.forEach((worker) => {
      const compat = worker.marketplaceCompatibility?.huggingface
      if (!compat) return

      compat.tasks.forEach((task: string) => {
        tasks.add(task)
      })
    })

    return Array.from(tasks)
  }

  // Get combined formats from selected workers
  const getAvailableFormats = () => {
    if (filters.workers.length === 0) {
      return options.availableFormats
    }

    const selectedWorkers = options.availableWorkers.filter((worker) => filters.workers.includes(worker.id))

    const formats = new Set<string>()
    selectedWorkers.forEach((worker) => {
      const compat = worker.marketplaceCompatibility?.huggingface
      if (!compat) return

      compat.formats.forEach((format: string) => {
        formats.add(format)
      })
    })

    return Array.from(formats)
  }

  // Get combined libraries from selected workers
  const getAvailableLibraries = () => {
    if (filters.workers.length === 0) {
      return options.availableLibraries
    }

    const selectedWorkers = options.availableWorkers.filter((worker) => filters.workers.includes(worker.id))

    const libraries = new Set<string>()
    selectedWorkers.forEach((worker) => {
      const compat = worker.marketplaceCompatibility?.huggingface
      if (!compat) return

      compat.libraries.forEach((library: string) => {
        libraries.add(library)
      })
    })

    return Array.from(libraries)
  }

  // Get parameter limits from selected workers
  const getParameterLimits = () => {
    if (filters.workers.length === 0) {
      return { min: 0.1, max: 1000 } // Default wide range
    }

    const selectedWorkers = options.availableWorkers.filter((worker) => filters.workers.includes(worker.id))

    let min = Infinity
    let max = -Infinity

    selectedWorkers.forEach((worker) => {
      const compat = worker.marketplaceCompatibility?.huggingface
      if (!compat) return

      if (compat.minParameters !== undefined) {
        min = Math.min(min, compat.minParameters)
      }
      if (compat.maxParameters !== undefined) {
        max = Math.max(max, compat.maxParameters)
      }
    })

    return {
      min: min === Infinity ? 0.1 : min,
      max: max === -Infinity ? 1000 : max,
    }
  }

  // Reset specific filter section
  const resetSection = (section: keyof Omit<HFFilterState, 'sort' | 'direction'>) => {
    if (section === 'workers' || section === 'tasks' || section === 'libraries' || section === 'formats') {
      updateFilters({ [section]: [] })
    } else {
      updateFilters({ [section]: undefined })
    }
  }

  if (collapsed) {
    return (
      <div className="w-12 bg-sidebar text-sidebar-foreground border-r border-sidebar-border flex flex-col items-center py-4">
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          onClick={onToggleCollapse}
          className="hover:bg-sidebar-accent/10 rounded-lg"
        >
          <ChevronDown className="w-5 h-5 text-muted-foreground rotate-180" />
        </Button>
      </div>
    )
  }

  return (
    <div className="w-80 bg-sidebar text-sidebar-foreground border-r border-sidebar-border h-full overflow-y-auto">
      <div className="p-4 border-b border-sidebar-border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-sidebar-foreground">Filters</h2>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={resetAllFilters}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-sidebar-accent-foreground"
            >
              <RotateCcw className="w-3 h-3" />
              Reset All
            </Button>
            {onToggleCollapse && (
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={onToggleCollapse}
                className="hover:bg-sidebar-accent/10 rounded"
              >
                <ChevronDown className="w-4 h-4 text-muted-foreground rotate-180" />
              </Button>
            )}
          </div>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search models..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-10 pr-4 py-2 text-sm"
          />
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Workers Filter */}
        <FilterSection
          title="Workers"
          onReset={() => resetSection('workers')}
          showReset={isSectionActive('workers')}
          isActive={isSectionActive('workers')}
        >
          <WorkerFilter
            workers={options.availableWorkers}
            selectedWorkers={filters.workers}
            onWorkersChange={(workers) => updateFilters({ workers })}
          />
        </FilterSection>

        {/* Sort Filter */}
        <FilterSection title="Sort" defaultExpanded={true}>
          <SortFilter
            sort={filters.sort}
            direction={filters.direction}
            onSortChange={(sort, direction) => updateFilters({ sort, direction })}
          />
        </FilterSection>

        {/* Tasks Filter */}
        <FilterSection
          title="Tasks"
          onReset={() => resetSection('tasks')}
          showReset={isSectionActive('tasks')}
          isActive={isSectionActive('tasks')}
        >
          <TaskFilter
            tasks={getAvailableTasks()}
            selectedTasks={filters.tasks}
            onTasksChange={(tasks) => updateFilters({ tasks })}
          />
        </FilterSection>

        {/* Parameters Filter */}
        <FilterSection
          title="Parameters"
          onReset={() => resetSection('minParameters')}
          showReset={filters.minParameters !== undefined || filters.maxParameters !== undefined}
          isActive={filters.minParameters !== undefined || filters.maxParameters !== undefined}
        >
          <ParameterFilter
            min={getParameterLimits().min}
            max={getParameterLimits().max}
            selectedMin={filters.minParameters}
            selectedMax={filters.maxParameters}
            onParametersChange={(minParameters, maxParameters) => updateFilters({ minParameters, maxParameters })}
          />
        </FilterSection>

        {/* Formats Filter */}
        <FilterSection
          title="Formats"
          onReset={() => resetSection('formats')}
          showReset={isSectionActive('formats')}
          isActive={isSectionActive('formats')}
        >
          <FormatFilter
            formats={getAvailableFormats()}
            libraries={getAvailableLibraries()}
            selectedFormats={filters.formats}
            selectedLibraries={filters.libraries}
            onFormatsChange={(formats) => updateFilters({ formats })}
            onLibrariesChange={(libraries) => updateFilters({ libraries })}
          />
        </FilterSection>

        {/* Languages Filter */}
        <FilterSection
          title="Languages"
          onReset={() => resetSection('languages')}
          showReset={isSectionActive('languages')}
          isActive={isSectionActive('languages')}
        >
          <LanguageFilter
            languages={options.availableLanguages}
            selectedLanguages={filters.languages || []}
            onLanguagesChange={(languages) => updateFilters({ languages })}
          />
        </FilterSection>

        {/* Licenses Filter */}
        <FilterSection
          title="Licenses"
          onReset={() => resetSection('licenses')}
          showReset={isSectionActive('licenses')}
          isActive={isSectionActive('licenses')}
        >
          <LicenseFilter
            licenses={options.availableLicenses}
            selectedLicenses={filters.licenses || []}
            onLicensesChange={(licenses) => updateFilters({ licenses })}
          />
        </FilterSection>
      </div>
    </div>
  )
}

export default HFFilterSidebar
