// TEAM-381: Search Results View - HuggingFace model search with filtering

import { useState, useEffect } from 'react'
import { Search, Download, Info } from 'lucide-react'
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  Badge,
  Button,
  Spinner,
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
} from '@rbee/ui/atoms'
import type { HFModel, FilterState } from './types'
import { detectArchitecture, detectFormat, filterModels, sortModels } from './utils'

interface SearchResultsViewProps {
  query: string
  filters: FilterState
  onSelect: (model: any) => void
}

export function SearchResultsView({ query, filters, onSelect }: SearchResultsViewProps) {
  const [results, setResults] = useState<HFModel[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch from HuggingFace API
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
        
        // Apply client-side filtering
        const filtered = filterModels(data, {
          formats: filters.formats,
          architectures: filters.architectures,
          openSourceOnly: filters.openSourceOnly,
        })
        
        // Sort results
        const sorted = sortModels(filtered, filters.sortBy)
        
        setResults(sorted)
      } catch (err) {
        console.error('HuggingFace search error:', err)
        setError(err instanceof Error ? err.message : 'Failed to search HuggingFace')
      } finally {
        setLoading(false)
      }
    }

    // Debounce search by 500ms
    const timeoutId = setTimeout(searchHF, 500)
    return () => clearTimeout(timeoutId)
  }, [query, filters])

  if (!query || query.length < 2) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <Search className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>Search HuggingFace</EmptyTitle>
          <EmptyDescription>
            Enter a search query to find models (e.g., &quot;llama&quot;, &quot;mistral&quot;, &quot;phi&quot;)
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Spinner className="h-8 w-8 mb-4" />
        <p className="text-muted-foreground">Searching HuggingFace for &quot;{query}&quot;...</p>
      </div>
    )
  }

  if (error) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <Info className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>Search Error</EmptyTitle>
          <EmptyDescription>{error}</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  if (results.length === 0) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <Search className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>No models found</EmptyTitle>
          <EmptyDescription>
            No models matching &quot;{query}&quot; with current filters
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
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
