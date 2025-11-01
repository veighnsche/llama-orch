// TEAM-381: Downloaded Models View - Shows models downloaded to disk

import { Play, Trash2, Info, HardDrive } from 'lucide-react'
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  Badge,
  Button,
  Skeleton,
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
} from '@rbee/ui/atoms'
import type { ModelInfo } from './types'

interface DownloadedModelsViewProps {
  models: ModelInfo[]
  loading: boolean
  error: Error | null
  selectedModel: ModelInfo | null
  onSelect: (model: ModelInfo) => void
  onLoad: (modelId: string) => void
  onDelete: (modelId: string) => void
}

export function DownloadedModelsView({
  models,
  loading,
  error,
  selectedModel,
  onSelect,
  onLoad,
  onDelete,
}: DownloadedModelsViewProps) {
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
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <Info className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>Error loading models</EmptyTitle>
          <EmptyDescription>{error.message}</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  if (models.length === 0) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <HardDrive className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>No models downloaded</EmptyTitle>
          <EmptyDescription>Search HuggingFace to download models</EmptyDescription>
        </EmptyHeader>
      </Empty>
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
            <TableCell>{(model.size_bytes / 1_000_000_000).toFixed(2)} GB</TableCell>
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
