// TEAM-381: Loaded Models View - Shows models loaded in RAM

import { Play, Trash2 } from 'lucide-react'
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  Badge,
  Button,
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
} from '@rbee/ui/atoms'
import type { ModelInfo } from './types'

interface LoadedModelsViewProps {
  models: ModelInfo[]
  selectedModel: ModelInfo | null
  onSelect: (model: ModelInfo) => void
  onUnload: (modelId: string) => void
}

export function LoadedModelsView({
  models,
  selectedModel,
  onSelect,
  onUnload,
}: LoadedModelsViewProps) {
  if (models.length === 0) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <Play className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>No models loaded in RAM</EmptyTitle>
          <EmptyDescription>Load a downloaded model to start inference</EmptyDescription>
        </EmptyHeader>
      </Empty>
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
