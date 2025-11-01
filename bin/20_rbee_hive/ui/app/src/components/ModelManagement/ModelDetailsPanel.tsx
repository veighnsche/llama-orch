// TEAM-381: Model Details Panel - Shows selected model details and actions

import { Play, Trash2 } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button } from '@rbee/ui/atoms'
import type { ModelInfo } from './types'

interface ModelDetailsPanelProps {
  model: ModelInfo | null
  onLoad: (modelId: string) => void
  onUnload: (modelId: string) => void
  onDelete: (modelId: string) => void
  isPending: boolean
}

export function ModelDetailsPanel({
  model,
  onLoad,
  onUnload,
  onDelete,
  isPending,
}: ModelDetailsPanelProps) {
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
            <div>{(model.size_bytes / 1_000_000_000).toFixed(2)} GB</div>
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
