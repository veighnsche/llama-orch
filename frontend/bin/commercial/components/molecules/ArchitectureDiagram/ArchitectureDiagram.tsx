import { cn } from '@/lib/utils'

export interface ArchitectureDiagramProps {
  variant?: 'simple' | 'detailed'
  showLabels?: boolean
  className?: string
}

export function ArchitectureDiagram({ variant = 'simple', showLabels = true, className }: ArchitectureDiagramProps) {
  return (
    <div className={cn('rounded-lg border border-border bg-card p-8', className)}>
      {showLabels && (
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full text-primary text-sm font-medium">
            The Bee Architecture
          </div>
        </div>
      )}

      <div className="space-y-8">
        {/* Queen */}
        <div className="flex flex-col items-center">
          <div className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-bold text-lg shadow-md">
            👑 Queen-rbee (Orchestrator)
          </div>
          <div className="h-8 w-0.5 bg-border my-2"></div>
        </div>

        {/* Hive Managers */}
        <div className="flex justify-center gap-4">
          <div className="bg-primary/10 text-primary px-4 py-2 rounded-lg font-medium text-sm border border-primary/20">
            🍯 Hive Manager 1
          </div>
          <div className="bg-primary/10 text-primary px-4 py-2 rounded-lg font-medium text-sm border border-primary/20">
            🍯 Hive Manager 2
          </div>
          <div className="bg-primary/10 text-primary px-4 py-2 rounded-lg font-medium text-sm border border-primary/20">
            🍯 Hive Manager 3
          </div>
        </div>

        <div className="flex justify-center gap-4">
          <div className="h-8 w-0.5 bg-border"></div>
          <div className="h-8 w-0.5 bg-border"></div>
          <div className="h-8 w-0.5 bg-border"></div>
        </div>

        {/* Workers */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-muted text-muted-foreground px-3 py-2 rounded-md text-xs font-medium border border-border text-center">
            🐝 Worker (CUDA)
          </div>
          <div className="bg-muted text-muted-foreground px-3 py-2 rounded-md text-xs font-medium border border-border text-center">
            🐝 Worker (Metal)
          </div>
          <div className="bg-muted text-muted-foreground px-3 py-2 rounded-md text-xs font-medium border border-border text-center">
            🐝 Worker (CPU)
          </div>
          <div className="bg-muted text-muted-foreground px-3 py-2 rounded-md text-xs font-medium border border-border text-center">
            🐝 Worker (CUDA)
          </div>
        </div>
      </div>
    </div>
  )
}
