// TEAM-382: Active workers grid view

import { Loader2, ServerOff } from 'lucide-react'
import { Empty, EmptyHeader, EmptyMedia, EmptyTitle, EmptyDescription } from '@rbee/ui/atoms'
import { WorkerCard } from './WorkerCard'
import type { ProcessStats } from './types'

interface ActiveWorkersViewProps {
  workers: ProcessStats[]
  loading: boolean
  error: Error | null
  onTerminate?: (pid: number) => void
  isTerminating?: boolean
}

export function ActiveWorkersView({
  workers,
  loading,
  error,
  onTerminate,
  isTerminating,
}: ActiveWorkersViewProps) {
  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <ServerOff className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>Failed to load workers</EmptyTitle>
          <EmptyDescription>{error.message}</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  // Empty state
  if (workers.length === 0) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia>
            <ServerOff className="h-12 w-12" />
          </EmptyMedia>
          <EmptyTitle>No active workers</EmptyTitle>
          <EmptyDescription>
            Spawn a worker to start processing inference requests
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  // Workers grid
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {workers.map((worker) => (
        <WorkerCard
          key={worker.pid}
          worker={worker}
          onTerminate={onTerminate}
          isTerminating={isTerminating}
        />
      ))}
    </div>
  )
}
