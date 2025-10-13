import { cn } from '@/lib/utils'

export type WorkerNode = {
  id: string
  label: string
  kind: 'cuda' | 'metal' | 'cpu'
}

export type BeeTopology =
  | { mode: 'single-pc'; hostLabel: string; workers: WorkerNode[] }
  | { mode: 'multi-host'; hosts: Array<{ hostLabel: string; workers: WorkerNode[] }> }

export interface BeeArchitectureProps {
  topology: BeeTopology
  className?: string
}

export function BeeArchitecture({ topology, className }: BeeArchitectureProps) {
  const getWorkerRing = (kind: WorkerNode['kind']) => {
    switch (kind) {
      case 'cuda':
        return 'ring-1 ring-amber-400/30'
      case 'metal':
        return 'ring-1 ring-sky-400/30'
      case 'cpu':
        return 'ring-1 ring-emerald-400/30'
    }
  }

  const getWorkerEmoji = (kind: WorkerNode['kind']) => {
    return '🐝'
  }

  const getWorkerSubLabel = (kind: WorkerNode['kind']) => {
    switch (kind) {
      case 'cuda':
        return 'CUDA'
      case 'metal':
        return 'Metal'
      case 'cpu':
        return 'CPU'
    }
  }

  const renderWorkerChip = (worker: WorkerNode, index: number) => (
    <div
      key={worker.id}
      className={cn(
        'flex items-center gap-2 rounded-lg border border-border bg-muted px-3 py-2',
        'animate-in zoom-in-50 duration-400',
        getWorkerRing(worker.kind)
      )}
      style={{ animationDelay: `${(index + 1) * 120}ms` }}
    >
      <span className="text-xl" aria-hidden="true">
        {getWorkerEmoji(worker.kind)}
      </span>
      <div className="text-sm">
        <div className="font-semibold text-foreground">{worker.label}</div>
        <div className="text-muted-foreground">{getWorkerSubLabel(worker.kind)}</div>
      </div>
    </div>
  )

  const renderSinglePC = () => {
    if (topology.mode !== 'single-pc') return null

    return (
      <>
        {/* Queen */}
        <div className="flex items-center gap-3 rounded-lg border border-primary/30 bg-primary/10 px-6 py-3 animate-in fade-in slide-in-from-top-2 duration-400 delay-100">
          <span className="text-2xl" aria-hidden="true">
            👑
          </span>
          <div>
            <div className="font-semibold text-foreground">queen-rbee</div>
            <div className="text-sm text-muted-foreground">Orchestrator (brain)</div>
          </div>
        </div>

        {/* Connector */}
        <div className="h-8 w-px bg-border" aria-hidden="true" />

        {/* Hive */}
        <div className="flex items-center gap-3 rounded-lg border border-border bg-muted px-6 py-3 animate-in fade-in slide-in-from-top-2 duration-400 delay-200">
          <span className="text-2xl" aria-hidden="true">
            🍯
          </span>
          <div>
            <div className="font-semibold text-foreground">rbee-hive</div>
            <div className="text-sm text-muted-foreground">Resource manager</div>
          </div>
        </div>

        {/* Connector */}
        <div className="h-8 w-px bg-border" aria-hidden="true" />

        {/* Host chassis */}
        <div className="w-full rounded-xl border border-border bg-card/50 p-6 animate-in fade-in slide-in-from-top-2 duration-400 delay-300">
          <div className="mb-4 text-center text-sm font-medium text-muted-foreground">{topology.hostLabel}</div>
          <div className="flex flex-wrap justify-center gap-4">{topology.workers.map(renderWorkerChip)}</div>
        </div>
      </>
    )
  }

  const renderMultiHost = () => {
    if (topology.mode !== 'multi-host') return null

    return (
      <>
        {/* Queen */}
        <div className="flex items-center gap-3 rounded-lg border border-primary/30 bg-primary/10 px-6 py-3 animate-in fade-in slide-in-from-top-2 duration-400 delay-100">
          <span className="text-2xl" aria-hidden="true">
            👑
          </span>
          <div>
            <div className="font-semibold text-foreground">queen-rbee</div>
            <div className="text-sm text-muted-foreground">Orchestrator (brain)</div>
          </div>
        </div>

        {/* Connector */}
        <div className="h-8 w-px bg-border" aria-hidden="true" />

        {/* Hive */}
        <div className="flex items-center gap-3 rounded-lg border border-border bg-muted px-6 py-3 animate-in fade-in slide-in-from-top-2 duration-400 delay-200">
          <span className="text-2xl" aria-hidden="true">
            🍯
          </span>
          <div>
            <div className="font-semibold text-foreground">rbee-hive</div>
            <div className="text-sm text-muted-foreground">Resource manager</div>
          </div>
        </div>

        {/* Connector */}
        <div className="h-8 w-px bg-border" aria-hidden="true" />

        {/* Multiple hosts */}
        <div className="grid w-full gap-6 sm:grid-cols-2 lg:grid-cols-3 animate-in fade-in slide-in-from-top-2 duration-400 delay-300">
          {topology.hosts.map((host, hostIndex) => (
            <div key={hostIndex} className="rounded-xl border border-border bg-card/50 p-4">
              <div className="mb-3 text-center text-sm font-medium text-muted-foreground">{host.hostLabel}</div>
              <div className="flex flex-col gap-3">{host.workers.map((worker, idx) => renderWorkerChip(worker, hostIndex * 10 + idx))}</div>
            </div>
          ))}
        </div>
      </>
    )
  }

  const getFigcaption = () => {
    if (topology.mode === 'single-pc') {
      const workerDesc = topology.workers
        .map((w) => `${w.kind.toUpperCase()} ${w.label}`)
        .join(', ')
      return `Topology diagram: queen-rbee orchestrates rbee-hive on a single PC with ${workerDesc}.`
    } else {
      const hostCount = topology.hosts.length
      return `Topology diagram: queen-rbee orchestrates rbee-hive across ${hostCount} host${hostCount > 1 ? 's' : ''}.`
    }
  }

  return (
    <figure className={cn('mx-auto mt-16 max-w-3xl rounded-xl border border-border bg-card p-8', className)}>
      <figcaption className="sr-only">{getFigcaption()}</figcaption>
      <h3 className="mb-6 text-center text-xl font-semibold text-card-foreground">The Bee Architecture</h3>
      <div className="flex flex-col items-center gap-6">
        {topology.mode === 'single-pc' ? renderSinglePC() : renderMultiHost()}
      </div>
    </figure>
  )
}
