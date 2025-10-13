import { TopologyDiagram, type TDNode, type TDEdge } from '@/components/organisms/TopologyDiagram'

export interface ArchitectureDiagramProps {
  variant?: 'simple' | 'detailed'
  showLabels?: boolean
  className?: string
}

export function ArchitectureDiagram({ variant = 'simple', showLabels = true, className }: ArchitectureDiagramProps) {
  // Define nodes
  const nodes: TDNode[] = [
    // Control tier
    {
      id: 'queen',
      label: 'Queen‑rbee (Orchestrator)',
      icon: '👑',
      tone: 'primary',
      tier: 'control',
      laneIndex: 0,
    },
    // Manager tier
    {
      id: 'manager-1',
      label: 'Hive Manager 1',
      icon: '🍯',
      tone: 'accent',
      tier: 'manager',
      laneIndex: 0,
    },
    {
      id: 'manager-2',
      label: 'Hive Manager 2',
      icon: '🍯',
      tone: 'accent',
      tier: 'manager',
      laneIndex: 1,
    },
    {
      id: 'manager-3',
      label: 'Hive Manager 3',
      icon: '🍯',
      tone: 'accent',
      tier: 'manager',
      laneIndex: 2,
    },
    // Execution tier
    {
      id: 'worker-cuda-1',
      label: 'Worker (CUDA)',
      icon: '🐝',
      tone: 'muted',
      tier: 'execution',
      laneIndex: 0,
    },
    {
      id: 'worker-metal',
      label: 'Worker (Metal)',
      icon: '🐝',
      tone: 'muted',
      tier: 'execution',
      laneIndex: 1,
    },
    {
      id: 'worker-cpu',
      label: 'Worker (CPU)',
      icon: '🐝',
      tone: 'muted',
      tier: 'execution',
      laneIndex: 2,
    },
    {
      id: 'worker-cuda-2',
      label: 'Worker (CUDA)',
      icon: '🐝',
      tone: 'muted',
      tier: 'execution',
      laneIndex: 3,
    },
  ]

  // Define edges
  const edges: TDEdge[] = [
    // Queen to Managers (control flow)
    { id: 'queen-m1', from: 'queen', to: 'manager-1', kind: 'control' },
    { id: 'queen-m2', from: 'queen', to: 'manager-2', kind: 'control' },
    { id: 'queen-m3', from: 'queen', to: 'manager-3', kind: 'control' },
    // Managers to Workers (control flow)
    { id: 'm1-w1', from: 'manager-1', to: 'worker-cuda-1', kind: 'control' },
    { id: 'm2-w2', from: 'manager-2', to: 'worker-metal', kind: 'control' },
    { id: 'm3-w3', from: 'manager-3', to: 'worker-cpu', kind: 'control' },
    { id: 'm1-w4', from: 'manager-1', to: 'worker-cuda-2', kind: 'control' },
    // Workers to Managers (telemetry)
    { id: 'w1-m1', from: 'worker-cuda-1', to: 'manager-1', kind: 'telemetry' },
    { id: 'w2-m2', from: 'worker-metal', to: 'manager-2', kind: 'telemetry' },
    { id: 'w3-m3', from: 'worker-cpu', to: 'manager-3', kind: 'telemetry' },
    { id: 'w4-m1', from: 'worker-cuda-2', to: 'manager-1', kind: 'telemetry' },
  ]

  return (
    <TopologyDiagram
      nodes={nodes}
      edges={edges}
      showLegend={variant === 'detailed'}
      showLaneLabels={variant === 'detailed'}
      className={className}
    />
  )
}
