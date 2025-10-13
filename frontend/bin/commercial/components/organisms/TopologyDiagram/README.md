# TopologyDiagram

Lightweight organism that renders system architecture diagrams with rails, nodes, and labeled connectors. Built with deterministic layout, SVG connectors, and full accessibility.

## Features

- **Deterministic layout**: 3-tier rail system (control, manager, execution) with absolute positioning
- **SVG connectors**: Crisp Bézier curves with arrow markers and port dots
- **Edge types**: Solid (control/task flow) and dashed (telemetry/health)
- **Animated flow**: Subtle dash animation on control edges
- **Motion hierarchy**: Staggered fade-in animations, respects `prefers-reduced-motion`
- **Accessibility**: Semantic HTML with ARIA labels and screen-reader descriptions
- **Responsive**: Percentage-based layout adapts 320px–1280px

## API

```tsx
type Tone = 'primary' | 'accent' | 'muted'
type Tier = 'control' | 'manager' | 'execution'

type TDNode = {
  id: string
  label: string
  icon?: React.ReactNode
  tone?: Tone
  tier: Tier
  laneIndex?: number
}

type TDEdge = {
  id: string
  from: string
  to: string
  label?: string
  kind?: 'control' | 'telemetry'
}

interface TopologyDiagramProps {
  nodes: TDNode[]
  edges: TDEdge[]
  showLegend?: boolean
  showLaneLabels?: boolean
  className?: string
  compact?: boolean
}
```

## Usage

```tsx
import { TopologyDiagram, type TDNode, type TDEdge } from '@/components/organisms/TopologyDiagram'

const nodes: TDNode[] = [
  {
    id: 'queen',
    label: 'Queen‑rbee (Orchestrator)',
    icon: '👑',
    tone: 'primary',
    tier: 'control',
    laneIndex: 0,
  },
  {
    id: 'manager-1',
    label: 'Hive Manager 1',
    icon: '🍯',
    tone: 'accent',
    tier: 'manager',
    laneIndex: 0,
  },
  // ... more nodes
]

const edges: TDEdge[] = [
  { id: 'queen-m1', from: 'queen', to: 'manager-1', kind: 'control' },
  { id: 'w1-m1', from: 'worker-1', to: 'manager-1', kind: 'telemetry' },
  // ... more edges
]

<TopologyDiagram
  nodes={nodes}
  edges={edges}
  showLegend={true}
  showLaneLabels={true}
/>
```

## Visual Design

### Rails
- **Control Rail**: Top tier (24% from top), typically contains orchestrator
- **Manager Rail**: Middle tier (50% from top), schedulers and coordinators
- **Execution Rail**: Bottom tier (76% from top), workers and runners
- Rails are subtle hairline guides (1px, `bg-border/40-50`), not boxed containers

### Node Tones
- **primary**: `bg-primary text-primary-foreground` with shadow (Queen)
- **accent**: `bg-primary/10 text-primary border-primary/25` (Managers)
- **muted**: `bg-muted text-muted-foreground border-border` (Workers)

### Edge Types
- **control**: Solid lines with arrows (task/job flow)
- **telemetry**: Dashed lines (health/metrics reporting)

## Animation

- Nodes fade in with staggered delays (80ms increments)
- Control edges have animated dash effect (`flow` keyframe)
- All animations respect `prefers-reduced-motion`
- `vectorEffect="non-scaling-stroke"` keeps edges crisp

## Accessibility

- Semantic `<figure>` with `aria-labelledby` and `aria-describedby`
- Screen-reader description explains relationships
- SVG marked `aria-hidden="true"` and `focusable="false"`
- Icons marked `aria-hidden="true"`
- Rails marked `aria-hidden="true"`
- Color contrast ≥ WCAG AA

## Implementation Notes

- Uses percentage-based absolute positioning for deterministic layout
- No DOM measurements required (zero `getBoundingClientRect` calls)
- SVG connectors use Bézier curves with shared midpoint Y
- Port dots rendered at node centers for visual clarity
- Nodes auto-distribute based on tier count (1-4 nodes supported)
- Compact mode reduces padding and min-height
- Single outer card container—no nested boxes
