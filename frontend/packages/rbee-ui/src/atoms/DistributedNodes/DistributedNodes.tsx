import { cn } from '@rbee/ui/utils'

export interface DistributedNodesProps {
  className?: string
}

/**
 * DistributedNodes - Theme-aware SVG background for cross-node orchestration sections
 *
 * Multiple node clusters with cross-cluster connections and data packets,
 * suggesting distributed coordination across infrastructure.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <DistributedNodes />
 * </div>
 * ```
 */
export function DistributedNodes({ className }: DistributedNodesProps) {
  return (
    <svg
      viewBox="0 0 1200 640"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('pointer-events-none', className)}
      aria-hidden="true"
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Cluster 1 - Left */}
      <g filter="url(#node-glow)">
        <circle cx="250" cy="200" r="8" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="300" cy="250" r="6" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="200" cy="280" r="6" className="fill-blue-500 dark:fill-blue-400" />
      </g>

      {/* Cluster 2 - Center */}
      <g filter="url(#node-glow)">
        <circle cx="600" cy="320" r="8" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="550" cy="380" r="6" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="650" cy="360" r="6" className="fill-blue-500 dark:fill-blue-400" />
      </g>

      {/* Cluster 3 - Right */}
      <g filter="url(#node-glow)">
        <circle cx="950" cy="240" r="8" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="900" cy="290" r="6" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="1000" cy="280" r="6" className="fill-blue-500 dark:fill-blue-400" />
      </g>

      {/* Inter-cluster connections - light theme */}
      <g className="dark:hidden" opacity="0.4">
        <line x1="300" y1="250" x2="550" y2="380" stroke="rgb(16 185 129)" strokeWidth="1.5" />
        <line x1="650" y1="360" x2="900" y2="290" stroke="rgb(16 185 129)" strokeWidth="1.5" />
      </g>

      {/* Inter-cluster connections - dark theme */}
      <g className="hidden dark:block" opacity="0.5">
        <line x1="300" y1="250" x2="550" y2="380" stroke="rgb(52 211 153)" strokeWidth="2" />
        <line x1="650" y1="360" x2="900" y2="290" stroke="rgb(52 211 153)" strokeWidth="2" />
      </g>

      {/* Data packets */}
      <g className="fill-emerald-500 dark:fill-emerald-400" opacity="0.6">
        <circle cx="425" cy="315" r="3" />
        <circle cx="775" cy="325" r="3" />
      </g>
    </svg>
  )
}
