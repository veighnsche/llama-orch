import { cn } from '@rbee/ui/utils'

export interface OrchestrationFlowProps {
  className?: string
}

/**
 * OrchestrationFlow - Theme-aware SVG background for solution/infrastructure sections
 *
 * Abstract orchestration pattern with multiple nodes working in harmony,
 * suggesting distributed coordination and resource management.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <OrchestrationFlow />
 * </div>
 * ```
 */
export function OrchestrationFlow({ className }: OrchestrationFlowProps) {
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
        <filter id="orch-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <radialGradient id="orch-gradient" cx="50%" cy="50%">
          <stop offset="0%" stopColor="rgb(59 130 246)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Central orchestrator */}
      <g filter="url(#orch-glow)">
        <circle cx="600" cy="320" r="12" className="fill-blue-500 dark:fill-blue-400" />
        <circle
          cx="600"
          cy="320"
          r="24"
          className="fill-none stroke-blue-500 dark:stroke-blue-400"
          strokeWidth="1.5"
          opacity="0.4"
        />
        <circle
          cx="600"
          cy="320"
          r="36"
          className="fill-none stroke-blue-500 dark:stroke-blue-400"
          strokeWidth="1"
          opacity="0.2"
        />
      </g>

      {/* Worker nodes arranged in circle */}
      <g filter="url(#orch-glow)">
        <circle cx="400" cy="200" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="800" cy="200" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="400" cy="440" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="800" cy="440" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="300" cy="320" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="900" cy="320" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
      </g>

      {/* Connection lines - light theme */}
      <g className="dark:hidden" opacity="0.35">
        <line x1="600" y1="320" x2="400" y2="200" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="600" y1="320" x2="800" y2="200" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="600" y1="320" x2="400" y2="440" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="600" y1="320" x2="800" y2="440" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="600" y1="320" x2="300" y2="320" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="600" y1="320" x2="900" y2="320" stroke="rgb(59 130 246)" strokeWidth="1.5" />
      </g>

      {/* Connection lines - dark theme */}
      <g className="hidden dark:block" opacity="0.45">
        <line x1="600" y1="320" x2="400" y2="200" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="600" y1="320" x2="800" y2="200" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="600" y1="320" x2="400" y2="440" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="600" y1="320" x2="800" y2="440" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="600" y1="320" x2="300" y2="320" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="600" y1="320" x2="900" y2="320" stroke="rgb(96 165 250)" strokeWidth="2" />
      </g>

      {/* Data flow particles */}
      <g className="fill-emerald-500 dark:fill-emerald-400" opacity="0.6">
        <circle cx="500" cy="260" r="2" />
        <circle cx="700" cy="260" r="2" />
        <circle cx="500" cy="380" r="2" />
        <circle cx="700" cy="380" r="2" />
      </g>

      {/* Background ambient glow */}
      <ellipse cx="600" cy="320" rx="400" ry="250" fill="url(#orch-gradient)" opacity="0.15" />
    </svg>
  )
}
