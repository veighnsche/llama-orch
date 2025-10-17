import { cn } from '@rbee/ui/utils'

export interface DeploymentFlowProps {
  className?: string
}

/**
 * DeploymentFlow - Theme-aware SVG background for deployment sections
 *
 * Abstract EU-blue flow diagram with four checkpoints and connecting lines,
 * suggesting enterprise deployment stages and compliance handoffs.
 * Adapts to light/dark themes.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-15">
 *   <DeploymentFlow />
 * </div>
 * ```
 */
export function DeploymentFlow({ className }: DeploymentFlowProps) {
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
        {/* Flow line gradient - light theme */}
        <linearGradient id="flow-gradient-light" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgb(59 130 246)" stopOpacity="0.2" />
          <stop offset="50%" stopColor="rgb(59 130 246)" stopOpacity="0.5" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0.2" />
        </linearGradient>

        {/* Flow line gradient - dark theme */}
        <linearGradient id="flow-gradient-dark" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgb(96 165 250)" stopOpacity="0.3" />
          <stop offset="50%" stopColor="rgb(96 165 250)" stopOpacity="0.6" />
          <stop offset="100%" stopColor="rgb(96 165 250)" stopOpacity="0.3" />
        </linearGradient>

        {/* Checkpoint glow filter */}
        <filter id="checkpoint-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Arrow marker - light theme */}
        <marker
          id="arrow-light"
          viewBox="0 0 10 10"
          refX="5"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="rgb(59 130 246)" opacity="0.6" />
        </marker>

        {/* Arrow marker - dark theme */}
        <marker
          id="arrow-dark"
          viewBox="0 0 10 10"
          refX="5"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="rgb(96 165 250)" opacity="0.7" />
        </marker>
      </defs>

      {/* Stage 1: Compliance Assessment */}
      <g filter="url(#checkpoint-glow)">
        <circle cx="200" cy="200" r="8" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="200" cy="200" r="16" className="fill-none stroke-blue-500 dark:stroke-blue-400" strokeWidth="1.5" opacity="0.4" />
        <text x="200" y="250" textAnchor="middle" className="fill-blue-600 dark:fill-blue-300 text-xs font-medium" opacity="0.6">
          1
        </text>
      </g>

      {/* Stage 2: Deployment & Configuration */}
      <g filter="url(#checkpoint-glow)">
        <circle cx="500" cy="280" r="8" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="500" cy="280" r="16" className="fill-none stroke-blue-500 dark:stroke-blue-400" strokeWidth="1.5" opacity="0.4" />
        <text x="500" y="330" textAnchor="middle" className="fill-blue-600 dark:fill-blue-300 text-xs font-medium" opacity="0.6">
          2
        </text>
      </g>

      {/* Stage 3: Compliance Validation */}
      <g filter="url(#checkpoint-glow)">
        <circle cx="700" cy="360" r="8" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="700" cy="360" r="16" className="fill-none stroke-blue-500 dark:stroke-blue-400" strokeWidth="1.5" opacity="0.4" />
        <text x="700" y="410" textAnchor="middle" className="fill-blue-600 dark:fill-blue-300 text-xs font-medium" opacity="0.6">
          3
        </text>
      </g>

      {/* Stage 4: Production Launch */}
      <g filter="url(#checkpoint-glow)">
        <circle cx="1000" cy="280" r="8" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="1000" cy="280" r="16" className="fill-none stroke-blue-500 dark:stroke-blue-400" strokeWidth="1.5" opacity="0.4" />
        <circle cx="1000" cy="280" r="24" className="fill-none stroke-amber-500 dark:stroke-amber-400" strokeWidth="1" opacity="0.5" />
        <text x="1000" y="330" textAnchor="middle" className="fill-blue-600 dark:fill-blue-300 text-xs font-medium" opacity="0.6">
          4
        </text>
      </g>

      {/* Flow lines - light theme */}
      <g className="dark:hidden">
        {/* Main flow path */}
        <path
          d="M 200 200 Q 350 240 500 280"
          fill="none"
          stroke="url(#flow-gradient-light)"
          strokeWidth="2"
          markerEnd="url(#arrow-light)"
        />
        <path
          d="M 500 280 Q 600 320 700 360"
          fill="none"
          stroke="url(#flow-gradient-light)"
          strokeWidth="2"
          markerEnd="url(#arrow-light)"
        />
        <path
          d="M 700 360 Q 850 320 1000 280"
          fill="none"
          stroke="url(#flow-gradient-light)"
          strokeWidth="2"
          markerEnd="url(#arrow-light)"
        />

        {/* Feedback loops (compliance iterations) */}
        <path
          d="M 700 340 Q 600 300 500 300"
          fill="none"
          stroke="rgb(245 158 11)"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity="0.4"
        />
        <path
          d="M 1000 300 Q 850 380 700 380"
          fill="none"
          stroke="rgb(245 158 11)"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity="0.4"
        />
      </g>

      {/* Flow lines - dark theme */}
      <g className="hidden dark:block">
        {/* Main flow path */}
        <path
          d="M 200 200 Q 350 240 500 280"
          fill="none"
          stroke="url(#flow-gradient-dark)"
          strokeWidth="2.5"
          markerEnd="url(#arrow-dark)"
        />
        <path
          d="M 500 280 Q 600 320 700 360"
          fill="none"
          stroke="url(#flow-gradient-dark)"
          strokeWidth="2.5"
          markerEnd="url(#arrow-dark)"
        />
        <path
          d="M 700 360 Q 850 320 1000 280"
          fill="none"
          stroke="url(#flow-gradient-dark)"
          strokeWidth="2.5"
          markerEnd="url(#arrow-dark)"
        />

        {/* Feedback loops (compliance iterations) */}
        <path
          d="M 700 340 Q 600 300 500 300"
          fill="none"
          stroke="rgb(251 191 36)"
          strokeWidth="1.5"
          strokeDasharray="4 4"
          opacity="0.5"
        />
        <path
          d="M 1000 300 Q 850 380 700 380"
          fill="none"
          stroke="rgb(251 191 36)"
          strokeWidth="1.5"
          strokeDasharray="4 4"
          opacity="0.5"
        />
      </g>

      {/* Background grid (subtle) */}
      <g opacity="0.15">
        <defs>
          <pattern id="deploy-grid" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.5"
              className="text-blue-500 dark:text-blue-400"
            />
          </pattern>
        </defs>
        <rect width="1200" height="640" fill="url(#deploy-grid)" />
      </g>
    </svg>
  )
}
