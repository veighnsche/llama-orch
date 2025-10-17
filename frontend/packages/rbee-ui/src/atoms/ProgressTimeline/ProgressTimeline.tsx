import { cn } from '@rbee/ui/utils'

export interface ProgressTimelineProps {
  className?: string
}

/**
 * ProgressTimeline - Theme-aware SVG background for real-time progress sections
 *
 * Horizontal timeline with progress markers and pulse on current step,
 * suggesting live tracking and monitoring.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <ProgressTimeline />
 * </div>
 * ```
 */
export function ProgressTimeline({ className }: ProgressTimelineProps) {
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
        <filter id="progress-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        
        <linearGradient id="progress-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgb(16 185 129)" stopOpacity="0.8" />
          <stop offset="60%" stopColor="rgb(16 185 129)" stopOpacity="0.8" />
          <stop offset="60%" stopColor="rgb(59 130 246)" stopOpacity="0.4" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0.4" />
        </linearGradient>
      </defs>

      {/* Timeline base - light theme */}
      <g className="dark:hidden">
        <rect x="200" y="315" width="800" height="10" className="fill-blue-500" opacity="0.2" rx="5" />
        <rect x="200" y="315" width="480" height="10" fill="url(#progress-gradient)" rx="5" />
      </g>

      {/* Timeline base - dark theme */}
      <g className="hidden dark:block">
        <rect x="200" y="315" width="800" height="10" className="fill-blue-400" opacity="0.3" rx="5" />
        <rect x="200" y="315" width="480" height="10" className="fill-emerald-400" opacity="0.6" rx="5" />
      </g>

      {/* Milestone markers */}
      <g filter="url(#progress-glow)">
        {/* Completed milestones */}
        <circle cx="300" cy="320" r="8" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="500" cy="320" r="8" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="680" cy="320" r="8" className="fill-amber-500 dark:fill-amber-400" />
        
        {/* Upcoming milestones */}
        <circle cx="850" cy="320" r="6" className="fill-blue-500 dark:fill-blue-400" opacity="0.5" />
        <circle cx="950" cy="320" r="6" className="fill-blue-500 dark:fill-blue-400" opacity="0.5" />
      </g>

      {/* Checkmarks on completed */}
      <g className="stroke-emerald-500 dark:stroke-emerald-400" strokeWidth="2" opacity="0.8">
        <path d="M 294 320 L 298 324 L 306 314" fill="none" />
        <path d="M 494 320 L 498 324 L 506 314" fill="none" />
      </g>

      {/* Current step pulse */}
      <g opacity="0.4">
        <circle cx="680" cy="320" r="16" className="fill-none stroke-amber-500 dark:stroke-amber-400" strokeWidth="1.5" />
        <circle cx="680" cy="320" r="24" className="fill-none stroke-amber-500 dark:stroke-amber-400" strokeWidth="1" />
        <circle cx="680" cy="320" r="32" className="fill-none stroke-amber-500 dark:stroke-amber-400" strokeWidth="0.5" />
      </g>

      {/* Progress percentage indicator */}
      <g className="fill-emerald-500 dark:fill-emerald-400" opacity="0.7">
        <text x="680" y="280" textAnchor="middle" fontSize="24" fontWeight="700">60%</text>
      </g>

      {/* Time markers */}
      <g className="fill-blue-500 dark:fill-blue-400" opacity="0.5">
        <text x="300" y="370" textAnchor="middle" fontSize="12">0s</text>
        <text x="500" y="370" textAnchor="middle" fontSize="12">2s</text>
        <text x="680" y="370" textAnchor="middle" fontSize="12">4s</text>
        <text x="850" y="370" textAnchor="middle" fontSize="12">6s</text>
      </g>
    </svg>
  )
}
