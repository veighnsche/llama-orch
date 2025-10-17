import { cn } from '@rbee/ui/utils'

export interface ComparisonGridProps {
  className?: string
}

/**
 * ComparisonGrid - Theme-aware SVG background for comparison sections
 *
 * Side-by-side vertical lanes with comparison points and indicators,
 * suggesting feature-by-feature evaluation.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <ComparisonGrid />
 * </div>
 * ```
 */
export function ComparisonGrid({ className }: ComparisonGridProps) {
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
        <filter id="comp-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Vertical lanes - light theme */}
      <g className="dark:hidden" opacity="0.3">
        <rect x="350" y="100" width="200" height="440" className="fill-blue-500" opacity="0.1" rx="8" />
        <rect x="650" y="100" width="200" height="440" className="fill-emerald-500" opacity="0.1" rx="8" />
      </g>

      {/* Vertical lanes - dark theme */}
      <g className="hidden dark:block" opacity="0.4">
        <rect x="350" y="100" width="200" height="440" className="fill-blue-400" opacity="0.15" rx="8" />
        <rect x="650" y="100" width="200" height="440" className="fill-emerald-400" opacity="0.15" rx="8" />
      </g>

      {/* Comparison lines */}
      <g className="stroke-amber-500 dark:stroke-amber-400" strokeWidth="1" opacity="0.4">
        <line x1="350" y1="200" x2="850" y2="200" strokeDasharray="4 4" />
        <line x1="350" y1="320" x2="850" y2="320" strokeDasharray="4 4" />
        <line x1="350" y1="440" x2="850" y2="440" strokeDasharray="4 4" />
      </g>

      {/* Comparison points */}
      <g filter="url(#comp-glow)">
        {/* Left lane points */}
        <circle cx="450" cy="200" r="6" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="450" cy="320" r="6" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="450" cy="440" r="6" className="fill-blue-500 dark:fill-blue-400" />

        {/* Right lane points */}
        <circle cx="750" cy="200" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="750" cy="320" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
        <circle cx="750" cy="440" r="6" className="fill-emerald-500 dark:fill-emerald-400" />
      </g>

      {/* Checkmarks (right lane advantages) */}
      <g className="stroke-emerald-500 dark:stroke-emerald-400" strokeWidth="2" opacity="0.6">
        <path d="M 742 200 L 748 206 L 758 194" fill="none" />
        <path d="M 742 320 L 748 326 L 758 314" fill="none" />
        <path d="M 742 440 L 748 446 L 758 434" fill="none" />
      </g>

      {/* Cross marks (left lane disadvantages) */}
      <g className="stroke-blue-500 dark:stroke-blue-400" strokeWidth="2" opacity="0.4">
        <line x1="444" y1="194" x2="456" y2="206" />
        <line x1="456" y1="194" x2="444" y2="206" />
      </g>
    </svg>
  )
}
