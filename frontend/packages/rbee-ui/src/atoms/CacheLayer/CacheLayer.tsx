import { cn } from '@rbee/ui/utils'

export interface CacheLayerProps {
  className?: string
}

/**
 * CacheLayer - Theme-aware SVG background for intelligent model management sections
 *
 * Layered cache with model icons and hit/miss indicators, suggesting
 * efficient resource management and optimization.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <CacheLayer />
 * </div>
 * ```
 */
export function CacheLayer({ className }: CacheLayerProps) {
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
        <filter id="cache-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Cache layers - light theme */}
      <g className="dark:hidden" opacity="0.3">
        <rect x="200" y="150" width="800" height="80" className="fill-blue-500" opacity="0.15" rx="6" />
        <rect x="200" y="280" width="800" height="80" className="fill-blue-500" opacity="0.2" rx="6" />
        <rect x="200" y="410" width="800" height="80" className="fill-blue-500" opacity="0.25" rx="6" />
      </g>

      {/* Cache layers - dark theme */}
      <g className="hidden dark:block" opacity="0.4">
        <rect x="200" y="150" width="800" height="80" className="fill-blue-400" opacity="0.2" rx="6" />
        <rect x="200" y="280" width="800" height="80" className="fill-blue-400" opacity="0.25" rx="6" />
        <rect x="200" y="410" width="800" height="80" className="fill-blue-400" opacity="0.3" rx="6" />
      </g>

      {/* Model rectangles */}
      <g filter="url(#cache-glow)">
        {/* L1 Cache - Hot models */}
        <rect x="250" y="170" width="60" height="40" className="fill-emerald-500 dark:fill-emerald-400" opacity="0.5" rx="3" />
        <rect x="350" y="170" width="60" height="40" className="fill-emerald-500 dark:fill-emerald-400" opacity="0.5" rx="3" />
        
        {/* L2 Cache - Warm models */}
        <rect x="250" y="300" width="60" height="40" className="fill-amber-500 dark:fill-amber-400" opacity="0.5" rx="3" />
        <rect x="350" y="300" width="60" height="40" className="fill-amber-500 dark:fill-amber-400" opacity="0.5" rx="3" />
        <rect x="450" y="300" width="60" height="40" className="fill-amber-500 dark:fill-amber-400" opacity="0.5" rx="3" />
        
        {/* L3 Cache - Cold models */}
        <rect x="250" y="430" width="60" height="40" className="fill-blue-500 dark:fill-blue-400" opacity="0.4" rx="3" />
        <rect x="350" y="430" width="60" height="40" className="fill-blue-500 dark:fill-blue-400" opacity="0.4" rx="3" />
      </g>

      {/* Arrow flows (cache hits) */}
      <g className="stroke-emerald-500 dark:stroke-emerald-400" strokeWidth="2" opacity="0.5">
        <path d="M 320 170 L 320 120 L 310 130 M 320 120 L 330 130" fill="none" />
        <path d="M 380 170 L 380 120 L 370 130 M 380 120 L 390 130" fill="none" />
      </g>

      {/* Layer labels */}
      <g className="fill-blue-500 dark:fill-blue-400" opacity="0.6">
        <text x="900" y="195" fontSize="14" fontWeight="600">L1</text>
        <text x="900" y="325" fontSize="14" fontWeight="600">L2</text>
        <text x="900" y="455" fontSize="14" fontWeight="600">L3</text>
      </g>
    </svg>
  )
}
