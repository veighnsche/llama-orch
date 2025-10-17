import { cn } from '@rbee/ui/utils'

export interface NetworkMeshProps {
  className?: string
}

/**
 * NetworkMesh - Theme-aware SVG background for problem/challenge sections
 *
 * Abstract network with broken connections and warning nodes, suggesting
 * infrastructure challenges and dependency risks. Adapts to light/dark themes.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <NetworkMesh />
 * </div>
 * ```
 */
export function NetworkMesh({ className }: NetworkMeshProps) {
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
        {/* Warning glow filter */}
        <filter id="warning-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Network nodes - some disconnected */}
      <g filter="url(#warning-glow)">
        {/* Connected nodes (blue) */}
        <circle cx="200" cy="200" r="5" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="400" cy="180" r="5" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="600" cy="220" r="5" className="fill-blue-500 dark:fill-blue-400" />
        
        {/* Warning nodes (amber) */}
        <circle cx="800" cy="200" r="6" className="fill-amber-500 dark:fill-amber-400" opacity="0.8" />
        <circle cx="1000" cy="240" r="6" className="fill-amber-500 dark:fill-amber-400" opacity="0.8" />
        
        {/* Disconnected nodes (red) */}
        <circle cx="300" cy="400" r="5" className="fill-red-500 dark:fill-red-400" opacity="0.7" />
        <circle cx="700" cy="420" r="5" className="fill-red-500 dark:fill-red-400" opacity="0.7" />
        <circle cx="900" cy="450" r="5" className="fill-red-500 dark:fill-red-400" opacity="0.7" />
      </g>

      {/* Connection lines - light theme */}
      <g className="dark:hidden" opacity="0.4">
        {/* Working connections */}
        <line x1="200" y1="200" x2="400" y2="180" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="400" y1="180" x2="600" y2="220" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        
        {/* Degraded connections */}
        <line x1="600" y1="220" x2="800" y2="200" stroke="rgb(245 158 11)" strokeWidth="1.5" strokeDasharray="4 4" />
        <line x1="800" y1="200" x2="1000" y2="240" stroke="rgb(245 158 11)" strokeWidth="1.5" strokeDasharray="4 4" />
        
        {/* Broken connections */}
        <line x1="300" y1="400" x2="400" y2="380" stroke="rgb(239 68 68)" strokeWidth="1" strokeDasharray="2 6" opacity="0.5" />
        <line x1="700" y1="420" x2="800" y2="400" stroke="rgb(239 68 68)" strokeWidth="1" strokeDasharray="2 6" opacity="0.5" />
      </g>

      {/* Connection lines - dark theme */}
      <g className="hidden dark:block" opacity="0.5">
        {/* Working connections */}
        <line x1="200" y1="200" x2="400" y2="180" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="400" y1="180" x2="600" y2="220" stroke="rgb(96 165 250)" strokeWidth="2" />
        
        {/* Degraded connections */}
        <line x1="600" y1="220" x2="800" y2="200" stroke="rgb(251 191 36)" strokeWidth="2" strokeDasharray="4 4" />
        <line x1="800" y1="200" x2="1000" y2="240" stroke="rgb(251 191 36)" strokeWidth="2" strokeDasharray="4 4" />
        
        {/* Broken connections */}
        <line x1="300" y1="400" x2="400" y2="380" stroke="rgb(248 113 113)" strokeWidth="1.5" strokeDasharray="2 6" opacity="0.6" />
        <line x1="700" y1="420" x2="800" y2="400" stroke="rgb(248 113 113)" strokeWidth="1.5" strokeDasharray="2 6" opacity="0.6" />
      </g>

      {/* Warning indicators */}
      <g opacity="0.3">
        {/* Alert triangles */}
        <path d="M 800 180 L 810 200 L 790 200 Z" className="fill-amber-500 dark:fill-amber-400" />
        <path d="M 1000 220 L 1010 240 L 990 240 Z" className="fill-amber-500 dark:fill-amber-400" />
        
        {/* Error crosses */}
        <g className="stroke-red-500 dark:stroke-red-400" strokeWidth="2">
          <line x1="295" y1="395" x2="305" y2="405" />
          <line x1="305" y1="395" x2="295" y2="405" />
          <line x1="695" y1="415" x2="705" y2="425" />
          <line x1="705" y1="415" x2="695" y2="425" />
        </g>
      </g>

      {/* Background grid (faded) */}
      <g opacity="0.08">
        <defs>
          <pattern id="problem-grid" x="0" y="0" width="60" height="60" patternUnits="userSpaceOnUse">
            <path
              d="M 60 0 L 0 0 0 60"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.5"
              className="text-slate-500"
            />
          </pattern>
        </defs>
        <rect width="1200" height="640" fill="url(#problem-grid)" />
      </g>
    </svg>
  )
}
