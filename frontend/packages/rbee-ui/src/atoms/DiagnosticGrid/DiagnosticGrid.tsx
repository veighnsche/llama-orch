import { cn } from '@rbee/ui/utils'

export interface DiagnosticGridProps {
  className?: string
}

/**
 * DiagnosticGrid - Theme-aware SVG background for error handling sections
 *
 * Diagnostic grid with error detection points and scan lines,
 * suggesting monitoring and troubleshooting capabilities.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <DiagnosticGrid />
 * </div>
 * ```
 */
export function DiagnosticGrid({ className }: DiagnosticGridProps) {
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
        <filter id="diag-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        
        <pattern id="diag-grid" x="0" y="0" width="80" height="80" patternUnits="userSpaceOnUse">
          <path
            d="M 80 0 L 0 0 0 80"
            fill="none"
            stroke="currentColor"
            strokeWidth="0.5"
            className="text-blue-500 dark:text-blue-400"
          />
        </pattern>
      </defs>

      {/* Diagnostic grid background */}
      <rect width="1200" height="640" fill="url(#diag-grid)" opacity="0.15" />

      {/* Status indicators */}
      <g filter="url(#diag-glow)">
        {/* Normal status (blue) */}
        <circle cx="300" cy="200" r="6" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="600" cy="180" r="6" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="900" cy="220" r="6" className="fill-blue-500 dark:fill-blue-400" />
        
        {/* Warning status (amber) */}
        <circle cx="400" cy="350" r="7" className="fill-amber-500 dark:fill-amber-400" opacity="0.8" />
        <circle cx="800" cy="380" r="7" className="fill-amber-500 dark:fill-amber-400" opacity="0.8" />
        
        {/* Error status (red) */}
        <circle cx="500" cy="500" r="8" className="fill-red-500 dark:fill-red-400" opacity="0.7" />
      </g>

      {/* Alert icons */}
      <g opacity="0.6">
        {/* Warning triangles */}
        <path d="M 400 330 L 410 350 L 390 350 Z" className="fill-amber-500 dark:fill-amber-400" />
        <path d="M 800 360 L 810 380 L 790 380 Z" className="fill-amber-500 dark:fill-amber-400" />
        
        {/* Error cross */}
        <g className="stroke-red-500 dark:stroke-red-400" strokeWidth="2.5">
          <line x1="494" y1="494" x2="506" y2="506" />
          <line x1="506" y1="494" x2="494" y2="506" />
        </g>
      </g>

      {/* Diagnostic scan lines */}
      <g className="stroke-blue-500 dark:stroke-blue-400" strokeWidth="1" opacity="0.3">
        <line x1="200" y1="320" x2="1000" y2="320" strokeDasharray="8 4" />
        <line x1="600" y1="100" x2="600" y2="540" strokeDasharray="8 4" />
      </g>

      {/* Scan crosshair */}
      <g className="stroke-blue-500 dark:stroke-blue-400" strokeWidth="1.5" opacity="0.4">
        <circle cx="600" cy="320" r="50" fill="none" />
        <line x1="550" y1="320" x2="570" y2="320" />
        <line x1="630" y1="320" x2="650" y2="320" />
        <line x1="600" y1="270" x2="600" y2="290" />
        <line x1="600" y1="350" x2="600" y2="370" />
      </g>
    </svg>
  )
}
