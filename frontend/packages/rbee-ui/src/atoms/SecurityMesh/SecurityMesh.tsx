import { cn } from '@rbee/ui/utils'

export interface SecurityMeshProps {
  className?: string
}

/**
 * SecurityMesh - Theme-aware SVG background for security sections
 *
 * Abstract security mesh with linked nodes and amber highlights, suggesting
 * hash-chains, zero-trust architecture, and time-bounded execution.
 * Adapts to light/dark themes.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-15">
 *   <SecurityMesh />
 * </div>
 * ```
 */
export function SecurityMesh({ className }: SecurityMeshProps) {
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
        {/* Hexagonal mesh pattern - light theme */}
        <pattern id="hex-mesh-light" x="0" y="0" width="60" height="52" patternUnits="userSpaceOnUse">
          <path
            d="M30 0 L45 13 L45 39 L30 52 L15 39 L15 13 Z"
            fill="none"
            stroke="rgb(59 130 246)"
            strokeWidth="0.5"
            className="opacity-30"
          />
        </pattern>

        {/* Hexagonal mesh pattern - dark theme */}
        <pattern id="hex-mesh-dark" x="0" y="0" width="60" height="52" patternUnits="userSpaceOnUse">
          <path
            d="M30 0 L45 13 L45 39 L30 52 L15 39 L15 13 Z"
            fill="none"
            stroke="rgb(96 165 250)"
            strokeWidth="0.75"
            className="opacity-40"
          />
        </pattern>

        {/* Glow filter for nodes */}
        <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Hexagonal mesh - light theme */}
      <rect width="1200" height="640" fill="url(#hex-mesh-light)" className="dark:hidden" />

      {/* Hexagonal mesh - dark theme */}
      <rect width="1200" height="640" fill="url(#hex-mesh-dark)" className="hidden dark:block" />

      {/* Security nodes - positioned at mesh intersections */}
      <g filter="url(#node-glow)">
        {/* Top layer nodes */}
        <circle cx="180" cy="130" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="360" cy="130" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="540" cy="130" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="720" cy="130" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="900" cy="130" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="1080" cy="130" r="3" className="fill-blue-500 dark:fill-blue-400" />

        {/* Middle layer nodes with amber accents */}
        <circle cx="120" cy="312" r="2.5" className="fill-amber-500 dark:fill-amber-400 opacity-70" />
        <circle cx="300" cy="312" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="480" cy="312" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="660" cy="312" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="840" cy="312" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="1020" cy="312" r="2.5" className="fill-amber-500 dark:fill-amber-400 opacity-70" />

        {/* Bottom layer nodes */}
        <circle cx="240" cy="494" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="420" cy="494" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="600" cy="494" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="780" cy="494" r="3" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="960" cy="494" r="3" className="fill-blue-500 dark:fill-blue-400" />
      </g>

      {/* Connection lines - light theme */}
      <g className="dark:hidden" opacity="0.35">
        {/* Horizontal connections */}
        <line x1="180" y1="130" x2="360" y2="130" stroke="rgb(59 130 246)" strokeWidth="1" />
        <line x1="360" y1="130" x2="540" y2="130" stroke="rgb(59 130 246)" strokeWidth="1" />
        <line x1="540" y1="130" x2="720" y2="130" stroke="rgb(59 130 246)" strokeWidth="1" />
        <line x1="720" y1="130" x2="900" y2="130" stroke="rgb(59 130 246)" strokeWidth="1" />
        <line x1="900" y1="130" x2="1080" y2="130" stroke="rgb(59 130 246)" strokeWidth="1" />

        <line x1="300" y1="312" x2="480" y2="312" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="480" y1="312" x2="660" y2="312" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="660" y1="312" x2="840" y2="312" stroke="rgb(59 130 246)" strokeWidth="1.5" />

        {/* Diagonal connections (hash-chain pattern) */}
        <line x1="360" y1="130" x2="300" y2="312" stroke="rgb(59 130 246)" strokeWidth="0.75" />
        <line x1="540" y1="130" x2="480" y2="312" stroke="rgb(59 130 246)" strokeWidth="0.75" />
        <line x1="720" y1="130" x2="660" y2="312" stroke="rgb(59 130 246)" strokeWidth="0.75" />
        <line x1="480" y1="312" x2="420" y2="494" stroke="rgb(59 130 246)" strokeWidth="0.75" />
        <line x1="660" y1="312" x2="600" y2="494" stroke="rgb(59 130 246)" strokeWidth="0.75" />

        {/* Amber accent lines (time-bounded execution) */}
        <line x1="120" y1="312" x2="300" y2="312" stroke="rgb(245 158 11)" strokeWidth="0.5" strokeDasharray="4 4" />
        <line x1="840" y1="312" x2="1020" y2="312" stroke="rgb(245 158 11)" strokeWidth="0.5" strokeDasharray="4 4" />
      </g>

      {/* Connection lines - dark theme */}
      <g className="hidden dark:block" opacity="0.45">
        {/* Horizontal connections */}
        <line x1="180" y1="130" x2="360" y2="130" stroke="rgb(96 165 250)" strokeWidth="1.5" />
        <line x1="360" y1="130" x2="540" y2="130" stroke="rgb(96 165 250)" strokeWidth="1.5" />
        <line x1="540" y1="130" x2="720" y2="130" stroke="rgb(96 165 250)" strokeWidth="1.5" />
        <line x1="720" y1="130" x2="900" y2="130" stroke="rgb(96 165 250)" strokeWidth="1.5" />
        <line x1="900" y1="130" x2="1080" y2="130" stroke="rgb(96 165 250)" strokeWidth="1.5" />

        <line x1="300" y1="312" x2="480" y2="312" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="480" y1="312" x2="660" y2="312" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="660" y1="312" x2="840" y2="312" stroke="rgb(96 165 250)" strokeWidth="2" />

        {/* Diagonal connections (hash-chain pattern) */}
        <line x1="360" y1="130" x2="300" y2="312" stroke="rgb(96 165 250)" strokeWidth="1" />
        <line x1="540" y1="130" x2="480" y2="312" stroke="rgb(96 165 250)" strokeWidth="1" />
        <line x1="720" y1="130" x2="660" y2="312" stroke="rgb(96 165 250)" strokeWidth="1" />
        <line x1="480" y1="312" x2="420" y2="494" stroke="rgb(96 165 250)" strokeWidth="1" />
        <line x1="660" y1="312" x2="600" y2="494" stroke="rgb(96 165 250)" strokeWidth="1" />

        {/* Amber accent lines (time-bounded execution) */}
        <line x1="120" y1="312" x2="300" y2="312" stroke="rgb(251 191 36)" strokeWidth="0.75" strokeDasharray="4 4" />
        <line x1="840" y1="312" x2="1020" y2="312" stroke="rgb(251 191 36)" strokeWidth="0.75" strokeDasharray="4 4" />
      </g>
    </svg>
  )
}
