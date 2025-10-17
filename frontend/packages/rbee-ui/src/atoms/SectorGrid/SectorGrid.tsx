import { cn } from '@rbee/ui/utils'

export interface SectorGridProps {
  className?: string
}

/**
 * SectorGrid - Theme-aware SVG background for industry use cases sections
 *
 * Abstract EU-blue grid of industry tiles—finance, healthcare, legal, government—
 * with soft amber accents. Premium dark UI with compliance theme.
 * Adapts to light/dark themes.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-15">
 *   <SectorGrid />
 * </div>
 * ```
 */
export function SectorGrid({ className }: SectorGridProps) {
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
        {/* Tile grid pattern - light theme */}
        <pattern id="sector-grid-light" x="0" y="0" width="120" height="120" patternUnits="userSpaceOnUse">
          <rect
            x="2"
            y="2"
            width="116"
            height="116"
            fill="none"
            stroke="rgb(59 130 246)"
            strokeWidth="1"
            className="opacity-25"
          />
          <rect
            x="8"
            y="8"
            width="104"
            height="104"
            fill="none"
            stroke="rgb(59 130 246)"
            strokeWidth="0.5"
            className="opacity-15"
          />
        </pattern>

        {/* Tile grid pattern - dark theme */}
        <pattern id="sector-grid-dark" x="0" y="0" width="120" height="120" patternUnits="userSpaceOnUse">
          <rect
            x="2"
            y="2"
            width="116"
            height="116"
            fill="none"
            stroke="rgb(96 165 250)"
            strokeWidth="1.5"
            className="opacity-35"
          />
          <rect
            x="8"
            y="8"
            width="104"
            height="104"
            fill="none"
            stroke="rgb(96 165 250)"
            strokeWidth="0.75"
            className="opacity-20"
          />
        </pattern>

        {/* Sector highlight glow */}
        <filter id="sector-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Grid pattern background - light theme */}
      <rect width="1200" height="640" fill="url(#sector-grid-light)" className="dark:hidden" />

      {/* Grid pattern background - dark theme */}
      <rect width="1200" height="640" fill="url(#sector-grid-dark)" className="hidden dark:block" />

      {/* Sector tiles (4 quadrants representing industries) */}
      <g filter="url(#sector-glow)">
        {/* Finance sector (top-left) */}
        <g opacity="0.6">
          <rect
            x="180"
            y="120"
            width="200"
            height="180"
            fill="none"
            stroke="rgb(59 130 246)"
            strokeWidth="2"
            className="dark:stroke-blue-400"
          />
          <circle cx="280" cy="210" r="4" className="fill-blue-500 dark:fill-blue-400" />
          <text
            x="280"
            y="165"
            textAnchor="middle"
            className="fill-blue-600 dark:fill-blue-300 text-xs font-medium"
            opacity="0.5"
          >
            FIN
          </text>
        </g>

        {/* Healthcare sector (top-right) */}
        <g opacity="0.6">
          <rect
            x="820"
            y="120"
            width="200"
            height="180"
            fill="none"
            stroke="rgb(59 130 246)"
            strokeWidth="2"
            className="dark:stroke-blue-400"
          />
          <circle cx="920" cy="210" r="4" className="fill-blue-500 dark:fill-blue-400" />
          <text
            x="920"
            y="165"
            textAnchor="middle"
            className="fill-blue-600 dark:fill-blue-300 text-xs font-medium"
            opacity="0.5"
          >
            MED
          </text>
        </g>

        {/* Legal sector (bottom-left) */}
        <g opacity="0.6">
          <rect
            x="180"
            y="340"
            width="200"
            height="180"
            fill="none"
            stroke="rgb(59 130 246)"
            strokeWidth="2"
            className="dark:stroke-blue-400"
          />
          <circle cx="280" cy="430" r="4" className="fill-blue-500 dark:fill-blue-400" />
          <text
            x="280"
            y="385"
            textAnchor="middle"
            className="fill-blue-600 dark:fill-blue-300 text-xs font-medium"
            opacity="0.5"
          >
            LEG
          </text>
        </g>

        {/* Government sector (bottom-right) */}
        <g opacity="0.6">
          <rect
            x="820"
            y="340"
            width="200"
            height="180"
            fill="none"
            stroke="rgb(59 130 246)"
            strokeWidth="2"
            className="dark:stroke-blue-400"
          />
          <circle cx="920" cy="430" r="4" className="fill-blue-500 dark:fill-blue-400" />
          <text
            x="920"
            y="385"
            textAnchor="middle"
            className="fill-blue-600 dark:fill-blue-300 text-xs font-medium"
            opacity="0.5"
          >
            GOV
          </text>
        </g>
      </g>

      {/* Connecting lines between sectors - light theme */}
      <g className="dark:hidden" opacity="0.3">
        <line x1="380" y1="210" x2="820" y2="210" stroke="rgb(59 130 246)" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="280" y1="300" x2="280" y2="340" stroke="rgb(59 130 246)" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="920" y1="300" x2="920" y2="340" stroke="rgb(59 130 246)" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="380" y1="430" x2="820" y2="430" stroke="rgb(59 130 246)" strokeWidth="1" strokeDasharray="4 4" />
      </g>

      {/* Connecting lines between sectors - dark theme */}
      <g className="hidden dark:block" opacity="0.4">
        <line x1="380" y1="210" x2="820" y2="210" stroke="rgb(96 165 250)" strokeWidth="1.5" strokeDasharray="4 4" />
        <line x1="280" y1="300" x2="280" y2="340" stroke="rgb(96 165 250)" strokeWidth="1.5" strokeDasharray="4 4" />
        <line x1="920" y1="300" x2="920" y2="340" stroke="rgb(96 165 250)" strokeWidth="1.5" strokeDasharray="4 4" />
        <line x1="380" y1="430" x2="820" y2="430" stroke="rgb(96 165 250)" strokeWidth="1.5" strokeDasharray="4 4" />
      </g>

      {/* Amber accent lines (compliance highlights) - light theme */}
      <g className="dark:hidden" opacity="0.25">
        <line x1="180" y1="120" x2="380" y2="120" stroke="rgb(245 158 11)" strokeWidth="1.5" />
        <line x1="820" y1="120" x2="1020" y2="120" stroke="rgb(245 158 11)" strokeWidth="1.5" />
        <line x1="180" y1="520" x2="380" y2="520" stroke="rgb(245 158 11)" strokeWidth="1.5" />
        <line x1="820" y1="520" x2="1020" y2="520" stroke="rgb(245 158 11)" strokeWidth="1.5" />
      </g>

      {/* Amber accent lines (compliance highlights) - dark theme */}
      <g className="hidden dark:block" opacity="0.35">
        <line x1="180" y1="120" x2="380" y2="120" stroke="rgb(251 191 36)" strokeWidth="2" />
        <line x1="820" y1="120" x2="1020" y2="120" stroke="rgb(251 191 36)" strokeWidth="2" />
        <line x1="180" y1="520" x2="380" y2="520" stroke="rgb(251 191 36)" strokeWidth="2" />
        <line x1="820" y1="520" x2="1020" y2="520" stroke="rgb(251 191 36)" strokeWidth="2" />
      </g>

      {/* Corner markers (compliance badges) */}
      <g opacity="0.4">
        <circle cx="180" cy="120" r="3" className="fill-amber-500 dark:fill-amber-400" />
        <circle cx="1020" cy="120" r="3" className="fill-amber-500 dark:fill-amber-400" />
        <circle cx="180" cy="520" r="3" className="fill-amber-500 dark:fill-amber-400" />
        <circle cx="1020" cy="520" r="3" className="fill-amber-500 dark:fill-amber-400" />
      </g>
    </svg>
  )
}
