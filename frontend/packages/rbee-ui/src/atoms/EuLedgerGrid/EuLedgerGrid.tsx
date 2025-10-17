import { cn } from '@rbee/ui/utils'

export interface EuLedgerGridProps {
  className?: string
}

/**
 * EuLedgerGrid - Theme-aware SVG background for compliance sections
 *
 * Abstract EU-blue ledger grid with glowing checkpoints, implying immutable
 * audit trails and data sovereignty. Adapts to light/dark themes.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-15">
 *   <EuLedgerGrid />
 * </div>
 * ```
 */
export function EuLedgerGrid({ className }: EuLedgerGridProps) {
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
        {/* Grid pattern - light theme */}
        <pattern id="grid-light" x="0" y="0" width="80" height="80" patternUnits="userSpaceOnUse">
          <path d="M 80 0 L 0 0 0 80" fill="none" stroke="rgb(59 130 246)" strokeWidth="0.5" className="opacity-40" />
        </pattern>

        {/* Grid pattern - dark theme */}
        <pattern id="grid-dark" x="0" y="0" width="80" height="80" patternUnits="userSpaceOnUse">
          <path d="M 80 0 L 0 0 0 80" fill="none" stroke="rgb(96 165 250)" strokeWidth="0.75" className="opacity-50" />
        </pattern>

        {/* Glow filter for checkpoints */}
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Radial gradient for ambient glow - light theme */}
        <radialGradient id="ambient-glow-light" cx="50%" cy="30%">
          <stop offset="0%" stopColor="rgb(59 130 246)" stopOpacity="0.05" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0" />
        </radialGradient>

        {/* Radial gradient for ambient glow - dark theme */}
        <radialGradient id="ambient-glow-dark" cx="50%" cy="30%">
          <stop offset="0%" stopColor="rgb(96 165 250)" stopOpacity="0.1" />
          <stop offset="100%" stopColor="rgb(96 165 250)" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Background ambient glow - light theme */}
      <rect width="1200" height="640" fill="url(#ambient-glow-light)" className="dark:hidden" />

      {/* Background ambient glow - dark theme */}
      <rect width="1200" height="640" fill="url(#ambient-glow-dark)" className="hidden dark:block" />

      {/* Grid pattern - light theme */}
      <rect width="1200" height="640" fill="url(#grid-light)" className="dark:hidden" />

      {/* Grid pattern - dark theme */}
      <rect width="1200" height="640" fill="url(#grid-dark)" className="hidden dark:block" />

      {/* Checkpoint nodes - positioned along grid intersections */}
      <g filter="url(#glow)">
        {/* Top row checkpoints */}
        <circle cx="240" cy="160" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="480" cy="160" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="720" cy="160" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="960" cy="160" r="4" className="fill-blue-500 dark:fill-blue-400" />

        {/* Middle row checkpoints with amber accents */}
        <circle cx="160" cy="320" r="3" className="fill-amber-500 dark:fill-amber-400 opacity-60" />
        <circle cx="400" cy="320" r="5" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="600" cy="320" r="5" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="840" cy="320" r="5" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="1040" cy="320" r="3" className="fill-amber-500 dark:fill-amber-400 opacity-60" />

        {/* Bottom row checkpoints */}
        <circle cx="320" cy="480" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="560" cy="480" r="4" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="800" cy="480" r="4" className="fill-blue-500 dark:fill-blue-400" />
      </g>

      {/* Connection lines between checkpoints - light theme */}
      <g className="dark:hidden" opacity="0.4">
        <line x1="240" y1="160" x2="480" y2="160" stroke="rgb(59 130 246)" strokeWidth="1" />
        <line x1="480" y1="160" x2="720" y2="160" stroke="rgb(59 130 246)" strokeWidth="1" />
        <line x1="720" y1="160" x2="960" y2="160" stroke="rgb(59 130 246)" strokeWidth="1" />

        <line x1="400" y1="320" x2="600" y2="320" stroke="rgb(59 130 246)" strokeWidth="1.5" />
        <line x1="600" y1="320" x2="840" y2="320" stroke="rgb(59 130 246)" strokeWidth="1.5" />

        <line x1="320" y1="480" x2="560" y2="480" stroke="rgb(59 130 246)" strokeWidth="1" />
        <line x1="560" y1="480" x2="800" y2="480" stroke="rgb(59 130 246)" strokeWidth="1" />

        {/* Vertical connections */}
        <line x1="480" y1="160" x2="400" y2="320" stroke="rgb(59 130 246)" strokeWidth="0.75" />
        <line x1="720" y1="160" x2="840" y2="320" stroke="rgb(59 130 246)" strokeWidth="0.75" />
        <line x1="600" y1="320" x2="560" y2="480" stroke="rgb(59 130 246)" strokeWidth="0.75" />
      </g>

      {/* Connection lines between checkpoints - dark theme */}
      <g className="hidden dark:block" opacity="0.5">
        <line x1="240" y1="160" x2="480" y2="160" stroke="rgb(96 165 250)" strokeWidth="1.5" />
        <line x1="480" y1="160" x2="720" y2="160" stroke="rgb(96 165 250)" strokeWidth="1.5" />
        <line x1="720" y1="160" x2="960" y2="160" stroke="rgb(96 165 250)" strokeWidth="1.5" />

        <line x1="400" y1="320" x2="600" y2="320" stroke="rgb(96 165 250)" strokeWidth="2" />
        <line x1="600" y1="320" x2="840" y2="320" stroke="rgb(96 165 250)" strokeWidth="2" />

        <line x1="320" y1="480" x2="560" y2="480" stroke="rgb(96 165 250)" strokeWidth="1.5" />
        <line x1="560" y1="480" x2="800" y2="480" stroke="rgb(96 165 250)" strokeWidth="1.5" />

        {/* Vertical connections */}
        <line x1="480" y1="160" x2="400" y2="320" stroke="rgb(96 165 250)" strokeWidth="1" />
        <line x1="720" y1="160" x2="840" y2="320" stroke="rgb(96 165 250)" strokeWidth="1" />
        <line x1="600" y1="320" x2="560" y2="480" stroke="rgb(96 165 250)" strokeWidth="1" />
      </g>

      {/* Subtle amber accent lines - light theme */}
      <g className="dark:hidden" opacity="0.3">
        <line x1="160" y1="320" x2="400" y2="320" stroke="rgb(245 158 11)" strokeWidth="0.5" strokeDasharray="4 4" />
        <line x1="840" y1="320" x2="1040" y2="320" stroke="rgb(245 158 11)" strokeWidth="0.5" strokeDasharray="4 4" />
      </g>

      {/* Subtle amber accent lines - dark theme */}
      <g className="hidden dark:block" opacity="0.35">
        <line x1="160" y1="320" x2="400" y2="320" stroke="rgb(251 191 36)" strokeWidth="0.75" strokeDasharray="4 4" />
        <line x1="840" y1="320" x2="1040" y2="320" stroke="rgb(251 191 36)" strokeWidth="0.75" strokeDasharray="4 4" />
      </g>
    </svg>
  )
}
