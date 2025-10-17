import { cn } from '@rbee/ui/utils'

export interface PricingTiersProps {
  className?: string
}

/**
 * PricingTiers - Theme-aware SVG background for pricing sections
 *
 * Layered horizontal bands representing pricing tiers with ascending pattern
 * and highlight on the popular/recommended tier.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <PricingTiers />
 * </div>
 * ```
 */
export function PricingTiers({ className }: PricingTiersProps) {
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
        <filter id="tier-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <linearGradient id="tier-gradient-1" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgb(59 130 246)" stopOpacity="0.1" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0.3" />
        </linearGradient>

        <linearGradient id="tier-gradient-2" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgb(59 130 246)" stopOpacity="0.2" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0.4" />
        </linearGradient>

        <linearGradient id="tier-gradient-3" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgb(59 130 246)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0.5" />
        </linearGradient>
      </defs>

      {/* Tier bands - light theme */}
      <g className="dark:hidden">
        {/* Basic tier */}
        <rect x="200" y="400" width="800" height="120" fill="url(#tier-gradient-1)" rx="8" />

        {/* Popular tier (highlighted) */}
        <rect x="200" y="260" width="800" height="120" fill="url(#tier-gradient-2)" rx="8" />
        <rect x="200" y="260" width="800" height="120" className="fill-amber-500" opacity="0.1" rx="8" />

        {/* Premium tier */}
        <rect x="200" y="120" width="800" height="120" fill="url(#tier-gradient-3)" rx="8" />
      </g>

      {/* Tier bands - dark theme */}
      <g className="hidden dark:block">
        {/* Basic tier */}
        <rect x="200" y="400" width="800" height="120" className="fill-blue-400" opacity="0.15" rx="8" />

        {/* Popular tier (highlighted) */}
        <rect x="200" y="260" width="800" height="120" className="fill-blue-400" opacity="0.25" rx="8" />
        <rect x="200" y="260" width="800" height="120" className="fill-amber-400" opacity="0.15" rx="8" />

        {/* Premium tier */}
        <rect x="200" y="120" width="800" height="120" className="fill-blue-400" opacity="0.35" rx="8" />
      </g>

      {/* Tier indicators */}
      <g filter="url(#tier-glow)">
        <circle cx="250" cy="460" r="5" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="250" cy="320" r="5" className="fill-blue-500 dark:fill-blue-400" />
        <circle cx="250" cy="180" r="5" className="fill-blue-500 dark:fill-blue-400" />
      </g>

      {/* Popular tier badge */}
      <g filter="url(#tier-glow)">
        <path
          d="M 600 280 L 610 300 L 632 304 L 616 320 L 620 342 L 600 332 L 580 342 L 584 320 L 568 304 L 590 300 Z"
          className="fill-amber-500 dark:fill-amber-400"
          opacity="0.6"
        />
      </g>

      {/* Ascending arrows */}
      <g className="stroke-blue-500 dark:stroke-blue-400" strokeWidth="2" opacity="0.3">
        <path d="M 950 480 L 950 440 L 940 450 M 950 440 L 960 450" fill="none" />
        <path d="M 950 340 L 950 300 L 940 310 M 950 300 L 960 310" fill="none" />
        <path d="M 950 200 L 950 160 L 940 170 M 950 160 L 960 170" fill="none" />
      </g>
    </svg>
  )
}
