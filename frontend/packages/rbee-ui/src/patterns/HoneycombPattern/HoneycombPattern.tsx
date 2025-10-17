import { cn } from '@rbee/ui/utils'
import * as React from 'react'

export interface HoneycombPatternProps {
  className?: string
  /** Unique ID for the pattern (required to avoid conflicts when multiple patterns exist) */
  id: string
  opacity?: number
  /** Size of honeycomb cells: 'small' for repeating pattern, 'large' for big cells */
  size?: 'small' | 'large'
  /** Gradient fade direction: 'radial' (all sides), 'bottom' (bottom only), 'none' */
  fadeDirection?: 'radial' | 'bottom' | 'none'
}

export const HoneycombPattern = React.forwardRef<HTMLDivElement, HoneycombPatternProps>(
  ({ className, id, opacity = 0.2, size = 'large', fadeDirection = 'radial' }, ref) => {
    const patternId = `honeycomb-${id}`
    const maskId = `fade-mask-${id}`
    const gradientId = `fade-gradient-${id}`

    return (
      <div ref={ref} className={cn('absolute inset-0 pointer-events-none select-none', className)} style={{ opacity }}>
        <svg className="w-full h-full text-muted-foreground" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <defs>
            {size === 'small' ? (
              <pattern id={patternId} x="0" y="0" width="50" height="86.6" patternUnits="userSpaceOnUse">
                <path
                  d="M25 58.9L0 44.64L0 14.29L25 0L50 14.29L50 44.64L25 58.9L25 86.6"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="M25 0L25 30.35L0 44.64L0 75L25 86.6L50 75L50 44.64L25 30.35"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
              </pattern>
            ) : (
              <pattern id={patternId} x="0" y="0" width="140" height="250" patternUnits="userSpaceOnUse">
                <path
                  d="M70 165L0 125L0 40L70 0L140 40L140 125L70 165L70 250"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="M70 0L70 85L0 125L0 210L70 250L140 210L140 125L70 85"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
              </pattern>
            )}
            {fadeDirection !== 'none' && (
              <mask id={maskId}>
                {fadeDirection === 'radial' && (
                  <radialGradient id={gradientId} cx="50%" cy="50%">
                    <stop offset="0%" stopColor="white" />
                    <stop offset="70%" stopColor="white" />
                    <stop offset="100%" stopColor="white" stopOpacity="0" />
                  </radialGradient>
                )}
                {fadeDirection === 'bottom' && (
                  <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="white" />
                    <stop offset="50%" stopColor="white" />
                    <stop offset="70%" stopColor="white" stopOpacity="0.8" />
                    <stop offset="85%" stopColor="white" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="white" stopOpacity="0" />
                  </linearGradient>
                )}
                <rect width="100%" height="100%" fill={`url(#${gradientId})`} />
              </mask>
            )}
          </defs>
          <rect
            width="100%"
            height="100%"
            fill={`url(#${patternId})`}
            mask={fadeDirection !== 'none' ? `url(#${maskId})` : undefined}
          />
        </svg>
      </div>
    )
  },
)

HoneycombPattern.displayName = 'HoneycombPattern'
