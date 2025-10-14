import * as React from 'react'
import { cn } from '@rbee/ui/utils'

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
  ({ className, id, opacity = 0.1, size = 'large', fadeDirection = 'radial' }, ref) => {
    const patternId = `honeycomb-${id}`
    const maskId = `fade-mask-${id}`
    const gradientId = `fade-gradient-${id}`

    return (
      <div
        ref={ref}
        className={cn('absolute inset-0 pointer-events-none select-none', className)}
        style={{ opacity }}
      >
        <svg
          className="w-full h-full text-muted-foreground"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <defs>
            {size === 'small' ? (
              <pattern id={patternId} x="0" y="0" width="40" height="69.28" patternUnits="userSpaceOnUse">
                <path
                  d="M20 47.12L0 35.71L0 11.43L20 0L40 11.43L40 35.71L20 47.12L20 69.28"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                />
                <path
                  d="M20 0L20 24.28L0 35.71L0 60L20 69.28L40 60L40 35.71L20 24.28"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                />
              </pattern>
            ) : (
              <pattern id={patternId} x="0" y="0" width="112" height="200" patternUnits="userSpaceOnUse">
                <path
                  d="M56 132L0 100L0 32L56 0L112 32L112 100L56 132L56 200"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                />
                <path
                  d="M56 0L56 68L0 100L0 168L56 200L112 168L112 100L56 68"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
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
  }
)

HoneycombPattern.displayName = 'HoneycombPattern'
