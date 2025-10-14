import * as React from 'react'
import { cn } from '@/lib/utils'

export interface BeeGlyphProps extends React.SVGAttributes<SVGSVGElement> {
  className?: string
}

export const BeeGlyph = React.forwardRef<SVGSVGElement, BeeGlyphProps>(
  ({ className, ...props }, ref) => {
    return (
      <svg
        ref={ref}
        className={cn('w-16 h-16 text-foreground', className)}
        viewBox="0 0 64 64"
        fill="currentColor"
        aria-hidden="true"
        {...props}
      >
        <circle cx="32" cy="32" r="24" />
        <path d="M20 32h24M32 20v24" stroke="white" strokeWidth="4" />
      </svg>
    )
  }
)

BeeGlyph.displayName = 'BeeGlyph'
