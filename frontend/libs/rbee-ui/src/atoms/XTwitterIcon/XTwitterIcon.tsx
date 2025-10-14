import * as React from 'react'
import { cn } from '@rbee/ui/utils'

export interface XTwitterIconProps extends React.SVGAttributes<SVGSVGElement> {
  className?: string
}

export const XTwitterIcon = React.forwardRef<SVGSVGElement, XTwitterIconProps>(
  ({ className, ...props }, ref) => {
    return (
      <svg
        ref={ref}
        className={cn('size-5', className)}
        viewBox="0 0 24 24"
        fill="currentColor"
        aria-hidden="true"
        {...props}
      >
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
      </svg>
    )
  }
)

XTwitterIcon.displayName = 'XTwitterIcon'
