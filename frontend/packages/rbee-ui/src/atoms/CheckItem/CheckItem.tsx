import { type ReactNode } from 'react'
import { cn } from '@rbee/ui/utils'

export interface CheckItemProps {
  /** Item content */
  children: ReactNode
  /** Additional CSS classes */
  className?: string
}

/**
 * CheckItem atom for consistent checklist items
 * Used in security crates, compliance pillars, and feature lists
 */
export function CheckItem({ children, className }: CheckItemProps) {
  return (
    <li className={cn('flex gap-2 text-sm text-muted-foreground', className)}>
      <span className="mt-0.5 h-4 w-4 shrink-0 text-chart-3" aria-hidden="true">
        ✓
      </span>
      <span>{children}</span>
    </li>
  )
}
