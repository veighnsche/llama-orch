import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

export interface ComplianceChipProps {
  /** Icon element (e.g., Lucide icon) */
  icon?: ReactNode
  /** Chip label */
  children: ReactNode
  /** Accessible label for screen readers */
  ariaLabel?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * ComplianceChip molecule for compliance proof indicators
 * Compact, chip-style badges with optional icons
 */
export function ComplianceChip({ icon, children, ariaLabel, className }: ComplianceChipProps) {
  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-card/40 px-3 py-1.5 text-xs',
        'transition-all duration-200 hover:border-border/80 hover:bg-card/60',
        className,
      )}
      role="status"
      aria-label={ariaLabel}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      <span>{children}</span>
    </div>
  )
}
