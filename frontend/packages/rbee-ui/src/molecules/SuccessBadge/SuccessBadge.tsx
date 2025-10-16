import { cn } from '@rbee/ui/utils'

export interface SuccessBadgeProps {
  /** Badge content */
  children: React.ReactNode
  /** Additional CSS classes */
  className?: string
}

/**
 * SuccessBadge molecule - success indicator badge with checkmark styling
 * Used for showing successful states or enabled features
 *
 * @example
 * <SuccessBadge>✓ Enabled</SuccessBadge>
 */
export function SuccessBadge({ children, className }: SuccessBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-full',
        'bg-chart-3/10 text-chart-3 px-3 py-1 text-xs font-semibold',
        className,
      )}
    >
      {children}
    </span>
  )
}

export { SuccessBadge as default }
