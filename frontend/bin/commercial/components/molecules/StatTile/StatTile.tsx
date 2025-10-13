import { cn } from '@/lib/utils'

export interface StatTileProps {
  /** Stat value (e.g., "100%", "7 Years") */
  value: string
  /** Stat label/explanation */
  label: string
  /** Optional help text for accessibility */
  helpText?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * StatTile molecule for compliance/trust statistics
 * Designed for enterprise hero sections with strong visual hierarchy
 */
export function StatTile({ value, label, helpText, className }: StatTileProps) {
  return (
    <div
      className={cn(
        'h-full rounded-xl border border-border/70 bg-card/50 p-5',
        'transition-all duration-200 hover:border-border hover:bg-card/70',
        className,
      )}
      role="group"
      aria-label={`${value} ${label}`}
    >
      <div className="mb-2 text-3xl font-bold text-primary">{value}</div>
      <div className="text-sm leading-snug text-muted-foreground">{label}</div>
      {helpText && <span className="sr-only">{helpText}</span>}
    </div>
  )
}
