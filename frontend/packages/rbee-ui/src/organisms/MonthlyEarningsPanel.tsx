import { ProgressBar } from '@rbee/ui/molecules'

export interface MonthlyEarningsPanelProps {
  /** Label for the month (e.g., "This Month") */
  monthLabel: string
  /** Earnings amount (e.g., "€2,847") */
  monthEarnings: string
  /** Growth indicator (e.g., "+23% vs last month") */
  monthGrowth: string
  /** Progress percentage for the decorative bar */
  progressPercentage: number
  /** Additional CSS classes */
  className?: string
}

/**
 * MonthlyEarningsPanel - Displays monthly earnings with growth indicator and progress bar
 *
 * @example
 * ```tsx
 * <MonthlyEarningsPanel
 *   monthLabel="This Month"
 *   monthEarnings="€2,847"
 *   monthGrowth="+23% vs last month"
 *   progressPercentage={68}
 * />
 * ```
 */
export function MonthlyEarningsPanel({
  monthLabel,
  monthEarnings,
  monthGrowth,
  progressPercentage,
  className,
}: MonthlyEarningsPanelProps) {
  return (
    <div className={`rounded-lg border border-primary/20 bg-primary/5 p-4 ${className || ''}`}>
      <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">{monthLabel}</div>
      <div className="mb-1 tabular-nums text-4xl font-extrabold text-foreground">{monthEarnings}</div>
      <div className="mb-3 text-sm font-medium text-emerald-400">{monthGrowth}</div>

      {/* Decorative progress bar */}
      <ProgressBar
        label=""
        percentage={progressPercentage}
        size="sm"
        showLabel={false}
        showPercentage={false}
        className="mt-2"
      />
    </div>
  )
}

export { MonthlyEarningsPanel as default }
