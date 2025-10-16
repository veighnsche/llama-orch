import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms/Card'
import { cn } from '@rbee/ui/utils'

export interface EarningsBreakdownRow {
  label: string
  value: string | number
}

export interface EarningsBreakdownProps {
  /** Card title */
  title: string
  /** Hourly rate row */
  hourlyRate: EarningsBreakdownRow
  /** Hours per month row */
  hoursPerMonth: EarningsBreakdownRow
  /** Utilization row with percentage */
  utilization: {
    label: string
    value: number // percentage 0-100
  }
  /** Commission row */
  commission: EarningsBreakdownRow
  /** Take-home row (highlighted) */
  takeHome: EarningsBreakdownRow
  /** Additional CSS classes */
  className?: string
}

/**
 * EarningsBreakdownCard - displays a detailed breakdown of earnings calculation
 *
 * @example
 * ```tsx
 * <EarningsBreakdownCard
 *   title="Breakdown"
 *   hourlyRate={{ label: "Hourly rate", value: "€0.45/hr" }}
 *   hoursPerMonth={{ label: "Hours per month", value: "600h" }}
 *   utilization={{ label: "Utilization", value: 80 }}
 *   commission={{ label: "rbee commission (15%)", value: "-€32" }}
 *   takeHome={{ label: "Your take-home", value: "€184" }}
 * />
 * ```
 */
export function EarningsBreakdownCard({
  title,
  hourlyRate,
  hoursPerMonth,
  utilization,
  commission,
  takeHome,
  className,
}: EarningsBreakdownProps) {
  return (
    <Card className={cn('bg-background/50', className)}>
      <CardHeader className="pb-0">
        <CardTitle className="text-sm font-mono">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 p-4 font-mono">
        {/* Hourly Rate */}
        <div className="flex justify-between text-sm">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">{hourlyRate.label}:</span>
          <span className="tabular-nums text-sm text-foreground">{hourlyRate.value}</span>
        </div>

        {/* Hours Per Month */}
        <div className="flex justify-between text-sm">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">{hoursPerMonth.label}:</span>
          <span className="tabular-nums text-sm text-foreground">{hoursPerMonth.value}</span>
        </div>

        {/* Utilization with Progress Bar */}
        <div>
          <div className="flex justify-between text-sm">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">{utilization.label}:</span>
            <span className="tabular-nums text-sm text-foreground">{utilization.value}%</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded bg-primary/15">
            <div className="h-full rounded bg-primary transition-all" style={{ width: `${utilization.value}%` }} />
          </div>
        </div>

        {/* Commission & Take-Home Section */}
        <div className="border-t border-border pt-3">
          <div className="flex justify-between text-sm">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">{commission.label}:</span>
            <span className="tabular-nums text-sm text-foreground">{commission.value}</span>
          </div>
          <div className="mt-2 flex justify-between font-medium">
            <span className="text-foreground">{takeHome.label}:</span>
            <span className="tabular-nums text-primary">{takeHome.value}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export { EarningsBreakdownCard as default }
