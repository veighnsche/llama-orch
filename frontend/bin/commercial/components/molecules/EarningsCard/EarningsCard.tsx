import { cn } from '@/lib/utils'

export interface EarningsStat {
  label: string
  value: string | number
}

export interface EarningsCardProps {
  title: string
  amount: number | string
  subtitle?: string
  stats: EarningsStat[]
  breakdown?: EarningsStat[]
  className?: string
}

export function EarningsCard({
  title,
  amount,
  subtitle,
  stats,
  breakdown,
  className,
}: EarningsCardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8',
        className
      )}
    >
      <h3 className="mb-6 text-xl font-bold text-foreground">{title}</h3>

      <div className="rounded-xl border border-primary/20 bg-primary/10 p-6 mb-6">
        <div className="mb-2 text-sm text-primary">Monthly Earnings</div>
        <div className="text-5xl font-bold text-foreground">€{amount}</div>
        {subtitle && (
          <div className="mt-2 text-sm text-muted-foreground">{subtitle}</div>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {stats.map((stat, index) => (
          <div
            key={index}
            className="rounded-lg border border-border bg-background/50 p-4"
          >
            <div className="mb-1 text-sm text-muted-foreground">
              {stat.label}
            </div>
            <div className="text-2xl font-bold text-foreground">
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {breakdown && breakdown.length > 0 && (
        <div className="mt-6 space-y-3">
          <div className="text-sm font-medium text-foreground">Breakdown</div>
          {breakdown.map((item, index) => (
            <div
              key={index}
              className="flex items-center justify-between text-sm"
            >
              <span className="text-muted-foreground">{item.label}</span>
              <span className="font-medium text-foreground">{item.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
