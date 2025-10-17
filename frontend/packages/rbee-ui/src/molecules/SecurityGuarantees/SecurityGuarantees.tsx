import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type SecurityGuaranteeStat = {
  value: string
  label: string
  ariaLabel?: string
}

export type SecurityGuaranteesProps = {
  heading: string
  stats: SecurityGuaranteeStat[]
  footnote: string
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function SecurityGuarantees({ heading, stats, footnote, className }: SecurityGuaranteesProps) {
  return (
    <Card className={className}>
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">{heading}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6 md:grid-cols-3">
          {stats.map((stat, idx) => (
            <div key={idx} className="text-center">
              <div className="mb-2 text-3xl font-bold text-primary" aria-label={stat.ariaLabel}>
                {stat.value}
              </div>
              <div className="text-sm text-foreground/85">{stat.label}</div>
            </div>
          ))}
        </div>
        <p className="mt-6 text-center text-xs text-muted-foreground">{footnote}</p>
      </CardContent>
    </Card>
  )
}
