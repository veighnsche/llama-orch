import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { StatsGrid } from '@rbee/ui/molecules'

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
        <StatsGrid
          stats={stats.map((stat) => ({
            value: stat.value,
            label: stat.label,
            helpText: stat.ariaLabel,
            valueTone: 'primary' as const,
          }))}
          variant="cards"
          columns={3}
        />
        <p className="mt-6 text-center text-xs text-muted-foreground">{footnote}</p>
      </CardContent>
    </Card>
  )
}
