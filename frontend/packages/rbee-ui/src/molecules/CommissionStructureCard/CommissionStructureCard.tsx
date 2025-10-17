import { Card, CardContent } from '@rbee/ui/atoms/Card'

export interface CommissionExample {
  label: string
  value: string
}

export interface CommissionStructureCardProps {
  /** Card title */
  title: string
  /** Standard commission label */
  standardCommissionLabel: string
  /** Standard commission value (e.g., "15%") */
  standardCommissionValue: string
  /** Standard commission description */
  standardCommissionDescription: string
  /** You keep label */
  youKeepLabel: string
  /** You keep value (e.g., "85%") */
  youKeepValue: string
  /** You keep description */
  youKeepDescription: string
  /** Example items */
  exampleItems: CommissionExample[]
  /** Example total label */
  exampleTotalLabel: string
  /** Example total value */
  exampleTotalValue: string
  /** Example badge text */
  exampleBadgeText: string
  /** Additional CSS classes */
  className?: string
}

/**
 * CommissionStructureCard - displays commission breakdown and examples
 *
 * @example
 * ```tsx
 * <CommissionStructureCard
 *   title="Commission Structure"
 *   standardCommissionLabel="Standard Commission"
 *   standardCommissionValue="15%"
 *   standardCommissionDescription="Covers marketplace operations"
 *   youKeepLabel="You Keep"
 *   youKeepValue="85%"
 *   youKeepDescription="No hidden fees"
 *   exampleItems={[...]}
 *   exampleTotalLabel="Your earnings"
 *   exampleTotalValue="€85.00"
 *   exampleBadgeText="Effective take-home: 85%"
 * />
 * ```
 */
export function CommissionStructureCard({
  title,
  standardCommissionLabel,
  standardCommissionValue,
  standardCommissionDescription,
  youKeepLabel,
  youKeepValue,
  youKeepDescription,
  exampleItems,
  exampleTotalLabel,
  exampleTotalValue,
  exampleBadgeText,
  className,
}: CommissionStructureCardProps) {
  return (
    <div className={className}>
      <h3 className="mb-6 text-2xl font-bold text-foreground">{title}</h3>
      <div className="space-y-4">
        {/* Standard Commission Card */}
        <Card className="bg-background/60 shadow-sm transition-transform hover:translate-y-0.5">
          <CardContent className="p-6">
            <div className="mb-4 flex items-center justify-between">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">{standardCommissionLabel}</div>
              <div className="tabular-nums text-2xl font-extrabold text-primary">{standardCommissionValue}</div>
            </div>
            <div className="text-sm text-muted-foreground">{standardCommissionDescription}</div>
          </CardContent>
        </Card>

        {/* You Keep Card */}
        <Card className="border-emerald-400/30 bg-emerald-400/10 shadow-sm transition-transform hover:translate-y-0.5">
          <CardContent className="p-6">
            <div className="mb-4 flex items-center justify-between">
              <div className="text-xs uppercase tracking-wide text-emerald-400">{youKeepLabel}</div>
              <div className="tabular-nums text-2xl font-extrabold text-emerald-400">{youKeepValue}</div>
            </div>
            <div className="text-sm text-emerald-400">{youKeepDescription}</div>
          </CardContent>
        </Card>

        {/* Example Table */}
        <Card className="bg-background/60 shadow-sm">
          <CardContent className="space-y-2 p-4 text-sm">
            {exampleItems.map((item, idx) => (
              <div key={idx} className="flex justify-between">
                <span className="text-muted-foreground">{item.label}</span>
                <span className="tabular-nums text-foreground">{item.value}</span>
              </div>
            ))}
            <div className="border-t border-border pt-2">
              <div className="flex justify-between font-semibold">
                <span className="text-foreground">{exampleTotalLabel}</span>
                <span className="tabular-nums text-primary">{exampleTotalValue}</span>
              </div>
            </div>
            <div className="mt-2 inline-flex rounded-full bg-primary/10 px-2.5 py-1 text-[11px] text-primary">
              {exampleBadgeText}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export { CommissionStructureCard as default }
