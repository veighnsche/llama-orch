'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import type { Provider, Row, RowGroup } from '@rbee/ui/molecules'
import { MatrixTable } from '@rbee/ui/molecules'
import { Check, X } from 'lucide-react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface PricingComparisonCTA {
  text: string
  href: string
  variant?: 'default' | 'outline'
}

export interface PricingComparisonTemplateProps {
  /** Section title */
  title: string
  /** Section subtitle */
  subtitle: string
  /** Last updated text */
  lastUpdated?: string
  /** Legend items configuration */
  legend: {
    includedText: string
    notAvailableText: string
  }
  /** Key differences list */
  keyDifferences: string[]
  /** Table columns (providers) */
  columns: Provider[]
  /** Table rows (features) */
  rows: Row[]
  /** Table groups */
  groups: RowGroup[]
  /** Table caption for accessibility */
  tableCaption: string
  /** CTA strip configuration */
  cta: {
    text: string
    buttons: PricingComparisonCTA[]
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * Pricing comparison table with feature matrix
 *
 * @example
 * ```tsx
 * <PricingComparisonTemplate
 *   title="Detailed Feature Comparison"
 *   subtitle="What changes across Home/Lab, Team, and Enterprise."
 *   lastUpdated="This month"
 *   legend={{
 *     includedText: 'Included',
 *     notAvailableText: 'Not available'
 *   }}
 *   keyDifferences={[
 *     'Team adds Web UI + collaboration',
 *     'Enterprise adds SLA + white-label + services'
 *   ]}
 *   columns={[...]}
 *   rows={[...]}
 *   groups={[...]}
 *   tableCaption="Feature availability comparison"
 *   cta={{
 *     text: 'Ready to get started?',
 *     buttons: [
 *       { text: 'Start with Team', href: '/signup' },
 *       { text: 'Talk to Sales', href: '/contact', variant: 'outline' }
 *     ]
 *   }}
 * />
 * ```
 */
export function PricingComparisonTemplate({
  title,
  subtitle,
  lastUpdated,
  legend,
  keyDifferences,
  columns,
  rows,
  groups,
  tableCaption,
  cta,
}: PricingComparisonTemplateProps) {
  return (
    <>
      {/* Header */}
      <div className="grid grid-cols-1 md:grid-cols-2 items-start gap-6 mb-6 animate-in fade-in slide-in-from-bottom-1 duration-500 ease-out">
        {/* Left: Title + Value Prop */}
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">{title}</h2>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>

        {/* Right: Legend + Decisions Panel */}
        <div className="space-y-4">
          {/* Legend */}
          <div className="text-sm text-muted-foreground">
            <div className="flex items-center gap-3 mb-1">
              <span className="inline-flex items-center gap-1">
                <Check className="h-4 w-4 text-chart-3" />
                {legend.includedText}
              </span>
              <span>•</span>
              <span className="inline-flex items-center gap-1">
                <X className="h-4 w-4 text-muted-foreground/40" />
                {legend.notAvailableText}
              </span>
            </div>
            {lastUpdated && <div className="text-xs">Last updated: {lastUpdated}</div>}
          </div>

          {/* Decisions Panel */}
          <Card className="p-4 text-sm text-muted-foreground leading-relaxed space-y-2">
            <div className="font-semibold text-foreground mb-2">Key Differences</div>
            <ul className="space-y-1.5 list-disc list-inside">
              {keyDifferences.map((diff, index) => (
                <li key={index}>{diff}</li>
              ))}
            </ul>
          </Card>
        </div>
      </div>

      {/* Comparison Table */}
      <Card className="rounded-xl border border-border bg-card shadow-sm overflow-hidden p-0">
        <div className="overflow-x-auto">
          <MatrixTable columns={columns} rows={rows} groups={groups} caption={tableCaption} />
        </div>
      </Card>

      {/* CTA Strip */}
      <div className="mt-6 flex flex-wrap items-center gap-3 justify-between rounded-xl border border-border p-4 bg-secondary animate-in fade-in slide-in-from-bottom-2 duration-500">
        <div className="text-sm font-medium">{cta.text}</div>
        <div className="flex gap-3">
          {cta.buttons.map((button, index) => (
            <Button key={index} size="default" variant={button.variant || 'default'} asChild>
              <a href={button.href}>{button.text}</a>
            </Button>
          ))}
        </div>
      </div>
    </>
  )
}
