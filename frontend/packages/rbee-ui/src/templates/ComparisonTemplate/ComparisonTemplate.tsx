import { Button } from '@rbee/ui/atoms/Button'
import type { Provider, Row } from '@rbee/ui/molecules/Tables/MatrixTable'
import { MatrixTable } from '@rbee/ui/molecules/Tables/MatrixTable'
import { Check, X } from 'lucide-react'
import Link from 'next/link'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type ComparisonLegendItem = {
  /** Rendered icon component */
  icon: React.ReactNode
  /** Label text */
  label: string
}

export type ComparisonCTA = {
  label: string
  href: string
  variant?: 'default' | 'ghost' | 'outline'
}

/**
 * ComparisonTemplate displays a feature comparison matrix table.
 * 
 * @example
 * ```tsx
 * <ComparisonTemplate
 *   columns={[
 *     { key: 'rbee', label: 'rbee', accent: true },
 *     { key: 'openai', label: 'OpenAI' },
 *   ]}
 *   rows={[
 *     { feature: 'Cost', values: { rbee: '$0', openai: '$20/mo' } },
 *   ]}
 *   legend={[
 *     { icon: <Check className="h-3.5 w-3.5" />, label: 'Available' },
 *   ]}
 * />
 * ```
 */
export type ComparisonTemplateProps = {
  /** Table column definitions */
  columns: Provider[]
  /** Table row data */
  rows: Row[]
  /** Legend items explaining symbols */
  legend?: ComparisonLegendItem[]
  /** Additional legend text */
  legendNote?: string
  /** Footer message */
  footerMessage?: string
  /** Call-to-action buttons */
  ctas?: ComparisonCTA[]
  /** Custom class name for the root element */
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function ComparisonTemplate({
  columns,
  rows,
  legend,
  legendNote,
  footerMessage,
  ctas,
  className,
}: ComparisonTemplateProps) {
  return (
    <div className={className}>
      <div className="max-w-6xl mx-auto space-y-6 animate-in fade-in-50 duration-500">
        {/* Legend */}
        {(legend || legendNote) && (
          <div className="text-xs text-muted-foreground flex flex-wrap gap-4 justify-center">
            {legend?.map((item, i) => (
              <span key={i} className="flex items-center gap-1.5">
                {item.icon}
                {item.label}
              </span>
            ))}
            {legendNote && <span>{legendNote}</span>}
          </div>
        )}

        {/* Comparison table */}
        <div className="rounded-xl ring-1 ring-border/60 bg-card overflow-hidden">
          <MatrixTable columns={columns} rows={rows} />
        </div>

        {/* Footer CTA */}
        {(footerMessage || ctas) && (
          <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center items-center">
            {footerMessage && (
              <p className="text-sm text-muted-foreground text-center sm:text-left font-sans">
                {footerMessage}
              </p>
            )}
            {ctas && ctas.length > 0 && (
              <div className="flex gap-3">
                {ctas.map((cta, i) => (
                  <Button key={i} asChild variant={cta.variant || 'default'} size="default">
                    <Link href={cta.href}>{cta.label}</Link>
                  </Button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
