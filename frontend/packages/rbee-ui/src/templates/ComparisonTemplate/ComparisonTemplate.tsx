'use client'

import { Legend } from '@rbee/ui/molecules/Legend'
import { MatrixCard } from '@rbee/ui/molecules/Tables/MatrixCard'
import type { Provider, Row } from '@rbee/ui/molecules/Tables/MatrixTable'
import { MatrixTable } from '@rbee/ui/molecules/Tables/MatrixTable'
import { SegmentedControl } from '@rbee/ui/molecules/SegmentedControl'
import { cn } from '@rbee/ui/utils'
import { useState } from 'react'

// Re-export LegendItem from Legend molecule for convenience
export type { LegendItem as ComparisonLegendItem } from '@rbee/ui/molecules/Legend'

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
  legend?: import('@rbee/ui/molecules/Legend').LegendItem[]
  /** Additional legend text */
  legendNote?: string
  /** Enable mobile card switcher view (default: false, desktop-only table) */
  showMobileCards?: boolean
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
  showMobileCards = false,
  className,
}: ComparisonTemplateProps) {
  const [selectedProviderKey, setSelectedProviderKey] = useState(columns[0]?.key || '')

  return (
    <div className={className}>
      <div className="max-w-6xl mx-auto space-y-6 animate-in fade-in-50 duration-500">
        {/* Legend */}
        <Legend items={legend} note={legendNote} />

        {/* Desktop Table */}
        <div
          className={cn(
            'rounded-xl ring-1 ring-border/60 bg-card overflow-hidden',
            showMobileCards ? 'hidden md:block' : 'block',
          )}
        >
          <MatrixTable columns={columns} rows={rows} />
        </div>

        {/* Mobile Cards */}
        {showMobileCards && (
          <div className="md:hidden">
            {/* Provider Switcher */}
            <SegmentedControl
              options={columns.map((col) => ({ key: col.key, label: col.label }))}
              value={selectedProviderKey}
              onChange={setSelectedProviderKey}
              className="mb-6"
            />

            {/* Single Card for Selected Provider */}
            <MatrixCard
              provider={columns.find((col) => col.key === selectedProviderKey) || columns[0]}
              rows={rows}
            />

            {/* Jump to Desktop Link (for screen readers) */}
            <a
              href="#comparison-table"
              className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
            >
              Jump to desktop table
            </a>
          </div>
        )}

      </div>
    </div>
  )
}
