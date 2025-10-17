'use client'

import { Legend } from '@rbee/ui/atoms/Legend'
import { MatrixCard, MatrixTable, type Provider, type Row } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import { useState } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type EnterpriseComparisonProps = {
  providers: Provider[]
  features: Row[]
  footnote: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseComparison({ providers, features, footnote }: EnterpriseComparisonProps) {
  const [selectedProvider, setSelectedProvider] = useState(0)

  return (
    <div>
      {/* Legend */}
      <Legend className="mb-8 animate-in fade-in-50" style={{ animationDelay: '100ms' }} />

      {/* Desktop Table (md+) */}
      <div className="hidden md:block animate-in fade-in-50" style={{ animationDelay: '150ms' }}>
        <MatrixTable columns={providers} rows={features} />
      </div>

      {/* Mobile Cards (<md) */}
      <div className="md:hidden animate-in fade-in-50" style={{ animationDelay: '150ms' }}>
        {/* Provider Switcher */}
        <div className="mb-6 flex items-center justify-center gap-2 rounded-lg border bg-card/60 p-1">
          {providers.map((provider, index) => (
            <button
              key={provider.key}
              onClick={() => setSelectedProvider(index)}
              className={cn(
                'flex-1 rounded-md px-3 py-2 text-xs font-medium transition-colors',
                selectedProvider === index
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground',
              )}
              aria-pressed={selectedProvider === index}
            >
              {provider.label}
            </button>
          ))}
        </div>

        {/* Single Card for Selected Provider */}
        <MatrixCard provider={providers[selectedProvider]} rows={features} />

        {/* Jump to Desktop Link (for screen readers) */}
        <a
          href="#comparison-h2"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
        >
          Jump to desktop table
        </a>
      </div>

      {/* Footnote */}
      <div className="mt-8 text-center text-sm text-muted-foreground">{footnote}</div>
    </div>
  )
}
