'use client'

import { MatrixCard, MatrixTable } from '@rbee/ui/molecules'
import { Legend } from '@rbee/ui/atoms/Legend'
import { cn } from '@rbee/ui/utils'
import { useState } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type Provider = {
  key: string
  label: string
  accent?: boolean
}

export type Feature = {
  name: string
  values: Record<string, boolean | string>
}

export type EnterpriseComparisonTemplateProps = {
  eyebrow: string
  heading: string
  description: string
  disclaimer: string
  providers: Provider[]
  features: Feature[]
  footnote: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseComparisonTemplate({
  eyebrow,
  heading,
  description,
  disclaimer,
  providers,
  features,
  footnote,
}: EnterpriseComparisonTemplateProps) {
  const [selectedProvider, setSelectedProvider] = useState(0)

  return (
    <section aria-labelledby="comparison-h2" className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        {/* Header Block */}
        <div className="mb-12 text-center animate-in fade-in-50 slide-in-from-bottom-2">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-primary">{eyebrow}</p>
          <h2 id="comparison-h2" className="mb-4 text-4xl font-bold text-foreground">
            {heading}
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            {description}
          </p>
          <p className="mt-3 text-xs text-muted-foreground/70">
            {disclaimer}
          </p>
        </div>

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
        <div className="mt-8 text-center text-sm text-muted-foreground">
          {footnote}
        </div>
      </div>
    </section>
  )
}
