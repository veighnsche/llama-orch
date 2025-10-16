import { Button } from '@rbee/ui/atoms/Button'
import { cn } from '@rbee/ui/utils'
import * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersUseCaseItem = {
  icon: React.ReactNode
  title: string
  subtitle?: string
  quote: string
  facts: { label: string; value: string }[]
  image?: { Component: React.ComponentType<any>; alt: string }
  highlight?: string
}

export type ProvidersUseCasesTemplateProps = {
  cases: ProvidersUseCaseItem[]
  ctas?: {
    primary?: { label: string; href: string }
    secondary?: { label: string; href: string }
  }
  className?: string
}

// ────────────────────────────────────────────────────────────────────────────
// CaseCard Component
// ────────────────────────────────────────────────────────────────────────────

function CaseCard({ caseData, index }: { caseData: ProvidersUseCaseItem; index: number }) {
  const delays = ['delay-75', 'delay-150', 'delay-200', 'delay-300']
  const delay = delays[index % delays.length]

  return (
    <div
      className={cn(
        'group min-h-[320px] rounded-2xl border/70 bg-gradient-to-b from-card/70 to-background/60 p-6 backdrop-blur transition-transform hover:translate-y-0.5 supports-[backdrop-filter]:bg-background/60 sm:p-7',
        'animate-in fade-in slide-in-from-bottom-2',
        delay,
      )}
    >
      {/* Header row */}
      <div className="mb-4 flex items-center gap-4">
        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-primary/10 transition-transform group-hover:scale-[1.02]">
          {React.cloneElement(caseData.icon as React.ReactElement<any>, {
            className: 'h-7 w-7 text-primary',
            'aria-hidden': true,
          })}
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-foreground">{caseData.title}</h3>
          {caseData.subtitle && <div className="text-xs text-muted-foreground">{caseData.subtitle}</div>}
        </div>
        {caseData.image?.Component && (
          <caseData.image.Component size={48} aria-label={caseData.image.alt} className="rounded-lg" />
        )}
      </div>

      {/* Optional highlight badge */}
      {caseData.highlight && (
        <div className="mb-3 inline-flex items-center rounded-full bg-primary/10 px-2.5 py-1 text-[11px] text-primary">
          {caseData.highlight}
        </div>
      )}

      {/* Quote block */}
      <p className="relative mb-4 text-pretty leading-relaxed text-muted-foreground">
        <span className="mr-1 text-primary">&ldquo;</span>
        {caseData.quote}
      </p>

      {/* Facts list */}
      <div className="space-y-2 text-sm">
        {caseData.facts.map((fact, idx) => {
          const isEarnings = fact.label.toLowerCase().includes('monthly')
          return (
            <div key={idx} className="flex justify-between">
              <span className="text-muted-foreground">{fact.label}</span>
              <span className={cn('tabular-nums text-foreground', isEarnings && 'font-semibold text-primary')}>
                {fact.value}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersUseCasesTemplate - Use cases section for GPU providers
 *
 * @example
 * ```tsx
 * <ProvidersUseCasesTemplate
 *   cases={[...]}
 *   ctas={{ primary: {...}, secondary: {...} }}
 * />
 * ```
 */
export function ProvidersUseCasesTemplate({ cases, ctas, className }: ProvidersUseCasesTemplateProps) {
  return (
    <div className={className}>
      {/* Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {cases.map((caseData, index) => (
          <CaseCard key={index} caseData={caseData} index={index} />
        ))}
      </div>

      {/* Micro-CTA rail */}
      {ctas && (ctas.primary || ctas.secondary) && (
        <div className="mt-8 text-center">
          <p className="mb-4 text-sm font-medium text-muted-foreground">Ready to join them?</p>
          <div className="flex flex-col items-center justify-center gap-2 sm:flex-row">
            {ctas.primary && (
              <Button asChild size="lg">
                <a href={ctas.primary.href}>{ctas.primary.label}</a>
              </Button>
            )}
            {ctas.secondary && (
              <Button asChild variant="outline" size="lg">
                <a href={ctas.secondary.href}>{ctas.secondary.label}</a>
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
