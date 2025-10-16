import { ProvidersCaseCard, type ProvidersCaseCardProps } from '@rbee/ui/molecules'
import type * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersUseCaseItem = {
  icon: React.ReactNode
  title: string
  subtitle?: string
  quote: string
  facts: { label: string; value: string }[]
  highlight?: string
}

export type ProvidersUseCasesTemplateProps = {
  cases: ProvidersUseCaseItem[]
  className?: string
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersUseCasesTemplate - Use cases section for GPU providers
 *
 * @example
 * ```tsx
 * <ProvidersUseCasesTemplate cases={[...]} />
 * ```
 */
export function ProvidersUseCasesTemplate({ cases, className }: ProvidersUseCasesTemplateProps) {
  return (
    <div className={className}>
      {/* Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {cases.map((caseData, index) => (
          <ProvidersCaseCard
            key={index}
            icon={caseData.icon}
            title={caseData.title}
            subtitle={caseData.subtitle}
            quote={caseData.quote}
            facts={caseData.facts}
            highlight={caseData.highlight}
            index={index}
          />
        ))}
      </div>
    </div>
  )
}
