import { IndustryCaseCard } from '@rbee/ui/organisms'
import type { ReactNode } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type IndustryCase = {
  icon: ReactNode
  industry: string
  segments: string
  badges: string[]
  summary: string
  challenges: string[]
  solutions: string[]
  href: string
}

export type EnterpriseUseCasesProps = {
  industryCases: IndustryCase[]
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseUseCases({ industryCases }: EnterpriseUseCasesProps) {
  return (
    <>
      {/* Industry Grid */}
      <div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] md:grid-cols-2">
          {industryCases.map((industryCase) => (
            <IndustryCaseCard
              key={industryCase.industry}
              icon={industryCase.icon}
              industry={industryCase.industry}
              segments={industryCase.segments}
              badges={industryCase.badges}
              summary={industryCase.summary}
              challenges={industryCase.challenges}
              solutions={industryCase.solutions}
              href={industryCase.href}
            />
          ))}
      </div>
    </>
  )
}
