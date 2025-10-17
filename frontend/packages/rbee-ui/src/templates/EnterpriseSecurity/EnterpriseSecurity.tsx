import { SecurityCard } from '@rbee/ui/organisms'
import type { ReactNode } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type SecurityCardData = {
  icon: ReactNode
  title: string
  subtitle: string
  intro: string
  bullets: string[]
  docsHref: string
}

export type EnterpriseSecurityProps = {
  securityCards: SecurityCardData[]
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseSecurity({ securityCards }: EnterpriseSecurityProps) {
  return (
    <>
      {/* Security Cards Grid */}
      <div className="animate-in fade-in-50 mb-12 mx-auto grid max-w-4xl gap-8 [animation-delay:120ms] md:grid-cols-2">
          {securityCards.map((card, idx) => (
            <SecurityCard
              key={idx}
              icon={card.icon}
              title={card.title}
              subtitle={card.subtitle}
              intro={card.intro}
              bullets={card.bullets}
              docsHref={card.docsHref}
            />
          ))}
      </div>
    </>
  )
}
