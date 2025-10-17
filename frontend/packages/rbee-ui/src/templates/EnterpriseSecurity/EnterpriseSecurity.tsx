import { SecurityGuarantees } from '@rbee/ui/molecules'
import { SecurityCard } from '@rbee/ui/organisms'
import Image from 'next/image'
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

export type SecurityGuarantee = {
  value: string
  label: string
  ariaLabel?: string
}

export type EnterpriseSecurityProps = {
  backgroundImage: {
    src: string
    alt: string
  }
  securityCards: SecurityCardData[]
  guarantees: {
    heading: string
    stats: SecurityGuarantee[]
    footnote: string
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseSecurity({
  backgroundImage,
  securityCards,
  guarantees,
}: EnterpriseSecurityProps) {
  return (
    <div className="relative">
      {/* Decorative background illustration */}
      <Image
        src={backgroundImage.src}
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[52rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt={backgroundImage.alt}
        aria-hidden="true"
      />

      <div className="relative z-10">
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

        {/* Security Guarantees */}
        <SecurityGuarantees
          heading={guarantees.heading}
          stats={guarantees.stats}
          footnote={guarantees.footnote}
          className="animate-in fade-in-50 [animation-delay:200ms]"
        />
      </div>
    </div>
  )
}
