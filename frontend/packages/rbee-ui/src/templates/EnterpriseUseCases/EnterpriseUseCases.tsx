import { IndustryCaseCard } from '@rbee/ui/organisms'
import Image from 'next/image'
import { type ReactNode } from 'react'

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
  backgroundImage: {
    src: string
    alt: string
  }
  industryCases: IndustryCase[]
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseUseCases({ backgroundImage, industryCases }: EnterpriseUseCasesProps) {
  return (
    <div className="relative">
      {/* Decorative background illustration */}
      <Image
        src={backgroundImage.src}
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-6 -z-10 hidden w-[50rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt={backgroundImage.alt}
        aria-hidden="true"
      />

      <div className="relative z-10">
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
      </div>
    </div>
  )
}
