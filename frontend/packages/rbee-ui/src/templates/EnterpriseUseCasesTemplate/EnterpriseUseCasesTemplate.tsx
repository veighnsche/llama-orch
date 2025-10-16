import { Button } from '@rbee/ui/atoms/Button'
import { IndustryCaseCard } from '@rbee/ui/molecules'
import type { ReactNode } from 'react'
import Image from 'next/image'
import Link from 'next/link'

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

export type CTAButton = {
  text: string
  href: string
  variant?: 'default' | 'outline'
}

export type IndustryLink = {
  text: string
  href: string
}

export type EnterpriseUseCasesTemplateProps = {
  backgroundImage: {
    src: string
    alt: string
  }
  industryCases: IndustryCase[]
  cta: {
    text: string
    buttons: CTAButton[]
    links: IndustryLink[]
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseUseCasesTemplate({ backgroundImage, industryCases, cta }: EnterpriseUseCasesTemplateProps) {
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

        {/* CTA Rail */}
        <div className="animate-in fade-in-50 rounded-2xl border border-primary/20 bg-primary/5 p-6 text-center [animation-delay:200ms]">
          <p className="mb-6 text-lg font-semibold text-foreground">{cta.text}</p>
          <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
            {cta.buttons.map((button, idx) => (
              <Button
                key={idx}
                size="lg"
                variant={button.variant}
                asChild
                className="transition-transform active:scale-[0.98]"
              >
                <Link href={button.href}>{button.text}</Link>
              </Button>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap justify-center gap-3 text-sm text-muted-foreground">
            {cta.links.map((link, idx) => (
              <>
                {idx > 0 && <span key={`sep-${idx}`}>•</span>}
                <Link key={idx} href={link.href} className="hover:text-primary hover:underline">
                  {link.text}
                </Link>
              </>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
