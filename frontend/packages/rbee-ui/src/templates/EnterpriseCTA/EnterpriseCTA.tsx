import { Button } from '@rbee/ui/atoms/Button'
import { CTAOptionCard } from '@rbee/ui/organisms'
import Link from 'next/link'
import type * as React from 'react'
import type { ReactNode } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type CTAOption = {
  icon: React.ReactNode
  title: string
  body: string
  tone?: 'primary' | 'outline'
  eyebrow?: string
  note?: string
  buttonText: string
  buttonHref: string
  buttonVariant?: 'default' | 'outline'
  buttonAriaLabel?: string
}

export type TrustStat = {
  value: string
  label: string
}

export type EnterpriseCTAProps = {
  eyebrow: string
  heading: string
  description: string
  trustStats: TrustStat[]
  ctaOptions: CTAOption[]
  footerCaption: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseCTA({
  eyebrow,
  heading,
  description,
  trustStats,
  ctaOptions,
  footerCaption,
}: EnterpriseCTAProps) {
  return (
    <section
      aria-labelledby="cta-h2"
      className="relative border-b border-border bg-gradient-to-b from-background via-primary/5 to-background px-6 py-24 overflow-hidden"
    >
      {/* Decorative Gradient */}
      <div className="pointer-events-none absolute inset-0 bg-radial-glow" aria-hidden="true" />

      <div className="relative mx-auto max-w-5xl">
        {/* Header Block */}
        <div className="mb-12 text-center animate-in fade-in-50 slide-in-from-bottom-2 duration-500">
          <p className="mb-2 text-sm font-sans font-semibold uppercase tracking-wide text-primary">{eyebrow}</p>
          <h2 id="cta-h2" className="mb-4 text-4xl font-bold text-foreground lg:text-5xl">
            {heading}
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">{description}</p>
        </div>

        {/* Trust Strip */}
        <div className="mb-12 grid gap-6 sm:grid-cols-4 text-center">
          {trustStats.map((stat, idx) => (
            <div key={idx} className="text-sm">
              <div className="font-semibold text-foreground">{stat.value}</div>
              <div className="text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* CTA Options Grid */}
        <div className="mb-12 grid gap-6 md:grid-cols-3 animate-in fade-in-50" style={{ animationDelay: '120ms' }}>
          {ctaOptions.map((option, idx) => (
            <CTAOptionCard
              key={idx}
              icon={option.icon}
              title={option.title}
              body={option.body}
              tone={option.tone}
              eyebrow={option.eyebrow}
              note={option.note}
              action={
                <Button
                  asChild
                  size="lg"
                  variant={option.buttonVariant}
                  className="w-full hover:translate-y-0.5 active:translate-y-[1px] transition-transform"
                  aria-label={option.buttonAriaLabel}
                >
                  <Link href={option.buttonHref}>{option.buttonText}</Link>
                </Button>
              }
            />
          ))}
        </div>

        {/* Footer Caption */}
        <p className="text-center text-sm text-muted-foreground">{footerCaption}</p>
      </div>
    </section>
  )
}
