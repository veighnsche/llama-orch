import { Button } from '@rbee/ui/atoms/Button'
import { StatsGrid } from '@rbee/ui/molecules'
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
  trustStats: TrustStat[]
  ctaOptions: CTAOption[]
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseCTA({ trustStats, ctaOptions }: EnterpriseCTAProps) {
  return (
    <>
      {/* Trust Strip */}
      <div className="mb-12">
        <StatsGrid stats={trustStats} variant="strip" columns={4} />
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
    </>
  )
}
