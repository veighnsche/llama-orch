'use client'

import { Badge, Button } from '@rbee/ui/atoms'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface PricingHeroAssuranceItem {
  text: string
  icon: LucideIcon
}

export interface PricingHeroTemplateProps {
  /** Badge text above the heading */
  badgeText: string
  /** Main heading (can include line breaks) */
  heading: ReactNode
  /** Description text */
  description: string
  /** Primary CTA button configuration */
  primaryCta: {
    text: string
    href?: string
  }
  /** Secondary CTA button configuration */
  secondaryCta: {
    text: string
    href?: string
  }
  /** Assurance items with icons */
  assuranceItems: PricingHeroAssuranceItem[]
  /** Visual illustration component */
  visual: ReactNode
  /** Aria label for the visual */
  visualAriaLabel: string
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * Hero section for the Pricing page
 *
 * @example
 * ```tsx
 * import { PricingScaleVisual } from '@rbee/ui/icons'
 * import { Sparkles } from 'lucide-react'
 *
 * <PricingHeroTemplate
 *   badgeText="Honest Pricing"
 *   heading={<>Start Free.<br /><span className="text-primary">Scale When Ready.</span></>}
 *   description="Every tier ships the full rbee orchestrator..."
 *   primaryCta={{ text: "View Plans" }}
 *   secondaryCta={{ text: "Talk to Sales" }}
 *   assuranceItems={[
 *     { text: 'Full orchestrator on every tier', icon: Sparkles },
 *     { text: 'No feature gates or limits', icon: Sparkles }
 *   ]}
 *   visual={<PricingScaleVisual size="100%" className="rounded-xl opacity-70" />}
 *   visualAriaLabel="Illustration showing rbee pricing scales"
 * />
 * ```
 */
export function PricingHeroTemplate({
  badgeText,
  heading,
  description,
  primaryCta,
  secondaryCta,
  assuranceItems,
  visual,
  visualAriaLabel,
}: PricingHeroTemplateProps) {
  return (
    <section
      aria-labelledby="pricing-hero-title"
      className="relative overflow-hidden py-24 lg:py-28 bg-gradient-to-b from-background to-card"
    >
      {/* Simple radial glow */}
      <div aria-hidden className="pointer-events-none absolute inset-0 opacity-50">
        <div className="absolute -top-1/3 right-[-20%] h-[60rem] w-[60rem] rounded-full bg-primary/5 blur-3xl" />
      </div>

      <div className="container mx-auto px-4">
        <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
          {/* Left: Text content */}
          <div className="max-w-2xl">
            <Badge variant="secondary" className="mb-4">
              {badgeText}
            </Badge>

            <h1 id="pricing-hero-title" className="text-5xl lg:text-6xl font-bold text-foreground tracking-tight">
              {heading}
            </h1>

            <p className="mt-6 text-xl text-muted-foreground leading-relaxed">{description}</p>

            {/* Buttons */}
            <div className="mt-8 flex gap-3">
              <Button size="lg" asChild={!!primaryCta.href}>
                {primaryCta.href ? <a href={primaryCta.href}>{primaryCta.text}</a> : primaryCta.text}
              </Button>
              <Button variant="secondary" size="lg" asChild={!!secondaryCta.href}>
                {secondaryCta.href ? <a href={secondaryCta.href}>{secondaryCta.text}</a> : secondaryCta.text}
              </Button>
            </div>

            {/* Assurance checkmarks */}
            <ul className="mt-6 grid grid-cols-2 gap-3 text-sm text-muted-foreground">
              {assuranceItems.map((item, index) => {
                const Icon = item.icon
                return (
                  <li key={index} className="flex items-center gap-2">
                    <Icon className="h-4 w-4 text-primary shrink-0" aria-hidden="true" />
                    <span>{item.text}</span>
                  </li>
                )
              })}
            </ul>
          </div>

          {/* Right: Visual illustration */}
          <div className="relative" aria-label={visualAriaLabel}>
            {visual}
          </div>
        </div>
      </div>
    </section>
  )
}
