'use client'

import { HeroTemplate } from '@rbee/ui/templates/HeroTemplate'
import type { ReactNode } from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface PricingHeroAssuranceItem {
  text: string
  icon: ReactNode
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
 *   visual={<PricingScaleVisual size="100%" className="rounded-md opacity-70" />}
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
    <HeroTemplate
      badge={{ variant: 'simple', text: badgeText }}
      headline={{ variant: 'custom', content: heading }}
      subcopy={description}
      subcopyMaxWidth="medium"
      proofElements={{
        variant: 'assurance',
        items: assuranceItems,
        columns: 2,
      }}
      ctas={{
        primary: {
          label: primaryCta.text,
          href: primaryCta.href,
        },
        secondary: {
          label: secondaryCta.text,
          href: secondaryCta.href,
          variant: 'secondary',
        },
      }}
      aside={visual}
      asideAriaLabel={visualAriaLabel}
      background={{
        variant: 'custom',
        className: 'pointer-events-none absolute inset-0 opacity-50',
      }}
      padding="spacious"
      headingId="pricing-hero-title"
    />
  )
}
