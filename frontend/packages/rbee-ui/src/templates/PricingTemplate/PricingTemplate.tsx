'use client'

import { PricingTier } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import Image from 'next/image'
import type * as React from 'react'
import { useState } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type PricingKickerBadge = {
  /** Rendered icon component */
  icon: React.ReactNode
  /** Badge label */
  label: string
}

export type PricingTierData = {
  title: string
  price: string
  priceYearly?: string
  period?: string
  features: string[]
  ctaText: string
  ctaHref: string
  ctaVariant?: 'default' | 'outline'
  highlighted?: boolean
  badge?: string
  footnote?: string
  saveBadge?: string
  className?: string
}

export type PricingFooter = {
  mainText: string
  subText: string
}

/**
 * PricingTemplate displays pricing tiers with monthly/yearly toggle.
 *
 * @example
 * ```tsx
 * <PricingTemplate
 *   kickerBadges={[
 *     { icon: <Unlock className="h-3.5 w-3.5" />, label: 'Open source' },
 *   ]}
 *   tiers={[
 *     { title: 'Free', price: '€0', features: [...], ctaText: 'Download', ctaHref: '/download' },
 *   ]}
 * />
 * ```
 */
export type PricingTemplateProps = {
  /** Kicker badges above title */
  kickerBadges?: PricingKickerBadge[]
  /** Pricing tier configurations */
  tiers: PricingTierData[]
  /** Editorial image to show below tiers */
  editorialImage?: {
    src: string
    alt: string
    width?: number
    height?: number
  }
  /** Footer reassurance text */
  footer?: PricingFooter
  /** Custom class name for the root element */
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function PricingTemplate({ kickerBadges, tiers, editorialImage, footer, className }: PricingTemplateProps) {
  const [isYearly, setIsYearly] = useState(false)

  return (
    <div className={className}>
      {/* Kicker badges */}
      {kickerBadges && kickerBadges.length > 0 && (
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground justify-center mb-8">
          {kickerBadges.map((badge, i) => (
            <span key={i} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-muted/50">
              {badge.icon}
              {badge.label}
            </span>
          ))}
        </div>
      )}

      {/* Billing toggle */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex items-center gap-2 text-sm bg-muted p-1 rounded">
          <button
            onClick={() => setIsYearly(false)}
            className={cn(
              'px-4 py-2 rounded-md font-medium transition-all',
              !isYearly ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
            )}
            aria-pressed={!isYearly}
          >
            Monthly
          </button>
          <button
            onClick={() => setIsYearly(true)}
            className={cn(
              'px-4 py-2 rounded-md font-medium transition-all inline-flex items-center gap-1.5',
              isYearly ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
            )}
            aria-pressed={isYearly}
          >
            Yearly
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-chart-3/20 text-chart-3 font-semibold">
              Save 2 months
            </span>
          </button>
        </div>
      </div>

      {/* Pricing grid */}
      <div className="grid grid-cols-12 gap-6 lg:gap-8 max-w-6xl mx-auto mt-6">
        {tiers.map((tier, i) => (
          <PricingTier
            key={i}
            title={tier.title}
            price={tier.price}
            priceYearly={tier.priceYearly}
            period={tier.period}
            features={tier.features}
            ctaText={tier.ctaText}
            ctaHref={tier.ctaHref}
            ctaVariant={tier.ctaVariant}
            highlighted={tier.highlighted}
            badge={tier.badge}
            footnote={tier.footnote}
            isYearly={isYearly}
            saveBadge={tier.saveBadge}
            className={tier.className}
          />
        ))}
      </div>

      {/* Editorial visual (desktop only) */}
      {editorialImage && (
        <div className="hidden lg:block mt-10">
          <Image
            src={editorialImage.src}
            width={editorialImage.width || 1100}
            height={editorialImage.height || 620}
            className="rounded ring-1 ring-border/60 shadow-sm mx-auto"
            alt={editorialImage.alt}
            priority
          />
        </div>
      )}

      {/* Footer reassurance */}
      {footer && (
        <div className="text-center mt-12 max-w-2xl mx-auto">
          <p className="text-muted-foreground font-sans">{footer.mainText}</p>
          <p className="text-[12px] text-muted-foreground/80 mt-2 font-sans">{footer.subText}</p>
        </div>
      )}
    </div>
  )
}
