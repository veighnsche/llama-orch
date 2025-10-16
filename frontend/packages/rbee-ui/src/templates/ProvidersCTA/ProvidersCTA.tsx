import { Button } from '@rbee/ui/atoms/Button'
import { StatsGrid } from '@rbee/ui/molecules'
import Image from 'next/image'
import type * as React from 'react'
import type { ReactNode } from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersCTAStat = {
  icon: React.ReactNode
  value: string
  label: string
}

export type ProvidersCTAProps = {
  badgeIcon: React.ReactNode
  badgeText: string
  title: string
  subtitle: string
  primaryCTA: {
    label: string
    ariaLabel: string
  }
  secondaryCTA: {
    label: string
    ariaLabel: string
  }
  disclaimerText: string
  stats: ProvidersCTAStat[]
  backgroundImage: {
    src: string
    alt: string
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersCTA - Final CTA section for GPU providers page
 *
 * @example
 * ```tsx
 * <ProvidersCTA
 *   badgeIcon={<Zap className="h-4 w-4" />}
 *   badgeText="Start earning today"
 *   title="Turn Idle GPUs Into Weekly Payouts"
 *   primaryCTA={{ label: "Start Earning Now", ariaLabel: "..." }}
 *   stats={[...]}
 *   backgroundImage={{ src: gpuEarnings, alt: "..." }}
 * />
 * ```
 */
export function ProvidersCTA({
  badgeIcon,
  badgeText,
  title,
  subtitle,
  primaryCTA,
  secondaryCTA,
  disclaimerText,
  stats,
  backgroundImage,
}: ProvidersCTAProps) {
  return (
    <section
      aria-labelledby="providers-cta-h2"
      className="relative overflow-hidden bg-gradient-to-b from-background via-amber-950/10 to-background px-6 py-24"
    >
      {/* Decorative Background Image - Repositioned to right edge */}
      <Image
        src={backgroundImage.src}
        width={960}
        height={540}
        className="pointer-events-none absolute -right-32 top-1/2 hidden -translate-y-1/2 opacity-[0.08] lg:block"
        alt={backgroundImage.alt}
        priority={false}
        style={{
          maskImage: 'radial-gradient(ellipse 70% 60% at 65% 50%, black 0%, black 30%, transparent 100%)',
          WebkitMaskImage: 'radial-gradient(ellipse 70% 60% at 65% 50%, black 0%, black 30%, transparent 100%)',
        }}
      />

      <div className="relative z-10 mx-auto max-w-4xl text-center">
        {/* Header Block */}
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 motion-reduce:animate-none">
          <div
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary"
            role="note"
            title="rbee (pronounced are-bee)"
          >
            {badgeIcon}
            {badgeText}
          </div>

          <h2 id="providers-cta-h2" className="mb-6 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            {title}
          </h2>

          <p className="mx-auto mb-8 max-w-2xl text-balance text-lg text-muted-foreground">{subtitle}</p>
        </div>

        {/* CTA Actions */}
        <div
          className="animate-in fade-in-50 [animation-delay:120ms] mb-4 flex flex-col items-center justify-center gap-3 motion-reduce:animate-none sm:flex-row sm:gap-4"
          aria-describedby="providers-cta-reassurance"
        >
          <Button
            size="lg"
            className="w-full bg-primary text-primary-foreground hover:bg-primary/90 sm:w-auto"
            aria-label={primaryCTA.ariaLabel}
          >
            {primaryCTA.label}
            <span className="ml-2">→</span>
          </Button>
          <Button
            size="lg"
            variant="outline"
            className="w-full border-border bg-transparent text-foreground hover:bg-secondary sm:w-auto"
            aria-label={secondaryCTA.ariaLabel}
          >
            {secondaryCTA.label}
          </Button>
        </div>

        {/* Micro-credibility */}
        <p className="mb-10 text-xs text-muted-foreground/70">{disclaimerText}</p>

        {/* Reassurance Bar */}
        <div
          id="providers-cta-reassurance"
          className="animate-in fade-in-50 [animation-delay:200ms] mt-10 text-sm text-muted-foreground motion-reduce:animate-none"
        >
          <StatsGrid variant="inline" columns={3} stats={stats} />
        </div>
      </div>
    </section>
  )
}
