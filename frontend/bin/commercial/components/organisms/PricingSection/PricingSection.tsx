'use client'

import { useState } from 'react'
import Image from 'next/image'
import { SectionContainer, PricingTier } from '@/components/molecules'
import { cn } from '@/lib/utils'
import { Shield, Zap, Layers, Unlock } from 'lucide-react'

export interface PricingSectionProps {
  /** Variant for different page contexts */
  variant?: 'home' | 'pricing'
  /** Override section title */
  title?: string
  /** Override section description */
  description?: string
  /** Show kicker badges */
  showKicker?: boolean
  /** Show editorial image below cards */
  showEditorialImage?: boolean
  /** Show footer reassurance text */
  showFooter?: boolean
  /** Custom class name */
  className?: string
}

export function PricingSection({
  variant = 'home',
  title,
  description,
  showKicker = true,
  showEditorialImage = true,
  showFooter = true,
  className,
}: PricingSectionProps) {
  const [isYearly, setIsYearly] = useState(false)

  // Variant-specific defaults
  const defaultTitle = variant === 'pricing' ? 'Simple, honest pricing.' : 'Start Free. Scale When Ready.'
  const defaultDescription =
    variant === 'pricing'
      ? "Every plan includes the full rbee orchestrator—no feature gates, no artificial limits. Start free and grow when you're ready."
      : 'Run rbee free at home. Add collaboration and governance when your team grows.'

  const finalTitle = title ?? defaultTitle
  const finalDescription = description ?? defaultDescription

  return (
    <SectionContainer
      title={finalTitle}
      description={finalDescription}
      className={className}
      kicker={
        showKicker ? (
          <div className="flex flex-wrap gap-2 text-xs text-muted-foreground justify-center">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-muted/50">
              <Unlock className="h-3.5 w-3.5" aria-hidden="true" />
              Open source
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-muted/50">
              <Zap className="h-3.5 w-3.5" aria-hidden="true" />
              OpenAI-compatible
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-muted/50">
              <Layers className="h-3.5 w-3.5" aria-hidden="true" />
              Multi-GPU
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-muted/50">
              <Shield className="h-3.5 w-3.5" aria-hidden="true" />
              No feature gates
            </span>
          </div>
        ) : undefined
      }
    >
      {/* Billing toggle */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex items-center gap-2 text-sm bg-muted p-1 rounded-lg">
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
        <PricingTier
          title="Home/Lab"
          price="€0"
          period="forever"
          features={[
            'Unlimited GPUs on your hardware',
            'OpenAI-compatible API',
            'Multi-modal models',
            'Active community support',
            'Open source core',
          ]}
          ctaText="Download rbee"
          ctaHref="/download"
          ctaVariant="outline"
          footnote="Local use. No feature gates."
          className="col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500"
        />

        <PricingTier
          title="Team"
          price="€99"
          priceYearly="€990"
          period="/month"
          features={[
            'Everything in Home/Lab',
            'Web UI for cluster & models',
            'Shared workspaces & quotas',
            'Priority support (business hours)',
            'Rhai policy templates (rate/data)',
          ]}
          ctaText="Start 30-Day Trial"
          ctaHref="/signup?plan=team"
          highlighted
          badge="Most Popular"
          footnote="Cancel anytime during trial."
          isYearly={isYearly}
          saveBadge="2 months free"
          className="col-span-12 md:col-span-4 order-first md:order-none motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-100"
        />

        <PricingTier
          title="Enterprise"
          price="Custom"
          features={[
            'Everything in Team',
            'Dedicated, isolated instances',
            'Custom SLAs & onboarding',
            'White-label & SSO options',
            'Enterprise security & support',
          ]}
          ctaText="Contact Sales"
          ctaHref="/contact?type=enterprise"
          ctaVariant="outline"
          footnote="We'll reply within 1 business day."
          className="col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-200"
        />
      </div>

      {/* Editorial visual (desktop only) */}
      {showEditorialImage && (
        <div className="hidden lg:block mt-10">
          <Image
            src="/images/pricing-hero.png"
            width={1100}
            height={620}
            className="rounded-2xl ring-1 ring-border/60 shadow-sm mx-auto"
            alt="Detailed isometric 3D illustration in dark mode showing a progression from left to right: a compact single-GPU homelab server rack (glowing neon teal accents) seamlessly transforming into a large-scale multi-node GPU cluster with interconnected nodes (amber and teal lighting). Clean editorial photography style with dramatic cinematic lighting, sharp focus on hardware details, floating UI panels showing metrics, dark navy background with subtle grid, professional tech marketing aesthetic, 4K quality, Octane render look"
            priority
          />
        </div>
      )}

      {/* Footer reassurance */}
      {showFooter && (
        <div className="text-center mt-12 max-w-2xl mx-auto">
          <p className="text-muted-foreground">
            {variant === 'pricing'
              ? 'Cancel anytime • No feature gates • Full orchestrator on every tier.'
              : 'Every plan includes the full rbee orchestrator. No feature gates. No artificial limits.'}
          </p>
          <p className="text-[12px] text-muted-foreground/80 mt-2">
            Prices exclude {variant === 'pricing' ? 'taxes' : 'VAT'}. OSS license applies to Home/Lab.
          </p>
        </div>
      )}
    </SectionContainer>
  )
}
