'use client'

import { Button } from '@rbee/ui/atoms/Button'
import Image, { type StaticImageData } from 'next/image'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface UseCasesHeroTemplateProps {
  /** Badge text displayed above the heading */
  badgeText: string
  /** Main heading text */
  heading: string
  /** Highlighted portion of the heading (gradient effect) */
  headingHighlight: string
  /** Subheading/description text */
  description: string
  /** Primary CTA button configuration */
  primaryCta: {
    text: string
    href: string
  }
  /** Secondary CTA button configuration */
  secondaryCta: {
    text: string
    href: string
  }
  /** Proof indicators displayed below CTAs */
  proofIndicators: Array<{
    text: string
    hasDot?: boolean
  }>
  /** Hero image */
  image: StaticImageData
  /** Alt text for the hero image */
  imageAlt: string
  /** Caption text below the image */
  imageCaption: string
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * Hero section for the Use Cases page
 * 
 * @example
 * ```tsx
 * import { useCasesHero } from '@rbee/ui/assets'
 * 
 * <UseCasesHeroTemplate
 *   badgeText="OpenAI-compatible"
 *   heading="Built for Those Who Value"
 *   headingHighlight="Independence"
 *   description="Own your AI infrastructure..."
 *   primaryCta={{ text: "Explore use cases", href: "#use-cases" }}
 *   secondaryCta={{ text: "See architecture", href: "#architecture" }}
 *   proofIndicators={[
 *     { text: "Self-hosted", hasDot: true },
 *     { text: "OpenAI-compatible" },
 *     { text: "CUDA · Metal · CPU" }
 *   ]}
 *   image={useCasesHero}
 *   imageAlt="Homelab setup with AI inference"
 *   imageCaption="Your models, your hardware — no lock-in."
 * />
 * ```
 */
export function UseCasesHeroTemplate({
  badgeText,
  heading,
  headingHighlight,
  description,
  primaryCta,
  secondaryCta,
  proofIndicators,
  image,
  imageAlt,
  imageCaption,
}: UseCasesHeroTemplateProps) {
  return (
    <section className="relative overflow-hidden py-20 lg:py-24 bg-gradient-to-b from-background to-card">
      {/* Soft radial glow */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <div className="absolute top-0 right-1/4 h-[40rem] w-[40rem] rounded-full bg-primary/10 blur-3xl" />
      </div>

      <div className="container mx-auto px-4">
        <div className="grid gap-12 lg:grid-cols-[6fr_5fr] lg:items-center">
          {/* Left: copy stack */}
          <div className="animate-in fade-in-50 slide-in-from-left-4">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border bg-card/50 px-4 py-1.5 text-sm text-muted-foreground">
              <span className="font-sans font-medium text-foreground">{badgeText}</span>
            </div>

            <h1 className="text-balance text-5xl lg:text-6xl xl:text-7xl font-bold text-foreground tracking-tight leading-[1.1]">
              {heading}{' '}
              <span className="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                {headingHighlight}
              </span>
            </h1>

            <p className="mt-6 text-lg lg:text-xl text-muted-foreground leading-relaxed max-w-xl">
              {description}
            </p>

            {/* Two clear CTAs */}
            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <Button className="h-12 px-8 text-base" asChild>
                <a href={primaryCta.href}>{primaryCta.text}</a>
              </Button>
              <Button variant="outline" className="h-12 px-8 text-base" asChild>
                <a href={secondaryCta.href}>{secondaryCta.text}</a>
              </Button>
            </div>

            {/* Proof indicators */}
            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted-foreground">
              {proofIndicators.map((indicator, index) => (
                <span key={index} className={indicator.hasDot ? "inline-flex items-center gap-2" : undefined}>
                  {indicator.hasDot && (
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" aria-hidden />
                  )}
                  {indicator.text}
                </span>
              ))}
            </div>
          </div>

          {/* Right: visual */}
          <div className="relative max-lg:order-first animate-in fade-in-50 slide-in-from-right-4">
            <div className="rounded-2xl border bg-card/50 p-4 backdrop-blur-sm">
              <Image
                src={image}
                width={1080}
                height={760}
                priority
                className="rounded-xl"
                alt={imageAlt}
              />
            </div>

            <p className="mt-4 text-center text-sm text-muted-foreground">{imageCaption}</p>
          </div>
        </div>
      </div>
    </section>
  )
}
