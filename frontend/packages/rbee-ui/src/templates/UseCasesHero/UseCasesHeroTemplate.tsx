'use client'

import { HeroTemplate } from '@rbee/ui/templates/HeroTemplate'
import Image from 'next/image'

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
  image: string
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
  const asideContent = (
    <div className="relative max-lg:order-first animate-in fade-in-50 slide-in-from-right-4">
      <div className="rounded-2xl border bg-card/50 p-4 backdrop-blur-sm">
        <Image src={image} width={1080} height={760} priority className="rounded-xl" alt={imageAlt} />
      </div>
      <p className="mt-4 text-center text-sm text-muted-foreground">{imageCaption}</p>
    </div>
  )

  return (
    <HeroTemplate
      badge={{ variant: 'simple', text: badgeText }}
      headline={{ variant: 'inline-highlight', content: heading, highlight: headingHighlight }}
      subcopy={description}
      subcopyMaxWidth="wide"
      proofElements={{
        variant: 'indicators',
        items: proofIndicators,
      }}
      ctas={{
        primary: {
          label: primaryCta.text,
          href: primaryCta.href,
        },
        secondary: {
          label: secondaryCta.text,
          href: secondaryCta.href,
          variant: 'outline',
        },
      }}
      aside={asideContent}
      asideAriaLabel={imageAlt}
      background={{
        variant: 'custom',
        className: 'pointer-events-none absolute inset-0',
      }}
      padding="compact"
      layout={{
        leftCols: 7,
        rightCols: 5,
      }}
    />
  )
}
