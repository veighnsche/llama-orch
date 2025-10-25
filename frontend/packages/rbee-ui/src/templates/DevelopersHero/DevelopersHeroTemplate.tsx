'use client'

import { GitHubIcon } from '@rbee/ui/atoms'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Card } from '@rbee/ui/atoms/Card'
import { TerminalWindow } from '@rbee/ui/molecules'
import { HeroTemplate } from '@rbee/ui/templates/HeroTemplate'
import Image from 'next/image'
import type * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface DevelopersHeroProps {
  /** Announcement badge configuration */
  badge: {
    text: string
    showPulse?: boolean
  }
  /** Main headline (first line) */
  headlineFirstLine: string
  /** Main headline (second line, gradient) */
  headlineSecondLine: string
  /** Subheadline / benefit description */
  subheadline: React.ReactNode
  /** Primary CTA button */
  primaryCta: {
    label: string
    href: string
  }
  /** Secondary CTA button */
  secondaryCta: {
    label: string
    href: string
  }
  /** Mobile-only tertiary link */
  tertiaryLink?: {
    label: string
    href: string
  }
  /** Trust badges/chips */
  trustBadges: string[]
  /** Terminal window content */
  terminal: {
    title: string
    command: string
    output: React.ReactNode
    stats: {
      gpu1: string
      gpu2: string
      cost: string
    }
  }
  /** Hardware montage image */
  hardwareImage: {
    src: string
    alt: string
  }
  /** Stat chips overlay on image */
  imageOverlayBadges: string[]
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * DevelopersHero - Hero section for developers page
 *
 * @example
 * ```tsx
 * <DevelopersHeroTemplate
 *   badge={{ text: "For developers who build with AI", showPulse: true }}
 *   headlineFirstLine="Build with AI."
 *   headlineSecondLine="Own your infrastructure."
 *   subheadline="Stop depending on external AI..."
 *   primaryCta={{ label: "Get started free", href: "#get-started" }}
 *   secondaryCta={{ label: "View on GitHub", href: "https://github.com/..." }}
 *   trustBadges={["Open source (GPL-3.0)", "OpenAI-compatible API"]}
 *   terminal={{ ... }}
 *   hardwareImage={{ src: image, alt: "..." }}
 *   imageOverlayBadges={["Zed & Cursor: drop-in via OpenAI API"]}
 * />
 * ```
 */
export function DevelopersHeroTemplate({
  badge,
  headlineFirstLine,
  headlineSecondLine,
  subheadline,
  primaryCta,
  secondaryCta,
  tertiaryLink,
  trustBadges,
  terminal,
  hardwareImage,
  imageOverlayBadges,
}: DevelopersHeroProps) {
  const asideContent = (
    <div className="space-y-6">
      {/* Terminal Window */}
      <div className="animate-in fade-in slide-in-from-right-2 duration-500 delay-300">
        <TerminalWindow title={terminal.title}>
          <div className="space-y-2">
            <div className="text-muted-foreground">
              <span className="text-chart-3">$</span> {terminal.command}
            </div>
            <div className="text-muted-foreground">
              <span className="animate-pulse">▊</span> Streaming tokens...
            </div>
            {terminal.output}
            <div className="flex items-center gap-4 text-muted-foreground pt-2">
              <div>GPU 1: {terminal.stats.gpu1}</div>
              <div>GPU 2: {terminal.stats.gpu2}</div>
              <div>Cost: {terminal.stats.cost}</div>
            </div>
          </div>
        </TerminalWindow>
      </div>

      {/* Hardware Montage */}
      <div className="animate-in fade-in duration-500 delay-400 relative">
        <Card className="relative overflow-hidden ring-1 ring-border/50 p-0">
          <Image
            src={hardwareImage.src}
            alt={hardwareImage.alt}
            width={840}
            height={525}
            className="rounded-md object-cover w-full aspect-video"
            priority={false}
            sizes="(min-width: 1024px) 420px, 100vw"
          />
          {/* Stat Chips Overlay */}
          <div className="absolute top-4 right-4 flex flex-col gap-2">
            {imageOverlayBadges.map((text) => (
              <Badge key={text} className="bg-background/90 backdrop-blur-sm border-border text-foreground shadow-lg">
                {text}
              </Badge>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )

  const trustBadgeItems = trustBadges.map((label) => ({
    text: label,
  }))

  const _secondaryCtaWithIcon = {
    ...secondaryCta,
    // GitHub icon will be handled in the button rendering
  }

  return (
    <HeroTemplate
      badge={{ variant: 'pulse', text: badge.text }}
      headline={{
        variant: 'custom',
        content: (
          <>
            <span className="block">{headlineFirstLine}</span>
            <span className="block bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
              {headlineSecondLine}
            </span>
          </>
        ),
      }}
      subcopy={subheadline}
      subcopyMaxWidth="medium"
      proofElements={{
        variant: 'badges',
        items: trustBadgeItems,
      }}
      ctas={{
        primary: {
          label: primaryCta.label,
          href: primaryCta.href,
          showIcon: true,
        },
        secondary: {
          label: (
            <>
              <GitHubIcon className="h-4 w-4 mr-2" aria-hidden="true" />
              {secondaryCta.label}
            </>
          ) as any,
          href: secondaryCta.href,
          variant: 'outline',
        },
        tertiary: tertiaryLink
          ? {
              label: tertiaryLink.label,
              href: tertiaryLink.href,
              mobileOnly: true,
            }
          : undefined,
      }}
      aside={asideContent}
      asideAriaLabel={hardwareImage.alt}
      background={{
        variant: 'custom',
        className:
          'absolute inset-0 bg-[radial-gradient(70%_60%_at_50%_-10%,hsl(var(--primary)/0.15),transparent_60%)] pointer-events-none',
      }}
      padding="spacious"
      layout={{
        leftCols: 7,
        rightCols: 5,
      }}
    />
  )
}
