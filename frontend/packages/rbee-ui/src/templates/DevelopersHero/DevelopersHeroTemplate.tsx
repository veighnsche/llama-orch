'use client'

import { BrandWordmark, GitHubIcon } from '@rbee/ui/atoms'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import { TerminalWindow } from '@rbee/ui/molecules'
import { ArrowRight, Check } from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'
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
  return (
    <section className="relative isolate overflow-hidden border-b border-border bg-background before:absolute before:inset-0 before:bg-[radial-gradient(70%_60%_at_50%_-10%,hsl(var(--primary)/0.15),transparent_60%)] before:pointer-events-none">
      <div className="relative mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
        <div className="lg:grid lg:grid-cols-12 lg:gap-12">
          {/* Left Column: Content */}
          <div className="lg:col-span-7 space-y-8">
            {/* Announcement Badge */}
            <div className="animate-in fade-in duration-500 delay-100" aria-live="polite" aria-atomic="true">
              <Badge
                variant="outline"
                className="inline-flex items-center gap-2 rounded-full border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary"
              >
                {badge.showPulse && (
                  <span className="relative flex h-2 w-2" aria-hidden="true">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
                  </span>
                )}
                {badge.text}
              </Badge>
            </div>

            {/* H1: Two-line lockup */}
            <h1 className="text-balance tracking-tight text-5xl sm:text-6xl lg:text-7xl font-bold">
              <span className="block animate-in fade-in-50 slide-in-from-bottom-1 duration-500 delay-150">
                {headlineFirstLine}
              </span>
              <span className="block animate-in fade-in-50 slide-in-from-bottom-1 duration-500 delay-250 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                {headlineSecondLine}
              </span>
            </h1>

            {/* Benefit Subline */}
            <div className="animate-in fade-in-50 duration-500 delay-300 text-balance text-xl leading-relaxed text-muted-foreground max-w-2xl">
              {subheadline}
            </div>

            {/* CTA Row */}
            <div className="animate-in fade-in zoom-in-50 duration-500 delay-400 flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <Button asChild size="lg" className="group">
                <Link href={primaryCta.href}>
                  {primaryCta.label}
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" aria-hidden="true" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline">
                <Link href={secondaryCta.href} target="_blank" rel="noopener noreferrer">
                  <GitHubIcon className="h-4 w-4" aria-hidden="true" />
                  {secondaryCta.label}
                </Link>
              </Button>
            </div>

            {/* Tertiary Link (Mobile Only) */}
            {tertiaryLink && (
              <div className="sm:hidden">
                <Link
                  href={tertiaryLink.href}
                  className="inline-flex items-center gap-2 text-sm text-primary hover:underline underline-offset-4"
                >
                  {tertiaryLink.label}
                  <ArrowRight className="h-3 w-3" aria-hidden="true" />
                </Link>
              </div>
            )}

            {/* Trust Chips */}
            <div className="flex flex-wrap items-center gap-3 text-sm">
              {trustBadges.map((label, index) => (
                <Badge
                  key={label}
                  variant="outline"
                  className={`animate-in fade-in duration-500 delay-[${500 + index * 100}ms] inline-flex items-center gap-2 rounded-full border px-3 py-1 text-muted-foreground`}
                >
                  <Check className="h-3 w-3" aria-hidden="true" />
                  {label}
                </Badge>
              ))}
            </div>
          </div>

          {/* Right Column: Visual Stack */}
          <div className="lg:col-span-5 mt-16 lg:mt-0 space-y-6">
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
                  className="rounded-xl object-cover w-full aspect-video"
                  priority={false}
                  sizes="(min-width: 1024px) 420px, 100vw"
                />
                {/* Stat Chips Overlay */}
                <div className="absolute top-4 right-4 flex flex-col gap-2">
                  {imageOverlayBadges.map((text) => (
                    <Badge
                      key={text}
                      className="bg-background/90 backdrop-blur-sm border-border text-foreground shadow-lg"
                    >
                      {text}
                    </Badge>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
