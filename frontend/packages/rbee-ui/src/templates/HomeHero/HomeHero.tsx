'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { HoneycombPattern } from '@rbee/ui/icons'
import { BulletListItem, FloatingKPICard, ProgressBar, PulseBadge, TerminalWindow } from '@rbee/ui/molecules'
import { ArrowRight, DollarSign, Star } from 'lucide-react'
import { useEffect, useState } from 'react'

export interface BulletItem {
  title: string
  variant?: 'check' | 'dot' | 'arrow'
  color?: 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'
}

export interface CTAButton {
  label: string
  href?: string
  onClick?: () => void
  showIcon?: boolean
  variant?: 'default' | 'outline'
  dataUmamiEvent?: string
}

export interface TrustBadge {
  type: 'github' | 'api' | 'cost'
  label: string
  href?: string
}

export interface GPUProgress {
  label: string
  percentage: number
}

export interface HomeHeroProps {
  // Badge
  badgeText: string

  // Headline
  headlinePrefix: string
  headlineHighlight: string

  // Subcopy
  subcopy: string

  // Bullets
  bullets: BulletItem[]

  // CTAs
  primaryCTA: CTAButton
  secondaryCTA: CTAButton

  // Trust badges
  trustBadges: TrustBadge[]

  // Terminal content
  terminalTitle: string
  terminalCommand: string
  terminalOutput: {
    loading: string
    ready: string
    prompt: string
    generating: string
  }

  // GPU utilization
  gpuPoolLabel: string
  gpuProgress: GPUProgress[]

  // Cost display
  costLabel: string
  costValue: string
}

export function HomeHero({
  badgeText,
  headlinePrefix,
  headlineHighlight,
  subcopy,
  bullets,
  primaryCTA,
  secondaryCTA,
  trustBadges,
  terminalTitle,
  terminalCommand,
  terminalOutput,
  gpuPoolLabel,
  gpuProgress,
  costLabel,
  costValue,
}: HomeHeroProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      setIsVisible(true)
    } else {
      const timer = setTimeout(() => setIsVisible(true), 50)
      return () => clearTimeout(timer)
    }
  }, [])

  const renderTrustBadge = (badge: TrustBadge, index: number) => {
    if (badge.type === 'github' && badge.href) {
      return (
        <li key={index}>
          <a
            href={badge.href}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 rounded-sm"
          >
            <Star className="h-5 w-5" aria-hidden="true" />
            <span className="text-sm font-sans">{badge.label}</span>
            <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" aria-hidden="true" />
          </a>
        </li>
      )
    }

    if (badge.type === 'api') {
      return (
        <li key={index} className="flex items-center gap-3 text-muted-foreground">
          <div
            className="h-5 px-1.5 flex items-center justify-center text-xs font-sans font-bold border rounded-sm"
            aria-hidden="true"
          >
            API
          </div>
          <span className="text-sm font-sans">{badge.label}</span>
        </li>
      )
    }

    if (badge.type === 'cost') {
      return (
        <li key={index} className="flex items-center gap-3 text-muted-foreground">
          <DollarSign className="h-5 w-5" aria-hidden="true" />
          <span className="text-sm font-sans">{badge.label}</span>
        </li>
      )
    }

    return null
  }

  return (
    <section
      aria-labelledby="hero-title"
      className="relative isolate min-h-[calc(100svh-3.5rem)] flex items-center overflow-hidden bg-gradient-to-b from-background to-card"
    >
      <HoneycombPattern id="hero" size="large" fadeDirection="radial" />

      <div className="container mx-auto px-4 py-24 relative z-10">
        <div className="grid lg:grid-cols-12 gap-12 items-center">
          {/* Cols 1–6: Messaging Stack */}
          <div className="lg:col-span-6 space-y-8">
            {/* Top Badge */}
            <PulseBadge text={badgeText} />

            {/* Headline */}
            <h1
              id="hero-title"
              className={`text-5xl md:text-6xl lg:text-6xl font-bold leading-tight text-balance transition-opacity duration-250 ${
                isVisible ? 'opacity-100' : 'opacity-0'
              }`}
            >
              {headlinePrefix}
              <br />
              <span className="text-primary">{headlineHighlight}</span>
            </h1>

            {/* Subcopy */}
            <p className="text-xl text-muted-foreground leading-8 max-w-[58ch]">{subcopy}</p>

            {/* Micro-proof bullets */}
            <ul className="space-y-2">
              {bullets.map((bullet, index) => (
                <BulletListItem
                  key={index}
                  title={bullet.title}
                  variant={bullet.variant || 'check'}
                  color={bullet.color || 'chart-3'}
                />
              ))}
            </ul>

            {/* CTA Group */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                size="lg"
                className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8 focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                aria-label={primaryCTA.label}
                data-umami-event={primaryCTA.dataUmamiEvent}
                onClick={primaryCTA.onClick}
                asChild={!!primaryCTA.href}
              >
                {primaryCTA.href ? (
                  <a href={primaryCTA.href}>
                    {primaryCTA.label}
                    {primaryCTA.showIcon && <ArrowRight className="ml-2 h-5 w-5" aria-hidden="true" />}
                  </a>
                ) : (
                  <>
                    {primaryCTA.label}
                    {primaryCTA.showIcon && <ArrowRight className="ml-2 h-5 w-5" aria-hidden="true" />}
                  </>
                )}
              </Button>
              <Button
                size="lg"
                variant={secondaryCTA.variant || 'outline'}
                className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                asChild={!!secondaryCTA.href}
                onClick={secondaryCTA.onClick}
              >
                {secondaryCTA.href ? <a href={secondaryCTA.href}>{secondaryCTA.label}</a> : <>{secondaryCTA.label}</>}
              </Button>
            </div>

            {/* Bottom Support Row: Trust Badges */}
            <ul className="flex flex-wrap gap-6 pt-4">{trustBadges.map(renderTrustBadge)}</ul>
          </div>

          {/* Cols 7–12: Visual Stack */}
          <div className="lg:col-span-6 space-y-12">
            {/* Terminal Window with Floating KPI */}
            <div className="relative max-w-[520px] lg:max-w-none mx-auto mb-8">
              <TerminalWindow title={terminalTitle}>
                <div className="space-y-3">
                  <div className="text-muted-foreground">
                    <span className="text-chart-3">$</span> {terminalCommand}
                  </div>
                  <div className="text-foreground pl-4">
                    <span className="text-primary">→</span> {terminalOutput.loading}
                  </div>
                  <div className="text-foreground pl-4">
                    <span className="text-chart-3">✓</span> {terminalOutput.ready}
                  </div>
                  <div className="text-muted-foreground pl-4">
                    <span className="text-chart-2">Prompt:</span> {terminalOutput.prompt}
                  </div>
                  <div className="text-foreground pl-4 leading-relaxed" aria-live="polite" aria-atomic="true">
                    <span className="text-primary animate-pulse" aria-hidden="true">
                      ▊
                    </span>{' '}
                    {terminalOutput.generating}
                  </div>

                  {/* GPU Utilization */}
                  <div className="pt-4 space-y-2">
                    <div className="text-muted-foreground text-xs font-sans">{gpuPoolLabel}</div>
                    <div className="space-y-1">
                      {gpuProgress.map((gpu, index) => (
                        <ProgressBar key={index} label={gpu.label} percentage={gpu.percentage} />
                      ))}
                    </div>
                  </div>

                  {/* Cost Counter */}
                  <div className="pt-2 flex items-center justify-between text-xs font-sans">
                    <span className="text-muted-foreground">{costLabel}</span>
                    <span className="text-chart-3 font-bold">{costValue}</span>
                  </div>
                </div>
              </TerminalWindow>

              {/* Floating KPI Card */}
              <FloatingKPICard />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
