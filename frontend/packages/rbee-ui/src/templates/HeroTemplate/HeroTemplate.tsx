'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { HoneycombPattern } from '@rbee/ui/icons'
import { BulletListItem, ComplianceChip, PulseBadge, StatsGrid } from '@rbee/ui/molecules'
import { ArrowRight, DollarSign, Star } from 'lucide-react'
import { useEffect, useState } from 'react'
import type {
  AssuranceItem,
  BadgeItem,
  BulletItem,
  ComplianceChipItem,
  HeroTemplateProps,
  IndicatorItem,
  StatItem,
  TrustBadgeItem,
} from './HeroTemplateProps'

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

export function HeroTemplate({
  badge,
  headline,
  subcopy,
  subcopyMaxWidth = 'medium',
  proofElements,
  ctas,
  trustElements,
  helperText,
  aside,
  asideAriaLabel,
  layout = {},
  background = { variant: 'gradient' },
  padding = 'default',
  animations = { enabled: true, stagger: true, direction: 'bottom' },
  headingId = 'hero-title',
}: HeroTemplateProps) {
  const [isVisible, setIsVisible] = useState(false)

  // Animation setup
  useEffect(() => {
    if (!animations.enabled) {
      setIsVisible(true)
      return
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      setIsVisible(true)
    } else {
      const timer = setTimeout(() => setIsVisible(true), 50)
      return () => clearTimeout(timer)
    }
  }, [animations.enabled])

  // Layout defaults
  const leftCols = layout.leftCols || 6
  const rightCols = layout.rightCols || 6
  const gap = layout.gap || 12
  const verticalSpacing = layout.verticalSpacing || 8

  // Padding classes
  const paddingClasses = {
    default: 'py-24',
    compact: 'py-20 lg:py-24',
    spacious: 'py-24 lg:py-32',
  }[padding]

  // Subcopy max width classes
  const subcopyMaxWidthClasses = {
    narrow: 'max-w-[58ch]',
    medium: 'max-w-2xl',
    wide: 'max-w-prose',
  }[subcopyMaxWidth]

  // Background rendering
  const renderBackground = () => {
    if (background.variant === 'honeycomb') {
      return (
        <HoneycombPattern
          id="hero"
          size={background.size || 'large'}
          fadeDirection={background.fadeDirection || 'radial'}
        />
      )
    }

    if (background.variant === 'radial') {
      return (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(60rem_40rem_at_20%_-10%,theme(colors.primary/8),transparent)]"
        />
      )
    }

    if (background.variant === 'custom') {
      return <div aria-hidden className={background.className} />
    }

    // Default gradient - no extra element needed
    return null
  }

  // Badge rendering
  const renderBadge = () => {
    if (!badge || badge.variant === 'none') return null

    if (badge.variant === 'pulse') {
      return <PulseBadge text={badge.text} />
    }

    if (badge.variant === 'icon') {
      return (
        <Badge variant="outline" className="mb-6 w-fit border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
          <div className="flex h-4 w-4 items-center justify-center">{badge.icon}</div>
          <span>{badge.text}</span>
        </Badge>
      )
    }

    if (badge.variant === 'simple') {
      return (
        <Badge variant="secondary" className="mb-4">
          {badge.text}
        </Badge>
      )
    }

    return null
  }

  // Headline rendering
  const renderHeadline = () => {
    const baseClasses = `text-5xl md:text-6xl lg:text-6xl font-bold leading-tight text-balance transition-opacity duration-250 ${
      isVisible ? 'opacity-100' : 'opacity-0'
    }`

    if (headline.variant === 'two-line-highlight') {
      return (
        <h1 id={headingId} className={baseClasses}>
          {headline.prefix}
          <br />
          <span className="text-primary">{headline.highlight}</span>
        </h1>
      )
    }

    if (headline.variant === 'inline-highlight') {
      return (
        <h1 id={headingId} className={baseClasses}>
          {headline.content}{' '}
          <span className="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            {headline.highlight}
          </span>
        </h1>
      )
    }

    if (headline.variant === 'custom') {
      return (
        <h1 id={headingId} className={baseClasses}>
          {headline.content}
        </h1>
      )
    }

    if (headline.variant === 'simple') {
      return (
        <h1 id={headingId} className={baseClasses}>
          {headline.content}
        </h1>
      )
    }

    return null
  }

  // Proof elements rendering
  const renderProofElements = () => {
    if (!proofElements || proofElements.variant === 'none') return null

    if (proofElements.variant === 'bullets') {
      return (
        <ul className="space-y-2">
          {proofElements.items.map((bullet: BulletItem, index) => (
            <BulletListItem
              key={index}
              title={bullet.title}
              variant={bullet.variant || 'check'}
              color={bullet.color || 'chart-3'}
            />
          ))}
        </ul>
      )
    }

    if (proofElements.variant === 'stats-tiles') {
      return (
        <StatsGrid
          variant="tiles"
          columns={proofElements.columns || 3}
          className="animate-in fade-in-50 [animation-delay:120ms]"
          stats={proofElements.items as StatItem[]}
        />
      )
    }

    if (proofElements.variant === 'stats-pills') {
      return (
        <StatsGrid variant="pills" columns={proofElements.columns || 3} stats={proofElements.items as StatItem[]} />
      )
    }

    if (proofElements.variant === 'badges') {
      return (
        <div className="flex flex-wrap gap-2">
          {proofElements.items.map((item: BadgeItem, index) => (
            <Badge key={index} variant="secondary">
              {item.text}
            </Badge>
          ))}
        </div>
      )
    }

    if (proofElements.variant === 'indicators') {
      return (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted-foreground">
          {proofElements.items.map((indicator: IndicatorItem, index) => (
            <span key={index} className={indicator.hasDot ? 'inline-flex items-center gap-2' : undefined}>
              {indicator.hasDot && <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" aria-hidden />}
              {indicator.text}
            </span>
          ))}
        </div>
      )
    }

    if (proofElements.variant === 'assurance') {
      return (
        <ul className={`grid grid-cols-${proofElements.columns || 2} gap-3 text-sm text-muted-foreground`}>
          {proofElements.items.map((item: AssuranceItem, index) => (
            <li key={index} className="flex items-center gap-2">
              <div className="h-4 w-4 text-primary shrink-0" aria-hidden="true">
                {item.icon}
              </div>
              <span>{item.text}</span>
            </li>
          ))}
        </ul>
      )
    }

    return null
  }

  // CTA rendering
  const renderCTAs = () => {
    return (
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <Button
            size="lg"
            variant={ctas.primary.variant || 'default'}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8 focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
            aria-label={ctas.primary.ariaLabel || ctas.primary.label}
            data-umami-event={ctas.primary.dataUmamiEvent}
            onClick={ctas.primary.onClick}
            asChild={!!ctas.primary.href}
          >
            {ctas.primary.href ? (
              <a href={ctas.primary.href}>
                {ctas.primary.label}
                {ctas.primary.showIcon && <ArrowRight className="ml-2 h-5 w-5" aria-hidden="true" />}
              </a>
            ) : (
              <>
                {ctas.primary.label}
                {ctas.primary.showIcon && <ArrowRight className="ml-2 h-5 w-5" aria-hidden="true" />}
              </>
            )}
          </Button>
          <Button
            size="lg"
            variant={ctas.secondary.variant || 'outline'}
            className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
            asChild={!!ctas.secondary.href}
            onClick={ctas.secondary.onClick}
            aria-label={ctas.secondary.ariaLabel}
          >
            {ctas.secondary.href ? <a href={ctas.secondary.href}>{ctas.secondary.label}</a> : ctas.secondary.label}
          </Button>
        </div>

        {/* Tertiary link (mobile only if specified) */}
        {ctas.tertiary && (
          <div className={ctas.tertiary.mobileOnly ? 'sm:hidden' : ''}>
            <a
              href={ctas.tertiary.href}
              className="inline-flex items-center gap-2 text-sm text-primary hover:underline underline-offset-4"
            >
              {ctas.tertiary.label}
              <ArrowRight className="h-3 w-3" aria-hidden="true" />
            </a>
          </div>
        )}
      </div>
    )
  }

  // Trust elements rendering
  const renderTrustElements = () => {
    if (!trustElements || trustElements.variant === 'none') return null

    if (trustElements.variant === 'badges') {
      return (
        <ul className="flex flex-wrap gap-6 pt-4">
          {trustElements.items.map((badge: TrustBadgeItem, index) => {
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
                    <ArrowRight
                      className="h-3 w-3 transition-transform group-hover:translate-x-0.5"
                      aria-hidden="true"
                    />
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
          })}
        </ul>
      )
    }

    if (trustElements.variant === 'chips') {
      return (
        <div className="flex flex-wrap items-center gap-3" aria-live="polite">
          {trustElements.items.map((chip: ComplianceChipItem, idx) => (
            <ComplianceChip key={idx} icon={chip.icon} ariaLabel={chip.ariaLabel}>
              {chip.label}
            </ComplianceChip>
          ))}
        </div>
      )
    }

    if (trustElements.variant === 'text') {
      return <p className="text-sm text-muted-foreground">{trustElements.text}</p>
    }

    return null
  }

  return (
    <section
      aria-labelledby={headingId}
      className={`relative isolate min-h-[calc(100svh-3.5rem)] flex items-center overflow-hidden bg-gradient-to-b from-background to-card ${paddingClasses}`}
    >
      {renderBackground()}

      <div className="container mx-auto px-4 relative z-10">
        <div className={`grid lg:grid-cols-12 gap-${gap} items-center`}>
          {/* Left: Messaging Stack */}
          <div className={`lg:col-span-${leftCols} space-y-${verticalSpacing}`}>
            {/* Badge */}
            {renderBadge()}

            {/* Headline */}
            {renderHeadline()}

            {/* Subcopy */}
            <p className={`text-xl text-muted-foreground leading-8 ${subcopyMaxWidthClasses}`}>{subcopy}</p>

            {/* Proof Elements */}
            {renderProofElements()}

            {/* CTA Group */}
            {renderCTAs()}

            {/* Helper Text */}
            {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}

            {/* Trust Elements */}
            {renderTrustElements()}
          </div>

          {/* Right: Aside (Page-specific visual) */}
          <div className={`lg:col-span-${rightCols}`} aria-label={asideAriaLabel}>
            {aside}
          </div>
        </div>
      </div>
    </section>
  )
}
