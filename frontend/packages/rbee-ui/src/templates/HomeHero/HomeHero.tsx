'use client'

import { FloatingKPICard, ProgressBar, TerminalWindow } from '@rbee/ui/molecules'
import { HeroTemplate } from '@rbee/ui/templates/HeroTemplate'

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

  // Floating KPI card
  floatingKPI?: {
    gpuPool?: { label: string; value: string }
    cost?: { label: string; value: string }
    latency?: { label: string; value: string }
  }
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
  floatingKPI,
}: HomeHeroProps) {
  const asideContent = (
    <div className="space-y-12">
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
        <FloatingKPICard {...floatingKPI} />
      </div>
    </div>
  )

  const bulletItems = bullets.map((bullet) => ({
    title: bullet.title,
    variant: bullet.variant || 'check',
    color: bullet.color || 'chart-3',
  }))

  const trustBadgeItems = trustBadges.map((badge) => ({
    type: badge.type,
    label: badge.label,
    href: badge.href,
  }))

  return (
    <HeroTemplate
      badge={{ variant: 'pulse', text: badgeText }}
      headline={{ variant: 'two-line-highlight', prefix: headlinePrefix, highlight: headlineHighlight }}
      subcopy={subcopy}
      subcopyMaxWidth="narrow"
      proofElements={{
        variant: 'bullets',
        items: bulletItems,
      }}
      ctas={{
        primary: {
          label: primaryCTA.label,
          href: primaryCTA.href,
          onClick: primaryCTA.onClick,
          showIcon: primaryCTA.showIcon,
          dataUmamiEvent: primaryCTA.dataUmamiEvent,
        },
        secondary: {
          label: secondaryCTA.label,
          href: secondaryCTA.href,
          onClick: secondaryCTA.onClick,
          variant: secondaryCTA.variant || 'outline',
        },
      }}
      trustElements={{
        variant: 'badges',
        items: trustBadgeItems,
      }}
      aside={asideContent}
      asideAriaLabel="Terminal demonstration showing AI inference with GPU utilization metrics"
      background={{ variant: 'honeycomb', size: 'large', fadeDirection: 'radial' }}
      padding="default"
      headingId="hero-title"
    />
  )
}
