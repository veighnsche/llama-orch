'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { Card, CardAction, CardContent } from '@rbee/ui/atoms/Card'
import { HeroTemplate } from '@rbee/ui/templates/HeroTemplate'
import { GPUListItem, IconCardHeader, MetricCard } from '@rbee/ui/molecules'
import { MonthlyEarningsPanel } from '@rbee/ui/organisms'
import type * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersHeroGPU = {
  name: string
  location: string
  earnings: string
  status: 'active' | 'idle'
}

export type ProvidersHeroProps = {
  kickerIcon: React.ReactNode
  kickerText: string
  headline: string
  supportingText: string
  stats: Array<{
    icon: React.ReactNode
    value: string
    label: string
  }>
  primaryCTA: {
    label: string
    ariaLabel: string
  }
  secondaryCTA: {
    label: string
    href: string
  }
  trustLine: string
  dashboard: {
    icon: React.ReactNode
    title: string
    statusBadge: string
    monthLabel: string
    monthEarnings: string
    monthGrowth: string
    progressPercentage: number
    totalHours: string
    avgRate: string
    gpuListTitle: string
    gpus: ProvidersHeroGPU[]
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersHero - Hero section for GPU providers page
 *
 * @example
 * ```tsx
 * <ProvidersHero
 *   kickerIcon={<Zap className="h-3.5 w-3.5" />}
 *   kickerText="💡 Turn Idle GPUs Into Income"
 *   headline="Your GPUs Can Pay You Every Month"
 *   // ... other props
 * />
 * ```
 */
export function ProvidersHero({
  kickerIcon,
  kickerText,
  headline,
  supportingText,
  stats,
  primaryCTA,
  secondaryCTA,
  trustLine,
  dashboard,
}: ProvidersHeroProps) {
  const asideContent = (
    <div className="flex items-center justify-center animate-in fade-in slide-in-from-bottom-2 delay-200 duration-700">
      <div className="relative w-full max-w-md lg:max-w-lg">
        {/* Glow effect */}
        <div className="absolute -inset-4 rounded-2xl bg-primary/20 blur-3xl" />

        {/* Card shell */}
        <Card className="relative bg-card/70 shadow-[0_10px_40px_-12px_rgb(0_0_0_/_0.35)] backdrop-blur">
          {/* Card header */}
          <IconCardHeader
            icon={dashboard.icon}
            title={dashboard.title}
            iconSize="md"
            iconTone="primary"
            titleClassName="text-sm font-medium text-muted-foreground"
            className="pb-5"
          />
          <CardAction className="absolute right-6 top-6">
            <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-400/30" variant="outline">
              {dashboard.statusBadge}
            </Badge>
          </CardAction>

          <CardContent>
            {/* This Month panel */}
            <MonthlyEarningsPanel
              monthLabel={dashboard.monthLabel}
              monthEarnings={dashboard.monthEarnings}
              monthGrowth={dashboard.monthGrowth}
              progressPercentage={dashboard.progressPercentage}
              className="mb-5"
            />

            {/* KPIs row */}
            <div className="mb-6 grid grid-cols-2 gap-3">
              <MetricCard label="Total Hours" value={dashboard.totalHours} />
              <MetricCard label="Avg Rate" value={dashboard.avgRate} />
            </div>

            {/* GPU list */}
            <div className="space-y-2">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {dashboard.gpuListTitle}
              </div>

              {dashboard.gpus.map((gpu, idx) => (
                <GPUListItem
                  key={idx}
                  name={gpu.name}
                  subtitle={gpu.location}
                  value={gpu.earnings}
                  label="this month"
                  status={gpu.status}
                  statusColor="bg-emerald-400"
                  className="group cursor-pointer transition-all hover:translate-x-0.5 hover:bg-background/70"
                />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )

  return (
    <HeroTemplate
      badge={{ variant: 'icon', text: kickerText, icon: kickerIcon }}
      headline={{ variant: 'simple', content: headline }}
      subcopy={supportingText}
      subcopyMaxWidth="medium"
      proofElements={{
        variant: 'stats-pills',
        items: stats,
        columns: 3,
      }}
      ctas={{
        primary: {
          label: (
            <>
              {primaryCTA.label}
              <span className="ml-2">→</span>
            </>
          ) as any,
          ariaLabel: primaryCTA.ariaLabel,
        },
        secondary: {
          label: secondaryCTA.label,
          href: secondaryCTA.href,
          variant: 'outline',
        },
      }}
      trustElements={{
        variant: 'text',
        text: trustLine,
      }}
      aside={asideContent}
      asideAriaLabel="Earnings dashboard showing monthly income from GPU rentals"
      background={{
        variant: 'custom',
        className:
          'absolute inset-0 bg-[radial-gradient(60rem_40rem_at_-10%_-20%,theme(colors.primary/15),transparent)]',
      }}
      padding="spacious"
      layout={{
        leftCols: 7,
        rightCols: 5,
      }}
    />
  )
}
