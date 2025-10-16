'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card, CardAction, CardContent } from '@rbee/ui/atoms/Card'
import { IconCardHeader, StatsGrid } from '@rbee/ui/molecules'
import { MonthlyEarningsPanel } from '@rbee/ui/organisms'
import Link from 'next/link'
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
  return (
    <section className="relative overflow-hidden bg-gradient-to-b from-background via-card to-background px-6 py-20 lg:py-28">
      {/* Background layer: subtle grid + beam */}
      <div className="absolute inset-0 bg-[radial-gradient(60rem_40rem_at_-10%_-20%,theme(colors.primary/15),transparent)]" />
      <div
        className="absolute inset-0 opacity-[0.15]"
        style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgb(148 163 184 / 0.15) 1px, transparent 0)`,
          backgroundSize: '40px 40px',
        }}
      />

      <div className="relative mx-auto max-w-7xl">
        <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:gap-16">
          {/* Left: Messaging */}
          <div className="flex flex-col justify-center animate-in fade-in slide-in-from-bottom-2 duration-700">
            {/* Kicker badge */}
            <Badge className="mb-5 w-fit animate-in fade-in zoom-in-95 duration-500" variant="outline">
              {kickerIcon}
              {kickerText}
            </Badge>

            {/* Headline */}
            <h1 className="mb-4 text-balance text-5xl font-extrabold leading-[1.05] tracking-tight text-foreground lg:text-6xl">
              {headline}
            </h1>

            {/* Supporting line */}
            <p className="mb-6 text-pretty text-lg leading-snug text-muted-foreground">{supportingText}</p>

            {/* Value bullets (stat pills) */}
            <StatsGrid variant="pills" columns={3} className="mb-6" stats={stats} />

            {/* Primary CTAs */}
            <div className="mb-5 flex flex-col gap-3 sm:flex-row">
              <Button
                size="lg"
                className="bg-primary text-primary-foreground transition-transform hover:bg-primary/90 active:scale-[0.98] animate-in fade-in slide-in-from-bottom-2 delay-150 duration-700"
                aria-label={primaryCTA.ariaLabel}
              >
                {primaryCTA.label}
                <span className="ml-2">→</span>
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border transition-transform hover:bg-secondary active:scale-[0.98]"
                asChild
              >
                <Link href={secondaryCTA.href}>{secondaryCTA.label}</Link>
              </Button>
            </div>

            {/* Micro-trust row */}
            <p className="text-sm text-muted-foreground">{trustLine}</p>
          </div>

          {/* Right: Earnings Dashboard Visual */}
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
                    <div className="rounded-lg border bg-background/70 p-3">
                      <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">Total Hours</div>
                      <div className="tabular-nums text-3xl font-bold text-foreground">{dashboard.totalHours}</div>
                    </div>
                    <div className="rounded-lg border bg-background/70 p-3">
                      <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">Avg Rate</div>
                      <div className="tabular-nums text-3xl font-bold text-foreground">{dashboard.avgRate}</div>
                    </div>
                  </div>

                  {/* GPU list */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {dashboard.gpuListTitle}
                    </div>

                    {dashboard.gpus.map((gpu, idx) => (
                      <div
                        key={idx}
                        className="group flex cursor-pointer items-center justify-between rounded-lg border bg-background/50 p-3 transition-all hover:translate-x-0.5 hover:bg-background/70"
                      >
                        <div className="flex items-center gap-2.5">
                          <div
                            className={`h-2 w-2 rounded-full ${
                              gpu.status === 'active' ? 'bg-emerald-400' : 'bg-muted-foreground'
                            }`}
                          />
                          <div>
                            <div className="text-sm font-medium text-foreground">{gpu.name}</div>
                            <div className="text-xs text-muted-foreground">{gpu.location}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="tabular-nums text-sm font-medium text-foreground">{gpu.earnings}</div>
                          <div className="text-xs text-muted-foreground">this month</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
