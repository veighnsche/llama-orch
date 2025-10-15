'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { ProgressBar, StatsGrid } from '@rbee/ui/molecules'
import { ArrowRight, Clock, DollarSign, Shield, Zap } from 'lucide-react'
import Link from 'next/link'

export function ProvidersHero() {
  return (
    <section className="relative overflow-hidden border-b border-border bg-gradient-to-b from-background via-card to-background px-6 py-20 lg:py-28">
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
              <Zap className="h-3.5 w-3.5" />💡 Turn Idle GPUs Into Income
            </Badge>

            {/* Headline */}
            <h1 className="mb-4 text-balance text-5xl font-extrabold leading-[1.05] tracking-tight text-foreground lg:text-6xl">
              Your GPUs Can Pay You Every Month
            </h1>

            {/* Supporting line */}
            <p className="mb-6 text-pretty text-lg leading-snug text-muted-foreground">
              Join the rbee marketplace and get paid when developers use your spare compute. Plug in once, set your
              price, and start earning automatically.
            </p>

            {/* Value bullets (stat pills) */}
            <StatsGrid
              variant="pills"
              columns={3}
              className="mb-6"
              stats={[
                {
                  icon: DollarSign,
                  value: '€50–200',
                  label: 'per GPU / month',
                },
                {
                  icon: Clock,
                  value: '24/7',
                  label: 'Passive income',
                },
                {
                  icon: Shield,
                  value: '100%',
                  label: 'Secure payouts',
                },
              ]}
            />

            {/* Primary CTAs */}
            <div className="mb-5 flex flex-col gap-3 sm:flex-row">
              <Button
                size="lg"
                className="bg-primary text-primary-foreground transition-transform hover:bg-primary/90 active:scale-[0.98] animate-in fade-in slide-in-from-bottom-2 delay-150 duration-700"
                aria-label="Start earning with rbee"
              >
                Start Earning
                <ArrowRight className="h-4 w-4" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border transition-transform hover:bg-secondary active:scale-[0.98]"
                asChild
              >
                <Link href="#earnings-calculator">Estimate My Payout</Link>
              </Button>
            </div>

            {/* Micro-trust row */}
            <p className="text-sm text-muted-foreground">
              No tech expertise needed • Set your own prices • Pause anytime
            </p>
          </div>

          {/* Right: Earnings Dashboard Visual */}
          <div className="flex items-center justify-center animate-in fade-in slide-in-from-bottom-2 delay-200 duration-700">
            <div className="relative w-full max-w-md lg:max-w-lg">
              {/* Glow effect */}
              <div className="absolute -inset-4 rounded-2xl bg-primary/20 blur-3xl" />

              {/* Card shell */}
              <div className="relative rounded-2xl border bg-card/70 p-6 shadow-[0_10px_40px_-12px_rgb(0_0_0_/_0.35)] backdrop-blur">
                {/* Card header */}
                <div className="mb-5 flex items-center justify-between">
                  <div className="text-sm font-medium text-muted-foreground">Your Earnings Dashboard</div>
                  <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-400/30" variant="outline">
                    Active
                  </Badge>
                </div>

                {/* This Month panel */}
                <div className="mb-5 rounded-lg border border-primary/20 bg-primary/5 p-4">
                  <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">This Month</div>
                  <div className="mb-1 tabular-nums text-4xl font-extrabold text-foreground">€156.80</div>
                  <div className="mb-3 text-sm font-medium text-emerald-400">+23%</div>

                  {/* Decorative progress bar */}
                  <ProgressBar
                    label=""
                    percentage={56}
                    size="sm"
                    showLabel={false}
                    showPercentage={false}
                    className="mt-2"
                  />
                </div>

                {/* KPIs row */}
                <div className="mb-6 grid grid-cols-2 gap-3">
                  <div className="rounded-lg border bg-background/70 p-3">
                    <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">Total Hours</div>
                    <div className="tabular-nums text-3xl font-bold text-foreground">487</div>
                  </div>
                  <div className="rounded-lg border bg-background/70 p-3">
                    <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">Avg Rate</div>
                    <div className="tabular-nums text-3xl font-bold text-foreground">€0.32/hr</div>
                  </div>
                </div>

                {/* GPU list */}
                <div className="space-y-2">
                  <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Your GPUs</div>

                  <div className="group flex cursor-pointer items-center justify-between rounded-lg border bg-background/50 p-3 transition-all hover:translate-x-0.5 hover:bg-background/70">
                    <div className="flex items-center gap-2.5">
                      <div className="h-2 w-2 rounded-full bg-emerald-400" />
                      <div>
                        <div className="text-sm font-medium text-foreground">RTX 4090</div>
                        <div className="text-xs text-muted-foreground">Gaming PC</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="tabular-nums text-sm font-medium text-foreground">€89.20</div>
                      <div className="text-xs text-muted-foreground">this month</div>
                    </div>
                  </div>

                  <div className="group flex cursor-pointer items-center justify-between rounded-lg border bg-background/50 p-3 transition-all hover:translate-x-0.5 hover:bg-background/70">
                    <div className="flex items-center gap-2.5">
                      <div className="h-2 w-2 rounded-full bg-emerald-400" />
                      <div>
                        <div className="text-sm font-medium text-foreground">RTX 3080</div>
                        <div className="text-xs text-muted-foreground">Workstation</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="tabular-nums text-sm font-medium text-foreground">€67.60</div>
                      <div className="text-xs text-muted-foreground">this month</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
