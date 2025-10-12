"use client"

import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, Zap, DollarSign, Shield } from "lucide-react"
import Link from "next/link"

export function ProvidersHero() {
  return (
    <section className="relative overflow-hidden border-b border-border bg-gradient-to-b from-background via-card to-background px-6 py-24 lg:py-32">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-background to-background" />

      <div className="relative mx-auto max-w-7xl">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
          {/* Left: Messaging */}
          <div className="flex flex-col justify-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
              <Zap className="h-4 w-4" />
              Turn Idle GPUs Into Income
            </div>

            <h1 className="mb-6 text-balance text-5xl font-bold leading-tight text-foreground lg:text-6xl">
              Your GPUs Are Worth More Than You Think
            </h1>

            <p className="mb-8 text-pretty text-xl leading-relaxed text-muted-foreground">
              Stop letting your gaming PC, workstation, or mining rig sit idle. Join the rbee marketplace and earn
              passive income by providing GPU power to developers who need it.
            </p>

            <div className="mb-8 grid gap-4 sm:grid-cols-3">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <DollarSign className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <div className="text-2xl font-bold text-foreground">€50-200</div>
                  <div className="text-sm text-muted-foreground">per GPU/month</div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <Zap className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <div className="text-2xl font-bold text-foreground">24/7</div>
                  <div className="text-sm text-muted-foreground">Passive income</div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <Shield className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <div className="text-2xl font-bold text-foreground">100%</div>
                  <div className="text-sm text-muted-foreground">Secure</div>
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-4 sm:flex-row">
              <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90">
                Start Earning Now
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border text-foreground hover:bg-secondary bg-transparent"
                asChild
              >
                <Link href="#earnings-calculator">Calculate Earnings</Link>
              </Button>
            </div>

            <p className="mt-6 text-sm text-muted-foreground">
              No technical expertise required • Set your own prices • Control your availability
            </p>
          </div>

          {/* Right: Visual */}
          <div className="flex items-center justify-center">
            <div className="relative w-full max-w-lg">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-primary/30 blur-3xl" />
              <div className="relative rounded-2xl border border-border bg-card/50 p-8 backdrop-blur-sm">
                <div className="mb-6 flex items-center justify-between">
                  <div className="text-sm font-medium text-muted-foreground">Your Earnings Dashboard</div>
                  <div className="rounded-full bg-chart-3/10 px-3 py-1 text-xs font-medium text-chart-3">
                    Active
                  </div>
                </div>

                <div className="mb-8 space-y-4">
                  <div>
                    <div className="mb-2 text-sm text-muted-foreground">This Month</div>
                    <div className="text-4xl font-bold text-foreground">€156.80</div>
                    <div className="text-sm text-chart-3">+23% from last month</div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg border border-border bg-background/50 p-4">
                      <div className="mb-1 text-sm text-muted-foreground">Total Hours</div>
                      <div className="text-2xl font-bold text-foreground">487</div>
                    </div>
                    <div className="rounded-lg border border-border bg-background/50 p-4">
                      <div className="mb-1 text-sm text-muted-foreground">Avg Rate</div>
                      <div className="text-2xl font-bold text-foreground">€0.32/hr</div>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="text-sm font-medium text-muted-foreground">Your GPUs</div>

                  <div className="flex items-center justify-between rounded-lg border border-border bg-background/50 p-3">
                    <div className="flex items-center gap-3">
                      <div className="h-2 w-2 rounded-full bg-chart-3" />
                      <div>
                        <div className="text-sm font-medium text-foreground">RTX 4090</div>
                        <div className="text-xs text-muted-foreground">Gaming PC</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-foreground">€89.20</div>
                      <div className="text-xs text-muted-foreground">this month</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between rounded-lg border border-border bg-background/50 p-3">
                    <div className="flex items-center gap-3">
                      <div className="h-2 w-2 rounded-full bg-chart-3" />
                      <div>
                        <div className="text-sm font-medium text-foreground">RTX 3080</div>
                        <div className="text-xs text-muted-foreground">Workstation</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-foreground">€67.60</div>
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
