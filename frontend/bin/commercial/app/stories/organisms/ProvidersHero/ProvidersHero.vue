<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersHero component -->
<script setup lang="ts">
import { Button } from '~/stories'
import { ArrowRight, Zap, DollarSign, Shield } from 'lucide-vue-next'

interface Props {
  badge?: string
  title?: string
  description?: string
  monthlyEarnings?: string
  availability?: string
  securityPercent?: string
  ctaPrimary?: string
  ctaSecondary?: string
  disclaimer?: string
}

const props = withDefaults(defineProps<Props>(), {
  badge: 'Turn Idle GPUs Into Income',
  title: 'Your GPUs Are Worth More Than You Think',
  description: 'Stop letting your gaming PC, workstation, or mining rig sit idle. Join the rbee marketplace and earn passive income by providing GPU power to developers who need it.',
  monthlyEarnings: '€50-200',
  availability: '24/7',
  securityPercent: '100%',
  ctaPrimary: 'Start Earning Now',
  ctaSecondary: 'Calculate Earnings',
  disclaimer: 'No technical expertise required • Set your own prices • Control your availability'
})
</script>

<template>
  <section class="relative overflow-hidden border-b border-border bg-gradient-to-b from-background via-secondary to-background px-6 py-24 lg:py-32">
    <div class="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-background to-background" />

    <div class="relative mx-auto max-w-7xl">
      <div class="grid gap-12 lg:grid-cols-2 lg:gap-16">
        <!-- Left: Messaging -->
        <div class="flex flex-col justify-center">
          <div class="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
            <Zap :size="16" />
            {{ badge }}
          </div>

          <h1 class="mb-6 text-balance text-5xl font-bold leading-tight text-foreground lg:text-6xl">
            {{ title }}
          </h1>

          <p class="mb-8 text-pretty text-xl leading-relaxed text-muted-foreground">
            {{ description }}
          </p>

          <div class="mb-8 grid gap-4 sm:grid-cols-3">
            <div class="flex items-center gap-3">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <DollarSign :size="20" class="text-primary" />
              </div>
              <div>
                <div class="text-2xl font-bold text-foreground">{{ monthlyEarnings }}</div>
                <div class="text-sm text-muted-foreground">per GPU/month</div>
              </div>
            </div>

            <div class="flex items-center gap-3">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Zap :size="20" class="text-primary" />
              </div>
              <div>
                <div class="text-2xl font-bold text-foreground">{{ availability }}</div>
                <div class="text-sm text-muted-foreground">Passive income</div>
              </div>
            </div>

            <div class="flex items-center gap-3">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Shield :size="20" class="text-primary" />
              </div>
              <div>
                <div class="text-2xl font-bold text-foreground">{{ securityPercent }}</div>
                <div class="text-sm text-muted-foreground">Secure</div>
              </div>
            </div>
          </div>

          <div class="flex flex-col gap-4 sm:flex-row">
            <Button size="lg" class="bg-primary text-primary-foreground hover:bg-primary/90">
              {{ ctaPrimary }}
              <ArrowRight :size="16" class="ml-2" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              class="border-border text-foreground hover:bg-secondary"
            >
              {{ ctaSecondary }}
            </Button>
          </div>

          <p class="mt-6 text-sm text-muted-foreground">
            {{ disclaimer }}
          </p>
        </div>

        <!-- Right: Visual -->
        <div class="flex items-center justify-center">
          <div class="relative w-full max-w-lg">
            <div class="absolute inset-0 bg-gradient-to-r from-primary/20 to-accent/20 blur-3xl" />
            <div class="relative rounded-2xl border border-border bg-card/50 p-8 backdrop-blur-sm">
              <div class="mb-6 flex items-center justify-between">
                <div class="text-sm font-medium text-muted-foreground">Your Earnings Dashboard</div>
                <div class="rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-400">
                  Active
                </div>
              </div>

              <div class="mb-8 space-y-4">
                <div>
                  <div class="mb-2 text-sm text-muted-foreground">This Month</div>
                  <div class="text-4xl font-bold text-foreground">€156.80</div>
                  <div class="text-sm text-green-400">+23% from last month</div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div class="rounded-lg border border-border bg-background/50 p-4">
                    <div class="mb-1 text-sm text-muted-foreground">Total Hours</div>
                    <div class="text-2xl font-bold text-foreground">487</div>
                  </div>
                  <div class="rounded-lg border border-border bg-background/50 p-4">
                    <div class="mb-1 text-sm text-muted-foreground">Avg Rate</div>
                    <div class="text-2xl font-bold text-foreground">€0.32/hr</div>
                  </div>
                </div>
              </div>

              <div class="space-y-3">
                <div class="text-sm font-medium text-muted-foreground">Your GPUs</div>

                <div class="flex items-center justify-between rounded-lg border border-border bg-background/50 p-3">
                  <div class="flex items-center gap-3">
                    <div class="h-2 w-2 rounded-full bg-green-400" />
                    <div>
                      <div class="text-sm font-medium text-foreground">RTX 4090</div>
                      <div class="text-xs text-muted-foreground">Gaming PC</div>
                    </div>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-medium text-foreground">€89.20</div>
                    <div class="text-xs text-muted-foreground">this month</div>
                  </div>
                </div>

                <div class="flex items-center justify-between rounded-lg border border-border bg-background/50 p-3">
                  <div class="flex items-center gap-3">
                    <div class="h-2 w-2 rounded-full bg-green-400" />
                    <div>
                      <div class="text-sm font-medium text-foreground">RTX 3080</div>
                      <div class="text-xs text-muted-foreground">Workstation</div>
                    </div>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-medium text-foreground">€67.60</div>
                    <div class="text-xs text-muted-foreground">this month</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
