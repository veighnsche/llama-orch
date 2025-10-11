<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersSolution component -->
<script setup lang="ts">
import { DollarSign, Shield, Sliders, Zap } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Turn Your GPUs Into a Revenue Stream',
  subtitle: 'rbee connects your idle GPUs with developers who need compute power. You set the price, control availability, and earn passive income.'
})

const gpuExamples = [
  { model: 'RTX 4090', vram: '24GB VRAM', power: '450W', earnings: '€180/mo', utilization: 'at 80% utilization' },
  { model: 'RTX 4080', vram: '16GB VRAM', power: '320W', earnings: '€140/mo', utilization: 'at 80% utilization' },
  { model: 'RTX 3080', vram: '10GB VRAM', power: '320W', earnings: '€90/mo', utilization: 'at 80% utilization' }
]
</script>

<template>
  <section class="border-b border-border bg-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
          {{ title }}
        </h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="mb-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <DollarSign :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Passive Income</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Earn €50-200/month per GPU while you sleep, game, or work
          </p>
        </div>

        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Sliders :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Full Control</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Set your own prices, availability windows, and usage limits
          </p>
        </div>

        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Shield :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Secure & Private</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Your data stays private. Sandboxed execution. No access to your files
          </p>
        </div>

        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Zap :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Easy Setup</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Install in 10 minutes. No technical expertise required
          </p>
        </div>
      </div>

      <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-12">
        <div class="grid gap-12 lg:grid-cols-2">
          <div>
            <h3 class="mb-6 text-2xl font-bold text-foreground">How It Works</h3>
            <div class="space-y-6">
              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  1
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Install rbee</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    One command installs rbee on your machine. Works on Windows, Mac, and Linux.
                  </div>
                </div>
              </div>

              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  2
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Configure Your GPUs</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    Set your pricing, availability windows, and usage limits through the web dashboard.
                  </div>
                </div>
              </div>

              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  3
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Join the Marketplace</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    Your GPUs appear in the rbee marketplace. Developers can rent your compute power.
                  </div>
                </div>
              </div>

              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  4
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Earn Passive Income</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    Get paid automatically. Track earnings in real-time. Withdraw anytime.
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="flex items-center justify-center">
            <div class="w-full max-w-md rounded-xl border border-border bg-background p-6">
              <div class="mb-4 text-sm font-medium text-muted-foreground">Example Earnings</div>
              <div class="mb-6 space-y-4">
                <div v-for="gpu in gpuExamples" :key="gpu.model" class="flex items-center justify-between">
                  <div>
                    <div class="text-sm font-medium text-foreground">{{ gpu.model }}</div>
                    <div class="text-xs text-muted-foreground">{{ gpu.vram }} • {{ gpu.power }}</div>
                  </div>
                  <div class="text-right">
                    <div class="text-lg font-bold text-primary">{{ gpu.earnings }}</div>
                    <div class="text-xs text-muted-foreground">{{ gpu.utilization }}</div>
                  </div>
                </div>
              </div>

              <div class="rounded-lg border border-primary/20 bg-primary/10 p-4">
                <div class="text-xs text-primary">
                  Earnings vary based on demand, your pricing, and availability. These are conservative estimates.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
