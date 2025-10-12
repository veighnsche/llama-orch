<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersHowItWorks component -->
<script setup lang="ts">
import { Download, Settings, Globe, Wallet } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
  setupTime?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Start Earning in 4 Simple Steps',
  subtitle: 'No technical expertise required. Get your GPUs earning in less than 15 minutes.',
  setupTime: '12 minutes'
})
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-background to-secondary px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
          {{ title }}
        </h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
        <div class="relative">
          <div class="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-accent">
            <Download :size="32" class="text-primary-foreground" />
          </div>
          <div class="mb-2 text-sm font-medium text-primary">Step 1</div>
          <h3 class="mb-3 text-xl font-bold text-foreground">Install rbee</h3>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            Download and install rbee with one command. Works on Windows, Mac, and Linux.
          </p>
          <div class="rounded-lg border border-border bg-background p-3">
            <code class="text-xs text-primary">curl -sSL rbee.dev/install.sh | sh</code>
          </div>
        </div>

        <div class="relative">
          <div class="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-accent">
            <Settings :size="32" class="text-primary-foreground" />
          </div>
          <div class="mb-2 text-sm font-medium text-primary">Step 2</div>
          <h3 class="mb-3 text-xl font-bold text-foreground">Configure Settings</h3>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            Set your pricing, availability windows, and usage limits through the intuitive web dashboard.
          </p>
          <ul class="space-y-2 text-sm text-muted-foreground">
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-primary" />
              Set hourly rate
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-primary" />
              Define availability
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-primary" />
              Set usage limits
            </li>
          </ul>
        </div>

        <div class="relative">
          <div class="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-accent">
            <Globe :size="32" class="text-primary-foreground" />
          </div>
          <div class="mb-2 text-sm font-medium text-primary">Step 3</div>
          <h3 class="mb-3 text-xl font-bold text-foreground">Join Marketplace</h3>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            Your GPUs automatically appear in the rbee marketplace. Developers can discover and rent your compute power.
          </p>
          <div class="rounded-lg border border-green-500/50 bg-green-500/20 p-3">
            <div class="text-xs font-medium text-green-400">Your GPUs are now live and earning!</div>
          </div>
        </div>

        <div class="relative">
          <div class="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-accent">
            <Wallet :size="32" class="text-primary-foreground" />
          </div>
          <div class="mb-2 text-sm font-medium text-primary">Step 4</div>
          <h3 class="mb-3 text-xl font-bold text-foreground">Get Paid</h3>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            Track earnings in real-time. Automatic payouts. Withdraw to your bank account or crypto wallet anytime.
          </p>
          <div class="space-y-2 text-sm text-muted-foreground">
            <div class="flex justify-between">
              <span>Payout frequency:</span>
              <span class="text-foreground">Weekly</span>
            </div>
            <div class="flex justify-between">
              <span>Minimum payout:</span>
              <span class="text-foreground">€25</span>
            </div>
          </div>
        </div>
      </div>

      <div class="mt-12 text-center">
        <p class="text-lg text-muted-foreground">
          Average setup time: <span class="font-bold text-primary">{{ setupTime }}</span>
        </p>
      </div>
    </div>
  </section>
</template>
