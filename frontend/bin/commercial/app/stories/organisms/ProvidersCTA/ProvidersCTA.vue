<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersCTA component -->
<script setup lang="ts">
import { Button } from '~/stories'
import { ArrowRight, Zap } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
  ctaPrimary?: string
  ctaSecondary?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Your GPUs Are Losing Money Every Second',
  subtitle: 'Stop letting your hardware sit idle. Join 500+ providers earning passive income on the rbee marketplace.',
  ctaPrimary: 'Start Earning Now',
  ctaSecondary: 'View Documentation'
})
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-background via-primary/10 to-background px-6 py-24">
    <div class="mx-auto max-w-4xl text-center">
      <div class="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
        <Zap :size="16" />
        Start Earning Today
      </div>

      <h2 class="mb-6 text-balance text-4xl font-bold text-foreground lg:text-5xl">
        {{ title }}
      </h2>

      <p class="mb-8 text-balance text-xl text-muted-foreground">
        {{ subtitle }}
      </p>

      <div class="mb-8 flex flex-col items-center justify-center gap-4 sm:flex-row">
        <Button size="lg" class="bg-primary text-primary-foreground hover:bg-primary/90">
          {{ ctaPrimary }}
          <ArrowRight :size="16" class="ml-2" />
        </Button>
        <Button size="lg" variant="outline" class="border-border text-foreground hover:bg-secondary">
          {{ ctaSecondary }}
        </Button>
      </div>

      <div class="grid gap-6 text-sm text-muted-foreground sm:grid-cols-3">
        <div>
          <div class="mb-1 font-medium text-foreground">Setup Time</div>
          <div>Less than 15 minutes</div>
        </div>
        <div>
          <div class="mb-1 font-medium text-foreground">Commission</div>
          <div>Only 15% (you keep 85%)</div>
        </div>
        <div>
          <div class="mb-1 font-medium text-foreground">Minimum Payout</div>
          <div>€25 (weekly payouts)</div>
        </div>
      </div>
    </div>
  </section>
</template>
