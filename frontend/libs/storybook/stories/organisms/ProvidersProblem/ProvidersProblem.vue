<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersProblem component -->
<script setup lang="ts">
import { AlertCircle, TrendingDown, Zap } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Your Hardware Is Losing Money Every Second',
  subtitle: 'That expensive GPU you bought? It sits idle 90% of the time, costing you electricity while earning nothing.'
})
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-background via-destructive/10 to-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
          {{ title }}
        </h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="grid gap-8 md:grid-cols-3">
        <div class="rounded-2xl border border-destructive/50 bg-gradient-to-b from-destructive/20 to-background p-8">
          <div class="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
            <TrendingDown :size="24" class="text-destructive" />
          </div>
          <h3 class="mb-3 text-xl font-bold text-foreground">Wasted Investment</h3>
          <p class="text-pretty leading-relaxed text-muted-foreground">
            You spent €1,500+ on a high-end GPU. It runs at full capacity maybe 10% of the time. The other 90%? Pure waste.
          </p>
        </div>

        <div class="rounded-2xl border border-destructive/50 bg-gradient-to-b from-destructive/20 to-background p-8">
          <div class="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
            <Zap :size="24" class="text-destructive" />
          </div>
          <h3 class="mb-3 text-xl font-bold text-foreground">Electricity Costs</h3>
          <p class="text-pretty leading-relaxed text-muted-foreground">
            Your GPU still draws power even when idle. You're paying €10-30/month in electricity for hardware that's doing nothing.
          </p>
        </div>

        <div class="rounded-2xl border border-destructive/50 bg-gradient-to-b from-destructive/20 to-background p-8">
          <div class="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
            <AlertCircle :size="24" class="text-destructive" />
          </div>
          <h3 class="mb-3 text-xl font-bold text-foreground">Missed Opportunity</h3>
          <p class="text-pretty leading-relaxed text-muted-foreground">
            Developers need GPU power and are willing to pay for it. Your idle hardware could be earning €50-200/month. Instead, it's earning nothing.
          </p>
        </div>
      </div>

      <div class="mt-12 rounded-2xl border border-border bg-card/50 p-8 text-center">
        <p class="text-balance text-xl font-medium text-card-foreground">
          Every hour your GPU sits idle is money left on the table. What if you could turn that waste into passive income?
        </p>
      </div>
    </div>
  </section>
</template>
