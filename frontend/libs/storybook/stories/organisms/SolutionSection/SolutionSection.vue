<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented SolutionSection component -->
<script setup lang="ts">
import { Anchor, DollarSign, Laptop, Shield } from 'lucide-vue-next'

interface Benefit {
  icon: 'DollarSign' | 'Shield' | 'Anchor' | 'Laptop'
  iconColor: string
  title: string
  description: string
}

interface Props {
  title?: string
  highlightText?: string
  description?: string
  benefits?: Benefit[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Your Hardware. Your Models.',
  highlightText: 'Your Control.',
  description: 'rbee orchestrates AI inference across every GPU in your home network—workstations, gaming PCs, Macs—turning idle hardware into a private AI infrastructure.',
  benefits: () => [
    {
      icon: 'DollarSign',
      iconColor: 'text-green-600 bg-green-100',
      title: 'Zero Ongoing Costs',
      description: 'Pay only for electricity. No subscriptions. No per-token fees.',
    },
    {
      icon: 'Shield',
      iconColor: 'text-blue-600 bg-blue-100',
      title: 'Complete Privacy',
      description: 'Code never leaves your network. GDPR-compliant by default.',
    },
    {
      icon: 'Anchor',
      iconColor: 'text-primary bg-accent',
      title: 'Never Changes',
      description: 'Models update only when YOU decide. No surprise breakages.',
    },
    {
      icon: 'Laptop',
      iconColor: 'text-muted-foreground bg-muted',
      title: 'Use All Your Hardware',
      description: 'Orchestrate across CUDA, Metal, CPU. Every GPU contributes.',
    },
  ],
})

const getIcon = (iconName: string) => {
  const icons = { DollarSign, Shield, Anchor, Laptop }
  return icons[iconName as keyof typeof icons]
}
</script>

<template>
  <section class="py-24 bg-secondary">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }} <span class="text-primary">{{ highlightText }}</span>
        </h2>
        <p class="text-xl text-muted-foreground leading-relaxed text-pretty">
          {{ description }}
        </p>
      </div>

      <!-- Architecture Diagram -->
      <div class="max-w-4xl mx-auto mb-16">
        <div class="bg-card border border-border rounded-lg p-8 shadow-lg">
          <div class="text-center mb-8">
            <div class="inline-flex items-center gap-2 px-4 py-2 bg-accent rounded-full text-accent-foreground text-sm font-medium">
              The Bee Architecture
            </div>
          </div>

          <div class="space-y-8">
            <!-- Queen -->
            <div class="flex flex-col items-center">
              <div class="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-bold text-lg shadow-md">
                👑 Queen-rbee (Orchestrator)
              </div>
              <div class="h-8 w-0.5 bg-border my-2"></div>
            </div>

            <!-- Hive Managers -->
            <div class="flex justify-center gap-4">
              <div class="bg-accent text-accent-foreground px-4 py-2 rounded-lg font-medium text-sm border border-border">
                🍯 Hive Manager 1
              </div>
              <div class="bg-accent text-accent-foreground px-4 py-2 rounded-lg font-medium text-sm border border-border">
                🍯 Hive Manager 2
              </div>
              <div class="bg-accent text-accent-foreground px-4 py-2 rounded-lg font-medium text-sm border border-border">
                🍯 Hive Manager 3
              </div>
            </div>

            <div class="flex justify-center gap-4">
              <div class="h-8 w-0.5 bg-border"></div>
              <div class="h-8 w-0.5 bg-border"></div>
              <div class="h-8 w-0.5 bg-border"></div>
            </div>

            <!-- Workers -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div class="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                🐝 Worker (CUDA)
              </div>
              <div class="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                🐝 Worker (Metal)
              </div>
              <div class="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                🐝 Worker (CPU)
              </div>
              <div class="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                🐝 Worker (CUDA)
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Key Benefits -->
      <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
        <div
          v-for="(benefit, index) in benefits"
          :key="index"
          class="bg-card border border-border rounded-lg p-6 space-y-3"
        >
          <div :class="['h-10 w-10 rounded-lg flex items-center justify-center', benefit.iconColor]">
            <component :is="getIcon(benefit.icon)" class="h-5 w-5" :stroke-width="benefit.icon === 'DollarSign' ? 3 : 2" />
          </div>
          <h3 class="text-lg font-bold text-card-foreground">{{ benefit.title }}</h3>
          <p class="text-muted-foreground text-sm leading-relaxed">
            {{ benefit.description }}
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
