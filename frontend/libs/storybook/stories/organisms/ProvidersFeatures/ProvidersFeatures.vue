<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersFeatures component -->
<script setup lang="ts">
import { ref, computed } from 'vue'
import { DollarSign, Shield, BarChart3, Clock, Sliders, Zap, type LucideIcon } from 'lucide-vue-next'

interface Feature {
  id: string
  icon: any
  title: string
  description: string
  details: string[]
}

const features: Feature[] = [
  {
    id: 'pricing',
    icon: DollarSign,
    title: 'Flexible Pricing Control',
    description: 'Set your own hourly rates based on GPU model, demand, and your preferences.',
    details: [
      'Dynamic pricing based on demand',
      'Set minimum and maximum rates',
      'Automatic price adjustments',
      'Competitive rate suggestions',
      'Seasonal pricing schedules',
      'Bulk discount options'
    ]
  },
  {
    id: 'availability',
    icon: Clock,
    title: 'Availability Management',
    description: 'Control exactly when your GPUs are available for rent.',
    details: [
      'Set availability windows (e.g., 9am-5pm)',
      'Weekend-only or weekday-only modes',
      'Vacation mode (pause earnings)',
      'Priority mode (your usage first)',
      'Auto-pause during gaming',
      'Calendar integration'
    ]
  },
  {
    id: 'security',
    icon: Shield,
    title: 'Security & Privacy',
    description: 'Your data and hardware are protected with enterprise-grade security.',
    details: [
      'Sandboxed execution (no file access)',
      'Encrypted communication',
      'No access to your personal data',
      'Malware scanning on all jobs',
      'Automatic security updates',
      'Insurance coverage included'
    ]
  },
  {
    id: 'analytics',
    icon: BarChart3,
    title: 'Earnings Dashboard',
    description: 'Track your earnings, utilization, and performance in real-time.',
    details: [
      'Real-time earnings tracking',
      'Utilization metrics per GPU',
      'Historical earnings charts',
      'Performance benchmarks',
      'Customer ratings and reviews',
      'Tax reporting exports'
    ]
  },
  {
    id: 'limits',
    icon: Sliders,
    title: 'Usage Limits',
    description: 'Set limits to protect your hardware and control costs.',
    details: [
      'Max hours per day/week/month',
      'Temperature monitoring and limits',
      'Power consumption caps',
      'Automatic cooldown periods',
      'Hardware health monitoring',
      'Warranty protection mode'
    ]
  },
  {
    id: 'performance',
    icon: Zap,
    title: 'Performance Optimization',
    description: 'Maximize your earnings with automatic optimization.',
    details: [
      'Automatic model selection',
      'Load balancing across GPUs',
      'Priority queue for high-paying jobs',
      'Idle detection and auto-start',
      'Performance benchmarking',
      'Earnings optimization suggestions'
    ]
  }
]

const activeFeatureId = ref(features[0].id)
const activeFeature = computed(() => features.find(f => f.id === activeFeatureId.value) || features[0])
</script>

<template>
  <section class="border-b border-border bg-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
          Everything You Need to Maximize Earnings
        </h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          Professional-grade tools to manage your GPU fleet and optimize your passive income.
        </p>
      </div>

      <div class="grid gap-8 lg:grid-cols-3">
        <!-- Feature Tabs -->
        <div class="space-y-2">
          <button
            v-for="feature in features"
            :key="feature.id"
            @click="activeFeatureId = feature.id"
            :class="[
              'w-full rounded-lg border p-4 text-left transition-all',
              activeFeatureId === feature.id
                ? 'border-primary bg-primary/10'
                : 'border-border bg-card/50 hover:border-border/70'
            ]"
          >
            <div class="flex items-center gap-3">
              <div
                :class="[
                  'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg',
                  activeFeatureId === feature.id ? 'bg-primary/20' : 'bg-secondary'
                ]"
              >
                <component
                  :is="feature.icon"
                  :size="20"
                  :class="activeFeatureId === feature.id ? 'text-primary' : 'text-muted-foreground'"
                />
              </div>
              <div class="font-medium text-foreground">{{ feature.title }}</div>
            </div>
          </button>
        </div>

        <!-- Feature Content -->
        <div class="lg:col-span-2">
          <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div class="mb-6 flex items-center gap-4">
              <div class="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                <component :is="activeFeature.icon" :size="32" class="text-primary" />
              </div>
              <div>
                <h3 class="mb-1 text-2xl font-bold text-foreground">{{ activeFeature.title }}</h3>
                <p class="text-muted-foreground">{{ activeFeature.description }}</p>
              </div>
            </div>

            <div class="grid gap-3 sm:grid-cols-2">
              <div
                v-for="(detail, index) in activeFeature.details"
                :key="index"
                class="flex items-start gap-3 rounded-lg border border-border bg-background/50 p-3"
              >
                <div class="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                <div class="text-sm leading-relaxed text-muted-foreground">{{ detail }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
