<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersSolution -->
<script setup lang="ts">
import { Cpu, DollarSign, Lock, Zap } from 'lucide-vue-next'

interface Benefit {
  icon: 'dollar' | 'lock' | 'zap' | 'cpu'
  title: string
  description: string
}

interface Worker {
  name: string
  type: string
  emoji: string
}

interface Props {
  title?: string
  subtitle?: string
  benefits?: Benefit[]
  showArchitecture?: boolean
  architectureTitle?: string
  workers?: Worker[]
}

withDefaults(defineProps<Props>(), {
  title: 'Your Hardware. Your Models. Your Control.',
  subtitle:
    'rbee orchestrates AI inference across every GPU in your home network—workstations, gaming PCs, Macs—turning idle hardware into a private AI infrastructure.',
  benefits: () => [
    {
      icon: 'dollar',
      title: 'Zero Ongoing Costs',
      description: 'Pay only for electricity. No subscriptions. No per-token fees.',
    },
    {
      icon: 'lock',
      title: 'Complete Privacy',
      description: 'Code never leaves your network. GDPR-compliant by default.',
    },
    {
      icon: 'zap',
      title: 'Never Changes',
      description: 'Models update only when YOU decide. No surprise breakages.',
    },
    {
      icon: 'cpu',
      title: 'Use All Your Hardware',
      description: 'Orchestrate across CUDA, Metal, CPU. Every GPU contributes.',
    },
  ],
  showArchitecture: true,
  architectureTitle: 'The Bee Architecture',
  workers: () => [
    { name: 'Worker 1', type: 'CUDA GPU', emoji: '🐝' },
    { name: 'Worker 2', type: 'Metal GPU', emoji: '🐝' },
    { name: 'Worker 3', type: 'CPU', emoji: '🐝' },
  ],
})

const getIcon = (iconType: string) => {
  switch (iconType) {
    case 'dollar':
      return DollarSign
    case 'lock':
      return Lock
    case 'zap':
      return Zap
    case 'cpu':
      return Cpu
    default:
      return Cpu
  }
}
</script>

<template>
  <section class="border-b border-border py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
        <p class="text-balance text-lg leading-relaxed text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="mx-auto mt-16 grid max-w-5xl gap-8 sm:grid-cols-2 lg:grid-cols-4">
        <div
          v-for="(benefit, index) in benefits"
          :key="index"
          class="group rounded-lg border border-border bg-secondary/50 p-6 transition-all hover:border-primary/50 hover:bg-secondary"
        >
          <div class="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <component :is="getIcon(benefit.icon)" class="h-6 w-6 text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-semibold text-foreground">{{ benefit.title }}</h3>
          <p class="text-balance text-sm leading-relaxed text-muted-foreground">
            {{ benefit.description }}
          </p>
        </div>
      </div>

      <!-- Architecture Diagram -->
      <div v-if="showArchitecture" class="mx-auto mt-16 max-w-3xl">
        <div class="rounded-lg border border-border bg-secondary/50 p-8">
          <h3 class="mb-6 text-center text-xl font-semibold text-foreground">{{ architectureTitle }}</h3>
          <div class="flex flex-col items-center gap-6">
            <div class="flex items-center gap-3 rounded-lg border border-primary/30 bg-primary/10 px-6 py-3">
              <span class="text-2xl">👑</span>
              <div>
                <div class="font-semibold text-foreground">queen-rbee</div>
                <div class="text-sm text-muted-foreground">Orchestrator (brain)</div>
              </div>
            </div>

            <div class="h-8 w-px bg-border" />

            <div class="flex items-center gap-3 rounded-lg border border-border bg-muted px-6 py-3">
              <span class="text-2xl">🍯</span>
              <div>
                <div class="font-semibold text-foreground">rbee-hive</div>
                <div class="text-sm text-muted-foreground">Resource manager</div>
              </div>
            </div>

            <div class="h-8 w-px bg-border" />

            <div class="grid gap-4 sm:grid-cols-3">
              <div
                v-for="(worker, index) in workers"
                :key="index"
                class="flex items-center gap-2 rounded-lg border border-border bg-muted px-4 py-2"
              >
                <span class="text-xl">{{ worker.emoji }}</span>
                <div class="text-sm">
                  <div class="font-semibold text-foreground">{{ worker.name }}</div>
                  <div class="text-muted-foreground">{{ worker.type }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
