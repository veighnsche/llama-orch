<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersProblem -->
<script setup lang="ts">
import { AlertTriangle, DollarSign, Lock } from 'lucide-vue-next'

interface Problem {
  icon: 'alert' | 'dollar' | 'lock'
  title: string
  description: string
  borderColor: string
  bgGradient: string
  iconBg: string
  iconColor: string
}

interface Props {
  title?: string
  subtitle?: string
  problems?: Problem[]
  warningText?: string
}

withDefaults(defineProps<Props>(), {
  title: 'The Hidden Risk of AI-Assisted Development',
  subtitle: "You're building complex codebases with AI assistance. But what happens when your provider changes the rules?",
  problems: () => [
    {
      icon: 'alert',
      title: 'The Model Changes',
      description:
        'Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your team is blocked.',
      borderColor: 'border-destructive/50',
      bgGradient: 'from-destructive/50 to-secondary',
      iconBg: 'bg-destructive/10',
      iconColor: 'text-destructive',
    },
    {
      icon: 'dollar',
      title: 'The Price Increases',
      description:
        '$20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of control.',
      borderColor: 'border-accent/50',
      bgGradient: 'from-accent/50 to-secondary',
      iconBg: 'bg-accent/10',
      iconColor: 'text-accent',
    },
    {
      icon: 'lock',
      title: 'The Provider Shuts Down',
      description:
        'API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes unmaintainable overnight.',
      borderColor: 'border-destructive/50',
      bgGradient: 'from-destructive/50 to-secondary',
      iconBg: 'bg-destructive/10',
      iconColor: 'text-destructive',
    },
  ],
  warningText:
    'Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external providers.',
})

const getIcon = (iconType: string) => {
  switch (iconType) {
    case 'alert':
      return AlertTriangle
    case 'dollar':
      return DollarSign
    case 'lock':
      return Lock
    default:
      return AlertTriangle
  }
}
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-destructive/20 via-background to-background py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
        <p class="text-balance text-lg leading-relaxed text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="mx-auto mt-16 grid max-w-5xl gap-8 sm:grid-cols-3">
        <div
          v-for="(problem, index) in problems"
          :key="index"
          :class="[
            'group relative overflow-hidden rounded-lg border p-8 transition-all',
            'bg-gradient-to-b',
            problem.borderColor,
            problem.bgGradient,
            'hover:border-opacity-100',
          ]"
        >
          <div :class="['mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg', problem.iconBg]">
            <component :is="getIcon(problem.icon)" :class="['h-6 w-6', problem.iconColor]" />
          </div>
          <h3 class="mb-3 text-xl font-semibold text-foreground">{{ problem.title }}</h3>
          <p class="text-balance leading-relaxed text-muted-foreground">
            {{ problem.description }}
          </p>
        </div>
      </div>

      <div class="mx-auto mt-12 max-w-2xl text-center">
        <p class="text-balance text-lg font-medium leading-relaxed text-destructive">
          {{ warningText }}
        </p>
      </div>
    </div>
  </section>
</template>
