<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersHero -->
<script setup lang="ts">
import { Button } from 'rbee-storybook/stories'
import { ArrowRight, Github, Check } from 'lucide-vue-next'

interface Props {
  badge?: string
  title?: string
  subtitle?: string
  description?: string
  primaryCta?: string
  secondaryCta?: string
  features?: string[]
  showTerminal?: boolean
}

withDefaults(defineProps<Props>(), {
  badge: 'For Developers Who Build with AI',
  title: 'Build with AI.',
  subtitle: 'Own Your Infrastructure.',
  description:
    'Stop depending on AI providers. Use rbee to orchestrate inference across ALL your home network hardware—GPUs, Macs, workstations—with zero ongoing costs.',
  primaryCta: 'Get Started Free',
  secondaryCta: 'View on GitHub',
  features: () => ['100% Open Source', 'OpenAI-Compatible API', 'Works with Zed & Cursor', 'No Cloud Required'],
  showTerminal: true,
})
</script>

<template>
  <section class="relative overflow-hidden border-b border-border bg-gradient-to-b from-background via-secondary to-background">
    <div class="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-background to-background" />

    <div class="relative mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
      <div class="mx-auto max-w-3xl text-center">
        <!-- Badge -->
        <div class="mb-8 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
          <span class="relative flex h-2 w-2">
            <span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
            <span class="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
          </span>
          {{ badge }}
        </div>

        <!-- Title -->
        <h1 class="mb-6 text-balance text-5xl font-bold tracking-tight text-foreground sm:text-6xl lg:text-7xl">
          {{ title }}
          <br />
          <span class="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            {{ subtitle }}
          </span>
        </h1>

        <!-- Description -->
        <p class="mb-8 text-balance text-xl leading-relaxed text-muted-foreground">
          {{ description }}
        </p>

        <!-- CTAs -->
        <div class="mb-12 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <Button size="lg" class="group bg-primary text-primary-foreground hover:bg-primary/90">
            {{ primaryCta }}
            <ArrowRight class="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Button>
          <Button size="lg" variant="outline" class="border-border text-foreground hover:bg-secondary bg-transparent">
            <Github class="mr-2 h-4 w-4" />
            {{ secondaryCta }}
          </Button>
        </div>

        <!-- Features -->
        <div class="flex flex-wrap items-center justify-center gap-6 text-sm text-muted-foreground">
          <div v-for="feature in features" :key="feature" class="flex items-center gap-2">
            <Check class="h-5 w-5 text-primary" />
            {{ feature }}
          </div>
        </div>
      </div>

      <!-- Animated Terminal -->
      <div v-if="showTerminal" class="mx-auto mt-16 max-w-4xl">
        <div class="overflow-hidden rounded-lg border border-border bg-secondary shadow-2xl">
          <div class="flex items-center gap-2 border-b border-border bg-muted px-4 py-3">
            <div class="h-3 w-3 rounded-full bg-destructive" />
            <div class="h-3 w-3 rounded-full bg-primary" />
            <div class="h-3 w-3 rounded-full bg-accent" />
            <span class="ml-2 text-sm text-muted-foreground">terminal</span>
          </div>
          <div class="p-6 font-mono text-sm">
            <div class="mb-2 text-muted-foreground">
              <span class="text-accent">$</span> rbee-keeper infer --model llama-3.1-70b --prompt "Generate API"
            </div>
            <div class="mb-4 text-muted-foreground/70">
              <span class="animate-pulse">▊</span> Streaming tokens...
            </div>
            <div class="space-y-1 text-foreground">
              <div class="animate-[fadeIn_0.5s_ease-in]">
                <span class="text-blue-400">export</span>
                <span class="text-primary">async</span>
                <span class="text-purple-400">function</span>
                <span class="text-accent">getUsers</span>() {
              </div>
              <div class="animate-[fadeIn_0.7s_ease-in] pl-4">
                <span class="text-blue-400">const</span> response =
                <span class="text-blue-400">await</span>
                <span class="text-accent">fetch</span>(<span class="text-primary">'/api/users'</span>)
              </div>
              <div class="animate-[fadeIn_0.9s_ease-in] pl-4">
                <span class="text-blue-400">return</span> response.<span class="text-accent">json</span>()
              </div>
              <div class="animate-[fadeIn_1.1s_ease-in]">}</div>
            </div>
            <div class="mt-4 flex items-center gap-4 text-muted-foreground/70">
              <div>GPU 1: 87%</div>
              <div>GPU 2: 92%</div>
              <div>Cost: $0.00</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
