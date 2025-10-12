<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersCTA -->
<script setup lang="ts">
import { Button } from 'rbee-storybook/stories'
import { ArrowRight, Github } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
  primaryCta?: string
  secondaryCta?: string
  footerText?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Stop Depending on AI Providers. Start Building today.',
  subtitle: "Join 500+ developers who've taken control of their AI infrastructure.",
  primaryCta: 'Get Started Free',
  secondaryCta: 'View Documentation',
  footerText: '100% open source. No credit card required. Install in 15 minutes.',
})
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-background via-primary/20 to-background py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
        <p class="mb-8 text-balance text-lg leading-relaxed text-muted-foreground">
          {{ subtitle }}
        </p>

        <div class="flex flex-col items-center justify-center gap-4 sm:flex-row">
          <Button size="lg" class="group bg-primary text-primary-foreground hover:bg-primary/90">
            {{ primaryCta }}
            <ArrowRight class="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Button>
          <Button size="lg" variant="outline" class="border-border text-foreground hover:bg-secondary bg-transparent">
            <Github class="mr-2 h-4 w-4" />
            {{ secondaryCta }}
          </Button>
        </div>

        <p class="mt-8 text-sm text-muted-foreground">
          {{ footerText }}
        </p>
      </div>
    </div>
  </section>
</template>
