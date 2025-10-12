<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseCTA -->
<script setup lang="ts">
import { Button } from '~/stories'
import { Calendar, FileText, MessageSquare } from 'lucide-vue-next'

interface CtaOption {
  icon: any
  title: string
  description: string
  buttonText: string
  variant: 'default' | 'outline'
}

interface Props {
  title?: string
  subtitle?: string
  options?: CtaOption[]
  disclaimer?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Ready to Meet Your Compliance Requirements?',
  subtitle: 'Schedule a demo with our compliance team or download our compliance documentation package.',
  options: () => [
    {
      icon: Calendar,
      title: 'Schedule Demo',
      description: '30-minute demo with our compliance team. See rbee in action.',
      buttonText: 'Book Demo',
      variant: 'default',
    },
    {
      icon: FileText,
      title: 'Compliance Pack',
      description: 'Download GDPR, SOC2, and ISO 27001 documentation.',
      buttonText: 'Download Docs',
      variant: 'outline',
    },
    {
      icon: MessageSquare,
      title: 'Talk to Sales',
      description: 'Discuss your specific compliance requirements.',
      buttonText: 'Contact Sales',
      variant: 'outline',
    },
  ],
  disclaimer: 'Enterprise support available 24/7. Typical deployment: 6-8 weeks from consultation to production.',
})
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-background via-primary/5 to-background px-6 py-24">
    <div class="mx-auto max-w-4xl text-center">
      <h2 class="mb-4 text-4xl font-bold text-foreground lg:text-5xl">{{ title }}</h2>
      <p class="mb-12 text-balance text-xl text-muted-foreground">
        {{ subtitle }}
      </p>

      <div class="mb-12 grid gap-6 md:grid-cols-3">
        <div v-for="(option, i) in options" :key="i" class="rounded-lg border border-border bg-card p-6">
          <div class="mb-4 flex justify-center">
            <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <component :is="option.icon" class="h-6 w-6 text-primary" />
            </div>
          </div>
          <h3 class="mb-2 font-semibold text-foreground">{{ option.title }}</h3>
          <p class="mb-4 text-sm leading-relaxed text-muted-foreground">
            {{ option.description }}
          </p>
          <Button
            :variant="option.variant"
            class="w-full"
            :class="option.variant === 'default' ? 'bg-primary text-primary-foreground hover:bg-primary/90' : 'border-border text-foreground hover:bg-secondary'"
          >
            {{ option.buttonText }}
          </Button>
        </div>
      </div>

      <p class="text-sm text-muted-foreground">
        {{ disclaimer }}
      </p>
    </div>
  </section>
</template>
