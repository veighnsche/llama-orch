<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented SocialProofSection component -->
<script setup lang="ts">
interface Metric {
  value: string
  label: string
  highlight?: boolean
}

interface Testimonial {
  name: string
  role: string
  avatarGradient: string
  quote: string
}

interface Props {
  title?: string
  metrics?: Metric[]
  testimonials?: Testimonial[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Trusted by Developers Who Value Independence',
  metrics: () => [
    { value: '1,200+', label: 'GitHub Stars' },
    { value: '500+', label: 'Active Installations' },
    { value: '8,000+', label: 'GPUs Orchestrated' },
    { value: '€0', label: 'Avg Monthly Cost', highlight: true },
  ],
  testimonials: () => [
    {
      name: 'Alex K.',
      role: 'Solo Developer',
      avatarGradient: 'from-blue-400 to-blue-600',
      quote: 'I was spending $80/month on Claude for coding. Now I run Llama 70B on my gaming PC and old workstation. Same quality, $0 cost. Never going back.',
    },
    {
      name: 'Sarah M.',
      role: 'CTO at StartupCo',
      avatarGradient: 'from-amber-400 to-amber-600',
      quote: 'We cut our AI costs from $500/month to zero by pooling our team\'s hardware. rbee just works. OpenAI-compatible API means no code changes.',
    },
    {
      name: 'Dr. Thomas R.',
      role: 'Research Lab Director',
      avatarGradient: 'from-green-400 to-green-600',
      quote: 'GDPR compliance was killing us. rbee let us build AI infrastructure on-premises. EU-only routing with Rhai scripts. Perfect solution.',
    },
  ],
})
</script>

<template>
  <section class="py-24 bg-secondary">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <!-- Metrics -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto mb-16">
        <div v-for="(metric, index) in metrics" :key="index" class="text-center">
          <div :class="['text-4xl font-bold mb-2', metric.highlight ? 'text-green-600' : 'text-primary']">
            {{ metric.value }}
          </div>
          <div class="text-muted-foreground">{{ metric.label }}</div>
        </div>
      </div>

      <!-- Testimonials -->
      <div class="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <div
          v-for="(testimonial, index) in testimonials"
          :key="index"
          class="bg-card border border-border rounded-lg p-6 space-y-4"
        >
          <div class="flex items-center gap-3">
            <div :class="['h-12 w-12 rounded-full bg-gradient-to-br', testimonial.avatarGradient]"></div>
            <div>
              <div class="font-bold text-card-foreground">{{ testimonial.name }}</div>
              <div class="text-sm text-muted-foreground">{{ testimonial.role }}</div>
            </div>
          </div>
          <p class="text-muted-foreground leading-relaxed">
            "{{ testimonial.quote }}"
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
