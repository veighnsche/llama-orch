<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersTestimonials -->
<script setup lang="ts">
interface Testimonial {
  quote: string
  author: string
  role: string
  avatar: string
}

interface Stat {
  value: string
  label: string
  highlighted?: boolean
}

interface Props {
  title?: string
  testimonials?: Testimonial[]
  stats?: Stat[]
}

withDefaults(defineProps<Props>(), {
  title: 'Trusted by Developers Who Value Independence',
  testimonials: () => [
    {
      quote:
        'I was spending $80/month on Claude for coding. Now I run Llama 70B on my gaming PC and old workstation. Same quality, $0 cost. Never going back.',
      author: 'Alex K.',
      role: 'Solo Developer',
      avatar: '👨‍💻',
    },
    {
      quote:
        "We cut our AI costs from $500/month to zero by pooling our team's hardware. rbee just works. OpenAI-compatible API means no code changes.",
      author: 'Sarah M.',
      role: 'CTO at StartupCo',
      avatar: '👩‍💼',
    },
    {
      quote:
        'The cascading shutdown guarantee is a game-changer. No more orphaned processes eating VRAM. Press Ctrl+C and everything shuts down cleanly.',
      author: 'Marcus T.',
      role: 'DevOps Engineer',
      avatar: '👨‍🔧',
    },
  ],
  stats: () => [
    { value: '1,200+', label: 'GitHub Stars', highlighted: false },
    { value: '500+', label: 'Active Installations', highlighted: false },
    { value: '8,000+', label: 'GPUs Orchestrated', highlighted: false },
    { value: '€0', label: 'Average Monthly Cost', highlighted: true },
  ],
})
</script>

<template>
  <section class="border-b border-border py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
      </div>

      <div class="mx-auto mt-16 grid max-w-6xl gap-8 sm:grid-cols-2 lg:grid-cols-3">
        <div v-for="(testimonial, index) in testimonials" :key="index" class="rounded-lg border border-border bg-secondary/50 p-6">
          <div class="mb-4 text-4xl">{{ testimonial.avatar }}</div>
          <p class="mb-4 text-balance leading-relaxed text-foreground">"{{ testimonial.quote }}"</p>
          <div>
            <div class="font-semibold text-foreground">{{ testimonial.author }}</div>
            <div class="text-sm text-muted-foreground">{{ testimonial.role }}</div>
          </div>
        </div>
      </div>

      <div class="mx-auto mt-12 grid max-w-4xl gap-8 sm:grid-cols-4">
        <div v-for="(stat, index) in stats" :key="index" class="text-center">
          <div :class="['mb-2 text-3xl font-bold', stat.highlighted ? 'text-primary' : 'text-foreground']">
            {{ stat.value }}
          </div>
          <div class="text-sm text-muted-foreground">{{ stat.label }}</div>
        </div>
      </div>
    </div>
  </section>
</template>
