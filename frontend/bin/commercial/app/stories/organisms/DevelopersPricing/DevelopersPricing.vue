<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersPricing -->
<script setup lang="ts">
import { Button } from '~/stories'
import { Check } from 'lucide-vue-next'

interface PricingTier {
  name: string
  price: string
  period: string
  description: string
  features: string[]
  cta: string
  highlighted?: boolean
}

interface Props {
  title?: string
  tiers?: PricingTier[]
  footerText?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Start Free. Scale When Ready.',
  tiers: () => [
    {
      name: 'Home/Lab',
      price: '$0',
      period: 'forever',
      description: 'For solo developers and hobbyists',
      features: [
        'Unlimited GPUs',
        'OpenAI-compatible API',
        'Multi-modal support',
        'Community support',
        '100% open source',
        'llama-orch-utils included',
      ],
      cta: 'Download Now',
      highlighted: false,
    },
    {
      name: 'Team',
      price: '€99',
      period: '/month',
      description: 'For small teams (5-10 devs)',
      features: [
        'Everything in Home/Lab',
        'Web UI management',
        'Team collaboration',
        'Priority support',
        'Rhai script templates',
        'Advanced monitoring',
      ],
      cta: 'Start 30-Day Trial',
      highlighted: true,
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      period: '',
      description: 'For large teams and enterprises',
      features: [
        'Everything in Team',
        'Dedicated instances',
        'Custom SLAs',
        'White-label option',
        'Enterprise support',
        'On-premises deployment',
      ],
      cta: 'Contact Sales',
      highlighted: false,
    },
  ],
  footerText: 'All tiers include the full rbee orchestrator. No feature gates. No artificial limits.',
})
</script>

<template>
  <section class="border-b border-border bg-secondary/30 py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
      </div>

      <div class="mx-auto mt-16 grid max-w-6xl gap-8 lg:grid-cols-3">
        <div
          v-for="tier in tiers"
          :key="tier.name"
          :class="[
            'relative rounded-lg border p-8',
            tier.highlighted
              ? 'border-primary bg-gradient-to-b from-primary/10 to-background'
              : 'border-border bg-secondary/50',
          ]"
        >
          <div
            v-if="tier.highlighted"
            class="absolute -top-4 left-1/2 -translate-x-1/2 rounded-full border border-primary bg-primary px-4 py-1 text-sm font-medium text-primary-foreground"
          >
            Most Popular
          </div>

          <div class="mb-6">
            <h3 class="mb-2 text-xl font-semibold text-foreground">{{ tier.name }}</h3>
            <div class="mb-2 flex items-baseline gap-1">
              <span class="text-4xl font-bold text-foreground">{{ tier.price }}</span>
              <span class="text-muted-foreground">{{ tier.period }}</span>
            </div>
            <p class="text-sm text-muted-foreground">{{ tier.description }}</p>
          </div>

          <ul class="mb-8 space-y-3">
            <li v-for="feature in tier.features" :key="feature" class="flex items-start gap-3">
              <Check class="h-5 w-5 flex-shrink-0 text-primary" />
              <span class="text-sm text-foreground">{{ feature }}</span>
            </li>
          </ul>

          <Button
            :class="[
              'w-full',
              tier.highlighted
                ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                : 'border-border bg-muted text-foreground hover:bg-secondary',
            ]"
            :variant="tier.highlighted ? 'default' : 'outline'"
          >
            {{ tier.cta }}
          </Button>
        </div>
      </div>

      <div class="mx-auto mt-12 max-w-2xl text-center">
        <p class="text-sm text-muted-foreground">
          {{ footerText }}
        </p>
      </div>
    </div>
  </section>
</template>
