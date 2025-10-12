<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented PricingSection component -->
<script setup lang="ts">
import { Check } from 'lucide-vue-next'
import { Button } from '~/stories'

interface PricingTier {
  name: string
  price: string
  period?: string
  description?: string
  features: string[]
  buttonText: string
  buttonVariant?: 'primary' | 'outline'
  highlighted?: boolean
  footer?: string
}

interface Props {
  title?: string
  tiers?: PricingTier[]
  footerText?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Start Free. Scale When Ready.',
  tiers: () => [
    {
      name: 'Home/Lab',
      price: '$0',
      period: 'forever',
      features: [
        'Unlimited GPUs',
        'OpenAI-compatible API',
        'Multi-modal support',
        'Community support',
        'Open source',
      ],
      buttonText: 'Download Now',
      buttonVariant: 'outline',
      footer: 'For solo developers, hobbyists, homelab enthusiasts',
    },
    {
      name: 'Team',
      price: '€99',
      period: '/month',
      description: '5-10 developers',
      features: [
        'Everything in Home/Lab',
        'Web UI management',
        'Team collaboration',
        'Priority support',
        'Rhai script templates',
      ],
      buttonText: 'Start 30-Day Trial',
      buttonVariant: 'primary',
      highlighted: true,
      footer: 'For small teams, startups',
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      description: 'Contact sales',
      features: [
        'Everything in Team',
        'Dedicated instances',
        'Custom SLAs',
        'White-label option',
        'Enterprise support',
      ],
      buttonText: 'Contact Sales',
      buttonVariant: 'outline',
      footer: 'For large teams, enterprises',
    },
  ],
  footerText: 'All tiers include the full rbee orchestrator. No feature gates. No artificial limits.',
})
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <div
          v-for="(tier, index) in tiers"
          :key="index"
          :class="[
            'rounded-lg p-8 space-y-6 relative',
            tier.highlighted
              ? 'bg-accent border-2 border-primary'
              : 'bg-card border-2 border-border',
          ]"
        >
          <div v-if="tier.highlighted" class="absolute -top-4 left-1/2 -translate-x-1/2">
            <span class="bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm font-medium">
              Most Popular
            </span>
          </div>

          <div>
            <h3 class="text-2xl font-bold text-card-foreground">{{ tier.name }}</h3>
            <div class="mt-4">
              <span class="text-4xl font-bold text-card-foreground">{{ tier.price }}</span>
              <span v-if="tier.period" class="text-muted-foreground ml-2">{{ tier.period }}</span>
            </div>
            <p v-if="tier.description" class="text-sm text-muted-foreground mt-1">
              {{ tier.description }}
            </p>
          </div>

          <ul class="space-y-3">
            <li
              v-for="(feature, fIndex) in tier.features"
              :key="fIndex"
              class="flex items-start gap-2"
            >
              <Check class="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
              <span
                :class="[
                  fIndex === 0 && index > 0 ? 'text-card-foreground font-medium' : 'text-muted-foreground',
                ]"
              >
                {{ feature }}
              </span>
            </li>
          </ul>

          <Button
            :variant="tier.buttonVariant === 'primary' ? 'default' : 'outline'"
            :class="[
              'w-full',
              tier.buttonVariant === 'primary'
                ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
                : '',
            ]"
          >
            {{ tier.buttonText }}
          </Button>

          <p v-if="tier.footer" class="text-sm text-muted-foreground text-center">
            {{ tier.footer }}
          </p>
        </div>
      </div>

      <p class="text-center text-muted-foreground mt-12 max-w-2xl mx-auto">
        {{ footerText }}
      </p>
    </div>
  </section>
</template>
