<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-002: Implemented PricingTiers organism -->
<script setup lang="ts">
import PricingCard from '../../molecules/PricingCard/PricingCard.vue'

interface Tier {
  title: string
  price: string
  priceSubtext: string
  description: string
  features: string[]
  buttonText: string
  buttonVariant?: 'default' | 'outline' | 'secondary' | 'destructive' | 'ghost' | 'link'
  highlighted?: boolean
  badge?: string
  teamSize?: string
}

const tiers: Tier[] = [
  {
    title: 'Home/Lab',
    price: '$0',
    priceSubtext: 'forever',
    description: 'For solo developers, hobbyists, homelab enthusiasts',
    features: [
      'Unlimited GPUs',
      'OpenAI-compatible API',
      'Multi-modal support',
      'Community support',
      'GPL open source',
      'CLI access',
      'Rhai scheduler',
    ],
    buttonText: 'Download Now',
    buttonVariant: 'outline',
  },
  {
    title: 'Team',
    price: '€99',
    priceSubtext: '/month',
    teamSize: '5-10 developers',
    description: 'For small teams, startups',
    features: [
      'Everything in Home/Lab',
      'Web UI management',
      'Team collaboration',
      'Priority support',
      'Rhai script templates',
      'Usage analytics',
      'Email support',
    ],
    buttonText: 'Start 30-Day Trial',
    buttonVariant: 'default',
    highlighted: true,
    badge: 'Most Popular',
  },
  {
    title: 'Enterprise',
    price: 'Custom',
    priceSubtext: '',
    teamSize: 'Contact sales',
    description: 'For large teams, enterprises',
    features: [
      'Everything in Team',
      'Dedicated instances',
      'Custom SLAs',
      'White-label option',
      'Enterprise support',
      'On-premises deployment',
      'Professional services',
    ],
    buttonText: 'Contact Sales',
    buttonVariant: 'outline',
  },
]
</script>

<template>
  <section class="py-24 bg-white">
    <div class="container mx-auto px-4">
      <div class="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <PricingCard
          v-for="tier in tiers"
          :key="tier.title"
          v-bind="tier"
        />
      </div>
    </div>
  </section>
</template>
