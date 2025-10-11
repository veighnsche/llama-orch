<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersPricing from './DevelopersPricing.vue'
</script>

<template>
  <Story title="organisms/DevelopersPricing">
    <Variant title="Default">
      <DevelopersPricing />
    </Variant>

    <Variant title="Two Tiers">
      <DevelopersPricing
        title=\"Simple Pricing\"
        :tiers=\"[
          {
            name: 'Free',
            price: '$0',
            period: 'forever',
            description: 'For individuals',
            features: ['Unlimited usage', 'Open source', 'Community support'],
            cta: 'Get Started',
            highlighted: false,
          },
          {
            name: 'Pro',
            price: '$49',
            period: '/month',
            description: 'For professionals',
            features: ['Everything in Free', 'Priority support', 'Advanced features'],
            cta: 'Start Trial',
            highlighted: true,
          },
        ]\"
      />
    </Variant>
  </Story>
</template>
