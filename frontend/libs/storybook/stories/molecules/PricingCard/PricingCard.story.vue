<!-- Created by: TEAM-FE-002 -->
<script setup lang="ts">
import PricingCard from './PricingCard.vue'
</script>

<template>
  <Story title="molecules/PricingCard">
    <Variant title="Home/Lab">
      <PricingCard
        title="Home/Lab"
        price="$0"
        priceSubtext="forever"
        description="For solo developers, hobbyists, homelab enthusiasts"
        :features="['Unlimited GPUs', 'OpenAI-compatible API', 'Multi-modal support', 'Community support', 'GPL open source', 'CLI access', 'Rhai scheduler']"
        buttonText="Download Now"
        buttonVariant="outline"
      />
    </Variant>

    <Variant title="Team">
      <PricingCard
        title="Team"
        price="€99"
        priceSubtext="/month"
        teamSize="5-10 developers"
        description="For small teams, startups"
        :features="['Everything in Home/Lab', 'Web UI management', 'Team collaboration', 'Priority support', 'Rhai script templates', 'Usage analytics', 'Email support']"
        buttonText="Start 30-Day Trial"
        buttonVariant="default"
        :highlighted="true"
        badge="Most Popular"
      />
    </Variant>

    <Variant title="Enterprise">
      <PricingCard
        title="Enterprise"
        price="Custom"
        priceSubtext=""
        teamSize="Contact sales"
        description="For large teams, enterprises"
        :features="['Everything in Team', 'Dedicated instances', 'Custom SLAs', 'White-label option', 'Enterprise support', 'On-premises deployment', 'Professional services']"
        buttonText="Contact Sales"
        buttonVariant="outline"
      />
    </Variant>
  </Story>
</template>
