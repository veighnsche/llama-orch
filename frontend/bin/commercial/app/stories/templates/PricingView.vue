<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-002: Implemented Pricing page assembly -->
<!-- TEAM-DX-006: Changed from barrel imports to local relative imports to fix Vite build issues -->
<script setup lang="ts">
import PricingHero from '../organisms/PricingHero/PricingHero.vue'
import PricingTiers from '../organisms/PricingTiers/PricingTiers.vue'
import PricingComparisonTable from '../organisms/PricingComparisonTable/PricingComparisonTable.vue'
import PricingFAQ from '../organisms/PricingFAQ/PricingFAQ.vue'
import EmailCapture from '../organisms/EmailCapture/EmailCapture.vue'
</script>

<template>
  <div class="pt-16">
    <PricingHero />
    <PricingTiers />
    <PricingComparisonTable />
    <PricingFAQ />
    <EmailCapture />
  </div>
</template>
