<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-007: Added comprehensive variants -->
<script setup lang="ts">
import ProvidersHero from './ProvidersHero.vue'
</script>

<template>
  <Story title="organisms/ProvidersHero">
    <Variant title="Default">
      <ProvidersHero />
    </Variant>
    
    <Variant title="Custom Earnings">
      <ProvidersHero
        monthly-earnings="€100-300"
        title="Maximize Your GPU Revenue"
      />
    </Variant>
  </Story>
</template>
