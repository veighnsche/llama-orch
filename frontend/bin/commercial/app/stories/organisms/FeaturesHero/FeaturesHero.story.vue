<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import FeaturesHero from './FeaturesHero.vue'
</script>

<template>
  <Story title="organisms/FeaturesHero">
    <Variant title="Default">
      <FeaturesHero />
    </Variant>

    <Variant title="Custom Title">
      <FeaturesHero
        title="Production-Ready Features."
        highlight="Developer-Friendly."
      />
    </Variant>

    <Variant title="Custom Highlight">
      <FeaturesHero
        highlight="Zero Complexity."
        description="Build AI infrastructure that just works. No cloud dependencies. No ongoing costs. Complete control over your AI stack."
      />
    </Variant>
  </Story>
</template>
