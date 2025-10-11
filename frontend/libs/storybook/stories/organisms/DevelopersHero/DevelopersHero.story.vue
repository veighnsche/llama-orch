<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersHero from './DevelopersHero.vue'
</script>

<template>
  <Story title="organisms/DevelopersHero">
    <Variant title="Default">
      <DevelopersHero />
    </Variant>

    <Variant title="Without Terminal">
      <DevelopersHero :show-terminal="false" />
    </Variant>

    <Variant title="Custom Content">
      <DevelopersHero
        badge="For AI Engineers"
        title="Build Faster."
        subtitle="Deploy Anywhere."
        description="Run AI models on your own infrastructure with complete control and zero vendor lock-in."
        primary-cta="Start Building"
        secondary-cta="Read Docs"
        :features="['Self-Hosted', 'Multi-GPU', 'Production Ready', 'Open Source']"
      />
    </Variant>
  </Story>
</template>
