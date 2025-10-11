<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import UseCasesHero from './UseCasesHero.vue'
</script>

<template>
  <Story title="organisms/UseCasesHero">
    <Variant title="Default">
      <UseCasesHero />
    </Variant>

    <Variant title="Custom Title">
      <UseCasesHero
        title="AI Infrastructure for Everyone"
        highlight="Freedom"
      />
    </Variant>

    <Variant title="Custom Description">
      <UseCasesHero
        description="Whether you're a solo developer or a large enterprise, rbee scales to meet your needs without vendor lock-in."
      />
    </Variant>
  </Story>
</template>
