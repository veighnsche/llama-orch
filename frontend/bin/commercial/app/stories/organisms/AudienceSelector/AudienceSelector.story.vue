<!-- TEAM-FE-004: Created AudienceSelector story -->
<script setup lang="ts">
import AudienceSelector from './AudienceSelector.vue'
</script>

<template>
  <Story title="organisms/AudienceSelector">
    <Variant title="Default">
      <AudienceSelector />
    </Variant>

    <Variant title="Custom Text">
      <AudienceSelector
        title="Find Your Solution"
        subtitle="Which path is right for you?"
        description="Discover how rbee can transform your AI infrastructure, whether you're a developer, provider, or enterprise."
      />
    </Variant>
  </Story>
</template>
