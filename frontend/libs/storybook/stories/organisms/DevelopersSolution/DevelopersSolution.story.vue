<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersSolution from './DevelopersSolution.vue'
</script>

<template>
  <Story title="organisms/DevelopersSolution">
    <Variant title="Default">
      <DevelopersSolution />
    </Variant>

    <Variant title="Without Architecture">
      <DevelopersSolution :show-architecture="false" />
    </Variant>

    <Variant title="Custom Content">
      <DevelopersSolution
        title="Self-Hosted AI Infrastructure"
        subtitle="Deploy AI models on your own servers with complete control and zero vendor lock-in."
        architecture-title="Distributed Architecture"
        :workers="[
          { name: 'Node 1', type: 'RTX 4090', emoji: '🚀' },
          { name: 'Node 2', type: 'M2 Max', emoji: '🚀' },
        ]"
      />
    </Variant>
  </Story>
</template>
