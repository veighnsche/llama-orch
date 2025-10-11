<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersFeatures from './DevelopersFeatures.vue'
</script>

<template>
  <Story title="organisms/DevelopersFeatures">
    <Variant title="Default">
      <DevelopersFeatures />
    </Variant>

    <Variant title="Custom Features">
      <DevelopersFeatures
        title="Key Features"
        :features="[
          {
            id: 'api',
            icon: 'code',
            title: 'REST API',
            description: 'Simple REST API for all operations',
            benefit: 'Easy integration',
            code: 'curl http://localhost:8080/api/models',
          },
          {
            id: 'cli',
            icon: 'terminal',
            title: 'CLI Tool',
            description: 'Command-line interface for management',
            benefit: 'Scriptable automation',
            code: 'rbee list models',
          },
        ]"
      />
    </Variant>
  </Story>
</template>
