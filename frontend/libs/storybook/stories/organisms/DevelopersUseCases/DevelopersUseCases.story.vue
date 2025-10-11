<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersUseCases from './DevelopersUseCases.vue'
</script>

<template>
  <Story title="organisms/DevelopersUseCases">
    <Variant title="Default">
      <DevelopersUseCases />
    </Variant>

    <Variant title="Custom Use Cases">
      <DevelopersUseCases
        title="Real-World Applications"
        :use-cases="[
          {
            icon: 'code',
            title: 'API Generation',
            scenario: 'Need to build REST APIs quickly',
            solution: 'Use AI to generate boilerplate and tests',
            outcome: '5x faster development',
          },
          {
            icon: 'wrench',
            title: 'Code Migration',
            scenario: 'Migrate from JavaScript to TypeScript',
            solution: 'Automated conversion with AI assistance',
            outcome: 'Weeks saved on migration',
          },
        ]"
      />
    </Variant>
  </Story>
</template>
