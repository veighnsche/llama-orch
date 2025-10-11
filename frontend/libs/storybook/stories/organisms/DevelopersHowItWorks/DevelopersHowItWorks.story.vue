<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersHowItWorks from './DevelopersHowItWorks.vue'
</script>

<template>
  <Story title="organisms/DevelopersHowItWorks">
    <Variant title="Default">
      <DevelopersHowItWorks />
    </Variant>

    <Variant title="Custom Steps">
      <DevelopersHowItWorks
        title="Get Started in 3 Steps"
        :steps="[
          {
            number: 1,
            title: 'Install',
            code: ['npm install @rbee/cli', 'rbee init'],
            language: 'bash',
          },
          {
            number: 2,
            title: 'Configure',
            code: ['rbee config set endpoint http://localhost:8080'],
            language: 'bash',
          },
          {
            number: 3,
            title: 'Deploy',
            code: ['rbee deploy --model llama-3.1-70b'],
            language: 'bash',
          },
        ]"
      />
    </Variant>
  </Story>
</template>
