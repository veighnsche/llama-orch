<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersCodeExamples from './DevelopersCodeExamples.vue'
</script>

<template>
  <Story title="organisms/DevelopersCodeExamples">
    <Variant title="Default">
      <DevelopersCodeExamples />
    </Variant>

    <Variant title="Single Example">
      <DevelopersCodeExamples
        title="Quick Start"
        subtitle="Get started with one simple example"
        :examples="[
          {
            title: 'Hello World',
            code: 'import { invoke } from \\'@llama-orch/utils\\'\\n\\nconst result = await invoke({ prompt: \\'Hello!\\' })',
            language: 'TypeScript',
          },
        ]"
      />
    </Variant>
  </Story>
</template>
