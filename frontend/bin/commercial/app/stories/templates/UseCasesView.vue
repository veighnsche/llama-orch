<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-009: Implemented Use Cases page assembly -->
<!-- TEAM-DX-006: Changed from barrel imports to local relative imports to fix Vite build issues -->
<script setup lang="ts">
import UseCasesHero from '../organisms/UseCasesHero/UseCasesHero.vue'
import UseCasesGrid from '../organisms/UseCasesGrid/UseCasesGrid.vue'
import IndustryUseCases from '../organisms/IndustryUseCases/IndustryUseCases.vue'
import EmailCapture from '../organisms/EmailCapture/EmailCapture.vue'
</script>

<template>
  <div class="pt-16">
    <UseCasesHero />
    <UseCasesGrid />
    <IndustryUseCases />
    <EmailCapture />
  </div>
</template>
