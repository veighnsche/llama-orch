<!-- Created by: TEAM-FE-009 -->
<!-- TEAM-FE-009: Implemented Use Cases page assembly -->
<!-- TEAM-FE-010: Adapted for Nuxt -->
<script setup lang="ts">
import {
  UseCasesHero,
  UseCasesGrid,
  IndustryUseCases,
  EmailCapture,
} from '~/stories'
</script>

<template>
  <div class="pt-16">
    <UseCasesHero />
    <UseCasesGrid />
    <IndustryUseCases />
    <EmailCapture />
  </div>
</template>
