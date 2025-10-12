<!-- Created by: TEAM-FE-009 -->
<!-- TEAM-FE-009: Implemented Features page assembly -->
<!-- TEAM-FE-010: Adapted for Nuxt -->
<script setup lang="ts">
import {
  FeaturesHero,
  CoreFeaturesTabs,
  CrossNodeOrchestration,
  IntelligentModelManagement,
  MultiBackendGPU,
  ErrorHandling,
  RealTimeProgress,
  SecurityIsolation,
  AdditionalFeaturesGrid,
  EmailCapture,
} from 'rbee-storybook/stories'
</script>

<template>
  <div class="pt-16">
    <FeaturesHero />
    <CoreFeaturesTabs />
    <CrossNodeOrchestration />
    <IntelligentModelManagement />
    <MultiBackendGPU />
    <ErrorHandling />
    <RealTimeProgress />
    <SecurityIsolation />
    <AdditionalFeaturesGrid />
    <EmailCapture />
  </div>
</template>
