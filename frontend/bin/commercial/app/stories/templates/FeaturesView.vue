<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-008: Implemented Features page assembly -->
<!-- TEAM-DX-006: Changed from barrel imports to local relative imports to fix Vite build issues -->
<script setup lang="ts">
import FeaturesHero from '../organisms/FeaturesHero/FeaturesHero.vue'
import CoreFeaturesTabs from '../organisms/CoreFeaturesTabs/CoreFeaturesTabs.vue'
import CrossNodeOrchestration from '../organisms/CrossNodeOrchestration/CrossNodeOrchestration.vue'
import IntelligentModelManagement from '../organisms/IntelligentModelManagement/IntelligentModelManagement.vue'
import MultiBackendGPU from '../organisms/MultiBackendGPU/MultiBackendGPU.vue'
import ErrorHandling from '../organisms/ErrorHandling/ErrorHandling.vue'
import RealTimeProgress from '../organisms/RealTimeProgress/RealTimeProgress.vue'
import SecurityIsolation from '../organisms/SecurityIsolation/SecurityIsolation.vue'
import AdditionalFeaturesGrid from '../organisms/AdditionalFeaturesGrid/AdditionalFeaturesGrid.vue'
import EmailCapture from '../organisms/EmailCapture/EmailCapture.vue'
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
